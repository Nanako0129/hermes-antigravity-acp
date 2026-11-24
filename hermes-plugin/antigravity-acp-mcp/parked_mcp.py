"""Loopback Streamable-HTTP MCP server whose ``tools/call`` parks instead of executing.

PlanC component 4. Same transport invariants as ``hermes_acp_mcp_bridge.mcp_server``
(127.0.0.1 ephemeral port, bearer via ``hmac.compare_digest``, Origin/Host checks,
declared-tool allowlist, handler socket timeout) with the executor replaced by "park":
the call is handed to ``on_call`` and the HTTP request blocks until the caller answers
it, abandons it, or the deadline passes. The server never runs a tool.

One server name, bearer token and URL path per ACP session (M1). Release ordering (M4),
for the caller: ``retire()`` first (stragglers refused), then ``session/cancel``, wait for
the prompt reply ``cancelled``, then ``release_parked()`` -- so a live prompt never sees
an error answer for a call it is still waiting on.
"""

from __future__ import annotations

import hashlib
import hmac
import http.server
import json
import secrets
import socketserver
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

DEFAULT_DEADLINE_SECONDS = 170.0  # agy's own MCP tools/call limit is 180 s (measured)
_SOCKET_TIMEOUT = 30.0  # L4: bounds slow-read clients; does not limit a parked wait
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")
_DROP = object()  # sentinel: close the HTTP connection without a JSON-RPC result
_MAX_COMPLETED_CACHE = 32
_MAX_COMPLETED_BYTES = 8 * 1024 * 1024


@dataclass
class _CompletedCall:
    name: str
    text: str
    is_error: bool
    images: list[tuple[str, str]]
    seq: int
    size_bytes: int
    delivery_ok: bool = False
    delivery_failed: bool = False
    inflight_sends: int = 0


def _canonical_signature(name: str, arguments: dict[str, Any]) -> str:
    return name + "\0" + json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class ParkedCall:
    """One suspended ``tools/call``. ``id`` is plugin-minted, unrelated to the JSON-RPC id.

    ``call_`` + hex: no provider prefix Hermes rewrites (``chatcmpl-tool-``,
    message_sanitization.py) and not ``call_<digits>``.
    """

    def __init__(self, name: str, arguments: dict[str, Any]) -> None:
        self.id = "call_" + uuid.uuid4().hex
        self.name = name
        self.arguments = arguments
        self.signature = _canonical_signature(name, arguments)
        self.state = "parked"  # parked | answered | abandoned | deadline
        self.text = ""
        self.is_error = False
        self.images: list[tuple[str, str]] = []  # (mimeType, base64) sent as MCP image content
        self.drop_connection = False
        self._epoch: int = 0
        self._completed_entry: _CompletedCall | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()

    def _finish(self, state: str, text: str, is_error: bool, drop: bool = False,
                images: list[tuple[str, str]] | None = None) -> bool:
        with self._lock:
            if self.state != "parked":
                return False
            self.text, self.is_error, self.drop_connection = text, is_error, drop
            self.images = list(images or [])
            self.state = state
        self._done.set()
        return True

    def answer(self, text: str, images: list[tuple[str, str]] | None = None, is_error: bool = False) -> bool:
        return self._finish("answered", text, is_error, images=images)

    def abandon(self, reason: str = "superseded", *, drop_connection: bool = False) -> bool:
        return self._finish("abandoned", reason, True, drop_connection)

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)


class _LoopbackHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _rpc_error(req_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def _tool_result(req_id: Any, text: str, is_error: bool, images: list[tuple[str, str]] = ()) -> dict[str, Any]:
    content = [{"type": "text", "text": text}]
    content += [{"type": "image", "mimeType": mime, "data": data} for mime, data in images]
    return {"jsonrpc": "2.0", "id": req_id, "result": {"content": content, "isError": is_error}}


class ParkedMcpServer:
    def __init__(
        self,
        tools: Sequence[dict[str, Any]],
        on_call: Callable[[ParkedCall], None],
        *,
        deadline: float = DEFAULT_DEADLINE_SECONDS,
        on_deadline: Callable[[ParkedCall], None] | None = None,
    ) -> None:
        """``tools``: MCP tool dicts (``name``, ``description``, ``inputSchema``)."""
        self.server_name = "hermes-" + secrets.token_hex(4)
        self._token = secrets.token_urlsafe(32)
        self._path = "/mcp/" + secrets.token_urlsafe(12)
        self._tools = {str(t["name"]): t for t in tools}
        self._on_call = on_call
        self._on_deadline = on_deadline
        self._deadline = deadline
        self._lock = threading.Lock()
        self._live: dict[bytes, ParkedCall] = {}  # sha256(signature) -> call parked or handed out
        self._completed: dict[bytes, _CompletedCall] = {}  # sha256(signature) -> answered call awaiting ACP completion
        self._completed_bytes: int = 0
        self._completed_seq: int = 0
        self._epoch: int = 0
        self._retired = False
        self._httpd: _LoopbackHTTPServer | None = None

    @property
    def tool_names(self) -> frozenset[str]:
        return frozenset(self._tools)

    @property
    def url(self) -> str:
        assert self._httpd is not None, "server not started"
        return f"http://127.0.0.1:{self._httpd.server_address[1]}{self._path}"

    def acp_mcp_server_config(self) -> dict[str, Any]:
        return {
            "type": "http",
            "name": self.server_name,
            "url": self.url,
            "headers": [{"name": "Authorization", "value": f"Bearer {self._token}"}],
        }

    # -- control -------------------------------------------------------------------

    def retire(self) -> None:
        """Every later ``tools/call`` gets ``isError`` at once and is never parked."""
        with self._lock:
            self._retired = True
            self._completed.clear()
            self._completed_bytes = 0
            self._epoch += 1

    def mark_completed(self, tool_name: str) -> bool:
        """Evict the oldest completed entry for ``tool_name`` once ACP reports it completed."""
        with self._lock:
            tier1 = [
                (k, e)
                for k, e in self._completed.items()
                if e.name == tool_name and e.delivery_ok and not e.delivery_failed
            ]
            candidates = tier1 or [
                (k, e)
                for k, e in self._completed.items()
                if e.name == tool_name and not e.delivery_failed
            ]
            if not candidates:
                return False
            sig_key, entry = min(candidates, key=lambda item: item[1].seq)
            del self._completed[sig_key]
            self._completed_bytes = max(0, self._completed_bytes - entry.size_bytes)
            return True

    def clear_completed(self, *, preserve_parked: bool = False) -> None:
        """Flush cached completed answers and advance the cache epoch."""
        with self._lock:
            self._completed.clear()
            self._completed_bytes = 0
            self._epoch += 1
            if preserve_parked:
                for call in self._live.values():
                    with call._lock:
                        if call.state == "parked":
                            call._epoch = self._epoch

    def release_parked(self, reason: str = "superseded", *, drop_connection: bool = True) -> None:
        """Release every live call: drop its HTTP connection without a result, or answer isError."""
        with self._lock:
            calls = list(self._live.values())
        for call in calls:
            call.abandon(reason, drop_connection=drop_connection)

    def close(self) -> None:
        self.retire()
        self.release_parked()
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    def start(self) -> ParkedMcpServer:
        bridge = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = _SOCKET_TIMEOUT

            def log_message(self, format: str, *args: Any) -> None:
                return

            def _send(self, status: int, payload: Any = None) -> None:
                body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                if body:
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _authorize(self) -> bool:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).hostname not in _LOOPBACK_HOSTS:
                    self._send(403, {"error": "forbidden origin"})
                    return False
                host = self.headers.get("Host")
                if host and urlparse("//" + host).hostname not in _LOOPBACK_HOSTS:
                    self._send(403, {"error": "forbidden host"})
                    return False
                auth = self.headers.get("Authorization") or ""
                if not auth.startswith("Bearer ") or not hmac.compare_digest(
                    auth[7:].encode(), bridge._token.encode()
                ):
                    self._send(401, {"error": "unauthorized"})
                    return False
                return True

            def _guarded(self) -> bool:
                if self.path.split("?", 1)[0] != bridge._path:
                    self._send(404, {"error": "not found"})
                    return False
                return self._authorize()

            def do_GET(self) -> None:
                if self._guarded():
                    self._send(405)

            def do_DELETE(self) -> None:
                if self._guarded():
                    self._send(200)

            def do_POST(self) -> None:
                if not self._guarded():
                    return
                try:
                    length = int(self.headers.get("Content-Length") or "0")
                except ValueError:
                    length = 0
                if not 0 < length <= 16 * 1024 * 1024:
                    self._send(400, {"error": "invalid request body size"})
                    return
                try:
                    msg = json.loads(self.rfile.read(length))
                except ValueError:
                    msg = None
                if not isinstance(msg, dict):
                    self._send(400, _rpc_error(None, -32700, "Parse error"))
                    return
                if msg.get("id") is None:  # notification
                    self._send(202)
                    return
                response, sig_key, entry = bridge._dispatch_rpc(msg["id"], msg.get("method"), msg.get("params"))
                if response is _DROP:
                    self.close_connection = True
                    return
                if entry is None:
                    try:
                        self._send(200, response)
                    except OSError:
                        self.close_connection = True
                else:
                    succeeded = False
                    try:
                        self._send(200, response)
                        succeeded = True
                    except OSError:
                        self.close_connection = True
                    finally:
                        with bridge._lock:
                            entry.inflight_sends = max(0, entry.inflight_sends - 1)
                            if sig_key is not None and bridge._completed.get(sig_key) is entry:
                                if succeeded:
                                    entry.delivery_ok = True
                                    entry.delivery_failed = False
                                elif not entry.delivery_ok and entry.inflight_sends == 0:
                                    entry.delivery_failed = True

        self._httpd = _LoopbackHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(
            target=self._httpd.serve_forever, name=f"mcp-parked-{self.server_name}", daemon=True
        ).start()
        return self

    # -- JSON-RPC ------------------------------------------------------------------

    def _handle_rpc(self, req_id: Any, method: Any, params: Any) -> Any:
        return self._dispatch_rpc(req_id, method, params)[0]

    def _dispatch_rpc(
        self, req_id: Any, method: Any, params: Any
    ) -> tuple[Any, bytes | None, _CompletedCall | None]:
        if method == "initialize":
            return (
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": self.server_name, "version": "0.1.0"},
                    },
                },
                None,
                None,
            )
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}, None, None
        if method == "tools/list":
            tools = [
                {"description": "", "inputSchema": {"type": "object", "properties": {}}, **t}
                for t in self._tools.values()
            ]
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}, None, None
        if method == "tools/call":
            return self._handle_tool_call_with_ref(req_id, params)
        return _rpc_error(req_id, -32601, f"Method not found: {method}"), None, None

    def _handle_tool_call_with_ref(
        self, req_id: Any, params: Any
    ) -> tuple[Any, bytes | None, _CompletedCall | None]:
        if not isinstance(params, dict):
            return _rpc_error(req_id, -32602, "Invalid params"), None, None
        name = str(params.get("name") or "").strip()
        arguments = params.get("arguments")
        arguments = {} if arguments is None else arguments
        if not isinstance(arguments, dict):
            return _rpc_error(req_id, -32602, "Tool arguments must be an object"), None, None
        if name not in self._tools:
            return _rpc_error(req_id, -32602, f"Undeclared tool: {name}"), None, None

        signature = _canonical_signature(name, arguments)
        sig_key = hashlib.sha256(signature.encode("utf-8")).digest()
        with self._lock:
            if self._retired:
                return _tool_result(req_id, "session retired", True), None, None
            if (cached := self._completed.get(sig_key)) is not None:
                cached.inflight_sends += 1
                cached.delivery_ok = False
                cached.delivery_failed = False
                return _tool_result(req_id, cached.text, cached.is_error, cached.images), sig_key, cached
            call = self._live.get(sig_key)  # M3: same signature shares the parked call
            if call is not None:
                with call._lock:
                    reusable = call.state == "parked" or (
                        call.state == "answered"
                        and not call.is_error
                        and not call.drop_connection
                        and call._epoch == self._epoch
                    )
            else:
                reusable = False
            fresh = not reusable
            if fresh:
                call = ParkedCall(name, arguments)
                call._epoch = self._epoch
                self._live[sig_key] = call
            assert call is not None
        if fresh:
            try:
                self._on_call(call)
            except Exception as exc:  # noqa: BLE001 - never leave an HTTP request parked on a failed hand-off
                call.abandon(f"hand-off failed: {type(exc).__name__}")
        if not call.wait(self._deadline) and call._finish(
            "deadline", f"tool call deadline exceeded after {self._deadline:g}s", True
        ):
            if self._on_deadline:
                self._on_deadline(call)
        completed_ref: tuple[bytes | None, _CompletedCall | None] = (None, None)
        with self._lock:
            if self._live.get(sig_key) is call:
                del self._live[sig_key]
                if (
                    call.state == "answered"
                    and not call.is_error
                    and not call.drop_connection
                    and not self._retired
                    and call._epoch == self._epoch
                ):
                    size_bytes = len(call.text.encode("utf-8")) + sum(
                        len(m.encode("utf-8")) + len(d.encode("utf-8")) for m, d in call.images
                    )
                    if size_bytes <= _MAX_COMPLETED_BYTES:
                        self._completed_seq += 1
                        entry = _CompletedCall(
                            name=name,
                            text=call.text,
                            is_error=call.is_error,
                            images=list(call.images),
                            seq=self._completed_seq,
                            size_bytes=size_bytes,
                            inflight_sends=1,
                        )
                        if (old := self._completed.pop(sig_key, None)) is not None:
                            self._completed_bytes -= old.size_bytes
                        self._completed[sig_key] = entry
                        self._completed_bytes += size_bytes
                        while (
                            len(self._completed) > _MAX_COMPLETED_CACHE
                            or self._completed_bytes > _MAX_COMPLETED_BYTES
                        ):
                            oldest_key = min(self._completed, key=lambda k: self._completed[k].seq)
                            evicted = self._completed.pop(oldest_key)
                            self._completed_bytes -= evicted.size_bytes
                        call._completed_entry = entry
                        completed_ref = (sig_key, entry)
            else:
                entry = call._completed_entry
                if entry is not None and self._completed.get(sig_key) is entry:
                    entry.inflight_sends += 1
                    entry.delivery_failed = False
                    completed_ref = (sig_key, entry)
        if call.drop_connection:
            return _DROP, None, None
        return _tool_result(req_id, call.text, call.is_error, call.images), completed_ref[0], completed_ref[1]
