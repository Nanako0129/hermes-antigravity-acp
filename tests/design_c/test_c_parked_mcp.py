import http.client
import json
import queue
import re
import threading
import time
from urllib.parse import urlparse

import pytest

from antigravity_acp_mcp.parked_mcp import ParkedMcpServer

TOOLS = [{"name": "terminal", "description": "run", "inputSchema": {"type": "object"}},
         {"name": "read_file"}]


class Client:
    def __init__(self, server, token=None, path=None, headers=None):
        cfg = server.acp_mcp_server_config()
        self.url = urlparse(cfg["url"])
        self.path = path or self.url.path
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        auth = cfg["headers"][0]["value"] if token is None else token
        if auth:
            self.headers["Authorization"] = auth

    def post(self, payload, timeout=10):
        conn = http.client.HTTPConnection(self.url.hostname, self.url.port, timeout=timeout)
        conn.request("POST", self.path, json.dumps(payload), self.headers)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, (json.loads(body) if body else None)

    def call(self, rid, name="terminal", args=None, timeout=10):
        return self.post({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                          "params": {"name": name, "arguments": args or {}}}, timeout)


@pytest.fixture
def make():
    servers = []

    def factory(**kw):
        parked = queue.Queue()
        server = ParkedMcpServer(TOOLS, parked.put, **kw).start()
        servers.append(server)
        return server, parked

    yield factory
    for s in servers:
        s.close()


def in_thread(fn, *a, **kw):
    out = []
    t = threading.Thread(target=lambda: out.append(fn(*a, **kw)))
    t.start()
    return t, out


def test_transport_invariants(make):
    server, _ = make()
    cfg = server.acp_mcp_server_config()
    assert re.fullmatch(r"hermes-[0-9a-f]{8}", server.server_name) and cfg["name"] == server.server_name
    assert urlparse(cfg["url"]).hostname == "127.0.0.1" and urlparse(cfg["url"]).path != "/mcp"
    listing = Client(server).post({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})[1]
    assert [t["name"] for t in listing["result"]["tools"]] == ["terminal", "read_file"]
    assert Client(server, token="Bearer wrong").post({"id": 1})[0] == 401
    assert Client(server, token="").post({"id": 1})[0] == 401
    assert Client(server, headers={"Origin": "http://evil.example"}).post({"id": 1})[0] == 403
    assert Client(server, headers={"Host": "evil.example"}).post({"id": 1})[0] == 403
    assert Client(server, path="/mcp").post({"id": 1})[0] == 404
    status, body = Client(server).call(2, name="rm_rf")
    assert status == 200 and body["error"]["code"] == -32602


def test_parked_call_blocks_then_returns_the_answered_text(make):
    server, parked = make()
    t, out = in_thread(Client(server).call, 42, "terminal", {"cmd": "ls"})
    call = parked.get(timeout=5)  # handed to the caller; nothing executes
    assert (call.name, call.arguments) == ("terminal", {"cmd": "ls"})
    assert re.fullmatch(r"call_[0-9a-f]{32}", call.id)  # plugin-minted, not the JSON-RPC id
    time.sleep(0.2)
    assert t.is_alive() and out == []  # still blocked
    assert call.answer("file1\nfile2") is True
    t.join(5)
    assert out[0][1] == {"jsonrpc": "2.0", "id": 42, "result": {
        "content": [{"type": "text", "text": "file1\nfile2"}], "isError": False}}
    assert call.answer("late") is False  # first answer wins


def test_answer_with_is_error_is_forwarded(make):
    server, parked = make()
    t, out = in_thread(Client(server).call, 1)
    parked.get(timeout=5).answer("denied", is_error=True)
    t.join(5)
    assert out[0][1]["result"] == {"content": [{"type": "text", "text": "denied"}], "isError": True}


def test_retired_endpoint_refuses_immediately_and_never_parks(make):
    server, parked = make()
    server.retire()
    started = time.monotonic()
    status, body = Client(server).call(1)
    assert time.monotonic() - started < 1.0
    assert body["result"]["isError"] is True and "retired" in body["result"]["content"][0]["text"]
    assert parked.empty()


def test_retire_does_not_release_a_call_already_parked(make):
    server, parked = make()
    t, out = in_thread(Client(server).call, 1)
    call = parked.get(timeout=5)
    server.retire()
    time.sleep(0.2)
    assert t.is_alive()  # M4: nothing is released until the caller says so
    call.answer("ok")
    t.join(5)
    assert out[0][1]["result"]["isError"] is False


def test_release_parked_drops_the_connection_without_a_result(make):
    server, parked = make()
    t, out = in_thread(lambda: _expect_disconnect(Client(server)))
    parked.get(timeout=5)
    server.retire()
    server.release_parked(drop_connection=True)
    t.join(5)
    assert out == ["disconnected"]


def _expect_disconnect(client):
    try:
        client.call(1)
    except (http.client.RemoteDisconnected, ConnectionError, http.client.BadStatusLine):
        return "disconnected"
    return "got a response"


def test_release_parked_can_answer_is_error_instead(make):
    server, parked = make()
    t, out = in_thread(Client(server).call, 1)
    parked.get(timeout=5)
    server.release_parked("superseded", drop_connection=False)
    t.join(5)
    assert out[0][1]["result"] == {"content": [{"type": "text", "text": "superseded"}], "isError": True}


def test_duplicate_signature_shares_one_parked_call(make):
    server, parked = make()
    first, out1 = in_thread(Client(server).call, 1, "terminal", {"a": 1, "b": [2, 3]})
    call = parked.get(timeout=5)
    second, out2 = in_thread(Client(server).call, 99, "terminal", {"b": [2, 3], "a": 1})  # same, reordered
    time.sleep(0.3)
    assert parked.empty()  # no second ParkedCall
    other, out3 = in_thread(Client(server).call, 3, "terminal", {"a": 2})  # different args: new call
    other_call = parked.get(timeout=5)
    assert other_call is not call
    call.answer("shared")
    other_call.answer("other")
    for t in (first, second, other):
        t.join(5)
    assert out1[0][1]["result"]["content"][0]["text"] == "shared" == out2[0][1]["result"]["content"][0]["text"]
    assert out2[0][1]["id"] == 99 and out3[0][1]["result"]["content"][0]["text"] == "other"
    # after the answer the same signature is a fresh call again
    again, _ = in_thread(Client(server).call, 4, "terminal", {"a": 1, "b": [2, 3]})
    assert parked.get(timeout=5) is not call
    server.release_parked(drop_connection=False)
    again.join(5)


def test_deadline_returns_is_error_and_notifies(make):
    expired = []
    server, parked = make(deadline=0.3, on_deadline=expired.append)
    started = time.monotonic()
    status, body = Client(server).call(1)
    assert 0.25 < time.monotonic() - started < 3
    assert body["result"]["isError"] is True and "deadline" in body["result"]["content"][0]["text"]
    assert expired == [parked.get_nowait()] and expired[0].state == "deadline"


def test_failed_handoff_does_not_leave_the_request_parked():
    def boom(call):
        raise RuntimeError("callback bug")

    server = ParkedMcpServer(TOOLS, boom, deadline=30).start()
    try:
        started = time.monotonic()
        _, body = Client(server).call(1)
        assert time.monotonic() - started < 5 and body["result"]["isError"] is True
    finally:
        server.close()
