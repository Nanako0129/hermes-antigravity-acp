import http.client
import json
import queue
import re
import threading
import time
from urllib.parse import urlparse

import pytest

from antigravity_acp_mcp.parked_mcp import _MAX_COMPLETED_BYTES, _MAX_COMPLETED_CACHE, ParkedMcpServer

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
    first.join(5)
    second.join(5)
    other_call.answer("other")
    other.join(5)
    assert out1[0][1]["result"]["content"][0]["text"] == "shared" == out2[0][1]["result"]["content"][0]["text"]
    assert out2[0][1]["id"] == 99 and out3[0][1]["result"]["content"][0]["text"] == "other"
    # Before mark_completed, a repeat returns the cached answer without parking a new call.
    status, cached_body = Client(server).call(4, "terminal", {"a": 1, "b": [2, 3]})
    assert status == 200 and cached_body["id"] == 4
    assert cached_body["result"]["content"][0]["text"] == "shared"
    assert parked.empty()
    # After mark_completed evicts the oldest terminal entry (call), the same signature parks freshly.
    assert server.mark_completed("terminal") is True
    again, _ = in_thread(Client(server).call, 5, "terminal", {"a": 1, "b": [2, 3]})
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


def _call_and_answer(server, parked, rid, name, args, answer_text, *, images=None, is_error=False):
    t, out = in_thread(Client(server).call, rid, name, args)
    call = parked.get(timeout=5)
    call.answer(answer_text, images=images, is_error=is_error)
    t.join(5)
    assert not t.is_alive() and len(out) == 1
    return out[0]


def test_completed_cache_bounds_bytes_oversized_and_hashed_keys(make):
    server, parked = make()
    chunk_3m = "x" * (3 * 1024 * 1024)
    # Two 3 MiB entries fit within the 8 MiB cap.
    _call_and_answer(server, parked, 1, "terminal", {"i": 1}, chunk_3m)
    _call_and_answer(server, parked, 2, "terminal", {"i": 2}, chunk_3m)
    assert len(server._completed) == 2
    assert server._completed_bytes == 6 * 1024 * 1024

    # Third 3 MiB entry evicts the oldest (i=1) to stay <= _MAX_COMPLETED_BYTES.
    _call_and_answer(server, parked, 3, "terminal", {"i": 3}, chunk_3m)
    assert len(server._completed) == 2
    assert server._completed_bytes == 6 * 1024 * 1024
    # Newer entries still hit cache.
    _, b2 = Client(server).call(20, "terminal", {"i": 2})
    _, b3 = Client(server).call(30, "terminal", {"i": 3})
    assert len(b2["result"]["content"][0]["text"]) == len(chunk_3m)
    assert len(b3["result"]["content"][0]["text"]) == len(chunk_3m)
    assert parked.empty()

    # Single entry > _MAX_COMPLETED_BYTES is skipped without evicting existing cached entries.
    oversized = "y" * (_MAX_COMPLETED_BYTES + 64)
    _, b_over = _call_and_answer(server, parked, 4, "terminal", {"i": "big"}, oversized)
    assert len(b_over["result"]["content"][0]["text"]) == len(oversized)
    assert len(server._completed) == 2
    assert server._completed_bytes == 6 * 1024 * 1024
    # Repeating the oversized call parks freshly.
    t_over, _ = in_thread(Client(server).call, 5, "terminal", {"i": "big"})
    c_over = parked.get(timeout=5)
    c_over.answer("small")
    t_over.join(5)
    assert server.mark_completed("terminal") is True  # evicts i=2

    # Large-argument calls use fixed 32-byte SHA-256 keys and respect _MAX_COMPLETED_CACHE.
    server.clear_completed()
    large_arg = "z" * (256 * 1024)
    for idx in range(_MAX_COMPLETED_CACHE + 4):
        _call_and_answer(server, parked, 100 + idx, "read_file", {"blob": large_arg, "n": idx}, f"ok-{idx}")
    assert len(server._completed) == _MAX_COMPLETED_CACHE
    assert all(isinstance(k, bytes) and len(k) == 32 for k in server._completed)
    assert server._completed_bytes == sum(e.size_bytes for e in server._completed.values())
    assert server._completed_bytes < 1024


def test_toctou_answered_call_sharing_and_no_zombie_secondary(make):
    # (a) Duplicate call arrives after call.answer() sets state="answered" and before first handler acquires _lock.
    parked = queue.Queue()
    on_call_count = 0
    first_woke = threading.Event()
    release_first = threading.Event()

    def on_call_a(call):
        nonlocal on_call_count
        on_call_count += 1
        orig_wait = call.wait
        first_seen = False

        def wrapped_wait(timeout):
            nonlocal first_seen
            res = orig_wait(timeout)
            if not first_seen:
                first_seen = True
                first_woke.set()
                release_first.wait(5)
            return res

        call.wait = wrapped_wait
        parked.put(call)

    server = ParkedMcpServer(TOOLS, on_call_a).start()
    try:
        t1, out1 = in_thread(Client(server).call, 1, "terminal", {"k": "toctou"})
        call = parked.get(timeout=5)
        call.answer("toctou-ok")
        assert first_woke.wait(5)
        # call.state is "answered", still in _live, not yet in _completed.
        status2, body2 = Client(server).call(2, "terminal", {"k": "toctou"})
        assert status2 == 200 and body2["result"]["content"][0]["text"] == "toctou-ok"
        release_first.set()
        t1.join(5)
        assert out1[0][1]["result"]["content"][0]["text"] == "toctou-ok"
        assert on_call_count == 1
        assert len(server._completed) == 1
        entry = next(iter(server._completed.values()))
        assert entry.delivery_ok is True and entry.inflight_sends == 0
    finally:
        server.close()

    # (b) Secondary waiter paused past mark_completed() never re-inserts a zombie entry.
    server2, parked2 = make()
    second_woke = threading.Event()
    release_second = threading.Event()
    t1, out1 = in_thread(Client(server2).call, 10, "terminal", {"k": "zombie"})
    call2 = parked2.get(timeout=5)
    orig_wait2 = call2.wait
    waiter_count = 0
    waiter_lock = threading.Lock()

    def wrapped_wait2(timeout):
        nonlocal waiter_count
        with waiter_lock:
            waiter_count += 1
            is_second = waiter_count == 2
        res = orig_wait2(timeout)
        if is_second:
            second_woke.set()
            release_second.wait(5)
        return res

    call2.wait = wrapped_wait2
    # Wait briefly until first waiter is blocked in orig_wait2 (it already entered wait before we wrapped,
    # so only the second waiter goes through wrapped_wait2).
    waiter_count = 1
    t2, out2 = in_thread(Client(server2).call, 11, "terminal", {"k": "zombie"})
    time.sleep(0.1)
    assert parked2.empty()
    call2.answer("no-zombie")
    t1.join(5)
    assert out1[0][1]["result"]["content"][0]["text"] == "no-zombie"
    assert second_woke.wait(5)
    assert server2.mark_completed("terminal") is True
    assert server2._completed == {} and server2._completed_bytes == 0
    release_second.set()
    t2.join(5)
    assert out2[0][1]["result"]["content"][0]["text"] == "no-zombie"
    assert server2._completed == {} and server2._completed_bytes == 0


def test_clear_completed_and_preserve_parked_block_stragglers():
    parked = queue.Queue()
    woke = {}
    release = {}

    def on_call(call):
        tag = call.arguments["id"]
        if tag in ("stale", "r1"):
            woke[tag] = threading.Event()
            release[tag] = threading.Event()
            orig_wait = call.wait

            def paused_wait(timeout):
                res = orig_wait(timeout)
                woke[tag].set()
                release[tag].wait(5)
                return res

            call.wait = paused_wait
        parked.put(call)

    server = ParkedMcpServer(TOOLS, on_call).start()
    try:
        # SEC-M3-3: clear_completed() blocks a delayed answered straggler from inserting into _completed.
        t_stale, out_stale = in_thread(Client(server).call, 1, "terminal", {"id": "stale"})
        c_stale = parked.get(timeout=5)
        c_stale.answer("stale-ans")
        assert woke["stale"].wait(5)
        server.clear_completed()
        release["stale"].set()
        t_stale.join(5)
        assert out_stale[0][1]["result"]["content"][0]["text"] == "stale-ans"
        assert server._completed == {} and server._completed_bytes == 0

        # SEC-M3-7: clear_completed(preserve_parked=True) blocks prior-round answered straggler while
        # preserving a currently parked call's epoch so it enters _completed when answered.
        t_r1, out_r1 = in_thread(Client(server).call, 2, "terminal", {"id": "r1"})
        c_r1 = parked.get(timeout=5)
        c_r1.answer("r1-ans")
        assert woke["r1"].wait(5)

        t_r2, out_r2 = in_thread(Client(server).call, 3, "terminal", {"id": "r2"})
        c_r2 = parked.get(timeout=5)
        server.clear_completed(preserve_parked=True)

        release["r1"].set()
        t_r1.join(5)
        assert out_r1[0][1]["result"]["content"][0]["text"] == "r1-ans"
        assert server._completed == {}

        c_r2.answer("r2-ans")
        t_r2.join(5)
        assert out_r2[0][1]["result"]["content"][0]["text"] == "r2-ans"
        assert len(server._completed) == 1
        # Repeating r2 hits cache; repeating r1 parks freshly.
        _, body_r2 = Client(server).call(4, "terminal", {"id": "r2"})
        assert body_r2["result"]["content"][0]["text"] == "r2-ans"
        assert parked.empty()
        t_r1_again, _ = in_thread(Client(server).call, 5, "terminal", {"id": "r1"})
        assert parked.get(timeout=5).arguments == {"id": "r1"}
        server.release_parked(drop_connection=False)
        t_r1_again.join(5)
    finally:
        server.close()


def test_delivery_failure_inflight_sends_images_and_error_skip(make):
    # 1) Initial OSError -> retry OSError -> retry succeeds -> mark_completed evicts.
    server, parked = make()
    handler_cls = server._httpd.RequestHandlerClass
    orig_send = handler_cls._send
    send_calls = 0

    def fail_first_two(self, status, payload=None):
        nonlocal send_calls
        if status == 200 and isinstance(payload, dict) and "result" in payload:
            send_calls += 1
            if send_calls <= 2:
                raise BrokenPipeError("simulated drop")
        return orig_send(self, status, payload)

    handler_cls._send = fail_first_two
    t1, out1 = in_thread(lambda: _expect_disconnect(Client(server)))
    call = parked.get(timeout=5)
    call.answer("recovered")
    t1.join(5)
    assert out1 == ["disconnected"]
    assert len(server._completed) == 1
    entry = next(iter(server._completed.values()))
    assert entry.delivery_failed is True and entry.delivery_ok is False and entry.inflight_sends == 0
    # mark_completed NEVER evicts a delivery_failed entry.
    assert server.mark_completed("terminal") is False
    assert len(server._completed) == 1

    # Second attempt (first retry) also raises OSError -> stays cached for another retry.
    assert _expect_disconnect(Client(server)) == "disconnected"
    assert parked.empty()
    assert entry.delivery_failed is True and entry.inflight_sends == 0
    assert server.mark_completed("terminal") is False

    # Third attempt (second retry) succeeds -> delivery_ok=True, mark_completed evicts it.
    status, body = Client(server).call(3)
    assert status == 200 and body["result"]["content"][0]["text"] == "recovered"
    assert parked.empty()
    assert entry.delivery_ok is True and entry.delivery_failed is False and entry.inflight_sends == 0
    assert server.mark_completed("terminal") is True
    assert server._completed == {} and server._completed_bytes == 0

    # 2) Slow-RST OSError while retry is in flight (inflight_sends > 0) does NOT set delivery_failed.
    send1_entered = threading.Event()
    send1_fail = threading.Event()
    send2_entered = threading.Event()
    send2_finish = threading.Event()
    inflight_mode = {"n": 0}

    def slow_rst_inflight(self, status, payload=None):
        if status == 200 and isinstance(payload, dict) and "result" in payload:
            inflight_mode["n"] += 1
            n = inflight_mode["n"]
            if n == 1:
                send1_entered.set()
                send1_fail.wait(5)
                raise ConnectionResetError("slow rst")
            if n == 2:
                send2_entered.set()
                send2_finish.wait(5)
        return orig_send(self, status, payload)

    handler_cls._send = slow_rst_inflight
    ta, outa = in_thread(lambda: _expect_disconnect(Client(server)))
    ca = parked.get(timeout=5)
    ca.answer("slow-rst-1")
    assert send1_entered.wait(5)
    tb, outb = in_thread(Client(server).call, 22)
    assert send2_entered.wait(5)
    entry2 = next(iter(server._completed.values()))
    assert entry2.inflight_sends == 2
    send1_fail.set()
    ta.join(5)
    assert outa == ["disconnected"]
    # Wait until send1's finally ran.
    deadline = time.monotonic() + 5
    while entry2.inflight_sends != 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert entry2.inflight_sends == 1 and entry2.delivery_failed is False
    send2_finish.set()
    tb.join(5)
    assert outb[0][1]["result"]["content"][0]["text"] == "slow-rst-1"
    assert entry2.delivery_ok is True and entry2.delivery_failed is False and entry2.inflight_sends == 0
    assert server.mark_completed("terminal") is True

    # 3) Slow-RST OSError AFTER retry already succeeded (delivery_ok=True) does NOT set delivery_failed.
    send3_entered = threading.Event()
    send3_fail = threading.Event()
    after_ok_mode = {"n": 0}

    def slow_rst_after_ok(self, status, payload=None):
        if status == 200 and isinstance(payload, dict) and "result" in payload:
            after_ok_mode["n"] += 1
            if after_ok_mode["n"] == 1:
                send3_entered.set()
                send3_fail.wait(5)
                raise ConnectionResetError("late rst after ok")
        return orig_send(self, status, payload)

    handler_cls._send = slow_rst_after_ok
    tc, outc = in_thread(lambda: _expect_disconnect(Client(server)))
    cc = parked.get(timeout=5)
    cc.answer("slow-rst-2")
    assert send3_entered.wait(5)
    status_d, body_d = Client(server).call(33)
    assert status_d == 200 and body_d["result"]["content"][0]["text"] == "slow-rst-2"
    entry3 = next(iter(server._completed.values()))
    assert entry3.delivery_ok is True and entry3.inflight_sends == 1
    send3_fail.set()
    tc.join(5)
    assert outc == ["disconnected"]
    deadline = time.monotonic() + 5
    while entry3.inflight_sends != 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert entry3.delivery_ok is True and entry3.delivery_failed is False
    assert server.mark_completed("terminal") is True

    # 4) Images preserved on cache hit; is_error=True never cached.
    handler_cls._send = orig_send
    _, img_first = _call_and_answer(
        server, parked, 40, "terminal", {"img": True}, "pic", images=[("image/png", "QUFB")]
    )
    _, img_cached = Client(server).call(41, "terminal", {"img": True})
    assert parked.empty()
    assert img_first["result"]["content"] == img_cached["result"]["content"] == [
        {"type": "text", "text": "pic"},
        {"type": "image", "mimeType": "image/png", "data": "QUFB"},
    ]
    assert server.mark_completed("terminal") is True

    _, err_first = _call_and_answer(server, parked, 50, "terminal", {"err": True}, "denied", is_error=True)
    assert err_first["result"]["isError"] is True
    assert server._completed == {}
    t_err, _ = in_thread(Client(server).call, 51, "terminal", {"err": True})
    assert parked.get(timeout=5).arguments == {"err": True}
    server.release_parked(drop_connection=False)
    t_err.join(5)
