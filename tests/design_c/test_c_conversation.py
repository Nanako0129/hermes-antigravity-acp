"""PlanC S3/S4: conversation turns against a fake agy that really calls the parked MCP endpoint."""

import contextlib
import json
import sys
import threading
import time

import pytest

from antigravity_acp_mcp.acp_connection import AcpConnection
from antigravity_acp_mcp.client import AntigravityAcpMcpClient
from antigravity_acp_mcp.conversation import (
    GUARD_TEXT,
    CancelledByUser,
    Conversation,
    RateLimited,
    StepTimeout,
)

from ._support import FAKE_AGY as FAKE, SYS, TOOLS, asst, events, make_conv, tool_rows

U1 = {"role": "user", "content": "check a and b"}
SAMPLE_429 = ("E1123 02:31:21.478230     119 errorreport.go:273] Error 429, Message: Resource has been exhausted "
              "(e.g. check quota)., Status: RESOURCE_EXHAUSTED, Details: []")


def test_two_hop_turn_is_one_prompt_then_continue_sends_only_the_delta(tmp_path):
    conv, log = make_conv(tmp_path, [
        [{"call": {"name": "fetch", "arguments": {"k": "a"}}},
         {"call": {"name": "fetch", "arguments": {"k": "b"}}}, {"text": "both fine"}],
        [{"text": "c fine"}],
    ])
    try:
        msgs = [SYS, U1]
        r1 = conv.run(msgs, TOOLS, "m")
        assert [c.arguments for c in r1["tool_calls"]] == [{"k": "a"}]
        msgs += [asst(r1["tool_calls"]), *tool_rows(r1["tool_calls"], ["A"])]
        r2 = conv.run(msgs, TOOLS, "m")
        assert [c.arguments for c in r2["tool_calls"]] == [{"k": "b"}]
        msgs += [asst(r2["tool_calls"]), *tool_rows(r2["tool_calls"], ["B"])]
        r3 = conv.run(msgs, TOOLS, "m")
        assert r3["tool_calls"] == [] and r3["text"] == "both fine"
        ev = events(log)
        assert [e["n"] for e in ev if e["ev"] == "prompt"] == [0]  # one ACP prompt for the whole turn
        assert [e["text"] for e in ev if e["ev"] == "answer"] == ["A", "B"]
        assert "SYS-MARK" in ev[1]["text"]  # system band in the prompt, no append
        meta = ev[0]["meta"]["antigravity"]
        assert set(meta) == {"disabledTools", "eagerMcpServers", "mcpMaxOutputBytes"}  # no systemPrompt append
        assert meta["eagerMcpServers"][0].startswith("hermes-")
        msgs += [{"role": "assistant", "content": "both fine (as stored by Hermes)"},
                 {"role": "user", "content": "now c"}]
        r4 = conv.run(msgs, TOOLS, "m")
        assert r4["text"] == "c fine"
        prompts = [e for e in events(log) if e["ev"] == "prompt"]
        assert prompts[1]["sid"] == prompts[0]["sid"] and prompts[1]["text"] == "now c"
    finally:
        conv.close()


def test_deadline_retires_and_replay_guard_blocks_reissue(tmp_path):
    conv, log = make_conv(tmp_path, [
        [{"call": {"name": "fetch", "arguments": {"k": "a"}}}, {"text": "never"}],
        [{"call": {"name": "fetch", "arguments": {"k": "a"}}}, {"echo_last": True}],
    ], call_deadline=0.5)
    try:
        msgs = [SYS, U1]
        r1 = conv.run(msgs, TOOLS, "m")
        time.sleep(1.5)  # Hermes approval takes longer than the call deadline
        assert any(e["ev"] == "cancel" for e in events(log))
        msgs += [asst(r1["tool_calls"]), *tool_rows(r1["tool_calls"], ["A-late"])]
        r2 = conv.run(msgs, TOOLS, "m")
        assert r2["tool_calls"] == []  # the re-issued call was not handed to Hermes again
        assert GUARD_TEXT in r2["text"]
        ev = events(log)
        prompts = [e for e in ev if e["ev"] == "prompt"]
        assert prompts[1]["sid"] != prompts[0]["sid"]  # fresh session after the cancel
        assert "A-late" in prompts[1]["text"] and "already executed" in prompts[1]["text"]
    finally:
        conv.close()


def test_new_user_row_while_parked_supersedes(tmp_path):
    conv, log = make_conv(tmp_path, [[{"call": {"name": "fetch", "arguments": {"k": "a"}}}], [{"text": "redo"}]])
    try:
        msgs = [SYS, U1]
        r1 = conv.run(msgs, TOOLS, "m")
        msgs += [asst(r1["tool_calls"]), *tool_rows(r1["tool_calls"], ["A"]),
                 {"role": "user", "content": "also c"}]
        r2 = conv.run(msgs, TOOLS, "m")
        assert r2["text"] == "redo"
        ev = events(log)
        assert any(e["ev"] == "cancel" for e in ev)
        assert len({e["sid"] for e in ev if e["ev"] == "prompt"}) == 2
    finally:
        conv.close()


def test_cancel_while_in_flight_returns_fast(tmp_path):
    conv, log = make_conv(tmp_path, [[{"silent": 20}]])
    try:
        err = {}
        token = threading.Event()

        def go():
            try:
                conv.run([SYS, U1], TOOLS, "m", token)
            except Exception as exc:  # noqa: BLE001
                err["e"] = exc

        th = threading.Thread(target=go)
        th.start()
        time.sleep(1.0)
        t = time.monotonic()
        conv.cancel(token)
        th.join(5)
        assert not th.is_alive() and time.monotonic() - t < 5
        assert isinstance(err.get("e"), CancelledByUser)
        assert any(e["ev"] == "cancel" for e in events(log))
    finally:
        conv.close()


def test_silent_model_step_times_out(tmp_path):
    conv, _ = make_conv(tmp_path, [[{"silent": 10}]], step_deadline=0.8)
    try:
        t = time.monotonic()
        with pytest.raises(StepTimeout):
            conv.run([SYS, U1], TOOLS, "m")
        assert time.monotonic() - t < 5
    finally:
        conv.close()


def test_silence_after_resume_also_times_out(tmp_path):
    conv, _ = make_conv(tmp_path, [[{"call": {"name": "fetch", "arguments": {"k": "a"}}}, {"silent": 10}]],
                        step_deadline=0.8)
    try:
        msgs = [SYS, U1]
        r1 = conv.run(msgs, TOOLS, "m")
        time.sleep(1.5)  # waiting on Hermes does not count as model inactivity
        msgs += [asst(r1["tool_calls"]), *tool_rows(r1["tool_calls"], ["A"])]
        with pytest.raises(StepTimeout):
            conv.run(msgs, TOOLS, "m")
    finally:
        conv.close()


def test_real_429_line_raises_rate_limited_fast(tmp_path):
    conv, _ = make_conv(tmp_path, [[{"stderr": SAMPLE_429}, {"silent": 10}]])
    try:
        t = time.monotonic()
        with pytest.raises(RateLimited) as info:
            conv.run([SYS, U1], TOOLS, "m")
        assert time.monotonic() - t < 5
        assert "Resource has been exhausted" not in str(info.value) or "429" in str(info.value)
    finally:
        conv.close()


def test_facade_tool_calls_stream_and_cancel(tmp_path, monkeypatch):
    log = tmp_path / "agy.log"
    behaviour = json.dumps({"script": [[{"call": {"name": "fetch", "arguments": {"k": "a"}}}, {"text": "done"}]],
                            "log": str(log)})
    client = AntigravityAcpMcpClient(acp_command=sys.executable, acp_args=[FAKE, behaviour])
    msgs = [SYS, {"role": "user", "content": "facade turn " + tmp_path.name}]
    c1 = client.chat.completions.create(model="m", messages=msgs, tools=TOOLS)
    tc = c1.choices[0].message.tool_calls
    assert c1.choices[0].finish_reason == "tool_calls" and json.loads(tc[0].function.arguments) == {"k": "a"}
    msgs += [{"role": "assistant", "content": "", "tool_calls": [
        {"id": tc[0].id, "type": "function", "function": {"name": "fetch", "arguments": tc[0].function.arguments}}]},
        {"role": "tool", "tool_call_id": tc[0].id, "content": "A"}]
    chunks = client.chat.completions.create(model="m", messages=msgs, tools=TOOLS, stream=True)
    assert chunks[0].choices[0].delta.content == "done"
    assert [e["n"] for e in events(log) if e["ev"] == "prompt"] == [0]
    client.cancel()  # nothing in flight: no-op
    client.close()


def test_cancel_before_run_starts(tmp_path):
    conv, log = make_conv(tmp_path, [[{"text": "never"}]])
    try:
        token = threading.Event()
        token.set()
        with pytest.raises(CancelledByUser):
            conv.run([SYS, U1], TOOLS, "m", token)
        assert not [e for e in events(log) if e["ev"] == "prompt"]
    finally:
        conv.close()


def test_rate_limit_is_not_sticky_next_request_uses_a_fresh_child(tmp_path):
    conv, _ = make_conv(tmp_path, [[{"stderr": SAMPLE_429}, {"silent": 10}], [{"text": "back"}]])
    try:
        with pytest.raises(RateLimited):
            conv.run([SYS, U1], TOOLS, "m")
        r = conv.run([SYS, U1, {"role": "user", "content": "again"}], TOOLS, "m")
        assert r["text"] == "back"
    finally:
        conv.close()


def test_straggler_from_retired_generation_is_refused():
    from antigravity_acp_mcp.parked_mcp import ParkedCall
    conv = Conversation("k", lambda key, cwd: None)
    try:
        conv.gen = 3
        call = ParkedCall("fetch", {"k": "a"})
        conv._on_call(2, call)
        assert call.state != "parked" and call.is_error and conv.events.empty()
    finally:
        conv.close()


def test_guest_while_in_flight_does_not_touch_the_owned_turn(tmp_path):
    from antigravity_acp_mcp.matcher import BTW_FORK_MARKER
    conv, log = make_conv(tmp_path, [[{"silent": 2}, {"text": "main done"}]])
    try:
        out = {}
        th = threading.Thread(target=lambda: out.update(r=conv.run([SYS, U1], TOOLS, "m")))
        th.start()
        time.sleep(0.8)
        fork = [SYS, U1, {"role": "user", "content": BTW_FORK_MARKER + " ... what is a?"}]
        assert conv.run(fork, [], "m") == {"guest": True}
        th.join(10)
        assert out["r"]["text"] == "main done"
        assert not any(e["ev"] == "cancel" for e in events(log))
    finally:
        conv.close()


def test_native_tool_update_trips_and_cancels(tmp_path):
    from antigravity_acp_mcp.conversation import ProviderError
    conv, log = make_conv(tmp_path, [[{"foreign": True}, {"silent": 10}]])
    try:
        with pytest.raises(ProviderError, match="native tool"):
            conv.run([SYS, U1], TOOLS, "m")
        assert any(e["ev"] == "cancel" for e in events(log))
    finally:
        conv.close()


def test_model_change_forces_a_fresh_session(tmp_path):
    conv, log = make_conv(tmp_path, [[{"text": "one"}], [{"text": "two"}]])
    try:
        msgs = [SYS, U1]
        conv.run(msgs, TOOLS, "m1")
        msgs += [{"role": "assistant", "content": "one"}, {"role": "user", "content": "next"}]
        conv.run(msgs, TOOLS, "m2")
        assert len({e["sid"] for e in events(log) if e["ev"] == "prompt"}) == 2
    finally:
        conv.close()


def test_guard_loop_stops_with_a_visible_message_and_guard_clears_after_end(tmp_path):
    from antigravity_acp_mcp.conversation import GUARD_LOOP_TEXT
    a = {"call": {"name": "fetch", "arguments": {"k": "a"}}}
    conv, _ = make_conv(tmp_path, [[a], [a, a, {"silent": 5}]], call_deadline=0.5)
    try:
        msgs = [SYS, U1]
        r1 = conv.run(msgs, TOOLS, "m")
        time.sleep(1.2)  # deadline -> retired
        msgs += [asst(r1["tool_calls"]), *tool_rows(r1["tool_calls"], ["A"])]
        r2 = conv.run(msgs, TOOLS, "m")
        assert r2["text"] == GUARD_LOOP_TEXT and r2["tool_calls"] == []
    finally:
        conv.close()


def test_cancel_while_the_session_is_opening_returns_within_5s(tmp_path):
    conv, _ = make_conv(tmp_path, [[{"text": "late"}]], new_delay=12)
    try:
        err, token = {}, threading.Event()

        def go():
            try:
                conv.run([SYS, U1], TOOLS, "m", token)
            except Exception as exc:  # noqa: BLE001
                err["e"] = exc

        th = threading.Thread(target=go)
        th.start()
        time.sleep(1.0)
        t = time.monotonic()
        conv.cancel(token)
        th.join(6)
        assert not th.is_alive() and time.monotonic() - t < 5
        assert isinstance(err.get("e"), CancelledByUser)
    finally:
        conv.close()


def test_btw_guest_while_the_owner_is_opening_is_a_guest(tmp_path):
    from antigravity_acp_mcp.matcher import BTW_FORK_MARKER
    conv, log = make_conv(tmp_path, [[{"text": "main done"}]], new_delay=2)
    try:
        out = {}
        th = threading.Thread(target=lambda: out.update(r=conv.run([SYS, U1], TOOLS, "m")))
        th.start()
        time.sleep(0.8)
        fork = [SYS, U1, {"role": "user", "content": BTW_FORK_MARKER + " ... what is a?"}]
        assert conv.run(fork, [], "m") == {"guest": True}
        th.join(10)
        assert out["r"]["text"] == "main done"
        assert not any(e["ev"] == "cancel" for e in events(log))
    finally:
        conv.close()


def test_failed_open_is_a_provider_error_and_not_live(tmp_path):
    from antigravity_acp_mcp.conversation import ProviderError

    def spawn(key, cwd):
        raise FileNotFoundError("no such launcher")

    conv = Conversation("k1", spawn)
    try:
        with pytest.raises(ProviderError, match="could not open"):
            conv.run([SYS, U1], TOOLS, "m")
        assert conv.live is False
    finally:
        conv.close()


def test_child_spawned_after_close_is_reaped(tmp_path):
    started, gate = [], threading.Event()
    log = tmp_path / "agy.log"
    behaviour = json.dumps({"script": [[{"text": "x"}]], "log": str(log)})

    def spawn(key, cwd):
        gate.wait(5)  # still spawning when the conversation is closed (close() removes its cwd)
        conn = AcpConnection(sys.executable, [FAKE, behaviour], str(tmp_path))
        conn.start()
        started.append(conn)
        return conn

    conv = Conversation("", spawn, stateless=True)
    token = threading.Event()
    def go():
        with contextlib.suppress(Exception):
            conv.run([SYS, U1], TOOLS, "m", token)

    th = threading.Thread(target=go)
    th.start()
    time.sleep(0.5)
    conv.cancel(token)
    conv.close()
    gate.set()
    th.join(10)
    deadline = time.monotonic() + 10
    while started and not started[0].exited and time.monotonic() < deadline:
        time.sleep(0.1)
    assert started and started[0].exited


def test_refused_native_attempt_does_not_cancel_the_turn(tmp_path):
    conv, log = make_conv(tmp_path, [[{"refused_native": True}, {"text": "fine"}]])
    try:
        assert conv.run([SYS, U1], TOOLS, "m")["text"] == "fine"
        assert not any(e["ev"] == "cancel" for e in events(log))
    finally:
        conv.close()
