import threading
from collections import Counter

from antigravity_acp_mcp.permission import PermissionGate

SERVER = "hermes-deadbeef"
CANCELLED = {"outcome": {"outcome": "cancelled"}}


def meta(server=SERVER, tool="terminal", is_mcp=True):
    return {"is_mcp_tool_call": is_mcp, "mcp": {"server": server, "tool": tool}}


def request(tool_meta, kinds=("allow_always", "allow_once", "reject_once"), title="terminal"):
    return {
        "toolCall": {"toolCallId": "t1", "title": title, "_meta": tool_meta},
        "options": [{"optionId": f"opt-{k}", "name": k, "kind": k} for k in kinds],
    }


def selected(option_id):
    return {"outcome": {"outcome": "selected", "optionId": option_id}}


def gate(**kw):
    return PermissionGate(SERVER, {"terminal", "read_file"}, **kw)


def test_allows_our_mcp_call_with_allow_once_only():
    assert gate().decide(request(meta())) == selected("opt-allow_once")


def test_title_only_match_is_rejected():
    assert gate().decide(request(None, title=f"{SERVER}: terminal")) == selected("opt-reject_once")
    assert gate().decide(request({"mcp": {"server": SERVER, "tool": "terminal"}})) == selected("opt-reject_once")
    assert gate().decide(request(meta(is_mcp="true"))) == selected("opt-reject_once")


def test_foreign_server_and_unknown_tool_are_rejected():
    assert gate().decide(request(meta(server="hermes-00000000"))) == selected("opt-reject_once")
    assert gate().decide(request(meta(tool="run_command"))) == selected("opt-reject_once")


def test_never_picks_allow_always():
    assert gate().decide(request(meta(), kinds=("allow_always", "reject_once"))) == selected("opt-reject_once")
    assert gate().decide(request(meta(), kinds=("allow_always",))) == CANCELLED
    assert gate().decide(request(meta(server="x"), kinds=("allow_always", "allow_once"))) == CANCELLED


def test_tripwire_fires_only_when_a_foreign_call_actually_runs():
    seen = []
    g = gate(on_foreign_tool_call=seen.append)
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "mine", "status": "in_progress", "_meta": meta()})
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "mine", "status": "completed"})
    g.observe_update({"sessionUpdate": "agent_message_chunk"})
    assert seen == []
    # a disabled native tool the model only tries: pending -> failed, never runs (live 2026-11-23)
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "n0", "status": "pending", "title": "write_to_file"})
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "n0", "status": "failed"})
    assert seen == []
    native = {"sessionUpdate": "tool_call", "toolCallId": "n1", "status": "in_progress", "title": "view_file"}
    g.observe_update(native)
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "n2", "status": "completed",
                      "_meta": meta(server="other")})
    assert seen[0] == native and len(seen) == 2


def test_permission_request_registers_the_call_as_ours():
    seen = []
    g = gate(on_foreign_tool_call=seen.append)
    g.decide(request(meta()))
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "t1", "status": "in_progress"})
    assert seen == []


def test_our_tool_completed_callback_on_followup_and_single_frame():
    completed, foreign = [], []
    g = gate(on_foreign_tool_call=foreign.append, on_our_tool_completed=completed.append)
    # 1) _our_meta on tool_call, followed by tool_call_update(completed) without _meta.
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "c1", "status": "in_progress",
                      "_meta": meta(tool="terminal")})
    assert completed == []
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "completed"})
    assert completed == ["terminal"] and foreign == []

    # 2) Registered via decide() then completed via tool_call_update without _meta.
    g.decide(request(meta(tool="read_file")))
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"})
    assert completed == ["terminal", "read_file"] and foreign == []

    # 3) Single-frame _our_meta + completed.
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "s1", "status": "completed",
                      "_meta": meta(tool="read_file")})
    assert completed == ["terminal", "read_file", "read_file"] and foreign == []


def test_duplicate_completed_is_idempotent_and_failed_does_not_fire():
    completed, foreign = [], []
    g = gate(on_foreign_tool_call=foreign.append, on_our_tool_completed=completed.append)
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "c1", "status": "in_progress", "_meta": meta()})
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "completed"})
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "completed"})
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "completed", "_meta": meta()})
    assert completed == ["terminal"] and foreign == []

    # status="failed" never fires on_our_tool_completed.
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "f1", "status": "in_progress", "_meta": meta()})
    g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": "f1", "status": "failed"})
    assert completed == ["terminal"] and foreign == []


def test_empty_call_id_does_not_disable_h1_or_fire_completion():
    completed, foreign = [], []
    g = gate(on_foreign_tool_call=foreign.append, on_our_tool_completed=completed.append)
    # Empty/missing toolCallId with our _meta must not register "" in _ours or fire completion (SEC-M3-6).
    g.decide({
        "toolCall": {"toolCallId": "", "title": "terminal", "_meta": meta()},
        "options": [{"optionId": "opt-allow_once", "name": "allow_once", "kind": "allow_once"}],
    })
    g.observe_update({"sessionUpdate": "tool_call", "toolCallId": "", "status": "completed", "_meta": meta()})
    g.observe_update({"sessionUpdate": "tool_call", "status": "completed", "_meta": meta()})
    assert completed == [] and foreign == []

    # Foreign tool with empty/missing toolCallId still trips H1.
    f1 = {"sessionUpdate": "tool_call", "toolCallId": "", "status": "in_progress", "title": "view_file"}
    f2 = {"sessionUpdate": "tool_call_update", "status": "completed", "title": "run_command"}
    g.observe_update(f1)
    g.observe_update(f2)
    assert foreign == [f1, f2] and completed == []


def test_concurrent_decide_and_observe_update_thread_safety():
    completed, foreign = [], []
    cb_lock = threading.Lock()

    def on_done(tool_name):
        with cb_lock:
            completed.append(tool_name)

    g = gate(on_foreign_tool_call=foreign.append, on_our_tool_completed=on_done)
    n_ids = 120
    barrier = threading.Barrier(4)
    errors = []

    def worker(offset, tool_name):
        try:
            barrier.wait(5)
            for i in range(n_ids):
                cid = f"cid-{offset}-{i}"
                g.decide({
                    "toolCall": {"toolCallId": cid, "title": tool_name, "_meta": meta(tool=tool_name)},
                    "options": [{"optionId": "opt-allow_once", "name": "allow_once", "kind": "allow_once"}],
                })
                g.observe_update({"sessionUpdate": "tool_call", "toolCallId": cid, "status": "in_progress",
                                  "_meta": meta(tool=tool_name)})
                g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": cid, "status": "completed"})
                g.observe_update({"sessionUpdate": "tool_call_update", "toolCallId": cid, "status": "completed"})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(0, "terminal")),
        threading.Thread(target=worker, args=(1, "read_file")),
        threading.Thread(target=worker, args=(2, "terminal")),
        threading.Thread(target=worker, args=(3, "read_file")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == []
    assert foreign == []
    assert Counter(completed) == {"terminal": 2 * n_ids, "read_file": 2 * n_ids}
