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
