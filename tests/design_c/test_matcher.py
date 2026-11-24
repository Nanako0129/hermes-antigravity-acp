"""PlanC S3 matcher table on synthetic Hermes message tails (no user conversation data)."""

from antigravity_acp_mcp.matcher import (
    BTW_FORK_MARKER,
    TurnView,
    canonical_args,
    classify,
    trailing_tool_round,
    view_after,
)

SYS = {"role": "system", "content": "You are Hermes."}
U1 = {"role": "user", "content": "check a and b"}


def call(cid, name="fetch", args='{"k": 1}'):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def asst_calls(*cids):
    return {"role": "assistant", "content": "", "tool_calls": [call(c) for c in cids]}


def tool(cid, text="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": text}


def test_first_request_replays():
    assert classify([SYS, U1], None).kind == "replay"


def test_resume_held_single_and_parallel_any_order():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a", "call_b"), live=True)
    msgs = [*base, asst_calls("call_a", "call_b"), tool("call_b", "B"), tool("call_a", "A")]
    d = classify(msgs, view)
    assert d.kind == "resume_held"
    assert d.tool_results == {"call_a": "A", "call_b": "B"}


def test_dropped_call_without_tool_row_replays():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a", "call_b"), live=True)
    msgs = [*base, asst_calls("call_a", "call_b"), tool("call_a")]  # dedupe/cap dropped call_b's row
    assert classify(msgs, view).kind == "replay"


def test_steer_row_after_tool_row_supersedes_live_prompt():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)
    msgs = [*base, asst_calls("call_a"), tool("call_a"), {"role": "user", "content": "also check c"}]
    assert classify(msgs, view).kind == "supersede"


def test_retry_of_same_create_replays():
    base = [SYS, U1]
    view = view_after(base, live=True)
    assert classify(base, view).kind == "replay"


def test_compression_rewrite_replays():
    base = [SYS, U1, {"role": "assistant", "content": "done"}, {"role": "user", "content": "next"}]
    view = view_after(base[:2])
    rewritten = [SYS, {"role": "user", "content": "[summary of earlier turns]"}, {"role": "user", "content": "next"}]
    assert classify(rewritten, view).kind == "replay"


def test_new_user_turn_after_idle_continues_with_delta_ignoring_assistant_text():
    base = [SYS, U1]
    view = view_after(base)  # prompt ended with text; Hermes stores its own rendering of it
    msgs = [*base, {"role": "assistant", "content": "a and b are fine (transformed)"},
            {"role": "user", "content": "thanks, now c"}]
    d = classify(msgs, view)
    assert d.kind == "continue"
    assert [m["role"] for m in d.delta] == ["user"]


def test_new_user_turn_on_retired_session_replays():
    base = [SYS, U1]
    v = view_after(base)
    view = TurnView(v.consumed, v.prefix_fp, retired=True)
    msgs = [*base, {"role": "assistant", "content": "x"}, {"role": "user", "content": "next"}]
    assert classify(msgs, view).kind == "replay"


def test_detached_tool_rows_after_cancel_replay():
    base = [SYS, U1]
    v = view_after(base, handed_out=("call_a",))
    view = TurnView(v.consumed, v.prefix_fp, handed_out=(), live=False, retired=True)  # deadline cancelled it
    msgs = [*base, asst_calls("call_a"), tool("call_a")]
    assert classify(msgs, view).kind == "replay"


def test_btw_guest_while_live_does_not_touch_owned_turn():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)
    fork = [*base, {"role": "user", "content": BTW_FORK_MARKER + " ...\n\nwhat is a?"}]
    assert classify(fork, view, tools=[]).kind == "guest"
    assert classify(fork, view, tools=[{"type": "function"}]).kind != "guest"  # positive match only


def test_unknown_shape_never_continues():
    base = [SYS, U1]
    view = view_after(base, handed_out=("call_a",), live=True)
    msgs = [*base, asst_calls("call_zz"), tool("call_zz")]
    assert classify(msgs, view).kind == "replay"


def test_trailing_tool_round_seed_and_canonical_args():
    msgs = [SYS, U1, asst_calls("call_a", "call_b"), tool("call_a")]
    assert trailing_tool_round(msgs) == {("fetch", canonical_args('{"k":1}'))}
    assert trailing_tool_round([SYS, U1]) == set()
    assert trailing_tool_round([SYS, U1, asst_calls("c1"), tool("c1"), {"role": "user", "content": "new"}]) == set()
    assert canonical_args('{"b": 2, "a": 1}') == canonical_args({"a": 1, "b": 2})


def test_explain_mismatch_reports_position_and_kind_without_content():
    from antigravity_acp_mcp.matcher import explain_mismatch
    base = [SYS, U1]
    view = view_after(base)
    edited = [SYS, {"role": "user", "content": "check a and b <memory-context>secret recall</memory-context>"},
              {"role": "assistant", "content": "x"}, {"role": "user", "content": "next"}]
    why = explain_mismatch(edited, list(view.seen))
    assert why.startswith("index 1: user text differs") and "secret" not in why
    assert classify(edited, view).kind == "replay"
