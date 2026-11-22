"""Tests for session-pinned warm process reuse and native systemPrompt.append."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent.parent / "hermes-plugin" / "antigravity-acp"

FAKE_SERVER = r"""
import itertools
import json
import os
import sys

log_path = sys.argv[1]
counter_path = sys.argv[2]
supports_sys_append = sys.argv[3] == "1"

with open(counter_path, "a", encoding="utf-8") as fh:
    fh.write("spawn\n")

sessions = itertools.count(1)
current_model = "fake-model-b-high"
mode_path = os.path.join(os.path.dirname(log_path), "mode.json")

def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()

def log(row):
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), **row}) + "\n")

def read_mode():
    if os.path.exists(mode_path):
        try:
            return json.loads(open(mode_path, encoding="utf-8").read())
        except Exception:
            return {}
    return {}

for line in sys.stdin:
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    method = msg.get("method")
    rid = msg.get("id")
    params = msg.get("params") or {}
    log({"method": method, "params": params})
    if method == "initialize":
        caps = {"loadSession": True}
        if supports_sys_append:
            caps["_meta"] = {"antigravity": {"systemPrompt": {"append": True}}}
        emit({"jsonrpc": "2.0", "id": rid, "result": {
            "protocolVersion": 1,
            "agentInfo": {"name": "fake-acp", "version": "1.5.2"},
            "agentCapabilities": caps,
            "authMethods": [],
        }})
    elif method == "session/new":
        sid = f"s-{os.getpid()}-{next(sessions)}"
        emit({"jsonrpc": "2.0", "id": rid, "result": {
            "sessionId": sid,
            "models": {
                "currentModelId": current_model,
                "availableModels": [
                    {"modelId": "fake-model-a", "name": "Fake A"},
                    {"modelId": "fake-model-b-high", "name": "Fake B High"},
                ],
            },
        }})
    elif method == "session/resume":
        mode = read_mode()
        if mode.get("fail_resume_once"):
            mode["fail_resume_once"] = False
            open(mode_path, "w", encoding="utf-8").write(json.dumps(mode))
            emit({"jsonrpc": "2.0", "id": rid,
                  "error": {"code": -32002, "message": "Session not found"}})
            continue
        sid = params.get("sessionId") or f"s-{os.getpid()}-{next(sessions)}"
        emit({"jsonrpc": "2.0", "id": rid, "result": {
            "sessionId": sid,
            "models": {
                "currentModelId": "fake-model-b-high",
                "availableModels": [
                    {"modelId": "fake-model-a", "name": "Fake A"},
                    {"modelId": "fake-model-b-high", "name": "Fake B High"},
                ],
            },
        }})
    elif method == "session/set_model":
        current_model = str(params.get("modelId") or current_model)
        emit({"jsonrpc": "2.0", "id": rid, "result": {}})
    elif method == "session/prompt":
        prompt_text = ((params.get("prompt") or [{}])[0].get("text") or "")
        mode = read_mode()
        reply_text = "ok-done"
        if mode.get("tool_on_first") and "Hermes continuation delta" not in prompt_text:
            reply_text = (
                '<tool_call>{"id":"call-1","type":"function",'
                '"function":{"name":"read_state","arguments":"{}"}}</tool_call>'
            )
        emit({"jsonrpc": "2.0", "method": "session/update", "params": {"update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": reply_text},
        }}})
        emit({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": "end_turn"}})
"""


def _tool_schema(name: str = "read_state") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read synthetic state.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def _load_client(
    monkeypatch, tmp_path: Path, *, supports_sys_append: bool, name_suffix: str
):
    monkeypatch.setenv(
        "HERMES_ANTIGRAVITY_ACP_STATE_FILE",
        str(tmp_path / "acp-conversation-state.json"),
    )
    monkeypatch.delenv("HERMES_ANTIGRAVITY_ACP_WORKSPACE", raising=False)
    monkeypatch.delenv("HERMES_ANTIGRAVITY_ACP_TRACE", raising=False)
    monkeypatch.delenv("HERMES_ANTIGRAVITY_ACP_TRACE_FILE", raising=False)
    registered = []

    class FakeProviderProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    providers = types.ModuleType("providers")
    providers.register_provider = registered.append
    providers_base = types.ModuleType("providers.base")
    providers_base.ProviderProfile = FakeProviderProfile
    monkeypatch.setitem(sys.modules, "providers", providers)
    monkeypatch.setitem(sys.modules, "providers.base", providers_base)

    mod_name = f"antigravity_acp_warm_sys_{name_suffix}"
    for key in list(sys.modules):
        if key == mod_name or key.startswith(mod_name + "."):
            monkeypatch.delitem(sys.modules, key, raising=False)
    spec = importlib.util.spec_from_file_location(
        mod_name,
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER.strip() + "\n", encoding="utf-8")
    log_path = tmp_path / "acp_wire.jsonl"
    counter_path = tmp_path / "spawns.txt"
    counter_path.touch()

    client_mod = importlib.import_module(f"{mod_name}.client")
    client = client_mod.AntigravityACPClient(
        acp_command=sys.executable,
        acp_args=[
            str(script),
            str(log_path),
            str(counter_path),
            "1" if supports_sys_append else "0",
        ],
        acp_cwd=str(tmp_path),
    )
    return client_mod, client, log_path, counter_path


def _read_rows(log_path: Path) -> list[dict]:
    return [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]


def test_warm_same_session_reuse_skips_spawn_resume_and_set_model(
    monkeypatch, tmp_path
) -> None:
    client_mod, client, log_path, counter_path = _load_client(
        monkeypatch, tmp_path, supports_sys_append=True, name_suffix="warm_reuse"
    )
    (tmp_path / "mode.json").write_text(json.dumps({"tool_on_first": True}))
    try:
        tools = [_tool_schema()]
        m1 = [
            {"role": "system", "content": "SYS_RULE_XYZ"},
            {"role": "user", "content": "inspect state"},
        ]
        r1 = client.chat.completions.create(
            model="fake-model-a", messages=m1, tools=tools
        )
        call = r1.choices[0].message.tool_calls[0]

        m2 = [
            *m1,
            {
                "role": "assistant",
                "content": r1.choices[0].message.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.function.name,
                "content": '{"ok":true}',
            },
        ]
        r2 = client.chat.completions.create(
            model="fake-model-a", messages=m2, tools=tools
        )
        assert r2.choices[0].message.content == "ok-done"
        assert r2.model == "fake-model-a"
        assert counter_path.read_text(encoding="utf-8").count("spawn") == 1

        methods = [row["method"] for row in _read_rows(log_path)]
        assert methods == [
            "initialize",
            "session/new",
            "session/set_model",
            "session/prompt",
            "session/prompt",
        ]
    finally:
        client_mod._shutdown_warm_pool()


def test_system_prompt_append_moves_system_band_out_of_user_prompt(
    monkeypatch, tmp_path
) -> None:
    client_mod, client, log_path, _ = _load_client(
        monkeypatch, tmp_path, supports_sys_append=True, name_suffix="sys_append"
    )
    try:
        msgs = [
            {"role": "system", "content": "CRITICAL_SOUL_MARKER_881"},
            {"role": "user", "content": "hello user request"},
        ]
        reply = client.chat.completions.create(
            model="fake-model-a", messages=msgs, tools=[_tool_schema()]
        )
        assert reply.choices[0].message.content == "ok-done"

        rows = _read_rows(log_path)
        new_row = next(r for r in rows if r["method"] == "session/new")
        sys_append = new_row["params"]["_meta"]["antigravity"]["systemPrompt"]["append"]
        assert "CRITICAL_SOUL_MARKER_881" in sys_append
        assert "<SYSTEM_INSTRUCTIONS>" in sys_append
        assert "<tool_call>" in sys_append
        assert new_row["params"]["_meta"]["agy"] == {"disabledTools": ["run_command"]}

        prompt_row = next(r for r in rows if r["method"] == "session/prompt")
        wire_user = prompt_row["params"]["prompt"][0]["text"]
        assert "CRITICAL_SOUL_MARKER_881" not in wire_user
        assert "<SYSTEM_INSTRUCTIONS>" not in wire_user
        assert "hello user request" in wire_user
        assert "<ACTIVE_REQUEST>" in wire_user
    finally:
        client_mod._shutdown_warm_pool()


def test_system_prompt_append_legacy_fallback_without_capability(
    monkeypatch, tmp_path
) -> None:
    client_mod, client, log_path, _ = _load_client(
        monkeypatch, tmp_path, supports_sys_append=False, name_suffix="legacy_fallback"
    )
    try:
        msgs = [
            {"role": "system", "content": "CRITICAL_SOUL_MARKER_881"},
            {"role": "user", "content": "hello user request"},
        ]
        client.chat.completions.create(
            model="fake-model-a", messages=msgs, tools=[_tool_schema()]
        )
        rows = _read_rows(log_path)
        new_row = next(r for r in rows if r["method"] == "session/new")
        assert new_row["params"]["_meta"] == {"agy": {"disabledTools": ["run_command"]}}
        wire_user = next(r for r in rows if r["method"] == "session/prompt")["params"]["prompt"][0]["text"]
        assert "CRITICAL_SOUL_MARKER_881" in wire_user
        assert "<SYSTEM_INSTRUCTIONS>" in wire_user
    finally:
        client_mod._shutdown_warm_pool()


def test_changed_system_prompt_or_cold_resume_reattaches_system_prompt_append(
    monkeypatch, tmp_path
) -> None:
    client_mod, client, log_path, counter_path = _load_client(
        monkeypatch, tmp_path, supports_sys_append=True, name_suffix="sys_update"
    )
    try:
        tools = [_tool_schema()]
        m1 = [
            {"role": "system", "content": "SYS_V1"},
            {"role": "user", "content": "turn 1"},
        ]
        r1 = client.chat.completions.create(
            model="fake-model-a", messages=m1, tools=tools
        )
        affinity = client_mod._conversation_affinity(m1, str(tmp_path))
        sid = client_mod.conversation_registry.get(affinity).session_id

        # If system_append_text changes (e.g. policy overlay state change) while
        # continuing the same warm session, session_meta_fingerprint detects it
        # and re-issues session/resume on the same warm process with the new append.
        answer, _ = client._run_prompt(
            "overlay update continuation",
            timeout_seconds=10.0,
            model="fake-model-a",
            affinity_key=affinity,
            resume_session_id=sid,
            system_append_text="SYS_V2_OVERLAY",
        )
        assert answer == "ok-done"
        assert counter_path.read_text(encoding="utf-8").count("spawn") == 1

        rows = _read_rows(log_path)
        resumes = [r for r in rows if r["method"] == "session/resume"]
        assert len(resumes) == 1
        assert (
            resumes[0]["params"]["_meta"]["antigravity"]["systemPrompt"]["append"]
            == "SYS_V2_OVERLAY"
        )

        # Cold restart after pool shutdown must re-send systemPrompt.append on
        # session/resume (server does not persist prior append across resume).
        client_mod._shutdown_warm_pool()
        m2 = [
            *m1,
            {"role": "assistant", "content": r1.choices[0].message.content},
            {"role": "user", "content": "turn 2"},
        ]
        r2 = client.chat.completions.create(
            model="fake-model-a", messages=m2, tools=tools
        )
        assert r2.choices[0].message.content == "ok-done"
        assert counter_path.read_text(encoding="utf-8").count("spawn") == 2
        resumes_after = [r for r in _read_rows(log_path) if r["method"] == "session/resume"]
        assert len(resumes_after) == 2
        assert "SYS_V1" in resumes_after[1]["params"]["_meta"]["antigravity"]["systemPrompt"]["append"]
    finally:
        client_mod._shutdown_warm_pool()


def test_warm_reuse_model_switch_issues_set_model(monkeypatch, tmp_path) -> None:
    client_mod, client, log_path, counter_path = _load_client(
        monkeypatch, tmp_path, supports_sys_append=True, name_suffix="model_switch"
    )
    try:
        tools = [_tool_schema()]
        m1 = [
            {"role": "system", "content": "SYS_SAME"},
            {"role": "user", "content": "turn 1"},
        ]
        client.chat.completions.create(
            model="fake-model-a", messages=m1, tools=tools
        )
        sys_append, _ = client_mod._split_prompt_sections(m1, tools=tools)
        affinity = client_mod._conversation_affinity(m1, str(tmp_path))
        sid = client_mod.conversation_registry.get(affinity).session_id
        # Continuing on the warm same-session process with a different model and
        # identical system_append skips session/resume but still issues set_model.
        client._run_prompt(
            "direct continuation",
            timeout_seconds=10.0,
            model="fake-model-b-high",
            affinity_key=affinity,
            resume_session_id=sid,
            system_append_text=sys_append,
        )
        assert counter_path.read_text(encoding="utf-8").count("spawn") == 1
        rows = _read_rows(log_path)
        assert not [r for r in rows if r["method"] == "session/resume"]
        set_models = [
            r["params"]["modelId"]
            for r in rows
            if r["method"] == "session/set_model"
        ]
        assert set_models == ["fake-model-a", "fake-model-b-high"]
    finally:
        client_mod._shutdown_warm_pool()


def test_resume_miss_replay_preserves_system_prompt_append_separation(
    monkeypatch, tmp_path
) -> None:
    client_mod, client, log_path, _ = _load_client(
        monkeypatch, tmp_path, supports_sys_append=True, name_suffix="resume_miss"
    )
    try:
        tools = [_tool_schema()]
        m1 = [
            {"role": "system", "content": "SYS_RESUME_MISS_MARKER"},
            {"role": "user", "content": "turn 1"},
        ]
        r1 = client.chat.completions.create(
            model="fake-model-a", messages=m1, tools=tools
        )
        # Force cold resume that returns Session not found -> triggers resume_miss replay.
        client_mod._shutdown_warm_pool()
        (tmp_path / "mode.json").write_text(json.dumps({"fail_resume_once": True}))
        m2 = [
            *m1,
            {"role": "assistant", "content": r1.choices[0].message.content},
            {"role": "user", "content": "turn 2 after miss"},
        ]
        r2 = client.chat.completions.create(
            model="fake-model-a", messages=m2, tools=tools
        )
        assert r2.choices[0].message.content == "ok-done"

        rows = _read_rows(log_path)
        new_rows = [r for r in rows if r["method"] == "session/new"]
        assert len(new_rows) == 2
        assert "SYS_RESUME_MISS_MARKER" in new_rows[1]["params"]["_meta"]["antigravity"]["systemPrompt"]["append"]
        last_prompt = [r for r in rows if r["method"] == "session/prompt"][-1]["params"]["prompt"][0]["text"]
        assert "SYS_RESUME_MISS_MARKER" not in last_prompt
        assert "<SYSTEM_INSTRUCTIONS>" not in last_prompt
        assert "turn 2 after miss" in last_prompt
    finally:
        client_mod._shutdown_warm_pool()
