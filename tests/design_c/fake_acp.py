"""Scriptable fake ACP agent for tests/c. argv[1] is a JSON behaviour dict:

early_update       send a session/update for the new session BEFORE the session/new reply
config_options     configOptions list returned by session/new
legacy_models      models.availableModels modelIds returned by session/new
prompt_images      initialize declares agentCapabilities.promptCapabilities.image
flood_stderr       bytes of stderr written (blocking) before the initialize reply
stderr_lines       lines written to stderr when exiting
exit_on            method on which to write stderr_lines and exit with exit_code
permission         params dict sent as session/request_permission during session/prompt
agent_request      method sent as an agent->client request during session/prompt
wait_cancel        after the permission request, read until session/cancel arrives
stop_reason        prompt result stopReason (default end_turn)
update_on_prompt   session/update payloads sent during session/prompt
x/dump returns env NAMES (+ ACP_MUX_* values), cwd and every message received.
"""

import json
import os
import sys

B = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
LOG = []


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read():
    line = sys.stdin.readline()
    if not line:
        sys.exit(0)
    msg = json.loads(line)
    LOG.append(msg)
    return msg


def update(sid, payload):
    emit({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": sid, "update": payload}})


def wait_response(rid):
    while (m := read()).get("id") != rid or "method" in m:
        pass
    return m


while True:
    msg = read()
    rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if method is None:
        continue
    if method == B.get("exit_on"):
        sys.stderr.write("\n".join(B.get("stderr_lines", [])) + "\n")
        sys.stderr.flush()
        sys.exit(B.get("exit_code", 3))
    if method == "initialize":
        for _ in range(B.get("flood_stderr", 0) // 100):
            sys.stderr.write("x" * 99 + "\n")  # blocks forever if the client never drains
        sys.stderr.flush()
        caps = {"promptCapabilities": {"image": True}} if B.get("prompt_images") else {}
        emit({"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": 1, "agentCapabilities": caps}})
    elif method == "session/new":
        sid = "sess-1"
        if B.get("early_update"):
            update(sid, {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "early"}})
        result = {"sessionId": sid, "configOptions": B.get("config_options", [])}
        if B.get("legacy_models"):
            result["models"] = {"availableModels": [{"modelId": m} for m in B["legacy_models"]]}
        emit({"jsonrpc": "2.0", "id": rid, "result": result})
    elif method == "session/prompt":
        sid = params["sessionId"]
        for u in B.get("update_on_prompt", []):
            update(sid, u)
        meta = {}
        if B.get("permission"):
            emit({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission",
                  "params": {"sessionId": sid, **B["permission"]}})
            if not B.get("wait_cancel"):
                meta["permissionResponse"] = wait_response(900)
        if B.get("agent_request"):
            emit({"jsonrpc": "2.0", "id": 901, "method": B["agent_request"], "params": {"sessionId": sid}})
            meta["agentRequestResponse"] = wait_response(901)
        if B.get("wait_cancel"):
            while read().get("method") != "session/cancel":
                pass
        reason = "cancelled" if B.get("wait_cancel") else B.get("stop_reason", "end_turn")
        emit({"jsonrpc": "2.0", "id": rid, "result": {"stopReason": reason, "_meta": meta}})
    elif method == "x/dump":
        emit({"jsonrpc": "2.0", "id": rid, "result": {
            "env_names": sorted(os.environ),
            "mux": {k: v for k, v in os.environ.items() if k.startswith("ACP_MUX_")},
            "cwd": os.getcwd(),
            "log": LOG,
        }})
    elif method == "x/error":
        emit({"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": "boom"}})
    elif rid is not None:
        emit({"jsonrpc": "2.0", "id": rid, "result": {}})
