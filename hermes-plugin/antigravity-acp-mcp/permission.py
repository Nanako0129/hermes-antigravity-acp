"""Permission gate (PlanC component 3, M2) and the detection-only native-tool tripwire (H1).

The gate allows a ``session/request_permission`` only for an MCP tool call to THIS
session's server and a tool in the active set, and only through an ``allow_once``
option. It never matches on titles and never picks ``allow_always``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Collection
from typing import Any

from .acp_connection import CANCELLED_OUTCOME

log = logging.getLogger(__name__)


def _our_meta(meta: Any, server_name: str, tool_names: Collection[str]) -> bool:
    if not isinstance(meta, dict) or meta.get("is_mcp_tool_call") is not True:
        return False
    mcp = meta.get("mcp")
    return (
        isinstance(mcp, dict)
        and mcp.get("server") == server_name
        and mcp.get("tool") in tool_names
    )


class PermissionGate:
    def __init__(
        self,
        server_name: str,
        tool_names: Collection[str],
        on_foreign_tool_call: Callable[[dict[str, Any]], None] | None = None,
        on_our_tool_completed: Callable[[str], None] | None = None,
    ) -> None:
        self.server_name = server_name
        self.tool_names = frozenset(tool_names)
        self._on_foreign = on_foreign_tool_call
        self._on_our_completed = on_our_tool_completed
        self._lock = threading.Lock()
        self._ours: set[str] = set()  # toolCallIds first seen carrying our _meta
        self._our_tools: dict[str, str] = {}  # toolCallId -> our tool name
        self._completed_ours: set[str] = set()  # toolCallIds already reported completed
        self._titles: dict[str, str] = {}  # foreign toolCallId -> title (for the log line only)

    def decide(self, params: dict[str, Any]) -> dict[str, Any]:
        """Response result for a ``session/request_permission`` request."""
        tool_call = params.get("toolCall") if isinstance(params.get("toolCall"), dict) else {}
        options = [o for o in params.get("options") or [] if isinstance(o, dict)]
        allowed = _our_meta(tool_call.get("_meta"), self.server_name, self.tool_names)
        for kind in ("allow_once", "reject_once") if allowed else ("reject_once",):
            for option in options:
                if option.get("kind") == kind and option.get("optionId"):
                    if kind == "allow_once":
                        call_id = str(tool_call.get("toolCallId") or "")
                        if call_id:
                            tool_name = str(tool_call["_meta"]["mcp"]["tool"])
                            with self._lock:
                                self._ours.add(call_id)
                                self._our_tools[call_id] = tool_name
                    return {"outcome": {"outcome": "selected", "optionId": option["optionId"]}}
        return CANCELLED_OUTCOME

    def observe_update(self, update: dict[str, Any]) -> None:
        """Tripwire, best effort: native tools may emit no ``tool_call`` update at all.

        Fires only when a call that is not ours reaches ``in_progress`` or ``completed`` (it actually runs).
        A disabled native tool the model merely tries arrives as ``pending``/``failed`` (live 2026-11-23:
        ``write_to_file`` attempted and refused) and is only logged: cancelling there cost a full replay.
        """
        kind = update.get("sessionUpdate")
        if kind not in ("tool_call", "tool_call_update"):
            return
        call_id = str(update.get("toolCallId") or "")
        status = str(update.get("status") or "")
        title = str(update.get("title") or "")[:60]

        completed_tool: str | None = None
        foreign = False
        log_title = "-"
        with self._lock:
            if _our_meta(update.get("_meta"), self.server_name, self.tool_names):
                tool_name = str(update["_meta"]["mcp"]["tool"])
                if call_id:
                    self._ours.add(call_id)
                    self._our_tools[call_id] = tool_name
                    if status == "completed" and call_id not in self._completed_ours:
                        self._completed_ours.add(call_id)
                        completed_tool = tool_name
            elif call_id and call_id in self._ours:
                if status == "completed" and call_id not in self._completed_ours:
                    self._completed_ours.add(call_id)
                    completed_tool = self._our_tools.get(call_id)
            else:
                if call_id and title:
                    self._titles[call_id] = title
                log_title = (self._titles.get(call_id) if call_id else title) or "-"
                foreign = True

        if completed_tool and self._on_our_completed:
            self._on_our_completed(completed_tool)
        if foreign:
            log.warning("acp-mcp: native tool_call %s status=%s title=%s", kind, status or "-", log_title)
            if status in ("in_progress", "completed") and self._on_foreign:
                self._on_foreign(update)
