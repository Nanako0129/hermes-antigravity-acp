"""Hermes model-provider registration (imported only by the built plugin's ``__init__.py``, never by tests)."""

from __future__ import annotations

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

from .client import _configured_command, _default_official_command


class AntigravityAcpMcpProfile(ProviderProfile):
    def create_client(self, **client_kwargs: Any) -> Any:
        from .client import AntigravityAcpMcpClient

        if not client_kwargs.get("command"):
            client_kwargs["command"] = _configured_command() or self.process_command
        if not client_kwargs.get("args"):
            client_kwargs["args"] = list(self.process_args)
        if not client_kwargs.get("base_url"):
            client_kwargs["base_url"] = self.base_url
        return AntigravityAcpMcpClient(**client_kwargs)

    def build_api_kwargs_extras(self, *, reasoning_config: dict | None = None, **_: Any) -> tuple[dict, dict]:
        """Hermes' reasoning effort reaches the client as ``reasoning_effort`` (mapped to agy's thinking level)."""
        if not isinstance(reasoning_config, dict):
            return {}, {}
        effort = "none" if reasoning_config.get("enabled") is False else str(reasoning_config.get("effort") or "")
        return {}, ({"reasoning_effort": effort} if effort else {})


antigravity_acp_mcp = AntigravityAcpMcpProfile(
    name="antigravity-acp-mcp",
    aliases=("agy-mcp",),
    display_name="Antigravity ACP (MCP tools)",
    description="Google Antigravity via the official ACP server; Hermes tools over a parked loopback MCP endpoint",
    api_mode="chat_completions",
    env_vars=(),
    base_url="acp://antigravity-mcp",
    auth_type="external_process",
    supports_vision=True,  # images go to agy as ACP image blocks / MCP image content
    process_command=_default_official_command(),
    process_args=(),
    process_command_env_vars=("HERMES_ANTIGRAVITY_ACP_MCP_COMMAND", "HERMES_ANTIGRAVITY_ACP_COMMAND"),
    process_args_env_var="HERMES_ANTIGRAVITY_ACP_MCP_ARGS",
    fallback_models=(),
)

register_provider(antigravity_acp_mcp)
