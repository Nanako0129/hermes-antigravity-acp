"""Opt-in plugin contract, hygiene, and registration tests for hermes-plugin/antigravity-acp-mcp/."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path
from typing import Any

from ._support import PLUGIN_DIR, ROOT

MANIFEST = PLUGIN_DIR / "plugin.yaml"
DEFAULT_PLUGIN_DIR = ROOT / "hermes-plugin" / "antigravity-acp"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_public_tree.py"


def _flat_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith((" ", "\t", "-")):
            raise AssertionError(f"nested manifest line is out of contract: {raw!r}")
        key, sep, value = raw.partition(":")
        assert sep, f"manifest line is not a key/value pair: {raw!r}"
        entries[key.strip()] = value.strip()
    return entries


def _load_audit_module() -> types.ModuleType:
    name = "_design_c_audit_public_tree"
    spec = importlib.util.spec_from_file_location(name, AUDIT_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def _load_isolated_plugin(package_name: str, plugin_dir: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        package_name,
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def test_opt_in_manifest_is_flat_valid_and_provider_only() -> None:
    entries = _flat_manifest(MANIFEST)
    assert entries["name"] == "antigravity-acp-mcp-provider"
    assert entries["kind"] == "model-provider"
    assert re.fullmatch(r"\d+\.\d+\.\d+", entries["version"].strip("\"'"))
    assert entries["description"]
    assert entries["author"]
    assert entries["requires_hermes"].strip("\"'") == ">=0.21.6"
    for key in (
        "provides_tools",
        "provides_hooks",
        "provides_middleware",
        "requires_env",
        "python_dependencies",
    ):
        assert entries.get(key) == "[]", f"{key} must stay [] for provider-only opt-in plugin"


def test_opt_in_plugin_registers_only_provider() -> None:
    registered: set[str] = set()
    for source in sorted(PLUGIN_DIR.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if isinstance(name, str) and name.startswith("register_"):
                registered.add(name)
    assert registered == {"register_provider"}


def test_opt_in_plugin_env_reads_stay_within_declared_surface() -> None:
    allowed_prefixes = ("HERMES_ANTIGRAVITY_ACP_",)
    allowed_exact = {
        "HERMES_SESSION_ID",
        "HERMES_SESSION_KEY",
        "HERMES_CRON_SESSION",
        "HERMES_HOME",
        "HOME",
    }
    offenders: set[str] = set()
    for source in sorted(PLUGIN_DIR.glob("*.py")):
        text = source.read_text(encoding="utf-8")
        for literal in re.findall(r"""environ\.get\(\s*["']([A-Z0-9_]+)["']""", text):
            if not (literal.startswith(allowed_prefixes) or literal in allowed_exact):
                offenders.add(f"{source.name}:{literal}")
    assert not offenders, f"undeclared environment reads: {sorted(offenders)}"


def test_opt_in_plugin_tree_passes_public_hygiene_scan() -> None:
    audit = _load_audit_module()
    forbidden_mux = "antigravity-acp-" + "file-mux"
    files = [
        path
        for path in sorted(PLUGIN_DIR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.relative_to(PLUGIN_DIR).parts
    ]
    assert len(files) == 12
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert audit.scan_text(text) == [], f"secret rule triggered in {path.name}"
        for token in audit.PERSONAL_OPS_TOKENS:
            assert token not in text, f"personal-ops token in {path.name}"
        assert forbidden_mux not in text, f"personal mux reference in {path.name}"


def test_opt_in_registration_non_collision_precedence_and_effort(monkeypatch) -> None:
    original_pkg = sys.modules.get("antigravity_acp_mcp")
    registered: list[Any] = []

    class StubProviderProfile:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    providers_mod = types.ModuleType("providers")
    providers_mod.register_provider = registered.append  # type: ignore[attr-defined]
    providers_base_mod = types.ModuleType("providers.base")
    providers_base_mod.ProviderProfile = StubProviderProfile  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "providers", providers_mod)
    monkeypatch.setitem(sys.modules, "providers.base", providers_base_mod)

    opt_in_pkg = "_isolated_opt_in_acp_mcp"
    default_pkg = "_isolated_default_acp"
    try:
        _load_isolated_plugin(default_pkg, DEFAULT_PLUGIN_DIR)
        _load_isolated_plugin(opt_in_pkg, PLUGIN_DIR)

        assert sys.modules.get("antigravity_acp_mcp") is original_pkg
        assert len(registered) == 2
        default_profile, opt_in_profile = registered

        assert opt_in_profile.name == "antigravity-acp-mcp"
        assert opt_in_profile.aliases == ("agy-mcp",)
        assert opt_in_profile.base_url == "acp://antigravity-mcp"
        assert opt_in_profile.fallback_models == ()
        assert opt_in_profile.supports_vision is True
        assert opt_in_profile.auth_type == "external_process"

        default_names = {default_profile.name, *default_profile.aliases}
        opt_in_names = {opt_in_profile.name, *opt_in_profile.aliases}
        assert default_names.isdisjoint(opt_in_names)
        assert default_profile.base_url != opt_in_profile.base_url

        # Runtime command precedence: MCP_COMMAND > ACP_COMMAND > profile.process_command,
        # while an explicit command= argument always wins.
        monkeypatch.delenv("HERMES_ANTIGRAVITY_ACP_MCP_COMMAND", raising=False)
        monkeypatch.delenv("HERMES_ANTIGRAVITY_ACP_COMMAND", raising=False)
        opt_in_profile.process_command = "/profile/official-launcher"
        assert opt_in_profile.create_client()._command == "/profile/official-launcher"

        monkeypatch.setenv("HERMES_ANTIGRAVITY_ACP_COMMAND", "/env/shared-launcher")
        assert opt_in_profile.create_client()._command == "/env/shared-launcher"

        monkeypatch.setenv("HERMES_ANTIGRAVITY_ACP_MCP_COMMAND", "/env/mcp-launcher")
        assert opt_in_profile.create_client()._command == "/env/mcp-launcher"

        assert opt_in_profile.create_client(command="/explicit/launcher")._command == "/explicit/launcher"

        # Effort mapping through build_api_kwargs_extras.
        assert opt_in_profile.build_api_kwargs_extras(reasoning_config=None) == ({}, {})
        assert opt_in_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False, "effort": "high"}
        ) == ({}, {"reasoning_effort": "none"})
        assert opt_in_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "xhigh"}
        ) == ({}, {"reasoning_effort": "xhigh"})
        assert opt_in_profile.build_api_kwargs_extras(reasoning_config={}) == ({}, {})
    finally:
        for mod_name in list(sys.modules):
            if (
                mod_name == opt_in_pkg
                or mod_name.startswith(opt_in_pkg + ".")
                or mod_name == default_pkg
                or mod_name.startswith(default_pkg + ".")
            ):
                sys.modules.pop(mod_name, None)


def test_default_official_command_prefers_user_bin_then_path_then_name(monkeypatch, tmp_path) -> None:
    from antigravity_acp_mcp import client as c

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(c.shutil, "which", lambda _name: None)
    launcher_name = "antigravity-acp-official.cmd" if c.os.name == "nt" else "antigravity-acp-official"
    assert c._default_official_command() == launcher_name

    monkeypatch.setattr(c.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    assert c._default_official_command() == f"/usr/local/bin/{launcher_name}"

    local_bin = tmp_path / ".local" / "bin"
    local_bin.mkdir(parents=True)
    launcher = local_bin / launcher_name
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    assert c._default_official_command() == str(launcher)
