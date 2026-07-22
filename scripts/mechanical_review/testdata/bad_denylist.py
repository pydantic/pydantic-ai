"""Synthetic production-like snippet for denylist tests."""

from pydantic_ai.ag_ui import something  # noqa: F401

def use_removed():
    load_mcp_servers()  # forbidden
    x = BuiltinToolCallPart()
    return x
