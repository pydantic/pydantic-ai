"""MCP server configuration loading utilities."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pydantic_core
from pydantic import BaseModel, Field
from pydantic.alias_generators import to_camel

from pydantic_ai.builtin_tools import MCPServerTool

__all__ = ('load_mcp_server_tools',)

# Pattern for environment variable expansion: ${VAR} or ${VAR:-default}
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}:]+)(:-([^}]*))?\}')


class _MCPServerEntry(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Configuration entry for a single MCP server."""

    url: str
    authorization_token: str | None = None
    description: str | None = None
    allowed_tools: list[str] | None = None
    headers: dict[str, str] | None = None


class _MCPServerToolConfig(BaseModel, alias_generator=to_camel, populate_by_name=True):
    """Configuration for loading MCP server tools from JSON."""

    mcp_servers: dict[str, _MCPServerEntry] = Field(alias='mcpServers')


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in a JSON structure.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
    """
    if isinstance(value, str):

        def replace_match(match: re.Match[str]) -> str:
            var_name = match.group(1)
            has_default = match.group(2) is not None
            default_value = match.group(3) if has_default else None

            if var_name in os.environ:
                return os.environ[var_name]
            elif has_default:
                return default_value or ''
            else:
                raise ValueError(f'Environment variable ${{{var_name}}} is not defined')

        return _ENV_VAR_PATTERN.sub(replace_match, value)
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    else:
        return value


def load_mcp_server_tools(config_path: str) -> list[MCPServerTool]:
    """Load MCPServerTool instances from a JSON config file.

    The JSON file should have the following structure:
    {
      "mcpServers": {
        "server-id": {
          "url": "https://example.com/mcp",
          "authorizationToken": "${TOKEN}",  // optional, supports env vars
          "description": "...",              // optional
          "allowedTools": ["tool1"],         // optional
          "headers": {"X-Key": "value"}      // optional
        }
      }
    }

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        List of MCPServerTool instances.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValidationError: If the JSON doesn't match the expected schema.
        ValueError: If an environment variable is not defined.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f'MCP config file not found: {config_path}')

    config_data = pydantic_core.from_json(path.read_bytes())
    expanded_data = _expand_env_vars(config_data)
    config = _MCPServerToolConfig.model_validate(expanded_data)

    tools: list[MCPServerTool] = []
    for server_id, entry in config.mcp_servers.items():
        tools.append(
            MCPServerTool(
                id=server_id,
                url=entry.url,
                authorization_token=entry.authorization_token,
                description=entry.description,
                allowed_tools=entry.allowed_tools,
                headers=entry.headers,
            )
        )
    return tools
