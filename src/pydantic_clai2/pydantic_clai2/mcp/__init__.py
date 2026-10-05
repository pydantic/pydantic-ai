"""Connect MCP servers and manage them with /mcp.

The built-in `mcp` plugin: `/mcp` manages MCP servers the way Code Puppy's `/mcp` does.

Servers live in `mcp.json` in the CLAI config folder, written by the `/mcp install` and `/mcp edit` form.
A repository's `.clai/mcp_servers.json` and Claude Code-style `.mcp.json` load after `/mcp trust accept`. Servers given as plugin
settings (`/plugins add mcp pydantic_clai2.mcp JSON`) still load, read-only.
"""

from collections.abc import Sequence

from pydantic_ai.capabilities import AgentCapability, Toolset
from pydantic_ai.toolsets import DynamicToolset
from pydantic_clai2.commands import Command
from pydantic_clai2.mcp._command import HELP, MCPCommand
from pydantic_clai2.mcp._form import EXAMPLES, ServerForm, edit_form, edit_in_editor, install_form, run_form
from pydantic_clai2.mcp._runtime import MCPServers, ServerEntry, State
from pydantic_clai2.mcp._settings import (
    OAUTH_TIMEOUT,
    HTTPServer,
    MCPSettings,
    RemoteServer,
    Server,
    ServerSettings,
    SSEServer,
    StdioServer,
    http_client,
)
from pydantic_clai2.mcp._store import CLAUDE_MCP_FILE, PROJECT_MCP_FILE, PROJECT_MCP_FILES, MCPStore, UserFile
from pydantic_clai2.mcp._tokens import SignIn, TokenStore, oauth, sign_in
from pydantic_clai2.plugins import Plugin, PluginHost, SessionEnd

__all__ = [
    'CLAUDE_MCP_FILE',
    'EXAMPLES',
    'HELP',
    'OAUTH_TIMEOUT',
    'PROJECT_MCP_FILE',
    'PROJECT_MCP_FILES',
    'HTTPServer',
    'MCPCommand',
    'MCPPlugin',
    'MCPServers',
    'MCPSettings',
    'MCPStore',
    'RemoteServer',
    'SSEServer',
    'Server',
    'ServerEntry',
    'ServerForm',
    'ServerSettings',
    'SignIn',
    'State',
    'StdioServer',
    'TokenStore',
    'UserFile',
    'edit_form',
    'edit_in_editor',
    'http_client',
    'install_form',
    'oauth',
    'run_form',
    'sign_in',
]


class MCPPlugin(Plugin[MCPSettings]):
    """Offer enabled servers to every run; nothing connects until a run or `/mcp start`."""

    def __init__(self, host: PluginHost[None], settings: MCPSettings, *, store: MCPStore | None = None) -> None:
        """`store` defaults to the user's `mcp.json`; tests pass their own."""
        super().__init__(host, settings)
        self.servers = MCPServers(store or MCPStore(), settings.servers)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (Toolset(DynamicToolset(self.servers.toolset, per_run_step=False)),)

    def get_commands(self) -> Sequence[Command]:
        command = MCPCommand(servers=self.servers)
        return (
            Command(
                name='mcp',
                description='Manage MCP servers: install, start, stop, status, logs, and more (/mcp help).',
                handler=command,
                complete=command.complete,
            ),
        )

    async def on_session_end(self, event: SessionEnd) -> None:
        await self.servers.close()
