"""The built-in `mcp` plugin: `/mcp` manages MCP servers the way Code Puppy's `/mcp` does.

Servers live in `mcp.json` in the CLAI config folder, written by the `/mcp install` and `/mcp edit` form.
A repository's `.clai/mcp_servers.json` loads after `/mcp trust accept`. Servers given as plugin
settings (`/plugins add mcp pydantic_clai2.mcp JSON`) still load, read-only.
"""

from pydantic_ai.capabilities import Toolset
from pydantic_ai.toolsets import DynamicToolset

from ..commands import Command
from ..plugins import PluginHost, SessionEnd
from ._command import HELP, MCPCommand
from ._form import EXAMPLES, ServerForm, edit_form, edit_in_editor, install_form, run_form
from ._runtime import MCPServers, ServerEntry, State
from ._settings import (
    HTTPServer,
    MCPSettings,
    RemoteServer,
    Server,
    ServerSettings,
    SSEServer,
    StdioServer,
    http_client,
)
from ._store import PROJECT_MCP_FILE, MCPStore, UserFile
from ._tokens import TokenStore, oauth

__all__ = [
    'EXAMPLES',
    'HELP',
    'PROJECT_MCP_FILE',
    'HTTPServer',
    'MCPCommand',
    'MCPServers',
    'MCPSettings',
    'MCPStore',
    'RemoteServer',
    'SSEServer',
    'Server',
    'ServerEntry',
    'ServerForm',
    'ServerSettings',
    'State',
    'StdioServer',
    'TokenStore',
    'UserFile',
    'activate',
    'edit_form',
    'edit_in_editor',
    'http_client',
    'install_form',
    'oauth',
    'run_form',
]


def activate(host: PluginHost[None], *, store: MCPStore | None = None) -> None:
    """Offer enabled servers to every run; nothing connects until a run or `/mcp start`."""
    servers = MCPServers(store or MCPStore(), host.settings(MCPSettings).servers)
    host.add(Toolset(DynamicToolset(servers.toolset, per_run_step=False)))
    command = MCPCommand(servers=servers)

    @host.on('session_end')
    async def release(_: SessionEnd) -> None:  # pyright: ignore[reportUnusedFunction]
        await servers.close()

    host.commands.register(
        Command(
            name='mcp',
            description='Manage MCP servers: install, start, stop, status, logs, and more (/mcp help).',
            handler=command,
            complete=command.complete,
        )
    )
