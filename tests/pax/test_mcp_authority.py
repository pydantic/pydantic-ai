"""A project MCP server has the authority of the file that configures it, which the session's folder trust sets.

Each test runs a Pax session against in-process FastMCP servers, and reads the authority of each server's
capability as the run registered it.
"""

import io
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from fastmcp import FastMCP
from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    Authority,
    CapabilityInstructionSource,
    InstructionDeltaPart,
    InstructionId,
    ModelMessage,
    ModelRequest,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCalls, FunctionModel
from pydantic_ai_pax.config.project_settings import ProjectSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.mcp import MCPPlugin, MCPServers, MCPSettings, MCPStore, ServerEntry, StdioServer
from pydantic_ai_pax.plugins import PluginHost, collect
from pydantic_ai_pax.runtime.session_runtime import create_shell

NOTE = 'Repository server note.'


def connect_in_process(monkeypatch: pytest.MonkeyPatch, *, include_instructions: bool = False) -> None:
    """Connect each server to an in-process FastMCP server, as `in_process` in `test_mcp_on_demand.py` does.

    Pax leaves a server's instructions out of the prompt; `include_instructions=True` sends `repo`'s, to show where
    they go.
    """
    repo = FastMCP('repo', instructions=NOTE)
    mine = FastMCP('mine')

    @repo.tool
    def lookup() -> str:
        return 'found'  # pragma: no cover

    @mine.tool
    def check() -> str:
        return 'clean'  # pragma: no cover

    def build(self: MCPServers, entry: ServerEntry) -> MCPToolset[None]:
        if entry.name == 'repo':
            return MCPToolset(repo, id='mcp_repo', include_instructions=include_instructions)
        return MCPToolset(mine, id='mcp_mine')

    monkeypatch.setattr(MCPServers, '_build', build)


@pytest.fixture(autouse=True)
def in_process(monkeypatch: pytest.MonkeyPatch) -> None:
    connect_in_process(monkeypatch)


class Recorded(AbstractCapability[None]):
    """The authority of each MCP server's capability, and the tools and instructions of each request."""

    def __init__(self) -> None:
        self.authorities: dict[str, Authority] = {}
        self.tools: list[str] = []
        self.instructions: list[str] = []

    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.authorities = {
            capability_id: capability._effective_authority
            for capability_id, capability in ctx.capabilities.items()
            if capability_id.startswith('mcp_')
        }
        return request_context

    async def model(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        self.tools = sorted(tool.name for tool in info.function_tools)
        self.instructions.append(info.instructions or '')
        yield 'done'


REPO_LINE = 'Repository catalog line.'


def project_file(folder: Path, *, eager: bool) -> None:
    """A Claude Code-style `.mcp.json` in `folder` with one server, `repo`, which describes itself."""
    servers = {'repo': {'command': 'unused', 'defer_loading': not eager, 'description': REPO_LINE}}
    (folder / '.mcp.json').write_text(json.dumps({'mcpServers': servers}))


async def run_session(folder: Path, recorded: Recorded, *, trusted: bool | None) -> list[ModelMessage]:
    """One prompt in a Pax session in `folder`, with your eager server `mine` and the accepted project file's `repo`.

    `trusted=None` leaves the folder undecided.
    """
    store = MCPStore(folder / 'config', workspace=folder)
    store.put('mine', StdioServer(type='stdio', command='unused', defer_loading=False))
    store.trust(*store.project_files())
    host: PluginHost[None] = PluginHost(name='mcp', console=Console(file=io.StringIO()), settings={})
    plugin = MCPPlugin(host, host.settings(MCPSettings), store=store)
    settings = SettingsStore(folder / 'config.db')
    if trusted is not None:
        settings.set_workspace_trust(folder.resolve(), trusted=trusted)
    shell = create_shell(
        Agent(FunctionModel(stream_function=recorded.model), deps_type=type(None), output_type=str),
        deps=None,
        plugins=[recorded, *collect(plugin).capabilities],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=settings,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=folder,
    )
    try:
        await shell.session.prompt('Hi.')
        return shell.session.messages
    finally:
        await shell.close('exit')
        await plugin.servers.close()


@pytest.mark.parametrize(
    ('trusted', 'expected'),
    [(True, 'operator'), (False, 'none'), (None, 'none')],
    ids=['trusted', 'untrusted', 'undecided'],
)
async def test_an_on_demand_project_server_has_the_authority_folder_trust_gives_its_file(
    tmp_path: Path, trusted: bool | None, expected: Authority
) -> None:
    """`/mcp trust accept` lets the repository's servers run; the folder's trust decides their authority."""
    project_file(tmp_path, eager=False)
    recorded = Recorded()
    await run_session(tmp_path, recorded, trusted=trusted)
    assert recorded.authorities == {'mcp_repo': expected}
    assert recorded.tools == ['load_capability', 'mine_check']


@pytest.mark.parametrize(('trusted', 'expected'), [(True, None), (False, 'none')], ids=['trusted', 'untrusted'])
async def test_an_eager_project_server_below_operator_is_its_own_lowered_capability(
    tmp_path: Path, trusted: bool, expected: Authority | None
) -> None:
    """At `'operator'` an eager project server is offered with your eager servers, as before; below it, on its own."""
    project_file(tmp_path, eager=True)
    recorded = Recorded()
    await run_session(tmp_path, recorded, trusted=trusted)
    assert recorded.authorities == ({} if expected is None else {'mcp_repo': expected})
    assert recorded.tools == ['mine_check', 'repo_lookup']


async def test_an_untrusted_project_servers_instructions_are_stated_as_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: below `'operator'`, core states the server's instructions in history, never in the instructions."""
    connect_in_process(monkeypatch, include_instructions=True)
    project_file(tmp_path, eager=True)
    recorded = Recorded()
    messages = await run_session(tmp_path, recorded, trusted=False)
    stated = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, InstructionDeltaPart)
    ]
    assert stated == [
        InstructionDeltaPart(id=InstructionId(CapabilityInstructionSource('mcp_repo')), content=NOTE, authority='none')
    ]
    assert all(NOTE not in instructions for instructions in recorded.instructions)


@pytest.mark.parametrize(('trusted', 'shown'), [(True, True), (False, False)], ids=['trusted', 'untrusted'])
async def test_an_untrusted_on_demand_servers_catalog_line_is_only_its_name(
    tmp_path: Path, trusted: bool, shown: bool
) -> None:
    """The `load_capability` catalog is in the instructions, so below `'operator'` it carries none of the repository's text."""
    project_file(tmp_path, eager=False)
    recorded = Recorded()
    await run_session(tmp_path, recorded, trusted=trusted)
    (instructions,) = recorded.instructions
    assert (REPO_LINE in instructions) is shown
    assert ('- mcp_repo: Tools from the repo MCP server.' in instructions) is not shown
