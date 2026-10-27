"""MCP servers with `defer_loading`: offered as on-demand capabilities, connected only once the model loads them."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastmcp import FastMCP
from mcp.types import Implementation
from pydantic import HttpUrl, JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2.mcp import (
    HTTPServer,
    MCPCommand,
    MCPPlugin,
    MCPServers,
    MCPSettings,
    MCPStore,
    ServerEntry,
    ServerForm,
    StdioServer,
    servers_menu,
)
from pydantic_clai2.plugins import LoadedPlugin, PluginHost, collect
from tests.clai2.dialog_script import Back, Pick, ScriptedDialogs, Type

Servers = dict[str, FastMCP]


def in_process(monkeypatch: pytest.MonkeyPatch, servers: Servers) -> None:
    """Connect each configured server to an in-process FastMCP server of the same name."""

    def build(self: MCPServers, entry: ServerEntry) -> MCPToolset[None]:
        return MCPToolset(servers[entry.name], id=f'mcp_{entry.name}')

    monkeypatch.setattr(MCPServers, '_build', build)


def docs_and_lint() -> Servers:
    docs = FastMCP('docs', instructions='Search the team docs.\nCall search first.')
    lint = FastMCP('lint')

    @docs.tool
    def search(query: str) -> str:
        return f'found {query}'

    @lint.tool
    def check() -> str:
        return 'clean'

    return {'docs': docs, 'lint': lint}


@dataclass(frozen=True, kw_only=True)
class Loaded:
    loaded: LoadedPlugin[None]
    servers: MCPServers
    store: MCPStore
    host: PluginHost[None]

    async def run(self, text: str) -> str:
        return await self.loaded.commands.execute_async(text)

    def agent(self, model: 'Scripted') -> Agent[None, str]:
        return Agent(FunctionModel(model), deps_type=type(None), capabilities=self.loaded.capabilities)


def make_plugin(tmp_path: Path) -> Loaded:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    host: PluginHost[None] = PluginHost(name='mcp', console=Console(file=io.StringIO()), settings={})
    plugin = MCPPlugin(host, host.settings(MCPSettings), store=store)
    return Loaded(loaded=collect(plugin), servers=plugin.servers, store=store, host=host)


def server(*, defer_loading: bool = False, description: str | None = None) -> StdioServer:
    """The command is never run: `in_process` connects the server instead."""
    return StdioServer(type='stdio', command='unused', defer_loading=defer_loading, description=description)


class Scripted:
    """A model that records what each request offered and answers from `replies`, then with text."""

    def __init__(self, *replies: ModelResponse) -> None:
        self.replies = list(replies)
        self.tools: list[list[str]] = []
        self.instructions: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tools.append(sorted(tool.name for tool in info.function_tools))
        self.instructions.append(info.instructions or '')
        return self.replies.pop(0) if self.replies else ModelResponse(parts=[TextPart('done')])


def call(tool: str, **args: str) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(tool, args)])


@pytest.mark.parametrize(
    'config',
    [
        {'type': 'stdio', 'command': 'x'},
        {'type': 'http', 'url': 'https://example.com/mcp'},
        {'type': 'sse', 'url': 'https://example.com/sse'},
    ],
)
def test_settings_accept_on_demand_for_every_type(config: dict[str, JsonValue]) -> None:
    eager = MCPSettings.model_validate({'servers': {'a': config}}).servers['a']
    assert not eager.defer_loading and eager.description is None, 'eager stays the default'
    on_demand = {**config, 'defer_loading': True, 'description': 'Docs.'}
    parsed = MCPSettings.model_validate({'servers': {'a': on_demand}}).servers['a']
    assert parsed.defer_loading and parsed.description == 'Docs.'
    with pytest.raises(ValidationError):
        MCPSettings.model_validate({'servers': {'a': {**config, 'description': ''}}})


async def test_eager_and_on_demand_servers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    in_process(monkeypatch, docs_and_lint())
    plugin = make_plugin(tmp_path)
    plugin.store.put('docs', server(defer_loading=True, description='Team docs search.'))
    plugin.store.put('lint', server())
    servers, run = plugin.servers, plugin.run
    docs, lint = servers.get('docs'), servers.get('lint')

    unused = Scripted()
    await plugin.agent(unused).run('Hi.')
    assert unused.tools == [['lint_check', 'load_capability']], "an on-demand server's tools stay out of the prompt"
    assert '- mcp_docs: Team docs search.' in unused.instructions[0]
    assert (servers.state(lint), servers.state(docs)) == ('running', 'ready'), 'nothing connects before a load'
    assert 'on demand (not loaded)' in await run('/mcp')

    both = ModelResponse(parts=[ToolCallPart('docs_search', {'query': 'setup'}), ToolCallPart('lint_check', {})])
    model = Scripted(call('load_capability', id='mcp_docs'), both)
    result = await plugin.agent(model).run('Search the docs.')
    assert model.tools[0] == ['lint_check', 'load_capability']
    assert model.tools[1] == ['docs_search', 'lint_check', 'load_capability'], 'loaded tools keep the name prefix'
    returns = {
        part.tool_name: part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert returns['docs_search'] == 'found setup' and returns['lint_check'] == 'clean'
    assert servers.state(docs) == 'running'
    assert servers.tools(docs) == ('docs_search',)

    plugin.host.conversation.replace_messages(result.all_messages())
    dashboard = await run('/mcp')
    assert '1 tools, on demand (loaded)' in dashboard and 'on demand' not in dashboard.split('lint')[1].split('\n')[0]
    details = await run('/mcp status docs')
    assert 'loading  on demand (loaded)' in details and 'about    Team docs search.' in details
    assert 'loading  eager (tools in every prompt)' in await run('/mcp status lint')
    assert 'about' not in await run('/mcp status lint')
    assert await run('/mcp tools docs') == 'docs_search', '/mcp tools lists an on-demand server too'

    resumed = Scripted()
    await plugin.agent(resumed).run('Again.', message_history=result.all_messages())
    assert resumed.tools == [['docs_search', 'lint_check', 'load_capability']], 'a resumed conversation keeps it loaded'

    plugin.host.conversation.replace_messages([])
    assert 'on demand (not loaded)' in await run('/mcp status docs'), 'loading belongs to the conversation'


async def test_a_loaded_server_that_cannot_connect_does_not_fail_the_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_process(monkeypatch, {'broken': FastMCP('broken')})

    async def refuse(self: MCPToolset[None]) -> MCPToolset[None]:
        raise ConnectionError('refused')

    monkeypatch.setattr(MCPToolset, '__aenter__', refuse)
    plugin = make_plugin(tmp_path)
    plugin.store.put('broken', server(defer_loading=True))
    model = Scripted(call('load_capability', id='mcp_broken'))
    result = await plugin.agent(model).run('Load it.')
    assert result.output == 'done'
    assert model.tools[1] == ['load_capability']
    assert '! broken' in await plugin.run('/mcp')


async def test_description_falls_back_to_the_server_then_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    servers_by_name = docs_and_lint() | {
        'long': FastMCP('long', instructions='  \n' + 'word ' * 100),
        'titled': FastMCP('titled'),
    }
    in_process(monkeypatch, servers_by_name)
    plugin = make_plugin(tmp_path)
    for name in servers_by_name:
        plugin.store.put(name, server(defer_loading=True))
    servers = plugin.servers
    assert servers.description(servers.get('docs')) == 'Tools from the docs MCP server.', (
        'nothing to ask before connecting'
    )

    assert 'Started docs' in await plugin.run('/mcp start docs')
    assert servers.description(servers.get('docs')) == 'Search the team docs.', "the first line of the server's own"
    await plugin.run('/mcp tools long')
    long = servers.description(servers.get('long'))
    assert long.endswith('...') and len(long) <= 200
    await plugin.run('/mcp start lint')
    assert servers.description(servers.get('lint')) == 'Tools from the lint MCP server.', 'no instructions, no title'

    monkeypatch.setattr(
        MCPToolset, 'server_info', property(lambda self: Implementation(name='t', title='Team tracker', version='1'))
    )
    await plugin.run('/mcp tools titled')
    assert servers.description(servers.get('titled')) == 'Team tracker'

    model = Scripted()
    await plugin.agent(model).run('Hi.')
    assert '- mcp_docs: Search the team docs.' in model.instructions[0]
    assert '- mcp_titled: Team tracker' in model.instructions[0]

    def unavailable(self: MCPToolset[None]) -> Implementation:
        raise AttributeError('this server did not send implementation info.')

    monkeypatch.setattr(MCPToolset, 'server_info', property(unavailable))
    await plugin.run('/mcp tools titled')
    assert servers.description(servers.get('titled')) == 'Tools from the titled MCP server.'


async def test_changing_on_demand_settings_keeps_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    in_process(monkeypatch, docs_and_lint())
    plugin = make_plugin(tmp_path)
    plugin.store.put('docs', server())
    assert 'Started docs' in await plugin.run('/mcp start docs')
    plugin.store.put('docs', server(defer_loading=True, description='Docs.'))
    await plugin.servers.sync()
    assert plugin.servers.state(plugin.servers.get('docs')) == 'running'
    await plugin.servers.close()


def test_form_toggles_on_demand_and_its_description(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config')
    form = ServerForm(store)
    assert 'Turn on On demand' in form.preview()
    assert 'Description: (from the server)' not in [option.label for option in form.options()]
    form.toggle_on_demand()
    labels = [option.label for option in form.options()]
    assert 'On demand: on' in labels and 'Description: (from the server)' in labels
    assert json.loads(form.config)['defer_loading'] is True and 'until the agent loads them' in form.preview()
    form.set_description('  Team docs.  ')
    assert form.description == 'Team docs.' and 'Description: Team docs.' in [option.label for option in form.options()]
    form.set_description('')
    assert 'description' not in json.loads(form.config), "empty uses the server's own"
    form.toggle_on_demand()
    assert 'defer_loading' not in json.loads(form.config), 'off is the default, so the JSON leaves it out'
    form.config = 'broken'
    form.toggle_on_demand()
    assert form.status == 'Fix the JSON before changing on-demand loading' and not form.on_demand
    assert form.description == ''


async def test_install_and_edit_round_trip_on_demand(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path / 'repo')
    (tmp_path / 'repo' / '.git').mkdir(parents=True)
    command = MCPCommand(servers=MCPServers(store))
    dialogs = ScriptedDialogs(
        [
            Pick(value='type'),
            Pick(value='http'),
            Pick(value='target'),
            Type(text='https://mcp.example.com/mcp'),
            Pick(value='on_demand'),
            Pick(value='description'),
            Type(text='Issue tracker: search and update tickets.'),
            Pick(value='name'),
            Type(text='tracker'),
            Pick(value='save'),
        ]
    )
    assert (await command.run(['install'], dialogs=dialogs, editor=None)).startswith('Added tracker.')
    saved = store.load().servers['tracker']
    assert isinstance(saved, HTTPServer) and saved.url == HttpUrl('https://mcp.example.com/mcp')
    assert saved.defer_loading and saved.description == 'Issue tracker: search and update tickets.'
    assert json.loads(store.path.read_text())['servers']['tracker']['defer_loading'] is True

    edit = ScriptedDialogs([Pick(value='description'), Back(), Pick(value='on_demand'), Pick(value='save')])
    assert (await command.run(['edit', 'tracker'], dialogs=edit, editor=None)).startswith('Updated tracker.')
    assert '"defer_loading": true' in edit.asked[0].description, 'the edit form starts from the saved JSON'
    edited = store.load().servers['tracker']
    assert not edited.defer_loading and edited.description == 'Issue tracker: search and update tickets.'


async def test_menu_rows_show_the_on_demand_state(tmp_path: Path) -> None:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path / 'repo')
    (tmp_path / 'repo' / '.git').mkdir(parents=True)
    menu = servers_menu(MCPCommand(servers=MCPServers(store)))
    store.put('docs', server(defer_loading=True))
    store.put('off', server(defer_loading=True).model_copy(update={'enabled': False}))
    notes = {row.key: row.note for row in menu.source.rows()}
    assert notes['server.docs'] == 'on demand (not loaded)'
    assert notes['server.off'] == 'Off until you start it'
