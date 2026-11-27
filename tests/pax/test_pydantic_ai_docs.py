"""The `pydantic_ai_docs` built-in: off by default, and once enabled, harness `PydanticAIDocs` as a deferred capability.

It reads the session repository's own `docs/` when that has the Pydantic AI layout, and GitHub otherwise.
"""

import functools
import io
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from rich.console import Console

import pydantic_ai
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.pydantic_ai_docs import _toolset as docs_toolset
from pydantic_ai_harness.pydantic_ai_docs._toolset import _PAGES  # pyright: ignore[reportPrivateUsage]
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.builtin_plugins.pydantic_ai_docs import DESCRIPTION
from pydantic_ai_pax.commands import Commands
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import SessionStart
from pydantic_ai_pax.plugins.loader import PluginLoader

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'pydantic_ai_docs')
REMOTE_PAGE = '# Hooks from GitHub\n'


@pytest.fixture(autouse=True)
def no_docs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The capability's default source, whatever the developer's environment points it at."""
    monkeypatch.delenv('PYDANTIC_AI_HARNESS_DOCS_PATH', raising=False)
    monkeypatch.setattr(pydantic_ai, '__version__', '2.76.0')


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[httpx2.Request]]:
    """Answer every docs fetch with `REMOTE_PAGE` in process, recording what was sent."""
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, text=REMOTE_PAGE)

    # Only the docs toolset's `httpx2` gets the mock client: replacing `httpx2.AsyncClient` itself would break the SDKs
    # that subclass it when they are first imported.
    mocked = SimpleNamespace(
        AsyncClient=functools.partial(httpx2.AsyncClient, transport=httpx2.MockTransport(respond)),
        HTTPError=httpx2.HTTPError,
    )
    monkeypatch.setattr(docs_toolset, 'httpx2', mocked)
    yield requests


def repository(root: Path, *, pages: Sequence[str]) -> Path:
    """A Git repository at `root` whose `docs/` holds `pages`, each a line naming itself; returns its `src/` folder."""
    (root / '.git').mkdir(parents=True)
    for page in pages:
        path = root / 'docs' / page
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'# Local {page}\n')
    (root / 'src').mkdir()
    return root / 'src'


def loader(tmp_path: Path, workspace: Path) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    return PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(BUILTIN,),
        workspace=lambda: workspace,
    )


def load_then_read(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load the deferred capability from its catalog line, read the hooks page, then answer with it."""
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        assert [tool.name for tool in info.function_tools] == ['load_capability']
        assert f'\n- pydantic_ai_docs: {DESCRIPTION}\n' in (info.instructions or '')
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'pydantic_ai_docs'})])
    if returns[0].tool_name == 'load_capability':
        assert 'read_pyai_docs' in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[ToolCallPart('read_pyai_docs', {'topic': 'hooks'})])
    return ModelResponse(parts=[TextPart(str(returns[0].content))])


async def read_hooks(tmp_path: Path, workspace: Path) -> str:
    plugins = loader(tmp_path, workspace)
    await plugins.load_all()
    await plugins.enable('pydantic_ai_docs')
    agent = Agent(
        FunctionModel(load_then_read),
        deps_type=type(None),
        capabilities=[LocalWorkspace(workspace), *plugins.capabilities()],
    )
    return (await agent.run('How do hooks work?')).output


async def test_inside_a_pydantic_ai_checkout_it_reads_the_repository_docs(
    tmp_path: Path, sent: list[httpx2.Request]
) -> None:
    """From a folder below the repository root, the root's `docs/` is read and nothing is fetched."""
    workspace = repository(tmp_path / 'pydantic-ai', pages=tuple(_PAGES.values()))

    assert await read_hooks(tmp_path, workspace) == '# Local hooks.md\n'
    assert sent == []


async def test_elsewhere_it_reads_github_at_the_installed_release(tmp_path: Path, sent: list[httpx2.Request]) -> None:
    """A `docs/` with only some of the pages is a project's own, so the page comes from GitHub."""
    workspace = repository(tmp_path / 'project', pages=('agent.md', 'hooks.md'))

    assert await read_hooks(tmp_path, workspace) == REMOTE_PAGE
    assert [str(request.url) for request in sent] == [
        'https://raw.githubusercontent.com/pydantic/pydantic-ai/v2.76.0/docs/hooks.md'
    ]


async def test_disabled_by_default_it_adds_nothing_to_requests(tmp_path: Path, sent: list[httpx2.Request]) -> None:
    plugins = loader(tmp_path, tmp_path)
    await plugins.load_all()
    assert [(entry.name, entry.state) for entry in plugins.entries()] == [('pydantic_ai_docs', 'disabled')]
    assert plugins.capabilities() == []

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools == []
        assert 'read_pyai_docs' not in (info.instructions or '')
        return ModelResponse(parts=[TextPart('no docs')])

    agent = Agent(FunctionModel(answer), deps_type=type(None), capabilities=plugins.capabilities())
    assert (await agent.run('How do hooks work?')).output == 'no docs'
    assert sent == []
