"""The `browser` built-in: off by default, and once enabled, a deferred headless `PlaywrightBrowser`.

Nothing here starts a browser or Playwright's driver, or reaches the network: the load-time Chromium check gets a
fake driver, and the browser calls the model makes are ones the egress policy refuses before Chromium would start.
Tests that need the `playwright` extra import it in their body, so the one about a Pax without it runs everywhere.
"""

import io
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path

import pytest
from rich.console import Console

import pydantic_ai_pax.builtin_plugins.browser as browser_module
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.builtin_plugins.browser import DESCRIPTION, install_chromium_command
from pydantic_ai_pax.commands import Commands
from pydantic_ai_pax.config import PluginSettings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import SessionStart
from pydantic_ai_pax.plugins.loader import PluginError, PluginLoader

BUILTIN = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'browser')
needs_playwright = pytest.mark.skipif(find_spec('playwright') is None, reason='needs the `playwright` extra')


@dataclass
class _Chromium:
    executable_path: str


@dataclass
class _Driver:
    chromium: _Chromium


def driver_reporting(executable: Path) -> Callable[[], AbstractAsyncContextManager[_Driver]]:
    """A stand-in for `async_playwright` whose Chromium lives at `executable`, so no driver process starts."""

    @asynccontextmanager
    async def driver() -> AsyncGenerator[_Driver]:
        yield _Driver(_Chromium(str(executable)))

    return driver


@pytest.fixture
def chromium(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An installed Chromium, as the load-time check sees it; delete the file to make it missing.

    The browser tools' own driver fails the test if anything starts it.
    """
    import playwright.async_api

    import pydantic_ai_harness.playwright._toolset as toolset

    executable = tmp_path / 'chrome'
    executable.touch()
    monkeypatch.setattr(playwright.async_api, 'async_playwright', driver_reporting(executable))

    def no_browser() -> None:
        raise AssertionError('the browser tools started Playwright')

    monkeypatch.setattr(toolset, 'async_playwright', no_browser)
    return executable


def loader(tmp_path: Path, output: io.StringIO, *, builtin: PluginSettings = BUILTIN) -> PluginLoader[None]:
    store = SettingsStore(tmp_path / 'config.db')
    return PluginLoader(
        store=store,
        console=Console(file=output, width=400),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(builtin,),
    )


def browse(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Load the browser from its catalog line, try a metadata address and an off-list site, then report back."""
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    if not returns:
        [load] = info.function_tools
        assert load.name == 'load_capability'
        assert f'\n- playwright_browser: {DESCRIPTION}\n' in (info.instructions or '')
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'playwright_browser'})])
    if returns[0].tool_name == 'load_capability':
        assert {'navigate', 'click', 'snapshot'} <= {tool.name for tool in info.function_tools}
        return ModelResponse(
            parts=[
                ToolCallPart('navigate', {'url': 'http://169.254.169.254/latest/meta-data/'}),
                ToolCallPart('navigate', {'url': 'https://elsewhere.test/'}),
            ]
        )
    return ModelResponse(parts=[TextPart('\n'.join(str(part.content) for part in returns))])


@needs_playwright
async def test_enabling_gives_the_model_a_browser_that_refuses_private_and_off_list_addresses(
    tmp_path: Path, chromium: Path
) -> None:
    from pydantic_ai_harness.playwright import PlaywrightBrowser

    output = io.StringIO()
    declared = BUILTIN.model_copy(update={'settings': {'allowed_domains': ['example.com']}})
    plugins = loader(tmp_path, output, builtin=declared)
    await plugins.load_all()
    await plugins.enable('browser')
    [capability] = plugins.capabilities()
    assert isinstance(capability, PlaywrightBrowser)
    assert (capability.headless, capability.block_private_addresses, capability.allowed_domains) == (
        True,
        True,
        ['example.com'],
    )
    assert output.getvalue() == '', 'Chromium is installed, so loading says nothing'
    agent = Agent(FunctionModel(browse), deps_type=type(None), capabilities=plugins.capabilities())

    result = await agent.run('Read the cloud metadata, then elsewhere.test.')

    assert result.output.splitlines() == [
        'Error: blocked private or link-local address: http://169.254.169.254/latest/meta-data/',
        'Error: domain not in allowed_domains: https://elsewhere.test/',
    ]
    await plugins.close('exit')


async def test_disabled_by_default_it_adds_nothing(tmp_path: Path) -> None:
    plugins = loader(tmp_path, io.StringIO())
    await plugins.load_all()
    assert [(entry.name, entry.state) for entry in plugins.entries()] == [('browser', 'disabled')]
    assert plugins.capabilities() == []
    await plugins.close('exit')


async def test_without_the_extra_enabling_names_the_install_command_and_the_launch_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def not_installed(name: str) -> None:
        return None

    monkeypatch.setattr(browser_module, 'find_spec', not_installed)
    # Pax's own interpreter, which has the Playwright whose Chromium build the browser uses.
    assert install_chromium_command() == f'{sys.executable} -m playwright install chromium'
    install = (
        f"Install Pax with it: uv tool install 'pydantic-ai-pax[playwright]', then run {install_chromium_command()}."
    )
    plugins = loader(tmp_path, io.StringIO())
    await plugins.load_all()
    with pytest.raises(PluginError, match='needs Playwright') as raised:
        await plugins.enable('browser')
    assert install in str(raised.value)
    assert plugins.capabilities() == []
    await plugins.close('exit')

    # The enabled declaration is saved, so the next launch tries again: it reports the same and starts.
    output = io.StringIO()
    relaunched = loader(tmp_path, output)
    await relaunched.load_all()
    assert install in output.getvalue()
    assert relaunched.capabilities() == []
    await relaunched.close('exit')


@needs_playwright
async def test_a_missing_chromium_is_reported_when_the_plugin_loads(tmp_path: Path, chromium: Path) -> None:
    chromium.unlink()
    output = io.StringIO()
    plugins = loader(tmp_path, output)
    await plugins.load_all()
    await plugins.enable('browser')
    assert output.getvalue() == f'The browser plugin needs Chromium. Run {install_chromium_command()}\n'
    assert len(plugins.capabilities()) == 1, 'the plugin still loads, and its tools give the same hint'
    await plugins.close('exit')


@needs_playwright
@pytest.mark.parametrize('failure', ['driver error', 'no driver process', 'hangs'])
async def test_a_chromium_check_that_fails_stays_quiet_and_the_plugin_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    import anyio
    import playwright.async_api

    @asynccontextmanager
    async def broken() -> AsyncGenerator[_Driver]:
        if failure == 'driver error':
            raise playwright.async_api.Error('Playwright driver did not start')
        if failure == 'no driver process':
            raise FileNotFoundError('node')
        await anyio.sleep_forever()
        yield _Driver(_Chromium(''))  # pragma: no cover

    monkeypatch.setattr(playwright.async_api, 'async_playwright', broken)
    # A check that never answers is cut off at once rather than after the real deadline.
    monkeypatch.setattr(browser_module, 'CHECK_TIMEOUT', 0)
    output = io.StringIO()
    plugins = loader(tmp_path, output)
    await plugins.load_all()
    await plugins.enable('browser')
    assert output.getvalue() == ''
    assert len(plugins.capabilities()) == 1
    await plugins.close('exit')
