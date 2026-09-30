"""Plugins running models under their own prefix with `PluginHost.model_provider`."""

import io
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Commands, set_completions
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import ModelProvider, PluginHost, SessionStart
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.model_menu import ModelMenu

PromptT = TypeVar('PromptT')

PLUGIN = """
from pydantic_ai.models.test import TestModel
from pydantic_clai2.plugins import PluginHost


def activate(host: PluginHost) -> None:
    host.model_provider(
        'echo-test', lambda name: TestModel(custom_output_text=f'{PREFIX} resolved {name}'), models=('hello',)
    )
"""


def echo(name: str) -> Model:
    return TestModel(custom_output_text=name)


def host() -> PluginHost[None]:
    return PluginHost[None](name='p', console=Console(file=io.StringIO()), settings={})


def test_host_records_a_provider_and_its_prefixed_names() -> None:
    plugin = host()
    provider = plugin.model_provider('echo-test', echo, models=['fast', 'smart'])
    assert plugin.model_providers == [provider]
    assert provider == ModelProvider(prefix='echo-test', resolve=echo, models=('fast', 'smart'))
    assert provider.names == ('echo-test:fast', 'echo-test:smart')


@pytest.mark.parametrize('prefix', ['', 'Echo', 'echo:x', '1echo', 'echo_test'])
def test_host_rejects_a_malformed_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match='lowercase letters, digits, and hyphens'):
        host().model_provider(prefix, echo)


@pytest.mark.parametrize('prefix', ['anthropic', 'openai', 'openai-codex', 'github-copilot', 'openrouter', 'vllm'])
def test_host_rejects_a_prefix_clai_already_runs(prefix: str) -> None:
    with pytest.raises(ValueError, match='provider CLAI already runs'):
        host().model_provider(prefix, echo)


async def test_loader_merges_providers_and_the_later_plugin_wins(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir(parents=True, exist_ok=True)
    (store.plugins_dir / 'a_first.py').write_text(PLUGIN.replace('{PREFIX}', 'first'))
    (store.plugins_dir / 'b_second.py').write_text(PLUGIN.replace('{PREFIX}', 'second'))
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
    )
    await loader.load_all()
    providers = loader.model_providers()
    assert list(providers) == ['echo-test']
    assert loader.model_names() == ['echo-test:hello']
    model = providers['echo-test'].resolve('x')
    assert isinstance(model, TestModel) and model.custom_output_text == 'second resolved x'


def test_completions_and_the_add_model_menu_offer_plugin_models(tmp_path: Path) -> None:
    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
        plugin_models=lambda: ('echo-test:hello',),
    )
    completions = tuple(set_completions(['model', ''], plugin_models=context.plugin_models()))
    assert {'echo-test:', 'echo-test:hello'} <= set(completions)
    assert 'echo-test:hello' not in set_completions(['model', ''])
    menu = ModelMenu(context)
    assert 'echo-test' in menu.providers()
    assert [model.name for model in menu.for_provider('echo-test').models] == ['echo-test:hello']


async def test_shell_runs_a_plugin_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    values = ['/set model echo-test:typed-name', 'hi', '/exit']

    class Prompt(Generic[PromptT]):
        def __init__(self, **kwargs: object) -> None:
            pass

        async def prompt_async(self, label: str, **kwargs: object) -> str:
            return values.pop(0)

    monkeypatch.setattr('pydantic_clai2._app.PromptSession', Prompt)
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir(parents=True, exist_ok=True)
    (store.plugins_dir / 'echo.py').write_text(PLUGIN.replace('{PREFIX}', 'plugin'))
    output = io.StringIO()
    await chat(
        Agent('test'), deps=None, settings=Settings(model='test'), console=Console(file=output, width=200), store=store
    )
    assert 'plugin resolved typed-name' in output.getvalue()
