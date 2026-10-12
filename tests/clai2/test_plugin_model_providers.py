"""Plugins running models under their own prefix with `Plugin.get_model_providers`."""

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Generic, TypeVar, cast

import pytest
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import chat
from pydantic_clai2._app import _ModelResolver  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Commands, set_completions
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import ModelProvider, Plugin, PluginHost, SessionStart, SettingsProvider, load_plugin
from pydantic_clai2.plugins.loader import PluginLoader
from pydantic_clai2.ui.menus.model_menu import ModelMenu, model_settings_command
from tests.clai2.menu_script import Script, pick

PromptT = TypeVar('PromptT')

PLUGIN = """
from pydantic_ai.models.test import TestModel
from pydantic_clai2.plugins import ModelProvider, Plugin


class Echo(Plugin):
    def get_model_providers(self):
        return [
            ModelProvider(
                prefix='echo-test',
                resolve=lambda name: TestModel(custom_output_text=f'{PREFIX} resolved {name}'),
                models=('hello',),
            )
        ]
"""


def echo(name: str) -> Model:
    return TestModel(custom_output_text=name)


def provider(prefix: str) -> ModelProvider:
    return ModelProvider(prefix=prefix, resolve=echo)


def test_plugin_declares_a_provider_and_its_prefixed_names() -> None:
    class Echoes(Plugin):
        def get_model_providers(self) -> Sequence[ModelProvider]:
            return (ModelProvider(prefix='echo-test', resolve=echo, models=('fast', 'smart')),)

    loaded = load_plugin(Echoes, PluginHost[None](name='p', console=Console(file=io.StringIO()), settings={}))
    [provider] = loaded.model_providers
    assert provider == ModelProvider(prefix='echo-test', resolve=echo, models=('fast', 'smart'))
    assert provider.names == ('echo-test:fast', 'echo-test:smart')
    model = provider.resolve('fast')
    assert isinstance(model, TestModel) and model.custom_output_text == 'fast'


@pytest.mark.parametrize('prefix', ['', 'Echo', 'echo:x', '1echo', 'echo_test'])
def test_host_rejects_a_malformed_prefix(prefix: str) -> None:
    with pytest.raises(ValueError, match='must start with a lowercase letter, followed by lowercase letters'):
        provider(prefix)


@pytest.mark.parametrize(
    'prefix',
    [
        *('anthropic', 'openai', 'openai-codex', 'github-copilot', 'openrouter', 'vllm'),
        # Aliases `infer_model` accepts that `known_model_names()` has no IDs for.
        *('openai-chat', 'openai-responses', 'azure', 'azure-responses', 'google-cloud', 'deepseek', 'vercel'),
    ],
)
def test_host_rejects_a_prefix_clai_already_runs(prefix: str) -> None:
    with pytest.raises(ValueError, match='provider CLAI already runs'):
        provider(prefix)


def test_host_rejects_a_known_provider_whose_sdk_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_sdk(provider: str) -> type:
        raise ImportError(f'install the {provider} SDK')

    monkeypatch.setattr('pydantic_clai2.plugins.infer_provider_class', missing_sdk)
    with pytest.raises(ValueError, match='provider CLAI already runs'):
        provider('echo-test')


def test_rejects_an_unknown_settings_from() -> None:
    with pytest.raises(ValueError, match=r"settings_from must be one of anthropic, google.*; got 'bedrock'"):
        ModelProvider(prefix='echo-test', resolve=echo, settings_from=cast(SettingsProvider, 'bedrock'))


async def test_plugin_models_take_the_settings_from_providers_controls(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir(parents=True, exist_ok=True)
    (store.plugins_dir / 'claude.py').write_text(
        'from pydantic_ai.models.test import TestModel\n\n'
        'from pydantic_clai2.plugins import ModelProvider, Plugin\n\n\n'
        'class Claude(Plugin):\n'
        '    def get_model_providers(self):\n'
        "        return [ModelProvider(prefix='claude-test', resolve=lambda name: TestModel(), settings_from='anthropic')]\n"
    )
    loader: PluginLoader[None] = PluginLoader(
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
    )
    await loader.load_all()
    model = 'claude-test:claude-fable-5-1'
    assert loader.settings_model(model) == 'anthropic:claude-fable-5-1'
    # Another account takes the same controls.
    assert loader.settings_model('claude-test@work:claude-fable-5-1') == 'anthropic:claude-fable-5-1'
    assert loader.settings_model('claude-test') == 'claude-test'
    assert loader.settings_model('echo-test:x') == 'echo-test:x'

    context = CommandContext(
        settings=Settings(model=None),
        store=store,
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
        settings_model=loader.settings_model,
    )
    store.add_model(name=model)
    direct = Script(lists=[pick('anthropic_effort'), MenuResult(cancelled=True)], choices=[pick('high')], texts=[])
    assert 'Saved anthropic_effort' in await model_settings_command(context, [model], runners=direct.runners)
    picked = Script(
        lists=[pick(model), pick('anthropic_effort'), MenuResult(cancelled=True), MenuResult(cancelled=True)],
        choices=[pick('low')],
        texts=[],
    )
    assert 'Saved anthropic_effort' in await model_settings_command(context, [], runners=picked.runners)
    assert store.model_settings(model) == {'anthropic_effort': 'low'}


async def test_resolver_routes_only_prefixed_names_to_plugins() -> None:
    echo_test = provider('echo-test')
    resolver = _ModelResolver(console=Console(file=io.StringIO()), plugins=lambda: {'echo-test': echo_test})
    assert await resolver.resolve('echo-test') == 'echo-test'
    assert await resolver.resolve('other:x') == 'other:x'
    model = await resolver.resolve('echo-test:fast')
    assert isinstance(model, TestModel) and model.custom_output_text == 'fast'


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
