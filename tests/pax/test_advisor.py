"""The `advisor` built-in: harness `Advisor` on a model Pax resolves through its own model services."""

import io
from pathlib import Path

import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.builtin_plugins.advisor import NO_MODEL, AdvisorPlugin, AdvisorSettings
from pydantic_ai_pax.commands import Commands
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.models.resolver import ModelResolver
from pydantic_ai_pax.plugins import ModelProvider, PluginHost, SessionStart, load_plugin
from pydantic_ai_pax.plugins.loader import PluginLoader

[DECLARATION] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'advisor']


def loader(store: SettingsStore, console: Console, *, resolve_model: ModelResolver | None = None) -> PluginLoader[None]:
    return PluginLoader(
        store=store,
        console=console,
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[DECLARATION],
        resolve_model=resolve_model.resolve if resolve_model is not None else None,
    )


def consulting(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """The main model: consult the advisor once, then answer with what it said."""
    returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
    usage = RequestUsage(input_tokens=10, output_tokens=5)
    if not returns:
        assert 'advisor' in [tool.name for tool in info.function_tools]
        return ModelResponse(parts=[ToolCallPart('advisor', {'prompt': 'Rewrite or patch?'})], usage=usage)
    return ModelResponse(parts=[TextPart(f'Following advice: {returns[0].content}')], usage=usage)


def test_declared_as_a_disabled_builtin() -> None:
    assert DECLARATION.factory == 'pydantic_ai_pax.builtin_plugins.advisor'
    assert not DECLARATION.enabled


async def test_disabled_it_adds_nothing(tmp_path: Path) -> None:
    output = io.StringIO()
    plugins = loader(SettingsStore(tmp_path / 'settings.db'), Console(file=output))
    try:
        await plugins.load_all()
        assert [entry.state for entry in plugins.entries()] == ['disabled']
        assert plugins.capabilities() == []
        assert output.getvalue() == ''
    finally:
        await plugins.close('exit')


async def test_enabled_without_a_model_says_what_to_set(tmp_path: Path) -> None:
    output = io.StringIO()
    plugins = loader(SettingsStore(tmp_path / 'settings.db'), Console(file=output, width=200))
    try:
        await plugins.load_all()
        await plugins.enable('advisor')
        assert [entry.state for entry in plugins.entries()] == ['enabled, loaded']
        assert plugins.capabilities() == [], 'no advisor tool without a model'
        assert NO_MODEL in output.getvalue()
        assert '/plugins configure advisor' in NO_MODEL
    finally:
        await plugins.close('exit')


async def test_the_model_consults_an_advisor_built_by_pax(tmp_path: Path) -> None:
    """The advisor model comes from Pax's resolver, here a plugin model prefix, and its request counts in the run."""
    advised: list[object] = []
    resolved: list[str] = []

    def advise(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        [request] = messages
        assert isinstance(request, ModelRequest)
        advised.extend(part.content for part in request.parts if isinstance(part, UserPromptPart))
        assert info.function_tools == [], 'the advisor gets none of the executor tools'
        return ModelResponse(parts=[TextPart('Patch it.')], usage=RequestUsage(input_tokens=1000, output_tokens=700))

    def build(name: str) -> Model:
        resolved.append(name)
        return FunctionModel(advise, model_name=f'fixture:{name}')

    store = SettingsStore(tmp_path / 'settings.db')
    store.save_plugin(DECLARATION.model_copy(update={'enabled': True, 'settings': {'model': 'fixture:wise'}}))
    console = Console(file=io.StringIO())
    provider = ModelProvider(prefix='fixture', resolve=build)
    resolver = ModelResolver(console=console, store=store, plugins=lambda: {'fixture': provider})
    plugins = loader(store, console, resolve_model=resolver)
    try:
        await plugins.load_all()
        result = await Agent(FunctionModel(consulting), deps_type=type(None), capabilities=plugins.capabilities()).run(
            'Fix the bug'
        )
    finally:
        await plugins.close('exit')

    assert result.output == 'Following advice: Patch it.'
    assert resolved == ['wise'], "built through Pax's resolver with the configured model"
    assert advised == ['Rewrite or patch?'], 'the advisor sees the prompt the model wrote, not the conversation'
    usage = result.usage
    assert (usage.requests, usage.input_tokens, usage.output_tokens) == (3, 1020, 710)


async def test_a_name_pax_leaves_to_core_stays_a_name() -> None:
    """A name the resolver hands back as a name is inferred by core, as `/model` does with it."""
    resolved: list[str] = []

    async def resolve(name: str) -> Model | str:
        resolved.append(name)
        return 'test'

    settings: dict[str, JsonValue] = {'model': 'core:plain'}
    host = PluginHost[None](
        name='advisor', console=Console(file=io.StringIO()), settings=settings, resolve_model=resolve
    )
    capabilities: tuple[AgentCapability[None], ...] = load_plugin(AdvisorPlugin, host).capabilities
    result = await Agent(FunctionModel(consulting), deps_type=type(None), capabilities=list(capabilities)).run(
        'Fix the bug'
    )

    assert resolved == ['core:plain']
    assert result.output == 'Following advice: success (no tool calls)'


async def test_a_model_pax_cannot_resolve_names_the_fix(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    console = Console(file=io.StringIO())
    settings: dict[str, JsonValue] = {'model': 'chain:missing'}
    host = PluginHost[None](
        name='advisor',
        console=console,
        settings=settings,
        resolve_model=ModelResolver(console=console, store=store).resolve,
    )
    agent = Agent(
        FunctionModel(consulting),
        deps_type=type(None),
        capabilities=list(load_plugin(AdvisorPlugin, host).capabilities),
    )

    with pytest.raises(UserError) as raised:
        await agent.run('Fix the bug')
    assert str(raised.value) == (
        'Advisor model chain:missing: No chain named missing. Create one with /model chains. '
        'Choose another with /plugins configure advisor.'
    )


@pytest.mark.parametrize('model', ['', 'no-provider', ' openai:gpt-6', 'openai:', 'openai@bad name:gpt-6'])
def test_settings_reject_a_model_pax_could_not_name(model: str) -> None:
    with pytest.raises(ValueError):
        AdvisorSettings(model=model)


def test_settings_menu_chooses_and_resets_the_model() -> None:
    saved: list[dict[str, JsonValue]] = []
    host = PluginHost[None](
        name='advisor', console=Console(file=io.StringIO()), settings={}, save_settings=saved.append
    )
    source = AdvisorPlugin.from_host(host).settings_menu().source
    [row] = source.rows()
    assert (row.key, row.label, source.current(row)) == ('model', 'Advisor model', row.default)

    assert source.problem(row, 'gpt-6') == 'Value error, Use a nonempty PROVIDER[@ACCOUNT]:MODEL or chain:NAME.'
    assert source.problem(row, 'anthropic@work:claude-opus-4-8') is None
    assert source.apply(row, 'chain:deep') == 'The advisor is chain:deep from the next prompt.'
    assert source.current(row) == 'chain:deep'
    assert source.reset(row) == 'The advisor has no model, so it adds nothing.'
    assert saved == [{'model': 'chain:deep'}, {'model': None}]
    assert source.current(row) == row.default
