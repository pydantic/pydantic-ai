"""CLAI's default model and per-family settings sit beneath agent and capability configuration.

Only what the user chose (a model, saved per-model settings) is passed to `agent.run`, so a
capability that selects a model or supplies settings, such as Logfire's `AgentControl`, takes
effect over CLAI's defaults. Without such a capability, the effective model and settings are
what they were when CLAI passed both to `agent.run`.
"""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import AbstractCapability, Capability, CapabilityOrdering, ResolveModelId, Thinking
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import Model, ModelResolutionContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModelSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from pydantic_clai2._app import create_shell, create_stock_agent
from pydantic_clai2.config import Settings, resolve_settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TurnStart
from pydantic_clai2.runtime._session import Session
from tests.clai2.menu_script import make_context


@dataclass
class Recorder:
    """Stands in for every model CLAI resolves, recording the name and the merged settings each request got.

    The model supports thinking, so the generic `thinking` setting is recorded too: core moves it
    out of the settings into the request parameters.
    """

    calls: list[tuple[str, ModelSettings]] = field(default_factory=list[tuple[str, ModelSettings]])

    def resolve(self, name: str) -> Model:
        def record(info: AgentInfo) -> None:
            settings = ModelSettings(**(info.model_settings or {}))
            if (thinking := info.model_request_parameters.thinking) is not None:
                settings['thinking'] = thinking
            self.calls.append((name, settings))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            record(info)
            return ModelResponse(parts=[TextPart('done')])

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            record(info)
            yield 'done'

        return FunctionModel(
            respond, stream_function=stream, model_name=name, profile=ModelProfile(supports_thinking=True)
        )


@dataclass
class Published(AbstractCapability[None]):
    """A capability selecting a model and supplying settings, like Logfire's `AgentControl`."""

    model: Model | str | None = None
    settings: ModelSettings | None = None

    def get_model(self) -> Model | str | None:
        return self.model

    def get_model_settings(self) -> ModelSettings | None:
        return self.settings


async def run_turn(
    tmp_path: Path,
    *,
    settings: Settings,
    saved: dict[str, dict[str, JsonValue]] | None = None,
    plugins: tuple[AbstractCapability[None], ...] = (),
    agent: Agent[None, str] | None = None,
    recorder: Recorder | None = None,
    resolve: Callable[[str], Model | str] | None = None,
) -> tuple[Recorder, Session[None, str]]:
    """One shell turn, on the stock agent unless `agent` is given; returns what the model saw and the session."""
    store = SettingsStore(tmp_path / 'config.db')
    for name, values in (saved or {}).items():
        store.save_model_settings(name, values)
    shell = create_shell(
        create_stock_agent() if agent is None else agent,
        deps=None,
        plugins=plugins,
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=settings,
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    recorder = recorder or Recorder()
    shell.session.resolve_model = resolve or recorder.resolve
    ended = await shell.run_turn(TurnStart(text='hello'), headless=True)
    assert ended.outcome == 'completed', ended.error
    return recorder, shell.session


@dataclass(frozen=True)
class Case:
    id: str
    model: str
    chosen: bool = True
    """Whether the user chose the model; otherwise it is CLAI's default, the stock agent's own model."""
    saved: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    effective: ModelSettings = field(default_factory=ModelSettings)
    run_level: ModelSettings = field(default_factory=ModelSettings)


CASES = [
    Case(
        id='openai-reasoning',
        model='openai:gpt-6',
        effective=snapshot(
            {
                'service_tier': 'default',
                'openai_reasoning_effort': 'medium',
                'openai_reasoning_context': 'all_turns',
                'openai_reasoning_mode': 'standard',
                'openai_reasoning_summary': 'detailed',
                'openai_text_verbosity': 'low',
                'thinking': True,
            }
        ),
        run_level=snapshot({}),
    ),
    Case(
        id='openai-reasoning-overrides',
        model='openrouter:openai/gpt-5.6:free',
        saved={'openai_reasoning_effort': 'high', 'thinking': None, 'max_tokens': 2048},
        effective=snapshot(
            {
                'max_tokens': 2048,
                'service_tier': 'default',
                'openai_reasoning_effort': 'high',
                'openai_reasoning_context': 'all_turns',
                'openai_reasoning_mode': 'standard',
                'openai_reasoning_summary': 'detailed',
                'openai_text_verbosity': 'low',
            }
        ),
        run_level=snapshot({'max_tokens': 2048, 'openai_reasoning_effort': 'high'}),
    ),
    Case(
        id='anthropic',
        model='anthropic:claude-sonnet-4-6',
        saved={
            'anthropic_thinking_mode': 'adaptive',
            'anthropic_effort': 'high',
            'anthropic_thinking_display': 'updates',
        },
        effective=snapshot(
            {
                'anthropic_cache': '5m',
                'anthropic_cache_instructions': '5m',
                'anthropic_cache_tool_definitions': '5m',
                'anthropic_effort': 'high',
                'anthropic_thinking': {'type': 'adaptive', 'display': 'updates'},
                'extra_headers': {'anthropic-beta': 'thinking-display-updates-2026-08-18'},
            }
        ),
        run_level=snapshot(
            {
                'anthropic_effort': 'high',
                'anthropic_thinking': {'type': 'adaptive', 'display': 'updates'},
                'extra_headers': {'anthropic-beta': 'thinking-display-updates-2026-08-18'},
            }
        ),
    ),
    Case(
        id='codex-default',
        model='openai-codex:gpt-6-astra',
        chosen=False,
        effective=snapshot(
            {
                'service_tier': 'default',
                'openai_reasoning_effort': 'medium',
                'openai_reasoning_context': 'all_turns',
                'openai_reasoning_mode': 'standard',
                'openai_reasoning_summary': 'detailed',
                'openai_text_verbosity': 'low',
                'thinking': True,
            }
        ),
        run_level=snapshot({}),
    ),
    Case(
        id='codex-default-fast',
        model='openai-codex:gpt-6-astra',
        chosen=False,
        saved={'service_tier': 'priority', 'custom_params': {'reasoning.summary': 'auto'}},
        effective=snapshot(
            {
                'service_tier': 'priority',
                'openai_reasoning_effort': 'medium',
                'openai_reasoning_context': 'all_turns',
                'openai_reasoning_mode': 'standard',
                'openai_reasoning_summary': 'detailed',
                'openai_text_verbosity': 'low',
                'extra_body': {'reasoning': {'summary': 'auto'}},
                'thinking': True,
            }
        ),
        run_level=snapshot({'service_tier': 'priority', 'extra_body': {'reasoning': {'summary': 'auto'}}}),
    ),
    Case(
        id='codex-chosen',
        model='openai-codex:gpt-5.6-luna',
        saved={'openai_text_verbosity': 'medium'},
        effective=snapshot(
            {
                'service_tier': 'default',
                'openai_reasoning_effort': 'medium',
                'openai_reasoning_context': 'all_turns',
                'openai_reasoning_mode': 'standard',
                'openai_reasoning_summary': 'detailed',
                'openai_text_verbosity': 'medium',
                'thinking': True,
            }
        ),
        run_level=snapshot({'openai_text_verbosity': 'medium'}),
    ),
    Case(
        id='copilot',
        model='github-copilot:gpt-5.6',
        saved={'parallel_tool_calls': False},
        effective=snapshot(
            {
                'parallel_tool_calls': False,
                'service_tier': 'default',
                'openai_reasoning_effort': 'medium',
                'openai_reasoning_context': 'all_turns',
                'openai_reasoning_mode': 'standard',
                'openai_reasoning_summary': 'detailed',
                'openai_text_verbosity': 'low',
                'thinking': True,
            }
        ),
        run_level=snapshot({'parallel_tool_calls': False}),
    ),
]


@pytest.mark.parametrize('stock', [True, False], ids=['stock', 'supplied'])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_effective_settings_unchanged(tmp_path: Path, case: Case, stock: bool) -> None:
    """The model sees what it saw when CLAI passed defaults and overrides together to `agent.run`.

    `before` replays that: the model run with `model_settings=context.model_settings(name)`, the
    merged family defaults and saved overrides. Now only the overrides reach `agent.run`. A
    supplied agent keeps its own model unless one is chosen, as `--agent` does.
    """
    after = Recorder()
    if stock:
        agent = None
        settings = resolve_settings({'model': case.model}) if case.chosen else Settings()
    else:
        agent = Agent(None if case.chosen else after.resolve(case.model))
        settings = resolve_settings({'model': case.model}) if case.chosen else Settings(model=None)
    _, session = await run_turn(
        tmp_path, settings=settings, saved={case.model: case.saved}, agent=agent, recorder=after
    )

    context, _ = make_context(tmp_path)
    before = Recorder()
    await Agent(before.resolve(case.model)).run('hello', model_settings=context.model_settings(case.model))

    assert after.calls == before.calls == [(case.model, case.effective)]
    # Read again before each request, as core resolves it, so edits saved mid-turn apply.
    run_level = session.model_settings
    assert callable(run_level)
    assert run_level(RunContext(deps=None, model=TestModel(), usage=RunUsage())) == case.run_level


async def test_capability_settings_beat_defaults_not_overrides(tmp_path: Path) -> None:
    """A capability's settings replace CLAI's family defaults; the user's saved settings still win."""
    name = 'openai-codex:gpt-6-astra'
    recorder, _ = await run_turn(
        tmp_path,
        settings=Settings(),
        saved={name: {'openai_text_verbosity': 'medium'}},
        plugins=(
            Published(
                settings=OpenAIResponsesModelSettings(openai_reasoning_effort='low', openai_text_verbosity='high')
            ),
        ),
    )
    assert recorder.calls == snapshot(
        [
            (
                'openai-codex:gpt-6-astra',
                {
                    'service_tier': 'default',
                    'openai_reasoning_effort': 'low',
                    'openai_reasoning_context': 'all_turns',
                    'openai_reasoning_mode': 'standard',
                    'openai_reasoning_summary': 'detailed',
                    'openai_text_verbosity': 'medium',
                    'thinking': True,
                },
            )
        ]
    )


@dataclass
class Outermost(Published):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')


@pytest.mark.parametrize('stock', [True, False], ids=['stock', 'supplied'])
async def test_outermost_capability_settings_beat_defaults(tmp_path: Path, stock: bool) -> None:
    """CLAI's family defaults sit beneath a capability's settings even when that capability is `outermost` too."""
    recorder = Recorder()
    outermost = Outermost(settings=OpenAIResponsesModelSettings(openai_reasoning_effort='low'))
    await run_turn(
        tmp_path,
        settings=Settings() if stock else Settings(model=None),
        plugins=(outermost,),
        agent=None if stock else Agent(recorder.resolve('openai-codex:gpt-6-astra'), deps_type=type(None)),
        recorder=recorder,
    )
    assert [settings.get('openai_reasoning_effort') for _, settings in recorder.calls] == ['low']


async def test_capability_model_beats_default_model(tmp_path: Path) -> None:
    """A capability's model replaces CLAI's default and resolves through CLAI; the GPT defaults do not follow it.

    The capability's Claude model gets its own family defaults, the prompt-caching TTLs, instead.

    `resolved_model()`, which `/compact` and session naming run on, still names the selected model,
    as `PLUGINS.md` documents.
    """
    recorder, session = await run_turn(
        tmp_path, settings=Settings(), plugins=(Published(model='anthropic:claude-sonnet-4-6'),)
    )
    assert recorder.calls == snapshot(
        [
            (
                'anthropic:claude-sonnet-4-6',
                {
                    'anthropic_cache': '5m',
                    'anthropic_cache_instructions': '5m',
                    'anthropic_cache_tool_definitions': '5m',
                },
            )
        ]
    )
    selected = await session.resolved_model()
    assert isinstance(selected, Model) and selected.model_name == 'openai-codex:gpt-6-astra'


@pytest.mark.parametrize('stock', [True, False], ids=['stock', 'supplied'])
async def test_saved_settings_stay_with_their_model(tmp_path: Path, stock: bool) -> None:
    """Settings saved for the default model are not sent to the model a capability selects instead."""
    name = 'openai-codex:gpt-6-astra'
    recorder = Recorder()
    # Without a chosen model nothing resolves names for a supplied agent, so its capability selects an instance.
    other = 'anthropic:claude-sonnet-4-6'
    published = Published(model=other if stock else recorder.resolve(other))
    await run_turn(
        tmp_path,
        settings=Settings() if stock else Settings(model=None),
        saved={name: {'service_tier': 'priority', 'custom_params': {'reasoning.summary': 'auto'}}},
        plugins=(published,) if stock else (),
        agent=None if stock else Agent(recorder.resolve(name), deps_type=type(None), capabilities=[published]),
        recorder=recorder,
    )
    # Only the Claude model's own caching defaults; nothing saved for the GPT model.
    cache = {'anthropic_cache': '5m', 'anthropic_cache_instructions': '5m', 'anthropic_cache_tool_definitions': '5m'}
    assert recorder.calls == [(other, cache)]


async def test_agent_capability_settings_beat_family_defaults(tmp_path: Path) -> None:
    """A capability on a supplied agent, such as `Thinking`, now takes effect over CLAI's family defaults.

    CLAI used to pass the family defaults to `agent.run`, where they overrode it. Agent-level
    `model_settings` still sit beneath them, as before.
    """
    recorder = Recorder()
    agent = Agent(
        recorder.resolve('openai:gpt-6'),
        model_settings=OpenAIResponsesModelSettings(openai_text_verbosity='high', max_tokens=100),
        capabilities=[Thinking(effort=False)],
    )
    await run_turn(tmp_path, settings=Settings(model=None), agent=agent, recorder=recorder)
    assert recorder.calls == snapshot(
        [
            (
                'openai:gpt-6',
                {
                    'openai_text_verbosity': 'low',
                    'max_tokens': 100,
                    'service_tier': 'default',
                    'openai_reasoning_effort': 'medium',
                    'openai_reasoning_context': 'all_turns',
                    'openai_reasoning_mode': 'standard',
                    'openai_reasoning_summary': 'detailed',
                    'thinking': False,
                },
            )
        ]
    )


@pytest.mark.parametrize('model', [None, 'anthropic:claude-sonnet-4-6'])
async def test_fork_keeps_how_the_model_was_chosen(tmp_path: Path, model: str | None) -> None:
    """A fork of a session on CLAI's default model leaves a capability free to replace it, unless the fork names one."""
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        create_stock_agent(),
        deps=None,
        plugins=(Published(model='openai:gpt-6-luna'),),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=store.load(),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    recorder = Recorder()
    shell.session.resolve_model = recorder.resolve
    fork = shell.fork_session(model, [])
    await fork.prompt('hello')
    assert recorder.calls[-1][0] == (model or 'openai:gpt-6-luna')


@pytest.mark.parametrize('stock', [True, False], ids=['stock', 'supplied'])
async def test_selected_model_resolves_through_clai_first(tmp_path: Path, stock: bool) -> None:
    """CLAI resolves the selected model before a resolver on the agent or a plugin, as when it did so before each run."""

    def claim(ctx: ModelResolutionContext[None], model_id: str) -> Model:
        raise AssertionError(f'{model_id!r} reached a resolver after CLAI')  # pragma: no cover

    resolver = ResolveModelId[None](claim)
    recorder, _ = await run_turn(
        tmp_path,
        settings=Settings() if stock else resolve_settings({'model': 'openai:gpt-6'}),
        plugins=(resolver,) if stock else (),
        agent=None if stock else Agent(None, deps_type=type(None), capabilities=[resolver]),
    )
    assert [name for name, _ in recorder.calls] == ['openai-codex:gpt-6-astra' if stock else 'openai:gpt-6']


@pytest.mark.parametrize('chosen', [False, True], ids=['capability', 'chosen'])
async def test_names_clai_does_not_handle_are_left_to_other_resolvers(tmp_path: Path, chosen: bool) -> None:
    """An alias only a plugin's resolver knows still resolves there, whether a capability or the user selected it."""
    recorder = Recorder()

    @dataclass
    class Alias(Published):
        async def resolve_model_id(self, ctx: ModelResolutionContext[None], *, model_id: str) -> Model | None:
            return recorder.resolve('anthropic:claude-sonnet-4-6') if model_id == 'my-alias' else None

    default = Settings().model
    await run_turn(
        tmp_path,
        settings=resolve_settings({'model': 'my-alias'}) if chosen else Settings(),
        plugins=(Alias(model=None if chosen else 'my-alias'),),
        recorder=recorder,
        # Like CLAI's resolver, which returns a name it has no integration for unchanged.
        resolve=lambda name: recorder.resolve(name) if name == default else name,
    )
    assert [name for name, _ in recorder.calls] == ['anthropic:claude-sonnet-4-6']


async def test_selected_name_clai_translates(tmp_path: Path) -> None:
    """A `resolve_model` that translates the selected name to another one runs the translated model."""
    recorder, session = await run_turn(
        tmp_path,
        settings=resolve_settings({'model': 'fast'}),
        resolve=lambda name: 'test' if name == 'fast' else name,
    )
    assert recorder.calls == []
    response = session.messages[-1]
    assert isinstance(response, ModelResponse) and response.model_name == 'test'


@pytest.mark.parametrize('source', ['saved', 'cli'])
async def test_chosen_model_beats_capability_model(tmp_path: Path, source: str) -> None:
    """A model the user chose is passed to `agent.run`, so it wins over a capability's."""
    name = 'openai-codex:gpt-6-astra'
    store = SettingsStore(tmp_path / 'config.db')
    if source == 'saved':
        store.set('model', name)
    settings = store.load() if source == 'saved' else resolve_settings({'model': name})
    recorder, _ = await run_turn(tmp_path, settings=settings, plugins=(Published(model='anthropic:claude-sonnet-4-6'),))
    assert [called for called, _ in recorder.calls] == [name]


@pytest.mark.parametrize('command', ['/set model', '/model', '/model add', '/add_model'])
async def test_set_model_chooses_and_reset_restores_default(tmp_path: Path, command: str) -> None:
    """Every way to pick a model by name makes a choice; resetting it, or changing another setting, leaves CLAI's default.

    `/model NAME` takes a model that is not in the saved list yet and adds it.
    """
    store = SettingsStore(tmp_path / 'config.db')
    shell = create_shell(
        create_stock_agent(),
        deps=None,
        plugins=(Published(model='anthropic:claude-sonnet-4-6'),),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=store.load(),
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    recorder = Recorder()
    shell.session.resolve_model = recorder.resolve

    async def ran() -> str:
        recorder.calls.clear()
        ended = await shell.run_turn(TurnStart(text='hello'), headless=True)
        assert ended.outcome == 'completed', ended.error
        return recorder.calls[-1][0]

    await shell.commands.execute_async('/set display.thinking false')
    assert await ran() == 'anthropic:claude-sonnet-4-6'
    await shell.commands.execute_async(f'{command} openai:gpt-6-luna')
    assert shell.session.model_chosen
    assert await ran() == 'openai:gpt-6-luna'
    shell.context.reset_setting('model')
    assert shell.session.model == 'openai-codex:gpt-6-astra'
    assert await ran() == 'anthropic:claude-sonnet-4-6'


async def test_settings_saved_mid_turn_match_the_merged_settings(tmp_path: Path) -> None:
    """A field saved or reset while a turn runs reaches its next request exactly as the merged settings would.

    The run-level overrides and the family defaults beneath them both read the store again per request.
    """
    name = 'openai:gpt-6'
    store = SettingsStore(tmp_path / 'config.db')
    context, _ = make_context(tmp_path)
    expected: list[ModelSettings] = []
    store.save_model_settings(name, {'openai_reasoning_effort': 'high'})
    # Reset back to the family default, then save another value.
    edits: list[dict[str, JsonValue]] = [{}, {'openai_reasoning_effort': 'low'}]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise NotImplementedError  # pragma: no cover

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        settings = ModelSettings(**(info.model_settings or {}))
        thinking = info.model_request_parameters.thinking
        assert thinking is not None
        settings['thinking'] = thinking
        recorded.append(settings)
        if edits:
            store.save_model_settings(name, edits.pop(0))
            expected.append(context.model_settings(name) or ModelSettings())
            yield {0: DeltaToolCall(name='noop', json_args='{}')}
        else:
            yield 'done'

    recorded: list[ModelSettings] = []
    model = FunctionModel(
        respond, stream_function=stream, model_name=name, profile=ModelProfile(supports_thinking=True)
    )
    tool = Capability[None](tools=[Tool(lambda: 'ok', name='noop', takes_ctx=False)])
    await run_turn(tmp_path, settings=resolve_settings({'model': name}), plugins=(tool,), resolve=lambda _: model)
    assert recorded[1:] == expected
    assert [dict(settings).get('openai_reasoning_effort') for settings in recorded] == ['high', 'medium', 'low']
