"""Capability *setup* wiring in `Agent.realtime_session`, shared with the graph run.

`realtime_session` and `run`/`iter` resolve capabilities through the same
`Agent._resolve_run_capabilities`, so a capability's setup contributions — instructions, native tools
(including under `override(native_tools=...)`), model settings, and toolsets — must reach a session
exactly as they reach a run. These pin that, guarding against the two silently diverging again (the
session used to drop capability instructions/model-settings and, under a native-tools override, drop a
capability-function's native tools). Network-free: a fake model records what `connect()` receives.
"""

from __future__ import annotations as _annotations

import asyncio
import contextvars
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._deferred_capabilities import (
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    parse_loaded_capabilities,
)
from pydantic_ai._instrumentation import get_instructions
from pydantic_ai.capabilities import Hooks, NativeTool, ProcessEventStream, WebSearch
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SpeechPart,
    SpeechPartDelta,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool, WebSearchTool
from pydantic_ai.realtime import (
    RealtimeEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    RealtimeSession,
    RealtimeTurnCompleteEvent,
)
from pydantic_ai.realtime.codec import (
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ResponseDone,
    ToolCall,
    ToolResult,
    UpdateTools,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ..conftest import IsDatetime

pytestmark = pytest.mark.anyio


class _Connection(RealtimeConnection):
    """Replays a fixed list of events (a lone `ResponseDone` by default) so the session drains."""

    def __init__(self, events: Sequence[RealtimeCodecEvent] = (ResponseDone(),)) -> None:
        self._events = events
        self.sent: list[RealtimeInput] = []

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        for event in self._events:
            yield event


class _RecordingModel(RealtimeModel):
    """A realtime model that records the arguments `realtime_session` passes to `connect`."""

    def __init__(
        self,
        *,
        settings: RealtimeModelSettings | None = None,
        supported_native_tools: frozenset[type[AbstractNativeTool]] = frozenset(),
        supports_tool_updates: bool = False,
        connection_events: Sequence[RealtimeCodecEvent] = (ResponseDone(),),
        connection: RealtimeConnection | None = None,
    ) -> None:
        self.settings = settings
        self._supported = supported_native_tools
        self._supports_tool_updates = supports_tool_updates
        self._connection_events = connection_events
        self.connection = connection
        self.instructions: str | None = None
        self.tools: list[ToolDefinition] | None = None
        self.native_tools: list[AbstractNativeTool] | None = None
        self.model_settings: RealtimeModelSettings | None = None

    @property
    def model_name(self) -> str:
        return 'gpt-realtime'

    @property
    def system(self) -> str:
        return 'openai'

    @property
    def profile(self) -> RealtimeModelProfile:
        return RealtimeModelProfile(
            supports_image_input=True,
            supports_manual_turn_control=True,
            supports_interruption=True,
            supports_output_truncation=True,
            supports_session_seeding=True,
            supports_tool_updates=self._supports_tool_updates,
            supported_native_tools=self._supported,
        )

    @asynccontextmanager
    async def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncGenerator[RealtimeConnection]:
        self.instructions = get_instructions(messages)
        self.tools = model_request_parameters.function_tools
        self.native_tools = model_request_parameters.native_tools
        self.model_settings = model_settings
        if self.connection is None:
            self.connection = _Connection(self._connection_events)
        yield self.connection


async def _drain(agent: Agent[None, str], model: _RecordingModel, **kwargs: object) -> list[RealtimeEvent]:
    events: list[RealtimeEvent] = []
    async with agent.realtime(model, **kwargs).session() as session:  # type: ignore[arg-type]
        async for event in session:  # pragma: no branch
            events.append(event)
    return events


async def test_capability_instructions_reach_session() -> None:
    """A capability's `get_instructions` is combined with the agent's, like a graph run."""

    class PirateCap(AbstractCapability[None]):
        def get_instructions(self) -> str:
            return 'Speak like a pirate.'

    agent = Agent(instructions='Be helpful.')
    model = _RecordingModel()
    await _drain(agent, model, capabilities=[PirateCap()])
    assert model.instructions is not None
    assert 'Be helpful.' in model.instructions
    assert 'Speak like a pirate.' in model.instructions


async def test_per_call_capabilities_are_bound_via_for_agent() -> None:
    """Per-call `capabilities=` are bound via `for_agent` before resolution, like `run`/`iter`.

    Regression: `realtime_session` skipped the `for_agent` binding, so a capability that overrides
    `for_agent` (e.g. the durability capabilities) was used unbound.
    """

    class BoundInstructionsCap(AbstractCapability[None]):
        def __init__(self, *, bound: bool = False) -> None:
            self._bound = bound

        def for_agent(self, agent: object) -> AbstractCapability[None]:
            return BoundInstructionsCap(bound=True)

        def get_instructions(self) -> str:
            return 'bound-instruction' if self._bound else 'unbound-instruction'

    agent = Agent(instructions='Base.')
    model = _RecordingModel()
    await _drain(agent, model, capabilities=[BoundInstructionsCap()])
    assert model.instructions is not None
    assert 'bound-instruction' in model.instructions
    assert 'unbound-instruction' not in model.instructions


async def test_root_capability_override_reaches_session() -> None:
    """`override(spec=...)` replaces the root capability and applies to a realtime session, like `iter`.

    Regression: `realtime_session` resolved from `self._root_capability`, ignoring the
    `override(root_capability=...)` set by `override(spec=...)`, so the overridden capability's
    contributions were silently dropped.
    """
    agent = Agent(instructions='Be helpful.')
    model = _RecordingModel()
    with agent.override(spec={'instructions': 'from override'}):
        await _drain(agent, model)
    assert model.instructions is not None
    assert 'from override' in model.instructions


async def test_regular_settings_do_not_reach_session() -> None:
    """Regular agent and capability settings do not leak into realtime settings."""

    class SettingsCap(AbstractCapability[None]):
        def get_model_settings(self) -> ModelSettings:
            return ModelSettings(temperature=0.3)

    agent = Agent(model_settings=ModelSettings(temperature=0.1))
    model = _RecordingModel(settings=RealtimeModelSettings(max_tokens=100, parallel_tool_calls=False))
    await _drain(
        agent,
        model,
        capabilities=[SettingsCap()],
        model_settings=RealtimeModelSettings(parallel_tool_calls=True),
    )
    assert model.model_settings == RealtimeModelSettings(max_tokens=100, parallel_tool_calls=True)


async def test_capability_toolset_reaches_session() -> None:
    """A capability's `get_toolset` contributes its tools to the session's tool set."""
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet(name: str) -> str:
        return f'Hello, {name}!'

    assert greet('World') == 'Hello, World!'

    class ToolsetCap(AbstractCapability[None]):
        def get_toolset(self) -> FunctionToolset[None]:
            return toolset

    agent = Agent()
    model = _RecordingModel()
    await _drain(agent, model, capabilities=[ToolsetCap()])
    assert model.tools is not None
    assert any(t.name == 'greet' for t in model.tools)


def _weather_toolset() -> FunctionToolset[None]:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def forecast(city: str) -> str:
        return f'Sunny in {city}.'

    return toolset


class _Weather(AbstractCapability[None]):
    """A deferred capability that contributes one tool, revealed when it loads."""

    id = 'weather'
    defer_loading = True

    def get_description(self) -> str:
        return 'Look up the weather.'

    def get_instructions(self) -> str | None:
        return 'Answer weather questions from the forecast tool.'

    def get_toolset(self) -> FunctionToolset[None]:
        return _weather_toolset()


class _ScriptedConnection(RealtimeConnection):
    """Replays turns, holding each one back until the session has answered the previous tool call.

    A real provider only speaks again once it has the tool result, and the reveal under test depends
    on that ordering: the follow-up call must be dispatched against the tool list the load produced.
    """

    def __init__(self, turns: Sequence[Sequence[RealtimeCodecEvent]]) -> None:
        self._turns = turns
        self.sent: list[RealtimeInput] = []
        self._answered = asyncio.Event()

    async def send(self, content: RealtimeInput) -> None:
        self.sent.append(content)
        if isinstance(content, ToolResult):
            self._answered.set()

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        for index, turn in enumerate(self._turns):
            if index:
                await self._answered.wait()
                self._answered.clear()
            for event in turn:
                yield event


@pytest.mark.parametrize('supports_tool_updates', [False, True])
async def test_deferred_capability_with_native_tools_raises_before_connect(supports_tool_updates: bool) -> None:
    """A deferred capability contributing *native* tools fails at session open on every provider.

    Re-advertising function tools is what `supports_tool_updates` buys; no realtime provider can turn
    on a server-side native tool mid-conversation, so the flag makes no difference here.
    """

    class DeferredCap(AbstractCapability[None]):
        id = 'deferred'
        defer_loading = True

        def get_native_tools(self) -> Sequence[AbstractNativeTool]:
            return [WebSearchTool()]

    agent = Agent()
    model = _RecordingModel(
        supported_native_tools=frozenset({WebSearchTool}), supports_tool_updates=supports_tool_updates
    )

    with pytest.raises(
        UserError,
        match=r"Realtime sessions cannot reveal native tools mid-session.*'deferred'",
    ):
        await _drain(agent, model, capabilities=[DeferredCap()])

    assert model.tools is None


async def test_deferred_capability_with_tools_raises_when_the_model_cannot_update_tools() -> None:
    """A tool-contributing deferred capability fails at open on a model that fixes tools at connect.

    Loading it would advertise nothing new, so the model would be told a capability is active while
    the tools it promised stay invisible — worse than the up-front error.
    """
    agent = Agent()
    model = _RecordingModel()

    with pytest.raises(
        UserError,
        match=r"does not support updating tools mid-session.*'weather'",
    ):
        await _drain(agent, model, capabilities=[_Weather()])

    assert model.tools is None


async def test_deferred_capability_tools_are_revealed_by_a_mid_session_update() -> None:
    """On a `supports_tool_updates` model, loading a deferred capability advertises its tools.

    The connect-time list holds `load_capability` and not the capability's own tools; the load sends
    a re-advertisement carrying them, ahead of the tool result that announces the load, so the model
    can never learn of the capability before its tools exist. The revealed tool is then callable, and
    the reveal is recorded in history the way a graph run records it.
    """
    connection = _ScriptedConnection(
        [
            [ToolCall(tool_call_id='tc_1', tool_name='load_capability', args='{"id": "weather"}')],
            [ToolCall(tool_call_id='tc_2', tool_name='forecast', args='{"city": "Lisbon"}'), ResponseDone()],
        ]
    )
    agent = Agent()
    model = _RecordingModel(supports_tool_updates=True, connection=connection)

    events: list[RealtimeEvent] = []
    async with agent.realtime(model, capabilities=[_Weather()]).session() as session:  # type: ignore[arg-type]
        async for event in session:  # pragma: no branch
            events.append(event)

    assert model.tools is not None
    assert [tool.name for tool in model.tools] == ['load_capability']

    update, load_result, forecast_result = connection.sent
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['load_capability', 'forecast']
    assert isinstance(load_result, ToolResult) and load_result.tool_call_id == 'tc_1'
    assert isinstance(forecast_result, ToolResult) and forecast_result.output == 'Sunny in Lisbon.'

    results = [e.part for e in events if isinstance(e, FunctionToolResultEvent)]
    assert [(part.tool_name, part.content) for part in results] == snapshot(
        [
            ('load_capability', {'instructions': 'Answer weather questions from the forecast tool.'}),
            ('forecast', 'Sunny in Lisbon.'),
        ]
    )

    history = session.all_messages()
    assert parse_loaded_capabilities(history) == {'weather'}
    assert [
        part
        for message in history
        for part in message.parts
        if isinstance(part, (LoadCapabilityCallPart, LoadCapabilityReturnPart, ToolAvailabilityDeltaPart))
    ] == snapshot(
        [
            LoadCapabilityCallPart(args='{"id": "weather"}', tool_call_id='tc_1'),
            LoadCapabilityReturnPart(
                content={'instructions': 'Answer weather questions from the forecast tool.'},
                tool_call_id='tc_1',
                timestamp=IsDatetime(),
            ),
            ToolAvailabilityDeltaPart(tools_added=['forecast'], tool_call_id='tc_1'),
        ]
    )


async def test_seeded_history_advertises_a_loaded_capabilitys_tools_at_connect() -> None:
    """A capability the seeded history already loaded needs no mid-session update, on any provider.

    Its tools go out with the connect-time list, so this works on a model that fixes its tools at
    connect (`supports_tool_updates` off) — the guard that rejects a tool-contributing deferred
    capability exempts an already-loaded one. Loading it again is refused as already available.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="What's the weather?")]),
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'weather'}, tool_call_id='tc_0')]),
        ModelRequest(
            parts=[
                LoadCapabilityReturnPart(content={'instructions': 'Use it.'}, tool_call_id='tc_0'),
                ToolAvailabilityDeltaPart(tools_added=['forecast'], tool_call_id='tc_0'),
            ]
        ),
    ]
    agent = Agent()
    model = _RecordingModel(
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='load_capability', args='{"id": "weather"}'),
            ResponseDone(),
        ],
    )
    events = await _drain(agent, model, capabilities=[_Weather()], message_history=history)

    assert model.tools is not None
    assert [tool.name for tool in model.tools] == ['load_capability', 'forecast']
    assert model.connection is not None
    assert not any(isinstance(frame, UpdateTools) for frame in model.connection.sent)  # type: ignore[attr-defined]

    retry = next(e.part for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(retry, RetryPromptPart)
    assert retry.content == snapshot(
        "Capability 'weather' is already available. Use its existing instructions and any tools it "
        'provides; do not call `load_capability` for it again.'
    )


async def test_seeded_history_advertises_an_already_revealed_deferred_tool_at_connect() -> None:
    """A plain `defer_loading=True` tool the seeded history already revealed advertises at connect.

    The session has no tool-search surface to reveal one *during* the call, which is why an unrevealed
    deferred tool is refused — but a history that already carries its
    `ToolAvailabilityDeltaPart` resolves it to visible, so there is nothing left to reveal and nothing
    to refuse. Works on a model that fixes its tools at connect (`supports_tool_updates` off).

    The local `search_tools` fallback rides along, because a deferred tool exists to build a corpus
    from. It is inert here rather than a dead end: this state is only reachable when *every* deferred
    tool is already revealed (an unrevealed one still fails the guard), so a search can only ever name
    tools the model can already see, which the session recognizes as a no-op reveal.
    """
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def unlock() -> ToolReturn[str]:  # pragma: no cover — only its recorded history is replayed here
        return ToolReturn(return_value='unlocked', tools=['withheld_tool'])

    @agent.tool_plain(defer_loading=True)
    def withheld_tool() -> str:  # pragma: no cover — the model never calls it in this test
        return 'withheld'

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='unlock the extras')]),
        ModelResponse(parts=[ToolCallPart(tool_name='unlock', args='{}', tool_call_id='tc_0')]),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='unlock', content='unlocked', tool_call_id='tc_0'),
                ToolAvailabilityDeltaPart(tools_added=['withheld_tool'], tool_call_id='tc_0'),
            ]
        ),
    ]
    model = _RecordingModel()
    await _drain(agent, model, message_history=history)

    assert model.tools is not None
    assert sorted(tool.name for tool in model.tools) == snapshot(['search_tools', 'unlock', 'withheld_tool'])


async def test_deferred_instruction_capability_loads_through_the_tool() -> None:
    """An instruction-only deferred capability works in a session on every provider.

    The catalog renders into the connect-time instructions, `load_capability` is advertised like any
    function tool, and the loaded instructions travel back as the tool call's own result — no
    mid-session tool reveal is involved.
    """

    class Pirate(AbstractCapability[None]):
        id = 'pirate'
        defer_loading = True

        def get_description(self) -> str:
            return 'Talk like a pirate.'

        def get_instructions(self) -> str | None:
            return 'Speak like a pirate at all times.'

    agent = Agent()
    model = _RecordingModel(
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='load_capability', args='{"id": "pirate"}'),
            ResponseDone(),
        ],
    )
    events = await _drain(agent, model, capabilities=[Pirate()])

    assert model.instructions is not None
    assert '- pirate: Talk like a pirate.' in model.instructions
    assert model.tools is not None
    assert any(tool.name == 'load_capability' for tool in model.tools)
    result_event = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    result_part = result_event.part
    assert isinstance(result_part, ToolReturnPart)
    assert result_part.tool_name == 'load_capability'
    assert 'Speak like a pirate at all times.' in str(result_part.content)


async def test_capability_native_tool_survives_native_tools_override() -> None:
    """A per-call capability's native tool is preserved on top of `override(native_tools=...)`.

    Regression: the session used to read the *unresolved* extra capabilities when an override was
    active, so a native tool that only materializes in a capability function's `for_run` (here, a
    lambda returning `NativeTool(WebSearchTool())`) was silently dropped. It must be preserved, exactly
    as in a graph run.
    """

    def web_search_cap(ctx: RunContext[None]) -> NativeTool[None]:
        # A capability *function*: its native tool only materializes when `for_run` resolves it.
        return NativeTool(WebSearchTool())

    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset({WebSearchTool}))
    with agent.override(native_tools=[]):  # replace the baseline; per-call cap tools stay on top
        await _drain(agent, model, capabilities=[web_search_cap])
    assert model.native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.native_tools)


async def test_capability_native_tool_without_override_reaches_session() -> None:
    """Without an override, a capability-contributed native tool still reaches the session."""
    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset({WebSearchTool}))
    await _drain(agent, model, capabilities=[NativeTool(WebSearchTool())])
    assert model.native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.native_tools)


async def test_dynamic_native_tool_function_resolves_at_connect() -> None:
    """A dynamic native-tool function resolves against the connect-time context, like `run`/`iter`.

    Regression: dynamic native-tool functions (`NativeTool(callable)`) were silently dropped from
    realtime sessions — only concrete `AbstractNativeTool` instances survived. They now resolve once
    at connect, like dynamic instructions: a session's tool list is fixed from the moment the
    connection opens.
    """

    def make_web_search(ctx: RunContext[None]) -> WebSearchTool:
        return WebSearchTool()

    async def make_none(ctx: RunContext[None]) -> None:
        return None  # a dynamic native tool resolving to None contributes nothing

    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset({WebSearchTool}))
    await _drain(agent, model, capabilities=[NativeTool(make_web_search), NativeTool(make_none)])
    assert model.native_tools is not None
    assert [type(t) for t in model.native_tools] == [WebSearchTool]


async def test_unsupported_capability_native_tool_raises_before_connect() -> None:
    """An unsupported native tool with no local fallback fails up front, before connecting.

    This runs the same native ↔ local-tool swap the classic agent-run path applies. With no local
    fallback configured, the swap raises the shared `UserError` that points the user at `local=...`.
    """
    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset())  # supports nothing
    with pytest.raises(UserError, match=r"not supported by this model.*WebSearch\(local='duckduckgo'\)"):
        await _drain(agent, model, capabilities=[NativeTool(WebSearchTool())])
    assert model.native_tools is None  # never connected


async def test_unsupported_native_tool_falls_back_to_local() -> None:
    """An unsupported native tool with a configured local fallback swaps to the local tool, not raise.

    Mirrors the classic path: the native tool is dropped (the model supports none) and the local
    DuckDuckGo function tool stays, so the session connects with no native tools.
    """
    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset())  # supports nothing
    await _drain(agent, model, capabilities=[WebSearch(local='duckduckgo')])
    assert model.native_tools == []  # native dropped, connected without it
    assert model.tools is not None
    assert any(t.name == 'duckduckgo_search' for t in model.tools)  # local fallback kept


async def test_supported_native_tool_drops_local_fallback() -> None:
    """When the native tool IS supported, the redundant local fallback is dropped, like the classic path."""
    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset({WebSearchTool}))
    await _drain(agent, model, capabilities=[WebSearch(local='duckduckgo')])
    assert model.native_tools is not None
    assert any(isinstance(t, WebSearchTool) for t in model.native_tools)  # native kept
    assert model.tools is not None
    assert not any(t.name == 'duckduckgo_search' for t in model.tools)  # redundant local dropped


async def test_local_fallback_tool_is_dispatched_through_tool_manager() -> None:
    """The swapped-in local fallback is a real function tool the session dispatches via the `ToolManager`.

    Drives a tool call for the local fallback through the session and asserts it executed and returned,
    proving the fallback isn't just present on the wire but wired into tool dispatch. Uses a plain
    recording callable as the local tool so the test stays network-free (the DuckDuckGo fallback used
    by the other cases would hit the network when invoked).
    """
    invoked: list[str] = []

    def local_search(query: str) -> str:
        invoked.append(query)
        return f'result for {query}'

    agent = Agent()
    model = _RecordingModel(
        supported_native_tools=frozenset(),  # unsupported → fall back to local
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='local_search', args='{"query": "hello"}'),
            ResponseDone(),
        ],
    )
    events = await _drain(agent, model, capabilities=[WebSearch(native=WebSearchTool(), local=local_search)])

    assert invoked == ['hello']  # the local callable ran, via the ToolManager
    result_event = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    result_part = result_event.part
    assert isinstance(result_part, ToolReturnPart)
    assert (result_part.tool_name, result_part.content) == ('local_search', 'result for hello')


async def test_run_lifecycle_hooks_fire_for_a_session() -> None:
    """Run hooks surround the full session and `wrap_run` cleanup runs when it closes."""
    calls: list[str] = []

    class LifecycleCapability(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            assert ctx.realtime
            calls.append('before_run')

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            calls.append(f'after_run:{result.output}')
            return result

        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            calls.append('wrap_run:before')
            try:
                return await handler()
            finally:
                calls.append('wrap_run:finally')

    model = _RecordingModel(connection_events=[OutputTranscript(text='final answer', is_final=True), ResponseDone()])
    agent = Agent(capabilities=[LifecycleCapability()], deps_type=type(None))

    async with agent.realtime(model).session() as session:
        calls.append('session:body')
        async for _ in session:
            pass

    assert calls == [
        'wrap_run:before',
        'before_run',
        'session:body',
        'wrap_run:finally',
        'after_run:final answer',
    ]
    assert session.result is not None
    assert session.result.output == 'final answer'
    assert session.result.new_messages() == session.new_messages()


async def test_wrap_run_recovers_session_error() -> None:
    """`wrap_run` recovery suppresses a session error and supplies `session.result`."""

    class RecoveryCapability(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            try:
                return await handler()
            except RuntimeError as error:
                assert str(error) == 'session failed'
                return AgentRunResult(output='wrap recovered')

    agent = Agent(capabilities=[RecoveryCapability()], deps_type=type(None))

    async with agent.realtime(_RecordingModel()).session() as session:
        raise RuntimeError('session failed')

    assert session.result is not None
    assert session.result.output == 'wrap recovered'


async def test_on_run_error_recovers_session_error() -> None:
    """`on_run_error` recovery suppresses a session error and supplies `session.result`."""

    class RecoveryCapability(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            assert str(error) == 'session failed'
            return AgentRunResult(output='error hook recovered')

    agent = Agent(capabilities=[RecoveryCapability()], deps_type=type(None))

    async with agent.realtime(_RecordingModel()).session() as session:
        raise RuntimeError('session failed')

    assert session.result is not None
    assert session.result.output == 'error hook recovered'


async def test_on_run_error_recovers_connect_failure() -> None:
    """A recovered *pre-session* failure (connecting) yields a closed session carrying the result.

    The lifecycle hooks suppress the connect error before any session was yielded; without the
    recovery yield in `_open_realtime_session`, `asynccontextmanager` would turn that clean exit
    into `RuntimeError: generator didn't yield` and defeat the documented run-error recovery.
    """

    class _FailingConnectModel(_RecordingModel):
        @asynccontextmanager
        async def connect(
            self,
            *,
            messages: Sequence[ModelMessage],
            model_settings: RealtimeModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> AsyncGenerator[RealtimeConnection]:
            raise RuntimeError('connect failed')
            yield  # pragma: no cover

    class RecoveryCapability(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            assert str(error) == 'connect failed'
            return AgentRunResult(output='recovered before connecting')

    agent = Agent(capabilities=[RecoveryCapability()], deps_type=type(None))

    async with agent.realtime(_FailingConnectModel()).session() as session:
        # No connection was ever opened, so the recovered session is yielded closed.
        with pytest.raises(UserError, match='session is closed'):
            await session.send('hello')

    assert session.result is not None
    assert session.result.output == 'recovered before connecting'


async def test_on_run_error_recovers_toolset_enter_failure() -> None:
    """A recovered failure *during resolution* (entering the toolset) also yields a closed session.

    This error fires inside `_resolve_realtime_session` after the lifecycle hooks are entered but
    before the resolution is yielded, exercising the resolver's own recovery yield.
    """

    class _FailingToolset(FunctionToolset[None]):
        async def __aenter__(self) -> _FailingToolset:
            raise RuntimeError('toolset failed')

    class RecoveryCapability(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            assert str(error) == 'toolset failed'
            return AgentRunResult(output='recovered before resolving')

    agent = Agent(capabilities=[RecoveryCapability()], toolsets=[_FailingToolset()], deps_type=type(None))

    async with agent.realtime(_RecordingModel()).session() as session:
        with pytest.raises(UserError, match='session is closed'):
            await session.send('hello')

    assert session.result is not None
    assert session.result.output == 'recovered before resolving'


async def test_unrecovered_session_error_propagates() -> None:
    """A session error still propagates unchanged when no run hook recovers it."""
    agent = Agent(deps_type=type(None))

    with pytest.raises(RuntimeError, match='session failed'):
        async with agent.realtime(_RecordingModel()).session():
            raise RuntimeError('session failed')


async def test_after_run_transforms_session_result() -> None:
    """An `after_run` result transformation becomes the public `session.result`."""

    class TransformCapability(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            return AgentRunResult(output=f'transformed: {result.output}')

    agent = Agent(capabilities=[TransformCapability()], deps_type=type(None))
    model = _RecordingModel(connection_events=[OutputTranscript(text='answer', is_final=True), ResponseDone()])

    async with agent.realtime(model).session() as session:
        async for _ in session:
            pass

    assert session.result is not None
    assert session.result.output == 'transformed: answer'


async def test_wrap_run_short_circuits_before_session_connects() -> None:
    """A `wrap_run` short-circuit returns a closed result-only session without connecting."""

    class ShortCircuitCapability(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            return AgentRunResult(output='short-circuited')

    model = _RecordingModel()
    agent = Agent(capabilities=[ShortCircuitCapability()], deps_type=type(None))

    async with agent.realtime(model).session() as session:
        assert session.closed
        assert session.result is not None
        assert session.result.output == 'short-circuited'

    assert model.instructions is None


async def test_short_circuit_then_recovered_caller_error_exits_cleanly() -> None:
    """A caller-body error after a `wrap_run` short-circuit, recovered by `on_run_error`, exits cleanly.

    The short-circuit path yields a closed session without connecting; if the caller then raises and
    `on_run_error` recovers, the lifecycle hooks suppress the error. Without the `yielded` guard on the
    short-circuit yields, `_resolve_realtime_session`/`_open_realtime_session` would resume past their
    exit stacks and yield a second time, which `asynccontextmanager` reports as
    `RuntimeError: generator didn't stop after athrow()`.
    """

    class ShortCircuitThenRecoverCapability(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            return AgentRunResult(output='short-circuited')

        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            assert str(error) == 'caller failed'
            return AgentRunResult(output='recovered after short-circuit')

    agent = Agent(capabilities=[ShortCircuitThenRecoverCapability()], deps_type=type(None))

    async with agent.realtime(_RecordingModel()).session() as session:
        assert session.closed
        raise RuntimeError('caller failed')

    assert session.result is not None


async def test_wrap_run_context_is_ambient_throughout_session() -> None:
    """Context set before `handler()` reaches instructions, caller code, and tool execution."""
    ambient: contextvars.ContextVar[str] = contextvars.ContextVar('realtime_lifecycle_ambient')
    seen: list[tuple[str, str | None]] = []

    class AmbientCapability(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            token = ambient.set('managed prompt')
            try:
                return await handler()
            finally:
                ambient.reset(token)

    agent = Agent(capabilities=[AmbientCapability()], deps_type=type(None))

    @agent.instructions
    def dynamic_instructions(ctx: RunContext[None]) -> str:
        seen.append(('instructions', ambient.get(None)))
        return 'Use the tool.'

    @agent.tool
    def inspect_context(ctx: RunContext[None]) -> str:
        seen.append(('tool', ambient.get(None)))
        return 'done'

    model = _RecordingModel(
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='inspect_context', args='{}'),
            ResponseDone(),
        ]
    )
    async with agent.realtime(model).session() as session:
        seen.append(('caller', ambient.get(None)))
        async for _ in session:
            pass

    assert seen == [
        ('instructions', 'managed prompt'),
        ('caller', 'managed prompt'),
        ('tool', 'managed prompt'),
    ]
    assert ambient.get(None) is None


async def test_run_context_realtime_is_false_for_classic_run() -> None:
    """`RunContext.realtime` distinguishes a classic run without importing model internals."""

    class ClassicCapability(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            assert not ctx.realtime

    await Agent(TestModel(), capabilities=[ClassicCapability()], deps_type=type(None)).run('hello')


async def test_wrap_run_event_stream_transforms_session_view_only() -> None:
    """The shared stream wrapper sees realtime-only events but cannot rewrite session history."""
    observed: list[AgentStreamEvent] = []

    class TransformStream(AbstractCapability[None]):
        async def wrap_run_event_stream(
            self,
            ctx: RunContext[None],
            *,
            stream: AsyncIterable[AgentStreamEvent],
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                observed.append(event)
                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, SpeechPartDelta):
                    event = replace(event, delta=replace(event.delta, transcript_delta='transformed'))
                yield event

    model = _RecordingModel(
        connection_events=[OutputTranscript(text='original '), OutputTranscript(text='transcript'), ResponseDone()]
    )
    agent = Agent(capabilities=[TransformStream()], deps_type=type(None))

    async with agent.realtime(model).session() as session:
        events = [event async for event in session]

    assert any(isinstance(event, RealtimeTurnCompleteEvent) for event in observed)
    delta = next(event.delta for event in events if isinstance(event, PartDeltaEvent))
    assert isinstance(delta, SpeechPartDelta)
    assert delta.transcript_delta == 'transformed'
    response = session.all_messages()[-1]
    assert isinstance(response.parts[0], SpeechPart)
    assert response.parts[0].transcript == 'original transcript'


async def test_event_stream_handler_receives_session_events() -> None:
    """`ProcessEventStream` provides the agent-level event handler for realtime sessions."""
    observed: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        observed.extend([event async for event in events])

    agent = Agent(capabilities=[ProcessEventStream(handler)], deps_type=type(None))
    await _drain(agent, _RecordingModel(connection_events=[OutputTranscript(text='hello'), ResponseDone()]))

    assert any(isinstance(event, PartStartEvent) for event in observed)
    assert any(isinstance(event, RealtimeTurnCompleteEvent) for event in observed)


async def test_on_event_hook_receives_session_events() -> None:
    """The `Hooks.on.event` convenience callback shares the realtime session vocabulary."""
    observed: list[AgentStreamEvent] = []
    hooks = Hooks()

    @hooks.on.event
    def observe(ctx: RunContext[None], event: AgentStreamEvent) -> AgentStreamEvent:
        observed.append(event)
        return event

    await _drain(
        Agent(capabilities=[hooks], deps_type=type(None)),
        _RecordingModel(connection_events=[OutputTranscript(text='hello'), ResponseDone()]),
    )

    assert any(isinstance(event, PartStartEvent) for event in observed)
    assert any(isinstance(event, RealtimeTurnCompleteEvent) for event in observed)


async def test_session_stream_wrapper_closes_after_early_break() -> None:
    """Stopping session iteration early closes the attached wrapper generator."""
    wrapper_closed = False

    class ClosingStream(AbstractCapability[None]):
        async def wrap_run_event_stream(
            self,
            ctx: RunContext[None],
            *,
            stream: AsyncIterable[AgentStreamEvent],
        ) -> AsyncIterable[AgentStreamEvent]:
            nonlocal wrapper_closed
            try:
                async for event in stream:  # pragma: no branch
                    yield event
            finally:
                wrapper_closed = True

    agent = Agent(capabilities=[ClosingStream()], deps_type=type(None))
    model = _RecordingModel(connection_events=[OutputTranscript(text='hello'), ResponseDone()])

    async with agent.realtime(model).session() as session:
        async for event in session:  # pragma: no branch
            assert isinstance(event, PartStartEvent)
            break

    assert wrapper_closed


async def test_for_run_state_flows_up_from_session_tool_to_after_run() -> None:
    """Mutable state on the `for_run` copy carries tool observations back to `after_run`."""
    checks: list[bool] = []

    class StatefulCapability(AbstractCapability[None]):
        tool_ran = False

        async def after_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, object],
            result: object,
        ) -> object:
            self.tool_ran = True
            return result

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            checks.append(self.tool_ran)
            return result

    agent = Agent(capabilities=[StatefulCapability()], deps_type=type(None))

    @agent.tool_plain
    def mark_state() -> str:
        return 'done'

    model = _RecordingModel(
        connection_events=[ToolCall(tool_call_id='tc_1', tool_name='mark_state', args='{}'), ResponseDone()]
    )
    async with agent.realtime(model).session() as session:
        async for _ in session:
            pass

    assert checks == [True]


async def test_run_context_exposes_realtime_session_and_merged_settings() -> None:
    """`ctx.realtime_session` is the live session once connected — `None` before, when only
    `ctx.realtime` can identify the run — and `ctx.model_settings` holds the merged
    `RealtimeModelSettings` for the whole session."""
    settings = RealtimeModelSettings(output_modality='audio')
    observed: list[object] = []

    class ContextCapability(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            # The session is constructed from the connection, so it cannot exist yet here.
            assert ctx.realtime
            assert ctx.realtime_session is None
            observed.append(dict(ctx.model_settings or {}))

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            observed.append(ctx.realtime_session)
            return result

    agent = Agent(capabilities=[ContextCapability()], deps_type=type(None))

    @agent.tool
    def check_session(ctx: RunContext[None]) -> str:
        observed.append(ctx.realtime_session)
        return 'done'

    model = _RecordingModel(
        settings=settings,
        connection_events=[ToolCall(tool_call_id='tc_1', tool_name='check_session', args='{}'), ResponseDone()],
    )
    async with agent.realtime(model).session() as session:
        async for _ in session:
            pass

    assert observed == [{'output_modality': 'audio'}, session, session]


async def test_external_cancellation_keeps_its_type_and_settles_history() -> None:
    """`task.cancel()` during a session propagates an untranslated `CancelledError` with history settled.

    A unit test because no recorded provider exchange can be interrupted mid-reply on replay. Pins
    the substrate whole-run cancellation (#6497) will attach run state to: external cancellation
    must keep its exception type — `asyncio.timeout()`, TaskGroup teardown, and Temporal all depend
    on that — while session teardown still records the cut-off reply as interrupted, so the
    conversation history survives the cancellation.
    """
    mid_reply = asyncio.Event()

    class _ParkedMidReply(_Connection):
        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            yield OutputTranscript(text='cut off mid-', is_final=False)
            mid_reply.set()
            await asyncio.Event().wait()

    class _ParkedModel(_RecordingModel):
        @asynccontextmanager
        async def connect(
            self,
            *,
            messages: Sequence[ModelMessage],
            model_settings: RealtimeModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> AsyncGenerator[RealtimeConnection]:
            yield _ParkedMidReply()

    sessions: list[RealtimeSession] = []
    agent = Agent(deps_type=type(None))

    async def talk() -> None:
        async with agent.realtime(_ParkedModel()).session() as session:
            sessions.append(session)
            async for _event in session:
                pass

    task = asyncio.create_task(talk())
    await mid_reply.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (session,) = sessions
    assert session.closed
    assert session.result is None
    response = next(message for message in session.all_messages() if isinstance(message, ModelResponse))
    assert response.state == 'interrupted'
    assert any(isinstance(part, SpeechPart) and part.transcript == 'cut off mid-' for part in response.parts)


@pytest.mark.parametrize('supports_tool_updates', [False, True])
async def test_agent_realtime_session_rejects_a_deferred_tool(supports_tool_updates: bool) -> None:
    # A `defer_loading=True` tool that no capability owns is hidden until tool *search* reveals it,
    # and a session drops the optional `ToolSearchTool`. Advertising it silently would hand the model
    # a `search_tools` affordance that finds the tool and then can't have it — a dead end. Being able
    # to re-advertise tools mid-session doesn't help: nothing in a session would ever trigger the
    # reveal. The guard fires before any dial, so no provider is involved.
    agent: Agent[None, str] = Agent()

    @agent.tool_plain(defer_loading=True)
    def withheld_tool() -> str:  # pragma: no cover — the session raises before any tool can run
        return 'withheld'

    with pytest.raises(
        UserError,
        match=r'no tool-search surface.*`defer_loading=True`.*"?\'withheld_tool\'"?',
    ):
        async with agent.realtime(_RecordingModel(supports_tool_updates=supports_tool_updates)).session():
            pass  # pragma: no cover


async def test_session_reveal_of_an_unloaded_capabilitys_tool_ends_the_session() -> None:
    """An ordinary tool revealing a still-unloaded capability's tool ends the session, as it ends a run.

    `ToolReturn.tools` may not smuggle a capability's tool past `load_capability` — the load is what
    activates the bundle's instructions and hooks. The graph raises the same `UserError` out of the
    run; here it surfaces through the session's event stream and the conversation stops.

    Reachable only now that a session can hold a tool-contributing deferred capability at all, and
    pinned rather than endorsed: whether a developer-error reveal should be allowed to end a live
    voice call is a maintainer question, flagged on the PR.
    """
    agent: Agent[None, str] = Agent()

    @agent.tool_plain
    def sneaky() -> ToolReturn[str]:
        return ToolReturn(return_value='done', tools=['forecast'])

    model = _RecordingModel(
        supports_tool_updates=True,
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='sneaky', args='{}'),
            ResponseDone(),
        ],
    )

    with pytest.raises(
        UserError,
        match=r"`ToolReturn.tools` cannot reveal 'forecast'.*belongs to capability 'weather'",
    ):
        await _drain(agent, model, capabilities=[_Weather()])

    assert model.connection is not None
    # The refusal happens before anything is sent, so the model never hears about the reveal.
    assert not any(isinstance(frame, UpdateTools) for frame in model.connection.sent)  # type: ignore[attr-defined]


@pytest.mark.parametrize('supports_tool_updates', [False, True])
async def test_agent_realtime_session_rejects_an_always_on_capabilitys_deferred_tool(
    supports_tool_updates: bool,
) -> None:
    # `defer_loading=True` on a tool inside an *always-on* capability is a tool-search gate, not a
    # capability-load gate — nothing in a session would ever reveal it, so it is refused like any
    # other search-gated tool. Only a tool a *deferred* capability gates has a reveal path.
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(defer_loading=True)
    def searchable() -> str:  # pragma: no cover — the session raises before any tool can run
        return 'searchable'

    class AlwaysOn(AbstractCapability[None]):
        id = 'always_on'

        def get_toolset(self) -> FunctionToolset[None]:
            return toolset

    agent = Agent()
    model = _RecordingModel(supports_tool_updates=supports_tool_updates)

    with pytest.raises(UserError, match=r"no tool-search surface.*'searchable'"):
        await _drain(agent, model, capabilities=[AlwaysOn()])

    assert model.tools is None


@pytest.mark.parametrize('supports_tool_updates', [False, True])
async def test_session_tool_reveal_is_a_no_op_like_a_standard_run(supports_tool_updates: bool) -> None:
    """`ToolReturn.tools` naming nothing revealable is a silent no-op, exactly as in a graph run.

    The graph rejects only a name owned by an unloaded capability and treats every other name — a
    typo, an already-visible tool — as a no-op (see `_reject_unloaded_capability_reveals`, and
    `ToolAvailabilityDeltaPart`'s "silent no-op by design"). The session records the reveal in history
    the same way, and nothing goes out on the wire either way: a model that can't re-advertise tools
    never tries, and one that can finds the advertised list unchanged and sends nothing.

    The repeated name pins the graph's call-level dedupe (`list(dict.fromkeys(tools))`): one name
    reveals once, however many times a single result asks for it.
    """
    agent = Agent()

    @agent.tool_plain
    def revealer() -> ToolReturn[str]:
        return ToolReturn(return_value='done', tools=['hidden_tool', 'hidden_tool'])

    model = _RecordingModel(
        supports_tool_updates=supports_tool_updates,
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='revealer', args='{}'),
            ResponseDone(),
        ],
    )
    events: list[RealtimeEvent] = []
    async with agent.realtime(model).session() as session:
        async for event in session:  # pragma: no branch
            events.append(event)

    assert model.connection is not None
    assert not any(isinstance(frame, UpdateTools) for frame in model.connection.sent)  # type: ignore[attr-defined]

    # Reaching here at all is the point: the reveal no longer raises out of the session.
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.content == 'done'
    assert result.part.outcome == 'success'

    assert [
        part
        for message in session.all_messages()
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ] == snapshot([ToolAvailabilityDeltaPart(tools_added=['hidden_tool'], tool_call_id='tc_1')])
