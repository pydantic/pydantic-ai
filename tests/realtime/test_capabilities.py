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

from pydantic_ai import Agent
from pydantic_ai._instrumentation import get_instructions
from pydantic_ai.capabilities import Hooks, NativeTool, ProcessEventStream, WebSearch
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import RunCancelled, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
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
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

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
        connection_events: Sequence[RealtimeCodecEvent] = (ResponseDone(),),
    ) -> None:
        self.settings = settings
        self._supported = supported_native_tools
        self._connection_events = connection_events
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
        yield _Connection(self._connection_events)


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


@pytest.mark.parametrize('contribution', ['tools', 'native_tools'])
async def test_deferred_capability_with_tools_raises_before_connect(contribution: str) -> None:
    """A deferred capability whose loading would have to reveal tools fails at session open.

    A session's tools are fixed when the connection opens, so a mid-session load could never make
    them available — silently loading less than promised is worse than the up-front error.
    """
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def greet() -> str:  # pragma: no cover — the session raises before any tool can run
        return 'Hello!'

    class DeferredCap(AbstractCapability[None]):
        id = 'deferred'
        defer_loading = True

        def get_toolset(self) -> FunctionToolset[None] | None:
            return toolset if contribution == 'tools' else None

        def get_native_tools(self) -> Sequence[AbstractNativeTool]:
            return [WebSearchTool()] if contribution == 'native_tools' else []

    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset({WebSearchTool}))

    with pytest.raises(
        UserError,
        match=r"Realtime sessions cannot reveal tools mid-session.*'deferred'",
    ):
        await _drain(agent, model, capabilities=[DeferredCap()])

    assert model.tools is None


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


async def test_before_run_can_cancel_realtime_session() -> None:
    """A unit test because cancellation before connection setup has no provider exchange to record."""

    class CancelBeforeRun(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            ctx.cancel()

    model = _RecordingModel()
    agent = Agent(capabilities=[CancelBeforeRun()], deps_type=type(None))

    with pytest.raises(RunCancelled) as exc_info:
        async with agent.realtime(model).session():
            pass

    assert exc_info.value.all_messages() == []
    assert model.instructions is None


async def test_cancel_on_finished_realtime_context_is_noop() -> None:
    """A retained context cannot cancel the task that happened to own an already-finished session."""
    contexts: list[RunContext[None]] = []

    class RetainContext(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            contexts.append(ctx)

    agent = Agent(capabilities=[RetainContext()], deps_type=type(None))
    async with agent.realtime(_RecordingModel(connection_events=[ResponseDone()])).session() as session:
        _ = [event async for event in session]

    (ctx,) = contexts
    ctx.cancel()
    assert await asyncio.sleep(0, result='unrelated work') == 'unrelated work'


async def test_nested_run_cancellation_in_before_run_uses_realtime_history() -> None:
    """A unit test because the behavior is lifecycle exception translation before any provider exchange."""
    inner_agent = Agent[None, str](TestModel(call_tools=['cancel_inner']), deps_type=type(None))

    @inner_agent.tool
    def cancel_inner(ctx: RunContext[None]) -> str:
        ctx.cancel()
        return 'discarded'

    class RunNested(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            await inner_agent.run('inner')

    outer_history = [ModelRequest(parts=[UserPromptPart('outer')])]
    outer_agent = Agent(capabilities=[RunNested()], deps_type=type(None))

    with pytest.raises(RunCancelled) as exc_info:
        async with outer_agent.realtime(_RecordingModel(), message_history=outer_history).session():
            pass

    assert exc_info.value.all_messages() == outer_history
    assert isinstance(exc_info.value.__cause__, RunCancelled)


async def test_run_error_hook_cannot_recover_realtime_cancellation() -> None:
    """A unit test because cancellation recovery is in-process lifecycle control flow, not provider behavior."""

    class RecoverCancellation(AbstractCapability[None]):
        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            return AgentRunResult(output='recovered')

    agent = Agent[None, str](capabilities=[RecoverCancellation()], deps_type=type(None))

    @agent.tool
    async def cancel(ctx: RunContext[None]) -> None:
        ctx.cancel()
        await asyncio.Event().wait()

    model = _RecordingModel(
        connection_events=[ToolCall(tool_call_id='tc', tool_name='cancel', args='{}'), ResponseDone()]
    )
    with pytest.raises(RunCancelled):
        async with agent.realtime(model).session() as session:
            async for _ in session:
                pass


async def test_external_cancellation_keeps_its_type_and_settles_history() -> None:
    """`task.cancel()` during a session propagates an untranslated `CancelledError` with history settled.

    A unit test because no recorded provider exchange can be interrupted mid-reply on replay. Pins
    the whole-run cancellation substrate attaches run state to: external cancellation
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
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task

    (session,) = sessions
    assert session.closed
    assert session.result is None
    response = next(message for message in session.all_messages() if isinstance(message, ModelResponse))
    assert response.state == 'interrupted'
    assert any(isinstance(part, SpeechPart) and part.transcript == 'cut off mid-' for part in response.parts)
    cancelled = RunCancelled.from_cancellation(exc_info.value)
    assert cancelled is not None
    assert cancelled.all_messages() == session.all_messages()


async def test_agent_realtime_session_rejects_a_deferred_tool() -> None:
    # A `defer_loading=True` tool is hidden until tool search reveals it, which a session whose tools
    # are fixed at connect can never do. Advertising it silently would hand the model a `search_tools`
    # affordance that finds the tool and then can't have it — a dead end — so the session refuses to
    # open, exactly as it does for a deferred *capability* that contributes tools. The guard fires
    # before any dial, so no provider is involved.
    agent: Agent[None, str] = Agent()

    @agent.tool_plain(defer_loading=True)
    def withheld_tool() -> str:  # pragma: no cover — the session raises before any tool can run
        return 'withheld'

    with pytest.raises(
        UserError,
        match=r'cannot reveal tools mid-session.*`defer_loading=True`.*"?\'withheld_tool\'"?',
    ):
        async with agent.realtime(_RecordingModel()).session():
            pass  # pragma: no cover


async def test_session_tool_reveal_is_a_no_op_like_a_standard_run() -> None:
    """`ToolReturn.tools` naming nothing revealable is a silent no-op, exactly as in a graph run.

    The graph rejects only a name owned by an unloaded capability and treats every other name — a
    typo, an already-visible tool — as a no-op (see `_reject_unloaded_capability_reveals`, and
    `ToolAvailabilityDeltaPart`'s "silent no-op by design"). A session used to raise on *any* reveal,
    which was both harsher than the graph and harsher than it needed to be: a session refuses
    `defer_loading=True` tools and tool-contributing deferred capabilities when it opens, so it holds
    nothing a reveal could have surfaced and nothing is lost by dropping the request.

    That also makes the unloaded-capability branch unreachable from a session — the capabilities that
    own hidden tools are exactly the ones a session refuses at connect — so it is covered on the graph
    side rather than here.
    """
    agent = Agent()

    @agent.tool_plain
    def revealer() -> ToolReturn[str]:
        return ToolReturn(return_value='done', tools=['hidden_tool'])

    model = _RecordingModel(
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='revealer', args='{}'),
            ResponseDone(),
        ],
    )
    events = await _drain(agent, model)

    # Reaching here at all is the point: the reveal no longer raises out of the session.
    result = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    assert isinstance(result.part, ToolReturnPart)
    assert result.part.content == 'done'
    assert result.part.outcome == 'success'
