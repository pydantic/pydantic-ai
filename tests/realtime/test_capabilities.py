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
from dataclasses import dataclass, replace
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart, LoadCapabilityReturnPart
from pydantic_ai._instrumentation import get_instructions
from pydantic_ai.capabilities import (
    BatchModelRequests,
    Capability,
    CombinedCapability,
    Hooks,
    Instrumentation,
    NativeTool,
    PrefixTools,
    ProcessEventStream,
    WebSearch,
    WrapperCapability,
)
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapOutputProcessHandler,
    WrapRunHandler,
)
from pydantic_ai.exceptions import RunCancelled, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    CapabilityEvent,
    CustomEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool, WebSearchTool
from pydantic_ai.output import OutputContext
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
    merge_realtime_profile,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings, ToolOrOutput
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ..realtime_connections import InferredConnection
from .test_session import (
    FakeRealtimeConnection,
    FakeRealtimeModel,
    _profile,  # pyright: ignore[reportPrivateUsage]
)


class _Connection(InferredConnection):
    """Replays a fixed list of events (a lone `ResponseDone` by default) so the session drains."""

    def __init__(self, events: Sequence[RealtimeCodecEvent] = (ResponseDone(),)) -> None:
        self._events = events
        self.sent: list[RealtimeInput] = []

    async def send_input(self, content: RealtimeInput) -> None:
        self.sent.append(content)

    async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
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
    """A deferred capability whose loading would have to reveal tools fails at session open on a model that can't take them.

    The model keeps the tools the session opened with, so a mid-session load could never make them
    available, and silently loading less than promised is worse than the up-front error. No model
    turns on a native tool mid-session.
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
        match=(
            r"Realtime sessions cannot turn on native tools mid-session.*'deferred'"
            if contribution == 'native_tools'
            else r"The 'gpt-realtime' realtime model keeps the tools a session opened with.*'deferred'"
        ),
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


async def test_load_capability_from_tool_is_rejected_in_a_session() -> None:
    """Loading a capability from code isn't supported in a session; the model's `load_capability` tool is.

    The `UserError` ends the session, like any other `UserError` a session tool raises.
    """

    class Pirate(AbstractCapability[None]):
        id = 'pirate'
        defer_loading = True

        def get_instructions(self) -> str | None:
            return 'Speak like a pirate at all times.'  # pragma: no cover

    agent = Agent()

    @agent.tool
    async def open_case(ctx: RunContext[object]) -> str:
        ctx.load_capability('pirate')
        return 'Case opened.'  # pragma: no cover

    model = _RecordingModel(
        connection_events=[ToolCall(tool_call_id='tc_1', tool_name='open_case', args='{}'), ResponseDone()],
    )

    with pytest.raises(UserError, match=r'`ctx.load_capability\(\)` is not supported in a realtime session\.'):
        await _drain(agent, model, capabilities=[Pirate()])


async def test_deferred_capability_without_instructions_loads_with_only_the_message() -> None:
    """A deferred capability with nothing to say still loads in a session, with no `instructions` in the return."""

    class Quiet(AbstractCapability[None]):
        id = 'quiet'
        defer_loading = True

        def get_description(self) -> str:
            return 'Says nothing.'

    model = _RecordingModel(
        connection_events=[
            ToolCall(tool_call_id='tc_1', tool_name='load_capability', args='{"id": "quiet"}'),
            ResponseDone(),
        ],
    )
    events = await _drain(Agent(), model, capabilities=[Quiet()])

    result_part = next(e for e in events if isinstance(e, FunctionToolResultEvent)).part
    assert isinstance(result_part, ToolReturnPart)
    assert result_part.content == {'message': "Capability 'quiet' loaded. Its instructions, if any, follow."}


async def test_seeded_history_keeps_a_loaded_capabilitys_instructions() -> None:
    """A capability a standard run loaded keeps its instructions in a session seeded with that run's history.

    The run states them in an `InstructionDeltaPart`, which seeding skips, and its load return only confirms the
    load, so the session states them with its own instructions. A capability the agent no longer has is dropped,
    as a standard run withdraws it.
    """

    class Pirate(AbstractCapability[None]):
        id = 'pirate'
        defer_loading = True

        def get_description(self) -> str:
            return 'Talk like a pirate.'

        def get_instructions(self) -> str | None:
            return 'Speak like a pirate at all times.'

    def load_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'pirate'})])
        return ModelResponse(parts=[TextPart('Arr.')])

    result = await Agent(FunctionModel(load_then_answer), deps_type=type(None), capabilities=[Pirate()]).run('Hello.')
    history = result.all_messages()

    model = _RecordingModel()
    await _drain(Agent(instructions='Be helpful.'), model, capabilities=[Pirate()], message_history=history)
    assert model.instructions is not None
    assert model.instructions.endswith('\n\nSpeak like a pirate at all times.')
    assert model.instructions.count('Speak like a pirate') == 1

    model = _RecordingModel()
    await _drain(Agent(instructions='Be helpful.'), model, message_history=history)
    assert model.instructions == 'Be helpful.'


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
        'after_run:final answer',
        'wrap_run:finally',
    ]
    assert session.result is not None
    assert session.result.output == 'final answer'
    assert session.result.new_messages() == session.new_messages()


async def test_closing_generator_that_owns_a_session_drains_wrap_run() -> None:
    """Closing a generator suspended inside a session cancels `wrap_run` and finishes its cleanup first."""
    calls: list[str] = []

    class LifecycleCapability(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            try:
                return await handler()
            except asyncio.CancelledError:
                calls.append('wrap_run:cancelled')
                raise
            finally:
                await asyncio.sleep(0)
                calls.append('wrap_run:finally')

    agent = Agent(capabilities=[LifecycleCapability()], deps_type=type(None))
    sessions: list[RealtimeSession] = []

    async def session_owner() -> AsyncGenerator[None]:
        async with agent.realtime(_RecordingModel(connection_events=[])).session() as session:
            sessions.append(session)
            yield

    generator = session_owner()
    await anext(generator)
    await generator.aclose()
    calls.append('closed')

    assert calls == ['wrap_run:cancelled', 'wrap_run:finally', 'closed']
    assert sessions[0].result is None


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


async def test_caller_error_after_short_circuit_propagates() -> None:
    """A short-circuit skips the inner lifecycle, including `on_run_error`."""

    on_error_called = False

    class ShortCircuitThenRecoverCapability(AbstractCapability[None]):
        async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
            return AgentRunResult(output='short-circuited')

        async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
            nonlocal on_error_called
            on_error_called = True  # pragma: no cover
            return AgentRunResult(output='recovered after short-circuit')  # pragma: no cover

    agent = Agent(capabilities=[ShortCircuitThenRecoverCapability()], deps_type=type(None))

    with pytest.raises(RuntimeError, match='caller failed'):
        async with agent.realtime(_RecordingModel()).session() as session:
            assert session.closed
            raise RuntimeError('caller failed')

    assert not on_error_called


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

    # The instructions are resolved at connect, then again before the tool's result and after the response.
    assert seen[:2] == [('instructions', 'managed prompt'), ('caller', 'managed prompt')]
    assert sorted(seen[2:]) == [
        ('instructions', 'managed prompt'),
        ('instructions', 'managed prompt'),
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


@dataclass
class _Booked(CustomEvent):
    where: str = ''


@dataclass
class _SnapshotSaved(CapabilityEvent, namespace='realtime_test_snapshots'):
    pass


async def test_emitted_events_reach_the_session_stream() -> None:
    """`ctx.emit` from run hooks and tools works in a session as in `iter()`: events emitted before the
    session connects lead its stream, a tool's land between its call and result events, and both reach
    `on_event`. An event emitted from `after_run`, once the stream has ended, is accepted but not
    delivered, also as in `iter()`."""
    hooks = Hooks()
    observed: list[str] = []
    after_run_emitted: list[_Booked] = []

    @hooks.on.before_run
    async def before_run(ctx: RunContext[None]) -> None:
        await ctx.emit(_Booked(where='before_run'))

    @hooks.on.after_run
    async def after_run(ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        after_run_emitted.append(await ctx.emit(_Booked(where='after_run')))
        return result

    @hooks.on.event
    def observe(ctx: RunContext[None], event: AgentStreamEvent) -> AgentStreamEvent:
        if isinstance(event, _Booked):
            observed.append(event.where)
        return event

    agent = Agent(capabilities=[hooks], deps_type=type(None))

    @agent.tool
    async def book(ctx: RunContext[None]) -> str:
        await ctx.emit(_Booked(where='tool'))
        return 'booked'

    events = await _drain(
        agent,
        _RecordingModel(connection_events=[ToolCall(tool_call_id='tc_1', tool_name='book', args='{}'), ResponseDone()]),
    )

    assert [type(event) for event in events] == [
        _Booked,
        PartStartEvent,
        PartEndEvent,
        FunctionToolCallEvent,
        _Booked,
        FunctionToolResultEvent,
    ]
    assert [event.where for event in events if isinstance(event, _Booked)] == ['before_run', 'tool']
    tool_event = events[4]
    assert isinstance(tool_event, _Booked)
    assert (tool_event.tool_call_id, tool_event.tool_name) == ('tc_1', 'book')
    assert observed == ['before_run', 'tool']
    assert [event.where for event in after_run_emitted] == ['after_run']


async def test_capability_event_emitted_from_after_run_closes_session_cleanly() -> None:
    """A capability that records a final snapshot in `after_run` and emits an event about it (as Harness
    `StepPersistence` does) no longer makes every normal session close raise."""

    class Snapshotting(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            await ctx.emit(_SnapshotSaved())
            return result

    assert await _drain(Agent(capabilities=[Snapshotting()], deps_type=type(None)), _RecordingModel()) == []


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


async def test_after_run_uncancel_cannot_complete_realtime_session() -> None:
    """A first-party cancellation remains terminal after a hook clears the owning task's cancellation count."""
    task = asyncio.current_task()
    assert task is not None

    class UncancelAfterRun(AbstractCapability[None]):
        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
            ctx.cancel()
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                task.uncancel()
            return result

    agent = Agent(capabilities=[UncancelAfterRun()], deps_type=type(None))
    with pytest.raises(RunCancelled):
        async with agent.realtime(_RecordingModel()).session() as session:
            async for _ in session:
                pass


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


class _RecoverCancellationInWrapRun(AbstractCapability[None]):
    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
        try:
            return await handler()
        except asyncio.CancelledError:
            return AgentRunResult(output='recovered')


class _RecoverCancellationInRunError(AbstractCapability[None]):
    async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
        return AgentRunResult(output='recovered')


@pytest.mark.parametrize('capability', [_RecoverCancellationInRunError(), _RecoverCancellationInWrapRun()])
async def test_run_hooks_cannot_recover_realtime_cancellation(capability: AbstractCapability[None]) -> None:
    """A unit test because cancellation recovery is in-process lifecycle control flow, not provider behavior."""
    agent = Agent[None, str](capabilities=[capability], deps_type=type(None))

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
        async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
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
    # A `defer_loading=True` tool is hidden until tool search reveals it, which a session on a model that
    # keeps the tools it opened with can never do. Advertising it silently would hand the model a `search_tools`
    # affordance that finds the tool and then can't have it — a dead end — so the session refuses to
    # open, exactly as it does for a deferred *capability* that contributes tools. The guard fires
    # before any dial, so no provider is involved.
    agent: Agent[None, str] = Agent()

    @agent.tool_plain(defer_loading=True)
    def withheld_tool() -> str:  # pragma: no cover — the session raises before any tool can run
        return 'withheld'

    with pytest.raises(
        UserError,
        match=r"keeps the tools a session opened with, so tools with `defer_loading=True` are not supported.*'withheld_tool'",
    ):
        async with agent.realtime(_RecordingModel()).session():
            pass  # pragma: no cover


async def test_session_tool_reveal_is_a_no_op_like_a_standard_run() -> None:
    """`ToolReturn.tools` naming nothing revealable is a silent no-op, exactly as in a graph run.

    The graph rejects only a name owned by an unloaded capability and treats every other name — a
    typo, an already-visible tool — as a no-op (see `_reject_unloaded_capability_reveals`, and
    `ToolAvailabilityDeltaPart`'s "silent no-op by design").
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


# --- Tools revealed mid-session --------------------------------------------------------------
#
# Loading a deferred capability, a `search_tools` match, or `ToolReturn.tools` shows the model new tools
# mid-session on a model whose profile sets `supports_tool_updates`, relaxing the rejections above.


_UPDATES = merge_realtime_profile(_profile(), RealtimeModelProfile(supports_tool_updates=True))
_NO_UPDATES = merge_realtime_profile(_profile(), RealtimeModelProfile(supports_tool_updates=False))


class _ScriptedConnection(FakeRealtimeConnection):
    """A connection whose events the test pushes as the conversation goes, the way a provider answers what it's sent."""

    def __init__(self) -> None:
        super().__init__([])
        self._queue: asyncio.Queue[RealtimeCodecEvent] = asyncio.Queue()

    def push(self, *events: RealtimeCodecEvent) -> None:
        for event in events:
            self._queue.put_nowait(event)

    async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
        while True:
            yield await self._queue.get()


def _call(tool_name: str, args: str = '{}', *, call_id: str) -> list[RealtimeCodecEvent]:
    return [ToolCall(tool_call_id=call_id, tool_name=tool_name, args=args), ResponseDone()]


_ANSWER: list[RealtimeCodecEvent] = [OutputTranscript(text='Done.', is_final=True), ResponseDone()]


async def _converse(
    session: RealtimeSession, connection: _ScriptedConnection, turns: Sequence[list[RealtimeCodecEvent]]
) -> list[RealtimeEvent]:
    """Push each turn once the tool results of the one before are in, and stop after the last turn completes."""
    pending = list(turns)
    connection.push(*pending.pop(0))
    events: list[RealtimeEvent] = []
    with anyio.fail_after(5):
        async for event in session:  # pragma: no branch
            events.append(event)
            if isinstance(event, FunctionToolResultEvent) and pending:
                connection.push(*pending.pop(0))
            elif isinstance(event, RealtimeTurnCompleteEvent) and not pending:
                break
    return events


def _forecast_capability() -> Capability[None]:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    async def get_forecast(city: str) -> str:
        """Look up the weather forecast for a city."""
        return f'Sunny in {city}.'

    return Capability(id='weather', description='Weather forecasts.', toolsets=[toolset], defer_loading=True)


def _sent_groups(connection: FakeRealtimeConnection) -> list[type]:
    return [type(content) for content in connection.sent]


async def test_loading_a_capability_reveals_its_tools() -> None:
    """The capability's tools go out ahead of the load's result, in the same send, and the model can call them."""
    connection = _ScriptedConnection()
    model = FakeRealtimeModel(connection, profile=_UPDATES)
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])

    async with agent.realtime(model).session() as session:
        events = await _converse(
            session,
            connection,
            [
                _call('load_capability', '{"id": "weather"}', call_id='load'),
                _call('get_forecast', '{"city": "Paris"}', call_id='forecast'),
                _ANSWER,
            ],
        )

    assert model.last_tools is not None
    assert [tool.name for tool in model.last_tools] == ['load_capability']
    update, load_result, forecast_result = connection.sent
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['load_capability', 'get_forecast']
    assert isinstance(load_result, ToolResult) and load_result.tool_call_id == 'load'
    assert isinstance(forecast_result, ToolResult) and forecast_result.output == 'Sunny in Paris.'
    results = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert [part.outcome for part in results] == ['success', 'success']

    messages = session.all_messages()
    load_call = messages[0].parts[0]
    assert isinstance(load_call, LoadCapabilityCallPart)
    assert load_call.capability_id == 'weather'
    load_request = messages[1]
    assert isinstance(load_request, ModelRequest)
    assert isinstance(load_request.parts[0], LoadCapabilityReturnPart)
    assert load_request.parts[1] == ToolAvailabilityDeltaPart(tools_added=['get_forecast'], tool_call_id='load')


async def test_a_loaded_capabilitys_prepare_tools_shapes_the_update() -> None:
    """The update carries the tools as the loaded capability's own `prepare_tools` hook prepares them, as a run's next request does."""
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    async def get_forecast(city: str) -> str:  # pragma: no cover
        """Look up the weather forecast for a city."""
        return f'Sunny in {city}.'

    class Weather(AbstractCapability[None]):
        id = 'weather'
        defer_loading = True

        def get_description(self) -> str:
            return 'Weather forecasts.'

        def get_toolset(self) -> FunctionToolset[None]:
            return toolset

        async def prepare_tools(self, ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [replace(tool_def, description='Prepared description.') for tool_def in tool_defs]

    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[Weather()])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "weather"}', call_id='load'), _ANSWER])

    update = connection.sent[0]
    assert isinstance(update, UpdateTools)
    assert [(tool.name, tool.description) for tool in update.tools][1:] == [('get_forecast', 'Prepared description.')]


async def test_a_standard_run_continues_a_session_that_revealed_tools() -> None:
    """History from a session that loaded a capability is history a standard run treats as loaded too."""
    connection = _ScriptedConnection()
    capability = _forecast_capability()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[capability])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "weather"}', call_id='load'), _ANSWER])

    seen: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.extend(tool.name for tool in info.function_tools)
        return ModelResponse(parts=[TextPart('Sunny.')])

    result = await agent.run('And tomorrow?', model=FunctionModel(respond), message_history=session.all_messages())
    assert result.output == 'Sunny.'
    assert seen == ['load_capability', 'get_forecast']


async def test_a_seeded_load_advertises_its_tools_at_connect_on_any_model() -> None:
    """A capability the history already loaded needs no update: its tools go out when the session opens."""
    capability = _forecast_capability()

    def load_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'weather'})])
        return ModelResponse(parts=[TextPart('Loaded.')])

    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[capability])
    history = (await agent.run('Hi.', model=FunctionModel(load_then_answer))).all_messages()

    connection = _ScriptedConnection()
    model = FakeRealtimeModel(connection, profile=_NO_UPDATES)
    async with agent.realtime(model, message_history=history).session() as session:
        events = await _converse(
            session, connection, [_call('get_forecast', '{"city": "Oslo"}', call_id='forecast'), _ANSWER]
        )

    assert model.last_tools is not None
    assert [tool.name for tool in model.last_tools] == ['load_capability', 'get_forecast']
    (result,) = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert result.content == 'Sunny in Oslo.'
    assert not any(isinstance(content, UpdateTools) for content in connection.sent)


async def test_loading_an_already_loaded_capability_is_refused() -> None:
    """A session knows what it loaded, so a second load is refused as in a standard run, and reveals nothing again."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        events = await _converse(
            session,
            connection,
            [
                _call('load_capability', '{"id": "weather"}', call_id='first'),
                _call('load_capability', '{"id": "weather"}', call_id='second'),
                _ANSWER,
            ],
        )

    results = [event.part for event in events if isinstance(event, FunctionToolResultEvent)]
    assert [part.outcome for part in results] == ['success', 'retried']
    assert _sent_groups(connection) == [UpdateTools, ToolResult, ToolResult]


async def test_tool_search_reveals_a_deferred_tool() -> None:
    """A `defer_loading=True` tool is found by `search_tools`, and goes out with its result."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain(defer_loading=True)
    async def get_stock_price(ticker: str) -> str:
        """Look up the current stock price for a ticker symbol."""
        return f'{ticker} is at 41.'

    model = FakeRealtimeModel(connection, profile=_UPDATES)
    async with agent.realtime(model).session() as session:
        await _converse(
            session,
            connection,
            [
                _call('search_tools', '{"queries": ["stock price"]}', call_id='search'),
                _call('get_stock_price', '{"ticker": "ACME"}', call_id='price'),
                _ANSWER,
            ],
        )

    assert model.last_tools is not None
    assert [tool.name for tool in model.last_tools] == ['search_tools']
    update, _, price = connection.sent
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['search_tools', 'get_stock_price']
    assert isinstance(price, ToolResult) and price.output == 'ACME is at 41.'
    # A search's result records its own matches, so no `ToolAvailabilityDeltaPart` is added for it.
    search_request = session.all_messages()[1]
    assert isinstance(search_request, ModelRequest)
    assert not any(isinstance(part, ToolAvailabilityDeltaPart) for part in search_request.parts)


async def test_a_tool_return_reveals_a_deferred_tool_once() -> None:
    """`ToolReturn.tools` reveals a tool as in a standard run, and a second reveal of it changes nothing."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain
    async def unlock() -> ToolReturn[str]:
        return ToolReturn(return_value='Unlocked.', tools=['secret', 'secret', 'no_such_tool'])

    @agent.tool_plain(defer_loading=True)
    async def secret() -> str:
        return 'The secret.'

    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(
            session,
            connection,
            [
                _call('unlock', call_id='first'),
                _call('unlock', call_id='second'),
                _call('secret', call_id='call'),
                _ANSWER,
            ],
        )

    assert _sent_groups(connection) == [UpdateTools, ToolResult, ToolResult, ToolResult]
    update = connection.sent[0]
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['unlock', 'search_tools', 'secret']
    reveals = [
        part
        for message in session.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    # As the graph records it: once, with the name that resolves to nothing kept.
    assert reveals == [ToolAvailabilityDeltaPart(tools_added=['secret', 'no_such_tool'], tool_call_id='first')]
    secret_result = connection.sent[-1]
    assert isinstance(secret_result, ToolResult) and secret_result.output == 'The secret.'


async def test_a_revealed_tool_outside_the_tool_choice_stays_hidden() -> None:
    """The update is narrowed by `tool_choice`, as the tools the session opened with were."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])

    @agent.tool_plain
    async def other() -> str:  # pragma: no cover
        return 'other'

    settings = RealtimeModelSettings(tool_choice=ToolOrOutput(function_tools=['load_capability', 'other']))
    model = FakeRealtimeModel(connection, profile=_UPDATES, settings=settings)
    async with agent.realtime(model).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "weather"}', call_id='load'), _ANSWER])

    update = connection.sent[0]
    assert isinstance(update, UpdateTools)
    assert [tool.name for tool in update.tools] == ['load_capability', 'other']


async def test_revealing_an_unloaded_capabilitys_tool_is_refused() -> None:
    """A tool another capability owns is revealed by loading that capability, as in a standard run."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[_forecast_capability()])

    @agent.tool_plain
    async def peek() -> ToolReturn[str]:
        return ToolReturn(return_value='Peeked.', tools=['get_forecast'])

    with pytest.raises(UserError, match=r"`ToolReturn\.tools` cannot reveal 'get_forecast'"):
        async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
            await _converse(session, connection, [_call('peek', call_id='peek'), _ANSWER])


async def test_a_reveal_a_model_cant_take_warns_once() -> None:
    """On a model that keeps the tools it opened with, a reveal the open couldn't foresee is recorded, and warned about."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))
    revealable = {'ready': False}

    @agent.toolset(per_run_step=True)
    async def late_tools(ctx: RunContext[None]) -> FunctionToolset[None] | None:
        if not revealable['ready']:
            return None
        toolset = FunctionToolset[None]()

        @toolset.tool_plain(defer_loading=True)
        async def late() -> str:  # pragma: no cover
            return 'late'

        return toolset

    @agent.tool_plain
    async def reveal() -> ToolReturn[str]:
        return ToolReturn(return_value='Revealed.', tools=['late'])

    async with agent.realtime(FakeRealtimeModel(connection, profile=_NO_UPDATES)).session() as session:
        revealable['ready'] = True
        with pytest.warns(
            UserWarning, match=r"A tool call revealed 'late'.*`supports_tool_updates` is `False`"
        ) as record:
            await _converse(
                session,
                connection,
                [
                    # Ahead of the step the tool joins in: a name that resolves to nothing yet changes nothing.
                    _call('reveal', call_id='first'),
                    _call('reveal', call_id='second'),
                    _call('reveal', call_id='third'),
                    _ANSWER,
                ],
            )

    assert len([warning for warning in record if 'revealed' in str(warning.message)]) == 1
    assert _sent_groups(connection) == [ToolResult, ToolResult, ToolResult]
    first_request = session.all_messages()[1]
    assert isinstance(first_request, ModelRequest)
    assert first_request.parts[1] == ToolAvailabilityDeltaPart(tools_added=['late'], tool_call_id='first')


async def test_an_instruction_only_capability_needs_no_update() -> None:
    """Loading a capability with nothing to reveal sends no update, on any model."""

    class Pirate(AbstractCapability[None]):
        id = 'pirate'
        defer_loading = True

        def get_instructions(self) -> str:
            return 'Speak like a pirate.'

    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None), capabilities=[Pirate()])
    async with agent.realtime(FakeRealtimeModel(connection, profile=_NO_UPDATES)).session() as session:
        await _converse(session, connection, [_call('load_capability', '{"id": "pirate"}', call_id='load'), _ANSWER])

    assert _sent_groups(connection) == [ToolResult]
    request = session.all_messages()[1]
    assert isinstance(request, ModelRequest)
    assert [type(part) for part in request.parts] == [LoadCapabilityReturnPart]


async def test_tools_still_unrevealed_on_a_model_without_updates_are_refused_at_open() -> None:
    """Of the deferred tools, only those history didn't reveal make a session on a model without updates refuse to open."""
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain(defer_loading=True)
    async def revealed() -> str:  # pragma: no cover
        return 'revealed'

    @agent.tool_plain(defer_loading=True)
    async def unrevealed() -> str:  # pragma: no cover
        return 'unrevealed'

    history: list[ModelMessage] = [ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['revealed'])])]
    model = FakeRealtimeModel(FakeRealtimeConnection([]), profile=_NO_UPDATES)
    with pytest.raises(UserError, match=r"`defer_loading=True` are not supported; remove it from: 'unrevealed'\."):
        async with agent.realtime(model, message_history=history).session():
            pass  # pragma: no cover


async def test_revealing_a_tool_the_model_has_changes_nothing() -> None:
    """Naming a tool the model can already call reveals nothing new, so no update goes out, as in a standard run."""
    connection = _ScriptedConnection()
    agent: Agent[None, str] = Agent(deps_type=type(None))

    @agent.tool_plain
    async def point() -> ToolReturn[str]:
        return ToolReturn(return_value='Pointed.', tools=['point'])

    async with agent.realtime(FakeRealtimeModel(connection, profile=_UPDATES)).session() as session:
        await _converse(session, connection, [_call('point', call_id='point'), _ANSWER])

    assert _sent_groups(connection) == [ToolResult]


# --- Hooks a session never calls ---------------------------------------------------------------


class _ModelRequestCap(AbstractCapability[None]):
    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return request_context  # pragma: no cover


class _GraphCap(AbstractCapability[None]):
    async def after_node_run(
        self, ctx: RunContext[None], *, node: AgentNode[None], result: NodeResult[None]
    ) -> NodeResult[None]:
        return result  # pragma: no cover

    async def wrap_output_process(
        self, ctx: RunContext[None], *, output_context: OutputContext, output: Any, handler: WrapOutputProcessHandler
    ) -> Any:
        return await handler(output)  # pragma: no cover


class _ToolHookCap(AbstractCapability[None]):
    def get_instructions(self) -> str:
        return 'Be brief.'

    async def before_tool_execute(
        self, ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        return args  # pragma: no cover


class _ModelRequestWrapper(WrapperCapability[None]):
    async def wrap_model_request(
        self, ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
    ) -> ModelResponse:
        return await handler(request_context)  # pragma: no cover


async def _on_model_request_error(
    ctx: RunContext[None], *, request_context: ModelRequestContext, error: Exception
) -> ModelResponse:
    raise error  # pragma: no cover


async def _before_tool_execute(
    ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
) -> dict[str, Any]:
    return args  # pragma: no cover


def _uncalled_hooks_warning(name: str, hooks: str) -> str:
    return (
        f'Capability {name} implements {hooks}, which realtime sessions never call, so it may not work as '
        'intended in this session. See https://pydantic.dev/docs/ai/realtime/capabilities/ for the hooks a '
        'session runs.'
    )


async def test_capability_with_uncalled_hook_warns() -> None:
    """A capability relying on `before_model_request` does nothing in a session, so opening one says so."""
    with pytest.warns(UserWarning) as record:
        await _drain(Agent[None, str](capabilities=[_ModelRequestCap()], deps_type=type(None)), _RecordingModel())
    assert [str(warning.message) for warning in record] == [
        _uncalled_hooks_warning('`_ModelRequestCap`', '`before_model_request`')
    ]


async def test_prepare_model_request_hook_warns() -> None:
    """A session makes no model request attempts, so a `prepare_model_request` hook never runs."""

    async def prepare(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        return request_context  # pragma: no cover

    with pytest.warns(UserWarning) as record:
        await _drain(
            Agent[None, str](capabilities=[Hooks(prepare_model_request=prepare)], deps_type=type(None)),
            _RecordingModel(),
        )
    assert [str(warning.message) for warning in record] == [
        _uncalled_hooks_warning('`Hooks`', '`prepare_model_request`')
    ]


async def test_batching_warns_that_a_session_sends_no_model_requests() -> None:
    """A session sends no model requests, so `BatchModelRequests`' `get_wrapper_model` has nothing to wrap."""
    with pytest.warns(UserWarning) as record:
        await _drain(Agent[None, str](capabilities=[BatchModelRequests()], deps_type=type(None)), _RecordingModel())
    assert [str(warning.message) for warning in record] == [
        _uncalled_hooks_warning('`BatchModelRequests`', '`get_wrapper_model`')
    ]


async def test_one_uncalled_hooks_warning_per_capability() -> None:
    """Each capability warns once per session, naming all of its uncalled hooks; nested ones are reached too."""
    graph_cap = _GraphCap()
    agent = Agent[None, str](
        capabilities=[
            graph_cap,
            CombinedCapability([graph_cap, Hooks(model_request_error=_on_model_request_error, id='errors')]),
            PrefixTools(wrapped=_ModelRequestCap(), prefix='x'),
            _ModelRequestWrapper(wrapped=_ToolHookCap()),
        ],
        deps_type=type(None),
    )
    for _ in range(2):
        with pytest.warns(UserWarning) as record:
            await _drain(agent, _RecordingModel())
        assert sorted(str(warning.message) for warning in record) == sorted(
            [
                _uncalled_hooks_warning('`_GraphCap`', '`after_node_run`, `wrap_output_process`'),
                _uncalled_hooks_warning("`Hooks` (id='errors')", '`on_model_request_error`'),
                _uncalled_hooks_warning('`_ModelRequestCap`', '`before_model_request`'),
                _uncalled_hooks_warning('`_ModelRequestWrapper`', '`wrap_model_request`'),
            ]
        )


class _ModelRequestInstrumentation(Instrumentation):
    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        return response  # pragma: no cover


async def test_subclass_of_exempt_built_in_warns_for_its_own_hooks() -> None:
    """An exempt built-in is measured against its own class, so hooks a subclass adds still warn."""
    with pytest.warns(UserWarning) as record:
        await _drain(
            Agent[None, str](capabilities=[_ModelRequestInstrumentation()], deps_type=type(None)), _RecordingModel()
        )
    assert [str(warning.message) for warning in record] == [
        _uncalled_hooks_warning("`_ModelRequestInstrumentation` (id='instrumentation')", '`after_model_request`')
    ]


@pytest.mark.parametrize(
    'capabilities',
    [
        pytest.param([], id='default-agent'),
        pytest.param([Instrumentation()], id='instrumentation'),
        pytest.param([_ToolHookCap()], id='supported-hooks'),
        pytest.param([Hooks(before_tool_execute=_before_tool_execute)], id='hooks-supported'),
        pytest.param([PrefixTools(wrapped=_ToolHookCap(), prefix='x')], id='wrapper-supported'),
        pytest.param(
            [PrefixTools(wrapped=CombinedCapability([_ToolHookCap(), Instrumentation()]), prefix='x')],
            id='wrapper-over-combined-supported',
        ),
    ],
)
async def test_capability_with_only_session_hooks_does_not_warn(capabilities: list[AbstractCapability[None]]) -> None:
    """Built-ins a session handles another way, and capabilities using only session hooks, open silently.

    The default agent auto-injects `ToolSearch` and the pending-message drain, which both implement
    `before_model_request`; `filterwarnings = error` turns any warning into a failure here.
    """
    await _drain(Agent[None, str](capabilities=capabilities, deps_type=type(None)), _RecordingModel())
