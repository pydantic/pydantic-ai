"""Capability *setup* wiring in `Agent.realtime_session`, shared with the graph run.

`realtime_session` and `run`/`iter` resolve capabilities through the same
`Agent._resolve_run_capabilities`, so a capability's setup contributions — instructions, native tools
(including under `override(native_tools=...)`), model settings, and toolsets — must reach a session
exactly as they reach a run. These pin that, guarding against the two silently diverging again (the
session used to drop capability instructions/model-settings and, under a native-tools override, drop a
capability-function's native tools). Network-free: a fake model records what `connect()` receives.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from pydantic_ai import Agent
from pydantic_ai._instrumentation import get_instructions
from pydantic_ai.capabilities import NativeTool, WebSearch
from pydantic_ai.capabilities.abstract import AbstractCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import FunctionToolResultEvent, ModelMessage, ToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, WebSearchTool
from pydantic_ai.realtime import (
    ModelResponseCompleteEvent,
    RealtimeEvent,
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
)
from pydantic_ai.realtime.codec import (
    OutputTranscript,
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
    ToolCall,
)
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

pytestmark = pytest.mark.anyio


class _Connection(RealtimeConnection):
    """Replays a fixed list of events (a lone `ModelResponseCompleteEvent` by default) so the session drains."""

    def __init__(self, events: Sequence[RealtimeCodecEvent] = (ModelResponseCompleteEvent(),)) -> None:
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
        connection_events: Sequence[RealtimeCodecEvent] = (ModelResponseCompleteEvent(),),
        lifecycle: list[str] | None = None,
    ) -> None:
        self.settings = settings
        self._supported = supported_native_tools
        self._connection_events = connection_events
        self._lifecycle = lifecycle
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
        if self._lifecycle is not None:
            self._lifecycle.append('connection opened')
        self.instructions = get_instructions(messages)
        self.tools = model_request_parameters.function_tools
        self.native_tools = model_request_parameters.native_tools
        self.model_settings = model_settings
        try:
            yield _Connection(self._connection_events)
        finally:
            if self._lifecycle is not None:
                self._lifecycle.append('connection closed')


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
            ModelResponseCompleteEvent(),
        ],
    )
    events = await _drain(agent, model, capabilities=[WebSearch(native=WebSearchTool(), local=local_search)])

    assert invoked == ['hello']  # the local callable ran, via the ToolManager
    result_event = next(e for e in events if isinstance(e, FunctionToolResultEvent))
    result_part = result_event.part
    assert isinstance(result_part, ToolReturnPart)
    assert (result_part.tool_name, result_part.content) == ('local_search', 'result for hello')


class _LifecycleCapability(AbstractCapability[None]):
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.allocated = False
        self.result: AgentRunResult[str] | None = None

    async def before_run(self, ctx: RunContext[None]) -> None:
        self.events.append('before run')
        self.allocated = True

    async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[str]) -> AgentRunResult[str]:
        self.events.append('after run')
        self.allocated = False
        self.result = result
        return result

    async def on_run_error(self, ctx: RunContext[None], *, error: BaseException) -> AgentRunResult[str]:
        self.events.append('run error')
        self.allocated = False
        raise error

    async def wrap_run(self, ctx: RunContext[None], *, handler: WrapRunHandler) -> AgentRunResult[str]:
        self.events.append('wrap run')
        raise AssertionError('`wrap_run` must not fire for realtime sessions')


async def test_run_lifecycle_success_and_result() -> None:
    events: list[str] = []
    capability = _LifecycleCapability(events)
    model = _RecordingModel(
        connection_events=[OutputTranscript(text='final answer', is_final=True), ModelResponseCompleteEvent()],
        lifecycle=events,
    )
    agent = Agent(capabilities=[capability], deps_type=type(None))

    async with agent.realtime(model).session() as session:
        events.append('session body')
        async for _ in session:
            pass

    assert events == ['connection opened', 'before run', 'session body', 'after run', 'connection closed']
    assert capability.allocated is False
    assert capability.result is session.result
    assert session.result is not None
    assert session.result.output == 'final answer'
    assert session.result.new_messages() == session.new_messages()


async def test_run_lifecycle_error_releases_and_reraises() -> None:
    events: list[str] = []
    capability = _LifecycleCapability(events)
    model = _RecordingModel(lifecycle=events)
    agent = Agent(capabilities=[capability], deps_type=type(None))

    with pytest.raises(RuntimeError, match='session failed'):
        async with agent.realtime(model).session():
            events.append('session body')
            raise RuntimeError('session failed')

    assert events == ['connection opened', 'before run', 'session body', 'run error', 'connection closed']
    assert capability.allocated is False


async def test_wrap_run_is_inert_for_realtime_session() -> None:
    events: list[str] = []
    capability = _LifecycleCapability(events)
    agent = Agent(capabilities=[capability], deps_type=type(None))

    async with agent.realtime(_RecordingModel()).session() as session:
        pass

    assert events == ['before run', 'after run']
    assert session.result is not None
    assert session.result.output == ''
