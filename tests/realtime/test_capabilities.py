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
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import AbstractNativeTool, WebSearchTool
from pydantic_ai.realtime import (
    RealtimeModel,
    RealtimeModelProfile,
    RealtimeModelSettings,
    TurnCompleteEvent,
)
from pydantic_ai.realtime.codec import (
    RealtimeCodecEvent,
    RealtimeConnection,
    RealtimeInput,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

pytestmark = pytest.mark.anyio


class _Connection(RealtimeConnection):
    """Yields a single `TurnCompleteEvent` so the session drains immediately."""

    async def send(self, content: RealtimeInput) -> None:
        pass

    async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
        yield TurnCompleteEvent()


class _RecordingModel(RealtimeModel):
    """A realtime model that records the arguments `realtime_session` passes to `connect`."""

    def __init__(
        self,
        *,
        settings: RealtimeModelSettings | None = None,
        supported_native_tools: frozenset[type[AbstractNativeTool]] = frozenset(),
    ) -> None:
        self.settings = settings
        self._supported = supported_native_tools
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
        yield _Connection()


async def _drain(agent: Agent[None, str], model: _RecordingModel, **kwargs: object) -> None:
    async with agent.realtime(model, **kwargs).session() as session:  # type: ignore[arg-type]
        async for _ in session:  # pragma: no branch - the single event ends the stream
            pass


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
    """A capability native tool the model doesn't support fails up front, before connecting."""
    agent = Agent()
    model = _RecordingModel(supported_native_tools=frozenset())  # supports nothing
    with pytest.raises(UserError, match='does not support'):
        await _drain(agent, model, capabilities=[NativeTool(WebSearchTool())])
    assert model.native_tools is None  # never connected
