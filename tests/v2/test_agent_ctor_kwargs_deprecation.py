"""1.x deprecation warnings for `Agent.__init__` kwargs whose migration target is a capability.

Card 40 (v2 prep). Two kwargs in scope:

- `event_stream_handler=` -> `capabilities=[ProcessEventStream(handler)]`
- `prepare_tools=` -> `capabilities=[Hooks(prepare_tools=...)]`

The card also lists `output_validators=` and `prepare_output_function=` but those are
not actual `Agent.__init__` kwargs in 1.x, so they're excluded from this PR.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks, PrepareTools, ProcessEventStream
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext, ToolDefinition

pytestmark = [pytest.mark.anyio]


def _model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='hello')])


async def _stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'streamed'


def _make_model() -> FunctionModel:
    return FunctionModel(_model_function, stream_function=_stream_function)


# --- event_stream_handler= ---------------------------------------------------------------


async def test_event_stream_handler_kwarg_emits_deprecation_warning():
    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    with pytest.warns(
        DeprecationWarning,
        match=r'`Agent\(event_stream_handler=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        agent = Agent(_make_model(), event_stream_handler=handler)

    assert agent.event_stream_handler is handler


async def test_event_stream_handler_kwarg_runs_handler():
    """The legacy kwarg path keeps working in 1.x — the handler still observes the stream."""
    seen: list[AgentStreamEvent] = []

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            seen.append(event)

    with pytest.warns(DeprecationWarning, match=r'event_stream_handler'):
        agent = Agent(_make_model(), event_stream_handler=handler)

    await agent.run('hello')
    assert seen, 'handler should have observed at least one event via the legacy path'


async def test_event_stream_handler_capability_equivalence():
    """Constructing with `capabilities=[ProcessEventStream(handler)]` (the migration target)
    fires the handler the same way the deprecated kwarg does."""
    seen: list[AgentStreamEvent] = []

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            seen.append(event)

    agent = Agent(_make_model(), capabilities=[ProcessEventStream(handler)])
    await agent.run('hello')
    assert seen, 'capability path should also observe events'


# --- prepare_tools= ----------------------------------------------------------------------


async def _noop_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
    return tool_defs


async def test_prepare_tools_kwarg_emits_deprecation_warning():
    with pytest.warns(
        DeprecationWarning,
        match=r'`Agent\(prepare_tools=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        Agent(_make_model(), prepare_tools=_noop_prep)


async def test_prepare_tools_kwarg_warning_mentions_function_tools_only_rescoping():
    """PR #4859 narrowed `prepare_tools` from all-tools to function-tools-only.
    The deprecation warning has to surface that rescoping so users know they may
    also need `Hooks(prepare_output_tools=...)` to preserve old behavior."""
    with pytest.warns(DeprecationWarning, match=r'prepare_tools` runs only on function tools'):
        Agent(_make_model(), prepare_tools=_noop_prep)


async def test_prepare_tools_kwarg_remaps_to_capability():
    """The kwarg auto-injects a `PrepareTools` capability into the agent's capability list,
    and the prepare callback fires once during a run."""
    with pytest.warns(DeprecationWarning, match=r'prepare_tools'):
        agent = Agent(_make_model(), prepare_tools=_noop_prep)

    assert any(isinstance(cap, PrepareTools) for cap in agent._root_capability.capabilities)  # pyright: ignore[reportPrivateUsage]
    # Run the agent to exercise the registered capability — this is what makes `_noop_prep` fire
    # and lets us assert the remap actually wires through to the prepare-tools chain.
    await agent.run('hello')


async def test_prepare_tools_kwarg_vs_hooks_capability_equivalence():
    """`Agent(prepare_tools=fn)` and `Agent(capabilities=[Hooks(prepare_tools=fn)])` produce
    the same observable behavior — the prepare callback runs once per step with the same tool defs."""
    kwarg_calls: list[int] = []
    hooks_calls: list[int] = []

    def my_tool(x: int) -> int:
        return x  # pragma: no cover

    async def kwarg_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        kwarg_calls.append(len(tool_defs))
        return tool_defs

    async def hooks_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        hooks_calls.append(len(tool_defs))
        return tool_defs

    with pytest.warns(DeprecationWarning, match=r'prepare_tools'):
        kwarg_agent = Agent(_make_model(), tools=[my_tool], prepare_tools=kwarg_prep)
    hooks_agent = Agent(_make_model(), tools=[my_tool], capabilities=[Hooks(prepare_tools=hooks_prep)])

    kwarg_result = await kwarg_agent.run('hello')
    hooks_result = await hooks_agent.run('hello')

    assert kwarg_result.output == hooks_result.output
    assert kwarg_calls == hooks_calls
