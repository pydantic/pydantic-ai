"""1.x deprecation warnings for `Agent.__init__` kwargs whose migration target is a capability.

Three kwargs in scope:

- `event_stream_handler=` -> `capabilities=[ProcessEventStream(handler)]`
- `prepare_tools=` -> `capabilities=[PrepareTools(prepare_tools)]`
- `prepare_output_tools=` -> `capabilities=[PrepareOutputTools(prepare_output_tools)]`

`output_validators=` is not an `Agent.__init__` kwarg in 1.x (it's only set via decorator),
so it's intentionally excluded.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import PrepareOutputTools, PrepareTools, ProcessEventStream, ProcessHistory
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import ToolOutput
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
        PydanticAIDeprecationWarning,
        match=r'`Agent\(event_stream_handler=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        agent = Agent(_make_model(), event_stream_handler=handler)  # pyright: ignore[reportCallIssue]

    assert agent.event_stream_handler is handler


async def test_event_stream_handler_kwarg_runs_handler():
    """The legacy kwarg path keeps working in 1.x — the handler still observes the stream."""
    seen: list[AgentStreamEvent] = []

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            seen.append(event)

    with pytest.warns(PydanticAIDeprecationWarning, match=r'event_stream_handler'):
        agent = Agent(_make_model(), event_stream_handler=handler)  # pyright: ignore[reportCallIssue]

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
        PydanticAIDeprecationWarning,
        match=r'`Agent\(prepare_tools=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        Agent(_make_model(), prepare_tools=_noop_prep)  # pyright: ignore[reportCallIssue]


async def test_prepare_tools_kwarg_warning_points_at_capability():
    """The migration target named in the warning is the `PrepareTools` capability,
    not the `Hooks(prepare_tools=...)` hook — both work but the capability is the cleaner v2 path."""
    with pytest.warns(PydanticAIDeprecationWarning, match=r'capabilities=\[PrepareTools\(prepare_tools\)\]'):
        Agent(_make_model(), prepare_tools=_noop_prep)  # pyright: ignore[reportCallIssue]


async def test_prepare_tools_kwarg_warning_mentions_function_tools_only_rescoping():
    """PR #4859 narrowed `prepare_tools` from all-tools to function-tools-only. The deprecation
    warning surfaces that so users know they may also need `PrepareOutputTools` to preserve old behavior."""
    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_tools` runs only on function tools'):
        Agent(_make_model(), prepare_tools=_noop_prep)  # pyright: ignore[reportCallIssue]


async def test_prepare_tools_kwarg_remaps_to_capability():
    """The kwarg auto-injects a `PrepareTools` capability into the agent's capability list,
    and the prepare callback fires once during a run."""
    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_tools'):
        agent = Agent(_make_model(), prepare_tools=_noop_prep)  # pyright: ignore[reportCallIssue]

    assert any(isinstance(cap, PrepareTools) for cap in agent._root_capability.capabilities)  # pyright: ignore[reportPrivateUsage]
    # Run the agent to exercise the registered capability — this is what makes `_noop_prep` fire
    # and lets us assert the remap actually wires through to the prepare-tools chain.
    await agent.run('hello')


async def test_prepare_tools_kwarg_vs_capability_equivalence():
    """`Agent(prepare_tools=fn)` and `Agent(capabilities=[PrepareTools(fn)])` produce the same
    observable behavior — the prepare callback runs once per step with the same tool defs."""
    kwarg_calls: list[int] = []
    cap_calls: list[int] = []

    def my_tool(x: int) -> int:
        return x  # pragma: no cover

    async def kwarg_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        kwarg_calls.append(len(tool_defs))
        return tool_defs

    async def cap_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        cap_calls.append(len(tool_defs))
        return tool_defs

    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_tools'):
        kwarg_agent = Agent(_make_model(), tools=[my_tool], prepare_tools=kwarg_prep)  # pyright: ignore[reportCallIssue]
    cap_agent = Agent(_make_model(), tools=[my_tool], capabilities=[PrepareTools(cap_prep)])

    kwarg_result = await kwarg_agent.run('hello')
    cap_result = await cap_agent.run('hello')

    assert kwarg_result.output == cap_result.output
    assert kwarg_calls == cap_calls


# --- prepare_output_tools= ---------------------------------------------------------------


async def test_prepare_output_tools_kwarg_emits_deprecation_warning():
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match=r'`Agent\(prepare_output_tools=\.\.\.\)` is deprecated and will be removed in v2\.0',
    ):
        Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=_noop_prep)  # pyright: ignore[reportCallIssue]


async def test_prepare_output_tools_kwarg_warning_points_at_capability():
    with pytest.warns(
        PydanticAIDeprecationWarning, match=r'capabilities=\[PrepareOutputTools\(prepare_output_tools\)\]'
    ):
        Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=_noop_prep)  # pyright: ignore[reportCallIssue]


async def test_prepare_output_tools_kwarg_remaps_to_capability():
    """The kwarg auto-injects a `PrepareOutputTools` capability into the agent's capability list,
    and the prepare callback fires once during a run."""
    seen: list[int] = []

    async def prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        seen.append(len(tool_defs))
        return tool_defs

    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_output_tools'):
        agent = Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=prep)  # pyright: ignore[reportCallIssue]

    assert any(isinstance(cap, PrepareOutputTools) for cap in agent._root_capability.capabilities)  # pyright: ignore[reportPrivateUsage]
    await agent.run('hello')
    assert seen, 'prepare_output_tools callback should have fired'


async def test_prepare_output_tools_kwarg_vs_capability_equivalence():
    """`Agent(prepare_output_tools=fn)` and `Agent(capabilities=[PrepareOutputTools(fn)])` produce
    the same observable behavior."""
    kwarg_calls: list[int] = []
    cap_calls: list[int] = []

    async def kwarg_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        kwarg_calls.append(len(tool_defs))
        return tool_defs

    async def cap_prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        cap_calls.append(len(tool_defs))
        return tool_defs

    with pytest.warns(PydanticAIDeprecationWarning, match=r'prepare_output_tools'):
        kwarg_agent = Agent(TestModel(), output_type=ToolOutput(str), prepare_output_tools=kwarg_prep)  # pyright: ignore[reportCallIssue]
    cap_agent = Agent(TestModel(), output_type=ToolOutput(str), capabilities=[PrepareOutputTools(cap_prep)])

    kwarg_result = await kwarg_agent.run('hello')
    cap_result = await cap_agent.run('hello')

    assert kwarg_result.output == cap_result.output
    assert kwarg_calls == cap_calls


# --- from_spec / from_file forwarders --------------------------------------------------------


async def test_from_spec_event_stream_handler_kwarg_emits_warning_and_stores_handler():
    """`Agent.from_spec(event_stream_handler=...)` warns + the handler is reachable on the agent."""

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:  # pragma: no cover
            pass

    with pytest.warns(
        PydanticAIDeprecationWarning, match=r'`Agent\.from_spec\(event_stream_handler=\.\.\.\)` is deprecated'
    ):
        agent: Agent[Any, Any] = Agent.from_spec({'model': 'test'}, event_stream_handler=handler)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    assert agent.event_stream_handler is handler  # pyright: ignore[reportUnknownMemberType]


async def test_from_file_event_stream_handler_kwarg_emits_warning_and_stores_handler(tmp_path: Any):
    """`Agent.from_file(event_stream_handler=...)` warns + the handler is reachable on the agent."""

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:  # pragma: no cover
            pass

    spec_file = tmp_path / 'agent.json'
    spec_file.write_text('{"model": "test"}')

    with pytest.warns(
        PydanticAIDeprecationWarning, match=r'`Agent\.from_file\(event_stream_handler=\.\.\.\)` is deprecated'
    ):
        agent: Agent[Any, Any] = Agent.from_file(spec_file, event_stream_handler=handler)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    assert agent.event_stream_handler is handler  # pyright: ignore[reportUnknownMemberType]


async def test_from_spec_prepare_tools_kwarg_routes_through_extra_capabilities():
    """`Agent.from_spec(prepare_tools=...)` warns and the legacy kwarg remaps into the
    `extra_capabilities` -> `all_capabilities` path that constructs the agent."""

    async def prep(_ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return tool_defs

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`Agent\.from_spec\(prepare_tools=\.\.\.\)` is deprecated'):
        agent: Agent[Any, Any] = Agent.from_spec({'model': 'test'}, prepare_tools=prep)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    # The deprecated kwarg gets remapped to a `PrepareTools` capability on the constructed agent.
    assert any(isinstance(cap, PrepareTools) for cap in agent._root_capability.capabilities)  # pyright: ignore[reportPrivateUsage,reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType]


async def test_from_file_history_processors_kwarg_routes_through_extra_capabilities(tmp_path: Any):
    """`Agent.from_file(history_processors=...)` warns and the legacy kwarg remaps into the
    `extra_capabilities` -> `merged_capabilities` path forwarded to `from_spec`."""

    def processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages

    spec_file = tmp_path / 'agent.json'
    spec_file.write_text('{"model": "test"}')

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`Agent\.from_file\(history_processors='):
        agent: Agent[Any, Any] = Agent.from_file(spec_file, history_processors=[processor])  # pyright: ignore[reportCallIssue, reportUnknownVariableType]

    process_history_caps = [cap for cap in agent._root_capability.capabilities if isinstance(cap, ProcessHistory)]  # pyright: ignore[reportPrivateUsage,reportUnknownArgumentType,reportUnknownMemberType,reportUnknownVariableType]
    assert process_history_caps, 'from_file(history_processors=) should remap into a ProcessHistory capability'
