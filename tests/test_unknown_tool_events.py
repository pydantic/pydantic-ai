"""Test that unknown tool calls emit proper events for streaming."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


async def test_unknown_tool_streaming_events():
    """Test that unknown tool calls emit both FunctionToolCallEvent and FunctionToolResultEvent during streaming."""

    def call_unknown_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls an unknown tool."""
        return ModelResponse(parts=[ToolCallPart('unknown_tool', {'arg': 'value'})])

    agent = Agent(FunctionModel(call_unknown_tool), retries=1)  # Allow 1 retry to see events

    # Add a known tool so we can test the "Available tools" message
    @agent.tool_plain
    def known_tool(x: int) -> int:
        return x * 2

    # Use the agent.iter API to manually iterate and collect events
    all_events: list[FunctionToolCallEvent | FunctionToolResultEvent] = []

    try:
        async with agent.iter('test') as run:
            # Iterate through the graph nodes and collect events
            async for node in run:
                if agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as event_stream:
                        async for event in event_stream:
                            all_events.append(event)
    except Exception:
        # Expected to fail due to unknown tool, but we should have captured events
        pass

        # We should get events for each unknown tool call attempt
    # With retries=1, we expect: 2 events for first attempt + events for retry attempts
    assert len(all_events) >= 2, f'Expected at least 2 events, got {len(all_events)}: {all_events}'

    # First event should be FunctionToolCallEvent
    assert isinstance(all_events[0], FunctionToolCallEvent)
    assert all_events[0].part.tool_name == 'unknown_tool'
    assert all_events[0].part.args == {'arg': 'value'}

    # Second event should be FunctionToolResultEvent
    assert isinstance(all_events[1], FunctionToolResultEvent)
    assert isinstance(all_events[1].result, RetryPromptPart)
    assert all_events[1].result.tool_name == 'unknown_tool'
    assert "Unknown tool name: 'unknown_tool'" in all_events[1].result.content
    assert 'Available tools: known_tool' in all_events[1].result.content

    # Verify the tool_call_id matches between first pair of events
    assert all_events[0].call_id == all_events[1].tool_call_id

    # Check that we have the key insight: unknown tool calls now emit events!
    tool_call_events = [e for e in all_events if isinstance(e, FunctionToolCallEvent)]
    tool_result_events = [e for e in all_events if isinstance(e, FunctionToolResultEvent)]

    # Each unknown tool call should produce both events
    assert len(tool_call_events) > 0, 'Should have at least one FunctionToolCallEvent'
    assert len(tool_result_events) > 0, 'Should have at least one FunctionToolResultEvent'
    assert all(e.part.tool_name == 'unknown_tool' for e in tool_call_events)
    assert all(isinstance(e.result, RetryPromptPart) for e in tool_result_events)


async def test_unknown_tool_streaming_events_no_tools():
    """Test unknown tool events when no tools are available."""

    def call_unknown_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls an unknown tool."""
        return ModelResponse(parts=[ToolCallPart('unknown_tool', {'arg': 'value'})])

    agent = Agent(FunctionModel(call_unknown_tool), retries=1)

    # Use the agent.iter API to manually iterate and collect events
    all_events: list[FunctionToolCallEvent | FunctionToolResultEvent] = []

    try:
        async with agent.iter('test') as run:
            # Iterate through the graph nodes and collect events
            async for node in run:
                if agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as event_stream:
                        async for event in event_stream:
                            all_events.append(event)
    except Exception:
        # Expected to fail due to unknown tool, but we should have captured events
        pass

    # Verify that we got both events for the unknown tool call
    assert len(all_events) >= 2

    # First event should be FunctionToolCallEvent
    assert isinstance(all_events[0], FunctionToolCallEvent)
    assert all_events[0].part.tool_name == 'unknown_tool'

    # Second event should be FunctionToolResultEvent
    assert isinstance(all_events[1], FunctionToolResultEvent)
    assert isinstance(all_events[1].result, RetryPromptPart)
    assert all_events[1].result.tool_name == 'unknown_tool'
    assert "Unknown tool name: 'unknown_tool'" in all_events[1].result.content
    assert 'No tools available.' in all_events[1].result.content


async def test_unknown_tool_events_with_output_type():
    """Test unknown tool events when using output type (structured response)."""

    def call_unknown_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls an unknown tool."""
        return ModelResponse(parts=[ToolCallPart('unknown_tool', {'arg': 'value'})])

    agent = Agent(FunctionModel(call_unknown_tool), output_type=str, retries=1)

    # Add a known tool
    @agent.tool_plain
    def known_tool(x: int) -> int:
        return x * 2

    # Use the agent.iter API to manually iterate and collect events
    all_events: list[FunctionToolCallEvent | FunctionToolResultEvent] = []

    try:
        async with agent.iter('test') as run:
            # Iterate through the graph nodes and collect events
            async for node in run:
                if agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as event_stream:
                        async for event in event_stream:
                            all_events.append(event)
    except Exception:
        # Expected to fail due to unknown tool, but we should have captured events
        pass

    # Verify that we got both events for the unknown tool call
    assert len(all_events) >= 2

    # First event should be FunctionToolCallEvent
    assert isinstance(all_events[0], FunctionToolCallEvent)
    assert all_events[0].part.tool_name == 'unknown_tool'

    # Second event should be FunctionToolResultEvent
    assert isinstance(all_events[1], FunctionToolResultEvent)
    assert isinstance(all_events[1].result, RetryPromptPart)
    assert all_events[1].result.tool_name == 'unknown_tool'
    # With output_type=str, we should still get the unknown tool error message
    # (The key insight is that we get events at all for unknown tools!)
    assert "Unknown tool name: 'unknown_tool'" in all_events[1].result.content


async def test_unknown_tool_events_multiple_calls():
    """Test unknown tool events when multiple tools are called, some unknown."""

    def call_mixed_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls both known and unknown tools."""
        return ModelResponse(
            parts=[
                ToolCallPart('known_tool', {'x': 1}),
                ToolCallPart('unknown_tool', {'arg': 'value'}),
                ToolCallPart('another_unknown', {'y': 2}),
            ]
        )

    agent = Agent(FunctionModel(call_mixed_tools), retries=1)

    @agent.tool_plain
    def known_tool(x: int) -> int:
        return x * 2

    # Use the agent.iter API to manually iterate and collect events
    all_events: list[FunctionToolCallEvent | FunctionToolResultEvent] = []

    try:
        async with agent.iter('test') as run:
            # Iterate through the graph nodes and collect events
            async for node in run:
                if agent.is_call_tools_node(node):
                    async with node.stream(run.ctx) as event_stream:
                        async for event in event_stream:
                            all_events.append(event)
    except Exception:
        # Expected to fail due to unknown tools, but we should have captured events
        pass

    # Should have events for known_tool, unknown_tool, and another_unknown
    # The key insight is that unknown tools now emit events!
    # We might not get all result events if some tools complete after the exception
    assert len(all_events) >= 4

    # Check that we have events for all tools
    tool_call_events = [e for e in all_events if isinstance(e, FunctionToolCallEvent)]
    tool_result_events = [e for e in all_events if isinstance(e, FunctionToolResultEvent)]

    # Should have called known_tool, unknown_tool, and another_unknown
    tool_names_called = {e.part.tool_name for e in tool_call_events}
    assert 'known_tool' in tool_names_called
    assert 'unknown_tool' in tool_names_called
    assert 'another_unknown' in tool_names_called

    # Verify that we captured events for unknown tools
    # The key insight: unknown tools now emit events instead of being silent!
    unknown_call_events = [e for e in tool_call_events if e.part.tool_name in ['unknown_tool', 'another_unknown']]
    assert len(unknown_call_events) >= 2  # Should have called both unknown tools

    # Verify that any result events for unknown tools are RetryPromptParts
    unknown_results = [e for e in tool_result_events if e.result.tool_name in ['unknown_tool', 'another_unknown']]
    assert all(isinstance(e.result, RetryPromptPart) for e in unknown_results)


async def test_unknown_tool_events_backward_compatibility():
    """Test that the change maintains backward compatibility with existing message history."""

    def call_unknown_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls an unknown tool."""
        return ModelResponse(parts=[ToolCallPart('unknown_tool', {'arg': 'value'})])

    agent = Agent(FunctionModel(call_unknown_tool), retries=0)

    @agent.tool_plain
    def known_tool(x: int) -> int:
        return x * 2

        # Test that the messages still work correctly

    with pytest.raises(Exception):  # Should raise because of retries=0
        await agent.run('test')

    # However, we can still get the message history
    from pydantic_ai import capture_run_messages

    with capture_run_messages() as messages:
        with pytest.raises(Exception):
            await agent.run('test')

    # Verify the message history contains the expected retry prompt
    assert len(messages) == 2
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[1], ModelResponse)

    # The model response should contain the unknown tool call
    assert len(messages[1].parts) == 1
    assert isinstance(messages[1].parts[0], ToolCallPart)
    assert messages[1].parts[0].tool_name == 'unknown_tool'
