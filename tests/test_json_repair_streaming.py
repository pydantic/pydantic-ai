"""Test that JSON repair works on malformed partial JSON during streaming.

This test simulates what happens when a model (like Claude's fine-grained tool
streaming) produces malformed JSON mid-stream. The JSON repair should fix the
syntax errors while preserving partial string values.

Run with:
    uv run pytest tests/test_json_repair_streaming.py -v
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

pytestmark = pytest.mark.anyio


class Whale(BaseModel):
    """A whale with a name and description."""

    name: str
    description: str | None = None


async def test_streaming_with_malformed_json_single_quotes():
    """Test that single quotes in streaming JSON are repaired.

    This simulates Claude's fine-grained tool streaming which can produce
    invalid JSON with single quotes instead of double quotes.
    """
    pytest.importorskip('fast_json_repair')

    async def malformed_stream(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        """Stream malformed JSON with single quotes."""
        assert agent_info.output_tools is not None
        output_tool_name = agent_info.output_tools[0].name

        # This is malformed JSON - single quotes instead of double quotes!
        # A well-behaved model would send: {"name": "Blue Whale", "description": "The largest animal"}
        # But Claude's fine-grained streaming sometimes sends broken JSON like this:
        malformed_json = "{'name': 'Blue Whale', 'description': 'The largest animal'}"

        yield {0: DeltaToolCall(name=output_tool_name)}
        # Stream it in chunks to simulate real streaming
        yield {0: DeltaToolCall(json_args=malformed_json[:20])}  # "{'name': 'Blue Whal"
        yield {0: DeltaToolCall(json_args=malformed_json[20:40])}  # "e', 'description': "
        yield {0: DeltaToolCall(json_args=malformed_json[40:])}  # "'The largest animal'}"

    agent: Agent[None, Whale] = Agent(
        FunctionModel(stream_function=malformed_stream),
        output_type=Whale,
    )

    async with agent.run_stream('Tell me about whales') as result:
        whales: list[Whale] = []
        async for whale in result.stream_output(debounce_by=None):
            whales.append(whale)

        # The final whale should have the correct values despite malformed JSON
        final_whale = whales[-1]
        assert final_whale.name == 'Blue Whale'
        assert final_whale.description == 'The largest animal'

    print('✅ Single quotes test passed!')


async def test_streaming_with_malformed_json_trailing_comma():
    """Test that trailing commas in streaming JSON are repaired."""
    pytest.importorskip('fast_json_repair')

    async def malformed_stream(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        """Stream malformed JSON with trailing comma."""
        assert agent_info.output_tools is not None
        output_tool_name = agent_info.output_tools[0].name

        # Malformed JSON with trailing comma
        malformed_json = '{"name": "Orca", "description": "Also known as killer whale",}'

        yield {0: DeltaToolCall(name=output_tool_name)}
        yield {0: DeltaToolCall(json_args=malformed_json[:25])}
        yield {0: DeltaToolCall(json_args=malformed_json[25:])}

    agent: Agent[None, Whale] = Agent(
        FunctionModel(stream_function=malformed_stream),
        output_type=Whale,
    )

    async with agent.run_stream('Tell me about orcas') as result:
        final_whale = await result.get_output()
        assert final_whale.name == 'Orca'
        assert final_whale.description == 'Also known as killer whale'

    print('✅ Trailing comma test passed!')


async def test_streaming_with_malformed_json_missing_brace():
    """Test that missing closing brace in streaming JSON is repaired."""
    pytest.importorskip('fast_json_repair')

    async def malformed_stream(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        """Stream malformed JSON with missing closing brace."""
        assert agent_info.output_tools is not None
        output_tool_name = agent_info.output_tools[0].name

        # Malformed JSON - missing closing brace
        malformed_json = '{"name": "Humpback", "description": "Known for singing"'

        yield {0: DeltaToolCall(name=output_tool_name)}
        yield {0: DeltaToolCall(json_args=malformed_json[:20])}
        yield {0: DeltaToolCall(json_args=malformed_json[20:])}

    agent: Agent[None, Whale] = Agent(
        FunctionModel(stream_function=malformed_stream),
        output_type=Whale,
    )

    async with agent.run_stream('Tell me about humpbacks') as result:
        final_whale = await result.get_output()
        assert final_whale.name == 'Humpback'
        assert final_whale.description == 'Known for singing'

    print('✅ Missing brace test passed!')


async def test_streaming_partial_string_preserved():
    """Test that partial string values are preserved during streaming with repair.

    This is the key test - when we repair partial JSON mid-stream, we should
    preserve the partial string value, not truncate it to null.
    """
    pytest.importorskip('fast_json_repair')

    async def malformed_partial_stream(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        """Stream malformed partial JSON with incomplete string."""
        assert agent_info.output_tools is not None
        output_tool_name = agent_info.output_tools[0].name

        # Malformed partial JSON - single quotes AND incomplete string!
        # This is the worst case: syntax error + partial value
        chunks = [
            "{'name': 'Beluga",  # Start with single quotes (malformed)
            "', 'description': 'White wha",  # Continue with partial string
            "le of the Arctic'}",  # Complete the JSON
        ]

        yield {0: DeltaToolCall(name=output_tool_name)}
        for chunk in chunks:
            yield {0: DeltaToolCall(json_args=chunk)}

    agent: Agent[None, Whale] = Agent(
        FunctionModel(stream_function=malformed_partial_stream),
        output_type=Whale,
    )

    async with agent.run_stream('Tell me about belugas') as result:
        whales: list[Whale] = []
        async for whale in result.stream_output(debounce_by=None):
            whales.append(whale)
            print(f'  Streamed: name={whale.name!r}, description={whale.description!r}')

        # Check the intermediate partial values were preserved
        # (not truncated to None by repair)
        assert len(whales) >= 2, 'Should have multiple streaming chunks'

        # Final value should be complete
        final_whale = whales[-1]
        assert final_whale.name == 'Beluga'
        assert final_whale.description == 'White whale of the Arctic'

    print('✅ Partial string preservation test passed!')


async def test_valid_json_still_works():
    """Sanity check - valid JSON should still work normally."""

    async def valid_stream(
        _messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[dict[int, DeltaToolCall]]:
        """Stream valid JSON."""
        assert agent_info.output_tools is not None
        output_tool_name = agent_info.output_tools[0].name

        valid_json = json.dumps({'name': 'Gray Whale', 'description': 'Migrates long distances'})

        yield {0: DeltaToolCall(name=output_tool_name)}
        yield {0: DeltaToolCall(json_args=valid_json[:20])}
        yield {0: DeltaToolCall(json_args=valid_json[20:])}

    agent: Agent[None, Whale] = Agent(
        FunctionModel(stream_function=valid_stream),
        output_type=Whale,
    )

    async with agent.run_stream('Tell me about gray whales') as result:
        final_whale = await result.get_output()
        assert final_whale.name == 'Gray Whale'
        assert final_whale.description == 'Migrates long distances'

    print('✅ Valid JSON test passed!')


if __name__ == '__main__':
    import asyncio

    async def main():
        print('🐋 Testing JSON repair on streaming chunks...\n')

        await test_valid_json_still_works()
        await test_streaming_with_malformed_json_single_quotes()
        await test_streaming_with_malformed_json_trailing_comma()
        await test_streaming_with_malformed_json_missing_brace()
        await test_streaming_partial_string_preserved()

        print('\n🎉 All streaming JSON repair tests passed!')

    asyncio.run(main())
