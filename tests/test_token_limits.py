from __future__ import annotations

import os

import pytest

from pydantic_ai import Agent
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.settings import ModelSettings


@pytest.mark.skipif(not os.getenv('ANTHROPIC_API_KEY'), reason='ANTHROPIC_API_KEY not set')
def test_real_agent_with_token_limited_tool():
    """Test that the agent can handle token limits by retrying with smaller chunks."""
    with override_allow_model_requests(allow_model_requests=True):
        # Set a very low token limit to trigger max_tokens error
        model_settings = ModelSettings(max_tokens=500)

        # Create an agent that requires a tool call for the result
        agent = Agent(model='anthropic:claude-3-5-sonnet-latest', model_settings=model_settings, retries=5)

        poem_content: str = ''

        @agent.tool_plain(retries=5)
        def write_poem(content: str, filename: str, append: bool = False) -> str:
            """Write a poem to a file. If append is True, append to the existing content. Otherwise, overwrite the existing content."""
            nonlocal poem_content
            if append:
                poem_content += content
                print(f'Appending to {filename}: {content}')
            else:
                poem_content = content
                print(f'Overwriting {filename}: {content}')
            return poem_content

        # Don't use capture_run_messages, just run directly
        _ = agent.run_sync(
            'Write a very long poem using the `write_poem` tool to write to poem.txt. '
            'The poem should be at least 500 words long. '
            'Use the write_poem tool to write the poem to poem.txt.'
        )

        # Verify that the final poem meets the length requirement
        words = poem_content.split()
        assert len(words) >= 500, f'Expected at least 500 words, got {len(words)}'

        # Check that multiple tool calls were made (splitting the poem into chunks)
        assert '\n\n' in poem_content, 'Expected poem to have multiple chunks with line breaks'
