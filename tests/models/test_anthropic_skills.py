from __future__ import annotations as _annotations

from typing import Any
from dataclasses import dataclass
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.builtin_tools import CodeExecutionTool
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

from ..conftest import try_import
from .test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaCodeExecutionToolResultBlock,
        BetaTextBlock,
        BetaUsage,
        BetaMessage,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

async def test_code_execution_with_skills(allow_model_requests: None):
    c = completion_message(
        [BetaTextBlock(text='response', type='text')],
        usage=BetaUsage(input_tokens=10, output_tokens=5),
    )
    mock_client = MockAnthropic.create_mock(c)
    m = AnthropicModel('claude-3-5-sonnet-20241022', provider=AnthropicProvider(anthropic_client=mock_client))
    
    # Create agent with CodeExecutionTool having skills
    # Skills should be a list of skill_id strings
    tool = CodeExecutionTool(skills=['pptx', 'xlsx'])
    agent = Agent(m, builtin_tools=[tool])

    await agent.run('test prompt')

    # Verify skills were passed in container, NOT in tool definition
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    extra_headers = completion_kwargs['extra_headers']
    container = completion_kwargs['container']
    
    assert tools == snapshot(
        [
            {
                'name': 'code_execution',
                'type': 'code_execution_20250825',
                # 'skills' should NOT be here
            }
        ]
    )
    assert 'skills' not in tools[0]
    assert 'skills-2025-10-02' in extra_headers['anthropic-beta']
    assert container == snapshot({
        'skills': [
            {'type': 'anthropic', 'skill_id': 'pptx', 'version': 'latest'},
            {'type': 'anthropic', 'skill_id': 'xlsx', 'version': 'latest'}
        ]
    })
