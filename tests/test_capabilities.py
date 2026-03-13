import os
from dataclasses import dataclass

import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    ExecutionEnvironment,
    Instructions,
    ModelSettingsCapability,
    Thinking,
    WebSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.environments.memory import MemoryEnvironment
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr

pytestmark = [
    pytest.mark.anyio,
]


def test_capability_types() -> None:
    assert CAPABILITY_TYPES == snapshot(
        {
            'ExecutionEnvironment': ExecutionEnvironment,
            'Instructions': Instructions,
            'ModelSettings': ModelSettingsCapability,
            'Thinking': Thinking,
            'WebSearch': WebSearch,
        }
    )


@pytest.mark.vcr()
async def test_agent(allow_model_requests: None):
    pytest.importorskip('anthropic')
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    api_key = os.getenv('ANTHROPIC_API_KEY', 'mock-value')
    agent = Agent(
        AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key=api_key)),
        capabilities=[
            Instructions("You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI."),
            Thinking(),
            ExecutionEnvironment(
                environment=MemoryEnvironment(
                    files={
                        'README.md': '# My Project\nThis is a simple project.\n',
                        'main.py': 'print("hello world")\n',
                    }
                ),
                include=['ls', 'read_file'],
            ),
            WebSearch(),
        ],
    )
    result = await agent.run('What files are in the project?')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='What files are in the project?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions="You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI.",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='The user wants to know what files are in the project. Let me list the directory contents.',
                        signature=IsStr(),
                        provider_name='anthropic',
                    ),
                    TextPart(content='Sure! Let me take a look at the project directory for you!'),
                    ToolCallPart(tool_name='ls', args={}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=2483,
                    output_tokens=83,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2483,
                        'output_tokens': 83,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id=IsStr(),
                finish_reason='tool_call',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ls',
                        content="""\
README.md (39 bytes)
main.py (21 bytes)\
""",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="You are 'JAK of all trades', a General Agent for Knowledge built with Pydantic AI.",
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
The project currently contains **2 files**:

1. 📄 **`README.md`** *(39 bytes)* - A Markdown file, likely containing documentation or a description of the project.
2. 🐍 **`main.py`** *(21 bytes)* - A Python source file, likely the main entry point of the application.

Would you like me to read the contents of either file?\
"""
                    )
                ],
                usage=RequestUsage(
                    input_tokens=2594,
                    output_tokens=100,
                    details={
                        'cache_creation_input_tokens': 0,
                        'cache_read_input_tokens': 0,
                        'input_tokens': 2594,
                        'output_tokens': 100,
                    },
                ),
                model_name='claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id=IsStr(),
                finish_reason='stop',
                run_id=IsStr(),
            ),
        ]
    )


def test_agent_from_spec_basic():
    """Test Agent.from_spec with basic capabilities."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'Instructions': 'You are a helpful agent.'},
                'Thinking',
                'WebSearch',
                {'ModelSettings': {'max_tokens': 4096}},
            ],
        }
    )
    assert agent.model is not None


def test_agent_from_spec_execution_environment():
    """Test Agent.from_spec with ExecutionEnvironment capability."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'ExecutionEnvironment': {'environment': 'memory', 'include': ['ls', 'read_file']}},
            ],
        }
    )
    assert agent.model is not None


def test_agent_from_spec_no_capabilities():
    """Test Agent.from_spec with no capabilities."""
    agent = Agent.from_spec({'model': 'test'})
    assert agent.model is not None


def test_agent_from_spec_unknown_capability():
    """Test Agent.from_spec with an unknown capability name."""
    with pytest.raises(ValueError, match="Capability 'Unknown' is not in the provided"):
        Agent.from_spec(
            {
                'model': 'test',
                'capabilities': ['Unknown'],
            }
        )


def test_agent_from_spec_bad_args():
    """Test Agent.from_spec with bad arguments for a capability."""
    with pytest.raises(ValueError, match="Failed to instantiate capability 'Instructions'"):
        Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {'Instructions': {'nonexistent_param': 'value'}},
                ],
            }
        )


@dataclass
class CustomCapability(AbstractCapability[None]):
    greeting: str = 'hello'


def test_agent_from_spec_custom_capability():
    """Test Agent.from_spec with a custom capability type."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'CustomCapability': 'world'},
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


def test_agent_from_spec_with_agent_spec_object():
    """Test Agent.from_spec with an AgentSpec instance."""
    from pydantic_ai._spec import NamedSpec
    from pydantic_ai.agent.spec import AgentSpec

    spec = AgentSpec(
        model='test',
        capabilities=[
            NamedSpec(name='Instructions', arguments=('You are helpful.',)),
            NamedSpec(name='Thinking', arguments=None),
        ],
    )
    agent = Agent.from_spec(spec)
    assert agent.model is not None
