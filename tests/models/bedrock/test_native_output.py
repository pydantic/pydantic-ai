"""Tests for Bedrock native structured output (`NativeOutput`).

Covers per-model coverage (Anthropic, Qwen, Google, MiniMax, Mistral, NVIDIA),
the `_native_output_format` builder, schema transforms (numerical constraints,
nested objects without `extra='forbid'`), streaming, and the unsupported-model
error path.
"""

from __future__ import annotations as _annotations

from enum import Enum
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.output import NativeOutput, OutputObjectDefinition
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsStr, try_import
from .conftest import Address, CityInfo, PersonQuery

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


async def test_bedrock_native_output_supported_model(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude Sonnet 4.6 via Bedrock: NativeOutput → outputConfig with json_schema."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-6', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    result = await agent.run('What is the capital of France? Give me the city name, country, and population.')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2161000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Paris","country":"France","population":2161000}')],
                usage=RequestUsage(input_tokens=211, output_tokens=18),
                model_name='us.anthropic.claude-sonnet-4-6',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_bedrock_native_output_unsupported_model_raises(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude 3.5 Sonnet (not 4.5): NativeOutput → raises UserError (model doesn't support it)."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20241022-v2:0', provider=bedrock_provider)

    agent = Agent(model, output_type=NativeOutput(CityInfo))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        agent.run_sync('Tell me about Berlin')


async def test_bedrock_native_output_qwen(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Qwen3 via Bedrock: NativeOutput → outputConfig with json_schema."""
    model = BedrockConverseModel('qwen.qwen3-32b-v1:0', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    result = await agent.run('What is the capital of France? Give me the city name, country, and population.')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2148000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Paris",
  "country": "France",
  "population": 2148000
}\
"""
                    )
                ],
                usage=RequestUsage(input_tokens=30, output_tokens=30),
                model_name='qwen.qwen3-32b-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_native_output_google(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Google Gemma 3 via Bedrock: NativeOutput → outputConfig with json_schema."""
    model = BedrockConverseModel('google.gemma-3-27b-it', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    result = await agent.run('What is the capital of France? Give me the city name, country, and population.')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2141000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Paris",
  "country": "France",
  "population": 2141000 \n\
}\
"""
                    )
                ],
                usage=RequestUsage(input_tokens=27, output_tokens=34),
                model_name='google.gemma-3-27b-it',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_native_output_minimax(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """MiniMax M2 via Bedrock: NativeOutput → outputConfig with json_schema."""
    model = BedrockConverseModel('minimax.minimax-m2', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    result = await agent.run('What is the capital of France? Give me the city name, country, and population.')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2110000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content="""\
I need to answer the user's question about Paris' capital status, including the country and population. I'll provide the city, which is Paris, France, and its population. As of 2023, the population for Paris is approximately 2.1 million within the city proper and over 12 million in the metropolitan area. I want to clarify whether the user prefers the city or metropolitan area population to avoid any ambiguity.

It's best to clarify that Paris has a population of about 2.1 million, and the metropolitan area is over 12 million. I want to avoid using ambiguous terms like "approximately 12.5 million" without specifying numbers. So, I can confidently say around 12.1 million for the metropolitan area as of 2023. I also need to confirm the capital of France is indeed Paris, which is essential. Let's keep it straightforward and concise!

I'm focusing on the capital of France, which is Paris. The city's population is about 2.1 million, and the metropolitan area has over 12 million residents. I think I should include this information without unnecessary formatting or disclaimers. They didn't ask for 2023 data specifically, but I'll mention "as of 2023" for clarity. So, I'll present the information in a neat format to make it easy for the user to read\
"""
                    ),
                    TextPart(
                        content="""\
{


  "city": "Paris",
  "country": "France",
  "population": 2110000
}\
"""
                    ),
                ],
                usage=RequestUsage(input_tokens=40, output_tokens=302),
                model_name='minimax.minimax-m2',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_native_output_mistral(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Mistral Large 3 via Bedrock: NativeOutput → outputConfig with json_schema."""
    model = BedrockConverseModel('mistral.mistral-large-3-675b-instruct', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    result = await agent.run('What is the capital of France? Give me the city name, country, and population.')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2102650))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{ "city": "Paris", "country": "France", "population": 2102650 }')],
                usage=RequestUsage(input_tokens=21, output_tokens=26),
                model_name='mistral.mistral-large-3-675b-instruct',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_native_output_nvidia(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """NVIDIA Nemotron Nano via Bedrock: NativeOutput → outputConfig with json_schema."""
    model = BedrockConverseModel('nvidia.nemotron-nano-12b-v2', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    result = await agent.run('What is the capital of France? Give me the city name, country, and population.')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2148000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content="""\
{
  "city": "Paris",
  "country": "France",
  "population": 2148000
}\
"""
                    )
                ],
                usage=RequestUsage(input_tokens=33, output_tokens=30),
                model_name='nvidia.nemotron-nano-12b-v2',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_native_output_nested_objects_without_extra_forbid(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Native output with nested objects that lack extra='forbid'.

    Bedrock requires additionalProperties: false on all object schemas for native output.
    prepare_request() forces strict=True for native output, so the transformer adds it
    even to nested objects that don't have extra='forbid' in their Pydantic config.
    """
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-6', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(PersonQuery))

    result = await agent.run('Look up John who lives at 123 Main St, Springfield')
    assert result.output == snapshot(
        PersonQuery(name='John', address=Address(street='123 Main St', city='Springfield'))
    )


def test_bedrock_native_output_format_structure():
    """Test that _native_output_format produces the correct AWS outputConfig structure."""
    params = ModelRequestParameters(
        output_mode='native',
        output_object=OutputObjectDefinition(
            json_schema={
                'type': 'object',
                'properties': {
                    'city': {'type': 'string'},
                    'country': {'type': 'string'},
                },
                'required': ['city', 'country'],
            },
            name='CityInfo',
            description='Information about a city',
        ),
    )

    result = BedrockConverseModel._native_output_format(params)  # pyright: ignore[reportPrivateUsage]

    assert result == snapshot(
        {
            'textFormat': {
                'type': 'json_schema',
                'structure': {
                    'jsonSchema': {
                        'name': 'CityInfo',
                        'schema': '{"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}}, "required": ["city", "country"]}',
                        'description': 'Information about a city',
                    }
                },
            }
        }
    )


def test_bedrock_native_output_format_auto_mode():
    """Test that _native_output_format returns None for auto output mode."""
    params = ModelRequestParameters(output_mode='auto')

    result = BedrockConverseModel._native_output_format(params)  # pyright: ignore[reportPrivateUsage]

    assert result is None


def test_bedrock_native_output_format_without_name_description():
    """Test that _native_output_format uses default name when not provided and omits description."""
    params = ModelRequestParameters(
        output_mode='native',
        output_object=OutputObjectDefinition(
            json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}},
        ),
    )

    result = BedrockConverseModel._native_output_format(params)  # pyright: ignore[reportPrivateUsage]

    assert result == snapshot(
        {
            'textFormat': {
                'type': 'json_schema',
                'structure': {
                    'jsonSchema': {
                        'name': 'final_result',
                        'schema': '{"type": "object", "properties": {"city": {"type": "string"}}}',
                    }
                },
            }
        }
    )


async def test_bedrock_native_output_numerical_constraints(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """NativeOutput with numeric constraints succeeds — transformer strips incompatible constraints."""

    class Priority(str, Enum):
        """Priority level."""

        LOW = 'low'
        MEDIUM = 'medium'
        HIGH = 'high'

    class TaskWithNumericalConstraints(BaseModel):
        """Task with numerical constraints (minimum, maximum, multipleOf)."""

        score: Annotated[float, Field(ge=0.0, le=100.0)]
        rating: Annotated[int, Field(multiple_of=5)]
        priority: Priority

    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(TaskWithNumericalConstraints))

    result = await agent.run('Rate this task: score 85.5 out of 100, rating 15, priority high.')
    assert result.output == snapshot(TaskWithNumericalConstraints(score=85.5, rating=15, priority=Priority.HIGH))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Rate this task: score 85.5 out of 100, rating 15, priority high.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"score": 85.5, "rating": 15, "priority": "high"}')],
                usage=RequestUsage(input_tokens=308, output_tokens=23),
                model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bedrock_native_output_stream(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude Sonnet 4.5 via Bedrock: NativeOutput with streaming."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    async with agent.run_stream(
        'What is the capital of France? Give me the city name, country, and population.'
    ) as result:
        output = await result.get_output()

    assert output == snapshot(CityInfo(city='Paris', country='France', population=2161000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France? Give me the city name, country, and population.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Paris","country":"France","population":2161000}')],
                usage=RequestUsage(input_tokens=210, output_tokens=18),
                model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'end_turn'},
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
