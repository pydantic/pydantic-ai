"""Tests for Bedrock strict tool calls.

Covers `ToolDefinition.strict` propagation onto `ToolSpecificationTypeDef`,
strict-vs-unsupported model behavior, mixed strict/non-strict runs, and the
intersection between strict tools and `NativeOutput`.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.output import NativeOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from ..._inline_snapshot import snapshot
from ...conftest import IsDatetime, IsStr, try_import
from .conftest import CityInfo, PersonQuery

with try_import() as imports_successful:
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def test_bedrock_strict_tool_definition_supported_model(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude Sonnet 4.5 via Bedrock: strict=True → strict field in tool definition."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)

    tool_def = ToolDefinition(
        name='get_weather',
        description='Get the weather for a city',
        parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
        strict=True,
    )

    result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]

    assert result == snapshot(
        {
            'toolSpec': {
                'name': 'get_weather',
                'inputSchema': {
                    'json': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
                },
                'description': 'Get the weather for a city',
                'strict': True,
            }
        }
    )


def test_bedrock_strict_tool_definition_unsupported_model(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude 3.5 Sonnet: strict=True specified but not sent (model doesn't support it)."""
    model = BedrockConverseModel('us.anthropic.claude-3-5-sonnet-20241022-v2:0', provider=bedrock_provider)

    tool_def = ToolDefinition(
        name='get_weather',
        description='Get the weather for a city',
        parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
        strict=True,
    )

    result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]

    assert result == snapshot(
        {
            'toolSpec': {
                'name': 'get_weather',
                'inputSchema': {
                    'json': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
                },
                'description': 'Get the weather for a city',
            }
        }
    )


def test_bedrock_strict_tool_definition_none(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Any model: strict=None → no strict field."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)

    tool_def = ToolDefinition(
        name='get_weather',
        description='Get the weather for a city',
        parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
        strict=None,
    )

    result = model._map_tool_definition(tool_def)  # pyright: ignore[reportPrivateUsage]

    assert result == snapshot(
        {
            'toolSpec': {
                'name': 'get_weather',
                'inputSchema': {
                    'json': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}
                },
                'description': 'Get the weather for a city',
            }
        }
    )


async def test_bedrock_strict_tool_supported_model(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude Sonnet 4.5 via Bedrock: strict=True tool with API call."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=True)
    def get_weather(city: str) -> str:
        return f'Weather in {city}: Sunny, 22°C'

    result = await agent.run("What's the weather in Paris?")
    assert result.output == snapshot(
        "The weather in Paris is currently sunny with a temperature of 22°C (approximately 72°F). It's a beautiful day!"
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content="What's the weather in Paris?", timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=560, output_tokens=53),
                model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='Weather in Paris: Sunny, 22°C',
                        tool_call_id=IsStr(),
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
                        content="The weather in Paris is currently sunny with a temperature of 22°C (approximately 72°F). It's a beautiful day!"
                    )
                ],
                usage=RequestUsage(input_tokens=637, output_tokens=31),
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


async def test_bedrock_mixed_strict_tool_run(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Exercise both strict=True and strict=False tool definitions against Bedrock."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(
        model,
        system_prompt='Always call `country_source` first, then call `capital_lookup` with that result before replying.',
    )

    @agent.tool_plain(strict=True)
    async def country_source() -> str:
        return 'Japan'

    @agent.tool_plain(strict=False)
    async def capital_lookup(country: str) -> str:
        if country == 'Japan':
            return 'Tokyo'
        return f'Unknown capital for {country}'  # pragma: no cover

    result = await agent.run('Use the registered tools and respond exactly as `Capital: <city>`.')
    assert result.output == snapshot('Capital: Tokyo')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='Always call `country_source` first, then call `capital_lookup` with that result before replying.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Use the registered tools and respond exactly as `Capital: <city>`.',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content="I'll help you find the capital city using the available tools."),
                    ToolCallPart(tool_name='country_source', args={}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=628, output_tokens=50),
                model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='country_source',
                        content='Japan',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='capital_lookup',
                        args={'country': 'Japan'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=691, output_tokens=53),
                model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='capital_lookup',
                        content='Tokyo',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Capital: Tokyo')],
                usage=RequestUsage(input_tokens=757, output_tokens=6),
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


async def test_bedrock_strict_false_tool_with_nested_objects(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Bedrock accepts strict=False tools without additionalProperties: false.

    When no strict flag is sent to the API, Bedrock does not validate the schema structure.
    This test confirms that strict=False tools with nested objects (lacking extra='forbid')
    are accepted by Bedrock without errors.
    """
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model)

    @agent.tool_plain(strict=False)
    async def lookup_person(query: PersonQuery) -> str:
        return f'{query.name} lives at {query.address.street}, {query.address.city}'

    result = await agent.run('Look up John who lives at 123 Main St, Springfield')
    assert isinstance(result.output, str)


async def test_bedrock_strict_tool_with_native_output(
    allow_model_requests: None,
    bedrock_provider: BedrockProvider,
):
    """Claude Sonnet 4.5 via Bedrock: strict=True tool + NativeOutput."""
    model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=bedrock_provider)
    agent = Agent(model, output_type=NativeOutput(CityInfo))

    @agent.tool_plain(strict=True)
    def lookup_population(city: str) -> int:
        return 2_161_000 if city == 'Paris' else 1_000_000

    result = await agent.run('Give me details about Paris including its population')

    assert result.output == snapshot(CityInfo(city='Paris', country='France', population=2161000))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Give me details about Paris including its population', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_population',
                        args={'city': 'Paris'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=747, output_tokens=53),
                model_name='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                timestamp=IsDatetime(),
                provider_name='bedrock',
                provider_url='https://bedrock-runtime.us-east-1.amazonaws.com',
                provider_details={'finish_reason': 'tool_use'},
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='lookup_population',
                        content=2161000,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Paris", "country": "France", "population": 2161000}')],
                usage=RequestUsage(input_tokens=816, output_tokens=23),
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
