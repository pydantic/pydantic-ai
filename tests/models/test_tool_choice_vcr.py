"""Cross-provider VCR tests for tool_choice functionality.

This module consolidates tool_choice integration tests across all providers.
Tests are parametrized by provider and scenario, using cassettes recorded
against live APIs.

Key behaviors tested:
- tool_choice='auto': Model decides whether to use tools
- tool_choice='none': Function tools disabled, text/output tools available
- tool_choice='required': Must use a tool (direct model requests)
- tool_choice=[list]: Specific tools only (direct model requests)
- ToolOrOutput: Specific function tools + output tools
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryImage,
    FilePart,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings, ToolChoice, ToolOrOutput
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import UsageLimits

from ..conftest import try_import

# =============================================================================
# Provider imports (conditional)
# =============================================================================

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as groq_available:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

with try_import() as mistral_available:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as huggingface_available:
    from pydantic_ai.models.huggingface import HuggingFaceModel
    from pydantic_ai.providers.huggingface import HuggingFaceProvider

with try_import() as openai_responses_available:
    from pydantic_ai.models.openai import OpenAIResponsesModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.vcr,
]


# =============================================================================
# Tool definitions
# =============================================================================


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'Sunny, 22C in {city}'


def get_time(timezone: str) -> str:
    """Get the current time in a timezone."""
    return f'14:30 in {timezone}'  # pragma: no cover


class CityInfo(BaseModel):
    """Structured output model for city information."""

    city: str
    summary: str


def make_tool_def(name: str, description: str, param_name: str) -> ToolDefinition:
    """Create a ToolDefinition for testing direct model requests."""
    return ToolDefinition(
        name=name,
        description=description,
        parameters_json_schema={
            'type': 'object',
            'properties': {param_name: {'type': 'string'}},
            'required': [param_name],
        },
    )


# =============================================================================
# Case dataclass
# =============================================================================


@dataclass(kw_only=True)
class Case:
    """A single test case for tool_choice behavior."""

    id: str
    provider: str
    tool_choice: ToolChoice
    expected_message_structure: Any  # snapshot() stored here per case
    tools: list[Callable[..., str]] = field(default_factory=lambda: [get_weather])
    output_type: OutputSpec[Any] | None = None
    prompt: str = "What's the weather in Paris?"
    model_name: str | None = None
    # Expected values - set to None to skip assertion
    expected_tool_choice_in_request: Any = None
    skip_reason: str | None = None
    use_direct_request: bool = False  # Use model.request() instead of agent.run() for required/list


# =============================================================================
# Provider configuration
# =============================================================================

PROVIDER_MODELS: dict[str, tuple[str, Callable[[], bool]]] = {
    'openai': ('gpt-5-mini', openai_available),
    'openai_responses': ('gpt-5-mini', openai_responses_available),
    'anthropic': ('claude-sonnet-4-5', anthropic_available),
    'groq': ('meta-llama/llama-4-scout-17b-16e-instruct', groq_available),
    'mistral': ('mistral-large-latest', mistral_available),
    'google': ('gemini-2.5-flash', google_available),
    'bedrock': ('us.anthropic.claude-sonnet-4-5-20250929-v1:0', bedrock_available),
    'huggingface': ('meta-llama/Llama-4-Scout-17B-16E-Instruct', huggingface_available),
}


def get_model(
    provider: str, api_keys: dict[str, str], bedrock_provider: Any = None, model_name: str | None = None
) -> Model:
    """Create a model instance for the given provider."""
    default_model_name, _ = PROVIDER_MODELS[provider]
    model_name = model_name or default_model_name

    if provider == 'openai':
        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'openai_responses':
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_keys['openai']))
    elif provider == 'anthropic':
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_keys['anthropic']))
    elif provider == 'groq':
        return GroqModel(model_name, provider=GroqProvider(api_key=api_keys['groq']))
    elif provider == 'mistral':
        return MistralModel(model_name, provider=MistralProvider(api_key=api_keys['mistral']))
    elif provider == 'google':
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_keys['google']))
    elif provider == 'bedrock':
        assert bedrock_provider is not None, 'bedrock_provider fixture required for bedrock tests'
        return BedrockConverseModel(model_name, provider=bedrock_provider)
    elif provider == 'huggingface':
        return HuggingFaceModel(
            model_name, provider=HuggingFaceProvider(api_key=api_keys['huggingface'], provider_name='together')
        )
    else:
        raise ValueError(f'Unknown provider: {provider}')  # pragma: no cover


def get_tool_choice_from_cassette(cassette: Any, provider: str) -> Any:
    """Extract tool_choice from cassette request body, handling provider differences."""
    import json

    if not cassette.requests:
        return None  # pragma: no cover

    # Find the POST request (skip any GET requests like HuggingFace's provider mapping)
    request = None
    for req in cassette.requests:
        if req.method == 'POST':
            request = req
            break
    if request is None:  # pragma: no cover
        return None
    # VCR stores body as bytes/string, need to parse JSON
    body_bytes = request.body
    if body_bytes is None:
        return None  # pragma: no cover

    try:
        body: dict[str, Any] = json.loads(body_bytes) if isinstance(body_bytes, (str, bytes)) else body_bytes
    except (json.JSONDecodeError, TypeError):  # pragma: no cover
        return None

    if provider == 'google':
        tool_config: dict[str, Any] = body.get('toolConfig', {})
        func_config: dict[str, Any] = tool_config.get('functionCallingConfig', {})
        return func_config.get('mode')
    elif provider == 'anthropic':
        tc = body.get('tool_choice', {})
        if isinstance(tc, dict):
            return tc.get('type')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return tc  # pragma: no cover
    elif provider == 'bedrock':
        tool_config = body.get('toolConfig', {})
        tool_choice = tool_config.get('toolChoice', {})
        if 'auto' in tool_choice:
            return 'auto'
        elif 'any' in tool_choice:
            return 'any'
        elif 'tool' in tool_choice:  # pragma: no cover
            return tool_choice['tool'].get('name')
        return None  # pragma: no cover
    else:
        # OpenAI, Groq, Mistral, HuggingFace use tool_choice directly
        return body.get('tool_choice')


class _MessageStructure(TypedDict):
    type: type[ModelRequest | ModelResponse]
    parts: list[type[ModelRequestPart | ModelResponsePart]]


def get_message_structure(messages: list[ModelRequest | ModelResponse]) -> list[_MessageStructure]:
    """Extract simplified message structure for snapshot comparison."""
    result: list[_MessageStructure] = []
    for msg in messages:
        result.append(
            {
                'type': type(msg),
                'parts': [type(p) for p in msg.parts],
            }
        )
    return result


# =============================================================================
# Test cases - each Case has its own snapshot() for expected_message_structure
# Snapshots will be populated when cassettes are recorded with --inline-snapshot=create
# =============================================================================

# fmt: off
CASES = [
    # === tool_choice='auto' - Model uses tool ===
    Case(
        id='openai-auto-uses-tool',
        provider='openai',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('auto'),
    ),
    Case(
        id='anthropic-auto-uses-tool',
        provider='anthropic',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('auto'),
    ),
    Case(
        id='groq-auto-uses-tool',
        provider='groq',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('auto'),
    ),
    Case(
        id='mistral-auto-uses-tool',
        provider='mistral',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('auto'),
    ),
    Case(
        id='google-auto-uses-tool',
        provider='google',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('AUTO'),
    ),
    Case(
        id='bedrock-auto-uses-tool',
        provider='bedrock',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('auto'),
    ),
    # Together backend returns 500 Internal Server Error on the second request when continuing
    # a tool conversation (first request with tool_choice='auto' succeeds and returns tool call,
    # but the follow-up request with tool result fails). Other backends (novita, nscale) either
    # don't support tool calling or return 400 errors. See test_together_500_on_tool_continuation.
    Case(
        id='huggingface-auto-uses-tool',
        provider='huggingface',
        tool_choice='auto',
        expected_message_structure=snapshot([]),
        expected_tool_choice_in_request=snapshot('auto'),
        skip_reason='Together backend 500s on tool continuation',
    ),
    Case(
        id='openai_responses-auto-uses-tool',
        provider='openai_responses',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ThinkingPart,ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('auto'),
    ),

    # === tool_choice='none' - Function tools disabled, text response ===
    Case(
        id='openai-none-text-response',
        provider='openai',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('none'),
    ),
    Case(
        id='anthropic-none-text-response',
        provider='anthropic',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('none'),
        prompt='Say hello',
    ),
    Case(
        id='groq-none-text-response',
        provider='groq',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('none'),
    ),
    Case(
        id='mistral-none-text-response',
        provider='mistral',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot(None),
    ),
    Case(
        id='google-none-text-response',
        provider='google',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('NONE'),
    ),
    Case(
        id='bedrock-none-text-response',
        provider='bedrock',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot(None),
    ),
    Case(
        id='huggingface-none-text-response',
        provider='huggingface',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        expected_tool_choice_in_request=snapshot('none'),
    ),
    Case(
        id='openai_responses-none-text-response',
        provider='openai_responses',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ThinkingPart,TextPart]}]),
        expected_tool_choice_in_request=snapshot('none'),
    ),

    # === tool_choice='required' - Must use tool (direct model request) ===
    Case(
        id='openai-required-forces-tool',
        provider='openai',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('required'),
        use_direct_request=True,
    ),
    Case(
        id='anthropic-required-forces-tool',
        provider='anthropic',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('any'),
        use_direct_request=True,
    ),
    Case(
        id='groq-required-forces-tool',
        provider='groq',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('required'),
        use_direct_request=True,
    ),
    Case(
        id='mistral-required-forces-tool',
        provider='mistral',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('required'),
        use_direct_request=True,
    ),
    Case(
        id='google-required-forces-tool',
        provider='google',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('ANY'),
        use_direct_request=True,
    ),
    Case(
        id='bedrock-required-forces-tool',
        provider='bedrock',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('any'),
        use_direct_request=True,
    ),
    Case(
        id='huggingface-required-forces-tool',
        provider='huggingface',
        tool_choice='required',
        expected_message_structure=snapshot(['ToolCallPart']),
        expected_tool_choice_in_request=snapshot('required'),
        use_direct_request=True,
    ),
    Case(
        id='openai_responses-required-forces-tool',
        provider='openai_responses',
        tool_choice='required',
        expected_message_structure=snapshot(['ThinkingPart','ToolCallPart']),
        expected_tool_choice_in_request=snapshot('required'),
        use_direct_request=True,
    ),

    # === tool_choice=['specific_tool'] - Force specific tool (direct model request) ===
    Case(
        id='openai-list-single-tool',
        provider='openai',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='anthropic-list-single-tool',
        provider='anthropic',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='groq-list-single-tool',
        provider='groq',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='mistral-list-single-tool',
        provider='mistral',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='google-list-single-tool',
        provider='google',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='bedrock-list-single-tool',
        provider='bedrock',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='huggingface-list-single-tool',
        provider='huggingface',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),
    Case(
        id='openai_responses-list-single-tool',
        provider='openai_responses',
        tool_choice=['get_weather'],
        expected_message_structure=snapshot(['ThinkingPart','ToolCallPart']),
        tools=[get_weather, get_time],
        use_direct_request=True,
    ),

    # === tool_choice='none' with structured output - output tool still works ===
    # No warning expected: tool_choice='none' disables function tools but output tools remain available
    Case(
        id='openai-none-with-output',
        provider='openai',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='anthropic-none-with-output',
        provider='anthropic',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='groq-none-with-output',
        provider='groq',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='mistral-none-with-output',
        provider='mistral',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='google-none-with-output',
        provider='google',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='bedrock-none-with-output',
        provider='bedrock',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='huggingface-none-with-output',
        provider='huggingface',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),
    Case(
        id='openai_responses-none-with-output',
        provider='openai_responses',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ThinkingPart,ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=CityInfo,
        prompt='Tell me about Paris',
    ),

    # === tool_choice='none' with text+structured output - triggers (tool_names, 'auto') branch ===
    # Using (str, CityInfo) tuple allows text output AND creates output tool, hitting the 'auto' mode path
    # No warning expected: tool_choice='none' disables function tools but output tools remain available
    Case(
        id='openai-none-with-output-text-allowed',
        provider='openai',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[TextPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='anthropic-none-with-output-text-allowed',
        provider='anthropic',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]}, {'type':ModelResponse,'parts':[ToolCallPart]}, {'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='groq-none-with-output-text-allowed',
        provider='groq',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]}, {'type':ModelResponse,'parts':[ToolCallPart]}, {'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='mistral-none-with-output-text-allowed',
        provider='mistral',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]}, {'type':ModelResponse,'parts':[ToolCallPart]}, {'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='google-none-with-output-text-allowed',
        provider='google',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]}, {'type':ModelResponse,'parts':[ToolCallPart]}, {'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='bedrock-none-with-output-text-allowed',
        provider='bedrock',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]}, {'type':ModelResponse,'parts':[ToolCallPart]}, {'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='huggingface-none-with-output-text-allowed',
        provider='huggingface',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]}, {'type':ModelResponse,'parts':[ToolCallPart]}, {'type': ModelRequest,'parts':[ToolReturnPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),
    Case(
        id='openai_responses-none-with-output-text-allowed',
        provider='openai_responses',
        tool_choice='none',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ThinkingPart,TextPart]}]),
        output_type=(str, CityInfo),
        prompt='Tell me about Paris briefly',
    ),

    # === ToolOrOutput - specific function tools + output tools ===
    Case(
        id='openai-tools-plus-output',
        provider='openai',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='anthropic-tools-plus-output',
        provider='anthropic',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='groq-tools-plus-output',
        provider='groq',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart,ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart,ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='mistral-tools-plus-output',
        provider='mistral',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='google-tools-plus-output',
        provider='google',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='bedrock-tools-plus-output',
        provider='bedrock',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='huggingface-tools-plus-output',
        provider='huggingface',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart,ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart,ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),
    Case(
        id='openai_responses-tools-plus-output',
        provider='openai_responses',
        tool_choice=ToolOrOutput(function_tools=['get_weather']),
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ThinkingPart,ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]},{'type':ModelResponse,'parts':[ThinkingPart,ThinkingPart, ThinkingPart, ThinkingPart, ThinkingPart, ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[get_weather, get_time],
        output_type=CityInfo,
        prompt='Get weather for Paris and summarize',
    ),

    # === Google-specific: output-only with no direct output allowed ===
    Case(
        id='google-auto-output-only-no-direct',
        provider='google',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[ToolCallPart]},{'type': ModelRequest,'parts':[ToolReturnPart]}]),
        tools=[],
        output_type=CityInfo,
        prompt='Tell me about Paris',
        expected_tool_choice_in_request=snapshot('ANY'),
    ),

    # === Google-specific: image-only output drops text modality ===
    Case(
        id='google-auto-image-only',
        provider='google',
        tool_choice='auto',
        expected_message_structure=snapshot([{'type': ModelRequest,'parts':[UserPromptPart]},{'type':ModelResponse,'parts':[FilePart]}]),
        tools=[],
        output_type=BinaryImage,
        prompt='Generate an image of a red kite in the sky',
        model_name='gemini-2.5-flash-image',
        expected_tool_choice_in_request=snapshot(None),
    ),
]
# fmt: on


def should_skip_case(case: Case) -> str | None:
    """Check if a case should be skipped based on provider availability."""
    if case.skip_reason:
        return case.skip_reason

    _, is_available = PROVIDER_MODELS.get(case.provider, (None, lambda: False))
    if callable(is_available):
        if not is_available():
            return f'{case.provider} not installed'  # pragma: no cover
    elif not is_available:  # pragma: no cover
        return f'{case.provider} not installed'

    return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def api_keys(
    openai_api_key: str,
    anthropic_api_key: str,
    groq_api_key: str,
    mistral_api_key: str,
    gemini_api_key: str,
    huggingface_api_key: str,
) -> dict[str, str]:
    """Collect all API keys into a dict."""
    return {
        'openai': openai_api_key,
        'anthropic': anthropic_api_key,
        'groq': groq_api_key,
        'mistral': mistral_api_key,
        'google': gemini_api_key,
        'huggingface': huggingface_api_key,
    }


# =============================================================================
# Main parametrized test
# =============================================================================


@pytest.mark.parametrize('case', CASES, ids=lambda c: c.id)
async def test_tool_choice(
    case: Case, api_keys: dict[str, str], bedrock_provider: Any, allow_model_requests: None, vcr: Any
):
    """Test tool_choice behavior across providers.

    This test verifies:
    1. The tool_choice setting is correctly sent to the provider API
    2. The message structure matches expected patterns for each scenario

    For 'required' and list tool_choice, uses direct model.request() since these
    force tool use on every request which would create infinite loops in agent.run().
    """
    skip_reason = should_skip_case(case)
    if skip_reason:
        pytest.skip(skip_reason)

    model = get_model(case.provider, api_keys, bedrock_provider, model_name=case.model_name)
    settings: ModelSettings = {'tool_choice': case.tool_choice}

    if case.use_direct_request:
        # Direct model.request() for required/list - single API call
        tool_defs = [make_tool_def('get_weather', 'Get weather for a city', 'city')]
        if len(case.tools) > 1:
            tool_defs.append(make_tool_def('get_time', 'Get time in a timezone', 'timezone'))

        params = ModelRequestParameters(
            function_tools=tool_defs,
            allow_text_output=True,
        )

        response = await model.request(
            [ModelRequest.user_text_prompt(case.prompt)],
            settings,
            params,
        )

        # Verify response has tool call(s) - simplified structure check
        response_structure = [type(p).__name__ for p in response.parts]
        assert response_structure == case.expected_message_structure

        # Verify tool_choice was sent correctly
        if case.expected_tool_choice_in_request is not None and vcr and vcr.requests:
            actual_tool_choice = get_tool_choice_from_cassette(vcr, case.provider)
            assert actual_tool_choice == case.expected_tool_choice_in_request, (
                f'Expected tool_choice={case.expected_tool_choice_in_request}, got {actual_tool_choice}'
            )
    else:
        # Agent-based test for auto, none, tools-plus-output
        if case.output_type:
            agent: Agent[None, Any] = Agent(model, tools=case.tools, output_type=case.output_type)
        else:
            agent = Agent(model, tools=case.tools)

        result = await agent.run(
            case.prompt,
            model_settings=settings,
            usage_limits=UsageLimits(output_tokens_limit=5000),
        )

        # Verify tool_choice was sent correctly (if expected value provided and cassette has requests)
        if case.expected_tool_choice_in_request is not None and vcr and vcr.requests:
            actual_tool_choice = get_tool_choice_from_cassette(vcr, case.provider)
            assert actual_tool_choice == case.expected_tool_choice_in_request, (
                f'Expected tool_choice={case.expected_tool_choice_in_request}, got {actual_tool_choice}'
            )

        # Verify message structure - snapshot stored per-case, updated with --inline-snapshot=create
        messages = result.all_messages()
        assert len(messages) >= 2, f'Expected at least 2 messages, got {len(messages)}'

        message_structure = get_message_structure(messages)
        assert message_structure == case.expected_message_structure


# =============================================================================
# Provider-specific edge case tests
# =============================================================================


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_thinking_mode_rejects_required(
    anthropic_api_key: str,
    allow_model_requests: None,
):
    """Anthropic with thinking mode enabled rejects tool_choice='required'."""
    from pydantic_ai.exceptions import UserError

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key),
    )
    agent: Agent[None, str] = Agent(model, tools=[get_weather])

    settings: AnthropicModelSettings = {
        'tool_choice': 'required',
        'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
    }

    with pytest.raises(UserError, match='thinking'):
        await agent.run("What's the weather?", model_settings=settings)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_thinking_mode_rejects_specific_tool(
    anthropic_api_key: str,
    allow_model_requests: None,
):
    """Anthropic with thinking mode enabled rejects tool_choice=[specific_tool]."""
    from pydantic_ai.exceptions import UserError

    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key=anthropic_api_key),
    )
    agent: Agent[None, str] = Agent(model, tools=[get_weather, get_time])

    settings: AnthropicModelSettings = {
        'tool_choice': ['get_weather'],
        'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1000},
    }

    with pytest.raises(UserError, match='thinking'):
        await agent.run("What's the weather?", model_settings=settings)
