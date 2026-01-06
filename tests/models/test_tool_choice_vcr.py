"""Cross-provider VCR tests for tool_choice functionality.

This module consolidates tool_choice integration tests across all providers.
Tests are parametrized by provider and scenario, using cassettes recorded
against live APIs.

Key behaviors tested:
- tool_choice='auto': Model decides whether to use tools
- tool_choice='none': Function tools disabled, text/output tools available
- tool_choice='required': Must use a tool (direct model requests)
- tool_choice=[list]: Specific tools only (direct model requests)
- ToolsPlusOutput: Specific function tools + output tools
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRequest, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings, ToolChoice, ToolsPlusOutput
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


@dataclass
class Case:
    """A single test case for tool_choice behavior."""

    id: str
    provider: str
    tool_choice: ToolChoice
    expected_message_structure: Any  # snapshot() stored here per case
    tools: list[Callable[..., str]] = field(default_factory=lambda: [get_weather])
    output_type: OutputSpec[Any] | None = None
    prompt: str = "What's the weather in Paris?"
    # Expected values - set to None to skip assertion
    expected_tool_choice_in_request: Any = None
    skip_reason: str | None = None
    expected_warning_match: str | None = None  # Regex pattern for expected UserWarning
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


def get_model(provider: str, api_keys: dict[str, str], bedrock_provider: Any = None) -> Model:
    """Create a model instance for the given provider."""
    model_name, _ = PROVIDER_MODELS[provider]

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
    type: str
    parts: list[str]


def get_message_structure(messages: list[ModelRequest | ModelResponse]) -> list[_MessageStructure]:
    """Extract simplified message structure for snapshot comparison."""
    result: list[_MessageStructure] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            result.append(
                {
                    'type': 'request',
                    'parts': [type(p).__name__ for p in msg.parts],
                }
            )
        else:
            result.append(
                {
                    'type': 'response',
                    'parts': [type(p).__name__ for p in msg.parts],
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
    Case('openai-auto-uses-tool', 'openai', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='auto'),
    Case('anthropic-auto-uses-tool', 'anthropic', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='auto'),
    Case('groq-auto-uses-tool', 'groq', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='auto'),
    Case('mistral-auto-uses-tool', 'mistral', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='auto'),
    Case('google-auto-uses-tool', 'google', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='AUTO'),
    Case('bedrock-auto-uses-tool', 'bedrock', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='auto'),
    # Together backend returns 500 Internal Server Error on the second request when continuing
    # a tool conversation (first request with tool_choice='auto' succeeds and returns tool call,
    # but the follow-up request with tool result fails). Other backends (novita, nscale) either
    # don't support tool calling or return 400 errors. See test_together_500_on_tool_continuation.
    Case('huggingface-auto-uses-tool', 'huggingface', 'auto', snapshot([]), expected_tool_choice_in_request='auto', skip_reason='Together backend 500s on tool continuation'),
    Case('openai_responses-auto-uses-tool', 'openai_responses', 'auto', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='auto'),

    # === tool_choice='none' - Function tools disabled, text response ===
    Case('openai-none-text-response', 'openai', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='none'),
    Case('anthropic-none-text-response', 'anthropic', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='none', prompt='Say hello'),
    Case('groq-none-text-response', 'groq', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='none'),
    Case('mistral-none-text-response', 'mistral', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request=None),
    Case('google-none-text-response', 'google', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request='NONE'),
    Case('bedrock-none-text-response', 'bedrock', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request=None),
    Case('huggingface-none-text-response', 'huggingface', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), expected_tool_choice_in_request=None),
    Case('openai_responses-none-text-response', 'openai_responses', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','TextPart']}]), expected_tool_choice_in_request='none'),

    # === tool_choice='required' - Must use tool (direct model request) ===
    Case('openai-required-forces-tool', 'openai', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='required', use_direct_request=True),
    Case('anthropic-required-forces-tool', 'anthropic', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='any', use_direct_request=True),
    Case('groq-required-forces-tool', 'groq', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='required', use_direct_request=True),
    Case('mistral-required-forces-tool', 'mistral', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='required', use_direct_request=True),
    Case('google-required-forces-tool', 'google', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='ANY', use_direct_request=True),
    Case('bedrock-required-forces-tool', 'bedrock', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='any', use_direct_request=True),
    Case('huggingface-required-forces-tool', 'huggingface', 'required', snapshot(['ToolCallPart']), expected_tool_choice_in_request='required', use_direct_request=True),
    Case('openai_responses-required-forces-tool', 'openai_responses', 'required', snapshot(['ThinkingPart','ToolCallPart']), expected_tool_choice_in_request='required', use_direct_request=True),

    # === tool_choice=['specific_tool'] - Force specific tool (direct model request) ===
    Case('openai-list-single-tool', 'openai', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('anthropic-list-single-tool', 'anthropic', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('groq-list-single-tool', 'groq', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('mistral-list-single-tool', 'mistral', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('google-list-single-tool', 'google', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('bedrock-list-single-tool', 'bedrock', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('huggingface-list-single-tool', 'huggingface', ['get_weather'], snapshot(['ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),
    Case('openai_responses-list-single-tool', 'openai_responses', ['get_weather'], snapshot(['ThinkingPart','ToolCallPart']), tools=[get_weather, get_time], use_direct_request=True),

    # === tool_choice='none' with structured output - output tool still works ===
    Case('openai-none-with-output', 'openai', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('anthropic-none-with-output', 'anthropic', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('groq-none-with-output', 'groq', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('mistral-none-with-output', 'mistral', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('google-none-with-output', 'google', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('bedrock-none-with-output', 'bedrock', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('huggingface-none-with-output', 'huggingface', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),
    Case('openai_responses-none-with-output', 'openai_responses', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]), output_type=CityInfo, prompt='Tell me about Paris',
         expected_warning_match="tool_choice='none'"),

    # === tool_choice='none' with text+structured output - triggers (tool_names, 'auto') branch ===
    # Using (str, CityInfo) tuple allows text output AND creates output tool, hitting the 'auto' mode path
    Case('openai-none-with-output-text-allowed', 'openai', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['TextPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('anthropic-none-with-output-text-allowed', 'anthropic', 'none', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('groq-none-with-output-text-allowed', 'groq', 'none', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('mistral-none-with-output-text-allowed', 'mistral', 'none', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('google-none-with-output-text-allowed', 'google', 'none', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('bedrock-none-with-output-text-allowed', 'bedrock', 'none', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('huggingface-none-with-output-text-allowed', 'huggingface', 'none', snapshot([{'type':'request','parts':['UserPromptPart']}, {'type':'response','parts':['ToolCallPart']}, {'type':'request','parts':['ToolReturnPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),
    Case('openai_responses-none-with-output-text-allowed', 'openai_responses', 'none', snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','TextPart']}]), output_type=(str, CityInfo), prompt='Tell me about Paris briefly',
         expected_warning_match="tool_choice='none'"),

    # === ToolsPlusOutput - specific function tools + output tools ===
    Case('openai-tools-plus-output', 'openai', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('anthropic-tools-plus-output', 'anthropic', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('groq-tools-plus-output', 'groq', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart','ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('mistral-tools-plus-output', 'mistral', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('google-tools-plus-output', 'google', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('bedrock-tools-plus-output', 'bedrock', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('huggingface-tools-plus-output', 'huggingface', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ToolCallPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart','ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
    Case('openai_responses-tools-plus-output', 'openai_responses', ToolsPlusOutput(function_tools=['get_weather']), snapshot([{'type':'request','parts':['UserPromptPart']},{'type':'response','parts':['ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']},{'type':'response','parts':['ThinkingPart','ThinkingPart','ToolCallPart']},{'type':'request','parts':['ToolReturnPart']}]),
         tools=[get_weather, get_time], output_type=CityInfo, prompt='Get weather for Paris and summarize'),
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

    model = get_model(case.provider, api_keys, bedrock_provider)
    settings: ModelSettings = {'tool_choice': case.tool_choice}

    import warnings

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

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter('always')
            result = await agent.run(
                case.prompt,
                model_settings=settings,
                usage_limits=UsageLimits(output_tokens_limit=5000),
            )

            # Check expected warning was raised
            if case.expected_warning_match:
                matching = [w for w in caught_warnings if case.expected_warning_match in str(w.message)]
                assert matching, (
                    f"Expected warning matching '{case.expected_warning_match}', got {[str(w.message) for w in caught_warnings]}"
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
