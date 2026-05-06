import asyncio
import json
import re
import sys
import warnings
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, Union

import pytest
from dirty_equals import IsJson
from pydantic import BaseModel, TypeAdapter, field_validator
from pydantic_core import ErrorDetails, to_json
from typing_extensions import Self

from pydantic_ai import (
    AbstractToolset,
    Agent,
    AgentStreamEvent,
    AudioUrl,
    BinaryContent,
    BinaryImage,
    CallDeferred,
    CombinedToolset,
    DocumentUrl,
    ExternalToolset,
    FilePart,
    FunctionToolset,
    ImageUrl,
    IncompleteToolCall,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelProfile,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelRetry,
    PrefixedToolset,
    RetryPromptPart,
    RunContext,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    VideoUrl,
    capture_run_messages,
)
from pydantic_ai._output import (
    NativeOutput,
    NativeOutputSchema,
    OutputSpec,
    PromptedOutput,
    TextOutput,
)
from pydantic_ai.agent import AgentRunResult, WrapperAgent
from pydantic_ai.builtin_tools import (
    CodeExecutionTool,
    MCPServerTool,
    WebSearchTool,
    WebSearchUserLocation,
)
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.messages import ModelResponseStreamEvent
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputObjectDefinition, StructuredDict, ToolOutput
from pydantic_ai.providers import Provider
from pydantic_ai.result import RunUsage
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition, ToolDenied
from pydantic_ai.usage import RequestUsage

if TYPE_CHECKING:
    from pydantic_ai.providers.alibaba import AlibabaProvider
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.cohere import CohereProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.fireworks import FireworksProvider
    from pydantic_ai.providers.github import GitHubProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_gla import GoogleGLAProvider  # pyright: ignore[reportDeprecated]
    from pydantic_ai.providers.google_vertex import GoogleVertexProvider  # pyright: ignore[reportDeprecated]
    from pydantic_ai.providers.grok import GrokProvider  # pyright: ignore[reportDeprecated]
    from pydantic_ai.providers.groq import GroqProvider
    from pydantic_ai.providers.heroku import HerokuProvider
    from pydantic_ai.providers.litellm import LiteLLMProvider
    from pydantic_ai.providers.mistral import MistralProvider
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider
    from pydantic_ai.providers.nebius import NebiusProvider
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from pydantic_ai.providers.ovhcloud import OVHcloudProvider
    from pydantic_ai.providers.sambanova import SambaNovaProvider
    from pydantic_ai.providers.together import TogetherProvider
    from pydantic_ai.providers.vercel import VercelProvider
else:
    try:
        from pydantic_ai.providers.alibaba import AlibabaProvider
        from pydantic_ai.providers.azure import AzureProvider
        from pydantic_ai.providers.cerebras import CerebrasProvider
        from pydantic_ai.providers.deepseek import DeepSeekProvider
        from pydantic_ai.providers.fireworks import FireworksProvider
        from pydantic_ai.providers.github import GitHubProvider
        from pydantic_ai.providers.grok import GrokProvider  # pyright: ignore[reportDeprecated]
        from pydantic_ai.providers.heroku import HerokuProvider
        from pydantic_ai.providers.moonshotai import MoonshotAIProvider
        from pydantic_ai.providers.nebius import NebiusProvider
        from pydantic_ai.providers.ollama import OllamaProvider
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.providers.openrouter import OpenRouterProvider
        from pydantic_ai.providers.ovhcloud import OVHcloudProvider
        from pydantic_ai.providers.sambanova import SambaNovaProvider
        from pydantic_ai.providers.together import TogetherProvider
        from pydantic_ai.providers.vercel import VercelProvider
    except ImportError:  # pragma: lax no cover
        AlibabaProvider = AzureProvider = CerebrasProvider = DeepSeekProvider = None  # type: ignore
        FireworksProvider = GitHubProvider = GrokProvider = HerokuProvider = None  # type: ignore
        MoonshotAIProvider = NebiusProvider = OllamaProvider = OpenAIProvider = None  # type: ignore
        OpenRouterProvider = OVHcloudProvider = SambaNovaProvider = None  # type: ignore
        TogetherProvider = VercelProvider = None  # type: ignore

    try:
        from pydantic_ai.providers.anthropic import AnthropicProvider
    except ImportError:  # pragma: lax no cover
        AnthropicProvider = None

    try:
        from pydantic_ai.providers.cohere import CohereProvider
    except ImportError:  # pragma: lax no cover
        CohereProvider = None

    try:
        from pydantic_ai.providers.google import GoogleProvider
    except ImportError:  # pragma: lax no cover
        GoogleProvider = None

    try:
        from pydantic_ai.providers.google_gla import GoogleGLAProvider  # pyright: ignore[reportDeprecated]
    except ImportError:  # pragma: lax no cover
        GoogleGLAProvider = None

    try:
        from pydantic_ai.providers.google_vertex import GoogleVertexProvider  # pyright: ignore[reportDeprecated]
    except ImportError:  # pragma: lax no cover
        GoogleVertexProvider = None

    try:
        from pydantic_ai.providers.groq import GroqProvider
    except ImportError:  # pragma: lax no cover
        GroqProvider = None

    try:
        from pydantic_ai.providers.litellm import LiteLLMProvider
    except ImportError:  # pragma: lax no cover
        LiteLLMProvider = None

    try:
        from pydantic_ai.providers.mistral import MistralProvider
    except ImportError:  # pragma: lax no cover
        MistralProvider = None

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInstance, IsNow, IsStr, TestEnv

pytestmark = pytest.mark.anyio

requires_openai = pytest.mark.skipif(OpenAIProvider is None, reason='openai not installed')  # pyright: ignore[reportUnnecessaryComparison]
requires_anthropic = pytest.mark.skipif(AnthropicProvider is None, reason='anthropic not installed')  # pyright: ignore[reportUnnecessaryComparison]
requires_cohere = pytest.mark.skipif(CohereProvider is None, reason='cohere not installed')  # pyright: ignore[reportUnnecessaryComparison]
requires_google = pytest.mark.skipif(GoogleProvider is None, reason='google-genai not installed')  # pyright: ignore[reportUnnecessaryComparison]
requires_google_gla = pytest.mark.skipif(GoogleGLAProvider is None, reason='google-gla deps not installed')  # pyright: ignore[reportUnnecessaryComparison, reportDeprecated]
requires_google_vertex = pytest.mark.skipif(GoogleVertexProvider is None, reason='google-auth not installed')  # pyright: ignore[reportUnnecessaryComparison, reportDeprecated]
requires_groq = pytest.mark.skipif(GroqProvider is None, reason='groq not installed')  # pyright: ignore[reportUnnecessaryComparison]
requires_litellm = pytest.mark.skipif(LiteLLMProvider is None, reason='litellm not installed')  # pyright: ignore[reportUnnecessaryComparison]
requires_mistral = pytest.mark.skipif(MistralProvider is None, reason='mistral not installed')  # pyright: ignore[reportUnnecessaryComparison]


def test_result_tuple():
    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), output_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert isinstance(result.run_id, str)
    assert result.output == ('foo', 'bar')
    assert result.response == snapshot(
        ModelResponse(
            parts=[ToolCallPart(tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr())],
            usage=RequestUsage(input_tokens=51, output_tokens=7),
            model_name='function:return_tuple:',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


class Person(BaseModel):
    name: str


# Generic classes for testing tool name sanitization with generic types
T = TypeVar('T')


class ResultGeneric(BaseModel, Generic[T]):
    """A generic result class."""

    value: T
    success: bool


class StringData(BaseModel):
    text: str


def test_result_list_of_models_with_stringified_response():
    def return_list(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # Simulate providers that return the nested payload as a JSON string under "response"
        args_json = json.dumps(
            {
                'response': json.dumps(
                    [
                        {'name': 'John Doe'},
                        {'name': 'Jane Smith'},
                    ]
                )
            }
        )
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_list), output_type=list[Person])

    result = agent.run_sync('Hello')
    assert result.output == snapshot(
        [
            Person(name='John Doe'),
            Person(name='Jane Smith'),
        ]
    )


class Foo(BaseModel):
    a: int
    b: str


def test_result_pydantic_model():
    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo)

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {'a': 1, 'b': 'foo'}


def test_result_pydantic_model_retry():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        if len(messages) == 1:
            args_json = '{"a": "wrong", "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo)

    assert agent.name is None

    result = agent.run_sync('Hello')
    assert agent.name == 'agent'
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": "wrong", "b": "foo"}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='final_result',
                        content=[
                            ErrorDetails(
                                type='int_parsing',
                                loc=('a',),
                                msg='Input should be a valid integer, unable to parse string as an integer',
                                input='wrong',
                            )
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 42, "b": "foo"}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=89, output_tokens=14),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.all_messages_json().startswith(b'[{"parts":[{"content":"Hello",')


def test_result_pydantic_model_validation_error():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 1, "b": "foo"}'
        else:
            args_json = '{"a": 1, "b": "bar"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    class Bar(BaseModel):
        a: int
        b: str

        @field_validator('b')
        def check_b(cls, v: str) -> str:
            if v == 'foo':
                raise ValueError('must not be foo')
            return v

    agent = Agent(FunctionModel(return_model), output_type=Bar)

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Bar)
    assert result.output.model_dump() == snapshot({'a': 1, 'b': 'bar'})
    messages_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in result.all_messages()]
    assert messages_part_kinds == snapshot(
        [
            ('request', ['user-prompt']),
            ('response', ['tool-call']),
            ('request', ['retry-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
        ]
    )

    user_retry = result.all_messages()[2]
    assert isinstance(user_retry, ModelRequest)
    retry_prompt = user_retry.parts[0]
    assert isinstance(retry_prompt, RetryPromptPart)
    assert retry_prompt.model_response() == snapshot("""\
1 validation error:
```json
[
  {
    "type": "value_error",
    "loc": [
      "b"
    ],
    "msg": "Value error, must not be foo",
    "input": "foo"
  }
]
```

Fix the errors and try again.\
""")


def test_output_validator():
    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        if len(messages) == 1:
            args_json = '{"a": 41, "b": "foo"}'
        else:
            args_json = '{"a": 42, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        assert ctx.tool_name == 'final_result'
        if o.a == 42:
            return o
        else:
            raise ModelRetry('"a" should be 42')

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)
    assert result.output.model_dump() == {'a': 42, 'b': 'foo'}
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 41, "b": "foo"}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='"a" should be 42',
                        tool_name='final_result',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args='{"a": 42, "b": "foo"}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=63, output_tokens=14),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_output_validator_retries():
    """Test that ctx.retry and ctx.max_retries are correctly tracked in RunContext for output validators."""
    retries_log: list[int] = []
    max_retries_log: list[int] = []
    target_retries = 3

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # Always return the same value, let the validator control retries
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=Foo, output_retries=target_retries)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        retries_log.append(ctx.retry)
        max_retries_log.append(ctx.max_retries)
        # Succeed on the last retry
        if ctx.retry == target_retries:
            return o
        else:
            raise ModelRetry(f'Retry {ctx.retry}')

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)

    # Should have been called target_retries + 1 times (0, 1, 2, 3)
    assert retries_log == [0, 1, 2, 3]
    assert max_retries_log == [target_retries] * (target_retries + 1)


def test_output_function_retries():
    """Test that ctx.retry and ctx.max_retries are correctly tracked in RunContext for output functions."""
    retries_log: list[int] = []
    max_retries_log: list[int] = []
    target_retries = 3

    def get_weather(ctx: RunContext[None], text: str) -> str:
        retries_log.append(ctx.retry)
        max_retries_log.append(ctx.max_retries)
        if ctx.retry == target_retries:
            return f'Weather: {text}'
        else:
            raise ModelRetry(f'Retry {ctx.retry}')

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='sunny')])

    agent = Agent(FunctionModel(return_model), output_type=TextOutput(get_weather), output_retries=target_retries)

    result = agent.run_sync('Hello')
    assert result.output == 'Weather: sunny'

    # Should have been called target_retries + 1 times (0, 1, 2, 3)
    assert retries_log == [0, 1, 2, 3]
    assert max_retries_log == [target_retries] * (target_retries + 1)


def test_tool_output_function_retries():
    """Test that ctx.retry and ctx.max_retries are correctly tracked in RunContext for tool output functions."""
    retries_log: list[int] = []
    max_retries_log: list[int] = []
    target_retries = 3

    def get_weather(ctx: RunContext[None], city: str) -> str:
        retries_log.append(ctx.retry)
        max_retries_log.append(ctx.max_retries)
        if ctx.retry == target_retries:
            return f'Weather in {city}'
        else:
            raise ModelRetry(f'Retry {ctx.retry}')

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_model), output_type=get_weather, output_retries=target_retries)

    result = agent.run_sync('Hello')
    assert result.output == 'Weather in Mexico City'

    # Should have been called target_retries + 1 times (0, 1, 2, 3)
    assert retries_log == [0, 1, 2, 3]
    assert max_retries_log == [target_retries] * (target_retries + 1)


def test_tool_output_max_retries_overrides_agent_retries():
    """ToolOutput.max_retries takes priority over Agent retries. Regression test for #4678."""
    retries_log: list[int] = []
    max_retries_log: list[int] = []
    target_retries = 5

    def get_weather(ctx: RunContext[None], city: str) -> str:
        retries_log.append(ctx.retry)
        max_retries_log.append(ctx.max_retries)
        if ctx.retry < target_retries:
            raise ModelRetry(f'Retry {ctx.retry}')
        return f'Weather in {city}'

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    # Agent retries=2 (lower than ToolOutput), ToolOutput max_retries=5
    # The ToolOutput value should take priority, allowing 5 retries
    agent = Agent(
        FunctionModel(return_model),
        output_type=ToolOutput(get_weather, max_retries=target_retries),
        tool_retries=2,
        output_retries=2,
    )

    result = agent.run_sync('Hello')
    assert result.output == 'Weather in Mexico City'
    assert retries_log == [0, 1, 2, 3, 4, 5]
    assert max_retries_log == [target_retries] * (target_retries + 1)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=6),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry 0',
                        tool_name='final_result',
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
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=60, output_tokens=12),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry 1',
                        tool_name='final_result',
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
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=69, output_tokens=18),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry 2',
                        tool_name='final_result',
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
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=78, output_tokens=24),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry 3',
                        tool_name='final_result',
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
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=87, output_tokens=30),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry 4',
                        tool_name='final_result',
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
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=96, output_tokens=36),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_tool_output_max_retries_per_tool():
    """Multiple ToolOutputs with different max_retries are tracked independently. Regression test for #4678."""
    a_max_retries_seen: list[int] = []
    b_max_retries_seen: list[int] = []

    def output_a(ctx: RunContext[None], value: str) -> str:
        a_max_retries_seen.append(ctx.max_retries)
        if ctx.retry < 3:
            raise ModelRetry(f'Retry A {ctx.retry}')
        return f'A: {value}'

    def output_b(ctx: RunContext[None], value: str) -> str:
        b_max_retries_seen.append(ctx.max_retries)
        raise ModelRetry(f'Retry B {ctx.retry}')

    tool_names: dict[str, str] = {}
    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        assert info.output_tools is not None
        tool_names.update({t.name: t.name for t in info.output_tools})

        name_a = next(n for n in tool_names if 'output_a' in n)
        name_b = next(n for n in tool_names if 'output_b' in n)

        # First call output_b to verify it sees max_retries=1, then switch to output_a
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(name_b, '{"value": "x"}')])
        return ModelResponse(parts=[ToolCallPart(name_a, '{"value": "hello"}')])

    # output_a gets 3 retries, output_b gets 1 — agent default is 0
    agent = Agent(
        FunctionModel(return_model),
        output_type=[
            ToolOutput(output_a, max_retries=3),
            ToolOutput(output_b, max_retries=1),
        ],
        tool_retries=0,
        output_retries=0,
    )

    result = agent.run_sync('Hello')
    assert result.output == 'A: hello'
    # output_b was called once and saw its own max_retries=1
    assert b_max_retries_seen == [1]
    # output_a was called 4 times (retries 0-3) and saw its own max_retries=3
    assert a_max_retries_seen == [3, 3, 3, 3]
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result_output_b',
                        args='{"value": "x"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry B 0',
                        tool_name='final_result_output_b',
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
                        tool_name='final_result_output_a',
                        args='{"value": "hello"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=61, output_tokens=10),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry A 0',
                        tool_name='final_result_output_a',
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
                        tool_name='final_result_output_a',
                        args='{"value": "hello"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=71, output_tokens=15),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry A 1',
                        tool_name='final_result_output_a',
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
                        tool_name='final_result_output_a',
                        args='{"value": "hello"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=81, output_tokens=20),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Retry A 2',
                        tool_name='final_result_output_a',
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
                        tool_name='final_result_output_a',
                        args='{"value": "hello"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=91, output_tokens=25),
                model_name='function:return_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result_output_a',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


class TestPartialOutput:
    """Tests for `ctx.partial_output` flag in output validators and output functions."""

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_streaming.py::TestPartialOutput` as well

    def test_output_validator_text(self):
        """Test that output validators receive correct value for `partial_output` with text output."""
        call_log: list[tuple[str, bool]] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Hello world!')])

        agent = Agent(FunctionModel(return_model))

        @agent.output_validator
        def validate_output(ctx: RunContext[None], output: str) -> str:
            call_log.append((output, ctx.partial_output))
            return output

        result = agent.run_sync('test')

        assert result.output == 'Hello world!'
        assert call_log == snapshot([('Hello world!', False)])

    def test_output_validator_structured(self):
        """Test that output validators receive correct value for `partial_output` with structured output."""
        call_log: list[tuple[Foo, bool]] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool_name = info.output_tools[0].name
            args_json = '{"a": 42, "b": "foo"}'
            return ModelResponse(parts=[ToolCallPart(tool_name, args_json)])

        agent = Agent(FunctionModel(return_model), output_type=Foo)

        @agent.output_validator
        def validate_output(ctx: RunContext[None], output: Foo) -> Foo:
            call_log.append((output, ctx.partial_output))
            return output

        result = agent.run_sync('test')

        assert result.output == Foo(a=42, b='foo')
        assert call_log == snapshot([(Foo(a=42, b='foo'), False)])

    def test_output_function_text(self):
        """Test that output functions receive correct value for `partial_output` with text output."""
        call_log: list[tuple[str, bool]] = []

        def process_output(ctx: RunContext[None], text: str) -> str:
            call_log.append((text, ctx.partial_output))
            return text.upper()

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Hello world!')])

        agent = Agent(FunctionModel(return_model), output_type=TextOutput(process_output))
        result = agent.run_sync('test')

        assert result.output == 'HELLO WORLD!'
        assert call_log == snapshot([('Hello world!', False)])

    def test_output_function_structured(self):
        """Test that output functions receive correct value for `partial_output` with structured output."""
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool_name = info.output_tools[0].name
            args_json = '{"a": 21, "b": "foo"}'
            return ModelResponse(parts=[ToolCallPart(tool_name, args_json)])

        agent = Agent(FunctionModel(return_model), output_type=process_foo)
        result = agent.run_sync('test')

        assert result.output == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    def test_output_function_structured_get_output(self):
        """Test that output functions receive correct value for `partial_output` with sync run."""
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool_name = info.output_tools[0].name
            args_json = '{"a": 21, "b": "foo"}'
            return ModelResponse(parts=[ToolCallPart(tool_name, args_json)])

        agent = Agent(FunctionModel(return_model), output_type=ToolOutput(process_foo, name='my_output'))
        result = agent.run_sync('test')

        assert result.output == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    def test_output_function_structured_stream_output_only(self):
        """Test that output functions receive correct value for `partial_output` with sync run."""
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool_name = info.output_tools[0].name
            args_json = '{"a": 21, "b": "foo"}'
            return ModelResponse(parts=[ToolCallPart(tool_name, args_json)])

        agent = Agent(FunctionModel(return_model), output_type=ToolOutput(process_foo, name='my_output'))
        result = agent.run_sync('test')

        assert result.output == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_streaming.py::TestPartialOutput` as well


def test_plain_response_then_tuple():
    call_index = 0

    def return_tuple(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_index

        assert info.output_tools is not None
        call_index += 1
        if call_index == 1:
            return ModelResponse(parts=[TextPart('hello')])
        else:
            args_json = '{"response": ["foo", "bar"]}'
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(return_tuple), output_type=ToolOutput(tuple[str, str]))

    result = agent.run_sync('Hello')
    assert result.output == ('foo', 'bar')
    assert call_index == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:return_tuple:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please include your response in a tool call.',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr())
                ],
                usage=RequestUsage(input_tokens=68, output_tokens=8),
                model_name='function:return_tuple:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result._output_tool_name == 'final_result'  # pyright: ignore[reportPrivateUsage]
    assert result.all_messages(output_tool_return_content='foobar')[-1] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='final_result', content='foobar', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )
    assert result.all_messages()[-1] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='final_result',
                    content='Final result processed.',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_output_tool_return_content_str_return():
    agent = Agent('test')

    result = agent.run_sync('Hello')
    assert result.output == 'success (no tool calls)'
    assert result.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            usage=RequestUsage(input_tokens=51, output_tokens=4),
            model_name='test',
            timestamp=IsDatetime(),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )

    msg = re.escape('Cannot set output tool return content when the return type is `str`.')
    with pytest.raises(ValueError, match=msg):
        result.all_messages(output_tool_return_content='foobar')


def test_output_tool_return_content_no_tool():
    agent = Agent('test', output_type=int)

    result = agent.run_sync('Hello')
    assert result.output == 0
    result._output_tool_name = 'wrong'  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(LookupError, match=re.escape("No tool call found with tool name 'wrong'.")):
        result.all_messages(output_tool_return_content='foobar')


def test_response_tuple():
    m = TestModel()

    agent = Agent(m, output_type=tuple[str, str])

    result = agent.run_sync('Hello')
    assert result.output == snapshot(('a', 'a'))

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.output_mode == 'tool'
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_output is False

    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 1
    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'response': {
                            'maxItems': 2,
                            'minItems': 2,
                            'prefixItems': [{'type': 'string'}, {'type': 'string'}],
                            'type': 'array',
                        }
                    },
                    'required': ['response'],
                    'type': 'object',
                },
                outer_typed_dict_key='response',
                kind='output',
            )
        ]
    )


def upcase(text: str) -> str:
    return text.upper()


@pytest.mark.parametrize(
    'input_union_callable',
    [
        lambda: Union[str, Foo],  # noqa: UP007
        lambda: Union[Foo, str],  # noqa: UP007
        lambda: str | Foo,
        lambda: Foo | str,
        lambda: [Foo, str],
        lambda: [TextOutput(upcase), ToolOutput(Foo)],
    ],
    ids=[
        'Union[str, Foo]',
        'Union[Foo, str]',
        'str | Foo',
        'Foo | str',
        '[Foo, str]',
        '[TextOutput(upcase), ToolOutput(Foo)]',
    ],
)
def test_response_union_allow_str(input_union_callable: Callable[[], Any]):
    try:
        union = input_union_callable()
    except TypeError:  # pragma: lax no cover
        pytest.skip('Python version does not support `|` syntax for unions')

    m = TestModel()
    agent: Agent[None, str | Foo] = Agent(m, output_type=union)

    got_tool_call_name = 'unset'

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return o

    assert agent._output_schema.allows_text  # pyright: ignore[reportPrivateUsage]

    result = agent.run_sync('Hello')
    assert isinstance(result.output, str)
    assert result.output.lower() == snapshot('success (no tool calls)')
    assert got_tool_call_name == snapshot(None)

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_output is True

    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 1

    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'a': {'type': 'integer'},
                        'b': {'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
@pytest.mark.parametrize(
    'union_code',
    [
        pytest.param('OutputType = Union[Foo, Bar]'),
        pytest.param('OutputType = [Foo, Bar]'),
        pytest.param('OutputType = [ToolOutput(Foo), ToolOutput(Bar)]'),
        pytest.param('OutputType = Foo | Bar'),
        pytest.param('OutputType: TypeAlias = Foo | Bar'),
        pytest.param(
            'type OutputType = Foo | Bar', marks=pytest.mark.skipif(sys.version_info < (3, 12), reason='3.12+')
        ),
    ],
)
def test_response_multiple_return_tools(create_module: Callable[[str], Any], union_code: str):
    module_code = f'''
from pydantic import BaseModel
from typing import Union
from typing_extensions import TypeAlias
from pydantic_ai import ToolOutput

class Foo(BaseModel):
    a: int
    b: str


class Bar(BaseModel):
    """This is a bar model."""

    b: str

{union_code}
    '''

    mod = create_module(module_code)

    m = TestModel()
    agent = Agent(m, output_type=mod.OutputType)
    got_tool_call_name = 'unset'

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Any) -> Any:
        nonlocal got_tool_call_name
        got_tool_call_name = ctx.tool_name
        return o

    result = agent.run_sync('Hello')
    assert result.output == mod.Foo(a=0, b='a')
    assert got_tool_call_name == snapshot('final_result_Foo')

    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.function_tools == snapshot([])
    assert m.last_model_request_parameters.allow_text_output is False

    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 2

    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_Foo',
                description='Foo: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {
                        'a': {'type': 'integer'},
                        'b': {'type': 'string'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
                kind='output',
            ),
            ToolDefinition(
                name='final_result_Bar',
                description='This is a bar model.',
                parameters_json_schema={
                    'properties': {'b': {'type': 'string'}},
                    'required': ['b'],
                    'title': 'Bar',
                    'type': 'object',
                },
                kind='output',
            ),
        ]
    )

    result = agent.run_sync('Hello', model=TestModel(seed=1))
    assert result.output == mod.Bar(b='b')
    assert got_tool_call_name == snapshot('final_result_Bar')


def test_output_type_generic_class_name_sanitization():
    """Test that generic class names with brackets are properly sanitized."""
    # This will have a name like "ResultGeneric[StringData]" which needs sanitization
    output_type = [ResultGeneric[StringData], ResultGeneric[int]]

    m = TestModel()
    agent = Agent(m, output_type=output_type)
    agent.run_sync('Hello')

    # The sanitizer should remove brackets from the generic type name
    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.output_tools is not None
    assert len(m.last_model_request_parameters.output_tools) == 2

    tool_names = [tool.name for tool in m.last_model_request_parameters.output_tools]
    assert tool_names == snapshot(['final_result_ResultGenericStringData', 'final_result_ResultGenericint'])


def test_output_type_with_two_descriptions():
    class MyOutput(BaseModel):
        """Description from docstring"""

        valid: bool

    m = TestModel()
    agent = Agent(m, output_type=ToolOutput(MyOutput, description='Description from ToolOutput'))
    result = agent.run_sync('Hello')
    assert result.output == snapshot(MyOutput(valid=False))
    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='Description from ToolOutput. Description from docstring',
                parameters_json_schema={
                    'properties': {'valid': {'type': 'boolean'}},
                    'required': ['valid'],
                    'title': 'MyOutput',
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_tool_output_union():
    class Foo(BaseModel):
        a: int
        b: str

    class Bar(BaseModel):
        c: bool

    m = TestModel()
    marker: ToolOutput[Foo | Bar] = ToolOutput(Foo | Bar, strict=False)  # type: ignore
    agent = Agent(m, output_type=marker)
    result = agent.run_sync('Hello')
    assert result.output == snapshot(Foo(a=0, b='a'))
    assert m.last_model_request_parameters is not None
    assert m.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    '$defs': {
                        'Bar': {
                            'properties': {'c': {'type': 'boolean'}},
                            'required': ['c'],
                            'title': 'Bar',
                            'type': 'object',
                        },
                        'Foo': {
                            'properties': {'a': {'type': 'integer'}, 'b': {'type': 'string'}},
                            'required': ['a', 'b'],
                            'title': 'Foo',
                            'type': 'object',
                        },
                    },
                    'properties': {'response': {'anyOf': [{'$ref': '#/$defs/Foo'}, {'$ref': '#/$defs/Bar'}]}},
                    'required': ['response'],
                    'type': 'object',
                },
                outer_typed_dict_key='response',
                strict=False,
                kind='output',
            )
        ]
    )


def test_output_type_function():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_with_run_context():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(ctx: RunContext[None], city: str) -> Weather:
        assert ctx is not None
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_bound_instance_method():
    class Weather(BaseModel):
        temperature: float
        description: str

        def get_weather(self, city: str) -> Self:
            return self

    weather = Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=weather.get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_bound_instance_method_with_run_context():
    class Weather(BaseModel):
        temperature: float
        description: str

        def get_weather(self, ctx: RunContext[None], city: str) -> Self:
            assert ctx is not None
            return self

    weather = Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=weather.get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_with_retry():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        if city != 'Mexico City':
            raise ModelRetry('City not found, I only know Mexico City')
        return Weather(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            args_json = '{"city": "New York City"}'
        else:
            args_json = '{"city": "Mexico City"}'

        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('New York City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='New York City',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "New York City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=53, output_tokens=7),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='City not found, I only know Mexico City',
                        tool_name='final_result',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=68, output_tokens=13),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_output_type_text_output_function_with_retry():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(ctx: RunContext[None], city: str) -> Weather:
        assert ctx is not None
        if city != 'Mexico City':
            raise ModelRetry('City not found, I only know Mexico City')
        return Weather(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            city = 'New York City'
        else:
            city = 'Mexico City'

        return ModelResponse(parts=[TextPart(content=city)])

    agent = Agent(FunctionModel(call_tool), output_type=TextOutput(get_weather))
    result = agent.run_sync('New York City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='New York City',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='New York City')],
                usage=RequestUsage(input_tokens=53, output_tokens=3),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='City not found, I only know Mexico City',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Mexico City')],
                usage=RequestUsage(input_tokens=70, output_tokens=5),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'output_type',
    [[str, str], [str, TextOutput(upcase)], [TextOutput(upcase), TextOutput(upcase)]],
)
def test_output_type_multiple_text_output(output_type: OutputSpec[str]):
    with pytest.raises(UserError, match='Only one `str` or `TextOutput` is allowed.'):
        Agent('test', output_type=output_type)


def test_output_type_text_output_invalid():
    def int_func(x: int) -> str:
        return str(int)  # pragma: no cover

    with pytest.raises(UserError, match='TextOutput must take a function taking a single `str` argument'):
        output_type: TextOutput[str] = TextOutput(int_func)  # type: ignore
        Agent('test', output_type=output_type)


def test_output_type_async_function():
    class Weather(BaseModel):
        temperature: float
        description: str

    async def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_with_custom_tool_name():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=ToolOutput(get_weather, name='get_weather'))
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='get_weather',
                description='The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            )
        ]
    )


def test_output_type_function_or_model():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=[get_weather, Weather])
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_get_weather',
                description='get_weather: The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            ),
            ToolDefinition(
                name='final_result_Weather',
                description='Weather: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {'temperature': {'type': 'number'}, 'description': {'type': 'string'}},
                    'required': ['temperature', 'description'],
                    'title': 'Weather',
                    'type': 'object',
                },
                kind='output',
            ),
        ]
    )


def test_output_type_text_output_function():
    def say_world(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='world')])

    agent = Agent(FunctionModel(say_world), output_type=TextOutput(upcase))
    result = agent.run_sync('hello')
    assert result.output == snapshot('WORLD')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='function:say_world:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_output_type_handoff_to_agent():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=get_weather)

    handoff_result = None

    async def handoff(city: str) -> Weather:
        result = await agent.run(f'Get me the weather in {city}')
        nonlocal handoff_result
        handoff_result = result
        return result.output

    def call_handoff_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    supervisor_agent = Agent(FunctionModel(call_handoff_tool), output_type=handoff)

    result = supervisor_agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Mexico City',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=6),
                model_name='function:call_handoff_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert handoff_result is not None
    assert handoff_result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Get me the weather in Mexico City',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"city": "Mexico City"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=57, output_tokens=6),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_output_type_multiple_custom_tools():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        return Weather(temperature=28.7, description='sunny')

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"city": "Mexico City"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(call_tool),
        output_type=[
            ToolOutput(get_weather, name='get_weather'),
            ToolOutput(Weather, name='return_weather'),
        ],
    )
    result = agent.run_sync('Mexico City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='get_weather',
                description='get_weather: The final response which ends this conversation',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'city': {'type': 'string'}},
                    'required': ['city'],
                    'type': 'object',
                },
                kind='output',
            ),
            ToolDefinition(
                name='return_weather',
                description='Weather: The final response which ends this conversation',
                parameters_json_schema={
                    'properties': {'temperature': {'type': 'number'}, 'description': {'type': 'string'}},
                    'required': ['temperature', 'description'],
                    'title': 'Weather',
                    'type': 'object',
                },
                kind='output',
            ),
        ]
    )


def test_output_type_structured_dict():
    PersonDict = StructuredDict(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['name', 'age'],
        },
        name='Person',
        description='A person',
    )
    AnimalDict = StructuredDict(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'species': {'type': 'string'},
            },
            'required': ['name', 'species'],
        },
        name='Animal',
        description='An animal',
    )

    output_tools = None

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        nonlocal output_tools
        output_tools = info.output_tools

        args_json = '{"name": "John Doe", "age": 30}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(call_tool),
        output_type=[PersonDict, AnimalDict],
    )

    result = agent.run_sync('Generate a person')

    assert result.output == snapshot({'name': 'John Doe', 'age': 30})
    assert output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result_Person',
                parameters_json_schema={
                    'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
                    'required': ['name', 'age'],
                    'title': 'Person',
                    'type': 'object',
                },
                description='A person',
                kind='output',
            ),
            ToolDefinition(
                name='final_result_Animal',
                parameters_json_schema={
                    'properties': {'name': {'type': 'string'}, 'species': {'type': 'string'}},
                    'required': ['name', 'species'],
                    'title': 'Animal',
                    'type': 'object',
                },
                description='An animal',
                kind='output',
            ),
        ]
    )


def test_output_type_structured_dict_nested():
    """Test StructuredDict with nested JSON schemas using $ref - Issue #2466."""
    # Schema with nested $ref that pydantic's generator can't resolve
    CarDict = StructuredDict(
        {
            '$defs': {
                'Tire': {
                    'type': 'object',
                    'properties': {'brand': {'type': 'string'}, 'size': {'type': 'integer'}},
                    'required': ['brand', 'size'],
                }
            },
            'type': 'object',
            'properties': {
                'make': {'type': 'string'},
                'model': {'type': 'string'},
                'tires': {'type': 'array', 'items': {'$ref': '#/$defs/Tire'}},
            },
            'required': ['make', 'model', 'tires'],
        },
        name='Car',
        description='A car with tires',
    )

    def call_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        # Verify the output tool schema has been properly transformed
        # The $refs should be inlined by InlineDefsJsonSchemaTransformer
        output_tool = info.output_tools[0]
        schema = output_tool.parameters_json_schema
        assert schema is not None

        assert schema == snapshot(
            {
                'properties': {
                    'make': {'type': 'string'},
                    'model': {'type': 'string'},
                    'tires': {
                        'items': {
                            'properties': {'brand': {'type': 'string'}, 'size': {'type': 'integer'}},
                            'required': ['brand', 'size'],
                            'type': 'object',
                        },
                        'type': 'array',
                    },
                },
                'required': ['make', 'model', 'tires'],
                'title': 'Car',
                'type': 'object',
            }
        )

        return ModelResponse(
            parts=[
                ToolCallPart(
                    output_tool.name, {'make': 'Toyota', 'model': 'Camry', 'tires': [{'brand': 'Michelin', 'size': 17}]}
                )
            ]
        )

    agent = Agent(FunctionModel(call_tool), output_type=CarDict)

    result = agent.run_sync('Generate a car')

    assert result.output == snapshot({'make': 'Toyota', 'model': 'Camry', 'tires': [{'brand': 'Michelin', 'size': 17}]})


def test_structured_dict_recursive_refs():
    class Node(BaseModel):
        nodes: list['Node'] | dict[str, 'Node']

    schema = Node.model_json_schema()
    assert schema == snapshot(
        {
            '$defs': {
                'Node': {
                    'properties': {
                        'nodes': {
                            'anyOf': [
                                {'items': {'$ref': '#/$defs/Node'}, 'type': 'array'},
                                {'additionalProperties': {'$ref': '#/$defs/Node'}, 'type': 'object'},
                            ],
                            'title': 'Nodes',
                        }
                    },
                    'required': ['nodes'],
                    'title': 'Node',
                    'type': 'object',
                }
            },
            '$ref': '#/$defs/Node',
        }
    )
    with pytest.raises(
        UserError,
        match=re.escape(
            '`StructuredDict` does not currently support recursive `$ref`s and `$defs`. See https://github.com/pydantic/pydantic/issues/12145 for more information.'
        ),
    ):
        StructuredDict(schema)


def test_default_structured_output_mode():
    class Foo(BaseModel):
        bar: str

    tool_model = TestModel(profile=ModelProfile(default_structured_output_mode='tool'))
    native_model = TestModel(
        profile=ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native'),
        custom_output_text=Foo(bar='baz').model_dump_json(),
    )
    prompted_model = TestModel(
        profile=ModelProfile(default_structured_output_mode='prompted'),
        custom_output_text=Foo(bar='baz').model_dump_json(),
    )

    tool_agent = Agent(tool_model, output_type=Foo)
    tool_agent.run_sync('Hello')
    assert tool_model.last_model_request_parameters is not None
    assert tool_model.last_model_request_parameters.output_mode == 'tool'
    assert tool_model.last_model_request_parameters.allow_text_output is False
    assert tool_model.last_model_request_parameters.output_object is None
    assert tool_model.last_model_request_parameters.output_tools == snapshot(
        [
            ToolDefinition(
                name='final_result',
                parameters_json_schema={
                    'properties': {'bar': {'type': 'string'}},
                    'required': ['bar'],
                    'title': 'Foo',
                    'type': 'object',
                },
                description='The final response which ends this conversation',
                kind='output',
            )
        ]
    )

    native_agent = Agent(native_model, output_type=Foo)
    native_agent.run_sync('Hello')
    assert native_model.last_model_request_parameters is not None
    assert native_model.last_model_request_parameters.output_mode == 'native'
    assert native_model.last_model_request_parameters.allow_text_output is True
    assert len(native_model.last_model_request_parameters.output_tools) == 0
    assert native_model.last_model_request_parameters.output_object == snapshot(
        OutputObjectDefinition(
            json_schema={
                'properties': {'bar': {'type': 'string'}},
                'required': ['bar'],
                'title': 'Foo',
                'type': 'object',
            },
            name='Foo',
        )
    )

    prompted_agent = Agent(prompted_model, output_type=Foo)
    prompted_agent.run_sync('Hello')
    assert prompted_model.last_model_request_parameters is not None
    assert prompted_model.last_model_request_parameters.output_mode == 'prompted'
    assert prompted_model.last_model_request_parameters.allow_text_output is True
    assert len(prompted_model.last_model_request_parameters.output_tools) == 0
    assert prompted_model.last_model_request_parameters.output_object == snapshot(
        OutputObjectDefinition(
            json_schema={
                'properties': {'bar': {'type': 'string'}},
                'required': ['bar'],
                'title': 'Foo',
                'type': 'object',
            },
            name='Foo',
        )
    )


def test_prompted_output():
    def return_city_location(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = CityLocation(city='Mexico City', country='Mexico').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_city_location)

    class CityLocation(BaseModel):
        """Description from docstring."""

        city: str
        country: str

    agent = Agent(
        m,
        output_type=PromptedOutput(CityLocation, name='City & Country', description='Description from PromptedOutput'),
    )

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city":"Mexico City","country":"Mexico"}')],
                usage=RequestUsage(input_tokens=56, output_tokens=7),
                model_name='function:return_city_location:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_prompted_output_with_template():
    def return_foo(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_foo)

    class Foo(BaseModel):
        bar: str

    agent = Agent(m, output_type=PromptedOutput(Foo, template='Gimme some JSON:'))

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(Foo(bar='baz'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"bar":"baz"}')],
                usage=RequestUsage(input_tokens=56, output_tokens=4),
                model_name='function:return_foo:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_prompted_output_with_template_and_instructions():
    def return_foo(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.instructions is not None
        assert 'Be helpful' in info.instructions
        assert 'Gimme some JSON:' in info.instructions
        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_foo)

    class Foo(BaseModel):
        bar: str

    agent = Agent(m, instructions='Be helpful', output_type=PromptedOutput(Foo, template='Gimme some JSON:'))

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(Foo(bar='baz'))


def test_prompted_output_with_template_false():
    def return_foo(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.prompted_output_template is False
        assert info.model_request_parameters.prompted_output_instructions is None
        assert info.instructions is None
        text = Foo(bar='baz').model_dump_json()
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_foo)

    class Foo(BaseModel):
        bar: str

    agent = Agent(m, output_type=PromptedOutput(Foo, template=False))

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(Foo(bar='baz'))


def test_prompted_output_with_defs():
    class Foo(BaseModel):
        """Foo description"""

        foo: str

    class Bar(BaseModel):
        """Bar description"""

        bar: str

    class Baz(BaseModel):
        """Baz description"""

        baz: str

    class FooBar(BaseModel):
        """FooBar description"""

        foo: Foo
        bar: Bar

    class FooBaz(BaseModel):
        """FooBaz description"""

        foo: Foo
        baz: Baz

    def return_foo_bar(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        text = '{"result": {"kind": "FooBar", "data": {"foo": {"foo": "foo"}, "bar": {"bar": "bar"}}}}'
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_foo_bar)

    agent = Agent(
        m,
        output_type=PromptedOutput(
            [FooBar, FooBaz], name='FooBar or FooBaz', description='FooBar or FooBaz description'
        ),
    )

    result = agent.run_sync('What is foo?')
    assert result.output == snapshot(FooBar(foo=Foo(foo='foo'), bar=Bar(bar='bar')))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is foo?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='{"result": {"kind": "FooBar", "data": {"foo": {"foo": "foo"}, "bar": {"bar": "bar"}}}}'
                    )
                ],
                usage=RequestUsage(input_tokens=53, output_tokens=17),
                model_name='function:return_foo_bar:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_native_output():
    def return_city_location(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            text = '{"city": "Mexico City"}'
        else:
            text = '{"city": "Mexico City", "country": "Mexico"}'
        return ModelResponse(parts=[TextPart(content=text)])

    m = FunctionModel(return_city_location)

    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(
        m,
        output_type=NativeOutput(CityLocation),
    )

    result = agent.run_sync('What is the capital of Mexico?')
    assert result.output == snapshot(CityLocation(city='Mexico City', country='Mexico'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of Mexico?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City"}')],
                usage=RequestUsage(input_tokens=56, output_tokens=5),
                model_name='function:return_city_location:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            ErrorDetails(
                                type='missing',
                                loc=('country',),
                                msg='Field required',
                                input={'city': 'Mexico City'},
                            ),
                        ],
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City", "country": "Mexico"}')],
                usage=RequestUsage(input_tokens=81, output_tokens=12),
                model_name='function:return_city_location:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_native_output_strict_mode():
    class CityLocation(BaseModel):
        city: str
        country: str

    agent = Agent(output_type=NativeOutput(CityLocation, strict=True))
    output_schema = agent._output_schema  # pyright: ignore[reportPrivateUsage]
    assert isinstance(output_schema, NativeOutputSchema)
    assert output_schema.object_def is not None
    assert output_schema.object_def.strict


def test_prompted_output_function_with_retry():
    class Weather(BaseModel):
        temperature: float
        description: str

    def get_weather(city: str) -> Weather:
        if city != 'Mexico City':
            raise ModelRetry('City not found, I only know Mexico City')
        return Weather(temperature=28.7, description='sunny')

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None

        if len(messages) == 1:
            args_json = '{"city": "New York City"}'
        else:
            args_json = '{"city": "Mexico City"}'

        return ModelResponse(parts=[TextPart(content=args_json)])

    agent = Agent(FunctionModel(call_tool), output_type=PromptedOutput(get_weather))
    result = agent.run_sync('New York City')
    assert result.output == snapshot(Weather(temperature=28.7, description='sunny'))
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='New York City',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "New York City"}')],
                usage=RequestUsage(input_tokens=53, output_tokens=6),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='City not found, I only know Mexico City',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"city": "Mexico City"}')],
                usage=RequestUsage(input_tokens=70, output_tokens=11),
                model_name='function:call_tool:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_run_with_history_new():
    m = TestModel()

    agent = Agent(m, system_prompt='Foobar')

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result1 = agent.run_sync('Hello')
    assert result1.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=53, output_tokens=9),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # if we pass new_messages, system prompt is inserted before the message_history messages
    result2 = agent.run_sync('Hello again', message_history=result1.new_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=53, output_tokens=9),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=55, output_tokens=13),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result2.new_messages() == result2.all_messages()[-2:]
    assert result2.output == snapshot('{"ret_a":"a-apple"}')
    assert result2._output_tool_name == snapshot(None)  # pyright: ignore[reportPrivateUsage]
    assert result2.usage() == snapshot(RunUsage(requests=1, input_tokens=55, output_tokens=13))
    new_msg_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in result2.all_messages()]
    assert new_msg_part_kinds == snapshot(
        [
            ('request', ['system-prompt', 'user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('response', ['text']),
            ('request', ['user-prompt']),
            ('response', ['text']),
        ]
    )
    assert result2.new_messages_json().startswith(b'[{"parts":[{"content":"Hello again",')

    # if we pass all_messages, system prompt is NOT inserted before the message_history messages,
    # so only one system prompt
    result3 = agent.run_sync('Hello again', message_history=result1.all_messages())
    # same as result2 except for datetimes
    assert result3.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=53, output_tokens=9),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=55, output_tokens=13),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result3.new_messages() == result3.all_messages()[-2:]
    assert result3.output == snapshot('{"ret_a":"a-apple"}')
    assert result3._output_tool_name == snapshot(None)  # pyright: ignore[reportPrivateUsage]
    assert result3.usage() == snapshot(RunUsage(requests=1, input_tokens=55, output_tokens=13))
    assert result3.timestamp() == IsNow(tz=timezone.utc)


def test_run_with_history_new_structured():
    m = TestModel()

    class Response(BaseModel):
        a: int

    agent = Agent(m, system_prompt='Foobar', output_type=Response)

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result1 = agent.run_sync('Hello')
    assert result1.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'a': 0},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=53, output_tokens=9),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    result2 = agent.run_sync('Hello again', message_history=result1.new_messages())
    assert result2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'a': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=53, output_tokens=9),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            # second call, notice no repeated system prompt
            ModelRequest(
                parts=[
                    UserPromptPart(content='Hello again', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'a': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=59, output_tokens=13),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result2.output == snapshot(Response(a=0))
    assert result2.new_messages() == result2.all_messages()[-3:]
    assert result2._output_tool_name == snapshot('final_result')  # pyright: ignore[reportPrivateUsage]
    assert result2.usage() == snapshot(RunUsage(requests=1, input_tokens=59, output_tokens=13))
    new_msg_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in result2.all_messages()]
    assert new_msg_part_kinds == snapshot(
        [
            ('request', ['system-prompt', 'user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('request', ['user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
        ]
    )
    assert result2.new_messages_json().startswith(b'[{"parts":[{"content":"Hello again",')


def test_run_with_history_ending_on_model_request_and_no_user_prompt():
    m = TestModel()
    agent = Agent(m)

    @agent.system_prompt(dynamic=True)
    async def system_prompt(ctx: RunContext) -> str:
        return f'System prompt: user prompt length = {len(ctx.prompt or [])}'

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='System prompt', dynamic_ref=system_prompt.__qualname__),
                UserPromptPart(content=['Hello', ImageUrl('https://example.com/image.jpg')]),
                UserPromptPart(content='How goes it?'),
            ],
            instructions='Original instructions',
        ),
    ]

    @agent.instructions
    async def instructions(ctx: RunContext) -> str:
        assert ctx.prompt == ['Hello', ImageUrl('https://example.com/image.jpg'), 'How goes it?']
        return 'New instructions'

    result = agent.run_sync(message_history=messages)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='System prompt: user prompt length = 3',
                        timestamp=IsDatetime(),
                        dynamic_ref=IsStr(),
                    ),
                    UserPromptPart(
                        content=['Hello', ImageUrl(url='https://example.com/image.jpg')],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='How goes it?',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='New instructions',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=61, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[-1:]


def test_run_with_history_ending_on_model_response_with_tool_calls_and_no_user_prompt():
    """Test that an agent run with message_history ending on ModelResponse starts with CallToolsNode."""

    def simple_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='Final response')])

    agent = Agent(FunctionModel(simple_response))

    @agent.tool_plain
    def test_tool() -> str:
        return 'Test response'

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='test_tool', args='{}', tool_call_id='call_123')]),
    ]

    result = agent.run_sync(message_history=message_history)

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='test_tool', args='{}', tool_call_id='call_123')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='test_tool',
                        content='Test response',
                        tool_call_id='call_123',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Final response')],
                usage=RequestUsage(input_tokens=53, output_tokens=4),
                model_name='function:simple_response:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[-2:]


def test_run_with_history_ending_on_model_response_with_tool_calls_and_user_prompt():
    """Test that an agent run raises error when message_history ends on ModelResponse with tool calls and there's a new prompt."""

    def simple_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='Final response')])  # pragma: no cover

    agent = Agent(FunctionModel(simple_response))

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='test_tool', args='{}', tool_call_id='call_123')]),
    ]

    with pytest.raises(
        UserError,
        match='Cannot provide a new user prompt when the message history contains unprocessed tool calls.',
    ):
        agent.run_sync(user_prompt='New question', message_history=message_history)


def test_run_with_history_ending_on_model_response_without_tool_calls_or_user_prompt():
    def simple_response(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='Final response')])  # pragma: no cover

    agent = Agent(FunctionModel(simple_response))

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[TextPart('world')]),
    ]

    result = agent.run_sync(message_history=message_history)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='world')],
                timestamp=IsDatetime(),
            ),
        ]
    )

    assert result.new_messages() == snapshot([])


async def test_message_history_ending_on_model_response_with_instructions():
    model = TestModel(custom_output_text='James likes cars in general, especially the Fiat 126p that his parents had.')
    summarize_agent = Agent(
        model,
        instructions="""
        Summarize this conversation to include all important facts about the user and
        what their interactions were about.
        """,
    )

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Hi, my name is James')]),
        ModelResponse(parts=[TextPart(content='Nice to meet you, James.')]),
        ModelRequest(parts=[UserPromptPart(content='I like cars')]),
        ModelResponse(parts=[TextPart(content='I like them too. Sport cars?')]),
        ModelRequest(parts=[UserPromptPart(content='No, cars in general.')]),
        ModelResponse(parts=[TextPart(content='Awesome. Which one do you like most?')]),
        ModelRequest(parts=[UserPromptPart(content='Fiat 126p')]),
        ModelResponse(parts=[TextPart(content="That's an old one, isn't it?")]),
        ModelRequest(parts=[UserPromptPart(content='Yes, it is. My parents had one.')]),
        ModelResponse(parts=[TextPart(content='Cool. Was it fast?')]),
    ]

    result = await summarize_agent.run(message_history=message_history)

    assert result.output == snapshot('James likes cars in general, especially the Fiat 126p that his parents had.')
    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=timezone.utc),
                instructions="""\
Summarize this conversation to include all important facts about the user and
        what their interactions were about.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='James likes cars in general, especially the Fiat 126p that his parents had.')],
                usage=RequestUsage(input_tokens=73, output_tokens=43),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_empty_response():
    def llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[])
        else:
            return ModelResponse(parts=[TextPart('ok here is text')])

    agent = Agent(FunctionModel(llm))

    result = agent.run_sync('Hello')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=51),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok here is text')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_empty_response_without_recovery():
    def llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    agent = Agent(FunctionModel(llm), output_type=tuple[str, int])

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(1\)'):
            agent.run_sync('Hello')

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=51),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=51),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_agent_message_history_includes_run_id() -> None:
    agent = Agent(TestModel(custom_output_text='testing run_id'))

    result = agent.run_sync('Hello')
    history = result.all_messages()

    run_ids = [message.run_id for message in history]
    assert run_ids == snapshot([IsStr(), IsStr()])
    assert len({*run_ids}) == snapshot(1)


def test_agent_message_history_includes_conversation_id() -> None:
    agent = Agent(TestModel(custom_output_text='testing conv_id'))

    result = agent.run_sync('Hello')

    assert result.conversation_id != result.run_id
    conversation_ids = [message.conversation_id for message in result.all_messages()]
    assert conversation_ids == [result.conversation_id, result.conversation_id]


def test_agent_conversation_id_resolves_from_message_history() -> None:
    agent = Agent(TestModel(custom_output_text='continuation'))

    first = agent.run_sync('first turn')
    second = agent.run_sync('second turn', message_history=first.all_messages())

    assert second.conversation_id == first.conversation_id
    new_conv_ids = [m.conversation_id for m in second.new_messages()]
    assert all(cid == first.conversation_id for cid in new_conv_ids)


def test_agent_conversation_id_explicit_override() -> None:
    agent = Agent(TestModel(custom_output_text='explicit'))

    result = agent.run_sync('hi', conversation_id='conv-app-42')
    assert result.conversation_id == 'conv-app-42'
    assert all(m.conversation_id == 'conv-app-42' for m in result.all_messages())


def test_agent_conversation_id_new_sentinel_forks() -> None:
    agent = Agent(TestModel(custom_output_text='fork'))

    first = agent.run_sync('first')
    forked = agent.run_sync('branch', message_history=first.all_messages(), conversation_id='new')

    assert forked.conversation_id != first.conversation_id
    new_conv_ids = [m.conversation_id for m in forked.new_messages()]
    assert all(cid == forked.conversation_id for cid in new_conv_ids)


def test_agent_conversation_id_available_in_run_context() -> None:
    captured: list[str | None] = []

    def capture_metadata(ctx: RunContext[None]) -> dict[str, Any]:
        captured.append(ctx.conversation_id)
        return {}

    agent = Agent(TestModel(custom_output_text='ctx'), metadata=capture_metadata)
    result = agent.run_sync('hi', conversation_id='conv-from-ctx')

    assert result.conversation_id == 'conv-from-ctx'
    assert captured
    assert all(cid == 'conv-from-ctx' for cid in captured)


async def test_agent_conversation_id_surfaces_on_iter_and_async_stream() -> None:
    """`conversation_id` is reachable from `AgentRun` and `StreamedRunResult`."""
    agent = Agent(TestModel(custom_output_text='surfaced'))

    async with agent.iter('hi', conversation_id='conv-iter') as agent_run:
        assert agent_run.conversation_id == 'conv-iter'
        async for _ in agent_run:
            pass

    async with agent.run_stream('hi', conversation_id='conv-async-stream') as stream:
        assert stream.conversation_id == 'conv-async-stream'
        await stream.get_output()


def test_agent_conversation_id_surfaces_on_sync_stream() -> None:
    """`conversation_id` is reachable from `StreamedRunResultSync`."""
    agent = Agent(TestModel(custom_output_text='surfaced'))
    result = agent.run_stream_sync('hi', conversation_id='conv-sync-stream')
    assert result.conversation_id == 'conv-sync-stream'


async def test_agent_run_result_conversation_id_property() -> None:
    """`AgentRunResult.conversation_id` returns the run's conversation ID."""
    agent = Agent(TestModel(custom_output_text='ok'))
    result = await agent.run('hi', conversation_id='conv-result')
    assert result.conversation_id == 'conv-result'


async def test_agent_run_result_metadata_available() -> None:
    agent = Agent(
        TestModel(custom_output_text='metadata output'),
        metadata=lambda ctx: {'prompt': ctx.prompt},
    )

    result = await agent.run('metadata prompt')
    assert result.output == 'metadata output'
    assert result.metadata == {'prompt': 'metadata prompt'}


async def test_agent_iter_metadata_surfaces_on_result() -> None:
    agent = Agent(TestModel(custom_output_text='iter metadata output'), metadata={'env': 'tests'})

    async with agent.iter('iter metadata prompt') as agent_run:
        async for _ in agent_run:
            pass

    assert agent_run.metadata == {'env': 'tests'}
    assert agent_run.result is not None
    assert agent_run.result.metadata == {'env': 'tests'}


async def test_agent_metadata_persisted_when_run_fails() -> None:
    agent = Agent(
        TestModel(),
        metadata=lambda ctx: {'prompt': ctx.prompt},
    )

    @agent.tool
    def explode(_: RunContext) -> str:
        raise RuntimeError('explode')

    failing_prompt = 'metadata failure prompt'
    captured_run = None

    with pytest.raises(RuntimeError, match='explode'):
        async with agent.iter(failing_prompt) as agent_run:
            captured_run = agent_run
            async for _ in agent_run:
                pass

    assert captured_run is not None
    assert captured_run.metadata == {'prompt': failing_prompt}
    assert captured_run.result is None


async def test_agent_metadata_recomputed_on_successful_run() -> None:
    agent = Agent(
        TestModel(custom_output_text='recomputed metadata'),
        metadata=lambda ctx: {'requests': ctx.usage.requests},
    )

    async with agent.iter('recompute metadata prompt') as agent_run:
        initial_metadata = agent_run.metadata
        async for _ in agent_run:
            pass

    assert initial_metadata == {'requests': 0}
    assert agent_run.metadata == {'requests': 1}
    assert agent_run.result is not None
    assert agent_run.result.metadata == {'requests': 1}


async def test_agent_metadata_override_with_dict() -> None:
    agent = Agent(TestModel(custom_output_text='override dict base'), metadata={'env': 'base'})

    with agent.override(metadata={'env': 'override'}):
        result = await agent.run('override dict prompt')

    assert result.metadata == {'env': 'override'}


async def test_agent_metadata_override_with_callable() -> None:
    agent = Agent(TestModel(custom_output_text='override callable base'), metadata={'env': 'base'})

    with agent.override(metadata=lambda ctx: {'computed': ctx.prompt}):
        result = await agent.run('callable override prompt')

    assert result.metadata == {'computed': 'callable override prompt'}


async def test_agent_run_metadata_kwarg_dict() -> None:
    agent = Agent(TestModel(custom_output_text='kwarg dict output'))

    result = await agent.run('kwarg dict prompt', metadata={'env': 'run'})

    assert result.metadata == {'env': 'run'}


async def test_agent_run_metadata_kwarg_callable() -> None:
    agent = Agent(TestModel(custom_output_text='kwarg callable output'))

    def run_meta(ctx: RunContext[None]) -> dict[str, Any]:
        return {'prompt': ctx.prompt}

    result = await agent.run('kwarg callable prompt', metadata=run_meta)

    assert result.metadata == {'prompt': 'kwarg callable prompt'}


async def test_agent_run_metadata_kwarg_merges_agent_metadata() -> None:
    agent = Agent(TestModel(custom_output_text='kwarg merge output'), metadata={'env': 'base', 'shared': 'agent'})

    result = await agent.run('kwarg merge prompt', metadata={'run': 'value', 'shared': 'run'})

    assert result.metadata == {'env': 'base', 'run': 'value', 'shared': 'run'}


async def test_agent_run_metadata_kwarg_ignored_with_override() -> None:
    agent = Agent(TestModel(custom_output_text='kwarg override output'), metadata={'env': 'base'})

    with agent.override(metadata={'env': 'override', 'override_only': True}):
        result = await agent.run('kwarg override prompt', metadata={'run_only': True})

    assert result.metadata == {'env': 'override', 'override_only': True}


def test_unknown_tool():
    def empty(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('foobar', '{}')])

    agent = Agent(FunctionModel(empty))

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'foobar' exceeded max retries count of 1"):
            agent.run_sync('Hello')
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='foobar',
                        content="Unknown tool name: 'foobar'. No tools available.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=65, output_tokens=4),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_unknown_tool_fix():
    def empty(m: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if len(m) > 1:
            return ModelResponse(parts=[TextPart('success')])
        else:
            return ModelResponse(parts=[ToolCallPart('foobar', '{}')])

    agent = Agent(FunctionModel(empty))

    result = agent.run_sync('Hello')
    assert result.output == 'success'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='foobar',
                        content="Unknown tool name: 'foobar'. No tools available.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success')],
                usage=RequestUsage(input_tokens=65, output_tokens=3),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_unknown_tool_multiple_retries():
    num_retries = 2

    def empty(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('foobar', '{}')])

    agent = Agent(FunctionModel(empty), tool_retries=num_retries, output_retries=num_retries)

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'foobar' exceeded max retries count of 2"):
            agent.run_sync('Hello')
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='foobar',
                        content="Unknown tool name: 'foobar'. No tools available.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=65, output_tokens=4),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        tool_name='foobar',
                        content="Unknown tool name: 'foobar'. No tools available.",
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=79, output_tokens=6),
                model_name='function:empty:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_unknown_tool_per_tool_retries_exceeded():
    """When output_retries > retries, the per-tool retry limit fires before the agent-level one."""

    def empty(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('foobar', '{}')])

    agent = Agent(FunctionModel(empty), tool_retries=1, output_retries=5)

    with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'foobar' exceeded max retries count of 1"):
        agent.run_sync('Hello')


def test_tool_exceeds_token_limit_error():
    def return_incomplete_tool(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        resp = ModelResponse(parts=[ToolCallPart('dummy_tool', args='{"foo": "bar",')])
        resp.finish_reason = 'length'
        return resp

    agent = Agent(FunctionModel(return_incomplete_tool), output_type=str)

    with pytest.raises(
        IncompleteToolCall,
        match=r'Model token limit \(10\) exceeded while generating a tool call, resulting in incomplete arguments\.',
    ):
        agent.run_sync('Hello', model_settings=ModelSettings(max_tokens=10))

    with pytest.raises(
        IncompleteToolCall,
        match=r'Model token limit \(provider default\) exceeded while generating a tool call, resulting in incomplete arguments\.',
    ):
        agent.run_sync('Hello')


def test_tool_exceeds_token_limit_but_complete_args():
    def return_complete_tool_but_hit_limit(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            resp = ModelResponse(parts=[ToolCallPart('dummy_tool', args='{"foo": "bar"}')])
            resp.finish_reason = 'length'
            return resp
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(return_complete_tool_but_hit_limit), output_type=str)

    @agent.tool_plain
    def dummy_tool(foo: str) -> str:
        return 'tool-ok'

    result = agent.run_sync('Hello')
    assert result.output == 'done'


def test_empty_response_with_finish_reason_length():
    def return_empty_response(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        resp = ModelResponse(parts=[])
        resp.finish_reason = 'length'
        return resp

    agent = Agent(FunctionModel(return_empty_response), output_type=str)

    with pytest.raises(
        UnexpectedModelBehavior,
        match=r'Model token limit \(10\) exceeded before any response was generated.',
    ):
        agent.run_sync('Hello', model_settings=ModelSettings(max_tokens=10))

    with pytest.raises(
        UnexpectedModelBehavior,
        match=r'Model token limit \(provider default\) exceeded before any response was generated.',
    ):
        agent.run_sync('Hello')


def test_thinking_only_response_with_finish_reason_length():
    def return_thinking_only(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        resp = ModelResponse(parts=[ThinkingPart(content='thinking...')])
        resp.finish_reason = 'length'
        return resp

    agent = Agent(FunctionModel(return_thinking_only), output_type=str)

    with pytest.raises(
        UnexpectedModelBehavior,
        match=r'Model token limit \(10\) exceeded before any response was generated.',
    ):
        agent.run_sync('Hello', model_settings=ModelSettings(max_tokens=10))

    with pytest.raises(
        UnexpectedModelBehavior,
        match=r'Model token limit \(provider default\) exceeded before any response was generated.',
    ):
        agent.run_sync('Hello')


def test_model_requests_blocked(env: TestEnv):
    try:
        env.set('GEMINI_API_KEY', 'foobar')
        agent = Agent('google-gla:gemini-3-flash-preview', output_type=tuple[str, str], defer_model_check=True)

        with pytest.raises(RuntimeError, match='Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False'):
            agent.run_sync('Hello')
    except ImportError:  # pragma: lax no cover
        pytest.skip('google-genai not installed')


def test_override_model(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent('google-gla:gemini-3-flash-preview', output_type=tuple[int, str], defer_model_check=True)

    with agent.override(model='test'):
        result = agent.run_sync('Hello')
        assert result.output == snapshot((0, 'a'))


def test_set_model(env: TestEnv):
    env.set('GEMINI_API_KEY', 'foobar')
    agent = Agent(output_type=tuple[int, str])

    agent.model = 'test'

    result = agent.run_sync('Hello')
    assert result.output == snapshot((0, 'a'))


def test_override_model_no_model():
    agent = Agent()

    with pytest.raises(UserError, match=r'`model` must either be set.+Even when `override\(model=...\)` is customiz'):
        with agent.override(model='test'):
            agent.run_sync('Hello')


async def test_agent_name():
    my_agent = Agent('test')

    assert my_agent.name is None

    await my_agent.run('Hello', infer_name=False)
    assert my_agent.name is None

    await my_agent.run('Hello')
    assert my_agent.name == 'my_agent'


async def test_agent_name_already_set():
    my_agent = Agent('test', name='fig_tree')

    assert my_agent.name == 'fig_tree'

    await my_agent.run('Hello')
    assert my_agent.name == 'fig_tree'


async def test_agent_name_changes():
    my_agent = Agent('test')

    await my_agent.run('Hello')
    assert my_agent.name == 'my_agent'

    new_agent = my_agent
    del my_agent

    await new_agent.run('Hello')
    assert new_agent.name == 'my_agent'


def test_agent_name_override():
    agent = Agent('test', name='custom_name')

    with agent.override(name='overridden_name'):
        agent.run_sync('Hello')
        assert agent.name == 'overridden_name'


def test_name_from_global(create_module: Callable[[str], Any]):
    module_code = """
from pydantic_ai import Agent

my_agent = Agent('test')

def foo():
    result = my_agent.run_sync('Hello')
    return result.output
"""

    mod = create_module(module_code)

    assert mod.my_agent.name is None
    assert mod.foo() == snapshot('success (no tool calls)')
    assert mod.my_agent.name == 'my_agent'


class OutputType(BaseModel):
    """Result type used by multiple tests."""

    value: str


class OutputTypeWithCount(BaseModel):
    """Result type with an additional int field, used to test schema validation failures."""

    value: str
    count: int


class TestMultipleToolCalls:
    """Tests for scenarios where multiple tool calls are made in a single response."""

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_streaming.py::TestMultipleToolCallsStreaming` as well

    def test_early_strategy_stops_after_first_final_result(self):
        """Test that 'early' strategy stops processing regular tools after first final result."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('deferred_tool', {'x': 3}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """Another tool that should not be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        result = agent.run_sync('test early strategy')
        messages = result.all_messages()

        # Verify no tools were called after final result
        assert tool_called == []

        # Verify we got tool returns for all calls
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test early strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args={'value': 'final'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='regular_tool', args={'x': 1}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='deferred_tool', args={'x': 3}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=53, output_tokens=17),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_early_strategy_does_not_call_additional_output_tools(self):
        """Test that 'early' strategy does not execute additional output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:  # pragma: no cover
            """Process second output."""
            output_tools_called.append('second')
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'first'}),
                    ToolCallPart('second_output', {'value': 'second'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='early',
        )

        result = agent.run_sync('test early output tools')

        # Verify the result came from the first output tool
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'first'

        # Verify only the first output tool was called
        assert output_tools_called == ['first']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test early output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args={'value': 'second'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_early_strategy_uses_first_final_result(self):
        """Test that 'early' strategy uses the first final result and ignores subsequent ones."""

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('final_result', {'value': 'second'}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='early')
        result = agent.run_sync('test multiple final results')
        messages = result.all_messages()

        # Verify the result came from the first final tool
        assert result.output.value == 'first'

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'second'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_early_strategy_with_final_result_in_middle(self):
        """Test that 'early' strategy stops at first final result, regardless of position."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                    ToolCallPart('deferred_tool', {'x': 5}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """A tool that should not be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        result = agent.run_sync('test early strategy with final result in middle')

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with final result in middle', timestamp=IsNow(tz=timezone.utc)
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args={'x': 1}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'final'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}, tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='deferred_tool',
                            args={'x': 5},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=58, output_tokens=22),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool', 'deferred_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_early_strategy_with_external_tool_call(self):
        """Test that early strategy handles external tool calls correctly.

        Streaming and non-streaming modes differ in how they choose the final result:
        - Streaming: First tool call (in response order) that can produce a final result (output or deferred)
        - Non-streaming: First output tool (if none called, all deferred tools become final result)

        See https://github.com/pydantic/pydantic-ai/issues/3636#issuecomment-3618800480 for details.
        """
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('external_tool'),
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('regular_tool', {'x': 1}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[OutputType, DeferredToolRequests],
            toolsets=[
                ExternalToolset(
                    tool_defs=[
                        ToolDefinition(
                            name='external_tool',
                            kind='external',
                        )
                    ]
                )
            ],
            end_strategy='early',
        )

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        result = agent.run_sync('test early strategy with external tool call')
        assert result.output == snapshot(OutputType(value='final'))
        messages = result.all_messages()

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with external tool call', timestamp=IsNow(tz=timezone.utc)
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='external_tool', tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='final_result',
                            args={'value': 'final'},
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='regular_tool',
                            args={'x': 1},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=57, output_tokens=11),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='external_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_early_strategy_with_deferred_tool_call(self):
        """Test that early strategy handles deferred tool calls correctly."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('deferred_tool'),
                    ToolCallPart('regular_tool', {'x': 1}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[str, DeferredToolRequests],
            end_strategy='early',
        )

        @agent.tool_plain
        def deferred_tool() -> int:
            raise CallDeferred

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            tool_called.append('regular_tool')
            return x

        result = agent.run_sync('test early strategy with deferred tool call')
        assert result.output == snapshot(
            DeferredToolRequests(calls=[ToolCallPart(tool_name='deferred_tool', tool_call_id=IsStr())])
        )
        messages = result.all_messages()

        # Verify regular tool was called
        assert tool_called == ['regular_tool']

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with deferred tool call', timestamp=IsNow(tz=timezone.utc)
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='deferred_tool', tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='regular_tool',
                            args={'x': 1},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=57, output_tokens=6),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=1,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool(self):
        """Test that 'early' strategy does not apply to tool calls when no output tool is called."""
        tool_called: list[str] = []
        agent = Agent(TestModel(), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        result = agent.run_sync('test early strategy with regular tool calls')

        # Verify the regular tool was executed
        assert tool_called == ['regular_tool']

        # Verify we got appropriate tool returns
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with regular tool calls',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='regular_tool',
                            args={'x': 0},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=57, output_tokens=4),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=0,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={'value': 'a'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=58, output_tokens=9),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_graceful_strategy_executes_function_tools_but_skips_output_tools(self):
        """Test that 'graceful' strategy executes function tools but skips remaining output tools."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('regular_tool', {'x': 42}),
                    ToolCallPart('another_tool', {'y': 2}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='graceful')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        result = agent.run_sync('test graceful strategy')

        # Verify the result came from the output tool
        assert result.output.value == 'first'

        # Verify all function tools were called
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test graceful strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='regular_tool', args={'x': 42}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=53, output_tokens=13),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=42,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_graceful_strategy_does_not_call_additional_output_tools(self):
        """Test that 'graceful' strategy does not execute additional output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:  # pragma: no cover
            """Process second output."""
            output_tools_called.append('second')
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'first'}),
                    ToolCallPart('second_output', {'value': 'second'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='graceful',
        )

        result = agent.run_sync('test graceful output tools')

        # Verify the result came from the first output tool
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'first'

        # Verify only the first output tool was called
        assert output_tools_called == ['first']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test graceful output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args={'value': 'second'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_graceful_strategy_uses_first_final_result(self):
        """Test that 'graceful' strategy uses the first final result and ignores subsequent ones."""

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('final_result', {'value': 'second'}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='graceful')
        result = agent.run_sync('test multiple final results')
        messages = result.all_messages()

        # Verify the result came from the first final tool
        assert result.output.value == 'first'

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'second'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_graceful_strategy_with_final_result_in_middle(self):
        """Test that 'graceful' strategy executes function tools but skips output and deferred tools."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 1}),
                    ToolCallPart('final_result', {'value': 'final'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                    ToolCallPart('deferred_tool', {'x': 5}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='graceful')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            tool_called.append('deferred_tool')
            return x + 1

        result = agent.run_sync('test graceful strategy with final result in middle')

        # Verify function tools were called but deferred tools were not
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got the correct final result
        assert result.output.value == 'final'

        # Verify we got appropriate tool returns
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test graceful strategy with final result in middle',
                            timestamp=IsNow(tz=timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args={'x': 1}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'final'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}, tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='deferred_tool',
                            args={'x': 5},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=58, output_tokens=22),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=1,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content=2,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool', 'deferred_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_strategy_executes_all_tools(self):
        """Test that 'exhaustive' strategy executes all tools while using first final result."""
        tool_called: list[str] = []

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('regular_tool', {'x': 42}),
                    ToolCallPart('final_result', {'value': 'first'}),
                    ToolCallPart('another_tool', {'y': 2}),
                    ToolCallPart('final_result', {'value': 'second'}),
                    ToolCallPart('unknown_tool', {'value': '???'}),
                    ToolCallPart('deferred_tool', {'x': 4}),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='exhaustive')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        result = agent.run_sync('test exhaustive strategy')

        # Verify the result came from the first final tool
        assert result.output.value == 'first'

        # Verify all regular tools were called
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args={'x': 42}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args={'y': 2}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args={'value': 'second'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args={'value': '???'}, tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='deferred_tool',
                            args={'x': 4},
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=53, output_tokens=27),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=42,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool', 'deferred_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_strategy_calls_all_output_tools(self):
        """Test that 'exhaustive' strategy executes all output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output."""
            output_tools_called.append('second')
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'first'}),
                    ToolCallPart('second_output', {'value': 'second'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        result = agent.run_sync('test exhaustive output tools')

        # Verify the result came from the first output tool
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'first'

        # Verify both output tools were called
        assert output_tools_called == ['first', 'second']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args={'value': 'first'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args={'value': 'second'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_strategy_invalid_first_valid_second_output(self):
        """Test that exhaustive strategy uses the second valid output when the first is invalid."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be invalid."""
            output_tools_called.append('first')
            raise ModelRetry('First output validation failed')

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will be valid."""
            output_tools_called.append('second')
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'invalid'}),
                    ToolCallPart('second_output', {'value': 'valid'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        result = agent.run_sync('test invalid first valid second')

        # Verify the result came from the second output tool (first was invalid)
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'

        # Verify both output tools were called
        assert output_tools_called == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test invalid first valid second', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args={'value': 'invalid'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args={'value': 'valid'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=55, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='First output validation failed',
                            tool_name='first_output',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_strategy_valid_first_invalid_second_output(self):
        """Test that exhaustive strategy uses the first valid output even when the second is invalid."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be valid."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will be invalid."""
            output_tools_called.append('second')
            raise ModelRetry('Second output validation failed')

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', {'value': 'invalid'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            output_retries=0,  # No retries - model must succeed first try
        )

        result = agent.run_sync('test valid first invalid second')

        # Verify the result came from the first output tool (second was invalid, but we ignore it)
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'

        # Verify both output tools were called
        assert output_tools_called == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test valid first invalid second', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args={'value': 'valid'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args={'value': 'invalid'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=55, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - output function execution failed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_strategy_with_tool_retry_and_final_result(self):
        """Test that exhaustive strategy doesn't increment retries when `final_result` exists and `ToolRetryError` occurs."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be valid."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will raise ModelRetry."""
            output_tools_called.append('second')
            raise ModelRetry('Second output validation failed')

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', {'value': 'invalid'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            output_retries=1,  # Allow 1 retry so `ToolRetryError` is raised
        )

        result = agent.run_sync('test exhaustive with tool retry')

        # Verify the result came from the first output tool
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'

        # Verify both output tools were called
        assert output_tools_called == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive with tool retry', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args={'value': 'valid'}, tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args={'value': 'invalid'}, tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=55, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        RetryPromptPart(
                            content='Second output validation failed',
                            tool_name='second_output',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_raises_unexpected_model_behavior(self):
        """Test that exhaustive strategy raises `UnexpectedModelBehavior` when all outputs have validation errors."""

        def process_output(output: OutputType) -> OutputType:  # pragma: no cover
            """A tool that should not be called."""
            assert False

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    # Missing 'value' field will cause validation error
                    ToolCallPart('output_tool', {'invalid_field': 'invalid'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_output, name='output_tool'),
            ],
            end_strategy='exhaustive',
        )

        with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(1\)'):
            agent.run_sync('test')

    def test_exhaustive_skips_output_tool_exceeding_retries_on_validation(self):
        """Exhaustive strategy skips an output tool that exceeds its per-tool retry limit during validation,
        when another output tool already produced a valid result."""

        def process_first(output: OutputType) -> OutputType:
            return output

        def process_second(output: OutputType) -> OutputType:  # pragma: no cover
            assert False

        call_count = 0

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    # First tool gets valid args, second gets invalid args
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', {'invalid': 'bad'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output', max_retries=0),
            ],
            end_strategy='exhaustive',
        )

        # First tool succeeds, second fails validation and exceeds max_retries=0
        # The run should succeed with the first tool's result
        result = agent.run_sync('test')
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'

    def test_exhaustive_skips_output_tool_exceeding_retries_on_execution(self):
        """Exhaustive strategy skips an output tool that exceeds its per-tool retry limit during execution,
        when another output tool already produced a valid result."""

        def process_first(output: OutputType) -> OutputType:
            return output

        def process_second(ctx: RunContext[None], value: str) -> str:
            raise ModelRetry('always fails')

        call_count = 0

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', '{"value": "hello"}'),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output', max_retries=0),
            ],
            end_strategy='exhaustive',
        )

        # First tool succeeds, second passes validation but fails execution and exceeds max_retries=0
        # The run should succeed with the first tool's result
        result = agent.run_sync('test')
        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'

    def test_multiple_final_result_are_validated_correctly(self):
        """Tests that if multiple final results are returned, but one fails validation, the other is used."""

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('final_result', {'bad_value': 'first'}, tool_call_id='first'),
                    ToolCallPart('final_result', {'value': 'second'}, tool_call_id='second'),
                ],
            )

        agent = Agent(FunctionModel(return_model), output_type=OutputType, end_strategy='early')
        result = agent.run_sync('test multiple final results')

        # Verify the result came from the second final tool
        assert result.output.value == 'second'

        # Verify we got appropriate tool returns
        assert result.new_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args={'bad_value': 'first'}, tool_call_id='first'),
                        ToolCallPart(tool_name='final_result', args={'value': 'second'}, tool_call_id='second'),
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=10),
                    model_name='function:return_model:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                ErrorDetails(
                                    type='missing',
                                    loc=('value',),
                                    msg='Field required',
                                    input={'bad_value': 'first'},
                                ),
                            ],
                            tool_name='final_result',
                            tool_call_id='first',
                            timestamp=IsDatetime(),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id='second',
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_exhaustive_strategy_second_output_schema_validation_fails(self):
        """Test exhaustive strategy when first output succeeds and second fails schema validation."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            output_tools_called.append('first')
            return output

        def process_second(output: OutputTypeWithCount) -> OutputTypeWithCount:  # pragma: no cover
            output_tools_called.append('second')
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', {'value': 'invalid', 'count': 'not_an_int'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        result = agent.run_sync('test exhaustive with schema validation failure')

        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'
        assert output_tools_called == ['first']

    def test_exhaustive_strategy_second_output_max_retries_exceeded(self):
        """Test exhaustive strategy when first output succeeds and second exceeds max retries."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            output_tools_called.append('first')
            return output

        def process_second(output: OutputTypeWithCount) -> OutputTypeWithCount:  # pragma: no cover
            output_tools_called.append('second')
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', {'value': 'invalid', 'count': 'not_an_int'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            output_retries=0,
        )

        result = agent.run_sync('test exhaustive with max retries exceeded')

        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'
        assert output_tools_called == ['first']

    def test_early_strategy_second_output_max_retries_exceeded(self):
        """Test early strategy when first output succeeds and second exceeds max retries."""

        def process_first(output: OutputType) -> OutputType:
            return output

        def process_second(output: OutputTypeWithCount) -> OutputTypeWithCount:  # pragma: no cover
            return output

        def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(
                parts=[
                    ToolCallPart('first_output', {'value': 'valid'}),
                    ToolCallPart('second_output', {'value': 'invalid', 'count': 'not_an_int'}),
                ],
            )

        agent = Agent(
            FunctionModel(return_model),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='early',
            output_retries=0,
        )

        result = agent.run_sync('test early with max retries exceeded')

        assert isinstance(result.output, OutputType)
        assert result.output.value == 'valid'

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_streaming.py::TestMultipleToolCallsStreaming` as well


async def test_model_settings_override() -> None:
    def return_settings(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(to_json(info.model_settings).decode())])

    my_agent = Agent(FunctionModel(return_settings))
    assert (await my_agent.run('Hello')).output == IsJson(None)
    assert (await my_agent.run('Hello', model_settings={'temperature': 0.5})).output == IsJson({'temperature': 0.5})

    my_agent = Agent(FunctionModel(return_settings), model_settings={'temperature': 0.1})
    assert (await my_agent.run('Hello')).output == IsJson({'temperature': 0.1})
    assert (await my_agent.run('Hello', model_settings={'temperature': 0.5})).output == IsJson({'temperature': 0.5})


async def test_empty_text_part():
    def return_empty_text(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(
            parts=[
                TextPart(''),
                ToolCallPart(info.output_tools[0].name, args_json),
            ],
        )

    agent = Agent(FunctionModel(return_empty_text), output_type=tuple[str, str])

    result = await agent.run('Hello')
    assert result.output == ('foo', 'bar')


def test_heterogeneous_responses_non_streaming() -> None:
    """Indicates that tool calls are prioritized over text in heterogeneous responses."""

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        parts: list[ModelResponsePart] = []
        if len(messages) == 1:
            parts = [TextPart(content='foo'), ToolCallPart('get_location', {'loc_name': 'London'})]
        else:
            parts = [TextPart(content='final response')]
        return ModelResponse(parts=parts)

    agent = Agent(FunctionModel(return_model))

    @agent.tool_plain
    async def get_location(loc_name: str) -> str:
        if loc_name == 'London':
            return json.dumps({'lat': 51, 'lng': 0})
        else:
            raise ModelRetry('Wrong location, please try again')  # pragma: no cover

    result = agent.run_sync('Hello')
    assert result.output == 'final response'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='foo'),
                    ToolCallPart(tool_name='get_location', args={'loc_name': 'London'}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=6),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_location',
                        content='{"lat": 51, "lng": 0}',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final response')],
                usage=RequestUsage(input_tokens=56, output_tokens=8),
                model_name='function:return_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_nested_capture_run_messages() -> None:
    agent = Agent('test')

    with capture_run_messages() as messages1:
        assert messages1 == []
        with capture_run_messages() as messages2:
            assert messages2 == []
            assert messages1 is messages2
            result = agent.run_sync('Hello')
            assert result.output == 'success (no tool calls)'

    assert messages1 == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert messages1 == messages2


def test_double_capture_run_messages() -> None:
    agent = Agent('test')

    with capture_run_messages() as messages:
        assert messages == []
        result = agent.run_sync('Hello')
        assert result.output == 'success (no tool calls)'
        result2 = agent.run_sync('Hello 2')
        assert result2.output == 'success (no tool calls)'
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_capture_run_messages_with_user_exception_does_not_contain_internal_errors() -> None:
    """Test that user exceptions within capture_run_messages context have clean stack traces."""
    agent = Agent('test')

    try:
        with capture_run_messages():
            agent.run_sync('Hello')
            raise ZeroDivisionError('division by zero')
    except Exception as e:
        assert e.__context__ is None


def test_dynamic_false_no_reevaluate():
    """When dynamic is false (default), the system prompt is not reevaluated
    i.e: SystemPromptPart(
            content="A",       <--- Remains the same when `message_history` is passed.
    )
    """
    agent = Agent('test', system_prompt='Foobar')

    dynamic_value = 'A'

    @agent.system_prompt
    async def func() -> str:
        return dynamic_value

    res = agent.run_sync('Hello')

    assert res.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(content=dynamic_value, timestamp=IsNow(tz=timezone.utc)),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='request',
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=53, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='response',
                conversation_id=IsStr(),
            ),
        ]
    )

    dynamic_value = 'B'

    res_two = agent.run_sync('World', message_history=res.all_messages())

    assert res_two.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content='A',  # Remains the same
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='request',
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=53, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='response',
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='World', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt')],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='request',
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=54, output_tokens=8),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='response',
                conversation_id=IsStr(),
            ),
        ]
    )

    assert res_two.new_messages() == res_two.all_messages()[-2:]


def test_dynamic_true_reevaluate_system_prompt():
    """When dynamic is true, the system prompt is reevaluated
    i.e: SystemPromptPart(
            content="B",       <--- Updated value
    )
    """
    agent = Agent('test', system_prompt='Foobar')

    dynamic_value = 'A'

    @agent.system_prompt(dynamic=True)
    async def func():
        return dynamic_value

    res = agent.run_sync('Hello')

    assert res.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content=dynamic_value,
                        dynamic_ref=func.__qualname__,
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='request',
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=53, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='response',
                conversation_id=IsStr(),
            ),
        ]
    )

    dynamic_value = 'B'

    res_two = agent.run_sync('World', message_history=res.all_messages())

    assert res_two.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Foobar', timestamp=IsNow(tz=timezone.utc)),
                    SystemPromptPart(
                        content='B',
                        dynamic_ref=func.__qualname__,
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='request',
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=53, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='response',
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='World', timestamp=IsNow(tz=timezone.utc), part_kind='user-prompt')],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='request',
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=54, output_tokens=8),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                kind='response',
                conversation_id=IsStr(),
            ),
        ]
    )

    assert res_two.new_messages() == res_two.all_messages()[-2:]


def test_dynamic_system_prompt_no_changes():
    """Test coverage for _reevaluate_dynamic_prompts branch where no parts are changed
    and the messages loop continues after replacement of parts.
    """
    agent = Agent('test')

    @agent.system_prompt(dynamic=True)
    async def dynamic_func() -> str:
        return 'Dynamic'

    result1 = agent.run_sync('Hello')

    # Create ModelRequest with non-dynamic SystemPromptPart (no dynamic_ref)
    manual_request = ModelRequest(parts=[SystemPromptPart(content='Static'), UserPromptPart(content='Manual')])

    # Mix dynamic and non-dynamic messages to trigger branch coverage
    result2 = agent.run_sync('Second call', message_history=result1.all_messages() + [manual_request])

    assert result2.output == 'success (no tool calls)'


def test_dynamic_system_prompt_none_return():
    """Test dynamic system prompts with None return values."""
    agent = Agent('test')

    dynamic_values = [None, 'DYNAMIC']

    @agent.system_prompt(dynamic=True)
    def dynamic_sys() -> str | None:
        return dynamic_values.pop(0)

    with capture_run_messages() as base_messages:
        agent.run_sync('Hi', model=TestModel(custom_output_text='baseline'))

    base_req = base_messages[0]
    assert isinstance(base_req, ModelRequest)
    sys_texts = [p.content for p in base_req.parts if isinstance(p, SystemPromptPart)]
    # The None value should have a '' placeholder due to keeping a reference to the dynamic prompt
    assert '' in sys_texts
    assert 'DYNAMIC' not in sys_texts

    # Run a second time to capture the updated system prompt
    with capture_run_messages() as messages:
        agent.run_sync('Hi', model=TestModel(custom_output_text='baseline'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    sys_texts = [p.content for p in req.parts if isinstance(p, SystemPromptPart)]
    # The None value should have a '' placeholder due to keep a reference to the dynamic prompt
    assert '' not in sys_texts
    assert 'DYNAMIC' in sys_texts


def test_system_prompt_none_return_are_omitted():
    """Test dynamic system prompts with None return values."""
    agent = Agent('test', system_prompt='STATIC')

    @agent.system_prompt
    def dynamic_sys() -> str | None:
        return None

    with capture_run_messages() as base_messages:
        agent.run_sync('Hi', model=TestModel(custom_output_text='baseline'))

    base_req = base_messages[0]
    assert isinstance(base_req, ModelRequest)
    sys_texts = [p.content for p in base_req.parts if isinstance(p, SystemPromptPart)]
    # The None value should be omitted
    assert 'STATIC' in sys_texts
    assert '' not in sys_texts


def test_capture_run_messages_tool_agent() -> None:
    agent_outer = Agent('test')
    agent_inner = Agent(TestModel(custom_output_text='inner agent result'))

    @agent_outer.tool_plain
    async def foobar(x: str) -> str:
        result_ = await agent_inner.run(x)
        return result_.output

    with capture_run_messages() as messages:
        result = agent_outer.run_sync('foobar')

    assert result.output == snapshot('{"foobar":"inner agent result"}')
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='foobar', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foobar',
                        content='inner agent result',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"foobar":"inner agent result"}')],
                usage=RequestUsage(input_tokens=54, output_tokens=11),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


class Bar(BaseModel):
    c: int
    d: str


def test_custom_output_type_sync() -> None:
    agent = Agent('test', output_type=Foo)

    assert agent.run_sync('Hello').output == snapshot(Foo(a=0, b='a'))
    assert agent.run_sync('Hello', output_type=Bar).output == snapshot(Bar(c=0, d='a'))
    assert agent.run_sync('Hello', output_type=str).output == snapshot('success (no tool calls)')
    assert agent.run_sync('Hello', output_type=int).output == snapshot(0)


async def test_custom_output_type_async() -> None:
    agent = Agent('test')

    result = await agent.run('Hello')
    assert result.output == snapshot('success (no tool calls)')

    result = await agent.run('Hello', output_type=Foo)
    assert result.output == snapshot(Foo(a=0, b='a'))
    result = await agent.run('Hello', output_type=int)
    assert result.output == snapshot(0)


def test_custom_output_type_invalid() -> None:
    agent = Agent('test')

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Any) -> Any:  # pragma: no cover
        return o

    with pytest.raises(UserError, match='Cannot set a custom run `output_type` when the agent has output validators'):
        agent.run_sync('Hello', output_type=int)


def test_binary_content_serializable():
    agent = Agent('test')

    content = BinaryContent(data=b'Hello', media_type='text/plain')
    result = agent.run_sync(['Hello', content])

    serialized = result.all_messages_json()
    assert json.loads(serialized) == snapshot(
        [
            {
                'parts': [
                    {
                        'content': [
                            'Hello',
                            {
                                'data': 'SGVsbG8=',
                                'media_type': 'text/plain',
                                'vendor_metadata': None,
                                'kind': 'binary',
                                'identifier': 'f7ff9e',
                            },
                        ],
                        'timestamp': IsStr(),
                        'part_kind': 'user-prompt',
                    }
                ],
                'timestamp': IsStr(),
                'instructions': None,
                'kind': 'request',
                'run_id': IsStr(),
                'conversation_id': IsStr(),
                'metadata': None,
            },
            {
                'parts': [
                    {
                        'content': 'success (no tool calls)',
                        'id': None,
                        'provider_name': None,
                        'part_kind': 'text',
                        'provider_details': None,
                    }
                ],
                'usage': {
                    'input_tokens': 56,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 4,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                },
                'model_name': 'test',
                'provider_name': None,
                'provider_details': None,
                'provider_response_id': None,
                'provider_url': None,
                'timestamp': IsStr(),
                'kind': 'response',
                'finish_reason': None,
                'run_id': IsStr(),
                'conversation_id': IsStr(),
                'metadata': None,
            },
        ]
    )

    # We also need to be able to round trip the serialized messages.
    messages = ModelMessagesTypeAdapter.validate_json(serialized)
    assert messages == result.all_messages()


def test_image_url_serializable_missing_media_type():
    agent = Agent('test')
    content = ImageUrl('https://example.com/chart.jpeg')
    result = agent.run_sync(['Hello', content])
    serialized = result.all_messages_json()
    assert json.loads(serialized) == snapshot(
        [
            {
                'parts': [
                    {
                        'content': [
                            'Hello',
                            {
                                'url': 'https://example.com/chart.jpeg',
                                'force_download': False,
                                'vendor_metadata': None,
                                'kind': 'image-url',
                                'media_type': 'image/jpeg',
                                'identifier': 'a72e39',
                            },
                        ],
                        'timestamp': IsStr(),
                        'part_kind': 'user-prompt',
                    }
                ],
                'timestamp': IsStr(),
                'instructions': None,
                'kind': 'request',
                'run_id': IsStr(),
                'conversation_id': IsStr(),
                'metadata': None,
            },
            {
                'parts': [
                    {
                        'content': 'success (no tool calls)',
                        'id': None,
                        'provider_name': None,
                        'part_kind': 'text',
                        'provider_details': None,
                    }
                ],
                'usage': {
                    'input_tokens': 51,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 4,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                },
                'model_name': 'test',
                'timestamp': IsStr(),
                'provider_name': None,
                'provider_details': None,
                'provider_url': None,
                'provider_response_id': None,
                'kind': 'response',
                'finish_reason': None,
                'run_id': IsStr(),
                'conversation_id': IsStr(),
                'metadata': None,
            },
        ]
    )

    # We also need to be able to round trip the serialized messages.
    messages = ModelMessagesTypeAdapter.validate_json(serialized)
    part = messages[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = part.content[1]
    assert isinstance(content, ImageUrl)
    assert content.media_type == 'image/jpeg'
    assert messages == result.all_messages()


def test_image_url_serializable():
    agent = Agent('test')

    content = ImageUrl('https://example.com/chart', media_type='image/jpeg')
    result = agent.run_sync(['Hello', content])

    serialized = result.all_messages_json()
    assert json.loads(serialized) == snapshot(
        [
            {
                'parts': [
                    {
                        'content': [
                            'Hello',
                            {
                                'url': 'https://example.com/chart',
                                'force_download': False,
                                'vendor_metadata': None,
                                'kind': 'image-url',
                                'media_type': 'image/jpeg',
                                'identifier': 'bdd86d',
                            },
                        ],
                        'timestamp': IsStr(),
                        'part_kind': 'user-prompt',
                    }
                ],
                'timestamp': IsStr(),
                'instructions': None,
                'kind': 'request',
                'run_id': IsStr(),
                'conversation_id': IsStr(),
                'metadata': None,
            },
            {
                'parts': [
                    {
                        'content': 'success (no tool calls)',
                        'id': None,
                        'provider_name': None,
                        'part_kind': 'text',
                        'provider_details': None,
                    }
                ],
                'usage': {
                    'input_tokens': 51,
                    'cache_write_tokens': 0,
                    'cache_read_tokens': 0,
                    'output_tokens': 4,
                    'input_audio_tokens': 0,
                    'cache_audio_read_tokens': 0,
                    'output_audio_tokens': 0,
                    'details': {},
                },
                'model_name': 'test',
                'timestamp': IsStr(),
                'provider_name': None,
                'provider_details': None,
                'provider_url': None,
                'provider_response_id': None,
                'kind': 'response',
                'finish_reason': None,
                'run_id': IsStr(),
                'conversation_id': IsStr(),
                'metadata': None,
            },
        ]
    )

    # We also need to be able to round trip the serialized messages.
    messages = ModelMessagesTypeAdapter.validate_json(serialized)
    part = messages[0].parts[0]
    assert isinstance(part, UserPromptPart)
    content = part.content[1]
    assert isinstance(content, ImageUrl)
    assert content.media_type == 'image/jpeg'
    assert messages == result.all_messages()


def test_tool_returning_binary_content_directly():
    """Test that a tool returning BinaryContent directly works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_image', {})])
        else:
            return ModelResponse(parts=[TextPart('Image received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_image() -> BinaryContent:
        """Return a simple image."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        return BinaryContent(png_data, media_type='image/png')

    # This should work without the serialization error
    result = agent.run_sync('Get an image')
    assert result.output == 'Image received'


def test_tool_returning_binary_content_with_identifier():
    """Test that a tool returning BinaryContent directly works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_image', {})])
        else:
            return ModelResponse(parts=[TextPart('Image received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_image() -> BinaryContent:
        """Return a simple image."""
        png_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf6\x178\x00\x00\x00\x00IEND\xaeB`\x82'
        return BinaryContent(png_data, media_type='image/png', identifier='image_id_1')

    # This should work without the serialization error
    result = agent.run_sync('Get an image')
    assert result.all_messages()[2] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_image',
                    content=IsInstance(BinaryContent),
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_tool_returning_file_url_with_identifier():
    """Test that a tool returning FileUrl subclasses with identifiers works correctly."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('get_files', {})])
        else:
            return ModelResponse(parts=[TextPart('Files received')])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def get_files():
        """Return various file URLs with custom identifiers."""
        return [
            ImageUrl(url='https://example.com/image.jpg', identifier='img_001'),
            VideoUrl(url='https://example.com/video.mp4', identifier='vid_002'),
            AudioUrl(url='https://example.com/audio.mp3', identifier='aud_003'),
            DocumentUrl(url='https://example.com/document.pdf', identifier='doc_004'),
        ]

    result = agent.run_sync('Get some files')
    assert result.all_messages()[2] == snapshot(
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='get_files',
                    content=[
                        ImageUrl(url='https://example.com/image.jpg', _identifier='img_001'),
                        VideoUrl(url='https://example.com/video.mp4', _identifier='vid_002'),
                        AudioUrl(url='https://example.com/audio.mp3', _identifier='aud_003'),
                        DocumentUrl(url='https://example.com/document.pdf', _identifier='doc_004'),
                    ],
                    tool_call_id=IsStr(),
                    timestamp=IsDatetime(),
                )
            ],
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_raise_error_when_system_prompt_is_set():
    agent = Agent('test', instructions='An instructions!')

    @agent.system_prompt
    def system_prompt() -> str:
        return 'A system prompt!'

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[
                SystemPromptPart(content='A system prompt!', timestamp=IsNow(tz=timezone.utc)),
                UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ],
            timestamp=IsNow(tz=timezone.utc),
            instructions='An instructions!',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_raise_error_when_instructions_is_set():
    agent = Agent('test', system_prompt='A system prompt!')

    @agent.instructions
    def instructions() -> str:
        return 'An instructions!'

    @agent.instructions
    def empty_instructions() -> str:
        return ''

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[
                SystemPromptPart(content='A system prompt!', timestamp=IsNow(tz=timezone.utc)),
                UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ],
            timestamp=IsNow(tz=timezone.utc),
            instructions='An instructions!',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_both_instructions_and_system_prompt_are_set():
    agent = Agent('test', instructions='An instructions!', system_prompt='A system prompt!')
    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[
                SystemPromptPart(content='A system prompt!', timestamp=IsNow(tz=timezone.utc)),
                UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc)),
            ],
            timestamp=IsNow(tz=timezone.utc),
            instructions='An instructions!',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_decorator_without_parenthesis():
    agent = Agent('test')

    @agent.instructions
    def instructions() -> str:
        return 'You are a helpful assistant.'

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
            timestamp=IsNow(tz=timezone.utc),
            instructions='You are a helpful assistant.',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_decorator_with_parenthesis():
    agent = Agent('test')

    @agent.instructions()
    def instructions_2() -> str:
        return 'You are a helpful assistant.'

    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
            timestamp=IsNow(tz=timezone.utc),
            instructions='You are a helpful assistant.',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_with_message_history():
    agent = Agent('test', instructions='You are a helpful assistant.')
    result = agent.run_sync(
        'Hello',
        message_history=[ModelRequest(parts=[SystemPromptPart(content='You are a helpful assistant')])],
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[SystemPromptPart(content='You are a helpful assistant', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=56, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[-2:]


def test_instructions_parameter_with_sequence():
    def instructions() -> str:
        return 'You are a potato.'

    def empty_instructions() -> str:
        return ''

    agent = Agent('test', instructions=('You are a helpful assistant.', empty_instructions, instructions))
    result = agent.run_sync('Hello')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
            timestamp=IsNow(tz=timezone.utc),
            instructions="""\
You are a helpful assistant.

You are a potato.\
""",
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_instructions_during_run():
    agent = Agent('test', instructions='You are a helpful assistant.')
    result = agent.run_sync('Hello', instructions='Your task is to greet people.')
    assert result.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
            timestamp=IsNow(tz=timezone.utc),
            instructions="""\
You are a helpful assistant.
Your task is to greet people.\
""",
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )

    result2 = agent.run_sync('Hello again!')
    assert result2.all_messages()[0] == snapshot(
        ModelRequest(
            parts=[UserPromptPart(content='Hello again!', timestamp=IsDatetime())],
            timestamp=IsNow(tz=timezone.utc),
            instructions='You are a helpful assistant.',
            run_id=IsStr(),
            conversation_id=IsStr(),
        )
    )


def test_multi_agent_instructions_with_structured_output():
    """Test that Agent2 uses its own instructions when called with Agent1's history.

    Reproduces issue #3207: when running agents sequentially with no user_prompt
    and structured output, Agent2's instructions were ignored.
    """

    class Output(BaseModel):
        text: str

    agent1 = Agent('test', instructions='Agent 1 instructions')
    agent2 = Agent('test', instructions='Agent 2 instructions', output_type=Output)

    result1 = agent1.run_sync('Hello')

    result2 = agent2.run_sync(message_history=result1.new_messages())
    messages = result2.new_messages()

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=timezone.utc),
                instructions='Agent 2 instructions',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'text': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=9),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Verify Agent2's retry requests used Agent2's instructions (not Agent1's)
    requests = [m for m in messages if isinstance(m, ModelRequest)]
    assert any(r.instructions == 'Agent 2 instructions' for r in requests)


def test_empty_final_response():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('foo'), ToolCallPart('my_tool', {'x': 1})])
        elif len(messages) == 3:
            return ModelResponse(parts=[TextPart('bar'), ToolCallPart('my_tool', {'x': 2})])
        else:
            return ModelResponse(parts=[])

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def my_tool(x: int) -> int:
        return x * 2

    result = agent.run_sync('Hello')
    assert result.output == 'bar'

    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='foo'),
                    ToolCallPart(tool_name='my_tool', args={'x': 1}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='bar'),
                    ToolCallPart(tool_name='my_tool', args={'x': 2}, tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=10),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool', content=4, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=53, output_tokens=10),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_agent_run_result_serialization() -> None:
    agent = Agent('test', output_type=Foo)
    result = agent.run_sync('Hello')

    # Check that dump_json doesn't raise an error
    adapter = TypeAdapter(AgentRunResult[Foo])
    serialized_data = adapter.dump_json(result)

    # Check that we can load the data back
    deserialized_result = adapter.validate_json(serialized_data)
    assert deserialized_result == result


def test_agent_repr() -> None:
    agent = Agent()
    assert repr(agent) == snapshot(
        "Agent(model=None, name=None, end_strategy='early', model_settings=None, output_type=<class 'str'>, instrument=None)"
    )


async def test_agent_context_manager_no_model():
    agent = Agent()
    async with agent:
        pass


def test_cached_async_http_client_deprecated():
    from pydantic_ai.models import cached_async_http_client  # pyright: ignore[reportDeprecated]

    with pytest.warns(DeprecationWarning, match='cached_async_http_client.*is deprecated'):
        cached_async_http_client()  # pyright: ignore[reportDeprecated]


@requires_openai
async def test_provider_lifecycle_closes_client():
    """Provider lifecycle closes owned HTTP client on exit.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider = OpenAIProvider(api_key='test-key')
    async with provider:
        http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert not http_client.is_closed
    assert http_client.is_closed


@requires_openai
async def test_provider_reentrant_lifecycle():
    """Reentrant provider lifecycle keeps client open until outermost exit.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider = OpenAIProvider(api_key='test-key')
    async with provider:
        http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            assert not http_client.is_closed
        assert not http_client.is_closed
    assert http_client.is_closed


@requires_openai
async def test_provider_aexit_without_aenter():
    """Calling __aexit__ without __aenter__ is a no-op (no crash).

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider = OpenAIProvider(api_key='test-key')
    await provider.__aexit__(None, None, None)
    # Clean up the owned http client to avoid ResourceWarning from __del__
    assert provider._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
    await provider._own_http_client.aclose()  # pyright: ignore[reportPrivateUsage]


@requires_openai
async def test_provider_aexit_without_aenter_then_async_with():
    """Bare __aexit__ before any __aenter__ must not corrupt later lifecycle state."""
    provider = OpenAIProvider(api_key='test-key')
    await provider.__aexit__(None, None, None)
    async with provider:
        assert provider._own_http_client is not None  # pyright: ignore[reportPrivateUsage]
        assert not provider._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]
    assert provider._own_http_client.is_closed  # pyright: ignore[reportPrivateUsage]


# TODO(v2): uncomment when we re-enable the ResourceWarning in Provider.__del__
# @requires_openai
# async def test_provider_del_warns_on_unclosed_client():
#     """Provider.__del__ warns if the HTTP client was never closed.
#
#     Regression test for PR #4421 (provider lifecycle management).
#     https://github.com/pydantic/pydantic-ai/pull/4421
#     """
#     provider = OpenAIProvider(api_key='test-key')
#     assert provider._own_http_client is not None
#     assert not provider._own_http_client.is_closed
#     with pytest.warns(ResourceWarning, match='was garbage collected with an open HTTP client'):
#         provider.__del__()
#     # Clean up
#     await provider._own_http_client.aclose()


@requires_openai
async def test_provider_reentry_after_close():
    """Provider can be re-entered after exit by recreating the HTTP client."""
    provider = OpenAIProvider(api_key='test-key')

    async with provider:
        first_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert not first_client.is_closed
    assert first_client.is_closed

    async with provider:
        second_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert not second_client.is_closed
        assert second_client is not first_client
    assert second_client.is_closed


@requires_google_gla
@pytest.mark.filterwarnings('ignore:`GoogleGLAProvider` is deprecated.:DeprecationWarning')
async def test_google_gla_provider_reentry_after_close():
    """GoogleGLAProvider restores base_url and API key header on re-entry."""
    provider = GoogleGLAProvider(api_key='test-key')  # pyright: ignore[reportDeprecated]

    async with provider:
        first_client = provider.client
        assert not first_client.is_closed
        assert str(first_client.base_url) == 'https://generativelanguage.googleapis.com/v1beta/models/'
        assert first_client.headers['X-Goog-Api-Key'] == 'test-key'
    assert first_client.is_closed

    async with provider:
        second_client = provider.client
        assert not second_client.is_closed
        assert second_client is not first_client
        assert str(second_client.base_url) == 'https://generativelanguage.googleapis.com/v1beta/models/'
        assert second_client.headers['X-Goog-Api-Key'] == 'test-key'
    assert second_client.is_closed


@requires_google_vertex
@pytest.mark.filterwarnings('ignore:`GoogleVertexProvider` is deprecated.:DeprecationWarning')
async def test_google_vertex_provider_reentry_after_close():
    """GoogleVertexProvider restores auth and base_url on re-entry."""
    provider = GoogleVertexProvider(service_account_file='/dev/null', project_id='test-project', region='us-central1')  # pyright: ignore[reportDeprecated]

    async with provider:
        first_client = provider.client
        assert not first_client.is_closed
        assert first_client.auth is not None
        assert 'us-central1' in str(first_client.base_url)
    assert first_client.is_closed

    async with provider:
        second_client = provider.client
        assert not second_client.is_closed
        assert second_client is not first_client
        assert second_client.auth is not None
        assert 'us-central1' in str(second_client.base_url)
    assert second_client.is_closed


@requires_openai
async def test_gateway_provider_reentry_after_close():
    """Gateway provider restores event_hooks on re-entry."""
    from pydantic_ai.providers.gateway import gateway_provider

    provider = gateway_provider('openai', api_key='test-key', base_url='https://gateway.example.com/proxy')

    async with provider:
        first_client = provider._own_http_client  # pyright: ignore[reportPrivateUsage]
        assert first_client is not None
        assert not first_client.is_closed
        assert len(first_client.event_hooks.get('request', [])) == 1
    assert first_client.is_closed

    async with provider:
        second_client = provider._own_http_client  # pyright: ignore[reportPrivateUsage]
        assert second_client is not None
        assert not second_client.is_closed
        assert second_client is not first_client
        assert len(second_client.event_hooks.get('request', [])) == 1
    assert second_client.is_closed


@requires_openai
async def test_azure_provider_lifecycle_closes_client():
    """Azure provider lifecycle closes owned HTTP client on exit.

    Regression test for PR #4421 (provider lifecycle management).
    https://github.com/pydantic/pydantic-ai/pull/4421
    """
    provider = AzureProvider(
        azure_endpoint='https://test.openai.azure.com',
        api_key='test-key',
        api_version='2024-02-01',
    )
    async with provider:
        http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert not http_client.is_closed
    assert http_client.is_closed


@pytest.mark.filterwarnings('ignore:`GrokProvider` is deprecated.:DeprecationWarning')
@pytest.mark.parametrize(
    'provider_factory',
    [
        pytest.param(lambda: OpenAIProvider(api_key='t'), marks=[requires_openai], id='openai'),
        pytest.param(lambda: AnthropicProvider(api_key='t'), marks=[requires_anthropic], id='anthropic'),
        pytest.param(lambda: GroqProvider(api_key='t'), marks=[requires_groq], id='groq'),
        pytest.param(lambda: MistralProvider(api_key='t'), marks=[requires_mistral], id='mistral'),
        pytest.param(lambda: CohereProvider(api_key='t'), marks=[requires_cohere], id='cohere'),
        pytest.param(lambda: GoogleProvider(api_key='t'), marks=[requires_google], id='google'),
        pytest.param(
            lambda: AzureProvider(azure_endpoint='https://t.openai.azure.com', api_key='t', api_version='2024-02-01'),
            marks=[requires_openai],
            id='azure',
        ),
        pytest.param(lambda: CerebrasProvider(api_key='t'), marks=[requires_openai], id='cerebras'),
        pytest.param(lambda: DeepSeekProvider(api_key='t'), marks=[requires_openai], id='deepseek'),
        pytest.param(lambda: FireworksProvider(api_key='t'), marks=[requires_openai], id='fireworks'),
        pytest.param(lambda: GitHubProvider(api_key='t'), marks=[requires_openai], id='github'),
        pytest.param(lambda: GrokProvider(api_key='t'), marks=[requires_openai], id='grok'),  # pyright: ignore[reportDeprecated]
        pytest.param(lambda: HerokuProvider(api_key='t'), marks=[requires_openai], id='heroku'),
        pytest.param(lambda: LiteLLMProvider(api_key='t'), marks=[requires_litellm], id='litellm'),
        pytest.param(lambda: MoonshotAIProvider(api_key='t'), marks=[requires_openai], id='moonshotai'),
        pytest.param(lambda: NebiusProvider(api_key='t'), marks=[requires_openai], id='nebius'),
        pytest.param(
            lambda: OllamaProvider(base_url='http://localhost:11434/v1'), marks=[requires_openai], id='ollama'
        ),
        pytest.param(lambda: OpenRouterProvider(api_key='t'), marks=[requires_openai], id='openrouter'),
        pytest.param(lambda: OVHcloudProvider(api_key='t'), marks=[requires_openai], id='ovhcloud'),
        pytest.param(lambda: SambaNovaProvider(api_key='t'), marks=[requires_openai], id='sambanova'),
        pytest.param(lambda: TogetherProvider(api_key='t'), marks=[requires_openai], id='together'),
        pytest.param(lambda: VercelProvider(api_key='t'), marks=[requires_openai], id='vercel'),
        pytest.param(lambda: AlibabaProvider(api_key='t'), marks=[requires_openai], id='alibaba'),
    ],
)
async def test_provider_reentry_recreates_http_client(provider_factory: Callable[[], Provider[Any]]):
    """All providers with _own_http_client properly close on exit and recreate on re-entry."""
    provider = provider_factory()
    assert provider._own_http_client is not None  # pyright: ignore[reportPrivateUsage]

    async with provider:
        first_client = provider._own_http_client  # pyright: ignore[reportPrivateUsage]
        assert first_client is not None
        assert not first_client.is_closed
    assert first_client.is_closed

    async with provider:
        second_client = provider._own_http_client  # pyright: ignore[reportPrivateUsage]
        assert second_client is not None
        assert not second_client.is_closed
        assert second_client is not first_client
    assert second_client.is_closed


def test_tool_call_with_validation_value_error_serializable():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('foo_tool', {'bar': 0})])
        elif len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('foo_tool', {'bar': 1})])
        else:
            return ModelResponse(parts=[TextPart('Tool returned 1')])

    agent = Agent(FunctionModel(llm))

    class Foo(BaseModel):
        bar: int

        @field_validator('bar')
        def validate_bar(cls, v: int) -> int:
            if v == 0:
                raise ValueError('bar cannot be 0')
            return v

    @agent.tool_plain
    def foo_tool(foo: Foo) -> int:
        return foo.bar

    result = agent.run_sync('Hello')
    assert json.loads(result.all_messages_json())[2] == snapshot(
        {
            'parts': [
                {
                    'content': [
                        {'type': 'value_error', 'loc': ['bar'], 'msg': 'Value error, bar cannot be 0', 'input': 0}
                    ],
                    'tool_name': 'foo_tool',
                    'tool_call_id': IsStr(),
                    'timestamp': IsStr(),
                    'part_kind': 'retry-prompt',
                }
            ],
            'timestamp': IsStr(),
            'instructions': None,
            'kind': 'request',
            'run_id': IsStr(),
            'conversation_id': IsStr(),
            'metadata': None,
        }
    )


def test_unsupported_output_mode():
    class Foo(BaseModel):
        bar: str

    def hello(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('hello')])  # pragma: no cover

    model = FunctionModel(hello, profile=ModelProfile(supports_tools=False, supports_json_schema_output=False))

    agent = Agent(model, output_type=NativeOutput(Foo))

    with pytest.raises(UserError, match='Native structured output is not supported by this model.'):
        agent.run_sync('Hello')

    agent = Agent(model, output_type=ToolOutput(Foo))

    with pytest.raises(UserError, match='Tool output is not supported by this model.'):
        agent.run_sync('Hello')

    agent = Agent(model, output_type=BinaryImage)

    with pytest.raises(UserError, match='Image output is not supported by this model.'):
        agent.run_sync('Hello')


def test_multimodal_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value='Data analysis completed successfully',
            content=[
                'Here are the analysis results:',
                ImageUrl('https://example.com/chart.jpg'),
                'The chart shows positive trends.',
            ],
            metadata={'foo': 'bar'},
        )

    result = agent.run_sync('Please analyze the data')

    # Verify final output
    assert result.output == 'Analysis completed'

    # Verify message history contains the expected parts

    # Verify the complete message structure using snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Please analyze the data', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Starting analysis'),
                    ToolCallPart(
                        tool_name='analyze_data',
                        args={},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='Data analysis completed successfully',
                        tool_call_id=IsStr(),
                        metadata={'foo': 'bar'},
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                    UserPromptPart(
                        content=[
                            'Here are the analysis results:',
                            ImageUrl(url='https://example.com/chart.jpg', identifier='672a5c'),
                            'The chart shows positive trends.',
                        ],
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Analysis completed')],
                usage=RequestUsage(input_tokens=70, output_tokens=6),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_plain_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> ToolReturn:
        return ToolReturn(
            return_value='Data analysis completed successfully',
            metadata={'foo': 'bar'},
        )

    result = agent.run_sync('Please analyze the data')

    # Verify final output
    assert result.output == 'Analysis completed'

    # Verify message history contains the expected parts

    # Verify the complete message structure using snapshot
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Please analyze the data', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content='Starting analysis'),
                    ToolCallPart(
                        tool_name='analyze_data',
                        args={},
                        tool_call_id=IsStr(),
                    ),
                ],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='analyze_data',
                        content='Data analysis completed successfully',
                        tool_call_id=IsStr(),
                        metadata={'foo': 'bar'},
                        timestamp=IsNow(tz=timezone.utc),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Analysis completed')],
                usage=RequestUsage(input_tokens=58, output_tokens=6),
                model_name='function:llm:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_many_multimodal_tool_response():
    """Test ToolReturn with custom content and tool return."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[TextPart('Starting analysis'), ToolCallPart('analyze_data', {})])
        else:
            return ModelResponse(  # pragma: no cover
                parts=[
                    TextPart('Analysis completed'),
                ]
            )

    agent = Agent(FunctionModel(llm))

    @agent.tool_plain
    def analyze_data() -> list[Any]:
        return [
            ToolReturn(
                return_value='Data analysis completed successfully',
                content=[
                    'Here are the analysis results:',
                    ImageUrl('https://example.com/chart.jpg'),
                    'The chart shows positive trends.',
                ],
                metadata={'foo': 'bar'},
            ),
            'Something else',
        ]

    with pytest.raises(
        UserError,
        match="The return value of tool 'analyze_data' contains invalid nested `ToolReturn` objects. `ToolReturn` should be used directly.",
    ):
        agent.run_sync('Please analyze the data')


def test_deprecated_kwargs_validation_agent_init():
    """Test that invalid kwargs raise UserError in Agent constructor."""
    with pytest.raises(UserError, match='Unknown keyword arguments: `usage_limits`'):
        Agent('test', usage_limits='invalid')  # type: ignore[call-arg]

    with pytest.raises(UserError, match='Unknown keyword arguments: `invalid_kwarg`'):
        Agent('test', invalid_kwarg='value')  # type: ignore[call-arg]

    with pytest.raises(UserError, match='Unknown keyword arguments: `foo`, `bar`'):
        Agent('test', foo='value1', bar='value2')  # type: ignore[call-arg]


def test_deprecated_kwargs_still_work():
    """Test that valid deprecated kwargs still work with warnings."""
    import warnings

    try:
        from pydantic_ai.mcp import MCPServerStdio

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')

            Agent(  # pyright: ignore[reportDeprecated]
                'test', mcp_servers=[MCPServerStdio('python', ['-m', 'tests.mcp_server'])]
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert '`mcp_servers` is deprecated' in str(w[0].message)
    except ImportError:
        pass


def test_override_toolsets():
    foo_toolset = FunctionToolset()

    @foo_toolset.tool_plain
    def foo() -> str:
        return 'Hello from foo'

    available_tools: list[list[str]] = []

    async def prepare_tools(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        nonlocal available_tools
        available_tools.append([tool_def.name for tool_def in tool_defs])
        return tool_defs

    agent = Agent('test', toolsets=[foo_toolset], prepare_tools=prepare_tools)

    @agent.tool_plain
    def baz() -> str:
        return 'Hello from baz'

    result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['baz', 'foo'])
    assert result.output == snapshot('{"baz":"Hello from baz","foo":"Hello from foo"}')

    bar_toolset = FunctionToolset()

    @bar_toolset.tool_plain
    def bar() -> str:
        return 'Hello from bar'

    with agent.override(toolsets=[bar_toolset]):
        result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['baz', 'bar'])
    assert result.output == snapshot('{"baz":"Hello from baz","bar":"Hello from bar"}')

    with agent.override(toolsets=[]):
        result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['baz'])
    assert result.output == snapshot('{"baz":"Hello from baz"}')

    result = agent.run_sync('Hello', toolsets=[bar_toolset])
    assert available_tools[-1] == snapshot(['baz', 'foo', 'bar'])
    assert result.output == snapshot('{"baz":"Hello from baz","foo":"Hello from foo","bar":"Hello from bar"}')

    with agent.override(toolsets=[]):
        result = agent.run_sync('Hello', toolsets=[bar_toolset])
    assert available_tools[-1] == snapshot(['baz'])
    assert result.output == snapshot('{"baz":"Hello from baz"}')


def test_override_tools():
    def foo() -> str:
        return 'Hello from foo'

    def bar() -> str:
        return 'Hello from bar'

    available_tools: list[list[str]] = []

    async def prepare_tools(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        nonlocal available_tools
        available_tools.append([tool_def.name for tool_def in tool_defs])
        return tool_defs

    agent = Agent('test', tools=[foo], prepare_tools=prepare_tools)

    result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['foo'])
    assert result.output == snapshot('{"foo":"Hello from foo"}')

    with agent.override(tools=[bar]):
        result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot(['bar'])
    assert result.output == snapshot('{"bar":"Hello from bar"}')

    with agent.override(tools=[]):
        result = agent.run_sync('Hello')
    assert available_tools[-1] == snapshot([])
    assert result.output == snapshot('success (no tool calls)')


def test_toolset_factory():
    toolset = FunctionToolset()

    @toolset.tool_plain
    def foo() -> str:
        return 'Hello from foo'

    available_tools: list[str] = []

    async def prepare_tools(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        nonlocal available_tools
        available_tools = [tool_def.name for tool_def in tool_defs]
        return tool_defs

    toolset_creation_counts: dict[str, int] = defaultdict(int)

    def via_toolsets_arg(ctx: RunContext[None]) -> AbstractToolset[None]:
        nonlocal toolset_creation_counts
        toolset_creation_counts['via_toolsets_arg'] += 1
        return toolset.prefixed('via_toolsets_arg')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('via_toolsets_arg_foo')])
        elif len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('via_toolset_decorator_foo')])
        else:
            return ModelResponse(parts=[TextPart('Done')])

    agent = Agent(FunctionModel(respond), toolsets=[via_toolsets_arg], prepare_tools=prepare_tools)

    @agent.toolset
    def via_toolset_decorator(ctx: RunContext[None]) -> AbstractToolset[None]:
        nonlocal toolset_creation_counts
        toolset_creation_counts['via_toolset_decorator'] += 1
        return toolset.prefixed('via_toolset_decorator')

    @agent.toolset(per_run_step=False)
    async def via_toolset_decorator_for_entire_run(ctx: RunContext[None]) -> AbstractToolset[None]:
        nonlocal toolset_creation_counts
        toolset_creation_counts['via_toolset_decorator_for_entire_run'] += 1
        return toolset.prefixed('via_toolset_decorator_for_entire_run')

    run_result = agent.run_sync('Hello')

    assert run_result._state.run_step == 3  # pyright: ignore[reportPrivateUsage]
    assert len(available_tools) == 3
    assert toolset_creation_counts == snapshot(
        defaultdict(int, {'via_toolsets_arg': 3, 'via_toolset_decorator': 3, 'via_toolset_decorator_for_entire_run': 1})
    )


def test_adding_tools_during_run():
    toolset = FunctionToolset()

    def foo() -> str:
        return 'Hello from foo'

    @toolset.tool_plain
    def add_foo_tool() -> str:
        toolset.add_function(foo)
        return 'foo tool added'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('add_foo_tool')])
        elif len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('foo')])
        else:
            return ModelResponse(parts=[TextPart('Done')])

    agent = Agent(FunctionModel(respond), toolsets=[toolset])
    result = agent.run_sync('Add the foo tool and run it')
    assert result.output == snapshot('Done')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add the foo tool and run it',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='add_foo_tool', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=57, output_tokens=2),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='add_foo_tool',
                        content='foo tool added',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foo', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=60, output_tokens=4),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foo',
                        content='Hello from foo',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done')],
                usage=RequestUsage(input_tokens=63, output_tokens=5),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_prepare_output_tools():
    @dataclass
    class AgentDeps:
        plan_presented: bool = False

    async def present_plan(ctx: RunContext[AgentDeps], plan: str) -> str:
        """
        Present the plan to the user.
        """
        ctx.deps.plan_presented = True
        return plan

    async def run_sql(ctx: RunContext[AgentDeps], purpose: str, query: str) -> str:
        """
        Run an SQL query.
        """
        return 'SQL query executed successfully'

    async def only_if_plan_presented(
        ctx: RunContext[AgentDeps], tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        return tool_defs if ctx.deps.plan_presented else []

    agent = Agent(
        model='test',
        deps_type=AgentDeps,
        tools=[present_plan],
        output_type=[ToolOutput(run_sql, name='run_sql')],
        prepare_output_tools=only_if_plan_presented,
    )

    result = agent.run_sync('Hello', deps=AgentDeps())
    assert result.output == snapshot('SQL query executed successfully')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='present_plan',
                        args={'plan': 'a'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='present_plan',
                        content='a',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_sql',
                        args={'purpose': 'a', 'query': 'a'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=12),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_sql',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_prepare_output_tools_receives_output_max_retries():
    """Regression for the bug surfaced in #4745 / #4859 design discussion: the
    `prepare_output_tools` callable must see `ctx.max_retries == max_output_retries`,
    not the function-tool retry budget. Also verifies `ctx.retry` advances per output retry
    and that per-tool retry counts in `ctx.retries` propagate (matching `prepare_tools`).
    """
    seen_retries: list[int] = []
    seen_max_retries: list[int] = []
    seen_per_tool_retries: list[int] = []

    target_retries = 3

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools
        # Always return the same output; the validator drives retries.
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"a": 1, "b": "foo"}')])

    async def prep(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        seen_retries.append(ctx.retry)
        seen_max_retries.append(ctx.max_retries)
        seen_per_tool_retries.append(ctx.retries.get(tool_defs[0].name, 0))
        return tool_defs

    agent = Agent(
        FunctionModel(return_model),
        output_type=Foo,
        output_retries=target_retries,
        tool_retries=1,  # tool retry budget — different from output, must NOT leak into prep ctx
        prepare_output_tools=prep,
    )

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        if ctx.retry == target_retries:
            return o
        raise ModelRetry(f'Retry {ctx.retry}')

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)

    # `prep` runs once per step (= target_retries + 1 calls). Each call should see the
    # OUTPUT retry budget, never the tool retry budget (`retries=1`).
    assert seen_max_retries == [target_retries] * (target_retries + 1)
    assert seen_retries == [0, 1, 2, 3]
    # Per-tool retry counts populated by `for_run_step` propagate too — matching
    # what `prepare_tools` sees on the function-tool side.
    assert seen_per_tool_retries == [0, 1, 2, 3]


def test_prepare_output_tools_sees_run_level_output_retries_override():
    """`prepare_output_tools` sees the run-level `output_retries` override on `ctx.max_retries`,
    not the agent-level default — so capability hooks observe the same budget the run will enforce.

    Regression for https://github.com/pydantic/pydantic-ai/pull/5075#discussion_r3170685538.
    """
    seen_max_retries: list[int] = []

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"a": 1, "b": "foo"}')])

    async def prep(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        seen_max_retries.append(ctx.max_retries)
        return tool_defs

    agent = Agent(
        FunctionModel(return_model),
        output_type=Foo,
        output_retries=5,
        prepare_output_tools=prep,
    )

    result = agent.run_sync('Hello', output_retries=2)
    assert isinstance(result.output, Foo)
    # Hook sees run-level override (2), not the agent-level default (5).
    assert seen_max_retries == [2]


async def test_explicit_context_manager():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    toolset = CombinedToolset([server1, PrefixedToolset(server2, 'prefix')])
    agent = Agent('test', toolsets=[toolset])

    async with agent:
        assert server1.is_running
        assert server2.is_running

        async with agent:
            assert server1.is_running
            assert server2.is_running


async def test_implicit_context_manager():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    toolset = CombinedToolset([server1, PrefixedToolset(server2, 'prefix')])
    agent = Agent('test', toolsets=[toolset])

    async with agent.iter(user_prompt='Hello'):
        assert server1.is_running
        assert server2.is_running


def test_parallel_mcp_calls():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    async def call_tools_parallel(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_none'),
                    ToolCallPart(tool_name='get_multiple_items'),
                ]
            )
        else:
            return ModelResponse(parts=[TextPart('finished')])

    server = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    agent = Agent(FunctionModel(call_tools_parallel), toolsets=[server])
    result = agent.run_sync('call tools in parallel')
    assert result.output == snapshot('finished')


async def test_parallel_tool_exception_cancels_sibling_tasks():
    """Non-CancelledError exceptions during parallel tool execution must cancel sibling tasks.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4423.
    Previously only asyncio.CancelledError triggered cleanup; any other exception
    left the remaining tasks running as orphaned asyncio tasks.
    """
    slow_tool_started = asyncio.Event()
    slow_tool_cancelled = asyncio.Event()

    async def call_two_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name='fast_failing_tool'),
                ToolCallPart(tool_name='slow_tool'),
            ]
        )

    agent = Agent(FunctionModel(call_two_tools))

    @agent.tool_plain
    async def fast_failing_tool() -> str:
        # Yield control so slow_tool can start, then raise.
        await asyncio.sleep(0)
        raise RuntimeError('boom')

    @agent.tool_plain
    async def slow_tool() -> str:
        slow_tool_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            slow_tool_cancelled.set()
            raise
        return 'done'  # pragma: no cover

    tasks_before = asyncio.all_tasks()
    with pytest.raises(RuntimeError, match='boom'):
        await agent.run('call tools')

    # Give the event loop a moment to process cancellations.
    await asyncio.sleep(0)

    # The slow tool must have started (confirming both tasks ran in parallel).
    assert slow_tool_started.is_set(), 'slow_tool never started — not running in parallel'
    # The slow tool must have been cancelled when fast_failing_tool raised.
    assert slow_tool_cancelled.is_set(), 'slow_tool was not cancelled after RuntimeError'
    # No new asyncio tasks should be left over from this run.
    leaked = asyncio.all_tasks() - tasks_before
    assert not leaked, f'Orphaned tasks remain: {leaked}'


@pytest.mark.parametrize('mode', ['argument', 'contextmanager'])
def test_sequential_calls(mode: Literal['argument', 'contextmanager']):
    """Test that tool calls are executed correctly when a `sequential` tool is present in the call."""

    async def call_tools_sequential(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name='call_first'),
                ToolCallPart(tool_name='call_first'),
                ToolCallPart(tool_name='call_first'),
                ToolCallPart(tool_name='call_first'),
                ToolCallPart(tool_name='call_first'),
                ToolCallPart(tool_name='call_first'),
                ToolCallPart(tool_name='increment_integer_holder'),
                ToolCallPart(tool_name='requires_approval'),
                ToolCallPart(tool_name='call_second'),
                ToolCallPart(tool_name='call_second'),
                ToolCallPart(tool_name='call_second'),
                ToolCallPart(tool_name='call_second'),
                ToolCallPart(tool_name='call_second'),
                ToolCallPart(tool_name='call_second'),
                ToolCallPart(tool_name='call_second'),
            ]
        )

    sequential_toolset = FunctionToolset()

    integer_holder: int = 1

    @sequential_toolset.tool_plain
    def call_first():
        nonlocal integer_holder
        assert integer_holder == 1

    @sequential_toolset.tool_plain(sequential=mode == 'argument')
    def increment_integer_holder():
        nonlocal integer_holder
        integer_holder = 2

    @sequential_toolset.tool_plain
    def requires_approval():
        from pydantic_ai.exceptions import ApprovalRequired

        raise ApprovalRequired()

    @sequential_toolset.tool_plain
    def call_second():
        nonlocal integer_holder
        assert integer_holder == 2

    agent = Agent(
        FunctionModel(call_tools_sequential), toolsets=[sequential_toolset], output_type=[str, DeferredToolRequests]
    )

    user_prompt = 'call a lot of tools'

    if mode == 'contextmanager':
        with agent.parallel_tool_call_execution_mode('sequential'):
            result = agent.run_sync(user_prompt)
    else:
        result = agent.run_sync(user_prompt)

    assert result.output == snapshot(
        DeferredToolRequests(approvals=[ToolCallPart(tool_name='requires_approval', tool_call_id=IsStr())])
    )
    assert integer_holder == 2


def test_set_mcp_sampling_model():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    test_model = TestModel()
    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = MCPServerStdio('python', ['-m', 'tests.mcp_server'], sampling_model=test_model)
    toolset = CombinedToolset([server1, PrefixedToolset(server2, 'prefix')])
    agent = Agent(None, toolsets=[toolset])

    with pytest.raises(UserError, match='No sampling model provided and no model set on the agent.'):
        agent.set_mcp_sampling_model()
    assert server1.sampling_model is None
    assert server2.sampling_model is test_model

    agent.model = test_model
    agent.set_mcp_sampling_model()
    assert server1.sampling_model is test_model
    assert server2.sampling_model is test_model

    function_model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Hello')]))
    with agent.override(model=function_model):
        agent.set_mcp_sampling_model()
        assert server1.sampling_model is function_model
        assert server2.sampling_model is function_model

    function_model2 = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('Goodbye')]))
    agent.set_mcp_sampling_model(function_model2)
    assert server1.sampling_model is function_model2
    assert server2.sampling_model is function_model2


def test_toolsets():
    toolset = FunctionToolset()

    @toolset.tool_plain
    def foo() -> str:
        return 'Hello from foo'  # pragma: no cover

    agent = Agent('test', toolsets=[toolset])
    assert toolset in agent.toolsets

    other_toolset = FunctionToolset()
    with agent.override(toolsets=[other_toolset]):
        assert other_toolset in agent.toolsets
        assert toolset not in agent.toolsets


async def test_wrapper_agent():
    async def event_stream_handler(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]):
        pass  # pragma: no cover

    foo_toolset = FunctionToolset()

    @foo_toolset.tool_plain
    def foo() -> str:
        return 'Hello from foo'  # pragma: no cover

    test_model = TestModel()
    agent = Agent(
        test_model,
        system_prompt='You are a wrapped agent',
        toolsets=[foo_toolset],
        output_type=Foo,
        event_stream_handler=event_stream_handler,
    )
    wrapper_agent = WrapperAgent(agent)
    assert [p.content for p in await wrapper_agent.system_prompt_parts()] == ['You are a wrapped agent']
    assert wrapper_agent.toolsets == agent.toolsets
    assert wrapper_agent.model == agent.model
    assert wrapper_agent.name == agent.name
    wrapper_agent.name = 'wrapped'
    assert wrapper_agent.name == 'wrapped'
    assert wrapper_agent.description == agent.description
    wrapper_agent.description = 'wrapped description'
    assert wrapper_agent.description == 'wrapped description'
    assert wrapper_agent.output_type == agent.output_type
    assert wrapper_agent.event_stream_handler == agent.event_stream_handler
    assert wrapper_agent.root_capability is agent.root_capability
    assert wrapper_agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'string'}},
            'title': 'Foo',
            'required': ['a', 'b'],
        }
    )
    assert wrapper_agent.output_json_schema(output_type=str) == snapshot({'type': 'string'})

    bar_toolset = FunctionToolset()

    @bar_toolset.tool_plain
    def bar() -> str:
        return 'Hello from bar'

    with wrapper_agent.override(toolsets=[bar_toolset]):
        async with wrapper_agent:
            async with wrapper_agent.iter(user_prompt='Hello') as run:
                async for _ in run:
                    pass

    assert run.result is not None
    assert run.result.output == snapshot(Foo(a=0, b='a'))
    assert test_model.last_model_request_parameters is not None
    assert [t.name for t in test_model.last_model_request_parameters.function_tools] == snapshot(['bar'])


async def test_thinking_only_response_retry():
    """Test that thinking-only responses trigger a retry mechanism."""

    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call: return thinking-only response
            return ModelResponse(
                parts=[ThinkingPart(content='Let me think about this...')],
                model_name='thinking-test-model',
            )
        else:
            # Second call: return proper response
            return ModelResponse(
                parts=[TextPart(content='Final answer')],
                model_name='thinking-test-model',
            )

    model = FunctionModel(model_function)
    agent = Agent(model, instructions='You are a helpful assistant.')

    result = await agent.run('Hello')

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='Let me think about this...')],
                usage=RequestUsage(input_tokens=51, output_tokens=6),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                instructions='You are a helpful assistant.',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Final answer')],
                usage=RequestUsage(input_tokens=63, output_tokens=8),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_retry_message_no_tools():
    """Test that thinking-only retry message does not suggest 'call a tool' when no function tools are registered."""
    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return ModelResponse(parts=[ThinkingPart(content='thinking...')])
        else:
            return ModelResponse(parts=[TextPart(content='result')])

    agent = Agent(FunctionModel(model_function))
    result = await agent.run('Hello')

    assert result.output == 'result'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='thinking...')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please return text.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='result')],
                usage=RequestUsage(input_tokens=63, output_tokens=3),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_thinking_only_response_retry_with_tool_output():
    """Test that thinking-only responses retry with tool-output guidance when text output is not allowed."""
    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        assert info.output_tools is not None

        if call_count == 1:
            return ModelResponse(parts=[ThinkingPart(content='thinking...')])
        else:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"a": 1, "b": "ok"}')])

    result = await Agent(FunctionModel(model_function), output_type=ToolOutput(Foo)).run('Hello')

    assert result.output == Foo(a=1, b='ok')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='thinking...')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please include your response in a tool call.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"a": 1, "b": "ok"}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=68, output_tokens=9),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_thinking_only_response_backward_looking_recovery():
    """Test that thinking-only response after a tool call recovers text from prior model response."""
    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First call: return text alongside a tool call
            return ModelResponse(
                parts=[
                    TextPart(content="Here's my analysis."),
                    ToolCallPart(tool_name='save_progress', args='{"data": "test"}'),
                ],
            )
        else:
            # Second call (after tool return): return thinking-only
            return ModelResponse(
                parts=[ThinkingPart(content='Nothing more to say.')],
            )

    agent = Agent(FunctionModel(model_function))

    @agent.tool_plain
    def save_progress(data: str) -> str:
        return 'saved'

    result = await agent.run('Hello')

    assert result.output == "Here's my analysis."
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    TextPart(content="Here's my analysis."),
                    ToolCallPart(tool_name='save_progress', args='{"data": "test"}', tool_call_id=IsStr()),
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=9),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='save_progress',
                        content='saved',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ThinkingPart(content='Nothing more to say.')],
                usage=RequestUsage(input_tokens=52, output_tokens=14),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_hitl_tool_approval():
    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='create_file',
                        args={'path': 'new_file.py', 'content': 'print("Hello, world!")'},
                        tool_call_id='create_file',
                    ),
                    ToolCallPart(
                        tool_name='delete_file', args={'path': 'ok_to_delete.py'}, tool_call_id='ok_to_delete'
                    ),
                    ToolCallPart(
                        tool_name='delete_file', args={'path': 'never_delete.py'}, tool_call_id='never_delete'
                    ),
                ]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    model = FunctionModel(model_function)

    agent = Agent(model, output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        return f'File {path!r} deleted'

    @agent.tool_plain
    def create_file(path: str, content: str) -> str:
        return f'File {path!r} created with content: {content}'

    result = await agent.run('Create new_file.py and delete ok_to_delete.py and never_delete.py')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create new_file.py and delete ok_to_delete.py and never_delete.py',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='create_file',
                        args={'path': 'new_file.py', 'content': 'print("Hello, world!")'},
                        tool_call_id='create_file',
                    ),
                    ToolCallPart(
                        tool_name='delete_file', args={'path': 'ok_to_delete.py'}, tool_call_id='ok_to_delete'
                    ),
                    ToolCallPart(
                        tool_name='delete_file', args={'path': 'never_delete.py'}, tool_call_id='never_delete'
                    ),
                ],
                usage=RequestUsage(input_tokens=60, output_tokens=23),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='create_file',
                        content='File \'new_file.py\' created with content: print("Hello, world!")',
                        tool_call_id='create_file',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot(
        DeferredToolRequests(
            approvals=[
                ToolCallPart(tool_name='delete_file', args={'path': 'ok_to_delete.py'}, tool_call_id='ok_to_delete'),
                ToolCallPart(tool_name='delete_file', args={'path': 'never_delete.py'}, tool_call_id='never_delete'),
            ]
        )
    )

    result = await agent.run(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={'ok_to_delete': True, 'never_delete': ToolDenied('File cannot be deleted')},
        ),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Create new_file.py and delete ok_to_delete.py and never_delete.py',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='create_file',
                        args={'path': 'new_file.py', 'content': 'print("Hello, world!")'},
                        tool_call_id='create_file',
                    ),
                    ToolCallPart(
                        tool_name='delete_file', args={'path': 'ok_to_delete.py'}, tool_call_id='ok_to_delete'
                    ),
                    ToolCallPart(
                        tool_name='delete_file', args={'path': 'never_delete.py'}, tool_call_id='never_delete'
                    ),
                ],
                usage=RequestUsage(input_tokens=60, output_tokens=23),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='create_file',
                        content='File \'new_file.py\' created with content: print("Hello, world!")',
                        tool_call_id='create_file',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delete_file',
                        content="File 'ok_to_delete.py' deleted",
                        tool_call_id='ok_to_delete',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='delete_file',
                        content='File cannot be deleted',
                        tool_call_id='never_delete',
                        timestamp=IsDatetime(),
                        outcome='denied',
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=78, output_tokens=24),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot('Done!')

    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delete_file',
                        content="File 'ok_to_delete.py' deleted",
                        tool_call_id='ok_to_delete',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='delete_file',
                        content='File cannot be deleted',
                        tool_call_id='never_delete',
                        timestamp=IsDatetime(),
                        outcome='denied',
                    ),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=78, output_tokens=24),
                model_name='function:model_function:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_run_with_deferred_tool_results_errors():
    agent = Agent('test')

    message_history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['Hello', 'world'])])]

    with pytest.raises(
        UserError,
        match='Tool call results were provided, but the message history does not contain a `ModelResponse`.',
    ):
        await agent.run(
            'Hello again',
            message_history=message_history,
            deferred_tool_results=DeferredToolResults(approvals={'create_file': True}),
        )

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[TextPart(content='Hello to you too!')]),
    ]

    with pytest.raises(
        UserError,
        match='Tool call results were provided, but the message history does not contain any unprocessed tool calls.',
    ):
        await agent.run(
            'Hello again',
            message_history=message_history,
            deferred_tool_results=DeferredToolResults(approvals={'create_file': True}),
        )

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='say_hello')]),
    ]

    with pytest.raises(
        UserError, match='Cannot provide a new user prompt when the message history contains unprocessed tool calls.'
    ):
        await agent.run('Hello', message_history=message_history)

    with pytest.raises(UserError, match='Tool call results need to be provided for all deferred tool calls.'):
        await agent.run(
            message_history=message_history,
            deferred_tool_results=DeferredToolResults(),
        )

    with pytest.raises(UserError, match='Tool call results were provided, but the message history is empty.'):
        await agent.run(
            'Hello again',
            deferred_tool_results=DeferredToolResults(approvals={'create_file': True}),
        )

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(
            parts=[
                ToolCallPart(tool_name='run_me', tool_call_id='run_me'),
                ToolCallPart(tool_name='run_me_too', tool_call_id='run_me_too'),
                ToolCallPart(tool_name='defer_me', tool_call_id='defer_me'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='run_me', tool_call_id='run_me', content='Success'),
                RetryPromptPart(tool_name='run_me_too', tool_call_id='run_me_too', content='Failure'),
            ]
        ),
    ]

    with pytest.raises(UserError, match="Tool call 'run_me' was already executed and its result cannot be overridden."):
        await agent.run(
            message_history=message_history,
            deferred_tool_results=DeferredToolResults(
                calls={'run_me': 'Failure', 'defer_me': 'Failure'},
            ),
        )

    with pytest.raises(
        UserError, match="Tool call 'run_me_too' was already executed and its result cannot be overridden."
    ):
        await agent.run(
            message_history=message_history,
            deferred_tool_results=DeferredToolResults(
                calls={'run_me_too': 'Success', 'defer_me': 'Failure'},
            ),
        )


async def test_user_prompt_with_deferred_tool_results():
    """Test that user_prompt can be provided alongside deferred_tool_results."""
    from pydantic_ai.exceptions import ApprovalRequired

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # First call: model requests tool approval
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='update_file', tool_call_id='update_file_1', args={'path': '.env', 'content': ''}
                    ),
                ]
            )
        # Second call: model responds to tool results and user prompt
        else:
            # Verify we received both tool results and user prompt
            last_request = messages[-1]
            assert isinstance(last_request, ModelRequest)
            has_tool_return = any(isinstance(p, ToolReturnPart) for p in last_request.parts)
            has_user_prompt = any(isinstance(p, UserPromptPart) for p in last_request.parts)
            assert has_tool_return, 'Expected tool return part in request'
            assert has_user_prompt, 'Expected user prompt part in request'

            # Get user prompt content
            user_prompt_content = next(p.content for p in last_request.parts if isinstance(p, UserPromptPart))
            return ModelResponse(parts=[TextPart(f'Approved and {user_prompt_content}')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def update_file(ctx: RunContext, path: str, content: str) -> str:
        if path == '.env' and not ctx.tool_call_approved:
            raise ApprovalRequired
        return f'File {path!r} updated'

    # First run: get deferred tool requests
    result = await agent.run('Update .env file')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1

    messages = result.all_messages()
    # Snapshot the message history after first run to show the state before deferred tool results
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Update .env file', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='update_file', tool_call_id='update_file_1', args={'path': '.env', 'content': ''}
                    )
                ],
                usage=RequestUsage(input_tokens=53, output_tokens=6),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # Second run: provide approvals AND user prompt
    results = DeferredToolResults(approvals={result.output.approvals[0].tool_call_id: True})
    result2 = await agent.run('continue with the operation', message_history=messages, deferred_tool_results=results)

    assert isinstance(result2.output, str)
    assert 'continue with the operation' in result2.output

    # Snapshot the new messages to show how tool results and user prompt are combined
    new_messages = result2.new_messages()
    assert new_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_file',
                        content="File '.env' updated",
                        tool_call_id='update_file_1',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='continue with the operation', timestamp=IsDatetime()),
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Approved and continue with the operation')],
                usage=RequestUsage(input_tokens=61, output_tokens=12),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_tool_requires_approval_no_output_type():
    """Adding a requires_approval tool without DeferredToolRequests in output type is allowed.

    The error is raised at runtime if the tool is called and no handler resolves it.
    """
    agent = Agent('test')

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> None:
        pass


async def test_consecutive_model_responses_in_history():
    received_messages: list[ModelMessage] | None = None

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal received_messages
        received_messages = messages
        return ModelResponse(
            parts=[
                TextPart('All right then, goodbye!'),
            ]
        )

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Hello...')]),
        ModelResponse(parts=[TextPart(content='...world!')]),
        ModelResponse(parts=[TextPart(content='Anything else I can help with?')]),
    ]

    m = FunctionModel(llm)
    agent = Agent(m)
    result = await agent.run('No thanks', message_history=history)

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello...',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='...world!'), TextPart(content='Anything else I can help with?')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='No thanks',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='All right then, goodbye!')],
                usage=RequestUsage(input_tokens=54, output_tokens=12),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='No thanks',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='All right then, goodbye!')],
                usage=RequestUsage(input_tokens=54, output_tokens=12),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello...',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='...world!'), TextPart(content='Anything else I can help with?')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='No thanks',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_override_instructions_basic():
    """Test that override can override instructions."""
    agent = Agent('test')

    @agent.instructions
    def instr_fn() -> str:
        return 'SHOULD_BE_IGNORED'

    with capture_run_messages() as base_messages:
        agent.run_sync('Hello', model=TestModel(custom_output_text='baseline'))

    base_req = base_messages[0]
    assert isinstance(base_req, ModelRequest)
    assert base_req.instructions == 'SHOULD_BE_IGNORED'

    with agent.override(instructions='OVERRIDE'):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'OVERRIDE'


def test_override_reset_after_context():
    """Test that instructions are reset after exiting the override context."""
    agent = Agent('test', instructions='ORIG')

    with agent.override(instructions='NEW'):
        with capture_run_messages() as messages_new:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    with capture_run_messages() as messages_orig:
        agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req_new = messages_new[0]
    assert isinstance(req_new, ModelRequest)
    req_orig = messages_orig[0]
    assert isinstance(req_orig, ModelRequest)
    assert req_new.instructions == 'NEW'
    assert req_orig.instructions == 'ORIG'


def test_override_none_clears_instructions():
    """Test that passing None for instructions clears all instructions."""
    agent = Agent('test', instructions='BASE')

    @agent.instructions
    def instr_fn() -> str:  # pragma: no cover - ignored under override
        return 'ALSO_BASE'

    with agent.override(instructions=None):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions is None


def test_override_instructions_callable_replaces_functions():
    """Override with a callable should replace existing instruction functions."""
    agent = Agent('test')

    @agent.instructions
    def base_fn() -> str:
        return 'BASE_FN'

    def override_fn() -> str:
        return 'OVERRIDE_FN'

    with capture_run_messages() as base_messages:
        agent.run_sync('Hello', model=TestModel(custom_output_text='baseline'))

    base_req = base_messages[0]
    assert isinstance(base_req, ModelRequest)
    assert base_req.instructions is not None
    assert 'BASE_FN' in base_req.instructions

    with agent.override(instructions=override_fn):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'OVERRIDE_FN'
    assert 'BASE_FN' not in req.instructions


async def test_override_instructions_async_callable():
    """Override with an async callable should be awaited."""
    agent = Agent('test')

    async def override_fn() -> str:
        await asyncio.sleep(0)
        return 'ASYNC_FN'

    with agent.override(instructions=override_fn):
        with capture_run_messages() as messages:
            await agent.run('Hi', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'ASYNC_FN'


def test_override_instructions_sequence_mixed_types():
    """Override can mix literal strings and functions."""
    agent = Agent('test', instructions='BASE')

    def override_fn() -> str:
        return 'FUNC_PART'

    def override_fn_2() -> str:
        return 'FUNC_PART_2'

    with agent.override(instructions=['OVERRIDE1', override_fn, 'OVERRIDE2', override_fn_2]):
        with capture_run_messages() as messages:
            agent.run_sync('Hello', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'OVERRIDE1\nOVERRIDE2\n\nFUNC_PART\n\nFUNC_PART_2'
    assert 'BASE' not in req.instructions


async def test_override_concurrent_isolation():
    """Test that concurrent overrides are isolated from each other."""
    agent = Agent('test', instructions='ORIG')

    async def run_with(instr: str) -> str | None:
        with agent.override(instructions=instr):
            with capture_run_messages() as messages:
                await agent.run('Hi', model=TestModel(custom_output_text='ok'))
            req = messages[0]
            assert isinstance(req, ModelRequest)
            return req.instructions

    a, b = await asyncio.gather(
        run_with('A'),
        run_with('B'),
    )

    assert a == 'A'
    assert b == 'B'


def test_override_replaces_instructions():
    """Test overriding instructions replaces the base instructions."""
    agent = Agent('test', instructions='ORIG_INSTR')

    with agent.override(instructions='NEW_INSTR'):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'NEW_INSTR'


def test_override_nested_contexts():
    """Test nested override contexts."""
    agent = Agent('test', instructions='ORIG')

    with agent.override(instructions='OUTER'):
        with capture_run_messages() as outer_messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

        with agent.override(instructions='INNER'):
            with capture_run_messages() as inner_messages:
                agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    outer_req = outer_messages[0]
    assert isinstance(outer_req, ModelRequest)
    inner_req = inner_messages[0]
    assert isinstance(inner_req, ModelRequest)

    assert outer_req.instructions == 'OUTER'
    assert inner_req.instructions == 'INNER'


async def test_override_async_run():
    """Test override with async run method."""
    agent = Agent('test', instructions='ORIG')

    with agent.override(instructions='ASYNC_OVERRIDE'):
        with capture_run_messages() as messages:
            await agent.run('Hi', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'ASYNC_OVERRIDE'


def test_override_with_dynamic_prompts():
    """Test override interacting with dynamic prompts."""
    agent = Agent('test')

    dynamic_value = 'DYNAMIC'

    @agent.system_prompt
    def dynamic_sys() -> str:
        return dynamic_value

    @agent.instructions
    def dynamic_instr() -> str:
        return 'DYNAMIC_INSTR'

    with capture_run_messages() as base_messages:
        agent.run_sync('Hi', model=TestModel(custom_output_text='baseline'))

    base_req = base_messages[0]
    assert isinstance(base_req, ModelRequest)
    assert base_req.instructions == 'DYNAMIC_INSTR'

    # Override should take precedence over dynamic instructions but leave system prompts intact
    with agent.override(instructions='OVERRIDE_INSTR'):
        with capture_run_messages() as messages:
            agent.run_sync('Hi', model=TestModel(custom_output_text='ok'))

    req = messages[0]
    assert isinstance(req, ModelRequest)
    assert req.instructions == 'OVERRIDE_INSTR'
    sys_texts = [p.content for p in req.parts if isinstance(p, SystemPromptPart)]
    # The dynamic system prompt should still be present since overrides target instructions only
    assert dynamic_value in sys_texts


def test_continue_conversation_that_ended_in_output_tool_call(allow_model_requests: None):
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) and p.tool_name == 'roll_dice' for p in messages[-1].parts):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'dice_roll': 4},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ]
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='roll_dice', args={}, tool_call_id='pyd_ai_tool_call_id__roll_dice')]
        )

    class Result(BaseModel):
        dice_roll: int

    agent = Agent(FunctionModel(llm), output_type=Result)

    @agent.tool_plain
    def roll_dice() -> int:
        return 4

    result = agent.run_sync('Roll me a dice.')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Roll me a dice.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='roll_dice', args={}, tool_call_id='pyd_ai_tool_call_id__roll_dice')],
                usage=RequestUsage(input_tokens=55, output_tokens=2),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='roll_dice',
                        content=4,
                        tool_call_id='pyd_ai_tool_call_id__roll_dice',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'dice_roll': 4},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=56, output_tokens=6),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    result = agent.run_sync('Roll me a dice again.', message_history=messages)
    new_messages = result.new_messages()
    assert new_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Roll me a dice again.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='roll_dice', args={}, tool_call_id='pyd_ai_tool_call_id__roll_dice')],
                usage=RequestUsage(input_tokens=66, output_tokens=8),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='roll_dice',
                        content=4,
                        tool_call_id='pyd_ai_tool_call_id__roll_dice',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'dice_roll': 4},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=67, output_tokens=12),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert not any(isinstance(p, ToolReturnPart) and p.tool_name == 'final_result' for p in new_messages[0].parts)


def test_agent_builtin_tools_runtime_vs_agent_level():
    """Test that runtime builtin_tools parameter is merged with agent-level builtin_tools."""
    model = TestModel()

    agent = Agent(
        model=model,
        builtin_tools=[
            WebSearchTool(),
            CodeExecutionTool(),
            MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'),
            MCPServerTool(id='github', url='https://api.githubcopilot.com/mcp'),
        ],
    )

    # Runtime tool with same unique ID should override agent-level tool
    with pytest.raises(Exception, match='TestModel does not support built-in tools'):
        agent.run_sync(
            'Hello',
            builtin_tools=[
                WebSearchTool(search_context_size='high'),
                MCPServerTool(id='example', url='https://mcp.example.com/mcp'),
                MCPServerTool(id='github', url='https://mcp.githubcopilot.com/mcp', authorization_token='token'),
            ],
        )

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == snapshot(
        [
            WebSearchTool(search_context_size='high'),
            CodeExecutionTool(),
            MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'),
            MCPServerTool(id='github', url='https://mcp.githubcopilot.com/mcp', authorization_token='token'),
            MCPServerTool(id='example', url='https://mcp.example.com/mcp'),
        ]
    )


def test_agent_override_builtin_tools_empty_runs_with_test_model():
    """Test that agent-level builtin tools can be removed when overriding the model."""
    model = TestModel()
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with agent.override(model=model, builtin_tools=[]):
        result = agent.run_sync('Hello')

    assert result.output == 'success (no tool calls)'
    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == []


def test_agent_override_builtin_tools_replaces_agent_level_tools():
    """Test that override builtin_tools replace, rather than append to, agent-level builtin tools."""
    model = TestModel()
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with (
        agent.override(builtin_tools=[CodeExecutionTool()]),
        pytest.raises(UserError, match='TestModel does not support built-in tools'),
    ):
        agent.run_sync('Hello')

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == snapshot([CodeExecutionTool()])


def test_agent_override_builtin_tools_preserves_runtime_additive_tools():
    """Test that runtime builtin_tools are still added to overridden builtin tools."""
    model = TestModel()
    agent = Agent(model=model, builtin_tools=[WebSearchTool()])

    with (
        agent.override(builtin_tools=[CodeExecutionTool()]),
        pytest.raises(UserError, match='TestModel does not support built-in tools'),
    ):
        agent.run_sync(
            'Hello',
            builtin_tools=[MCPServerTool(id='example', url='https://mcp.example.com/mcp')],
        )

    assert model.last_model_request_parameters is not None
    assert model.last_model_request_parameters.builtin_tools == snapshot(
        [CodeExecutionTool(), MCPServerTool(id='example', url='https://mcp.example.com/mcp')]
    )


async def test_run_with_unapproved_tool_call_in_history():
    def should_not_call_model(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise ValueError('The agent should not call the model.')  # pragma: no cover

    agent = Agent(
        model=FunctionModel(function=should_not_call_model),
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(requires_approval=True)
    def delete_file() -> None:
        print('File deleted.')  # pragma: no cover

    messages = [
        ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ModelResponse(parts=[ToolCallPart(tool_name='delete_file')]),
    ]

    result = await agent.run(message_history=messages)

    assert result.all_messages() == messages
    assert result.output == snapshot(
        DeferredToolRequests(approvals=[ToolCallPart(tool_name='delete_file', tool_call_id=IsStr())])
    )


async def test_message_history():
    def llm(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('ok here is text')])

    agent = Agent(FunctionModel(llm))

    async with agent.iter(
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='Hello')]),
        ],
    ) as run:
        async for _ in run:
            pass
        assert run.new_messages() == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='ok here is text')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:llm:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        assert run.new_messages_json().startswith(b'[{"parts":[{"content":"ok here is text",')
        assert run.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Hello',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='ok here is text')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:llm:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
        assert run.all_messages_json().startswith(b'[{"parts":[{"content":"Hello",')


@dataclass
class UserContext:
    location: str | None


async def prepared_web_search(ctx: RunContext[UserContext]) -> WebSearchTool | None:
    if not ctx.deps.location:
        return None

    return WebSearchTool(
        search_context_size='medium',
        user_location=WebSearchUserLocation(city=ctx.deps.location),
    )


async def test_dynamic_builtin_tool_configured():
    model = TestModel()
    agent = Agent(model, builtin_tools=[prepared_web_search], deps_type=UserContext)

    user_context = UserContext(location='London')

    with pytest.raises(UserError, match='TestModel does not support built-in tools'):
        await agent.run('Hello', deps=user_context)

    assert model.last_model_request_parameters is not None
    tools = model.last_model_request_parameters.builtin_tools
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.user_location is not None
    assert tool.user_location.get('city') == 'London'
    assert tool.search_context_size == 'medium'


async def test_dynamic_builtin_tool_omitted():
    model = TestModel()
    agent = Agent(model, builtin_tools=[prepared_web_search], deps_type=UserContext)

    user_context = UserContext(location=None)

    await agent.run('Hello', deps=user_context)

    assert model.last_model_request_parameters is not None
    tools = model.last_model_request_parameters.builtin_tools
    assert len(tools) == 0


async def test_mixed_static_and_dynamic_builtin_tools():
    model = TestModel()

    static_tool = CodeExecutionTool()
    agent = Agent(model, builtin_tools=[static_tool, prepared_web_search], deps_type=UserContext)

    # Case 1: Dynamic tool returns None
    with pytest.raises(UserError, match='TestModel does not support built-in tools'):
        await agent.run('Hello', deps=UserContext(location=None))

    assert model.last_model_request_parameters is not None
    tools = model.last_model_request_parameters.builtin_tools
    assert len(tools) == 1
    assert tools[0] == static_tool

    # Case 2: Dynamic tool returns a tool
    with pytest.raises(UserError, match='TestModel does not support built-in tools'):
        await agent.run('Hello', deps=UserContext(location='Paris'))

    assert model.last_model_request_parameters is not None
    tools = model.last_model_request_parameters.builtin_tools
    assert len(tools) == 2
    assert tools[0] == static_tool
    dynamic_tool = tools[1]
    assert isinstance(dynamic_tool, WebSearchTool)
    assert dynamic_tool.user_location is not None
    assert dynamic_tool.user_location.get('city') == 'Paris'


def sync_dynamic_tool(ctx: RunContext[UserContext]) -> WebSearchTool:
    """Verify that synchronous functions work."""
    return WebSearchTool(search_context_size='low')


async def test_sync_dynamic_tool():
    model = TestModel()
    agent = Agent(model, builtin_tools=[sync_dynamic_tool], deps_type=UserContext)

    with pytest.raises(UserError, match='TestModel does not support built-in tools'):
        await agent.run('Hello', deps=UserContext(location='London'))

    assert model.last_model_request_parameters is not None
    tools = model.last_model_request_parameters.builtin_tools
    assert len(tools) == 1
    assert isinstance(tools[0], WebSearchTool)
    assert tools[0].search_context_size == 'low'


async def test_dynamic_tool_in_run_call():
    """Verify dynamic tools can be passed to agent.run()."""
    model = TestModel()
    agent = Agent(model, deps_type=UserContext)

    with pytest.raises(UserError, match='TestModel does not support built-in tools'):
        await agent.run('Hello', deps=UserContext(location='Berlin'), builtin_tools=[prepared_web_search])

    assert model.last_model_request_parameters is not None
    tools = model.last_model_request_parameters.builtin_tools
    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.user_location is not None
    assert tool.user_location.get('city') == 'Berlin'


async def test_central_content_filter_handling():
    """
    Test that the agent graph correctly raises ContentFilterError
    when a model returns finish_reason='content_filter' AND empty content.
    """

    async def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[],
            model_name='test-model',
            finish_reason='content_filter',
            provider_details={'finish_reason': 'content_filter'},
        )

    model = FunctionModel(function=filtered_response, model_name='test-model')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered. Finish reason: 'content_filter'"):
        await agent.run('Trigger filter')


async def test_central_content_filter_with_partial_content():
    """
    Test that the agent graph returns partial content (does not raise exception)
    even if finish_reason='content_filter', provided parts are not empty.
    """

    async def filtered_response(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('Partially generated content...')], model_name='test-model', finish_reason='content_filter'
        )

    model = FunctionModel(function=filtered_response, model_name='test-model')
    agent = Agent(model)

    # Should NOT raise ContentFilterError
    result = await agent.run('Trigger filter')
    assert result.output == 'Partially generated content...'


async def test_agent_allows_none_output_empty_response():
    """Test that Agent(output_type=str | None) succeeds on empty response."""

    async def empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    model = FunctionModel(function=empty_model)
    agent = Agent(model, output_type=str | None)

    result = await agent.run('hello')
    assert result.output is None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=51),
                model_name='function:empty_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_agent_allows_none_output_after_tool():
    """Test that Agent(output_type=str | None) succeeds after tool call with no final text."""
    call_count = 0

    async def tool_then_empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='noop', args={}, tool_call_id='123')])
        return ModelResponse(parts=[])

    model = FunctionModel(function=tool_then_empty_model)
    agent = Agent(model, output_type=str | None)

    @agent.tool_plain
    def noop() -> str:
        return 'done'

    result = await agent.run('hello')
    assert result.output is None
    assert call_count == 2
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='noop', args={}, tool_call_id='123')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='function:tool_then_empty_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='noop',
                        content='done',
                        tool_call_id='123',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:tool_then_empty_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_agent_allows_none_output_validator_called():
    """Test that output validators are called when returning None on empty response."""

    async def empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    validator_called = False
    model = FunctionModel(function=empty_model)
    agent = Agent(model, output_type=str | None)

    @agent.output_validator
    async def validate_output(ctx: RunContext[None], output: str | None) -> str | None:
        nonlocal validator_called
        validator_called = True
        assert output is None
        return output

    result = await agent.run('hello')
    assert result.output is None
    assert validator_called
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=51),
                model_name='function:empty_model:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_agent_allows_none_output_validator_retry():
    """Test that output validator raising ModelRetry triggers a retry when output is None."""

    async def model_then_text(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[])
        return ModelResponse(parts=[TextPart(content='hello')])

    model = FunctionModel(function=model_then_text)
    agent = Agent(model, output_type=str | None)

    @agent.output_validator
    async def reject_none(ctx: RunContext[None], output: str | None) -> str | None:
        if output is None:
            raise ModelRetry('None not acceptable, please respond')
        return output

    result = await agent.run('hello')
    assert result.output == 'hello'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=51),
                model_name='function:model_then_text:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='None not acceptable, please respond',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=RequestUsage(input_tokens=65, output_tokens=1),
                model_name='function:model_then_text:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_agent_still_fails_if_none_not_allowed():
    """Test that Agent(output_type=str) still fails on empty response."""

    async def empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    model = FunctionModel(function=empty_model)
    agent = Agent(model, output_type=str)

    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
        await agent.run('hello')


def test_agent_output_type_bare_none_error():
    """Test that Agent(output_type=None) raises a clear error."""
    with pytest.raises(UserError, match='At least one output type must be provided other than `None`'):
        Agent('test', output_type=None)  # type: ignore[arg-type]


async def test_agent_allows_none_output_tool_mode_none_via_tool():
    """Test that `int | None` exposes a separate `final_result_NoneType` tool the model can call."""
    seen_tool_names: list[str] = []

    async def call_none_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        seen_tool_names[:] = [t.name for t in info.output_tools]
        none_tool = next(t for t in info.output_tools if 'NoneType' in t.name)
        return ModelResponse(
            parts=[ToolCallPart(tool_name=none_tool.name, args={'response': None}, tool_call_id='pyd_ai_id')]
        )

    agent = Agent(FunctionModel(function=call_none_tool), output_type=int | None)
    result = await agent.run('hello')
    assert result.output is None
    assert seen_tool_names == snapshot(['final_result_int', 'final_result_NoneType'])


async def test_agent_allows_none_output_tool_mode_int_via_tool():
    """Test that `int | None` still resolves the int branch via the int tool."""

    async def call_int_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        int_tool = next(t for t in info.output_tools if t.name == 'final_result_int')
        return ModelResponse(
            parts=[ToolCallPart(tool_name=int_tool.name, args={'response': 42}, tool_call_id='pyd_ai_id')]
        )

    agent = Agent(FunctionModel(function=call_int_tool), output_type=int | None)
    result = await agent.run('hello')
    assert result.output == 42


async def test_agent_allows_none_output_tool_mode_empty_response():
    """Test that `int | None` still falls back to `None` on an empty response."""

    async def empty(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    agent = Agent(FunctionModel(function=empty), output_type=int | None)
    result = await agent.run('hello')
    assert result.output is None


async def test_agent_allows_none_output_native_structured_none():
    """Test that `NativeOutput(int | None)` returns `None` when the model emits the NoneType branch."""

    async def native_none(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=json.dumps({'result': {'kind': 'NoneType', 'data': {'response': None}}}))]
        )

    agent = Agent(FunctionModel(function=native_none), output_type=NativeOutput([int, type(None)]))
    result = await agent.run('hello')
    assert result.output is None


async def test_agent_allows_none_output_prompted_structured_none():
    """Test that `PromptedOutput(int | None)` returns `None` when the model emits the NoneType branch."""

    async def prompted_none(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content=json.dumps({'result': {'kind': 'NoneType', 'data': {'response': None}}}))]
        )

    agent = Agent(FunctionModel(function=prompted_none), output_type=PromptedOutput([int, type(None)]))
    result = await agent.run('hello')
    assert result.output is None


async def test_agent_allows_none_output_tool_output_union_null():
    """Test that `ToolOutput(int | None)` accepts a JSON `null` via the tool's `anyOf` schema."""

    async def call_final_result(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result', args={'response': None}, tool_call_id='pyd_ai_id')]
        )

    agent = Agent(FunctionModel(function=call_final_result), output_type=ToolOutput(int | None))  # type: ignore[arg-type]
    result = await agent.run('hello')
    assert result.output is None


async def test_agent_allows_none_output_explicit_none_tool():
    """Test that `[ToolOutput(int), ToolOutput(type(None))]` surfaces one tool per type."""

    async def call_none_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(tool_name='final_result_NoneType', args={'response': None}, tool_call_id='pyd_ai_id')]
        )

    agent = Agent(
        FunctionModel(function=call_none_tool),
        output_type=[ToolOutput(int), ToolOutput(type(None))],
    )
    result = await agent.run('hello')
    assert result.output is None


# region Dynamic model_settings


def _text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('response')])


def _model_with_settings(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
    temperature = info.model_settings.get('temperature') if info.model_settings else None
    return ModelResponse(parts=[TextPart(f'max_tokens={max_tokens} temperature={temperature}')])


class TestCallableAgentLevelSettings:
    """Test agent-level callable model_settings."""

    def test_callable_agent_settings(self):
        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=200)

        agent = Agent(FunctionModel(_model_with_settings), model_settings=dynamic_settings)
        result = agent.run_sync('Hello')
        assert result.output == 'max_tokens=200 temperature=None'

    def test_callable_receives_run_context(self):
        contexts: list[RunContext[str]] = []

        def dynamic_settings(ctx: RunContext[str]) -> ModelSettings:
            contexts.append(ctx)
            return ModelSettings(max_tokens=50)

        agent = Agent(FunctionModel(_text_model), deps_type=str, model_settings=dynamic_settings)
        agent.run_sync('Hello', deps='test-deps')

        assert len(contexts) >= 1
        assert contexts[0].deps == 'test-deps'

    def test_callable_sees_model_settings_from_model(self):
        """The callable should see `ctx.model_settings` set to the model's base settings."""
        seen_settings: list[ModelSettings | None] = []

        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            seen_settings.append(ctx.model_settings)
            return ModelSettings(max_tokens=100)

        # FunctionModel has no settings (None), so ctx.model_settings should be None
        agent = Agent(FunctionModel(_text_model), model_settings=dynamic_settings)
        agent.run_sync('Hello')

        assert len(seen_settings) >= 1
        assert seen_settings[0] is None


class TestCallableRunLevelSettings:
    """Test run-level callable model_settings."""

    def test_callable_run_settings(self):
        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.9)

        agent = Agent(FunctionModel(_model_with_settings))
        result = agent.run_sync('Hello', model_settings=dynamic_settings)
        assert result.output == 'max_tokens=None temperature=0.9'

    def test_callable_run_sees_merged_agent_settings(self):
        """Run-level callable should see merged model+agent settings via ctx.model_settings."""
        seen_settings: list[ModelSettings | None] = []

        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            seen_settings.append(ctx.model_settings)
            return ModelSettings(temperature=0.5)

        agent = Agent(FunctionModel(_text_model), model_settings=ModelSettings(max_tokens=100))
        agent.run_sync('Hello', model_settings=run_settings)

        assert len(seen_settings) >= 1
        assert seen_settings[0] is not None
        assert seen_settings[0].get('max_tokens') == 100


class TestMixedStaticAndCallableSettings:
    """Test mixing static and callable model_settings."""

    def test_static_agent_callable_run(self):
        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.8)

        agent = Agent(
            FunctionModel(_model_with_settings),
            model_settings=ModelSettings(max_tokens=100),
        )
        result = agent.run_sync('Hello', model_settings=run_settings)
        assert result.output == 'max_tokens=100 temperature=0.8'

    def test_callable_agent_static_run(self):
        def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=150)

        agent = Agent(FunctionModel(_model_with_settings), model_settings=agent_settings)
        result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.6))
        assert result.output == 'max_tokens=150 temperature=0.6'

    def test_callable_agent_callable_run(self):
        def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=200)

        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.4)

        agent = Agent(FunctionModel(_model_with_settings), model_settings=agent_settings)
        result = agent.run_sync('Hello', model_settings=run_settings)
        assert result.output == 'max_tokens=200 temperature=0.4'


class TestPerStepSettingsResolution:
    """Test that callable model_settings is called before each model request."""

    def test_called_each_step(self):
        call_count = 0
        step_numbers: list[int] = []

        def dynamic_settings(ctx: RunContext[None]) -> ModelSettings:
            nonlocal call_count
            call_count += 1
            step_numbers.append(ctx.run_step)
            return ModelSettings(max_tokens=100)

        def multi_step_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[ToolCallPart('my_tool', args='{"x": 1}')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(multi_step_model), model_settings=dynamic_settings)

        @agent.tool_plain
        def my_tool(x: int) -> str:
            return f'result-{x}'

        agent.run_sync('Hello')

        assert call_count >= 2
        assert step_numbers == sorted(step_numbers)

    def test_step_dependent_settings(self):
        """Settings can vary based on run_step."""

        def step_dependent_settings(ctx: RunContext[None]) -> ModelSettings:
            if ctx.run_step <= 1:
                return ModelSettings(max_tokens=100)
            return ModelSettings(max_tokens=500)

        settings_per_step: list[tuple[int, int | None]] = []

        def tracking_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
            step = len(messages)
            settings_per_step.append((step, max_tokens))
            if step == 1:
                return ModelResponse(parts=[ToolCallPart('my_tool', args='{"x": 1}')])
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(tracking_model), model_settings=step_dependent_settings)

        @agent.tool_plain
        def my_tool(x: int) -> str:
            return f'result-{x}'

        agent.run_sync('Hello')

        assert settings_per_step[0][1] == 100
        assert settings_per_step[1][1] == 500


class TestDynamicSettingsPrecedence:
    """Test that run > agent > model precedence is maintained with callables."""

    def test_callable_run_overrides_callable_agent(self):
        def agent_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.2)

        def run_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(temperature=0.8)

        agent = Agent(FunctionModel(_model_with_settings), model_settings=agent_settings)
        result = agent.run_sync('Hello', model_settings=run_settings)
        assert result.output == 'max_tokens=None temperature=0.8'


class TestOverrideWithModelSettings:
    """Test the override() context manager with model_settings."""

    def test_override_with_static(self):
        agent = Agent(FunctionModel(_model_with_settings))

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=42 temperature=None'

    def test_override_with_callable(self):
        def override_settings(ctx: RunContext[None]) -> ModelSettings:
            return ModelSettings(max_tokens=99)

        agent = Agent(FunctionModel(_model_with_settings))

        with agent.override(model_settings=override_settings):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=99 temperature=None'

    def test_override_replaces_agent_settings(self):
        """Override model_settings should replace agent-level settings."""
        agent = Agent(
            FunctionModel(_model_with_settings),
            model_settings=ModelSettings(max_tokens=100, temperature=0.5),
        )

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=42 temperature=None'

    def test_override_ignores_run_settings(self):
        """When override is set, run-level model_settings should be ignored."""
        agent = Agent(FunctionModel(_model_with_settings))

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello', model_settings=ModelSettings(temperature=0.9))
            assert result.output == 'max_tokens=42 temperature=None'

    def test_override_resets_after_context(self):
        """After exiting override context, original settings should be restored."""
        agent = Agent(
            FunctionModel(_model_with_settings),
            model_settings=ModelSettings(max_tokens=100),
        )

        with agent.override(model_settings=ModelSettings(max_tokens=42)):
            result = agent.run_sync('Hello')
            assert result.output == 'max_tokens=42 temperature=None'

        result = agent.run_sync('Hello')
        assert result.output == 'max_tokens=100 temperature=None'


def test_output_validator_retry_consistency_across_paths():
    """Output validators see global retry info, matching the text-output path.

    Regression test for https://github.com/pydantic/pydantic-ai/issues/4385:
    the text path sets ctx.retry/max_retries to the global output retry counter,
    but the tool-output path was using the per-tool counter, causing inconsistent
    ctx.retry and ctx.max_retries values in @agent.output_validator.

    Using ToolOutput(max_retries=5) with output_retries=2 exposes the bug:
    without the fix, the validator would see max_retries=5 (per-tool value)
    instead of max_retries=2 (global output_retries, matching the text path).
    """
    retries_log: list[int] = []
    max_retries_log: list[int] = []

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(return_model),
        output_type=ToolOutput(Foo, max_retries=5),
        output_retries=2,
    )

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        retries_log.append(ctx.retry)
        max_retries_log.append(ctx.max_retries)
        if ctx.retry == 2:
            return o
        raise ModelRetry(f'Retry {ctx.retry}')

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)

    assert retries_log == [0, 1, 2]
    assert max_retries_log == [2, 2, 2]


def test_output_validator_exceeds_output_retries():
    """Output validator that never succeeds should respect output_retries limit.

    When the output_validator always raises ModelRetry, the agent should stop
    after output_retries attempts. The per-tool limit from ToolManager enforces
    this via output_retries flowing into ToolsetTool.max_retries.
    """
    retries_log: list[int] = []

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(return_model),
        output_type=ToolOutput(Foo),
        output_retries=2,
    )

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        retries_log.append(ctx.retry)
        raise ModelRetry(f'Retry {ctx.retry}')

    with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries \\(2\\)'):
        agent.run_sync('Hello')

    assert retries_log == [0, 1, 2]


async def test_concurrent_runs_output_retry_isolation():
    """Concurrent runs on the same agent must not share output retry state.

    OutputToolset.for_run_step returns a shallow copy so each run gets
    isolated _output_retry_count values across await points.
    """
    retries_by_run: dict[str, list[int]] = {'fast': [], 'slow': []}

    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        assert info.output_tools is not None
        args_json = '{"a": 1, "b": "foo"}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    agent = Agent(
        FunctionModel(return_model),
        output_type=ToolOutput(Foo),
        output_retries=3,
    )

    @agent.output_validator
    async def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        # Use the prompt to identify which run this is
        run_id = 'slow' if 'slow' in (ctx.prompt or '') else 'fast'
        retries_by_run[run_id].append(ctx.retry)
        if ctx.retry < 2:
            if run_id == 'slow':
                await asyncio.sleep(0.05)
            raise ModelRetry(f'{run_id} retry {ctx.retry}')
        return o

    result_fast, result_slow = await asyncio.gather(
        agent.run('fast'),
        agent.run('slow'),
    )

    assert isinstance(result_fast.output, Foo)
    assert isinstance(result_slow.output, Foo)
    assert retries_by_run['fast'] == [0, 1, 2]
    assert retries_by_run['slow'] == [0, 1, 2]


def test_output_validator_retry_counter_with_tool_switch():
    """Global retry counter tracks across output tool switches.

    When the model switches from one output tool to another, the global
    retry counter (visible to output validators via ctx.retry) keeps
    incrementing. Each tool's per-tool counter is independent.
    """
    validator_retries: list[int] = []
    validator_max_retries: list[int] = []

    def output_a(value: str) -> str:
        return value

    def output_b(value: str) -> str:
        return value

    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        assert info.output_tools is not None
        tool_names: dict[str, str] = {}
        tool_names.update({t.name: t.name for t in info.output_tools})

        name_a = next(n for n in tool_names if 'output_a' in n)
        name_b = next(n for n in tool_names if 'output_b' in n)

        call_count += 1
        # First call: output_b (will fail validation), second+: output_a
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(name_b, '{"value": "x"}')])
        return ModelResponse(parts=[ToolCallPart(name_a, '{"value": "hello"}')])

    agent = Agent(
        FunctionModel(return_model),
        output_type=[
            ToolOutput(output_a, max_retries=3),
            ToolOutput(output_b, max_retries=1),
        ],
        tool_retries=0,
        output_retries=0,
    )

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: str) -> str:
        validator_retries.append(ctx.retry)
        validator_max_retries.append(ctx.max_retries)
        if ctx.retry < 2:
            raise ModelRetry(f'Retry {ctx.retry}')
        return o

    result = agent.run_sync('Hello')
    assert result.output == 'hello'

    # Global retry counter increments across tool switches
    assert validator_retries == [0, 1, 2]
    # max_retries reflects the agent-level default (0) since output_retries not set
    assert validator_max_retries == [0, 0, 0]


def test_output_tool_validation_vs_execution_retry_counting():
    """Both validation failures and execution failures increment the global retry counter.

    Validation failure (bad args from model) and execution failure (ModelRetry from
    output function) both go through process_tool_calls and should both increment
    ctx.state.output_retries_used for output validator context tracking.
    """
    validator_retries: list[int] = []
    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        assert info.output_tools is not None
        call_count += 1
        tool_name = info.output_tools[0].name
        # First call: send invalid args (validation failure path)
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, '{"bad_field": 1}')])
        # Subsequent calls: send valid args (execution path)
        return ModelResponse(parts=[ToolCallPart(tool_name, '{"a": 1, "b": "foo"}')])

    agent = Agent(
        FunctionModel(return_model),
        output_type=ToolOutput(Foo),
        output_retries=5,
    )

    @agent.output_validator
    def validate_output(ctx: RunContext[None], o: Foo) -> Foo:
        validator_retries.append(ctx.retry)
        if ctx.retry < 2:
            raise ModelRetry(f'Retry {ctx.retry}')
        return o

    result = agent.run_sync('Hello')
    assert isinstance(result.output, Foo)

    # retry 0: never reached validator (validation failure from bad args)
    # retry 1: reached validator, raised ModelRetry (execution failure)
    # retry 2: reached validator, succeeded
    assert validator_retries == [1, 2]


@dataclass(kw_only=True)
class OutputRetryBudgetCase:
    """One row in the output-retry budget precedence matrix.

    `init`/`override`/`run` flow into `Agent(...)`, `agent.override(...)`, and
    `agent.run_sync(...)` respectively. The test asserts the effective budget
    by exhausting it with an always-retrying output_validator and confirming
    the validator saw exactly `range(expected_budget + 1)` invocations.

    All cases run on the tool-output path; text-path parity has its own test
    (`test_text_path_honors_output_retry_budget`) below.
    """

    id: str
    init: dict[str, Any]
    override: dict[str, Any] | None
    run: dict[str, Any]
    expected_budget: int
    expects_deprecation_warning: bool


OUTPUT_RETRY_BUDGET_CASES = [
    OutputRetryBudgetCase(
        id='run-arg-caps-agent-default',
        init={'output_retries': 5},
        override=None,
        run={'output_retries': 2},
        expected_budget=2,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='run-arg-beats-spec',
        init={'output_retries': 5},
        override=None,
        run={'output_retries': 2, 'spec': {'output_retries': 4}},
        expected_budget=2,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='spec-only-beats-agent-default',
        init={'output_retries': 5},
        override=None,
        run={'spec': {'output_retries': 2}},
        expected_budget=2,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='override-kwarg-caps-agent-default',
        init={'output_retries': 5},
        override={'output_retries': 2},
        run={},
        expected_budget=2,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='override-spec-honored',
        init={'output_retries': 5},
        override={'spec': {'output_retries': 2}},
        run={},
        expected_budget=2,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='override-beats-run-arg',
        init={'output_retries': 5},
        override={'output_retries': 1},
        run={'output_retries': 10},
        expected_budget=1,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='deprecated-retries-cascades',
        init={'retries': 5},
        override=None,
        run={},
        expected_budget=5,
        expects_deprecation_warning=True,
    ),
    OutputRetryBudgetCase(
        id='tool-retries-no-cascade',
        init={'tool_retries': 5},
        override=None,
        run={},
        expected_budget=1,
        expects_deprecation_warning=False,
    ),
    OutputRetryBudgetCase(
        id='retries-cascades-even-when-tool-retries-set',
        init={'retries': 5, 'tool_retries': 3},
        override=None,
        run={},
        expected_budget=5,
        expects_deprecation_warning=True,
    ),
]


@pytest.mark.parametrize('case', OUTPUT_RETRY_BUDGET_CASES, ids=lambda c: c.id)
def test_output_retry_budget_resolution(case: OutputRetryBudgetCase):
    """Effective output-retry budget = override > run arg > spec > agent default.

    Exhausts the resolved budget with an always-retrying output_validator and asserts the
    validator saw `range(expected_budget + 1)` invocations and the exhaustion error reports
    the expected budget.
    """
    retries_log: list[int] = []

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"a": 1, "b": "foo"}')])

    deprecation_ctx = (
        pytest.warns(DeprecationWarning, match=r'`retries` is deprecated and will be removed in v2')
        if case.expects_deprecation_warning
        else nullcontext()
    )
    with warnings.catch_warnings():
        if not case.expects_deprecation_warning:
            warnings.simplefilter('error', DeprecationWarning)
        with deprecation_ctx:
            agent = Agent(FunctionModel(return_model), output_type=ToolOutput(Foo), **case.init)

    @agent.output_validator
    def always_retry(ctx: RunContext[None], o: Foo) -> Foo:
        retries_log.append(ctx.retry)
        raise ModelRetry(f'retry {ctx.retry}')

    expected_msg = rf'Exceeded maximum output retries \({case.expected_budget}\)'
    override_ctx = agent.override(**case.override) if case.override is not None else nullcontext()

    with override_ctx:
        with pytest.raises(UnexpectedModelBehavior, match=expected_msg):
            agent.run_sync('Hello', **case.run)

    assert retries_log == list(range(case.expected_budget + 1))


def test_text_path_honors_output_retry_budget():
    """Parity check: the text-output path enforces the run-level `output_retries` override too.

    The matrix above runs on the tool-output path; this test verifies the same precedence applies
    when the model returns a `TextPart` and the agent's `output_type` is `str`.
    """
    retries_log: list[int] = []

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('hello')])

    agent = Agent(FunctionModel(return_model), output_type=str, output_retries=5)

    @agent.output_validator
    def always_retry(ctx: RunContext[None], o: str) -> str:
        retries_log.append(ctx.retry)
        raise ModelRetry(f'retry {ctx.retry}')

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(2\)'):
        agent.run_sync('Hello', output_retries=2)

    assert retries_log == [0, 1, 2]


def test_output_retries_run_override_without_validators():
    """Run-level `output_retries` takes effect — and isolates from the shared agent-level toolset —
    even when no output_validator is registered.

    Exercises the shared-toolset clone branch: with `output_schema == self._output_schema` and no
    output_validators registered, `Agent.iter` must clone the shared `self._output_toolset` before
    mutating `max_retries` so the agent-level toolset retains its original budget for subsequent runs.
    Enforcement runs through `ToolManager._check_max_retries` (per-tool), not `consume_output_retry`.
    """
    call_count = 0

    def return_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        assert info.output_tools is not None
        # Always return invalid args → output function raises ModelRetry via validation path
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"bad_field": 1}')])

    agent = Agent(FunctionModel(return_model), output_type=ToolOutput(Foo), output_retries=10)
    shared_toolset = agent._output_toolset  # pyright: ignore[reportPrivateUsage]
    assert shared_toolset is not None
    assert shared_toolset.max_retries == 10

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum output retries \(2\)'):
        agent.run_sync('Hello', output_retries=2)

    # 3 attempts: initial + 2 retries, capped by the run override at 2 (not the agent default 10)
    assert call_count == 3
    # The shared agent-level toolset's `max_retries` was preserved — confirms the run cloned before mutating.
    assert shared_toolset.max_retries == 10


@pytest.mark.parametrize(
    'kwargs, expected_warning',
    [
        ({}, None),
        ({'tool_retries': 5}, None),
        ({'output_retries': 5}, None),
        ({'tool_retries': 5, 'output_retries': 5}, None),
        ({'retries': 5}, r'`retries` is deprecated.*Use `tool_retries` instead'),
        ({'retries': 5, 'output_retries': 3}, r'`retries` is deprecated.*Use `tool_retries` instead'),
        (
            {'retries': 5, 'tool_retries': 3},
            r'`retries` is deprecated.*only setting `output_retries` via the legacy 1\.x cascade',
        ),
    ],
    ids=[
        'no-kwargs',
        'tool_retries-only',
        'output_retries-only',
        'tool_retries-and-output_retries',
        'retries-only-warns',
        'retries-and-output_retries-warns-generic',
        'retries-and-tool_retries-warns-cascade-callout',
    ],
)
def test_agent_init_deprecation_warning(kwargs: dict[str, Any], expected_warning: str | None):
    """Cover the `Agent.__init__` deprecation warning text for every kwarg combination.

    The both-passed case (`retries=` + `tool_retries=`) gets a sharper message that calls out the
    silent cascade to `output_retries`; everything else either warns generically or stays silent.
    """
    if expected_warning is None:
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            Agent('test', **kwargs)
    else:
        with pytest.warns(DeprecationWarning, match=expected_warning):
            Agent('test', **kwargs)


def test_unknown_tool_with_valid_tool_does_not_exhaust_retries():
    """Unknown tool calls should not increment the global retry counter.

    When the model returns both an unknown tool and a valid tool in the same
    response, the unknown tool is handled via per-tool retries (ModelRetry)
    downstream. The global retry counter should only reflect output validation
    retries, not unknown-tool retries, so valid tools keep working.

    We set retries=2 (per-tool max for unknown tools) and output_retries=1
    (global max for output validation) to isolate the bug: per-tool retries
    allow 2 rounds of the unknown tool, but the old code's global increment
    would exhaust output_retries after just 2 rounds.
    """
    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return ModelResponse(
                parts=[
                    ToolCallPart('nonexistent_tool', '{}'),
                    ToolCallPart('valid_tool', '{"x": 1}'),
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_function), tool_retries=2, output_retries=1)

    @agent.tool_plain
    def valid_tool(x: int) -> str:
        return f'result={x}'

    result = agent.run_sync('Hello')
    assert result.output == 'done'
    assert call_count == 3


# endregion


# region Image output validators


async def test_image_output_validators_run():
    """Test that output validators are called and can modify image output."""

    def return_image(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'original', media_type='image/png'))])

    image_profile = ModelProfile(supports_image_output=True)
    agent = Agent(FunctionModel(return_image, profile=image_profile), output_type=BinaryImage)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: BinaryImage) -> BinaryImage:
        return BinaryImage(data=b'modified', media_type=output.media_type)

    result = await agent.run('Give me an image')
    assert isinstance(result.output, BinaryImage)
    assert result.output.data == b'modified', 'validator should be able to modify the output'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Give me an image', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[FilePart(content=BinaryImage(data=b'original', media_type='image/png'))],
                usage=RequestUsage(input_tokens=54, output_tokens=8),
                model_name='function:return_image:',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_image_output_validator_model_retry():
    """Test that ModelRetry from image validator raises UnexpectedModelBehavior in streaming."""

    class ImageStreamedResponse(StreamedResponse):
        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            self._usage = RequestUsage()
            yield self._parts_manager.handle_part(
                vendor_part_id=0,
                part=FilePart(content=BinaryImage(data=b'fake-png', media_type='image/png')),
            )

        @property
        def model_name(self) -> str:
            return 'image-model'

        @property
        def provider_name(self) -> str:
            return 'test'

        @property
        def provider_url(self) -> str:
            return 'https://test.example.com'

        @property
        def timestamp(self) -> datetime:
            return datetime(2024, 1, 1)

    class ImageStreamModel(Model):
        @property
        def system(self) -> str:  # pragma: no cover
            return 'test'

        @property
        def model_name(self) -> str:
            return 'image-model'

        @property
        def base_url(self) -> str:  # pragma: no cover
            return 'https://test.example.com'

        async def request(  # pragma: no cover
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'fake-png', media_type='image/png'))])

        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext | None = None,
        ) -> AsyncIterator[StreamedResponse]:
            yield ImageStreamedResponse(model_request_parameters=model_request_parameters)

    image_profile = ModelProfile(supports_image_output=True)
    agent = Agent(ImageStreamModel(profile=image_profile), output_type=BinaryImage)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: BinaryImage) -> BinaryImage:
        raise ModelRetry('image validation failed')

    with pytest.raises(
        UnexpectedModelBehavior,
        match='Output validation failed during streaming, and retries are not supported',
    ):
        async with agent.run_stream('Give me an image') as stream:
            await stream.get_output()


async def test_image_output_validators_run_stream():
    """Test that output validators are called and can modify image output during streaming."""

    class ImageStreamedResponse(StreamedResponse):
        async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
            self._usage = RequestUsage()
            yield self._parts_manager.handle_part(
                vendor_part_id=0,
                part=FilePart(content=BinaryImage(data=b'original', media_type='image/png')),
            )

        @property
        def model_name(self) -> str:
            return 'image-model'

        @property
        def provider_name(self) -> str:
            return 'test'

        @property
        def provider_url(self) -> str:
            return 'https://test.example.com'

        @property
        def timestamp(self) -> datetime:
            return datetime(2024, 1, 1)

    class ImageStreamModel(Model):
        @property
        def system(self) -> str:  # pragma: no cover
            return 'test'

        @property
        def model_name(self) -> str:
            return 'image-model'

        @property
        def base_url(self) -> str:  # pragma: no cover
            return 'https://test.example.com'

        async def request(  # pragma: no cover
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'original', media_type='image/png'))])

        @asynccontextmanager
        async def request_stream(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
            run_context: RunContext | None = None,
        ) -> AsyncIterator[StreamedResponse]:
            yield ImageStreamedResponse(model_request_parameters=model_request_parameters)

    image_profile = ModelProfile(supports_image_output=True)
    agent = Agent(ImageStreamModel(profile=image_profile), output_type=BinaryImage)

    @agent.output_validator
    def validate_output(ctx: RunContext[None], output: BinaryImage) -> BinaryImage:
        return BinaryImage(data=b'modified', media_type=output.media_type)

    async with agent.run_stream('Give me an image') as stream:
        result = await stream.get_output()

    assert isinstance(result, BinaryImage)
    assert result.data == b'modified', 'validator should be able to modify the output'
    assert stream.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Give me an image', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[FilePart(content=BinaryImage(data=b'original', media_type='image/png'))],
                model_name='image-model',
                timestamp=IsDatetime(),
                provider_name='test',
                provider_url='https://test.example.com',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


# endregion
