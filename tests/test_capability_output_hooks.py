"""Tests for capability output validation and processing hooks.

Split out of `test_capabilities.py` per #7304.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel, ValidationError

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    HandleDeferredToolCalls,
    ToolSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    UserError,
)
from pydantic_ai.messages import (
    BinaryImage,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import NativeOutput, OutputContext, PromptedOutput, TextOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .capability_models import (
    MyOutput,
    make_text_response,
)
from .conftest import IsDatetime, IsStr, iter_message_parts

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


# --- Output hook tests ---


class TestBeforeOutputValidate:
    """before_output_validate can transform raw output before parsing."""

    async def test_structured_prompted_output(self):
        """before_output_validate transforms raw text before Pydantic validation for PromptedOutput."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "not_a_number"}')])

        @dataclass
        class FixJsonCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                if isinstance(output, str):
                    return output.replace('"not_a_number"', '42')
                return output  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[FixJsonCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)

    async def test_plain_str_output(self):
        """For plain str output, validate hooks are skipped; process hooks fire instead."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello world')

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                # The uncovered body is the assertion: this hook must not fire for plain text.
                log.append('validate')  # pragma: no cover
                return output  # pragma: no cover

            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append(f'process:{output}')
                assert output_context.mode == 'text'
                assert output_context.output_type is str
                assert output_context.has_function is False
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == 'hello world'
        # Validate hooks do NOT fire for plain text; only process hooks fire
        assert log == ['process:hello world']

    async def test_text_output_function(self):
        """For TextOutput, validate hooks are skipped; process hooks fire and call the function."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append(f'before:{output}')
                assert output_context.has_function is True
                return output

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == 'WORLD'
        assert log == ['before:world']

    async def test_can_transform_text_before_function(self):
        """before_output_process can modify text before the TextOutput function runs."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class PrependCap(AbstractCapability[Any]):
            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                assert isinstance(output, str)
                return f'hello {output}'

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[PrependCap()])
        result = await agent.run('greet')
        assert result.output == 'HELLO WORLD'


class TestOnOutputValidateError:
    """on_output_validate_error can recover from validation errors."""

    async def test_recover_from_invalid_json(self):
        """on_output_validate_error can fix raw output and return corrected data."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])

        @dataclass
        class RecoverCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                # Recovery replaces the validation result; for structured output
                # the execute step (call()) returns this as-is when there's no function.
                return {'value': 99}

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RecoverCap()])
        result = await agent.run('hello')
        # The error hook bypasses Pydantic validation, so the output is the raw dict
        assert result.output == {'value': 99}
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": "bad"}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_default_reraises(self):
        """Without an error hook, validation errors propagate normally as retries."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput))
        result = await agent.run('hello')
        # Model retries and eventually gets it right
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": "bad"}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'int_parsing',
                                    'loc': ('value',),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'bad',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=87, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestOnOutputValidateErrorModelRetry:
    """on_output_validate_error can raise ModelRetry to trigger a retry with a custom message."""

    async def test_error_hook_raises_model_retry(self):
        """on_output_validate_error raises ModelRetry, which becomes a retry prompt."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class RetryHookCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                raise ModelRetry('Please return a valid integer for value')

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RetryHookCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": "bad"}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Please return a valid integer for value',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestModelRetryFromOutputHooks:
    """Hooks can raise ModelRetry to trigger a model retry."""

    async def test_before_output_validate_raises_model_retry(self):
        """before_output_validate can raise ModelRetry to skip validation and retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class RejectNegativeCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                if isinstance(output, str) and '-1' in output:
                    raise ModelRetry('Negative values are not allowed')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectNegativeCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": -1}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Negative values are not allowed',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=65, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_output_validate_raises_model_retry(self):
        """after_output_validate can raise ModelRetry to reject validated output."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": 0}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class RejectZeroCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                # Validated output is a MyOutput instance (Pydantic returns model instances)
                if isinstance(output, MyOutput) and output.value == 0:
                    raise ModelRetry('Zero is not a valid value')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectZeroCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 0}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Zero is not a valid value',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=66, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_output_process_raises_model_retry(self):
        """after_output_process can raise ModelRetry to reject the execution result."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='short')])
            return ModelResponse(parts=[TextPart(content='this is long enough')])

        @dataclass
        class MinLengthCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, str) and len(output) < 10:
                    raise ModelRetry('Output too short, please elaborate')
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[MinLengthCap()])
        result = await agent.run('hello')
        assert result.output == 'this is long enough'
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='short')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Output too short, please elaborate',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='this is long enough')],
                    usage=RequestUsage(input_tokens=65, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_output_process_model_retry_skips_error_hook(self):
        """ModelRetry from wrap_output_process bypasses on_output_process_error."""
        error_hook_called = False
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='bad')])
            return ModelResponse(parts=[TextPart(content='good')])

        @dataclass
        class WrapRetryCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, handler: Any
            ) -> Any:
                result = await handler(output)
                if result == 'bad':
                    raise ModelRetry('Bad output, please try again')
                return result

            # The uncovered body is the assertion: this hook must not be called.
            async def on_output_process_error(  # pragma: no cover
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, error: Exception
            ) -> Any:
                nonlocal error_hook_called
                error_hook_called = True
                raise error

        agent = Agent(FunctionModel(model_fn), capabilities=[WrapRetryCap()])
        result = await agent.run('hello')
        assert result.output == 'good'
        assert call_count == 2
        assert not error_hook_called  # ModelRetry skips on_output_process_error
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='bad')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Bad output, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good')],
                    usage=RequestUsage(input_tokens=65, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_output_process_raises_model_retry(self):
        """before_output_process can raise ModelRetry to skip execution."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": 0}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class RejectBeforeExecCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value == 0:
                    raise ModelRetry('Cannot execute with zero value')
                return output

        agent = Agent(
            FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectBeforeExecCap()]
        )
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 0}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Cannot execute with zero value',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 5}')],
                    usage=RequestUsage(input_tokens=65, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_before_validate_raises_model_retry(self):
        """ModelRetry from before_output_validate on a tool output includes tool_call_id."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": -1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class RejectNegativeCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                if (
                    isinstance(output, str)
                    and '-1' in output
                    or isinstance(output, dict)
                    and output.get('value', 0) < 0
                ):
                    raise ModelRetry('Negative values not allowed')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[RejectNegativeCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": -1}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Negative values not allowed',
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 42}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=62, output_tokens=8),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_after_execute_raises_model_retry(self):
        """ModelRetry from after_output_process on a tool output triggers retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": 0}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 10}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class RejectZeroCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value == 0:
                    raise ModelRetry('Zero not allowed')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[RejectZeroCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 0}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Zero not allowed',
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 10}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=61, output_tokens=8),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_validation_failure(self):
        """Invalid output tool args trigger retry through output validate hooks."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": "bad"}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput)
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": "bad"}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'int_parsing',
                                    'loc': ('value',),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'bad',
                                }
                            ],
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 42}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=89, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_error_hook_raises_model_retry(self):
        """on_output_validate_error raises ModelRetry for output tool, includes tool_call_id."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": "bad"}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class RetryOnErrorCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                raise ModelRetry('Please provide a valid integer')

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[RetryOnErrorCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": "bad"}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Please provide a valid integer',
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 42}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=63, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestOutputToolWithOutputFunction:
    """Output tools with output functions that raise ModelRetry."""

    async def test_output_function_model_retry(self):
        """An output function on a tool output type that raises ModelRetry triggers a retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 10}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        def my_output_fn(output: MyOutput) -> MyOutput:
            if output.value < 5:
                raise ModelRetry('Value must be >= 5')
            return output

        agent = Agent(FunctionModel(model_fn), output_type=my_output_fn)
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert call_count == 2

    async def test_output_function_model_retry_with_hooks(self):
        """Output function ModelRetry works correctly when output hooks are present."""
        log: list[str] = []
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 10}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        def my_output_fn(output: MyOutput) -> MyOutput:
            if output.value < 5:
                raise ModelRetry('Value must be >= 5')
            return output

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(f'execute:{output}')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=my_output_fn, capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert call_count == 2
        # Execute hook fires for both attempts (retry + success)
        assert len(log) == 2


class TestWrapOutputValidate:
    """wrap_output_validate provides full middleware control around validation."""

    async def test_wrap_can_observe(self):
        """wrap_output_validate can observe without modifying."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 10}')])

        @dataclass
        class WrapCap(AbstractCapability[Any]):
            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('before')
                result = await handler(output)
                log.append('after')
                return result

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[WrapCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert log == ['before', 'after']

    async def test_wrap_can_transform_input(self):
        """wrap_output_validate can transform the output before passing to handler."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "oops"}')])

        @dataclass
        class TransformCap(AbstractCapability[Any]):
            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                # Fix the input before validation
                fixed = '{"value": 7}' if isinstance(output, str) else output
                return await handler(fixed)

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[TransformCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=7)

    async def test_wrap_can_catch_and_recover(self):
        """wrap_output_validate can catch validation errors and return a fallback."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='not json at all')])

        @dataclass
        class RecoverWrapCap(AbstractCapability[Any]):
            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                try:
                    return await handler(output)
                except (ValidationError, ModelRetry):
                    return {'value': 0}

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RecoverWrapCap()])
        result = await agent.run('hello')
        # The wrap recovery bypasses Pydantic validation, so the output is the raw dict
        assert result.output == {'value': 0}


class TestAfterOutputProcess:
    """after_output_process can transform the final result after execution."""

    async def test_transform_structured_result(self):
        """after_output_process transforms the result of structured output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class DoubleResultCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                assert isinstance(output, MyOutput)
                return MyOutput(value=output.value * 2)

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[DoubleResultCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)

    async def test_transform_plain_text_result(self):
        """after_output_process can transform plain text output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class UpperCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return output.upper() if isinstance(output, str) else output

        agent = Agent(FunctionModel(model_fn), capabilities=[UpperCap()])
        result = await agent.run('hello')
        assert result.output == 'HELLO'

    async def test_transform_text_function_result(self):
        """after_output_process fires after TextOutput function has executed."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class WrapResultCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                # output is already 'WORLD' from upcase
                return f'[{output}]'

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[WrapResultCap()])
        result = await agent.run('hello')
        assert result.output == '[WORLD]'


class TestToolOutputWithOutputHooks:
    """Output hooks fire for tool-based output, nested inside tool hooks."""

    async def test_output_hooks_fire_for_tool_output(self):
        """Output hooks fire when the output type uses tool mode."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class OutputLogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'before_output_validate:{output_context.mode}')
                return output

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_output_validate')
                return output

            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('before_output_process')
                return output

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_output_process')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[OutputLogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert 'before_output_validate:tool' in log
        assert 'after_output_validate' in log
        assert 'before_output_process' in log
        assert 'after_output_process' in log

    async def test_output_hooks_fire_without_tool_hooks(self):
        """Output tools use output hooks only — tool hooks do NOT fire."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class BothHooksCap(AbstractCapability[Any]):
            # The uncovered body is the assertion: this hook must not be called.
            async def before_tool_validate(  # pragma: no cover
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'tool_validate:{call.tool_name}')
                return args

            # The uncovered body is the assertion: this hook must not be called.
            async def before_tool_execute(  # pragma: no cover
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                log.append(f'tool_execute:{call.tool_name}')
                return args

            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('output_validate')
                return output

            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('output_process')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[BothHooksCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        # Only output hooks fire for output tools — tool hooks are skipped
        assert 'tool_validate:final_result' not in log
        assert 'tool_execute:final_result' not in log
        assert 'output_validate' in log
        assert 'output_process' in log

    async def test_after_output_process_transforms_tool_output(self):
        """after_output_process can transform the result of tool-based output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 5}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class DoubleOutputCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                if isinstance(output, MyOutput):
                    return MyOutput(value=output.value * 2)
                return output  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[DoubleOutputCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)


class TestHookComposition:
    """Multiple capabilities with output hooks compose correctly."""

    async def test_multiple_before_output_validate(self):
        """Multiple capabilities' before_output_validate hooks chain in order."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('cap1')
                return output

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('cap2')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[Cap1(), Cap2()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=1)
        assert log == ['cap1', 'cap2']

    async def test_chained_transformations(self):
        """Multiple capabilities can chain transformations in before_output_validate."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class AddExclamation(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return f'{output}!' if isinstance(output, str) else output

        @dataclass
        class AddQuestion(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return f'{output}?' if isinstance(output, str) else output

        agent = Agent(FunctionModel(model_fn), capabilities=[AddExclamation(), AddQuestion()])
        result = await agent.run('hello')
        # after hooks run in reversed order: AddQuestion first, then AddExclamation
        assert result.output == 'hello?!'


class TestHooksClassOutputDecorators:
    """Test decorator registration for output hooks with Hooks class."""

    async def test_before_output_validate_decorator(self):
        """Hooks.on.before_output_validate registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_validate
        def fix_output(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_output_validate')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 3}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=3)
        assert log == ['before_output_validate']

    async def test_after_output_validate_decorator(self):
        """Hooks.on.after_output_validate registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.after_output_validate
        async def after_validate(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: Any,
        ) -> Any:
            log.append('after_output_validate')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 4}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=4)
        assert log == ['after_output_validate']

    async def test_wrap_output_validate_decorator(self):
        """Hooks.on.output_validate (wrap) registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.output_validate
        async def wrap_validate(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            handler: Any,
        ) -> Any:
            log.append('wrap_start')
            result = await handler(output)
            log.append('wrap_end')
            return result

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert log == ['wrap_start', 'wrap_end']

    async def test_on_output_validate_error_decorator(self):
        """Hooks.on.output_validate_error can recover from validation failures."""
        hooks = Hooks()

        @hooks.on.output_validate_error
        async def recover(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            error: ValidationError | ModelRetry,
        ) -> Any:
            return {'value': 999}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='not valid json')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        # Error recovery bypasses Pydantic validation, so the output is the raw dict
        assert result.output == {'value': 999}

    async def test_before_output_process_decorator(self):
        """Hooks.on.before_output_process registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_process
        async def before_exec(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_output_process')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 6}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=6)
        assert log == ['before_output_process']

    async def test_after_output_process_decorator(self):
        """Hooks.on.after_output_process transforms the final result."""
        hooks = Hooks()

        @hooks.on.after_output_process
        async def double_output(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: Any,
        ) -> Any:
            if isinstance(output, MyOutput):
                return MyOutput(value=output.value * 2)
            return output  # pragma: no cover

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 7}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=14)

    async def test_wrap_output_process_decorator(self):
        """Hooks.on.output_process (wrap) registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.output_process
        async def wrap_exec(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            handler: Any,
        ) -> Any:
            log.append('exec_start')
            result = await handler(output)
            log.append('exec_end')
            return result

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 8}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=8)
        assert log == ['exec_start', 'exec_end']

    async def test_sync_hook_auto_wrapping(self):
        """Sync output hook functions are auto-wrapped to async."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_process
        def sync_hook(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: Any,
        ) -> Any:
            log.append('sync_before')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'hello'
        assert log == ['sync_before']


class TestOutputHookFullLifecycle:
    """Test the full output hook lifecycle fires in the correct order."""

    async def test_full_validate_and_execute_order(self):
        """All output hooks fire in the expected order for structured text output."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        @dataclass
        class FullLifecycleCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_validate')
                return output

            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('wrap_validate:before')
                result = await handler(output)
                log.append('wrap_validate:after')
                return result

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_validate')
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_execute')
                return output

            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('wrap_execute:before')
                result = await handler(output)
                log.append('wrap_execute:after')
                return result

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_execute')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[FullLifecycleCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=1)
        assert log == [
            'before_validate',
            'wrap_validate:before',
            'wrap_validate:after',
            'after_validate',
            'before_execute',
            'wrap_execute:before',
            'wrap_execute:after',
            'after_execute',
        ]
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 1}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_full_lifecycle_with_tool_output(self):
        """All output hooks fire in order for tool-based output."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 100}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class FullLifecycleCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_validate')
                assert output_context.mode == 'tool'
                assert output_context.tool_call is not None
                assert output_context.tool_def is not None
                return output

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_validate')
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_execute')
                return output

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_execute')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[FullLifecycleCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=100)
        assert log == [
            'before_validate',
            'after_validate',
            'before_execute',
            'after_execute',
        ]
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 100}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestOutputContext:
    """OutputContext is populated correctly for different output modes."""

    async def test_output_context_for_prompted_output(self):
        """OutputContext has correct fields for prompted text output."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'prompted'
        assert oc.output_type is MyOutput
        assert oc.object_def is not None
        assert oc.has_function is False
        assert oc.tool_call is None
        assert oc.tool_def is None

    async def test_output_context_for_plain_text(self):
        """OutputContext has correct fields for plain text output (via process hooks)."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'text'
        assert oc.output_type is str
        assert oc.object_def is None
        assert oc.has_function is False

    async def test_output_context_for_text_function(self):
        """OutputContext has correct fields for TextOutput function (via process hooks)."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'text'
        assert oc.output_type is str
        assert oc.has_function is True

    async def test_output_context_for_tool_output(self):
        """OutputContext has correct fields for tool-based output, including tool_call and tool_def."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'tool'
        assert oc.output_type is MyOutput
        assert oc.object_def is not None
        assert oc.has_function is False
        assert oc.tool_call is not None
        assert oc.tool_call.tool_name == 'final_result'
        assert oc.tool_def is not None
        assert oc.tool_def.name == 'final_result'
        assert oc.tool_def.kind == 'output'


class TestWrapOutputProcess:
    """wrap_output_process provides full middleware control around execution."""

    async def test_wrap_can_observe(self):
        """wrap_output_process can observe without modifying."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class WrapCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('before')
                result = await handler(output)
                log.append('after')
                return result

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[WrapCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert log == ['before', 'after']

    async def test_wrap_can_replace_result(self):
        """wrap_output_process can replace the result entirely."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class ReplaceCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                await handler(output)  # Call handler but ignore result
                return MyOutput(value=0)

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[ReplaceCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=0)


class TestOnOutputProcessError:
    """on_output_process_error can recover from execution failures."""

    async def test_recover_from_output_function_error(self):
        """on_output_process_error catches errors from output functions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('trigger error')

        def failing_func(text: str) -> str:
            raise ValueError('output function failed')

        @dataclass
        class RecoverCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'recovered'

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(failing_func), capabilities=[RecoverCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='trigger error')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_default_reraises(self):
        """Without a recovery hook, output execution errors propagate."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('trigger error')

        def failing_func(text: str) -> str:
            raise ValueError('output function failed')

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(failing_func))
        with pytest.raises(ValueError, match='output function failed'):
            await agent.run('hello')


class TestRunSync:
    """Output hooks work with run_sync as well as run."""

    def test_before_output_validate_with_run_sync(self):
        """Output hooks fire correctly with agent.run_sync."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 77}')])

        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_validate
        def log_hook(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_validate')
            return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=77)
        assert log == ['before_validate']


class TestOutputHookErrorPaths:
    """Test error paths to ensure correct error wrapping and hook firing."""

    def test_on_output_validate_error_reraise_wraps_in_tool_retry(self):
        """When on_output_validate_error re-raises ValidationError, it's wrapped in ToolRetryError causing retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not valid json')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        error_log: list[str] = []

        @dataclass
        class ErrorLogCapability(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append(f'validate_error: {type(error).__name__}')
                raise error  # Re-raise — should cause retry

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[ErrorLogCapability()],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert len(error_log) == 1
        assert error_log[0] == 'validate_error: ValidationError'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='not valid json')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected ident at line 1 column 2',
                                    'input': 'not valid json',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_on_output_process_error_recovery(self):
        """on_output_process_error can recover from output function failure."""

        def bad_function(value: int) -> str:
            raise ValueError('value too small')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        @dataclass
        class RecoverCapability(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'recovered value'

        agent = Agent(
            FunctionModel(model_fn),
            output_type=bad_function,
            capabilities=[RecoverCapability()],
        )
        result = agent.run_sync('hello')
        assert result.output == 'recovered value'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 42}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
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

    def test_composed_on_output_validate_error_chain(self):
        """Multiple capabilities' on_output_validate_error hooks chain correctly."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[TextPart(content='invalid')])
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        error_log: list[str] = []

        @dataclass
        class FirstCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('first_error')
                raise error

        @dataclass
        class SecondCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('second_error')
                raise error

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[FirstCap(), SecondCap()],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=1)
        # Both error hooks should have been called (reverse order per composition)
        assert 'second_error' in error_log
        assert 'first_error' in error_log
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='invalid')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected value at line 1 column 1',
                                    'input': 'invalid',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 1}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_composed_on_output_process_error_chain(self):
        """Multiple capabilities' on_output_process_error hooks chain correctly."""

        def failing_func(value: int) -> str:
            raise ValueError('intentional')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        @dataclass
        class FirstCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'recovered_by_first'

        @dataclass
        class SecondCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                raise error  # Don't recover, pass to next cap

        agent = Agent(
            FunctionModel(model_fn),
            output_type=failing_func,
            capabilities=[FirstCap(), SecondCap()],
        )
        result = agent.run_sync('hello')
        assert result.output == 'recovered_by_first'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 42}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
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

    def test_hooks_output_validate_error_decorator(self):
        """Test on_output_validate_error via Hooks decorator API."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[TextPart(content='bad json')])
            return ModelResponse(parts=[TextPart(content='{"value": 99}')])

        hooks = Hooks()

        @hooks.on.output_validate_error
        async def handle_error(
            ctx: RunContext[Any],
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            error: ValidationError | ModelRetry,
        ) -> Any:
            raise error  # Re-raise to trigger retry

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=99)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='bad json')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected value at line 1 column 1',
                                    'input': 'bad json',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 99}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_hooks_output_process_error_decorator(self):
        """Test on_output_process_error via Hooks decorator API."""

        def bad_function(value: int) -> str:
            raise ValueError('intentional failure')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 10}')])

        hooks = Hooks()

        @hooks.on.output_process_error
        async def handle_error(
            ctx: RunContext[Any],
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'fallback result'

        agent = Agent(
            FunctionModel(model_fn),
            output_type=bad_function,
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == 'fallback result'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 10}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
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

    def test_tool_output_validate_error_hook_not_triggered_on_valid_data(self):
        """For tool output with valid data, on_output_validate_error does not fire."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        hooks = Hooks()
        error_log: list[str] = []

        @hooks.on.before_output_validate
        def log_validate(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            error_log.append('before_validate')
            return output

        agent = Agent(
            FunctionModel(model_fn),
            output_type=MyOutput,
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=42)
        assert error_log == ['before_validate']  # Validate fires but no error

    def test_wrapper_capability_output_hooks_delegate(self):
        """WrapperCapability delegates output hooks to wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        log: list[str] = []

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('inner_before_validate')
                return output

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('inner_after_execute')
                return output

        @dataclass
        class OuterCap(WrapperCapability[Any]):
            pass

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[OuterCap(wrapped=InnerCap())],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=5)
        assert 'inner_before_validate' in log
        assert 'inner_after_execute' in log


class TestDefaultOutputErrorHooks:
    """Test that default (no override) error hooks work correctly via retry."""

    def test_default_on_output_validate_error_causes_retry(self):
        """Default on_output_validate_error re-raises, triggering model retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not json')])
            return ModelResponse(parts=[TextPart(content='{"value": 7}')])

        # Hooks with only a before_output_validate hook (no error hook override).
        # Default on_output_validate_error re-raises → ToolRetryError → model retry.
        hooks = Hooks()

        @hooks.on.before_output_validate
        def noop(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=7)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='not json')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected ident at line 1 column 2',
                                    'input': 'not json',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 7}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_default_on_output_process_error_reraises(self):
        """Default on_output_process_error re-raises the error."""

        def failing_func(value: int) -> str:
            raise ValueError('intentional')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        # Hooks with only a before_output_process hook (no error hook override).
        hooks = Hooks()

        @hooks.on.before_output_process
        def noop(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            return output

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[hooks])
        with pytest.raises(ValueError, match='intentional'):
            agent.run_sync('hello')


class TestStreamingOutputHooks:
    """Output hooks fire during streaming (partial and final validation)."""

    async def test_output_hooks_fire_during_streaming(self):
        """Validate hooks fire on partial attempts; execute hooks fire only when partial validation succeeds."""

        hook_calls: list[tuple[str, bool]] = []

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            # Stream the JSON response in chunks
            yield {0: DeltaToolCall(name='final_result', json_args='{"val')}
            yield {0: DeltaToolCall(json_args='ue": 42}')}

        @dataclass
        class StreamLogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                hook_calls.append(('before_validate', ctx.partial_output))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                hook_calls.append(('after_execute', ctx.partial_output))
                return output

        agent = Agent(FunctionModel(stream_function=stream_fn), output_type=MyOutput, capabilities=[StreamLogCap()])
        async with agent.run_stream('hello') as stream:
            outputs = [o async for o in stream.stream_output(debounce_by=None)]
        assert outputs[-1] == MyOutput(value=42)
        # Validate hooks fire on partial attempts AND the final result
        validate_calls = [(phase, partial) for phase, partial in hook_calls if phase == 'before_validate']
        assert any(partial for _, partial in validate_calls), 'Expected at least one partial validation call'
        assert any(not partial for _, partial in validate_calls), 'Expected at least one final validation call'
        # Execute hooks fire only when validation succeeds (partial or final)
        execute_calls = [(phase, partial) for phase, partial in hook_calls if phase == 'after_execute']
        assert any(not partial for _, partial in execute_calls), 'Expected at least one final execute call'
        assert stream.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 42}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=4),
                    model_name='function::stream_fn',
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

    async def test_union_output_hooks_fire_during_streaming(self):
        """Union output types: hooks fire during partial and final validation, with the kind
        resolved per-invocation so concurrent streams can't clobber each other."""

        class TypeA(BaseModel):
            value: int

        class TypeB(BaseModel):
            name: str

        hook_calls: list[tuple[str, bool]] = []

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            yield {0: DeltaToolCall(name='final_result_TypeA', json_args='{"va')}
            yield {0: DeltaToolCall(json_args='lue": 7}')}

        @dataclass
        class StreamLogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                hook_calls.append(('before_validate', ctx.partial_output))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                hook_calls.append(('after_execute', ctx.partial_output))
                return output

        agent = Agent(
            FunctionModel(stream_function=stream_fn),
            output_type=[TypeA, TypeB],
            capabilities=[StreamLogCap()],
        )
        async with agent.run_stream('hello') as stream:
            outputs = [o async for o in stream.stream_output(debounce_by=None)]
        assert isinstance(outputs[-1], TypeA)
        assert outputs[-1].value == 7
        # Validate hooks fire on partial attempts AND final
        assert any(partial for phase, partial in hook_calls if phase == 'before_validate')
        assert any(not partial for phase, partial in hook_calls if phase == 'before_validate')
        # Execute hooks fire on final at minimum
        assert any(not partial for phase, partial in hook_calls if phase == 'after_execute')


class TestOutputHookEdgeCases:
    """Tests for edge cases to ensure full coverage of output hook code paths."""

    def test_before_output_validate_transforms_text_to_dict(self):
        """before_output_validate can transform raw text to a pre-parsed dict."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ignored raw text')])

        @dataclass
        class PreParseCapability(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                # Transform text to a pre-parsed dict
                return {'value': 99}

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[PreParseCapability()],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=99)

    def test_streaming_output_hooks_fire_on_partial(self):
        """Process hooks fire for plain text output (validate hooks are skipped)."""
        from pydantic_ai.models.function import FunctionModel

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='hello world')])

        @dataclass
        class StreamLogCapability(AbstractCapability[Any]):
            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append(f'before_process partial={ctx.partial_output}')
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[StreamLogCapability()])
        result = agent.run_sync('hello')
        assert result.output == 'hello world'
        assert any('before_process' in entry for entry in log)

    def test_no_capability_fast_path_structured_raw_validation_error(self):
        """`ObjectOutputProcessor.hook_validate` — used by streaming paths without retries —
        must let `ValidationError` propagate unwrapped.
        """
        from pydantic_ai._output import ObjectOutputProcessor

        processor = ObjectOutputProcessor(output=MyOutput)

        ctx = RunContext(
            deps=None,
            model=None,  # pyright: ignore[reportArgumentType]
            usage=None,  # pyright: ignore[reportArgumentType]
            prompt='test',
            run_step=0,
            retry=0,
            max_retries=3,
            trace_include_content=False,
            tracer=NoOpTracer(),
            instrumentation_version=0,
        )
        with pytest.raises(ValidationError):
            processor.hook_validate('not valid json', run_context=ctx)

    def test_no_capability_fast_path_union_raw_validation_error(self):
        """Same as above but for `UnionOutputProcessor.hook_validate`."""
        from pydantic_ai._output import UnionOutputProcessor

        processor = UnionOutputProcessor(outputs=[MyOutput])

        ctx = RunContext(
            deps=None,
            model=None,  # pyright: ignore[reportArgumentType]
            usage=None,  # pyright: ignore[reportArgumentType]
            prompt='test',
            run_step=0,
            retry=0,
            max_retries=3,
            trace_include_content=False,
            tracer=NoOpTracer(),
            instrumentation_version=0,
        )
        with pytest.raises(ValidationError):
            processor.hook_validate('not valid json', run_context=ctx)

    def test_output_toolset_call_tool_raises(self):
        """`OutputToolset.call_tool` exists only to satisfy `AbstractToolset` — output tools go
        through `ToolManager.validate_output_tool_call` / `execute_output_tool_call`, never
        through the normal toolset path. Calling `call_tool` directly must raise.
        """
        import asyncio

        from pydantic_ai._output import OutputToolset

        toolset = OutputToolset.build([MyOutput])
        assert toolset is not None
        toolset.max_retries = 1  # Agent normally sets this; required by `get_tools`

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # pyright: ignore[reportArgumentType]
                usage=None,  # pyright: ignore[reportArgumentType]
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            tools = await toolset.get_tools(ctx)
            tool_name = next(iter(tools))
            tool = tools[tool_name]
            await toolset.call_tool(tool_name, {}, ctx, tool)

        with pytest.raises(NotImplementedError, match='validate_output_tool_call'):
            asyncio.run(run())

    def test_hooks_on_output_process_via_hooks_class(self):
        """Test wrap_output_process via Hooks decorator API."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 10}')])

        hooks = Hooks()
        execute_log: list[str] = []

        @hooks.on.output_process
        async def wrap_exec(
            ctx: RunContext[Any],
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            handler: Any,
        ) -> Any:
            execute_log.append('wrap_execute_before')
            result = await handler(output)
            execute_log.append('wrap_execute_after')
            return result

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=10)
        assert execute_log == ['wrap_execute_before', 'wrap_execute_after']


class TestErrorHookCoveragePaths:
    """Tests to exercise error hook delegation paths (abstract defaults, wrapper, hooks chaining)."""

    def test_bare_capability_default_on_output_validate_error(self):
        """A bare AbstractCapability subclass with no error hook override exercises default `raise error`."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not json')])
            return ModelResponse(parts=[TextPart(content='{"value": 3}')])

        @dataclass
        class BareCap(AbstractCapability[Any]):
            """Has no hook overrides — uses all defaults."""

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[BareCap()])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=3)
        assert call_count == 2  # First attempt failed, retried

    def test_bare_capability_default_on_output_process_error(self):
        """A bare AbstractCapability subclass with no error hook override lets execute errors propagate."""

        def failing_func(value: int) -> str:
            raise ValueError('execute fail')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        @dataclass
        class BareCap(AbstractCapability[Any]):
            pass

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[BareCap()])
        with pytest.raises(ValueError, match='execute fail'):
            agent.run_sync('hello')

    def test_wrapper_on_output_validate_error_delegates(self):
        """WrapperCapability delegates on_output_validate_error to the wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='invalid')])
            return ModelResponse(parts=[TextPart(content='{"value": 8}')])

        error_log: list[str] = []

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('inner_error')
                raise error

        @dataclass
        class OuterWrap(WrapperCapability[Any]):
            pass

        agent = Agent(
            FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[OuterWrap(wrapped=InnerCap())]
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=8)
        assert 'inner_error' in error_log

    def test_wrapper_on_output_process_error_delegates(self):
        """WrapperCapability delegates on_output_process_error to the wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        def failing_func(value: int) -> str:
            raise ValueError('exec fail')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'wrapper_recovered'

        @dataclass
        class OuterWrap(WrapperCapability[Any]):
            pass

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[OuterWrap(wrapped=InnerCap())])
        result = agent.run_sync('hello')
        assert result.output == 'wrapper_recovered'

    def test_hooks_on_output_process_error_chaining(self):
        """Hooks class on_output_process_error re-raises, chaining errors."""

        def failing_func(value: int) -> str:
            raise ValueError('original')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        hooks = Hooks()

        @hooks.on.output_process_error
        async def first_handler(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any], error: Exception
        ) -> Any:
            raise ValueError('chained')  # Re-raise different error

        @hooks.on.output_process_error
        async def second_handler(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any], error: Exception
        ) -> Any:
            return 'recovered'  # This one recovers

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[hooks])
        result = agent.run_sync('hello')
        assert result.output == 'recovered'


class TestUnionOutputWithHooks:
    """Tests for UnionOutputProcessor with output hooks — verifying clean validate/call decomposition."""

    def test_union_output_hooks_fire_for_both_phases(self):
        """Union output types properly split into validate (Pydantic) and execute (function call) phases."""

        class TypeA(BaseModel):
            kind: str = 'a'
            value: int

        class TypeB(BaseModel):
            kind: str = 'b'
            name: str

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "TypeA", "data": {"value": 42}}}')])

        @dataclass
        class LogCapability(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_validate')
                return output

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_validate')
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('before_execute')
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('after_execute')
                return output

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[LogCapability()],
        )
        result = agent.run_sync('hello')
        assert isinstance(result.output, TypeA)
        assert result.output.value == 42
        # Both validate and execute hooks should fire
        assert 'before_validate' in log
        assert 'after_validate' in log
        assert 'before_execute' in log
        assert 'after_execute' in log

    def test_union_output_process_hook_transforms_result(self):
        """Execute hooks can transform the result for union output types."""

        class TypeA(BaseModel):
            value: int

        class TypeB(BaseModel):
            name: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "TypeA", "data": {"value": 5}}}')])

        @dataclass
        class DoubleCapability(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                assert isinstance(output, TypeA)
                output.value *= 2
                return output

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[DoubleCapability()],
        )
        result = agent.run_sync('hello')
        assert isinstance(result.output, TypeA)
        assert result.output.value == 10

    def test_union_with_multi_arg_output_function_runs(self):
        """A multi-arg output function in a union must actually execute.

        Regression: `UnionOutputProcessor.hook_execute` previously isinstance-checked the
        validated dict against the function's first-arg type, which always failed for
        multi-arg functions, so the function was silently bypassed.
        """
        executed: list[tuple[int, str]] = []

        def combine(x: int, y: str) -> str:
            executed.append((x, y))
            return f'{x}:{y}'

        class Other(BaseModel):
            value: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Emit the discriminated union shape that PromptedOutput expects, selecting the
            # `combine` branch with the dict the multi-arg function will receive.
            return ModelResponse(
                parts=[TextPart(content='{"result": {"kind": "combine", "data": {"x": 7, "y": "ok"}}}')]
            )

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput([combine, Other]))
        result = agent.run_sync('hello')
        assert result.output == '7:ok'
        assert executed == [(7, 'ok')]

    def test_union_resolve_by_type_skips_multi_arg_inners(self):
        """When a process hook swaps the semantic value to a different type, `hook_execute`
        falls through to `_resolve_inner_for_value`. That fallback can't pick a multi-arg
        function inner because its `output_type` is just the first arg's type — it should
        skip multi-arg inners and only consider single-value inners (BaseModel, primitives).
        """

        def combine(x: int, y: str) -> str:  # pragma: no cover
            return f'{x}:{y}'

        class Single(BaseModel):
            value: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "Single", "data": {"value": 1}}}')])

        @dataclass
        class SwapToInt(AbstractCapability[Any]):
            """Swap the validated `Single` instance for a bare `int` during the process
            phase, so the value no longer matches `Single`'s type. The fallthrough resolver
            should iterate inners — skip `combine` (multi-arg, can't isinstance-check),
            and not find any matching single-value inner for `int` since `Single` is the
            only single-value inner and the int isn't a `Single`."""

            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
                handler: Callable[[Any], Awaitable[Any]],
            ) -> Any:
                return await handler(99)

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([combine, Single]),
            capabilities=[SwapToInt()],
        )
        # No matching inner found → semantic returned unmodified.
        result = agent.run_sync('hello')
        assert result.output == 99

    def test_union_on_output_validate_error_fires(self):
        """on_output_validate_error fires for union output when validation fails."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not json')])
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "MyOutput", "data": {"value": 1}}}')])

        error_log: list[str] = []

        @dataclass
        class ErrorLogCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('validate_error')
                raise error

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([MyOutput, MyOutput]),
            capabilities=[ErrorLogCap()],
        )
        result = agent.run_sync('hello')
        assert isinstance(result.output, MyOutput)
        assert call_count == 2
        assert 'validate_error' in error_log

    async def test_union_error_hook_recovery(self):
        """on_output_validate_error can recover for union types without crashing."""

        class TypeA(BaseModel):
            a_val: int

        class TypeB(BaseModel):
            b_val: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Return invalid union JSON — missing 'result' envelope
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverUnionCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                # Recover with a pre-built result
                return TypeA(a_val=42)

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[RecoverUnionCap()],
        )
        result = await agent.run('hello')
        assert result.output == TypeA(a_val=42)

    async def test_union_error_hook_recovery_second_type(self):
        """Error recovery matching the second union type exercises the isinstance loop."""

        class TypeA(BaseModel):
            a_val: int

        class TypeB(BaseModel):
            b_val: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverUnionCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                # Recover with TypeB — the second union member — so isinstance(output, TypeA)
                # fails first, then isinstance(output, TypeB) succeeds
                return TypeB(b_val='recovered')

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[RecoverUnionCap()],
        )
        result = await agent.run('hello')
        assert result.output == TypeB(b_val='recovered')

    async def test_union_error_hook_recovery_with_primitive(self):
        """Union mixing a BaseModel with a primitive (`Foo | bool | None`).

        `bool` gets an `outer_typed_dict_key='response'` wrapper; recovery must rewrap the
        primitive into the inner processor's dict shape before calling.
        """

        class Foo(BaseModel):
            x: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverPrimitiveCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                return True  # recover with a bool, matching the second union member

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, bool]),
            capabilities=[RecoverPrimitiveCap()],
        )
        result = await agent.run('hello')
        assert result.output is True

    async def test_union_error_hook_recovery_with_generic(self):
        """Union mixing a BaseModel with a generic (`Foo | list[Bar]`).

        `isinstance(x, list[Bar])` raises `TypeError`; resolution must fall back to the
        generic origin (`list`) so the recovered list-valued output still maps to its
        inner processor.
        """

        class Foo(BaseModel):
            x: int

        class Bar(BaseModel):
            y: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverListCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                return [Bar(y=1), Bar(y=2)]

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, list[Bar]]),
            capabilities=[RecoverListCap()],
        )
        result = await agent.run('hello')
        assert result.output == [Bar(y=1), Bar(y=2)]

    async def test_union_after_validate_hook_swaps_union_member(self):
        """`after_output_validate` can return a value of a different union member.

        If the validated kind was `Foo` but a hook returned a `Bar`, `hook_execute` must
        fall through to type-based resolution instead of passing a `Bar` to `Foo`'s inner
        processor.
        """

        class Foo(BaseModel):
            kind: str = 'Foo'
            x: int

        class Bar(BaseModel):
            kind: str = 'Bar'
            y: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "Foo", "data": {"x": 1}}}')])

        @dataclass
        class SwapUnionCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                # Model said "Foo", hook swaps to "Bar" — execute must route to Bar's processor.
                return Bar(y=42)

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, Bar]),
            capabilities=[SwapUnionCap()],
        )
        result = await agent.run('hello')
        assert result.output == Bar(y=42)

    async def test_union_hook_returns_unknown_type_passes_through(self):
        """If a hook returns a value matching NO union member, `hook_execute` passes it through.

        The output function (if any) doesn't run, and the value reaches the user as-is —
        better than silently dropping to `None`.
        """

        class Foo(BaseModel):
            x: int

        class Bar(BaseModel):
            y: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverUnknownCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                return 'not in union'  # str isn't Foo or Bar

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, Bar]),
            capabilities=[RecoverUnknownCap()],
        )
        result = await agent.run('hello')
        assert result.output == 'not in union'


class TestTextFunctionOutputCallHook:
    """Tests that TextFunctionOutputProcessor.call() is exercised through execute hooks."""

    def test_text_function_execute_hook_wraps_call(self):
        """Execute hooks wrap the text function call (processor.call)."""

        def uppercase(text: str) -> str:
            return text.upper()

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='hello world')])

        @dataclass
        class ExecLogCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, handler: Any
            ) -> Any:
                log.append(f'input: {output}')
                result = await handler(output)
                log.append(f'output: {result}')
                return result

        agent = Agent(
            FunctionModel(model_fn),
            output_type=TextOutput(uppercase),
            capabilities=[ExecLogCap()],
        )
        result = agent.run_sync('hello')
        assert result.output == 'HELLO WORLD'
        assert log == ['input: hello world', 'output: HELLO WORLD']


class TestNativeOutputWithHooks:
    """Output hooks fire for native structured output mode."""

    async def test_hooks_fire_for_native_output(self):
        """Output hooks fire with mode='native' for NativeOutput."""
        log: list[tuple[str, str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 7}')])

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append(('before_validate', output_context.mode))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('after_execute', output_context.mode))
                return output

        agent = Agent(FunctionModel(model_fn), output_type=NativeOutput(MyOutput), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=7)
        assert log == [('before_validate', 'native'), ('after_execute', 'native')]

    async def test_before_validate_transforms_native_output(self):
        """before_output_validate can transform raw text before native output parsing."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])

        @dataclass
        class FixCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                if isinstance(output, str):
                    return output.replace('"bad"', '42')
                return output  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=NativeOutput(MyOutput), capabilities=[FixCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)

    async def test_model_retry_from_native_output_hook(self):
        """ModelRetry from output hooks triggers retry for native output."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class RejectCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value < 0:
                    raise ModelRetry('Value must be non-negative')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=NativeOutput(MyOutput), capabilities=[RejectCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2


class TestImageOutputWithHooks:
    """Image output fires process hooks (not validate hooks, since there's no parsing)."""

    async def test_process_hooks_fire_for_image_output(self):
        """Process hooks fire for image output; validate hooks are skipped."""
        log: list[str] = []

        def return_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'test-png', media_type='image/png'))])

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                # The uncovered body is the assertion: this hook must not fire for images.
                log.append('validate')  # pragma: no cover
                return output  # pragma: no cover

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(f'process:{output_context.mode}')
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('after_process')
                assert isinstance(output, BinaryImage)
                return output

        image_profile = ModelProfile(supports_image_output=True)
        agent = Agent(
            FunctionModel(return_image, profile=image_profile), output_type=BinaryImage, capabilities=[LogCap()]
        )
        result = await agent.run('hello')
        assert isinstance(result.output, BinaryImage)
        assert result.output.data == b'test-png'
        # Process hooks fire; validate hooks do NOT (no parsing for images)
        assert log == ['process:image', 'after_process']

    async def test_image_process_hook_can_transform(self):
        """Process hooks can transform image output."""

        def return_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'original', media_type='image/png'))])

        @dataclass
        class TransformCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, BinaryImage):
                    return BinaryImage(data=b'transformed', media_type=output.media_type)
                return output  # pragma: no cover

        image_profile = ModelProfile(supports_image_output=True)
        agent = Agent(
            FunctionModel(return_image, profile=image_profile), output_type=BinaryImage, capabilities=[TransformCap()]
        )
        result = await agent.run('hello')
        assert isinstance(result.output, BinaryImage)
        assert result.output.data == b'transformed'


class TestAutoModeOutputWithHooks:
    """Output hooks fire for auto mode (which delegates to tool or text based on model)."""

    async def test_hooks_fire_for_auto_mode_tool_path(self):
        """Auto mode that resolves to tool output fires output hooks."""
        log: list[tuple[str, str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Auto mode with default tool profile — model uses output tools
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 99}', tool_call_id='call-1')]
                )
            return ModelResponse(parts=[TextPart(content='{"value": 99}')])  # pragma: no cover

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append(('before_validate', output_context.mode))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('after_execute', output_context.mode))
                return output

        # Default auto mode — FunctionModel defaults to tool mode
        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=99)
        assert log == [('before_validate', 'tool'), ('after_execute', 'tool')]


class TestHookSemanticValue:
    """Output hooks see the **semantic value** (what the model was asked to produce), not the
    internal dict-wrapped form used by the validator pipeline.

    This is intentionally different from *tool* call hooks, which always see `dict[str, Any]`
    (matching the tool schema the model satisfies). For outputs, users think of
    `Agent(output_type=T)` as "the model produces a T", so hooks should see T.
    """

    async def _run_and_capture(
        self,
        *,
        output_type: Any,
        model_fn: Any,
    ) -> tuple[Any, list[tuple[str, Any]]]:
        log: list[tuple[str, Any]] = []

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('after_validate', output))
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('before_process', output))
                return output

        agent = Agent(FunctionModel(model_fn), output_type=output_type, capabilities=[CaptureCap()])
        result = await agent.run('hello')
        return result.output, log

    async def test_case_a_bare_basemodel_tool_output(self):
        """Case A: `Agent(output_type=MyOutput)` — hooks see the BaseModel instance."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"value": 42}')])

        output, log = await self._run_and_capture(output_type=MyOutput, model_fn=model_fn)
        assert output == MyOutput(value=42)
        assert log == [('after_validate', MyOutput(value=42)), ('before_process', MyOutput(value=42))]

    async def test_case_b_bare_int_tool_output(self):
        """Case B: `Agent(output_type=int)` — hooks see `42`, not `{'response': 42}`."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"response": 42}')])

        output, log = await self._run_and_capture(output_type=int, model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_case_c_function_basemodel_arg(self):
        """Case C: `def f(data: MyOutput) -> int` — hooks see `MyOutput(...)`, not `{'data': MyOutput(...)}`."""

        def double(data: MyOutput) -> int:
            return data.value * 2

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"value": 21}')])

        output, log = await self._run_and_capture(output_type=double, model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', MyOutput(value=21)), ('before_process', MyOutput(value=21))]

    async def test_case_d_function_primitive_arg(self):
        """Case D: `def f(data: int) -> str` — hooks see `42`, not `{'data': 42}`."""

        def stringify(data: int) -> str:
            return f'got {data}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"data": 42}')])

        output, log = await self._run_and_capture(output_type=stringify, model_fn=model_fn)
        assert output == 'got 42'
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_case_e_function_multiple_args(self):
        """Case E: multi-arg function — hooks see the dict (genuine multi-value input)."""

        def combine(data: MyOutput, other: str) -> str:
            return f'{data.value}:{other}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"data": {"value": 7}, "other": "x"}')])

        output, log = await self._run_and_capture(output_type=combine, model_fn=model_fn)
        assert output == '7:x'
        # Multi-arg: hooks see the dict
        assert log == [
            ('after_validate', {'data': MyOutput(value=7), 'other': 'x'}),
            ('before_process', {'data': MyOutput(value=7), 'other': 'x'}),
        ]

    async def test_native_output_unwraps_primitive(self):
        """NativeOutput(int) — hooks see `42`, not `{'response': 42}`."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"response": 42}')])

        output, log = await self._run_and_capture(output_type=NativeOutput(int), model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_native_output_unwraps_function_basemodel(self):
        """NativeOutput(func-with-basemodel-arg) — hooks see the BaseModel, not the wrap dict."""

        def double(data: MyOutput) -> int:
            return data.value * 2

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 21}')])

        output, log = await self._run_and_capture(output_type=NativeOutput(double), model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', MyOutput(value=21)), ('before_process', MyOutput(value=21))]

    async def test_prompted_output_unwraps_primitive(self):
        """PromptedOutput(int) — hooks see `42`, not `{'response': 42}`."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"response": 42}')])

        output, log = await self._run_and_capture(output_type=PromptedOutput(int), model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_prompted_output_unwraps_function_primitive(self):
        """PromptedOutput(func-with-primitive-arg) — hooks see the primitive value."""

        def stringify(data: int) -> str:
            return f'got {data}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"data": 42}')])

        output, log = await self._run_and_capture(output_type=PromptedOutput(stringify), model_fn=model_fn)
        assert output == 'got 42'
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_output_validator_sees_final_processed_value(self):
        """Output validators see the final value (after function call), not the wrapped form."""

        def double(data: MyOutput) -> int:
            return data.value * 2

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"value": 21}')])

        seen: list[Any] = []
        agent = Agent(FunctionModel(model_fn), output_type=double)

        @agent.output_validator
        def validate(v: int) -> int:
            seen.append(v)
            return v

        result = await agent.run('hello')
        assert result.output == 42
        # Validator sees the post-process value (function's return), an int
        assert seen == [42]

    async def test_hook_transform_at_semantic_boundary(self):
        """A hook can transform the semantic value and the transformed value flows through correctly."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"response": 10}')])

        @dataclass
        class DoubleCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                return output * 2  # transform the semantic int value

        agent = Agent(FunctionModel(model_fn), output_type=int, capabilities=[DoubleCap()])
        result = await agent.run('hello')
        assert result.output == 20

    async def test_dict_output_type_contains_unwrap_key(self):
        """Regression: `output_type=dict[str, Any]` where the dict contains the unwrap key
        ('response') must not be mistaken for an already-wrapped value during re-wrap.
        """

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            # The dict itself contains a 'response' key — the same key used as the outer wrapper
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"response": {"response": "yes", "other": "stuff"}}')])

        output, log = await self._run_and_capture(output_type=dict[str, Any], model_fn=model_fn)
        # Hook sees the inner dict (unwrapped)
        assert log == [
            ('after_validate', {'response': 'yes', 'other': 'stuff'}),
            ('before_process', {'response': 'yes', 'other': 'stuff'}),
        ]
        # Final output is the full inner dict — NOT just "yes" (which would happen if re-wrap
        # was skipped due to the buggy "already wrapped" check)
        assert output == {'response': 'yes', 'other': 'stuff'}


class TestHookExceptionHandling:
    """ValidationError/ModelRetry raised from before_* and after_* hooks should trigger retry,
    matching the behavior when raised from wrap_output_validate/wrap_output_process.
    """

    async def test_validation_error_from_after_output_validate_triggers_retry(self):
        """ValidationError from after_output_validate should be caught and trigger model retry."""
        from pydantic import TypeAdapter

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class StricterCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                # Additional Pydantic validation: reject negative values
                if isinstance(output, MyOutput) and output.value < 0:
                    # Simulate Pydantic validation failing
                    TypeAdapter(int).validate_python('not_an_int')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[StricterCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2  # retry happened

    async def test_validation_error_from_after_output_process_triggers_retry(self):
        """ValidationError from after_output_process should be caught and trigger model retry."""
        from pydantic import TypeAdapter

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class StricterCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value < 0:
                    TypeAdapter(int).validate_python('not_an_int')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[StricterCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2

    async def test_model_retry_from_before_output_process_triggers_retry(self):
        """ModelRetry from before_output_process should trigger model retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class RejectCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value < 0:
                    raise ModelRetry('Value must be non-negative')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2


# region HandleDeferredToolCalls


async def test_deferred_tool_handler_approve():
    """HandleDeferredToolCalls capability auto-approves a requires_approval tool inline."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done!')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done!'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={'x': 5}, tool_call_id='call1')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content=50,
                        tool_call_id='call1',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deferred_tool_handler_deny():
    """HandleDeferredToolCalls capability denies a requires_approval tool inline, producing a `ToolReturnPart(outcome='denied')`."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Understood, denied.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Not allowed.') for call in requests.approvals}
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood, denied.'
    # The denial must surface in message history as outcome='denied', not a successful return.
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].tool_call_id == 'call1'
    assert tool_returns[0].outcome == 'denied'
    assert tool_returns[0].content == 'Not allowed.'


async def test_deferred_tool_handler_no_output_type_needed():
    """When handler resolves all deferred calls, DeferredToolRequests is not needed in output type."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 3}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Result received.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # Note: output_type is just str, no DeferredToolRequests
    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 100

    result = await agent.run('Hello')
    assert result.output == 'Result received.'


async def test_deferred_tool_handler_none_fallback():
    """When no handler is present, deferred tools bubble up as DeferredToolRequests output."""

    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise ApprovalRequired

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1


async def test_deferred_tool_handler_partial_resolution():
    """Handler resolves some calls, remaining bubble up as DeferredToolRequests output."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('tool_a', {}, tool_call_id='a1'),
                ToolCallPart('tool_b', {}, tool_call_id='b1'),
            ]
        )

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Only approve tool_a, leave tool_b unresolved
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a done'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b done'  # pragma: no cover

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_name == 'tool_b'


async def test_deferred_tool_handler_sync_handler():
    """HandleDeferredToolCalls works with a sync handler function."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('OK')])

    def handle_deferred_sync(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred_sync)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'OK'


async def test_deferred_tool_handler_accumulation():
    """Two capabilities each resolve different deferred calls."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, tool_call_id='a1'),
                    ToolCallPart('tool_b', {}, tool_call_id='b1'),
                ]
            )
        return ModelResponse(parts=[TextPart('Both done.')])

    def handler_a(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    def handler_b(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # handler_a resolved tool_a, so we only see tool_b
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[
            HandleDeferredToolCalls(handler=handler_a),
            HandleDeferredToolCalls(handler=handler_b),
        ],
    )

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a result'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b result'

    result = await agent.run('Hello')
    assert result.output == 'Both done.'


async def test_deferred_tool_handler_unresolved_no_output_type_error():
    """Unresolved deferred calls without DeferredToolRequests in output type raises UserError."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    # Handler returns None → does not resolve
    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults()  # Empty results → nothing resolved

    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    with pytest.raises(UserError, match='DeferredToolRequests'):
        await agent.run('Hello')


async def test_deferred_tool_handler_external_call():
    """HandleDeferredToolCalls capability resolves an externally-executed tool."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 3}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Got it.')])

    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.messages import ToolReturn

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Simulate external execution: return a ToolReturn with metadata
        return DeferredToolResults(
            calls={
                call.tool_call_id: ToolReturn(return_value='external result', metadata={'source': 'ext'})
                for call in requests.calls
            }
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        raise CallDeferred

    result = await agent.run('Hello')
    assert result.output == 'Got it.'


async def test_deferred_tool_handler_via_handle_call():
    """handle_call(resolve_deferred=True) resolves deferred tools inline via ToolManager."""

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('All done.')])

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        result = await ctx.tool_manager.handle_call(inner_call)
        return f'inner returned: {result}'

    @agent.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_handle_call_wrap_validation_errors_false():
    """`wrap_validation_errors=False` propagates through deferred-tool resolution.

    Regression for the case where a sandboxed caller (`handle_call(wrap_validation_errors=False)`)
    invokes a tool that requires approval: after the handler approves, the re-execution must
    keep the raw-error contract — `ModelRetry` from the approved tool body should propagate
    as-is, not wrapped as `ToolRetryError`.
    """

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('Done.')])

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            await ctx.tool_manager.handle_call(inner_call, wrap_validation_errors=False)
        except ModelRetry as e:
            return f'raw ModelRetry: {e}'
        return 'no error'  # pragma: no cover

    @agent.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('post-approval retry')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # outer_tool caught the raw ModelRetry from the approved inner_tool body and surfaced it
    # in its return value; if wrap_validation_errors hadn't been forwarded through
    # _resolve_single_deferred, outer_tool would have seen a ToolRetryError instead.
    inner_message = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_name == 'outer_tool' for part in msg.parts)
    )
    outer_return = next(
        part for part in inner_message.parts if isinstance(part, ToolReturnPart) and part.tool_name == 'outer_tool'
    )
    assert outer_return.content == 'raw ModelRetry: post-approval retry'


async def test_deferred_tool_handler_via_handle_call_no_handler():
    """handle_call(resolve_deferred=True) re-raises when no handler is available."""
    from pydantic_ai.toolsets import FunctionToolset

    # inner_tool is only available via ToolManager, not as a top-level agent tool
    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'  # pragma: no cover

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('OK')])

    agent = Agent(FunctionModel(llm), toolsets=[inner_toolset])

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            result = await ctx.tool_manager.handle_call(inner_call)
            return f'inner returned: {result}'  # pragma: no cover
        except ApprovalRequired:
            return 'inner needs approval'

    result = await agent.run('Hello')
    assert result.output == 'OK'


async def test_deferred_tool_handler_build_results_helper():
    """DeferredToolRequests.build_results() creates a DeferredToolResults."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'Done.'


def test_deferred_tool_requests_build_results_validates_ids():
    """build_results rejects result IDs that don't match a pending request of the right kind."""
    requests = DeferredToolRequests(
        approvals=[ToolCallPart('a', {}, tool_call_id='approval_1')],
        calls=[ToolCallPart('b', {}, tool_call_id='call_1')],
    )

    # Mis-routed ID: tool-result provided for something in the approvals list.
    with pytest.raises(ValueError, match=r'calls.*not in.*DeferredToolRequests.calls'):
        requests.build_results(calls={'approval_1': 'oops'})

    # Unknown ID entirely.
    with pytest.raises(ValueError, match=r'approvals.*not in.*DeferredToolRequests.approvals'):
        requests.build_results(approvals={'unknown_id': True})

    # Happy path still works.
    results = requests.build_results(approvals={'approval_1': True}, calls={'call_1': 'result'})
    assert results.approvals == {'approval_1': True}
    assert results.calls == {'call_1': 'result'}


def test_deferred_tool_requests_remaining_cross_category_ids_do_not_resolve():
    """remaining() only resolves requests with a same-kind result, never a mis-keyed one."""
    approval = ToolCallPart('a', {}, tool_call_id='approval_1')
    call = ToolCallPart('b', {}, tool_call_id='call_1')
    requests = DeferredToolRequests(
        approvals=[approval],
        calls=[call],
        metadata={'approval_1': {'kind': 'approval'}, 'call_1': {'kind': 'call'}},
    )

    mis_keyed = DeferredToolResults(approvals={'call_1': True}, calls={'approval_1': 'result'})
    assert requests.remaining(mis_keyed) == requests

    matching = DeferredToolResults(approvals={'approval_1': True}, calls={'call_1': 'result'})
    assert requests.remaining(matching) is None

    approval_only = DeferredToolResults(approvals={'approval_1': True})
    assert requests.remaining(approval_only) == DeferredToolRequests(
        calls=[call], metadata={'call_1': {'kind': 'call'}}
    )


async def test_deferred_tool_handler_ignores_cross_category_ids():
    """A cross-category handler result does not execute an external call."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='call_1')])
        raise AssertionError('A cross-category result must not resume the model')  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={'call_1': True})

    agent = Agent(
        FunctionModel(model),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )
    calls = 0

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        nonlocal calls
        calls += 1
        raise CallDeferred

    result = await agent.run('go')

    assert calls == 1
    assert result.output == DeferredToolRequests(calls=[ToolCallPart('external_tool', {}, tool_call_id='call_1')])


def test_deferred_tool_requests_build_results_approve_all():
    """approve_all=True approves every pending approval not explicitly specified."""
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart('a', {}, tool_call_id='approval_1'),
            ToolCallPart('b', {}, tool_call_id='approval_2'),
            ToolCallPart('c', {}, tool_call_id='approval_3'),
        ],
    )

    # Explicit deny wins; the other two get auto-approved.
    results = requests.build_results(
        approvals={'approval_1': False},
        approve_all=True,
    )
    assert results.approvals['approval_1'] is False
    assert isinstance(results.approvals['approval_2'], ToolApproved)
    assert isinstance(results.approvals['approval_3'], ToolApproved)


async def test_deferred_tool_handler_wrapper_capability():
    """HandleDeferredToolCalls works through WrapperCapability delegation."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # PrefixTools wraps HandleDeferredToolCalls — tests WrapperCapability delegation
    inner = HandleDeferredToolCalls(handler=handle_deferred)
    agent = Agent(
        FunctionModel(llm),
        capabilities=[inner.prefix_tools('ns')],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'Done.'


async def test_deferred_tool_handler_external_call_plain_value():
    """HandleDeferredToolCalls resolves an external call with a plain value (not ToolReturn)."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Got it.')])

    from pydantic_ai.exceptions import CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: 'plain string result' for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool_plain
    def my_tool() -> str:
        raise CallDeferred

    result = await agent.run('Hello')
    assert result.output == 'Got it.'


async def test_deferred_tool_handler_re_deferred_with_metadata():
    """When an approved tool re-raises ApprovalRequired, it stays unresolved with metadata."""

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        # Always requires approval — even when approved, raises again with metadata
        raise ApprovalRequired(metadata={'attempt': call_count})

    result = await agent.run('Hello')
    # Tool re-raised after approval → goes to remaining → becomes output
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.metadata.get('call1') == {'attempt': 2}


async def test_deferred_tool_handler_denied_via_batch():
    """Batch path deny via handler produces a `ToolReturnPart(outcome='denied')` in message history."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Understood.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Policy denied.') for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood.'
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].outcome == 'denied'
    assert tool_returns[0].content == 'Policy denied.'


async def test_deferred_tool_handler_batch_deny_via_bool_and_default():
    """Batch path: covers `approvals[id] = False` AND default `ToolDenied()` as separate calls."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('needs_approval', {'x': 1}, tool_call_id='bool_false'),
                    ToolCallPart('needs_approval', {'x': 2}, tool_call_id='default_denied'),
                ]
            )
        return ModelResponse(parts=[TextPart('ok')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={
                'bool_false': False,
                'default_denied': ToolDenied(),  # no custom message
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x  # pragma: no cover

    result = await agent.run('go')
    assert result.output == 'ok'
    tool_returns = {p.tool_call_id: p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)}
    assert tool_returns['bool_false'].outcome == 'denied'
    assert tool_returns['bool_false'].content == ToolDenied().message
    assert tool_returns['default_denied'].outcome == 'denied'
    assert tool_returns['default_denied'].content == ToolDenied().message


async def test_deferred_tool_handler_batch_approve_via_tool_approved_default():
    """Batch path: covers `approvals[id] = ToolApproved()` (default, no override_args)."""
    from pydantic_ai.tools import ToolApproved

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('needs_approval', {'x': 7}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 2

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].outcome != 'denied'
    assert tool_returns[0].content == 14


async def test_deferred_tool_handler_batch_external_tool_return_metadata():
    """Batch path: handler-supplied external `ToolReturn(value, metadata)` lands on the return part."""
    from pydantic_ai.messages import ToolReturn as _ToolReturn

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: _ToolReturn(
                    return_value='computed', metadata={'source': 'external'}, content='user extra'
                )
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'done'
    messages = result.all_messages()
    tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'computed'
    assert tool_returns[0].metadata == {'source': 'external'}
    # The `content` field on ToolReturn becomes a UserPromptPart.
    from pydantic_ai.messages import UserPromptPart

    user_extras = [p for p in iter_message_parts(messages, ModelRequest, UserPromptPart) if p.content == 'user extra']
    assert len(user_extras) == 1


async def test_deferred_tool_handler_batch_external_model_retry():
    """Batch path: handler-supplied `ModelRetry` in `calls` surfaces as a `RetryPromptPart`, not a tool return."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('retried')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('try again') for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    messages = result.all_messages()
    retry_parts = list(iter_message_parts(messages, ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].content == 'try again'
    tool_returns = [p for p in iter_message_parts(messages, ModelRequest, ToolReturnPart) if p.tool_call_id == 'c1']
    assert tool_returns == []


async def test_deferred_tool_handler_batch_external_retry_prompt_part():
    """Batch path: handler-supplied `RetryPromptPart` in `calls` surfaces as a retry (names stamped from the deferred call)."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('retried')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    retry_parts = list(iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].tool_name == 'external_tool'
    assert retry_parts[0].content == 'retry via part'


async def test_deferred_tool_handler_via_handle_call_external_tool_return():
    """Per-call path: handler-supplied external `ToolReturn(value, metadata)` is returned verbatim from handle_call."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.messages import ToolReturn as _ToolReturn
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={call.tool_call_id: _ToolReturn(return_value='ext', metadata={'k': 'v'}) for call in requests.calls}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    # Per-call path returns whatever the handler supplied verbatim — full ToolReturn wrapper preserved.
    assert isinstance(captured_result, _ToolReturn)
    assert captured_result.return_value == 'ext'
    assert captured_result.metadata == {'k': 'v'}


async def test_deferred_tool_handler_via_handle_call_tool_failed():
    """Per-call path: handler-supplied `ToolFailed` raises `ToolFailedError`, matching a tool that raises `ToolFailed` in-process."""
    from pydantic_ai.exceptions import CallDeferred, ToolFailed, ToolFailedError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={call.tool_call_id: ToolFailed('backend unavailable') for call in requests.calls}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_error: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext[object]) -> str:
        nonlocal captured_error
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
        except ToolFailedError as e:
            captured_error = e
        return 'done'

    await agent.run('go')
    assert captured_error is not None
    assert captured_error.tool_failed.tool_name == 'inner_tool'
    assert captured_error.tool_failed.tool_call_id == 'inner_1'
    assert captured_error.tool_failed.content == 'backend unavailable'
    assert captured_error.tool_failed.outcome == 'failed'


def test_deferred_tool_handler_serialization_name():
    """HandleDeferredToolCalls is not spec-constructible."""
    assert HandleDeferredToolCalls.get_serialization_name() is None


async def test_deferred_tool_handler_via_handle_call_with_resolve():
    """handle_call(resolve_deferred=True) goes through _resolve_single_deferred happy path.

    This exercises the per-call resolution path used by CodeMode-style callers.
    """
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved result'

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        # Call inner_tool via handle_call — exercises _resolve_single_deferred
        result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        # _resolve_single_deferred returns result_part.content
        assert result == 'approved result'
        return f'got: {result}'

    result = await agent.run('go')
    assert result.output == 'final'
    # Verify the inner tool was called (tool return visible in messages)
    tool_returns = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'caller_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'got: approved result'


async def test_deferred_tool_handler_approved_tool_returns_tool_return():
    """Approved tool returning a ToolReturn preserves metadata and user content."""
    from pydantic_ai.messages import ToolReturn as _ToolReturn, UserPromptPart

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='result', metadata={'source': 'tool'}, content='user prompt extra')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # Verify ToolReturn.metadata preserved
    tool_returns = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart) if p.tool_name == 'my_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].metadata == {'source': 'tool'}
    # Verify ToolReturn.content appears as UserPromptPart
    user_parts = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if p.content == 'user prompt extra'
    ]
    assert len(user_parts) == 1


async def test_deferred_tool_handler_approved_tool_raises_model_retry():
    """Approved tool that raises ModelRetry produces a RetryPromptPart."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Retried and done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('try again')

    result = await agent.run('Hello')
    assert result.output == 'Retried and done.'
    # Verify the retry happened
    retry_parts = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart) if p.tool_name == 'my_tool'
    ]
    assert len(retry_parts) == 1


async def test_deferred_tool_handler_approved_tool_override_args():
    """Approved tool with ToolApproved(override_args=...) uses the override."""
    from pydantic_ai.tools import ToolApproved

    received_x = None

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Override the args: replace x=5 with x=42
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        nonlocal received_x
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        received_x = x
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    assert received_x == 42  # Override was applied


async def test_deferred_tool_handler_via_handle_call_retry():
    """handle_call path: approved tool raising ModelRetry propagates ToolRetryError."""
    from pydantic_ai.exceptions import ToolRetryError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()
    retry_count = 0

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        nonlocal retry_count
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        retry_count += 1
        raise ModelRetry('try again')

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no retry'  # pragma: no cover
        except ToolRetryError:
            return 'got retry'

    result = await agent.run('go')
    assert result.output == 'final'
    assert retry_count == 1


async def test_deferred_tool_handler_re_deferred_without_metadata():
    """Approved tool that re-raises without metadata — no metadata added to remaining."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        # No metadata
        raise ApprovalRequired

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    # No metadata set (tool raised without metadata both times)
    assert 'call1' not in result.output.metadata


async def test_deferred_tool_handler_mixed_unresolved_and_re_deferred():
    """Handler resolves some, another call is re-deferred — both end up in remaining."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('re_raising_tool', {}, tool_call_id='re1'),
                ToolCallPart('unhandled_tool', {}, tool_call_id='un1'),
            ]
        )

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Only approve the re-raising one; leave unhandled_tool unresolved
        return DeferredToolResults(
            approvals={call.tool_call_id: True for call in requests.approvals if call.tool_name == 're_raising_tool'}
        )

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def re_raising_tool(ctx: RunContext) -> str:
        # Always raises — even after approval
        raise ApprovalRequired

    @agent.tool
    def unhandled_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    # Both calls in remaining: unhandled_tool (never resolved) + re_raising_tool (re-deferred after approval)
    approval_ids = {call.tool_call_id for call in result.output.approvals}
    assert 're1' in approval_ids
    assert 'un1' in approval_ids


async def test_deferred_tool_handler_re_deferred_as_call_deferred():
    """Approved tool that re-raises CallDeferred (not ApprovalRequired) stays in remaining.calls."""
    from pydantic_ai.exceptions import CallDeferred

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise CallDeferred (external execution needed)
        raise CallDeferred(metadata={'reason': 'external'})

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    # Should be in calls (external), not approvals
    assert len(result.output.calls) == 1
    assert len(result.output.approvals) == 0
    assert result.output.metadata == {'call1': {'reason': 'external'}}


async def test_deferred_tool_handler_via_handle_call_preserves_tool_return():
    """handle_call(resolve_deferred=True) preserves `ToolReturn` wrapper (metadata, user content).

    The non-deferred `handle_call` path returns whatever the tool returned verbatim.
    The deferred path should do the same — critical for CodeMode-style callers that
    check `isinstance(result, ToolReturn)` to preserve metadata on nested return parts.
    """
    from pydantic_ai.messages import ToolReturn as _ToolReturn
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='actual result', metadata={'source': 'inner'}, content='user extra')

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        captured_result = result
        return 'done'

    await agent.run('go')
    # handle_call returned the ToolReturn wrapper verbatim, not the unwrapped content
    assert isinstance(captured_result, _ToolReturn)
    assert captured_result.return_value == 'actual result'
    assert captured_result.metadata == {'source': 'inner'}
    assert captured_result.content == 'user extra'


async def test_deferred_tool_handler_via_handle_call_denied_via_bool():
    """When a handler denies via `approvals[id] = False`, handle_call returns `ToolDenied()` with the default denial message."""
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: False for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured
        assert ctx.tool_manager is not None
        captured = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'caught' if isinstance(captured, ToolDenied) else 'no denial'

    await agent.run('go')
    assert isinstance(captured, ToolDenied)
    assert captured == ToolDenied()


async def test_deferred_tool_handler_via_handle_call_override_args():
    """When a handler approves with override_args, handle_call executes the tool with those args."""
    from pydantic_ai.tools import ToolApproved
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext, x: int) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return f'x={x}'

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={'x': 1}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    assert captured_result == 'x=42'


async def test_deferred_tool_handler_via_handle_call_external_plain_value():
    """When a handler supplies an external-call plain value, handle_call returns it verbatim."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: 'external value' for call in requests.calls})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    assert captured_result == 'external value'


async def test_deferred_tool_handler_via_handle_call_external_model_retry():
    """When a handler supplies a `ModelRetry` external-call result, handle_call raises `ToolRetryError`."""
    from pydantic_ai.exceptions import CallDeferred, ModelRetry, ToolRetryError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('retry please') for call in requests.calls})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught: ToolRetryError | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ToolRetryError as e:
            caught = e
            return 'caught'

    await agent.run('go')
    assert caught is not None
    assert caught.tool_retry.content == 'retry please'
    assert caught.tool_retry.tool_name == 'inner_tool'
    assert caught.tool_retry.tool_call_id == 'inner_1'


async def test_deferred_tool_handler_via_handle_call_external_retry_prompt_part():
    """When a handler supplies a `RetryPromptPart` external-call result, handle_call raises `ToolRetryError` with the part."""
    from pydantic_ai.exceptions import CallDeferred, ToolRetryError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught: ToolRetryError | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ToolRetryError as e:
            caught = e
            return 'caught'

    await agent.run('go')
    assert caught is not None
    assert caught.tool_retry.content == 'retry via part'
    # The helper stamps the real tool name / id onto the prompt part.
    assert caught.tool_retry.tool_name == 'inner_tool'
    assert caught.tool_retry.tool_call_id == 'inner_1'


async def test_deferred_tool_handler_via_handle_call_denied_returns_message():
    """When a handler denies a deferred call, handle_call returns the custom `ToolDenied` value verbatim."""
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied(message='not today') for call in requests.approvals}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured
        assert ctx.tool_manager is not None
        captured = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'caught' if isinstance(captured, ToolDenied) else 'no denial'

    await agent.run('go')
    assert isinstance(captured, ToolDenied)
    assert captured == ToolDenied(message='not today')


async def test_deferred_tool_handler_via_handle_call_re_raises_new_exception():
    """After approval, if tool re-raises CallDeferred (not ApprovalRequired), the new exception type is propagated."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()
    call_count = 0

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise a *different* deferral type with new metadata
        raise CallDeferred(metadata={'reason': 'external-after-approval'})

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught_exc_type: type | None = None
    caught_metadata: dict[str, Any] | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught_exc_type, caught_metadata
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except (CallDeferred, ApprovalRequired) as e:
            caught_exc_type = type(e)
            caught_metadata = e.metadata
            return 'caught'

    result = await agent.run('go')
    assert result.output == 'final'
    # The new CallDeferred exception should surface, not the original ApprovalRequired
    assert caught_exc_type is CallDeferred
    assert caught_metadata == {'reason': 'external-after-approval'}


async def test_deferred_tool_handler_via_handle_call_handler_resolves_wrong_id():
    """handle_call path: handler returns results for wrong ID → remaining non-empty → raises original exc."""
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Resolve a non-existent ID — our tool's ID stays in remaining
        return DeferredToolResults(approvals={'wrong_id': True})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ApprovalRequired:
            return 'caught'

    result = await agent.run('go')
    assert result.output == 'final'


async def test_deferred_tool_handler_via_hooks_decorator():
    """`@hooks.on.deferred_tool_calls` resolves deferred calls inline."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done!')])

    hooks = Hooks()

    @hooks.on.deferred_tool_calls
    async def handler(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done!'


async def test_deferred_tool_handler_via_hooks_constructor_kwarg_and_accumulation():
    """`Hooks(deferred_tool_calls=...)` accumulates results across registered handlers."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, tool_call_id='a1'),
                    ToolCallPart('tool_b', {}, tool_call_id='b1'),
                    ToolCallPart('tool_c', {}, tool_call_id='c1'),
                ]
            )
        return ModelResponse(parts=[TextPart('All done.')])

    def handle_a(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    hooks = Hooks(deferred_tool_calls=handle_a)

    @hooks.on.deferred_tool_calls
    async def handle_rest(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # tool_a was already resolved by handle_a; this handler sees only tool_b and tool_c
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    @hooks.on.deferred_tool_calls
    async def never_called(  # pragma: no cover
        ctx: RunContext, *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        # All calls should already be resolved by the previous handler — this is the early-break path
        raise AssertionError('Should not be called: all requests already resolved')

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b'

    @agent.tool
    def tool_c(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'c'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_hooks_returns_none_when_unhandled():
    """`Hooks` returns None from the deferred-tool-calls hook when no registered handler resolves anything."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    hooks = Hooks()

    @hooks.on.deferred_tool_calls
    async def declines(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        return None

    @hooks.on.deferred_tool_calls
    async def empty(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # Empty results count as "didn't handle"
        return DeferredToolResults()

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[hooks],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    # Falls through to bubble-up since no handler resolved anything
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
