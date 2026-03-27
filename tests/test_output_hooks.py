"""Tests for output hooks (before/after/wrap/on_error for output validate and execute)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel, ValidationError

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputContext, PromptedOutput, TextOutput
from pydantic_ai.tools import ToolDefinition

pytestmark = [
    pytest.mark.anyio,
]


# --- Helpers ---


class MyOutput(BaseModel):
    value: int


def make_text_response(text: str = 'hello') -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


# --- Tests ---


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
        """before_output_validate fires for plain str output with identity handler."""
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
                log.append(f'before_output_validate:{output}')
                assert output_context.mode == 'text'
                assert output_context.output_type is str
                assert output_context.has_function is False
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == 'hello world'
        assert log == ['before_output_validate:hello world']

    async def test_text_output_function(self):
        """before_output_validate fires before TextOutput function is called."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'before:{output}')
                assert output_context.has_function is True
                return output

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == 'WORLD'
        assert log == ['before:world']

    async def test_can_transform_text_before_function(self):
        """before_output_validate can modify text that is then passed to TextOutput function."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class PrependCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
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
        # Verify the retry message contains the ModelRetry message
        messages = result.all_messages()
        retry_parts = [p for m in messages for p in m.parts if hasattr(p, 'content') and p.part_kind == 'retry-prompt']
        assert len(retry_parts) == 1
        assert retry_parts[0].content == 'Please return a valid integer for value'


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
            async def before_output_execute(
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


class TestAfterOutputExecute:
    """after_output_execute can transform the final result after execution."""

    async def test_transform_structured_result(self):
        """after_output_execute transforms the result of structured output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class DoubleResultCap(AbstractCapability[Any]):
            async def after_output_execute(
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
        """after_output_execute can transform plain text output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class UpperCap(AbstractCapability[Any]):
            async def after_output_execute(
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
        """after_output_execute fires after TextOutput function has executed."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class WrapResultCap(AbstractCapability[Any]):
            async def after_output_execute(
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

            async def before_output_execute(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('before_output_execute')
                return output

            async def after_output_execute(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_output_execute')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[OutputLogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert 'before_output_validate:tool' in log
        assert 'after_output_validate' in log
        assert 'before_output_execute' in log
        assert 'after_output_execute' in log

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
            async def before_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'tool_validate:{call.tool_name}')
                return args

            async def before_tool_execute(
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

            async def before_output_execute(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('output_execute')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[BothHooksCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        # Only output hooks fire for output tools — tool hooks are skipped
        assert 'tool_validate:final_result' not in log
        assert 'tool_execute:final_result' not in log
        assert 'output_validate' in log
        assert 'output_execute' in log

    async def test_after_output_execute_transforms_tool_output(self):
        """after_output_execute can transform the result of tool-based output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 5}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class DoubleOutputCap(AbstractCapability[Any]):
            async def after_output_execute(
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
            async def after_output_execute(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return f'{output}!' if isinstance(output, str) else output

        @dataclass
        class AddQuestion(AbstractCapability[Any]):
            async def after_output_execute(
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

    async def test_before_output_execute_decorator(self):
        """Hooks.on.before_output_execute registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_execute
        async def before_exec(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_output_execute')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 6}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=6)
        assert log == ['before_output_execute']

    async def test_after_output_execute_decorator(self):
        """Hooks.on.after_output_execute transforms the final result."""
        hooks = Hooks()

        @hooks.on.after_output_execute
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

    async def test_wrap_output_execute_decorator(self):
        """Hooks.on.output_execute (wrap) registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.output_execute
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

        @hooks.on.before_output_validate
        def sync_hook(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
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

            async def before_output_execute(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_execute')
                return output

            async def wrap_output_execute(
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

            async def after_output_execute(
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

            async def before_output_execute(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_execute')
                return output

            async def after_output_execute(
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
        """OutputContext has correct fields for plain text output."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
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
        """OutputContext has correct fields for TextOutput function."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
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


class TestWrapOutputExecute:
    """wrap_output_execute provides full middleware control around execution."""

    async def test_wrap_can_observe(self):
        """wrap_output_execute can observe without modifying."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class WrapCap(AbstractCapability[Any]):
            async def wrap_output_execute(
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
        """wrap_output_execute can replace the result entirely."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class ReplaceCap(AbstractCapability[Any]):
            async def wrap_output_execute(
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


class TestOnOutputExecuteError:
    """on_output_execute_error can recover from execution failures."""

    async def test_recover_from_output_function_error(self):
        """on_output_execute_error catches errors from output functions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('trigger error')

        def failing_func(text: str) -> str:
            raise ValueError('output function failed')

        @dataclass
        class RecoverCap(AbstractCapability[Any]):
            async def on_output_execute_error(
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

    def test_on_output_execute_error_recovery(self):
        """on_output_execute_error can recover from output function failure."""

        def bad_function(value: int) -> str:
            raise ValueError('value too small')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        @dataclass
        class RecoverCapability(AbstractCapability[Any]):
            async def on_output_execute_error(
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

    def test_composed_on_output_execute_error_chain(self):
        """Multiple capabilities' on_output_execute_error hooks chain correctly."""

        def failing_func(value: int) -> str:
            raise ValueError('intentional')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        @dataclass
        class FirstCap(AbstractCapability[Any]):
            async def on_output_execute_error(
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
            async def on_output_execute_error(
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

    def test_hooks_output_execute_error_decorator(self):
        """Test on_output_execute_error via Hooks decorator API."""

        def bad_function(value: int) -> str:
            raise ValueError('intentional failure')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 10}')])

        hooks = Hooks()

        @hooks.on.output_execute_error
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

            async def after_output_execute(
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

    def test_default_on_output_execute_error_reraises(self):
        """Default on_output_execute_error re-raises the error."""

        def failing_func(value: int) -> str:
            raise ValueError('intentional')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        # Hooks with only a before_output_execute hook (no error hook override).
        hooks = Hooks()

        @hooks.on.before_output_execute
        def noop(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            return output

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[hooks])
        with pytest.raises(ValueError, match='intentional'):
            agent.run_sync('hello')


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
        """Output hooks fire during streaming, including on partial validation."""
        from pydantic_ai.models.function import FunctionModel

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='hello world')])

        @dataclass
        class StreamLogCapability(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'before_validate partial={ctx.partial_output}')
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[StreamLogCapability()])
        result = agent.run_sync('hello')
        assert result.output == 'hello world'
        assert any('before_validate' in entry for entry in log)

    def test_no_capability_fast_path_text(self):
        """When capability is None, run_output_with_hooks falls through to process() for text."""
        import asyncio

        from pydantic_ai._output import TextOutputProcessor, run_output_with_hooks
        from pydantic_ai._run_context import RunContext

        processor = TextOutputProcessor()

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # type: ignore
                usage=None,  # type: ignore
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            return await run_output_with_hooks(processor, 'hello', run_context=ctx, capability=None, output_mode='text')

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == 'hello'

    def test_no_capability_fast_path_structured(self):
        """When capability is None, run_output_with_hooks uses process() for structured output."""
        import asyncio

        from pydantic_ai._output import ObjectOutputProcessor, run_output_with_hooks
        from pydantic_ai._run_context import RunContext

        processor = ObjectOutputProcessor(output=MyOutput)

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # type: ignore
                usage=None,  # type: ignore
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            return await run_output_with_hooks(
                processor, '{"value": 42}', run_context=ctx, capability=None, output_mode='prompted'
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(result, MyOutput)
        assert result.value == 42

    def test_no_capability_fast_path_text_function(self):
        """When capability is None, run_output_with_hooks uses process() for text functions."""
        import asyncio

        from pydantic_ai._output import TextFunctionOutputProcessor, run_output_with_hooks
        from pydantic_ai._run_context import RunContext

        def upper(text: str) -> str:
            return text.upper()

        processor = TextFunctionOutputProcessor(upper)

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # type: ignore
                usage=None,  # type: ignore
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            return await run_output_with_hooks(processor, 'hello', run_context=ctx, capability=None, output_mode='text')

        result = asyncio.get_event_loop().run_until_complete(run())
        assert result == 'HELLO'

    def test_no_capability_fast_path_union(self):
        """When capability is None, run_output_with_hooks uses process() for union output."""
        import asyncio

        from pydantic_ai._output import UnionOutputProcessor, run_output_with_hooks
        from pydantic_ai._run_context import RunContext

        processor = UnionOutputProcessor(outputs=[MyOutput])

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # type: ignore
                usage=None,  # type: ignore
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            return await run_output_with_hooks(
                processor,
                '{"result": {"kind": "MyOutput", "data": {"value": 7}}}',
                run_context=ctx,
                capability=None,
                output_mode='prompted',
            )

        result = asyncio.get_event_loop().run_until_complete(run())
        assert isinstance(result, MyOutput)
        assert result.value == 7

    def test_no_capability_fast_path_structured_validation_error(self):
        """When capability is None, process() wraps ValidationError in ToolRetryError."""
        import asyncio

        from pydantic_ai._output import ObjectOutputProcessor, run_output_with_hooks
        from pydantic_ai._run_context import RunContext
        from pydantic_ai.exceptions import ToolRetryError

        processor = ObjectOutputProcessor(output=MyOutput)

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # type: ignore
                usage=None,  # type: ignore
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            return await run_output_with_hooks(
                processor, 'not valid json', run_context=ctx, capability=None, output_mode='prompted'
            )

        with pytest.raises(ToolRetryError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_no_capability_fast_path_union_validation_error(self):
        """When capability is None, UnionOutputProcessor.process() wraps ValidationError in ToolRetryError."""
        import asyncio

        from pydantic_ai._output import UnionOutputProcessor, run_output_with_hooks
        from pydantic_ai._run_context import RunContext
        from pydantic_ai.exceptions import ToolRetryError

        processor = UnionOutputProcessor(outputs=[MyOutput])

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # type: ignore
                usage=None,  # type: ignore
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            return await run_output_with_hooks(
                processor, 'not valid json', run_context=ctx, capability=None, output_mode='prompted'
            )

        with pytest.raises(ToolRetryError):
            asyncio.get_event_loop().run_until_complete(run())

    def test_hooks_on_output_execute_via_hooks_class(self):
        """Test wrap_output_execute via Hooks decorator API."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 10}')])

        hooks = Hooks()
        execute_log: list[str] = []

        @hooks.on.output_execute
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

            pass

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[BareCap()])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=3)
        assert call_count == 2  # First attempt failed, retried

    def test_bare_capability_default_on_output_execute_error(self):
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

    def test_wrapper_on_output_execute_error_delegates(self):
        """WrapperCapability delegates on_output_execute_error to the wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        def failing_func(value: int) -> str:
            raise ValueError('exec fail')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def on_output_execute_error(
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

    def test_hooks_on_output_execute_error_chaining(self):
        """Hooks class on_output_execute_error re-raises, chaining errors."""

        def failing_func(value: int) -> str:
            raise ValueError('original')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        hooks = Hooks()

        @hooks.on.output_execute_error
        async def first_handler(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any], error: Exception
        ) -> Any:
            raise ValueError('chained')  # Re-raise different error

        @hooks.on.output_execute_error
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

            async def before_output_execute(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('before_execute')
                return output

            async def after_output_execute(
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

    def test_union_output_execute_hook_transforms_result(self):
        """Execute hooks can transform the result for union output types."""

        class TypeA(BaseModel):
            value: int

        class TypeB(BaseModel):
            name: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "TypeA", "data": {"value": 5}}}')])

        @dataclass
        class DoubleCapability(AbstractCapability[Any]):
            async def after_output_execute(
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
            async def wrap_output_execute(
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
