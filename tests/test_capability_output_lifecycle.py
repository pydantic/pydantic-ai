"""Output lifecycle ordering tests.

Split from test_capabilities.py to keep that module below the repository file-size limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputContext, PromptedOutput, TextOutput
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .capability_models import make_text_response
from .conftest import IsDatetime, IsStr

pytestmark = pytest.mark.anyio


class MyOutput(BaseModel):
    value: int


class TestOutputHookFullLifecycle:
    """Test the full output hook lifecycle fires in the correct order."""

    async def test_wrap_output_validate_encloses_before_and_after(self):
        call_order: list[str] = []

        @dataclass
        class LifecycleCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                call_order.append('before')
                return output

            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                call_order.append('after')
                return output

            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                call_order.append('wrap:before')
                result = await handler(output)
                call_order.append('wrap:after')
                return result

        agent = Agent(
            FunctionModel(lambda messages, info: make_text_response('{"value": 1}')),
            output_type=PromptedOutput(MyOutput),
            capabilities=[LifecycleCap()],
        )
        await agent.run('hello')
        assert call_order == ['wrap:before', 'before', 'after', 'wrap:after']

    async def test_wrap_output_process_encloses_before_and_after(self):
        call_order: list[str] = []

        @dataclass
        class LifecycleCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                call_order.append('before')
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                call_order.append('after')
                return output

            async def wrap_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, handler: Any
            ) -> Any:
                call_order.append('wrap:before')
                result = await handler(output)
                call_order.append('wrap:after')
                return result

        agent = Agent(
            FunctionModel(lambda messages, info: make_text_response('hello')),
            capabilities=[LifecycleCap()],
        )
        await agent.run('hello')
        assert call_order == ['wrap:before', 'before', 'after', 'wrap:after']

    async def test_wrap_output_validate_short_circuit_skips_before_and_after(self):
        call_order: list[str] = []

        @dataclass
        class ShortCircuitCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                call_order.append('before')  # pragma: no cover
                return output  # pragma: no cover

            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                call_order.append('after')  # pragma: no cover
                return output  # pragma: no cover

            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                call_order.append('wrap')
                return MyOutput(value=2)

        agent = Agent(
            FunctionModel(lambda messages, info: make_text_response('invalid')),
            output_type=PromptedOutput(MyOutput),
            capabilities=[ShortCircuitCap()],
        )
        result = await agent.run('hello')
        assert result.output == MyOutput(value=2)
        assert call_order == ['wrap']

    async def test_wrap_output_process_short_circuit_skips_before_and_after(self):
        call_order: list[str] = []

        @dataclass
        class ShortCircuitCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                call_order.append('before')  # pragma: no cover
                return output  # pragma: no cover

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                call_order.append('after')  # pragma: no cover
                return output  # pragma: no cover

            async def wrap_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, handler: Any
            ) -> Any:
                call_order.append('wrap')
                return 'short-circuited'

        agent = Agent(
            FunctionModel(lambda messages, info: make_text_response('hello')),
            capabilities=[ShortCircuitCap()],
        )
        result = await agent.run('hello')
        assert result.output == 'short-circuited'
        assert call_order == ['wrap']

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
            'wrap_validate:before',
            'before_validate',
            'after_validate',
            'wrap_validate:after',
            'wrap_execute:before',
            'before_execute',
            'after_execute',
            'wrap_execute:after',
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
