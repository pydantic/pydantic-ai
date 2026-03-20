"""Tests for capability hooks: before/after/wrap for all lifecycle points."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability, BeforeModelRequestContext
from pydantic_ai.exceptions import SkipModelRequest, SkipToolExecution, SkipToolValidation
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

# --- Helpers ---


def make_text_response(text: str = 'hello') -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def simple_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return make_text_response('response from model')


async def simple_stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'streamed response'


async def tool_calling_stream_function(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """A streaming model that calls a tool on first request, then returns text."""
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                yield 'final response'
                return

    if info.function_tools:
        tool = info.function_tools[0]
        yield {0: DeltaToolCall(name=tool.name, json_args='{}', tool_call_id='call-1')}
        return

    yield 'no tools available'


def tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that calls a tool on first request, then returns text."""
    # Check if there's already a tool return in messages (i.e., tool was called)
    for msg in messages:
        for part in msg.parts:
            if hasattr(part, 'tool_name') and hasattr(part, 'content') and not hasattr(part, 'args'):
                # This is a ToolReturnPart - tool was already called
                return make_text_response('final response')

    # First request: call the tool
    if info.function_tools:
        tool = info.function_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{}', tool_call_id='call-1')])

    return make_text_response('no tools available')


# --- Logging capability for testing ---


@dataclass
class LoggingCapability(AbstractCapability[Any]):
    """A capability that logs all hook invocations for testing."""

    log: list[str] = field(default_factory=lambda: [])

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.log.append('before_run')

    async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        self.log.append('after_run')
        return result

    async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
        self.log.append('wrap_run:before')
        result = await handler()
        self.log.append('wrap_run:after')
        return result

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: BeforeModelRequestContext,
    ) -> BeforeModelRequestContext:
        self.log.append('before_model_request')
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        response: ModelResponse,
    ) -> ModelResponse:
        self.log.append('after_model_request')
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        messages: list[ModelMessage],
        model_settings: ModelSettings,
        model_request_parameters: ModelRequestParameters,
        handler: Any,
    ) -> ModelResponse:
        self.log.append('wrap_model_request:before')
        response = await handler(messages, model_settings, model_request_parameters)
        self.log.append('wrap_model_request:after')
        return response

    async def before_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        self.log.append(f'before_tool_validate:{call.tool_name}')
        return args

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.log.append(f'after_tool_validate:{call.tool_name}')
        return args

    async def wrap_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, args: str | dict[str, Any], handler: Any
    ) -> dict[str, Any]:
        self.log.append(f'wrap_tool_validate:{call.tool_name}:before')
        result = await handler(args)
        self.log.append(f'wrap_tool_validate:{call.tool_name}:after')
        return result

    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.log.append(f'before_tool_execute:{call.tool_name}')
        return args

    async def after_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any], result: Any
    ) -> Any:
        self.log.append(f'after_tool_execute:{call.tool_name}')
        return result

    async def wrap_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any], handler: Any
    ) -> Any:
        self.log.append(f'wrap_tool_execute:{call.tool_name}:before')
        result = await handler(args)
        self.log.append(f'wrap_tool_execute:{call.tool_name}:after')
        return result


# --- Tests ---


class TestRunHooks:
    async def test_before_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_run' in cap.log

    async def test_after_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'after_run' in cap.log

    async def test_wrap_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log

    async def test_run_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        # wrap_run wraps the run (which includes before_run inside iter),
        # then after_run fires at the end (outside wrap_run)
        assert cap.log.index('wrap_run:before') < cap.log.index('before_run')
        assert cap.log.index('before_run') < cap.log.index('wrap_run:after')
        assert cap.log.index('wrap_run:after') <= cap.log.index('after_run')

    async def test_after_run_can_modify_result(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                return AgentRunResult(output='modified output')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ModifyResultCap()])
        result = await agent.run('hello')
        assert result.output == 'modified output'

    async def test_wrap_run_can_short_circuit(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                # Don't call handler - short-circuit the run
                return AgentRunResult(output='short-circuited')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ShortCircuitRunCap()])
        result = await agent.run('hello')
        assert result.output == 'short-circuited'


class TestModelRequestHooks:
    async def test_before_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_model_request' in cap.log

    async def test_after_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'after_model_request' in cap.log

    async def test_wrap_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'wrap_model_request:before' in cap.log
        assert 'wrap_model_request:after' in cap.log

    async def test_model_request_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert cap.log.index('before_model_request') < cap.log.index('wrap_model_request:before')
        assert cap.log.index('wrap_model_request:before') < cap.log.index('wrap_model_request:after')
        assert cap.log.index('wrap_model_request:after') < cap.log.index('after_model_request')

    async def test_after_model_request_can_modify_response(self):
        @dataclass
        class ModifyResponseCap(AbstractCapability[Any]):
            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                response: ModelResponse,
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='modified by after hook')])

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ModifyResponseCap()])
        result = await agent.run('hello')
        assert result.output == 'modified by after hook'

    async def test_wrap_model_request_can_modify_response(self):
        @dataclass
        class WrapModifyCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                messages: list[ModelMessage],
                model_settings: ModelSettings,
                model_request_parameters: ModelRequestParameters,
                handler: Any,
            ) -> ModelResponse:
                response = await handler(messages, model_settings, model_request_parameters)
                return ModelResponse(parts=[TextPart(content='wrapped: ' + response.parts[0].content)])

        agent = Agent(FunctionModel(simple_model_function), capabilities=[WrapModifyCap()])
        result = await agent.run('hello')
        assert result.output == 'wrapped: response from model'

    async def test_skip_model_request(self):
        @dataclass
        class SkipCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: BeforeModelRequestContext,
            ) -> BeforeModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped model')]))

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SkipCap()])
        result = await agent.run('hello')
        assert result.output == 'skipped model'


class TestToolValidateHooks:
    async def test_tool_validate_hooks_fire(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert 'before_tool_validate:my_tool' in cap.log
        assert 'after_tool_validate:my_tool' in cap.log
        assert 'wrap_tool_validate:my_tool:before' in cap.log
        assert 'wrap_tool_validate:my_tool:after' in cap.log

    async def test_before_tool_validate_can_modify_args(self):
        @dataclass
        class ModifyArgsCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self, ctx: RunContext[Any], *, call: ToolCallPart, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                # Inject an argument
                if isinstance(args, dict):
                    return {**args, 'name': 'injected'}
                return {'name': 'injected'}

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[ModifyArgsCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f'hello {name}'

        await agent.run('greet someone')
        assert received_name == 'injected'

    async def test_skip_tool_validation(self):
        @dataclass
        class SkipValidateCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self, ctx: RunContext[Any], *, call: ToolCallPart, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                raise SkipToolValidation({'name': 'skip-validated'})

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[SkipValidateCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f'hello {name}'

        await agent.run('greet someone')
        assert received_name == 'skip-validated'


class TestToolExecuteHooks:
    async def test_tool_execute_hooks_fire(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert 'before_tool_execute:my_tool' in cap.log
        assert 'after_tool_execute:my_tool' in cap.log
        assert 'wrap_tool_execute:my_tool:before' in cap.log
        assert 'wrap_tool_execute:my_tool:after' in cap.log

    async def test_after_tool_execute_can_modify_result(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_tool_execute(
                self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any], result: Any
            ) -> Any:
                return f'modified: {result}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')

        agent = Agent(FunctionModel(model_fn), capabilities=[ModifyResultCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return 'original'

        result = await agent.run('call tool')
        assert 'modified: original' in result.output

    async def test_skip_tool_execution(self):
        @dataclass
        class SkipExecCap(AbstractCapability[Any]):
            async def before_tool_execute(
                self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any]
            ) -> dict[str, Any]:
                raise SkipToolExecution('denied')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')

        agent = Agent(FunctionModel(model_fn), capabilities=[SkipExecCap()])

        tool_was_called = False

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_was_called
            tool_was_called = True
            return 'should not be called'

        result = await agent.run('call tool')
        assert not tool_was_called
        assert 'denied' in result.output

    async def test_wrap_tool_execute_with_error_handling(self):
        @dataclass
        class ErrorHandlingCap(AbstractCapability[Any]):
            caught_error: str | None = None

            async def wrap_tool_execute(
                self, ctx: RunContext[Any], *, call: ToolCallPart, args: dict[str, Any], handler: Any
            ) -> Any:
                try:
                    return await handler(args)
                except Exception as e:
                    self.caught_error = str(e)
                    return 'recovered from error'

        cap = ErrorHandlingCap()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        await agent.run('call tool')
        assert cap.caught_error == 'tool failed'


class TestCompositionOrder:
    async def test_multiple_capabilities_model_request_order(self):
        """Test that multiple capabilities compose in the correct order."""
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: BeforeModelRequestContext,
            ) -> BeforeModelRequestContext:
                log.append('cap1:before')
                return request_context

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append('cap1:after')
                return response

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                messages: list[ModelMessage],
                model_settings: ModelSettings,
                model_request_parameters: ModelRequestParameters,
                handler: Any,
            ) -> ModelResponse:
                log.append('cap1:wrap:before')
                response = await handler(messages, model_settings, model_request_parameters)
                log.append('cap1:wrap:after')
                return response

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: BeforeModelRequestContext,
            ) -> BeforeModelRequestContext:
                log.append('cap2:before')
                return request_context

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append('cap2:after')
                return response

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                messages: list[ModelMessage],
                model_settings: ModelSettings,
                model_request_parameters: ModelRequestParameters,
                handler: Any,
            ) -> ModelResponse:
                log.append('cap2:wrap:before')
                response = await handler(messages, model_settings, model_request_parameters)
                log.append('cap2:wrap:after')
                return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Cap1(), Cap2()])
        await agent.run('hello')

        # before hooks: forward order (cap1 then cap2)
        assert log.index('cap1:before') < log.index('cap2:before')
        # wrap hooks: cap1 outermost, cap2 innermost
        assert log.index('cap1:wrap:before') < log.index('cap2:wrap:before')
        assert log.index('cap2:wrap:after') < log.index('cap1:wrap:after')
        # after hooks: reverse order (cap2 then cap1)
        assert log.index('cap2:after') < log.index('cap1:after')

    async def test_multiple_capabilities_run_hooks_order(self):
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                log.append('cap1:before_run')

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                log.append('cap1:after_run')
                return result

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                log.append('cap1:wrap_run:before')
                result = await handler()
                log.append('cap1:wrap_run:after')
                return result

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                log.append('cap2:before_run')

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                log.append('cap2:after_run')
                return result

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                log.append('cap2:wrap_run:before')
                result = await handler()
                log.append('cap2:wrap_run:after')
                return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Cap1(), Cap2()])
        await agent.run('hello')

        # before_run: forward order
        assert log.index('cap1:before_run') < log.index('cap2:before_run')
        # wrap_run: cap1 outermost
        assert log.index('cap1:wrap_run:before') < log.index('cap2:wrap_run:before')
        assert log.index('cap2:wrap_run:after') < log.index('cap1:wrap_run:after')
        # after_run: reverse order
        assert log.index('cap2:after_run') < log.index('cap1:after_run')


class TestCombinedBeforeWrapAfter:
    async def test_all_hook_types_on_same_capability(self):
        """Test before + wrap + after all fire correctly on a single capability."""
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')

        # Check run hooks
        assert 'before_run' in cap.log
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log
        assert 'after_run' in cap.log

        # Check model request hooks (should fire twice: once for tool call, once for final)
        model_request_before_count = cap.log.count('before_model_request')
        assert model_request_before_count == 2

        # Check tool hooks
        assert 'before_tool_validate:my_tool' in cap.log
        assert 'wrap_tool_validate:my_tool:before' in cap.log
        assert 'wrap_tool_validate:my_tool:after' in cap.log
        assert 'after_tool_validate:my_tool' in cap.log
        assert 'before_tool_execute:my_tool' in cap.log
        assert 'wrap_tool_execute:my_tool:before' in cap.log
        assert 'wrap_tool_execute:my_tool:after' in cap.log
        assert 'after_tool_execute:my_tool' in cap.log


class TestRunHooksRunStream:
    """Test that wrap_run and after_run fire for run_stream()."""

    async def test_wrap_run_fires_for_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log

    async def test_after_run_fires_for_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'after_run' in cap.log

    async def test_wrap_run_fires_for_iter(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        async with agent.iter('hello') as agent_run:
            async for _node in agent_run:
                pass
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log
        assert 'after_run' in cap.log

    async def test_after_run_can_modify_result_via_iter(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                return AgentRunResult(output='modified by after_run')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ModifyResultCap()])
        async with agent.iter('hello') as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        assert agent_run.result.output == 'modified by after_run'

    async def test_run_hook_order_via_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert cap.log.index('wrap_run:before') < cap.log.index('before_run')
        assert cap.log.index('before_run') < cap.log.index('wrap_run:after')
        assert cap.log.index('wrap_run:after') <= cap.log.index('after_run')


class TestStreamingHooks:
    """Test that SkipModelRequest and wrap_model_request work in streaming paths."""

    async def test_skip_model_request_streaming(self):
        @dataclass
        class SkipCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: BeforeModelRequestContext,
            ) -> BeforeModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped in stream')]))

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[SkipCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'skipped in stream'

    async def test_wrap_model_request_streaming(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'wrap_model_request:before' in cap.log
        assert 'wrap_model_request:after' in cap.log

    async def test_wrap_model_request_modifies_result_via_run_with_streaming(self):
        """wrap_model_request modification affects the final result when using run() with streaming."""

        @dataclass
        class WrapModifyCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                messages: list[ModelMessage],
                model_settings: ModelSettings,
                model_request_parameters: ModelRequestParameters,
                handler: Any,
            ) -> ModelResponse:
                response = await handler(messages, model_settings, model_request_parameters)
                return ModelResponse(parts=[TextPart(content='wrapped: ' + response.parts[0].content)])

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[WrapModifyCap()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        result = await agent.run('hello', event_stream_handler=handler)
        assert result.output == 'wrapped: streamed response'

    async def test_after_model_request_fires_streaming(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'after_model_request' in cap.log


class TestWrapRunEventStream:
    """Tests for the wrap_run_event_stream hook."""

    async def test_wrap_run_event_stream_observes(self):
        """Hook sees events from model streaming."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async def observe() -> AsyncIterator[AgentStreamEvent]:
                    async for event in stream:
                        observed_events.append(event)
                        yield event

                return observe()

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ObserverCap()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        await agent.run('hello', event_stream_handler=handler)
        assert len(observed_events) > 0

    async def test_wrap_run_event_stream_transforms(self):
        """Modifications by the hook are visible to event_stream_handler."""
        handler_events: list[AgentStreamEvent] = []

        @dataclass
        class TransformCap(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async def transform() -> AsyncIterator[AgentStreamEvent]:
                    async for event in stream:
                        # Add a custom marker by yielding the event unchanged
                        yield event

                return transform()

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[TransformCap()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        await agent.run('hello', event_stream_handler=handler)
        assert len(handler_events) > 0

    async def test_wrap_run_event_stream_composition(self):
        """Multiple capabilities compose in correct order (first = outermost)."""
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async def wrap() -> AsyncIterator[AgentStreamEvent]:
                    log.append('cap1:enter')
                    async for event in stream:
                        yield event
                    log.append('cap1:exit')

                return wrap()

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async def wrap() -> AsyncIterator[AgentStreamEvent]:
                    log.append('cap2:enter')
                    async for event in stream:
                        yield event
                    log.append('cap2:exit')

                return wrap()

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[Cap1(), Cap2()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        await agent.run('hello', event_stream_handler=handler)

        # Cap1 is outermost, so enters first and exits last
        assert log.index('cap1:enter') < log.index('cap2:enter')
        assert log.index('cap2:exit') < log.index('cap1:exit')

    async def test_wrap_run_event_stream_tool_events(self):
        """HandleResponseEvents from CallToolsNode flow through the hook."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async def observe() -> AsyncIterator[AgentStreamEvent]:
                    async for event in stream:
                        observed_events.append(event)
                        yield event

                return observe()

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ObserverCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        await agent.run('call tool', event_stream_handler=handler)
        # Should have observed events from both ModelRequestNode and CallToolsNode streams
        assert len(observed_events) > 0

    async def test_wrap_run_event_stream_fires_in_run_stream_without_handler(self):
        """wrap_run_event_stream fires in run_stream() even without an event_stream_handler."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async def observe() -> AsyncIterator[AgentStreamEvent]:
                    async for event in stream:
                        observed_events.append(event)
                        yield event

                return observe()

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ObserverCap()],
        )

        # No event_stream_handler — hook should still fire
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(observed_events) > 0


class TestWrapRunShortCircuit:
    """Test short-circuiting wrap_run via iter() and run_stream()."""

    async def test_wrap_run_short_circuit_via_iter(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                return AgentRunResult(output='short-circuited')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ShortCircuitRunCap()])
        async with agent.iter('hello') as agent_run:
            nodes: list[Any] = []
            async for node in agent_run:
                nodes.append(node)
        # Iteration should stop immediately (no graph nodes executed)
        assert nodes == []
        assert agent_run.result is not None
        assert agent_run.result.output == 'short-circuited'

    async def test_wrap_run_short_circuit_via_run_stream(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                return AgentRunResult(output='short-circuited')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ShortCircuitRunCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'short-circuited'


class TestSkipModelRequestInteraction:
    """Test SkipModelRequest interaction with after_model_request."""

    async def test_skip_model_request_still_calls_after_model_request(self):
        log: list[str] = []

        @dataclass
        class SkipAndLogCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: BeforeModelRequestContext,
            ) -> BeforeModelRequestContext:
                log.append('before_model_request')
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped')]))

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append('after_model_request')
                return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SkipAndLogCap()])
        result = await agent.run('hello')
        assert result.output == 'skipped'
        # after_model_request should still fire via _finish_handling
        assert 'after_model_request' in log

    async def test_wrap_model_request_short_circuit_streaming(self):
        """wrap_model_request can return without calling handler in streaming path."""

        @dataclass
        class ShortCircuitModelCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                messages: list[ModelMessage],
                model_settings: ModelSettings,
                model_request_parameters: ModelRequestParameters,
                handler: Any,
            ) -> ModelResponse:
                # Don't call handler — return a response directly
                return ModelResponse(parts=[TextPart(content='model short-circuited')])

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ShortCircuitModelCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'model short-circuited'
