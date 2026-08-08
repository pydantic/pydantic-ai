"""Tests for capability hooks (`LoggingCapability`, node/run/tool hook lifecycle).

Split out of `test_capabilities.py`, which had grown past the repository's file-size limit.
"""

from __future__ import annotations

import asyncio
import contextvars
import re
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from types import NoneType
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_ai._run_context import RunContext
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    ToolSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.hooks import Hooks, HookTimeoutError
from pydantic_ai.exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    ToolFailed,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelRequestContext,
    ModelResolutionContext,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import ToolOutput
from pydantic_ai.result import FinalResult
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .capability_models import (
    make_text_response,
    simple_model_function,
    simple_stream_function,
    tool_calling_model,
    tool_calling_stream_function,
)
from .conftest import IsDatetime, IsStr

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


class MyOutput(BaseModel):
    value: int


def _build_run_context(deps: Any = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), run_step=0)

# --- Hooks test helpers ---


@dataclass
class _ReplacingCapability(AbstractCapability[Any]):
    """Capability that replaces ModelRequestNode with a fresh copy in before_node_run.

    Used to test that streaming + node replacement doesn't cause double model execution.
    """

    replaced: bool = field(default=False, init=False)

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        from pydantic_ai import ModelRequestNode

        if isinstance(node, ModelRequestNode) and not self.replaced:
            self.replaced = True
            return ModelRequestNode(request=node.request)  # pyright: ignore[reportUnknownVariableType]
        return node  # pyright: ignore[reportUnknownVariableType]


# Defined at module scope so pydantic-ai can resolve the annotation under `from __future__ import annotations`.
class SingleBaseModelArg(BaseModel):
    label: str = 'default'


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
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self.log.append('before_model_request')
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self.log.append('after_model_request')
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: Any,
        handler: Any,
    ) -> ModelResponse:
        self.log.append('wrap_model_request:before')
        response = await handler(request_context)
        self.log.append('wrap_model_request:after')
        return response

    async def before_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        self.log.append(f'before_tool_validate:{call.tool_name}')
        return args

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.log.append(f'after_tool_validate:{call.tool_name}')
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: Any,
    ) -> dict[str, Any]:
        self.log.append(f'wrap_tool_validate:{call.tool_name}:before')
        result = await handler(args)
        self.log.append(f'wrap_tool_validate:{call.tool_name}:after')
        return result

    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.log.append(f'before_tool_execute:{call.tool_name}')
        return args

    async def after_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], result: Any
    ) -> Any:
        self.log.append(f'after_tool_execute:{call.tool_name}')
        return result

    async def wrap_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], handler: Any
    ) -> Any:
        self.log.append(f'wrap_tool_execute:{call.tool_name}:before')
        result = await handler(args)
        self.log.append(f'wrap_tool_execute:{call.tool_name}:after')
        return result

    async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
        self.log.append('on_run_error')
        raise error

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        self.log.append(f'before_node_run:{type(node).__name__}')
        return node

    async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
        self.log.append(f'after_node_run:{type(node).__name__}')
        return result

    async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
        self.log.append(f'on_node_run_error:{type(node).__name__}')
        raise error

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        self.log.append('on_model_request_error')
        raise error

    async def on_tool_validate_error(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
    ) -> dict[str, Any]:
        self.log.append(f'on_tool_validate_error:{call.tool_name}')
        raise error

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        self.log.append(f'on_tool_execute_error:{call.tool_name}')
        raise error


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

    async def test_wrap_run_can_recover_from_error(self):
        """wrap_run can catch errors from handler() and return a recovery result."""

        @dataclass
        class ErrorRecoveryCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    return AgentRunResult(output='recovered from error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[ErrorRecoveryCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered from error'

    async def test_wrap_run_error_propagates_without_recovery(self):
        """Without recovery in wrap_run, errors propagate normally."""

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model))
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_wrap_run_recovery_via_iter(self):
        """wrap_run error recovery works when using agent.iter() too."""

        @dataclass
        class ErrorRecoveryCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    return AgentRunResult(output='recovered via iter')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[ErrorRecoveryCap()])
        async with agent.iter('hello') as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        assert agent_run.result.output == 'recovered via iter'


class TestModelRequestHooks:
    async def test_before_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_model_request' in cap.log

    @pytest.mark.parametrize(
        ('mode', 'streaming'),
        [('run', False), ('run_stream', True), ('event_stream_handler', True)],
    )
    async def test_before_model_request_sees_selection_context(self, mode: str, streaming: bool):
        """`before_model_request` sees the selected model ID and effective streaming mode."""
        contexts: list[ModelRequestContext] = []

        @dataclass
        class CaptureContext(AbstractCapability[None]):
            async def before_model_request(
                self,
                ctx: RunContext[None],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                contexts.append(request_context)
                return request_context

        agent = Agent('test', deps_type=type(None), capabilities=[CaptureContext()], defer_model_check=True)
        if mode == 'run_stream':
            async with agent.run_stream('hello') as result:
                await result.get_output()
        elif mode == 'event_stream_handler':

            async def handle_events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
                async for _ in stream:
                    pass

            await agent.run('hello', event_stream_handler=handle_events)
        else:
            await agent.run('hello')

        assert [(context.model_id, context.streaming) for context in contexts] == [('test', streaming)]

    async def test_withdrawn_bootstrap_model_id_does_not_leak_to_default(self):
        """A bootstrap model contribution withdrawn by `for_run` must not leak its selection string as provenance."""
        model_ids: list[str | None] = []

        @dataclass
        class BootstrapModel(AbstractCapability[None]):
            def get_model(self) -> str:
                return 'bootstrap-alias'

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return AbstractCapability()

            async def resolve_model_id(
                self, ctx: ModelResolutionContext[None], *, model_id: KnownModelName | str
            ) -> Model | None:
                return TestModel() if model_id == 'bootstrap-alias' else None

        @dataclass
        class CaptureModelId(AbstractCapability[None]):
            async def before_model_request(
                self,
                ctx: RunContext[None],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                model_ids.append(request_context.model_id)
                return request_context

        agent = Agent(TestModel(), deps_type=NoneType, capabilities=[BootstrapModel(), CaptureModelId()])
        await agent.run('hello')

        assert model_ids == [None]

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
                request_context: ModelRequestContext,
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
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                response = await handler(request_context)
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
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped model')]))

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SkipCap()])
        result = await agent.run('hello')
        assert result.output == 'skipped model'

    async def test_before_model_request_swaps_model(self):
        call_log: list[str] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_log.append('swap_model')
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapModelCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                request_context.model = swap_target
                return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SwapModelCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert call_log == ['swap_model']

    async def test_wrap_model_request_swaps_model(self):
        call_log: list[str] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_log.append('swap_model')
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapInWrapCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
            ) -> ModelResponse:
                request_context.model = swap_target
                return await handler(request_context)

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SwapInWrapCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert call_log == ['swap_model']

    async def test_before_model_request_swaps_model_streaming(self):
        call_log: list[str] = []

        async def swap_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            call_log.append('swap_stream')
            yield 'from swap stream'

        swap_target = FunctionModel(stream_function=swap_stream_fn)

        @dataclass
        class SwapModelCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                request_context.model = swap_target
                return request_context

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[SwapModelCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'from swap stream'
        assert call_log == ['swap_stream']

    async def test_run_context_model_unchanged_after_swap(self):
        observed_models: list[Any] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('from swap model')

        original_model = FunctionModel(simple_model_function)
        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapAndObserveCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                observed_models.append(ctx.model)
                request_context.model = swap_target
                return request_context

        agent = Agent(original_model, capabilities=[SwapAndObserveCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert observed_models[0] is original_model

    async def test_hooks_before_model_request_swaps_model(self):
        call_log: list[str] = []
        hooks = Hooks()

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_log.append('swap_model')
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @hooks.on.before_model_request
        async def _(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            request_context.model = swap_target
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert call_log == ['swap_model']

    async def test_after_model_request_sees_wrap_swap(self):
        """after_model_request sees the model swapped during wrap_model_request."""
        after_models: list[Any] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapInWrapAndObserveCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
            ) -> ModelResponse:
                request_context.model = swap_target
                return await handler(request_context)

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                after_models.append(request_context.model)
                return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SwapInWrapAndObserveCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert after_models[0] is swap_target


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
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                # Inject an argument
                if isinstance(args, dict):
                    return {**args, 'name': 'injected'}  # pragma: no cover
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
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
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

    async def test_tool_def_matches_called_tool(self):
        """Verify tool_def is the correct ToolDefinition for the tool being called."""
        received_tool_defs: list[ToolDefinition] = []

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                received_tool_defs.append(tool_def)
                return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[CaptureCap()])

        @agent.tool_plain(description='Say hello')
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert len(received_tool_defs) == 1
        td = received_tool_defs[0]
        assert td.name == 'my_tool'
        assert td.description == 'Say hello'
        assert td.kind == 'function'


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
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
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
            return make_text_response('no tools')  # pragma: no cover

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
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
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
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[SkipExecCap()])

        tool_was_called = False

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_was_called
            tool_was_called = True  # pragma: no cover
            return 'should not be called'  # pragma: no cover

        result = await agent.run('call tool')
        assert not tool_was_called
        assert 'denied' in result.output

    async def test_wrap_tool_execute_with_error_handling(self):
        @dataclass
        class ErrorHandlingCap(AbstractCapability[Any]):
            caught_error: str | None = None

            async def wrap_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                handler: Any,
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

    async def test_hooks_receive_dict_args_for_single_base_model_tool(self):
        """Validate and execute hooks receive dict-shaped args when the tool has a single BaseModel parameter.

        The JSON schema sent to the model unwraps the BaseModel, so the model generates its fields at the
        top level. Pydantic's validator returns a BaseModel instance directly, but the framework wraps it
        as `{param_name: model}` so hooks and `call_tool` always see a dict.
        """
        captured_args: list[tuple[str, dict[str, Any]]] = []

        @dataclass
        class CapturingCap(AbstractCapability[Any]):
            async def after_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                captured_args.append(('validate', args))
                return args

            async def wrap_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                handler: Any,
            ) -> Any:
                captured_args.append(('execute', args))
                return await handler(args)

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[CapturingCap()])

        @agent.tool_plain
        def my_tool(payload: SingleBaseModelArg) -> str:
            return f'got {payload.label}'

        await agent.run('call the tool')
        assert captured_args == [
            ('validate', {'payload': SingleBaseModelArg()}),
            ('execute', {'payload': SingleBaseModelArg()}),
        ]

    async def test_tool_hooks_skip_output_tools(self):
        """Tool hooks don't fire for internal output tools (#5111).

        Output tools deliver structured output to the user via `result.output`; they're not
        user-facing tool calls. Firing hooks on them lets e.g. `after_tool_execute` return a
        `ToolReturn` that leaks through to `result.output` instead of the typed value.
        """

        class MyOutput(BaseModel):
            answer: str

        hooks = Hooks()

        @hooks.on.after_tool_execute
        async def wrap_result(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            result: Any,
        ) -> ToolReturn:
            return ToolReturn(return_value=result, content='extra context')

        cap = LoggingCapability()
        agent = Agent(
            TestModel(custom_output_args={'answer': 'hi'}),
            output_type=MyOutput,
            capabilities=[cap, hooks],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        result = await agent.run('call tool and answer')

        # Function tool still fires every tool hook.
        assert 'before_tool_validate:my_tool' in cap.log
        assert 'after_tool_validate:my_tool' in cap.log
        assert 'wrap_tool_validate:my_tool:before' in cap.log
        assert 'wrap_tool_validate:my_tool:after' in cap.log
        assert 'before_tool_execute:my_tool' in cap.log
        assert 'after_tool_execute:my_tool' in cap.log
        assert 'wrap_tool_execute:my_tool:before' in cap.log
        assert 'wrap_tool_execute:my_tool:after' in cap.log
        # Output tool does not appear in any hook log entry.
        assert all('final_result' not in entry for entry in cap.log)
        # Regression for #5111: the ToolReturn from `after_tool_execute` would have corrupted
        # `result.output` if output tool hooks still fired.
        assert result.output == MyOutput(answer='hi')


class TestCompositionOrder:
    async def test_multiple_capabilities_model_request_order(self):
        """Test that multiple capabilities compose in the correct order."""
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append('cap1:before')
                return request_context

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append('cap1:after')
                return response

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                log.append('cap1:wrap:before')
                response = await handler(request_context)
                log.append('cap1:wrap:after')
                return response

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append('cap2:before')
                return request_context

            async def after_model_request(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
            ) -> ModelResponse:
                log.append('cap2:after')
                return response

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                log.append('cap2:wrap:before')
                response = await handler(request_context)
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
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped in stream')]))

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[SkipCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'skipped in stream'

    async def test_skip_model_request_from_wrap_model_request(self):
        """SkipModelRequest raised inside wrap_model_request is handled in non-streaming."""

        @dataclass
        class WrapSkipCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='wrap-skipped')]))

        agent = Agent(FunctionModel(simple_model_function), capabilities=[WrapSkipCap()])
        result = await agent.run('hello')
        assert result.output == 'wrap-skipped'

    async def test_skip_model_request_from_wrap_model_request_streaming(self):
        """SkipModelRequest raised inside wrap_model_request during streaming is handled."""

        @dataclass
        class WrapSkipCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='wrap-skipped in stream')]))

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[WrapSkipCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'wrap-skipped in stream'

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
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                response = await handler(request_context)
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
                async for event in stream:
                    observed_events.append(event)
                    yield event

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
                async for event in stream:
                    yield event

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
                log.append('cap1:enter')
                async for event in stream:
                    yield event
                log.append('cap1:exit')

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                log.append('cap2:enter')
                async for event in stream:
                    yield event
                log.append('cap2:exit')

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
                async for event in stream:
                    observed_events.append(event)
                    yield event

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
                async for event in stream:
                    observed_events.append(event)
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ObserverCap()],
        )

        # No event_stream_handler — hook should still fire
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(observed_events) > 0

    async def test_wrap_run_event_stream_fires_in_run_without_handler(self):
        """wrap_run_event_stream fires in run() even without an event_stream_handler."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async for event in stream:
                    observed_events.append(event)
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ObserverCap()],
        )

        # No event_stream_handler — hook should still fire via forced streaming
        result = await agent.run('hello')
        assert result.output is not None
        assert any(isinstance(e, PartStartEvent) for e in observed_events)


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
                nodes.append(node)  # pragma: no cover
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
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append('before_model_request')
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped')]))

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
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
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
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


class TestPrepareToolsHook:
    async def test_filter_function_tools(self):
        """Capability can filter out function tools by name."""

        @dataclass
        class HideToolCap(AbstractCapability[Any]):
            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [td for td in tool_defs if td.name != 'hidden_tool']

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f'tools: {sorted(tool_names)}')

        agent = Agent(FunctionModel(model_fn), capabilities=[HideToolCap()])

        @agent.tool_plain
        def hidden_tool() -> str:
            return 'hidden'  # pragma: no cover

        @agent.tool_plain
        def visible_tool() -> str:
            return 'visible'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == "tools: ['visible_tool']"

    async def test_receives_function_tools_only(self):
        """`prepare_tools` receives **function** tools only. Output tools route to
        `prepare_output_tools` (with `ctx.max_retries` reflecting the output retry budget)."""

        @dataclass
        class CountKindsCap(AbstractCapability[Any]):
            seen_kinds: list[str] = field(default_factory=list[str])

            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                self.seen_kinds = sorted({td.kind for td in tool_defs})
                return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        cap = CountKindsCap()
        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        await agent.run('hello')
        assert cap.seen_kinds == ['function']

    async def test_modify_tool_description(self):
        """Capability can modify tool descriptions."""
        from dataclasses import replace as dc_replace

        @dataclass
        class PrefixDescriptionCap(AbstractCapability[Any]):
            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [dc_replace(td, description=f'[PREFIXED] {td.description}') for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs = [t.description for t in info.function_tools]
            return make_text_response(f'descriptions: {descs}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrefixDescriptionCap()])

        @agent.tool_plain
        def my_tool() -> str:
            """Original description."""
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        assert '[PREFIXED] Original description.' in result.output

    async def test_chaining_order(self):
        """Multiple capabilities chain prepare_tools in forward order."""

        @dataclass
        class AddSuffixCap(AbstractCapability[Any]):
            suffix: str

            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                from dataclasses import replace as dc_replace

                return [dc_replace(td, description=f'{td.description}{self.suffix}') for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs = [t.description for t in info.function_tools]
            return make_text_response(f'{descs}')

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[AddSuffixCap(suffix='_A'), AddSuffixCap(suffix='_B')],
        )

        @agent.tool_plain
        def tool() -> str:
            """desc"""
            return 'r'  # pragma: no cover

        result = await agent.run('hello')
        # A runs first, then B, so suffix order is _A_B
        assert 'desc_A_B' in result.output


class TestPrepareOutputToolsHook:
    async def test_only_receives_output_tools(self):
        """`prepare_output_tools` receives only output tools — function tools route to
        `prepare_tools`."""

        @dataclass
        class CountKindsCap(AbstractCapability[Any]):
            seen_kinds: list[str] = field(default_factory=list[str])

            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                self.seen_kinds = [td.kind for td in tool_defs]
                return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        cap = CountKindsCap()
        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        await agent.run('hello')
        assert cap.seen_kinds == ['output'], f'expected only output tools, got {cap.seen_kinds}'

    async def test_filter_output_tools(self):
        """Capability can hide output tools from the model."""

        class Out(BaseModel):
            value: str

        @dataclass
        class HideCap(AbstractCapability[Any]):
            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return []  # hide all output tools

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response(f'output_tools: {len(info.output_tools)}')

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, ToolOutput(Out, name='out')],
            capabilities=[HideCap()],
        )

        result = await agent.run('hello')
        assert result.output == 'output_tools: 0'

    async def test_run_context_carries_output_max_retries(self):
        """`prepare_output_tools` ctx.max_retries reflects the agent-level output retry budget,
        matching the contract of output hooks (and unlike `prepare_tools` which doesn't have
        a tool-specific retry budget at preparation time)."""
        seen: list[tuple[int, int]] = []

        @dataclass
        class CaptureCtxCap(AbstractCapability[Any]):
            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                seen.append((ctx.retry, ctx.max_retries))
                return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 7}', tool_call_id='c1')]
            )

        agent = Agent(
            FunctionModel(model_fn),
            output_type=MyOutput,
            retries={'tools': 4, 'output': 4},
            capabilities=[CaptureCtxCap()],
        )
        await agent.run('hello')
        assert seen == [(0, 4)]

    async def test_chaining_order(self):
        """Multiple capabilities chain `prepare_output_tools` in forward order."""
        from dataclasses import replace as dc_replace

        @dataclass
        class AddSuffixCap(AbstractCapability[Any]):
            suffix: str

            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [dc_replace(td, description=f'{td.description or ""}{self.suffix}') for td in tool_defs]

        descs: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs.extend(t.description for t in info.output_tools)
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        agent = Agent(
            FunctionModel(model_fn),
            output_type=MyOutput,
            capabilities=[AddSuffixCap(suffix='_A'), AddSuffixCap(suffix='_B')],
        )
        await agent.run('hello')
        assert descs and descs[0] is not None and descs[0].endswith('_A_B')


class TestWrapNodeRunHook:
    async def test_observe_nodes(self):
        """wrap_node_run can observe all nodes in the agent run."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']

    async def test_observe_nodes_with_tools(self):
        """wrap_node_run fires for each node including tool call round-trips."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('hello')
        # UserPrompt -> ModelRequest (calls tool) -> CallTools (executes tool) ->
        # ModelRequest (gets final response) -> CallTools (produces End)
        assert cap.nodes == [
            'UserPromptNode',
            'ModelRequestNode',
            'CallToolsNode',
            'ModelRequestNode',
            'CallToolsNode',
        ]

    async def test_works_with_iter_next(self):
        """wrap_node_run fires when driving iter() with next()."""
        from pydantic_graph import End

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']

    async def test_bare_async_for_mixed_with_next_does_not_double_run_nodes(self):
        """Advancing inside the loop body doesn't make bare iteration re-run the same node.

        `__anext__` advances the node it last yielded, so a loop body that calls `next()` itself would
        otherwise run that node — and every one of its hooks — a second time. It checks where the graph
        actually is instead, which makes mixing the two drive styles safe rather than silently doubling
        side effects.
        """

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return node

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            async for node in agent_run:
                if not isinstance(node, End):
                    await agent_run.next(node)

        assert cap.nodes == snapshot(['UserPromptNode', 'ModelRequestNode', 'CallToolsNode'])

    async def test_bare_async_for_mixed_with_next_after_wrap_node_run_short_circuit(self):
        """A wrapper short-circuit advances the graph so bare iteration does not run the node again."""

        @dataclass
        class ShortCircuitCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                if Agent.is_model_request_node(node):
                    return End(FinalResult(output='short-circuited'))
                return await handler(node)

        cap = ShortCircuitCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            async for node in agent_run:
                if not isinstance(node, End):
                    await agent_run.next(node)

        assert cap.nodes == snapshot(['UserPromptNode', 'ModelRequestNode'])
        assert agent_run.result is not None
        assert agent_run.result.output == 'short-circuited'

    async def test_bare_async_for_mixed_with_next_after_replacing_node_and_short_circuiting(self):
        """A wrapper short-circuit advances the graph after `before_node_run` replaces the node."""

        @dataclass
        class ReplaceAndShortCircuitCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.nodes.append(type(node).__name__)
                if Agent.is_model_request_node(node):
                    return replace(node)
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                if Agent.is_model_request_node(node):
                    return End(FinalResult(output='short-circuited'))
                return await handler(node)

        cap = ReplaceAndShortCircuitCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            async for node in agent_run:
                if not isinstance(node, End):
                    await agent_run.next(node)

        assert cap.nodes == snapshot(['UserPromptNode', 'ModelRequestNode'])
        assert agent_run.result is not None
        assert agent_run.result.output == 'short-circuited'

    async def test_bare_async_for_mixed_with_next_after_wrap_node_run_recovers_error(self):
        """A wrapper that handles the model error advances the graph past its ErrorMarker."""

        @dataclass
        class RecoverErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                try:
                    return await handler(node)
                except RuntimeError:
                    return End(FinalResult(output='recovered'))

        def model_error(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(model_error), capabilities=[RecoverErrorCap()])

        async with agent.iter('hello') as agent_run:
            async for node in agent_run:
                if not isinstance(node, End):
                    await agent_run.next(node)

        assert agent_run.result is not None
        assert agent_run.result.output == 'recovered'

    async def test_bare_async_for_after_wrap_node_run_retries_a_failed_node(self):
        """A wrapper that recovers from an error by returning a node re-runs it, rather than re-raising."""

        @dataclass
        class RetryOnErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                try:
                    return await handler(node)
                except RuntimeError:
                    # The graph is holding an `ErrorMarker`; hand back the node to run again.
                    return node

        attempts = 0

        def model_error_once(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError('model exploded')
            return ModelResponse(parts=[TextPart(content='second time lucky')])

        agent = Agent(FunctionModel(model_error_once), capabilities=[RetryOnErrorCap()])

        nodes: list[str] = []
        async with agent.iter('hello') as agent_run:
            async for node in agent_run:
                nodes.append(type(node).__name__)

        assert nodes == snapshot(['UserPromptNode', 'ModelRequestNode', 'ModelRequestNode', 'CallToolsNode', 'End'])
        assert attempts == 2
        assert agent_run.result is not None
        assert agent_run.result.output == 'second time lucky'

    async def test_wrap_node_run_replacing_the_handler_result_ends_the_run(self):
        """A wrapper that runs the handler and then overrides its result ends the run there."""

        @dataclass
        class OverrideResultCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                result = await handler(node)
                if Agent.is_model_request_node(node):
                    # The handler advanced the graph to `CallToolsNode`; end the run instead.
                    return End(FinalResult(output='overridden'))
                return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[OverrideResultCap()])

        nodes: list[str] = []
        async with agent.iter('hello') as agent_run:
            async for node in agent_run:
                nodes.append(type(node).__name__)

        assert nodes == snapshot(['UserPromptNode', 'ModelRequestNode', 'End'])
        assert agent_run.result is not None
        assert agent_run.result.output == 'overridden'
        assert agent_run.next_node == End(FinalResult(output='overridden'))

        result = await agent.run('hello')
        assert result.output == 'overridden'

    async def test_bare_async_for_fires_wrap_node_run(self):
        """Bare `async for` fires `wrap_node_run`, matching `next()` driving and `agent.run()`."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            async with agent.iter('hello') as agent_run:
                async for _node in agent_run:
                    pass
        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']
        assert w == []

    async def test_works_with_manual_next(self):
        """wrap_node_run fires when using manual next() driving."""
        from pydantic_graph import End

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']

    async def test_chaining_nests_correctly(self):
        """Multiple capabilities compose wrap_node_run as nested middleware."""
        log: list[str] = []

        @dataclass
        class OrderedCap(AbstractCapability[Any]):
            name: str

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'{self.name}:before:{type(node).__name__}')
                result = await handler(node)
                log.append(f'{self.name}:after:{type(result).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[OrderedCap(name='outer'), OrderedCap(name='inner')],
        )
        await agent.run('hello')
        # For UserPromptNode: outer wraps inner
        assert log[0] == 'outer:before:UserPromptNode'
        assert log[1] == 'inner:before:UserPromptNode'
        assert log[2] == 'inner:after:ModelRequestNode'
        assert log[3] == 'outer:after:ModelRequestNode'



# --- Node run lifecycle hook tests ---


class TestNodeRunHooks:
    async def test_before_node_run_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_node_run:UserPromptNode' in cap.log
        assert 'before_node_run:ModelRequestNode' in cap.log
        assert 'before_node_run:CallToolsNode' in cap.log

    async def test_after_node_run_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'after_node_run:UserPromptNode' in cap.log
        assert 'after_node_run:ModelRequestNode' in cap.log
        assert 'after_node_run:CallToolsNode' in cap.log

    async def test_node_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        # For each node, before fires before after
        for node_name in ('UserPromptNode', 'ModelRequestNode', 'CallToolsNode'):
            before_idx = cap.log.index(f'before_node_run:{node_name}')
            after_idx = cap.log.index(f'after_node_run:{node_name}')
            assert before_idx < after_idx


# --- Run error hook tests ---


class TestRunErrorHooks:
    async def test_on_run_error_fires_on_failure(self):
        cap = LoggingCapability()

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')
        assert 'on_run_error' in cap.log

    async def test_on_run_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'on_run_error' not in cap.log

    async def test_on_run_error_can_transform_error(self):
        @dataclass
        class TransformErrorCap(AbstractCapability[Any]):
            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                raise ValueError('transformed error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[TransformErrorCap()])
        with pytest.raises(ValueError, match='transformed error'):
            await agent.run('hello')

    async def test_on_run_error_can_recover(self):
        @dataclass
        class RecoverRunCap(AbstractCapability[Any]):
            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                return AgentRunResult(output='recovered')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[RecoverRunCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered'

    async def test_on_run_error_not_called_when_wrap_run_recovers(self):
        @dataclass
        class WrapRecoveryCap(AbstractCapability[Any]):
            log: list[str] = field(default_factory=lambda: [])

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    self.log.append('wrap_run:caught')
                    return AgentRunResult(output='wrap_recovered')

            # The uncovered body is the assertion: this hook must not be called.
            async def on_run_error(  # pragma: no cover
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                self.log.append('on_run_error')
                raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = WrapRecoveryCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'wrap_recovered'
        assert 'wrap_run:caught' in cap.log
        assert 'on_run_error' not in cap.log

    async def test_on_run_error_fires_via_iter(self):
        from pydantic_graph import End

        @dataclass
        class RecoverRunCap(AbstractCapability[Any]):
            called: bool = False

            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                self.called = True
                return AgentRunResult(output='recovered via iter')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = RecoverRunCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):  # pragma: no branch
                node = await agent_run.next(node)
        assert cap.called
        assert agent_run.result is not None
        assert agent_run.result.output == 'recovered via iter'


# --- Node run error hook tests ---


class TestNodeRunErrorHooks:
    async def test_on_node_run_error_fires(self):
        cap = LoggingCapability()

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')
        assert 'on_node_run_error:ModelRequestNode' in cap.log

    async def test_on_node_run_error_can_recover_with_end(self):
        from pydantic_ai.result import FinalResult
        from pydantic_graph import End

        @dataclass
        class RecoverNodeCap(AbstractCapability[Any]):
            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: BaseException) -> Any:
                return End(FinalResult(output='recovered'))

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = RecoverNodeCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)
        assert isinstance(node, End)
        assert node.data.output == 'recovered'

    async def test_on_node_run_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert all('on_node_run_error' not in entry for entry in cap.log)


# --- Model request error hook tests ---


class TestModelRequestErrorHooks:
    async def test_on_model_request_error_fires(self):
        cap = LoggingCapability()

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')
        assert 'on_model_request_error' in cap.log

    async def test_on_model_request_error_can_recover(self):
        @dataclass
        class RecoverModelCap(AbstractCapability[Any]):
            async def on_model_request_error(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='recovered response')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[RecoverModelCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered response'

    async def test_on_model_request_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'on_model_request_error' not in cap.log

    async def test_default_on_model_request_error_reraises(self):
        """Default on_model_request_error re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            def get_instructions(self):
                return 'Be helpful.'

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[MinimalCap()])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_default_on_model_request_error_reraises_streaming(self):
        """Default on_model_request_error re-raises in streaming path (wrap_task error after stream consumed)."""

        @dataclass
        class PostProcessFailCap(AbstractCapability[Any]):
            """wrap_model_request that fails AFTER handler returns (post-processing error)."""

            def get_instructions(self):
                return 'Be helpful.'

            async def wrap_model_request(self, ctx: RunContext[Any], *, request_context: Any, handler: Any) -> Any:
                await handler(request_context)
                raise RuntimeError('post-processing exploded')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[PostProcessFailCap()],
        )
        with pytest.raises(RuntimeError, match='post-processing exploded'):
            async with agent.run_stream('hello') as stream:
                await stream.get_output()


# --- Tool validate error hook tests ---


class TestToolValidateErrorHooks:
    async def test_on_tool_validate_error_fires_on_validation_failure(self):
        cap = LoggingCapability()

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "correct"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[cap])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        await agent.run('greet someone')
        assert 'on_tool_validate_error:greet' in cap.log

    async def test_on_tool_validate_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert all('on_tool_validate_error' not in entry for entry in cap.log)

    async def test_on_tool_validate_error_can_recover(self):
        @dataclass
        class RecoverValidateCap(AbstractCapability[Any]):
            async def on_tool_validate_error(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
            ) -> dict[str, Any]:
                return {'name': 'recovered-name'}

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[RecoverValidateCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert received_name == 'recovered-name'
        assert 'hello recovered-name' in result.output

    async def test_default_on_tool_validate_error_reraises(self):
        """The default on_tool_validate_error re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            def get_instructions(self):
                return 'Be helpful.'

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "correct"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert 'hello correct' in result.output

    async def test_args_validator_deferral_is_not_a_validate_error(self):
        """A deferral raised by an `args_validator` passes through the validate hooks as control flow.

        Like an execute-stage deferral, it's not an error, so `on_tool_validate_error` doesn't fire
        and the tool is never executed. `after_tool_validate` still runs: it guards validated
        arguments, and a deferred call is queued with exactly those.
        """
        cap = LoggingCapability()

        def my_validator(ctx: RunContext[Any], x: int) -> None:
            raise ApprovalRequired()

        agent = Agent(TestModel(), output_type=[str, DeferredToolRequests], capabilities=[cap])

        @agent.tool_plain(args_validator=my_validator)
        def my_tool(x: int) -> int:  # pragma: no cover
            return x

        result = await agent.run('call the tool')
        assert isinstance(result.output, DeferredToolRequests)
        assert [entry for entry in cap.log if 'tool_validate' in entry or 'tool_execute' in entry] == snapshot(
            ['before_tool_validate:my_tool', 'wrap_tool_validate:my_tool:before', 'after_tool_validate:my_tool']
        )


# --- `after_tool_validate` as a policy gate on deferred calls ---


@dataclass
class ArgsGateCap(AbstractCapability[Any]):
    """An `after_tool_validate` policy gate: records the args it sees, and can reject or rewrite them."""

    reject_first: bool = False
    rewrite: dict[str, Any] | None = None
    seen: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.seen.append(dict(args))
        if self.reject_first and len(self.seen) == 1:
            raise ModelRetry('policy says no')
        return self.rewrite if self.rewrite is not None else args


class TestAfterToolValidateOnDeferral:
    """`after_tool_validate` guards validated arguments, so a deferral must not bypass it.

    Before this was fixed, an `args_validator` deferral escaped `wrap_tool_validate` and skipped the
    hook entirely, so a deployment using it as an authorization gate had a privileged call queued for
    approval without ever passing the gate.
    """

    async def test_runs_when_args_validator_defers(self):
        """The gate sees the validated args, and the call is still deferred once it passes."""
        cap = ArgsGateCap()
        agent = Agent(TestModel(), output_type=[str, DeferredToolRequests], capabilities=[cap])

        def my_validator(ctx: RunContext[Any], x: int) -> None:
            raise ApprovalRequired(metadata={'from': 'args_validator'})

        # `retries=0` pins that the deferral still doesn't consume the retry budget.
        @agent.tool(args_validator=my_validator, retries=0)
        def my_tool(ctx: RunContext[Any], x: int) -> int:  # pragma: no cover
            return x

        events: list[AgentStreamEvent | AgentRunResultEvent[Any]] = []
        async with agent.run_stream_events('call the tool') as stream:
            async for event in stream:
                events.append(event)

        result = events[-1]
        assert isinstance(result, AgentRunResultEvent)
        assert result.result.output == snapshot(
            DeferredToolRequests(
                approvals=[
                    ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool')
                ],
                metadata={'pyd_ai_tool_call_id__my_tool': {'from': 'args_validator'}},
            )
        )
        assert cap.seen == snapshot([{'x': 0}])
        assert [e.args_valid for e in events if isinstance(e, FunctionToolCallEvent)] == [True]

    async def test_rejection_wins_over_the_deferral(self):
        """A gate that rejects is honored: the model gets the retry, not a queued approval request."""
        cap = ArgsGateCap(reject_first=True)
        agent = Agent(TestModel(), output_type=[str, DeferredToolRequests], capabilities=[cap])

        def my_validator(ctx: RunContext[Any], x: int) -> None:
            raise ApprovalRequired()

        @agent.tool(args_validator=my_validator, retries=1)
        def my_tool(ctx: RunContext[Any], x: int) -> int:  # pragma: no cover
            return x

        result = await agent.run('call the tool')
        retries = [
            part.content
            for msg in result.all_messages()
            if isinstance(msg, ModelRequest)
            for part in msg.parts
            if isinstance(part, RetryPromptPart)
        ]
        assert retries == snapshot(['policy says no'])
        # The second attempt passes the gate, so that one defers.
        assert isinstance(result.output, DeferredToolRequests)
        assert cap.seen == snapshot([{'x': 0}, {'x': 0}])

    async def test_still_runs_after_a_recovered_validation_failure(self):
        """Regression pin: the failure path already ran the gate, via `on_tool_validate_error`."""

        @dataclass
        class RecoverCap(AbstractCapability[Any]):
            async def on_tool_validate_error(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
            ) -> dict[str, Any]:
                return {'name': 'recovered'}

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            return ModelResponse(parts=[ToolCallPart(tool_name='greet', args='{"wrong": 1}', tool_call_id='call-1')])

        gate = ArgsGateCap()
        agent = Agent(FunctionModel(bad_args_model), capabilities=[RecoverCap(), gate])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert result.output == snapshot('got: hello recovered')
        assert gate.seen == snapshot([{'name': 'recovered'}])

    async def test_deferral_carries_the_hooks_args(self):
        """A deferred call carries `after_tool_validate`'s output — what a non-deferred call would use.

        `ValidatedToolCall.validated_args` has no public observable on the deferral path (the request
        holds the model's original `ToolCallPart`, and resuming re-validates from it), so the
        contract is pinned directly on the tool manager.
        """
        toolset = FunctionToolset[None]()

        def my_validator(ctx: RunContext[None], x: int) -> None:
            raise ApprovalRequired()

        @toolset.tool(args_validator=my_validator)
        def my_tool(ctx: RunContext[None], x: int) -> int:  # pragma: no cover
            return x

        cap = ArgsGateCap(rewrite={'x': 99})
        manager = await ToolManager[None](toolset=toolset, root_capability=cap).for_run_step(_build_run_context())

        validated = await manager.validate_tool_call(ToolCallPart('my_tool', {'x': 0}, tool_call_id='call-1'))
        assert validated.args_valid is True
        assert isinstance(validated.deferral, ApprovalRequired)
        assert validated.validated_args == snapshot({'x': 99})
        assert cap.seen == snapshot([{'x': 0}])


# --- Deferrals raised from tool hooks ---


@dataclass
class DeferringHookCap(AbstractCapability[Any]):
    """Raises a deferral from the single hook position named by `where` (none, if `where` is empty)."""

    where: str
    exc_type: type[ApprovalRequired] | type[CallDeferred] = ApprovalRequired

    def _maybe(self, position: str) -> None:
        if self.where == position:
            raise self.exc_type(metadata={'from': position})

    async def before_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
    ) -> Any:
        self._maybe('before_tool_validate')
        return args

    async def wrap_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, handler: Any
    ) -> Any:
        self._maybe('wrap_tool_validate_before')
        result = await handler(args)
        self._maybe('wrap_tool_validate_after')
        return result

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
    ) -> Any:
        self._maybe('after_tool_validate')
        return args

    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
    ) -> Any:
        self._maybe('before_tool_execute')
        return args

    async def wrap_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, handler: Any
    ) -> Any:
        self._maybe('wrap_tool_execute_before')
        result = await handler(args)
        self._maybe('wrap_tool_execute_after')
        return result

    async def after_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, result: Any
    ) -> Any:
        self._maybe('after_tool_execute')
        return result


class TestToolHookDeferrals:
    """A tool call may only be deferred once its arguments are known to be valid.

    That admits the tool's `args_validator` (covered by `tests/test_tools.py`), the validation hooks
    that run after validation, and every execution hook; the validation hooks that run before it get
    a `UserError` instead of the bare exception aborting the run.
    """

    @pytest.mark.parametrize('exc_type', [ApprovalRequired, CallDeferred])
    @pytest.mark.parametrize('where', ['after_tool_validate', 'wrap_tool_validate_after'])
    async def test_validate_hook_defers_once_args_are_valid(
        self, where: str, exc_type: type[ApprovalRequired] | type[CallDeferred]
    ):
        """Hooks holding validated args defer the call, exactly like an `args_validator` does."""
        executed: list[int] = []

        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringHookCap(where=where, exc_type=exc_type)],
        )

        # `retries=0` pins that deferring doesn't consume the retry budget: charging it would raise
        # `UnexpectedModelBehavior` here rather than deferring.
        @agent.tool(retries=0)
        def my_tool(ctx: RunContext[Any], x: int) -> int:  # pragma: no cover
            executed.append(x)
            return x

        events: list[AgentStreamEvent | AgentRunResultEvent[Any]] = []
        async with agent.run_stream_events('call the tool') as stream:
            async for event in stream:
                events.append(event)

        result = events[-1]
        assert isinstance(result, AgentRunResultEvent)
        requests = result.result.output
        assert isinstance(requests, DeferredToolRequests)
        deferred = requests.approvals if exc_type is ApprovalRequired else requests.calls
        assert [call.tool_name for call in deferred] == ['my_tool']
        assert requests.metadata == {deferred[0].tool_call_id: {'from': where}}
        assert [e.args_valid for e in events if isinstance(e, FunctionToolCallEvent)] == [True]
        assert executed == []

    @pytest.mark.parametrize('exc_type', [ApprovalRequired, CallDeferred])
    @pytest.mark.parametrize('where', ['before_tool_validate', 'wrap_tool_validate_before'])
    async def test_validate_hook_cannot_defer_before_args_are_valid(
        self, where: str, exc_type: type[ApprovalRequired] | type[CallDeferred]
    ):
        """Deferring before validation has run is a usage error, not a bare exception aborting the run."""
        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringHookCap(where=where, exc_type=exc_type)],
        )

        @agent.tool
        def my_tool(ctx: RunContext[Any], x: int) -> int:  # pragma: no cover
            return x

        # The full wording is pinned once by `test_on_tool_validate_error_cannot_defer`; here we pin
        # that each rejected position is named, so a user can find the hook that raised.
        hook_name = 'wrap_tool_validate' if where.startswith('wrap') else where
        with pytest.raises(UserError, match=re.escape(f'`{hook_name}` raised `{exc_type.__name__}`')):
            await agent.run('call the tool')

    async def test_on_tool_validate_error_cannot_defer(self):
        """The error hook only runs when validation failed, so it has no valid arguments to defer."""

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='greet', args='{"wrong": 1}', tool_call_id='call-1')])

        @dataclass
        class DeferringValidateErrorCap(AbstractCapability[Any]):
            async def on_tool_validate_error(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
            ) -> Any:
                raise ApprovalRequired()

        agent = Agent(
            FunctionModel(bad_args_model),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringValidateErrorCap()],
        )

        @agent.tool_plain
        def greet(name: str) -> str:  # pragma: no cover
            return f'hello {name}'

        with pytest.raises(UserError) as exc_info:
            await agent.run('greet someone')
        assert str(exc_info.value) == snapshot(
            "`on_tool_validate_error` raised `ApprovalRequired`, but a tool call can only be deferred once its arguments have been validated. Raise it from `after_tool_validate`, from the tool's `args_validator`, or from `before_tool_execute` instead."
        )

    async def test_hook_deferral_replaces_an_args_validator_deferral(self):
        """When both defer, the hook wins: it runs later and is the policy layer.

        The `args_validator` asks for approval; `after_tool_validate` — which runs even for an
        already-deferred call — defers for external execution instead, and that's what the run
        surfaces, with the hook's metadata.
        """
        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringHookCap(where='after_tool_validate', exc_type=CallDeferred)],
        )

        def my_validator(ctx: RunContext[Any], x: int) -> None:
            raise ApprovalRequired(metadata={'from': 'args_validator'})

        @agent.tool(args_validator=my_validator, retries=0)
        def my_tool(ctx: RunContext[Any], x: int) -> int:  # pragma: no cover
            return x

        result = await agent.run('call the tool')
        assert result.output == snapshot(
            DeferredToolRequests(
                calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool')],
                metadata={'pyd_ai_tool_call_id__my_tool': {'from': 'after_tool_validate'}},
            )
        )

    @pytest.mark.parametrize('exc_type', [ApprovalRequired, CallDeferred])
    @pytest.mark.parametrize(
        'where',
        ['before_tool_execute', 'wrap_tool_execute_before', 'wrap_tool_execute_after', 'after_tool_execute'],
    )
    async def test_execute_hook_defers(self, where: str, exc_type: type[ApprovalRequired] | type[CallDeferred]):
        """Every execution hook can defer: by then the arguments are validated.

        The two positions that run after the tool body defer a call whose side effects already
        happened and whose result is discarded — documented, not fixed here.
        """
        executed: list[int] = []

        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringHookCap(where=where, exc_type=exc_type)],
        )

        @agent.tool(retries=0)
        def my_tool(ctx: RunContext[Any], x: int) -> int:
            executed.append(x)
            return x

        events: list[AgentStreamEvent | AgentRunResultEvent[Any]] = []
        async with agent.run_stream_events('call the tool') as stream:
            async for event in stream:
                events.append(event)

        result = events[-1]
        assert isinstance(result, AgentRunResultEvent)
        requests = result.result.output
        assert isinstance(requests, DeferredToolRequests)
        deferred = requests.approvals if exc_type is ApprovalRequired else requests.calls
        assert [call.tool_name for call in deferred] == ['my_tool']
        assert requests.metadata == {deferred[0].tool_call_id: {'from': where}}
        assert [e.args_valid for e in events if isinstance(e, FunctionToolCallEvent)] == [True]
        assert executed == ([0] if where.endswith('_after') or where == 'after_tool_execute' else [])

    @pytest.mark.parametrize('exc_type', [ApprovalRequired, CallDeferred])
    async def test_execute_error_hook_defers(self, exc_type: type[ApprovalRequired] | type[CallDeferred]):
        """The execution error hook can replace a tool failure with a deferral."""

        @dataclass
        class DeferringExecuteErrorCap(AbstractCapability[Any]):
            async def on_tool_execute_error(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Exception
            ) -> Any:
                raise exc_type(metadata={'from': 'on_tool_execute_error'})

        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringExecuteErrorCap()],
        )

        @agent.tool(retries=0)
        def my_tool(ctx: RunContext[Any], x: int) -> int:
            raise RuntimeError('tool failed')

        result = await agent.run('call the tool')
        requests = result.output
        assert isinstance(requests, DeferredToolRequests)
        deferred = requests.approvals if exc_type is ApprovalRequired else requests.calls
        assert [call.tool_name for call in deferred] == ['my_tool']
        assert requests.metadata == {deferred[0].tool_call_id: {'from': 'on_tool_execute_error'}}

    async def test_hooks_that_defer_nowhere_leave_the_call_alone(self):
        """Control case: the same capability without a deferral runs the tool and returns its result.

        Pins that it's the deferral, not the hooks themselves, that changes any of the above.
        """
        executed: list[int] = []

        agent = Agent(
            TestModel(),
            output_type=[str, DeferredToolRequests],
            capabilities=[DeferringHookCap(where='')],
        )

        @agent.tool(retries=0)
        def my_tool(ctx: RunContext[Any], x: int) -> int:
            executed.append(x)
            return x

        result = await agent.run('call the tool')
        assert result.output == snapshot('{"my_tool":0}')
        assert executed == [0]


# --- Tool execute error hook tests ---


class TestToolExecuteErrorHooks:
    async def test_on_tool_execute_error_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call the tool')
        assert 'on_tool_execute_error:my_tool' in cap.log

    async def test_on_tool_execute_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert all('on_tool_execute_error' not in entry for entry in cap.log)

    async def test_on_tool_execute_error_can_recover(self):
        @dataclass
        class RecoverExecCap(AbstractCapability[Any]):
            async def on_tool_execute_error(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'fallback result'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[RecoverExecCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert 'fallback result' in result.output


# --- Hooks capability tests ---


class TestHooksCapability:
    """Tests for the Hooks decorator-based capability."""

    async def test_decorator_registration(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        @hooks.on.after_model_request
        async def log_response(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            call_log.append('after_model_request')
            return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['before_model_request', 'after_model_request']

    async def test_constructor_form(self):
        call_log: list[str] = []

        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Hooks(before_model_request=log_request)])
        await agent.run('hello')
        assert call_log == ['before_model_request']

    async def test_multiple_hooks_same_event(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def first(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('first')
            return request_context

        @hooks.on.before_model_request
        async def second(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('second')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['first', 'second']

    async def test_tool_names_filtering(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_tool_execute(tools=['target_tool'])
        async def filtered(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            call_log.append(f'filtered:{call.tool_name}')
            return args

        @hooks.on.after_tool_execute
        async def unfiltered(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], result: Any
        ) -> Any:
            call_log.append(f'unfiltered:{call.tool_name}')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def target_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert 'filtered:target_tool' in call_log
        assert 'unfiltered:target_tool' in call_log

    async def test_wrap_model_request(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.model_request
        async def wrap(ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any) -> ModelResponse:
            call_log.append('wrap_start')
            result = await handler(request_context)
            call_log.append('wrap_end')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['wrap_start', 'wrap_end']

    async def test_wrap_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.run
        async def wrap(ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            call_log.append('wrap_run_start')
            result = await handler()
            call_log.append('wrap_run_end')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['wrap_run_start', 'wrap_run_end']

    async def test_on_error_recovery(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def recover(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='recovered')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'recovered'

    async def test_sync_function_auto_wrapping(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        def sync_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('sync_hook')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['sync_hook']

    async def test_timeout(self):
        hooks = Hooks()

        @hooks.on.before_model_request(timeout=0.01)
        async def slow_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            await asyncio.sleep(10)
            return request_context  # pragma: no cover

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        with pytest.raises(HookTimeoutError) as exc_info:
            await agent.run('hello')
        assert exc_info.value.hook_name == 'before_model_request'
        assert exc_info.value.func_name == 'slow_hook'
        assert exc_info.value.timeout == 0.01
        assert isinstance(exc_info.value, AgentRunError)
        assert isinstance(exc_info.value, TimeoutError)

    async def test_has_wrap_node_run(self):
        hooks = Hooks()
        with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
            assert hooks.has_wrap_node_run is False  # type: ignore[reportDeprecated]

        nodes_seen: list[str] = []

        @hooks.on.node_run
        async def wrap(ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
            nodes_seen.append(type(node).__name__)
            return await handler(node)

        with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
            assert hooks.has_wrap_node_run is True  # type: ignore[reportDeprecated]

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert len(nodes_seen) > 0

    async def test_composition_with_other_capabilities(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def hooks_before(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('hooks_before')
            return request_context

        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks, cap])
        await agent.run('hello')
        assert 'hooks_before' in call_log
        assert 'before_model_request' in cap.log

    async def test_before_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_run
        async def on_start(ctx: RunContext[Any]) -> None:
            call_log.append('before_run')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['before_run']

    async def test_after_run(self):
        hooks = Hooks()
        outputs: list[str] = []

        @hooks.on.after_run
        async def on_end(ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            outputs.append(result.output)
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run('hello')
        assert outputs == [result.output]

    async def test_repr(self):
        hooks = Hooks()
        assert repr(hooks) == 'Hooks({})'

        @hooks.on.before_model_request
        async def hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            return request_context

        assert repr(hooks) == "Hooks({'before_model_request': 1})"

        # Verify the registered hook actually works
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')

    async def test_on_model_request_error_reraise(self):
        """Error hooks that re-raise propagate the error to the caller."""

        hooks = Hooks()

        @hooks.on.model_request_error
        async def log_and_reraise(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_on_run_error_reraise(self):
        """on_run_error hooks that re-raise propagate the error."""

        hooks = Hooks()

        @hooks.on.run_error
        async def log_and_reraise(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_on_run_error_recovery(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def recover(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output='recovered from run error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'recovered from run error'

    async def test_on_run_error_chaining(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def first_handler(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            raise ValueError('transformed by first')

        @hooks.on.run_error
        async def second_handler(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output=f'caught: {error}')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('original error')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert 'transformed by first' in result.output

    async def test_error_hook_chaining(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def first(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise ValueError('transformed')

        @hooks.on.model_request_error
        async def second(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=f'recovered: {error}')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('original')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert 'transformed' in result.output

    async def test_wrap_run_event_stream(self):
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.run_event_stream
        async def observe_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                events_seen.append(type(event).__name__)
                yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_hooks_with_streaming_run(self):
        """Hooks capability used during a streaming run exercises the default wrap_run_event_stream path."""

        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'before_model_request' in call_log

    async def test_node_run_hooks(self):
        """Exercise before_node_run, after_node_run, and node_run (wrap) via .on namespace."""
        hooks = Hooks()
        nodes_seen: list[str] = []

        @hooks.on.before_node_run
        async def before(ctx: RunContext[Any], *, node: Any) -> Any:
            nodes_seen.append(f'before:{type(node).__name__}')
            return node

        @hooks.on.after_node_run
        async def after(ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            nodes_seen.append(f'after:{type(node).__name__}')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert any('before:' in n for n in nodes_seen)
        assert any('after:' in n for n in nodes_seen)

    async def test_node_run_error_hook(self):
        """on.node_run_error fires when a node fails."""
        hooks = Hooks()
        error_log: list[str] = []

        @hooks.on.node_run_error
        async def handle(ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            error_log.append(f'error:{type(error).__name__}')
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('node exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='node exploded'):
            await agent.run('hello')
        assert any('error:RuntimeError' in e for e in error_log)

    async def test_on_event_hook(self):
        """on.event fires for each stream event and can modify events."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_on_event_hook_fires_in_run(self):
        """on.event fires in run() even without an event_stream_handler."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert result.output is not None
        assert 'PartStartEvent' in events_seen

    async def test_wrap_run_event_stream_fires_in_run(self):
        """on.run_event_stream fires in run() even without an event_stream_handler."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.run_event_stream
        async def observe_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                events_seen.append(type(event).__name__)
                yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert result.output is not None
        assert 'PartStartEvent' in events_seen

    async def test_on_event_with_run_event_stream(self):
        """on.event and on.run_event_stream can be used together."""
        hooks = Hooks()
        event_log: list[str] = []
        stream_log: list[str] = []

        @hooks.on.event
        async def per_event(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            event_log.append(type(event).__name__)
            return event

        @hooks.on.run_event_stream
        async def wrap_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            stream_log.append('started')
            async for event in stream:
                yield event
            stream_log.append('finished')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(event_log) > 0
        assert stream_log == ['started', 'finished']

    async def test_prepare_tools_hook(self):
        """on.prepare_tools filters tool definitions."""
        hooks = Hooks()

        @hooks.on.prepare_tools
        async def hide_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if not td.name.startswith('hidden_')]

        tool_called = False

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def visible_tool() -> str:
            nonlocal tool_called
            tool_called = True
            return 'visible'

        @agent.tool_plain
        def hidden_tool() -> str:
            return 'hidden'  # pragma: no cover

        await agent.run('call tool')
        assert tool_called

    async def test_prepare_output_tools_hook(self):
        """`on.prepare_output_tools` filters output tool definitions — model only sees the
        non-filtered ones."""
        hooks = Hooks()

        @hooks.on.prepare_output_tools
        async def hide_secret(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if td.name != 'secret_output']

        seen_output_tools: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            seen_output_tools.extend(td.name for td in info.output_tools)
            # Call the only remaining (non-filtered) output tool
            return ModelResponse(parts=[ToolCallPart('public_output', {'value': 'ok'})])

        class SecretOutput(BaseModel):
            value: str

        class PublicOutput(BaseModel):
            value: str

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[
                ToolOutput(SecretOutput, name='secret_output'),
                ToolOutput(PublicOutput, name='public_output'),
            ],
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert isinstance(result.output, PublicOutput)
        assert seen_output_tools == ['public_output']

    async def test_tool_validate_hooks(self):
        """Exercise before/after/wrap tool_validate and on_tool_validate_error."""
        hooks = Hooks()
        validate_log: list[str] = []

        @hooks.on.before_tool_validate
        async def before_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
        ) -> Any:
            validate_log.append('before_validate')
            return args

        @hooks.on.after_tool_validate
        async def after_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            validate_log.append('after_validate')
            return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert 'before_validate' in validate_log
        assert 'after_validate' in validate_log

    async def test_wrap_tool_validate_hook(self):
        """Exercise on.tool_validate (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_validate
        async def wrap_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, handler: Any
        ) -> dict[str, Any]:
            wrap_log.append('wrap_start')
            result = await handler(args)
            wrap_log.append('wrap_end')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert wrap_log == ['wrap_start', 'wrap_end']

    async def test_tool_validate_error_hook(self):
        """on.tool_validate_error can recover from validation failures."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def recover_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            return {'name': 'recovered'}

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert 'hello recovered' in result.output

    async def test_wrap_tool_execute_hook(self):
        """Exercise on.tool_execute (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_execute
        async def wrap_exec(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], handler: Any
        ) -> Any:
            wrap_log.append('exec_start')
            result = await handler(args)
            wrap_log.append('exec_end')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert wrap_log == ['exec_start', 'exec_end']

    async def test_tool_execute_error_hook(self):
        """on.tool_execute_error can recover from tool execution failures."""
        hooks = Hooks()

        @hooks.on.tool_execute_error
        async def recover_exec(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'fallback result'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert 'fallback result' in result.output

    async def test_tool_validate_error_reraise(self):
        """on.tool_validate_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def reraise(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            raise error

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "ok"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        await agent.run('greet someone')

    async def test_tool_execute_error_reraise(self):
        """on.tool_execute_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_execute_error
        async def reraise(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            raise error

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call tool')

    async def test_get_serialization_name(self):
        assert Hooks.get_serialization_name() is None

    async def test_default_on_tool_execute_error_reraises(self):
        """The default on_tool_execute_error just re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            """Capability that doesn't override error hooks."""

            def get_instructions(self):
                return 'Be helpful.'

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call the tool')


# --- Context var propagation tests ---

_test_cv: contextvars.ContextVar[str] = contextvars.ContextVar('_test_cv')


class TestContextVarPropagation:
    """Context vars set in wrap_run propagate to all hooks in the outer task."""

    async def test_wrap_run_contextvar_visible_in_node_hooks(self):
        """A capability that sets a contextvar in wrap_run should have it
        visible in another capability's node-level hooks via agent.run()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('from-wrap-run')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.seen.append(('wrap_node_run', _test_cv.get(None)))
                return await handler(node)

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                self.seen.append(('after_node_run', _test_cv.get(None)))
                return result

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                self.seen.append(('after_run', _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run('hello')

        for hook_name, value in reader.seen:
            assert value == 'from-wrap-run', f'{hook_name} did not see contextvar'

    async def test_wrap_run_contextvar_visible_via_iter_next(self):
        """Context vars set in wrap_run are visible when using agent.iter() + next()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('from-iter')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                self.seen.append(('after_run', _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        for hook_name, value in reader.seen:
            assert value == 'from-iter', f'{hook_name} did not see contextvar'

    async def test_contextvar_cleaned_up_after_run(self):
        """Context vars set in wrap_run are restored after the run completes."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('temporary')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        await agent.run('hello')

        # After the run, the contextvar should be cleaned up
        assert _test_cv.get(None) is None

    async def test_contextvar_cleaned_up_on_early_iter_exit(self):
        """Context vars are restored even when the caller exits iter() early."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('early-exit')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        async with agent.iter('hello') as agent_run:
            # Exit immediately without driving any nodes
            _ = agent_run.next_node

        # Context var must be cleaned up even though we abandoned the run
        assert _test_cv.get(None) is None

    async def test_before_run_contextvar_propagates(self):
        """Context vars set in before_run (not wrap_run) also propagate."""

        @dataclass
        class Setter(AbstractCapability):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                _test_cv.set('from-before-run')

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run('hello')

        for hook_name, value in reader.seen:
            assert value == 'from-before-run', f'{hook_name} did not see contextvar'

    async def test_contextvar_visible_in_on_run_error(self):
        """Context vars set in wrap_run are visible in on_run_error."""

        @dataclass
        class SetterWithRecovery(AbstractCapability):
            seen_in_error: str | None = None

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('error-path')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                self.seen_in_error = _test_cv.get(None)
                return AgentRunResult(output='recovered')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = SetterWithRecovery()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run('hello')

        assert result.output == 'recovered'
        assert cap.seen_in_error == 'error-path'



# --- Tests for double-execution bug fix (streaming + before_node_run replacement) ---


class TestNodeStreamingWithHooks:
    """Tests that node streaming with event_stream_handler doesn't cause double model execution
    when before_node_run replaces a node."""

    async def test_before_node_run_replacement_no_double_execution(self):
        """When before_node_run replaces a ModelRequestNode and event_stream_handler is set,
        the model should be called exactly once (not twice)."""
        model_call_count = 0

        async def counting_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield 'streamed response'

        cap = _ReplacingCapability()
        agent = Agent(FunctionModel(simple_model_function, stream_function=counting_stream), capabilities=[cap])

        events_received: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events_received.append(event)

        result = await agent.run('hello', event_stream_handler=handler)
        assert result.output == 'streamed response'
        assert model_call_count == 1, f'Model was called {model_call_count} times, expected 1'
        assert len(events_received) > 0

    async def test_hook_ordering_with_event_stream_handler(self):
        """before_node_run fires BEFORE streaming events, wrap_node_run wraps the streaming,
        and after_node_run fires after graph advancement."""
        log: list[str] = []

        @dataclass
        class OrderTrackingCapability(AbstractCapability[Any]):
            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                log.append(f'before:{type(node).__name__}')
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'wrap:enter:{type(node).__name__}')
                result = await handler(node)
                log.append(f'wrap:exit:{type(node).__name__}')
                return result

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                log.append(f'after:{type(node).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[OrderTrackingCapability()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass
            log.append('stream:consumed')

        await agent.run('hello', event_stream_handler=handler)

        # For ModelRequestNode: before → wrap:enter → stream:consumed → wrap:exit → after
        mr_before = log.index('before:ModelRequestNode')
        mr_wrap_enter = log.index('wrap:enter:ModelRequestNode')
        stream_consumed_idx = log.index('stream:consumed')
        mr_wrap_exit = log.index('wrap:exit:ModelRequestNode')
        mr_after = log.index('after:ModelRequestNode')
        assert mr_before < mr_wrap_enter < stream_consumed_idx < mr_wrap_exit < mr_after

    async def test_run_stream_before_node_run_replacement_no_double_execution(self):
        """Same as the run() test but for run_stream(): before_node_run replacement
        should not cause double model execution."""
        model_call_count = 0

        async def counting_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield 'streamed response'

        cap = _ReplacingCapability()
        agent = Agent(FunctionModel(simple_model_function, stream_function=counting_stream), capabilities=[cap])

        async with agent.run_stream('hello') as streamed:
            output = await streamed.get_output()

        assert output == 'streamed response'
        assert model_call_count == 1, f'Model was called {model_call_count} times, expected 1'

    async def test_run_stream_skips_wrap_and_after_for_the_final_model_request(self):
        """`run_stream()` hands back the result mid-stream, so the final `ModelRequestNode` only gets `before_node_run`.

        Pinning the documented exception to "node hooks fire however the run is driven": that node's
        `wrap_node_run`/`after_node_run` are deliberately skipped, while the `SetFinalResult` node
        that ends the run gets the full lifecycle.
        """
        log: list[str] = []

        @dataclass
        class NodeHookCap(AbstractCapability[Any]):
            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                log.append(f'before:{type(node).__name__}')
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'wrap:{type(node).__name__}')
                return await handler(node)

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                log.append(f'after:{type(node).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[NodeHookCap()],
        )

        async with agent.run_stream('hello') as streamed:
            await streamed.get_output()

        assert log == snapshot(
            [
                'before:UserPromptNode',
                'wrap:UserPromptNode',
                'after:UserPromptNode',
                'before:ModelRequestNode',
                'before:SetFinalResult',
                'wrap:SetFinalResult',
                'after:SetFinalResult',
            ]
        )

    async def test_on_node_run_error_fires_in_run_stream(self):
        """on_node_run_error in run_stream() fires when wrap_node_run raises during graph advancement."""
        error_log: list[str] = []

        @dataclass
        class WrapErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                # Raise on CallToolsNode — after UserPromptNode and ModelRequestNode pass through.
                # ModelRequestNode with tool calls doesn't produce a FinalResultEvent in run_stream(),
                # so it falls through to wrap_node_run; CallToolsNode is next and triggers the error.
                from pydantic_ai._agent_graph import CallToolsNode

                if isinstance(node, CallToolsNode):
                    raise RuntimeError('wrap error')
                return await handler(node)

            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
                error_log.append(type(node).__name__)
                raise error

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[WrapErrorCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        with pytest.raises(RuntimeError, match='wrap error'):
            async with agent.run_stream('hello') as _streamed:
                pass

        assert error_log == ['CallToolsNode']


# --- ToolFailed and ModelRetry from hooks tests ---


class _BeforeToolFailedCap(AbstractCapability[Any]):
    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        raise ToolFailed('failed before execution')


class _WrapToolFailedCap(AbstractCapability[Any]):
    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        try:
            return await handler(args)
        except RuntimeError as e:
            raise ToolFailed('failed during wrapper') from e


class _AfterToolFailedCap(AbstractCapability[Any]):
    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        raise ToolFailed('failed after execution')


class _OnToolExecuteErrorFailedCap(AbstractCapability[Any]):
    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        raise ToolFailed('failed while handling error')


class _BeforeToolValidateFailedCap(AbstractCapability[Any]):
    async def before_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        raise ToolFailed('failed before validation')


class _OnToolValidateErrorFailedCap(AbstractCapability[Any]):
    async def on_tool_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        error: ValidationError | ModelRetry,
    ) -> dict[str, Any]:
        raise ToolFailed('failed while handling validation error')


class _WrapToolValidateFailedCap(AbstractCapability[Any]):
    async def wrap_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: Any,
    ) -> dict[str, Any]:
        raise ToolFailed('failed during validate wrapper')


def _tool_failed_roundtrip_model(tool_args: str) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return make_text_response(f'got: {part.outcome}:{part.content}')
        if info.function_tools:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.function_tools[0].name, args=tool_args, tool_call_id='call-1')]
            )
        return make_text_response('no tools')  # pragma: no cover

    return model_fn


def _assert_failed_tool_result(result: AgentRunResult[Any], expected_message: str) -> None:
    assert result.output == f'got: failed:{expected_message}'

    parts = [part for msg in result.all_messages() for part in msg.parts]
    tool_return = next(part for part in parts if isinstance(part, ToolReturnPart))
    assert tool_return.outcome == 'failed'
    assert tool_return.content == expected_message
    assert not any(isinstance(part, RetryPromptPart) for part in parts)


class TestToolFailedFromHooks:
    """Tests for raising ToolFailed from capability tool hooks."""

    @pytest.mark.parametrize('hook_name', ['before', 'wrap', 'after', 'on_error'])
    async def test_tool_execute_hook_tool_failed(self, hook_name: str):
        tool_call_count = 0
        cap_type, expected_message, tool_should_run = {
            'before': (_BeforeToolFailedCap, 'failed before execution', False),
            'wrap': (_WrapToolFailedCap, 'failed during wrapper', True),
            'after': (_AfterToolFailedCap, 'failed after execution', True),
            'on_error': (_OnToolExecuteErrorFailedCap, 'failed while handling error', True),
        }[hook_name]

        agent = Agent(FunctionModel(_tool_failed_roundtrip_model('{}')), capabilities=[cap_type()])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            if hook_name in {'wrap', 'on_error'}:
                raise RuntimeError('tool failed')
            return 'tool result'

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, expected_message)
        assert tool_call_count == int(tool_should_run)

    async def test_deferred_tool_validate_hook_tool_failed(self):
        """Deferred tool validation can return a failed tool result instead of a deferred request."""
        tool_call_count = 0

        agent = Agent(
            FunctionModel(_tool_failed_roundtrip_model('{}')),
            capabilities=[_BeforeToolValidateFailedCap()],
            output_type=[str, DeferredToolRequests],
            retries={'tools': 0, 'output': 2},
        )

        @agent.tool_plain(requires_approval=True)
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, 'failed before validation')
        assert tool_call_count == 0

    @pytest.mark.parametrize(
        ('capability', 'tool_args', 'expected_message'),
        [
            pytest.param(
                _BeforeToolValidateFailedCap(), '{"x":1}', 'failed before validation', id='before_tool_validate'
            ),
            pytest.param(
                _OnToolValidateErrorFailedCap(),
                '{"x":"bad"}',
                'failed while handling validation error',
                id='on_tool_validate_error',
            ),
            pytest.param(
                _WrapToolValidateFailedCap(), '{"x":1}', 'failed during validate wrapper', id='wrap_tool_validate'
            ),
        ],
    )
    async def test_tool_validate_hook_tool_failed(
        self, capability: AbstractCapability[Any], tool_args: str, expected_message: str
    ):
        """Non-deferred tool validation hooks can report a failed tool result instead of retrying."""
        tool_call_count = 0

        agent = Agent(
            FunctionModel(_tool_failed_roundtrip_model(tool_args)),
            capabilities=[capability],
            retries={'tools': 0, 'output': 2},
        )

        @agent.tool_plain
        def my_tool(x: int) -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return f'tool result: {x}'  # pragma: no cover

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, expected_message)
        assert tool_call_count == 0

    async def test_args_validator_tool_failed(self):
        """An `args_validator` raising `ToolFailed` reports a failed tool result instead of retrying."""
        tool_call_count = 0
        expected_message = 'failed in args validator'

        def validate_args(ctx: RunContext[Any]) -> None:
            raise ToolFailed(expected_message)

        agent = Agent(
            FunctionModel(_tool_failed_roundtrip_model('{}')),
            retries={'tools': 0, 'output': 2},
        )

        @agent.tool_plain(args_validator=validate_args)
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, expected_message)
        assert tool_call_count == 0


class TestModelRetryFromHooks:
    """Tests for raising ModelRetry from capability hooks."""

    async def test_after_model_request_model_retry(self):
        """after_model_request raises ModelRetry — model is called again with retry prompt."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_text_response('bad response')
            return make_text_response('good response')

        @dataclass
        class RetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Response was bad, please try again')
                return response

        cap = RetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'good response'
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
                    parts=[TextPart(content='bad response')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Response was bad, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=66, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_model_request_model_retry_max_retries(self):
        """after_model_request raises ModelRetry repeatedly — hits output_retries."""

        @dataclass
        class AlwaysRetryCap(AbstractCapability[Any]):
            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                raise ModelRetry('always bad')

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[AlwaysRetryCap()],
            retries={'output': 2},
        )
        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
            await agent.run('hello')

    async def test_after_model_request_model_retry_streaming(self):
        """after_model_request raises ModelRetry during streaming with tool calls — model is called again."""
        call_count = 0

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return a tool call that after_model_request will reject
                yield {0: DeltaToolCall(name='my_tool', json_args='{}', tool_call_id='call-1')}
            elif call_count == 2:
                # Second call (after retry): return text
                yield 'good response'
            else:
                yield 'unexpected'  # pragma: no cover

        @dataclass
        class RetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Response was bad, please try again')
                return response

        cap = RetryCap()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=stream_fn),
            capabilities=[cap],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        async with agent.run_stream('hello') as streamed:
            result = await streamed.get_output()
        assert result == 'good response'
        assert call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Response was bad, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_streaming_short_circuit(self):
        """wrap_model_request raises ModelRetry without calling handler during streaming."""

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'good response'

        @dataclass
        class ShortCircuitRetryCap(AbstractCapability[Any]):
            call_count: int = 0

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                self.call_count += 1
                if self.call_count == 1:
                    # Short-circuit: don't call handler, raise ModelRetry
                    raise ModelRetry('Short-circuit retry')
                return await handler(request_context)

        cap = ShortCircuitRetryCap()
        agent = Agent(FunctionModel(simple_model_function, stream_function=stream_fn), capabilities=[cap])
        async with agent.run_stream('hello') as streamed:
            result = await streamed.get_output()
        assert result == 'good response'
        assert cap.call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Short-circuit retry',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_streaming_after_handler(self):
        """wrap_model_request raises ModelRetry after calling handler during streaming (tool call scenario)."""
        call_count = 0

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tool call that wrap hook will reject
                yield {0: DeltaToolCall(name='my_tool', json_args='{}', tool_call_id='call-1')}
            else:
                yield 'good response'

        @dataclass
        class AfterHandlerRetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                response = await handler(request_context)
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Post-handler retry')
                return response

        cap = AfterHandlerRetryCap()
        agent = Agent(FunctionModel(simple_model_function, stream_function=stream_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        async with agent.run_stream('hello') as streamed:
            result = await streamed.get_output()
        assert result == 'good response'
        assert call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Post-handler retry',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry(self):
        """wrap_model_request raises ModelRetry after calling handler — triggers retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_text_response('first attempt')
            return make_text_response('second attempt')

        @dataclass
        class WrapRetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                response = await handler(request_context)
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Wrap says retry')
                return response

        cap = WrapRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'second attempt'
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
                    parts=[TextPart(content='first attempt')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Wrap says retry',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='second attempt')],
                    usage=RequestUsage(input_tokens=63, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_skips_on_error(self):
        """wrap_model_request raising ModelRetry should NOT call on_model_request_error."""
        on_error_called = False

        @dataclass
        class WrapRetrySkipErrorCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                raise ModelRetry('retry please')

            # The uncovered body is the assertion: this hook must not be called.
            async def on_model_request_error(  # pragma: no cover
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                error: Exception,
            ) -> ModelResponse:
                nonlocal on_error_called
                on_error_called = True
                raise error

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[WrapRetrySkipErrorCap()], retries={'output': 1}
        )
        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
            await agent.run('hello')
        assert not on_error_called

    async def test_on_model_request_error_model_retry(self):
        """on_model_request_error raises ModelRetry to recover via retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('model failed')
            return make_text_response('recovered response')

        @dataclass
        class ErrorRetryCap(AbstractCapability[Any]):
            async def on_model_request_error(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                error: Exception,
            ) -> ModelResponse:
                raise ModelRetry('Model failed, please try again')

        cap = ErrorRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'recovered response'
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Model failed, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='recovered response')],
                    usage=RequestUsage(input_tokens=65, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_execute_model_retry(self):
        """after_tool_execute raises ModelRetry — tool retry prompt sent to model, tool retried on success."""
        tool_call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Always call the tool — after retry, the hook won't raise again
            if info.function_tools:
                # Check if we already got a tool return (second call succeeded)
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class AfterExecRetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
            ) -> Any:
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Tool result is bad, try again')
                return result

        cap = AfterExecRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        assert tool_call_count == 2  # Tool called twice: first rejected by hook, second succeeds
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Tool result is bad, try again',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=65, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_execute_model_retry(self):
        """before_tool_execute raises ModelRetry — tool execution is skipped, then succeeds on retry."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Always call the tool — after retry, the hook won't raise again
            if info.function_tools:
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        hooks = Hooks[Any]()
        hook_called = False

        @hooks.on.before_tool_execute
        async def reject_first(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal hook_called
            if not hook_called:
                hook_called = True
                raise ModelRetry('Not ready to execute, try again')
            return args

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Not ready to execute, try again',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=65, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_execute_validation_error(self):
        """after_tool_execute raises ValidationError — converted to ToolRetryError for retry."""

        tool_call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.function_tools:
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class ValErrCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
            ) -> Any:
                if not self.retried:
                    self.retried = True
                    # Simulate a user hook doing additional Pydantic validation
                    TypeAdapter(int).validate_python('not_an_int')
                return result

        cap = ValErrCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        assert tool_call_count == 2  # Retried after ValidationError
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
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
                                    'loc': (),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'not_an_int',
                                }
                            ],
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=88, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=90, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_execute_validation_error(self):
        """before_tool_execute raises ValidationError — converted to ToolRetryError for retry."""

        tool_call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.function_tools:
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class ValErrCap(AbstractCapability[Any]):
            retried: bool = False

            async def before_tool_execute(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
            ) -> dict[str, Any]:
                if not self.retried:
                    self.retried = True
                    TypeAdapter(int).validate_python('not_an_int')
                return args

        cap = ValErrCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        # Tool only called once — before_tool_execute ValidationError prevented first call
        assert tool_call_count == 1
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
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
                                    'loc': (),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'not_an_int',
                                }
                            ],
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=88, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=90, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_tool_execute_model_retry_skips_on_error(self):
        """wrap_tool_execute raising ModelRetry should NOT call on_tool_execute_error."""
        on_error_called = False

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got retry')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class WrapExecRetryCap(AbstractCapability[Any]):
            async def wrap_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                handler: Any,
            ) -> Any:
                raise ModelRetry('Wrap says retry tool')

            # The uncovered body is the assertion: this hook must not be called.
            async def on_tool_execute_error(  # pragma: no cover
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                error: Exception,
            ) -> Any:
                nonlocal on_error_called
                on_error_called = True
                raise error

        agent = Agent(FunctionModel(model_fn), capabilities=[WrapExecRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')
        assert result.output == 'got retry'
        assert not on_error_called
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Wrap says retry tool',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got retry')],
                    usage=RequestUsage(input_tokens=63, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_on_tool_execute_error_model_retry(self):
        """on_tool_execute_error raises ModelRetry to recover via retry."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got retry after error')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class ErrorRetryCap(AbstractCapability[Any]):
            async def on_tool_execute_error(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                error: Exception,
            ) -> Any:
                raise ModelRetry('Tool errored, please retry')

        agent = Agent(FunctionModel(model_fn), capabilities=[ErrorRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert result.output == 'got retry after error'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Tool errored, please retry',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got retry after error')],
                    usage=RequestUsage(input_tokens=63, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_validate_model_retry(self):
        """after_tool_validate raises ModelRetry — validation retry sent to model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got validation retry')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class AfterValRetryCap(AbstractCapability[Any]):
            async def after_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                raise ModelRetry('Validated args are bad')

        agent = Agent(FunctionModel(model_fn), capabilities=[AfterValRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')
        assert result.output == 'got validation retry'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Validated args are bad',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got validation retry')],
                    usage=RequestUsage(input_tokens=63, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_validate_model_retry(self):
        """before_tool_validate raises ModelRetry — validation retry sent to model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got pre-validation retry')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class BeforeValRetryCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                raise ModelRetry('Args look bad before validation')

        agent = Agent(FunctionModel(model_fn), capabilities=[BeforeValRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')
        assert result.output == 'got pre-validation retry'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Args look bad before validation',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got pre-validation retry')],
                    usage=RequestUsage(input_tokens=64, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestCtxAgentInCapability:
    """Test that ctx.agent is available in capability hooks."""

    async def test_ctx_agent_in_hooks(self):
        hook_agent_names: list[str | None] = []

        @dataclass
        class AgentTrackingCap(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                assert ctx.agent is not None
                hook_agent_names.append(ctx.agent.name)

            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                assert ctx.agent is not None
                hook_agent_names.append(ctx.agent.name)
                return request_context

        agent = Agent(FunctionModel(simple_model_function), name='hook_test_agent', capabilities=[AgentTrackingCap()])
        await agent.run('hello')
        assert hook_agent_names == ['hook_test_agent', 'hook_test_agent']


