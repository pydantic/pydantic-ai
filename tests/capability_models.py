"""Model stubs shared by the capability test modules.

These live outside `test_capabilities.py` so that the capability tests can be split across
several modules without either duplicating the stubs or importing one test module from another.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import ModelOverloadedError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetFunc
from pydantic_ai.usage import RequestUsage, RunUsage


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

    yield 'no tools available'  # pragma: no cover


def tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that calls a tool on first request, then returns text."""
    # Check if there's already a tool return in messages (i.e., tool was called)
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return make_text_response('final response')

    # First request: call the tool
    if info.function_tools:
        tool = info.function_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{}', tool_call_id='call-1')])

    return make_text_response('no tools available')  # pragma: no cover


@dataclass
class CustomCapability(AbstractCapability):
    greeting: str = 'hello'


@dataclass
class ToolsetFuncCapability(AbstractCapability):
    """A capability that returns a ToolsetFunc instead of an AbstractToolset."""

    def get_toolset(self) -> ToolsetFunc:
        def make_toolset(ctx: RunContext) -> AbstractToolset:
            toolset = FunctionToolset()

            @toolset.tool_plain
            def greet(name: str) -> str:
                """Greet someone by name."""
                return f'Hello, {name}!'

            return toolset

        return make_toolset


def _noop_greet(name: str) -> str:
    return f'Hello, {name}!'  # pragma: no cover


def _build_run_context(deps: Any = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), run_step=0)


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


async def _registered_capability_context(
    *capabilities: AbstractCapability,
) -> tuple[dict[str, AbstractCapability], set[str]]:
    captured_capabilities: dict[str, AbstractCapability] = {}
    captured_available_ids: set[str] = set()

    @dataclass
    class CaptureCapabilities(AbstractCapability):
        async def before_model_request(
            self, ctx: RunContext, request_context: ModelRequestContext
        ) -> ModelRequestContext:
            captured_capabilities.update(ctx.capabilities)
            captured_available_ids.update(ctx.active_capability_ids)
            return request_context

    agent = Agent(
        FunctionModel(lambda _messages, _info: make_text_response('done')),
        capabilities=[*capabilities, CaptureCapabilities()],
    )
    await agent.run('capture capabilities')
    capability_ids = {id(capability) for capability in capabilities}
    captured_capabilities = {
        capability_id: capability
        for capability_id, capability in captured_capabilities.items()
        if id(capability) in capability_ids
    }
    captured_available_ids &= set(captured_capabilities)
    return captured_capabilities, captured_available_ids


build_run_context = _build_run_context
noop_greet = _noop_greet
registered_capability_context = _registered_capability_context


class MyOutput(BaseModel):
    value: int


@dataclass
class _BilledThenFailingStream(StreamedResponse):
    """Reports usage, as Anthropic does on `message_start`, then fails before any output, as on `overloaded_error`."""

    _get_raises: bool = False
    _job_id: str | None = None

    def get(self) -> ModelResponse:
        if self._get_raises:
            raise RuntimeError('a custom `get()` that fails on a failed stream')
        return super().get()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        self._usage = RequestUsage(input_tokens=100, output_tokens=5)
        self.provider_response_id = self._job_id
        raise ModelOverloadedError(model_name='billed', message='overloaded')
        yield  # pragma: no cover

    @property
    def model_name(self) -> str:
        return 'billed'

    @property
    def provider_name(self) -> str:
        return 'test'

    @property
    def provider_url(self) -> None:
        return None

    @property
    def timestamp(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


class BilledThenFailsModel(Model):
    """A model whose first `fail_times` streams are billed 100 input and 5 output tokens and fail before their first event.

    Later streams are `answer`'s. With `get_raises`, the failing stream's `get()` raises too. `job_id` is the failing
    stream's `provider_response_id`, like a background job's on a poll.
    """

    def __init__(
        self, answer: Model, *, fail_times: int = 1, get_raises: bool = False, job_id: str | None = None
    ) -> None:
        super().__init__()
        self.answer = answer
        self.fail_times = fail_times
        self.get_raises = get_raises
        self.job_id = job_id

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        raise NotImplementedError  # pragma: no cover

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        if self.fail_times:
            self.fail_times -= 1
            yield _BilledThenFailingStream(
                model_request_parameters=model_request_parameters, _get_raises=self.get_raises, _job_id=self.job_id
            )
        else:
            async with self.answer.request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream

    @property
    def model_name(self) -> str:
        return 'billed'

    @property
    def system(self) -> str:
        return 'test'
