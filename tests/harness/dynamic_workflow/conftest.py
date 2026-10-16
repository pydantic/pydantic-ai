"""Shared helpers for the `DynamicWorkflow` tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflowToolset


def echo_agent(name: str) -> Agent[object, str]:
    """A sub-agent answering `<name>:<task>`, so a test can see which agent ran on what."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        task = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        return ModelResponse(parts=[TextPart(f'{name}:{task}')])

    return Agent(FunctionModel(respond), name=name)


def run_ctx(
    *, workspace: Workspace | None = None, tracer: Tracer | None = None, trace_include_content: bool = False
) -> RunContext[object]:
    extra: dict[str, Any] = {}
    if workspace is not None:
        extra['workspace'] = workspace
    if tracer is not None:
        extra['tracer'] = tracer
    return RunContext[object](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=1,
        trace_include_content=trace_include_content,
        **extra,
    )


async def call_workflow_tool(
    ts: DynamicWorkflowToolset[object], tool_args: dict[str, Any], ctx: RunContext[object] | None = None
) -> Any:
    """Call `run_workflow` (or, by `tool_name`, another tool) directly on a toolset."""
    ctx = ctx or run_ctx()
    tool_name = tool_args.pop('tool_name', ts.tool_name)
    tools = await ts.get_tools(ctx)
    return await ts.call_tool(tool_name, tool_args, ctx, tools[tool_name])


def recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def spans_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def scripted_model(calls: Sequence[tuple[str, dict[str, Any]]]) -> FunctionModel:
    """A model making `calls` one step at a time, then answering with the last tool result.

    Streams too, since a run with event listeners is a streamed run.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(calls):
            tool_name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(tool_name, args, tool_call_id=f'c{step}')])
        part = messages[-1].parts[-1]
        assert isinstance(part, ToolReturnPart | RetryPromptPart)
        return ModelResponse(
            parts=[TextPart(part.model_response_str() if isinstance(part, ToolReturnPart) else part.model_response())]
        )

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        (part,) = respond(messages, info).parts
        if isinstance(part, ToolCallPart):
            yield {
                0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id)
            }
        else:
            assert isinstance(part, TextPart)
            yield part.content

    return FunctionModel(respond, stream_function=stream)
