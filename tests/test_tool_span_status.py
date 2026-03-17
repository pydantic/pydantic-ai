
from __future__ import annotations as _annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.result import DeferredToolRequests


@pytest.fixture
def otel_setup():
    exporter = InMemorySpanExporter()
    processor = SimpleSpanProcessor(exporter)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    return provider, exporter

@pytest.mark.anyio
async def test_tool_retry_span_status(otel_setup):
    provider, exporter = otel_setup

    async def model_logic(messages, info):
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart('retry_tool', {'x': 5}, 'call_1')])
        return ModelResponse(parts=[TextPart('Done')])
    
    model = FunctionModel(model_logic)
    agent = Agent(model=model)

    @agent.tool_plain
    async def retry_tool(x: int) -> str:
        if x < 10:
            raise ModelRetry('Too small')
        return f'Accepted {x}'

    agent.instrument = InstrumentationSettings(tracer_provider=provider)
    await agent.run('Run retry tool')

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name in ('execute_tool:retry_tool', 'running tool')]
    assert len(tool_spans) > 0
    for span in tool_spans:
        assert span.status.status_code != StatusCode.ERROR
        # Also ensure no exception events are recorded for ModelRetry/ToolRetryError
        for event in span.events:
            if event.name == 'exception':
                exc_type = event.attributes.get('exception.type')
                assert exc_type not in ('ModelRetry', 'ToolRetryError', 'pydantic_ai.exceptions.ToolRetryError')

class MyOutput(BaseModel):
    x: int

@pytest.mark.anyio
async def test_output_function_retry_span_status(otel_setup):
    provider, exporter = otel_setup

    async def my_output_function(x: int) -> MyOutput:
        if x < 10:
             raise ModelRetry('Too small')
        return MyOutput(x=x)

    async def model_logic(messages, info):
        if not any(isinstance(m, ModelResponse) for m in messages):
            # Model calls the output tool
            return ModelResponse(parts=[ToolCallPart('final_result', {'x': 5}, 'call_1')])
        return ModelResponse(parts=[TextPart('Done')])
    
    model = FunctionModel(model_logic)
    agent = Agent(model=model, output_type=my_output_function)

    agent.instrument = InstrumentationSettings(tracer_provider=provider)
    try:
        await agent.run('Run output tool')
    except Exception:
        pass

    spans = exporter.get_finished_spans()
    found = False
    for span in spans:
        if span.name in ('execute_tool:final_result', 'running output function'):
            found = True
            assert span.status.status_code != StatusCode.ERROR
    assert found

@pytest.mark.anyio
async def test_tool_deferred_span_status(otel_setup):
    provider, exporter = otel_setup

    async def model_logic(messages, info):
        return ModelResponse(parts=[ToolCallPart('deferred_tool', {'x': 5}, 'call_1')])
    
    model = FunctionModel(model_logic)
    agent = Agent(model=model, output_type=str | DeferredToolRequests)

    @agent.tool_plain
    async def deferred_tool(x: int) -> str:
        raise CallDeferred()

    agent.instrument = InstrumentationSettings(tracer_provider=provider)
    result = await agent.run('Run deferred tool')
    assert isinstance(result.output, DeferredToolRequests)

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name in ('execute_tool:deferred_tool', 'running tool')]
    assert len(tool_spans) > 0
    for span in tool_spans:
        assert span.status.status_code != StatusCode.ERROR

@pytest.mark.anyio
async def test_tool_approval_span_status(otel_setup):
    provider, exporter = otel_setup

    async def model_logic(messages, info):
        return ModelResponse(parts=[ToolCallPart('approval_tool', {'x': 5}, 'call_1')])
    
    model = FunctionModel(model_logic)
    agent = Agent(model=model, output_type=str | DeferredToolRequests)

    @agent.tool_plain
    async def approval_tool(x: int) -> str:
        raise ApprovalRequired()

    agent.instrument = InstrumentationSettings(tracer_provider=provider)
    result = await agent.run('Run approval tool')
    assert isinstance(result.output, DeferredToolRequests)

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if s.name in ('execute_tool:approval_tool', 'running tool')]
    assert len(tool_spans) > 0
    for span in tool_spans:
        assert span.status.status_code != StatusCode.ERROR
