from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Literal

import anyio
import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import TypeAdapter

from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.capabilities import (
    AbstractCapability,
    Hooks,
    Instrumentation,
    ModelFallback,
    OutputRetryEvent,
    SelectModel,
)
from pydantic_ai.messages import CapabilityEvent, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from tests.test_capability_model_fallback_output import Answer


@dataclass
class RecordRetries(AbstractCapability[object]):
    events: list[OutputRetryEvent] = field(default_factory=list[OutputRetryEvent])

    async def on_event(self, ctx: RunContext[object], *, event: AgentStreamEvent) -> None:
        if isinstance(event, OutputRetryEvent):
            self.events.append(event)


class TestOutputRetryEvent:
    async def test_output_policy_starts_from_selected_model(self) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        model = TestModel(custom_output_text='draft')
        agent = Agent(
            capabilities=[
                ModelFallback(TestModel(custom_output_text='ready'), output_retries=1),
                SelectModel(lambda ctx: model),
                Instrumentation(settings=InstrumentationSettings(tracer_provider=provider)),
            ]
        )

        @agent.output_validator
        def validate(ctx: RunContext[object], value: str) -> str:
            if ctx.retry == 0:
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        assert sum(span.name == 'model_fallback.output_fallback' for span in exporter.get_finished_spans()) == 1

    @pytest.mark.parametrize('include_content', [False, True])
    @pytest.mark.parametrize('producer', ['test', None])
    async def test_retry_events_and_single_trace(self, include_content: bool, producer: str | None) -> None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        recorder = RecordRetries()
        hooks = Hooks[object]()

        @hooks.on.after_model_request
        def set_producer(
            ctx: RunContext[object], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            response.model_name = producer
            return response

        fallback = ModelFallback(
            TestModel(custom_output_text='draft'), TestModel(custom_output_text='ready'), output_retries=1
        )
        fallback.id = 'fallback'
        agent = Agent(
            capabilities=[
                fallback,
                recorder,
                hooks,
                Instrumentation(
                    settings=InstrumentationSettings(tracer_provider=provider, include_content=include_content)
                ),
            ],
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        assert (await agent.run('Answer.')).output == 'ready'
        [event] = recorder.events
        assert event.source == 'response_output'
        assert event.response.text == 'draft'
        assert event.response.model_name == producer
        assert (event.retry.cause, event.retry.attempt, event.retry.max_retries) == ('model_retry', 1, 1)
        assert event.capability_id == 'fallback'
        adapter = TypeAdapter(CapabilityEvent)
        assert adapter.validate_json(adapter.dump_json(event)) == event
        spans = exporter.get_finished_spans()
        [transition] = [s for s in spans if s.name == 'model_fallback.output_fallback']
        attributes: dict[str, object] = dict(transition.attributes or {})
        assert attributes.pop('model_fallback.producer', None) == producer
        assert attributes == snapshot(
            {
                'model_fallback.model': 'test',
                'model_fallback.source': 'response_output',
            }
        )
        assert len({s.context.trace_id for s in spans if s.context is not None}) == 1
        assert len([s for s in spans if s.parent is None]) == 1

    async def test_concurrent_runs_and_history_start_at_first_level(self) -> None:
        recorder = RecordRetries()
        agent = Agent(
            capabilities=[
                ModelFallback(
                    TestModel(custom_output_text='draft'), TestModel(custom_output_text='ready'), output_retries=1
                ),
                recorder,
            ]
        )

        @agent.output_validator
        def validate(value: str) -> str:
            if value == 'draft':
                raise ModelRetry('Finish the draft.')
            return value

        async def run() -> None:
            assert (await agent.run('Answer.')).output == 'ready'

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            group.start_soon(run)
        first = await agent.run('Answer.')
        assert (await agent.run('Answer again.', message_history=first.all_messages())).output == 'ready'
        assert [event.response.text for event in recorder.events] == ['draft'] * 4

    @pytest.mark.parametrize('method', ['run', 'handler', 'events'])
    async def test_failed_output_sibling_is_reported_without_another_request(
        self, method: Literal['run', 'handler', 'events']
    ) -> None:
        calls: list[str] = []

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            calls.append('primary')
            yield {
                0: DeltaToolCall(name='first', json_args='{"number": "bad"}', tool_call_id='one'),
                1: DeltaToolCall(name='second', json_args='{"number": 42}', tool_call_id='two'),
            }

        recorder = RecordRetries()
        agent = Agent(
            output_type=[ToolOutput(Answer, name='first'), ToolOutput(Answer, name='second')],
            end_strategy='exhaustive',
            capabilities=[
                ModelFallback(
                    FunctionModel(stream_function=stream),
                    TestModel(custom_output_args={'number': -1}),
                    output_retries=1,
                ),
                recorder,
            ],
        )
        streamed_events: list[AgentStreamEvent] = []
        if method == 'handler':

            async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
                streamed_events.extend([event async for event in stream])

            assert (await agent.run('Answer.', event_stream_handler=collect)).output == Answer(number=42)
        elif method == 'events':
            async with agent.run_stream_events('Answer.') as event_stream:
                async for event in event_stream:
                    if isinstance(event, AgentRunResultEvent):
                        assert event.result.output == Answer(number=42)
                    else:
                        streamed_events.append(event)
        else:
            assert (await agent.run('Answer.')).output == Answer(number=42)
        [event] = recorder.events
        assert (event.source, event.tool_name, event.tool_call_id) == ('output_tool', 'first', 'one')
        assert (event.retry.attempt, event.retry.max_retries) == (1, 1)
        assert calls == ['primary']
        if method != 'run':
            assert [e for e in streamed_events if isinstance(e, OutputRetryEvent)] == [event]
