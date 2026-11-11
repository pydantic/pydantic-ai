"""`SubAgents` says on the delegation's `execute_tool` span how it ended and which model it ran on."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.subagents import SubAgent, SubAgents


def instrumented(*, include_content: bool = False) -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=include_content)
    return Instrumentation(settings=settings), exporter


def delegating(times: int = 1) -> FunctionModel:
    """A parent model that delegates `do it` to `worker` `times` times, then replies with text."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls = sum(isinstance(message, ModelResponse) for message in messages)
        if calls < times:
            args = {'agent_name': 'worker', 'task': 'do it'}
            return ModelResponse(parts=[ToolCallPart('delegate_task', args, tool_call_id=f'c{calls}')])
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(respond)


def delegation_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == 'execute_tool delegate_task']


def sub_agents_attributes(span: ReadableSpan) -> dict[str, Any]:
    return {key: value for key, value in (span.attributes or {}).items() if key.startswith('sub_agents.')}


async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    await asyncio.sleep(1)
    return ModelResponse(parts=[TextPart('late')])  # pragma: no cover - cancelled by the timeout


def unexpected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise UnexpectedModelBehavior('kaboom')


def crash(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError('m', 'provider down')


def noop() -> str:
    return 'x'


class TestSubAgentsTelemetry:
    @pytest.mark.parametrize(
        ('sub_agent', 'outcome', 'model'),
        [
            (SubAgent(Agent(TestModel(), name='worker')), 'ok', 'test:test'),
            (
                SubAgent(Agent(FunctionModel(slow), name='worker'), timeout_seconds=0.01),
                'timeout',
                'function:function:slow:',
            ),
            (
                SubAgent(Agent(TestModel(), name='worker', tools=[noop]), usage_limits=UsageLimits(request_limit=1)),
                'budget',
                'test:test',
            ),
            (SubAgent(Agent(FunctionModel(unexpected), name='worker')), 'failed', 'function:function:unexpected:'),
            (
                SubAgent(Agent(FunctionModel(crash), name='worker'), contain_errors=True),
                'contained',
                'function:function:crash:',
            ),
        ],
    )
    async def test_how_a_delegation_ended_is_recorded(self, sub_agent: SubAgent[object], outcome: str, model: str):
        instrumentation, exporter = instrumented()
        await Agent(delegating(), capabilities=[SubAgents(agents=[sub_agent]), instrumentation]).run('go')

        [span] = delegation_spans(exporter)
        assert sub_agents_attributes(span) == {'sub_agents.outcome': outcome, 'sub_agents.model': model}

    async def test_a_timeout_is_a_successful_call_and_quotes_nothing_without_content(self):
        worker = SubAgent(Agent(FunctionModel(slow), name='worker'), timeout_seconds=0.01)
        instrumentation, exporter = instrumented()
        await Agent(delegating(), capabilities=[SubAgents(agents=[worker]), instrumentation]).run('go')

        [span] = delegation_spans(exporter)
        assert span.status.status_code == StatusCode.UNSET
        assert sub_agents_attributes(span)['sub_agents.outcome'] == 'timeout'
        assert 'do it' not in json.dumps(dict(span.attributes or {}))

    async def test_an_uncontained_crash_is_an_error_with_its_outcome(self):
        worker = SubAgent(Agent(FunctionModel(crash), name='worker'))
        instrumentation, exporter = instrumented()
        with pytest.raises(ModelAPIError):
            await Agent(delegating(), capabilities=[SubAgents(agents=[worker]), instrumentation]).run('go')

        [span] = delegation_spans(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert sub_agents_attributes(span) == {
            'sub_agents.outcome': 'error',
            'sub_agents.model': 'function:function:crash:',
        }

    async def test_an_exhausted_max_calls_budget_is_recorded_without_a_model(self):
        worker = SubAgent(Agent(TestModel(), name='worker'), max_calls=1)
        instrumentation, exporter = instrumented()
        await Agent(delegating(times=2), capabilities=[SubAgents(agents=[worker]), instrumentation]).run('go')

        assert [sub_agents_attributes(span) for span in delegation_spans(exporter)] == snapshot(
            [{'sub_agents.outcome': 'ok', 'sub_agents.model': 'test:test'}, {'sub_agents.outcome': 'max_calls'}]
        )

    async def test_an_uninstrumented_run_leaves_the_applications_span_alone(self):
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        worker = SubAgent(Agent(TestModel(), name='worker'))
        with provider.get_tracer('app').start_as_current_span('app request'):
            await Agent(delegating(), capabilities=[SubAgents(agents=[worker])]).run('go')

        assert [(span.name, dict(span.attributes or {})) for span in exporter.get_finished_spans()] == [
            ('app request', {})
        ]
