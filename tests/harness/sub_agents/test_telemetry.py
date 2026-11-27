"""`SubAgents` says on the delegation's `execute_tool` span how it ended and which model it ran on."""

from __future__ import annotations

import anyio
import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.sub_agents import DynamicAgents, SubAgent, SubAgents
from tests.harness._recording_durability import RecordingDurability
from tests.harness.conftest import Instrumented


def calling(*calls: tuple[str, dict[str, str]]) -> FunctionModel:
    """A parent model that makes each `(tool name, arguments)` call in turn, then replies with text."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step < len(calls):
            tool_name, args = calls[step]
            return ModelResponse(parts=[ToolCallPart(tool_name, args, tool_call_id=f'c{step}')])
        return ModelResponse(parts=[TextPart('all done')])

    return FunctionModel(respond)


def delegating(times: int = 1, agent_name: str = 'worker') -> FunctionModel:
    """A parent model that delegates `do it` to `agent_name` `times` times, then replies with text."""
    return calling(*[('delegate_task', {'agent_name': agent_name, 'task': 'do it'})] * times)


def delegation_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == 'execute_tool delegate_task']


def sub_agents_attributes(span: ReadableSpan) -> dict[str, AttributeValue]:
    return {key: value for key, value in (span.attributes or {}).items() if key.startswith('sub_agents.')}


async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    await anyio.sleep_forever()
    # Cancelled by the timeout.
    return ModelResponse(parts=[TextPart('late')])  # pragma: no cover


def unexpected(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise UnexpectedModelBehavior('kaboom')


def crash(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError('m', 'provider down')


def noop() -> str:
    return 'x'


def calls_stop(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('stop', {}, tool_call_id='s1')])


def cancelling_agent() -> Agent[object, str]:
    """A sub-agent whose `stop` tool cancels its own run."""
    agent = Agent(FunctionModel(calls_stop), name='worker')

    @agent.tool
    def stop(ctx: RunContext[object]) -> str:
        ctx.cancel()
        return 'ignored'

    return agent


class TestSubAgentsTelemetry:
    @pytest.mark.parametrize(
        ('sub_agent', 'outcome', 'model'),
        [
            # A plain `ok` is covered by the first delegation in the `max_calls` test.
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
            (SubAgent(cancelling_agent()), 'cancelled', 'function:function:calls_stop:'),
            # The model that answered, not the fallback chain.
            (SubAgent(Agent(FallbackModel(FunctionModel(crash), TestModel()), name='worker')), 'ok', 'test:test'),
        ],
    )
    async def test_how_a_delegation_ended_is_recorded(
        self,
        sub_agent: SubAgent[object],
        outcome: str,
        model: str,
        instrumented: Instrumented,
    ):
        instrumentation, exporter = instrumented(False)
        await Agent(delegating(), capabilities=[SubAgents(agents=[sub_agent]), instrumentation]).run('go')

        [span] = delegation_spans(exporter)
        assert sub_agents_attributes(span) == {
            'sub_agents.agent_name': 'worker',
            'sub_agents.outcome': outcome,
            'sub_agents.model': model,
        }

    @pytest.mark.parametrize(
        ('isolated', 'expected'),
        [
            # The unit runs in the delegating run's context, so its own run records the model on the same span.
            pytest.param(
                False,
                {'sub_agents.agent_name': 'worker', 'sub_agents.outcome': 'ok', 'sub_agents.model': 'test:test'},
                id='inline',
            ),
            # As on a worker: only the delegating side's record, without the model, reaches this span.
            pytest.param(True, {'sub_agents.agent_name': 'worker', 'sub_agents.outcome': 'ok'}, id='isolated'),
        ],
    )
    async def test_a_delegation_in_its_own_durable_unit_records_its_outcome(
        self, isolated: bool, expected: dict[str, AttributeValue], instrumented: Instrumented
    ):
        # Created agents give plain delegation a durable unit of its own.
        sub_agents = SubAgents[object](
            agents=[SubAgent(Agent(TestModel(), name='worker'))],
            dynamic_agents=DynamicAgents(factory=lambda spec: Agent(TestModel())),
        )
        durability = RecordingDurability(isolated=isolated)
        instrumentation, exporter = instrumented(False)
        await Agent(delegating(), name='durable', capabilities=[sub_agents, durability, instrumentation]).run('go')

        assert 'durable__capability__sub_agents.delegate' in {name for name, _ in durability.calls}
        [span] = delegation_spans(exporter)
        assert sub_agents_attributes(span) == expected

    @pytest.mark.parametrize('include_content', [False, True])
    async def test_a_created_agents_name_is_content(self, include_content: bool, instrumented: Instrumented):
        create = ('create_agent', {'name': 'poet', 'description': 'd', 'instructions': 'i'})
        delegate = ('delegate_task', {'agent_name': 'poet', 'task': 'do it'})
        sub_agents = SubAgents[object](models={'fast': TestModel()}, dynamic_agents=DynamicAgents(models=['fast']))
        instrumentation, exporter = instrumented(include_content)
        await Agent(calling(create, delegate), capabilities=[sub_agents, instrumentation]).run('go')

        [span] = delegation_spans(exporter)
        name = {'sub_agents.agent_name': 'poet'} if include_content else {}
        assert sub_agents_attributes(span) == {**name, 'sub_agents.outcome': 'ok', 'sub_agents.model': 'test:test'}

    async def test_an_uncontained_crash_is_an_error_with_its_outcome(self, instrumented: Instrumented):
        worker = SubAgent(Agent(FunctionModel(crash), name='worker'))
        instrumentation, exporter = instrumented(False)
        with pytest.raises(ModelAPIError):
            await Agent(delegating(), capabilities=[SubAgents(agents=[worker]), instrumentation]).run('go')

        [span] = delegation_spans(exporter)
        assert span.status.status_code == StatusCode.ERROR
        assert sub_agents_attributes(span) == {
            'sub_agents.agent_name': 'worker',
            'sub_agents.outcome': 'error',
            'sub_agents.model': 'function:function:crash:',
        }

    async def test_an_exhausted_max_calls_budget_is_recorded_without_a_model(self, instrumented: Instrumented):
        worker = SubAgent(Agent(TestModel(), name='worker'), max_calls=1)
        instrumentation, exporter = instrumented(False)
        await Agent(delegating(times=2), capabilities=[SubAgents(agents=[worker]), instrumentation]).run('go')

        assert [sub_agents_attributes(span) for span in delegation_spans(exporter)] == snapshot(
            [
                {'sub_agents.agent_name': 'worker', 'sub_agents.outcome': 'ok', 'sub_agents.model': 'test:test'},
                {'sub_agents.agent_name': 'worker', 'sub_agents.outcome': 'max_calls'},
            ]
        )
