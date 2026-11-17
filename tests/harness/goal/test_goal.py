"""Public completion and retry behavior of `Goal`."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.goal import Goal
from tests.harness._recording_durability import RecordingDurability


async def complete(ctx: RunContext[object], output: object) -> str | None:
    return None if output == 'done' else 'Produce the completed work.'


class Result(BaseModel):
    done: bool


def _traced_context(include_content: bool) -> tuple[RunContext[None], InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        tracer=provider.get_tracer('test'),
        trace_include_content=include_content,
    )
    return ctx, exporter


class TestGoal:
    async def test_retries_with_goal_and_gap(self) -> None:
        calls = 0

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            assert info.instructions is not None
            assert 'Finish the work' in info.instructions
            assert 'unattended' in info.instructions
            if calls == 1:
                return ModelResponse(parts=[TextPart('Should I continue?')])
            (feedback,) = messages[-1].parts
            assert isinstance(feedback, UserPromptPart)
            assert 'Goal not met: Finish the work\nWhat remains: Produce the completed work.' in str(feedback.content)
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(FunctionModel(respond), capabilities=[Goal(goal='Finish the work', verify=complete)])
        assert (await agent.run('Start')).output == 'done'
        assert calls == 2

    async def test_exhaustion_is_bounded(self) -> None:
        checks = 0

        async def reject(ctx: RunContext[object], output: object) -> str:
            nonlocal checks
            checks += 1
            return 'Not done.'

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart('Not finished')])

        agent = Agent(
            FunctionModel(respond),
            retries={'output': 2},
            capabilities=[Goal(goal='Finish', verify=reject)],
        )
        with pytest.raises(UnexpectedModelBehavior, match='retries'):
            await agent.run('Start')
        assert checks == 3

    async def test_usage_limits_remain_authoritative(self) -> None:
        agent = Agent(TestModel(), retries={'output': 5}, capabilities=[Goal(goal='Finish', verify=complete)])
        with pytest.raises(UsageLimitExceeded):
            await agent.run('Start', usage_limits=UsageLimits(request_limit=1))

    async def test_interactive_skips_verifier(self) -> None:
        async def fail(ctx: RunContext[object], output: object) -> None:
            pytest.fail('Interactive runs must not enforce completion')  # pragma: no cover

        goal = Goal(goal='Finish', verify=fail, headless=False)
        assert goal.get_instructions() == 'Your goal is:\nFinish'
        agent = Agent(TestModel(custom_output_text='Which file?'), capabilities=[goal])
        assert (await agent.run('Start')).output == 'Which file?'

    async def test_typed_output_and_dependencies(self) -> None:
        async def verify(ctx: RunContext[str], output: object) -> str | None:
            assert ctx.deps == 'evidence'
            assert isinstance(output, Result)
            assert ctx.messages
            return None

        agent = Agent(
            TestModel(custom_output_args={'done': True}),
            deps_type=str,
            output_type=Result,
            capabilities=[Goal(goal='Finish', verify=verify)],
        )
        assert (await agent.run('Start', deps='evidence')).output.done

    async def test_verifier_errors_propagate(self) -> None:
        async def fail(ctx: RunContext[object], output: object) -> None:
            raise ValueError('Evidence unavailable')

        agent = Agent(TestModel(), capabilities=[Goal(goal='Finish', verify=fail)])
        with pytest.raises(ValueError, match='Evidence unavailable'):
            await agent.run('Start')

    async def test_empty_gap_is_configuration_error(self) -> None:
        async def empty(ctx: RunContext[object], output: object) -> str:
            return ' '

        agent = Agent(TestModel(), capabilities=[Goal(goal='Finish', verify=empty)])
        with pytest.raises(UserError, match='nonempty explanation'):
            await agent.run('Start')

    async def test_reuse_and_multiple_goals(self) -> None:
        goal = Goal(goal='Finish', verify=complete)
        assert goal.get_serialization_name() is None
        agent = Agent(
            TestModel(custom_output_text='done'),
            capabilities=[goal, Goal(goal='Also finish', verify=complete, id='also')],
        )
        for _ in range(2):
            assert (await agent.run('Start')).output == 'done'

    def test_goals_sharing_an_id_must_agree(self) -> None:
        goal = Goal(goal='Finish', verify=complete)
        assert Agent(TestModel(), capabilities=[goal, Goal(goal='Finish', verify=complete)])
        with pytest.raises(UserError, match=r"disagree on 'goal'\. Give each its own `id`"):
            Agent(TestModel(), capabilities=[goal, Goal(goal='Also finish', verify=complete)])

    @pytest.mark.parametrize('goal', ['', ' \n'])
    def test_empty_goal(self, goal: str) -> None:
        with pytest.raises(UserError, match='nonempty goal'):
            Goal(goal=goal, verify=complete)

    @pytest.mark.parametrize('include_content', [False, True])
    @pytest.mark.parametrize('accepted', [False, True])
    async def test_telemetry(self, *, include_content: bool, accepted: bool) -> None:
        ctx, exporter = _traced_context(include_content)
        goal = Goal(goal='Finish', verify=complete)
        output_context = OutputContext(mode='text', output_type=str, object_def=None, has_function=False)
        if accepted:
            assert await goal.after_output_process(ctx, output_context=output_context, output='done') == 'done'
        else:
            with pytest.raises(ModelRetry):
                await goal.after_output_process(ctx, output_context=output_context, output='question')
        (span,) = exporter.get_finished_spans()
        assert span.name == 'goal.verify'
        expected: dict[str, str | bool] = {'goal.met': accepted}
        if include_content:
            expected['goal.description'] = 'Finish'
            if not accepted:
                expected['goal.gap'] = 'Produce the completed work.'
        assert dict(span.attributes or {}) == expected

    @pytest.mark.parametrize('include_content', [False, True])
    async def test_verifier_error_telemetry_follows_content_setting(self, include_content: bool) -> None:
        async def fail(ctx: RunContext[None], output: object) -> None:
            raise ValueError(f'Evidence unavailable for {output}')

        ctx, exporter = _traced_context(include_content)
        output_context = OutputContext(mode='text', output_type=str, object_def=None, has_function=False)
        with pytest.raises(ValueError):
            await Goal(goal='Finish', verify=fail).after_output_process(
                ctx, output_context=output_context, output='secret'
            )
        (span,) = exporter.get_finished_spans()
        if include_content:
            assert span.status.description == 'ValueError: Evidence unavailable for secret'
            assert [event.name for event in span.events] == ['exception']
        else:
            assert span.status.description is None
            assert span.events == ()

    async def test_partial_output_is_not_verified(self) -> None:
        ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), partial_output=True)
        goal = Goal(goal='Finish', verify=complete)
        output_context = OutputContext(mode='text', output_type=str, object_def=None, has_function=False)
        assert await goal.after_output_process(ctx, output_context=output_context, output='partial') == 'partial'

    async def test_verification_dispatches_as_a_durable_operation(self) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='goal_agent',
            capabilities=[Goal(goal='Finish', verify=complete), RecordingDurability()],
        )
        assert (await agent.run('Start')).output == 'done'
        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        assert 'goal_agent__capability__goal.verify' in {name for name, _ in bound.calls}
