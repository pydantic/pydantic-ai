"""Public completion and retry behavior of `Goal`."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities.instrumentation import Instrumentation
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.goal import Goal
from pydantic_ai_harness.wrap_up_at_limits import WrapUpAtLimits
from tests.harness._recording_durability import RecordingDurability


async def complete(ctx: RunContext[object], output: object) -> str | None:
    return None if output == 'done' else 'Produce the completed work.'


def question_then_done(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('Which file?' if len(messages) == 1 else 'done')])


class Result(BaseModel):
    done: bool


def _instrumented(include_content: bool) -> tuple[Instrumentation, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    settings = InstrumentationSettings(tracer_provider=provider, include_content=include_content)
    return Instrumentation(settings=settings), exporter


def _goal_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    spans = [span for span in exporter.get_finished_spans() if span.name == 'goal.verify']
    assert all(span.parent is not None for span in spans)
    return spans


class TestGoal:
    @pytest.mark.parametrize('headless', [True, False])
    async def test_retries_with_goal_and_gap(self, headless: bool) -> None:
        instructions: list[str | None] = []
        feedback: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions.append(info.instructions)
            if len(messages) > 1:
                (part,) = messages[-1].parts
                assert isinstance(part, UserPromptPart)
                feedback.append(str(part.content))
            return question_then_done(messages, info)

        goal = Goal(goal='Finish the work', verify=complete, headless=headless)
        agent = Agent(FunctionModel(respond), capabilities=[goal])
        assert (await agent.run('Start')).output == 'done'
        (first, second) = instructions
        assert first == second
        assert {'instructions': first, 'retry': feedback} == snapshot(
            {
                True: {
                    'instructions': """\
Your goal is:
Finish the work
This run is unattended: nobody is available to answer questions. Work toward the goal before producing a final answer. Make reasonable decisions within the supplied constraints. Do not invent missing credentials, bypass approvals, or claim success without evidence. If completion is impossible, explain the blocker accurately.\
""",
                    'retry': [
                        """\
Your previous response failed the application's validation (this is automated feedback, not a message from the user):
<pyai:retry_feedback>
Goal not met: Finish the work
What remains: Produce the completed work.
Nobody is available to answer questions. Continue within the supplied constraints; do not bypass approvals or fabricate success.
</pyai:retry_feedback>\
"""
                    ],
                },
                False: {
                    'instructions': """\
Your goal is:
Finish the work\
""",
                    'retry': [
                        """\
Your previous response failed the application's validation (this is automated feedback, not a message from the user):
<pyai:retry_feedback>
Goal not met: Finish the work
What remains: Produce the completed work.
Continue within the supplied constraints; do not bypass approvals or fabricate success.
</pyai:retry_feedback>\
"""
                    ],
                },
            }
        )[headless]

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

    @pytest.mark.parametrize('final_answer', ['done', 'Which file?'])
    async def test_wrap_up_answer_still_needs_the_goal_met(self, final_answer: str) -> None:
        def research(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.model_settings and info.model_settings.get('tool_choice') == 'none':
                return ModelResponse(parts=[TextPart(final_answer)])
            return ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'x'})])

        agent = Agent(FunctionModel(research), capabilities=[WrapUpAtLimits(), Goal(goal='Finish', verify=complete)])

        @agent.tool_plain
        def look_up(topic: str) -> str:
            return topic

        limits = UsageLimits(request_limit=2)
        if final_answer == 'done':
            assert (await agent.run('Start', usage_limits=limits)).output == 'done'
        else:
            with pytest.raises(UsageLimitExceeded):
                await agent.run('Start', usage_limits=limits)

    async def test_without_verifier_only_instructs(self) -> None:
        goal = Goal(goal='Finish', headless=False)
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
        with pytest.raises(UserError, match=r'The `Goal` verifier must return `None` or a nonempty explanation\.'):
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

    async def test_goals_sharing_an_id_must_agree(self) -> None:
        checks: list[object] = []

        async def verify(ctx: RunContext[object], output: object) -> None:
            checks.append(output)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.instructions is not None
            assert info.instructions.count('Your goal is:') == 1
            return ModelResponse(parts=[TextPart('done')])

        agent = Agent(
            FunctionModel(respond),
            capabilities=[Goal(goal='Finish', verify=verify), Goal(goal='Finish', verify=verify)],
        )
        assert (await agent.run('Start')).output == 'done'
        assert checks == ['done']
        goal = Goal(goal='Finish', verify=complete)
        with pytest.raises(UserError, match=r"disagree on 'goal'\. Give each its own `id`"):
            Agent(TestModel(), capabilities=[goal, Goal(goal='Also finish', verify=complete)])

    @pytest.mark.parametrize('goal', ['', ' \n'])
    def test_empty_goal(self, goal: str) -> None:
        with pytest.raises(UserError, match=r'`Goal` requires a nonempty `goal`\.'):
            Goal(goal=goal, verify=complete)

    @pytest.mark.parametrize('include_content', [False, True])
    async def test_telemetry(self, include_content: bool) -> None:
        instrumentation, exporter = _instrumented(include_content)
        agent = Agent(
            FunctionModel(question_then_done),
            capabilities=[instrumentation, Goal(goal='Finish', verify=complete)],
        )
        assert (await agent.run('Start')).output == 'done'
        rejected, accepted = (dict(span.attributes or {}) for span in _goal_spans(exporter))
        content = {'goal.description': 'Finish'} if include_content else {}
        gap = {'goal.gap': 'Produce the completed work.'} if include_content else {}
        assert rejected == {'goal.met': False, **content, **gap}
        assert accepted == {'goal.met': True, **content}

    @pytest.mark.parametrize('include_content', [False, True])
    async def test_verifier_error_telemetry_follows_content_setting(self, include_content: bool) -> None:
        async def fail(ctx: RunContext[object], output: object) -> None:
            raise ValueError(f'Evidence unavailable for {output}')

        instrumentation, exporter = _instrumented(include_content)
        agent = Agent(
            TestModel(custom_output_text='secret'),
            capabilities=[instrumentation, Goal(goal='Finish', verify=fail)],
        )
        with pytest.raises(ValueError):
            await agent.run('Start')
        (span,) = _goal_spans(exporter)
        if include_content:
            assert span.status.description == 'ValueError: Evidence unavailable for secret'
            assert [event.name for event in span.events] == ['exception']
        else:
            assert span.status.description is None
            assert span.events == ()

    async def test_partial_streamed_output_is_not_verified(self) -> None:
        checked: list[object] = []

        async def verify(ctx: RunContext[object], output: object) -> None:
            checked.append(output)

        agent = Agent(
            TestModel(custom_output_text='the work is done'),
            capabilities=[Goal(goal='Finish', verify=verify)],
        )
        async with agent.run_stream('Start') as result:
            streamed = [output async for output in result.stream_output(debounce_by=None)]
        assert len(streamed) > 1
        assert checked == ['the work is done']

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

    async def test_durable_verifier_sees_json_output_and_its_usage_reaches_the_run(self) -> None:
        judge = Agent(TestModel())
        seen: list[object] = []

        async def verify(ctx: RunContext[object], output: object) -> None:
            seen.append(output)
            await judge.run('Is it done?', usage=ctx.usage)

        agent = Agent(
            TestModel(custom_output_args={'done': True}),
            name='goal_agent',
            output_type=Result,
            capabilities=[Goal(goal='Finish', verify=verify), RecordingDurability(isolated=True)],
        )
        result = await agent.run('Start')
        assert result.output == Result(done=True)
        assert seen == [{'done': True}]
        assert result.usage.requests == 2
