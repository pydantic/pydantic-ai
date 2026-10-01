"""Tests for `ToolCallJudge`."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import is_dataclass
from typing import Any, Literal

import pytest
from inline_snapshot import snapshot
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer

from pydantic_ai import Agent, AgentSpec, DeferredToolRequests
from pydantic_ai.exceptions import ApprovalRequired, SkipToolExecution, UsageLimitExceeded, UserError
from pydantic_ai.messages import (
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolResults, RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness.tool_call_judge import ToolCallJudge, ToolCallVerdict
from tests.harness._recording_durability import RecordingDurability

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _judge_model(
    answer: Literal['yes', 'no', 'unsure'],
    *,
    confidence: float | None = None,
    prompts: list[str] | None = None,
    instructions: list[str] | None = None,
    provider_details: dict[str, object] | None = None,
) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if prompts is not None:
            prompts.extend(
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
        if instructions is not None and info.instructions is not None:
            instructions.append(info.instructions)
        output_tool = info.output_tools[0]
        details = provider_details
        if details is None and confidence is not None:
            details = {'confidence': {'response': confidence}}
        return ModelResponse(
            parts=[ToolCallPart(output_tool.name, {'response': answer})],
            provider_details=details,
        )

    return FunctionModel(respond)


def _yielding_judge_model(answer: Literal['yes', 'no', 'unsure'], *, calls: list[str]) -> FunctionModel:
    """A judge that yields to the event loop, so parallel judgements overlap."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append(answer)
        await asyncio.sleep(0)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'response': answer})])

    return FunctionModel(respond)


def _error_model() -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('judge unavailable')

    return FunctionModel(respond)


def _outer_model(*calls: ToolCallPart, returns: list[ToolReturnPart] | None = None) -> FunctionModel:
    """Request `calls` on the first step, then finish, recording the tool returns it saw."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if tool_returns:
            if returns is not None:
                returns.extend(tool_returns)
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=list(calls))

    return FunctionModel(respond)


def _judge(model: FunctionModel, **kwargs: Any) -> ToolCallJudge[None]:
    kwargs.setdefault('tools', ['danger'])
    kwargs.setdefault('question', 'Would this cause harm?')
    kwargs.setdefault('denial_message', 'judge blocked {tool_name}')
    return ToolCallJudge(model, **kwargs)


def _agent(
    judge: ToolCallJudge[None],
    outer: FunctionModel,
    *,
    ran: list[str],
    requires_approval: bool = False,
    extra: ToolCallJudge[None] | None = None,
) -> Agent[None, str | DeferredToolRequests]:
    capabilities = [judge] if extra is None else [judge, extra]
    agent = Agent[None, 'str | DeferredToolRequests'](
        outer,
        deps_type=type(None),
        capabilities=capabilities,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(requires_approval=requires_approval)
    def danger(x: int) -> str:
        ran.append(f'danger:{x!r}')
        return f'danger ran with {x!r}'

    @agent.tool_plain
    def safe(x: int) -> str:
        ran.append('safe')
        return 'safe ran'

    return agent


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


def _ctx(
    *,
    tracer: Tracer | None = None,
    trace_include_content: bool = False,
    messages: list[ModelMessage] | None = None,
) -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        tracer=tracer if tracer is not None else NoOpTracer(),
        trace_include_content=trace_include_content,
        messages=messages or [],
    )


def _tool_def(name: str = 'danger', *, metadata: dict[str, Any] | None = None) -> ToolDefinition:
    return ToolDefinition(name=name, metadata=metadata)


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return spans[0]


class TestConfiguration:
    def test_is_a_dataclass_with_capability_fields(self) -> None:
        judge = _judge(_judge_model('no'), id='gate', description='blocks risky calls')
        assert is_dataclass(judge)
        assert judge.id == 'gate'
        assert judge.description == 'blocks risky calls'
        assert judge.tools == ['danger']

    def test_judges_every_tool_by_default(self) -> None:
        judge = ToolCallJudge(_judge_model('no'), question='Would this cause harm?')
        assert judge.tools == 'all'

    def test_rejects_an_empty_question(self) -> None:
        with pytest.raises(UserError, match='question must not be empty'):
            ToolCallJudge(_judge_model('no'), question='   ')

    def test_rejects_an_empty_denial_message(self) -> None:
        with pytest.raises(UserError, match='denial_message must not be empty'):
            _judge(_judge_model('no'), denial_message='')

    def test_rejects_an_unknown_denial_message_placeholder(self) -> None:
        with pytest.raises(UserError, match='invalid `denial_message` placeholder'):
            _judge(_judge_model('no'), denial_message='blocked {reason}')

    def test_rejects_a_non_positive_conversation_window(self) -> None:
        with pytest.raises(UserError, match='conversation_window must be at least 1 token'):
            _judge(_judge_model('no'), conversation_window=0)

    def test_agent_spec_builds_the_serializable_configuration(self) -> None:
        judge = ToolCallJudge[None].from_spec(
            model='test',
            question='Would this cause harm?',
            tools=['danger'],
            include_conversation=True,
            conversation_window=100,
            on_uncertain='ask',
            denial_message='blocked',
            id='gate',
        )
        assert judge.model == 'test'
        assert judge.tools == ['danger']
        assert judge.include_conversation is True
        assert judge.conversation_window == 100
        assert judge.on_uncertain == 'ask'
        assert judge.denial_message == 'blocked'
        assert judge.id == 'gate'

    def test_agent_spec_schema_includes_configuration_and_base_fields(self) -> None:
        schema = AgentSpec.model_json_schema_with_capabilities([ToolCallJudge])
        params = schema['$defs']['spec_params_ToolCallJudge']
        properties = params['properties']

        assert params['additionalProperties'] is False
        assert set(properties) == {
            'model',
            'question',
            'tools',
            'include_conversation',
            'conversation_window',
            'on_uncertain',
            'denial_message',
            'id',
            'description',
            'defer_loading',
        }
        assert set(params['required']) == {'model', 'question'}
        assert ToolCallJudge.get_serialization_name() == 'ToolCallJudge'

    def test_agent_spec_loads_the_capability(self) -> None:
        agent = Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {
                        'ToolCallJudge': {
                            'model': 'test',
                            'question': 'Would this cause harm?',
                            'tools': ['danger'],
                        }
                    }
                ],
            },
            custom_capability_types=[ToolCallJudge],
        )

        capability = agent.root_capability.capabilities[-1]
        assert isinstance(capability, ToolCallJudge)
        assert capability.question == 'Would this cause harm?'
        assert capability.tools == ['danger']
        assert capability.on_uncertain == 'block'


class TestDecisions:
    async def test_no_lets_the_call_run(self) -> None:
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('no'), on_verdict=verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )
        result = await agent.run('go')
        assert result.output == 'done'
        assert ran == ['danger:1']
        assert [(v.tool_name, v.verdict, v.answer) for v in verdicts] == [('danger', 'allow', 'no')]

    async def test_yes_blocks_the_call_before_the_body_runs(self) -> None:
        ran: list[str] = []
        returns: list[ToolReturnPart] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('yes', confidence=0.9), on_verdict=verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'), returns=returns),
            ran=ran,
        )
        result = await agent.run('go')
        assert result.output == 'done'
        assert ran == []
        assert [part.content for part in returns] == ['judge blocked danger']
        assert returns[0].metadata == {
            'tool_call_judge': {'answer': 'yes', 'confidence': 0.9, 'question': 'Would this cause harm?'}
        }
        assert [(v.verdict, v.answer, v.confidence) for v in verdicts] == [('block', 'yes', 0.9)]

    @pytest.mark.parametrize(
        ('model', 'answer'),
        [(_judge_model('unsure'), 'unsure'), (_error_model(), None)],
        ids=['unsure', 'model-error'],
    )
    async def test_uncertainty_blocks_by_default(self, model: FunctionModel, answer: str | None) -> None:
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(model, on_verdict=verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )
        await agent.run('go')
        assert ran == []
        assert [(v.verdict, v.answer) for v in verdicts] == [('block', answer)]

    async def test_uncertainty_can_be_allowed_through(self) -> None:
        ran: list[str] = []
        agent = _agent(
            _judge(_judge_model('unsure'), on_uncertain='allow'),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )
        await agent.run('go')
        assert ran == ['danger:1']

    async def test_uncertainty_can_be_handed_to_a_person(self) -> None:
        ran: list[str] = []
        agent = _agent(
            _judge(_judge_model('unsure'), on_uncertain='ask'),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )
        result = await agent.run('go')
        assert isinstance(result.output, DeferredToolRequests)
        assert [call.tool_name for call in result.output.approvals] == ['danger']
        assert ran == []

    async def test_the_judge_sees_validated_arguments_not_the_raw_json(self) -> None:
        prompts: list[str] = []
        ran: list[str] = []
        agent = _agent(
            _judge(_judge_model('no', prompts=prompts)),
            _outer_model(ToolCallPart('danger', {'x': '1'}, tool_call_id='call-1')),
            ran=ran,
        )
        await agent.run('go')
        assert ran == ['danger:1']
        assert prompts == ['<tool_call>\n{"tool_name": "danger", "arguments": {"x": 1}}\n</tool_call>']

    async def test_the_question_leads_the_judge_instructions(self) -> None:
        instructions: list[str] = []
        agent = _agent(
            _judge(_judge_model('no', instructions=instructions)),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=[],
        )
        await agent.run('go')
        assert instructions[0].startswith('Would this cause harm?')
        assert 'untrusted' in instructions[0]


class TestToolSelection:
    async def test_unselected_tools_run_unjudged(self) -> None:
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('yes'), on_verdict=verdicts.append),
            _outer_model(ToolCallPart('safe', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )
        await agent.run('go')
        assert ran == ['safe']
        assert verdicts == []

    async def test_a_mixed_response_judges_only_the_selected_call(self) -> None:
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('yes'), on_verdict=verdicts.append),
            _outer_model(
                ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
                ToolCallPart('safe', {'x': 2}, tool_call_id='call-2'),
            ),
            ran=ran,
        )
        await agent.run('go')
        assert ran == ['safe']
        assert [(v.tool_name, v.tool_call_id) for v in verdicts] == [('danger', 'call-1')]

    async def test_all_judges_every_tool(self) -> None:
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('yes'), tools='all', on_verdict=verdicts.append),
            _outer_model(
                ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
                ToolCallPart('safe', {'x': 2}, tool_call_id='call-2'),
            ),
            ran=ran,
        )
        await agent.run('go')
        assert ran == []
        assert sorted(v.tool_name for v in verdicts) == ['danger', 'safe']

    async def test_a_predicate_selector_decides_per_tool(self) -> None:
        def only_danger(ctx: RunContext[None], tool_def: ToolDefinition) -> bool:
            return tool_def.name == 'danger'

        judge = _judge(_judge_model('yes'), tools=only_danger)
        args = await judge.before_tool_execute(
            _ctx(), call=ToolCallPart('safe', {}, tool_call_id='c'), tool_def=_tool_def('safe'), args={}
        )
        assert args == {}

    async def test_a_metadata_selector_decides_per_tool(self) -> None:
        judge = _judge(_judge_model('yes'), tools={'risk': 'high'})
        args = await judge.before_tool_execute(
            _ctx(),
            call=ToolCallPart('danger', {}, tool_call_id='c'),
            tool_def=_tool_def(metadata={'risk': 'low'}),
            args={},
        )
        assert args == {}
        with pytest.raises(SkipToolExecution):
            await judge.before_tool_execute(
                _ctx(),
                call=ToolCallPart('danger', {}, tool_call_id='c'),
                tool_def=_tool_def(metadata={'risk': 'high'}),
                args={},
            )


class TestHumanApproval:
    async def test_a_requires_approval_tool_is_judged_only_after_the_person_approves(self) -> None:
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('no'), on_verdict=verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
            requires_approval=True,
        )

        first = await agent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        assert [call.tool_name for call in first.output.approvals] == ['danger']
        assert verdicts == [], 'the judge must not run while the call is waiting on a person'

        second = await agent.run(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': True}),
        )
        assert second.output == 'done'
        assert ran == ['danger:1']
        assert [v.verdict for v in verdicts] == ['allow']

    async def test_the_judge_still_blocks_a_call_a_person_approved(self) -> None:
        ran: list[str] = []
        returns: list[ToolReturnPart] = []
        agent = _agent(
            _judge(_judge_model('yes')),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'), returns=returns),
            ran=ran,
            requires_approval=True,
        )

        first = await agent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        await agent.run(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': True}),
        )
        assert ran == []
        assert [part.content for part in returns] == ['judge blocked danger']

    async def test_an_approved_call_the_judge_is_still_unsure_about_runs(self) -> None:
        """`on_uncertain='ask'` must reach the tool: core re-judges the approved call."""
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('unsure'), on_uncertain='ask', on_verdict=verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )

        first = await agent.run('go')
        assert isinstance(first.output, DeferredToolRequests)
        assert [call.tool_name for call in first.output.approvals] == ['danger']

        second = await agent.run(
            message_history=first.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals={'call-1': True}),
        )
        assert second.output == 'done'
        assert ran == ['danger:1'], 'an approved call must run instead of being asked about again'
        assert [v.verdict for v in verdicts] == ['ask', 'allow']

    async def test_ask_defers_the_call_for_a_person(self) -> None:
        judge = _judge(_judge_model('unsure'), on_uncertain='ask')
        with pytest.raises(ApprovalRequired):
            await judge.before_tool_execute(
                _ctx(),
                call=ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
                tool_def=_tool_def(),
                args={'x': 1},
            )


class TestUsageAccounting:
    async def test_parallel_judgements_share_the_runs_request_budget(self) -> None:
        """Judging sibling calls concurrently must not bill past the run's `request_limit`."""
        judge_calls: list[str] = []
        ran: list[str] = []
        agent = _agent(
            _judge(_yielding_judge_model('no', calls=judge_calls)),
            _outer_model(
                ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
                ToolCallPart('danger', {'x': 2}, tool_call_id='call-2'),
                ToolCallPart('danger', {'x': 3}, tool_call_id='call-3'),
            ),
            ran=ran,
        )

        usage = RunUsage()
        with pytest.raises(UsageLimitExceeded):
            await agent.run('go', usage=usage, usage_limits=UsageLimits(request_limit=2))

        assert usage.requests == 2, 'the judges billed past the run request limit'
        assert judge_calls == ['no'], 'only the judgement the budget affords reaches the model'


class TestDurableExecution:
    """The judgement is a durable operation, addressed by the capability's `id`."""

    def _durable_agent(self, judge: ToolCallJudge[None], *, name: str) -> tuple[Agent[None, str], list[str]]:
        ran: list[str] = []
        agent: Agent[None, str] = Agent(
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            name=name,
            deps_type=type(None),
            capabilities=[judge, RecordingDurability()],
        )

        @agent.tool_plain
        def danger(x: int) -> str:
            ran.append(f'danger:{x!r}')
            return 'danger ran'

        return agent, ran

    async def test_the_judgement_dispatches_as_a_durable_operation(self) -> None:
        """Without this the judge's model call runs uncheckpointed and repeats on every replay."""
        judge = _judge(_judge_model('no'), id='refund-judge')
        agent, ran = self._durable_agent(judge, name='judged')

        result = await agent.run('go')

        assert result.output == 'done'
        assert ran == ['danger:1']
        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        assert 'judged__capability__refund-judge.judge' in {name for name, _ in bound.calls}

    async def test_a_judge_without_an_id_cannot_bind_to_a_durable_agent(self) -> None:
        """A durable operation is addressed by the `id`, and this capability has no default one."""
        with pytest.raises(UserError, match='needs an explicit `id`'):
            self._durable_agent(_judge(_judge_model('no')), name='unnamed_judge')

    async def test_an_id_leaves_a_plain_run_unchanged(self) -> None:
        """Naming a judge is what durability needs; it changes nothing about an ordinary run."""
        ran: list[str] = []
        verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('yes'), id='refund-judge', on_verdict=verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
        )

        result = await agent.run('go')

        assert result.output == 'done'
        assert ran == []
        assert [v.verdict for v in verdicts] == ['block']


class TestComposition:
    async def test_the_first_judge_that_blocks_stops_the_call(self) -> None:
        ran: list[str] = []
        first_verdicts: list[ToolCallVerdict] = []
        second_verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('yes'), on_verdict=first_verdicts.append),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
            extra=_judge(_judge_model('no'), on_verdict=second_verdicts.append),
        )
        await agent.run('go')
        assert ran == []
        assert [v.verdict for v in first_verdicts] == ['block']
        assert second_verdicts == [], 'a blocked call short-circuits the remaining judges'

    async def test_both_judges_must_allow_the_call(self) -> None:
        ran: list[str] = []
        second_verdicts: list[ToolCallVerdict] = []
        agent = _agent(
            _judge(_judge_model('no')),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=ran,
            extra=_judge(_judge_model('no'), on_verdict=second_verdicts.append),
        )
        await agent.run('go')
        assert ran == ['danger:1']
        assert [v.verdict for v in second_verdicts] == ['allow']


class TestConversationContext:
    async def test_the_conversation_is_withheld_by_default(self) -> None:
        prompts: list[str] = []
        agent = _agent(
            _judge(_judge_model('no', prompts=prompts)),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=[],
        )
        await agent.run('delete everything in /etc')
        assert '<conversation>' not in prompts[0]
        assert 'delete everything in /etc' not in prompts[0]

    async def test_the_conversation_can_be_included(self) -> None:
        prompts: list[str] = []
        agent = _agent(
            _judge(_judge_model('no', prompts=prompts), include_conversation=True),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=[],
        )
        await agent.run('delete everything in /etc')
        assert prompts[0].startswith('<conversation>\nuser: delete everything in /etc')
        assert prompts[0].endswith('</tool_call>')

    async def test_the_conversation_window_keeps_the_most_recent_turns(self) -> None:
        prompts: list[str] = []
        agent = _agent(
            _judge(_judge_model('no', prompts=prompts), include_conversation=True, conversation_window=3),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=[],
        )
        await agent.run('a very long instruction that will not fit in the window')
        conversation = prompts[0].split('\n\n')[0]
        assert conversation == '<conversation>\nwith {"x":1}\n</conversation>', 'only the tail survives the clamp'
        assert 'a very long instruction' not in prompts[0]

    async def test_the_conversation_renders_observable_behavior_only(self) -> None:
        prompts: list[str] = []
        judge = _judge(_judge_model('no', prompts=prompts), include_conversation=True)
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='you are a helpful agent'),
                    UserPromptPart(content=['look at this', TextContent(content='and this'), ImageUrl(url='u')]),
                    UserPromptPart(content=[ImageUrl(url='u')]),
                    ToolReturnPart(tool_name='reader', content='file body', tool_call_id='t1'),
                    RetryPromptPart(content='try again', tool_name='reader', tool_call_id='t3'),
                    RetryPromptPart(content='bad output', tool_call_id='t4'),
                ]
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content='private reasoning'),
                    NativeToolReturnPart(tool_name='web_search', content='hits', tool_call_id='t2'),
                    TextPart(content='on it'),
                    TextPart(content=''),
                    ToolCallPart('reader', {'path': 'a'}, tool_call_id='t5'),
                    NativeToolCallPart('web_search', {'q': 'b'}, tool_call_id='t6'),
                    SpeechPart(speaker='assistant', transcript='spoken reply'),
                    SpeechPart(speaker='assistant'),
                ]
            ),
        ]
        await judge.before_tool_execute(
            _ctx(messages=messages),
            call=ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
            tool_def=_tool_def(),
            args={'x': 1},
        )
        conversation = prompts[0].split('</conversation>')[0].removeprefix('<conversation>\n')
        assert conversation == snapshot("""\
user: look at this and this
tool reader returned: file body
retry (reader): try again

Fix the errors and try again.
retry (output): Validation feedback:
bad output

Fix the errors and try again.
native tool web_search returned: hits
assistant: on it
assistant called tool reader with {"path":"a"}
assistant called native tool web_search with {"q":"b"}
assistant: spoken reply
""")
        assert 'you are a helpful agent' not in conversation, 'the system prompt is the agent configuration'
        assert 'private reasoning' not in conversation, 'thinking is not observable behavior'


class TestObservability:
    async def test_the_span_records_the_verdict_and_confidence_without_arguments(self) -> None:
        tracer, exporter = _recording_tracer()
        judge = _judge(_judge_model('yes', confidence=0.75))
        with pytest.raises(SkipToolExecution):
            await judge.before_tool_execute(
                _ctx(tracer=tracer),
                call=ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
                tool_def=_tool_def(),
                args={'x': 1},
            )
        attributes = _only_span(exporter).attributes or {}
        assert attributes['tool_call_judge.tool'] == 'danger'
        assert attributes['tool_call_judge.tool_call_id'] == 'call-1'
        assert attributes['tool_call_judge.model_result'] == 'yes'
        assert attributes['tool_call_judge.verdict'] == 'block'
        assert attributes['tool_call_judge.confidence'] == 0.75
        assert 'tool_call_judge.arguments' not in attributes

    @pytest.mark.parametrize(
        'provider_details',
        [{}, {'confidence': 'high'}, {'confidence': {'response': 2.0}}, {'confidence': {'other': 0.5}}],
        ids=['missing', 'not-a-mapping', 'out-of-range', 'wrong-key'],
    )
    async def test_unusable_provider_confidence_is_ignored(self, provider_details: dict[str, object]) -> None:
        tracer, exporter = _recording_tracer()
        judge = _judge(_judge_model('no', provider_details=provider_details))
        await judge.before_tool_execute(
            _ctx(tracer=tracer),
            call=ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
            tool_def=_tool_def(),
            args={'x': 1},
        )
        assert 'tool_call_judge.confidence' not in (_only_span(exporter).attributes or {})

    async def test_the_span_records_arguments_only_when_content_is_enabled(self) -> None:
        tracer, exporter = _recording_tracer()
        judge = _judge(_judge_model('no'))
        await judge.before_tool_execute(
            _ctx(tracer=tracer, trace_include_content=True),
            call=ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
            tool_def=_tool_def(),
            args={'x': 1},
        )
        assert (_only_span(exporter).attributes or {})['tool_call_judge.arguments'] == '{"x": 1}'

    async def test_the_span_records_a_model_failure_and_the_fail_closed_verdict(self) -> None:
        tracer, exporter = _recording_tracer()
        judge = _judge(_error_model())
        with pytest.raises(SkipToolExecution):
            await judge.before_tool_execute(
                _ctx(tracer=tracer),
                call=ToolCallPart('danger', {'x': 1}, tool_call_id='call-1'),
                tool_def=_tool_def(),
                args={'x': 1},
            )
        attributes = _only_span(exporter).attributes or {}
        assert attributes['tool_call_judge.model_result'] == 'error'
        assert attributes['tool_call_judge.error.type'] == 'RuntimeError'
        assert attributes['tool_call_judge.verdict'] == 'block'

    @pytest.mark.parametrize('host_instrument', ['redacted', 'off'])
    async def test_the_internal_judge_does_not_trace_its_prompt(self, host_instrument: str) -> None:
        """Global instrumentation must not record arguments the host agent redacts or does not trace."""
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        agent = _agent(
            _judge(_judge_model('no')),
            _outer_model(ToolCallPart('danger', {'x': 987654321}, tool_call_id='call-1')),
            ran=[],
        )
        agent.instrument = (
            InstrumentationSettings(tracer_provider=provider, include_content=False)
            if host_instrument == 'redacted'
            else False
        )
        Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
        try:
            await agent.run('go')
        finally:
            Agent.instrument_all(False)

        spans = exporter.get_finished_spans()
        assert 'invoke_agent tool_call_judge' not in {span.name for span in spans}
        assert not [span.name for span in spans if '987654321' in str(span.attributes)]

    async def test_the_judge_usage_is_added_to_the_run(self) -> None:
        agent = _agent(
            _judge(_judge_model('no')),
            _outer_model(ToolCallPart('danger', {'x': 1}, tool_call_id='call-1')),
            ran=[],
        )
        result = await agent.run('go')
        assert result.usage.requests == 3, 'two outer requests plus one judge request'


class TestVerdictCallback:
    def test_the_callback_is_kept_as_configured(self) -> None:
        seen: list[ToolCallVerdict] = []
        callback: Callable[[ToolCallVerdict], None] = seen.append
        judge = _judge(_judge_model('no'), on_verdict=callback)
        assert judge.on_verdict is callback
