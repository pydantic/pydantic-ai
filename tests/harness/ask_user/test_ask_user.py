"""Tests for the AskUser capability."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import (
    DECLINED,
    TOOL_NAME,
    AskUserAnswer,
    AskUserAnsweredEvent,
    AskUserRequest,
    AskUserRequestedEvent,
    AskUserResponse,
    Question,
    QuestionOption,
    check_response,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def question(header: str = 'Approach', *, multi_select: bool = False, labels: Sequence[str] = ('A', 'B')) -> Question:
    return Question(
        header=header,
        question=f'Which {header.lower()}?',
        options=tuple(QuestionOption(label=label, description=f'Pick {label}') for label in labels),
        multi_select=multi_select,
    )


def raw_questions() -> list[dict[str, object]]:
    return [
        {
            'header': 'Approach',
            'question': 'How should we do it?',
            'options': [{'label': 'Refactor', 'description': 'Rewrite the module'}, {'label': 'Patch'}],
        },
        {
            'header': 'Targets',
            'question': 'Which files?',
            'options': [{'label': 'api.py'}, {'label': 'db.py'}, {'label': 'ui.py'}],
            'multi_select': True,
        },
    ]


class ScriptedAnswerer:
    """Answers with a fixed response and keeps what it was asked."""

    def __init__(self, response: AskUserResponse | None = None) -> None:
        self.response = response
        self.requests: list[AskUserRequest] = []

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        self.requests.append(request)
        if self.response is not None:
            return self.response
        answers = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
        return AskUserResponse(answers=tuple(answers))


class Observer(AbstractCapability[None]):
    """Watches the events without being the answerer."""

    def __init__(self) -> None:
        self.requested: list[AskUserRequestedEvent] = []
        self.answered: list[AskUserAnsweredEvent] = []

    @on_event(AskUserRequestedEvent)
    async def _requested(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        self.requested.append(event)

    @on_event(AskUserAnsweredEvent)
    async def _answered(self, ctx: RunContext[None], event: AskUserAnsweredEvent) -> None:
        self.answered.append(event)


def calling(arguments: object) -> FunctionModel:
    """A model that calls the tool once with `arguments`, then answers with the tool result.

    Streams too, since a run with event listeners is a streamed run.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(TOOL_NAME, {'questions': arguments}, tool_call_id='c1')])
        part = messages[-1].parts[-1]
        if isinstance(part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(part.model_response_str())])
        assert isinstance(part, RetryPromptPart)
        return ModelResponse(parts=[TextPart(part.model_response())])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        (part,) = respond(messages, info).parts
        if isinstance(part, ToolCallPart):
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id='c1')}
        else:
            assert isinstance(part, TextPart)
            yield part.content

    return FunctionModel(respond, stream_function=stream)


class TestAskUser:
    async def test_answers_reach_the_model_keyed_by_header(self) -> None:
        answerer = ScriptedAnswerer(
            AskUserResponse(
                answers=(
                    AskUserAnswer(header='Approach', selected=('Refactor',)),
                    AskUserAnswer(header='Targets', selected=('api.py', 'db.py')),
                )
            )
        )
        agent = Agent(calling(raw_questions()), capabilities=[AskUser(answerer=answerer)])
        result = await agent.run('go')
        assert json.loads(result.output) == {'Approach': ['Refactor'], 'Targets': ['api.py', 'db.py']}
        (request,) = answerer.requests
        assert [q.header for q in request.questions] == ['Approach', 'Targets']
        assert request.questions[1].multi_select is True
        assert request.questions[0].options[0].description == 'Rewrite the module'

    async def test_declining_tells_the_model_without_failing_the_run(self) -> None:
        answerer = ScriptedAnswerer(AskUserResponse(cancelled=True))
        agent = Agent(calling(raw_questions()), capabilities=[AskUser(answerer=answerer)])
        result = await agent.run('go')
        assert result.output == DECLINED

    async def test_events_are_observable_without_being_the_answerer(self) -> None:
        answerer = ScriptedAnswerer()
        observer = Observer()
        agent = Agent(
            calling(raw_questions()), deps_type=type(None), capabilities=[AskUser(answerer=answerer), observer]
        )
        await agent.run('go')
        (requested,) = observer.requested
        (answered,) = observer.answered
        assert requested.request is answerer.requests[0]
        assert answered.request_id == requested.request.id
        assert answered.response.cancelled is False
        assert requested.tool_name == TOOL_NAME and answered.tool_call_id == 'c1'

    async def test_request_event_reaches_listeners_before_the_answerer(self) -> None:
        order: list[str] = []

        class Ordered(AbstractCapability[None]):
            @on_event(AskUserRequestedEvent)
            async def _requested(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
                order.append('requested')

        async def answer(request: AskUserRequest) -> AskUserResponse:
            order.append('answerer')
            return AskUserResponse(cancelled=True)

        capabilities: list[AbstractCapability[None]] = [AskUser(answerer=answer), Ordered()]
        agent = Agent(calling(raw_questions()), deps_type=type(None), capabilities=capabilities)
        await agent.run('go')
        assert order == ['requested', 'answerer']

    @pytest.mark.parametrize(
        ('arguments', 'complaint'),
        [
            pytest.param([], 'at least 1 item', id='no questions'),
            pytest.param([{**raw_questions()[0], 'options': [{'label': 'Only'}]}], 'at least 2 items', id='one option'),
            pytest.param(
                [{**raw_questions()[0], 'options': [{'label': 'Same'}, {'label': 'same'}]}],
                'option labels must be unique',
                id='duplicate labels',
            ),
            pytest.param(
                [raw_questions()[0], raw_questions()[0]], 'question headers must be unique', id='duplicate headers'
            ),
            pytest.param([{**raw_questions()[0], 'header': ' '}], 'at least 1 character', id='blank header'),
            pytest.param(
                [{**raw_questions()[0], 'options': [{'label': 'Other', 'free_text': True}, {'label': 'B'}]}],
                'Extra inputs are not permitted',
                id='unsupported option field',
            ),
            pytest.param(
                [{**raw_questions()[0], 'allow_other': True}],
                'Extra inputs are not permitted',
                id='unsupported question field',
            ),
            pytest.param(
                [{**raw_questions()[0], 'options': [{'label': 'A\x1b]52;c;aGVsbG8=\x07'}, {'label': 'B'}]}],
                'must not contain control characters',
                id='escape sequence in a label',
            ),
            pytest.param(
                [{**raw_questions()[0], 'header': 'Two\nlines'}],
                'must not contain control characters',
                id='newline in a header',
            ),
        ],
    )
    async def test_bad_schemas_are_returned_to_the_model_as_retries(self, arguments: object, complaint: str) -> None:
        answerer = ScriptedAnswerer()
        agent = Agent(calling(arguments), capabilities=[AskUser(answerer=answerer)])
        result = await agent.run('go')
        assert complaint in result.output
        assert answerer.requests == []

    async def test_ten_questions_is_the_limit(self) -> None:
        eleven = [{**raw_questions()[0], 'header': f'Q{i}'} for i in range(11)]
        agent = Agent(calling(eleven), capabilities=[AskUser(answerer=ScriptedAnswerer())])
        result = await agent.run('go')
        assert 'at most 10 items' in result.output

    async def test_tool_is_listed_with_instructions(self) -> None:
        model = TestModel(call_tools=[])
        agent = Agent(model, capabilities=[AskUser(answerer=ScriptedAnswerer())])
        result = await agent.run('hello')
        assert model.last_model_request_parameters is not None
        (tool,) = model.last_model_request_parameters.function_tools
        assert tool.name == TOOL_NAME
        schema = tool.parameters_json_schema['properties']['questions']
        assert schema['minItems'] == 1 and schema['maxItems'] == 10
        first = result.all_messages()[0]
        assert isinstance(first, ModelRequest)
        assert first.instructions is not None and TOOL_NAME in first.instructions


class TestCheckResponse:
    def request(self) -> AskUserRequest:
        return AskUserRequest(questions=(question(), question('Targets', multi_select=True, labels=('x', 'y'))))

    def test_complete_response_passes(self) -> None:
        response = AskUserResponse(
            answers=(
                AskUserAnswer(header='Approach', selected=('A',)),
                AskUserAnswer(header='Targets', selected=('x', 'y')),
            )
        )
        check_response(self.request(), response)

    def test_cancelled_response_passes_empty(self) -> None:
        check_response(self.request(), AskUserResponse(cancelled=True))

    @pytest.mark.parametrize(
        ('response', 'complaint'),
        [
            pytest.param(
                AskUserResponse(cancelled=True, answers=(AskUserAnswer(header='Approach', selected=('A',)),)),
                'cancelled response carries no answers',
                id='cancelled with answers',
            ),
            pytest.param(
                AskUserResponse(answers=(AskUserAnswer(header='Nope', selected=('A',)),)),
                "unknown or repeated question 'Nope'",
                id='unknown header',
            ),
            pytest.param(
                AskUserResponse(
                    answers=(
                        AskUserAnswer(header='Approach', selected=('A',)),
                        AskUserAnswer(header='Approach', selected=('B',)),
                    )
                ),
                "unknown or repeated question 'Approach'",
                id='repeated header',
            ),
            pytest.param(
                AskUserResponse(answers=(AskUserAnswer(header='Approach', selected=('Z',)),)),
                "picked options it does not offer: ['Z']",
                id='unknown label',
            ),
            pytest.param(
                AskUserResponse(answers=(AskUserAnswer(header='Approach', selected=()),)),
                'picked nothing',
                id='nothing picked',
            ),
            pytest.param(
                AskUserResponse(answers=(AskUserAnswer(header='Approach', selected=('A', 'B')),)),
                'single-select',
                id='several on single-select',
            ),
            pytest.param(
                AskUserResponse(
                    answers=(
                        AskUserAnswer(header='Approach', selected=('A',)),
                        AskUserAnswer(header='Targets', selected=('x', 'x')),
                    )
                ),
                'picked the same option twice',
                id='duplicate pick on multi-select',
            ),
            pytest.param(
                AskUserResponse(answers=(AskUserAnswer(header='Approach', selected=('A',)),)),
                "unanswered questions: ['Targets']",
                id='missing answer',
            ),
        ],
    )
    def test_misfits_are_rejected(self, response: AskUserResponse, complaint: str) -> None:
        with pytest.raises(ValueError, match=re.escape(complaint)):
            check_response(self.request(), response)

    async def test_a_broken_answerer_fails_the_run_but_still_releases_observers(self) -> None:
        answerer = ScriptedAnswerer(AskUserResponse(answers=(AskUserAnswer(header='Nope', selected=('A',)),)))
        observer = Observer()
        agent = Agent(
            calling(raw_questions()), deps_type=type(None), capabilities=[AskUser(answerer=answerer), observer]
        )
        with pytest.raises(ValueError, match='unknown or repeated question'):
            await agent.run('go')
        assert len(observer.requested) == 1 and len(observer.answered) == 1
        assert observer.answered[0].response is answerer.response


class TestSchema:
    def test_question_text_and_descriptions_may_span_lines(self) -> None:
        asked = Question(
            header='Style',
            question='Which style?\nPick one.',
            options=(QuestionOption(label='A', description='First\nline'), QuestionOption(label='B')),
        )
        assert asked.question == 'Which style?\nPick one.'
        with pytest.raises(ValidationError, match='control characters'):
            QuestionOption(label='A', description='tab\there')

    def test_whitespace_is_stripped(self) -> None:
        option = QuestionOption(label='  Keep  ')
        assert option.label == 'Keep' and option.description is None

    def test_long_description_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match='at most 200 characters'):
            QuestionOption(label='x', description='d' * 201)

    def test_questions_are_immutable_once_validated(self) -> None:
        asked = question()
        with pytest.raises(ValidationError, match='frozen'):
            asked.header = 'Changed'
        with pytest.raises(ValidationError, match='frozen'):
            asked.options[0].label = 'Changed'
        assert isinstance(asked.options, tuple)

    def test_request_ids_are_unique(self) -> None:
        first, second = AskUserRequest(questions=(question(),)), AskUserRequest(questions=(question(),))
        assert first.id != second.id


class TestCustomAnswers:
    async def test_custom_and_selected_answers_reach_model_and_observers(self) -> None:
        response = AskUserResponse(
            answers=(
                AskUserAnswer(header='Approach', custom_answer='Use a different approach\nKeep 中文'),
                AskUserAnswer(header='Targets', selected=('api.py',)),
            )
        )
        observer = Observer()
        agent = Agent(
            calling(raw_questions()),
            deps_type=type(None),
            capabilities=[
                AskUser(answerer=ScriptedAnswerer(response)),
                observer,
            ],
        )
        result = await agent.run('go')
        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert returns[0].content == {'Approach': ['Use a different approach\nKeep 中文'], 'Targets': ['api.py']}
        assert observer.answered[0].response == response

    @pytest.mark.parametrize(
        'custom, selected, message',
        [
            ('', (), 'must not be blank'),
            (' \n ', (), 'must not be blank'),
            ('custom', ('A',), 'cannot also select'),
            ('bad\x1b[31m', (), 'control characters'),
            ('bad\ttext', (), 'control characters'),
        ],
    )
    def test_invalid_custom_answers(self, custom: str, selected: tuple[str, ...], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            check_response(
                AskUserRequest(questions=(question(),)),
                AskUserResponse(answers=(AskUserAnswer(header='Approach', selected=selected, custom_answer=custom),)),
            )

    def test_custom_answer_replaces_multi_select(self) -> None:
        check_response(
            AskUserRequest(questions=(question(multi_select=True),)),
            AskUserResponse(answers=(AskUserAnswer(header='Approach', custom_answer='Neither'),)),
        )
