from __future__ import annotations as _annotations

import re
from typing import Any, Literal

import pytest
from pytest_mock import MockerFixture

from .._inline_snapshot import snapshot
from ..conftest import BinaryContent, iter_message_parts, try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, SystemPromptPart, ToolCallPart
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.settings import ModelSettings
    from pydantic_evals.evaluators.llm_as_a_judge import (
        GEvalOutput,
        GradingOutput,
        _build_prompt,  # pyright: ignore[reportPrivateUsage]
        _judge_g_eval,  # pyright: ignore[reportPrivateUsage]
        _judge_input_output,  # pyright: ignore[reportPrivateUsage]
        _judge_input_output_agent,  # pyright: ignore[reportPrivateUsage]
        _judge_input_output_expected,  # pyright: ignore[reportPrivateUsage]
        _judge_input_output_expected_agent,  # pyright: ignore[reportPrivateUsage]
        _judge_output,  # pyright: ignore[reportPrivateUsage]
        _judge_output_agent,  # pyright: ignore[reportPrivateUsage]
        _judge_output_expected,  # pyright: ignore[reportPrivateUsage]
        _judge_output_expected_agent,  # pyright: ignore[reportPrivateUsage]
        _stringify,  # pyright: ignore[reportPrivateUsage]
        judge_g_eval,
        judge_input_output,
        judge_input_output_expected,
        judge_output,
        judge_output_expected,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


@pytest.fixture(autouse=True)
def _default_judge_model(monkeypatch: pytest.MonkeyPatch):
    """Keep mocked judge tests independent of provider credentials."""
    monkeypatch.setattr('pydantic_evals.evaluators.llm_as_a_judge._default_model', TestModel())


def test_grading_output():
    """Test GradingOutput model."""
    # Test with pass=True
    output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    assert output.reason == 'Test passed'
    assert output.pass_ is True
    assert output.score == 1.0

    # Test with pass=False
    output = GradingOutput(reason='Test failed', pass_=False, score=0.0)
    assert output.reason == 'Test failed'
    assert output.pass_ is False
    assert output.score == 0.0

    # Test with alias
    output = GradingOutput.model_validate({'reason': 'Test passed', 'pass': True, 'score': 1.0})
    assert output.reason == 'Test passed'
    assert output.pass_ is True
    assert output.score == 1.0

    schema = GradingOutput.model_json_schema()
    assert schema['properties']['reason']['description'] == ('A concise 1-2 sentence justification for the verdict.')


@pytest.mark.anyio
async def test_judge_prompts_constrain_reason():
    """Every judge sends the concise-reason instruction in its system prompt (#5034).

    Exercises each public judge helper against a `FunctionModel` that captures the
    system prompt actually delivered to the model, so dropping the instruction from
    any of the four judge agents fails this test.
    """
    captured: list[str] = []

    async def capture_system_prompt(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.append(
            '\n'.join(part.content for part in iter_message_parts(messages, ModelRequest, SystemPromptPart))
        )
        assert info.output_tools is not None
        args = '{"reason": "ok", "pass": true, "score": 1.0}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    model = FunctionModel(capture_system_prompt)

    await judge_output('output', 'rubric', model=model)
    await judge_input_output('input', 'output', 'rubric', model=model)
    await judge_input_output_expected('input', 'output', 'expected', 'rubric', model=model)
    await judge_output_expected('output', 'expected', 'rubric', model=model)

    assert len(captured) == 4
    for system_prompt in captured:
        assert 'concise 1-2 sentence justification' in system_prompt
        assert 'Do not include your reasoning process' in system_prompt


async def test_judge_output_with_text_support_retries_a_null_reason():
    """A nullable public result must not weaken the text judge's output schema."""
    requests = 0

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert info.output_tools is not None
        reason = None if requests == 1 else 'The output contains a greeting.'
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, {'reason': reason, 'pass': True, 'score': 1.0})]
        )

    result = await judge_output('Hello world', 'Content contains a greeting', model=FunctionModel(answer))

    assert requests == 2
    assert result == GradingOutput(reason='The output contains a greeting.', pass_=True, score=1.0)


async def test_judge_g_eval_with_text_support_retries_a_null_reason():
    """G-Eval keeps requiring a reason from judges that can generate text."""
    requests = 0

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert info.output_tools is not None
        reason = None if requests == 1 else 'The output is clear.'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'reason': reason, 'score': 4})])

    result = await judge_g_eval('Clear output.', 'clarity', ['Read it.'], model=FunctionModel(answer))

    assert requests == 2
    assert result == GEvalOutput(reason='The output is clear.', score=4)


async def test_judge_output_without_text_support():
    """A verdict-only judge gets one boolean question and reports no invented reason."""
    schemas: list[dict[str, object]] = []

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        output_tool = info.output_tools[0]
        schemas.append(output_tool.parameters_json_schema)
        return ModelResponse(parts=[ToolCallPart(output_tool.name, {'pass': True})])

    model = FunctionModel(answer, profile={'supports_text_output': False})
    result = await _judge_output('Hello world', 'Content contains a greeting', model=model, allow_reasonless=True)

    assert result.reason is None
    assert result.pass_ is True
    assert result.score == 1.0
    assert schemas == snapshot(
        [
            {
                'additionalProperties': False,
                'properties': {
                    'pass': {
                        'description': 'Is the statement in <Rubric> true for <Output>?',
                        'type': 'boolean',
                    },
                },
                'required': ['pass'],
                'title': 'BinaryGrading',
                'type': 'object',
            }
        ]
    )


@pytest.mark.parametrize(
    'variant,kwargs,expected_question',
    [
        pytest.param(
            'output',
            {'output': 'O', 'rubric': 'R'},
            'Is the statement in <Rubric> true for <Output>?',
            id='output',
        ),
        pytest.param(
            'input_output',
            {'inputs': 'I', 'output': 'O', 'rubric': 'R'},
            'Is the statement in <Rubric> true for <Output>, taking <Input> into account?',
            id='input and output',
        ),
        pytest.param(
            'output_expected',
            {'output': 'O', 'expected_output': 'E', 'rubric': 'R'},
            'Is the statement in <Rubric> true for <Output>, taking <ExpectedOutput> into account?',
            id='output and expected output',
        ),
        pytest.param(
            'input_output_expected',
            {'inputs': 'I', 'output': 'O', 'expected_output': 'E', 'rubric': 'R'},
            'Is the statement in <Rubric> true for <Output>, taking <Input> and <ExpectedOutput> into account?',
            id='input, output and expected output',
        ),
        pytest.param(
            'input_output_expected',
            {'inputs': None, 'output': 'O', 'expected_output': None, 'rubric': 'R'},
            'Is the statement in <Rubric> true for <Output>?',
            id='the helper takes both but the caller passed neither',
        ),
    ],
)
async def test_the_verdict_question_names_only_the_sections_the_prompt_carries(
    variant: str, kwargs: dict[str, Any], expected_question: str
):
    """A judge that cannot write a reason is asked about the sections it was given, and no others."""
    # Resolve the helper inside the test body so the module-level skip applies when the optional
    # `pydantic-evals` dependency is missing: a name in a `parametrize` is read at collection time.
    judge: Any = {
        'output': _judge_output,
        'input_output': _judge_input_output,
        'output_expected': _judge_output_expected,
        'input_output_expected': _judge_input_output_expected,
    }[variant]
    questions: list[str] = []

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        output_tool = info.output_tools[0]
        properties = output_tool.parameters_json_schema['properties']
        questions.append(properties['pass']['description'])
        return ModelResponse(parts=[ToolCallPart(output_tool.name, {'pass': True})])

    model = FunctionModel(answer, profile={'supports_text_output': False})
    await judge(**kwargs, model=model, allow_reasonless=True)

    assert questions == [expected_question]


async def test_a_verdict_that_is_not_a_boolean_is_refused():
    """A judge that cannot write a reason still has to answer the one question it was asked."""

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'pass': 'yes'})])

    model = FunctionModel(answer, profile={'supports_text_output': False})
    with pytest.raises(ValueError, match="Judge returned an invalid verdict: 'yes'"):
        await _judge_output('Hello world', 'Content contains a greeting', model=model, allow_reasonless=True)


async def test_judge_g_eval_without_text_support():
    """A reason-less G-Eval judge gets a normalized rubric and returns the requested integer scale."""
    schemas: list[dict[str, object]] = []

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        output_tool = info.output_tools[0]
        schemas.append(output_tool.parameters_json_schema)
        return ModelResponse(parts=[ToolCallPart(output_tool.name, {'score': 3})])

    model = FunctionModel(answer, profile={'supports_text_output': False})
    result = await _judge_g_eval(
        'Clear output.', 'clarity', ['Read it.'], score_range=(1, 5), model=model, allow_reasonless=True
    )

    assert result.reason is None
    assert result.score == 4
    assert schemas == snapshot(
        [
            {
                'type': 'object',
                'properties': {
                    'score': {
                        'description': 'What score does the output earn according to the criteria and evaluation steps?',
                        'anyOf': [
                            {'const': 0, 'description': '1: the worst score according to the evaluation criteria.'},
                            {'const': 1, 'description': '2: an intermediate score between the worst and best.'},
                            {'const': 2, 'description': '3: an intermediate score between the worst and best.'},
                            {'const': 3, 'description': '4: an intermediate score between the worst and best.'},
                            {'const': 4, 'description': '5: the best score according to the evaluation criteria.'},
                        ],
                    }
                },
                'required': ['score'],
                'additionalProperties': False,
                'title': 'GEvalScore',
            }
        ]
    )


async def test_judge_g_eval_without_text_support_rejects_an_invalid_score():
    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'score': 'high'})])

    model = FunctionModel(answer, profile={'supports_text_output': False})
    with pytest.raises(ValueError, match="Judge returned an invalid score: 'high'"):
        await _judge_g_eval(
            'Clear output.', 'clarity', ['Read it.'], score_range=(1, 5), model=model, allow_reasonless=True
        )


async def test_judge_g_eval_without_text_support_rejects_too_many_score_levels():
    """An oversized rubric fails before the grading agent can make a request."""

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('request should not be made')  # pragma: no cover

    model = FunctionModel(answer, profile={'supports_text_output': False})
    with pytest.raises(
        UserError,
        match=re.escape(
            '`score_range` can contain at most 20 levels for a judge that does not support text output; '
            'got 21 in (0, 20).'
        ),
    ):
        await _judge_g_eval(
            'Clear output.', 'clarity', ['Read it.'], score_range=(0, 20), model=model, allow_reasonless=True
        )


async def test_public_judge_helpers_require_a_reason():
    """The existing helper result types stay strict; evaluators own reasonless results."""

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('request should not be made')  # pragma: no cover

    model = FunctionModel(answer, profile={'supports_text_output': False})
    with pytest.raises(UserError, match='Use the `LLMJudge` evaluator'):
        await judge_output('Clear output.', 'Content is clear.', model=model)
    with pytest.raises(UserError, match='Use the `GEval` evaluator'):
        await judge_g_eval('Clear output.', 'clarity', ['Read it.'], model=model)


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('judge', ['grading', 'g_eval'])
async def test_text_judges_support_fallback_models(wrapped: bool, judge: Literal['grading', 'g_eval']):
    """Composite models without an aggregate profile keep the existing text-judge path."""

    async def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        output_tool = info.output_tools[0]
        properties = output_tool.parameters_json_schema['properties']
        args = (
            {'reason': 'The output is clear.', 'pass': True, 'score': 1.0}
            if 'pass' in properties
            else {'reason': 'The output is clear.', 'score': 4}
        )
        return ModelResponse(parts=[ToolCallPart(output_tool.name, args)])

    model = FallbackModel(FunctionModel(answer), FunctionModel(answer))
    judge_model = WrapperModel(model) if wrapped else model

    if judge == 'grading':
        result = await judge_output('Clear output.', 'Content is clear.', model=judge_model)
    else:
        result = await judge_g_eval('Clear output.', 'clarity', ['Read it.'], model=judge_model)

    assert result.reason == 'The output is clear.'


def test_stringify():
    """Test _stringify function."""
    # Test with string
    assert _stringify('test') == 'test'

    # Test with dict
    assert _stringify({'key': 'value'}) == '{"key":"value"}'

    # Test with list
    assert _stringify([1, 2, 3]) == '[1,2,3]'

    # Test with custom object
    class CustomObject:
        def __repr__(self):
            return 'CustomObject()'

    obj = CustomObject()
    assert _stringify(obj) == 'CustomObject()'

    # Test with non-JSON-serializable object
    class NonSerializable:
        def __repr__(self):
            return 'NonSerializable()'

    obj = NonSerializable()
    assert _stringify(obj) == 'NonSerializable()'


@pytest.mark.parametrize(
    'variant,kwargs,expected_tags',
    [
        ('output', {'output': 'O', 'rubric': 'R'}, ['Output', 'Rubric']),
        ('input_output', {'output': 'O', 'rubric': 'R', 'inputs': 'I'}, ['Input', 'Output', 'Rubric']),
        (
            'output_expected',
            {'output': 'O', 'rubric': 'R', 'expected_output': 'E'},
            ['Output', 'ExpectedOutput', 'Rubric'],
        ),
        (
            'input_output_expected',
            {'output': 'O', 'rubric': 'R', 'inputs': 'I', 'expected_output': 'E'},
            ['Input', 'Output', 'ExpectedOutput', 'Rubric'],
        ),
    ],
)
def test_build_prompt_section_order_matches_few_shot_examples(
    variant: str, kwargs: dict[str, str], expected_tags: list[str]
) -> None:
    """The runtime prompt and every few-shot example in the matching judge's system prompt must
    share the same section order: `Input → Output → ExpectedOutput → Rubric` (matching the
    `judge_input_output_expected` naming), with the rubric (the instruction) last, after all
    the context it applies to. See issue #6110."""
    # Resolve the agent inside the test body so the module-level skip applies when the optional
    # `pydantic-evals` dependency is missing (the agents only exist on a successful import).
    agent: Agent[None, GradingOutput] = {
        'output': _judge_output_agent,
        'input_output': _judge_input_output_agent,
        'output_expected': _judge_output_expected_agent,
        'input_output_expected': _judge_input_output_expected_agent,
    }[variant]

    prompt, _ = _build_prompt(**kwargs)
    assert isinstance(prompt, str)
    assert re.findall(r'<(\w+)>', prompt) == expected_tags

    system_prompt = agent._system_prompts[0]  # pyright: ignore[reportPrivateUsage]
    assert isinstance(system_prompt, str)
    # Check every example block, not the deduped set, so a single drifting example is caught.
    example_tags = re.findall(r'<(\w+)>', system_prompt)
    assert example_tags and len(example_tags) % len(expected_tags) == 0
    for start in range(0, len(example_tags), len(expected_tags)):
        assert example_tags[start : start + len(expected_tags)] == expected_tags


@pytest.mark.anyio
async def test_judge_output_mock(mocker: MockerFixture):
    """Test judge_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string output
    grading_output = await judge_output('Hello world', 'Content contains a greeting')
    assert isinstance(grading_output, GradingOutput)
    assert grading_output.reason == 'Test passed'
    assert grading_output.pass_ is True
    assert grading_output.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nContent contains a greeting\n</Rubric>' in call_args[0]


@pytest.mark.anyio
async def test_judge_output_with_model_settings_mock(mocker: MockerFixture):
    """Test judge_output function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    grading_output = await judge_output(
        'Hello world settings',
        'Content contains a greeting with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(grading_output, GradingOutput)
    assert grading_output.reason == 'Test passed with settings'
    assert grading_output.pass_ is True
    assert grading_output.score == 1.0

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert '<Output>\nHello world settings\n</Output>' in call_args[0]
    assert '<Rubric>\nContent contains a greeting with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs


@pytest.mark.anyio
async def test_judge_input_output_mock(mocker: MockerFixture):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string input and output
    result = await judge_input_output('Hello', 'Hello world', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0]
    assert '<Input>\nHello\n</Input>' in call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input\n</Rubric>' in call_args[0]


async def test_judge_input_output_binary_content_list_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    result = await judge_input_output([image_content, image_content], 'Hello world', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    raw_prompt = mock_run.call_args[0][0]

    # 1) It must be a list
    assert isinstance(raw_prompt, list), 'Expected prompt to be a list when passing binary'

    # 2) The BinaryContent you passed in should be one of the elements
    assert image_content in raw_prompt, 'Expected the exact BinaryContent instance to be in the prompt list'


async def test_judge_binary_output_mock(mocker: MockerFixture, image_content: BinaryContent) -> None:
    """Test judge_output function when binary content is to be judged"""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    result = await judge_output(output=image_content, rubric='dummy rubric')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    call_args, *_ = mock_run.call_args

    assert call_args == snapshot((['<Output>', image_content, '</Output>', '<Rubric>', 'dummy rubric', '</Rubric>'],))


async def test_judge_input_output_binary_content_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_input_output function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    result = await judge_input_output(image_content, 'Hello world', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    mock_run.assert_called_once()
    raw_prompt = mock_run.call_args[0][0]

    # 1) It must be a list
    assert isinstance(raw_prompt, list), 'Expected prompt to be a list when passing binary'

    # 2) The BinaryContent you passed in should be one of the elements
    assert image_content in raw_prompt, 'Expected the exact BinaryContent instance to be in the prompt list'


@pytest.mark.anyio
async def test_judge_input_output_with_model_settings_mock(mocker: MockerFixture):
    """Test judge_input_output function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    result = await judge_input_output(
        'Hello settings',
        'Hello world with settings',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert '<Input>\nHello settings\n</Input>' in call_args[0]
    assert '<Output>\nHello world with settings\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs


@pytest.mark.anyio
async def test_judge_input_output_expected_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_input_output_expected function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string input and output
    result = await judge_input_output_expected('Hello', 'Hello world', 'Hello', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    call_args = mock_run.call_args[0]
    assert call_args == snapshot(
        (
            """\
<Input>
Hello
</Input>
<Output>
Hello world
</Output>
<ExpectedOutput>
Hello
</ExpectedOutput>
<Rubric>
Output contains input
</Rubric>\
""",
        )
    )

    result = await judge_input_output_expected(image_content, 'Hello world', 'Hello', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args = mock_run.call_args[0]
    assert call_args == snapshot(
        (
            [
                '<Input>',
                image_content,
                '</Input>',
                '<Output>',
                'Hello world',
                '</Output>',
                '<ExpectedOutput>',
                'Hello',
                '</ExpectedOutput>',
                '<Rubric>',
                'Output contains input',
                '</Rubric>',
            ],
        )
    )


@pytest.mark.anyio
async def test_judge_input_output_expected_with_model_settings_mock(
    mocker: MockerFixture, image_content: BinaryContent
):
    """Test judge_input_output_expected function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    result = await judge_input_output_expected(
        'Hello settings',
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args
    assert call_args == snapshot(
        (
            """\
<Input>
Hello settings
</Input>
<Output>
Hello world with settings
</Output>
<ExpectedOutput>
Hello
</ExpectedOutput>
<Rubric>
Output contains input with settings
</Rubric>\
""",
        )
    )
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs

    result = await judge_input_output_expected(
        image_content,
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )

    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args
    assert call_args == snapshot(
        (
            [
                '<Input>',
                image_content,
                '</Input>',
                '<Output>',
                'Hello world with settings',
                '</Output>',
                '<ExpectedOutput>',
                'Hello',
                '</ExpectedOutput>',
                '<Rubric>',
                'Output contains input with settings',
                '</Rubric>',
            ],
        )
    )
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs

    result = await judge_input_output_expected(
        123,
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )

    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args

    assert call_args == snapshot(
        (
            """\
<Input>
123
</Input>
<Output>
Hello world with settings
</Output>
<ExpectedOutput>
Hello
</ExpectedOutput>
<Rubric>
Output contains input with settings
</Rubric>\
""",
        )
    )

    result = await judge_input_output_expected(
        [123],
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )

    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args

    assert call_args == snapshot(
        (
            """\
<Input>
123
</Input>
<Output>
Hello world with settings
</Output>
<ExpectedOutput>
Hello
</ExpectedOutput>
<Rubric>
Output contains input with settings
</Rubric>\
""",
        )
    )


@pytest.mark.anyio
async def test_judge_output_expected_mock(mocker: MockerFixture):
    """Test judge_output_expected function with mocked agent."""
    # Mock the agent run method
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    # Test with string output and expected output
    result = await judge_output_expected('Hello world', 'Hello', 'Output contains input')
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed'
    assert result.pass_ is True
    assert result.score == 1.0

    # Verify the agent was called with correct prompt
    call_args = mock_run.call_args[0]
    assert '<Input>' not in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input\n</Rubric>' in call_args[0]


@pytest.mark.anyio
async def test_judge_output_expected_with_model_settings_mock(mocker: MockerFixture, image_content: BinaryContent):
    """Test judge_output_expected function with model_settings and mocked agent."""
    mock_result = mocker.MagicMock()
    mock_result.output = GradingOutput(reason='Test passed with settings', pass_=True, score=1.0)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    test_model_settings = ModelSettings(temperature=1)

    result = await judge_output_expected(
        'Hello world with settings',
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert '<Input>' not in call_args[0]
    assert '<ExpectedOutput>\nHello\n</ExpectedOutput>' in call_args[0]
    assert '<Output>\nHello world with settings\n</Output>' in call_args[0]
    assert '<Rubric>\nOutput contains input with settings\n</Rubric>' in call_args[0]
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs

    result = await judge_output_expected(
        image_content,
        'Hello',
        'Output contains input with settings',
        model_settings=test_model_settings,
    )
    assert isinstance(result, GradingOutput)
    assert result.reason == 'Test passed with settings'
    assert result.pass_ is True
    assert result.score == 1.0

    call_args, call_kwargs = mock_run.call_args
    assert call_args == snapshot(
        (
            [
                '<Output>',
                image_content,
                '</Output>',
                '<ExpectedOutput>',
                'Hello',
                '</ExpectedOutput>',
                '<Rubric>',
                'Output contains input with settings',
                '</Rubric>',
            ],
        )
    )
    assert call_kwargs['model_settings'] == test_model_settings
    # Check if 'model' kwarg is passed, its value will be the default model or None
    assert 'model' in call_kwargs


async def test_judge_g_eval_mock(mocker: MockerFixture):
    """`judge_g_eval` builds a numbered-steps prompt and returns a `GEvalOutput`."""
    mock_result = mocker.MagicMock()
    mock_result.output = GEvalOutput(reason='good', score=4)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    result = await judge_g_eval(
        'The cat sat on the mat.',
        'coherence',
        ['Step A.', 'Step B.'],
        score_range=(1, 5),
    )
    assert result == GEvalOutput(reason='good', score=4)

    prompt = mock_run.call_args[0][0]
    assert 'coherence' in prompt
    assert 'between 1 and 5' in prompt
    assert (
        'Evaluation steps (apply each step in order):\n1. Step A.\n2. Step B.\n\nProduce a single integer score'
        in prompt
    )


async def test_judge_g_eval_validates_score_range():
    with pytest.raises(ValueError, match='`score_range` must satisfy min < max'):
        await judge_g_eval('out', 'c', ['s'], score_range=(5, 5))


async def test_judge_g_eval_requires_evaluation_steps():
    with pytest.raises(ValueError, match='`evaluation_steps` must contain at least one step'):
        await judge_g_eval('out', 'c', [])


async def test_judge_g_eval_rejects_out_of_range_score(mocker: MockerFixture):
    """A judge response outside `score_range` raises instead of recording a misleading value."""
    mock_result = mocker.MagicMock()
    mock_result.output = GEvalOutput(reason='overenthusiastic', score=100)
    mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    with pytest.raises(ValueError, match='outside the requested `score_range`'):
        await judge_g_eval('out', 'coherence', ['Check.'], score_range=(1, 5))
