from __future__ import annotations as _annotations

import warnings
from typing import TYPE_CHECKING

import pytest
from pytest_mock import MockerFixture

from pydantic_ai.models.test import TestModel

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import EvaluationReason, GembaScore, GEval
    from pydantic_evals.evaluators.common import DEFAULT_EVALUATORS
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.evaluators.llm_as_a_judge import (
        GembaScoreOutput,
        GEvalOutput,
        judge_g_eval,
        judge_gemba_da,
        judge_gemba_sqm,
    )
    from pydantic_evals.otel._errors import SpanTreeRecordingError

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


if TYPE_CHECKING or imports_successful():

    def _ctx_geval(
        inputs: object,
        *,
        output: object,
        metadata: object = None,
    ) -> EvaluatorContext[object, object, object]:
        return EvaluatorContext[object, object, object](
            name='test',
            inputs=inputs,
            metadata=metadata,
            expected_output=None,
            output=output,
            duration=0.0,
            _span_tree=SpanTreeRecordingError('spans were not recorded'),
            attributes={},
            metrics={},
        )

    def _ctx_gemba(
        inputs: str,
        output: str,
        *,
        expected_output: str | None = None,
    ) -> EvaluatorContext[str, str, object]:
        return EvaluatorContext[str, str, object](
            name='test',
            inputs=inputs,
            metadata=None,
            expected_output=expected_output,
            output=output,
            duration=0.0,
            _span_tree=SpanTreeRecordingError('spans were not recorded'),
            attributes={},
            metrics={},
        )


async def test_geval_delegates_to_judge_g_eval(mocker: MockerFixture):
    mock = mocker.patch(
        'pydantic_evals.evaluators.quality.judge_g_eval',
        return_value=GEvalOutput(reason='Clear and well structured', score=4),
    )

    ctx = _ctx_geval(inputs='Explain gravity.', output='Gravity pulls masses together.')
    evaluator = GEval(
        criteria='coherence',
        evaluation_steps=[
            'Read the output carefully.',
            'Check that each sentence follows logically from the previous one.',
            'Assign a score from 1 (incoherent) to 5 (fully coherent).',
        ],
        include_input=True,
    )
    result = await evaluator.evaluate(ctx)

    assert result == snapshot(EvaluationReason(value=4, reason='Clear and well structured'))
    call_kwargs = mock.call_args.kwargs
    call_args = mock.call_args.args
    assert call_args == ('Gravity pulls masses together.', 'coherence', evaluator.evaluation_steps, (1, 5))
    assert call_kwargs == {'inputs': 'Explain gravity.', 'model': None, 'model_settings': None}


async def test_geval_default_hides_input(mocker: MockerFixture):
    mock = mocker.patch(
        'pydantic_evals.evaluators.quality.judge_g_eval',
        return_value=GEvalOutput(reason='n/a', score=3),
    )

    ctx = _ctx_geval(inputs='secret', output='answer')
    evaluator = GEval(criteria='fluency', evaluation_steps=['Check grammar.'])
    await evaluator.evaluate(ctx)

    assert mock.call_args.kwargs['inputs'] is None


async def test_evaluation_name_customizes_report_name_without_warning():
    """`evaluation_name` names the result for `GEval`/`GembaScore` without the deprecated-attribute warning."""
    assert GEval(criteria='c', evaluation_steps=['s']).get_default_evaluation_name() == 'GEval'
    assert GembaScore(source_lang='English', target_lang='French').get_default_evaluation_name() == 'GembaScore'

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any deprecation warning becomes a test failure
        assert (
            GEval(criteria='c', evaluation_steps=['s'], evaluation_name='coherence').get_default_evaluation_name()
            == 'coherence'
        )
        assert (
            GembaScore(
                source_lang='English', target_lang='French', evaluation_name='quality'
            ).get_default_evaluation_name()
            == 'quality'
        )


async def test_gemba_score_da(mocker: MockerFixture):
    mock = mocker.patch(
        'pydantic_evals.evaluators.quality.judge_gemba_da',
        return_value=GembaScoreOutput(reason='Accurate', score=92),
    )

    ctx = _ctx_gemba('Hello world', 'Bonjour le monde', expected_output='Bonjour le monde')
    evaluator = GembaScore(source_lang='English', target_lang='French')
    result = await evaluator.evaluate(ctx)

    assert result == snapshot(EvaluationReason(value=92, reason='Accurate'))
    call_args = mock.call_args.args
    call_kwargs = mock.call_args.kwargs
    assert call_args == ('Hello world', 'Bonjour le monde', 'English', 'French')
    assert call_kwargs == {'reference': 'Bonjour le monde', 'model': None, 'model_settings': None}


async def test_gemba_score_sqm_no_reference(mocker: MockerFixture):
    mock = mocker.patch(
        'pydantic_evals.evaluators.quality.judge_gemba_sqm',
        return_value=GembaScoreOutput(reason='Mostly fine', score=5),
    )

    ctx = _ctx_gemba('Hello', 'Hola', expected_output=None)
    evaluator = GembaScore(source_lang='English', target_lang='Spanish', score_type='SQM')
    result = await evaluator.evaluate(ctx)

    assert result == snapshot(EvaluationReason(value=5, reason='Mostly fine'))
    assert mock.call_args.kwargs['reference'] is None


async def test_quality_evaluators_registered_in_defaults():
    """`GEval` and `GembaScore` deserialize from YAML configs via `DEFAULT_EVALUATORS`."""
    names = {cls.get_serialization_name() for cls in DEFAULT_EVALUATORS}
    assert {'GEval', 'GembaScore'}.issubset(names)


async def test_model_instance_serialized_as_string():
    """`GEval`/`GembaScore` serialize a `Model` instance as its `model_id`, matching `LLMJudge`."""
    model = TestModel()
    evaluators = [
        GEval(criteria='coherence', evaluation_steps=['step'], model=model),
        GembaScore(source_lang='English', target_lang='French', model=model),
    ]
    for evaluator in evaluators:
        assert evaluator.build_serialization_arguments()['model'] == model.model_id

    # A string model name is already serializable and passes through unchanged.
    geval = GEval(criteria='coherence', evaluation_steps=['step'], model='openai:gpt-5.2')
    assert geval.build_serialization_arguments()['model'] == 'openai:gpt-5.2'


async def test_judge_g_eval_prompt_shape(mocker: MockerFixture):
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
    # The numbered steps must be consistently un-indented; a naive `dedent()` of an interpolated
    # multi-line block leaves step 2+ at column 0 while the rest keeps its indent.
    assert (
        'Evaluation steps (apply each step in order):\n1. Step A.\n2. Step B.\n\nProduce a single integer score'
        in prompt
    )


async def test_judge_g_eval_validates_score_range():
    with pytest.raises(ValueError, match='score_range'):
        await judge_g_eval('out', 'c', ['s'], score_range=(5, 5))


async def test_judge_gemba_da_prompt_shape(mocker: MockerFixture):
    mock_result = mocker.MagicMock()
    mock_result.output = GembaScoreOutput(reason='great', score=95)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    await judge_gemba_da(
        'Hello',
        'Bonjour',
        source_lang='English',
        target_lang='French',
        reference='Salut',
    )
    prompt = mock_run.call_args[0][0]
    assert 'from English to French' in prompt
    assert '0 to 100' in prompt
    # Source, reference, and translation lines must share consistent (zero) indentation; the
    # reference line previously broke `dedent()` and pushed the translation line to column 0.
    assert 'English source: "Hello"\nFrench human reference: "Salut"\nFrench translation: "Bonjour"' in prompt


async def test_judge_gemba_sqm_prompt_shape(mocker: MockerFixture):
    mock_result = mocker.MagicMock()
    mock_result.output = GembaScoreOutput(reason='ok', score=4)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    await judge_gemba_sqm(
        'Hello',
        'Hola',
        source_lang='English',
        target_lang='Spanish',
    )
    prompt = mock_run.call_args[0][0]
    assert 'Scalar Quality Metrics' in prompt
    assert '0 to 6' in prompt
    # No reference block when `reference` is None:
    assert 'human reference' not in prompt
    # Source and translation lines are consecutive and un-indented when no reference is present.
    assert 'English source: "Hello"\nSpanish translation: "Hola"' in prompt


async def test_judge_gemba_da_prompt_no_reference(mocker: MockerFixture):
    mock_result = mocker.MagicMock()
    mock_result.output = GembaScoreOutput(reason='ok', score=80)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    await judge_gemba_da('Hello', 'Bonjour', source_lang='English', target_lang='French')
    prompt = mock_run.call_args[0][0]
    assert 'human reference' not in prompt
    assert 'English source: "Hello"\nFrench translation: "Bonjour"' in prompt


async def test_judge_gemba_sqm_prompt_with_reference(mocker: MockerFixture):
    mock_result = mocker.MagicMock()
    mock_result.output = GembaScoreOutput(reason='ok', score=5)
    mock_run = mocker.patch('pydantic_ai.agent.AbstractAgent.run', return_value=mock_result)

    await judge_gemba_sqm('Hello', 'Hola', source_lang='English', target_lang='Spanish', reference='Buenos días')
    prompt = mock_run.call_args[0][0]
    assert 'English source: "Hello"\nSpanish human reference: "Buenos días"\nSpanish translation: "Hola"' in prompt
