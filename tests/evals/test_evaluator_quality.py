from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pydantic_core import to_jsonable_python
from pytest_mock import MockerFixture

from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import (
        AnswerRelevance,
        ContextPrecision,
        ContextRecall,
        EvaluationReason,
        EvaluatorContext,
        Faithfulness,
        GembaScore,
        GEval,
        Hallucination,
        HasQuestion,
        OutputConfig,
        QuestionWithContext,
    )
    from pydantic_evals.evaluators.common import DEFAULT_EVALUATORS
    from pydantic_evals.evaluators.llm_as_a_judge import (
        GembaScoreOutput,
        GEvalOutput,
        GradingOutput,
        judge_g_eval,
        judge_gemba_da,
        judge_gemba_sqm,
    )
    from pydantic_evals.otel._errors import SpanTreeRecordingError

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


if TYPE_CHECKING or imports_successful():

    @dataclass
    class QInputs:
        """Dataset input shape matching `HasQuestion`."""

        question: str

    @dataclass
    class RagInputs:
        """Dataset input shape matching `QuestionWithContext`."""

        question: str
        context: list[str]

    def _ctx_qc(
        question: str,
        context: list[str],
        *,
        output: object = 'some answer',
        expected_output: object = None,
        metadata: object = None,
    ) -> EvaluatorContext[QuestionWithContext, object, object]:
        return EvaluatorContext[QuestionWithContext, object, object](
            name='test',
            inputs=RagInputs(question=question, context=context),
            metadata=metadata,
            expected_output=expected_output,
            output=output,
            duration=0.0,
            _span_tree=SpanTreeRecordingError('spans were not recorded'),
            attributes={},
            metrics={},
        )

    def _ctx_q(
        question: str,
        *,
        output: object = 'some answer',
        expected_output: object = None,
        metadata: object = None,
    ) -> EvaluatorContext[HasQuestion, object, object]:
        return EvaluatorContext[HasQuestion, object, object](
            name='test',
            inputs=QInputs(question=question),
            metadata=metadata,
            expected_output=expected_output,
            output=output,
            duration=0.0,
            _span_tree=SpanTreeRecordingError('spans were not recorded'),
            attributes={},
            metrics={},
        )

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

    def _patch_judge(
        mocker: MockerFixture,
        target: str,
        *,
        reason: str = 'ok',
        pass_: bool = True,
        score: float = 0.9,
    ):
        grading_output = GradingOutput(reason=reason, pass_=pass_, score=score)
        return mocker.patch(target, return_value=grading_output)


async def test_faithfulness(mocker: MockerFixture):
    """`Faithfulness` routes through `judge_input_output` with a context-aware rubric."""
    mock = _patch_judge(
        mocker,
        'pydantic_evals.evaluators.quality.judge_input_output',
        reason='All supported',
        pass_=True,
        score=1.0,
    )

    ctx = _ctx_qc(
        'Where is Paris?',
        ['Paris is the capital of France.'],
        output='Paris is in France.',
    )
    evaluator = Faithfulness()
    result = await evaluator.evaluate(ctx)

    assert to_jsonable_python(result) == snapshot(
        {
            'Faithfulness_score': {'value': 1.0, 'reason': 'All supported'},
            'Faithfulness_pass': {'value': True, 'reason': 'All supported'},
        }
    )
    assert mock.call_count == 1
    call_args, _ = mock.call_args
    inputs_payload, output_arg, rubric_arg, model_arg, settings_arg = call_args
    assert inputs_payload == {
        'question': 'Where is Paris?',
        'context': ['Paris is the capital of France.'],
    }
    assert output_arg == 'Paris is in France.'
    assert 'fraction of factual claims' in rubric_arg
    assert model_arg is None
    assert settings_arg is None


async def test_answer_relevance(mocker: MockerFixture):
    """`AnswerRelevance` pulls `question` directly from `ctx.inputs`."""
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.quality.judge_input_output')

    ctx = _ctx_q('What is 2+2?', output='4')
    evaluator = AnswerRelevance()
    await evaluator.evaluate(ctx)

    inputs_arg, output_arg, rubric_arg, *_ = mock.call_args[0]
    assert inputs_arg == {'question': 'What is 2+2?'}
    assert output_arg == '4'
    assert 'directly answer' in rubric_arg


async def test_context_precision(mocker: MockerFixture):
    """`ContextPrecision` shows the context as both the input's context field and as the output."""
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.quality.judge_input_output')

    passages = [
        'Paris is the capital of France.',
        'The Seine flows through Paris.',
        'Lyon is a city in France.',
    ]
    ctx = _ctx_qc('What is the capital of France?', passages, output='unused')
    evaluator = ContextPrecision()
    await evaluator.evaluate(ctx)

    inputs_arg, output_arg, rubric_arg, *_ = mock.call_args[0]
    assert inputs_arg == {
        'question': 'What is the capital of France?',
        'context': passages,
    }
    assert output_arg == passages
    assert 'fraction of the context that is relevant' in rubric_arg


async def test_context_recall_with_expected(mocker: MockerFixture):
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.quality.judge_input_output_expected')

    ctx = _ctx_qc(
        'What is the capital of France?',
        ['France is a country in Western Europe.'],
        output='Paris',
        expected_output='Paris is the capital of France',
    )
    evaluator = ContextRecall()
    await evaluator.evaluate(ctx)

    inputs_arg, output_arg, expected_arg, rubric_arg, *_ = mock.call_args[0]
    assert inputs_arg == {
        'question': 'What is the capital of France?',
        'context': ['France is a country in Western Europe.'],
    }
    assert output_arg == 'Paris'
    assert expected_arg == 'Paris is the capital of France'
    assert 'ground-truth answer' in rubric_arg


async def test_context_recall_skips_when_no_expected(mocker: MockerFixture):
    """When no ground truth is available, the evaluator skips rather than fabricating one."""
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.quality.judge_input_output_expected')

    ctx = _ctx_qc('q', ['some context'], output='a', expected_output=None)
    evaluator = ContextRecall()
    result = await evaluator.evaluate(ctx)

    assert result == {}
    assert mock.call_count == 0


async def test_hallucination_detector_semantics(mocker: MockerFixture):
    """`Hallucination` fires the assertion when the judge says claims are unsupported."""
    mock = _patch_judge(
        mocker,
        'pydantic_evals.evaluators.quality.judge_input_output',
        reason='Contains unsupported claim about Lyon',
        pass_=True,
        score=0.4,
    )

    ctx = _ctx_qc(
        'What is the capital of France?',
        ['Paris is the capital of France.'],
        output='Paris, and Lyon is the capital of Germany.',
    )
    evaluator = Hallucination()
    result = await evaluator.evaluate(ctx)

    assert to_jsonable_python(result) == snapshot(
        {
            'Hallucination_score': {'value': 0.4, 'reason': 'Contains unsupported claim about Lyon'},
            'Hallucination_pass': {'value': True, 'reason': 'Contains unsupported claim about Lyon'},
        }
    )
    rubric_arg = mock.call_args[0][2]
    assert 'pass=true ONLY if one or more' in rubric_arg


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


async def test_gemba_score_da(mocker: MockerFixture):
    mock = mocker.patch(
        'pydantic_evals.evaluators.quality.judge_gemba_da',
        return_value=GembaScoreOutput(reason='Accurate', score=92),
    )
    mocker.patch('pydantic_evals.evaluators.quality.judge_gemba_sqm')

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
    """Quality-pack evaluators deserialize from YAML configs via `DEFAULT_EVALUATORS`."""
    names = {cls.__name__ for cls in DEFAULT_EVALUATORS}
    assert {
        'Faithfulness',
        'AnswerRelevance',
        'ContextPrecision',
        'ContextRecall',
        'Hallucination',
        'GEval',
        'GembaScore',
    }.issubset(names)


async def test_score_and_assertion_configuration(mocker: MockerFixture):
    """The `score` / `assertion` output configs mirror `LLMJudge`."""
    _patch_judge(mocker, 'pydantic_evals.evaluators.quality.judge_input_output')

    ctx = _ctx_qc('q', ['c'], output='a')
    evaluator = Faithfulness(
        score=OutputConfig(evaluation_name='my_score', include_reason=False),
        assertion=False,
    )
    result = await evaluator.evaluate(ctx)
    assert to_jsonable_python(result) == snapshot({'my_score': 0.9})


async def test_assertion_only_disables_score(mocker: MockerFixture):
    """Passing `score=False` suppresses the score output while keeping the assertion."""
    _patch_judge(
        mocker,
        'pydantic_evals.evaluators.quality.judge_input_output',
        reason='looks fine',
        pass_=True,
        score=0.5,
    )

    ctx = _ctx_qc('q', ['c'], output='a')
    evaluator = Faithfulness(score=False)
    result = await evaluator.evaluate(ctx)
    assert to_jsonable_python(result) == snapshot({'Faithfulness': {'value': True, 'reason': 'looks fine'}})


async def test_model_and_settings_are_plumbed_through(mocker: MockerFixture):
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.quality.judge_input_output')
    settings = ModelSettings(temperature=0.0)

    evaluator = AnswerRelevance(model='openai:gpt-5.2', model_settings=settings)
    await evaluator.evaluate(_ctx_q('What is 2+2?'))

    assert mock.call_args.args[3] == 'openai:gpt-5.2'
    assert mock.call_args.args[4] == settings


@pytest.mark.anyio
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
    assert '1. Step A.' in prompt
    assert '2. Step B.' in prompt
    assert 'coherence' in prompt
    assert 'between 1 and 5' in prompt


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
    assert '"Hello"' in prompt
    assert '"Bonjour"' in prompt
    assert 'French human reference: "Salut"' in prompt
    assert '0 to 100' in prompt


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
