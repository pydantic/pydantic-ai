from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from pydantic_core import to_jsonable_python
from pytest_mock import MockerFixture

from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import OutputConfig, ragas
    from pydantic_evals.evaluators.common import DEFAULT_EVALUATORS
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.evaluators.llm_as_a_judge import GradingOutput
    from pydantic_evals.evaluators.ragas import (
        AnswerRelevance,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
        HasQuestion,
        QuestionWithContext,
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
        'pydantic_evals.evaluators.ragas.judge_input_output',
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
            'ragas.Faithfulness_score': {'value': 1.0, 'reason': 'All supported'},
            'ragas.Faithfulness_pass': {'value': True, 'reason': 'All supported'},
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
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.ragas.judge_input_output')

    ctx = _ctx_q('What is 2+2?', output='4')
    evaluator = AnswerRelevance()
    await evaluator.evaluate(ctx)

    inputs_arg, output_arg, rubric_arg, *_ = mock.call_args[0]
    assert inputs_arg == {'question': 'What is 2+2?'}
    assert output_arg == '4'
    assert 'directly answer' in rubric_arg


async def test_context_precision(mocker: MockerFixture):
    """`ContextPrecision` shows the context as both the input's context field and as the output."""
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.ragas.judge_input_output')

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
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.ragas.judge_input_output_expected')

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
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.ragas.judge_input_output_expected')

    ctx = _ctx_qc('q', ['some context'], output='a', expected_output=None)
    evaluator = ContextRecall()
    result = await evaluator.evaluate(ctx)

    assert result == {}
    assert mock.call_count == 0


async def test_ragas_evaluators_registered_in_defaults():
    """Ragas presets deserialize from YAML configs via `DEFAULT_EVALUATORS` under their namespaced names."""
    names = {cls.get_serialization_name() for cls in DEFAULT_EVALUATORS}
    assert {
        'ragas.Faithfulness',
        'ragas.AnswerRelevance',
        'ragas.ContextPrecision',
        'ragas.ContextRecall',
    }.issubset(names)


async def test_model_instance_serialized_as_string():
    """Each Ragas preset serializes a `Model` instance as its `model_id`, matching `LLMJudge`."""
    model = TestModel()
    evaluators = [
        Faithfulness(model=model),
        AnswerRelevance(model=model),
        ContextPrecision(model=model),
        ContextRecall(model=model),
    ]
    for evaluator in evaluators:
        assert evaluator.build_serialization_arguments()['model'] == model.model_id

    # A string model name is already serializable and passes through unchanged.
    assert AnswerRelevance(model='openai:gpt-5.2').build_serialization_arguments()['model'] == 'openai:gpt-5.2'


async def test_namespaced_serialization_name():
    """The presets live under the `ragas.` namespace so bare metric names stay free."""
    assert ragas.Faithfulness.get_serialization_name() == 'ragas.Faithfulness'
    assert ragas.ContextRecall.get_serialization_name() == 'ragas.ContextRecall'


async def test_score_and_assertion_configuration(mocker: MockerFixture):
    """The `score` / `assertion` output configs mirror `LLMJudge`."""
    _patch_judge(mocker, 'pydantic_evals.evaluators.ragas.judge_input_output')

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
        'pydantic_evals.evaluators.ragas.judge_input_output',
        reason='looks fine',
        pass_=True,
        score=0.5,
    )

    ctx = _ctx_qc('q', ['c'], output='a')
    evaluator = Faithfulness(score=False)
    result = await evaluator.evaluate(ctx)
    assert to_jsonable_python(result) == snapshot({'ragas.Faithfulness': {'value': True, 'reason': 'looks fine'}})


async def test_model_and_settings_are_plumbed_through(mocker: MockerFixture):
    mock = _patch_judge(mocker, 'pydantic_evals.evaluators.ragas.judge_input_output')
    settings = ModelSettings(temperature=0.0)

    evaluator = AnswerRelevance(model='openai:gpt-5.2', model_settings=settings)
    await evaluator.evaluate(_ctx_q('What is 2+2?'))

    assert mock.call_args.args[3] == 'openai:gpt-5.2'
    assert mock.call_args.args[4] == settings
