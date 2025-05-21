from __future__ import annotations as _annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from dirty_equals import HasRepr
from inline_snapshot import snapshot
from pydantic import BaseModel

from ..conftest import try_import
from .utils import render_table

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire

    from pydantic_evals import Case, Dataset
    from pydantic_evals.dataset import increment_eval_metric, set_eval_attribute
    from pydantic_evals.evaluators import EvaluationResult, Evaluator, EvaluatorOutput, LLMJudge, Python
    from pydantic_evals.evaluators.context import EvaluatorContext

    @dataclass
    class MockEvaluator(Evaluator[object, object, object]):
        """This is just for testing purposes. It just returns the wrapped value."""

        output: EvaluatorOutput

        def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluatorOutput:
            return self.output

    from pydantic_evals.reporting import ReportCase

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover


@pytest.fixture(autouse=True)
def use_logfire(capfire: CaptureLogfire):
    assert capfire


class TaskInput(BaseModel):
    query: str


class TaskOutput(BaseModel):
    answer: str
    confidence: float = 1.0


class TaskMetadata(BaseModel):
    difficulty: str = 'easy'
    category: str = 'general'


@pytest.fixture
def example_cases() -> list[Case[TaskInput, TaskOutput, TaskMetadata]]:
    return [
        Case(
            name='case1',
            inputs=TaskInput(query='What is 2+2?'),
            expected_output=TaskOutput(answer='4'),
            metadata=TaskMetadata(difficulty='easy'),
        ),
        Case(
            name='case2',
            inputs=TaskInput(query='What is the capital of France?'),
            expected_output=TaskOutput(answer='Paris'),
            metadata=TaskMetadata(difficulty='medium', category='geography'),
        ),
    ]


@pytest.fixture
def example_dataset(
    example_cases: list[Case[TaskInput, TaskOutput, TaskMetadata]],
) -> Dataset[TaskInput, TaskOutput, TaskMetadata]:
    return Dataset[TaskInput, TaskOutput, TaskMetadata](cases=example_cases)


@pytest.fixture
def simple_evaluator() -> type[Evaluator[TaskInput, TaskOutput, TaskMetadata]]:
    @dataclass
    class SimpleEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
            if ctx.expected_output is None:  # pragma: no cover
                return {'result': 'no_expected_output'}

            return {
                'correct': ctx.output.answer == ctx.expected_output.answer,
                'confidence': ctx.output.confidence,
            }

    return SimpleEvaluator


async def test_dataset_init(
    example_cases: list[Case[TaskInput, TaskOutput, TaskMetadata]],
    simple_evaluator: type[Evaluator[TaskInput, TaskOutput, TaskMetadata]],
):
    """Test Dataset initialization."""
    dataset = Dataset(cases=example_cases, evaluators=[simple_evaluator()])

    assert len(dataset.cases) == 2
    assert dataset.cases[0].name == 'case1'
    assert dataset.cases[1].name == 'case2'
    assert len(dataset.evaluators) == 1


async def test_add_evaluator(
    example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata],
    simple_evaluator: type[Evaluator[TaskInput, TaskOutput, TaskMetadata]],
):
    """Test adding evaluators to a dataset."""
    assert len(example_dataset.evaluators) == 0

    example_dataset.add_evaluator(simple_evaluator())
    assert len(example_dataset.evaluators) == 1

    @dataclass
    class MetadataEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):  # pragma: no cover
            """Evaluator that uses metadata."""
            if ctx.metadata is None:
                return {'result': 'no_metadata'}

            return {
                'difficulty': ctx.metadata.difficulty,
                'category': ctx.metadata.category,
            }

    example_dataset.add_evaluator(MetadataEvaluator())
    assert len(example_dataset.evaluators) == 2

    dataset = Dataset[TaskInput, TaskOutput, TaskMetadata](
        cases=[
            Case(
                name='My Case 1',
                inputs=TaskInput(query='What is 1+1?'),
            ),
            Case(
                name='My Case 2',
                inputs=TaskInput(query='What is 2+2?'),
            ),
        ]
    )
    dataset.add_evaluator(Python('ctx.output > 0'))
    dataset.add_evaluator(Python('ctx.output == 2'), specific_case='My Case 1')
    dataset.add_evaluator(Python('ctx.output == 4'), specific_case='My Case 2')
    with pytest.raises(ValueError) as exc_info:
        dataset.add_evaluator(Python('ctx.output == 4'), specific_case='My Case 3')
    assert str(exc_info.value) == snapshot("Case 'My Case 3' not found in the dataset")

    assert dataset.model_dump(mode='json', exclude_defaults=True, context={'use_short_form': True}) == {
        'cases': [
            {
                'evaluators': [{'Python': 'ctx.output == 2'}],
                'inputs': {'query': 'What is 1+1?'},
                'name': 'My Case 1',
            },
            {
                'evaluators': [{'Python': 'ctx.output == 4'}],
                'inputs': {'query': 'What is 2+2?'},
                'name': 'My Case 2',
            },
        ],
        'evaluators': [{'Python': 'ctx.output > 0'}],
    }


async def test_evaluate(
    example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata],
    simple_evaluator: type[Evaluator[TaskInput, TaskOutput, TaskMetadata]],
):
    """Test evaluating a dataset."""
    example_dataset.add_evaluator(simple_evaluator())

    async def mock_task(inputs: TaskInput) -> TaskOutput:
        if inputs.query == 'What is 2+2?':
            return TaskOutput(answer='4')
        elif inputs.query == 'What is the capital of France?':
            return TaskOutput(answer='Paris')
        return TaskOutput(answer='Unknown')  # pragma: no cover

    report = await example_dataset.evaluate(mock_task)

    assert report is not None
    assert len(report.cases) == 2
    assert report.cases[0].model_dump() == snapshot(
        {
            'assertions': {
                'correct': {
                    'name': 'correct',
                    'reason': None,
                    'source': {'name': 'SimpleEvaluator', 'arguments': None},
                    'value': True,
                }
            },
            'attributes': {},
            'expected_output': {'answer': '4', 'confidence': 1.0},
            'inputs': {'query': 'What is 2+2?'},
            'labels': {},
            'metadata': {'category': 'general', 'difficulty': 'easy'},
            'metrics': {},
            'name': 'case1',
            'output': {'answer': '4', 'confidence': 1.0},
            'scores': {
                'confidence': {
                    'name': 'confidence',
                    'reason': None,
                    'source': {'name': 'SimpleEvaluator', 'arguments': None},
                    'value': 1.0,
                }
            },
            'span_id': '0000000000000003',
            'task_duration': 1.0,
            'total_duration': 6.0,
            'trace_id': '00000000000000000000000000000001',
        }
    )


async def test_evaluate_with_concurrency(
    example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata],
    simple_evaluator: type[Evaluator[TaskInput, TaskOutput, TaskMetadata]],
):
    """Test evaluating a dataset with concurrency limits."""
    example_dataset.add_evaluator(simple_evaluator())

    async def mock_task(inputs: TaskInput) -> TaskOutput:
        if inputs.query == 'What is 2+2?':
            return TaskOutput(answer='4')
        elif inputs.query == 'What is the capital of France?':
            return TaskOutput(answer='Paris')
        return TaskOutput(answer='Unknown')  # pragma: no cover

    report = await example_dataset.evaluate(mock_task, max_concurrency=1)

    assert report is not None
    assert len(report.cases) == 2
    assert report.cases[0].model_dump() == snapshot(
        {
            'assertions': {
                'correct': {
                    'name': 'correct',
                    'reason': None,
                    'source': {'name': 'SimpleEvaluator', 'arguments': None},
                    'value': True,
                }
            },
            'attributes': {},
            'expected_output': {'answer': '4', 'confidence': 1.0},
            'inputs': {'query': 'What is 2+2?'},
            'labels': {},
            'metadata': {'category': 'general', 'difficulty': 'easy'},
            'metrics': {},
            'name': 'case1',
            'output': {'answer': '4', 'confidence': 1.0},
            'scores': {
                'confidence': {
                    'name': 'confidence',
                    'reason': None,
                    'source': {'name': 'SimpleEvaluator', 'arguments': None},
                    'value': 1.0,
                }
            },
            'span_id': '0000000000000003',
            'task_duration': 1.0,
            'total_duration': 3.0,
            'trace_id': '00000000000000000000000000000001',
        }
    )


async def test_evaluate_with_failing_task(
    example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata],
    simple_evaluator: type[Evaluator[TaskInput, TaskOutput, TaskMetadata]],
):
    """Test evaluating a dataset with a failing task."""
    example_dataset.add_evaluator(simple_evaluator())

    async def failing_task(inputs: TaskInput) -> TaskOutput:
        if inputs.query == 'What is 2+2?':
            raise ValueError('Task error')
        return TaskOutput(answer='Paris')

    # TODO: Should we include the exception in the result rather than bubbling up the error?
    with pytest.raises(ExceptionGroup) as exc_info:
        await example_dataset.evaluate(failing_task)
    assert exc_info.value == HasRepr(
        repr(ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Task error')]))
    )


async def test_evaluate_with_failing_evaluator(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with a failing evaluator."""

    class FailingEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
            raise ValueError('Evaluator error')

    example_dataset.add_evaluator(FailingEvaluator())

    async def mock_task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer='4')

    with pytest.raises(ExceptionGroup) as exc_info:
        await example_dataset.evaluate(mock_task)

    assert exc_info.value == HasRepr(
        repr(
            ExceptionGroup(
                'unhandled errors in a TaskGroup',
                [
                    ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Evaluator error')]),
                    ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Evaluator error')]),
                ],
            )
        )
    )


async def test_increment_eval_metric(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test the increment_eval_metric function."""

    async def my_task(inputs: TaskInput) -> TaskOutput:
        for _ in inputs.query:
            increment_eval_metric('chars', 1)

        increment_eval_metric('phantom', 0)  # doesn't get created due to being zero

        set_eval_attribute('is_about_france', 'France' in inputs.query)
        return TaskOutput(answer=f'answer to {inputs.query}')

    report = await example_dataset.evaluate(my_task)
    assert report.cases == snapshot(
        [
            ReportCase(
                name='case1',
                inputs=TaskInput(query='What is 2+2?'),
                metadata=TaskMetadata(difficulty='easy', category='general'),
                expected_output=TaskOutput(answer='4', confidence=1.0),
                output=TaskOutput(answer='answer to What is 2+2?', confidence=1.0),
                metrics={'chars': 12},
                attributes={'is_about_france': False},
                scores={},
                labels={},
                assertions={},
                task_duration=1.0,
                total_duration=3.0,
                trace_id='00000000000000000000000000000001',
                span_id='0000000000000003',
            ),
            ReportCase(
                name='case2',
                inputs=TaskInput(query='What is the capital of France?'),
                metadata=TaskMetadata(difficulty='medium', category='geography'),
                expected_output=TaskOutput(answer='Paris', confidence=1.0),
                output=TaskOutput(answer='answer to What is the capital of France?', confidence=1.0),
                metrics={'chars': 30},
                attributes={'is_about_france': True},
                scores={},
                labels={},
                assertions={},
                task_duration=1.0,
                total_duration=3.0,
                trace_id='00000000000000000000000000000001',
                span_id='0000000000000007',
            ),
        ]
    )


async def test_repeated_name_outputs(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test the increment_eval_metric function."""

    async def my_task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=f'answer to {inputs.query}')

    example_dataset.add_evaluator(MockEvaluator({'output': 'a'}))
    example_dataset.add_evaluator(MockEvaluator({'output': 'b'}))
    example_dataset.add_evaluator(MockEvaluator({'output': 'c'}))

    report = await example_dataset.evaluate(my_task)
    assert report.cases == snapshot(
        [
            ReportCase(
                name='case1',
                inputs=TaskInput(query='What is 2+2?'),
                metadata=TaskMetadata(difficulty='easy', category='general'),
                expected_output=TaskOutput(answer='4', confidence=1.0),
                output=TaskOutput(answer='answer to What is 2+2?', confidence=1.0),
                metrics={},
                attributes={},
                scores={},
                labels={
                    'output': EvaluationResult(
                        name='output', value='a', reason=None, source=MockEvaluator(output={'output': 'a'})
                    ),
                    'output_2': EvaluationResult(
                        name='output', value='b', reason=None, source=MockEvaluator(output={'output': 'b'})
                    ),
                    'output_3': EvaluationResult(
                        name='output', value='c', reason=None, source=MockEvaluator(output={'output': 'c'})
                    ),
                },
                assertions={},
                task_duration=1.0,
                total_duration=6.0,
                trace_id='00000000000000000000000000000001',
                span_id='0000000000000003',
            ),
            ReportCase(
                name='case2',
                inputs=TaskInput(query='What is the capital of France?'),
                metadata=TaskMetadata(difficulty='medium', category='geography'),
                expected_output=TaskOutput(answer='Paris', confidence=1.0),
                output=TaskOutput(answer='answer to What is the capital of France?', confidence=1.0),
                metrics={},
                attributes={},
                scores={},
                labels={
                    'output': EvaluationResult(
                        name='output', value='a', reason=None, source=MockEvaluator(output={'output': 'a'})
                    ),
                    'output_2': EvaluationResult(
                        name='output', value='b', reason=None, source=MockEvaluator(output={'output': 'b'})
                    ),
                    'output_3': EvaluationResult(
                        name='output', value='c', reason=None, source=MockEvaluator(output={'output': 'c'})
                    ),
                },
                assertions={},
                task_duration=1.0,
                total_duration=4.0,
                trace_id='00000000000000000000000000000001',
                span_id='0000000000000007',
            ),
        ]
    )


async def test_genai_attribute_collection(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    async def my_task(inputs: TaskInput) -> TaskOutput:
        with logfire.span(
            'my chat span',
            **{  # type: ignore
                'gen_ai.operation.name': 'chat',
                'gen_ai.usage.input_tokens': 1,
                'gen_ai.usage.details.special_tokens': 2,
                'other_attribute': 3,
            },
        ):
            with logfire.span('some other span'):
                pass
        return TaskOutput(answer=f'answer to {inputs.query}')

    report = await example_dataset.evaluate(my_task)
    assert report.cases == snapshot(
        [
            ReportCase(
                name='case1',
                inputs=TaskInput(query='What is 2+2?'),
                metadata=TaskMetadata(difficulty='easy', category='general'),
                expected_output=TaskOutput(answer='4', confidence=1.0),
                output=TaskOutput(answer='answer to What is 2+2?', confidence=1.0),
                metrics={'requests': 1, 'input_tokens': 1, 'special_tokens': 2},
                attributes={},
                scores={},
                labels={},
                assertions={},
                task_duration=5.0,
                total_duration=7.0,
                trace_id='00000000000000000000000000000001',
                span_id='0000000000000003',
            ),
            ReportCase(
                name='case2',
                inputs=TaskInput(query='What is the capital of France?'),
                metadata=TaskMetadata(difficulty='medium', category='geography'),
                expected_output=TaskOutput(answer='Paris', confidence=1.0),
                output=TaskOutput(answer='answer to What is the capital of France?', confidence=1.0),
                metrics={'requests': 1, 'input_tokens': 1, 'special_tokens': 2},
                attributes={},
                scores={},
                labels={},
                assertions={},
                task_duration=5.0,
                total_duration=7.0,
                trace_id='00000000000000000000000000000001',
                span_id='000000000000000b',
            ),
        ]
    )


async def test_serialization_to_yaml(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata], tmp_path: Path):
    """Test serializing a dataset to YAML."""
    yaml_path = tmp_path / 'test_cases.yaml'
    example_dataset.to_file(yaml_path)

    assert yaml_path.exists()

    # Test loading back
    loaded_dataset = Dataset[TaskInput, TaskOutput, TaskMetadata].from_file(yaml_path)
    assert len(loaded_dataset.cases) == 2
    assert loaded_dataset.cases[0].name == 'case1'
    assert loaded_dataset.cases[0].inputs.query == 'What is 2+2?'


async def test_serialization_to_json(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata], tmp_path: Path):
    """Test serializing a dataset to JSON."""
    json_path = tmp_path / 'test_cases.json'
    example_dataset.to_file(json_path, fmt='json')  # purposely specify fmt, for coverage reasons

    assert json_path.exists()

    # Test loading back
    loaded_dataset = Dataset[TaskInput, TaskOutput, TaskMetadata].from_file(json_path)
    assert len(loaded_dataset.cases) == 2
    assert loaded_dataset.cases[0].name == 'case1'
    assert loaded_dataset.cases[0].inputs.query == 'What is 2+2?'

    raw = json.loads(json_path.read_text())
    schema = raw['$schema']
    assert isinstance(schema, str)
    assert (tmp_path / schema).exists()


def test_serialization_errors(tmp_path: Path):
    with pytest.raises(ValueError) as exc_info:
        Dataset[TaskInput, TaskOutput, TaskMetadata].from_file(tmp_path / 'test_cases.abc')

    assert str(exc_info.value) == snapshot(
        "Could not infer format for filename 'test_cases.abc'. Use the `fmt` argument to specify the format."
    )


async def test_from_text():
    """Test creating a dataset from text."""
    dataset_dict = {
        'cases': [
            {
                'name': '1',
                'inputs': {'query': 'What is the capital of Germany?'},
                'expected_output': {'answer': 'Berlin', 'confidence': 0.9},
                'metadata': {'difficulty': 'hard', 'category': 'geography'},
            },
            {
                'name': '2',
                'inputs': {'query': 'What is the capital of Germany?'},
                'expected_output': {'answer': 'Berlin', 'confidence': 0.9},
                'metadata': {'difficulty': 'hard', 'category': 'geography'},
                'evaluators': [{'LLMJudge': 'my rubric'}],
            },
        ],
        'evaluators': [{'LLMJudge': 'my rubric'}],
    }

    loaded_dataset = Dataset[TaskInput, TaskOutput, TaskMetadata].from_text(json.dumps(dataset_dict))
    assert loaded_dataset.cases == snapshot(
        [
            Case(
                name='1',
                inputs=TaskInput(query='What is the capital of Germany?'),
                metadata=TaskMetadata(difficulty='hard', category='geography'),
                expected_output=TaskOutput(answer='Berlin', confidence=0.9),
                evaluators=(),
            ),
            Case(
                name='2',
                inputs=TaskInput(query='What is the capital of Germany?'),
                metadata=TaskMetadata(difficulty='hard', category='geography'),
                expected_output=TaskOutput(answer='Berlin', confidence=0.9),
                evaluators=(LLMJudge(rubric='my rubric'),),
            ),
        ]
    )
    assert loaded_dataset.evaluators == snapshot([LLMJudge(rubric='my rubric')])


async def test_from_text_failure():
    """Test creating a dataset from text."""
    dataset_dict = {
        'cases': [
            {
                'name': 'text_case',
                'inputs': {'query': 'What is the capital of Germany?'},
                'expected_output': {'answer': 'Berlin', 'confidence': 0.9},
                'metadata': {'difficulty': 'hard', 'category': 'geography'},
                'evaluators': ['NotAnEvaluator'],
            }
        ],
        'evaluators': ['NotAnEvaluator'],
    }
    with pytest.raises(ExceptionGroup) as exc_info:
        Dataset[TaskInput, TaskOutput, TaskMetadata].from_text(json.dumps(dataset_dict))
    assert exc_info.value == HasRepr(
        repr(
            ExceptionGroup(
                '2 error(s) loading evaluators from registry',
                [
                    ValueError(
                        "Evaluator 'NotAnEvaluator' is not in the provided `custom_evaluator_types`. Valid choices: ['Equals', 'EqualsExpected', 'Contains', 'IsInstance', 'MaxDuration', 'LLMJudge', 'HasMatchingSpan']. If you are trying to use a custom evaluator, you must include its type in the `custom_evaluator_types` argument."
                    ),
                    ValueError(
                        "Evaluator 'NotAnEvaluator' is not in the provided `custom_evaluator_types`. Valid choices: ['Equals', 'EqualsExpected', 'Contains', 'IsInstance', 'MaxDuration', 'LLMJudge', 'HasMatchingSpan']. If you are trying to use a custom evaluator, you must include its type in the `custom_evaluator_types` argument."
                    ),
                ],
            )
        )
    )

    dataset_dict = {
        'cases': [
            {
                'name': 'text_case',
                'inputs': {'query': 'What is the capital of Germany?'},
                'expected_output': {'answer': 'Berlin', 'confidence': 0.9},
                'metadata': {'difficulty': 'hard', 'category': 'geography'},
                'evaluators': ['LLMJudge'],
            }
        ],
        'evaluators': ['LLMJudge'],
    }
    with pytest.raises(ExceptionGroup) as exc_info:
        Dataset[TaskInput, TaskOutput, TaskMetadata].from_text(json.dumps(dataset_dict))
    if sys.version_info >= (3, 10):
        assert exc_info.value == HasRepr(  # pragma: lax no cover
            repr(
                ExceptionGroup(
                    '2 error(s) loading evaluators from registry',
                    [
                        ValueError(
                            "Failed to instantiate evaluator 'LLMJudge' for dataset: LLMJudge.__init__() missing 1 required positional argument: 'rubric'"
                        ),
                        ValueError(
                            "Failed to instantiate evaluator 'LLMJudge' for case 'text_case': LLMJudge.__init__() missing 1 required positional argument: 'rubric'"
                        ),
                    ],
                )
            )
        )
    else:
        assert exc_info.value == HasRepr(  # pragma: lax no cover
            repr(
                ExceptionGroup(
                    '2 error(s) loading evaluators from registry',
                    [
                        ValueError(
                            "Failed to instantiate evaluator 'LLMJudge' for dataset: __init__() missing 1 required positional argument: 'rubric'"
                        ),
                        ValueError(
                            "Failed to instantiate evaluator 'LLMJudge' for case 'text_case': __init__() missing 1 required positional argument: 'rubric'"
                        ),
                    ],
                )
            )
        )


async def test_duplicate_evaluator_failure(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    @dataclass
    class FirstEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):  # pragma: no cover
            return False

    @dataclass
    class SecondEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):  # pragma: no cover
            return False

    SecondEvaluator.__name__ = FirstEvaluator.__name__
    with pytest.raises(ValueError) as exc_info:
        Dataset[TaskInput, TaskOutput, TaskMetadata].from_dict(
            {'cases': []}, custom_evaluator_types=(FirstEvaluator, SecondEvaluator)
        )
    assert str(exc_info.value) == snapshot("Duplicate evaluator class name: 'FirstEvaluator'")


async def test_invalid_evaluator_output_type(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test that an invalid evaluator output type raises an error."""
    invalid_evaluator = Python(expression='...')
    example_dataset.add_evaluator(invalid_evaluator)

    async def mock_task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer='4')

    with pytest.raises(ExceptionGroup) as exc_info:
        await example_dataset.evaluate(mock_task)
    assert exc_info.value == HasRepr(
        repr(
            ExceptionGroup(
                'unhandled errors in a TaskGroup',
                [
                    ExceptionGroup(
                        'unhandled errors in a TaskGroup',
                        [
                            ValueError(
                                "Python(expression='...').evaluate returned a value of an invalid type: Ellipsis."
                            )
                        ],
                    ),
                    ExceptionGroup(
                        'unhandled errors in a TaskGroup',
                        [
                            ValueError(
                                "Python(expression='...').evaluate returned a value of an invalid type: Ellipsis."
                            )
                        ],
                    ),
                ],
            )
        )
    )


async def test_dataset_evaluate_with_failing_task(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with a failing task."""

    async def failing_task(inputs: TaskInput) -> TaskOutput:
        raise ValueError('Task failed')

    with pytest.raises(ExceptionGroup) as exc_info:
        await example_dataset.evaluate(failing_task)
    assert exc_info.value == HasRepr(
        repr(ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Task failed'), ValueError('Task failed')]))
    )


async def test_dataset_evaluate_with_failing_evaluator(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with a failing evaluator."""

    class FailingEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> bool:
            raise ValueError('Evaluator failed')

    example_dataset.add_evaluator(FailingEvaluator())

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    with pytest.raises(ExceptionGroup) as exc_info:
        await example_dataset.evaluate(task)
    assert exc_info.value == HasRepr(
        repr(
            ExceptionGroup(
                'unhandled errors in a TaskGroup',
                [
                    ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Evaluator failed')]),
                    ExceptionGroup('unhandled errors in a TaskGroup', [ValueError('Evaluator failed')]),
                ],
            )
        )
    )


async def test_dataset_evaluate_with_invalid_evaluator_result(
    example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata],
):
    """Test evaluating a dataset with an evaluator that returns an invalid result type."""

    @dataclass
    class MyObject:
        pass

    class InvalidEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> Any:
            return MyObject()  # Return an invalid type

    example_dataset.add_evaluator(InvalidEvaluator())

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    with pytest.raises(ExceptionGroup) as exc_info:
        await example_dataset.evaluate(task)
    assert exc_info.value == HasRepr(
        repr(
            ExceptionGroup(
                'unhandled errors in a TaskGroup',
                [
                    ExceptionGroup(
                        'unhandled errors in a TaskGroup',
                        [
                            ValueError(
                                'test_dataset_evaluate_with_invalid_evaluator_result.<locals>.InvalidEvaluator().evaluate returned a value of an invalid type: test_dataset_evaluate_with_invalid_evaluator_result.<locals>.MyObject().'
                            )
                        ],
                    ),
                    ExceptionGroup(
                        'unhandled errors in a TaskGroup',
                        [
                            ValueError(
                                'test_dataset_evaluate_with_invalid_evaluator_result.<locals>.InvalidEvaluator().evaluate returned a value of an invalid type: test_dataset_evaluate_with_invalid_evaluator_result.<locals>.MyObject().'
                            )
                        ],
                    ),
                ],
            )
        )
    )


async def test_dataset_evaluate_with_custom_name(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with a custom task name."""

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    report = await example_dataset.evaluate(task, name='custom_task')
    assert report.name == 'custom_task'


async def test_dataset_evaluate_with_sync_task(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with a synchronous task."""

    def sync_task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    report = await example_dataset.evaluate(lambda x: asyncio.sleep(0, sync_task(x)))
    assert report.name == '<lambda>'
    assert len(report.cases) == 2


async def test_dataset_evaluate_with_no_expected_output(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with no expected output."""
    case = Case(
        name='no_output',
        inputs=TaskInput(query='hello'),
        metadata=TaskMetadata(difficulty='easy'),
    )
    dataset = Dataset(cases=[case])

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    report = await dataset.evaluate(task)
    assert len(report.cases) == 1
    assert report.cases[0].name == 'no_output'


async def test_dataset_evaluate_with_no_metadata(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with no metadata."""
    case = Case(
        name='no_metadata',
        inputs=TaskInput(query='hello'),
        expected_output=TaskOutput(answer='HELLO'),
    )
    dataset = Dataset(cases=[case])

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    report = await dataset.evaluate(task)
    assert len(report.cases) == 1
    assert report.cases[0].name == 'no_metadata'


async def test_dataset_evaluate_with_empty_cases(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with no cases."""
    dataset = Dataset(cases=[])

    async def task(inputs: TaskInput) -> TaskOutput:  # pragma: no cover
        return TaskOutput(answer=inputs.query.upper())

    report = await dataset.evaluate(task)
    assert len(report.cases) == 0


async def test_dataset_evaluate_with_multiple_evaluators(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with multiple evaluators."""

    class FirstEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> int:
            return len(ctx.output.answer)

    class SecondEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> int:
            return len(ctx.output.answer) + 1

    example_dataset.add_evaluator(FirstEvaluator())
    example_dataset.add_evaluator(SecondEvaluator())

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer=inputs.query.upper())

    report = await example_dataset.evaluate(task)
    assert len(report.cases) == 2
    assert len(report.cases[0].scores) == 2


@pytest.mark.anyio
async def test_unnamed_cases():
    dataset = Dataset[TaskInput, TaskOutput, TaskMetadata](
        cases=[
            Case(
                name=None,
                inputs=TaskInput(query='What is 1+1?'),
            ),
            Case(
                name='My Case',
                inputs=TaskInput(query='What is 2+2?'),
            ),
            Case(
                name=None,
                inputs=TaskInput(query='What is 1+2?'),
            ),
        ]
    )

    async def task(inputs: TaskInput) -> TaskOutput:
        return TaskOutput(answer='4')

    result = await dataset.evaluate(task)
    assert [case.name for case in dataset.cases] == [None, 'My Case', None]
    assert [case.name for case in result.cases] == ['Case 1', 'My Case', 'Case 3']


@pytest.mark.anyio
async def test_duplicate_case_names():
    with pytest.raises(ValueError) as exc_info:
        Dataset[TaskInput, TaskOutput, TaskMetadata](
            cases=[
                Case(
                    name='My Case',
                    inputs=TaskInput(query='What is 1+1?'),
                ),
                Case(
                    name='My Case',
                    inputs=TaskInput(query='What is 2+2?'),
                ),
            ]
        )
    assert str(exc_info.value) == "Duplicate case name: 'My Case'"

    dataset = Dataset[TaskInput, TaskOutput, TaskMetadata](
        cases=[
            Case(
                name='My Case',
                inputs=TaskInput(query='What is 1+1?'),
            ),
        ]
    )
    dataset.add_case(
        name='My Other Case',
        inputs=TaskInput(query='What is 2+2?'),
    )

    with pytest.raises(ValueError) as exc_info:
        dataset.add_case(
            name='My Case',
            inputs=TaskInput(query='What is 1+2?'),
        )
    assert str(exc_info.value) == "Duplicate case name: 'My Case'"


def test_add_invalid_evaluator():
    class NotAnEvaluator:
        pass

    class SimpleEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):  # pragma: no cover
            return False

    dataset = Dataset[TaskInput, TaskOutput, TaskMetadata](cases=[])

    with pytest.raises(ValueError) as exc_info:
        dataset.model_json_schema_with_evaluators((NotAnEvaluator,))  # type: ignore
    assert str(exc_info.value).startswith('All custom evaluator classes must be subclasses of Evaluator')

    with pytest.raises(ValueError) as exc_info:
        dataset.model_json_schema_with_evaluators((SimpleEvaluator,))
    assert str(exc_info.value).startswith('All custom evaluator classes must be decorated with `@dataclass`')


def test_import_generate_dataset():
    # This function is tough to test in an interesting way outside an example...
    # This at least ensures importing it doesn't fail.
    # TODO: Add an "example" that actually makes use of this functionality
    from pydantic_evals.generation import generate_dataset

    assert generate_dataset


def test_evaluate_non_serializable_inputs():
    @dataclass
    class MyInputs:
        output_type: type[str] | type[int]

    my_dataset = Dataset[MyInputs, Any, Any](
        cases=[
            Case(
                name='str',
                inputs=MyInputs(output_type=str),
                expected_output='abc',
            ),
            Case(
                name='int',
                inputs=MyInputs(output_type=int),
                expected_output=123,
            ),
        ],
    )

    async def my_task(my_inputs: MyInputs) -> int | str:
        if issubclass(my_inputs.output_type, str):
            return my_inputs.output_type('abc')
        else:
            return my_inputs.output_type(123)

    report = my_dataset.evaluate_sync(task=my_task)
    assert [c.inputs for c in report.cases] == snapshot([MyInputs(output_type=str), MyInputs(output_type=int)])

    table = report.console_table(include_input=True)
    assert render_table(table) == snapshot("""\
                                        Evaluation Summary: my_task
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Inputs                                                                             ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ str      │ test_evaluate_non_serializable_inputs.<locals>.MyInputs(output_type=<class 'str'>) │     1.0s │
├──────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ int      │ test_evaluate_non_serializable_inputs.<locals>.MyInputs(output_type=<class 'int'>) │     1.0s │
├──────────┼────────────────────────────────────────────────────────────────────────────────────┼──────────┤
│ Averages │                                                                                    │     1.0s │
└──────────┴────────────────────────────────────────────────────────────────────────────────────┴──────────┘
""")
