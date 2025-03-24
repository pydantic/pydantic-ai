import json
import sys
from functools import partial
from pathlib import Path

import pytest
from dirty_equals import HasRepr
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire
from pydantic import BaseModel

from pydantic_evals import Case, Dataset
from pydantic_evals.dataset import increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import python
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.reporting import ReportCase

pytestmark = pytest.mark.anyio

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup


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


async def simple_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Simple evaluator that checks if the output matches the expected output."""
    if ctx.expected_output is None:
        return {'result': 'no_expected_output'}

    return {
        'correct': ctx.output.answer == ctx.expected_output.answer,
        'confidence': ctx.output.confidence,
    }


async def metadata_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Evaluator that uses metadata."""
    if ctx.metadata is None:
        return {'result': 'no_metadata'}

    return {
        'difficulty': ctx.metadata.difficulty,
        'category': ctx.metadata.category,
    }


async def test_case_init():
    """Test Case initialization."""
    case = Case(
        name='test',
        inputs=TaskInput(query='What is 2+2?'),
        expected_output=TaskOutput(answer='4'),
        metadata=TaskMetadata(difficulty='easy'),
        evaluators=(simple_evaluator,),
    )

    assert case.name == 'test'
    assert case.inputs.query == 'What is 2+2?'
    assert case.expected_output is not None
    assert case.expected_output.answer == '4'
    assert case.metadata is not None
    assert case.metadata.difficulty == 'easy'
    assert len(case.evaluators) == 1


async def test_dataset_init(example_cases: list[Case[TaskInput, TaskOutput, TaskMetadata]]):
    """Test Dataset initialization."""
    dataset = Dataset(cases=example_cases, evaluators=[simple_evaluator])

    assert len(dataset.cases) == 2
    assert dataset.cases[0].name == 'case1'
    assert dataset.cases[1].name == 'case2'
    assert len(dataset.evaluators) == 1


async def test_add_evaluator(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test adding evaluators to a dataset."""
    assert len(example_dataset.evaluators) == 0

    example_dataset.add_evaluator(simple_evaluator)
    assert len(example_dataset.evaluators) == 1

    example_dataset.add_evaluator(metadata_evaluator)
    assert len(example_dataset.evaluators) == 2


async def test_evaluate(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset."""
    example_dataset.add_evaluator(simple_evaluator)

    async def mock_task(inputs: TaskInput) -> TaskOutput:
        if inputs.query == 'What is 2+2?':
            return TaskOutput(answer='4')
        elif inputs.query == 'What is the capital of France?':
            return TaskOutput(answer='Paris')
        return TaskOutput(answer='Unknown')

    report = await example_dataset.evaluate(mock_task)

    assert report is not None
    assert len(report.cases) == 2
    assert report.cases[0].model_dump() == snapshot(
        {
            'assertions': {'correct': {'name': 'correct', 'reason': None, 'source': 'simple_evaluator', 'value': True}},
            'attributes': {},
            'expected_output': {'answer': '4', 'confidence': 1.0},
            'inputs': {'query': 'What is 2+2?'},
            'labels': {},
            'metadata': {'category': 'general', 'difficulty': 'easy'},
            'metrics': {},
            'name': 'case1',
            'output': {'answer': '4', 'confidence': 1.0},
            'scores': {
                'confidence': {'name': 'confidence', 'reason': None, 'source': 'simple_evaluator', 'value': 1.0},
                'correct': {'name': 'correct', 'reason': None, 'source': 'simple_evaluator', 'value': True},
            },
            'span_id': '0000000000000003',
            'task_duration': 1.0,
            'total_duration': 6.0,
            'trace_id': '00000000000000000000000000000001',
        }
    )


async def test_evaluate_with_concurrency(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with concurrency limits."""
    example_dataset.add_evaluator(simple_evaluator)

    async def mock_task(inputs: TaskInput) -> TaskOutput:
        if inputs.query == 'What is 2+2?':
            return TaskOutput(answer='4')
        elif inputs.query == 'What is the capital of France?':
            return TaskOutput(answer='Paris')
        return TaskOutput(answer='Unknown')

    report = await example_dataset.evaluate(mock_task, max_concurrency=1)

    assert report is not None
    assert len(report.cases) == 2
    assert report.cases[0].model_dump() == snapshot(
        {
            'assertions': {'correct': {'name': 'correct', 'reason': None, 'source': 'simple_evaluator', 'value': True}},
            'attributes': {},
            'expected_output': {'answer': '4', 'confidence': 1.0},
            'inputs': {'query': 'What is 2+2?'},
            'labels': {},
            'metadata': {'category': 'general', 'difficulty': 'easy'},
            'metrics': {},
            'name': 'case1',
            'output': {'answer': '4', 'confidence': 1.0},
            'scores': {
                'confidence': {'name': 'confidence', 'reason': None, 'source': 'simple_evaluator', 'value': 1.0},
                'correct': {'name': 'correct', 'reason': None, 'source': 'simple_evaluator', 'value': True},
            },
            'span_id': '0000000000000003',
            'task_duration': 1.0,
            'total_duration': 3.0,
            'trace_id': '00000000000000000000000000000001',
        }
    )


async def test_evaluate_with_failing_task(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test evaluating a dataset with a failing task."""
    example_dataset.add_evaluator(simple_evaluator)

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

    async def failing_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
        raise ValueError('Evaluator error')

    example_dataset.add_evaluator(failing_evaluator)

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

        set_eval_attribute('is_about_france', 'France' in inputs.query)
        return TaskOutput(answer=f'answer to {inputs.query}')

    report = await example_dataset.evaluate(my_task)
    assert report.cases == [
        ReportCase(
            name='case1',
            inputs={'query': 'What is 2+2?'},
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
            inputs={'query': 'What is the capital of France?'},
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
    example_dataset.to_file(json_path)

    assert json_path.exists()

    # Test loading back
    loaded_dataset = Dataset[TaskInput, TaskOutput, TaskMetadata].from_file(json_path)
    assert len(loaded_dataset.cases) == 2
    assert loaded_dataset.cases[0].name == 'case1'
    assert loaded_dataset.cases[0].inputs.query == 'What is 2+2?'


async def test_from_text(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test creating a dataset from text."""
    dataset_dict = {
        'cases': [
            {
                'name': 'text_case',
                'inputs': {'query': 'What is the capital of Germany?'},
                'expected_output': {'answer': 'Berlin', 'confidence': 0.9},
                'metadata': {'difficulty': 'hard', 'category': 'geography'},
            }
        ]
    }

    json_text = json.dumps(dataset_dict)

    loaded_dataset = Dataset[TaskInput, TaskOutput, TaskMetadata].from_text(json_text)
    assert loaded_dataset.cases == [
        Case(
            name='text_case',
            inputs=TaskInput(query='What is the capital of Germany?'),
            metadata=TaskMetadata(difficulty='hard', category='geography'),
            expected_output=TaskOutput(answer='Berlin', confidence=0.9),
            evaluators=(),
        )
    ]


async def test_invalid_evaluator_result_type(example_dataset: Dataset[TaskInput, TaskOutput, TaskMetadata]):
    """Test that an invalid evaluator result type raises an error."""
    invalid_evaluator = partial(python, expression='...')
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
                                "EvaluatorSpec(call='python', args=[], kwargs={'expression': '...'}) returned a value of an invalid type: Ellipsis."
                            )
                        ],
                    ),
                    ExceptionGroup(
                        'unhandled errors in a TaskGroup',
                        [
                            ValueError(
                                "EvaluatorSpec(call='python', args=[], kwargs={'expression': '...'}) returned a value of an invalid type: Ellipsis."
                            )
                        ],
                    ),
                ],
            )
        )
    )
