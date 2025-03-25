# Pydantic Evals

Pydantic Evals is a powerful evaluation framework designed to help you systematically test and evaluate the quality of your code, especially when working with LLM-powered applications.

!!! note "In Beta"
    Pydantic Evals support was [introduced](https://github.com/pydantic/pydantic-ai/pull/935) in v0.0.44 and is currently in beta. The API is subject to change. The documentation is incomplete.


## Installation

To install the Pydantic Evals package, you can use:

```bash
pip/uv-add pydantic-evals
```

## Datasets and Cases

The foundation of Pydantic Evals is the concept of datasets and test cases:

- [`Case`][pydantic_evals.Case]: A single test scenario consisting of inputs, expected outputs, metadata, and optional evaluators.
- [`Dataset`][pydantic_evals.Dataset]: A collection of test cases designed to evaluate a specific task or function.

```python {title="simple_eval_dataset.py"}
from pydantic_evals import Case, Dataset

case1 = Case(  # (1)!
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)

dataset = Dataset(cases=[case1])  # (2)!
```

1. Create a test case
2. Create a dataset from cases, in most real world cases you would have multiple cases `#!python Dataset(cases=[case1, case2, case3])`

## Evaluators

Evaluators are the components that analyze and score the results of your task when tested against a case.

Pydantic Evals includes several built-in evaluators and allows you to create custom evaluators:

```python {title="simple_eval_evaluator.py"}
from simple_eval_dataset import dataset

from pydantic_evals.evaluators.common import IsInstance  # (1)!
from pydantic_evals.evaluators.context import EvaluatorContext

dataset.add_evaluator(IsInstance(type_name='str'))  # (2)!


async def my_evaluator(ctx: EvaluatorContext[str, str]) -> float:  # (3)!
    if ctx.output == ctx.expected_output:
        return 1.0
    elif (
        isinstance(ctx.output, str)
        and ctx.expected_output.lower() in ctx.output.lower()
    ):
        return 0.8
    else:
        return 0.0


dataset.add_evaluator(my_evaluator)
```
1. Import built-in evaluators, here we import [`IsInstance`][pydantic_evals.evaluators.IsInstance].
2. Add built-in evaluators [`IsInstance`][pydantic_evals.evaluators.IsInstance] to the dataset.
3. Create a custom evaluator function that takes an [`EvaluatorContext`][pydantic_evals.evaluators.context.EvaluatorContext] and returns a simple score.

## Evaluation Process

The evaluation process involves running a task against all cases in a dataset:

Putting the above two examples together and using the more declarative `evaluators` kwarg to [`Dataset`][pydantic_evals.Dataset]:

```python {title="simple_eval_complete.py"}
import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

logfire.configure()

case1 = Case(  # (1)!
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)


class MyEvaluator(Evaluator[str, str]):
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset = Dataset(
    cases=[case1],
    evaluators=[IsInstance(type_name='str'), MyEvaluator()],  # (3)!
)


async def guess_city(question: str) -> str:  # (4)!
    return 'Paris'


report = dataset.evaluate_sync(guess_city)  # (5)!
report.print(include_input=True, include_output=True)  # (6)!
"""
                                    Evaluation Summary: guess_city
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID     ┃ Inputs                         ┃ Outputs ┃ Scores            ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ simple_case │ What is the capital of France? │ Paris   │ MyEvaluator: 1.00 │ ✔          │    123µs │
├─────────────┼────────────────────────────────┼─────────┼───────────────────┼────────────┼──────────┤
│ Averages    │                                │         │ MyEvaluator: 1.00 │ 100.0% ✔   │    123µs │
└─────────────┴────────────────────────────────┴─────────┴───────────────────┴────────────┴──────────┘
"""
```

1. Create a [test case][pydantic_evals.Case] as above
2. Also create a custom evaluator function as above
3. Create a [`Dataset`][pydantic_evals.Dataset] with test cases, also set the [`evaluators`][pydantic_evals.Dataset.evaluators] when creating the dataset
4. Our function to evaluate.
5. Run the evaluation with [`evaluate_sync`][pydantic_evals.Dataset.evaluate_sync], which runs the function against all test cases in the dataset, and returns an [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] object.
6. Print the report with [`print`][pydantic_evals.reporting.EvaluationReport.print], which shows the results of the evaluation, including input and output.

## Evaluation with `LlmJudge`

In this example we evaluate a method for generating recipes based on customer orders.

```python {title="judge_recipes.py"}
from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.format_as_xml import format_as_xml
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance, LlmJudge


class CustomerOrder(BaseModel):  # (1)!
    dish_name: str
    dietary_restriction: str | None = None


class Recipe(BaseModel):
    ingredients: list[str]
    steps: list[str]


recipe_agent = Agent(
    'groq:llama-3.3-70b-versatile',
    result_type=Recipe,
    system_prompt=(
        'Generate a recipe to cook the dish that meets the dietary restrictions.'
    ),
)


async def transform_recipe(customer_order: CustomerOrder) -> Recipe:  # (2)!
    r = await recipe_agent.run(format_as_xml(customer_order))
    return r.data


recipe_dataset = Dataset[CustomerOrder, Recipe, Any](  # (3)!
    cases=[
        Case(
            name='vegetarian_recipe',
            inputs=CustomerOrder(
                dish_name='Spaghetti Bolognese', dietary_restriction='vegetarian'
            ),
            expected_output=None,  # (4)
            metadata={'focus': 'vegetarian'},
            evaluators=(
                LlmJudge(  # (5)!
                    rubric='Recipe should not contain meat or animal products',
                ),
            ),
        ),
        Case(
            name='gluten_free_recipe',
            inputs=CustomerOrder(
                dish_name='Chocolate Cake', dietary_restriction='gluten-free'
            ),
            expected_output=None,
            metadata={'focus': 'gluten-free'},
            # Case-specific evaluator with a focused rubric
            evaluators=(
                LlmJudge(
                    rubric='Recipe should not contain gluten or wheat products',
                ),
            ),
        ),
    ],
    evaluators=[  # (6)!
        IsInstance(type_name='Recipe'),
        LlmJudge(
            rubric='Recipe should have clear steps and relevant ingredients',
            include_input=True,
            model='anthropic:claude-3-7-sonnet-latest',  # (7)!
        ),
    ],
)


report = recipe_dataset.evaluate_sync(transform_recipe)
print(report)
"""
     Evaluation Summary: transform_recipe
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID            ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ vegetarian_recipe  │ ✔✔✔        │    123µs │
├────────────────────┼────────────┼──────────┤
│ gluten_free_recipe │ ✔✔✔        │    123µs │
├────────────────────┼────────────┼──────────┤
│ Averages           │ 100.0% ✔   │    123µs │
└────────────────────┴────────────┴──────────┘
"""
```

1. Define models for our task — Input for recipe generation task and output of the task.
2. Define our recipe generation function - this is the task we want to evaluate.
3. Create a dataset with different test cases and different rubrics.
4. No expected output, we'll let the LLM judge the quality.
5. Case-specific evaluator with a focused rubric using [`LlmJudge`][pydantic_evals.evaluators.LlmJudge].
6. Dataset-level evaluators that apply to all cases, including a general quality rubric for all recipes
7. By default `LlmJudge` uses `openai:gpt-4o`, here we use a specific Anthropic model.

### Generating Test Datasets

Pydantic Evals allows you to generate test datasets using LLMs with [`generate_dataset`][pydantic_evals.examples.generate_dataset]:

```python {title="generate_dataset_example.py"}
from pydantic import BaseModel, Field

from pydantic_evals.examples import generate_dataset


class QuestionInputs(BaseModel):
    """Model for question inputs."""

    question: str = Field(description='A question to answer')
    context: str | None = Field(None, description='Optional context for the question')


class AnswerOutput(BaseModel):
    """Model for expected answer outputs."""

    answer: str = Field(description='The answer to the question')
    confidence: float = Field(description='Confidence level (0-1)', ge=0, le=1)


class MetadataType(BaseModel):
    """Metadata model for test cases."""

    difficulty: str = Field(description='Difficulty level (easy, medium, hard)')
    category: str = Field(description='Question category')


async def xmain():
    dataset = await generate_dataset(
        inputs_type=QuestionInputs,
        output_type=AnswerOutput,
        metadata_type=MetadataType,
        n_examples=2,
        extra_instructions="""
        Generate question-answer pairs about world capitals and landmarks.
        Make sure to include both easy and challenging questions.
        """,
    )
    dataset.to_file('questions_cases.yaml')
```

## Advanced Usage

### Saving and Loading Datasets

Datasets can be saved to and loaded from files (YAML or JSON format):

```python {title="save_load_dataset_example.py"}
from pathlib import Path

from judge_recipes import CustomerOrder, Recipe, recipe_dataset

from pydantic_evals import Dataset

recipe_dataset.to_file('recipe_transform_tests.yaml')
serialized = Path('recipe_transform_tests.yaml').read_text()
print(serialized)
"""
# yaml-language-server: $schema=recipe_transform_tests_schema.json
cases:
- name: vegetarian_recipe
  inputs:
    dish_name: Pasta Bolognese
    dietary_restriction: vegetarian
  metadata:
    focus: vegetarian
  expected_output: null
  evaluators:
  - LlmJudge: Recipe should not contain meat or animal products
- name: gluten_free_recipe
  inputs:
    dish_name: Chocolate Cake
    dietary_restriction: gluten-free
  metadata:
    focus: gluten-free
  expected_output: null
  evaluators:
  - LlmJudge: Recipe should not contain gluten or wheat products
evaluators:
- IsInstance: Recipe
- LlmJudge:
    rubric: Recipe should have clear steps and relevant ingredients
    include_input: true
"""

# Load dataset from file
# In a real scenario, you'd specify the actual file path
loaded_dataset = Dataset[CustomerOrder, Recipe, dict].from_text(serialized)

print(f'Loaded dataset with {len(loaded_dataset.cases)} cases')
#> Loaded dataset with 2 cases

# Clean up
Path('recipe_transform_tests.yaml').unlink()
Path('recipe_transform_tests_schema.json').unlink()
```

### Parallel Evaluation

You can control concurrency during evaluation (this might be useful to prevent exceeding a rate limit):

```python {title="parallel_evaluation_example.py"}
import asyncio
import time

from pydantic_evals import Case, Dataset

# Create a dataset with multiple test cases
dataset = Dataset(
    cases=[
        Case(
            name=f'case_{i}',
            inputs=i,
            expected_output=i * 2,
        )
        for i in range(5)
    ]
)


async def double_number(input_value: int) -> int:
    """Function that simulates work by sleeping for a second before returning double the input."""
    await asyncio.sleep(0.1)  # Simulate work
    return input_value * 2


# Run evaluation with unlimited concurrency
t0 = time.time()
report_default = dataset.evaluate_sync(double_number)
print(f'Evaluation took less than 0.2s: {time.time() - t0 < 0.2}')
#> Evaluation took less than 0.2s: True
report_default.print()
"""
  Evaluation Summary:
     double_number
┏━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━┩
│ case_0   │  101.0ms │
├──────────┼──────────┤
│ case_1   │  101.0ms │
├──────────┼──────────┤
│ case_2   │  101.0ms │
├──────────┼──────────┤
│ case_3   │  101.0ms │
├──────────┼──────────┤
│ case_4   │  101.0ms │
├──────────┼──────────┤
│ Averages │  101.0ms │
└──────────┴──────────┘
"""

# Run evaluation with limited concurrency
t0 = time.time()
report_limited = dataset.evaluate_sync(double_number, max_concurrency=1)
print(f'Evaluation took more than 0.5s: {time.time() - t0 > 0.5}')
#> Evaluation took more than 0.5s: True

report_limited.print()
"""
  Evaluation Summary:
     double_number
┏━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━┩
│ case_0   │  101.0ms │
├──────────┼──────────┤
│ case_1   │  101.0ms │
├──────────┼──────────┤
│ case_2   │  101.0ms │
├──────────┼──────────┤
│ case_3   │  101.0ms │
├──────────┼──────────┤
│ case_4   │  101.0ms │
├──────────┼──────────┤
│ Averages │  101.0ms │
└──────────┴──────────┘
"""
```

### OpenTelemetry Integration

Pydantic Evals integrates with OpenTelemetry for tracing and metrics:

```python {title="opentelemetry_example.py"}
import asyncio
from typing import Any

import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.otel.span_tree import SpanQuery, as_predicate

# Configure logfire for OpenTelemetry integration
logfire.configure()


class SpanTracingEvaluator(Evaluator[str, str]):
    """Evaluator that analyzes the span tree generated during function execution."""

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> dict[str, Any]:
        # Get the span tree from the context
        span_tree = ctx.span_tree
        if span_tree is None:
            return {'has_spans': False, 'performance_score': 0.0}

        # Find all spans with "processing" in the name
        processing_spans = span_tree.find_all(lambda node: 'processing' in node.name)

        # Calculate total processing time
        total_processing_time = sum(
            (span.duration.total_seconds() for span in processing_spans), 0.0
        )

        # Check for error spans
        error_query: SpanQuery = {'name_contains': 'error'}
        has_errors = span_tree.any(as_predicate(error_query))

        # Calculate a performance score (lower is better)
        performance_score = 1.0 if total_processing_time < 0.5 else 0.5

        return {
            'has_spans': True,
            'has_errors': has_errors,
            'performance_score': 0 if has_errors else performance_score,
        }


async def process_text(text: str) -> str:
    """Function that processes text with OpenTelemetry instrumentation."""
    with logfire.span('process_text'):
        # Simulate initial processing
        with logfire.span('text_processing'):
            await asyncio.sleep(0.1)
            processed = text.strip().lower()

        # Simulate additional processing
        with logfire.span('additional_processing'):
            if 'error' in processed:
                with logfire.span('error_handling'):
                    logfire.error(f'Error detected in text: {text}')
                    return f'Error processing: {text}'
            await asyncio.sleep(0.2)
            processed = processed.replace(' ', '_')

        return f'Processed: {processed}'


# Create test cases
dataset = Dataset(
    cases=[
        Case(
            name='normal_text',
            inputs='Hello World',
            expected_output='Processed: hello_world',
        ),
        Case(
            name='text_with_error',
            inputs='Contains error marker',
            expected_output='Error processing: Contains error marker',
        ),
    ],
    evaluators=[SpanTracingEvaluator()],
)

# Run evaluation - spans are automatically captured since logfire is configured
report = dataset.evaluate_sync(process_text)

# Print the report
report.print(include_input=True, include_output=True)
"""
                                                    Evaluation Summary: process_text
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID         ┃ Inputs                ┃ Outputs                                 ┃ Scores                   ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ normal_text     │ Hello World           │ Processed: hello_world                  │ performance_score: 1.00  │ ✔✗         │  101.0ms │
├─────────────────┼───────────────────────┼─────────────────────────────────────────┼──────────────────────────┼────────────┼──────────┤
│ text_with_error │ Contains error marker │ Error processing: Contains error marker │ performance_score: 0     │ ✔✔         │  101.0ms │
├─────────────────┼───────────────────────┼─────────────────────────────────────────┼──────────────────────────┼────────────┼──────────┤
│ Averages        │                       │                                         │ performance_score: 0.500 │ 75.0% ✔    │  101.0ms │
└─────────────────┴───────────────────────┴─────────────────────────────────────────┴──────────────────────────┴────────────┴──────────┘
"""
```
