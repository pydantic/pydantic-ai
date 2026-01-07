# Core Concepts

This page explains the key concepts in Pydantic Evals and how they work together.

## Overview

Pydantic Evals is built around these core concepts:

- **[`Dataset`][pydantic_evals.dataset.Dataset]** - A static definition containing test cases and evaluators
- **[`Case`][pydantic_evals.dataset.Case]** - A single test scenario with inputs and optional expected outputs
- **[`Evaluator`][pydantic_evals.evaluators.Evaluator]** - Logic for scoring or validating outputs
- **Experiment** - The act of running a task function against all cases in a dataset. (This corresponds to a call to `Dataset.evaluate`.)
- **[`EvaluationReport`][pydantic_evals.reporting.EvaluationReport]** - The results from running an experiment

The key distinction is between:

- **Definition** (`Dataset` with `Case`s and `Evaluator`s) - what you want to test
- **Execution** (Experiment) - running your task against those tests
- **Results** (`EvaluationReport`) - what happened during the experiment

## Unit Testing Analogy

A helpful way to think about Pydantic Evals:

| Unit Testing | Pydantic Evals |
|--------------|----------------|
| Test function | [`Case`][pydantic_evals.dataset.Case] + [`Evaluator`][pydantic_evals.evaluators.Evaluator] |
| Test suite | [`Dataset`][pydantic_evals.dataset.Dataset] |
| Running tests (`pytest`) | **Experiment** (`dataset.evaluate(task)`) |
| Test report | [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] |
| `assert` | Evaluator returning `bool` |

**Key Difference**: AI systems are probabilistic, so instead of simple pass/fail, evaluations can have:

- Quantitative scores (0.0 to 1.0)
- Qualitative labels ("good", "acceptable", "poor")
- Pass/fail assertions with explanatory reasons

Just like you can run `pytest` multiple times on the same test suite, you can run multiple experiments on the same dataset to compare different implementations or track changes over time.

## Dataset

A [`Dataset`][pydantic_evals.dataset.Dataset] is a collection of test cases and evaluators that define an evaluation suite.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import IsInstance

dataset = Dataset(
    name='my_eval_suite',  # Optional name
    cases=[
        Case(inputs='test input', expected_output='test output'),
    ],
    evaluators=[
        IsInstance(type_name='str'),
    ],
)
```

### Key Features

- **Type-safe**: Generic over `InputsT`, `OutputT`, and `MetadataT` types
- **Serializable**: Can be saved to/loaded from YAML or JSON files
- **Evaluable**: Run against any function with matching input/output types

### `Dataset`-Level vs `Case`-Level Evaluators

Evaluators can be defined at two levels:

- **`Dataset`-level**: Apply to all cases in the dataset
- **`Case`-level**: Apply only to specific cases

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, IsInstance

dataset = Dataset(
    cases=[
        Case(
            name='special_case',
            inputs='test',
            expected_output='TEST',
            evaluators=[
                # This evaluator only runs for this case
                EqualsExpected(),
            ],
        ),
    ],
    evaluators=[
        # This evaluator runs for ALL cases
        IsInstance(type_name='str'),
    ],
)
```

## Experiments

An **Experiment** is what happens when you execute a task function against all cases in a dataset. This is the bridge between your static test definition (the Dataset) and your results (the EvaluationReport).

### Running an Experiment

You run an experiment by calling [`evaluate()`][pydantic_evals.dataset.Dataset.evaluate] or [`evaluate_sync()`][pydantic_evals.dataset.Dataset.evaluate_sync] on a dataset:

```python
from pydantic_evals import Case, Dataset

# Define your dataset (static definition)
dataset = Dataset(
    cases=[
        Case(inputs='hello', expected_output='HELLO'),
        Case(inputs='world', expected_output='WORLD'),
    ],
)

# Define your task
def uppercase_task(text: str) -> str:
    return text.upper()

# Run the experiment (execution)
report = dataset.evaluate_sync(uppercase_task)
```

### What Happens During an Experiment

When you run an experiment:

1. **Setup**: The dataset loads all cases and evaluators
2. **Execution**: For each case:
    1. The task function is called with `case.inputs`
    2. Execution time is measured and OpenTelemetry spans are captured (if `logfire` is configured)
    3. The outputs of the task function for each case are recorded
3. **Evaluation**: For each case output:
    1. All dataset-level evaluators are run
    2. Case-specific evaluators are run (if any)
    3. Results are collected (scores, assertions, labels)
4. **Reporting**: All results are aggregated into an [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport]

### Multiple Experiments from One Dataset

A key feature of Pydantic Evals is that you can run the same dataset against different task implementations:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    cases=[
        Case(inputs='hello', expected_output='HELLO'),
    ],
    evaluators=[EqualsExpected()],
)


# Original implementation
def task_v1(text: str) -> str:
    return text.upper()


# Improved implementation (with exclamation)
def task_v2(text: str) -> str:
    return text.upper() + '!'


# Compare results
report_v1 = dataset.evaluate_sync(task_v1)
report_v2 = dataset.evaluate_sync(task_v2)

avg_v1 = report_v1.averages()
avg_v2 = report_v2.averages()
print(f'V1 pass rate: {avg_v1.assertions if avg_v1 and avg_v1.assertions else 0}')
#> V1 pass rate: 1.0
print(f'V2 pass rate: {avg_v2.assertions if avg_v2 and avg_v2.assertions else 0}')
#> V2 pass rate: 0
```

This allows you to:

- **Compare implementations** across versions
- **Track performance** over time
- **A/B test** different approaches
- **Validate changes** before deployment

## Case

A [`Case`][pydantic_evals.dataset.Case] represents a single test scenario with specific inputs and optional expected outputs.

```python
from pydantic_evals import Case
from pydantic_evals.evaluators import EqualsExpected

case = Case(
    name='test_uppercase',  # Optional, but recommended for reporting
    inputs='hello world',  # Required: inputs to your task
    expected_output='HELLO WORLD',  # Optional: expected output
    metadata={'category': 'basic'},  # Optional: arbitrary metadata
    evaluators=[EqualsExpected()],  # Optional: case-specific evaluators
)
```

### Case Components

#### Inputs
The inputs to pass to the task being evaluated. Can be any type:

```python
from pydantic import BaseModel

from pydantic_evals import Case


class MyInputModel(BaseModel):
    field1: str


# Simple types
Case(inputs='hello')
Case(inputs=42)

# Complex types
Case(inputs={'query': 'What is AI?', 'max_tokens': 100})
Case(inputs=MyInputModel(field1='value'))
```

#### Expected Output
The expected result, used by evaluators like [`EqualsExpected`][pydantic_evals.evaluators.EqualsExpected]:

```python
from pydantic_evals import Case

Case(
    inputs='2 + 2',
    expected_output='4',
)
```

If no `expected_output` is provided, evaluators that require it (like `EqualsExpected`) will skip that case.

#### Metadata
Arbitrary data that evaluators can access via [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext]:

```python
from pydantic_evals import Case

Case(
    inputs='question',
    metadata={
        'difficulty': 'hard',
        'category': 'math',
        'source': 'exam_2024',
    },
)
```

Metadata is useful for:

- Filtering cases during analysis
- Providing context to evaluators
- Organizing test suites

#### Evaluators

Cases can have their own evaluators that only run for that specific case. This is particularly powerful for building comprehensive evaluation suites where different cases have different requirements - if you could write one evaluator rubric that worked perfectly for all cases, you'd just incorporate it into your agent instructions. Case-specific [`LLMJudge`][pydantic_evals.evaluators.LLMJudge] evaluators are especially useful for quickly building maintainable golden datasets by describing what "good" looks like for each scenario. See [Case-specific evaluators](evaluators/overview.md#case-specific-evaluators) for a more detailed explanation and examples.

## Evaluator

An [`Evaluator`][pydantic_evals.evaluators.Evaluator] assesses the output of your task and returns one or more scores, labels, or assertions. Each score, label or assertion can also have an optional string-value reason associated.

### Evaluator Types

Evaluators return different types of results:

| Return Type | Purpose | Example |
|-------------|---------|---------|
| `bool` | **Assertion** - Pass/fail check | `True` → ✔, `False` → ✗ |
| `int` or `float` | **Score** - Numeric quality metric | `0.95`, `87` |
| `str` | **Label** - Categorical result | `"correct"`, `"hallucination"` |

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ExactMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output  # Assertion


@dataclass
class Confidence(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        # Analyze output and return confidence score
        return 0.95  # Score


@dataclass
class Classifier(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> str:
        if 'error' in ctx.output.lower():
            return 'error'  # Label
        return 'success'
```

Evaluators can also return instances of [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason], and dictionaries mapping labels to output values.
See the [custom evaluator return types](evaluators/custom.md#return-types) docs for more detail.

### EvaluatorContext

All evaluators receive an [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] containing:

- `name`: Case name (optional)
- `inputs`: Task inputs
- `metadata`: Case metadata (optional)
- `expected_output`: Expected output (optional)
- `output`: Actual output from task
- `duration`: Task execution time in seconds
- `span_tree`: OpenTelemetry spans (if `logfire` is configured)
- `attributes`: Custom attributes dict
- `metrics`: Custom metrics dict

### Multiple Evaluations

Evaluators can return multiple results by returning a dictionary:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class MultiCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool | float | str]:
        return {
            'is_valid': isinstance(ctx.output, str),  # Assertion
            'length': len(ctx.output),  # Metric
            'category': 'long' if len(ctx.output) > 100 else 'short',  # Label
        }
```

### Evaluation Reasons

Add explanations to your evaluations using [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason]:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


@dataclass
class SmartCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        if ctx.output == ctx.expected_output:
            return EvaluationReason(
                value=True,
                reason='Exact match with expected output',
            )
        return EvaluationReason(
            value=False,
            reason=f'Expected {ctx.expected_output!r}, got {ctx.output!r}',
        )
```

Reasons appear in reports when using `include_reasons=True`.

## Evaluation Report

An [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] is the result of running an experiment. It contains all the data from executing your task against the dataset's cases and running all evaluators.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    cases=[Case(inputs='hello', expected_output='HELLO')],
    evaluators=[EqualsExpected()],
)


def my_task(text: str) -> str:
    return text.upper()


# Run an experiment
report = dataset.evaluate_sync(my_task)

# Print to console
report.print()
"""
    Evaluation Summary: my_task
┏━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID  ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Case 1   │ ✔          │     10ms │
├──────────┼────────────┼──────────┤
│ Averages │ 100.0% ✔   │     10ms │
└──────────┴────────────┴──────────┘
"""

# Access data programmatically
for case in report.cases:
    print(f'{case.name}: {case.scores}')
    #> Case 1: {}
```

### Report Structure

The [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] contains:

- `name`: Experiment name
- `cases`: List of successful case evaluations
- `failures`: List of failed executions
- `trace_id`: OpenTelemetry trace ID (optional)
- `span_id`: OpenTelemetry span ID (optional)

### ReportCase

Each successfulcase result contains:

**Case data:**

- `name`: Case name
- `inputs`: Task inputs
- `metadata`: Case metadata (optional)
- `expected_output`: Expected output (optional)
- `output`: Actual output from task

**Evaluation results:**

- `scores`: Dictionary of numeric scores from evaluators
- `labels`: Dictionary of categorical labels from evaluators
- `assertions`: Dictionary of pass/fail assertions from evaluators

**Performance data:**

- `task_duration`: Task execution time
- `total_duration`: Total time including evaluators

**Additional data:**

- `metrics`: Custom metrics dict
- `attributes`: Custom attributes dict

**Tracing:**

- `trace_id`: OpenTelemetry trace ID (optional)
- `span_id`: OpenTelemetry span ID (optional)

**Errors:**

- `evaluator_failures`: List of evaluator errors

## Data Model Relationships

Here's how the core concepts relate to each other:

### Static Definition

- A **Dataset** contains:
  - Many **Cases** (test scenarios with inputs and expected outputs)
  - Many **Evaluators** (logic for scoring outputs)

### Execution (Experiment)

When you call `dataset.evaluate(task)`, an **Experiment** runs:

- The **Task** function is executed against all **Cases** in the **Dataset**
- All **Evaluators** are run (both dataset-level and case-specific) against each output as appropriate
- One **EvaluationReport** is produced as the final output

### Results

- An **EvaluationReport** contains:
  - Results for each **Case** (inputs, outputs, scores, assertions, labels)
  - Summary statistics (averages, pass rates)
  - Performance data (durations)
  - Tracing information (OpenTelemetry spans)

### Key Relationships

- **One Dataset → Many Experiments**: You can run the same dataset against different task implementations or multiple times to track changes
- **One Experiment → One Report**: Each time you call `dataset.evaluate(...)`, you get one report
- **One Experiment → Many Case Results**: The report contains results for every case in the dataset


## Next Steps

- **[Evaluators Overview](evaluators/overview.md)** - When to use different evaluator types
- **[Built-in Evaluators](evaluators/built-in.md)** - Complete reference of provided evaluators
- **[Custom Evaluators](evaluators/custom.md)** - Write your own evaluation logic
- **[Dataset Management](how-to/dataset-management.md)** - Save, load, and generate datasets
