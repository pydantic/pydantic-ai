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
- [`Dataset`][pydantic_evals.Case]: A collection of test cases designed to evaluate a specific task or function.

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
1. Import built-in evaluators, here we import [`is_instance`][pydantic_evals.evaluators.is_instance].
2. Add built-in evaluators [`is_instance`][pydantic_evals.evaluators.is_instance] to the dataset.
3. Create a custom evaluator function that takes an [`EvaluatorContext`][pydantic_evals.evaluators.context.EvaluatorContext] and returns a simple score.

## Evaluation Process

The evaluation process involves running a task against all cases in a dataset:

Putting the above two examples together and using the more declarative `evaluators` kwarg to `Dataset`:

```python {title="simple_eval_complete.py"}
import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.common import IsInstance
from pydantic_evals.evaluators.context import EvaluatorContext

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
│ simple_case │ What is the capital of France? │ Paris   │ IsInstance: True  │ ✔          │    123µs │
│             │                                │         │ MyEvaluator: 1.00 │            │          │
├─────────────┼────────────────────────────────┼─────────┼───────────────────┼────────────┼──────────┤
│ Averages    │                                │         │ IsInstance: 1.00  │ 100.0% ✔   │    123µs │
│             │                                │         │ MyEvaluator: 1.00 │            │          │
└─────────────┴────────────────────────────────┴─────────┴───────────────────┴────────────┴──────────┘
"""
```

1. Create a [test case][pydantic_evals.Case] as above
2. Also create a custom evaluator function as above
3. Create a [`Dataset`][pydantic_evals.Dataset] with test cases, also set the [`evaluators`][pydantic_evals.Dataset.evaluators] when creating the dataset
4. Our function to evaluate.
5. Run the evaluation with [`evaluate_sync`][pydantic_evals.Dataset.evaluate_sync], which runs the function against all test cases in the dataset, and returns an [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] object.
6. Print the report with [`print`][pydantic_evals.reporting.EvaluationReport.print], which shows the results of the evaluation, including input and output.

## LLM as a Judge

Pydantic Evals integrates seamlessly with LLMs for both evaluation and dataset generation:

You can use LLMs to evaluate the quality of outputs:

```python
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output


async def judge_case(inputs, output):
    """Judge the output based on a rubric."""
    rubric = 'The output should be accurate, complete, and relevant to the inputs.'
    return await judge_input_output(inputs, output, rubric)
```

### Generating Test Datasets

Pydantic Evals allows you to generate test datasets using LLMs:

# TODO: Make the following self-contained, in particular, it needs to define MyInputs and MyOutput
# It should also make use of extra_instructions as appropriate
```python
# from pydantic_evals.examples import generate_dataset
#
# async def main():
#     await generate_dataset(
#         path='my_test_cases.yaml',
#         inputs_type=MyInputs,
#         output_type=MyOutput,
#         metadata_type=dict,
#         n_examples=5
#     )
```

## Advanced Usage

### Saving and Loading Datasets

Datasets can be saved to and loaded from files (YAML or JSON format):

# TODO: Add full, self-contained example


### Parallel Evaluation

You can control concurrency during evaluation:

# TODO: Add full, self-contained example

### OpenTelemetry Integration

Pydantic Evals integrates with OpenTelemetry for tracing and metrics:

# TODO: Add full, self-contained example

## Example: Time Range Evaluation

Here's a complete example of using Pydantic Evals for evaluating a function used to select a time range based on a user prompt:

# TODO: Make the following a full, self-contained example; in particular, it should not depend on external files.
# (It could use the from_text method rather than from_file..)
# It should also actually implement an example function for inferring the time range...
```python
# from typing import Any
#
# from pydantic import BaseModel
#
# from pydantic_evals import Dataset
# from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluatorOutput
# from pydantic_evals.evaluators.common import IsInstance, LlmJudge
# from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output
#
#
# # Define input and output models
# class TimeRangeInputs(BaseModel):
#     query: str
#     context: str | None = None
#
#
# class TimeRangeResponse(BaseModel):
#     start_time: str | None = None
#     end_time: str | None = None
#     error: str | None = None
#
#
# # Load dataset
# dataset = Dataset[TimeRangeInputs, TimeRangeResponse, Any].from_file(
#     'test_cases.yaml', custom_evaluator_types=[IsInstance, LlmJudge]
# )
#
#
# # Create a custom evaluator
# class TimeRangeEvaluator(Evaluator[TimeRangeInputs, TimeRangeResponse]):
#     async def evaluate(
#         self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]
#     ) -> EvaluatorOutput:
#         rubric = 'The output should be a reasonable time range to select for the given inputs.'
#         result = await judge_input_output(ctx.inputs, ctx.output, rubric)
#         return {
#             'is_reasonable': 'yes' if result.pass_ else 'no',
#             'accuracy': result.score,
#         }
#
#
# dataset.add_evaluator(TimeRangeEvaluator())
#
#
# # Define function to test
# async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
#     # Your implementation here
#     pass
#
#
# # Run evaluation
# report = dataset.evaluate_sync(infer_time_range)
# report.print(include_input=True, include_output=True)
```
