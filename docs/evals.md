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

```python {title="llm_judge_example.py"}
import asyncio
from typing import Any

from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output


async def judge_case(inputs: Any, output: Any):
    """Judge the output based on a rubric."""
    rubric = 'The output should be accurate, complete, and relevant to the inputs.'
    return await judge_input_output(inputs, output, rubric)


async def main():
    inputs = 'What is the capital of France?'
    output = 'Paris is the capital of France.'
    result = await judge_case(inputs, output)
    print(f'Judge result: {result}')


if __name__ == '__main__':
    asyncio.run(main())
```

### Generating Test Datasets

Pydantic Evals allows you to generate test datasets using LLMs:

```python {title="generate_dataset_example.py"}
import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from pydantic_evals.examples import generate_dataset


class QuestionInputs(BaseModel):
    """Model for question inputs."""

    question: str = Field(..., description='A question to answer')
    context: Optional[str] = Field(
        None, description='Optional context for the question'
    )


class AnswerOutput(BaseModel):
    """Model for expected answer outputs."""

    answer: str = Field(..., description='The answer to the question')
    confidence: float = Field(..., description='Confidence level (0-1)', ge=0, le=1)


class MetadataType(BaseModel):
    """Metadata model for test cases."""

    difficulty: str = Field(..., description='Difficulty level (easy, medium, hard)')
    category: str = Field(..., description='Question category')


async def main():
    # In a real scenario, you'd typically save this to a file
    dataset = await generate_dataset(
        # path='my_test_cases.yaml',  # Uncomment to save to file
        inputs_type=QuestionInputs,
        output_type=AnswerOutput,
        metadata_type=MetadataType,
        n_examples=2,
        extra_instructions="""
        Generate question-answer pairs about world capitals and landmarks.
        Make sure to include both easy and challenging questions.
        """,
    )

    # For demonstration, print the generated dataset
    for case in dataset.cases:
        print(f'Case: {case.name}')
        print(f'Inputs: {case.inputs}')
        print(f'Expected output: {case.expected_output}')
        print(f'Metadata: {case.metadata}')
        print('---')


if __name__ == '__main__':
    asyncio.run(main())
```

## Advanced Usage

### Saving and Loading Datasets

Datasets can be saved to and loaded from files (YAML or JSON format):

```python {title="save_load_dataset_example.py"}
from typing import List, Optional

from pydantic import BaseModel

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.common import IsInstance


# Define input and output types
class QueryInput(BaseModel):
    query: str
    filters: Optional[List[str]] = None


class QueryResult(BaseModel):
    results: List[str]
    total_count: int


# Create a dataset
dataset = Dataset(
    cases=[
        Case(
            name="basic_search",
            inputs=QueryInput(query="python", filters=["programming"]),
            expected_output=QueryResult(results=["Python tutorial", "Python basics"], total_count=2),
            metadata={"importance": "high"}
        ),
        Case(
            name="filtered_search",
            inputs=QueryInput(query="python", filters=["books"]),
            expected_output=QueryResult(results=["Python cookbook"], total_count=1),
            metadata={"importance": "medium"}
        ),
    ],
    evaluators=[IsInstance(type_name="QueryResult")]
)

# Save dataset to file
# In a real scenario, you'd save to an actual file path
dataset.to_file("search_tests.yaml")
print("Dataset saved to search_tests.yaml")

# Load dataset from file
# In a real scenario, you'd specify the actual file path
loaded_dataset = Dataset[QueryInput, QueryResult, dict].from_text("""
cases:
- name: basic_search
  inputs:
    query: python
    filters:
    - programming
  expected_output:
    results:
    - Python tutorial
    - Python basics
    total_count: 2
  metadata:
    importance: high
- name: filtered_search
  inputs:
    query: python
    filters:
    - books
  expected_output:
    results:
    - Python cookbook
    total_count: 1
  metadata:
    importance: medium
""")

print(f"Loaded dataset with {len(loaded_dataset.cases)} cases")
```

### Parallel Evaluation

You can control concurrency during evaluation:

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
        for i in range(10)
    ]
)


async def double_number(input_value: int) -> int:
    """Function that simulates work by sleeping for a second before returning double the input."""
    await asyncio.sleep(1)  # Simulate work
    return input_value * 2


async def main():
    # Run evaluation with default concurrency (usually based on CPU count)
    start_default = time.time()
    report_default = await dataset.evaluate(double_number)
    duration_default = time.time() - start_default

    # Run evaluation with limited concurrency
    start_limited = time.time()
    report_limited = await dataset.evaluate(double_number, concurrency=2)
    duration_limited = time.time() - start_limited

    # Print results
    print(f'Default concurrency completed in {duration_default:.2f} seconds')
    print(f'Limited concurrency (2) completed in {duration_limited:.2f} seconds')

    # Print summary reports
    print('\nDefault concurrency report:')
    report_default.print()

    print('\nLimited concurrency report:')
    report_limited.print()


if __name__ == '__main__':
    asyncio.run(main())
```

### OpenTelemetry Integration

Pydantic Evals integrates with OpenTelemetry for tracing and metrics:

```python {title="opentelemetry_example.py"}
import asyncio
import logfire
from typing import Dict, Any

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.otel.span_tree import SpanTree, SpanQuery, as_predicate
from pydantic_evals.otel import context_subtree


# Configure logfire for OpenTelemetry integration
logfire.configure()


class SpanTracingEvaluator(Evaluator[str, str]):
    """Evaluator that analyzes the span tree generated during function execution."""
    
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> Dict[str, Any]:
        # Get the span tree from the context
        span_tree = ctx.span_tree
        if span_tree is None:
            return {"has_spans": False, "performance_score": 0.0}
        
        # Find all spans with "processing" in the name
        processing_spans = span_tree.find_all(
            lambda node: "processing" in node.name
        )
        
        # Calculate total processing time
        total_processing_time = sum(
            (span.duration.total_seconds() for span in processing_spans),
            0.0
        )
        
        # Check for error spans
        error_query: SpanQuery = {"name_contains": "error"}
        has_errors = span_tree.any(as_predicate(error_query))
        
        # Calculate a performance score (lower is better)
        performance_score = 1.0 if total_processing_time < 0.5 else 0.5
        
        return {
            "has_spans": True,
            "processing_time": total_processing_time,
            "has_errors": has_errors,
            "performance_score": 0 if has_errors else performance_score
        }


async def process_text(text: str) -> str:
    """Function that processes text with OpenTelemetry instrumentation."""
    with logfire.span("process_text"):
        # Simulate initial processing
        with logfire.span("text_processing"):
            await asyncio.sleep(0.1)
            processed = text.strip().lower()
        
        # Simulate additional processing
        with logfire.span("additional_processing"):
            if "error" in processed:
                with logfire.span("error_handling"):
                    logfire.error(f"Error detected in text: {text}")
                    return f"Error processing: {text}"
            await asyncio.sleep(0.2)
            processed = processed.replace(" ", "_")
        
        return f"Processed: {processed}"


async def main():
    # Create test cases
    dataset = Dataset(
        cases=[
            Case(
                name="normal_text",
                inputs="Hello World",
                expected_output="Processed: hello_world",
            ),
            Case(
                name="text_with_error",
                inputs="Contains error marker",
                expected_output="Error processing: Contains error marker",
            ),
        ],
        evaluators=[SpanTracingEvaluator()]
    )
    
    # Track spans during evaluation
    async def traced_process_text(input_text: str) -> str:
        with context_subtree() as tree:
            result = await process_text(input_text)
        ctx.span_tree = tree  # Store the span tree in the context
        return result
    
    # Run evaluation
    report = await dataset.evaluate(traced_process_text)
    
    # Print the report
    report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Example: Time Range Evaluation

Here's a complete example of using Pydantic Evals for evaluating a function used to select a time range based on a user prompt:

```python {title="time_range_evaluation_example.py"}
import asyncio
import datetime
from typing import Any

from pydantic import BaseModel

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance
from pydantic_evals.evaluators.llm_as_a_judge import judge_input_output


# Define input and output models
class TimeRangeInputs(BaseModel):
    query: str
    context: str | None = None


class TimeRangeResponse(BaseModel):
    start_time: str | None = None
    end_time: str | None = None
    error: str | None = None


# Create a custom evaluator
class TimeRangeEvaluator(Evaluator[TimeRangeInputs, TimeRangeResponse]):
    async def evaluate(
        self, ctx: EvaluatorContext[TimeRangeInputs, TimeRangeResponse]
    ) -> dict[str, Any]:
        rubric = """
        Evaluate the time range based on:
        1. Whether it reasonably matches the user query
        2. Whether the format is valid (ISO format preferred)
        3. Whether start_time is before end_time
        4. Whether an appropriate error is returned when the input is unclear
        """
        result = await judge_input_output(ctx.inputs, ctx.output, rubric)
        
        # Perform additional checks
        valid_format = True
        valid_range = True
        
        if ctx.output.start_time and ctx.output.end_time:
            try:
                start = datetime.datetime.fromisoformat(ctx.output.start_time)
                end = datetime.datetime.fromisoformat(ctx.output.end_time)
                valid_range = start <= end
            except ValueError:
                valid_format = False
        
        return {
            "llm_score": result.score,
            "valid_format": valid_format,
            "valid_range": valid_range,
            "overall_score": result.score * 0.7 + (0.3 if valid_format and valid_range else 0)
        }


# Define function to test
async def infer_time_range(inputs: TimeRangeInputs) -> TimeRangeResponse:
    """Infer a time range from a natural language query."""
    query = inputs.query.lower()
    now = datetime.datetime.now()
    
    if "today" in query:
        start = datetime.datetime(now.year, now.month, now.day, 0, 0, 0)
        end = start + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
        return TimeRangeResponse(
            start_time=start.isoformat(),
            end_time=end.isoformat()
        )
    elif "yesterday" in query:
        start = datetime.datetime(now.year, now.month, now.day, 0, 0, 0) - datetime.timedelta(days=1)
        end = start + datetime.timedelta(days=1) - datetime.timedelta(seconds=1)
        return TimeRangeResponse(
            start_time=start.isoformat(),
            end_time=end.isoformat()
        )
    elif "last week" in query:
        end = datetime.datetime(now.year, now.month, now.day, 0, 0, 0)
        start = end - datetime.timedelta(days=7)
        return TimeRangeResponse(
            start_time=start.isoformat(),
            end_time=end.isoformat()
        )
    else:
        return TimeRangeResponse(
            error="Could not determine time range from query"
        )


async def main():
    # Create test dataset
    dataset = Dataset(
        cases=[
            Case(
                name="today_query",
                inputs=TimeRangeInputs(query="Show me data for today"),
                expected_output=None,  # We'll let the evaluator judge correctness
                metadata={"expected": "today"}
            ),
            Case(
                name="yesterday_query",
                inputs=TimeRangeInputs(query="What happened yesterday?"),
                expected_output=None,
                metadata={"expected": "yesterday"}
            ),
            Case(
                name="last_week_query",
                inputs=TimeRangeInputs(query="Show stats from last week"),
                expected_output=None,
                metadata={"expected": "last_week"}
            ),
            Case(
                name="ambiguous_query",
                inputs=TimeRangeInputs(query="Show me some data"),
                expected_output=TimeRangeResponse(error="Could not determine time range from query"),
                metadata={"expected": "error"}
            ),
        ],
        evaluators=[IsInstance(type_name="TimeRangeResponse"), TimeRangeEvaluator()]
    )
    
    # Run evaluation
    report = await dataset.evaluate(infer_time_range)
    
    # Print the report
    report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    asyncio.run(main())
```
