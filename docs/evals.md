---
title: Pydantic Evals
---

# Pydantic Evals

**Pydantic Evals** is a powerful evaluation framework for systematically testing and evaluating AI systems, from simple LLM calls to complex multi-agent applications.

## Design Philosophy

!!! note "Code-First Approach"
    Pydantic Evals follows a code-first philosophy where all evaluation components are defined in Python. This differs from platforms with web-based configuration. You write and run evals in code, and can write the results to disk or view them in your terminal or in [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/).

!!! danger "Evals are an Emerging Practice"
    Unlike unit tests, evals are an emerging art/science. Anyone who claims to know exactly how your evals should be defined can safely be ignored. We've designed Pydantic Evals to be flexible and useful without being too opinionated.


## Quick Navigation

**Getting Started:**

- [Installation](#installation)
- [Quick Start](evals/quick-start.md)
- [Core Concepts](evals/core-concepts.md)

**Evaluators:**

- [Evaluators Overview](evals/evaluators/overview.md) - Compare evaluator types and learn when to use each approach
- [Built-in Evaluators](evals/evaluators/built-in.md) - Complete reference for exact match, instance checks, and other ready-to-use evaluators
- [LLM as a Judge](evals/evaluators/llm-judge.md) - Use LLMs to evaluate subjective qualities, complex criteria, and natural language outputs
- [Custom Evaluators](evals/evaluators/custom.md) - Implement domain-specific scoring logic and custom evaluation metrics
- [Span-Based Evaluation](evals/evaluators/span-based.md) - Evaluate internal agent behavior (tool calls, execution flow) using OpenTelemetry traces. Essential for complex agents where correctness depends on _how_ the answer was reached, not just the final output. Also ensures eval assertions align with production telemetry.

**How-To Guides:**

- [Logfire Integration](evals/how-to/logfire-integration.md) - Visualize results
- [Dataset Management](evals/how-to/dataset-management.md) - Save, load, generate
- [Concurrency & Performance](evals/how-to/concurrency.md) - Control parallel execution
- [Retry Strategies](evals/how-to/retry-strategies.md) - Handle transient failures
- [Metrics & Attributes](evals/how-to/metrics-attributes.md) - Track custom data

**Examples:**

- [Simple Validation](evals/examples/simple-validation.md) - Basic example

**Reference:**

- [API Documentation](api/pydantic_evals/dataset.md)

## Code-First Evaluation

Pydantic Evals follows a **code-first approach** where you define all evaluation components (datasets, experiments, tasks, cases and evaluators) in Python code, or as serialized data loaded by Python code. This differs from platforms with fully web-based configuration.

When you run an _Experiment_ you'll see a progress indicator and can print the results wherever you run your python code (IDE, terminal, etc). You also get a report object back that you can serialize and store or send to a notebook or other application for further visualization and analysis.

If you are using [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/), your experiment results automatically appear in the Logfire web interface for visualization, comparison, and collaborative analysis. Logfire serves as a observability layer - you write and run evals in code, then view and analyze results in the web UI.

## Installation

To install the Pydantic Evals package, run:

```bash
pip/uv-add pydantic-evals
```

`pydantic-evals` does not depend on `pydantic-ai`, but has an optional dependency on `logfire` if you'd like to
use OpenTelemetry traces in your evals, or send evaluation results to [logfire](https://pydantic.dev/logfire).

```bash
pip/uv-add 'pydantic-evals[logfire]'
```

## Pydantic Evals Data Model

Pydantic Evals is built around a simple data model:

### Data Model Diagram

```
Dataset (1) ──────────── (Many) Case
│                        │
│                        │
└─── (Many) Experiment ──┴─── (Many) Case results
     │
     └─── (1) Task
     │
     └─── (Many) Evaluator
```

### Key Relationships

1. **Dataset → Cases**: One Dataset contains many Cases
2. **Dataset → Experiments**: One Dataset can be used across many Experiments over time
3. **Experiment → Case results**: One Experiment generates results by executing each Case
4. **Experiment → Task**: One Experiment evaluates one defined Task
5. **Experiment → Evaluators**: One Experiment uses multiple Evaluators. Dataset-wide Evaluators are run against all Cases, and Case-specific Evaluators against their respective Cases

### Data Flow

1. **Dataset creation**: Define cases and evaluators in YAML/JSON, or directly in Python
2. **Experiment execution**: Run `dataset.evaluate_sync(task_function)`
3. **Cases run**: Each Case is executed against the Task
4. **Evaluation**: Evaluators score the Task outputs for each Case
5. **Results**: All Case results are collected into a summary report

!!! note "A metaphor"

    A useful metaphor (although not perfect) is to think of evals like a **Unit Testing** framework:

    - **Cases + Evaluators** are your individual unit tests - each one
    defines a specific scenario you want to test, complete with inputs
    and expected outcomes. Just like a unit test, a case asks: _"Given
    this input, does my system produce the right output?"_

    -  **Datasets** are like test suites - they are the scaffolding that holds your unit
    tests together. They group related cases and define shared
    evaluation criteria that should apply across all tests in the suite.

    - **Experiments** are like running your entire test suite and getting a
    report. When you execute `dataset.evaluate_sync(my_ai_function)`,
    you're running all your cases against your AI system and
    collecting the results - just like running `pytest` and getting a
    summary of passes, failures, and performance metrics.

    The key difference from traditional unit testing is that AI systems are
    probabilistic. If you're type checking you'll still get a simple pass/fail,
    but scores for text outputs are likely qualitative and/or categorical,
    and more open to interpretation.

For a deeper understanding, see [Core Concepts](evals/core-concepts.md).

## Datasets and Cases

In Pydantic Evals, everything begins with [`Dataset`][pydantic_evals.dataset.Dataset]s and [`Case`][pydantic_evals.dataset.Case]s:

- **[`Dataset`][pydantic_evals.dataset.Dataset]**: A collection of test Cases designed for the evaluation of a specific task or function
- **[`Case`][pydantic_evals.dataset.Case]**: A single test scenario corresponding to Task inputs, with optional expected outputs, metadata, and case-specific evaluators

```python {title="simple_eval_dataset.py"}
from pydantic_evals import Case, Dataset

case1 = Case(
    name='simple_case',
    inputs='What is the capital of France?',
    expected_output='Paris',
    metadata={'difficulty': 'easy'},
)

dataset = Dataset(cases=[case1])
```

_(This example is complete, it can be run "as is")_

See [Dataset Management](evals/how-to/dataset-management.md) to learn about saving, loading, and generating datasets.

## Evaluators

[`Evaluator`][pydantic_evals.evaluators.Evaluator]s analyze and score the results of your Task when tested against a Case.

These can be deterministic, code-based checks (such as testing model output format with a regex, or checking for the appearance of PII or sensitive data), or they can assess non-deterministic model outputs for qualities like accuracy, precision/recall, hallucinations, or instruction-following.

While both kinds of testing are useful in LLM systems, classical code-based tests are cheaper and easier than tests which require either human or machine review of model outputs.

Pydantic Evals includes several [built-in evaluators](evals/evaluators/built-in.md) and allows you to define [custom evaluators](evals/evaluators/custom.md):

```python {title="simple_eval_evaluator.py" requires="simple_eval_dataset.py"}
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance

from simple_eval_dataset import dataset

dataset.add_evaluator(IsInstance(type_name='str'))  # (1)!


@dataclass
class MyEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:  # (2)!
        if ctx.output == ctx.expected_output:
            return 1.0
        elif (
            isinstance(ctx.output, str)
            and ctx.expected_output.lower() in ctx.output.lower()
        ):
            return 0.8
        else:
            return 0.0


dataset.add_evaluator(MyEvaluator())
```

1. You can add built-in evaluators to a dataset using the [`add_evaluator`][pydantic_evals.dataset.Dataset.add_evaluator] method.
2. This custom evaluator returns a simple score based on whether the output matches the expected output.

_(This example is complete, it can be run "as is")_

Learn more:

- [Evaluators Overview](evals/evaluators/overview.md) - When to use different types
- [Built-in Evaluators](evals/evaluators/built-in.md) - Complete reference
- [LLM Judge](evals/evaluators/llm-judge.md) - Using LLMs as evaluators
- [Custom Evaluators](evals/evaluators/custom.md) - Write your own logic
- [Span-Based Evaluation](evals/evaluators/span-based.md) - Analyze execution traces

## Running Experiments

Performing evaluations involves running a task against all cases in a dataset, also known as running an "experiment".

Putting the above two examples together and using the more declarative `evaluators` kwarg to [`Dataset`][pydantic_evals.dataset.Dataset]:

```python {title="simple_eval_complete.py"}
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

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
    evaluators=[IsInstance(type_name='str'), MyEvaluator()],  # (2)!
)


async def guess_city(question: str) -> str:  # (3)!
    return 'Paris'


report = dataset.evaluate_sync(guess_city)  # (4)!
report.print(include_input=True, include_output=True, include_durations=False)  # (5)!
"""
                              Evaluation Summary: guess_city
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Case ID     ┃ Inputs                         ┃ Outputs ┃ Scores            ┃ Assertions ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ simple_case │ What is the capital of France? │ Paris   │ MyEvaluator: 1.00 │ ✔          │
├─────────────┼────────────────────────────────┼─────────┼───────────────────┼────────────┤
│ Averages    │                                │         │ MyEvaluator: 1.00 │ 100.0% ✔   │
└─────────────┴────────────────────────────────┴─────────┴───────────────────┴────────────┘
"""
```

1. Create a [test case][pydantic_evals.dataset.Case] as above
2. Create a [`Dataset`][pydantic_evals.dataset.Dataset] with test cases and [`evaluators`][pydantic_evals.dataset.Dataset.evaluators]
3. Our function to evaluate.
4. Run the evaluation with [`evaluate_sync`][pydantic_evals.dataset.Dataset.evaluate_sync], which runs the function against all test cases in the dataset, and returns an [`EvaluationReport`][pydantic_evals.reporting.EvaluationReport] object.
5. Print the report with [`print`][pydantic_evals.reporting.EvaluationReport.print], which shows the results of the evaluation. We have omitted duration here just to keep the printed output from changing from run to run.

_(This example is complete, it can be run "as is")_

See [Quick Start](evals/quick-start.md) for more examples and [Concurrency & Performance](evals/how-to/concurrency.md) to learn about controlling parallel execution.

## API Reference

For comprehensive coverage of all classes, methods, and configuration options, see the detailed [API Reference documentation](https://ai.pydantic.dev/api/pydantic_evals/dataset/).

## Next Steps

<!-- TODO - this would be the perfect place for a full tutorial or case study  -->
1. **Start with simple evaluations** using [Quick Start](evals/quick-start.md)
2. **Understand the data model** with [Core Concepts](evals/core-concepts.md)
3. **Explore built-in evaluators** in [Built-in Evaluators](evals/evaluators/built-in.md)
4. **Integrate with Logfire** for visualization: [Logfire Integration](evals/how-to/logfire-integration.md)
5. **Build comprehensive test suites** with [Dataset Management](evals/how-to/dataset-management.md)
6. **Implement custom evaluators** for domain-specific metrics: [Custom Evaluators](evals/evaluators/custom.md)
