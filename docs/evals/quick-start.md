# Pydantic Evals

**Pydantic Evals** is a powerful evaluation framework for systematically testing and evaluating AI systems, from simple LLM calls to complex multi-agent applications.

## What is Pydantic Evals?

Pydantic Evals helps you:

- **Create test datasets** with type-safe structured inputs and expected outputs
- **Run evaluations** against your AI systems with automatic concurrency
- **Score results** using deterministic checks, LLM judges, or custom evaluators
- **Generate reports** with detailed metrics, assertions, and performance data
- **Track changes** by comparing evaluation runs over time
- **Integrate with Logfire** for visualization and collaborative analysis

## Installation

```bash
pip install pydantic-evals
```

For OpenTelemetry tracing and Logfire integration:

```bash
pip install 'pydantic-evals[logfire]'
```

## Quick Start

While evaluations are typically used to test AI systems, the Pydantic Evals framework works with any function call. To demonstrate the core functionality, we'll start with a simple, deterministic example.

Here's a complete example of evaluating a simple text transformation function:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected

# Create a dataset with test cases
dataset = Dataset(
    cases=[
        Case(
            name='uppercase_basic',
            inputs='hello world',
            expected_output='HELLO WORLD',
        ),
        Case(
            name='uppercase_with_numbers',
            inputs='hello 123',
            expected_output='HELLO 123',
        ),
    ],
    evaluators=[
        EqualsExpected(),  # Check exact match with expected_output
        Contains(value='HELLO', case_sensitive=True),  # Check contains "HELLO"
    ],
)


# Define the function to evaluate
def uppercase_text(text: str) -> str:
    return text.upper()


# Run the evaluation
report = dataset.evaluate_sync(uppercase_text)

# Print the results
report.print()
"""
        Evaluation Summary: uppercase_text
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID                ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ uppercase_basic        │ ✔✔         │     10ms │
├────────────────────────┼────────────┼──────────┤
│ uppercase_with_numbers │ ✔✔         │     10ms │
├────────────────────────┼────────────┼──────────┤
│ Averages               │ 100.0% ✔   │     10ms │
└────────────────────────┴────────────┴──────────┘
"""
```

Output:

```
                  Evaluation Summary: uppercase_text
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Case ID                 ┃ Assertions ┃ Duration ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩
│ uppercase_basic         │ ✔✔         │     10ms │
├─────────────────────────┼────────────┼──────────┤
│ uppercase_with_numbers  │ ✔✔         │     10ms │
├─────────────────────────┼────────────┼──────────┤
│ Averages                │ 100.0% ✔   │     10ms │
└─────────────────────────┴────────────┴──────────┘
```

## Key Concepts

Understanding a few core concepts will help you get the most out of Pydantic Evals:

- **[`Dataset`][pydantic_evals.Dataset]** - A collection of test cases and (optional) evaluators
- **[`Case`][pydantic_evals.Case]** - A single test scenario with inputs and optional expected outputs and case-specific evaluators
- **[`Evaluator`][pydantic_evals.evaluators.Evaluator]** - A function that scores or validates task outputs
- **[`EvaluationReport`][pydantic_evals.reporting.EvaluationReport]** - Results from running an evaluation

For a deeper dive, see [Core Concepts](core-concepts.md).

## Common Use Cases

### Deterministic Validation

Test that your AI system produces correctly-structured outputs:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, IsInstance

dataset = Dataset(
    cases=[
        Case(inputs={'data': 'required_key present'}, expected_output={'result': 'success'}),
    ],
    evaluators=[
        IsInstance(type_name='dict'),
        Contains(value='required_key'),
    ],
)
```

### LLM-as-a-Judge Evaluation

Use an LLM to evaluate subjective qualities like accuracy or helpfulness:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

dataset = Dataset(
    cases=[
        Case(inputs='What is the capital of France?', expected_output='Paris'),
    ],
    evaluators=[
        LLMJudge(
            rubric='Response is accurate and helpful',
            include_input=True,
            model='anthropic:claude-sonnet-4-5',
        )
    ],
)
```

### Performance Testing

Ensure your system meets performance requirements:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import MaxDuration

dataset = Dataset(
    cases=[
        Case(inputs='test input', expected_output='test output'),
    ],
    evaluators=[
        MaxDuration(seconds=2.0),
    ],
)
```

## Next Steps

Explore the documentation to learn more:

- **[Core Concepts](core-concepts.md)** - Understand the data model and evaluation flow
- **[Built-in Evaluators](evaluators/built-in.md)** - Learn about all available evaluators
- **[Custom Evaluators](evaluators/custom.md)** - Write your own evaluation logic
- **[Dataset Management](how-to/dataset-management.md)** - Save, load, and generate datasets
- **[Examples](examples/simple-validation.md)** - Practical examples for common scenarios
