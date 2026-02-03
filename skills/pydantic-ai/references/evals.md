# Evals Reference

Source: `pydantic_evals/`

Pydantic Evals is a code-first evaluation framework for testing AI systems.

## Installation

```bash
pip/uv-add pydantic-evals

# With Logfire integration
pip/uv-add 'pydantic-evals[logfire]'
```

## Basic Evaluation

```python {title="simple_eval_complete.py"}
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance

case1 = Case(
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
    evaluators=[IsInstance(type_name='str'), MyEvaluator()],
)


async def guess_city(question: str) -> str:
    return 'Paris'


report = dataset.evaluate_sync(guess_city)
report.print(include_input=True, include_output=True, include_durations=False)
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

## Data Model

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

## Dataset and Cases

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

Datasets can be saved/loaded from YAML/JSON:

```python
# Save
dataset.to_file('dataset.yaml')

# Load
dataset = Dataset[str, str].from_file('dataset.yaml')
```

## Evaluators

### Built-in Evaluators

```python {title="simple_eval_evaluator.py" requires="simple_eval_dataset.py"}
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.evaluators.common import IsInstance

from simple_eval_dataset import dataset

dataset.add_evaluator(IsInstance(type_name='str'))


@dataclass
class MyEvaluator(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
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

Common built-ins:
- `IsInstance(type_name='str')` — Check output type
- `Equals()` — Exact match
- `Contains(substring='...')` — Substring check

### Custom Evaluators

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class LengthEvaluator(Evaluator[str, str]):
    max_length: int = 100

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if len(ctx.output) <= self.max_length:
            return 1.0
        return max(0.0, 1.0 - (len(ctx.output) - self.max_length) / 100)
```

### LLM Judge Evaluator

Use an LLM to evaluate outputs:

```python
from pydantic_ai import Agent
from pydantic_evals.evaluators import LLMJudge

judge_agent = Agent('openai:gpt-4o')

evaluator = LLMJudge(
    agent=judge_agent,
    rubric='Rate the response accuracy from 0-1. 1 = perfect, 0 = wrong.',
)
dataset.add_evaluator(evaluator)
```

## Running Experiments

```python
# Async
report = await dataset.evaluate(my_function)

# Sync
report = dataset.evaluate_sync(my_function)

# Print results
report.print()
report.print(include_input=True, include_output=True)

# Access data
for case_result in report.case_results:
    print(case_result.name, case_result.scores)
```

## EvaluatorContext Fields

```python
ctx.inputs           # Case inputs
ctx.output           # Task output
ctx.expected_output  # Expected output (if defined)
ctx.metadata         # Case metadata dict
ctx.duration         # Task execution time
```

## Logfire Integration

View results in Logfire web UI:

```python
import logfire
from pydantic_evals import Dataset

logfire.configure()

dataset = Dataset(cases=[...], evaluators=[...])
report = dataset.evaluate_sync(my_function)
# Results automatically appear in Logfire
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Dataset` | `pydantic_evals.Dataset` | Collection of test cases |
| `Case` | `pydantic_evals.Case` | Single test scenario |
| `Evaluator` | `pydantic_evals.evaluators.Evaluator` | Base evaluator class |
| `EvaluatorContext` | `pydantic_evals.evaluators.EvaluatorContext` | Context passed to evaluators |
| `EvaluationReport` | `pydantic_evals.reporting.EvaluationReport` | Experiment results |
| `IsInstance` | `pydantic_evals.evaluators.IsInstance` | Type check evaluator |
| `LLMJudge` | `pydantic_evals.evaluators.LLMJudge` | LLM-based evaluator |
