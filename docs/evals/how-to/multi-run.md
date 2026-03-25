# Multi-Run Evaluation

Run each case multiple times to measure variability and get more reliable aggregate results.

## Overview

AI systems are inherently stochastic — the same input can produce different outputs across runs. The `repeat` parameter lets you run each case multiple times and automatically aggregates the results, giving you a clearer picture of your system's typical behavior.

## Basic Usage

Pass `repeat` to [`evaluate()`][pydantic_evals.dataset.Dataset.evaluate] or [`evaluate_sync()`][pydantic_evals.dataset.Dataset.evaluate_sync]:

```python
from pydantic_evals import Case, Dataset

dataset = Dataset(
    cases=[
        Case(name='greeting', inputs='Say hello'),
        Case(name='farewell', inputs='Say goodbye'),
    ]
)


def task(inputs: str) -> str:
    return inputs.upper()


# Run each case 5 times
report = dataset.evaluate_sync(task, repeat=5)

# 2 cases × 5 repeats = 10 total runs
print(len(report.cases))
#> 10
```

When `repeat > 1`, each run gets an indexed name like `greeting [1/5]`, `greeting [2/5]`, etc., while the original case name is preserved in [`source_case_name`][pydantic_evals.reporting.ReportCase.source_case_name] for grouping.

## Accessing Grouped Results

Use [`case_groups()`][pydantic_evals.reporting.EvaluationReport.case_groups] to access runs organized by original case, with per-group aggregated statistics:

```python
from pydantic_evals import Case, Dataset

dataset = Dataset(
    cases=[
        Case(name='greeting', inputs='Say hello'),
        Case(name='farewell', inputs='Say goodbye'),
    ]
)


def task(inputs: str) -> str:
    return inputs.upper()


report = dataset.evaluate_sync(task, repeat=3)

groups = report.case_groups()
assert groups is not None  # None for single-run (repeat=1)

print(len(groups))
#> 2

group_names = [g.name for g in groups]
print(group_names)
#> ['greeting', 'farewell']

# Each group has 3 runs and aggregated statistics
for group in groups:
    assert len(group.runs) == 3
    assert len(group.failures) == 0
    assert group.summary.task_duration > 0
```

Each [`ReportCaseGroup`][pydantic_evals.reporting.ReportCaseGroup] contains:

- `name` — the original case name
- `runs` — the individual [`ReportCase`][pydantic_evals.reporting.ReportCase] results
- `failures` — any runs that raised exceptions
- `summary` — a [`ReportCaseAggregate`][pydantic_evals.reporting.ReportCaseAggregate] with averaged scores, metrics, labels, assertions, and durations

## Aggregation

With `repeat > 1`, the report's [`averages()`][pydantic_evals.reporting.EvaluationReport.averages] uses a two-level aggregation strategy:

1. **Per-group averages**: Each case's runs are averaged into a group summary
2. **Cross-group averages**: The group summaries are averaged to produce the final result

This ensures each original case contributes equally to the overall averages, regardless of how many runs succeeded or failed.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected

dataset = Dataset(
    cases=[
        Case(name='easy', inputs='hello', expected_output='HELLO'),
        Case(name='hard', inputs='world', expected_output='WORLD'),
    ],
    evaluators=[EqualsExpected()],
)


def task(inputs: str) -> str:
    return inputs.upper()


report = dataset.evaluate_sync(task, repeat=3)

averages = report.averages()
assert averages is not None
print(f'Overall assertion rate: {averages.assertions}')
#> Overall assertion rate: 1.0
```

## Default Behavior

When `repeat=1` (the default), behavior is identical to a standard evaluation — no run indexing, no `source_case_name`, and `case_groups()` returns `None`:

```python
from pydantic_evals import Case, Dataset

dataset = Dataset(cases=[Case(name='test', inputs='hello')])


def task(inputs: str) -> str:
    return inputs.upper()


report = dataset.evaluate_sync(task)  # repeat=1 by default

assert report.case_groups() is None
assert all(c.source_case_name is None for c in report.cases)
```

## Next Steps

- **[Concurrency & Performance](concurrency.md)** — Control parallel execution with `max_concurrency`
- **[Metrics & Attributes](metrics-attributes.md)** — Track custom metrics across runs
- **[Logfire Integration](logfire-integration.md)** — Visualize multi-run results
