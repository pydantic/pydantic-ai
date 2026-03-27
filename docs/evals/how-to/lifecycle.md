# Case Lifecycle Hooks

Control per-case setup, context preparation, and teardown during evaluation using [`CaseLifecycle`][pydantic_evals.lifecycle.CaseLifecycle].

## Overview

[`CaseLifecycle`][pydantic_evals.lifecycle.CaseLifecycle] provides hooks at each stage of case evaluation. You pass a lifecycle **class** (not an instance) to [`Dataset.evaluate`][pydantic_evals.dataset.Dataset.evaluate], and a new instance is created for each case, so instance attributes naturally hold case-specific state.

## Evaluation Flow

Each case follows this flow:

1. **`setup()`** — called before task execution
2. **Task runs**
3. **`prepare_context()`** — called after task, before evaluators
4. **Evaluators run**
5. **`teardown()`** — called after evaluators complete

## Per-Case Setup and Teardown

Use `setup()` and `teardown()` when each case needs its own environment — for example, creating a database, starting a service, or preparing fixtures driven by case metadata. Since a new lifecycle instance is created for each case, instance attributes are naturally case-scoped:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import ReportCase, ReportCaseFailure


class SetupFromMetadata(CaseLifecycle[str, str, dict]):
    async def setup(self) -> None:
        prefix = (self.case.metadata or {}).get('prefix', '')
        self.prefix = prefix

    async def prepare_context(
        self, ctx: EvaluatorContext[str, str, dict]
    ) -> EvaluatorContext[str, str, dict]:
        ctx.metrics['prefix_length'] = len(self.prefix)
        return ctx

    async def teardown(
        self,
        result: ReportCase[str, str, dict] | ReportCaseFailure[str, str, dict],
    ) -> None:
        pass  # Clean up resources here


dataset = Dataset(
    cases=[
        Case(name='no_prefix', inputs='hello', metadata={'prefix': ''}),
        Case(name='with_prefix', inputs='hello', metadata={'prefix': 'PREFIX:'}),
    ]
)

report = dataset.evaluate_sync(lambda inputs: inputs.upper(), lifecycle=SetupFromMetadata)

metrics = {c.name: c.metrics for c in report.cases}
print(metrics['no_prefix']['prefix_length'])
#> 0
print(metrics['with_prefix']['prefix_length'])
#> 7
```

The case metadata drives per-case behavior without needing custom [`Case`][pydantic_evals.dataset.Case] subclasses or serialization.

### Conditional Teardown

The `teardown()` hook receives the full result, so you can vary cleanup logic based on success or failure — for example, keeping test environments up for manual inspection when a case fails:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import ReportCase, ReportCaseFailure

cleaned_up: list[str] = []


class ConditionalCleanup(CaseLifecycle[str, str, dict]):
    async def setup(self) -> None:
        self.resource_id = self.case.name

    async def teardown(
        self,
        result: ReportCase[str, str, dict] | ReportCaseFailure[str, str, dict],
    ) -> None:
        keep_on_failure = (self.case.metadata or {}).get('keep_on_failure', False)
        if isinstance(result, ReportCaseFailure) and keep_on_failure:
            pass  # Keep resource for inspection
        else:
            cleaned_up.append(self.resource_id)


dataset = Dataset(
    cases=[
        Case(name='success_case', inputs='hello', metadata={'keep_on_failure': True}),
        Case(name='failure_case', inputs='fail', metadata={'keep_on_failure': True}),
    ]
)


def task(inputs: str) -> str:
    if inputs == 'fail':
        raise ValueError('intentional failure')
    return inputs.upper()


report = dataset.evaluate_sync(task, max_concurrency=1, lifecycle=ConditionalCleanup)

print(cleaned_up)
#> ['success_case']
```

## Preparing Evaluator Context

The `prepare_context()` hook runs after the task completes but before evaluators see the context. This can be used to add metrics or attributes based on the task output, span tree, or any other state — for example, deriving metrics from instrumented spans (like tool call counts or API latency), or computing values from external resources set up during `setup()`:

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle


class EnrichMetrics(CaseLifecycle):
    async def prepare_context(self, ctx: EvaluatorContext) -> EvaluatorContext:
        ctx.metrics['output_length'] = len(str(ctx.output))
        return ctx


@dataclass
class CheckLength(Evaluator):
    max_length: int = 50

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.metrics.get('output_length', 0) <= self.max_length


dataset = Dataset(
    cases=[Case(name='short', inputs='hi'), Case(name='long', inputs='hello world')],
    evaluators=[CheckLength()],
)

report = dataset.evaluate_sync(lambda inputs: inputs.upper(), lifecycle=EnrichMetrics)

for case in report.cases:
    print(f'{case.name}: output_length={case.metrics["output_length"]}')
    #> short: output_length=2
    #> long: output_length=11
```

## Type Parameters

[`CaseLifecycle`][pydantic_evals.lifecycle.CaseLifecycle] is generic over the same three type parameters as [`Case`][pydantic_evals.dataset.Case]: `InputsT`, `OutputT`, and `MetadataT`. All three default to `Any`, so you can omit them when your hooks don't need type-specific access:

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle


# Works with any dataset — no type parameters needed
class GenericMetricEnricher(CaseLifecycle):
    async def prepare_context(self, ctx: EvaluatorContext) -> EvaluatorContext:
        ctx.metrics['custom'] = 42
        return ctx


dataset = Dataset(cases=[Case(inputs='test')])
report = dataset.evaluate_sync(lambda inputs: inputs, lifecycle=GenericMetricEnricher)

print(report.cases[0].metrics['custom'])
#> 42
```

## Next Steps

- **[Metrics & Attributes](metrics-attributes.md)** — Recording metrics inside tasks
- **[Custom Evaluators](../evaluators/custom.md)** — Using enriched metrics in evaluators
- **[Span-Based Evaluation](../evaluators/span-based.md)** — Analyzing execution traces
