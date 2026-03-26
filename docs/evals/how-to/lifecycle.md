# Case Lifecycle Hooks

Control per-case setup, context enrichment, and teardown during evaluation using [`CaseLifecycle`][pydantic_evals.lifecycle.CaseLifecycle].

## Overview

When running evaluations, you may need to:

- **Set up resources** before each case (databases, services, fixtures)
- **Enrich the evaluator context** with metrics derived from span trees or external state
- **Clean up resources** after evaluators complete, with behavior that varies based on success or failure

[`CaseLifecycle`][pydantic_evals.lifecycle.CaseLifecycle] provides hooks at each stage of case evaluation. You pass a lifecycle **class** (not an instance) to [`Dataset.evaluate`][pydantic_evals.dataset.Dataset.evaluate], and a new instance is created for each case, so instance attributes naturally hold case-specific state.

## Evaluation Flow

Each case follows this flow:

1. **`setup()`** — called before task execution
2. **Task runs**
3. **`prepare_context()`** — called after task, before evaluators
4. **Evaluators run**
5. **`teardown()`** — called after evaluators complete

## Enriching Metrics

The most common use case is enriching the evaluator context with metrics derived from the span tree — for example, counting tool calls or measuring API latency — before evaluators see it:

```python {test="skip"}
from collections import Counter

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle


class EnrichToolMetrics(CaseLifecycle):
    async def prepare_context(self, ctx: EvaluatorContext) -> EvaluatorContext:
        tool_spans = [
            s
            for s in ctx.span_tree.find(name='.*')
            if 'gen_ai.tool.name' in s.attributes
        ]
        counts = Counter(s.attributes['gen_ai.tool.name'] for s in tool_spans)
        ctx.metrics['tool_call_count'] = sum(counts.values())
        ctx.metrics['unique_tool_count'] = len(counts)
        for name, count in counts.items():
            ctx.metrics[f'tool_calls:{name}'] = count
        return ctx


dataset = Dataset(cases=[Case(name='test', inputs='hello')])
report = await dataset.evaluate(my_task, lifecycle=EnrichToolMetrics)
```

Without lifecycle hooks, metrics set via [`increment_eval_metric`][pydantic_evals.increment_eval_metric] inside the task are finalized before evaluators run, and cannot be updated afterward. The `prepare_context` hook runs in between, giving you full access to the span tree and the ability to modify metrics before evaluators see them.

## Per-Case Setup and Teardown

Use `setup()` and `teardown()` when each case needs its own environment. Since a new lifecycle instance is created for each case, instance attributes are naturally case-scoped:

```python {test="skip"}
from contextvars import ContextVar

from pydantic_evals import Case, Dataset
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import ReportCase, ReportCaseFailure

_db: ContextVar['TestDB'] = ContextVar('db')


class DatabaseLifecycle(CaseLifecycle[str, dict, dict]):
    async def setup(self) -> None:
        self.db = await create_test_db()
        seed_rows = (self.case.metadata or {}).get('seed_rows', 0)
        if seed_rows:
            await self.db.insert_seed_data(seed_rows)
        _db.set(self.db)

    async def teardown(
        self,
        result: ReportCase[str, dict, dict] | ReportCaseFailure[str, dict, dict],
    ) -> None:
        await self.db.drop()


dataset = Dataset(
    cases=[
        Case(name='empty_db', inputs='add 3 users', metadata={'seed_rows': 0}),
        Case(name='existing_data', inputs='delete inactive', metadata={'seed_rows': 100}),
    ]
)

report = await dataset.evaluate(my_task, lifecycle=DatabaseLifecycle)
```

The case metadata drives per-case behavior without needing custom `Case` subclasses or serialization.

### Conditional Teardown

The `teardown()` hook receives the full result, so you can vary cleanup logic based on success or failure — for example, keeping test environments up for manual inspection when a case fails:

```python {test="skip"}
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import ReportCase, ReportCaseFailure


class KeepOnFailure(CaseLifecycle[str, str, None]):
    async def setup(self) -> None:
        self.env = await create_test_environment()

    async def teardown(
        self,
        result: ReportCase[str, str, None] | ReportCaseFailure[str, str, None],
    ) -> None:
        if isinstance(result, ReportCaseFailure) and self.keep_failures:
            print(f'Keeping environment for failed case: {self.case.name}')
        else:
            await self.env.destroy()
```

## Combining Hooks

All three hooks can be used together. For example, setting up a database, enriching metrics from it, and cleaning up afterward:

```python {test="skip"}
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle
from pydantic_evals.reporting import ReportCase, ReportCaseFailure


class FullLifecycle(CaseLifecycle[str, dict, dict]):
    async def setup(self) -> None:
        self.db = await create_test_db()
        seed_rows = (self.case.metadata or {}).get('seed_rows', 0)
        if seed_rows:
            await self.db.insert_seed_data(seed_rows)

    async def prepare_context(
        self, ctx: EvaluatorContext[str, dict, dict]
    ) -> EvaluatorContext[str, dict, dict]:
        ctx.metrics['final_row_count'] = await self.db.count()
        return ctx

    async def teardown(
        self,
        result: ReportCase[str, dict, dict] | ReportCaseFailure[str, dict, dict],
    ) -> None:
        await self.db.drop()
```

## Type Parameters

[`CaseLifecycle`][pydantic_evals.lifecycle.CaseLifecycle] is generic over the same three type parameters as [`Case`][pydantic_evals.dataset.Case]: `InputsT`, `OutputT`, and `MetadataT`. All three default to `object`, so you can omit them when your hooks don't need type-specific access:

```python {test="skip"}
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.lifecycle import CaseLifecycle


# Works with any dataset — no type parameters needed
class GenericMetricEnricher(CaseLifecycle):
    async def prepare_context(self, ctx: EvaluatorContext) -> EvaluatorContext:
        ctx.metrics['custom'] = 42
        return ctx
```

## Next Steps

- **[Metrics & Attributes](metrics-attributes.md)** — Recording metrics inside tasks
- **[Custom Evaluators](../evaluators/custom.md)** — Using enriched metrics in evaluators
- **[Span-Based Evaluation](../evaluators/span-based.md)** — Analyzing execution traces
