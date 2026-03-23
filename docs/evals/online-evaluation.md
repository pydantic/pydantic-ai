# Online Evaluation

Online evaluation lets you attach evaluators to production (or staging) functions so that every call — or a sampled subset — is automatically evaluated in the background. The same [`Evaluator`][pydantic_evals.evaluators.Evaluator] classes used with [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate] work here; the difference is in *how* they're wired up, not *what* they are.

## When to Use Online Evaluation

Online evaluation is useful when you want to:

- **Monitor production quality** — continuously score LLM outputs against rubrics
- **Catch regressions** — detect degradation in agent behavior across deploys
- **Collect evaluation data** — build datasets from real traffic for offline analysis
- **Control costs** — sample expensive LLM judges on a fraction of traffic while running cheap checks on everything

For testing against curated datasets before deployment, use [offline evaluation](quick-start.md) with [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate] instead.

## Quick Start

The simplest way to start is with the [`evaluate()`][pydantic_evals.online.evaluate] decorator, which attaches evaluators to any function:

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import OnlineEvalConfig, wait_for_evaluations


# A custom evaluator — same class you'd use with Dataset.evaluate()
@dataclass
class OutputNotEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


# Collect results for demonstration
results_log: list[str] = []


async def log_results(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        results_log.append(f'{r.name}={r.value}')


config = OnlineEvalConfig(default_sink=log_results)


@config.evaluate(OutputNotEmpty())
async def summarize(text: str) -> str:
    # In a real app, this would call an LLM
    return f'Summary of: {text}'


async def main():
    result = await summarize('Pydantic AI is a Python agent framework.')
    print(result)
    #> Summary of: Pydantic AI is a Python agent framework.

    await wait_for_evaluations()
    print(results_log)
    #> ['OutputNotEmpty=True']


asyncio.run(main())
```

The decorated function runs normally and returns immediately — evaluators execute in the background without blocking the caller.

## Core Concepts

### OnlineEvaluator

Different evaluators need different settings. A cheap heuristic should run on 100% of traffic; an expensive LLM judge might run on 1%. [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] wraps an [`Evaluator`][pydantic_evals.evaluators.Evaluator] with per-evaluator configuration:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator


@dataclass
class IsHelpful(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) > 10


# Cheap evaluator: run on every request
always_check = OnlineEvaluator(evaluator=IsHelpful(), sample_rate=1.0)

# Expensive evaluator: run on 1% of requests, limit concurrency
# rare_check = OnlineEvaluator(
#     evaluator=LLMJudge(rubric="Is the response helpful?"),
#     sample_rate=0.01,
#     max_concurrency=5,
# )
```

When you pass a bare [`Evaluator`][pydantic_evals.evaluators.Evaluator] to the [`evaluate()`][pydantic_evals.online.evaluate] decorator, it's automatically wrapped in an [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] with the config's default sample rate.

### OnlineEvalConfig

[`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig] holds cross-evaluator defaults (sink, sample rate, metadata). There's a global default instance, plus you can create custom instances for different configurations:

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import OnlineEvalConfig, wait_for_evaluations


@dataclass
class IsNonEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


results_log: list[str] = []


async def log_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        results_log.append(f'{r.name}={r.value}')


my_eval = OnlineEvalConfig(
    default_sink=log_sink,
    default_sample_rate=1.0,
    metadata={'service': 'my-app'},
)


@my_eval.evaluate(IsNonEmpty())
async def my_function(query: str) -> str:
    return f'Answer to: {query}'


async def main():
    result = await my_function('What is 2+2?')
    print(result)
    #> Answer to: What is 2+2?
    await wait_for_evaluations()
    print(results_log)
    #> ['IsNonEmpty=True']


asyncio.run(main())
```

### EvaluationSink

[`EvaluationSink`][pydantic_evals.online.EvaluationSink] is a protocol for evaluation result destinations. Results flow through pluggable sinks rather than being hardcoded to one destination.

The built-in [`CallbackSink`][pydantic_evals.online.CallbackSink] wraps any callable (sync or async) that accepts results, failures, and context. You can also pass a bare callable wherever a sink is expected — it's auto-wrapped in a `CallbackSink`.

For custom sinks, implement the [`EvaluationSink`][pydantic_evals.online.EvaluationSink] protocol:

```python
from collections.abc import Sequence

from pydantic_evals.evaluators import (
    EvaluationResult,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import SpanReference


class PrintSink:
    """Prints evaluation results to stdout."""

    async def submit(
        self,
        *,
        results: Sequence[EvaluationResult],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext,
        span_reference: SpanReference | None,
    ) -> None:
        for r in results:
            print(f'  {r.name}: {r.value}')
        for f in failures:
            print(f'  FAILED {f.name}: {f.error_message}')
```

## Sampling

Control evaluation frequency with per-evaluator sample rates to balance quality monitoring against cost.

### Static Sample Rates

A `sample_rate` between 0.0 and 1.0 sets the probability of evaluating each call:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator


@dataclass
class QuickCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


# Run on every request
always = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=1.0)

# Run on 10% of requests
sometimes = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=0.1)

# Never run (effectively disabled)
never = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=0.0)
```

### Dynamic Sample Rates

Pass a callable to enable runtime-configurable sampling. The callable returns a `float` (probability) or `bool` (always/never):

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator

_CURRENT_RATE = 0.5


def get_current_rate() -> float:
    return _CURRENT_RATE


@dataclass
class QuickCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


dynamic = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=get_current_rate)
```

This enables integration with feature flags, managed variables, or configuration systems — change `_CURRENT_RATE` at runtime without redeploying.

### Disabling Evaluation

Use [`disable_evaluation()`][pydantic_evals.online.disable_evaluation] to suppress all online evaluation in a scope. This is especially useful in tests:

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import (
    OnlineEvalConfig,
    disable_evaluation,
    wait_for_evaluations,
)


@dataclass
class OutputCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


results_log: list[str] = []


async def log_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        results_log.append(f'{r.name}={r.value}')


config = OnlineEvalConfig(default_sink=log_sink)


@config.evaluate(OutputCheck())
async def my_function(x: int) -> int:
    return x * 2


async def main():
    # Evaluators suppressed inside this block
    with disable_evaluation():
        result = await my_function(21)
        print(result)
        #> 42

    await wait_for_evaluations()
    print(f'evaluations run: {len(results_log)}')
    #> evaluations run: 0

    # Evaluators resume outside the block
    await my_function(21)
    await wait_for_evaluations()
    print(f'evaluations run: {len(results_log)}')
    #> evaluations run: 1


asyncio.run(main())
```

## Conditional Evaluation (Gating)

For cost control, expensive evaluators can be gated behind cheap checks. The `gate` parameter on [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] is a callable that receives the [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] and returns whether the evaluator should run:

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import (
    OnlineEvalConfig,
    OnlineEvaluator,
    wait_for_evaluations,
)

results_log: list[str] = []


async def log_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        results_log.append(r.name)


@dataclass
class DetailedAnalysis(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> float:
        return len(str(ctx.output)) / 100.0


config = OnlineEvalConfig(default_sink=log_sink)


# Only run the expensive evaluator on long outputs
@config.evaluate(
    OnlineEvaluator(
        evaluator=DetailedAnalysis(),
        gate=lambda ctx: len(str(ctx.output)) > 20,
    )
)
async def generate(prompt: str) -> str:
    return f'Response to: {prompt}'


async def main():
    await generate('hi')  # output is 15 chars — gate blocks
    await wait_for_evaluations()
    print(f'after short output: {len(results_log)} evaluations')
    #> after short output: 0 evaluations

    await generate('tell me a long story about dragons')  # output is 49 chars — gate allows
    await wait_for_evaluations()
    print(f'after long output: {len(results_log)} evaluations')
    #> after long output: 1 evaluations


asyncio.run(main())
```

The gate is checked **after** sampling, so it's only called for requests that were already sampled. Gates can be sync or async. If a gate raises an exception, the evaluator is skipped and the exception is logged.

## Sync Function Support

The [`evaluate()`][pydantic_evals.online.evaluate] decorator works with both async and sync functions:

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import OnlineEvalConfig, wait_for_evaluations

results_log: list[str] = []


async def log_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        results_log.append(f'{r.name}={r.value}')


@dataclass
class OutputCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


config = OnlineEvalConfig(default_sink=log_sink)


@config.evaluate(OutputCheck())
def process(text: str) -> str:
    return text.upper()


result = process('hello')
print(result)
#> HELLO

# wait_for_evaluations() awaits async tasks and joins background threads
asyncio.run(wait_for_evaluations())
print(results_log)
#> ['OutputCheck=True']
```

For sync functions called outside an async context, evaluators are dispatched in a background thread. When called from within an async context (e.g., an event loop is running), evaluators are dispatched as async tasks on that loop.

## Per-Evaluator Sink Overrides

Individual evaluators can override the config's default sink. This is useful when different evaluators need to send results to different destinations:

```python
import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_evals.evaluators import (
    EvaluationResult,
    Evaluator,
    EvaluatorContext,
    EvaluatorFailure,
)
from pydantic_evals.online import (
    OnlineEvalConfig,
    OnlineEvaluator,
    wait_for_evaluations,
)

default_log: list[str] = []
special_log: list[str] = []


async def default_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        default_log.append(r.name)


async def special_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        special_log.append(r.name)


@dataclass
class FastCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


@dataclass
class ImportantCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


config = OnlineEvalConfig(default_sink=default_sink)


@config.evaluate(
    FastCheck(),  # uses default sink
    OnlineEvaluator(evaluator=ImportantCheck(), sink=special_sink),  # uses special sink
)
async def my_function(x: int) -> int:
    return x


async def main():
    await my_function(42)
    await wait_for_evaluations()

    print(f'default: {default_log}')
    #> default: ['FastCheck']
    print(f'special: {special_log}')
    #> special: ['ImportantCheck']


asyncio.run(main())
```

## Re-running Evaluators from Stored Data

A key capability of online evaluation is re-running evaluators without re-executing the original function. This is useful when you want to evaluate historical data with updated rubrics, or run additional evaluators on existing traces.

### run_evaluators

[`run_evaluators()`][pydantic_evals.online.run_evaluators] runs a list of evaluators against an [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] and returns the results:

```python
import asyncio
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import run_evaluators
from pydantic_evals.otel.span_tree import SpanTree


@dataclass
class LengthCheck(Evaluator):
    min_length: int = 10

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) >= self.min_length


@dataclass
class HasKeyword(Evaluator):
    keyword: str = 'hello'

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return self.keyword in str(ctx.output).lower()


async def main():
    # Build a context manually (in practice, you'd get this from stored data)
    ctx = EvaluatorContext(
        name='example',
        inputs={'query': 'greet the user'},
        output='Hello! How can I help you today?',
        expected_output=None,
        metadata=None,
        duration=0.5,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    results, failures = await run_evaluators(
        [LengthCheck(min_length=10), HasKeyword(keyword='hello')],
        ctx,
    )

    for r in results:
        print(f'{r.name}: {r.value}')
        #> LengthCheck: True
        #> HasKeyword: True
    print(f'failures: {len(failures)}')
    #> failures: 0


asyncio.run(main())
```

### EvaluatorContextSource Protocol

For fetching context data from external storage (like Logfire), the [`EvaluatorContextSource`][pydantic_evals.online.EvaluatorContextSource] protocol defines a batch-first interface. Use [`rebuild_context()`][pydantic_evals.online.rebuild_context] and [`rebuild_contexts()`][pydantic_evals.online.rebuild_contexts] to reconstruct `EvaluatorContext` objects from stored data:

```python
import asyncio
from collections.abc import Sequence

from pydantic_evals.online import (
    EvaluatorContextData,
    SpanReference,
    rebuild_context,
    rebuild_contexts,
)
from pydantic_evals.otel.span_tree import SpanTree


class MyContextSource:
    """Example source that fetches context from a hypothetical store."""

    def __init__(self, store: dict[str, EvaluatorContextData]) -> None:
        self._store = store

    async def fetch(self, span: SpanReference) -> EvaluatorContextData:
        return self._store[span.span_id]

    async def fetch_many(self, spans: Sequence[SpanReference]) -> list[EvaluatorContextData]:
        return [self._store[s.span_id] for s in spans]


async def main():
    source = MyContextSource({
        'span_abc': EvaluatorContextData(
            inputs={'query': 'What is AI?'},
            output='AI is artificial intelligence.',
            metadata={'model': 'gpt-4o'},
            duration=1.2,
            span_tree=SpanTree(),
        ),
        'span_def': EvaluatorContextData(
            inputs={'query': 'What is ML?'},
            output='ML is machine learning.',
            metadata={'model': 'gpt-4o'},
            duration=0.8,
            span_tree=SpanTree(),
        ),
    })

    # Rebuild a single context
    ctx = await rebuild_context(source, SpanReference(trace_id='t1', span_id='span_abc'))
    print(f'inputs: {ctx.inputs}')
    #> inputs: {'query': 'What is AI?'}
    print(f'output: {ctx.output}')
    #> output: AI is artificial intelligence.

    # Rebuild multiple contexts in a batch
    spans = [
        SpanReference(trace_id='t1', span_id='span_abc'),
        SpanReference(trace_id='t1', span_id='span_def'),
    ]
    contexts = await rebuild_contexts(source, spans)
    print(f'batch size: {len(contexts)}')
    #> batch size: 2


asyncio.run(main())
```

## Concurrency Control

Each [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] has a `max_concurrency` limit (default: 10). When the limit is reached, new evaluation requests for that evaluator are **dropped** (not queued), and a warning is logged. This prevents expensive evaluators from consuming unbounded resources:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator


@dataclass
class ExpensiveCheck(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        # Imagine this calls an LLM
        return True


# Allow at most 3 concurrent evaluations
limited = OnlineEvaluator(
    evaluator=ExpensiveCheck(),
    sample_rate=0.1,
    max_concurrency=3,
)
```

!!! note
    Dropped evaluations are logged at WARNING level via the `pydantic_evals.online` logger. Monitor this in production to tune `max_concurrency` appropriately.

## API Reference

The complete API for the `pydantic_evals.online` module is documented in the [API reference](../api/pydantic_evals/online.md).

Key classes and functions:

| Name | Description |
|------|-------------|
| [`evaluate()`][pydantic_evals.online.evaluate] | Decorator to attach evaluators (uses global config) |
| [`configure()`][pydantic_evals.online.configure] | Configure the global default config |
| [`disable_evaluation()`][pydantic_evals.online.disable_evaluation] | Context manager to suppress evaluation |
| [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig] | Cross-evaluator configuration |
| [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] | Per-evaluator configuration wrapper |
| [`EvaluationSink`][pydantic_evals.online.EvaluationSink] | Protocol for result destinations |
| [`CallbackSink`][pydantic_evals.online.CallbackSink] | Built-in sink wrapping a callable |
| [`SpanReference`][pydantic_evals.online.SpanReference] | Identifies a span for result association |
| [`run_evaluators()`][pydantic_evals.online.run_evaluators] | Run evaluators on a context directly |
| [`rebuild_context()`][pydantic_evals.online.rebuild_context] | Reconstruct context from stored data |
| [`rebuild_contexts()`][pydantic_evals.online.rebuild_contexts] | Batch-reconstruct contexts |

## Next Steps

- **[Custom Evaluators](evaluators/custom.md)** — Write evaluators for your domain
- **[Built-in Evaluators](evaluators/built-in.md)** — Use ready-made evaluators
- **[Logfire Integration](how-to/logfire-integration.md)** — Visualize evaluation results in Logfire
- **[Quick Start](quick-start.md)** — Offline evaluation with `Dataset.evaluate()`
