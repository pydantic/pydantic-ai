# Online Evaluation

Online evaluation lets you attach evaluators to production (or staging) functions so that every call (or a sampled subset) is automatically evaluated in the background. The same [`Evaluator`][pydantic_evals.evaluators.Evaluator] classes used with [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate] work here; the difference is just in how they're wired up.

## When to Use Online Evaluation

Online evaluation is useful when you want to:

- **Monitor production quality:** continuously score LLM outputs against rubrics
- **Catch regressions:** detect degradation in agent behavior across deploys
- **Collect evaluation data:** build datasets from real traffic for offline analysis
- **Control costs:** sample expensive LLM judges on a fraction of traffic while running cheap checks on everything

For testing against curated datasets before deployment, use [offline evaluation](quick-start.md) with [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate] instead.

## Quick Start

The [`evaluate()`][pydantic_evals.online.evaluate] decorator attaches evaluators to any function. Evaluators run in the background without blocking the caller:

```python
import asyncio
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import configure, evaluate, wait_for_evaluations

results_log: list[str] = []
configure(default_sink=lambda results, failures, ctx: results_log.extend(f'{r.name}={r.value}' for r in results))


@dataclass
class OutputNotEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


@evaluate(OutputNotEmpty())
async def summarize(text: str) -> str:
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

This uses the module-level [`configure()`][pydantic_evals.online.configure] and [`evaluate()`][pydantic_evals.online.evaluate] functions, which delegate to a global [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig]. For multiple configurations or isolated setups, create your own config instances (see [OnlineEvalConfig](#onlineevalconfig) below).

## Core Concepts

### OnlineEvaluator

Different evaluators need different settings. A cheap heuristic could run on 100% of traffic; an expensive LLM judge might run on 1%. [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] wraps an [`Evaluator`][pydantic_evals.evaluators.Evaluator] with per-evaluator configuration:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.online import OnlineEvaluator


@dataclass
class IsHelpful(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) > 10


# Cheap evaluator: run on every request
always_check = OnlineEvaluator(evaluator=IsHelpful(), sample_rate=1.0)

# Expensive evaluator: run on 1% of requests, limit concurrency
rare_check = OnlineEvaluator(
    evaluator=LLMJudge(rubric='Is the response helpful?'),
    sample_rate=0.01,
    max_concurrency=5,
)
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

The built-in [`CallbackSink`][pydantic_evals.online.CallbackSink] wraps any callable (sync or async) that accepts results, failures, and context. You can also pass a bare callable wherever a sink is expected — it's auto-wrapped in a [`CallbackSink`][pydantic_evals.online.CallbackSink].

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

!!! note "Evaluators require at least one sink"
    Evaluators are designed to produce evaluation results, not side effects — use sinks for
    any actions you want to take on results (logging, alerting, storing, etc.).
    Because of this, **evaluators are skipped entirely when no sinks are configured**
    (neither a per-evaluator sink nor a default sink). This avoids wasted work when there's
    nowhere to send results.

    If you need side effects produced _within_ an evaluator itself, configure a no-op sink
    to ensure the evaluator runs:

    ```python
    from pydantic_evals.online import OnlineEvalConfig

    config = OnlineEvalConfig(default_sink=lambda results, failures, context: None)
    ```

## Sampling

Control evaluation frequency with per-evaluator sample rates to balance quality monitoring against cost.

!!! note
    Sampling is decided **before** the decorated function runs. When no evaluators are sampled for a given call, the function executes without any additional instrumentation overhead (no logfire span or span tree capture).

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

Pass a callable to enable runtime-configurable or input-dependent sampling. The callable receives a [`SamplingContext`][pydantic_evals.online.SamplingContext] with the evaluator instance, function inputs, config metadata, and a per-call random seed, and returns a `float` (probability) or `bool` (always/never):

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator, SamplingContext


def get_current_rate(ctx: SamplingContext) -> float:
    return 0.5


@dataclass
class QuickCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


dynamic = OnlineEvaluator(evaluator=QuickCheck(), sample_rate=get_current_rate)
```

This enables integration with feature flags, managed variables, or configuration systems — for example, you could replace `get_current_rate` with a function that reads from a remote config service at runtime, allowing you to change the probability without redeploying the application.

You can also use the [`SamplingContext`][pydantic_evals.online.SamplingContext] to make sampling decisions based on the function inputs:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator, SamplingContext


def sample_long_inputs(ctx: SamplingContext) -> bool:
    """Only evaluate calls with long input text."""
    return len(str(ctx.inputs.get('text', ''))) > 100


@dataclass
class QualityCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return len(str(ctx.output)) > 10


expensive = OnlineEvaluator(evaluator=QualityCheck(), sample_rate=sample_long_inputs)
```

### Correlated Sampling

By default, each evaluator samples independently. With three evaluators each at 10%, roughly 27% of calls incur evaluation overhead (`1 − 0.9³`). If you'd prefer that the *same* 10% of calls run *all* evaluators, set `sampling_mode='correlated'`:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvalConfig, OnlineEvaluator


@dataclass
class CheckA(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


@dataclass
class CheckB(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


config = OnlineEvalConfig(
    default_sink=lambda results, failures, ctx: None,
    sampling_mode='correlated',
)

# Both run on the same ~10% of calls
check_a = OnlineEvaluator(evaluator=CheckA(), sample_rate=0.1)
check_b = OnlineEvaluator(evaluator=CheckB(), sample_rate=0.1)
```

In correlated mode, a single random `call_seed` (uniformly distributed between 0.0 and 1.0) is generated per function call and shared across all evaluators. An evaluator runs when `call_seed < sample_rate`, so lower-rate evaluators' calls are always a subset of higher-rate ones, and the total overhead probability equals the maximum rate rather than accumulating.

The `call_seed` is also available on [`SamplingContext`][pydantic_evals.online.SamplingContext] for custom `sample_rate` callables that want to implement their own correlated logic regardless of mode.

### Disabling Evaluation

Use [`disable_evaluation()`][pydantic_evals.online.disable_evaluation] to suppress all online evaluation in a scope. This may be useful in tests:

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

## Conditional Evaluation

For cost control, you can run expensive evaluation logic conditionally within a single custom evaluator. Return a mapping where you only include keys for checks that have run — checks you don't want to perform can simply be omitted from the results:

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
    wait_for_evaluations,
)

results_log: list[str] = []


async def log_sink(
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    context: EvaluatorContext,
) -> None:
    for r in results:
        results_log.append(f'{r.name}={r.value}')


@dataclass
class ConditionalAnalysis(Evaluator):
    """Runs a cheap check on every call, and an expensive check only on long outputs."""

    def evaluate(self, ctx: EvaluatorContext) -> dict[str, float | bool]:
        output = str(ctx.output)
        results: dict[str, float | bool] = {
            'has_content': len(output) > 0,
        }
        # Only run the expensive analysis on long outputs
        if len(output) > 20:
            # pretend the following line is expensive..
            results['detail_score'] = len(output) / 100.0
        return results


config = OnlineEvalConfig(default_sink=log_sink)


@config.evaluate(ConditionalAnalysis())
async def generate(prompt: str) -> str:
    return f'Response to: {prompt}'


async def main():
    await generate('hi')  # short output — only cheap check runs
    await wait_for_evaluations()
    print(results_log)
    #> ['has_content=True']

    results_log.clear()
    await generate('tell me a long story about dragons')  # long output — both checks run
    await wait_for_evaluations()
    print(sorted(results_log))
    #> ['detail_score=0.47', 'has_content=True']


asyncio.run(main())
```

This pattern lets you combine cheap and expensive checks in one evaluator, avoiding unnecessary work when conditions aren't met.

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


async def main():
    # Sync decorated functions work from async contexts too
    result = process('hello')
    print(result)
    #> HELLO

    await wait_for_evaluations()
    print(results_log)
    #> ['OutputCheck=True']


asyncio.run(main())
```

Sync decorated functions work from both sync and async contexts. When a running event loop is available, evaluators are dispatched as background tasks on that loop. Otherwise, a background thread with its own event loop is spawned.

## Per-Evaluator Sink Overrides

Individual evaluators can override the config's default sink. This is useful if different evaluators need to send results to different destinations:

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
    # Normally EvaluatorContext would not be manually constructed —
    # it is built automatically by the @evaluate decorator or OnlineEvaluation capability,
    # or from an EvaluatorContextSource (see below).
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

For fetching context data from external storage (like Pydantic Logfire), implement the [`EvaluatorContextSource`][pydantic_evals.online.EvaluatorContextSource] protocol. It defines `fetch()` and `fetch_many()` methods that return [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] objects from stored data:

```python
import asyncio
from collections.abc import Sequence

from pydantic_evals.evaluators import EvaluatorContext
from pydantic_evals.online import SpanReference
from pydantic_evals.otel.span_tree import SpanTree


class MyContextSource:
    """Example source that fetches context from a hypothetical store."""

    def __init__(self, store: dict[str, EvaluatorContext]) -> None:
        self._store = store

    async def fetch(self, span: SpanReference) -> EvaluatorContext:
        return self._store[span.span_id]

    async def fetch_many(self, spans: Sequence[SpanReference]) -> list[EvaluatorContext]:
        return [self._store[s.span_id] for s in spans]


def _make_context(
    *,
    inputs: object = None,
    output: object = None,
    metadata: object = None,
    duration: float = 0.0,
) -> EvaluatorContext:
    # Normally EvaluatorContext would not be manually constructed —
    # it is built automatically by the @evaluate decorator or OnlineEvaluation capability.
    return EvaluatorContext(
        name=None,
        inputs=inputs,
        output=output,
        expected_output=None,
        metadata=metadata,
        duration=duration,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


async def main():
    source = MyContextSource({
        'span_abc': _make_context(
            inputs={'query': 'What is AI?'},
            output='AI is artificial intelligence.',
            metadata={'model': 'gpt-4o'},
            duration=1.2,
        ),
        'span_def': _make_context(
            inputs={'query': 'What is ML?'},
            output='ML is machine learning.',
            metadata={'model': 'gpt-4o'},
            duration=0.8,
        ),
    })

    # Fetch a single context
    ctx = await source.fetch(SpanReference(trace_id='t1', span_id='span_abc'))
    print(f'inputs: {ctx.inputs}')
    #> inputs: {'query': 'What is AI?'}
    print(f'output: {ctx.output}')
    #> output: AI is artificial intelligence.

    # Fetch multiple contexts in a batch
    spans = [
        SpanReference(trace_id='t1', span_id='span_abc'),
        SpanReference(trace_id='t1', span_id='span_def'),
    ]
    contexts = await source.fetch_many(spans)
    print(f'batch size: {len(contexts)}')
    #> batch size: 2


asyncio.run(main())
```

## Concurrency Control

Each [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] has a `max_concurrency` limit (default: 10). When the limit is reached, new evaluation requests for that evaluator are **dropped** (not queued). This prevents expensive evaluators from consuming unbounded resources:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvaluator


@dataclass
class ExpensiveCheck(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        # Imagine this makes a slow call to an LLM
        return True


# Allow at most 3 concurrent evaluations
limited = OnlineEvaluator(
    evaluator=ExpensiveCheck(),
    sample_rate=0.1,
    max_concurrency=3,
)
```

To react to dropped evaluations, set `on_max_concurrency` on the [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] or as a default on [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig]. The callback receives the [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] that would have been evaluated, and can be sync or async:

```python
import warnings
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnlineEvalConfig, OnlineEvaluator


@dataclass
class ExpensiveCheck(Evaluator):
    async def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


def warn_on_drop(ctx: EvaluatorContext) -> None:
    warnings.warn('Evaluation dropped due to max concurrency', stacklevel=1)


# Per-evaluator handler
limited = OnlineEvaluator(
    evaluator=ExpensiveCheck(),
    max_concurrency=3,
    on_max_concurrency=warn_on_drop,
)

# Or set a global default for all evaluators in a config
config = OnlineEvalConfig(on_max_concurrency=warn_on_drop)
```

!!! note
    If neither the per-evaluator nor the config-level `on_max_concurrency` is set, dropped evaluations are silently ignored.

## Error Handling

There are two types of error handling:

- **`on_sampling_error`**: Called synchronously when a `sample_rate` callable raises. Receives the exception and the [`Evaluator`][pydantic_evals.evaluators.Evaluator]. Must be sync (not async). If set, the evaluator is skipped. If not set, the exception **propagates to the caller**.
- **`on_error`**: Called when an exception occurs in a `sink` or `on_max_concurrency` callback. Receives the exception, [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext], [`Evaluator`][pydantic_evals.evaluators.Evaluator], and a location string. Can be sync or async. If not set, exceptions are **silently suppressed**.

Set these on [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig] for global defaults, or on [`OnlineEvaluator`][pydantic_evals.online.OnlineEvaluator] to override per-evaluator:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import OnErrorLocation, OnlineEvalConfig, OnlineEvaluator


def log_errors(
    exc: Exception,
    ctx: EvaluatorContext,
    evaluator: Evaluator,
    location: OnErrorLocation,
) -> None:
    print(f'[{location}] {type(exc).__name__}: {exc}')


@dataclass
class MyCheck(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return True


# Global default — applies to all evaluators in this config
config = OnlineEvalConfig(
    default_sink=lambda results, failures, context: None,
    on_error=log_errors,
)

# Per-evaluator override
custom = OnlineEvaluator(evaluator=MyCheck(), on_error=log_errors)
```

Key behaviors:

- **Evaluator exceptions** are handled by converting them to [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure] objects passed to sinks — they do not go through `on_error`.
- **One evaluator's error doesn't affect siblings** — each evaluator runs in its own task with isolated error handling.
- **One sink's error doesn't affect other sinks** — each sink submission is wrapped individually.
- **If `on_error` itself raises**, the exception is silently suppressed to protect sibling evaluators.
- **If no `on_error` is set**, exceptions are silently suppressed — this is the safe default.

## API Reference

The complete API for the `pydantic_evals.online` module is documented in the [API reference](../api/pydantic_evals/online.md).

## Next Steps

- **[Custom Evaluators](evaluators/custom.md)** — Write evaluators for your domain
- **[Built-in Evaluators](evaluators/built-in.md)** — Use ready-made evaluators
- **[Logfire Integration](how-to/logfire-integration.md)** — Visualize evaluation results in Logfire
- **[Quick Start](quick-start.md)** — Offline evaluation with [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate]
