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

The [`evaluate()`][pydantic_evals.online.evaluate] decorator attaches evaluators to any function. Evaluators run in the background without blocking the caller, and results are emitted as OpenTelemetry events:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import evaluate


@dataclass
class OutputNotEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


@evaluate(OutputNotEmpty())
async def summarize(text: str) -> str:
    return f'Summary of: {text}'
```

Wire up OTel export (e.g. `logfire.configure()`) elsewhere in your application startup so that the emitted `gen_ai.evaluation.result` events reach your backend.

Each decorated call emits one `gen_ai.evaluation.result` OTel event per evaluator result, following the [OTel GenAI evaluation semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult). This mirrors how offline evaluation emits OTel spans via `logfire_span`: if any OTel SDK is configured in the process (via `logfire.configure()`, the OTel SDK directly, or a vendor instrumentation), events flow to your backend; if not, emission is a cheap no-op.

To additionally handle results in Python code — for alerting, bespoke aggregation, in-memory test capture, or non-OTel destinations — register a [sink](#sinks). Sinks run *in addition to* OTel event emission.

The module-level [`configure()`][pydantic_evals.online.configure] and [`evaluate()`][pydantic_evals.online.evaluate] functions delegate to a global [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig]. For multiple configurations or isolated setups, create your own config instances (see [OnlineEvalConfig](#onlineevalconfig) below).

## Target

Each decorated function (or agent) emits results tagged with a **target** — a name that groups results in downstream sinks and dashboards. By default the target is the decorated function's `__name__`, but you can override it with `target=...`:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import evaluate


@dataclass
class OutputNotEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


# Default: target='summarize' (function name)
@evaluate(OutputNotEmpty())
async def summarize(text: str) -> str: ...


# Override: use a friendly name
@evaluate(OutputNotEmpty(), target='customer_support')
async def run_agent(prompt: str) -> str: ...
```

The target name is supplied to sinks on every `submit()` call as a plain `str` — a single sink instance handles any number of decorated functions or agents.

For agent capabilities, the target name is taken from the agent's own `name` attribute (see [Agent Integration](#agent-integration)); to categorize or route on agent-ness, add metadata on the config (e.g., `metadata={'kind': 'agent'}`).

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

[`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig] holds cross-evaluator defaults (sample rate, metadata, optional additional sinks, OTel-emission toggle). There's a global default instance, plus you can create custom instances for different configurations:

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

### Sinks

OTel event emission is the default observability surface for online evaluation (see [Default OTel event emission](#default-otel-event-emission)). Sinks are for *additional* handling in Python code — in-memory test capture, alerting, fan-out to non-OTel destinations, or bespoke aggregation. [`EvaluationSink`][pydantic_evals.online.EvaluationSink] is the protocol; multiple sinks can be registered on a single config.

The built-in [`CallbackSink`][pydantic_evals.online.CallbackSink] wraps any callable (sync or async) that accepts results, failures, and context. You can also pass a bare callable wherever a sink is expected — it's auto-wrapped in a [`CallbackSink`][pydantic_evals.online.CallbackSink].

For custom sinks, implement the [`EvaluationSink`][pydantic_evals.online.EvaluationSink] protocol. Each `submit()` call receives a [`SinkPayload`][pydantic_evals.online.SinkPayload] bundling the results, failures, context, span reference, and target from one or more evaluators that ran for a given function call:

```python
from pydantic_evals.online import SinkPayload


class PrintSink:
    """Prints evaluation results to stdout."""

    async def submit(self, payload: SinkPayload) -> None:
        for r in payload.results:
            version = f' ({r.evaluator_version})' if r.evaluator_version else ''
            print(f'  [{payload.target}] {r.name}{version}: {r.value}')
        for f in payload.failures:
            version = f' ({f.evaluator_version})' if f.evaluator_version else ''
            print(f'  [{payload.target}] FAILED {f.name}{version}: {f.error_message}')
```

`payload.results` and `payload.failures` may cover one or more evaluators from a single function call — when multiple evaluators share a sink, their results are batched into a single `submit()` call. Each result carries its own attribution (name, `evaluator_version` on [`EvaluationResult`][pydantic_evals.evaluators.EvaluationResult] and [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure], and source spec), so sinks can separate them downstream; see [Evaluator Versioning](#evaluator-versioning). The `payload.target` identifies the function or agent being evaluated (see [Target](#target)).

!!! note "`SinkPayload` is extensible"
    Additional fields may be added to [`SinkPayload`][pydantic_evals.online.SinkPayload] in future releases. Read only the attributes your sink needs, and do not instantiate `SinkPayload` directly in application code.

### Default OTel event emission

Every dispatched evaluator emits one `gen_ai.evaluation.result` OTel log event per [`EvaluationResult`][pydantic_evals.evaluators.EvaluationResult] or [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure], unconditionally — no sink registration required. Events are parented to the span that produced them, so they appear nested under the original function call in the trace. If no OTel SDK is configured in the process, emission is a cheap no-op.

Each event has `event.name = 'gen_ai.evaluation.result'` and a short human-readable body (e.g. `evaluation: accuracy=0.87`, or `evaluation: accuracy failed: <error>`). Emission follows the [OpenTelemetry GenAI evaluation semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult), with these attributes:

| Attribute | Source | Notes |
|---|---|---|
| `gen_ai.evaluation.name` | [`EvaluationResult.name`][pydantic_evals.evaluators.EvaluationResult] / [`EvaluatorFailure.name`][pydantic_evals.evaluators.EvaluatorFailure] | The evaluator class name when the `evaluate()` method returns a scalar, or the mapping key when it returns `{'accuracy': ..., 'score': ...}`. |
| `gen_ai.evaluation.score.value` | the result's scalar return value | Populated for `bool` (`True`→`1.0`, `False`→`0.0`) and numeric returns. Omitted for `str` returns. |
| `gen_ai.evaluation.score.label` | the result's scalar return value | Populated for `bool` (`True`→`'pass'`, `False`→`'fail'`) and `str` returns (used directly as the label). Omitted for numeric returns. |
| `gen_ai.evaluation.explanation` | `EvaluationResult.reason` (success) or `EvaluatorFailure.error_message` (failure) | Omitted when absent. Set this via `reason=...` when constructing an `EvaluationResult` inside a custom evaluator. |
| `error.type` | fixed string, failure events only | `'pydantic_evals.EvaluatorFailure'` when the evaluator raised; absent on successful evaluations. |
| `gen_ai.evaluation.target` | `@evaluate(target=...)` or agent `name` | See [Target](#target). |
| `gen_ai.evaluation.evaluator_version` | `Evaluator.evaluator_version` class attribute | Omitted when the class doesn't set it. See [Evaluator Versioning](#evaluator-versioning). |
| `gen_ai.evaluation.evaluator_source` | JSON-serialized [`EvaluatorSpec`][pydantic_evals.evaluators.evaluator.EvaluatorSpec] | Identifies the evaluator class and its constructor arguments, so downstream queries can group by evaluator identity without relying on `name` alone (two different `LLMJudge(rubric=...)` instances share a name but have different sources). |

OTel baggage entries (if any) are also attached to each event as attributes — configurable via `include_baggage` on the config. The `gen_ai.*` and `error.type` attributes above always win on conflict with baggage.

For example, a `ToneCheck(evaluator_version='v2')` evaluator whose `evaluate()` returns `True`, decorated as `@evaluate(ToneCheck(), target='customer_support')`, emits one event with:

- `gen_ai.evaluation.name = 'ToneCheck'`
- `gen_ai.evaluation.score.value = 1.0`, `gen_ai.evaluation.score.label = 'pass'`
- `gen_ai.evaluation.target = 'customer_support'`
- `gen_ai.evaluation.evaluator_version = 'v2'`
- `gen_ai.evaluation.evaluator_source = '{"name": "ToneCheck", "arguments": null}'`

The `gen_ai.evaluation.*` extensions beyond the current semconv sit in the `gen_ai.*` namespace on the assumption that upstream will land analogous attributes there; if OTel adopts different names we'll realign.

To disable the default emission (e.g. in a test harness that only wants to assert on a custom sink), set `emit_otel_events=False` on the config:

```python
from pydantic_evals.online import OnlineEvalConfig

config = OnlineEvalConfig(emit_otel_events=False)
```

#### Evaluator Versioning

Set `evaluator_version` as a class attribute on an [`Evaluator`][pydantic_evals.evaluators.Evaluator] subclass to stamp every result it emits with a version string — surfaced as `gen_ai.evaluation.evaluator_version` on emitted events and as `evaluator_version` on each [`EvaluationResult`][pydantic_evals.evaluators.EvaluationResult] and [`EvaluatorFailure`][pydantic_evals.evaluators.EvaluatorFailure]. This lets trend lines and dashboards filter out results produced by retired evaluator versions without deleting historical rows — useful when you change an LLM judge's prompt or rework a heuristic in a way that invalidates prior scores:

```python
from dataclasses import dataclass

from pydantic_evals.evaluators import Evaluator, EvaluatorContext


@dataclass
class ToneCheck(Evaluator):
    evaluator_version = 'v2'  # bumped after prompt rewrite

    def evaluate(self, ctx: EvaluatorContext) -> str:
        return 'neutral'
```

The version applies to all results the evaluator produces (so one evaluator class maps to one version, even when the evaluator returns a mapping of named results).

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

This enables integration with feature flags, managed variables, or configuration systems — for example, you could replace `get_current_rate` with a function that reads from a remote config service (such as [Logfire managed variables](https://logfire.pydantic.dev/docs/reference/advanced/managed-variables/)) at runtime, allowing you to change the probability without redeploying the application.

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

## Agent Integration

The [`OnlineEvaluation`][pydantic_evals.online_capability.OnlineEvaluation] capability brings online evaluation to Pydantic AI agents. Instead of decorating a function, you add the capability to your agent. As with the `@evaluate` decorator, evaluators dispatch in the background and results are emitted as OTel events by default — no sink registration required:

```python
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online_capability import OnlineEvaluation


@dataclass
class OutputNotEmpty(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return bool(ctx.output)


agent = Agent(
    'openai:gpt-5.2',
    name='assistant',
    capabilities=[OnlineEvaluation(evaluators=[OutputNotEmpty()])],
)
```

The target name written to each emitted event is the agent's own `name` attribute, so events from `agent = Agent(..., name='assistant')` land under `gen_ai.evaluation.target = 'assistant'`. If the agent has no name, the target falls back to the literal string `'agent'`.

After each completed agent run, the capability:

1. Samples evaluators based on their `sample_rate` configuration
2. Builds an [`EvaluatorContext`][pydantic_evals.evaluators.EvaluatorContext] from the run result (output, prompt, token usage, duration, span tree) — `context.name` is populated with the agent run's `run_id`
3. Dispatches evaluators asynchronously in the background
4. Returns control to the caller without waiting for evaluators to finish

To attach additional sinks or override sampling defaults, pass an [`OnlineEvalConfig`][pydantic_evals.online.OnlineEvalConfig] — same as with the `@evaluate` decorator: `OnlineEvaluation(evaluators=[...], config=OnlineEvalConfig(default_sample_rate=0.1))`.

The capability supports all the same features as the [`@evaluate()`][pydantic_evals.online.evaluate] decorator: sampling, per-evaluator sinks, concurrency control, and error handling. The `config` parameter is optional and defaults to the global [`DEFAULT_CONFIG`][pydantic_evals.online.DEFAULT_CONFIG].

!!! note
    [`OnlineEvaluation`][pydantic_evals.online_capability.OnlineEvaluation] wraps [`agent.run()`][pydantic_ai.Agent.run], [`agent.run_stream()`][pydantic_ai.Agent.run_stream], and [`agent.iter()`][pydantic_ai.Agent.iter] when the run reaches a final result. For streaming runs, evaluators are dispatched only after the final result is available and the surrounding context manager exits. The same delayed-dispatch behavior applies when driving an [`agent.iter()`][pydantic_ai.Agent.iter] run to completion, which is generally the preferred streaming API.

## API Reference

The complete API for the `pydantic_evals.online` module is documented in the [API reference](../api/pydantic_evals/online.md).

## Next Steps

- **[Custom Evaluators](evaluators/custom.md)** — Write evaluators for your domain
- **[Built-in Evaluators](evaluators/built-in.md)** — Use ready-made evaluators
- **[Logfire Integration](how-to/logfire-integration.md)** — Visualize evaluation results in Logfire
- **[Quick Start](quick-start.md)** — Offline evaluation with [`Dataset.evaluate()`][pydantic_evals.dataset.Dataset.evaluate]
