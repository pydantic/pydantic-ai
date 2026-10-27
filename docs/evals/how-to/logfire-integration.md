---
description: "Send Pydantic Evals experiments to Pydantic Logfire to view results, inspect each case and its full trace, and compare evaluation runs over time."
---

# Logfire Integration

Visualize and analyze evaluation results using Pydantic Logfire.

## Overview

Pydantic Evals uses OpenTelemetry to record traces of the evaluation process. These traces contain all the information from your evaluation reports, plus full tracing from the execution of your task function.

You can send these traces to any OpenTelemetry-compatible backend, including [Pydantic Logfire](https://logfire.pydantic.dev/docs/guides/web-ui/evals/).

## Installation

Install the optional logfire dependency:

```bash
pip install 'pydantic-evals[logfire]'
```

## Basic Setup

Configure Logfire before running evaluations:

```python {title="basic_logfire_setup.py"}
import logfire

from pydantic_evals import Case, Dataset

# Configure Logfire
logfire.configure(
    send_to_logfire='if-token-present',  # (1)!
)


# Your evaluation code
def my_task(inputs: str) -> str:
    return f'result for {inputs}'


dataset = Dataset(name='logfire_demo', cases=[Case(name='test', inputs='example')])
report = dataset.evaluate_sync(my_task)
```

1. Sends data to Logfire only if the `LOGFIRE_TOKEN` environment variable is set

That's it! Your evaluation traces will now appear in the Logfire web UI as long as you have the `LOGFIRE_TOKEN` environment variable set.

## What Gets Sent to Logfire

When you run an evaluation, Logfire receives:

1. **Evaluation metadata**
    1. Dataset name
    1. Number of cases
    1. Evaluator names
2. **Per-case data**
    1. Inputs and outputs
    1. Expected outputs
    1. Metadata
    1. Execution duration
3. **Evaluation results**
    1. Scores, assertions, and labels
    1. Reasons (if included)
    1. Evaluator failures
4. **Task execution traces**
    1. All OpenTelemetry spans from your task function
    1. Tool calls (for Pydantic AI agents)
    1. API calls, database queries, etc.

For the exact spans and attributes, see the [span reference](#span-reference).

## Viewing Results in Logfire

### Evaluation Overview

Logfire provides a special table view for evaluation results on the root evaluation span:

![Logfire Evals Overview](../../img/logfire-evals-overview.png)

This view shows:

- Case names
- Pass/fail status
- Scores and assertions
- Execution duration
- Quick filtering and sorting

### Individual Case Details

Click any case to see detailed inputs and outputs:

![Logfire Evals Case](../../img/logfire-evals-case.png)

### Full Trace View

View the complete execution trace including all spans generated during evaluation:

![Logfire Evals Case Trace](../../img/logfire-evals-case-trace.png)

This is especially useful for:

- Debugging failed cases
- Understanding performance bottlenecks
- Analyzing tool usage patterns
- Writing span-based evaluators

## Analyzing Traces

### Comparing Runs

Run the same evaluation multiple times and compare in Logfire:

```python
from pydantic_evals import Case, Dataset


def original_task(inputs: str) -> str:
    return f'original result for {inputs}'


def improved_task(inputs: str) -> str:
    return f'improved result for {inputs}'


dataset = Dataset(name='comparison', cases=[Case(name='test', inputs='example')])

# Run 1: Original implementation
report1 = dataset.evaluate_sync(original_task)

# Run 2: Improved implementation
report2 = dataset.evaluate_sync(improved_task)

# Compare in Logfire by filtering by timestamp or attributes
```

### Debugging Failed Cases

Find failed cases quickly:

1. Search for `service_name = 'my_service_evals' AND is_exception` (replace with the actual service name you are using)
2. View the full span tree to see where the failure occurred
3. Inspect attributes and logs for error messages

## Span-Based Evaluation

Logfire integration enables powerful span-based evaluators. See [Span-Based Evaluation](../evaluators/span-based.md) for details.

Example: Verify specific tools were called:

```python
import logfire

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import HasMatchingSpan

logfire.configure(send_to_logfire='if-token-present')


def my_agent(inputs: str) -> str:
    return f'result for {inputs}'


dataset = Dataset(
    name='logfire_demo',
    cases=[Case(name='test', inputs='example')],
    evaluators=[
        HasMatchingSpan(
            query={'name_contains': 'search_tool'},
            evaluation_name='used_search',
        ),
    ],
)

report = dataset.evaluate_sync(my_agent)
```

The span tree is available in both:

- Your evaluator code (via `ctx.span_tree`)
- Logfire UI (visual trace view)

## Span Reference {#span-reference}

This section lists the OpenTelemetry spans and attributes that [`Dataset.evaluate`][pydantic_evals.dataset.Dataset.evaluate] emits, for building queries, dashboards, or other tools that read or produce these spans.

!!! note
    As the [version policy](../../version-policy.md) allows, these span attributes may change in minor releases as Evals support in Logfire evolves.

The spans listed here use the instrumentation scope `pydantic-evals`; spans from your task keep their own scopes. Each span's OpenTelemetry name is its message template, such as `case: {case_name}`, except the evaluator span, which is named `evaluator: {evaluator_name}` with the message template `Calling evaluator: {evaluator_name}`. The message template is in the `logfire.msg_template` attribute and the rendered message, such as `case: my_case`, in `logfire.msg`. Attribute values that are not strings, numbers, or booleans are JSON-encoded strings, and `logfire.json_schema` describes them.

A run produces this span tree:

```text
evaluate {name}
├── case: {case_name}                 (one per case, or per case and run with `repeat`)
│   ├── execute {task}
│   │   └── spans from your task
│   └── evaluator: {evaluator_name}   (one per evaluator)
└── report_evaluator: {evaluator_name} (one per report evaluator)
```

### Experiment span

Name `evaluate {name}`. Set when the span starts:

| Attribute | Value |
|---|---|
| `name` | The `name` argument, defaulting to `task_name` |
| `task_name` | The `task_name` argument, defaulting to the task function's name |
| `dataset_name` | The dataset's `name` |
| `n_cases` | The number of cases in the dataset, not multiplied by `repeat` |
| `gen_ai.operation.name` | Always `'experiment'` |
| `metadata` | The `metadata` argument; only set when given |
| `logfire.experiment.repeat` | The `repeat` argument; only set when greater than 1 |

Set when all cases and report evaluators have finished:

| Attribute | Value |
|---|---|
| `assertion_pass_rate` | The fraction of passing assertions across successful cases, the same value as `averages.assertions` below (with `repeat`, the mean of each case's pass rate across its runs); only set when there are assertions |
| `logfire.experiment.metadata` | `{"n_cases": ..., "repeat": ..., "metadata": ..., "averages": ...}`, where `repeat` and `metadata` are only present when they are on the experiment span, and `averages` is a [`ReportCaseAggregate`][pydantic_evals.reporting.ReportCaseAggregate] that is only present when there is a successful case |
| `logfire.experiment.analyses` | The list of [report evaluator](../evaluators/report-evaluators.md) analyses; only set when there are any |
| `logfire.experiment.report_evaluator_failures` | A list of `{"name", "error_message", "error_stacktrace", "source"}` objects for report evaluators that raised; only set when there are any |

### Case span

Name `case: {case_name}`. Set when the span starts:

| Attribute | Value |
|---|---|
| `task_name` | The task function's name |
| `case_name` | The case name; with `repeat`, the run-indexed name such as `my_case [1/3]` |
| `inputs` | The case's `inputs` |
| `metadata` | The case's `metadata` |
| `expected_output` | The case's `expected_output` |
| `logfire.experiment.source_case_name` | The original case name; only set when `repeat` is greater than 1 |

Set after the task succeeds:

| Attribute | Value |
|---|---|
| `output` | The task's output |
| `task_duration` | The task's duration in seconds |
| `metrics` | Metrics recorded with [`increment_eval_metric`][pydantic_evals.dataset.increment_eval_metric], plus metrics extracted from the task's spans |
| `attributes` | Attributes recorded with [`set_eval_attribute`][pydantic_evals.dataset.set_eval_attribute] |

Set after all evaluators have finished:

| Attribute | Value |
|---|---|
| `assertions` | Results with `bool` values |
| `scores` | Results with `int` or `float` values |
| `labels` | Results with `str` values |

Each of `assertions`, `scores`, and `labels` maps a result name to an [`EvaluationResult`][pydantic_evals.evaluators.EvaluationResult] object:

```json
{"accuracy": {"name": "accuracy", "value": 0.87, "reason": null, "source": {"name": "MyEvaluator", "arguments": null}, "evaluator_version": "v2"}}
```

`source` is the evaluator's [`EvaluatorSpec`][pydantic_evals.evaluators.evaluator.EvaluatorSpec], and `evaluator_version` comes from [`get_evaluator_version`][pydantic_evals.evaluators.Evaluator.get_evaluator_version]. When two results in a case share a name, the later ones get the keys `name_2`, `name_3`, and so on, while the `name` field keeps the original name.

If the task raises, the exception is recorded on its `execute {task}` span and on the case span, and the attributes set after the task are absent. With `retry_task`, each failed attempt's `execute {task}` span records that attempt's exception. An exception from a case lifecycle hook is also recorded on the case span.

### Task and evaluator spans

| Span | Message | Attribute |
|---|---|---|
| `execute {task}` | `execute {task}` | `task`: the task function's name. With `retry_task`, each attempt gets its own span. |
| `evaluator: {evaluator_name}` | `Calling evaluator: {evaluator_name}` | `evaluator_name`: the evaluator's [`get_default_evaluation_name`][pydantic_evals.evaluators.Evaluator.get_default_evaluation_name]. An exception raised by the evaluator is recorded on this span, not on the case span. |
| `report_evaluator: {evaluator_name}` | `report_evaluator: {evaluator_name}` | `evaluator_name`: the report evaluator's serialization name. |

[Online evaluation](../online-evaluation.md#default-otel-event-emission) emits `gen_ai.evaluation.result` log events instead of these spans.

## Troubleshooting

### No Data Appearing in Logfire

Check:

1. **Token is set**: `echo $LOGFIRE_TOKEN`
2. **Configuration is correct**:
   ```python {test="skip"}
   import logfire

   logfire.configure(send_to_logfire=True)  # Force sending
   ```
3. **Network connectivity**: Check firewall settings
4. **Project exists**: Verify project name in Logfire UI

### Traces Missing Spans

If some spans are missing:

1. **Ensure logfire is configured before imports**:
   ```python
   import logfire

   logfire.configure()  # Must be first
   ```

2. **Check instrumentation**: Ensure your code has enabled all instrumentations you want:
   ```python
   import logfire

   logfire.instrument_pydantic_ai()
   logfire.instrument_httpx(capture_all=True)
   ```

## Best Practices

### 1. Configure Early

Always configure Logfire before running evaluations:

```python
import logfire

from pydantic_evals import Case, Dataset

logfire.configure(send_to_logfire='if-token-present')


# Now import and run evaluations
def task(inputs: str) -> str:
    return f'result for {inputs}'


dataset = Dataset(name='logfire_demo', cases=[Case(name='test', inputs='example')])
dataset.evaluate_sync(task)
```

### 2. Use Descriptive Service Names And Environments

```python
import logfire

logfire.configure(
    service_name='rag-pipeline-evals',
    environment='development',
)
```

### 3. Review Periodically

- Check Logfire regularly to identify patterns
- Look for consistently failing cases
- Analyze performance trends
- Adjust evaluators based on insights

## Next Steps

- **[Span-Based Evaluation](../evaluators/span-based.md)** - Use OpenTelemetry spans in evaluators
- **[Logfire Documentation](https://logfire.pydantic.dev/docs/guides/web-ui/evals/)** - Complete Logfire guide
- **[Metrics & Attributes](metrics-attributes.md)** - Add custom data to traces
