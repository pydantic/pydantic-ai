# Agentic Evaluators

Deterministic, span-based evaluators that grade an agent's *trajectory* — the sequence and arguments of tool calls — rather than just its final output.

!!! note "Requires Logfire"
    These evaluators read from the OpenTelemetry span tree captured during
    task execution, so [`logfire`](../how-to/logfire-integration.md) must be
    installed and configured:
    ```bash
    pip install 'pydantic-evals[logfire]'
    ```
    If spans aren't available, each evaluator returns a `False` result with
    a reason pointing at logfire configuration, rather than raising.

!!! warning "Locally-executed tools only"
    These evaluators see tools whose execution produces a local OpenTelemetry
    span — i.e. tools that Pydantic AI invokes itself. Provider-native or
    server-side builtin tools (such as OpenAI's file search or Anthropic's
    web search) don't produce local spans and are therefore invisible to
    these evaluators. Use [`HasMatchingSpan`](span-based.md#hasmatchingspan-evaluator)
    against the provider's own spans, or the model's output, to assess those.

## Overview

Agentic evaluators answer a class of "did the agent do the right thing?" questions that pure input/output checks can't:

- **Tool coverage** — did the agent call the specific tools it was supposed to? ([`ToolCorrectness`][pydantic_evals.evaluators.ToolCorrectness])
- **Trajectory shape** — did it call them in the right order, or at least use the right set? ([`TrajectoryMatch`][pydantic_evals.evaluators.TrajectoryMatch])
- **Argument quality** — did the tool receive the expected inputs? ([`ArgumentCorrectness`][pydantic_evals.evaluators.ArgumentCorrectness])
- **Budget discipline** — did the agent finish within a tool-call and/or model-request budget? ([`StepEfficiency`][pydantic_evals.evaluators.StepEfficiency])
- **Retries** — did the model get tool arguments right on the first try, or did it need retries? ([`RetryCount`][pydantic_evals.evaluators.RetryCount])

They are all deterministic, never call an LLM, and are cheap enough to run on every case in every experiment.

## ToolCorrectness

Assert that the agent called a specific **multiset** of tools. Repeated names require repeated calls.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ToolCorrectness

dataset = Dataset(
    name='rag_agent',
    cases=[Case(inputs='Summarize the latest papers on X')],
    evaluators=[
        ToolCorrectness(
            expected_tools=['search', 'rerank', 'generate'],
            allow_extra=True,
        ),
    ],
)
```

**Parameters:**

- `expected_tools` (`list[str]`): Tool names the agent is expected to call. Order doesn't matter; duplicates are significant — `['search', 'search']` requires two `search` calls.
- `allow_extra` (`bool`, default `True`): If `False`, any tool not listed in `expected_tools` fails the check.
- `evaluation_name` (`str | None`): Custom name in reports.

**Returns:** [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a `bool` value. The `reason` names missing and unexpected tools.

## TrajectoryMatch

Compare the actual ordered list of tool names to an expected one, using one of three modes.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import TrajectoryMatch

dataset = Dataset(
    name='ordered_tools',
    cases=[Case(inputs='Process and file this request')],
    evaluators=[
        TrajectoryMatch(
            expected_trajectory=['validate', 'enrich', 'submit'],
            order='in_order',
        ),
    ],
)
```

**Parameters:**

- `expected_trajectory` (`list[str]`): Expected ordered list of tool names.
- `order` (`Literal['exact', 'in_order', 'any_order']`, default `'in_order'`):
    - `'exact'` — `1.0` iff the sequences are equal, else `0.0`.
    - `'in_order'` — F1 computed from the longest common subsequence (LCS). Precision = `LCS / len(actual)`, recall = `LCS / len(expected)`. Allows extra calls interleaved with the expected order.
    - `'any_order'` — multiset overlap: `|multiset(actual) ∩ multiset(expected)| / len(expected)`.
- `evaluation_name` (`str | None`): Custom name in reports.

**Returns:** [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a `float` value in `[0.0, 1.0]`. For `order='in_order'`, the reason text spells out LCS, precision, recall, and F1 so the score is reproducible from the mismatch.

For example, if `expected = ['a', 'b', 'c']` and the agent called `['a', 'x', 'b']`, the LCS is `['a', 'b']` (length 2), giving precision `2/3`, recall `2/3`, and F1 `≈ 0.667`.

## ArgumentCorrectness

Check that a specific tool call received particular arguments.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ArgumentCorrectness

dataset = Dataset(
    name='support_agent',
    cases=[Case(inputs='Refund order 12345')],
    evaluators=[
        ArgumentCorrectness(
            tool_name='issue_refund',
            expected_arguments={'order_id': '12345'},
            match_mode='subset',
            occurrence='first',
        ),
    ],
)
```

**Parameters:**

- `tool_name` (`str`): The tool to inspect.
- `expected_arguments` (`dict[str, Any]`): Expected argument keys/values.
- `match_mode` (`Literal['exact', 'subset']`, default `'subset'`):
    - `'subset'` — every expected key/value is present in the actual arguments.
    - `'exact'` — deep equality; unexpected keys also fail.
- `occurrence` (`Literal['first', 'last'] | int`, default `'first'`): Which invocation to inspect if the tool is called multiple times. Integer indexes are 0-based.
- `evaluation_name` (`str | None`): Custom name in reports.

**Returns:** [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a `bool` value.

**Graceful degradation:** this evaluator doesn't crash when arguments aren't available — for example, when the agent was instrumented with `include_content=False`, the evaluator returns `False` with a reason explaining the situation so your reports still make sense.

## StepEfficiency

Assert that the agent stayed within tool-call and/or model-request budgets.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import StepEfficiency

dataset = Dataset(
    name='budget_aware',
    cases=[Case(inputs='Draft a short reply')],
    evaluators=[
        StepEfficiency(max_tool_calls=5, max_model_requests=3),
    ],
)
```

**Parameters:**

- `max_tool_calls` (`int | None`): Maximum allowed locally-executed tool calls. `None` disables the check.
- `max_model_requests` (`int | None`): Maximum allowed model (chat) requests. `None` disables the check. Prefers `ctx.metrics['requests']` when available, otherwise counts LLM request spans.
- `evaluation_name` (`str | None`): Unused — results are emitted under fixed keys.

**Returns:** a mapping under the following **stable, documented** keys (only keys whose budget is configured are included):

- `'tool_calls_under_budget'` — set when `max_tool_calls` is provided.
- `'model_requests_under_budget'` — set when `max_model_requests` is provided.

Stable key names make reports render consistently across runs and across cases.

## RetryCount

Surface the number of tool-call retries the agent had to issue. Each retry is a `ToolRetryError` raised inside the tool wrapper — covering Pydantic AI's tool argument-validation failures, tool-raised [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], missing tool names, and structured-output validation. A non-zero count means the model did not get its arguments right on the first try.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import RetryCount

dataset = Dataset(
    name='retry_aware',
    cases=[Case(inputs='Run the daily report')],
    evaluators=[
        # Surface the metric every run; also fail when retries exceed 2.
        RetryCount(max_retries=2),
    ],
)
```

**How retries are detected:** Pydantic AI marks tool spans whose call ended in a retry with the `pydantic_ai.tool.retry` attribute. `pydantic_evals.dataset._extract_span_tree_metrics` reads that attribute into `ctx.metrics['retries']` automatically; `RetryCount` prefers that metric and falls back to walking the span tree directly.

**Parameters:**

- `max_retries` (`int | None`): Maximum allowed retries before the run is over budget. `None` disables the budget check; only the count metric is returned.
- `evaluation_name` (`str | None`): Unused — results are emitted under fixed keys.

**Returns:** a mapping under these **stable, documented** keys:

- `'retries'` — always set, an [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] whose `value` is the retry count (`int`). Useful for charting the retry-rate trend over time.
- `'retries_under_budget'` — set when `max_retries` is provided. `True` when within budget.

## Recipes

### RAG agent

Check that the retrieval pipeline runs *search → rerank → generate*, with no unexpected tool calls.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ToolCorrectness, TrajectoryMatch

dataset = Dataset(
    name='rag_pipeline',
    cases=[Case(inputs='Find papers on in-context learning')],
    evaluators=[
        ToolCorrectness(
            expected_tools=['search', 'rerank', 'generate'],
            allow_extra=False,
        ),
        TrajectoryMatch(
            expected_trajectory=['search', 'rerank', 'generate'],
            order='exact',
        ),
    ],
)
```

### Multi-tool agent where order matters

Allow occasional retries, but require the main steps to happen in order.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import TrajectoryMatch

dataset = Dataset(
    name='ordered_with_slack',
    cases=[Case(inputs='Process shipment 99')],
    evaluators=[
        TrajectoryMatch(
            expected_trajectory=['validate', 'enrich', 'submit'],
            order='in_order',  # F1-based: extra calls allowed, order must be preserved
        ),
    ],
)
```

### Support agent with `ArgumentCorrectness` and `StepEfficiency`

Verify that the right action was taken with the right inputs — within a reasonable number of steps.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ArgumentCorrectness, StepEfficiency

dataset = Dataset(
    name='refund_handling',
    cases=[
        Case(
            name='valid_refund',
            inputs={'query': 'Refund my order', 'order_id': '12345'},
            evaluators=[
                ArgumentCorrectness(
                    tool_name='issue_refund',
                    expected_arguments={'order_id': '12345'},
                ),
            ],
        ),
    ],
    evaluators=[
        StepEfficiency(max_tool_calls=4, max_model_requests=2),
    ],
)
```

### Task completion with `LLMJudge` that uses the tool-call trajectory

For tasks where deterministic checks aren't enough, you can layer an
[`LLMJudge`][pydantic_evals.evaluators.LLMJudge] on top of any of these
evaluators. A common pattern is to include a summary of the tool-call
trajectory in the judge's rubric context by writing a small custom evaluator
that surfaces the trajectory into `ctx.attributes`:

```python
from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, LLMJudge


@dataclass
class RecordTrajectory(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        # Read span attributes to build a plain-text trajectory summary.
        tool_names = [
            node.attributes['gen_ai.tool.name']
            for node in ctx.span_tree
            if 'gen_ai.tool.name' in node.attributes
            and not str(node.attributes.get('logfire.msg', '')).startswith('running output function:')
        ]
        ctx.attributes['tool_trajectory'] = ', '.join(str(n) for n in tool_names) or '(none)'
        return True


dataset = Dataset(
    name='task_completion',
    cases=[Case(inputs='Resolve ticket 42')],
    evaluators=[
        RecordTrajectory(),
        LLMJudge(
            rubric=(
                'The agent completed the task correctly. Consider whether the '
                'tool trajectory (available in attributes as `tool_trajectory`) '
                'is reasonable for the given input.'
            ),
            include_input=True,
        ),
    ],
)
```

This pattern keeps `LLMJudge` deterministic where possible, and leaves the
qualitative, open-ended judgement to the LLM without requiring access to the
raw span tree from the rubric.

## Next steps

- [Span-Based Evaluation](span-based.md) — low-level span queries via `HasMatchingSpan` and `SpanQuery`
- [Custom Evaluators](custom.md) — write your own evaluation logic
- [Built-in Evaluators](built-in.md) — complete reference of other evaluator types
