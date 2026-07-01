# Agentic Evaluators

Deterministic, span-based evaluators that grade an agent's *trajectory* — the sequence and arguments of tool calls — rather than just its final output.

!!! note "Requires Logfire"
    These evaluators read from the OpenTelemetry span tree captured during
    task execution, so [`logfire`](../how-to/logfire-integration.md) must be
    installed and configured:
    ```bash
    pip install 'pydantic-evals[logfire]'
    ```
    If spans aren't available, each evaluator returns a failing result
    (`False` for the boolean evaluators, `0.0` for `TrajectoryMatch`) with
    a reason pointing at logfire configuration, rather than raising.

!!! warning "Locally-executed tools only"
    These evaluators see tools whose execution produces a local OpenTelemetry
    span — i.e. tools that Pydantic AI invokes itself. Provider-native or
    server-side builtin tools (such as OpenAI's file search or Anthropic's
    web search) don't produce local spans and are therefore invisible to
    these evaluators. Use [`HasMatchingSpan`][pydantic_evals.evaluators.HasMatchingSpan]
    against the provider's own spans, or the model's output, to assess those.

!!! note "What counts as a tool call"
    Every execution *attempt* produces a span, discriminated as follows:

    - An attempt that ended in an error — the tool body raised an exception,
      or requested a retry via `ModelRetry` — is **not** counted by default;
      pass `include_failed=True` to count every attempt. The exception:
      [`MaxToolCalls`][pydantic_evals.evaluators.MaxToolCalls] counts failed
      attempts by default (they still consume budget); pass
      `include_failed=False` there to count only successful calls.
    - A deferred call (`ApprovalRequired` / `CallDeferred`) is **never**
      counted: it did not execute in this run.
    - All matching spans in the captured trace are counted, including tool
      calls made by nested sub-agents (agent-as-tool delegation). If you
      delegate to sub-agents that call their own tools, account for those
      calls in your expectations and budgets.

## Overview

Agentic evaluators answer a class of "did the agent do the right thing?" questions that pure input/output checks can't:

- **Tool coverage** — did the agent call the specific tools it was supposed to? ([`ToolCorrectness`][pydantic_evals.evaluators.ToolCorrectness])
- **Trajectory shape** — did it call them in the right order, or at least use the right set? ([`TrajectoryMatch`][pydantic_evals.evaluators.TrajectoryMatch])
- **Argument quality** — did the tool receive the expected inputs? ([`ArgumentCorrectness`][pydantic_evals.evaluators.ArgumentCorrectness])
- **Budget discipline** — did the agent finish within a tool-call and/or model-request budget? ([`MaxToolCalls`][pydantic_evals.evaluators.MaxToolCalls], [`MaxModelRequests`][pydantic_evals.evaluators.MaxModelRequests])

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
        ),
    ],
)
```

**Parameters:**

- `expected_tools` (`list[str]`): Tool names the agent is expected to call. Order doesn't matter; duplicates are significant — `['search', 'search']` requires two `search` calls.
- `allow_extra` (`bool`, default `False`): By default, any tool call not listed in `expected_tools` fails the check. Set to `True` to only require that the expected tools were called, permitting extras.
- `include_failed` (`bool`, default `False`): Whether to count tool-call attempts that ended in an error.
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
    - `'in_order'` — F1 computed from the longest common subsequence (LCS). Precision = `LCS / len(actual)`, recall = `LCS / len(expected)`. Allows extra calls interleaved with the expected order, but they reduce precision.
    - `'any_order'` — F1 computed from the multiset intersection. Precision = `overlap / len(actual)`, recall = `overlap / len(expected)`. Order is ignored, but extra and missing calls both reduce the score.
- `include_failed` (`bool`, default `False`): Whether the trajectory includes tool-call attempts that ended in an error.
- `evaluation_name` (`str | None`): Custom name in reports.

**Returns:** [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a `float` value in `[0.0, 1.0]`. For the F1-based modes, the reason text spells out the overlap, precision, recall, and F1 so the score is reproducible from the mismatch.

For example, if `expected = ['a', 'b', 'c']` and the agent called `['a', 'x', 'b']`, the LCS is `['a', 'b']` (length 2), giving precision `2/3`, recall `2/3`, and F1 `≈ 0.667`.

If both the expected and actual trajectories are empty, all modes score `1.0`; if only one of them is empty, all modes score `0.0`.

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
    - `'subset'` — every expected key/value is present in the actual arguments. Note that this applies only to top-level keys: an expected *value* (including a nested dict) must compare equal to the actual value in full.
    - `'exact'` — deep equality; unexpected keys also fail.
- `occurrence` (`Literal['first', 'last'] | int`, default `'first'`): Which invocation to inspect if the tool is called multiple times. Integer indexes are 0-based.
- `include_failed` (`bool`, default `False`): Whether tool-call attempts that ended in an error are considered. When `True`, each attempt counts as a separate occurrence.
- `evaluation_name` (`str | None`): Custom name in reports.

**Returns:** [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a `bool` value.

**Graceful degradation:** this evaluator doesn't crash when arguments aren't available — for example, when the agent was instrumented with `include_content=False`, the evaluator returns `False` with a reason explaining the situation so your reports still make sense.

## MaxToolCalls and MaxModelRequests

Assert that the agent stayed within a tool-call and/or model-request budget. These follow the same shape as [`MaxDuration`][pydantic_evals.evaluators.MaxDuration]: one budget per evaluator, each reported as its own boolean assertion.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import MaxModelRequests, MaxToolCalls

dataset = Dataset(
    name='budget_aware',
    cases=[Case(inputs='Draft a short reply')],
    evaluators=[
        MaxToolCalls(max_calls=5),
        MaxModelRequests(max_requests=3),
    ],
)
```

**Parameters:**

- `MaxToolCalls`: `max_calls` (`int`) — maximum allowed locally-executed tool calls. `include_failed` (`bool`, default `True`) controls whether attempts that ended in an error count against the budget (by default they do — they still consumed time and tokens).
- `MaxModelRequests`: `max_requests` (`int`) — maximum allowed model (chat) requests. Prefers the `requests` value from `ctx.metrics` when available, otherwise counts LLM request spans directly (both use the same criteria).
- Both accept `evaluation_name` (`str | None`) to customize the name in reports — useful when the same budget check appears at both the dataset and case level.

**Returns:** [`EvaluationReason`][pydantic_evals.evaluators.EvaluationReason] with a `bool` value. The `reason` includes the observed count and the budget.

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
            order='in_order',  # F1-based: extra calls only reduce precision, order must be preserved
        ),
    ],
)
```

### Support agent with `ArgumentCorrectness` and budget checks

Verify that the right action was taken with the right inputs — within a reasonable number of steps.

```python
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import (
    ArgumentCorrectness,
    MaxModelRequests,
    MaxToolCalls,
)

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
        MaxToolCalls(max_calls=4),
        MaxModelRequests(max_requests=2),
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
