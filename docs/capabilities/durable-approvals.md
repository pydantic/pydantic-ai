---
description: "Wait for a person's approval inside a durable Pydantic AI agent run with DurableApprovals: the durable execution suspends until the decision arrives, then the run continues."
---

# Durable Approvals

[`DurableApprovals`][pydantic_ai.capabilities.DurableApprovals] is a [capability](overview.md) that resolves [deferred tool calls](../deferred-tools.md), the ones that need a person's approval or an external result, inside a [durable](../durable_execution/overview.md) agent run. Without it, a durable run that reaches such a call ends with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output, and your code has to wait for the answer and start a second run with the message history and the [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]. With it, the run hands a resume token to your `announce` function, suspends until the token is resolved with the decision, and continues in the same durable execution.

The durable engine records the token, the announcement, and the decision, so the person can take hours or days to answer, and a replayed run does not announce again. On engines that can suspend an execution, no compute is held while the run waits.

| Engine | Support |
|---|---|
| [AWS Lambda durable functions](../harness/aws-lambda.md#approvals) | The token is a Lambda callback ID |
| Temporal, DBOS, Prefect, and other engines | Not supported yet: a call that needs a decision fails the run with a `UserError`. End the run with `DeferredToolRequests` output instead |

Engine authors can add support by implementing the [external event primitive](#waiting-for-other-external-events), as described in [Building a durable execution backend](../durable_execution/backends.md#waiting-for-external-events).

```python {title="durable_approvals.py" test="skip" lint="skip"}
from datetime import timedelta

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import DurableApprovals
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability


async def announce(ctx: RunContext, requests: DeferredToolRequests, token: str) -> None:
    calls = ', '.join(f'{call.tool_name} {call.tool_call_id}' for call in requests.approvals)
    await approvals_channel.post(f'Approve {calls}? Answer with token {token}')  # (1)!


agent = Agent(
    '{{ bedrock:medium }}',
    name='support',
    capabilities=[AWSLambdaDurability(), DurableApprovals(announce, timeout=timedelta(days=1))],
)


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: int) -> str:
    return f'Refunded {amount} for order {order_id}'
```

1. `announce` runs once per decision, inside a durable unit that is not run again on replay. Like any durable unit it can be retried when it fails, so make the announcement idempotent, for example by keying the message or database row on the token.

## The decision

Resolve the token with the JSON form of [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults], keyed by the `tool_call_id`s of the pending calls in the `requests` passed to `announce`. Every key is optional:

```json
{
  "approvals": {
    "call_1": true,
    "call_2": false,
    "call_3": {"kind": "tool-denied", "message": "Refunds over $100 need a manager."},
    "call_4": {"kind": "tool-approved", "override_args": {"amount": 100}}
  },
  "calls": {
    "call_5": {"status": "shipped"},
    "call_6": {"kind": "tool-failed", "message": "The warehouse API is down."}
  },
  "metadata": {"call_1": {"approved_by": "alice"}}
}
```

- An approval is `true`, `false` (denied with the default message), a [`ToolDenied`][pydantic_ai.tools.ToolDenied] with the `message` the model is shown, or a [`ToolApproved`][pydantic_ai.tools.ToolApproved] with `override_args` to run the tool with instead of the model's arguments.
- An external call's result is any JSON value the model is shown, or a [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] (`"kind": "tool-failed"`) or [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] (`"kind": "model-retry"`).
- `metadata` reaches the tool as [`RunContext.tool_call_metadata`][pydantic_ai.tools.RunContext.tool_call_metadata].

From Python, build the decision as a `DeferredToolResults` and serialize it:

```python {title="durable_approval_decision.py"}
from pydantic import TypeAdapter

from pydantic_ai import DeferredToolResults, ToolDenied

decision = DeferredToolResults(approvals={'call_1': True, 'call_3': ToolDenied('Refunds over $100 need a manager.')})
print(TypeAdapter(DeferredToolResults).dump_json(decision).decode())
"""
{"calls":{},"approvals":{"call_1":true,"call_3":{"message":"Refunds over $100 need a manager.","kind":"tool-denied"}},"metadata":{}}
"""
```

Calls the decision leaves out stay pending: the next [`handle_deferred_tool_calls`](../deferred-tools.md#resolving-deferred-calls-with-a-handler) handler gets them, and otherwise the run ends with them as its `DeferredToolRequests` output. A decision that is not valid `DeferredToolResults`, or that names a call that is not pending (an answer sent for an earlier wait, say), fails the run with a [`UserError`][pydantic_ai.exceptions.UserError]: the decision is already recorded, so a replay fails the same way.

## Timeouts

`timeout` bounds the wait. When it elapses, or the token is resolved with a failure instead of a decision, every pending approval is denied and every pending external call fails, with the message `No decision on this tool call arrived: <reason>`. The run then continues, so the model can tell the user what happened. Without a `timeout`, the run waits as long as the engine allows, which on AWS Lambda is the execution timeout.

## Composition

`DurableApprovals` orders itself [innermost](custom.md#ordering), so every other deferred tool call handler gets the calls first, and the run waits only for what they leave pending. A [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] that auto-approves read-only tools, or Pydantic AI Harness's [`ApprovalReviewer`](../harness/approval-reviewer.md) that approves low-risk calls on the person's behalf, still runs before the person is asked, wherever it is listed.

`DurableApprovals` emits no telemetry of its own: the run's [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] and [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] mark the wait in the event stream, and the engine's execution history shows it.

## Waiting for other external events

`DurableApprovals` is built on [`wait_for_external_event`][pydantic_ai.durable_exec.wait_for_external_event], which you can call from other code that runs in the durable run itself, such as a capability hook, to wait for any JSON payload:

```python {title="wait_for_review.py" test="skip" lint="skip"}
from datetime import timedelta

from pydantic_ai.durable_exec import wait_for_external_event

review = await wait_for_external_event(
    'editor_review',
    announce=lambda token: send_review_request(draft, token),
    timeout=timedelta(hours=4),
)
```

It raises [`ExternalEventError`][pydantic_ai.durable_exec.ExternalEventError] when the timeout elapses or the token is resolved with a failure. Don't call it from a tool: on engines where a tool call is itself a durable unit, the wait can't be recorded inside it, and AWS Lambda rejects it with a `UserError`.

To run an agent that uses `DurableApprovals` without a durable engine, for example in a test, install an [`ExternalEventWaiter`][pydantic_ai.durable_exec.ExternalEventWaiter] that returns a decision with [`external_event_scope`][pydantic_ai.durable_exec.external_event_scope] around the run.
