---
description: "Wait for a person's approval inside a Pydantic AI agent run with AwaitApproval: in process by default, or suspended and recorded by a durable engine such as AWS Lambda durable functions."
---

# Await Approval

[`AwaitApproval`][pydantic_ai.capabilities.AwaitApproval] is a [capability](overview.md) that resolves [deferred tool calls](../deferred-tools.md), the ones that need a person's approval or an external result, inside the agent run. It hands the pending calls and a resume token to your `announce` function, waits until code you write elsewhere resolves the token with the person's decision, and continues the same run: approved calls execute, denied ones are reported to the model, and external results are filled in.

It works in any run. Without a durable engine, the wait happens in your process and [`resolve_external_event`][pydantic_ai.external_events.resolve_external_event] ends it. In a [durable](../durable_execution/overview.md) run on an engine that supports it, the engine records the wait and suspends the run while it lasts, so the person can take days and a replay does not ask them again.

## How it relates to `HandleDeferredToolCalls`

`AwaitApproval` is a [`handle_deferred_tool_calls`](../deferred-tools.md#resolving-deferred-calls-with-a-handler) capability, like [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]. A `HandleDeferredToolCalls` handler can also wait for a person itself, by sending them a message and awaiting their reply, and in a plain process that works. In a durable run it falls short in two ways:

- The handler is not journaled. When the engine replays the run, it runs again and asks the person a second time.
- The execution stays alive while it waits, holding compute and running into the engine's invocation limits, such as AWS Lambda's 15-minute invocation timeout.

`AwaitApproval` waits through [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event], which a durable engine can implement: it records the token, the announcement and the decision, and suspends the execution until the decision arrives. Without a durable engine it waits in process, which is what a `HandleDeferredToolCalls` handler would do too.

Ending the run with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output and starting a new run with the decision remains an option, for a run that can't stay open while it waits and has no durable engine to keep it.

## `announce` and the answer

`announce` does not get the answer. It delivers the question and the token to where a person will see it: a Slack message with Approve and Deny buttons, an email with a link, a row in a database table an admin UI lists. The answer comes back separately, through code you write elsewhere, such as the webhook behind the buttons, an admin endpoint or a CLI command, which resolves the token with the decision:

- Without a durable engine, call [`resolve_external_event(token, decision)`][pydantic_ai.external_events.resolve_external_event] in the process the run is waiting in, or [`fail_external_event(token, reason)`][pydantic_ai.external_events.fail_external_event] to end the wait without a decision.
- On AWS Lambda durable functions, the token is a callback ID: send the decision with `SendDurableExecutionCallbackSuccess`, or end the wait with `SendDurableExecutionCallbackFailure`. See [Approvals](../harness/aws-lambda.md#approvals).

`announce` can be sync or async; a sync function runs in a worker thread, like a sync tool. In a durable run it runs once, inside a durable unit that is not run again on replay, but the engine can retry it when it fails, so make it idempotent, for example by keying the message or database row on the token.

## A round trip in one process

This FastAPI app lists the pending approvals and resolves one when the approver posts a decision:

```python {title="await_approval.py"}
import asyncio
from datetime import timedelta

from fastapi import FastAPI

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import AwaitApproval
from pydantic_ai.external_events import resolve_external_event

pending: dict[str, DeferredToolRequests] = {}  # (1)!


def announce(ctx: RunContext, requests: DeferredToolRequests, token: str) -> None:
    pending[token] = requests  # (2)!


agent = Agent(
    '{{ openai:medium }}',
    capabilities=[AwaitApproval(announce, timeout=timedelta(hours=1))],
)


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: int) -> str:
    return f'Refunded ${amount} for order {order_id}'


app = FastAPI()


@app.get('/approvals')
async def list_approvals() -> dict[str, list[str]]:
    return {
        token: [f'{call.tool_name}({call.args_as_json_str()})' for call in requests.approvals]
        for token, requests in pending.items()
    }


@app.post('/approvals/{token}')
async def decide(token: str, approved: bool) -> None:
    requests = pending.pop(token)
    decision = {'approvals': {call.tool_call_id: approved for call in requests.approvals}}
    resolve_external_event(token, decision)  # (3)!


async def main():
    run = asyncio.create_task(agent.run('Refund order 1234, it arrived broken.'))
    while not pending:  # (4)!
        await asyncio.sleep(0.01)
    print(list((await list_approvals()).values()))
    #> [['issue_refund({"order_id":"1234","amount":50})']]
    [token] = pending
    await decide(token, approved=True)
    result = await run
    print(result.output)
    #> I've refunded $50 for order 1234.
```

1. The pending approvals, by token. A real app keeps them in a database, so an admin UI can list them.
2. Delivers the question and the token; it does not wait for the answer.
3. Ends the wait the run is in. The run continues with the decision: the approved refund runs, and the model writes its reply.
4. Stands in for the approver, who opens the approvals page and clicks Approve, which posts to `/approvals/{token}`.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

The wait lives in the process: if the process restarts, the run and its wait are gone, and resolving the old token raises a [`UserError`][pydantic_ai.exceptions.UserError]. So does a token that is wrong, from another process, or already resolved, timed out or cancelled. For waits that must survive restarts, use a durable engine, or end the run with `DeferredToolRequests` output and store the conversation as [Pausing a conversation](../deferred-tools.md#pausing-a-conversation) shows.

## The same agent on AWS Lambda

Under AWS Lambda durable functions, add [`AWSLambdaDurability`](../harness/aws-lambda.md) and run the agent from a durable handler. The token is now a Lambda callback ID, the wait is recorded and suspends the execution, and the approval endpoint resolves the token through the Lambda API instead of `resolve_external_event`:

```python {title="await_approval_lambda.py"}
import json
from datetime import timedelta
from typing import Any

import boto3
from aws_durable_execution_sdk_python import DurableContext, durable_execution
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import AwaitApproval

lambda_client = boto3.client('lambda')


def announce(ctx: RunContext, requests: DeferredToolRequests, callback_id: str) -> None:
    ...  # (1)!


agent = Agent(
    '{{ bedrock:medium }}',
    name='support',
    capabilities=[AWSLambdaDurability(), AwaitApproval(announce, timeout=timedelta(days=1))],
)


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: int) -> str:
    return f'Refunded ${amount} for order {order_id}'


@durable_execution
@durable_agent_handler
async def handler(event: dict[str, Any], context: DurableContext) -> str:
    result = await agent.run(str(event['prompt']))
    return result.output


def decide(callback_id: str, tool_call_ids: list[str], approved: bool) -> None:  # (2)!
    decision = {'approvals': {tool_call_id: approved for tool_call_id in tool_call_ids}}
    lambda_client.send_durable_execution_callback_success(
        CallbackId=callback_id, Result=json.dumps(decision)
    )
```

1. Write the callback ID and `requests` to where the approver will see them, such as a DynamoDB table your admin UI lists, or a Slack message whose buttons carry the callback ID.
2. Runs in your approval endpoint, which can be another Lambda function or any service with permission to call `lambda:SendDurableExecutionCallbackSuccess`.

Run outside a durable handler, for example in local development, the same agent waits in process, and `resolve_external_event` resolves its token.

## Engines

| Where the run executes | How it waits |
|---|---|
| A plain process | In the process; resolve the token with `resolve_external_event` |
| [AWS Lambda durable functions](../harness/aws-lambda.md#approvals) | A Lambda callback; resolve it with `SendDurableExecutionCallbackSuccess` |
| Temporal, DBOS, Prefect, and other durable engines | Not supported yet: a call that needs a decision fails the run with a `UserError`, since a wait in process would not be recorded. End the run with `DeferredToolRequests` output instead |

Engine authors can add support as [Building a durable execution backend](../durable_execution/backends.md#waiting-for-external-events) describes.

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

Whoever can resolve the token decides the calls, including their arguments through `override_args` and the `metadata` tools read, so a tool must not treat that metadata as proof of who approved. Pydantic AI checks only the decision's shape and that it names pending calls. Treat the token as a bearer credential, in process as on an engine, and authorize the approver in the code that resolves it.

From Python, build the decision as a `DeferredToolResults` and serialize it:

```python {title="await_approval_decision.py"}
from pydantic import TypeAdapter

from pydantic_ai import DeferredToolResults, ToolDenied

decision = DeferredToolResults(approvals={'call_1': True, 'call_3': ToolDenied('Refunds over $100 need a manager.')})
print(TypeAdapter(DeferredToolResults).dump_json(decision).decode())
"""
{"calls":{},"approvals":{"call_1":true,"call_3":{"message":"Refunds over $100 need a manager.","kind":"tool-denied"}},"metadata":{}}
"""
```

Calls the decision leaves out stay pending, and the run ends with them as its `DeferredToolRequests` output, so include `DeferredToolRequests` in the agent's `output_type` if a decision may leave calls out. A decision that is not valid `DeferredToolResults`, or that names a call that is not pending (an answer sent for an earlier wait, say), fails the run with a [`UserError`][pydantic_ai.exceptions.UserError]. In a durable run the decision is already recorded, so a replay fails the same way.

## Timeouts

`timeout` bounds the wait. When it elapses, or the token is resolved with a failure instead of a decision, every pending approval is denied and every pending external call fails, with the message `No decision on this tool call arrived: <reason>`. The run then continues, so the model can tell the user what happened. Without a `timeout`, an in-process wait has no limit, and a durable one lasts as long as the engine allows, which on AWS Lambda is the execution timeout.

## Composition

`AwaitApproval` orders itself [innermost](custom.md#ordering), so every other deferred tool call handler gets the calls first, and the person is asked only about what they leave pending. A [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] that auto-approves read-only tools, or Pydantic AI Harness's [`ApprovalReviewer`](../harness/approval-reviewer.md) that approves low-risk calls on the person's behalf, still runs before the person is asked, wherever it is listed.

`AwaitApproval` emits no telemetry of its own: the run's [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] and [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] mark the wait in the event stream, and a durable engine's execution history shows it.

## Waiting for other external events

`AwaitApproval` is built on [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event], which you can call from other code that runs in the agent run, such as a capability hook, or outside a run, to wait for any JSON payload:

```python {title="wait_for_review.py"}
from datetime import timedelta

from pydantic_ai.external_events import resolve_external_event, wait_for_external_event


def send_review_request(token: str) -> None:
    resolve_external_event(token, {'verdict': 'publish'})  # (1)!


async def main():
    review = await wait_for_external_event(
        'editor_review', announce=send_review_request, timeout=timedelta(hours=4)
    )
    print(review)
    #> {'verdict': 'publish'}
```

1. Resolves the token right away to keep the example short. A real `announce` sends the token to the editor, and your review UI resolves it later.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

It raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] when the timeout elapses or the token is resolved with a failure. Don't call it from a tool that runs in a durable unit: the wait has to be recorded by the run itself, so it raises a `UserError` there.
