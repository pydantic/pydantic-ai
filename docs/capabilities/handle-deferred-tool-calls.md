---
description: "Resolve deferred tool calls inside a Pydantic AI run with HandleDeferredToolCalls: approve, deny or answer them from a handler, or hand a resume token to a person and wait in process or suspended and recorded by a durable engine such as AWS Lambda durable functions."
---

# Handle Deferred Tool Calls

[`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] is a [capability](overview.md) that adds no tool for the model. It resolves the [deferred tool calls](../deferred-tools.md) that other tools raise: tools that [require approval](../deferred-tools.md#human-in-the-loop-tool-approval), [external tools](../deferred-tools.md#external-tool-execution) and [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] calls, and the questions Pydantic AI Harness's [`AskUser`](../harness/ask-user.md) asks with `answerer=None`. Without it, the run ends with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output. With it, your handler resolves the calls and the same run continues: approved calls execute, denied ones are reported to the model, and external results and answers are filled in.

```python {title="handle_deferred_tool_calls.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults


async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    return requests.build_results(approve_all=True)  # (1)!


agent = Agent('{{ openai:medium }}', capabilities=[HandleDeferredToolCalls(handle_deferred)])
```

1. Auto-approve every call that's waiting on approval. Real handlers typically inspect `requests.approvals` and `requests.calls` and decide per call: check a policy, ask an operator, or execute an external call.

## Two forms of handler

The handler is called with the run context and the pending [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], and takes one of two forms:

- **`handler(ctx, requests)` answers in place.** It returns [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] with results for some or all pending calls, or `None` to decline. The next deferred tool call handler gets what it leaves, and calls that no handler resolves end the run as `DeferredToolRequests` output as usual. It can be sync or async.
- **`handler(ctx, requests, token)` can also hand a resume token on.** A third required positional parameter selects this form. The handler answers in place the same way, or hands the token to whoever will answer and returns `None`. The run then waits until the token is resolved with the answer.

Use the first form when the handler decides without waiting on anything outside the process. In a [durable](../durable_execution/overview.md) run, it isn't recorded: when the engine replays the run, the handler runs again. Use the second form when the decision comes from outside, such as a person, or when a durable engine should record the answer. The handler then runs through [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event], which a durable engine can implement. The engine records the token and the answer, and suspends the execution until the answer arrives. Without a durable engine, the wait happens in your process.

Ending the run with `DeferredToolRequests` output and starting a new run with the answer, as [Deferred Tools](../deferred-tools.md) shows, remains an option when the run can't stay open while it waits and has no durable engine to keep it.

## A handler that takes a token

A handler that takes a token does one of two things:

- **Answer in place**: return the answer as `DeferredToolResults`, as a terminal prompt, a bot, a judge model or a policy does.
- **Hand the token on**: deliver the question and the token to where a person will see it and return `None`. That could be a Slack message with Approve and Deny buttons, an email with a link, or a row in a database table an admin UI lists. The answer comes back separately, through code you write elsewhere, such as the webhook behind the buttons, which resolves the token with it:
    - Without a durable engine, call [`resolve_external_event(token, answer)`][pydantic_ai.external_events.resolve_external_event] in the process the run is waiting in. To end the wait without an answer, call [`fail_external_event(token, reason)`][pydantic_ai.external_events.fail_external_event].
    - On AWS Lambda durable functions, the token is a callback ID: send the answer with `SendDurableExecutionCallbackSuccess`, or end the wait with `SendDurableExecutionCallbackFailure`. See [Approvals](../harness/aws-lambda.md#approvals).

The handler can be sync or async; a sync function runs in a worker thread, like a sync tool. An answer it returns goes through its JSON form, as a resolved answer does, so the run sees the same value either way. In a durable run, the handler runs once, inside a durable unit that records its answer and isn't run again on replay. The engine can retry it when it fails, so make it return quickly and keep it idempotent, for example by keying the message or database row on the token. Anything that can take a while, such as a person, should answer through the token.

A token handler that wants to leave every call to later handlers returns empty results, `requests.build_results()`. Returning `None` would make the run wait for the token.

`requests.kind(tool_call_id)` tells you what each pending call waits for: an `'approval'`, an answer to a `'question'` (such as those `AskUser` asks), or an `'external'` result.

## Answering in place

This handler approves refunds up to $100 itself and denies the rest, recording who answered:

```python {title="deferred_tool_policy.py"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls


def refund_policy(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
    return requests.build_results(
        approvals={call.tool_call_id: call.args_as_dict()['amount'] <= 100 for call in requests.approvals},
        answered_by={call.tool_call_id: 'refund-policy' for call in requests.approvals},
    )


agent = Agent('{{ openai:medium }}', capabilities=[HandleDeferredToolCalls(refund_policy)])


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: int) -> str:
    return f'Refunded ${amount} for order {order_id}'


result = agent.run_sync('Refund order 1234, it arrived broken.')
print(result.output)
#> I've refunded $50 for order 1234.
```

`answered_by` is reported in the run's [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent]. Pydantic AI does not check it.

To have a durable engine record the policy's answer, so a replay reuses it instead of deciding again, give the handler a `token` parameter: `def refund_policy(ctx, requests, token)`. It still answers in place.

## A round trip in one process

This FastAPI app hands the token on: it lists the pending approvals and resolves one when the approver posts a decision:

```python {title="deferred_tool_approval_endpoint.py"}
import asyncio
from datetime import timedelta

from fastapi import FastAPI

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.external_events import resolve_external_event

pending: dict[str, DeferredToolRequests] = {}  # (1)!


def ask(ctx: RunContext, requests: DeferredToolRequests, token: str) -> None:
    pending[token] = requests  # (2)!


agent = Agent(
    '{{ openai:medium }}',
    capabilities=[HandleDeferredToolCalls(ask, timeout=timedelta(hours=1))],
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
2. Hands the question and the token on, and returns `None`: the answer arrives through the token.
3. Ends the wait the run is in. The run continues with the decision: the approved refund runs, and the model writes its reply.
4. Stands in for the approver, who opens the approvals page and clicks Approve, which posts to `/approvals/{token}`.

_(To run this example, add `asyncio.run(main())`; no other changes are needed.)_

The wait lives in the process: if the process restarts, the run and its wait are gone, and resolving the old token raises a [`UserError`][pydantic_ai.exceptions.UserError]. So does a token that is wrong, from another process, or already resolved, timed out or cancelled. For waits that must survive restarts, use a durable engine, or end the run with `DeferredToolRequests` output and store the conversation as [Pausing a conversation](../deferred-tools.md#pausing-a-conversation) shows.

## The same agent on AWS Lambda

Under AWS Lambda durable functions, add [`AWSLambdaDurability`](../harness/aws-lambda.md) and run the agent from a durable handler. The token is now a Lambda callback ID. The wait is recorded and suspends the execution, and the approval endpoint resolves the token through the Lambda API instead of `resolve_external_event`:

```python {title="deferred_tool_calls_lambda.py"}
import json
from datetime import timedelta
from typing import Any

import boto3
from aws_durable_execution_sdk_python import DurableContext, durable_execution
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import HandleDeferredToolCalls

lambda_client = boto3.client('lambda')


def ask(ctx: RunContext, requests: DeferredToolRequests, callback_id: str) -> None:
    ...  # (1)!


agent = Agent(
    '{{ bedrock:medium }}',
    name='support',
    capabilities=[AWSLambdaDurability(), HandleDeferredToolCalls(ask, timeout=timedelta(days=1))],
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

Run outside a durable handler, for example in local development, the same agent waits in process, and `resolve_external_event` resolves its token. A token handler that answers in place works the same on Lambda: its answer is recorded, and the execution continues without suspending.

## Engines

How a handler that takes a token waits depends on where the run executes. A handler without a token is called the same way everywhere.

| Where the run executes | How it waits |
|---|---|
| A plain process | In the process; resolve the token with `resolve_external_event` |
| [AWS Lambda durable functions](../harness/aws-lambda.md#approvals) | A Lambda callback; resolve it with `SendDurableExecutionCallbackSuccess` |
| Temporal, DBOS, Prefect, and other durable engines | Not supported yet: the engine refuses the wait, so a call that needs an answer fails the run with a `UserError`. Use a handler without a token, or end the run with `DeferredToolRequests` output instead |

The engine refuses rather than waiting in process because the wait has to be recorded. When the engine replays the run, for example after a worker restart, an unrecorded handler would run again: the person would be asked a second time and could answer differently than the answer the run already acted on. The replay would then diverge from the engine's recorded history (on DBOS, for example, steps recorded after the wait would no longer match). And when the process restarts, the wait and any answer sent to it would be lost.

Engine authors can add support as [Building a durable execution backend](../durable_execution/backends.md#waiting-for-external-events) describes.

## The answer

Answer with [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults], or resolve the token with its JSON form. Key it by the `tool_call_id`s of the pending calls in the `requests` passed to the handler. Every key is optional:

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
  "metadata": {"call_1": {"ticket": "OPS-12"}},
  "answered_by": {"call_1": "alice"}
}
```

- An approval is one of:
    - `true`;
    - `false`, which denies the call with the default message;
    - a [`ToolDenied`][pydantic_ai.tools.ToolDenied] with the `message` the model is shown;
    - a [`ToolApproved`][pydantic_ai.tools.ToolApproved] with `override_args` to run the tool with instead of the model's arguments.
- An external call's result or a question's answer is any JSON value the model is shown, or a [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] (`"kind": "tool-failed"`) or [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] (`"kind": "model-retry"`).
- `metadata` reaches the tool as [`RunContext.tool_call_metadata`][pydantic_ai.tools.RunContext.tool_call_metadata].
- `answered_by` says who answered each call, and is reported in the run's `DeferredToolResultsEvent`.

Whoever can resolve the token decides the calls. That includes their arguments, through `override_args`, the `metadata` tools read, and the `answered_by` the run reports. So a tool must not treat `metadata` or `answered_by` as proof of who approved. Pydantic AI checks only the answer's shape and that it names pending calls. Treat the token as a bearer credential, in process as on an engine, and authorize the approver in the code that resolves it.

From Python, build the answer as a `DeferredToolResults` and serialize it:

```python {title="deferred_tool_decision.py"}
from pydantic import TypeAdapter

from pydantic_ai import DeferredToolResults, ToolDenied

decision = DeferredToolResults(approvals={'call_1': True, 'call_3': ToolDenied('Refunds over $100 need a manager.')})
print(TypeAdapter(DeferredToolResults).dump_json(decision).decode())
"""
{"calls":{},"approvals":{"call_1":true,"call_3":{"message":"Refunds over $100 need a manager.","kind":"tool-denied"}},"metadata":{},"answered_by":{}}
"""
```

An answer doesn't have to cover every pending call. The calls it leaves out stay pending, and the run ends with them as its `DeferredToolRequests` output. That fails the run unless `DeferredToolRequests` is in the agent's `output_type`. An answer through the token that is not valid `DeferredToolResults`, or that names a call that is not pending (an answer sent for an earlier wait, say), fails the run with a [`UserError`][pydantic_ai.exceptions.UserError]. In a durable run the answer is already recorded, so a replay fails the same way.

## Timeouts and names

`timeout` and `name` configure the wait, so they need a handler that takes a token. Passing them with a handler that doesn't raises a [`UserError`][pydantic_ai.exceptions.UserError].

`timeout` bounds the wait. The wait can end without an answer in two ways: the timeout elapses, or the token is resolved with a failure. Either way, every pending approval is denied and every pending external call fails, with the message `No decision on this tool call arrived: <reason>`. The run then continues, so the model can tell the user what happened. In process, the timeout also bounds the handler: an async handler still running when it elapses is cancelled. Without a `timeout`, an in-process wait has no limit, and a durable one lasts as long as the engine allows, which on AWS Lambda is the execution timeout.

`name` names the wait in a durable engine's history, `'deferred_tool_calls'` by default. On AWS Lambda, `HandleDeferredToolCalls(ask, name='approve_refund')` names the callback `{agent}__external_event.approve_refund` and the step that calls `ask` `{agent}__external_event.approve_refund.ask`. Keep it stable across deployments.

## Composition

Several deferred tool call handlers can share an agent. Each gets the calls the ones before it left pending.

- A handler without a token keeps the position you list it in.
- A handler that takes a token orders itself [innermost](custom.md#ordering), so it is only asked about what every other handler left pending, wherever it is listed. Those other handlers include a `HandleDeferredToolCalls` that auto-approves read-only tools, Pydantic AI Harness's [`ApprovalReviewer`](../harness/approval-reviewer.md) that approves low-risk calls on the person's behalf, and its [`AskUser`](../harness/ask-user.md) that answers its own questions with its answerer.

`HandleDeferredToolCalls` emits no telemetry of its own. The run's [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] and [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] mark the handler in the event stream, and a durable engine's execution history shows a wait.

[`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] ends the run on a call to a tool declared with `requires_approval=True` or as external without running deferred tool call handlers ([#12686](https://github.com/pydantic/pydantic-ai/issues/12686)). For those tools, use [`run()`][pydantic_ai.agent.AbstractAgent.run], [`iter()`][pydantic_ai.agent.AbstractAgent.iter] or [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events].

See [Resolving deferred calls with a handler](../deferred-tools.md#resolving-deferred-calls-with-a-handler) for how this fits into the wider deferred-tools flow.

## Migrating from `RequestInput`

`RequestInput` is deprecated. `RequestInput(ask, timeout=..., name=...)` is `HandleDeferredToolCalls(ask, timeout=..., name=...)`: an `ask` that takes `(ctx, requests, token)` is a handler that takes a token, and works the same. `RequestInput` passes the token whatever `ask`'s signature, but `HandleDeferredToolCalls` only passes it to a handler whose third positional parameter is required. If `ask` declares the token with a default or as `*args`, make it a required parameter when you switch, or the handler is called without a token and its `None` declines instead of waiting. `RequestInput` keeps working, with a deprecation warning, until the next major version.

## Waiting for other external events

A handler that takes a token is built on [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event], which you can call yourself to wait for any JSON payload, in or outside an agent run:

```python {title="wait_for_review.py"}
from datetime import timedelta

from pydantic_ai.external_events import resolve_external_event, wait_for_external_event


def send_review_request(token: str) -> None:
    resolve_external_event(token, {'verdict': 'publish'})  # (1)!


async def main():
    review = await wait_for_external_event(
        'editor_review', ask=send_review_request, timeout=timedelta(hours=4)
    )
    print(review)
    #> {'verdict': 'publish'}
```

1. Resolves the token right away to keep the example short. A real `ask` sends the token to the editor and returns `None`, and your review UI resolves it later. An `ask` that can answer itself returns the payload instead.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

`wait_for_external_event` raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] when the timeout elapses or the token is resolved with a failure. Don't call it from a tool that runs in a durable unit: the wait has to be recorded by the run itself, so it raises a `UserError` there. In the agent runs of a durable engine that doesn't support waiting, the engine refuses it with a `UserError` too.
