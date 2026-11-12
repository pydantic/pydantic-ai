---
description: "Ask for approvals, answers and external results inside a Pydantic AI agent run with RequestInput: answered in place, or waited for in process or suspended and recorded by a durable engine such as Temporal or AWS Lambda durable functions."
---

# Request Input

[`RequestInput`][pydantic_ai.capabilities.RequestInput] is a [capability](overview.md) that resolves [deferred tool calls](../deferred-tools.md), the ones that need a person's approval, an answer to a question or an external result, inside the agent run. It passes the pending calls and a resume token to your `ask` function, which either answers right away or hands the token to whoever will answer, and continues the same run with the answer: approved calls execute, denied ones are reported to the model, and external results and answers are filled in.

It works in any run. Without a durable engine, a wait happens in your process and [`resolve_external_event`][pydantic_ai.external_events.resolve_external_event] ends it. In a [durable](../durable_execution/overview.md) run on an engine that supports it, the engine records the answer and suspends the run while it waits, so the person can take days and a replay does not ask them again.

## How it relates to `HandleDeferredToolCalls`

`RequestInput` is a [`handle_deferred_tool_calls`](../deferred-tools.md#resolving-deferred-calls-with-a-handler) capability, like [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]. A `HandleDeferredToolCalls` handler can also wait for a person itself, by sending them a message and awaiting their reply, and in a plain process that works. In a durable run it falls short in two ways:

- The handler is not journaled. When the engine replays the run, it runs again and asks the person a second time.
- The execution stays alive while it waits, holding compute and running into the engine's invocation limits, such as AWS Lambda's 15-minute invocation timeout.

`RequestInput` asks through [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event], which a durable engine can implement: it records the token and the answer, and suspends the execution until the answer arrives. Without a durable engine it waits in process, which is what a `HandleDeferredToolCalls` handler would do too. Use `HandleDeferredToolCalls` for handlers that decide without waiting on anything outside the process, and `RequestInput` for those that do.

Ending the run with [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output and starting a new run with the answer remains an option, for a run that can't stay open while it waits and has no durable engine to keep it.

## `ask` and the answer

`ask` is called with the run context, the pending [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] and a resume token, and does one of two things:

- **Answer in place**: return the answer as [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]. A terminal prompt, a bot, a judge model or a policy does this.
- **Hand the token on**: deliver the question and the token to where a person will see it, such as a Slack message with Approve and Deny buttons, an email with a link, or a row in a database table an admin UI lists, and return `None`. The answer comes back separately, through code you write elsewhere, such as the webhook behind the buttons, which resolves the token with it:
    - Without a durable engine, call [`resolve_external_event(token, answer)`][pydantic_ai.external_events.resolve_external_event] in the process the run is waiting in, or [`fail_external_event(token, reason)`][pydantic_ai.external_events.fail_external_event] to end the wait without an answer.
    - On Temporal, the token names the workflow: send the answer with [`resolve_external_event(client, token, answer)`][pydantic_ai.durable_exec.temporal.resolve_external_event] from `pydantic_ai.durable_exec.temporal`, or end the wait with [`fail_external_event(client, token, reason)`][pydantic_ai.durable_exec.temporal.fail_external_event]. See [The same agent on Temporal](#the-same-agent-on-temporal).
    - On AWS Lambda durable functions, the token is a callback ID: send the answer with `SendDurableExecutionCallbackSuccess`, or end the wait with `SendDurableExecutionCallbackFailure`. See [Approvals](../harness/aws-lambda.md#approvals).

`ask` can be sync or async; a sync function runs in a worker thread, like a sync tool. An answer it returns goes through its JSON form, as a resolved answer does, so the run sees the same value either way. In a durable run, `ask` runs once, inside a durable unit that records its answer and is not run again on replay. The engine can retry it when it fails, so make it return quickly and keep it idempotent, for example by keying the message or database row on the token. Anything that can take a while, such as a person, should answer through the token.

`requests.kind(tool_call_id)` tells you what each pending call waits for: an `'approval'`, an answer to a `'question'` (such as Pydantic AI Harness's [`AskUser`](../harness/ask-user.md) asks), or an `'external'` result.

## Answering in place

This `ask` approves refunds up to $100 itself and denies the rest, recording who answered:

```python {title="request_input_policy.py"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai.capabilities import RequestInput


def refund_policy(ctx: RunContext, requests: DeferredToolRequests, token: str) -> DeferredToolResults:
    return requests.build_results(
        approvals={call.tool_call_id: call.args_as_dict()['amount'] <= 100 for call in requests.approvals},
        answered_by={call.tool_call_id: 'refund-policy' for call in requests.approvals},
    )


agent = Agent('{{ openai:medium }}', capabilities=[RequestInput(refund_policy)])


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: int) -> str:
    return f'Refunded ${amount} for order {order_id}'


result = agent.run_sync('Refund order 1234, it arrived broken.')
print(result.output)
#> I've refunded $50 for order 1234.
```

`answered_by` is reported in the run's [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent]. Pydantic AI does not check it.

## A round trip in one process

This FastAPI app hands the token on: it lists the pending approvals and resolves one when the approver posts a decision:

```python {title="request_input.py"}
import asyncio
from datetime import timedelta

from fastapi import FastAPI

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import RequestInput
from pydantic_ai.external_events import resolve_external_event

pending: dict[str, DeferredToolRequests] = {}  # (1)!


def ask(ctx: RunContext, requests: DeferredToolRequests, token: str) -> None:
    pending[token] = requests  # (2)!


agent = Agent(
    '{{ openai:medium }}',
    capabilities=[RequestInput(ask, timeout=timedelta(hours=1))],
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

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

The wait lives in the process: if the process restarts, the run and its wait are gone, and resolving the old token raises a [`UserError`][pydantic_ai.exceptions.UserError]. So does a token that is wrong, from another process, or already resolved, timed out or cancelled. For waits that must survive restarts, use a durable engine, or end the run with `DeferredToolRequests` output and store the conversation as [Pausing a conversation](../deferred-tools.md#pausing-a-conversation) shows.

## The same agent on AWS Lambda

Under AWS Lambda durable functions, add [`AWSLambdaDurability`](../harness/aws-lambda.md) and run the agent from a durable handler. The token is now a Lambda callback ID, the wait is recorded and suspends the execution, and the approval endpoint resolves the token through the Lambda API instead of `resolve_external_event`:

```python {title="request_input_lambda.py"}
import json
from datetime import timedelta
from typing import Any

import boto3
from aws_durable_execution_sdk_python import DurableContext, durable_execution
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import RequestInput

lambda_client = boto3.client('lambda')


def ask(ctx: RunContext, requests: DeferredToolRequests, callback_id: str) -> None:
    ...  # (1)!


agent = Agent(
    '{{ bedrock:medium }}',
    name='support',
    capabilities=[AWSLambdaDurability(), RequestInput(ask, timeout=timedelta(days=1))],
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

Run outside a durable handler, for example in local development, the same agent waits in process, and `resolve_external_event` resolves its token. An `ask` that answers in place, like the refund policy above, works the same on Lambda: its answer is recorded, and the execution continues without suspending.

## The same agent on Temporal

Under Temporal, add [`TemporalDurability`](../durable_execution/temporal.md) and run the agent from a workflow. The workflow records the answer and waits without holding a worker, for as long as the timeout allows. The token names the workflow, and the approval endpoint resolves it by signalling the workflow with [`resolve_external_event`][pydantic_ai.durable_exec.temporal.resolve_external_event] from `pydantic_ai.durable_exec.temporal`, which takes a Temporal client. The workflow declares no signal for it:

```python {title="request_input_temporal.py" test="skip"}
from datetime import timedelta

from temporalio import workflow
from temporalio.client import Client

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import RequestInput
from pydantic_ai.durable_exec.temporal import TemporalDurability, resolve_external_event


async def ask(ctx: RunContext, requests: DeferredToolRequests, token: str) -> None:
    ...  # (1)!


agent = Agent(
    '{{ openai:medium }}',
    name='support',
    capabilities=[TemporalDurability(), RequestInput(ask, timeout=timedelta(days=1))],
)


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: int) -> str:
    return f'Refunded ${amount} for order {order_id}'


@workflow.defn
class SupportWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await agent.run(prompt)
        return result.output


async def decide(client: Client, token: str, tool_call_ids: list[str], approved: bool) -> None:  # (2)!
    decision = {'approvals': {tool_call_id: approved for tool_call_id in tool_call_ids}}
    await resolve_external_event(client, token, decision)
```

1. Write the token and `requests` to where the approver will see them, such as a database table your admin UI lists, or a Slack message whose buttons carry the token. `ask` runs in a local activity on the worker running the workflow, so it can do I/O, and it gets the run's own context, so `ctx.deps` doesn't have to be serializable. Under the workflow sandbox it's the sandbox's copy of `ask`; see [Waiting for approvals and answers](../durable_execution/temporal.md#waiting-for-approvals-and-answers) for sharing its clients with the worker.
2. Runs in your approval endpoint, with a client connected with [`PydanticAIPlugin`][pydantic_ai.durable_exec.temporal.PydanticAIPlugin] to the workflow's namespace.

An `ask` that answers in place works the same on Temporal: its answer is recorded, and the workflow continues without waiting. See [Waiting for approvals and answers](../durable_execution/temporal.md#waiting-for-approvals-and-answers) for what the workflow history records.

## Engines

| Where the run executes | How it waits |
|---|---|
| A plain process | In the process; resolve the token with `resolve_external_event` |
| [Temporal](../durable_execution/temporal.md#waiting-for-approvals-and-answers) | A workflow condition; resolve the token with `resolve_external_event` from `pydantic_ai.durable_exec.temporal`, which signals the workflow |
| [AWS Lambda durable functions](../harness/aws-lambda.md#approvals) | A Lambda callback; resolve it with `SendDurableExecutionCallbackSuccess` |
| DBOS, Prefect, and other durable engines | Not supported yet: the engine refuses the wait, so a call that needs an answer fails the run with a `UserError`. End the run with `DeferredToolRequests` output instead |

The engine refuses rather than waiting in process because the wait has to be recorded. A wait in process would not be recorded. When the engine replays the run, for example after a worker restart, `ask` would run again: the person would be asked a second time and could answer differently than the answer the run already acted on, so the replay would diverge from the engine's recorded history (on DBOS, for example, steps recorded after the wait would no longer match). And when the process restarts, the wait and any answer sent to it would be lost.

Engine authors can add support as [Building a durable execution backend](../durable_execution/backends.md#waiting-for-external-events) describes.

## The answer

Answer with [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults], or resolve the token with its JSON form, keyed by the `tool_call_id`s of the pending calls in the `requests` passed to `ask`. Every key is optional:

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

- An approval is `true`, `false` (denied with the default message), a [`ToolDenied`][pydantic_ai.tools.ToolDenied] with the `message` the model is shown, or a [`ToolApproved`][pydantic_ai.tools.ToolApproved] with `override_args` to run the tool with instead of the model's arguments.
- An external call's result or a question's answer is any JSON value the model is shown, or a [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] (`"kind": "tool-failed"`) or [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] (`"kind": "model-retry"`).
- `metadata` reaches the tool as [`RunContext.tool_call_metadata`][pydantic_ai.tools.RunContext.tool_call_metadata].
- `answered_by` says who answered each call, and is reported in the run's `DeferredToolResultsEvent`.

Whoever can resolve the token decides the calls, including their arguments through `override_args`, the `metadata` tools read and the `answered_by` the run reports, so a tool must not treat either as proof of who approved. Pydantic AI checks only the answer's shape and that it names pending calls. In process and on AWS Lambda, treat the token as a bearer credential. On Temporal, the token is in the workflow history, so the boundary is permission to signal the workflow; see [Waiting for approvals and answers](../durable_execution/temporal.md#waiting-for-approvals-and-answers). Either way, authorize the approver in the code that resolves it.

From Python, build the answer as a `DeferredToolResults` and serialize it:

```python {title="request_input_decision.py"}
from pydantic import TypeAdapter

from pydantic_ai import DeferredToolResults, ToolDenied

decision = DeferredToolResults(approvals={'call_1': True, 'call_3': ToolDenied('Refunds over $100 need a manager.')})
print(TypeAdapter(DeferredToolResults).dump_json(decision).decode())
"""
{"calls":{},"approvals":{"call_1":true,"call_3":{"message":"Refunds over $100 need a manager.","kind":"tool-denied"}},"metadata":{},"answered_by":{}}
"""
```

An answer must cover every pending call unless `DeferredToolRequests` is in the agent's `output_type`: calls it leaves out stay pending, and the run ends with them as its `DeferredToolRequests` output, which fails the run when that output type isn't allowed. An answer that is not valid `DeferredToolResults`, or that names a call that is not pending (an answer sent for an earlier wait, say), fails the run with a [`UserError`][pydantic_ai.exceptions.UserError]. In a durable run the answer is already recorded, so a replay fails the same way.

## Timeouts

`name` names the wait in a durable engine's history, `'deferred_tool_calls'` by default: on AWS Lambda, `RequestInput(ask, name='approve_refund')` names the callback `{agent}__external_event.approve_refund` and the step that calls `ask` `{agent}__external_event.approve_refund.ask`. Keep it stable across deployments. On Temporal, it summarizes the local activity that calls `ask` and the timer that bounds the wait.

`timeout` bounds the wait. When it elapses, or the token is resolved with a failure instead of an answer, every pending approval is denied and every pending external call fails, with the message `No decision on this tool call arrived: <reason>`. The run then continues, so the model can tell the user what happened. In process, the timeout also bounds `ask`: an async `ask` still running when it elapses is cancelled. Without a `timeout`, an in-process wait has no limit, and a durable one lasts as long as the engine allows, which on Temporal is the workflow's execution timeout and on AWS Lambda the execution timeout.

## Composition

`RequestInput` orders itself [innermost](custom.md#ordering), so every other deferred tool call handler gets the calls first, and `ask` is only called with what they leave pending. A [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] that auto-approves read-only tools, Pydantic AI Harness's [`ApprovalReviewer`](../harness/approval-reviewer.md) that approves low-risk calls on the person's behalf, or its [`AskUser`](../harness/ask-user.md) that answers its own questions with its answerer, still runs before `ask`, wherever it is listed.

`RequestInput` emits no telemetry of its own: the run's [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] and [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] mark the wait in the event stream, and a durable engine's execution history shows it.

[`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] ends the run on a call to a tool declared with `requires_approval=True` or as external without running deferred tool call handlers ([#12686](https://github.com/pydantic/pydantic-ai/issues/12686)): for those, use [`run()`][pydantic_ai.agent.AbstractAgent.run], [`iter()`][pydantic_ai.agent.AbstractAgent.iter] or [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events].

## Waiting for other external events

`RequestInput` is built on [`wait_for_external_event`][pydantic_ai.external_events.wait_for_external_event], which you can call yourself to wait for any JSON payload, in or outside an agent run:

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

It raises [`ExternalEventError`][pydantic_ai.external_events.ExternalEventError] when the timeout elapses or the token is resolved with a failure. Don't call it from a tool that runs in a durable unit: the wait has to be recorded by the run itself, so it raises a `UserError` there. In the agent runs of a durable engine that doesn't support waiting, the engine refuses it with a `UserError` too.
