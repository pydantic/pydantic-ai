# AWS Lambda Durability

`AWSLambdaDurability` makes an agent resumable on [AWS Lambda durable
functions](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html). Every model
request, function tool call, MCP call, and dynamic-toolset resolution is checkpointed as a durable
step, so an invocation that times out, fails, or is retried continues from the last completed step
instead of repeating the work already paid for.

Lambda keeps a log of durable operations. When an execution resumes, the handler runs again from
the top and completed steps return their stored results rather than executing. Without
checkpointing, a resumed run would repeat every model request and tool call.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

![A Lambda handler decorated with durable_execution and durable_agent_handler runs an agent named support that has the AWSLambdaDurability capability. In the first invocation, each model request and tool call runs as a durable step, and Lambda records its result in the checkpoint log under a step name such as support__model.request; the invocation times out during the second model request. The resumed invocation runs the handler again from the top: the two completed steps return their recorded results without calling the model or the tool again, and only the unfinished model request runs, is recorded, and finishes the run.](https://pydantic.dev/docs/ai/img/harness/aws-lambda.svg)

## Installation

uv:

```bash
uv add "pydantic-ai-harness[aws-lambda]"
```

pip:

```bash
pip install "pydantic-ai-harness[aws-lambda]"
```

The quick start below uses a Bedrock provider model, which needs the Bedrock SDK from
`pydantic-ai-slim[bedrock]`:

uv:

```bash
uv add "pydantic-ai-harness[aws-lambda]" "pydantic-ai-slim[bedrock]"
```

pip:

```bash
pip install "pydantic-ai-harness[aws-lambda]" "pydantic-ai-slim[bedrock]"
```

## Quick start

Attach the capability when you build the agent, then adapt an async handler body with
`durable_agent_handler`:

```python {title="handler.py" test="skip" lint="skip"}
from typing import Any

from aws_durable_execution_sdk_python import DurableContext, durable_execution
from pydantic_ai import Agent

from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

agent = Agent(
    'bedrock:us.amazon.nova-pro-v1:0',
    name='support',
    capabilities=[AWSLambdaDurability()],
)


@agent.tool_plain
def get_weather(city: str) -> str:
    return f'It is sunny in {city}.'


@durable_execution
@durable_agent_handler
async def handler(event: dict[str, Any], context: DurableContext) -> str:
    result = await agent.run(str(event['prompt']))
    return result.output
```

Attach the capability at construction rather than per run, so the toolset wrapping and the
deployed step shape stay the same across invocations.

Deploy with a durable configuration and invoke a published version, since in-flight executions are
pinned to the version that started them:

```bash
aws lambda create-function \
  --function-name support-agent \
  --runtime python3.13 \
  --handler handler.handler \
  --role <ROLE_ARN> \
  --zip-file fileb://support-agent.zip \
  --timeout 300 --memory-size 1024 \
  --durable-config '{"ExecutionTimeout":3600,"RetentionPeriodInDays":7}'

aws lambda publish-version --function-name support-agent
```

!!! warning "A run is durable only inside the durable handler bridge"
    Attaching `AWSLambdaDurability` does not by itself make a run durable. Only a run entered through
    `durable_agent_handler` or `run_durable` is checkpointed. Calling `agent.run_sync(...)`, or
    awaiting the agent from your own `asyncio.run(...)`, produces a fully working but
    **non-durable** run, with no warning.

## Requirements

The agent needs a `name` (or `AWSLambdaDurability(name=...)`), and every leaf toolset needs a unique
`id`. Both are part of every step name, so both are checked when the agent is constructed: an agent
without a name raises a `UserError` from `Agent(...)`, as does a toolset that has no `id` or shares
one with another toolset. Tools registered directly on the agent live in a toolset whose id renders
as `<agent>`, so `@agent.tool_plain def get_weather` is checkpointed as
`{name}__function_toolset__<agent>.call_tool:get_weather`.

## How the sync handler and the async agent connect

Lambda's durable API is synchronous: `context.step(...)` blocks, and every step has to be created
on the thread Lambda invoked. An agent run is async. `durable_agent_handler` uses `run_durable` to
bridge the two: it hosts the async handler body on a background event loop and services its steps
on the Lambda handler thread, so all steps are created in one continuous sequence. A step body
hands its work back to the agent loop and blocks until it finishes, which keeps the loop free while
the handler thread waits.

The async body can await several agent runs and do async work between them, and all of those
sections share the same bridge and step sequence. Await the runs one after another, not
concurrently with `asyncio.gather` or tasks: a resumed execution matches checkpoints by the order
steps are reached, so concurrent runs that reach their steps in a different order on replay are
served each other's recorded results. A synchronous handler must call `run_durable` separately for
each async section, and concurrent calls are rejected.

Rules for the handler:

- `@durable_execution` must be the outermost decorator because its wrapper is what Lambda invokes.
  Reversing the order raises a `UserError` when the handler is defined.
- `run_durable` blocks the calling thread, so it cannot be called from inside a running event loop.
  Call it directly from a synchronous handler. It remains available when the bridge needs to be
  entered somewhere other than the top of a handler.
- Durable steps cannot nest. A tool that starts another durable agent run is rejected with an
  explanatory error rather than deadlocking.

The loop is reused across invocations of a warm execution environment, so loop-bound resources like
a provider's cached HTTP client stay valid between them. A run abandoned by a suspension or an error
is therefore cancelled before the handler returns, and `run_durable` waits `cancel_timeout` seconds
(5 by default) for it to unwind. Raise that for a workload whose cleanup is genuinely slow. If the
timeout expires, the abandoned cleanup keeps running on a retired loop for at most the retired
loop's grace period and can overlap the next warm invocation. Do not share mutable module-global
state between the agent run and the handler. An external side effect from cleanup, such as writing
to a store, releasing a shared lock, or emitting a metric, can also land during a later invocation.
Loop-bound resources are isolated: the next invocation gets a fresh loop, so the abandoned cleanup
cannot touch resources such as that invocation's provider HTTP client.

At the end of the grace period the abandoned cleanup is cancelled again, and the retired loop stops
a few iterations later (on Python 3.11, after jobs already submitted to its default executor
finish). A cleanup still running by then is left unfinished. A cleanup that cancels every task on
its loop is not cancelled again; the loop stops at the deadline either way.

Loop reuse has one more consequence: do not detach background work from a tool with
`asyncio.create_task()` or by leaving executor work unawaited. Detached work is not checkpointed and
is not covered by durable execution's guarantees, and it can outlive the invocation that started it.

## Approvals

To have a person approve a tool call, add [`RequestInput`](https://pydantic.dev/docs/ai/capabilities/request-input/) next to `AWSLambdaDurability`.
When the model calls a tool that needs approval (`requires_approval=True` or `ApprovalRequired`),
external execution (`CallDeferred`) or an answer to a question, the run creates a Lambda callback and
calls your `ask` function with the callback ID inside a step. `ask` can answer right away, and the
step records its answer, so the run continues without suspending and a replay reuses the answer. To
let a person answer, `ask` writes the callback ID where the approver will see it and returns `None`,
and the run suspends: the invocation returns, and no compute is billed while the execution waits.
The answer comes back through your approval endpoint, which sends the decision to the callback.
Lambda then invokes the function again, the handler replays from its checkpoints without calling the
model or `ask` again, and the run continues with the approved calls executed and the denied ones
reported to the model.

```python {title="approvals_handler.py"}
import json
from datetime import timedelta
from typing import Any

import boto3
from aws_durable_execution_sdk_python import DurableContext, durable_execution

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities import RequestInput
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, durable_agent_handler

lambda_client = boto3.client('lambda')


def ask(ctx: RunContext, requests: DeferredToolRequests, callback_id: str) -> None:
    ...  # Write the callback ID and `requests` where the approver will see them.


agent = Agent(
    'bedrock:us.amazon.nova-pro-v1:0',
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


def decide(callback_id: str, tool_call_ids: list[str], approved: bool) -> None:
    # Runs in your approval endpoint, with permission to call `lambda:SendDurableExecutionCallbackSuccess`.
    decision = {'approvals': {tool_call_id: approved for tool_call_id in tool_call_ids}}
    lambda_client.send_durable_execution_callback_success(
        CallbackId=callback_id, Result=json.dumps(decision)
    )
```

From the command line, send the same decision with:

```bash
aws lambda send-durable-execution-callback-success \
  --callback-id <CALLBACK_ID> \
  --cli-binary-format raw-in-base64-out \
  --result '{"approvals": {"<TOOL_CALL_ID>": true}}'
```

The decision is the JSON form of `DeferredToolResults` keyed by tool call ID: an approval can also
deny the call with a message for the model, or approve it with different arguments, and
[The answer](https://pydantic.dev/docs/ai/capabilities/request-input/#the-answer) lists every form. `send-durable-execution-callback-failure` ends
the wait without a decision, like a timeout: every pending approval is denied and every pending
external call fails, with a message saying why, and the run continues.

The callback's `timeout` is set from `RequestInput(timeout=...)`, rounded up to whole seconds, and
counts from when the callback is created, just before `ask` runs. Without one, the execution waits
until its `ExecutionTimeout`. When `ask` answers right away, nothing waits on the callback: it stays
open until its timeout, and resolving it has no effect on the run. Without a `timeout`, that callback
has none, so it is never closed; when `ask` may answer in place, set `timeout` so it expires. The `ask` step is a step like any
other: it is not run again on replay, `step_config` applies to it, and the default retry policy can
run it again when it fails, so make it idempotent. Both are named after `RequestInput(name=...)`,
`deferred_tool_calls` by default. An approved call's tool appears twice in the history as
`call_tool:<tool>` steps: once for the attempt that asked for approval, and once for the approved
run. A decision must cover every pending call unless `DeferredToolRequests` is in the agent's
`output_type`. A checkpointed tool cannot wait for a decision itself, because it runs inside a step;
that is rejected with a `UserError`.

Questions from Pydantic AI Harness's [`AskUser`](https://pydantic.dev/docs/ai/harness/ask-user/) wait the same way: its answerer runs in the `ask`
step, and an answerer that takes a resume token can hand the question on and let the answer arrive
through the callback.

Outside a durable handler, as in local development, the same agent waits in process instead, and
`resolve_external_event` from `pydantic_ai.external_events` resolves its token: see
[A round trip in one process](https://pydantic.dev/docs/ai/capabilities/request-input/#a-round-trip-in-one-process). A `HandleDeferredToolCalls` handler
that waited for the approver itself would hold the invocation, and its 15-minute timeout, for the
whole wait, and would ask again on every replay; ending the run with `DeferredToolRequests` output and
starting a second run with the results also still works.

## What gets checkpointed

Step names are built from the agent's `name` and each toolset's `id`:

| Step name | Operation |
|---|---|
| `{name}__model.request` | one model request segment |
| `{name}__model.request_stream` | one streamed model request segment |
| `{name}__model.compact_messages` | one model message-compaction operation |
| `{name}__model.cancel_suspended_response` | tearing down a suspended response |
| `{name}__capability__{capability_id}.{operation}` | an operation contributed by another capability |
| `{name}__function_toolset__{id}.validate_args` | validating a function tool call's arguments |
| `{name}__function_toolset__{id}.call_tool:{tool}` | a function tool call |
| `{name}__mcp_server__{id}.get_tools` | listing an MCP server's tools |
| `{name}__mcp_server__{id}.get_instructions` | an MCP server's instructions |
| `{name}__mcp_server__{id}.call_tool` | an MCP tool call |
| `{name}__dynamic_toolset__{id}.get_tools` | resolving a dynamic toolset |
| `{name}__dynamic_toolset__{id}.validate_args` | validating a dynamic toolset call's arguments |
| `{name}__dynamic_toolset__{id}.call_tool:{tool}` | a dynamic toolset's tool call |
| `{name}__event_stream_handler` | one event delivered to an `event_stream_handler` |
| `{name}__external_event.deferred_tool_calls` | the callback a [`RequestInput`](#approvals) wait suspends on |
| `{name}__external_event.deferred_tool_calls.ask` | calling `ask` with that callback's ID, and its answer |
| `{name}__external_event.ask_user_question` | the callback an `AskUser` question waits on |
| `{name}__external_event.ask_user_question.ask` | calling the `AskUser` answerer, and its answer |

A model operation that does not use the agent's default model records its model id in the step name
(for example, `{name}__model.request.{model_id}`), so the execution history shows which model each
checkpoint came from. The default model keeps the plain, suffix-less name.

A step served from its checkpoint emits no telemetry: [instrumentation](https://pydantic.dev/docs/ai/integrations/logfire/)
records a model request's or tool call's span, and a model request's token and cost metrics, in the
invocation that ran it, and an agent run's span in each invocation that did work for it. A resumed
execution's traces show only the work that invocation did, and costs are counted once.

## Constraints

- **Steps are at least once, and retried by default.** A step is checkpointed after it runs, so an
  interruption between a tool's side effect and its checkpoint re-runs the tool when the execution
  resumes. On top of that, the SDK's default retry policy is six attempts with exponential backoff
  (5s to 60s), applied to every model request and every tool call. Keep tool side effects
  idempotent. `AT_MOST_ONCE_PER_RETRY` alone is not enough to make a tool run once: it prevents
  re-execution after an interruption *within* an attempt, but the retry policy still starts further
  attempts that do execute the body. For a tool that must not repeat, set both:

    ```python {names="defined"}
    from aws_durable_execution_sdk_python.config import StepSemantics
    from aws_durable_execution_sdk_python.retries import RetryPresets

    metadata={'aws_lambda': {'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY,
                             'retry_strategy': RetryPresets.none()}}
    ```

- **Retries stack.** Pydantic AI and provider clients have their own retry logic. Leaving those
  enabled alongside the step retry policy multiplies the attempts and mishandles `Retry-After`;
  disable one side.
- **Tool calls run one at a time.** A step's identity comes from the order steps are reached, so
  concurrently scheduled tool calls could claim each other's checkpoints when the execution
  resumes. Inside a durable handler the run is switched to sequential tool execution. Outside one
  the agent keeps its configured parallelism.
  [`DynamicWorkflow`](../dynamic_workflow/README.md#running-under-other-durable-engines) sub-agents are the
  exception: they run concurrently, each in a branch of a Lambda `parallel` operation, which numbers
  its steps on its own.
- **Changing the shape of the run breaks in-flight executions.** A resumed execution matches
  checkpoints by the order operations are reached, so anything that changes the number or order of
  steps breaks executions started under the old code: adding or removing a tool or MCP server,
  flipping a `metadata={'aws_lambda': False}` opt-out, or adding an `event_stream_handler` (which
  switches the model step to `model.request_stream` and adds handler steps). SDK 2.0 and later
  compare each step's type and name on replay, so these changes, and renaming the agent, a toolset
  `id` or a tool, or changing the id a non-default model runs under (a `models=` key or the `model=`
  passed to `agent.run`), which is [part of its step names](#what-gets-checkpointed), fail the
  execution with `NonDeterministicExecutionError`. On SDK 1.x a mismatch is not detected: a step is
  served another step's recorded result, which usually fails to decode but can return the wrong
  data. Either way, deploy under a new published version and let in-flight executions drain on the
  old one.
- **Step results must survive the SDK serializer.** Results are checkpointed through the Lambda
  SDK's serializer. Tool results are encoded by Pydantic first, so structured returns such as
  `ToolReturn` and `BinaryContent` round-trip; a value Pydantic cannot serialize does not.
- **Events cannot leave a running durable execution.** `run_stream` and `iter` do work inside the
  handler and are checkpointed normally, but a durable execution returns a single value when it
  completes, so there is no channel to stream tokens to a caller while it runs. An
  `event_stream_handler` works: model events are handled live inside the model step and each
  agent-level event is checkpointed in its own step.
- **`ctx.enqueue()` is not available inside a durable step**, whether a checkpointed tool or an
  `event_stream_handler` (which runs inside the model step for model events and its own step for
  agent events), because a resumed execution serves the recorded step output and would drop the
  enqueued messages. Enqueue from handler-level code instead.
- **Checkpoints hold run data.** Model responses, tool results, and the error messages of failed
  steps (which can include a provider's error response body) are stored in the execution's durable
  state for the `RetentionPeriodInDays` of the durable configuration. Anyone who can read the
  execution history can read them, so keep the retention period short for sensitive workloads.
- **Budgets.** A durable execution allows 30,000 operations and 100 MB of cumulative checkpointed
  state. A turn costs one model step plus one step per tool call, so the operation budget is
  generous, but large tool results consume the state budget: return a reference (an S3 key, say)
  rather than a blob.
- **Batched requests wait in the invocation.** Under [`BatchModelRequests`](https://pydantic.dev/docs/ai/core-concepts/batch/#durable-execution), submitting a
  request and each check on its batch are durable steps, but the waits between checks are plain
  sleeps: the invocation has to stay up for as long as the batch takes, within its timeout, and a
  recovered run sleeps again through the waits it already did.

## Per-tool configuration

Tool metadata under the `aws_lambda` key configures that tool's step. It accepts the `StepConfig`
fields `retry_strategy`, `step_semantics`, and `serdes`:

```python {test="skip" lint="skip"}
from aws_durable_execution_sdk_python.config import StepSemantics
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset(id='billing')


@toolset.tool_plain(metadata={'aws_lambda': {'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY}})
def charge_card(amount: int) -> str:
    return f'charged {amount}'
```

`metadata={'aws_lambda': False}` opts a tool out of checkpointing entirely, so it runs inline on
every attempt. Use it for cheap, side-effect-free tools whose result is not worth a checkpoint. MCP
tools cannot opt out, because they perform I/O that must not re-run when the execution resumes.

`AWSLambdaDurability(step_config=...)` sets the base configuration for every step. Per-tool metadata
overrides it key by key, so a tool that sets only `step_semantics` keeps the base `retry_strategy`.

## Composition with other capabilities

`AWSLambdaDurability` orders itself innermost, so any other capability's contribution to a model
request is already applied inside the durable step. Attach it alongside other capabilities as usual.

## Further reading

- [AWS Lambda durable functions](https://docs.aws.amazon.com/lambda/latest/dg/durable-functions.html)
- [AWS Durable Execution SDK for Python](https://github.com/aws/aws-durable-execution-sdk-python)
- [Pydantic AI durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/)
- [AWS Lambda Durability source code](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/aws_lambda/)
- [Pydantic AI Harness version policy](https://pydantic.dev/docs/ai/harness/#version-policy)
