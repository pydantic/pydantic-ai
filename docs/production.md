---
description: "Take a Pydantic AI agent to production: crash recovery, human approval, conversation state, cost limits, provider failover, tracing, evals, testing, frontends and deployment."
---

# Going to Production

Everything on this page is open source and runs on your infrastructure. [Pydantic AI](index.md), [Pydantic Evals](evals.md) and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) are MIT-licensed Python packages that run in your own process, and durable execution runs on an engine you operate (or that engine's own cloud). There is no Pydantic-hosted agent runtime. Two hosted products are optional: [Pydantic Logfire](logfire.md) for observability, where any OpenTelemetry backend works instead, and the [Pydantic AI Gateway](gateway.md) for one key across providers, where calling providers directly works instead.

The rest of the page answers the questions teams ask before shipping an agent, each with the short answer and a link to the page that covers it in full.

| Before you ship, you want to know... | Short answer | Where it lives |
|---|---|---|
| [What happens when the process crashes mid-run?](#crash) | Run the agent on a durable execution engine | Core, plus an engine you run |
| [How does a human approve a risky action, and the run resume days later?](#approval) | Deferred tools end the run; a later run resumes it | Core |
| [How do I keep conversation state?](#state) | Store the message history, or let `StepPersistence` do it | Core, or Harness |
| [How do I bound cost and runaway loops?](#cost) | `UsageLimits` per run, `SpendLimits` per day, tenant or month | Core, and Harness |
| [What happens when a provider is down?](#outages) | `FallbackModel`, transport retries, or Gateway failover | Core, or hosted Gateway |
| [How do I see what happened?](#observability) | OpenTelemetry traces of every run | Core, to any OTel backend or Logfire |
| [How do I know it's still good?](#evals) | Evals on datasets, on trajectories and on live traffic | Pydantic Evals |
| [How do I test without API calls?](#testing) | `TestModel`, `FunctionModel` and `Agent.override` | Core |
| [How do I limit what the agent can touch?](#safety) | Approval, guardrails, a tool call judge, sandboxes | Core, and Harness |
| [How do I serve it to a frontend?](#frontend) | AG-UI or Vercel AI streams from any ASGI app | Core |
| [How do I deploy it?](#deploy) | As your own Python process: a web app, a worker or a function | Your infrastructure |

## Which pieces for which job

Start with the smallest set that does the job, and add the rest as you need it:

| Job | Install | Start with | Add for production |
|---|---|---|---|
| A typed agent behind an API | `pydantic-ai`, or `pydantic-ai-slim` with your [provider's extra](install.md#slim-install) | [Agents](agent.md), [Output](output.md), [Dependencies](dependencies.md) | [Usage limits](agent.md#usage-limits), a [fallback model](models/overview.md#fallback-model), [instrumentation](logfire.md), [tests](testing.md), a [UI adapter](ui/overview.md) |
| A durable workflow | `pydantic-ai-slim` with one engine's extra, such as `[dbos]`, `[temporal]` or `[prefect]` | [Durable execution](durable_execution/overview.md) and one engine's page | [Deferred tools](deferred-tools.md) for approval, [Persistence](persistence.md) for state between runs |
| A coding or long-running agent | `pydantic-ai-harness` | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) and [Coder](https://pydantic.dev/docs/ai/harness/coder/) | A [sandbox](https://pydantic.dev/docs/ai/harness/#execution-environments), [`SpendLimits`](https://pydantic.dev/docs/ai/harness/spend/), [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/), [Harness durable execution](https://pydantic.dev/docs/ai/harness/durable-execution/) |
| An eval suite | `pydantic-evals` (included in `pydantic-ai`) | [Evals quick start](evals/quick-start.md) | [Trajectory evaluators](evals/evaluators/agentic.md), [online evaluation](evals/online-evaluation.md) |

## What happens when the process crashes mid-run? {#crash}

Without durable execution, a run in flight dies with the process: you start it again, or continue it from the last snapshot [`StepPersistence`](#checkpointer) saved. With it, each model request and tool call runs as a durable step on an engine that journals the result, so a restarted run replays completed steps from the journal and continues from the one that was in flight, without paying for completed model requests again or re-running completed tool calls.

You attach durability to an ordinary agent as a [capability](capabilities/overview.md), and the run is durable when it executes inside the engine's workflow:

```python {title="production_durable.py" test="skip"}
from dbos import DBOS, DBOSConfig

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSDurability

DBOS(config=DBOSConfig(name='support', system_database_url='postgresql://...'))

agent = Agent('anthropic:claude-fable-5-1', name='support', capabilities=[DBOSDurability()])


@DBOS.workflow()
async def answer(question: str) -> str:
    result = await agent.run(question)  # resumes from its last completed step after a crash
    return result.output
```

A step that was in flight when the process died runs again from its start, so a tool with side effects should be safe to repeat, for example by passing an idempotency key to the API it calls.

Pydantic AI integrates with [Temporal](durable_execution/temporal.md), [DBOS](durable_execution/dbos.md), [Prefect](durable_execution/prefect.md), [Restate](durable_execution/restate.md), [AWS Lambda durable functions](https://pydantic.dev/docs/ai/harness/aws-lambda/), [Kitaru](durable_execution/kitaru.md), [Apache Airflow](durable_execution/airflow.md) and [Absurd](https://pydantic.dev/docs/ai/harness/absurd/), and the [backend builder](durable_execution/backends.md) integrates any other engine. DBOS runs in your process against Postgres or SQLite; Temporal and Restate run a server that your workers or services connect to; Prefect and Airflow fit when you already orchestrate with them. Each engine's page says which calls it wraps for you and which you mark yourself, and an agent takes [one engine at a time](durable_execution/overview.md).

### What you get compared to a checkpointer {#checkpointer}

In frameworks built around a checkpointer, one mechanism saves state after each step and covers crash recovery, pausing for a human and resuming a thread. Pydantic AI splits those jobs between pieces that compose on one agent:

- **Crash recovery inside a run** comes from durable execution, at the granularity of a single model request or tool call. A crash in the middle of a long tool loop resumes at the model request or tool call that was running, with everything before it replayed from the journal.
- **Saving and continuing between runs** comes from [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) (Harness): it saves a continuable snapshot after every completed tool cycle and when a run fails, so you can continue a run from its latest snapshot or fork it from any saved one, later and in another process if you like, with in-memory, file, SQLite or MongoDB stores, or your own.
- **Knowing whether a side effect happened** comes from the `StepPersistence` tool-effect ledger, which records when each tool call started and finished, so after a crash you can tell a call that completed from one whose effect is unknown, before deciding to run it again.
- **Pausing for a human** needs no live process at all: [deferred tools](#approval) end the run cleanly and a later run picks it up.

`StepPersistence` is aware of durable execution: its writes are durable operations, so replaying a durable run does not write its events twice. A snapshot holds the run's message history, which is the state an agent run is made of; state that a capability keeps outside the messages, a workspace's files and resuming in the middle of a step are not part of it, and that last one is what durable execution is for. [Persistence](persistence.md) lays out which piece answers which question.

## How does a human approve a risky action, and the run resume days later? {#approval}

Mark the tool as needing approval. When the model calls it, the run ends with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output listing the pending calls, and nothing stays in memory while you wait. Store the message history, ask a person, and when they answer, minutes or weeks later, start a new run from that history with their decision:

```python {title="production_approval.py"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ModelMessagesTypeAdapter

agent = Agent('anthropic:claude-fable-5-1', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: float) -> str:
    return f'Refunded {amount} on order {order_id}'


async def start(prompt: str) -> bytes:
    result = await agent.run(prompt)
    if isinstance(result.output, DeferredToolRequests):
        ...  # ask a person about each call in result.output.approvals
    return result.all_messages_json()  # store this in your database


async def resume(stored: bytes, tool_call_id: str, approved: bool) -> str | DeferredToolRequests:
    result = await agent.run(
        message_history=ModelMessagesTypeAdapter.validate_json(stored),
        deferred_tool_results=DeferredToolResults(approvals={tool_call_id: approved}),
    )
    return result.output
```

The same mechanism hands a call to a frontend or a background worker to execute, and an approval can come back with edited arguments or a denial message the model sees. The [UI adapters](ui/overview.md) carry approvals to and from the browser, and the [CLI](cli.md) prompts for them. When the approver is in the same process, a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves calls inline without ending the run. All of it is on [Deferred Tools](deferred-tools.md).

For an agent with nobody watching, [`ToolCallJudge`](https://pydantic.dev/docs/ai/harness/tool-call-judge/) (Harness) asks a second model whether a call may run, and can escalate the calls it is unsure about to a person.

!!! warning "Approval guards against the model, not the client"
    An approval submitted by a client is trusted as given. Authenticate the endpoint, and check authorization inside the tool for sensitive actions; see the [trust boundary for client-supplied history](message-history.md#trust-boundary-for-client-supplied-history).

## How do I keep conversation state? {#state}

A conversation's state is its message history. Store [`result.new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages] under a [`conversation_id`](message-history.md#correlating-runs-with-run_id-and-conversation_id) in your own database, serialized with [`ModelMessagesTypeAdapter`](message-history.md#storing-and-loading-messages-to-json), and pass the history back as `message_history` on the next run. A `jsonb` column is enough; no schema migration is needed when Pydantic AI adds a message part.

To skip writing that code, [`StepPersistence`](https://pydantic.dev/docs/ai/harness/step-persistence/) (Harness) stores runs for you, with continue and fork. For what an agent should remember about a user across conversations, [`Memory`](https://pydantic.dev/docs/ai/harness/memory/) (Harness) keeps notes the agent writes itself. [Persistence](persistence.md) compares all of these, and [compaction](capabilities/compaction.md) keeps a long conversation inside the context window.

## How do I bound cost and runaway loops? {#cost}

[`UsageLimits`][pydantic_ai.usage.UsageLimits] caps one run's requests, tool calls, tokens and cost in USD, and raises [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] once a limit is hit:

```python {title="production_limits.py"}
from decimal import Decimal

from pydantic_ai import Agent, UsageLimits

agent = Agent('openai:gpt-6-sol', tool_timeout=30)

limits = UsageLimits(request_limit=25, tool_calls_limit=50, cost_limit=Decimal('0.50'))


async def handle(prompt: str) -> str:
    result = await agent.run(prompt, usage_limits=limits)
    return result.output
```

For budgets longer than one run, such as a daily ceiling, a monthly one or a per-tenant share, [`SpendLimits`](https://pydantic.dev/docs/ai/harness/spend/) (Harness) prices every response and refuses the next request once a budget is spent, with a counter that several worker processes can share. The [Gateway](gateway.md) adds spending caps per project, user and key on the provider side. [Timeouts](timeouts.md) bound each model request, tool call and hook, [cancellation](agent.md#cancelling-a-run) stops a run from a stop button or a timer, and [`max_concurrency`](agent.md#concurrency-limiting) caps how many runs an agent executes at once.

## What happens when a provider is down? {#outages}

[`FallbackModel`](models/overview.md#fallback-model) tries the next model when a request fails with an API error, or when a response fails a check you define:

```python {title="production_fallback.py"}
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

agent = Agent(FallbackModel('anthropic:claude-fable-5-1', 'openai:gpt-6-sol'))
```

Underneath that, [transport retries](retries.md#transport-retries) retry rate limits and 5xx responses with backoff that respects `Retry-After`; [Retries](retries.md) explains how those layers multiply. The [Gateway](gateway.md) can do failover and load balancing between providers serving the same model on its side, configured as [gateway endpoints](gateway.md#gateway-endpoints).

## How do I see what happened? {#observability}

Pydantic AI emits OpenTelemetry traces of every run: each model request with its tokens and cost, each tool call with its arguments and result, and the run around them. Send them to [Logfire](logfire.md) or to [any OpenTelemetry backend](logfire.md#otel-without-logfire):

```python {title="production_tracing.py"}
import logfire

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
```

Without the Logfire SDK, [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all] does the same against your own tracer provider. [`run_id` and `conversation_id`](message-history.md#correlating-runs-with-run_id-and-conversation_id) are on every run, so a conversation reloaded from storage stays correlated across its runs. [Logfire](logfire.md) also covers which data a span includes and how to leave prompts and completions out of it.

## How do I know it's still good? {#evals}

[Pydantic Evals](evals.md) runs your agent, or any Python function, against a dataset of cases in code, and scores the results with your own evaluators, built-in checks or an [LLM judge](evals/evaluators/llm-judge.md). [Trajectory evaluators](evals/evaluators/agentic.md) grade the sequence and arguments of the agent's tool calls, not just its final answer. [Online evaluation](evals/online-evaluation.md) attaches the same evaluators to production traffic, scoring every call or a sample in the background and emitting the results as OpenTelemetry events.

During a run, [`TrajectoryJudge`](https://pydantic.dev/docs/ai/harness/trajectory-judge/) (Harness) reviews a long run every few model requests and steers it back when it drifts.

## How do I test without API calls? {#testing}

Swap the model for [`TestModel`][pydantic_ai.models.test.TestModel], which calls your tools and returns data that matches your output type, or [`FunctionModel`][pydantic_ai.models.function.FunctionModel], which answers with a function you write. [`Agent.override`][pydantic_ai.agent.Agent.override] swaps it in without touching your application code, and [`ALLOW_MODEL_REQUESTS = False`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] makes any real model request fail the test:

```python {title="production_testing.py"}
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False

agent = Agent('anthropic:claude-fable-5-1', instructions='Answer support questions.')


def test_support_agent():
    with agent.override(model=TestModel()):
        result = agent.run_sync('Where is my order?')
    assert result.output == 'success (no tool calls)'
```

`Agent('test')` uses `TestModel` directly, which is handy before you have an API key. [Testing](testing.md) covers fixtures, asserting on the messages a run exchanged, and `FunctionModel`.

## How do I limit what the agent can touch? {#safety}

Give the agent only the tools the current caller may use: [build the toolset per run](toolsets.md#dynamically-building-a-toolset) or [filter it](toolsets.md#filtering-tools) against the user in your [dependencies](dependencies.md), and treat a history sent by a client as [untrusted input](message-history.md#loading-untrusted-history). Beyond [approval](#approval), Pydantic AI Harness adds:

- [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) on input, output and tool calls, which block, redact, retry or require approval, with ready-made secret and PII detectors.
- [`ToolCallJudge`](https://pydantic.dev/docs/ai/harness/tool-call-judge/), which blocks a tool call a second model judges risky, before the tool function runs.
- [`FileSystem`](https://pydantic.dev/docs/ai/harness/filesystem/) with [path patterns](https://pydantic.dev/docs/ai/harness/filesystem/#pattern-filtering) that allow, deny or make paths read-only; `.env` files, keys, secrets and `.git` are read-only by default.
- [Sandboxes](https://pydantic.dev/docs/ai/harness/#execution-environments) for shell commands and model-written code, such as [bubblewrap](https://pydantic.dev/docs/ai/harness/bubblewrap-sandbox/) locally and [E2B](https://pydantic.dev/docs/ai/harness/e2b-sandbox/) or [Modal](https://pydantic.dev/docs/ai/harness/modal-sandbox/) remotely.
- A [prompt injection defender](https://pydantic.dev/docs/ai/harness/prompt-injection-defender/), built on StackOne Defender, that scans tool results such as emails, tickets and web pages.

## How do I serve it to a frontend? {#frontend}

A [UI adapter](ui/overview.md) turns a request from your frontend into an agent run and streams its text, thinking and tool calls back over the [AG-UI](ui/ag-ui.md) or [Vercel AI](ui/vercel-ai.md) protocol, from FastAPI or any Starlette app:

```python {title="production_frontend.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('anthropic:claude-fable-5-1')
app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)
```

Approvals for [deferred tools](#approval) travel over the same stream. For a durable run, the endpoint starts a workflow and streams its events back, as [Temporal](durable_execution/temporal.md#streaming-events-to-a-frontend-with-workflow-streams) shows. [Interfaces](interfaces.md) lists every other surface: a built-in [web chat UI](web.md), the [CLI](cli.md), editors over ACP, other agents over A2A, [voice](realtime/overview.md) and GitHub.

## How do I deploy it? {#deploy}

An agent is a Python object in your process, so you deploy it the way you deploy the rest of your Python: there is nothing Pydantic-specific to host. Typical shapes:

- **A web service**: run the agent inside your FastAPI, Starlette, Django or Flask handler, as [above](#frontend), and scale it like any other web app.
- **A worker**: start runs from a queue or schedule; a durable engine's workers, such as [Temporal](durable_execution/temporal.md), [DBOS queues](durable_execution/dbos.md) or [Prefect](durable_execution/prefect.md) deployments, give you retries and recovery with it.
- **A serverless function**: run on [AWS Lambda durable functions](https://pydantic.dev/docs/ai/harness/aws-lambda/) so a timed-out or retried invocation resumes instead of starting over.
- **A voice agent**: see [realtime deployment](realtime/deployment.md).
- **An agent on a repository**: [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) run an agent on issues, pull requests or a schedule.

Keep [model prices up to date](agent.md#keeping-model-prices-up-to-date) so cost limits and traces price new models, and pin your versions: Pydantic AI follows its [version policy](version-policy.md), and Pydantic AI Harness is on 0.x releases whose API may change between minor versions, with deprecation warnings and migration notes when it does.
