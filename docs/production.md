---
description: "Take a Pydantic AI agent to production: crash recovery, human approval, conversation state, cost limits, provider failover, API keys, tracing, evals, prompt rollouts, testing, frontends and deployment."
---

# Going to Production

Everything on this page is open source and runs on your infrastructure. [Pydantic AI](index.md), [Pydantic Evals](evals.md) and [Pydantic AI Harness](harness/index.md) are MIT-licensed Python packages that run in your own process, and durable execution runs on an engine you operate (or that engine's own cloud). There is no Pydantic-hosted agent runtime. Two hosted products plug in with a couple of lines when you want them: [Pydantic Logfire](logfire.md), which shows every run, model request and tool call as a trace next to the rest of your application, with a [free tier](https://pydantic.dev/pricing/) (any OpenTelemetry backend works instead); and the [Pydantic AI Gateway](gateway.md), one key for models across providers with spending caps, where calling providers directly works instead.

The rest of the page answers the questions teams ask before shipping an agent, each with the short answer and a link to the page that covers it in full, and puts the answers together in [one runnable service](#end-to-end).

| Before you ship, you want to know... | Short answer | Where it lives |
|---|---|---|
| [What happens when the process crashes mid-run?](#crash) | Run the agent on a [durable execution](durable_execution/overview.md) engine | Core, plus an engine you run |
| [How does a human approve a risky action, and the run resume days later?](#approval) | [Deferred tools](deferred-tools.md) end the run; a later run resumes it | Core |
| [How do I keep conversation state?](#state) | Store the run's [`Conversation`][pydantic_ai.conversation.Conversation], or let [`StepPersistence`](harness/step-persistence.md) do it | Core, or [Harness](harness/index.md) |
| [How do I bound cost and runaway loops?](#cost) | [`UsageLimits`][pydantic_ai.usage.UsageLimits] per run, [`SpendLimits`](harness/spend.md) per day, tenant or month | Core, and [Harness](harness/index.md) |
| [What happens when a provider is down?](#outages) | [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], [transport retries](retries.md#transport-retries), or Gateway failover | Core, or the hosted [Gateway](gateway.md) |
| [Where do API keys go?](#keys) | Environment variables, or one Gateway key for every provider | Core, or the hosted [Gateway](gateway.md) |
| [How do I see what happened?](#observability) | [OpenTelemetry traces](capabilities/instrumentation.md) of every run | Core, to [Logfire](logfire.md) or [any OTel backend](logfire.md#otel-without-logfire) |
| [How do I know it's still good?](#evals) | Evals on datasets, on trajectories and on live traffic | [Pydantic Evals](evals.md) |
| [How do I ship a prompt or model change?](#changes) | Compare eval runs, then roll out a [managed prompt](harness/managed-prompt.md) | [Pydantic Evals](evals.md), Harness and [Logfire](logfire.md) |
| [How do I test without API calls?](#testing) | [`TestModel`][pydantic_ai.models.test.TestModel], [`FunctionModel`][pydantic_ai.models.function.FunctionModel] and [`Agent.override`][pydantic_ai.agent.Agent.override] | Core |
| [How do I limit what the agent can touch?](#safety) | Approval, [guardrails](harness/guardrails.md), a [tool call judge](harness/tool-call-judge.md), [sandboxes](harness/index.md#execution-environments) | Core, and [Harness](harness/index.md) |
| [How do I serve it to a frontend?](#frontend) | [AG-UI](ui/ag-ui.md) or [Vercel AI](ui/vercel-ai.md) streams from any ASGI app | Core |
| [How do I deploy it?](#deploy) | As your own Python process: a web app, a worker or a function | Your infrastructure |

## Which pieces for which job

Start with the smallest set that does the job, and add the rest as you need it:

| Job | Install | Start with | Add for production |
|---|---|---|---|
| A typed agent behind an API | `pydantic-ai`, or `pydantic-ai-slim` with your [provider's extra](install.md#slim-install) | [Agents](agent.md), [Output](output.md), [Dependencies](dependencies.md) | [Usage limits](agent.md#usage-limits), a [fallback model](models/overview.md#fallback-model), [instrumentation](logfire.md), [tests](testing.md), a [UI adapter](ui/overview.md) |
| A durable workflow | `pydantic-ai-slim` with one engine's extra, such as `[dbos]`, `[temporal]` or `[prefect]`; `pydantic-ai-harness` with `[absurd]` or `[pgtask]` | [Durable execution](durable_execution/overview.md) and one engine's page | [Deferred tools](deferred-tools.md) for approval, [Persistence](persistence.md) for state between runs |
| A coding or long-running agent | `pydantic-ai-harness` | [Pydantic AI Harness](harness/index.md) and [Coder](harness/coder.md) | A [sandbox](harness/index.md#execution-environments), [`SpendLimits`](harness/spend.md), [`StepPersistence`](harness/step-persistence.md), [workspace capabilities under an engine](harness/durable-execution.md) |
| An eval suite | `pydantic-evals` (included in `pydantic-ai`) | [Evals quick start](evals/quick-start.md) | [Trajectory evaluators](evals/evaluators/agentic.md), [online evaluation](evals/online-evaluation.md) |

## A complete service {#end-to-end}

One file with the pieces most production agents need: an HTTP API, a tool that waits for a person to approve it, durable execution on [DBOS](durable_execution/dbos.md), and OpenTelemetry traces. It needs a model provider's API key and nothing else: DBOS keeps its state in a SQLite file here (use Postgres in production), and traces go to [Logfire](logfire.md) only when `LOGFIRE_TOKEN` is set, or to any OTel collector named by `OTEL_EXPORTER_OTLP_ENDPOINT`.

```bash
pip/uv-add "pydantic-ai[dbos]" fastapi uvicorn
```

```python {title="production_service.py" test="skip"}
from typing import Any

import logfire
from dbos import DBOS
from fastapi import FastAPI

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.durable_exec.dbos import DBOSDurability

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

app = FastAPI()
DBOS(fastapi=app, config={'name': 'support', 'system_database_url': 'sqlite:///support.sqlite'})

agent = Agent(
    '{{ anthropic:medium }}',
    name='support',
    instructions='Help customers with their orders.',
    output_type=[str, DeferredToolRequests],
    capabilities=[DBOSDurability()],
)


@agent.tool_plain(requires_approval=True)
@DBOS.step()  # DBOSDurability doesn't wrap function tools, so mark side effects as steps
def issue_refund(order_id: str, amount: float) -> str:
    return f'Refunded {amount} on order {order_id}'  # call your payments API here


@DBOS.workflow()
async def support_run(prompt: str) -> str:
    await DBOS.set_event_async('state', {'status': 'running'})
    result = await agent.run(prompt)
    while isinstance(output := result.output, DeferredToolRequests):
        pending = {call.tool_call_id: call.args_as_dict() for call in output.approvals}
        await DBOS.set_event_async('state', {'status': 'awaiting_approval', 'pending': pending})
        approvals: dict[str, bool] | None = await DBOS.recv_async('approvals', timeout_seconds=7 * 24 * 3600)
        if approvals is None:  # nobody answered within a week
            approvals = {call_id: False for call_id in pending}
        await DBOS.set_event_async('state', {'status': 'running'})
        result = await agent.run(
            conversation=result.conversation,
            deferred_tool_results=output.build_results(approvals=approvals),
        )
    await DBOS.set_event_async('state', {'status': 'done', 'output': output})
    return output


@app.post('/runs')
async def start_run(prompt: str) -> dict[str, str]:
    handle = await DBOS.start_workflow_async(support_run, prompt)
    return {'run_id': handle.get_workflow_id()}


@app.get('/runs/{run_id}')
async def get_run(run_id: str) -> dict[str, Any] | None:
    return await DBOS.get_event_async(run_id, 'state', timeout_seconds=0)  # None until the workflow starts


@app.post('/runs/{run_id}/approvals')
async def approve(run_id: str, approvals: dict[str, bool]) -> None:
    await DBOS.send_async(run_id, approvals, topic='approvals')
```

Run it with `uvicorn production_service:app`, start a run with `POST /runs?prompt=...`, read its status and any pending calls with `GET /runs/{run_id}`, and approve or deny the pending refund with `POST /runs/{run_id}/approvals` and a body like `{"<tool_call_id>": true}`. Kill the server at any point and start it again: DBOS recovers the workflow, replays the model requests and the `issue_refund` step that already finished from its database, and goes back to waiting for the approval if that is where it was. The HTTP requests only start, read and signal the workflow, so the run does not depend on any of them staying open. The sections below explain each piece and the alternatives.

## What happens when the process crashes mid-run? {#crash}

Without [durable execution](durable_execution/overview.md), a run in flight dies with the process: you start it again, or continue it from the last snapshot [`StepPersistence`](#checkpointer) saved. With it, each model request runs as a durable step on an engine that journals the result, and so does each tool call the engine wraps or you mark; which tool calls an engine wraps for you differs per engine (DBOS, for one, leaves function tools to you, as in the [service above](#end-to-end)). A restarted run replays completed steps from the journal and continues from the one that was in flight, without paying for completed model requests again or re-running completed steps.

You attach durability to an ordinary agent as a [capability](capabilities/overview.md), and the run is durable when it executes inside the engine's workflow, as `support_run` does in the [service above](#end-to-end).

A step that was in flight when the process died runs again from its start, so a tool with side effects should be safe to repeat, for example by passing an idempotency key to the API it calls. Model requests, tool calls and output validation each [retry on their own layer](retries.md) before a run fails, and a run that still fails raises an exception such as [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior], with [`capture_run_messages`][pydantic_ai.capture_run_messages] to see what it exchanged; see [Model errors](agent.md#model-errors).

The [service above](#end-to-end) uses DBOS because it runs in your process against Postgres or SQLite, with no server to operate; [Temporal](durable_execution/temporal.md) and [Restate](durable_execution/restate.md) run a server that your workers connect to, [Prefect](durable_execution/prefect.md) and [Airflow](durable_execution/airflow.md) fit when you already orchestrate with them, and [Absurd](harness/absurd.md) and [pgtask](harness/pgtask.md) (Harness) run on Postgres alone. [Durable Execution](durable_execution/overview.md) lists every supported engine and links each one's page, which says which calls it wraps for you and which you mark yourself; an agent takes one engine at a time.

### What you get compared to a checkpointer {#checkpointer}

In frameworks built around a checkpointer, one mechanism saves state after each step and covers crash recovery, pausing for a human and resuming a thread. Pydantic AI splits those jobs between pieces that compose on one agent:

- **Crash recovery inside a run** comes from durable execution, at the granularity of a single model request or journaled tool call. A crash in the middle of a long tool loop resumes at the step that was running, with every completed step before it replayed from the journal.
- **Saving and continuing between runs** comes from [`StepPersistence`](harness/step-persistence.md) (Harness): it saves a continuable snapshot at every settled step and when a run fails, so you can continue a run from its latest snapshot or fork it from any saved one, later and in another process if you like, with built-in stores or your own.
- **Knowing whether a side effect happened** comes from the `StepPersistence` tool-effect ledger, which records when each tool call started and finished, so after a crash you can tell a call that completed from one whose effect is unknown, before deciding to run it again.
- **Pausing for a human** needs no live process at all: [deferred tools](#approval) end the run cleanly and a later run picks it up.

`StepPersistence` is aware of durable execution: its writes are durable operations, so replaying a durable run does not write its events twice. A snapshot holds the run's message history, which is the state an agent run is made of; state that a capability keeps outside the messages, a workspace's files and resuming in the middle of a step are not part of it, and that last one is what durable execution is for. [Persistence](persistence.md) lays out which piece answers which question.

## How does a human approve a risky action, and the run resume days later? {#approval}

[Mark the tool as needing approval](deferred-tools.md#human-in-the-loop-tool-approval). When the model calls it, the run ends with a [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests] output listing the pending calls, and nothing stays in memory while you wait. Store the run's [`Conversation`][pydantic_ai.conversation.Conversation], which keeps the pending calls as [`deferred_tool_requests`][pydantic_ai.conversation.Conversation.deferred_tool_requests], ask a person, and when they answer, minutes or weeks later, continue the conversation with their decision:

```python {title="production_approval.py"}
from pydantic_ai import Agent, ConversationTypeAdapter, DeferredToolRequests

agent = Agent('{{ anthropic:medium }}', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: float) -> str:
    return f'Refunded {amount} on order {order_id}'


async def start(prompt: str) -> bytes:
    result = await agent.run(prompt)
    if isinstance(result.output, DeferredToolRequests):
        ...  # ask a person about each call in result.output.approvals
    return ConversationTypeAdapter.dump_json(result.conversation)  # store under result.conversation_id


async def resume(stored: bytes, approvals: dict[str, bool]) -> str | DeferredToolRequests:
    conversation = ConversationTypeAdapter.validate_json(stored)
    assert conversation.deferred_tool_requests is not None
    result = await agent.run(
        conversation=conversation,
        deferred_tool_results=conversation.deferred_tool_requests.build_results(approvals=approvals),
    )
    return result.output
```

The same mechanism [hands a call to a frontend or a background worker](deferred-tools.md#external-tool-execution) to execute, and an approval can come back with edited arguments or a denial message the model sees. The [UI adapters](ui/overview.md) carry approvals to and from the browser, and the [CLI](cli.md) prompts for them. Under a durable engine, the workflow can instead wait for the decision itself, as `support_run` does in the [service above](#end-to-end) with DBOS messages; the engine keeps the wait in its database, so a restart picks it back up. When the approver is in the same process, a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves calls inline without ending the run. All of it is on [Deferred Tools](deferred-tools.md).

For an agent with nobody watching, [`ToolCallJudge`](harness/tool-call-judge.md) (Harness) asks a second model whether a call may run, and can escalate the calls it is unsure about to a person. When people are watching but most calls are harmless, [`ApprovalReviewer`](harness/approval-reviewer.md) (Harness) lets a second model approve the low-risk ones and leaves the rest to the person.

!!! warning "Approval guards against the model, not the client"
    An approval submitted by a client is trusted as given. Authenticate the endpoint, and check authorization inside the tool for sensitive actions; see the [trust boundary for client-supplied history](message-history.md#trust-boundary-for-client-supplied-history).

## How do I keep conversation state? {#state}

A conversation's state is its [`Conversation`][pydantic_ai.conversation.Conversation]: the [messages](message-history.md), the running usage that [`UsageLimits`](#cost) budgets keep counting across turns and approvals, the `conversation_id` that [correlates its runs](message-history.md#correlating-runs-with-run_id-and-conversation_id), and any pending deferred calls. Store [`result.conversation`][pydantic_ai.agent.AgentRunResult.conversation] in your own database, serialized with [`ConversationTypeAdapter`][pydantic_ai.conversation.ConversationTypeAdapter] and keyed by its `conversation_id`, and pass it back as `conversation=` on the next run. A `jsonb` column is enough; no schema migration is needed when Pydantic AI adds a message part. [Persistence](persistence.md#storing-a-conversation-yourself) also shows an append-only variant that keeps writes proportional to each turn.

To skip writing that code, [`StepPersistence`](harness/step-persistence.md) (Harness) stores runs for you, with continue and fork. For what an agent should remember about a user across conversations, [`Memory`](harness/memory.md) (Harness) keeps notes the agent writes itself. [Persistence](persistence.md) compares all of these, and [compaction](capabilities/compaction.md) keeps a long conversation inside the context window.

## How do I bound cost and runaway loops? {#cost}

[`UsageLimits`][pydantic_ai.usage.UsageLimits] caps one run's requests, tool calls, tokens and cost in USD, and raises [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] once a limit is hit:

```python {title="production_limits.py"}
from decimal import Decimal

from pydantic_ai import Agent, UsageLimits

agent = Agent('{{ openai:medium }}', tool_timeout=30)

limits = UsageLimits(request_limit=25, tool_calls_limit=50, cost_limit=Decimal('0.50'))


async def handle(prompt: str) -> str:
    result = await agent.run(prompt, usage_limits=limits)
    return result.output
```

Cost is best-effort: a model [genai-prices](https://github.com/pydantic/genai-prices) has no pricing data for isn't counted (with a `cost_limit`, a run that can't be priced at all emits [`CostNotFoundWarning`][pydantic_ai.exceptions.CostNotFoundWarning]), so keep `request_limit` alongside `cost_limit`; see [usage limits](agent.md#usage-limits). Keep [model prices up to date](agent.md#keeping-model-prices-up-to-date) so cost limits and traces price new models.

For budgets longer than one run, such as a daily ceiling, a monthly one or a per-tenant share, [`SpendLimits`](harness/spend.md) (Harness) prices every response and refuses the next request once a budget is spent, with a counter that several worker processes can share. Its USD budgets have the same gap: an unpriced response counts as free unless you set `on_unpriced='raise'` or supply your own `price`, while its token budgets still hold. The [Gateway](gateway.md#key-features) enforces its own daily, weekly and monthly spending caps per project, user and API key before a request reaches the provider. [Timeouts](timeouts.md) bound [each model request, tool call and hook](timeouts.md#bounding-how-long-a-step-takes) and [a whole run](timeouts.md#bounding-a-whole-run) (the `tool_timeout=30` above is one), [cancellation](agent.md#cancelling-a-run) stops a run from a stop button or a timer, [`max_concurrency`](agent.md#concurrency-limiting) caps how many runs an agent executes at once, and [`ConcurrencyLimitedModel`][pydantic_ai.ConcurrencyLimitedModel] caps [concurrent requests to a model](models/overview.md#http-request-concurrency) to stay under a provider's rate limits.

The cheapest request is one the provider has cached: the [`Caching`][pydantic_ai.capabilities.Caching] capability turns on [prompt caching](capabilities/caching.md), so a multi-turn run stops paying full price for its instructions, tools and history on every request, and each model request's span reports [prompt-cache health](logfire.md#prompt-cache-health), so you notice when a change breaks the cache. The [token and cost metrics](logfire.md#emitted-metrics) break spend down by provider and model, in Logfire or any OpenTelemetry backend.

## What happens when a provider is down? {#outages}

[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] tries the next model when a request fails with an API error, or when a response fails a check you define; see [Fallback Model](models/overview.md#fallback-model):

```python {title="production_fallback.py"}
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

agent = Agent(FallbackModel('{{ anthropic:medium }}', '{{ openai:medium }}'))
```

[Provider SDK clients](retries.md#provider-sdk-retries) already retry some failures on their own. For rate limits and 5xx responses with backoff that respects `Retry-After`, install a [retrying transport](retries.md#transport-retries) on the HTTP client; nothing retries at that layer by default. [Retries](retries.md) explains how the layers multiply. The [Gateway](gateway.md) can do failover and load balancing between providers serving the same model on its side, configured as [gateway endpoints](gateway.md#gateway-endpoints).

## Where do API keys go? {#keys}

Each provider reads its key from an environment variable, such as `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (each [model page](models/overview.md) names its own), or takes `api_key=` on its provider class, so you load keys from your secret manager like any other credential. With the [Gateway](gateway.md), the application holds one `PYDANTIC_AI_GATEWAY_API_KEY` for every provider, and each Gateway key can carry its own spending cap, so you can issue one per service or environment; see the [Gateway quick start](gateway.md#quick-start).

## How do I see what happened? {#observability}

Pydantic AI emits OpenTelemetry traces of every run: each model request with its tokens and cost, each tool call with its arguments and result, and the run around them. Send them to [Logfire](logfire.md) or to [any OpenTelemetry backend](logfire.md#otel-without-logfire):

```python {title="production_tracing.py"}
import logfire

logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
```

With Logfire, getting there takes `logfire auth` and `logfire projects new` on your machine and a write token in production; a coding agent can do it for you from [pydantic.dev/ai-setup.md](https://pydantic.dev/ai-setup.md), and the free tier covers getting started. The same SDK instruments [FastAPI](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/) and [the HTTP requests to the provider](logfire.md#monitoring-http-requests), so a run shows up inside the request that started it, with the raw prompt it sent. [Using Logfire](logfire.md#using-logfire) walks through it.

Without the Logfire SDK, [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all] or the [`Instrumentation`](capabilities/instrumentation.md) capability does the same against your own tracer provider; see [OTel without Logfire](logfire.md#otel-without-logfire). [`run_id` and `conversation_id`](message-history.md#correlating-runs-with-run_id-and-conversation_id) are on every run, so a conversation reloaded from storage stays correlated across its runs, and [run metadata](agent.md#run-metadata) such as a tenant or user ID lands on the run span, so you can filter traces by it.

Telemetry includes prompts, completions and tool arguments by default. To keep them out, set [`include_content=False`](logfire.md#excluding-prompts-and-completions) on [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings], and [`include_binary_content=False`](logfire.md#excluding-binary-content) leaves out images, audio and documents.

## How do I know it's still good? {#evals}

[Pydantic Evals](evals.md) runs your agent, or any Python function, against a dataset of cases in code, and scores the results with your own evaluators, built-in checks or an [LLM judge](evals/evaluators/llm-judge.md). [Trajectory evaluators](evals/evaluators/agentic.md) grade the sequence and arguments of the agent's tool calls, not just its final answer. [Online evaluation](evals/online-evaluation.md) attaches the same evaluators to production traffic, scoring every call or a sample in the background and emitting the results as OpenTelemetry events.

During a run, [`TrajectoryJudge`](harness/trajectory-judge.md) (Harness) reviews a long run every few model requests and steers it back when it drifts.

## How do I ship a prompt or model change? {#changes}

Before you ship a prompt or model change, run the same [evals](#evals) dataset against it and compare the reports; [sending experiments to Logfire](evals/how-to/logfire-integration.md) keeps every run side by side with the trace of each case, and [dataset management](evals/how-to/dataset-management.md) covers keeping datasets in YAML next to your code. To change instructions without a redeploy, [`ManagedPrompt`](harness/managed-prompt.md) (Harness) backs them with a Logfire-managed prompt that you version, label and roll out from the Logfire UI, and an [agent spec](agent-spec.md) keeps the model, instructions and capabilities in a YAML or JSON file instead of code.

## How do I test without API calls? {#testing}

Swap the model for [`TestModel`][pydantic_ai.models.test.TestModel], which calls your tools and returns data that matches your output type, or [`FunctionModel`][pydantic_ai.models.function.FunctionModel], which answers with a function you write. [`Agent.override`][pydantic_ai.agent.Agent.override] swaps it in without touching your application code, and [`ALLOW_MODEL_REQUESTS = False`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] makes any real model request fail the test:

```python {title="production_testing.py"}
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False

agent = Agent('{{ anthropic:medium }}', instructions='Answer support questions.')


def test_support_agent():
    with agent.override(model=TestModel()):
        result = agent.run_sync('Where is my order?')
    assert result.output == 'success (no tool calls)'
```

`Agent('test')` uses `TestModel` directly, which is handy before you have an API key. [Testing](testing.md) covers [pytest fixtures](testing.md#overriding-model-via-pytest-fixtures), asserting on the messages a run exchanged, and [`FunctionModel`](testing.md#unit-testing-with-functionmodel).

## How do I limit what the agent can touch? {#safety}

Give the agent only the tools the current caller may use: [build the toolset per run](toolsets.md#dynamically-building-a-toolset) or [filter it](toolsets.md#filtering-tools) against the user in your [dependencies](dependencies.md), and treat a history sent by a client as [untrusted input](message-history.md#loading-untrusted-history). Beyond [approval](#approval), [Pydantic AI Harness](harness/index.md) adds:

- [Guardrails](harness/guardrails.md) on input, output and tool calls, which block, redact, retry or require approval, with ready-made secret and PII detectors.
- [`ToolCallJudge`](harness/tool-call-judge.md), which blocks a tool call a second model judges risky, before the tool function runs.
- [`FileSystem`](harness/filesystem.md) with [path patterns](harness/filesystem.md#pattern-filtering) that allow, deny or make paths read-only; `.env` files, keys, secrets and `.git` are read-only by default.
- [Sandboxes](harness/index.md#execution-environments) for shell commands and model-written code, such as [bubblewrap](harness/bubblewrap-sandbox.md) locally and [E2B](harness/e2b-sandbox.md) or [Modal](harness/modal-sandbox.md) remotely.
- A [prompt injection defender](harness/prompt-injection-defender.md), built on StackOne Defender, that scans tool results such as emails, tickets and web pages.

## How do I serve it to a frontend? {#frontend}

A [UI adapter](ui/overview.md) turns a request from your frontend into an agent run and streams its text, thinking and tool calls back over the [AG-UI](ui/ag-ui.md) or [Vercel AI](ui/vercel-ai.md) protocol, from FastAPI or any Starlette app:

```python {title="production_frontend.py"}
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter

agent = Agent('{{ anthropic:medium }}')
app = FastAPI()


@app.post('/chat')
async def chat(request: Request) -> Response:
    return await VercelAIAdapter.dispatch_request(request, agent=agent)
```

Approvals for [deferred tools](#approval) travel over the same stream. The adapter endpoint is not an authentication boundary: run it inside your own authenticated route, as the [trust model for client-submitted messages](ui/overview.md#trust-model-for-client-submitted-messages) explains, along with the defaults that keep system prompts and file access on your side. For a durable run or a background job, the endpoint starts the run elsewhere and [streams its events back](ui/overview.md#running-the-agent-elsewhere), as [Temporal's workflow streams](durable_execution/temporal.md#streaming-events-to-a-frontend-with-workflow-streams) do. [Interfaces](interfaces.md) lists every other way to serve an agent; the built-in [web chat UI](web.md) is for local development and debugging, not production.

## How do I deploy it? {#deploy}

An agent is a Python object in your process, so you deploy it the way you deploy the rest of your Python: there is nothing Pydantic-specific to host. Typical shapes:

- **A web service**: run the agent inside your FastAPI, Starlette, Django or Flask handler, as [above](#frontend), and scale it like any other web app. A conversation's state lives in what you [store](#state) or in the durable engine, not in the process, so any instance can continue any conversation, and health checks, graceful shutdown and autoscaling are your web framework's and platform's.
- **A worker**: start runs from a queue or schedule; a durable engine's workers, such as [Temporal](durable_execution/temporal.md), [DBOS queues](durable_execution/dbos.md#durable-execution) or [Prefect deployments](durable_execution/prefect.md#deployments-and-scheduling), give you retries and recovery with it; [Render Workflows](harness/render-workflows.md) (Harness) run model requests and supported tool calls as separate tasks, each with its own retries, timeout and compute.
- **A serverless function**: run on [AWS Lambda durable functions](harness/aws-lambda.md) so a timed-out or retried invocation resumes instead of starting over.
- **A voice agent**: see [realtime deployment](realtime/deployment.md).
- **An agent on a repository**: [GitHub Agentic Workflows](harness/gh-aw.md) run an agent on issues, pull requests or a schedule.

In a long-running server, runs are async and share one event loop: [`max_concurrency`](agent.md#concurrency-limiting) caps how many run at once (with a queue limit for backpressure), [`UseThreadExecutor`](capabilities/thread-executor.md) gives sync tools a bounded thread pool, and a slow tool can run as a [background tool](harness/background-tools.md) (Harness) while the agent keeps working.

Pin your versions: Pydantic AI follows its [version policy](version-policy.md), and Pydantic AI Harness is on [0.x releases](harness/index.md#version-policy) whose API may change between minor versions.
