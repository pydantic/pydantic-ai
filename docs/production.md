---
description: "Take a Pydantic AI agent to production: crash recovery, human approval, conversation state, cost limits, provider failover, API keys, tracing, evals, prompt rollouts, testing, frontends and deployment."
---

# Going to Production

[Pydantic AI](index.md), [Pydantic Evals](evals.md) and [Pydantic AI Harness](harness/index.md) are open-source, MIT-licensed Python packages that run in your own process, and durable execution runs on an engine you operate (or that engine's own cloud): there is no Pydantic-hosted agent runtime. Two hosted Pydantic services plug in with a couple of lines when you want them: [Pydantic Logfire](logfire.md), which shows every run, model request and tool call as a trace next to the rest of your application, with a [free tier](https://pydantic.dev/pricing/) (any OpenTelemetry backend works instead); and the [Pydantic AI Gateway](gateway.md), one key for models across providers with spending caps, where calling providers directly works instead.

Pydantic AI is open source, with [commercial enterprise support from Pydantic](enterprise-support.md), and runs on [Azure OpenAI](models/azure.md), [Amazon Bedrock](models/bedrock.md) and [Google Vertex AI](models/google-cloud.md) as well as on OpenAI, Anthropic, Gemini and [models you host yourself](models/ollama.md). [Pydantic AI for Enterprise](enterprise.md) answers the questions security, platform and procurement teams ask: what leaves your infrastructure, which models you can use, approvals and audit trails, central model access and spending caps through the Gateway, Logfire's compliance and self-hosting, and support.

The rest of the page answers the questions teams ask before shipping an agent, each with the short answer and a link to the page that covers it in full, after [one runnable service](#end-to-end) that puts the answers together.

| Before you ship, you want to know... | Short answer | Install |
|---|---|---|
| [What happens when the process crashes mid-run?](#crash) | Run the agent on a [durable execution](durable_execution/overview.md) engine | `pydantic-ai-slim` with one engine's extra, such as `[dbos]` or `[temporal]` |
| [How does a human approve a risky action, and the run resume days later?](#approval) | [Deferred tools](deferred-tools.md) end the run; a later run resumes it | `pydantic-ai` |
| [How do I keep conversation state?](#state) | Store the run's `Conversation` in your database | `pydantic-ai` |
| [How do I bound cost and runaway loops?](#cost) | `UsageLimits` per run, `SpendLimits` per day, tenant or month | `pydantic-ai`, and `pydantic-ai-harness` for `SpendLimits` |
| [What happens when a provider is down?](#outages) | A fallback model, transport retries, or Gateway failover | `pydantic-ai`, or the hosted [Gateway](gateway.md) |
| [Where do API keys go?](#keys) | Environment variables, or one Gateway key for every provider | `pydantic-ai`, or the hosted [Gateway](gateway.md) |
| [How do I see what happened?](#observability) | OpenTelemetry traces of every run, in [Logfire](logfire.md) or any OTel backend | `pydantic-ai` (includes `logfire`) |
| [How do I test and evaluate changes?](#quality) | Unit tests with `TestModel`, then evals on datasets and live traffic | `pydantic-ai` (includes `pydantic-evals`) |
| [How do I limit what the agent can touch?](#safety) | Per-caller tools, approval, and a sandbox for model-written code | `pydantic-ai`, and `pydantic-ai-harness` for sandboxes |
| [How do I serve it to a frontend?](#frontend) | [AG-UI](ui/ag-ui.md) or [Vercel AI](ui/vercel-ai.md) streams from any ASGI app | `pydantic-ai` |
| [How do I deploy it?](#deploy) | As your own Python process: a web app, a worker or a function | Your infrastructure |

For systems already running Pydantic AI in production, with their code or case studies, see [Built with Pydantic AI](built-with.md).

## An end-to-end service {#end-to-end}

One file with the pieces most production agents need: an HTTP API, a tool that waits for a person to approve it, durable execution on [DBOS](durable_execution/dbos.md), and OpenTelemetry traces. It needs a model provider's API key and nothing else: DBOS keeps its state in a SQLite file here (use Postgres in production), and traces go to [Logfire](logfire.md) when a Logfire token is available, from `LOGFIRE_TOKEN` or the credentials the Logfire CLI saved locally, or to any OTel collector named by `OTEL_EXPORTER_OTLP_ENDPOINT`. To stay short, it leaves out what you add before exposing it: authentication and checks that the caller owns the run, deduplication of repeated requests, and [usage limits](#cost).

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
    try:
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
    except Exception as e:  # so GET /runs/{run_id} doesn't keep reporting 'running'
        await DBOS.set_event_async('state', {'status': 'failed', 'error': str(e)})
        raise


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

Without [durable execution](durable_execution/overview.md), a run in flight dies with the process: you start it again, or continue it from the last [`StepPersistence`](#checkpointer) snapshot. With it, an engine journals each model request and each tool call it wraps or you mark, and a restarted run replays completed steps from the journal and retries the one that was in flight, without paying for completed model requests again. Which tool calls an engine wraps for you differs (DBOS leaves function tools to you, as in the [service above](#end-to-end)). The engines that ship in Pydantic AI and the Harness attach to an ordinary agent as a [capability](capabilities/overview.md) and take effect when the run executes inside the engine's workflow or handler; Restate wraps the agent in a class of its own instead, and Airflow in an operator.

A step that was in flight when the process died runs again from its start, so make a tool with side effects safe to repeat, for example by passing an idempotency key to the API it calls. Model requests, tool calls and output validation each [retry on their own layer](retries.md) before a run fails with an exception such as [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior]; see [Model errors](agent.md#model-errors).

The [service above](#end-to-end) uses DBOS because it runs in your process against Postgres or SQLite, with no server to operate; [Temporal](durable_execution/temporal.md) and [Restate](durable_execution/restate.md) run a server that your workers connect to, [Prefect](durable_execution/prefect.md) and [Airflow](durable_execution/airflow.md) fit when you already orchestrate with them, and [Absurd](harness/absurd.md) and [pgtask](harness/pgtask.md) (Harness) run on Postgres alone. [Durable Execution](durable_execution/overview.md) lists every engine and what each one wraps for you; an agent takes one engine at a time.

### What you get compared to a checkpointer {#checkpointer}

In frameworks built around a checkpointer, one mechanism saves state after each step and covers crash recovery, pausing for a human and resuming a thread. Pydantic AI splits those jobs:

- **Crash recovery inside a run** is durable execution, at the granularity of one model request or journaled tool call.
- **Pausing for a human** needs no live process: [deferred tools](#approval) end the run and a later run picks it up.
- **Continuing between runs** is the stored [`Conversation`](#state). [`StepPersistence`](harness/step-persistence.md) (Harness) adds message-history snapshots at each settled tool-call boundary, to continue or fork a run from, and a tool-effect ledger that records whether each tool call started, returned or raised: a call that never finished has an unknown outcome, so check the external system before running it again. Snapshots hold messages only, not the usage and pending deferred calls a `Conversation` carries, so they complement storing it rather than replace it.

[Persistence](persistence.md) lays out which piece answers which question.

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


async def resume(
    stored: bytes, approvals: dict[str, bool]
) -> tuple[bytes, str | DeferredToolRequests]:
    conversation = ConversationTypeAdapter.validate_json(stored)
    assert conversation.deferred_tool_requests is not None
    result = await agent.run(
        conversation=conversation,
        deferred_tool_results=conversation.deferred_tool_requests.build_results(approvals=approvals),
    )
    return ConversationTypeAdapter.dump_json(result.conversation), result.output  # store it again
```

Store the returned conversation after every continuation: the model can ask for another approval, and the conversation is what answers it.

The same mechanism [hands a call to a frontend or a background worker](deferred-tools.md#external-tool-execution) to execute, and an approval can come back with edited arguments or a denial message the model sees. The [UI adapters](ui/overview.md) and the [CLI](cli.md) carry approvals; under a durable engine the workflow can wait for the decision itself, as `support_run` does in the [service above](#end-to-end); and when the approver is in the same process, a [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls] handler resolves calls inline without ending the run. All of it is on [Deferred Tools](deferred-tools.md).

For an agent with nobody watching, [`ToolCallJudge`](harness/tool-call-judge.md) (Harness) asks a second model whether a call may run and escalates the calls it is unsure about to a person; [`ApprovalReviewer`](harness/approval-reviewer.md) (Harness) lets a second model approve the low-risk calls and leaves the rest to the person.

!!! warning "Approval guards against the model, not the client"
    An approval submitted by a client is trusted as given. Authenticate the endpoint, and check authorization inside the tool for sensitive actions; see the [trust boundary for client-supplied history](message-history.md#trust-boundary-for-client-supplied-history).

## How do I keep conversation state? {#state}

A conversation's state is its [`Conversation`][pydantic_ai.conversation.Conversation]: the [messages](message-history.md), the running usage that [`UsageLimits`](#cost) keep counting across turns and approvals, the `conversation_id` that [correlates its runs](message-history.md#correlating-runs-with-run_id-and-conversation_id), and any pending deferred calls. Store [`result.conversation`][pydantic_ai.agent.AgentRunResult.conversation] in your own database, serialized with [`ConversationTypeAdapter`][pydantic_ai.conversation.ConversationTypeAdapter] and keyed by its `conversation_id` (a `jsonb` column is enough), and pass it back as `conversation=` on the next run.

A conversation ID is a key, not a credential: check that the caller may access the conversation before you load it, and serialize concurrent updates to one conversation, for example with a version column or a per-conversation lock, so two instances don't continue the same stored version. [Persistence](persistence.md) covers an append-only variant, [`StepPersistence`](#checkpointer), and [`Memory`](harness/memory.md) (Harness) for what an agent should remember about a user across conversations; [compaction](capabilities/compaction.md) keeps a long conversation inside the context window.

## How do I bound cost and runaway loops? {#cost}

[`UsageLimits`][pydantic_ai.usage.UsageLimits] caps one run's agent-loop requests, successful tool calls, tokens and cost in USD, and raises [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded] once a limit is hit. `request_limit` counts turns of the agent loop, not every attempt sent to a provider, and tokens and cost are checked after a response arrives, so the request that crosses a limit has already been paid for:

```python {title="production_limits.py"}
from decimal import Decimal

from pydantic_ai import Agent, UsageLimits

agent = Agent('{{ openai:medium }}', tool_timeout=30)

limits = UsageLimits(request_limit=25, tool_calls_limit=50, cost_limit=Decimal('0.50'))


async def handle(prompt: str) -> str:
    result = await agent.run(prompt, usage_limits=limits)
    return result.output
```

The response that crosses a token or cost limit stays in the run's messages, so a run continued from what [`capture_run_messages`][pydantic_ai.capture_run_messages] captured, with or without a new prompt, doesn't pay for it again; see [usage limits](agent.md#usage-limits).

Cost is best-effort: a model [genai-prices](https://github.com/pydantic/genai-prices) has no pricing data for isn't counted, so keep `request_limit` alongside `cost_limit` and keep [model prices up to date](agent.md#keeping-model-prices-up-to-date); see [usage limits](agent.md#usage-limits).

A run that hits a limit returns no result, but the work it did before that is not lost: [capture its messages and resume from them](agent.md#recovering-a-failed-run) instead of re-running the prompt and repeating its tool calls. To have the model produce its output before the limit instead, [tell it how many requests it has left](agent.md#wrapping-up-before-the-request-limit).

For budgets longer than one run, such as a daily, monthly or per-tenant ceiling, [`SpendLimits`](harness/spend.md) (Harness) refuses the next request once a budget is spent. It is a brake, not a reservation: the request that crosses the line completes and concurrent runs can each pass the check, so spend can overshoot; budgets that span worker processes or restarts need a shared store instead of the default in-memory one; and an unpriced response counts as free unless you set `on_unpriced='raise'`. The [Gateway](gateway.md#key-features) enforces daily, weekly and monthly spending caps per project, user and API key before a request reaches the provider.

[Timeouts](timeouts.md) bound [each model request, tool call and hook](timeouts.md#bounding-how-long-a-step-takes) and [a whole run](timeouts.md#bounding-a-whole-run) (the `tool_timeout=30` above is one), [cancellation](agent.md#cancelling-a-run) stops a run from a stop button or a timer, [`max_concurrency`](agent.md#concurrency-limiting) caps how many runs an agent executes at once, and [`ConcurrencyLimitedModel`][pydantic_ai.ConcurrencyLimitedModel] caps [concurrent requests to a model](models/overview.md#http-request-concurrency) within one process (concurrency, not requests per minute, so still handle the provider's rate-limit responses).

The [`Caching`][pydantic_ai.capabilities.Caching] capability turns on [prompt caching](capabilities/caching.md), so a multi-turn run stops paying full price for its instructions, tools and history on every request; [prompt-cache health](logfire.md#prompt-cache-health) on each request span shows when a change breaks the cache, and [token and cost metrics](logfire.md#emitted-metrics) break spend down by provider and model.

## What happens when a provider is down? {#outages}

The [`ModelFallback`](capabilities/model-fallback.md) capability tries the next model when a request fails with an API error, or when a non-streamed response fails a check you define (streamed responses, as a [frontend](#frontend) gets, are accepted as they are):

```python {title="production_fallback.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelFallback

agent = Agent('{{ anthropic:medium }}', capabilities=[ModelFallback('{{ openai:medium }}')])
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

With Logfire, getting there takes `uvx logfire-cli wizard --task pydantic-ai` on your machine, which also has your coding agent add the instrumentation, and a write token in production; the free tier covers getting started. The same SDK instruments [FastAPI](https://logfire.pydantic.dev/docs/integrations/web-frameworks/fastapi/) and [the HTTP requests to the provider](logfire.md#monitoring-http-requests), so a run shows up inside the request that started it, with the raw prompt it sent. [Using Logfire](logfire.md#using-logfire) walks through it.

Without the Logfire SDK, [`Agent.instrument_all()`][pydantic_ai.agent.Agent.instrument_all] or the [`Instrumentation`](capabilities/instrumentation.md) capability does the same against your own tracer provider; see [OTel without Logfire](logfire.md#otel-without-logfire). Every run carries its [`run_id` and `conversation_id`](message-history.md#correlating-runs-with-run_id-and-conversation_id), and [run metadata](agent.md#run-metadata) such as a tenant or user ID lands on the run span, so you can filter traces by it.

Telemetry includes prompts, completions and tool arguments by default. To keep them out of Pydantic AI's spans, set [`include_content=False`](logfire.md#excluding-prompts-and-completions) on [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings], and [`include_binary_content=False`](logfire.md#excluding-binary-content) leaves out the bytes of images, audio and documents. Neither touches run metadata, your own spans or captured HTTP request bodies, so configure those separately.

## How do I test and evaluate changes? {#quality}

**Unit tests** swap the model for [`TestModel`][pydantic_ai.models.test.TestModel], which calls your tools and returns data that matches your output type, or [`FunctionModel`][pydantic_ai.models.function.FunctionModel], which answers with a function you write. [`Agent.override`][pydantic_ai.agent.Agent.override] swaps it in without touching your application code, and [`ALLOW_MODEL_REQUESTS = False`][pydantic_ai.models.ALLOW_MODEL_REQUESTS] makes any real model request fail the test:

```python {title="production_testing.py"}
from pydantic_ai import Agent, models
from pydantic_ai.models.test import TestModel

models.ALLOW_MODEL_REQUESTS = False

agent = Agent(
    '{{ anthropic:medium }}',
    instructions='Answer support questions.',
    defer_model_check=True,  # build the model on first use, so tests need no API key
)


def test_support_agent():
    with agent.override(model=TestModel()):
        result = agent.run_sync('Where is my order?')
    assert result.output == 'success (no tool calls)'
```

[Testing](testing.md) covers [pytest fixtures](testing.md#overriding-model-via-pytest-fixtures), asserting on the messages a run exchanged, and [`FunctionModel`](testing.md#unit-testing-with-functionmodel).

**Evals** check what unit tests can't: whether the model's answers are good. [Pydantic Evals](evals.md) runs your agent against a dataset of cases in code and scores the results with your own evaluators, built-in checks or an [LLM judge](evals/evaluators/llm-judge.md), and [trajectory evaluators](evals/evaluators/agentic.md) grade the agent's tool calls, not just its final answer.

**To roll out a prompt or model change**, run the same dataset against it and compare the reports; [sending experiments to Logfire](evals/how-to/logfire-integration.md) keeps every run side by side with the trace of each case. [`ManagedPrompt`](harness/managed-prompt.md) (Harness) backs instructions with a Logfire-managed prompt that you version, label and roll out from the Logfire UI without a redeploy, and an [agent spec](agent-spec.md) keeps the model, instructions and capabilities in a YAML or JSON file instead of code.

**[Online evaluation](evals/online-evaluation.md)** attaches the same evaluators to production traffic, scoring in the background and emitting the results as OpenTelemetry events. That background work lives in your process, not on a job queue: [sample](evals/online-evaluation.md#sampling) and [cap the concurrency](evals/online-evaluation.md#concurrency-control) of expensive evaluators such as an LLM judge, and await [`wait_for_evaluations()`][pydantic_evals.online.wait_for_evaluations] on shutdown so pending ones finish. To steer a long run while it happens, [`TrajectoryJudge`](harness/trajectory-judge.md) (Harness) reviews it every few model requests.

## How do I limit what the agent can touch? {#safety}

- **Authorize tools and conversation access**: give the agent only the tools the current caller may use, by [building the toolset per run](toolsets.md#dynamically-building-a-toolset) or [filtering it](toolsets.md#filtering-tools) against the user in your [dependencies](dependencies.md); check that the caller may access the [conversation](#state) it continues; and treat a history sent by a client as [untrusted input](message-history.md#loading-untrusted-history).
- **Require approval for sensitive actions**: [deferred tools](#approval) put a person, or a second model with [`ToolCallJudge`](harness/tool-call-judge.md), in front of the call.
- **Isolate model-written code**: run shell commands and generated code in a [sandbox](harness/index.md#execution-environments), such as [bubblewrap](harness/bubblewrap-sandbox.md) locally or [E2B](harness/e2b-sandbox.md) or [Modal](harness/modal-sandbox.md) remotely. [`FileSystem`](harness/filesystem.md) path patterns guard what the model's file tools write, but they are checks, not isolation, and don't bound shell commands.

[Pydantic AI Harness](harness/index.md#capabilities) adds more, such as [guardrails](harness/guardrails.md) with secret and PII detectors and a [prompt injection defender](harness/prompt-injection-defender.md) for tool results.

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

Approvals for [deferred tools](#approval) travel over the same stream. The adapter endpoint is not an authentication boundary: run it inside your own authenticated route, as the [trust model for client-submitted messages](ui/overview.md#trust-model-for-client-submitted-messages) explains. For a durable run or a background job, the endpoint starts the run elsewhere and [streams its events back](ui/overview.md#running-the-agent-elsewhere). [Interfaces](interfaces.md) lists every other way to serve an agent; the built-in [web chat UI](web.md) is for local development, not production.

## How do I deploy it? {#deploy}

An agent is a Python object in your process, so you deploy it the way you deploy the rest of your Python: there is nothing Pydantic-specific to host. Typical shapes:

- **A web service**: run the agent inside your FastAPI, Starlette, Django or Flask handler, as [above](#frontend), and scale it like any other web app. A conversation's state lives in what you [store](#state) or in the durable engine, not in the process, so any instance can continue any conversation.
- **A worker**: start runs from a queue or schedule, on a durable engine's workers such as [Temporal](durable_execution/temporal.md), [DBOS queues](durable_execution/dbos.md#durable-execution) or [Prefect deployments](durable_execution/prefect.md#deployments-and-scheduling), or on [Render Workflows](harness/render-workflows.md) (Harness), which run model requests and supported tool calls as separate tasks.
- **A serverless function**: run on [AWS Lambda durable functions](harness/aws-lambda.md) so a timed-out or retried invocation resumes instead of starting over.
- **A voice agent**: a [realtime session](realtime/overview.md) holds a provider connection open for the whole call, so it needs long-lived compute rather than request-scoped functions; see [hosting a realtime backend](realtime/deployment.md#hosting).
- **An agent on a repository**: [GitHub Agentic Workflows](harness/gh-aw.md) run an agent on issues, pull requests or a schedule.

In a long-running server, runs are async and share one event loop: [`max_concurrency`](agent.md#concurrency-limiting) caps how many run at once (with a queue limit for backpressure), [`UseThreadExecutor`](capabilities/thread-executor.md) gives sync tools a bounded thread pool, and a slow tool can run as a [background tool](harness/background-tools.md) (Harness) while the agent keeps working.

Pin your versions: Pydantic AI follows its [version policy](version-policy.md), and Pydantic AI Harness has [its own](harness/index.md#version-policy). With a durable engine, a deploy also changes what in-flight runs replay against: adding a tool or changing a type the engine stored can break a run started under the old code, so let those runs drain on the old version or follow the engine's versioning guidance, such as [Temporal's](durable_execution/temporal.md#agent-run-context-and-dependencies) or [AWS Lambda's](harness/aws-lambda.md#constraints).
