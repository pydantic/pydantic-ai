# Going to Production

The questions teams ask before shipping an agent, with the short answer and where the full one
lives. The agent you build runs in the user's own process: there is no Pydantic-hosted agent
runtime, so "deploying" means deploying their Python application.

The docs page this mirrors is [Going to Production](https://pydantic.dev/docs/ai/overview/production/).

## Start from how the run begins

Who is waiting decides the shape of the whole thing. Ask before designing.

| How it starts | What you build | What to get right |
|---|---|---|
| A person waiting in a web app | The agent runs inside a route of their application | [Stream to the frontend](#serve-it-to-a-frontend) and store each run's `Conversation` [in their database](#keep-conversation-state) |
| Nobody waiting: a queue, event or schedule | A job in their task queue, or a step in their pipeline | Usually returns [structured output](./AGENTS-CORE.md) for their code to store; decide [what a crash costs](#decide-what-a-crash-costs) |
| A person waiting in a terminal | The run lives as long as the session, so nothing to host | `Agent.to_cli()`, or `Coder` from the Harness for a coding agent; to keep a conversation past the session, loop around `agent.run()` and store the `Conversation` |

## Decide what a crash costs

A run in flight **dies with the process**. For most agents, starting it again is fine, and that is
the right default: do not reach for durable execution because production sounds like it needs it.

Reach for more only when one of these is true:

- Starting the run over would repeat a lot of paid-for work or side effects.
- The run is a workflow that itself waits days, for an [approval](#human-approval-that-resumes-later).

Then there are two levels, and they are not the same:

- **`StepPersistence`** (Harness) continues from the last tool-call snapshot, with no engine to run.
- **Durable execution** survives a crash partway through a step, on an engine the user operates:
  `pydantic-ai-slim` with one engine's extra, such as `[dbos]` or `[temporal]`. See
  [Orchestration and Integrations](./ORCHESTRATION-AND-INTEGRATIONS.md).

## Bound cost and runaway loops

Both, and they answer different questions:

- `UsageLimits` per run: this run cannot exceed N requests or tokens.
- `SpendLimits` (Harness) per day, tenant or month: this tenant cannot exceed a budget.

A loop that never terminates is a cost incident, so set the per-run limit on every agent that runs
unattended, not just the expensive ones.

## Survive a provider outage

A fallback model, transport-level retries, or Gateway failover. Pick by what fails: a fallback model
covers a provider being down, retries cover a flaky connection, and the Gateway covers both without
the user's code changing.

## Where API keys go

Environment variables, or one Pydantic AI Gateway key standing in for every provider. Never in
source, never in a prompt, and never echoed into telemetry (see **Telemetry safety** in `SKILL.md`).

## Keep conversation state

Store the run's `Conversation` in the user's database; a `ConversationStore` does it for them. This
is the piece most often got wrong by re-sending a prompt instead of resuming, which repeats the tool
calls. See [Input and History](./INPUT-AND-HISTORY.md) and **A failed run's work isn't lost** in
`SKILL.md`.

## Human approval that resumes later

[Deferred tools](./TOOLS-ADVANCED.md) end the run, and a later run resumes it. The run does not sit
in memory waiting, which is why the gap can be days and why this works in a plain web app without
durable execution.

## Limit what the agent can touch

Three separate controls, often needed together: per-caller tool exposure, approval on the risky
tools, and a sandbox for model-written code. `LocalWorkspace` isolates nothing: for untrusted code
use a sandbox provider. See [Workspaces](./WORKSPACES.md).

## See what happened

OpenTelemetry traces of every run, model request and tool call, in Logfire or any OTel backend. Set
this up by default, before shipping rather than after the first incident; `SKILL.md` has the setup
and the credential handling.

## Test and evaluate changes

Unit tests with `TestModel` first, then evals on datasets and on live traffic. The second half is
the one teams skip: a prompt change that passes unit tests can still regress quality, and only an
eval catches it. See [Testing and Debugging](./TESTING-AND-DEBUGGING.md).

## Serve it to a frontend

AG-UI or Vercel AI, streaming from any ASGI app. See
[Orchestration and Integrations](./ORCHESTRATION-AND-INTEGRATIONS.md).

## Deploy it

As the user's own Python process: a web app, a worker, a pipeline step or a function. Nothing of
ours runs anywhere else, which is usually the answer a platform team is looking for.

## For enterprise questions

What leaves their infrastructure, which models they may use, approvals and audit trails, central
model access and spending caps, and compliance or self-hosting are answered by
[Pydantic AI for Enterprise](https://pydantic.dev/docs/ai/overview/enterprise/). Pydantic AI is
MIT-licensed and runs on Azure OpenAI, Amazon Bedrock and Google Vertex AI as well as on models the
user hosts themselves.

For how real systems are structured, see
[Built with Pydantic AI](https://pydantic.dev/docs/ai/overview/built-with/).
