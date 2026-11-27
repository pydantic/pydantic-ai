# Going to Production

The questions teams ask before shipping an agent, with the short answer and where the full one
lives. The agent you build runs in the user's own process: there is no Pydantic-hosted agent
runtime, so "deploying" means deploying their Python application.

The docs page this mirrors is [Going to Production](https://pydantic.dev/docs/ai/production/).

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

- **`StepPersistence`** (Harness) continues from the last tool-call snapshot, with no engine to run. A tool call in the ledger that never finished has an unknown outcome, so check the external system before re-running it. Snapshots hold messages only, not the usage and pending deferred calls a `Conversation` carries, so they complement storing it rather than replace it.
- **Durable execution** replays completed model requests and journaled tool calls after a crash, and **re-runs the one that was in flight from its start**. So a tool with side effects has to be safe to repeat, for instance by passing an idempotency key to the API it calls. This is the point to get right: a crash mid-call is *not* covered for you. Runs on an engine the user operates, via `pydantic-ai-slim` with one engine's extra such as `[dbos]` or `[temporal]`. See [Use Durable Execution](./ORCHESTRATION-AND-INTEGRATIONS.md#use-durable-execution).

## Bound cost and runaway loops

Both, and they answer different questions:

- `UsageLimits` per run: this run cannot exceed N requests or tokens.
- `SpendLimits` (Harness) per day, tenant or month: this tenant cannot exceed a budget. Two caveats that only matter in production: its default store is in memory, so a budget resets on every restart and is not shared between worker processes, and it is a brake rather than a reservation, so concurrent runs can overshoot. See [Spend](https://pydantic.dev/docs/ai/harness/spend/). Where neither caveat is acceptable, the Gateway enforces daily, weekly and monthly caps per project, user and key before the request reaches the provider.

A loop that never terminates is a cost incident, so set the per-run limit on every agent that runs
unattended, not just the expensive ones.

## Survive a provider outage

Inside an agent, the two capabilities are `RetryOnModelErrors`, which honors `Retry-After`, and `ModelFallback`, which moves to another model; they compose. See
[Handle Provider Failures](./AGENTS-CORE.md#handle-provider-failures).

Do not reach for a retrying HTTP transport here: that is for code outside an agent, and it stacks with the SDK's own retries. The Gateway can fail over between providers serving the same model, when they are configured as gateway endpoints, which is narrower than "it covers outages for you".

## Where API keys go

Environment variables, or one Pydantic AI Gateway key standing in for every provider. Never in
source, never in a prompt, and never echoed into telemetry (see **Telemetry safety** in `SKILL.md`).

## Keep conversation state

Store the run's `Conversation` in the user's database; a `ConversationStore` does it for them. **A conversation ID is a key, not a credential**: check that the caller may access a conversation before loading it, and serialize concurrent updates to one conversation, with a version column or a per-conversation lock, so two instances do not continue the same stored version. This
is the piece most often got wrong by re-sending a prompt instead of resuming, which repeats the tool
calls. See [Input and History](./INPUT-AND-HISTORY.md) and **A failed run's work isn't lost** in
`SKILL.md`.

## Human approval that resumes later

[Deferred tools](./TOOLS-ADVANCED.md) end the run, and a later run resumes it. The run does not sit
in memory waiting, which is why the gap can be days and why this works in a plain web app without
durable execution.

## Limit what the agent can touch

Three separate controls, often needed together: per-caller tool exposure
([Organize or Restrict Which Tools an Agent Can Use](./TOOLS-CORE.md#organize-or-restrict-which-tools-an-agent-can-use)),
approval on the risky tools
([Require Tool Approval](./TOOLS-ADVANCED.md#require-tool-approval-human-in-the-loop)),
and a sandbox for model-written code. `LocalWorkspace` isolates nothing: for untrusted code
use a sandbox provider. See [Workspaces](./WORKSPACES.md).

## See what happened

OpenTelemetry traces of every run, model request and tool call, in Logfire or any OTel backend. Set
this up by default, before shipping rather than after the first incident; `SKILL.md` has the setup
and the credential handling.

## Test and evaluate changes

Unit tests with `TestModel` first, then evals on datasets and on live traffic. The second half is
the one teams skip: a prompt change that passes unit tests can still regress quality, and only an
eval catches it. The two halves live in different references: [Testing and Debugging](./TESTING-AND-DEBUGGING.md) for the unit tests, and
[Systematically Verify Agent Behavior with Evals](./ORCHESTRATION-AND-INTEGRATIONS.md#systematically-verify-agent-behavior-with-evals) for the
evals, which is also where to start for [online evaluation](https://pydantic.dev/docs/ai/evals/online-evaluation/) of live traffic.

## Serve it to a frontend

`AGUIAdapter` or `VercelAIAdapter`, streaming from any ASGI app. The skill content is in
[Input and History](./INPUT-AND-HISTORY.md#correlate-runs-with-run_id-and-conversation_id).

**The adapter endpoint is not an authentication boundary.** Run it inside the application's own authenticated route, because messages submitted by a client are untrusted. Mounting `dispatch_request` on a bare route hands anyone who finds it an agent run.

## Deploy it

As the user's own Python process: a web app, a worker, a pipeline step or a function. Nothing of
ours runs anywhere else, which is usually the answer a platform team is looking for.

## For enterprise questions

What leaves their infrastructure, which models they may use, approvals and audit trails, central
model access and spending caps, and compliance or self-hosting are answered by
[Pydantic AI for Enterprise](https://pydantic.dev/docs/ai/production/enterprise/). Pydantic AI is
MIT-licensed and runs on Azure OpenAI, Amazon Bedrock and Google Vertex AI as well as on models the
user hosts themselves.

For how real systems are structured, see
[Built with Pydantic AI](https://pydantic.dev/docs/ai/overview/built-with/).
