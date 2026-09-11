# Pydantic AI vs Vercel AI SDK

The Vercel AI SDK is the default way to talk to models from TypeScript, and its real strength isn't
the agent loop, it's the wire between your server and your React app. Streaming text into a UI,
rendering tool calls as they happen, asking the user to approve one, resuming after they answer: it
does all of that well, and no Python library matches it, ours included.

Version 7 has grown agent machinery too. There's a `ToolLoopAgent`, helpers for building an agent UI
stream, a provider registry, middleware, tool approval errors as first-class types, and
`uploadSkill` for provider-hosted skills. Cancellation is the standard JavaScript idiom: pass an
`abortSignal`.

So the real question here is a simple one: which language does your agent live in?

## If your product is TypeScript

Use the AI SDK. Running Python for the agent means a service boundary, a deployment, and a second
language in your repository, and unless you need something specific from the Python side, that trade
usually isn't worth it.

The reasonable middle is a split: the AI SDK owns the browser and the streaming, and a Python agent
sits behind it as an HTTP endpoint. Pydantic AI has UI adapters, including one for the AI SDK's own
protocol, so the front end doesn't need to know what language answered.

## If your agent lives in Python

Then the comparison is worth having, and it's mostly about how much is decided for you.

**Cancellation.** `abortSignal` aborts the request. What you keep afterwards is whatever you collected
while streaming. In Pydantic AI, stopping raises `RunCancelled`, that exception carries the whole
conversation, and passing it to the next run continues from there.

**How a response becomes a value.** The AI SDK picks a sensible strategy for you. Pydantic AI makes it
explicit, so the same schema can arrive as a tool call, as native structured output, or as text you
transform. That matters when a provider's structured output mode is unreliable for your schema and you
want to switch how the answer comes back without touching the schema or the tools.

**Trusted state.** The AI SDK's runtime context is a loosely typed bag that travels with the call.
Pydantic AI's `deps_type` is a typed argument on the agent: tools read it through `RunContext`, and
the type checker knows the shape.

**Crash recovery.** There isn't a first-party story in the AI SDK; there's no durable agent export in
version 7. Pydantic AI runs can be wrapped by Temporal, DBOS, Prefect, Restate, Kitaru, or Airflow
without changing the agent.

## Side by side

| | Vercel AI SDK 7.0.97 | Pydantic AI 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| Streaming to a UI | Its whole reason for existing | Adapters, including for the AI SDK's protocol |
| The agent loop | `ToolLoopAgent`, with `stopWhen` for control | The loop is a value; `agent.iter()` drives it step by step |
| Stopping a run | `abortSignal` aborts the request | `RunCancelled` carries the conversation; resume is a normal run |
| Trusted state | A loosely typed runtime context | `deps_type` plus `RunContext`: a typed dependency API |
| Structured output | Chosen for you | You choose: tool call, native, or transformed text |
| Skills | `uploadSkill` to a provider | Capabilities that load on demand and round-trip to YAML |
| Budgets | `stopWhen` on steps; no money limit | Requests, tool calls, tokens, and `cost_limit` in USD when pricing data is available, checked after each response |
| Crash recovery | Not first-party | Six engines wrap the agent object |
| Testing offline | Subclass their provider spec yourself; it works well | `TestModel` and `FunctionModel` included |
| Evals | Not first-party | `pydantic-evals` in your test suite using the agent's own types |

## FAQ

**Can I use both?**
Yes, and it's a common setup. The AI SDK handles the UI stream; Pydantic AI answers behind it through
a UI adapter.

**Is Pydantic AI a drop-in replacement?**
No, it's a different language. What ports is the thinking: tools, prompts, schemas.

---

*Checked against `ai` 7.0.97 and Pydantic AI 2.42 on 2026-09-10. The AI SDK facts come from installing the
package and reading its exports and type definitions, `ToolLoopAgent`, `uploadSkill`, the approval error
types, `abortSignal`, `stopWhen`, and the absence of any durable-agent export. We recheck this page's version
pins and behaviour claims each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
