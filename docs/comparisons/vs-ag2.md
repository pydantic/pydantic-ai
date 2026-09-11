# Pydantic AI vs AG2

If you remember AG2 as the community continuation of AutoGen, `ConversableAgent`, `UserProxyAgent`,
`GroupChat`, that library is gone. Version 1.0 is a rewrite, and the old names aren't importable from
the top level any more. Anything you read in an AutoGen-era tutorial no longer applies, which is worth
knowing before you either adopt it or inherit a codebase that uses it.

What replaced it landed on a lot of the same ideas we did. AG2
1.0.4 has `Agent`, a durable `Task` with a checkpoint store and `resume_from`, an `AgentSpec` that
describes an agent as data, typed dependency injection through `Inject` and `Depends`,
`ResponseSchema` for structured output, `Task.cancel()`, and a `TestConfig` for scripting a model
offline. Plus first-party modules for the Agent Client Protocol, agent-to-agent messaging,
human-in-the-loop, evaluation and knowledge. Of everything on these pages, AG2 landed nearest to
where we did.

This isn't a comparison between a typed framework and an untyped one. Both are typed. The difference
that decides the rest is where durability comes from.

## Where durability comes from

AG2 puts it in the framework. A `Task` takes a `checkpoint_store` and a `resume_from`, and cancelling
is a state change on the task envelope: `Task.cancel()` moves it to cancelled and peers see the event.
For a checkpointed task graph, that's a coherent design, and it works with no extra infrastructure.

Pydantic AI puts it outside. A run is an ordinary coroutine, so a durable engine wraps the agent:
Temporal, DBOS, and Prefect ship in the repository, and Restate, Kitaru, and Airflow adapters live in
those projects. The trade is real. Theirs works with no infrastructure; ours means you already run
something like Temporal, and in exchange the retry policy, the durability guarantees, and the
operational tooling are that engine's, not a framework's reimplementation of them.

Cancellation splits the same way. AG2's is a property of a durable task. Ours is a property of a run:
a `CancellationToken` usable from another thread and across several runs at once, a tool that can stop
its own run with `ctx.cancel()`, and a `RunCancelled` exception carrying the history so the next run
picks up where it stopped.

Both libraries can also describe an agent as data. AG2's `AgentSpec` carries a name, prompt, tool
names, and a response schema. Ours round-trips to YAML, generates a JSON schema file for editors, and
checks prompt templates against `deps_type` when you load a spec through
[`Agent.from_spec`][pydantic_ai.Agent.from_spec]. The two specs are the same idea, different shape;
translating one is mechanical but manual.

## Side by side

| | AG2 1.0.4 | Pydantic AI 2.42 |
|---|---|---|
| Agent as data | `AgentSpec` with name, prompt, tools, response schema | `AgentSpec` round-tripping to YAML, with a generated JSON schema |
| Prompt templates | Spec fields | Checked against `deps_type` at `Agent.from_spec` load |
| Typed dependencies | `Inject` and `Depends` type what tools receive | `deps_type` plus `RunContext`: a typed dependency API |
| Structured output | `ResponseSchema` | `output_type`, with explicit control over how it goes over the wire |
| Durability | `Task` with a checkpoint store and `resume_from` | Six engines wrap the agent; you pick which |
| Stopping a run | `Task.cancel()` on the task envelope | `CancellationToken` across runs, `ctx.cancel()` in a tool, resumable history |
| Testing offline | `TestConfig` scripts model events, including tool calls and errors | `TestModel` and `FunctionModel`, plus a global block on real calls |
| Budgets | No money limit | `cost_limit` in USD when pricing data is available, checked after each response; pair with `request_limit` |
| Protocols | ACP and A2A first-party | ACP through the harness (experimental) |
| Migration | The AutoGen-era API is gone at 1.0 | |

## FAQ

**I have AutoGen-era code. What now?**
It won't run on AG2 1.x unchanged either, the classic API is gone. If a rewrite is happening
regardless, that's the moment to compare rather than assume.

**Are the two AgentSpecs compatible?**
No. Same idea, different shape. Translating one is mechanical but manual.

---

*Checked against ag2 1.0.4 and Pydantic AI 2.42 on 2026-09-10. The AG2 facts come from reading the installed
package: top-level exports (`Agent`, `AgentSpec`, `Task`, `Inject`, `Depends`, `ResponseSchema`), `Task`
parameters including `checkpoint_store` and `resume_from`, `Task.cancel`, and `TestConfig`. A full scripted
`Task.run()` was not completed, so runtime behaviour is not claimed here. We recheck this page's version pins
and behaviour claims each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
