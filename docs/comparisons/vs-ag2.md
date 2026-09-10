# Pydantic AI vs AG2

If you remember AG2 as the community continuation of AutoGen — `ConversableAgent`, `UserProxyAgent`,
`GroupChat` — that library is gone. Version 1.0 is a rewrite, and the old names aren't importable from
the top level any more. Anything you read in an AutoGen-era tutorial no longer applies, which is worth
knowing before you either adopt it or inherit a codebase that uses it.

What replaced it is interesting, because it converged on a lot of the same ideas we did. AG2 1.0.4 has
`Agent`, a durable `Task` with a checkpoint store and `resume_from`, an `AgentSpec` describing an agent
as data, typed dependency injection through `Inject` and `Depends`, `ResponseSchema` for structured
output, `Task.cancel()`, and a `TestConfig` for scripting a model offline. There are first-party
modules for the Agent Client Protocol, agent-to-agent messaging, human-in-the-loop, evaluation, and
knowledge.

So this isn't a comparison between a typed framework and an untyped one. Both are typed. The
differences are narrower and mostly about how far the types reach and where durability comes from.

## Agents as data, and what validates them

Both libraries can describe an agent as a specification rather than code. AG2's `AgentSpec` carries a
name, prompt, tool names, and a response schema. Ours does the same job, round-trips to YAML and back,
and generates a JSON schema file so an editor can check it:

```python {title="spec_data_roundtrip.py"}
"""The agent is data: spec -> YAML + JSON schema -> a running agent.

The same declarative spec validates its templates against typed deps on
the dict path (Agent.from_spec), writes itself to a file with a
companion schema, loads back (Agent.from_file), and runs offline.
"""
import os
import tempfile

from pydantic import BaseModel

from pydantic_ai import Agent, AgentSpec


class Ctx(BaseModel):
    version: str

spec_data = {
    'name': 'checker',
    'model': 'test',  # offline stub backend
    'instructions': 'Reply with {{version}}.',
    'capabilities': [],
}
agent = Agent.from_spec(spec_data, deps_type=Ctx)  # templates meet typed deps here (dict path)
spec = AgentSpec.from_dict(spec_data)  # the same data, as a file-exportable spec

with tempfile.TemporaryDirectory() as d:
    yaml_path = os.path.join(d, 'agent.yaml')
    schema_path = os.path.join(d, 'agent.schema.json')
    spec.to_file(yaml_path, schema_path=schema_path)
    agent2 = Agent.from_file(yaml_path, deps_type=Ctx)  # validated again on load
    exists = os.path.exists(schema_path)

result = agent2.run_sync('go', deps=Ctx(version='v2'))
print(f'spec -> YAML + schema file (exists={exists}) -> running agent, offline')
#> spec -> YAML + schema file (exists=True) -> running agent, offline
print(f'loaded-from-file output: {result.output!r}')
#> loaded-from-file output: 'success (no tool calls)'
assert exists
assert result.output == 'success (no tool calls)'


```


The part we'd point at is what happens to the prompt. A Pydantic AI spec's template is checked against
your dependencies type when the spec loads, so `{{customer_nme}}` fails immediately with
`TemplateSchemaError: Field 'customer_nme' not found in schema` rather than rendering as empty text at
two in the morning. Worth being precise about our own limit: that check runs when a spec is loaded from
a dictionary or a file, not when you build an `AgentSpec` object directly in Python.

## Where durability comes from

AG2 puts it in the framework. A `Task` takes a `checkpoint_store` and a `resume_from`, and cancelling
is a state change on the task envelope — `Task.cancel()` moves it to cancelled and peers see the event.
For a checkpointed task graph, that's a coherent design and arguably a nicer fit than ours.

Pydantic AI puts it outside. A run is an ordinary coroutine, so a durable engine wraps the agent:
Temporal, DBOS, and Prefect ship in the repository, and Restate, Kitaru, and Airflow adapters live in
those projects. The trade is real. Theirs works with no infrastructure; ours means you already run
something like Temporal, and in exchange the retry policy, the durability guarantees, and the operational
tooling are that engine's, not a framework's reimplementation of them.

Cancellation splits the same way. AG2's is a property of a durable task. Ours is a property of a run:
a `CancellationToken` usable from another thread and across several runs at once, a tool that can stop
its own run with `ctx.cancel()`, and a `RunCancelled` exception carrying the history so the next run
picks up where it stopped.

## Side by side

| | AG2 1.0.4 | Pydantic AI 2.42 |
|---|---|---|
| Agent as data | `AgentSpec` with name, prompt, tools, response schema | `AgentSpec` round-tripping to YAML, with a generated JSON schema |
| Prompt validation | Not checked against a dependencies type | Checked at load; a bad field raises `TemplateSchemaError` |
| Typed dependencies | `Inject` and `Depends` type what tools receive | `deps_type` types it *and* keeps it out of the model's reach |
| Structured output | `ResponseSchema` | `output_type`, with explicit control over how it goes over the wire |
| Durability | `Task` with a checkpoint store and `resume_from` | Six engines wrap the agent; you pick which |
| Stopping a run | `Task.cancel()` on the task envelope | `CancellationToken` across runs, `ctx.cancel()` in a tool, resumable history |
| Testing offline | `TestConfig` scripts model events, including tool calls and errors | `TestModel` and `FunctionModel`, plus a global block on real calls |
| Protocols | ACP and A2A first-party | ACP through the harness |
| Migration | The AutoGen-era API is gone at 1.0 | — |

## Choose AG2 when

- A checkpointed task graph is how you think about the work, and you want that without running an
  orchestration engine.
- Its first-party ACP and A2A modules match your interoperability plans.
- You're already on AG2 1.x and it's working.

## Choose Pydantic AI when

- You want durability from an engine your company already operates and trusts.
- You want the model kept away from identity and credentials, not just tool inputs typed.
- You want a spec whose prompt is validated before it ever runs.
- You want stopping a run to give you back a conversation you can resume.

## FAQ

**I have AutoGen-era code. What now?**
It won't run on AG2 1.x unchanged either — the classic API is gone. If a rewrite is happening
regardless, that's the moment to compare rather than assume.

**What does AG2 do better?**
Durable tasks with no infrastructure behind them, and a genuinely broad first-party protocol story for
a project of its size.

**Are the two AgentSpecs compatible?**
No. Same idea, different shape. Translating one is mechanical but manual.

---

*Checked against ag2 1.0.4 and Pydantic AI 2.42 on 2026-09-10. The AG2 facts come from reading the
installed package: top-level exports, `Task` parameters including `checkpoint_store` and `resume_from`,
`Task.cancel`, and `TestConfig`. A full scripted `Task.run()` was not completed, so runtime behaviour
is not claimed here. The Pydantic AI example is executed by this repository's test suite. We recheck this page's
version pins and behaviour claims each time Pydantic AI ships a minor release; if something here has
gone stale, [tell us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
