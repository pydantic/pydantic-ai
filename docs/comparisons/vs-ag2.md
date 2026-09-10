# Pydantic AI vs AG2

**AG2** is the protocol-rich framework — the v1 rewrite ships a checkpointed `Task` state
machine (`checkpoint_store`, `resume_from`), its own `AgentSpec`, `Inject`/`Depends` typed inputs,
`ResponseSchema`, and envelopes for cancel/expire/fail.

**Pydantic AI** is an agent that is data — a spec that fails at load, serializes with a
companion schema, and runs offline — plus durable wraps and typed, repairable history.

*Verified against `ag2 1.0.4` (2026-09-10). Pydantic AI claims below are self-contained scripts —
offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | AG2 | Pydantic AI |
|---|---|---|
| Agent as data | `AgentSpec` + `ResponseSchema` (the closest spec story among competitors) | `AgentSpec`: templates validated against typed deps at load; YAML + JSON schema + round-trip (proven below) |
| Durability | Checkpointed `Task` with `resume_from` — a durable *state machine* | Six engine wraps; the run stays an ordinary coroutine |
| Cancellation | Durable envelope: `Task.cancel()` flips metadata, emits `TaskCancelled`; no in-flight abort | Typed in-process cancel (`ctx.cancel()`, token) + durable units raise an explanatory error (replay-safety) |
| Typed inputs | `Inject`/`Depends`/`ResponseSchema` | One `deps_type` from construction through tools, specs, tests, evals |
| History | Task state | Typed, repairable message history |

## Prove it yourself

Their spec protocol is the closest to ours — so prove the difference where it matters: validation at
load, a file + companion schema, and an agent that runs offline from that file. YAML round-trips need
the optional `spec` extra for PyYAML (`pip install 'pydantic-ai[spec]'`):

```python {title="spec_data_roundtrip.py"}
"""The agent is data: spec -> YAML + JSON schema -> a running agent.

The same declarative spec validates its templates against typed deps at
load (on the dict/YAML path), writes itself to a file with a companion
schema, loads back, and runs offline.
"""
import os
import tempfile

from pydantic import BaseModel

from pydantic_ai import Agent, AgentSpec

class Ctx(BaseModel):
    version: str

spec = AgentSpec.from_dict(
    {
        'name': 'checker',
        'model': 'test',  # offline stub backend
        'instructions': 'Reply with {{version}}.',
        'tools': [],
        'capabilities': [],
    }
)
agent = Agent.from_spec(spec, deps_type=Ctx)

with tempfile.TemporaryDirectory() as d:
    yaml_path = os.path.join(d, 'agent.yaml')
    schema_path = os.path.join(d, 'agent.schema.json')
    spec.to_file(yaml_path, schema_path=schema_path)
    agent2 = Agent.from_file(yaml_path, deps_type=Ctx)  # validated again on load
    exists = os.path.exists(schema_path)

result = agent2.run_sync('go', deps=Ctx(version='v2'))
print(f'spec -> YAML + schema file (exists={exists}) -> running agent, offline')
print(f'loaded-from-file output: {result.output!r}')
assert exists
assert result.output == 'success (no tool calls)'
```

```text
spec -> YAML + schema file (exists=True) -> running agent, offline
loaded-from-file output: 'success (no tool calls)'
```


## Key differences

**AG2.** the v1 rewrite is genuinely different — a durable task state machine with envelopes
is the right shape when the *whole system* is checkpointed, and their protocol surface (ACP, A2A,
hitl, live) is broad.

**Pydantic AI.** the spec is validated against typed deps at load (dict/YAML path), serializes with a schema
file, and the loaded agent runs offline. Durable units wrap the same ordinary run; `ctx.cancel()`
inside a unit raises an explanatory error instead of replay-diverging.

## When to choose AG2

You are building a checkpointed, protocol-spanning system (multi-agent with ACP/A2A/live) and your
durable unit of truth is the task state machine.

## When to choose Pydantic AI

You want an agent that is data you can validate, ship, and run — with the loop staying yours, typed
history, and durable engines you choose.

## Summary

Both treat the agent as data. Ours validates it against deps at load and runs it offline:
'success (no tool calls)'.

*AG2 behavior pinned to 1.0.4 (installed, probed); records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*