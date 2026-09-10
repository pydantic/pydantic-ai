# Pydantic AI vs AG2

You're choosing a Python agent framework and have narrowed it to [Pydantic AI](../agent.md) and AG2.
This page makes the call — and lets you check the evidence yourself: every snippet runs offline,
no API keys.

## Pydantic AI fits if you need

- the **agent as data**: a spec validated against types at load, shipped as YAML + schema, run offline
- **typed, repairable history** on an ordinary loop
- durability by wrapping six engines — not adopting one state machine

## Why the answers differ

Both treat the agent as data — AG2 as a checkpointed task protocol, ours as a validated spec on an ordinary loop. Where it shows: validation at load, a file + schema, and offline execution.

## See it work

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

## The details

| What you get | AG2 | Pydantic AI |
|---|---|---|
|---|---|---|
| Agent as data | `AgentSpec` + `ResponseSchema` (the closest spec story among competitors) | `AgentSpec`: templates validated against typed deps at load; YAML + JSON schema + round-trip (proven below) |
| Durability | Checkpointed `Task` with `resume_from` — a durable *state machine* | Six engine wraps; the run stays an ordinary coroutine |
| Cancellation | Durable envelope: `Task.cancel()` flips metadata, emits `TaskCancelled`; no in-flight abort | Typed in-process cancel (`ctx.cancel()`, token) + durable units raise an explanatory error (replay-safety) |
| Typed inputs | `Inject`/`Depends`/`ResponseSchema` | One `deps_type` from construction through tools, specs, tests, evals |
| History | Task state | Typed, repairable message history |

## If this answer doesn't fit you

If your system is built around checkpointed task state machines and envelopes across protocols (ACP/A2A/live), AG2's v1 design is the deliberate architecture for that. Their spec protocol is the closest competitor to what this page demonstrates — the difference is what validated agent at load time.

---

---

*Versions: ag2 1.0.4; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
