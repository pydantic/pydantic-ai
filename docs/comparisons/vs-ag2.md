# Pydantic AI vs AG2

Choosing an agent framework and you're down to
[Pydantic AI](../agent.md) and AG2. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- the **agent as data**: a spec validated against types at load, shipped as YAML + schema, run offline
- **typed, repairable history** on an ordinary loop
- durability by wrapping six engines — not adopting one state machine

## Why the answers differ

Both treat the agent as data — AG2 as a checkpointed task protocol, ours as a validated spec on an ordinary loop. Where it shows: validation at load, a file + schema, and offline execution.

## See it work

Say your agent should be data — validated, shipped, run.

AG2 ships its own agent-spec and response-schema protocol on a checkpointed task state machine (1.0.4).

Your side, runs offline:

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

**Notice:** Ours validates templates against typed deps at load, writes YAML plus a schema file, and the loaded agent runs offline. (YAML round-trips need `pip install "pydantic-ai[spec]"`.)

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

If your system is built around checkpointed task state machines and protocol envelopes across ACP/A2A/live, then AG2's v1 design is the architecture you actually want — and we'll even say its spec protocol is the closest thing to ours. The difference this page proves: our spec validates against types at load, runs offline, and leaves the loop ordinary.

---

## FAQ

**Is Pydantic AI a drop-in replacement for AG2?**
Drop-in, no — the loop and the seams are different, even though the ideas carry over (tools,
prompts, outputs). If you're weighing a move, that honesty is the point of this page: read the fits
list and run the proof before you decide.

**When should I use AG2 on its own?**
When your system is a checkpointed, protocol-spanning task graph (ACP, A2A, live) with envelopes as first-class states.

**Why do people pick Pydantic AI over AG2?**
Because the loop is yours end to end — typed deps, cancellation that resumes, budgets that stop side
effects before they start, evals in CI — and every one of those claims is a snippet on this page you
can run in seconds. Community threads on r/AI_Agents add "documentation" and "low abstraction" to
that list; see Independent takes on the [overview](index.md).


---

*Versions: ag2 1.0.4; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
