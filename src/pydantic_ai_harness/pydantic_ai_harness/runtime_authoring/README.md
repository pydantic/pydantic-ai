# RuntimeAuthoring

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.runtime_authoring import RuntimeAuthoring
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

Let an agent author, validate, and persist real pydantic-ai capabilities at runtime.

## The problem

A coding agent often discovers, mid-task, that it wants a behavior its host does not
yet have: a guardrail, an extra instruction, a tool, a request hook. The capability
surface to express that already exists -- but only a developer can write a capability
class, wire it into the agent, and restart. The agent itself cannot extend its own host
while it runs.

## The solution

`RuntimeAuthoring` exposes three tools:

- `author_capability(name, code)` -- write `code` to `<directory>/<name>.py`, import it,
  and validate it. Validation requires exactly one
  `pydantic_ai.capabilities.AbstractCapability` subclass that constructs with no
  arguments; the side-effect-free static getters (`get_instructions`, `get_toolset`,
  `get_native_tools`, `get_model_settings`, `get_serialization_name`) are exercised. The
  async lifecycle hooks are not run -- they need a live `RunContext`.
- `list_authored_capabilities()` -- list authored capabilities with status and any
  validation error.
- `disable_authored_capability(name)` -- stop a capability from being injected.

A "hook" is not a standalone object in pydantic-ai -- it is a method on a capability. So
authoring a hook means authoring a capability that overrides one lifecycle method.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.runtime_authoring import RuntimeAuthoring

authoring = RuntimeAuthoring(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[authoring])
```

## Activation boundary

A capability **cannot** be added to a live, already-executing run. pydantic-ai resolves
the effective capability set once at the start of each run (the run's `root_capability`
is fixed; there is no setter). So an authored capability is live on the **next**
`agent.run(...)`, not the run that authored it. This mirrors Loopy's runtime personas,
which are usable on the next delegate call rather than mid-execution -- but one notch
coarser: a persona adds no tools or hooks (it rides a single generic `delegate` tool), so
it is usable later in the same run, whereas a full capability contributes tools and hooks
that only exist once the run's toolset and capability chain are assembled at run start.

### Integration contract

The orchestrator drives the loop, so it owns the one-line contract: thread the store's
active capabilities into each run. With `agent.run(..., capabilities=...)`, the authored
capability is live on the very next loop iteration -- no process restart.

```python
history = None
while not done:
    extra = authoring.store.load_active()
    result = await agent.run(next_prompt, message_history=history, capabilities=extra)
    history = result.all_messages()
```

Because the capabilities also persist to disk (`<directory>/<name>.py` plus a
`manifest.json` index), a fresh process picks them up by constructing a new
`RuntimeAuthoring` over the same `directory` and calling `store.load_active()`.

`manifest.json` records each capability's name, module file, class name, status, and last
validation error -- the surface a UI can read to show what the agent has authored.

## Trust boundary and the sandboxed alternative

Authoring executes arbitrary Python in-process at import, construction, and run time. That
is the same trust boundary an agent that already runs shell commands and edits files
operates under, which is the deliberate choice here.

The sandboxed alternative is the dormant `pa` Monty hook-slot registration system in the
Loopy tree (`pa/slots.py`, `pa/registration_tools.py`, `pa/capability.py`
`PaRegistrations`, `pa/registrations.py`, `pa/registration_runtime.py`,
`pa/monty_bridge.py`). It wires type-checked, resource-limited, allowlist-gated
`pydantic_monty` snippets into typed hook slots instead of importing native `.py`. It
trades native power for sandboxing and is also next-run-only. `RuntimeAuthoring` chooses
native authoring; this note records the alternative so the tradeoff is on file.

## Typing

Imported authored code is dynamic, but nothing typed `Any` crosses back into the harness:
every value pulled from an authored module is narrowed with `isinstance`/`issubclass`
before use, and loaded instances are typed `AbstractCapability[object]`. Because
`AgentDepsT` is contravariant, an `AbstractCapability[object]` is accepted by any agent's
`capabilities=` parameter.
