# Runtime Capability Creation

Runtime capability creation lets an agent write, validate, and persist Pydantic AI
capabilities during one run for activation on the next.

`CapabilityCreation` is for capabilities authored by the agent as Python source. For
capabilities written or selected by application code, see
[Building Custom Capabilities](https://pydantic.dev/docs/ai/capabilities/custom/).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/capability_creation/)

## The problem

A coding agent often discovers, mid-task, that it wants a behavior its host does not
yet have: a guardrail, an extra instruction, a tool, a request hook. The capability
surface to express that already exists -- but only a developer can write a capability
class, wire it into the agent, and restart. Without runtime capability creation, the
agent cannot author that extension during a run and make it available to the next run.

## The solution

`CapabilityCreation` exposes three tools:

- `author_capability(name, code)` -- write `code` to `<directory>/<name>.py`, import it,
  and validate it. Validation requires exactly one
  `pydantic_ai.capabilities.AbstractCapability` subclass that constructs with no
  arguments; the side-effect-free static getters (`get_instructions`, `get_toolset`,
  `get_native_tools`, `get_model_settings`, `get_serialization_name`) are exercised. The
  async lifecycle hooks are not run -- they need a live `RunContext`.
- `list_authored_capabilities()` -- list authored capabilities with status and any
  validation error.
- `disable_authored_capability(name)` -- stop a capability from being injected on the
  next run.

A "hook" is not a standalone object in pydantic-ai -- it is a method on a capability. So
authoring a hook means authoring a capability that overrides one lifecycle method.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import CapabilityCreation

creation = CapabilityCreation(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[creation])
```

`CapabilityCreation` also contributes static, cache-stable system-prompt guidance
explaining these tools. Leave `guidance=None` for the default text, or pass your own
string; set `guidance=''` to omit it entirely.

## Activation boundary

Writing and validation happen in the current run; activation happens on the next run.
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
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness import CapabilityCreation

creation = CapabilityCreation(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[creation])

history = None
done = False
next_prompt = 'Start the task.'
while not done:
    extra = creation.store.load_active()
    result = await agent.run(next_prompt, message_history=history, capabilities=extra)
    history = result.all_messages()
    # ... decide `next_prompt` and `done` from `result` ...
```

Because the capabilities also persist to disk (`<directory>/<name>.py` plus a
`manifest.json` index), a fresh process picks them up by constructing a new
`CapabilityCreation` over the same `directory` and calling `store.load_active()`.

`manifest.json` records each capability's name, module file, class name, status, and last
validation error -- the surface a UI can read to show what the agent has authored.

Capability names must be lowercase letters, digits, and underscores, starting with a
letter; reusing a name replaces the previous capability of that name.

## Trust boundary

Authoring executes arbitrary Python in-process at import, construction, and run time. That
is the same trust boundary an agent that already runs shell commands and edits files
operates under, which is the deliberate choice here. Do not point it at a directory whose
contents you would not run yourself, and treat authored capabilities as code the agent is
executing on your host.

Because authored capabilities hold live code, they are not spec-serializable
(`get_serialization_name()` returns `None`) and are persisted as source rather than as an
agent spec.

## Typing

Imported authored code is dynamic, but nothing typed `Any` crosses back into the harness:
every value pulled from an authored module is narrowed with `isinstance`/`issubclass`
before use, and loaded instances are typed `AbstractCapability[object]`. Because
`AgentDepsT` is contravariant, an `AbstractCapability[object]` is accepted by any agent's
`capabilities=` parameter.
