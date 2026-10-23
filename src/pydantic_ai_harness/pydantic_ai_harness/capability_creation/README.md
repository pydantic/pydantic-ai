# Runtime Capability Creation

Runtime capability creation lets an agent write, validate, and persist Pydantic AI
capabilities during one run for activation on the next.

`CapabilityCreation` is for capabilities authored by the agent as Python source. For
capabilities written or selected by application code, see
[Building Custom Capabilities](https://pydantic.dev/docs/ai/capabilities/custom/).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/capability_creation/)

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
  arguments. It calls the static getters (`get_instructions`, `get_toolset`,
  `get_native_tools`, `get_model_settings`, `get_serialization_name`) and checks that the
  agent can use what they return. The async lifecycle hooks are not run -- they need a
  live `RunContext`.
- `list_authored_capabilities()` -- list authored capabilities with status and any
  validation error.
- `disable_authored_capability(name)` -- mark a capability disabled, so `load_active()`
  skips it.

A "hook" is not a standalone object in pydantic-ai -- it is a method on a capability. So
authoring a hook means authoring a capability that overrides one lifecycle method.

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import CapabilityCreation

creation = CapabilityCreation(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace('.'), creation])
```

`directory` is a path on the machine running the agent, relative to the process's current
directory, not a workspace path.

`CapabilityCreation` also contributes static, cache-stable system-prompt guidance
explaining these tools. Leave `guidance=None` for the default text, or pass your own
string; set `guidance=''` to omit it entirely.

## Activation boundary

Writing and validation happen in the current run; activation happens on the next run.
A capability **cannot** be added to a live, already-executing run. pydantic-ai resolves
the effective capability set once at the start of each run (the run's `root_capability`
is fixed; there is no setter). So an authored capability is live on the **next**
`agent.run(...)`, not the run that authored it. Authoring writes and validates the
capability immediately, but its tools and hooks only exist once the next run's toolset and
capability chain are assembled at run start.

### Integration contract

The successful `author_capability` result tells the model the capability does not take effect
in the current run. Writing and validating a capability does not schedule or inject it automatically.

The orchestrator drives the loop, so it owns the one-line contract: thread the store's
active capabilities into each run. With `agent.run(..., capabilities=...)`, the authored
capability is live on the very next loop iteration -- no process restart:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import CapabilityCreation

creation = CapabilityCreation(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-5', capabilities=[LocalWorkspace('.'), creation])

history = None
done = False
next_prompt = 'Start the task.'
while not done:
    extra = creation.store.load_active()
    result = await agent.run(next_prompt, message_history=history, capabilities=extra)
    history = result.all_messages()
    # ... decide `next_prompt` and `done` from `result` ...
```

`store.load_active()` re-imports every active authored capability and runs the same checks
as `author_capability` before returning it. An entry that fails (corrupt source,
construction error, a getter with a wrong return type) is skipped rather than raised, and
its error is recorded in the manifest, so one bad capability does not block the others.

Because the capabilities also persist to disk (`<directory>/<name>.py` plus a
`manifest.json` index), a fresh process picks them up by constructing a new
`CapabilityCreation` over the same `directory` and calling `store.load_active()`.

`manifest.json` records each capability's name, module file, class name, status, and last
validation error -- the surface a UI can read to show what the agent has authored. The
manifest is written atomically. A manifest that cannot be read (malformed JSON, or a field
value this version does not accept) is read as empty, and `author_capability` and
`disable_authored_capability` raise `UserError` instead of replacing it, so its entries are
kept until you repair or remove it.

The store serializes its own tool calls, but not other processes or agents: point each
`directory` at one agent at a time, or concurrent authoring can lose manifest updates.

Capability names are at most 64 lowercase letters, digits, and underscores, starting with a
letter; reusing a name replaces the previous capability of that name. Code that fails
validation is still written to disk (so it can be inspected) and recorded with its
`last_error` set; `load_active()` skips it.

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), each authoring, listing, and disabling
call to the store is recorded, so a recovered run reuses the result instead of
writing the authored module and manifest again. Temporal and Prefect record
each tool call in its own activity or task. DBOS runs function tools in
workflow code, so there the store call runs as its own step.

The records are named after the capability's `id`, which defaults to
`capability_creation`, so durable execution needs no configuration.

Inside a Temporal or DBOS workflow, a run refuses capabilities passed with
`agent.run(..., capabilities=...)`, because those engines register a capability's
activities or steps when the capability is bound to the agent. The integration contract
above therefore cannot activate authored capabilities there. A Prefect flow accepts an
authored capability that only adds hooks, instructions, or model settings, and refuses one
whose `get_toolset()` contributes a function, MCP, or dynamic toolset. The authoring tools
work under all three.

## Trust boundary

`CapabilityCreation` imports model-written Python into the agent's own process, on this
machine. A run therefore refuses to start unless its
[workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/) is a writable `LocalWorkspace`: next to a
sandbox the model's code would run outside the sandbox, and a read-only workspace promises the
model that it changes nothing.

Authoring executes arbitrary Python in-process at import, construction, and run time. That
is the same trust boundary an agent that already runs shell commands and edits files
operates under, which is the deliberate choice here. Do not point it at a directory whose
contents you would not run yourself, and treat authored capabilities as code the agent is
executing on your host.

Authored code first runs when the model calls `author_capability`, not on the next run:
validation executes the module body, the constructor, and the static getters in the current
run, with the agent process's environment and permissions, even when validation then
rejects the code. Validation waits for that code to return, so authored code that blocks
also blocks the tool call, and cancelling the run does not interrupt it. Loads from
different threads run one at a time, so it also holds up every other authored capability
load in the process, including `store.load_active()`.

If the agent also has file or shell tools (such as `FileSystem` or `Shell`) over a
workspace that contains `directory`, as with `LocalWorkspace('.')` and
`directory=Path('.authored')`, it can edit authored files and `manifest.json` directly,
without going through `author_capability`. Keep `directory` outside that workspace if every
authored file should pass through validation.

Because authored capabilities hold live code, they are not spec-serializable
(`get_serialization_name()` returns `None`) and are persisted as source rather than as an
agent spec.

## Telemetry

`CapabilityCreation` emits no spans of its own. Core's tool spans record each
`author_capability`, `list_authored_capabilities`, and `disable_authored_capability` call,
and authoring and validation run inside the `author_capability` span.
