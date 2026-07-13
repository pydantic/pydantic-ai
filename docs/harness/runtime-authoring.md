---
title: Runtime Authoring
description: Let an agent write, validate, and persist real pydantic-ai capabilities at runtime, live on the next run.
---

# Runtime Authoring

`RuntimeAuthoring` lets an agent author, validate, and persist real pydantic-ai capabilities while it runs. It exposes three tools that let the model write a capability class to disk as Python source, validate it immediately, and manage the set of authored capabilities. Each authored capability is a real `pydantic_ai.capabilities.AbstractCapability` subclass, so it can contribute instructions, model settings, a toolset, native tools, or a lifecycle hook.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/runtime_authoring/)

> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

## The problem

A coding agent often discovers, mid-task, that it wants a behavior its host does not yet have: a guardrail, an extra instruction, a tool, a request hook. The capability surface to express that already exists -- but normally only a developer can write a capability class, wire it into the agent, and restart. The agent itself cannot extend its own host while it runs.

## The solution

`RuntimeAuthoring` exposes three tools:

- `author_capability(name, code)` -- write `code` to `<directory>/<name>.py`, import it, and validate it. Validation requires exactly one `pydantic_ai.capabilities.AbstractCapability` subclass that constructs with no arguments; the side-effect-free static getters (`get_instructions`, `get_toolset`, `get_native_tools`, `get_model_settings`, `get_serialization_name`) are exercised. The async lifecycle hooks are not run -- they need a live `RunContext`.
- `list_authored_capabilities()` -- list authored capabilities with their status and any validation error.
- `disable_authored_capability(name)` -- stop a capability from being injected on the next run.

A "hook" is not a standalone object in pydantic-ai -- it is a method on a capability. So authoring a hook means authoring a capability that overrides one lifecycle method. A single overridden hook is a valid capability.

## Usage

Construct `RuntimeAuthoring` with a `directory` for the authored files, then add it to the agent's `capabilities`:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.runtime_authoring import RuntimeAuthoring

authoring = RuntimeAuthoring(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[authoring])
```

The agent can now call `author_capability`, `list_authored_capabilities`, and `disable_authored_capability`. `RuntimeAuthoring` also contributes static, cache-stable system-prompt guidance explaining these tools. Leave `guidance=None` for the default text, or pass your own string; set `guidance=''` to omit it entirely.

## Activation boundary

A capability **cannot** be added to a live, already-executing run. pydantic-ai resolves the effective capability set once at the start of each run (the run's root capability is fixed; there is no setter). So an authored capability is live on the **next** `agent.run(...)`, not the run that authored it. Authoring writes and validates the capability immediately, but its tools and hooks only exist once the next run's toolset and capability chain are assembled at run start.

### Integration contract

The orchestrator drives the loop, so it owns the one-line contract: thread the store's active capabilities into each run via `agent.run(..., capabilities=...)`. With that in place, the authored capability is live on the very next loop iteration -- no process restart:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.runtime_authoring import RuntimeAuthoring

authoring = RuntimeAuthoring(directory=Path('.authored'))
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[authoring])

history = None
done = False
next_prompt = 'Start the task.'
while not done:
    extra = authoring.store.load_active()
    result = await agent.run(next_prompt, message_history=history, capabilities=extra)
    history = result.all_messages()
    # ... decide `next_prompt` and `done` from `result` ...
```

`authoring.store` is the disk-backed `CapabilityStore` over the same `directory`. `store.load_active()` re-imports and re-constructs every active authored capability for injection into the next run. Entries that fail to load (corrupt source, construction error) are skipped, not raised, so one bad capability never blocks the rest.

## Persistence

Authored capabilities persist to disk: each is one `<directory>/<name>.py` file, indexed by a sibling `manifest.json`. A fresh process picks them up by constructing a new `RuntimeAuthoring` over the same `directory` and calling `store.load_active()`.

`manifest.json` records each capability's name, module file, class name, status (`active` or `disabled`), and last validation error. That is the surface a UI can read to show what the agent has authored. The manifest is written atomically (temp file plus `os.replace`), so a crash mid-write never leaves a partial file that reads back as "no capabilities".

Capability names must be lowercase letters, digits, and underscores, starting with a letter. Reusing a name replaces the previous capability of that name. A code that imports but fails validation is still written to disk (so it can be inspected) and recorded with its `last_error` set; `load_active()` skips it.

## Trust boundary

`RuntimeAuthoring` executes arbitrary Python in-process at import, construction, and run time. That is the same trust boundary an agent that already runs shell commands and edits files operates under, which is the deliberate choice here. Do not point it at a directory whose contents you would not run yourself, and treat authored capabilities as code the agent is executing on your host.

Because authored capabilities hold live code, they are not spec-serializable (`get_serialization_name()` returns `None`) and are persisted as source rather than as an [agent spec](/ai/core-concepts/agent-spec/).

## Typing

Imported authored code is dynamic, but nothing typed `Any` crosses back into the harness: every value pulled from an authored module is narrowed with `isinstance`/`issubclass` before use, and loaded instances are typed `AbstractCapability[object]`. Because `AgentDepsT` is contravariant, an `AbstractCapability[object]` is accepted by any agent's `capabilities=` parameter.

## API reference

::: pydantic_ai_harness.runtime_authoring.RuntimeAuthoring

::: pydantic_ai_harness.runtime_authoring.CapabilityStore
