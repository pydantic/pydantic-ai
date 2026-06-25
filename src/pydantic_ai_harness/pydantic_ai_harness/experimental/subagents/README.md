# SubAgents

> [!WARNING]
> **Experimental.** This capability lives under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import it from the
> experimental path -- there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.subagents import SubAgent, SubAgents
> ```
>
> Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence **all**
> harness experimental warnings with a single filter (no per-capability lines needed):
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

Let an agent delegate self-contained tasks to named child agents.

## The problem

A single agent that does everything accumulates a large tool set and a long context. Splitting the work across specialized sub-agents keeps each context focused, but wiring up delegation by hand means writing a tool per agent, forwarding deps, threading usage limits, and telling the model what it can delegate to.

## The solution

`SubAgents` takes a sequence of `SubAgent` entries and exposes a single `delegate_task(agent_name, task)` tool. Each delegation runs the chosen sub-agent in its own run -- with its own message history, so it never sees the parent conversation -- and returns its output to the parent. The available sub-agents are listed in the system prompt as a static instruction, so the listing stays in the cached prefix.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.subagents import SubAgent, SubAgents

researcher = Agent('anthropic:claude-sonnet-4-6', name='researcher', description='Researches a topic and reports findings')
writer = Agent('anthropic:claude-sonnet-4-6', name='writer', description='Turns notes into polished prose')

orchestrator = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[SubAgents(agents=[SubAgent(researcher), SubAgent(writer)])],
)

result = orchestrator.run_sync('Research the history of TLS and write a one-paragraph summary.')
print(result.output)
```

A delegate's name -- how the parent model refers to it, and how it is listed in the prompt -- is the agent's own `name`, or a `SubAgent(name=...)` override. Two delegates resolving to the same name is an error, and an agent with no name and no override is rejected.

## The tool

| Tool | Purpose |
|---|---|
| `delegate_task(agent_name, task)` | Run the named sub-agent on a self-contained task and return its output. |

- The sub-agent runs with its own message history, so `task` must be self-contained.
- An unknown `agent_name` raises `ModelRetry`, so the model can correct itself.
- The result returned to the parent is `str(result.output)`.

## Deps, usage, tools, and capabilities

- **Deps are forwarded.** The parent run's `deps` are passed to each sub-agent, so sub-agents share the parent's `AgentDepsT` (enforced by the type signature -- every sub-agent is an `AbstractAgent[AgentDepsT, Any]`).
- **Usage is shared by default.** The parent's `usage` is passed to each sub-agent run, so token usage aggregates and a parent `usage_limits` applies across the whole agent tree. Set `forward_usage=False` to give each sub-agent run its own accounting.
- **Tools can be inherited.** With `inherit_tools=True`, the parent agent's own tools (registered directly or via `toolsets`) are added to each sub-agent run, on top of the sub-agent's own. Tools contributed by the parent's capabilities are not inherited: they are bound to capability instances registered in the parent run, and would arrive without the hooks and instructions they depend on. Use `shared_capabilities` to give sub-agents a capability. This also excludes the delegate tool itself, so a sub-agent can't recurse into further delegation. Off by default.
- **Capabilities can be shared.** `shared_capabilities` are applied to every sub-agent run -- e.g. give all sub-agents a common guardrail, memory, or planning capability without rebuilding each `Agent`.
- **Sub-agent events can be streamed.** Pass an `event_stream_handler` and it's forwarded to each sub-agent run, so the sub-agent's model-streaming and tool events surface to the caller (the handler receives the sub-agent's own `RunContext`).

## Per-delegate run controls

Each `SubAgent` carries its own budgets, so one delegate's controls do not touch the others. A `SubAgent` with no controls set runs with the `SubAgents` defaults.

```python
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.experimental.subagents import SubAgent, SubAgents

# reproducer and librarian are Agent instances, as in the example above.
orchestrator = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[
        SubAgents(
            agents=[
                SubAgent(reproducer, usage_limits=UsageLimits(request_limit=35), timeout_seconds=600, max_calls=1),
                SubAgent(librarian, usage_limits=UsageLimits(request_limit=18), timeout_seconds=300, max_calls=2),
            ]
        )
    ],
)
```

| Field | Effect |
|---|---|
| `usage_limits` | A request/token budget for one delegation. The child runs with its own usage accounting, so the budget counts only that child's requests and tokens (not the parent's or siblings'), even when `forward_usage=True`. The tradeoff: that child's tokens no longer aggregate into the parent's `usage`. Reaching the budget is a soft outcome (see below), not a run-stopping `UsageLimitExceeded`. |
| `timeout_seconds` | A wall-clock budget for one delegation. When the child exceeds it, its run is cancelled and the parent gets a soft steering message instead of hanging on the child. The cancelled child's `event_stream_handler` (if any) stops receiving events without a terminal event. |
| `max_calls` | The maximum number of delegations to this sub-agent per parent run. Once reached, further delegations return a soft budget-exhausted message without running the child. Counts are scoped to one `Agent.run` (a `run_id`) and cleared when it ends, so each parent run and each level of a nested tree budgets independently. |
| `on_failure` | A steering message returned to the parent for any soft degradation of this delegate, in place of the built-in default. Setting it also makes child failures soft (see below). |

## Failure handling

A *soft outcome* returns a steering message to the parent as a normal tool result, so its model reads the message and decides what to do next (rather than immediately re-delegating, which a `ModelRetry` invites). A timeout, a reached `usage_limits` budget, and an exhausted `max_calls` budget are always soft. When `on_failure` is set, the message it carries replaces the built-in default for these outcomes.

A sub-agent run that fails with a *soft model error* (`ModelRetry`, `UnexpectedModelBehavior`, e.g. it exhausted its own retries) is, by default, converted into a `ModelRetry` for the parent -- so the parent's model sees `Sub-agent '<name>' failed: ...` and can react. Set `on_failure` for that delegate to make its failures soft instead: the child error returns the `on_failure` message as a normal tool result.

Hard errors propagate to stop the whole run. A `UsageLimitExceeded` from a child that has *no* per-delegate `usage_limits` (so it shares the parent's accounting) means the whole tree is out of budget and propagates; a child reaching its *own* `usage_limits` is soft, as above.

## Discovery

The sub-agents are listed in the system prompt via `get_instructions`, using each agent's `description` (or a `SubAgent(description=...)` override). A sub-agent with no description is listed by name alone.

## Loading sub-agents from disk

A repo's markdown agent definitions become delegates without writing any `Agent` code. By default every `*.md` file under the conventional folders is loaded as a sub-agent, alongside the explicitly-passed `agents`.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.subagents import SubAgents

orchestrator = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[SubAgents(inherit_tools=True)],  # auto-loads ./.agents/agents/ and ~/.agents/agents/
)
```

`agent_folders` controls where definitions come from. It defaults to `'agents'`, the conventional layout:

- A folder-name `str` (the default `'agents'`): for the project root (cwd) then the home root, load from `<root>/.agents/<name>/`, falling back to `<root>/.claude/<name>/` when `<root>/.agents/` is absent.
- A sequence of paths loads from exactly those folders, in order.
- `None` disables disk loading, exposing only the explicitly-passed `agents`.

### Definition format

A definition is a markdown file with optional frontmatter:

```markdown
---
name: researcher
description: Researches a topic and reports findings
tools: Read, Grep
---
You research topics. Report your findings, each with a source.
```

- `name` is the delegate name (how the parent refers to it and how it is listed). It falls back to the filename stem when absent.
- `description` drives the prompt listing.
- The markdown body becomes the agent's instructions.
- `tools` (or `allowed-tools`) is a comma-separated string or a YAML block list. See "Tools" below.
- `model` and `color` are ignored: the model is inherited from the parent (see below), and `color` has no pyai equivalent.

Frontmatter is read by a small, dependency-free parser limited to those keys (`pyyaml` is not a harness dependency). Full YAML frontmatter is not supported.

### Models and effort

Disk agents inherit the parent run's model by default. Per agent, the caller can override the model and set a thinking/effort level via `agent_overrides`, keyed by the agent's name:

```python
from pydantic_ai_harness.experimental.subagents import AgentOverride, SubAgents

SubAgents(
    agent_folders='agents',
    agent_overrides={'researcher': AgentOverride(model='anthropic:claude-sonnet-4-6', effort='high')},
)
```

Every agent the capability builds runs at a minimum thinking-effort floor. `MINIMUM_EFFORT_FLOOR` and the `clamp_effort(level, floor=...)` helper are exported so an orchestrator can apply the same floor to its own agents (that orchestrator-side application is the caller's responsibility). `clamp_effort` maps `None`/`False` to the floor, leaves `True` (provider-default effort) unchanged, and raises a concrete level below the floor up to it. Effort is applied through pyai's `ModelSettings.thinking`.

### Tools

A disk agent gets no tools by default (`inherit_tools` is `False`); set `inherit_tools=True` to expose the parent's tools to it through the `inherit_tools` mechanism, in which case its `tools` frontmatter is ignored. To map the frontmatter tool names to specific toolsets instead, pass a `tool_resolver`: it receives each tool name (so it can honor entries like `Bash(git:*)`) and returns the toolsets that provide it, or `None` for an unknown name, which is skipped with a warning.

```python
def resolve(tool_name: str):
    return TOOLSETS.get(tool_name)  # -> Sequence[AgentToolset[object]] | None

SubAgents(agent_folders='agents', tool_resolver=resolve)
```

### Precedence

When the same name appears in more than one source, the higher-precedence one wins and the others are skipped with a warning: explicitly-passed `agents` first, then the project folder, then the home folder (and, for an explicit path sequence, earlier paths before later ones). A duplicate name within the explicitly-passed `agents` list is still an error.

## Configuration

```python
SubAgents(
    agents=(),             # Sequence[SubAgent[AgentDepsT]] -- each pairs an agent with its run controls
    agent_folders='agents',# folder-name str (convention) | Sequence[Path] | None (disable)
    agent_overrides={},    # Mapping[str, AgentOverride] -- per-disk-agent model/effort override
    tool_resolver=None,    # Callable[[str], Sequence[AgentToolset[object]] | None] -- disk-agent tool mapping
    forward_usage=True,    # share the parent's usage with sub-agent runs
    inherit_tools=False,   # expose the parent's own tools to sub-agents (capability tools excluded)
    shared_capabilities=(),# capabilities applied to every sub-agent run
    event_stream_handler=None,  # forwarded to each sub-agent run to stream its events
    tool_name='delegate_task',
)
```

```python
SubAgent(
    agent,                 # AbstractAgent[AgentDepsT, Any] -- the child agent to run
    name=None,             # delegate name; defaults to the agent's own `name`
    description=None,      # prompt-listing description; defaults to the agent's own `description`
    usage_limits=None,     # per-delegation request/token budget (isolated accounting)
    timeout_seconds=None,  # per-delegation wall-clock budget
    max_calls=None,        # max delegations to this sub-agent per parent run
    on_failure=None,       # steering message for soft degradations of this delegate
)
```

`SubAgents` is not serializable via the agent spec (it holds live `Agent` instances), so `get_serialization_name()` returns `None`.

## Notes

- Sub-agents can themselves have `SubAgents`, forming a tree. Share `usage` (the default) and set a `usage_limits` on the top-level run to bound the whole tree.
- Delegations the model issues in parallel run as independent sub-agent runs.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Multi-agent applications](https://ai.pydantic.dev/multi-agent-applications/)
