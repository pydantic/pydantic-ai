---
title: Subagents
description: "Let a Pydantic AI agent delegate tasks to named subagents through one delegate_task tool, with per-subagent budgets, model choice, and Markdown agent files."
---

# Subagents

`SubAgents` lets an agent delegate self-contained tasks to named child agents. It takes a sequence of `SubAgent` entries and exposes a single `delegate_task(agent_name, task)` tool. Each delegation runs the chosen sub-agent in its own run -- with its own message history, so it never sees the parent conversation -- and returns its output to the parent.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/subagents/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

A single agent that does everything accumulates a large tool set and a long context. Splitting the work across specialized sub-agents keeps each context focused, but wiring up delegation by hand means writing a tool per agent, forwarding deps, threading usage limits, and telling the model what it can delegate to.

## The solution

`SubAgents` takes a sequence of `SubAgent` entries and exposes a single `delegate_task(agent_name, task)` tool. Each delegation runs the chosen sub-agent in its own run -- with its own message history, so it never sees the parent conversation -- and returns its output to the parent. The available sub-agents are listed in the system prompt as a static instruction, so the listing stays in the cached prefix.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent, SubAgents

researcher = Agent('anthropic:claude-opus-5-5', name='researcher', description='Researches a topic and reports findings')
writer = Agent('anthropic:claude-opus-5-5', name='writer', description='Turns notes into polished prose')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
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
- With a `models` menu configured, the tool takes an extra `model` argument (see below).

## Deps, usage, tools, and capabilities

- **Deps are forwarded.** The parent run's `deps` are passed to each sub-agent, so sub-agents share the parent's `AgentDepsT` (enforced by the type signature -- every sub-agent is an `AbstractAgent[AgentDepsT, Any]`).
- Delegated agents run in the parent's workspace, including wrappers such as `ReadOnlyWorkspace`.
- **Usage is shared by default.** The parent's `usage` is passed to each sub-agent run, so token usage aggregates and a parent `usage_limits` applies across the whole agent tree. Set `forward_usage=False` to give each sub-agent run its own accounting.
- **The parent's tools are not passed on.** A sub-agent runs with its own tools. To delegate to an agent that has all of the parent's tools and capabilities, use `include_self=True` (see "Delegating to the agent itself" below).
- **Capabilities can be shared.** `shared_capabilities` are applied to every sub-agent run -- e.g. give all sub-agents a common guardrail, memory, or planning capability without rebuilding each `Agent`.
- **Sub-agent events can be streamed.** Pass an `event_stream_handler` and it's forwarded to each sub-agent run, so the sub-agent's model-streaming and tool events surface to the caller (the handler receives the sub-agent's own `RunContext`).

`inherit_tools=True` is deprecated and emits a `HarnessDeprecationWarning`. It adds the parent agent's own tools (registered via `tools=` or `toolsets=`) to each sub-agent run, minus the delegate tool, but not the tools contributed by the parent's capabilities. Bind the tools an explicit sub-agent needs directly to its `Agent`, use `include_self=True` to delegate to a fresh run of the agent with everything bound to it, or use a `tool_resolver` to give disk-loaded agents tools.

## Delegating to the agent itself

With `include_self=True`, the roster also lists the running agent itself, as `self`. A delegation to `self` starts a fresh run of that agent (`RunContext.agent`) on the parent run's model (or the `models` option the parent picks), with no parent conversation. Because the child is the same `Agent`, it has every capability, toolset, and instruction bound to it, and they register again in the child run: a guardrail, approval gate, or audit hook bound next to `SubAgents` sees the tool calls the delegate makes, not only the `delegate_task` call. This is what [`Coder`](coder.md) uses by default.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgents

agent = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[SubAgents(include_self=True)],
)
```

- **Only what is bound to the `Agent` carries over.** Capabilities, toolsets, instructions, and model settings passed to the parent's `run()` are not part of the agent, so the delegate does not get them. Passing `SubAgents(include_self=True)` itself to `run()` raises a `UserError` when the run starts, since the delegate would come up without it.
- **Delegation depth is capped.** The delegate carries `delegate_task` too, so `max_depth` (default `3`, counting the top-level run) bounds the tree: the top-level run delegates, its delegates delegate once more, and a run at the limit gets neither `delegate_task` nor the sub-agent listing. The limit applies to every delegation through `SubAgents`, including explicit rosters.
- **Delegates inherit everything, including what may not suit a sub-task.** An `AskUser` capability bound to the agent can prompt the user from inside a delegation, and the delegate returns the agent's own `output_type`, rendered with `str()`.
- The deprecated `inherit_tools` does not apply to `self`, whose tools are already the parent's. The name `self` is reserved: an explicit delegate with that name is an error, and a disk definition with that name is skipped with a warning.

## Per-delegate run controls

Each `SubAgent` carries its own budgets, so one delegate's controls do not touch the others. A `SubAgent` with no controls set runs with the `SubAgents` defaults.

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import SubAgent, SubAgents

reproducer = Agent('anthropic:claude-opus-5-5', instructions='Reproduce the reported bug from a minimal script.')
librarian = Agent('anthropic:claude-opus-5-5', instructions='Find relevant docs, issues, and prior art.')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
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
| `models` | Which keys of the `SubAgents` model menu this delegate may run on, and which one it runs on by default: the first key listed. See "Per-delegation model selection" below. |
| `usage_limits` | A request/token budget for one delegation. The child runs with its own usage accounting, so the budget counts only that child's requests and tokens (not the parent's or siblings'). With `forward_usage=True`, the child's usage is added to the parent's usage after the delegation. Reaching the budget is a soft outcome (see below), not a run-stopping `UsageLimitExceeded`. |
| `timeout_seconds` | A wall-clock budget for one delegation. When the child exceeds it, its run is cancelled and the parent gets a soft steering message instead of hanging on the child. The cancelled child's `event_stream_handler` (if any) stops receiving events without a terminal event. |
| `max_calls` | The maximum number of delegations to this sub-agent per parent run. Once reached, further delegations return a soft budget-exhausted message without running the child. Counts are scoped to one `Agent.run` (a `run_id`) and cleared when it ends, so each parent run and each level of a nested tree budgets independently. |
| `on_failure` | A steering message returned to the parent for any soft degradation of this delegate, in place of the built-in default. Setting it also makes child failures soft (see below). |
| `contain_errors` | Whether an unexpected crash in this delegate is caught and returned to the parent as a bounded `ModelRetry` instead of aborting the parent run (see below). Unset inherits the `SubAgents(contain_errors=...)` default (off). |

## Per-delegation model selection

The orchestrator knows how hard a task is at the moment it writes the brief, so it is the right place to decide which model runs it. Configure a `models` menu and the delegate tool gains a `model` argument that names one of its keys.

```python
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import SubAgent, SubAgents
from pydantic_ai_harness.subagents import ModelOption

reviewer = Agent('anthropic:claude-opus-5-5', name='reviewer', description='Reviews a diff')
linter = Agent('anthropic:claude-opus-5-5', name='linter', description='Runs the linter and reports failures')

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        SubAgents(
            agents=[SubAgent(reviewer), SubAgent(linter, models=['fast'])],
            models={
                'fast': 'anthropic:claude-haiku-4-5',
                'standard': 'anthropic:claude-sonnet-5',
                'deep': ModelOption(
                    'anthropic:claude-opus-5-5',
                    description='hard reasoning, multi-file changes',
                    settings=ModelSettings(thinking='xhigh'),
                ),
            },
        )
    ],
)
```

- **Off by default.** With no `models` menu the `model` argument is not in the tool schema at all, and every delegation runs exactly as it did before.
- **The keys are the interface.** They are listed in the system prompt with each entry's model and description, and the tool schema offers them as an enum, so the model picks from the menu instead of inventing a model name. Name them for the job (`'fast'`, `'deep'`), not for the vendor.
- **An entry is a model or a `ModelOption`.** `ModelOption(model, description=..., settings=...)` adds a routing hint and per-option `ModelSettings`, so one key can mean "same model, more thinking". Those settings merge over the sub-agent's own `model_settings`, which keeps the parts the option does not set.
- **Resolution order.** The key the parent passed, else the delegate's first allowed key (see below), else the delegate's own model, else the parent run's model.
- **A delegate can be restricted.** `SubAgent(linter, models=['fast'])` pins that delegate to `fast`: the first listed key is what it runs on when the parent passes no `model`, and the others are refused with a `ModelRetry`. The restriction is rendered in the prompt listing (`- linter: Runs the linter (models: fast)`). Restricting to a key the menu does not define is a `ValueError` at construction.
- **A rejected key costs nothing.** An unknown or unavailable key comes back as a `ModelRetry` listing the valid options, before the delegate's `max_calls` budget is charged.

## Failure handling

A *soft outcome* returns a steering message to the parent as a normal tool result, so its model reads the message and decides what to do next (rather than immediately re-delegating, which a `ModelRetry` invites). A timeout, a reached `usage_limits` budget, and an exhausted `max_calls` budget are always soft. When `on_failure` is set, the message it carries replaces the built-in default for these outcomes.

A sub-agent run that fails with a *soft model error* (`ModelRetry`, `UnexpectedModelBehavior`, e.g. it exhausted its own retries) is, by default, converted into a `ModelRetry` for the parent -- so the parent's model sees `Sub-agent '<name>' failed: ...` and can react by re-delegating. The delegate tool defaults to `tool_retries=2`, so the parent aborts only after that many consecutive delegate failures; the counter resets after any successful delegation. Raise `tool_retries` to tolerate a flakier sub-agent, or set `None` to inherit the parent agent's default tool retries. Set `on_failure` for a delegate to make its failures soft instead: the child error returns the `on_failure` message as a normal tool result.

Hard errors propagate to stop the whole run. A `UsageLimitExceeded` from a child that has *no* per-delegate `usage_limits` (so it shares the parent's accounting) means the whole tree is out of budget and propagates; a child reaching its *own* `usage_limits` is soft, as above.

An *unexpected crash* -- any other exception the child raises, such as a provider `ModelAPIError`/`FallbackExceptionGroup` or a plain `ValueError` from a bad tool argument -- propagates by default and aborts the parent run. Set `contain_errors=True` (per delegate, or as the `SubAgents` default) to catch it and return it to the parent as a bounded `ModelRetry` instead, so one delegate crash cannot kill the whole run. Containment stays loud: the exception rides the retry message (`Sub-agent '<name>' crashed: ...`), it is logged via the standard `logging` module, and `tool_retries` still bounds consecutive crashes into an abort. This is orthogonal to `on_failure` -- a contained crash always raises the loud retry, never the soft `on_failure` return, so a genuine bug is never masked as success. Cancellation, a shared `UsageLimitExceeded`, pydantic-ai control-flow signals (`CallDeferred`, `ApprovalRequired`, the `Skip*` signals), and `UserError` bypass containment regardless of `contain_errors`. Cancellation covers both kinds: external cancellation (`asyncio.CancelledError`) propagates as-is, and a child's own first-party cancellation (`RunContext.cancel()` inside the child, raising `RunCancelled`) leaves the delegate tool uncontained, after which pydantic-ai isolates it as a failed `delegate_task` return the parent model can react to, not a crash retry that invites re-delegation.

## Events

`SubAgents` emits typed capability events in the `sub_agents` namespace so a host can show a delegation as it runs, and how it ended, without parsing the delegate tool's arguments and result:

| Event | Dispatch | When | Payload |
|---|---|---|---|
| `DelegationStartEvent` | stream | a delegation passed every check and the child run is about to start | `agent_name`, `task`, `truncated`, `model` (the menu key, or `None`), `inherits_tools` |
| `DelegationEndEvent` | stream | the delegation settled into what the parent receives | `agent_name`, `outcome`, `output`, `truncated`, `usage`, `duration_seconds` |

Both are notifications. One delegation is one `delegate_task` call, so a start and its end share the `tool_call_id` core stamps on every event; that is how a subscriber pairs them when the model delegates in parallel.

`outcome` follows the failure handling above: `ok` (the child's output went back to the parent), `timeout`, `budget` (the child's own `usage_limits`), `failed` (a soft model error, returned as `on_failure` or raised as a `ModelRetry`), or `contained` (a crash `contain_errors` caught). `output` is what the delegate tool hands back to the parent in each case: the child's output, the steering message, or the retry text. It is emitted from inside the tool, before any `ToolGuardrail` result guard screens that text for the model; a host that needs the screened version reads the `ToolReturnPart` in core's `FunctionToolResultEvent`. `usage` is the child's own `RunUsage` when it has separate accounting (`usage_limits` set, or `forward_usage=False`) and `None` when it accrues into the parent's usage, where its share is not separable.

A delegation refused before the child runs (an unknown sub-agent, a model key off the menu, an exhausted `max_calls` budget) emits nothing; the tool result says why. An exception that propagates out of the delegate tool (a shared usage limit, an uncontained crash, a cancellation) ends without an end event. A `SubAgentToolset` registered directly in `Agent(toolsets=[...])` has no owning capability and emits nothing. As with any capability event, a listener that raises aborts the parent run.

`task` and `output` are cut at `MAX_EVENT_TEXT_CHARS` (4096) with a `truncated` flag, so a persisted or forwarded event stream cannot be flooded by one verbose delegation.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai_harness.subagents import DelegationEndEvent, SubAgent, SubAgents


class ReportDelegations(AbstractCapability):
    @on_event(DelegationEndEvent)
    async def on_delegation_end(self, ctx, event: DelegationEndEvent) -> None:
        print(f'{event.agent_name}: {event.outcome} in {event.duration_seconds:.1f}s')


researcher = Agent('anthropic:claude-opus-5-5', name='researcher')
agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SubAgents(agents=[SubAgent(researcher)]), ReportDelegations()],
)
```

Nested model streaming from the child run is not an event concern; pass an `event_stream_handler` for that.

See [capability events](../capabilities/overview.md#capability-events) for how `@on_event` works.

`SubAgents` emits no OpenTelemetry spans of its own: the child run is a core agent run with its own spans nested under the parent's tool-call span, and the events above carry the outcome a trace would only show as an exception or a tool result.

## Discovery

The sub-agents are listed in the system prompt via `get_instructions`, using each agent's `description` (or a `SubAgent(description=...)` override). A sub-agent with no description is listed by name alone.

## Loading sub-agents from disk

A repo's agent definitions can become delegates without writing any `Agent` code. With `agent_folders` set, every Claude-style `*.md` file and Codex-style `*.toml` file under those folders is loaded as a sub-agent, alongside the explicitly-passed `agents`. Files are read in sorted filename order.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgents

orchestrator = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[SubAgents(agent_folders='agents')],  # loads .agents/agents/ from the run's workspace
)
```

Definitions are read at the start of every run from the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/), so a sandbox's agent files are found and nothing is read from your home directory. To read them from somewhere else, such as definitions that ship with your application, pass `workspace=LocalWorkspaceBackend('/app')`.

`agent_folders` controls which folders are read:

- A folder-name `str` (`'agents'` is the conventional layout): load from `.agents/<name>/`, `.claude/<name>/`, and `.codex/<name>/` under the workspace's working directory, in that order, so a workspace that uses `.agents/` for something else (such as skills) still loads agents from the others. A run without a workspace skips them.
- A sequence of workspace paths, absolute or relative to the working directory, loads from exactly those folders, in order.
- `None`, the default, disables disk loading, exposing only the explicitly-passed `agents`.

Earlier releases loaded the conventional folders by default. Pass `agent_folders='agents'` to keep doing so.

Until this release, the folders were read from this machine, including the home folder `~/.agents/agents/`. A run without a workspace now fails at its start when given a path sequence. Convention discovery warns once when it skips a folder in the current or home directory that the workspace does not reach, naming the folder and the `workspace=` that reads it.

### Definition format

A definition is a Claude-style markdown file with optional frontmatter, or a Codex-style TOML file. Markdown:

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

A Codex-style standalone TOML file:

```toml
name = "reviewer"
description = "Reviews code for bugs"
developer_instructions = "Inspect the code and report findings. Do not edit files."
tools = ["Read", "Grep"]
```

- `name`, `description`, and `developer_instructions` are required nonempty strings.
- `tools` (or `allowed-tools`) is optional: a list of strings or a comma-separated string, not both keys.
- `model`, `effort`, `model_reasoning_effort`, and `color` are ignored with a warning; use `agent_overrides` for models and effort.
- Any other key, including sandbox or permission settings, skips that file with a warning rather than silently granting broader tools. The older `[agents.<name>] config_file` layout is not supported.
- TOML is parsed with the standard library `tomllib`, so it needs Python 3.11 or newer; on 3.10 TOML files are skipped with a warning.

Nothing in a definition file is executed. A malformed or invalid file is skipped with a warning without blocking the others.

### Models and effort

Disk agents inherit the parent run's model by default. Per agent, the caller can override the model and set a thinking/effort level via `agent_overrides`, keyed by the agent's name:

```python
from pydantic_ai_harness import SubAgents
from pydantic_ai_harness.subagents import AgentOverride

SubAgents(
    agent_folders='agents',
    agent_overrides={'researcher': AgentOverride(model='anthropic:claude-opus-5-5', effort='high')},
)
```

When `effort` is unset, the disk agent adds no thinking setting, so the inherited model's defaults apply. An explicit value, including `False` or `'minimal'`, is passed through unchanged via Pydantic AI's `ModelSettings.thinking`.

`MINIMUM_EFFORT_FLOOR` and `clamp_effort(level, floor=...)` remain importable for compatibility but are deprecated. `SubAgents` no longer uses them. Pass `AgentOverride(effort=...)` when a disk agent needs an explicit level, or apply an application-specific floor outside the capability.

### Tools

Without a `tool_resolver`, a disk agent gets no tools and its `tools` frontmatter is ignored. To map the frontmatter tool names to toolsets, pass a `tool_resolver`: it receives each tool name (so it can honor entries like `Bash(git:*)`) and returns the toolsets that provide it, or `None` for an unknown name, which is skipped with a warning.

```python {names="defined"}
from collections.abc import Sequence

from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness import SubAgents

TOOLSETS: dict[str, Sequence[AgentToolset[object]]] = {}  # your tool name -> toolsets mapping


def resolve(tool_name: str) -> Sequence[AgentToolset[object]] | None:
    return TOOLSETS.get(tool_name)

SubAgents(agent_folders='agents', tool_resolver=resolve)
```

### Precedence

When the same name appears in more than one source, the higher-precedence one wins and the others are skipped with a warning: explicitly-passed `agents` first; for convention discovery, the workspace's `.agents/` folder before its `.claude/` folder; and for an explicit path sequence, earlier folders before later ones. A duplicate name within the explicitly-passed `agents` list is still an error.

## Configuration

```python {names="defined"}
from pydantic_ai_harness import SubAgents

SubAgents(
    agents=(),             # Sequence[SubAgent[AgentDepsT]] -- each pairs an agent with its run controls
    models={},             # Mapping[str, Model | str | ModelOption] -- per-delegation model menu (off when empty)
    agent_folders=None,    # folder-name str ('agents' is conventional) | Sequence[str | Path] workspace paths | None
    agent_overrides={},    # Mapping[str, AgentOverride] -- per-disk-agent model/effort override
    tool_resolver=None,    # Callable[[str], Sequence[AgentToolset[object]] | None] -- disk-agent tool mapping
    forward_usage=True,    # share the parent's usage with sub-agent runs
    inherit_tools=False,   # deprecated: use include_self=True, or tool_resolver for disk agents
    shared_capabilities=(),# capabilities applied to every sub-agent run
    event_stream_handler=None,  # forwarded to each sub-agent run to stream its events
    tool_name='delegate_task',
    tool_retries=2,        # extra delegate-tool attempts after a sub-agent error before aborting (None inherits the agent default)
    contain_errors=False,  # default for SubAgent.contain_errors: contain an unexpected crash as a bounded retry
    workspace=None,        # WorkspaceBackend to read agent_folders from instead of the run's workspace
    include_self=False,    # also list the running agent itself as the delegate `self`
    max_depth=3,           # delegation levels, counting the top-level run
)
```

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness import SubAgent

agent = Agent('anthropic:claude-opus-5-5')

SubAgent(
    agent,                 # AbstractAgent[AgentDepsT, Any] -- the child agent to run
    name=None,             # delegate name; defaults to the agent's own `name`
    description=None,      # prompt-listing description; defaults to the agent's own `description`
    models=None,           # Sequence[str] -- menu keys this delegate may run on; the first is its default
    usage_limits=None,     # per-delegation request/token budget (isolated accounting)
    timeout_seconds=None,  # per-delegation wall-clock budget
    max_calls=None,        # max delegations to this sub-agent per parent run
    on_failure=None,       # steering message for soft degradations of this delegate
    contain_errors=None,   # contain an unexpected crash as a bounded retry; None inherits the SubAgents default
)
```

`SubAgents` is not serializable via the [agent spec](../agent-spec.md) (it holds live `Agent` instances), so `get_serialization_name()` returns `None`.

## Notes

- Sub-agents can themselves have `SubAgents`, forming a tree. Each `SubAgents` stops offering delegation once the run is at its own `max_depth`, so a delegate with a higher limit can go deeper than its parent's. Share `usage` (the default) and set a `usage_limits` on the top-level run to bound the whole tree.
- Delegations the model issues in parallel run as independent sub-agent runs.

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Multi-agent applications](../multi-agent-applications.md)

## API reference

::: pydantic_ai_harness.subagents.SubAgents

::: pydantic_ai_harness.subagents.SubAgent

::: pydantic_ai_harness.subagents.ModelOption

::: pydantic_ai_harness.subagents.AgentOverride

::: pydantic_ai_harness.subagents.DelegationStartEvent

::: pydantic_ai_harness.subagents.DelegationEndEvent

## Managed delegation sessions

`SubAgents` keeps its existing foreground-only behavior unless a caller explicitly
opens and binds `DelegationTasks`. This owner adds `background` and `resume` to the
delegate tool's schema, gives every child a stable conversation ID, and owns every
worker until shutdown. Use `DelegationReports` on the parent run to deliver settled
background reports through core's `SystemPromptPart` queue. The reports are
explicitly automated, untrusted data, not user instructions or permission grants.
No extra agent loop is implemented. Reports default to `priority='when_idle'`
so active parents finish their current work first. A host that starts an idle
continuation should use `DelegationReports(..., priority='asap')` and
`agent.run(None, ...)`: pending reports then enter the first model request, with
no synthetic user prompt. The host owns wake-up scheduling between runs.

```python {test="skip"}
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai_harness.subagents import DelegationReports, DelegationTasks, SubAgents

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[SubAgents(include_self=True)])
tasks = DelegationTasks(directory=Path('.task-history'))

async def converse():
    async with tasks.opened():
        with tasks.bind():
            result = await agent.run(
                'Delegate an independent investigation in the background.',
                conversation_id='review',
                capabilities=[DelegationReports(tasks, conversation_id='review')],
            )
            print(result.output)
            # Keep this owner open across subsequent parent turns.
```

`opened()` drains workers on exit. Keep workspace and plugin resources alive outside
that scope. Detached execution is refused for run-owned non-local workspaces;
foreground delegation still works. `max_depth=4` counts the main run and allows
three child layers. An explicitly configured non-default `SubAgents.max_depth`
still takes precedence. Ordinary `SubAgents` retains its original depth default.

`background(task_id)` releases a foreground waiter without restarting the child.
`await cancel(task_id)` stops and drains that child and its descendants. A user stop
blocks model-requested resume until the application explicitly calls
`await allow_resume(task_id)`. `one_shot` names never resume. A resume uses the same
child ID and its independent history, a new run ID, and the current direct parent.
A child waits for its own descendants and consumes their reports before its final
output settles. Reports are routed to the direct parent; an idle parent receives
pending reports on its next explicitly started run. Enqueue delivery is acknowledged
only when core emits `EnqueuedMessagesEvent`, and acknowledgements are persisted.

An observer receives `DelegationTaskEvent`, with the task identity and an optional
correlated child stream event. Managed start/end events carry `task_id` and
`parent_id`. Managed cancellation and uncontained exceptions produce terminal
outcomes; unmanaged events and exception propagation keep their original contract.
Metadata and final/interrupted histories are atomically saved under `directory`.
Pass `step_store` to checkpoint through `StepPersistence` and recover a process-killed
child's latest frontier. Loading an interrupted record never executes its tools.
Inspect possible partial effects before an explicit resume.

Managed children with `forward_usage=True` share live usage accounting and inherit
parent ceilings. A per-child budget is converted to an absolute ceiling at launch;
concurrent sibling spend may reach that ceiling earlier, but cannot bypass the
parent's budget. The ordinary unmanaged per-child accounting contract is unchanged.

`agents`, `aliases`, and `instructions` extend the roster and guidance only inside
the bound scope; they do not add delegation to an agent without `SubAgents`.
`SubAgent(read_only=True)` wraps the child's workspace in `ReadOnlyWorkspace`.
Also give that agent only trusted read-only capabilities: arbitrary Python tools
can bypass the workspace API. CLAI's Explore and Plan specialists use filesystem
readers and expose no shell, code execution, or parent plugin tools.

Core's agent/model/tool spans provide execution telemetry; task IDs, parent IDs,
and child run IDs provide correlation. This owner adds no logging exporter.
