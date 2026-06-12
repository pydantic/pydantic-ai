# SubAgents

> [!WARNING]
> **Experimental.** This capability lives under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import it from the
> experimental path -- there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.subagents import SubAgents
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

`SubAgents` takes a name-to-agent mapping and exposes a single `delegate_task(agent_name, task)` tool. Each delegation runs the chosen sub-agent in its own run -- with its own message history, so it never sees the parent conversation -- and returns its output to the parent. The available sub-agents are listed in the system prompt as a static instruction, so the listing stays in the cached prefix.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.subagents import SubAgents

researcher = Agent('anthropic:claude-sonnet-4-6', name='researcher', description='Researches a topic and reports findings')
writer = Agent('anthropic:claude-sonnet-4-6', name='writer', description='Turns notes into polished prose')

orchestrator = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[SubAgents(agents={'researcher': researcher, 'writer': writer})],
)

result = orchestrator.run_sync('Research the history of TLS and write a one-paragraph summary.')
print(result.output)
```

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
- **Tools can be inherited.** With `inherit_tools=True`, the parent agent's tools are added to each sub-agent run (on top of the sub-agent's own). The delegate tool itself is filtered out, so a sub-agent can't recurse into further delegation. Off by default.
- **Capabilities can be shared.** `shared_capabilities` are applied to every sub-agent run -- e.g. give all sub-agents a common guardrail, memory, or planning capability without rebuilding each `Agent`.
- **Sub-agent events can be streamed.** Pass an `event_stream_handler` and it's forwarded to each sub-agent run, so the sub-agent's model-streaming and tool events surface to the caller (the handler receives the sub-agent's own `RunContext`).

## Per-delegate run controls

`limits` maps a sub-agent name to a `SubAgentLimits`, giving one delegate its own budgets without touching the others. A name absent from `limits` runs with the `SubAgents` defaults.

```python
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.experimental.subagents import SubAgentLimits, SubAgents

# reproducer and librarian are Agent instances, as in the example above.
orchestrator = Agent(
    'anthropic:claude-opus-4-7',
    capabilities=[
        SubAgents(
            agents={'reproducer': reproducer, 'librarian': librarian},
            limits={
                'reproducer': SubAgentLimits(usage_limits=UsageLimits(request_limit=35), timeout_seconds=600, max_calls=1),
                'librarian': SubAgentLimits(usage_limits=UsageLimits(request_limit=18), timeout_seconds=300, max_calls=2),
            },
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

The sub-agents are listed in the system prompt via `get_instructions`, using each agent's `description` (or a per-name `descriptions` override). A sub-agent with no description is listed by name alone.

## Configuration

```python
SubAgents(
    agents={},             # Mapping[str, AbstractAgent[AgentDepsT, Any]] -- name -> agent
    descriptions=None,     # optional per-name description overrides for the prompt listing
    forward_usage=True,    # share the parent's usage with sub-agent runs
    inherit_tools=False,   # expose the parent's tools to sub-agents (delegate tool excluded)
    shared_capabilities=(),# capabilities applied to every sub-agent run
    event_stream_handler=None,  # forwarded to each sub-agent run to stream its events
    tool_name='delegate_task',
    limits={},             # Mapping[str, SubAgentLimits] -- per-delegate run controls
)
```

`SubAgents` is not serializable via the agent spec (it holds live `Agent` instances), so `get_serialization_name()` returns `None`.

## Notes

- Sub-agents can themselves have `SubAgents`, forming a tree. Share `usage` (the default) and set a `usage_limits` on the top-level run to bound the whole tree.
- Delegations the model issues in parallel run as independent sub-agent runs.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Multi-agent applications](https://ai.pydantic.dev/multi-agent-applications/)
