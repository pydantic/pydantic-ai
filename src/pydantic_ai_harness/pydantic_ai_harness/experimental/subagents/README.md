# SubAgents

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

## Failure handling

If a sub-agent run fails with a *soft* model error (`ModelRetry`, `UnexpectedModelBehavior`, e.g. it exhausted its own retries), the failure is converted into a `ModelRetry` for the parent -- so the parent's model sees `Sub-agent '<name>' failed: …` and can react. Hard errors (e.g. `UsageLimitExceeded`) propagate to stop the whole run.

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
)
```

`SubAgents` is not serializable via the agent spec (it holds live `Agent` instances), so `get_serialization_name()` returns `None`.

## Notes

- Sub-agents can themselves have `SubAgents`, forming a tree. Share `usage` (the default) and set a `usage_limits` on the top-level run to bound the whole tree.
- Delegations the model issues in parallel run as independent sub-agent runs.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Multi-agent applications](https://ai.pydantic.dev/multi-agent-applications/)
