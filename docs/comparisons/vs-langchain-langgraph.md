# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph: `create_agent()` returns a `CompiledStateGraph`. Pydantic AI is a
typed value. Pause, durability, and tests attach to it; they don't redraw it as nodes.

LangSmith is hosted tracing. We emit OpenTelemetry you already run.

## Side by side

| | LangChain & LangGraph | Pydantic AI |
|---|---|---|
| An agent is | A `CompiledStateGraph` | A typed `Agent` |
| Pause | `interrupt()` replays the node, unless you split first or use HITL | `requires_approval=True` at the tool |
| Crash recovery | Checkpointers | The same agent, inside Temporal, DBOS, or Prefect |
| Trusted state | `context_schema` / `state_schema` | A typed object your tools read; the model never sees it |
| Cancel | `abort()` on the experimental v3 stream only | In-process stop; not through Temporal / DBOS / Prefect |
| Test offline | Fake chat models raise on `bind_tools` | A fake model you script; no API key |
| Tracing | LangSmith | OpenTelemetry GenAI names, when you turn them on |
| Integrations | Far larger catalogue | Smaller; call theirs from a tool |

## Pause at the tool

`interrupt()` inside a LangGraph node restarts that node from line one: the model call and anything
above the pause run again. `HumanInTheLoopMiddleware` (Deep Agents: `interrupt_on`) pauses before a
named tool and does not replay. `interrupt_after` on a node you split out first also doesn't. The
documented fix is to split the graph before you know where you'll want to stop.

Ours pauses at the tool. `requires_approval=True` plus
[`DeferredToolRequests`][pydantic_ai.DeferredToolRequests] in `output_type` ends the run holding the
pending call. Resume is a second `run`. Lookup once, payout once.

Deep Agents is the same graph: `create_deep_agent()` returns a `CompiledStateGraph`.

## FAQ

**Do I have to redraw the agent as a graph to pause?** No. `requires_approval=True` on the tool. The
run ends holding the pending call. Resume is a second `run`.

**Can I get a coding agent without Deep Agents?** Yes.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) on the same typed [`Agent`][pydantic_ai.Agent].
A LangChain connector you already have can be a tool on that agent.
