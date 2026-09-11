# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph: `create_agent()` returns a `CompiledStateGraph`. Pydantic AI is a
typed value. Pause, durability, and tests attach to it; they don't redraw it as nodes.

LangSmith is a hosted tracing product. We don't ship one.

## Side by side

| | LangChain & LangGraph 1.4.0 / 1.2.11 | Pydantic AI 2.42 |
|---|---|---|
| An agent is | A `CompiledStateGraph` | A typed `Agent` |
| Pause | `interrupt()` replays the node, unless you split first or use HITL | `requires_approval=True` at the tool |
| Crash recovery | Checkpointers | Six engines wrap the agent |
| Trusted state | `context_schema` / `state_schema` | `deps_type` plus `RunContext` |
| Cancel | `abort()` on the experimental v3 stream only | `CancellationToken` in-process; not through Temporal/DBOS/Prefect |
| Test offline | Fake chat models raise on `bind_tools` | `TestModel` / `FunctionModel` |
| Tracing | LangSmith | OpenTelemetry GenAI conventions, when enabled |
| Integrations | Far larger catalogue | Smaller; call theirs from a tool |

## Pause without paying twice

`HumanInTheLoopMiddleware` (Deep Agents: `interrupt_on`) pauses before a named tool and does not
replay. Any other pause is `interrupt()` inside your node, and resume restarts that node from line
one. We ran a refund node that calls the model, writes an audit row, then interrupts:

| Pausing with | The model call | A side effect above the pause |
|---|---|---|
| `HumanInTheLoopMiddleware` / `interrupt_on` | runs once | runs once |
| `interrupt()` inside the node doing the work | **runs twice** | **runs twice** |
| `interrupt_after` on a node you split out first | runs once | runs once |
| Pydantic AI, `requires_approval=True` | runs once | runs once |

Row two is a second model bill. Row three is the documented fix: split the graph before you know
where you'll want to stop.

Ours pauses at the tool. `requires_approval=True` plus
[`DeferredToolRequests`][pydantic_ai.DeferredToolRequests] in `output_type` ends the run holding the
pending call. Resume is a second `run`. Lookup once, payout once.

Deep Agents is the same graph: `create_deep_agent()` returns a `CompiledStateGraph`, so it inherits
the replay. [pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness) is capabilities on
the same agent. Their sandbox backend is [Daytona](https://www.daytona.io/) or one you plug in; ours
is a Modal account you bring.

## FAQ

**Can I use both?** Yes. Us for the agent, LangChain for a connector that exists only there.

**Drop-in?** No. Tools and prompts carry. The graph does not.

---

*langchain 1.4.0 / langgraph 1.2.11 / langchain-core 1.6.2 / deepagents 0.7.13, installed.
`create_agent` and `create_deep_agent` both annotate `CompiledStateGraph`.
`FakeListChatModel.bind_tools([])` and `FakeMessagesListChatModel.bind_tools([])` raise
`NotImplementedError`. `abort()` is on `@beta` `GraphRunStream`. Interrupt traces re-run
2026-09-11. Pydantic AI 2.42.
[Tell us](https://github.com/pydantic/pydantic-ai/issues/new) if a pin goes stale.*
