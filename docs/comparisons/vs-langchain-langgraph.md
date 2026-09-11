# Pydantic AI vs LangChain & LangGraph

A LangChain **agent** is a graph. In 1.x, `create_agent()` returns a `CompiledStateGraph`, the
`langchain` package requires `langgraph>=1.2.11`, and `AgentExecutor` is gone. Chat models, prompts,
and retrievers can stay graph-free. Agents cannot.

Pydantic AI is a typed Python value. Pause, durability, and tests attach to that value. They don't
turn it into a graph.

LangSmith is a hosted tracing product. We don't ship one.

## Pause without paying twice

LangChain's `HumanInTheLoopMiddleware` (Deep Agents: `interrupt_on`) pauses before a named tool.
Resume replays nothing: the middleware is its own graph node. If that's the only pause you need, pick
on other grounds.

Any other pause means `interrupt()` inside your own node. Resume restarts that node from its first
line. We ran a refund node that calls the model, writes an audit row, then interrupts:

| Pausing with | The model call | A side effect above the pause |
|---|---|---|
| `HumanInTheLoopMiddleware` / `interrupt_on` | runs once | runs once |
| `interrupt()` inside the node doing the work | **runs twice** | **runs twice** |
| `interrupt_after` on a node you split out first | runs once | runs once |
| Pydantic AI, `requires_approval=True` | runs once | runs once |

Row two is a second model bill and a second audit row, or a second card charge. Row three is the
documented fix: split the graph *before* you know where you'll want to stop.

Pydantic AI pauses at the tool. Mark it `requires_approval=True`, include
[`DeferredToolRequests`][pydantic_ai.DeferredToolRequests] in `output_type`, and the run ends holding
the pending call. Resume is a second `run` from that boundary. The lookup runs once. The payout runs
once.

That's the node rule. Time travel and forking are the upside of it. Replay is the price.

**Deep Agents** is the same graph one layer up: `create_deep_agent()` returns a `CompiledStateGraph`.
Filesystem, subagents, and skills inherit `interrupt()`.
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness) is capabilities on the same
agent object, so the pause above still applies. Their hosted sandbox works the afternoon you install
it; ours is a Modal account you bring. Deep Agents also pins `langchain-anthropic` and
`langchain-google-genai` as hard dependencies.

## Side by side

| | LangChain & LangGraph (1.4.0 / 1.2.11) | Pydantic AI (2.42) |
|---|---|---|
| An agent is | A `CompiledStateGraph` | A typed value; the run is a coroutine |
| Pause | `interrupt()` replays the node, unless you split first or use HITL middleware | `requires_approval=True` at the tool |
| Crash recovery | Checkpointers, a graph feature | Six engines wrap the agent: Temporal, DBOS, Prefect in-tree; Restate, Kitaru, Airflow outside |
| Trusted state | `context_schema` via `runtime.context`; `state_schema` is graph state | `deps_type` plus `RunContext`: a typed dependency API |
| Cancel | `abort()` on the experimental v3 stream; nothing on `invoke()` / `stream()` | `CancellationToken` in-process; `RunCancelled` with history. Not through Temporal/DBOS/Prefect (`UserError`) |
| Test an agent offline | Every fake chat model in `langchain-core` raises `NotImplementedError` on `bind_tools`; a short `BaseChatModel` subclass works | `TestModel` / `FunctionModel`; `ALLOW_MODEL_REQUESTS = False` |
| Tracing | LangSmith; OpenInference spans carry zero `gen_ai.*` | OpenTelemetry GenAI conventions when instrumentation is enabled |
| Integrations | Far larger catalogue | Smaller; call theirs from a tool if you need one |

## FAQ

**Can I use both?**
Yes. Pydantic AI for the agent, LangChain for a connector that exists only there.

**Is it a drop-in replacement?**
No. Tools and prompts carry over. The graph does not. The state dict splits into deps, history, and
whatever your app already uses for workflow state.

**Time travel?**
Not as a product. History is a list you own, so a fork is a slice. There is no checkpoint browser.

---

*Measured 2026-09-10 against langchain 1.4.0, langgraph 1.2.11, langchain-core 1.6.2, deepagents 0.7.13, and
Pydantic AI 2.42. The interrupt traces were re-run offline on 2026-09-11 against those pins. `abort()` lives
on `GraphRunStream` from `stream_events(version='v3')`; both are `@beta`. We recheck this page's version pins
and behaviour claims each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
