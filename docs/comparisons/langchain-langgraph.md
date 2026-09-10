# Pydantic AI vs LangChain & LangGraph

**LangChain, at its best:** the largest ecosystem in agent tooling — every integration, every
provider, a huge community, and durable workflows through LangGraph.

**Pydantic AI, at its best:** a typed, async-first Python framework where the agent loop is a plain
value — no graph DSL, no runtime to adopt, capabilities you wire yourself.

*Verified against langchain 1.3.1 / langgraph 1.2.1 (2026-09-10). Pydantic AI claims below run
offline: `uv run -m pydantic_ai_examples.comparisons.graph_is_a_value`.*

## Quick comparison

| What you get | LangChain/LangGraph | Pydantic AI |
|---|---|---|
| Structure without a DSL | Adopt a `StateGraph` to get checkpoints, retries, streaming | Plain async; the loop is a value you drive |
| Pause for a human | `interrupt()` — graph state; resuming **re-runs the node's LLM call** | `ctx.cancel()` / deferred tools — pause at the tool boundary, resume clean |
| Trusted state | `context_schema` / state dict flows through the loop | `deps_type` — model cannot choose or see it |
| Extend the agent | Middleware (LIFO order) intercepts steps | Capabilities: bundle tools + instructions + settings + hooks, orderable, serializable |
| Durability | Checkpointers are a graph feature — you must use LangGraph | Engines wrap the same agent (Temporal/DBOS/Prefect/Restate/Kitaru/Airflow) |
| Cancellation | None: interrupt is state; killing a run = kill the task | Typed: `RunCancelled` with resumable history; external `CancelledError` preserved |
| Events | Super-step graph events | Part/tool/result/final typed events; capabilities can transform the stream |
| Offline tests | `GenericFakeChatModel` can't `bind_tools` (probed) | `TestModel`/`FunctionModel` drive the whole pipeline deterministically |

## Prove it yourself

The loop is a value, not an abstraction to adopt. One run, iterated node by node, offline:

```snippet {path="/examples/pydantic_ai_examples/comparisons/graph_is_a_value.py"}
```

```
nodes in one run: UserPromptNode -> ModelRequestNode -> CallToolsNode -> ModelRequestNode -> CallToolsNode -> End
the loop is a plain value: iterate it, drive it manually, or let a capability transform it
```

That's the structural difference in one line: LangGraph's graph is the abstraction *you must adopt*
to get checkpoints and structure; here the run is already a value, and `agent.iter()` is one way to
drive it — the same value a capability can wrap, a durable engine can wrap, and a spec can describe.

## Key differences

**Their best:** the ecosystem is the offer — integrations, community, fast tinkering. LangGraph's
checkpointed workflows are genuinely durable. Middleware is a real intercept seam.

**Ours:** the seams are typed and the run is yours. Interrupt-vs-cancel is the sharpest example:
`interrupt()` in LangGraph pauses by graph state and **resuming restarts the enclosing node** (the
LLM call re-runs — probed); our pause sits at the tool-call boundary, the run stops or continues
without re-running finished work, and cancellation ends in a catchable `RunCancelled` carrying the
resumable history. Typed deps mean the model can't reach trusted state; `deps_type` also validates
tool signatures, spec templates, tests, and evals. Durability wraps the agent — six engines, none of
them ours, all through the public interface.

## When to choose LangChain

You live in its ecosystem — the integration count is real, and LangGraph's checkpointed workflows are
the offered way to get durable structure. If you've already standardized on it, the migration skill
exists: [`skills-langchain-to-pydantic-ai`](https://github.com/pydantic/skills-langchain-to-pydantic-ai).

## When to choose Pydantic AI

You're building a production agent: typed deps as a security boundary, budgets that halt before side
effects, cancellation as a typed outcome, specs that fail at load, evals in CI, durability by
wrapping instead of by rewriting as a graph. For the row-by-row runnable evidence, see
[the production agent](production-agents.md).

## Summary

LangChain's best is its ecosystem; LangGraph's best is checkpointed workflows. Neither ships a typed
deps boundary, a tool-boundary pause, or cancellation as a typed outcome. Their graph is the
abstraction you adopt; our loop is a value you drive.

*Pydantic AI claims verified offline; LangChain/LangGraph behavior probed against 1.3.1 / 1.2.1
(records in the [framework-comparison series](https://github.com/pydantic/pydantic-ai-notes)).*