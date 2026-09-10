# Pydantic AI vs LangChain & LangGraph

Choosing an agent framework and you're down to
[Pydantic AI](../agent.md) and LangChain/LangGraph. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- state the model **cannot choose or see**, as a typed boundary (`deps_type`)
- **pauses at the tool boundary** that resume without re-running finished work
- **cancellation as a typed outcome** — a tool or a thread stops the run, history survives, resume is an ordinary run
- **durability by wrapping** (six engines) instead of expressing the agent as a graph

## Why the answers differ

LangGraph's structure is the graph you adopt; here the run is a value you drive. The consequences are the seams: their `interrupt()` resumes by re-running the enclosing node (the LLM call repeats); our pause sits at the tool call boundary. Deps, budgets, cancellation, and evals all hang off that difference. An independent review (Speakeasy, 2026-03) puts the cost of their structure plainly: no token-budget management (context bloat silently degrades long runs), and a full copy of the graph state per step, so checkpoints grow with your payloads.

One more data point, measured by someone else: an independent comparison (Speakeasy, 2026-03, [90-day five-framework benchmark](https://www.speakeasy.com)) scored developer experience 8/10 for Pydantic AI against 5/10 for LangChain, and credited type validation with catching 23 bugs in development that would have reached production.
## See it work

Say you need a loop with a human-safe pause.

In LangGraph, pausing is `interrupt()` — it's graph state, and resuming restarts the enclosing node, so the model call runs again (1.2.1, verified).

Your side, runs offline:

```python {title="graph_is_a_value.py"}
"""The agent loop is a value you can drive — no graph DSL required.

Pydantic AI doesn't need you to adopt a graph abstraction for structure.
The run is plain async code over a typed value; when you do want the
graph, it's the same value, iterated node by node.
"""
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('double', {'n': 21})])
    return ModelResponse(parts=[TextPart('42')])


agent = Agent(FunctionModel(model))


@agent.tool
def double(ctx, n: int) -> int:
    return n * 2


async def main():
    nodes = []
    async with agent.iter('what is 21*2?') as run:
        async for node in run:
            nodes.append(type(node).__name__)
    print('nodes in one run:', ' -> '.join(nodes))
    print('the loop is a plain value: iterate it, drive it manually, or let a capability transform it')


asyncio.run(main())


```

```text
nodes in one run: UserPromptNode -> ModelRequestNode -> CallToolsNode -> ModelRequestNode -> CallToolsNode -> End
the loop is a plain value: iterate it, drive it manually, or let a capability transform it
```

**Notice:** The nodes you just saw are a value you drive: the pause, the cancel, and every seam your app needs live on the loop — not inside a DSL you adopt.

## The details

| What you get | LangChain & LangGraph | Pydantic AI |
|---|---|---|
|---|---|---|
| Structure without a DSL | Adopt a `StateGraph` to get checkpoints, retries, streaming | Plain async; the loop is a value you drive |
| Pause for a human | `interrupt()` — graph state; resuming **re-runs the node's LLM call** | `ctx.cancel()` / deferred tools — pause at the tool boundary, resume clean |
| Trusted state | `context_schema` / state dict flows through the loop | `deps_type` — model cannot choose or see it |
| Extend the agent | Middleware (LIFO order) intercepts steps | Capabilities: bundle tools + instructions + settings + hooks, orderable, serializable |
| Durability | Checkpointers are a graph feature — you must use LangGraph | Engines wrap the same agent (Temporal/DBOS/Prefect/Restate/Kitaru/Airflow) |
| Cancellation | None: interrupt is state; killing a run = kill the task | Typed: `RunCancelled` with resumable history; external `CancelledError` preserved |
| Events | Super-step graph events | Part/tool/result/final typed events; capabilities can transform the stream |
| Offline tests | `GenericFakeChatModel` can't `bind_tools` | `TestModel`/`FunctionModel` drive the whole pipeline deterministically |

## If this answer doesn't fit you

If your product genuinely lives in the LangChain ecosystem — its integration breadth, its community patterns, its checkpointed workflows — that's a real thing to build on, and we're not going to argue you out of it. What this page does is name what the ecosystem doesn't hand you: the seams on the left. And if you're on board with those and want to transform your project, we have the skills to walk you over — [skills-langchain-to-pydantic-ai](https://github.com/pydantic/skills-langchain-to-pydantic-ai), plus a [Deep Agents migration playbook](https://github.com/pydantic/skills-deepagents-migration) if that's your flavor.

---

## FAQ

**Is Pydantic AI a drop-in replacement for LangChain & LangGraph?**
Drop-in, no — the loop and the seams are different, even though the ideas carry over (tools,
prompts, outputs). If you're weighing a move, that honesty is the point of this page: read the fits
list and run the proof before you decide.

**When should I use LangChain & LangGraph on its own?**
When your product genuinely lives in the LangChain ecosystem — the integration breadth, the community patterns, the checkpointed workflows — that's a real thing to build on, and it's not what this page is arguing against.

**Why do people pick Pydantic AI over LangChain & LangGraph?**
Because the loop is yours end to end — typed deps, cancellation that resumes, budgets that stop side
effects before they start, evals in CI — and every one of those claims is a snippet on this page you
can run in seconds. Community threads on r/AI_Agents add "documentation" and "low abstraction" to
that list; see Independent takes on the [overview](index.md).


---

*Versions: langchain 1.3.1 / langgraph 1.2.1; Pydantic AI 2.42.0 — 2026-09-10. Snippets re-executed by this repository's tests.*
