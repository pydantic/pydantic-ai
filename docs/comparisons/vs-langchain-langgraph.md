# Pydantic AI vs LangChain & LangGraph

"LangChain" covers several things, and this page is about one of them.

If you use LangChain for chat models, prompts, output parsers, retrievers, or a chain, LangGraph
never enters the picture — `langchain-core` doesn't depend on it, and plenty of production LangChain
code has no graph in it anywhere. Build an **agent**, though, and it does. In 1.x, `create_agent()`
returns a `CompiledStateGraph`, the `langchain` package requires `langgraph>=1.2.11`, and the old
`AgentExecutor` is gone. There is no longer a non-graph agent path, which is why this page treats the
two as one thing: comparing agents means comparing against the graph.

**Deep Agents** sits one layer up: a coding-agent harness with a filesystem, sandbox, subagents, and
skills, built on the same graph. Its counterpart on our side is
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness), so this page compares those
too — [further down](#one-layer-up-deep-agents-and-the-harness). **LangSmith** is a hosted platform
for traces and datasets; we don't ship one, we emit OpenTelemetry and you point it where you like.

Pydantic AI is the agent framework. An agent is a typed Python value, the run is an ordinary
coroutine, and everything else — durability, approvals, budgets, evals — attaches to that value
instead of changing its shape.

Where that stops being a matter of taste is the subject of the rest of this page.

## Where you're allowed to stop

Every agent that spends money eventually has to stop and ask a person. Both frameworks do this, and
for the common case they now do it the same way — worth saying plainly before the part where they
come apart.

LangChain ships a `HumanInTheLoopMiddleware`, which Deep Agents exposes as `interrupt_on`. Point it at
a tool and the run pauses before that tool executes. Resuming replays nothing: the middleware is its
own graph node, so the model call that chose the tool sits behind a finished checkpoint. If approving
a tool call is the only pause you need, this page has no argument to make and you should pick on other
grounds.

The difference starts when the pause you want isn't a tool call. A budget check partway through a long
deliberation, a review of a draft before the agent continues — there's no middleware for those, so you
write `interrupt()` inside your own node. Resume that, and the node restarts from its first line. We
ran a refund node that calls the model, writes an audit row, then interrupts:

| Pausing with | The model call | A side effect above the pause |
|---|---|---|
| `HumanInTheLoopMiddleware` / `interrupt_on` | runs once | runs once |
| `interrupt()` inside the node doing the work | **runs twice** | **runs twice** |
| `interrupt_after` on a node you split out first | runs once | runs once |
| Pydantic AI, `requires_approval=True` | runs once | runs once |

Row two is the one that costs money. You pay for that model call a second time, the audit row is
written twice, and so is anything else above the `interrupt()` — a card charge, an email, a counter.

Row three is the documented fix, and it works. But look at what it asks: to move a pause, you redraw
the graph. Where your agent may safely stop is decided by how you split it into nodes, which you did
before you knew where you would want to stop.

??? example "How we measured this"

    Three scripts, run offline against langchain 1.4.0 / langgraph 1.2.11. All three share one stub
    model, which appends to a list every time it is called and returns a canned tool call, so nothing
    here needs a network or an API key:

    ```python {test="skip" lint="skip"}
    class CountingModel(BaseChatModel):
        calls: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
            trail.append('MODEL_CALL')
            self.calls += 1
            if self.calls == 1:
                msg = AIMessage(content='', tool_calls=[{'name': 'issue_refund', 'args': {...}, 'id': 'c1'}])
            else:
                msg = AIMessage(content='Refunded A-4471.')
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def bind_tools(self, tools, **kw):
            return self
    ```

    That subclass is also the answer to "can you test a LangChain agent offline?" — it drives
    `create_agent` fine. There just isn't a test model in the box.

    First, an `interrupt()` inside the node that holds the work:

    ```python {test="skip" lint="skip"}
    def refund_node(state: S):
        trail.append('LLM_CALL')
        trail.append('AUDIT_ROW_WRITTEN')
        decision = interrupt({'question': 'approve refund of 38.00?'})
        trail.append(f'PAID_OUT:{decision}')
        return {'log': [f'done:{decision}']}
    ```

    ```text
    trail after resume: ['LLM_CALL', 'AUDIT_ROW_WRITTEN', 'LLM_CALL', 'AUDIT_ROW_WRITTEN', 'PAID_OUT:yes']
    LLM_CALL count: 2
    AUDIT_ROW_WRITTEN count: 2
    ```

    The same agent paused with `HumanInTheLoopMiddleware(interrupt_on={'issue_refund': True})`:

    ```text
    trail after resume: ['MODEL_CALL', 'PAID_OUT', 'MODEL_CALL']
    MODEL_CALL count: 2      # two different calls, neither repeated
    PAID_OUT count: 1
    ```

    And with the approval split into its own node, paused by `interrupt_after=['decide']`:

    ```text
    trail after resume: ['LLM_CALL', 'AUDIT_ROW_WRITTEN', 'PAID_OUT']
    LLM_CALL count: 1
    ```

In Pydantic AI the pause is at the tool call, and that is the only rule. A tool marked
`requires_approval=True` doesn't run; the run ends and hands you the pending call:

```python {title="approval_pause.py"}
from pydantic_ai import Agent, DeferredToolRequests

trail: list[str] = []

agent = Agent('openai:gpt-5.2', output_type=[str, DeferredToolRequests])


@agent.tool_plain
def look_up_order(order_id: str) -> str:
    trail.append('AUDIT_ROW_WRITTEN')
    return f'{order_id}: delivered=False, charged=38.00'


@agent.tool_plain(requires_approval=True)
def issue_refund(order_id: str, amount: float) -> str:
    trail.append('PAID_OUT')
    return f'refunded {amount} on {order_id}'


first = agent.run_sync('Refund order A-4471, the customer never received it.')
requests = first.output
assert isinstance(requests, DeferredToolRequests)
print('paused, waiting on:', [call.tool_name for call in requests.approvals])
#> paused, waiting on: ['issue_refund']
print('trail at the pause:', trail)
#> trail at the pause: ['AUDIT_ROW_WRITTEN']

# Hours later, in another process, a human says yes.
results = requests.build_results(approve_all=True)
second = agent.run_sync(message_history=first.all_messages(), deferred_tool_results=results)

print('trail after resume:', trail)
#> trail after resume: ['AUDIT_ROW_WRITTEN', 'PAID_OUT']
print('audit rows written:', trail.count('AUDIT_ROW_WRITTEN'))
#> audit rows written: 1
print(second.output)
#> Refunded A-4471.
```

The lookup ran once. The payout ran once, after approval. Between the two runs there is nothing live —
no held connection, no parked task — just a list of messages you can put in a database and read back
tomorrow.

`requires_approval=True` on the tool is the whole change. And when the pause you want isn't a tool
call, the answer is the same shape rather than a different mechanism: a `CancellationToken` or a tool
calling `ctx.cancel()` ends the run in `RunCancelled`, carrying the same resumable history, and that
works whichever way you started the run.

LangGraph can stop a run too, but only down one path: `stream_events(version='v3')` returns a
`GraphRunStream` with an `abort()`, and that method warns it's experimental when you touch it. Ordinary
`invoke()` and `stream()` have nothing, so stopping those means cancelling whatever task is executing
them. What `abort()` gives you is also a different thing — it closes the graph iterator so in-flight
nodes see `GeneratorExit`, and you keep whatever the checkpointer happened to write. Ours is a value
you catch.

## Why the two behave differently

LangGraph's unit of execution is the node, so its unit of recovery is also the node. A checkpoint
records which nodes finished; resuming means running the unfinished one, and a node that was
interrupted halfway is unfinished from the top. That is a coherent design — it is what makes time
travel and forking work — and the replay is the price of it.

Pydantic AI's unit of execution is the tool call, because the run is a plain loop over messages
and not a graph. Everything downstream follows from that:

- **Durability is added to the agent, not built into it.** `TemporalDurability()` is a capability you
  add to the same agent
  object and gives you Temporal's retries and crash recovery. So do the DBOS and Prefect wrappers in
  the repo, and the Restate, Kitaru, and Airflow adapters outside it. In LangGraph, durability is the
  checkpointer, and the checkpointer is a graph feature — you get crash recovery by expressing your
  control flow as a graph.
- **Trusted state is a separate typed argument.** `deps_type` holds your database handle, the
  customer ID, the API client. Tools read it; the model never sees it and cannot name it. LangChain's
  `context_schema` flows through the same state the loop reads from.
- **History is typed, owned data.** `first.all_messages()` is a list of Pydantic models you can
  serialize, inspect, edit, and hand to the next run. LangGraph's checkpoint is a copy of the whole
  state dict per step, which is why checkpoint size tracks your payload size.
- **Cancelling is a typed outcome, not a killed task.** A `CancellationToken` stops one or several
  runs from another thread, a tool can call `ctx.cancel()`, and the run ends in `RunCancelled`
  carrying the history — which resumes like any other, from any entry point. LangGraph's `abort()` is
  experimental, exists only on the v3 stream, and closes the iterator instead of returning you a
  result.

## One layer up: Deep Agents and the harness

When people say "Pydantic AI" they usually mean two libraries: the agent framework, and
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness), which adds what a coding agent
needs — a filesystem, a shell, subagents, skills, memory, compaction, planning, spend tracking, a
prompt-injection defender, and `CodeMode`, which collapses many tool calls into one sandboxed Python
program on the [Monty](https://github.com/pydantic/monty) runtime.

LangChain's answer at that layer is **Deep Agents**, and it is a serious one. Version 0.7.13 ships a
filesystem with `ls`, `read_file`, `write_file`, `edit_file`, `delete`, `glob`, `grep`, and `execute`;
subagents; skills; memory; summarisation; a permissions layer; and backends for state, a store, the
local filesystem, a local shell, and a hosted `LangSmithSandbox` — all behind a `BackendProtocol` you
can implement yourself. If you want a sandbox that works the afternoon you install it, theirs is
hosted and ours is a Modal account you bring.

The difference is what these features are made of, and it is the same difference as the rest of this
page. `create_deep_agent()` returns a `CompiledStateGraph`, and every Deep Agents feature is LangGraph
middleware on it. So the harness inherits the graph: the filesystem, subagents, and skills are nodes
and middleware hooks, and pausing anywhere the middleware doesn't already pause takes you back to
`interrupt()` and the node rule above.

Our harness features are capabilities — the same unit as any other capability, on the same loop. A
capability bundles tools, instructions, settings, and hooks together, and can defer its own loading
until the model asks for it. `FileSystem` composes onto a support agent as readily as onto a coding
agent, because there is no harness shape to adopt. Everything on this page — the tool-boundary pause,
the deps boundary, resumable cancellation, the durable engine wrappers — applies to a harness agent
unchanged, because it is the same agent.

Two honest notes. Deep Agents pins `langchain-anthropic` and `langchain-google-genai` as hard
dependencies, so its default surface leans on those two providers, while ours is provider-agnostic by
construction. And several harness capabilities of ours are still marked experimental — planning,
subagents, compaction, and runtime authoring among them — where Deep Agents' equivalents are shipped
and in use.


## The comparison, row by row

| | LangChain & LangGraph (1.4.0 / 1.2.11) | Pydantic AI (2.42) |
|---|---|---|
| Getting structure | `create_agent()` compiles to a `StateGraph`; retries, streaming, and checkpoints are graph features | Plain async; the run is a value, and `agent.iter()` exposes it node by node when you want that |
| Pausing for a human | `interrupt()`, replaying the enclosing node; or `interrupt_after` on a node you split out in advance | `requires_approval=True` on the tool; the run ends and resumes from the tool boundary |
| Crash recovery | Checkpointers, part of LangGraph | Six engines wrap the agent object: Temporal, DBOS, Prefect in-tree; Restate, Kitaru, Airflow outside |
| Trusted state | `context_schema`, part of the state the loop reads | `deps_type`, a separate typed argument tools read and the model cannot see |
| Cancellation | `abort()` on the experimental `stream_events(version='v3')` stream; nothing on `invoke()` or `stream()` | `CancellationToken`, `ctx.cancel()`, `RunCancelled` with resumable history, from any entry point |
| Extending the agent | Middleware, wrapping in LIFO order around each pass | Capabilities, bundling tools, instructions, settings, and hooks as one unit that can also load on demand |
| Testing offline | Three fake chat models ship in `langchain-core`, and all three raise `NotImplementedError` on `bind_tools`, so none can drive an agent; a dozen-line `BaseChatModel` subclass does work | `TestModel` calls your tools with no scripting; `FunctionModel` scripts them; `ALLOW_MODEL_REQUESTS = False` blocks real providers globally |
| Tracing | LangSmith is the paved road; OpenInference spans are available but carry zero `gen_ai.*` attributes | The OpenTelemetry GenAI semantic conventions, 36 `gen_ai.*` attributes, straight into the dashboards you already have |
| Budgets | `recursion_limit` caps graph depth; no token or money ceiling on the run | Requests, tool calls and tokens per run, plus `cost_limit` in USD across 41 providers, checked before the next request |
| Evals | Datasets and experiments in LangSmith | `pydantic-evals` in your test suite, sharing the agent's own types, no platform |
| Deploying an agent as config | Graphs are code | `AgentSpec` round-trips to YAML and validates prompt templates against `deps_type` at load |

## Moving an existing project

Tools port almost unchanged — a function with type hints and a docstring is a tool in both. Prompts
port unchanged. What doesn't port is the graph: nodes become ordinary control flow, and the state
dict splits into three things that were previously one, which is usually the point at which people
say the migration was worth it. Your deps become `deps_type`, your conversation becomes
`message_history`, and your workflow state becomes whatever your application already uses for state.

If what you're moving is a Deep Agents application rather than a plain LangChain agent, the target is
[pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness) rather than Pydantic AI on its
own, and the file, shell, and sub-agent tools you're relying on have direct counterparts there.

## When LangChain is the right answer

- You need an integration that exists there and nowhere else. The catalogue is far larger than ours
  and that is a real reason to choose it.
- Your problem genuinely is a graph — long-running workflows with branches, joins, and human review
  at known points — and you want time travel and forking over checkpoint history.
- Your team is already fluent in it and shipping. Rewriting a working system to change the shape of
  its pause is not a good trade.

## When Pydantic AI is the right answer

- You want to pause anywhere without redesigning the agent, and you don't want to pay for repeated
  work when you do.
- Credentials and identity must sit somewhere the model cannot reach.
- You want crash recovery from a specific engine your company already runs, without expressing the
  agent as a graph to get it.
- You want the agent's tests to run in CI, offline, deterministically, alongside everything else.

## FAQ

**Can I use both?**
Yes, and people do. The usual split is Pydantic AI for the agent and LangChain for a specific
integration, called from a tool like any other library.

**Is Pydantic AI a drop-in replacement?**
No. Tools and prompts carry over; the graph does not. Budget for rewriting control flow, and expect
the state dict to split into deps, history, and application state.

**Does Pydantic AI have anything like LangGraph's time travel?**
Not as a feature with that name. History is a list you own, so forking a conversation is slicing the
list and running from there — but there is no checkpoint browser, and no equivalent of replaying an
arbitrary node.

**What does Pydantic AI not have?**
No hosted platform, no managed deployment, and a much smaller integration catalogue. If you want a
tracing dashboard that works the day you install it, LangSmith is a product and we don't ship one.

---

*Measured 2026-09-10 against langchain 1.4.0, langgraph 1.2.11, langchain-core 1.6.2, deepagents 0.7.13, and
Pydantic AI 2.42. Both LangGraph traces come from scripts that log every model call and side effect across a
pause and a resume — one through `HumanInTheLoopMiddleware`, one through a hand-written `interrupt()` — run
against a stub chat model with no network. The dependency and `create_agent` return-type claims are read from
the installed distributions. `GraphRunStream.abort()` was reached by calling `stream_events(version='v3')` on
a compiled graph and confirmed to raise a `LangChainBetaWarning`; `invoke()` and `stream()` were checked for a
stop method and have none. The probes are the ones shown above, and they need nothing beyond those two
packages and no network. The Pydantic AI snippet on this page is executed by this repository's test suite on
every commit, so its output is what it printed. The `gen_ai.*` counts are distinct semantic-convention
attribute names found in each installed package's source; ours were also captured from a live run through a
plain OpenTelemetry exporter. We recheck this page's version pins and behaviour claims each time Pydantic AI
ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
