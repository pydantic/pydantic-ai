# The production agent

Every framework demo can run an agent. Production needs more. This page walks the checklist that
matters when the agent ships — each row is a runnable snippet, offline, no API keys. Run any of them
with:

```bash
uv run -m pydantic_ai_examples.comparisons.<name>
```

(Individual steps don't need side-by-side comparison: the point is which framework ships the seam at
all. Competitor frameworks that lack a row are named per row; their best features are stated on their
own pages.)

## 1. Trusted state: a boundary the model cannot cross

The DB password lives in `deps`. Only the tool may read it. The model receives tool definitions —
nothing else — and its request payload never contains the secret.

```snippet {path="/examples/pydantic_ai_examples/comparisons/deps_boundary.py"}
```

```
request payload contained the db password: False
tool executed with deps ('done'); model saw only tool definitions
```

No competitor has this seam: other frameworks pass "context"/"inputs" through the loop, where the
model's request history can echo it (LangChain `context_schema`, OpenAI `TContext`, CrewAI inputs,
ADK invocation context).

## 2. Capabilities that load on demand

Request 1: the refund tool is absent from the request payload. Request 2: the model asks to load the
capability. Request 3: the tool exists. The model cannot touch what it hasn't loaded.

```snippet {path="/examples/pydantic_ai_examples/comparisons/deferred_capability.py"}
```

```
requests: 4
  req1: tools=['load_capability'] cap_instruction_in_messages=False
  req2: tools=['load_capability'] cap_instruction_in_messages=False
  req3: tools=['load_capability', 'refund_status'] cap_instruction_in_messages=True
  req4: tools=['load_capability', 'refund_status'] cap_instruction_in_messages=True
deferred tool visible before load_capability: False
```

This ran offline against a stub model; the same protocol ran live against Anthropic with the tool
server-hidden (`defer_loading=True`) until loaded. Extend the pattern: capabilities bundle tools +
instructions + settings + hooks, order themselves, serialize into `AgentSpec`, and can observe or
transform the run's event stream.

## 3. Cancellation is a typed, resumable outcome

A tool may stop the run. `ctx.cancel()` requests it; the run ends in a catchable `RunCancelled`
carrying everything completed before the stop — resume by passing that history to the next run.

```snippet {path="/examples/pydantic_ai_examples/comparisons/cancel_from_tool.py"}
```

```
run ended with RunCancelled; completed work preserved (2 message(s))
cancellation is a typed, catchable, resumable outcome
```

And from outside: a stop button in another thread interrupts a blocked synchronous run.

```snippet {path="/examples/pydantic_ai_examples/comparisons/cancel_token_thread.py"}
```

```
blocked run_sync interrupted from another thread -> RunCancelled
```

Compare: OpenAI SDK cancels the streamed run only; Claude SDK = kill the subprocess; LangGraph =
interrupt is graph state and resuming re-runs the node's LLM call; smolagents/CrewAI = kill the
thread; Google ADK exposes no user cancellation API; AG2 cancels via a durable envelope.

## 4. Budgets halt before side effects, not after

The model asks for two tool calls in one response; the limit allows one. The whole batch is
rejected — `UsageLimitExceeded` — and *neither* tool ran.

```snippet {path="/examples/pydantic_ai_examples/comparisons/usage_limits_atomic.py"}
```

```
UsageLimitExceeded: The next tool call(s) would exceed the tool_calls_limit of 1...
tool executions that happened: 0
```

Budget checks precede execution. A framework that stops mid-batch lets the first side effect happen
and calls it a limit.

## 5. History repairs itself

A run that dies mid-tool leaves a dangling tool call — invalid for any provider. The next run closes
it out before the request goes out.

```snippet {path="/examples/pydantic_ai_examples/comparisons/history_repair.py"}
```

```
outgoing request carried a synthesized result for t1: ['The tool call was interrupted before a result was produced.']
history was provider-valid: no malformed pairing sent to the model
```

## 6. Specs fail at load, not at 3 a.m.

A template typo errors against the typed deps schema at construction, naming the field:

```snippet {path="/examples/pydantic_ai_examples/comparisons/spec_validation.py"}
```

```
TemplateSchemaError: 1 error(s) found:
  - non_existent_field: Field 'non_existent_field' not found in schema
valid template -> Agent('support') runs offline
```

The same spec is data: YAML, schema file, publish, load — construction-time validation holds on the
dict/YAML path (a pre-built Python `AgentSpec` object skips it: validate via `from_spec(dict, ...)`).

## 7. The run is an event stream you can observe or transform

Parts, tool calls, results, the final result — typed events, streamed. No opinion about what you do
with them: your auditor, your UI, your SSE adapter, or a capability transforming the stream.

```snippet {path="/examples/pydantic_ai_examples/comparisons/event_stream.py"}
```

```
events observed: ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent', 'AgentRunResultEvent']
final output: '42' (streamed while it happened)
```

## 8. Evals in CI, typed, offline

Same types as the agent, same harness as CI: dataset → evaluators → report.

```snippet {path="/examples/pydantic_ai_examples/comparisons/evals_ci.py"}
```

```
assertions passed: 100%
[Evaluation summary table]
```

## 9. Durability is a wrapper, not a rewrite

One agent source; the engine is the import. Temporal, DBOS, and Prefect agents wrap the same
definition. This one needs the engine installed — it's a reference, not an offline proof:

```snippet {path="/examples/pydantic_ai_examples/comparisons/durability_wrap.py"}
```

```
same agent source under:
  - TemporalAgent(agent, task_queue="tq")   # Temporal
  - DBOSAgent(agent, workflow_name="wf")    # DBOS
  - PrefectAgent(agent, task_name="t")      # Prefect
agent definition changes: 0 lines
```

Restate, Kitaru, and Airflow ship the same shape. Everything else here stays yours.

## What we don't ship (same tone)

- No first-party managed agent server.
- No TS/JS framework; our UI story is adapters + agents writing JSON.
- Curated integrations count, not exhaustive.
- `run_sync` can't be nested inside async code; a worker-thread tool can't be force-stopped.

*Versions verified 2026-09-10: pydantic-ai 2.42.0, pydantic-evals 2.42.0. Snippets are CI tests in
this repo; probe records in the [framework-comparison series](https://github.com/pydantic/pydantic-ai-notes).*