# Project-Scale LangGraph Migration Patterns

Use this reference for substantial LangGraph graphs and production services. These are architecture studies, not drop-in ports.

## Contents

- [Open Deep Research](#open-deep-research)
- [Route Deep Agents separately](#route-deep-agents-separately)
- [Choose a migration boundary](#choose-a-migration-boundary)

## Open Deep Research

Source study: [langchain-ai/open_deep_research at `b764481`](https://github.com/langchain-ai/open_deep_research/tree/b764481fca7f0dbf00b2c70239bd97cea59d1059), reviewed 2026-07-17. Treat this as an example; inspect the target checkout because the default branch may have changed.

Evidence surfaces: [`deep_researcher.py`](https://github.com/langchain-ai/open_deep_research/blob/b764481fca7f0dbf00b2c70239bd97cea59d1059/src/open_deep_research/deep_researcher.py) defines the graph phases and supervisor/researcher loops; [`state.py`](https://github.com/langchain-ai/open_deep_research/blob/b764481fca7f0dbf00b2c70239bd97cea59d1059/src/open_deep_research/state.py) defines state and reducers; [`configuration.py`](https://github.com/langchain-ai/open_deep_research/blob/b764481fca7f0dbf00b2c70239bd97cea59d1059/src/open_deep_research/configuration.py) owns phase-specific models and limits; [`utils.py`](https://github.com/langchain-ai/open_deep_research/blob/b764481fca7f0dbf00b2c70239bd97cea59d1059/src/open_deep_research/utils.py) owns search/MCP selection and token recovery; [`run_evaluate.py`](https://github.com/langchain-ai/open_deep_research/blob/b764481fca7f0dbf00b2c70239bd97cea59d1059/tests/run_evaluate.py) and [`evaluators.py`](https://github.com/langchain-ai/open_deep_research/blob/b764481fca7f0dbf00b2c70239bd97cea59d1059/tests/evaluators.py) define end-to-end evaluation.

### How it works

- A top-level `StateGraph` clarifies the request, writes a research brief, runs a research-supervisor subgraph, and generates the final report.
- The supervisor uses structured tool calls to create multiple research topics, caps fan-out, invokes researcher subgraphs concurrently, aggregates notes, and reflects until a limit or completion signal.
- Each researcher runs its own model/tool loop over search, MCP, and reflection tools, then compresses its history into a focused result.
- Configuration selects different models for summarization, research, compression, and final writing.
- Typed state plus reducers accumulate messages, raw notes, compressed notes, counters, the brief, and the report.
- LangSmith datasets and a domain benchmark evaluate the complete report, not just unit-level syntax.

### Pydantic AI migration shape

Keep the four-phase topology. Do not collapse it into one giant agent prompt.

1. Model configuration and runtime clients as typed dependencies.
2. Use small Pydantic AI agents with typed outputs for clarification and research-brief creation.
3. Implement the supervisor loop in plain async Python or `pydantic_graph`. Let the supervisor produce a typed list of research tasks, cap it in application code, and run researcher agents under an explicit semaphore and shared usage budget.
4. Give each researcher provider-adaptive `WebSearch`, explicit search tools, and/or MCP toolsets. Preserve completion limits and capture raw evidence separately from compressed output.
5. Use a final writer agent whose input contract is the brief plus bounded findings. Preserve truncation, citation, and provenance behavior.
6. Port the existing benchmark cases to `pydantic_evals`; compare report quality, citation validity, latency, requests, and cost before cutover.

The source hand-builds an inner ReAct loop. A Pydantic AI `Agent` already owns a model/tool loop, so keep the outer supervisor graph while replacing inner researcher nodes one at a time.

Critical parity risks: bounded parallelism, reducer and result order, per-phase model settings, raw-note provenance, token-limit recovery, native versus local search behavior, structured-output retry, cancellation, and total usage propagation across child agents.

### Safest first slice

Start with clarification or research-brief generation rather than the supervisor or final writer. Freeze a framework-neutral request and typed result, run source and target against the same benchmark cases, and shadow outputs before routing traffic. This proves model configuration, instructions, structured output, tracing, and caller adaptation without changing parallel research, citations, or write side effects.

## Route Deep Agents separately

If inventory finds `create_deep_agent`, `HarnessProfile`, Deep Agents backends, repository skills, virtual files, planning, compaction, subagent registries, or sandbox/deployment orchestration, use the dedicated `migrate-deep-agents-to-pydantic-ai` skill. It owns Deep Agents defaults, Pydantic AI Harness capability selection, Open SWE, remote sandboxes, background work, and Deep Agents project playbooks.

Do not run both migration skills over the same project. This skill may still inventory shared LangGraph or LangChain primitives inside a mixed repository, but the Deep Agents skill owns the final migration ledger and plan.

## Choose a migration boundary

Choose the smallest boundary that produces user-visible value:

- **Tool boundary:** wrap existing LangChain tools temporarily when the agent loop can move first.
- **Agent boundary:** put a new Pydantic AI agent behind the existing graph node or service endpoint.
- **Graph boundary:** replace one subgraph while retaining the LangGraph deployment runtime temporarily.
- **Product boundary:** expose old and new runtimes behind one application-owned request/event protocol.

Prefer a reversible boundary. Avoid a flag deep inside tool execution that can mix frameworks within one side-effectful run; route the entire run or subgraph to one implementation.

For a production product boundary, make these surfaces explicit:

| Surface | Minimum contract |
|---|---|
| request | version, authenticated context, stable run/thread identity, limits, runtime route |
| event | version, correlation and sequence IDs, lifecycle/tool/output event types, reconnect semantics |
| deferred action | typed approval/cancel/resume request and durable correlation, or explicit `none` with the retained owner |
| final result | typed success/error outcome, usage, remaining side effects, and trace correlation |

List retained webhook, queue, scheduler, thread, sandbox, credential, CI, and deployment owners individually. Put middleware order, tool visibility, streaming compatibility, mid-run steering, usage limits, retries, and cancellation in the parity ledger even when the first slice does not exercise them; mark each `preserved`, `not applicable`, or `deferred with owner`.
