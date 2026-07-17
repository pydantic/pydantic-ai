# Validated Workaround Recipes

Use this reference after finding a non-1:1 mapping. The job is not complete until the gap has a working Pydantic AI construction, a parity probe, and a named owner—or is an explicit blocker.

## Contents

- [Required outcomes](#required-outcomes)
- [Choose the replacement mechanism](#choose-the-replacement-mechanism)
- [Prompt replacement and history](#prompt-replacement-and-history)
- [Middleware and local tool retry](#middleware-and-local-tool-retry)
- [Structured output and return-direct tools](#structured-output-and-return-direct-tools)
- [Approval and durable resume](#approval-and-durable-resume)
- [Workflow state, fan-out, and reducers](#workflow-state-fan-out-and-reducers)
- [Streaming and event contracts](#streaming-and-event-contracts)
- [Limits and persisted budgets](#limits-and-persisted-budgets)
- [Evidence rules](#evidence-rules)

## Required outcomes

Assign every semantic-gap row one outcome:

| Outcome | Meaning |
|---|---|
| `validated-native` | A Pydantic AI primitive preserves the required behavior in a deterministic parity spike. |
| `validated-adapter` | Small application or capability code closes the gap, and the spike proves the public contract. |
| `integration-required` | A concrete construction exists, but the target project/version, backend, provider, or operational boundary has not passed its own parity test. |
| `blocker` | No acceptable construction has passed. Do not migrate that slice. |

Do not leave a row at “redesign” or “not 1:1.” Select the most native mechanism that can preserve the behavior, write the smallest source/target reproduction, and measure both. Keep spike code outside the product and skill; promote only stable contract tests.

## Choose the replacement mechanism

| Source behavior | First Pydantic AI construction to spike | Fallback owner |
|---|---|---|
| current developer policy | `instructions` or one dynamic `@agent.instructions` function | model-request hook |
| replace-not-compose prompt middleware | ordered `Hooks.on.before_model_request` replacements | request adapter |
| missing/untrusted historical system prompt | `ReinjectSystemPrompt(replace_existing=True)` | server history adapter |
| before/after/wrap middleware | `Hooks`; use `on.model_request`, `on.tool_execute`, or another wrapper hook for nesting | custom `AbstractCapability` |
| same-handler transient tool retry | `Hooks.on.tool_execute` or service-client retry | application service |
| model-correctable tool arguments | `ModelRetry` / tool retry budget | agent |
| provider-native structured response | `NativeOutput` | provider-specific adapter |
| tool-based structured response | `ToolOutput` | output adapter |
| `return_direct=True` | named output function wrapped in `ToolOutput` | explicit application route |
| HITL tool approval | deferred tools plus authenticated pending-action store | durable workflow |
| checkpoint/replay | `DBOSAgent`, `TemporalAgent`, `PrefectAgent`, Restate, Kitaru, or another selected durable runtime | application workflow |
| graph state and reducers | typed application state or `pydantic_graph`; branch-local results plus explicit reducer | orchestration layer |
| complete execution with events | `run(event_stream_handler=...)`, `run_stream_events()`, or `iter()` | event adapter |
| per-run request/tool limits | `UsageLimits` and shared `RunUsage` | application budget service |
| persisted thread/graph-step limits | per-thread serialization or transactional reservation plus settlement/reconciliation | workflow store |

## Prompt replacement and history

If the desired policy is composition, put all policy fragments in `instructions`. If the source truly uses last-replacement-wins behavior, reproduce that rather than accidentally strengthening or weakening the prompt.

The simplest validated route is to collapse a last-replacement-wins source chain into one effective dynamic instruction function, provided overwritten prompt functions are pure and do not own required side effects. If evaluating every replacement is itself behavioral, ordered `before_model_request` hooks can replace the latest request instructions.

Do not rely on a success-only `after_run` hook to clean request instructions: failed or cancelled runs bypass it. If the source does not checkpoint request prompts, omit boundary-only instruction fields in the application message serializer, or use a reversible run wrapper whose `finally` path is proved under success, failure, and cancellation. Probe at least two dependency values plus a continued conversation and compare both actual model input and persisted application history.

For UI or database history that omits system prompts, use:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    model,
    system_prompt='Authoritative server policy',
    capabilities=[ReinjectSystemPrompt(replace_existing=True)],
)
```

`replace_existing=True` is important for untrusted history: it removes client-supplied system prompts before adding the server policy. This solves prompt authority, not general message conversion. Keep a fail-closed converter for the exact LangChain message subset the application retains; reject tool, reasoning, multimodal, or provider metadata until their round-trip is tested.

## Middleware and local tool retry

Pydantic AI `Hooks` wrapper hooks preserve nested control flow and can short-circuit by returning a response without calling the handler. Register separate capabilities in source order and trace their exact before/after sequence. Current decorator names omit `wrap_`: use `@hooks.on.model_request` and `@hooks.on.tool_execute`.

Use a tool-execution wrapper when LangChain retries the same handler locally. A validated shape is:

```python
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import Hooks

retry_hooks = Hooks()

@retry_hooks.on.tool_execute(tools=['lookup'])
async def retry_lookup(
    ctx: RunContext,
    *,
    call,
    tool_def,
    args: dict[str, Any],
    handler,
) -> Any:
    for attempt in range(2):
        try:
            return await handler(args)
        except TransientServiceError:
            if attempt == 1:
                raise
    raise AssertionError('unreachable')
```

This can match LangChain at two handler executions but only the original tool-request model round plus finalization. `ModelRetry` is deliberately different: it asks the model to issue a corrected call and therefore adds a model request. Add exception filtering, backoff, timeout, observability, exhaustion behavior, and idempotency appropriate to the service; never blindly retry side effects.

Use a custom capability rather than a pile of hooks when a source middleware unit owns tools, instructions, settings, and lifecycle behavior together. Use a model wrapper for provider failover/transport policy, and application code for routing or persisted workflow-state jumps.

## Structured output and return-direct tools

Match the source transport explicitly:

```python
from pydantic_ai import Agent, NativeOutput, ToolOutput

native_agent = Agent(model, output_type=NativeOutput(Answer))
tool_agent = Agent(model, output_type=ToolOutput(Answer, name='Answer'))
```

A parity probe must inspect the model request: native mode has an output object and no output tool; tool mode exposes the named output tool. Also compare invalid-output retries, co-emitted function tools, final DTO, and provider fallback.

When a LangChain tool uses `return_direct=True`, and the model must choose it as a terminal action, migrate it as an output function rather than a normal function tool:

```python
from pydantic_ai import Agent, RunContext, ToolOutput

def export_report(ctx: RunContext[Deps], report_id: str) -> str:
    return ctx.deps.reports.export(ctx.deps.customer_id, report_id)

agent = Agent(
    model,
    deps_type=Deps,
    output_type=ToolOutput(export_report, name='export_report'),
)
```

A disposable skill-development spike on Pydantic AI `2.10.1.dev24` observed one output-function execution and one model call. Treat that as a candidate construction, not validation for the target project: rerun the source/target probe before assigning `validated-native`. It changes the model contract from “optional ordinary tool” to “terminal output choice,” so use a union/list of output choices when other terminal outcomes exist. Output functions can run on partial values under `run_stream()`; guard side effects with `ctx.partial_output` or use a complete-execution API.

## Approval and durable resume

Use `requires_approval=True` or raise `ApprovalRequired`, then persist the returned `DeferredToolRequests`. The application record should include:

- application approval ID and Pydantic tool-call ID;
- authenticated tenant, initiator, and permitted approver;
- tool name and original validated arguments;
- serializable dependencies or stable references to reconstruct them;
- complete server-owned message history;
- pending/consuming/completed status, result, and one-time consumption version;
- business idempotency key for the protected effect.

On resume, authenticate first, claim the pending row atomically, reconstruct the original history/dependencies, and pass server-created `DeferredToolResults`. Cache the completed result. Unknown, foreign, modified, or already-consumed approvals must fail before the agent or tool executes.

A disposable SQLite development spike exercised this adapter across a fresh service instance, including foreign-principal rejection and a simulated crash after the tool. The tool was attempted twice after recovery but its unique business idempotency key produced one durable effect. This is design evidence, not target-project validation or a claim that arbitrary external APIs are exactly once; keep the outcome `integration-required` until the selected store passes the same probe.

For long waits and process recovery, put the loop in a durable runtime. Pydantic AI supplies wrappers such as `DBOSAgent`, `TemporalAgent`, and `PrefectAgent`; Restate and Kitaru also provide integrations. A disposable two-process DBOS development spike with a stable agent name and workflow ID returned the persisted first result without repeating the model call. This narrows the candidate design but does not change the target outcome from `integration-required`. For DBOS, decorate non-deterministic or I/O tool functions with `@DBOS.step`; they are not automatically durable merely because the agent is wrapped.

Run a real process-kill/restart test against the selected production backend. Keep stable agent/toolset IDs, serializable dependencies, and explicit workflow signals/events for approval. Do not generalize DBOS evidence to Temporal, Prefect, Restate, Kitaru, or a different database.

## Workflow state, fan-out, and reducers

Keep model messages out of workflow state. Persist plans, next step, joins, pending actions, counters, and domain progress in a typed workflow record or durable runtime. Pass only model protocol history through `message_history`.

For a fixed fan-out, create branch-local tasks and reduce after they finish. Preserve source ordering deliberately, for example by keeping the input index and sorting before the reducer. Use `asyncio.TaskGroup` when sibling cancellation on failure is required; use `gather(..., return_exceptions=True)` only when partial results are part of the contract. Put concurrency limits in a semaphore.

Use `pydantic_graph` when the state machine itself needs typed, inspectable nodes and joins. It is not a checkpoint replacement: current graph APIs do not supply LangGraph-style built-in persistence. Wrap the workflow in a durable runtime or persist transitions in application code.

For model-emitted function tools, Pydantic AI runs tools concurrently by default. Set `sequential=True` on a barrier tool, or use `with agent.parallel_tool_call_execution_mode('sequential')` for a whole run. These settings control tool execution; they do not reproduce LangGraph reducers or `Send` branch state.

## Streaming and event contracts

Do not expose framework events as the public API. Normalize source and target into an application envelope with version, run ID, sequence, correlation/tool-call ID, event kind, payload, and terminal/error semantics.

Use `run_stream()` only when committing the first matching output is the intended contract. When all function tools, retries, and side effects must complete, use `run(event_stream_handler=...)`, `run_stream_events()`, or `iter()`, then emit the application terminal event from the completed `AgentRunResult`.

Probe token/tool/final ordering, co-emitted output and tools, consumer cancellation, cleanup, backpressure, reconnect cursor, and duplicate delivery. Durable runtimes have different streaming constraints: validate the selected wrapper rather than assuming core-agent streaming behavior survives unchanged.

## Limits and persisted budgets

Use `UsageLimits` for Pydantic AI units and reuse one `RunUsage` object when several runs or child agents share a budget:

```python
from pydantic_ai import RunUsage, UsageLimits

usage = RunUsage()
limits = UsageLimits(request_limit=12, tool_calls_limit=20)
result = await agent.run(prompt, usage=usage, usage_limits=limits)
```

This does not implement LangGraph `recursion_limit` or LangChain's persisted per-thread model-call counter. Keep a workflow-step counter and thread budget in the application store. For a strict shared budget, either serialize all work for one thread or reserve capacity transactionally before dispatch with compare-and-swap/row locking, then settle measured usage and reconcile abandoned reservations after crashes. A check-before-call followed by an update-after-call is not safe under concurrency or process failure. Preserve the source failure shape if callers depend on a synthetic terminal message rather than an exception.

Test boundary minus one, boundary, and boundary plus one. Pydantic AI checks a parallel tool-call batch atomically against `tool_calls_limit`; prove whether the source instead allows a prefix. Include child-agent usage, replayed steps, and resumed runs.

## Evidence rules

Keep workaround spikes disposable and deterministic:

1. Print versions, module origins, and the Pydantic AI checkout SHA.
2. Use real LangChain/LangGraph and Pydantic AI APIs with fake/function models; do not mock the behavior under comparison.
3. Count model calls, tool calls, side effects, persisted records, and events.
4. Add failure and restart injection where the workaround owns retries, durability, or approval.
5. State the narrow guarantee proved and what remains unproved.
6. Delete the spike after extracting the recipe; add a stable product test for the public contract.

Re-run the relevant recipe against the target project's installed versions. These constructions were exercised during skill development, but framework and integration behavior can change.
