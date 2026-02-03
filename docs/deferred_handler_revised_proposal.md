### Description

# Deferred Handler Minimal-Core Proposal Draft

## Context
We need a minimal-core `deferred_tool_handler` proposal that preserves blocking approvals while keeping policy and UI in third-party packages. This draft should enable an initial replacement of `pydantic-ai-blocking-approval` without expanding core beyond the loop mechanics.

---

# Proposal: `deferred_tool_handler` for Inline Resolution of Deferred Tools (Minimal Core)

## Summary

Add an optional `deferred_tool_handler` parameter to `Agent` and all run variants (`run`, `run_sync`, `run_stream`, `run_stream_sync`, and `iter`) that enables inline resolution of deferred tool calls (approvals and external calls). When a handler is provided, it must return results for all deferred calls in that response; missing results raise `UserError`. When no handler is provided, deferred calls bubble up as `DeferredToolRequests` (current behavior).

The core stays minimal: it only defines how deferred tool requests are surfaced to a handler and how results are applied. Approval policy, UI, caching, and "blocked" semantics remain in third-party packages.

## Goals

- Enable blocking, inline approval workflows without re-implementing the agent loop.
- Preserve current behavior when no handler is provided.
- Keep approval policy and UI in third-party packages.

## Non-Goals

- Define approval policy rules in core.
- Add new approval UI or caching APIs in core.

## Motivation

PydanticAI's current deferred tools pattern works well for batch/review workflows where tool calls are collected, reviewed externally (dashboard, email, async process), and then resumed with a new `agent.run()` call.

However, for interactive scenarios (CLI agents, coding assistants, multi-step dangerous operations), a blocking pattern is more appropriate:

1. The agent proposes tool calls
2. The user reviews and approves/denies inline
3. Approved tools execute immediately
4. The agent sees results and continues
5. Repeat until task complete

Currently, implementing this requires:
- Re-implementing the agent loop externally
- Managing message history across multiple `run()` calls
- Complex state handling for multi-agent scenarios (see #3274)

A first-class `deferred_tool_handler` makes this pattern easy while keeping approval policy out of core.

## Proposal

### API

```python
from typing import Callable, Awaitable
from pydantic_ai import Agent, RunContext, DeferredToolRequests, DeferredToolResults

DeferredToolHandler = Callable[
    [RunContext[Deps], DeferredToolRequests],
    DeferredToolResults | Awaitable[DeferredToolResults]
]

# On Agent constructor (optional)
agent = Agent(
    'openai:gpt-4o',
    deferred_tool_handler=my_handler,  # Optional
)

# On run methods (optional override)
result = await agent.run(
    prompt,
    deferred_tool_handler=my_handler,  # Overrides agent default
)

# On iter (optional override)
async with agent.iter(
    prompt,
    deferred_tool_handler=my_handler,  # Overrides agent default
) as agent_run:
    async for node in agent_run:
        ...
```

If you want the default (non-blocking) behavior, **omit** `deferred_tool_handler` entirely. When the parameter is provided it must be a callable.

The handler receives `RunContext[Deps]` for consistency with other PydanticAI patterns, giving access to dependencies (database connections, config, etc.) useful for policy decisions.

**Type locations:** `DeferredToolRequests`, `DeferredToolResults`, `ToolApproved`, `ToolDenied`, and `RunContext` are defined in `pydantic_ai.tools` and re-exported from `pydantic_ai`.

**Naming note:** `DeferredToolResults` already exists, but the name is a bit misleading since it also includes approvals/denials. A follow-up could rename it (e.g., `DeferredToolResolutions` or `DeferredToolOutcomes`) with a deprecation alias for backward compatibility.

### DeferredToolResults structure

See [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults] for the exact structure.

### Core Behavior

When `deferred_tool_handler` is set and an LLM response contains deferred tool calls, execution happens in two phases:

1. Execute non-deferred tools from the response, collecting any deferrals (pre-marked external/approval-required tools, plus those discovered via `CallDeferred`/`ApprovalRequired` during execution).
2. Call `deferred_tool_handler(ctx, requests)` with all deferred calls from that response and await the result.
3. Validate the handler returned results for all deferred calls (else raise `UserError`).
4. Apply the returned `DeferredToolResults` to the pending tool calls.
5. Execute approved deferred calls (can be parallel).
6. Continue the agent loop with tool results.

The key invariant is that the handler sees all deferred calls from a single model response before any of those deferred calls execute. Non-deferred tools still execute immediately; if you need a user decision before any side effects, mark the tool as `requires_approval` or raise `ApprovalRequired`.

When `deferred_tool_handler` is `None` (default):

- Current behavior unchanged, return `DeferredToolRequests` to caller

### What the LLM Sees

A key difference between the current offline flow and the blocking handler flow is what the LLM perceives:

**Current offline flow (no handler):**
```
LLM: "I'll delete those files" → calls delete_files(...)
                ↓
Tool raises ApprovalRequired
                ↓
run() returns DeferredToolRequests to caller
(no tool result is sent to the model in this run)
                ↓
[time passes, external review happens]
                ↓
Caller resumes: run(deferred_tool_results={...approved...})
                ↓
Tool actually executes
                ↓
LLM sees: "Deleted 3 files: foo.txt, bar.txt, baz.txt"
```

**Blocking flow with handler (approved):**
```
LLM: "I'll delete those files" → calls delete_files(...)
                ↓
Tool raises ApprovalRequired
                ↓
Handler called → user approves → ToolApproved()
                ↓
Tool actually executes
                ↓
LLM sees: "Deleted 3 files: foo.txt, bar.txt, baz.txt"  ← normal result!
                ↓
Loop continues in same run()
```

**Blocking flow with handler (denied):**
```
LLM: "I'll delete those files" → calls delete_files(...)
                ↓
Handler called → user denies → ToolDenied(message="User denied: too risky")
                ↓
Tool does NOT execute
                ↓
LLM sees tool result: "User denied: too risky"
                ↓
Loop continues - LLM can adapt: "I understand, let me try a safer approach..."
```

**What happens on denial:**
1. The tool function is **never called** - no side effects occur
2. A synthetic tool result is returned to the LLM containing the denial message
3. The agent loop continues with this result in the message history
4. The LLM can react to the denial (try alternative approach, ask for clarification, give up gracefully)

This matches how tool execution errors are already handled - the LLM receives an error message as the tool result and can adapt its behavior.

**Key insight:** With the blocking handler, from the LLM's perspective, deferred tools behave like normal tools—they either succeed with real results or fail with a denial message. The LLM doesn't know or care that a human was consulted in the middle. This is a cleaner mental model compared to "your request was saved for later review" which leaves things in limbo.

### Defaults and Edge Cases

- **No partial handling (v1)**: The handler must return results for **all** deferred tool calls in that response. Missing entries raise `UserError`.
- **External calls**: The handler must return results for deferred external calls as well. Missing entries raise `UserError`. Results are sent to the model as tool outputs (non-matching values are wrapped as tool returns). To signal failure, return a `ModelRetry` or `RetryPromptPart` instead of a plain value.
- **Deferral mapping**: `ApprovalRequired` and `requires_approval=True` map to `requests.approvals`; `CallDeferred` and external tool calls map to `requests.calls`.
- **Denial mapping**: `DeferredToolResults.approvals` accepts `ToolApproved`/`ToolDenied` or booleans. `True` is treated as `ToolApproved()`, `False` as `ToolDenied()` (default message). Use `ToolDenied(message=...)` for a custom denial message.
- **Re-deferral during inline handling**: If an approved tool call raises `CallDeferred` or `ApprovalRequired` again, the run ends with a new `DeferredToolRequests` containing those re-deferrals. Already-executed tools from the same response are not rolled back.
- **Output type requirement**: If deferred calls are returned (no handler, or a tool re-defers), `DeferredToolRequests` must be included in `output_type`, otherwise raise `UserError` (consistent with current behavior).
- **Metadata propagation**: No automatic pass-through. Tool execution only receives `DeferredToolResults.metadata` (same as the current resume flow). If a handler wants request metadata available during execution, it should explicitly copy it into `DeferredToolResults.metadata`.
- **Handler exceptions**: If the handler raises, the run fails and the exception propagates. If you want deny-all or fallback behavior, catch exceptions inside the handler.
- **Handler precedence**:
- If a handler is provided to `run()`/`run_sync()`/`run_stream()`/`run_stream_sync()`/`iter()`, it overrides the `Agent` default.
  - If a handler is not provided, the `Agent` default is used.
  - Note: When using a handler, you do not need `DeferredToolRequests` in `output_type` unless you expect tools to re-defer.

### Sync vs Async and Streaming

- `run_sync()` runs the regular `run()` in an event loop, so async handlers work fine. No special restriction needed.
- `run_stream()` and `run_stream_sync()` keep their current semantics: they stop after the first final output and do not pause/resume mid-stream.
  - The handler is invoked only when the run continues to tool execution (i.e., before any final output has been streamed).
  - If a final output has already been streamed, deferred tool calls in that response are not handled inline and the handler is not invoked; they are effectively skipped and not returned as `DeferredToolRequests`.
  - If you need streaming plus guaranteed tool execution and approvals, use [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] with an event stream handler or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter].

### Granularity: Per LLM Response

The handler is called once per LLM response that contains deferred tools, not once per tool call or once for the entire run.

This design enables:

| Benefit | Explanation |
|---------|-------------|
| Full context per turn | User sees all tool calls that are not pre-approved before any of those deferred calls execute |
| Batch decisions | "Approve all reads", "Deny all writes" |
| Parallel execution | Approved tools from same response can run concurrently |
| Blocking between turns | Model's next response waits for these results |

```
LLM Response 1 -> [Tool A, B, C] -> handler -> approve/deny -> execute -> results
LLM Response 2 -> [Tool D, E] -> handler -> approve/deny -> execute -> results
...
Final response -> output
```

## Extension Points for Approval Policies (Non-Core)

The core does not define approval policy. Third-party packages can implement policy and UI using the hook provided by `deferred_tool_handler`.

### 1) Per-Call Policy Decisions

A wrapper/toolset can decide per call whether to:

- **pre_approve** (execute immediately)
- **needs_approval** (raise `ApprovalRequired`)
- **blocked** (prevent execution without prompting)

This can be done today by raising `ApprovalRequired` or raising a policy-specific error. The `deferred_tool_handler` is the bridge that enables inline blocking without core policy logic.

### 2) Metadata Pass-Through for UI

Metadata attached to deferred tools should be preserved in `DeferredToolRequests.metadata`, and passed through to tool execution by default (see **Metadata propagation** above). If the handler returns `DeferredToolResults.metadata`, it overrides request metadata on a per-key basis.

This enables third-party packages to include UI context (descriptions, policy reasons, worker name, etc.) without core types.

**Recommended (non-binding) metadata keys:**
- `approval_description`: human-readable description of the action
- `approval_reason`: why approval is needed
- `approval_policy`: e.g., "blocked" to indicate policy-blocked without prompting
- `toolset_id`, `worker`: for multi-agent UI clarity

Handlers can use these hints to decide whether to prompt the user or auto-deny.

**Design note: Supporting richer approval systems**

The metadata pass-through is intentionally flexible to support more sophisticated approval models. For example, a capability-based approval system could attach:

```python
raise ApprovalRequired(metadata={
    "required_capabilities": ["fs.write", "net.external"],
    "missing_capabilities": ["net.external"],
    "isolation_profile": "unisolated",
    "tool_name": name,
})
```

The handler then interprets these capabilities to:
- Prompt for missing capability grants
- Manage grant lifetime (per-run, inheritable to child workers)
- Apply isolation profile rules (e.g., auto-grant in isolated environments)
- Cache grants scoped to stable profiles

If the user grants the missing capability, the handler can attach it in `DeferredToolResults.metadata`, for example:

```python
results.metadata[call.tool_call_id] = {"granted_capabilities": ["net.external"]}
```

This keeps the core minimal (just metadata pass-through) while enabling rich policy logic in third-party packages.

### Appendix: Suggested metadata conventions

| Policy state | Suggested metadata | Notes |
|-------------|--------------------|-------|
| `pre_approved` | _Not deferred_ | The call executes immediately and does not appear in `DeferredToolRequests`. |
| `needs_approval` | `approval_policy: "needs_approval"` (or omit `approval_policy`) | Default prompt behavior. |
| `blocked` | `approval_policy: "blocked"` | Handler should auto-deny without prompting. |

### 3) Blocked vs Denied Semantics

Core does not define a new "blocked" result type. Third-party handlers may implement "blocked" by auto-returning a `ToolDenied(message="Blocked: ...")` based on metadata.

This keeps core minimal while allowing richer UX in external packages.

## Examples

### CLI with Blocking Approval (Third-Party Policy)

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import (
    DeferredToolRequests,
    DeferredToolResults,
    ToolApproved,
    ToolDenied,
)

def cli_handler(ctx: RunContext[MyDeps], requests: DeferredToolRequests) -> DeferredToolResults:
    results = DeferredToolResults()
    for call in requests.approvals:
        meta = requests.metadata.get(call.tool_call_id, {})
        if meta.get("approval_policy") == "blocked":
            results.approvals[call.tool_call_id] = ToolDenied(message="Blocked by policy")
            continue

        desc = meta.get("approval_description") or f"{call.tool_name}({call.args})"
        response = input(f"Approve {desc}? [y/n]: ")
        if response.lower() == "y":
            results.approvals[call.tool_call_id] = ToolApproved()
        else:
            results.approvals[call.tool_call_id] = ToolDenied(message="User denied")
    # External calls must also be handled in v1
    for call in requests.calls:
        # This string becomes the tool result seen by the model
        results.calls[call.tool_call_id] = "External call not handled by CLI handler"
    return results

agent = Agent('openai:gpt-4o', output_type=str)
result = agent.run_sync("Clean up logs", deferred_tool_handler=cli_handler)
```

### Auto-Approve for Tests (possible extension)

```python
def auto_approve_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
    results = DeferredToolResults()
    for call in requests.approvals:
        results.approvals[call.tool_call_id] = ToolApproved()
    for call in requests.calls:
        # This string becomes the tool result seen by the model
        results.calls[call.tool_call_id] = "External call not handled by auto-approve handler"
    return results

result = agent.run_sync(prompt, deferred_tool_handler=auto_approve_all)
```

### Hybrid Flow (Not Supported in v1)

v1 requires handlers to return results for **all** deferred tool calls in a response. If you need hybrid flows (inline approvals for some calls, external review for others), use the existing deferred-tools resume pattern instead.

### Fail-Closed Defaults (Handler-Side)

Since the handler must return results for all deferred calls, a "fail closed" policy can fill in defaults for any requests it doesn't explicitly handle:

```python
from pydantic_ai.tools import ToolDenied

def fail_closed_handler(ctx: RunContext[MyDeps], requests: DeferredToolRequests) -> DeferredToolResults:
    results = DeferredToolResults()

    # Approvals: deny anything not explicitly approved
    for call in requests.approvals:
        results.approvals.setdefault(call.tool_call_id, ToolDenied("Denied by policy"))

    # External calls: synthesize an error result if you don't execute them
    for call in requests.calls:
        results.calls.setdefault(call.tool_call_id, "External call not handled")

    return results
```

## Implementation Notes

### Internal Loop Change

```python
# Proposed: handler resolves deferrals inline
async def run(..., deferred_tool_handler=None):
    while True:
        response = await model.call(messages)

        # Phase 1: execute non-deferred tools and collect any deferrals
        deferred = collect_predeferred(response)
        execute_non_deferred_tools(response, deferred)  # may add CallDeferred/ApprovalRequired

        if deferred.has_pending:
            if deferred_tool_handler:
                results = await resolve_maybe_async(deferred_tool_handler(ctx, deferred))
                # Validate the handler provided results for all deferred calls
                validate_results_complete(deferred, results)  # Raise UserError on missing entries
                apply_results(results, response)
                execute_deferred_tools(response, results)
            else:
                return DeferredToolRequests(...)
        ...
```

### Testing (when implemented)

- Missing handler results raise `UserError`.
- `DeferredToolResults.approvals` supports bools and `ToolApproved`/`ToolDenied`.
- `ToolDenied.message` is returned to the model verbatim.
- No implicit pass-through: only `DeferredToolResults.metadata` is available to tools.

### Compatibility

- Default `deferred_tool_handler=None` preserves current behavior.
- Existing code using `DeferredToolRequests` output type continues to work.
- No breaking changes.

## Comparison with Alternatives

### External Loop Helper

An alternative is a helper function that wraps the loop externally:

```python
async def run_with_deferred(agent, prompt, handler):
    result = await agent.run(prompt)
    while isinstance(result.output, DeferredToolRequests):
        results = await handler(result.output)
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=results,
        )
    return result
```

Downsides:
- Each iteration is a separate `run()` call
- Message history management is caller's responsibility
- Streaming behavior may differ across iterations
- Doesn't compose well with other run options

### Per-Tool Handler

```python
approval_handler: Callable[[ToolCallPart], ToolApproved | ToolDenied]
```

Downsides:
- Can't show batch context ("model wants to do A, B, C")
- Can't make batch decisions ("approve all reads")
- Prevents parallel execution of approved tools
- Doesn't handle external tools pattern

## Related Issues

- [#3274](https://github.com/pydantic/pydantic-ai/issues/3274) - Human in the Loop Approval for Multi Agent Systems
- [#3488](https://github.com/pydantic/pydantic-ai/issues/3488) - Allow `user_prompt` with deferred tool approval

## Use Cases

| Use Case | Pattern |
|---------|---------|
| CLI coding assistant | Blocking approval with stdin prompts |
| IDE extension | Blocking approval with UI dialogs |
| Slack/Discord bot | Async handler awaiting user reaction |
| CI/CD pipeline | Auto-deny or auto-approve based on policy |
| Testing | Auto-approve handler |
| Web app with dashboard | Async handler with external review UI |
| Multi-agent orchestration | Handler that manages nested agent approvals |

## Conclusion

This proposal enables blocking, inline approval workflows while keeping the core minimal. Key design decisions:

1. **Same types, different timing**: The handler produces `DeferredToolResults`—the same type used in the existing offline flow—but inline during the run.
2. **Transparent to LLM**: From the LLM's perspective, approved tools behave like normal tools (real results), denied tools behave like failed tools (error message). No "pending review" limbo state.
3. **No partial handling in v1**: The handler must resolve all deferred calls in a response; missing results raise `UserError`.
4. **Consistent API**: Uses `RunContext[Deps]` like other PydanticAI patterns, works uniformly across `run()`, `run_sync()`, `run_stream()`, `run_stream_sync()`, and `iter()`.


### References
