# ADR 0001: Deferred Tool Handler v1 (No Partial Handling)

Date: 2026-02-03
Status: Proposed

## Context

PydanticAI's deferred tools pattern currently requires callers to manage an external loop: call [`Agent.run()`][pydantic_ai.agent.Agent.run], check for [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], gather approvals, then call [`Agent.run()`][pydantic_ai.agent.Agent.run] again with [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]. This works well for asynchronous review workflows, but creates friction for interactive scenarios where blocking inline approval is natural.

## Decision Drivers

- CLI agents need blocking approval with stdin prompts.
- IDE extensions need blocking approval with UI dialogs.
- Multi-agent orchestration systems need nested approval handling.
- The two-run pattern requires re-implementing the agent loop externally.
- Message history management across runs is error-prone.

## Considered Alternatives

### 1) Subclass [`Agent`][pydantic_ai.agent.Agent] and override a handler method

```python
class MyAgent(Agent):
    def handle_deferred_tools(self, requests: DeferredToolRequests) -> DeferredToolResults:
        return my_approval_logic(requests)
```

Rejected because:

- Per-run overrides are awkward (would need a subclass per use case).
- Composition is harder; runtime swapping of handlers is clumsy.
- This does not align with PydanticAI's functional/compositional style.

### 2) Handler object with a `handle()` method

```python
class ApprovalHandler(Protocol):
    def handle(self, requests: DeferredToolRequests) -> DeferredToolResults: ...

agent = Agent('openai:gpt-5', deferred_tool_handler=MyHandler())
```

Rejected because:

- Adds ceremony without clear benefit over plain callables.
- Python's callable protocol already supports stateful handlers via classes with `__call__`.
- Plain functions are simpler for most use cases.

### 3) Hybrid method that delegates to a handler

```python
class Agent:
    def handle_deferred_tools(self, requests):
        if self._handler:
            return self._handler(requests)
        raise NotImplementedError
```

Rejected because:

- Mixes OO override patterns with delegation.
- Creates ambiguity about which extension point to use.
- No compelling advantage over pure delegation.

### 4) External loop helper function

```python
async def run_with_approval(agent, prompt, handler):
    result = await agent.run(prompt)
    while isinstance(result.output, DeferredToolRequests):
        results = await handler(result.output)
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=results,
        )
    return result
```

Rejected because:

- Each iteration is a separate [`Agent.run()`][pydantic_ai.agent.Agent.run] call with overhead.
- Message history management is caller's responsibility.
- Streaming behavior differs across iterations.
- It does not compose well with other run options (usage tracking, metadata).
- Users would have to copy-paste this pattern.

### 5) Per-tool handler callback

```python
approval_handler: Callable[[ToolCallPart], ToolApproved | ToolDenied]

agent = Agent('openai:gpt-5', approval_handler=per_tool_handler)
```

Rejected because:

- Loses batch context ("model wants A, B, and C").
- Cannot make batch decisions ("approve all reads, deny all writes").
- Forces approval UI to be sequential.
- Discourages parallel execution of approved tools.
- Does not handle the external tools pattern.

## Decision

Adopt a **per-model-response callable handler** passed as the `deferred_tool_handler` parameter on [`Agent`][pydantic_ai.agent.Agent] and all run variants: [`Agent.run()`][pydantic_ai.agent.Agent.run] / [`Agent.run_sync()`][pydantic_ai.agent.Agent.run_sync] / [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream] / [`Agent.run_stream_sync()`][pydantic_ai.agent.Agent.run_stream_sync] / [`Agent.iter()`][pydantic_ai.agent.AbstractAgent.iter].

The handler receives all deferred tool calls from a single model response as a batch, enabling full context, batch decisions, and parallel execution. Handlers can be plain functions or objects implementing `__call__`.

### v1 constraint: no partial handling

For v1, **partial handling is not supported**.

When a handler is provided and the model response contains deferred tool calls:

- The handler **must** return results for **all** deferred tool calls in that response.
- If any deferred tool call is missing from the returned [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults], the run raises [`UserError`][pydantic_ai.exceptions.UserError].
- The results are applied through the existing tool-call execution pipeline without modification.

This keeps the behavior and validation rules identical to the existing "resume with [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]" flow: tool-call results must be complete and match all deferred calls.

## Consequences

Benefits:

- A single [`Agent.run()`][pydantic_ai.agent.Agent.run] call can handle an entire conversation with approvals.
- No external loop or message history management is needed.
- Natural fit for CLI, IDE, and interactive scenarios.
- Backward compatible; default `None` preserves existing behavior.

Trade-offs:

- [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream] and [`Agent.run_stream_sync()`][pydantic_ai.agent.Agent.run_stream_sync] keep current semantics: they do not pause/resume mid-stream. The handler is invoked only when the run proceeds to tool execution (before any final output has been streamed), so deferred tools in a response that already produced a final output are not handled inline.
- Handler errors during the agent loop require clear error propagation.
- Hybrid flows (inline approvals + external review in the same run) are not supported in v1.
- No automatic metadata pass-through. Tool execution only receives `DeferredToolResults.metadata` (same as the current resume flow). If a handler wants request metadata available during execution, it must explicitly copy it into `DeferredToolResults.metadata`.
- Wrapper agents and durable-exec adapters should forward the handler parameter to keep the public API consistent across environments.

Rationale for metadata handling:

- Capability checks can be enforced entirely in the handler; the tool can rely on `tool_call_approved=True` to avoid re-deferring on the second execution.
- If a tool needs additional execution-time context (e.g., `granted_capabilities`, audit tags), the handler can attach that explicitly in `DeferredToolResults.metadata`.
- This keeps behavior consistent with the existing resume flow and avoids implicit propagation of request metadata into execution.

Motivation for streaming semantics:

- Preserves the existing [`Agent.run_stream()`][pydantic_ai.agent.Agent.run_stream] / [`Agent.run_stream_sync()`][pydantic_ai.agent.Agent.run_stream_sync] contract (stop after first final output) and avoids surprising pauses/resumes mid-stream.
- Keeps streaming behavior consistent with current expectations while still allowing inline approvals when the run continues to tool execution.
- Users who need streaming plus guaranteed tool execution can use [`Agent.run()`][pydantic_ai.agent.Agent.run] with event streaming or [`Agent.iter()`][pydantic_ai.agent.AbstractAgent.iter].

Motivation for deferral discovery:

- Some deferrals are only discovered at tool execution time (via [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or [`CallDeferred`][pydantic_ai.exceptions.CallDeferred]) rather than being pre-marked.
- Preserving this behavior avoids changing tool contracts or forcing tools to predeclare all deferrals.
- Inline handling therefore needs to accommodate deferrals discovered during execution, even if this adds an extra phase internally.
- In practice this implies a two-pass process within a single model response: execute tools, collect deferrals, then invoke the handler and execute approved deferred calls.

## Execution Semantics and Conditional Approval

- Parallel vs sequential execution is unchanged; keep using existing tool settings such as the `Tool` `sequential` flag (see [`Tool`][pydantic_ai.tools.Tool]).
- There is no separate "async mode" for tool execution. Async tools run on the event loop, while sync tools are executed in a thread; the handler sees a single batch of deferred calls per model response.
- Conditional approvals should be expressed by raising [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] at runtime when a tool decides approval is necessary.
- Unconditional approvals should use `requires_approval=True` so calls are pre-marked as deferred.
- If a tool decides approval does not make sense for given arguments (e.g., invalid or unsafe), prefer raising [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] or returning a normal tool result rather than prompting for approval.

## Related Issues

- [#3959](https://github.com/pydantic/pydantic-ai/issues/3959) — This feature
- [#3274](https://github.com/pydantic/pydantic-ai/issues/3274) — Human in the Loop Approval for Multi Agent Systems
- [#3488](https://github.com/pydantic/pydantic-ai/issues/3488) — Allow `user_prompt` with deferred tool approval

## See Also

- [Function Tools](../tools.md) — Basic tool concepts and registration
- [Advanced Tool Features](../tools-advanced.md) — Custom schemas, dynamic tools, and execution details
- [Toolsets](../toolsets.md) — Managing collections of tools, including `ExternalToolset` for external tools
- [Message History](../message-history.md) — Working with message history for deferred tools
