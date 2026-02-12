# ADR 0001: Deferred Tool Handler v1

Date: 2026-02-03
Status: Proposed

## Context

PydanticAI's deferred tools pattern requires callers to manage an external loop: call [`Agent.run()`][pydantic_ai.agent.Agent.run], check for [`DeferredToolRequests`][pydantic_ai.tools.DeferredToolRequests], gather approvals, then call [`Agent.run()`][pydantic_ai.agent.Agent.run] again with [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]. This works well for asynchronous review workflows but creates friction for interactive scenarios where inline blocking approval is natural.

## Decision Drivers

- CLI agents need blocking approval via stdin prompts
- IDE extensions need blocking approval via UI dialogs
- Multi-agent orchestration needs nested approval handling
- The two-run pattern forces callers to re-implement the agent loop
- Message history management across runs is error-prone

## Considered Alternatives

### 1) Subclass Agent and override a handler method

```python
class MyAgent(Agent):
    def handle_deferred_tools(self, requests: DeferredToolRequests) -> DeferredToolResults:
        return my_approval_logic(requests)
```

**Rejected:** Per-run overrides require a subclass per use case. Composition and runtime swapping are clumsy. Does not align with PydanticAI's functional/compositional style.

### 2) Handler object with a `handle()` method

```python
class ApprovalHandler(Protocol):
    def handle(self, requests: DeferredToolRequests) -> DeferredToolResults: ...

agent = Agent('openai:gpt-5', deferred_tool_handler=MyHandler())
```

**Rejected:** Adds ceremony without benefit—Python's callable protocol already supports stateful handlers via `__call__`. Plain functions are simpler for most cases.

### 3) Hybrid method that delegates to a handler

```python
class Agent:
    def handle_deferred_tools(self, requests):
        if self._handler:
            return self._handler(requests)
        raise NotImplementedError
```

**Rejected:** Mixes OO override patterns with delegation, creating ambiguity about which extension point to use.

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

**Rejected:** Each iteration incurs separate [`Agent.run()`][pydantic_ai.agent.Agent.run] overhead. Callers must manage message history. Streaming behavior differs across iterations. Does not compose well with other run options. Users would copy-paste this pattern.

### 5) Per-tool handler callback

```python
approval_handler: Callable[[ToolCallPart], ToolApproved | ToolDenied]

agent = Agent('openai:gpt-5', approval_handler=per_tool_handler)
```

**Rejected:** Loses batch context ("model wants A, B, and C"). Cannot make batch decisions ("approve all reads, deny all writes"). Forces sequential approval UI. Discourages parallel execution. Does not handle external tools.

## Decision

Adopt a **per-model-response callable handler** passed as `deferred_tool_handler` on [`Agent`][pydantic_ai.agent.Agent] and all run methods ([`run()`][pydantic_ai.agent.Agent.run], [`run_sync()`][pydantic_ai.agent.Agent.run_sync], [`run_stream()`][pydantic_ai.agent.Agent.run_stream], [`run_stream_sync()`][pydantic_ai.agent.Agent.run_stream_sync], [`iter()`][pydantic_ai.agent.AbstractAgent.iter]).

The handler receives all deferred tool calls from a single model response as a batch, enabling:

- Full context for approval decisions
- Batch decisions ("approve all reads, deny all writes")
- Parallel execution of approved tools

Handlers can be plain functions or objects implementing `__call__`.

### v1 Constraint: No Partial Handling

For v1, **partial handling is not supported**. When a handler is provided:

- The handler **must** return results for **all** deferred tool calls in the response
- Missing results raise [`UserError`][pydantic_ai.exceptions.UserError]
- Results flow through the existing tool-call execution pipeline unchanged

This keeps validation identical to the existing "resume with [`DeferredToolResults`][pydantic_ai.tools.DeferredToolResults]" flow.

## Consequences

### Benefits

- A single [`Agent.run()`][pydantic_ai.agent.Agent.run] call handles an entire conversation with approvals
- No external loop or message history management needed
- Natural fit for CLI, IDE, and interactive scenarios
- Backward compatible—default `None` preserves existing behavior

### Trade-offs

- **Streaming semantics unchanged:** [`run_stream()`][pydantic_ai.agent.Agent.run_stream] / [`run_stream_sync()`][pydantic_ai.agent.Agent.run_stream_sync] do not pause/resume mid-stream. The handler is invoked only when the run proceeds to tool execution (before final output is streamed). Users needing streaming with guaranteed tool execution can use [`run()`][pydantic_ai.agent.Agent.run] with event streaming or [`iter()`][pydantic_ai.agent.AbstractAgent.iter].

- **No hybrid flows:** Inline approvals and external review cannot be mixed in the same run (v1 limitation).

- **No automatic metadata pass-through:** Tool execution receives only `DeferredToolResults.metadata`. Handlers must explicitly copy request metadata if tools need it. This keeps behavior consistent with the existing resume flow.

- **Handler errors** during the agent loop require clear error propagation.

- **Wrapper agents and durable-exec adapters** should forward the handler parameter to maintain API consistency.

### Deferral Discovery

Some deferrals are discovered at execution time (via [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or [`CallDeferred`][pydantic_ai.exceptions.CallDeferred]) rather than being pre-marked. Inline handling accommodates this with a two-pass process per model response: execute tools, collect deferrals, invoke handler, execute approved deferred calls.

## Execution Semantics

- **Parallel vs sequential:** Unchanged. Use existing tool settings like [`Tool.sequential`][pydantic_ai.tools.Tool].
- **Async execution:** No separate "async mode". Async tools run on the event loop; sync tools run in a thread. The handler sees a single batch per model response.
- **Conditional approval:** Raise [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or [`CallDeferred`][pydantic_ai.exceptions.CallDeferred] at runtime when a tool decides approval is necessary.
- **Unconditional approval:** Use `requires_approval=True` to pre-mark calls as deferred.
- **Invalid arguments:** Prefer raising [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] or returning a normal result over prompting for approval.

## Related Issues

- [#3959](https://github.com/pydantic/pydantic-ai/issues/3959) — Deferred tool handler
- [#3274](https://github.com/pydantic/pydantic-ai/issues/3274) — Human-in-the-loop approval for multi-agent systems
- [#3488](https://github.com/pydantic/pydantic-ai/issues/3488) — Allow `user_prompt` with deferred tool approval

## See Also

- [Deferred Tools](../deferred-tools.md) — User documentation with usage examples
- [Function Tools](../tools.md) — Tool concepts and registration
- [Advanced Tool Features](../tools-advanced.md) — Custom schemas, dynamic tools, execution details
- [Toolsets](../toolsets.md) — Managing tool collections, including `ExternalToolset`
- [Message History](../message-history.md) — Working with message history for deferred tools
