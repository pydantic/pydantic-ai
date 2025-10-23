# Idempotency in Pydantic AI

## Summary

**Pydantic AI does not have built-in idempotency support** for preventing duplicate processing when a client retries a request. Developers need to implement their own idempotency mechanisms at the application level.

## What Pydantic AI Provides

### 1. No Request-Level Idempotency Keys

The `Agent.run()`, `Agent.run_sync()`, and `Agent.run_stream()` methods do not accept:
- Request IDs
- Idempotency keys/tokens
- Client request identifiers
- Deduplication parameters

Each call is treated as a fresh, independent request to the LLM.

**Source**: `pydantic_ai_slim/pydantic_ai/agent/abstract.py:161-211`

The run method signature:
```python
async def run(
    self,
    user_prompt: str | Sequence[_messages.UserContent] | None = None,
    *,
    output_type: OutputSpec[RunOutputDataT] | None = None,
    message_history: Sequence[_messages.ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    model: models.Model | models.KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: _usage.UsageLimits | None = None,
    usage: _usage.RunUsage | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
) -> AgentRunResult[Any]:
```

No idempotency-related parameters are present.

### 2. HTTP Request Retries (Not Idempotency)

The `pydantic_ai.retries` module provides HTTP-level retry functionality for **transient failures** like rate limits or network errors. This is **NOT** the same as idempotency.

**Source**: `pydantic_ai_slim/pydantic_ai/retries.py`

Key differences:
- **Retries**: Automatically retry a *failed* request (network timeout, 429 rate limit, etc.)
- **Idempotency**: Prevent *duplicate execution* when a client thinks a request failed but it actually succeeded

The retry transports (`TenacityTransport`, `AsyncTenacityTransport`) handle:
- Network failures and timeouts
- HTTP 429 (rate limiting) with `Retry-After` header support
- Server errors (5xx)
- Exponential backoff strategies

But they do **not** prevent duplicate processing if:
1. Server processes request successfully
2. Network fails while sending response back
3. Client times out and retries
4. Server processes the same request again

### 3. Durable Execution (Partial Idempotency)

Pydantic AI integrates with **DBOS** and **Temporal** for durable execution, which provides *workflow-level* idempotency, but not *request-level* idempotency.

#### DBOS Approach

**Source**: `docs/durable_execution/dbos.md`

- Wraps `Agent.run()` as a DBOS workflow
- Checkpoints workflow state and step outputs to a database (Postgres or SQLite)
- If a workflow crashes, it resumes from the last completed step
- Model requests and tool calls run as DBOS steps

**Idempotency characteristics**:
- **Workflow recovery**: If the process crashes mid-execution, DBOS will resume from the last checkpoint
- **Step idempotency**: DBOS can retry failed steps, but doesn't prevent duplicate workflow execution
- **No built-in request deduplication**: Multiple calls with the same input will create multiple workflows

**Key limitation**: DBOS doesn't prevent a client from submitting the same request twice. You'd need to build your own workflow ID generation based on request content or client-provided idempotency keys.

#### Temporal Approach

**Source**: `docs/durable_execution/temporal.md`

- Wraps `Agent.run()` as a Temporal workflow
- Uses replay mechanism to recover from failures
- Model requests, tool calls, and MCP communication run as activities
- Workflow state is tracked by Temporal Server

**Idempotency characteristics**:
- **Workflow IDs**: Temporal uses workflow IDs to track execution
- **At-most-once execution**: Temporal guarantees a workflow with a given ID executes at most once
- **Resumption on failure**: If a workflow is interrupted, it resumes from the last recorded state

**Potential for idempotency**: You could implement request-level idempotency by:
```python
# Example (not in Pydantic AI by default):
workflow_id = f"agent-run-{hash(user_prompt)}-{client_request_id}"
output = await client.execute_workflow(
    GeographyWorkflow.run,
    args=[prompt],
    id=workflow_id,  # Temporal will reject duplicate workflow IDs
    task_queue='geography',
)
```

However, this is **not built into Pydantic AI** - you must implement it yourself.

## What Developers Must Build

For true request-level idempotency (preventing duplicate processing of retried requests), developers need to implement:

### 1. Request Identifier Generation
```python
# Client-side or server-side
idempotency_key = str(uuid.uuid4())  # or hash-based on content
```

### 2. Request Deduplication Store
- Cache/database to track processed requests
- Store mapping: `idempotency_key -> result`
- TTL for cleanup (e.g., 24 hours)

Options:
- Redis: `SET idempotency:{key} {result} EX 86400 NX`
- Postgres: Table with unique constraint on idempotency key
- In-memory cache (not recommended for production)

### 3. Application-Level Check Before Running Agent
```python
async def idempotent_run(agent, prompt, idempotency_key):
    # Check if we've already processed this request
    cached_result = await cache.get(f"idempotency:{idempotency_key}")
    if cached_result:
        return cached_result

    # Lock to prevent concurrent duplicate processing
    async with cache.lock(f"idempotency:{idempotency_key}:lock", timeout=30):
        # Double-check after acquiring lock
        cached_result = await cache.get(f"idempotency:{idempotency_key}")
        if cached_result:
            return cached_result

        # Run the agent
        result = await agent.run(prompt)

        # Cache the result
        await cache.set(
            f"idempotency:{idempotency_key}",
            result.model_dump_json(),
            ex=86400  # 24 hour TTL
        )

        return result
```

### 4. Integration with Durable Execution (Optional)

If using Temporal:
```python
# Generate deterministic workflow ID from idempotency key
workflow_id = f"agent-{agent.name}-{idempotency_key}"

output = await client.execute_workflow(
    MyAgentWorkflow.run,
    args=[prompt],
    id=workflow_id,  # Temporal ensures at-most-once execution
    task_queue='my-queue',
)
```

If using DBOS:
```python
# You'd need to implement similar logic with DBOS workflow IDs
# DBOS doesn't have built-in deduplication by default
```

## Recommendations

1. **For simple use cases**: Implement application-level idempotency with Redis/Postgres
2. **For durable workflows**: Use Temporal with deterministic workflow IDs
3. **For HTTP retries**: Use `AsyncTenacityTransport` for transient failures
4. **For production systems**: Combine all three:
   - HTTP retries for network resilience
   - Idempotency keys for request deduplication
   - Durable execution for workflow reliability

## Related Code References

- Agent run interface: `pydantic_ai_slim/pydantic_ai/agent/abstract.py:161`
- HTTP retries: `pydantic_ai_slim/pydantic_ai/retries.py`
- DBOS integration: `pydantic_ai_slim/pydantic_ai/durable_exec/dbos/`
- Temporal integration: `pydantic_ai_slim/pydantic_ai/durable_exec/temporal/`
- DBOS documentation: `docs/durable_execution/dbos.md`
- Temporal documentation: `docs/durable_execution/temporal.md`

## Conclusion

**Idempotency is the developer's responsibility.** Pydantic AI provides building blocks (HTTP retries, durable execution) but does not handle request-level idempotency out of the box. You must implement your own idempotency key system and deduplication logic at the application layer.
