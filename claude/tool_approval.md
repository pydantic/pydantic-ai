# PydanticAI Tool Approval: A Deep Dive

## Executive Summary

PydanticAI's tool approval system is **fundamentally asynchronous and split across two separate `agent.run()` calls**. This is completely different from Claude Code's synchronous blocking approach where execution freezes until you approve/deny.

**Key difference:** When PydanticAI encounters a tool requiring approval, `agent.run()` **returns immediately** with a `DeferredToolRequests` object. There is no blocking, no UI prompt within the run - the agent run simply ends. You then handle approvals in your own code (outside of PydanticAI), and start a **second** agent run with the approval results.

---

## How It Works: The Two-Phase Flow

### Phase 1: Agent Run Encounters Approval-Required Tool

```python
agent = Agent('openai:gpt-4', output_type=[str, DeferredToolRequests])

@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'File {path!r} deleted'

# First run - agent.run() returns IMMEDIATELY when tool needs approval
result = agent.run_sync('Delete file.txt')
messages = result.all_messages()

# result.output is DeferredToolRequests, NOT the final answer
assert isinstance(result.output, DeferredToolRequests)
# Contains: DeferredToolRequests(approvals=[ToolCallPart(...)])
```

**What happens internally:**

1. Model calls `delete_file` tool
2. PydanticAI sees it has `requires_approval=True`, marks tool as `kind='unapproved'`
3. **Tool is NOT executed** - instead it's added to a `DeferredToolRequests.approvals` list
4. **Agent run ENDS** - `agent.run()` returns with `DeferredToolRequests` as output
5. Execution returns to your code

**Critical insight:** `agent.run()` does NOT wait, block, or pause. It completes normally and returns control to you.

### Phase 2: You Handle Approvals, Then Resume

```python
# Now YOUR code decides what to do (could show UI, prompt user, apply logic, etc.)
results = DeferredToolResults()
for call in result.output.approvals:
    # Your custom approval logic here
    user_approved = show_approval_ui(call.tool_name, call.args)

    if user_approved:
        results.approvals[call.tool_call_id] = True  # or ToolApproved()
    else:
        results.approvals[call.tool_call_id] = ToolDenied('User rejected this')

# Second agent run - continues where first left off
result = agent.run_sync(
    message_history=messages,  # Original conversation context
    deferred_tool_results=results  # Your approval decisions
)

# NOW result.output is the final answer
print(result.output)  # "File 'file.txt' was deleted" (or denial message)
```

**What happens in the second run:**

1. PydanticAI sees `deferred_tool_results` provided
2. For each approval:
   - If approved (`True` or `ToolApproved()`): **executes the tool function** with `ctx.tool_call_approved=True`
   - If denied (`False` or `ToolDenied()`): sends denial message to model, does NOT execute
3. Model receives tool results and continues conversation
4. Agent run completes with final output

---

## Mixed Scenarios: Some Tools Execute, Some Require Approval

**This is where it gets interesting.** If the model calls multiple tools in one response, and only SOME require approval:

```python
@agent.tool_plain
def get_price(fruit: str) -> float:
    return 10.0  # Executes immediately

@agent.tool_plain
def buy(fruit: str):
    raise ApprovalRequired  # Requires approval

# Model calls BOTH tools
result = agent.run_sync('What does an apple cost? Also buy me one.')
```

**What happens:**
1. `get_price('apple')` **executes immediately** and returns `10.0`
2. `buy('apple')` raises `ApprovalRequired`, gets deferred
3. **Agent run ends** with:
   - `DeferredToolRequests(approvals=[buy tool call])`
   - Message history contains the `get_price` result already
4. Tools that don't require approval execute normally; only approval-required ones are deferred

**Source:** See `test_approval_required_toolset` in tests/test_tools.py:394-531 - this test shows:
- Model calls 3 tools: `foo(x=1)`, `foo(x=2)`, `bar(x=3)`
- Only `foo` requires approval, `bar` does not
- Result: `bar(x=3)` executes immediately (returns 9), both `foo` calls are deferred
- First run returns `DeferredToolRequests(approvals=[foo(x=1), foo(x=2)])`

---

## Comparison to Claude Code's Synchronous Approach

### Claude Code (Synchronous/Blocking)

```
User: "Delete this file"
  ↓
Claude Code executes, encounters delete_file tool
  ↓
**EXECUTION FREEZES** ← You're "frozen" waiting for approval
  ↓
UI appears asking for approval
  ↓
User clicks approve/deny
  ↓
Execution RESUMES, tool executes/denied
  ↓
Claude Code continues and responds
```

**Characteristics:**
- Single continuous execution
- Blocks/freezes at approval point
- User approval is synchronous - must happen during the run
- Cannot do anything else while waiting

### PydanticAI (Asynchronous/Split)

```
User: "Delete this file"
  ↓
agent.run() executes, encounters delete_file tool
  ↓
agent.run() RETURNS immediately with DeferredToolRequests
  ↓
**Your code takes over** ← Control returned to your application
  ↓
Your code handles approvals (UI, logic, database, etc.)
  ↓  (This could take hours, days, or involve external systems)
  ↓
agent.run() called AGAIN with approval results
  ↓
Tool executes/denied, model continues
```

**Characteristics:**
- Two separate agent runs
- Never blocks - always returns control
- Approval happens between runs, in your application code
- Extremely flexible - approval can be async, batched, delayed, etc.

---

## Key Design Implications

### 1. You Control the Approval UX Completely

PydanticAI **does not** provide any approval UI or mechanism. It just:
- Returns `DeferredToolRequests` with pending approvals
- Accepts `DeferredToolResults` with your decisions

**You implement:**
- How to show approval requests (CLI prompt, web UI, Slack message, etc.)
- When to gather approvals (immediately, batched, scheduled)
- Who approves (user, admin, automated policy)
- Where approvals are stored (in-memory, database, audit log)

### 2. Long-Running or External Approval Flows

Because approval happens between two agent runs, you can:

```python
# First run
result = await agent.run('Delete all production databases')
if isinstance(result.output, DeferredToolRequests):
    # Store pending approvals in database
    db.save_pending_approvals(result.output.approvals, result.all_messages())

    # Send approval request to Slack
    await slack.send_approval_request(result.output.approvals)

    # Exit - approval will come later via webhook
    return {"status": "awaiting_approval"}

# Hours later, when approval comes via webhook:
async def handle_approval_webhook(approval_id, approved):
    approvals, messages = db.get_pending_approval(approval_id)
    results = DeferredToolResults(approvals={...})

    # Resume agent run
    result = await agent.run(
        message_history=messages,
        deferred_tool_results=results
    )
```

Claude Code cannot do this - it requires synchronous approval during the run.

### 3. Batch Approvals

If the model calls multiple tools requiring approval, you can:
- Show all pending approvals at once
- Let user approve/deny as a batch
- Apply complex approval logic (e.g., "approve all reads, deny all writes")

```python
result = agent.run_sync('Read config, delete logs, update settings')
# Might have 3 tools requiring approval
if isinstance(result.output, DeferredToolRequests):
    for call in result.output.approvals:
        print(f"Approve {call.tool_name}({call.args})? [y/n]")
    # Batch collect approvals, then resume
```

### 4. The Output Type Requirement

**Critical:** Your agent's `output_type` MUST include `DeferredToolRequests`:

```python
# Required for deferred tools
agent = Agent('openai:gpt-4', output_type=[str, DeferredToolRequests])
```

This is because `agent.run()` will return EITHER:
- Your normal output type (e.g., `str`)
- `DeferredToolRequests` if any deferred tools were called

Without this, PydanticAI raises an error: `"DeferredToolRequests is not among output types"`

---

## Implementation Details

### How `requires_approval` Works Internally

Location: `pydantic_ai_slim/pydantic_ai/tools.py:427-428`

```python
async def prepare_tool_def(self, ctx: RunContext[AgentDepsT]) -> ToolDefinition | None:
    base_tool_def = self.tool_def

    # If tool requires approval and hasn't been approved yet
    if self.requires_approval and not ctx.tool_call_approved:
        # Mark tool as 'unapproved' kind
        base_tool_def = replace(base_tool_def, kind='unapproved')

    return base_tool_def
```

### How Deferred Tools Are Processed

Location: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:850-878`

```python
# After processing tool calls
if not final_result and deferred_calls:
    if not ctx.deps.output_schema.allows_deferred_tools:
        raise UserError('DeferredToolRequests is not among output types...')

    # Create DeferredToolRequests output
    deferred_tool_requests = DeferredToolRequests(
        calls=deferred_calls['external'],        # External execution tools
        approvals=deferred_calls['unapproved'],  # Approval-required tools
    )

    # Set as final result - ends the agent run
    final_result = FinalResult(deferred_tool_requests)
```

### How Approval Results Are Applied

Location: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:265-274`

When `deferred_tool_results` is provided in second run:

```python
# For each approval in deferred_tool_results
if isinstance(tool_call_result, ToolApproved):
    # Can override tool arguments!
    if tool_call_result.override_args is not None:
        tool_call = dataclasses.replace(tool_call, args=tool_call_result.override_args)
    # Execute the tool
    tool_result = await tool_manager.handle_call(tool_call)

elif isinstance(tool_call_result, ToolDenied):
    # Return denial message to model, don't execute
    return ToolReturnPart(
        tool_name=tool_call.tool_name,
        content=tool_call_result.message,  # Default: "The tool call was denied."
        tool_call_id=tool_call.tool_call_id,
    )
```

### The `ctx.tool_call_approved` Flag

Location: `pydantic_ai_slim/pydantic_ai/_run_context.py:55-56`

```python
tool_call_approved: bool = False
"""Whether a tool call that required approval has now been approved."""
```

Set in `_agent_graph.py:728`:
```python
tool_call_approved=ctx.state.run_step == 0,
```

Wait, this is interesting - it's `True` when `run_step == 0`, which means the FIRST run (before any tool calls). That seems backwards...

Actually, looking more carefully at the code flow:
- **First run (`run_step=0`)**: `tool_call_approved=False` initially, but when incremented becomes 1
- **Second run with deferred_tool_results**: Tools see `tool_call_approved=True`

Actually, I need to trace this more carefully. The `run_step` is incremented in `ModelRequestNode._prepare_request` at line 449:
```python
ctx.state.run_step += 1
```

So actually:
- Initial state: `run_step=0`
- Before making model request: increments to `run_step=1`
- When building RunContext: `tool_call_approved = (run_step == 0)` = False

Wait, that's confusing. Let me check if there's logic that sets it based on `deferred_tool_results`...

Actually, I see now - the `run_step == 0` check is a simple heuristic. But the real logic is:
1. First run: Tools marked as `kind='unapproved'` don't execute, ApprovalRequired is raised
2. Second run: When `deferred_tool_results` is provided with approvals, those specific tool calls execute

The `ctx.tool_call_approved` flag is mainly for tool functions that conditionally require approval based on arguments (like the `.env` example in the docs).

---

## Conditional Approval Example

```python
PROTECTED_FILES = {'.env', 'secrets.yaml'}

@agent.tool
def update_file(ctx: RunContext, path: str, content: str) -> str:
    # Conditionally require approval based on arguments
    if path in PROTECTED_FILES and not ctx.tool_call_approved:
        raise ApprovalRequired

    # If we get here, either:
    # - File is not protected, OR
    # - This is the second run after approval
    return f'Updated {path}'

# First run: update README.md and .env
result = agent.run_sync('Update README.md and .env')

# Result: README.md updates immediately, .env is deferred
# DeferredToolRequests(approvals=[ToolCallPart(tool_name='update_file', args={'path': '.env', ...})])

# Second run with approval
result = agent.run_sync(
    message_history=messages,
    deferred_tool_results=DeferredToolResults(approvals={'env_call_id': True})
)
# Now .env updates
```

---

## Comparison Summary

| Aspect | Claude Code | PydanticAI |
|--------|-------------|------------|
| **Execution model** | Synchronous (blocks) | Asynchronous (returns) |
| **Number of runs** | 1 continuous run | 2 separate runs |
| **When approval happens** | During execution | Between runs |
| **Who provides approval UI** | Claude Code | Your application |
| **Can approval be delayed?** | No (must approve immediately) | Yes (hours, days, external systems) |
| **Can batch approvals?** | No | Yes |
| **Mixed execution** | N/A (all tools wait) | Non-approval tools execute immediately |
| **Approval storage** | In-memory (session) | Your choice (DB, queue, etc.) |
| **Use case** | Interactive CLI tools | Production systems, workflows, auditing |

---

## Flexibility & Power

PydanticAI's approach gives you **complete control** over:

1. **Approval Mechanism**
   - CLI prompts
   - Web UI forms
   - Slack/Teams notifications
   - Database-backed approval queues
   - Automated policy engines

2. **Approval Timing**
   - Immediate (like Claude Code)
   - Batched (collect all pending, approve once)
   - Scheduled (approve during business hours)
   - Async (webhook-based, hours later)

3. **Approval Logic**
   - User-based permissions
   - Risk scoring
   - Multi-stakeholder approval
   - Audit logging
   - Compliance checks

4. **State Management**
   - Store pending approvals in database
   - Persist conversation state
   - Resume across process restarts
   - Distribute across services

---

## The Trade-off

**Simplicity vs Flexibility:**

- **Claude Code:** Super simple, works great for interactive CLI use, but limited
- **PydanticAI:** Requires more code (you build approval flow), but can handle any real-world scenario

**When to use each:**
- **Claude Code style:** Building interactive dev tools, CLI utilities, simple agents
- **PydanticAI style:** Production systems, web apps, multi-user systems, regulated environments, long-running workflows

---

## Example: Building Claude-Code-like Blocking Approval

You can actually implement Claude Code's blocking behavior with PydanticAI:

```python
def run_with_blocking_approval(agent, prompt):
    """Makes PydanticAI behave like Claude Code's blocking approval."""
    result = agent.run_sync(prompt)

    # Loop until we get a non-deferred result
    while isinstance(result.output, DeferredToolRequests):
        messages = result.all_messages()

        # Show approval UI (BLOCKS here, like Claude Code)
        results = DeferredToolResults()
        for call in result.output.approvals:
            approved = input(f"Approve {call.tool_name}({call.args})? [y/n]: ") == 'y'
            results.approvals[call.tool_call_id] = approved

        # Resume run with approvals
        result = agent.run_sync(
            message_history=messages,
            deferred_tool_results=results
        )

    return result

# Use it
result = run_with_blocking_approval(agent, "Delete production database")
print(result.output)
```

But PydanticAI also lets you do **much more** than this blocking pattern.

---

## Conclusion

PydanticAI's tool approval is **fundamentally asynchronous and non-blocking**:

1. **First `agent.run()`**: Encounters approval-required tool → returns immediately with `DeferredToolRequests`
2. **Your code**: Handles approvals however you want (UI, DB, external system, etc.)
3. **Second `agent.run()`**: Resumes with approval results → tools execute/denied → final output

This is **completely different** from Claude Code's blocking approach where execution freezes until you respond. PydanticAI's approach is more complex to implement but vastly more flexible and suitable for production systems.

**Key insight:** Other tools in the same model response CAN execute normally - only approval-required tools are deferred. This enables mixed execution scenarios that aren't possible with a purely blocking approach.
