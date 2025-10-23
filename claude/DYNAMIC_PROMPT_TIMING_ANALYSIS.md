# Dynamic Prompt Re-evaluation Timing: Deep Dive Analysis

## TL;DR: The Critical Finding

**Dynamic prompts are NOT re-evaluated mid-run after tool execution.**

They are ONLY re-evaluated when:
1. Starting a new `agent.run()` call
2. With `message_history` from a previous run

This means for agent switching, you **must** use a two-run pattern.

---

## Execution Flow Analysis

### The Graph Node Cycle

The agent execution graph has three node types:

```
UserPromptNode → ModelRequestNode → CallToolsNode
      ↑                                   ↓
      └────── (only on new run) ──────────┘
                                           ↓
                              ModelRequestNode (after tools)
                                           ↓
                              CallToolsNode (final result)
                                           ↓
                                         End
```

### Where Dynamic Prompts ARE Evaluated

**Location**: `UserPromptNode.run()` - Lines 233-237

```python
# Build the run context after `ctx.deps.prompt` has been updated
run_context = build_run_context(ctx)

if messages:
    await self._reevaluate_dynamic_prompts(messages, run_context)

if next_message:
    await self._reevaluate_dynamic_prompts([next_message], run_context)
```

This ONLY happens in `UserPromptNode`, which is the **first node** in the graph.

### After Tool Execution: What Happens

**Location**: `CallToolsNode._handle_tool_calls()` - Lines 492-496

After tools execute:

```python
else:
    instructions = await ctx.deps.get_instructions(run_context)
    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
        _messages.ModelRequest(parts=output_parts, instructions=instructions)
    )
```

Notice: It creates a **ModelRequestNode**, NOT a UserPromptNode!

### ModelRequestNode Does NOT Re-evaluate Dynamic Prompts

**Location**: `ModelRequestNode._prepare_request()` - Lines 268-306

When the ModelRequestNode prepares a request:

```python
ctx.state.message_history.append(self.request)

ctx.state.run_step += 1

run_context = build_run_context(ctx)

# This will raise errors for any tool name conflicts
ctx.deps.tool_manager = await ctx.deps.tool_manager.for_run_step(run_context)

original_history = ctx.state.message_history[:]
message_history = await _process_message_history(original_history, ctx.deps.history_processors, run_context)
# ... continues
```

**Key observation**: No call to `_reevaluate_dynamic_prompts()`!

It processes message history through `HistoryProcessor`s, but does NOT re-evaluate dynamic prompts.

---

## Complete Execution Flow

### Scenario: Single agent.run() with Tool Call

```
agent.run("Switch to Alice")
    │
    ↓
1. UserPromptNode
    │  ├─ _reevaluate_dynamic_prompts(messages) ← HAPPENS HERE (for existing messages)
    │  └─ Returns: ModelRequestNode(request=user_message)
    │
    ↓
2. ModelRequestNode
    │  ├─ Appends request to message_history
    │  ├─ Calls model
    │  └─ Returns: CallToolsNode(model_response)
    │
    ↓
3. CallToolsNode
    │  ├─ Model says: "Sure, switching to Alice" + calls change_agent('alice_v1')
    │  ├─ Tool executes: ctx.deps.current_agent_id = 'alice_v1'
    │  ├─ Collects tool return: "Switched to Alice"
    │  └─ Returns: ModelRequestNode(request=tool_returns) ← NOT UserPromptNode!
    │
    ↓
4. ModelRequestNode (2nd time)
    │  ├─ Appends request to message_history
    │  ├─ NO dynamic prompt re-evaluation ← CRITICAL!
    │  ├─ Calls model with SAME system prompt (Jarvis)
    │  └─ Returns: CallToolsNode(model_response)
    │
    ↓
5. CallToolsNode (2nd time)
    │  ├─ Model responds (still as Jarvis!)
    │  └─ Returns: End(final_result)
```

### Scenario: Separate agent.run() Calls

```
# First run
result1 = agent.run("Switch to Alice", deps=deps)
    │
    ↓
1. UserPromptNode
    │  └─ Dynamic prompt evaluated: returns Jarvis prompt
    ↓
2. ModelRequestNode → Jarvis calls change_agent tool
    ↓
3. CallToolsNode → Tool updates deps.current_agent_id = 'alice_v1'
    ↓
4. ModelRequestNode → Jarvis says "Sure, getting Alice"
    ↓
5. End

# Second run (separate call)
result2 = agent.run("Hello", message_history=result1.all_messages(), deps=deps)
    │
    ↓
1. UserPromptNode
    │  ├─ _reevaluate_dynamic_prompts(messages) ← HAPPENS HERE!
    │  │  └─ Reads deps.current_agent_id = 'alice_v1'
    │  │  └─ Updates SystemPromptPart to Alice's prompt
    │  └─ Returns: ModelRequestNode
    ↓
2. ModelRequestNode → Alice responds!
```

---

## Why This Matters

### For "Immediate Switch" (Same Run)

If you wanted Alice to respond in the same run:

```python
result = agent.run("Switch to Alice, then Alice tell me a joke")
```

❌ **This won't work!** Because:
1. Jarvis calls change_agent tool
2. deps.current_agent_id changes to 'alice_v1'
3. Graph continues to ModelRequestNode (NOT UserPromptNode)
4. Dynamic prompt is NOT re-evaluated
5. Model still has Jarvis's system prompt
6. Response is still from Jarvis's perspective

### For "Clean Handoff" (Separate Runs)

This is what actually works:

```python
# Run 1: Jarvis handles the switch
deps = AgentDeps(current_agent_id='jarvis_v1')
result1 = agent.run("Switch to Alice", deps=deps)
# Jarvis says: "Sure, let me get Alice for you"
# Tool executes: deps.current_agent_id = 'alice_v1'

# Run 2: Alice responds
result2 = agent.run(
    "Tell me a joke",
    message_history=result1.all_messages(),
    deps=deps  # Still has current_agent_id='alice_v1'
)
# Alice responds!
```

✅ **This works!** Because:
1. Run 2 starts with `UserPromptNode`
2. `_reevaluate_dynamic_prompts()` is called
3. Sees deps.current_agent_id = 'alice_v1'
4. Updates system prompt to Alice's
5. Alice responds

---

## Code Evidence

### UserPromptNode is the Graph Entry Point

**Location**: `agent/__init__.py:626-634` (where the graph run starts)

```python
start_node = _agent_graph.UserPromptNode[AgentDepsT](
    user_prompt=user_prompt,
    deferred_tool_results=deferred_tool_results,
    instructions=instructions_literal,
    instructions_functions=instructions_functions,
    system_prompts=self._system_prompts,
    system_prompt_functions=self._system_prompt_functions,
    system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
)
```

The graph always starts with UserPromptNode.

### Graph Flow After Tools

**Location**: `_agent_graph.py:492-496`

After tools execute, the next node is:

```python
self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
    _messages.ModelRequest(parts=output_parts, instructions=instructions)
)
```

NOT `UserPromptNode`! The graph doesn't cycle back to the start.

### _reevaluate_dynamic_prompts Implementation

**Location**: `_agent_graph.py:135-157`

```python
async def _reevaluate_dynamic_prompts(
    self, messages: list[_messages.ModelMessage], run_context: RunContext[DepsT]
) -> None:
    """Reevaluate any `SystemPromptPart` with dynamic_ref in the provided messages."""
    if self.system_prompt_dynamic_functions:
        for msg in messages:
            if isinstance(msg, _messages.ModelRequest):
                reevaluated_message_parts: list[_messages.ModelRequestPart] = []
                for part in msg.parts:
                    if isinstance(part, _messages.SystemPromptPart) and part.dynamic_ref:
                        if runner := self.system_prompt_dynamic_functions.get(part.dynamic_ref):
                            updated_part_content = await runner.run(run_context)
                            part = _messages.SystemPromptPart(updated_part_content, dynamic_ref=part.dynamic_ref)

                    reevaluated_message_parts.append(part)

                if reevaluated_message_parts != msg.parts:
                    msg.parts = reevaluated_message_parts
```

This updates `SystemPromptPart`s in the message history. But it's only called in `UserPromptNode`.

---

## Implications for Agent Switching

### Pattern 1: Two-Run Handoff (Recommended)

```python
@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_id: str) -> str:
    """Switch to a different agent."""
    ctx.deps.current_agent_id = agent_id
    # This change will take effect on the NEXT run
    return f"Switched to {AGENT_REGISTRY[agent_id].name}"

# Usage
deps = AgentDeps(current_agent_id='jarvis_v1')

# Run 1: Jarvis handles switch
result1 = agent.run("Get Alice please", deps=deps)
print(result1.output)
#> "Sure, let me get Alice for you"

# Run 2: Alice takes over
result2 = agent.run(
    "Hello Alice!",
    message_history=result1.all_messages(),
    deps=deps  # Now has current_agent_id='alice_v1'
)
print(result2.output)
#> "Hello! I'm Alice, how can I help?"
```

**Why this works**:
- Run 1 ends with Jarvis's response
- Tool modifies deps.current_agent_id
- Run 2 starts fresh → UserPromptNode → dynamic prompts re-evaluated
- Alice's prompt is loaded
- Alice responds

### Pattern 2: Single-Run (Won't Switch Mid-Run)

```python
# This DOESN'T achieve mid-run switch
result = agent.run(
    "Switch to Alice, then tell me a joke",
    deps=deps
)
# Jarvis calls the tool AND tells the joke (doesn't switch to Alice mid-run)
```

**Why this doesn't switch**:
- Graph doesn't return to UserPromptNode after tool execution
- Dynamic prompts not re-evaluated
- Same system prompt throughout the run

### Pattern 3: Using ToolOutput to Force End

You COULD potentially use a ToolOutput to end the run immediately after the switch:

```python
from pydantic_ai import ToolOutput

switch_tool_output = ToolOutput(str, name='agent_switch_complete')

agent = Agent(
    'openai:gpt-4o',
    deps_type=AgentDeps,
    output_type=[str, switch_tool_output]
)

@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_id: str) -> str:
    """Switch agent - this ends the current run."""
    ctx.deps.current_agent_id = agent_id
    # Returning via output tool ends the run
    return f"Switched to {AGENT_REGISTRY[agent_id].name}"
```

But even this still requires a second run for the new agent to respond!

---

## Conclusion

### The Truth About dynamic=True

The `dynamic=True` flag means:

✅ **"Re-evaluate when UserPromptNode runs"**

Which happens:
- At the start of `agent.run()`
- When processing existing message_history

❌ **NOT "Re-evaluate after every tool call within a run"**

The graph does NOT cycle back to UserPromptNode after tool execution within a single run.

### Recommended Architecture

For agent switching in flutter-agent-api:

```python
# Agent definition
@agent.system_prompt(dynamic=True)
def current_agent_prompt(ctx: RunContext[AgentDeps]) -> str:
    return AGENT_REGISTRY[ctx.deps.current_agent_id].prompt

@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_id: str) -> str:
    """Transfers conversation to another agent."""
    old_name = AGENT_REGISTRY[ctx.deps.current_agent_id].name
    new_name = AGENT_REGISTRY[agent_id].name

    ctx.deps.current_agent_id = agent_id

    # Handoff message that current agent says
    return f"Transferring you to {new_name} now."

# Server endpoint handling
async def handle_message(user_message: str, session: Session):
    deps = AgentDeps(current_agent_id=session.current_agent_id)

    result = await agent.run(
        user_message,
        message_history=session.message_history,
        deps=deps
    )

    # Save updated state
    session.current_agent_id = deps.current_agent_id  # May have changed!
    session.message_history = result.all_messages()

    return result.output
```

The key insight: Each HTTP request = one `agent.run()` call, which starts with UserPromptNode, so dynamic prompts ARE re-evaluated between requests. Perfect for your use case!

---

## Testing to Confirm

You can verify this behavior:

```python
async def test_dynamic_prompt_timing():
    """Verify dynamic prompts don't change mid-run."""

    # Track dynamic prompt evaluations
    calls = []

    @agent.system_prompt(dynamic=True)
    def tracked_prompt(ctx: RunContext[AgentDeps]) -> str:
        agent_id = ctx.deps.current_agent_id
        calls.append(agent_id)
        return AGENT_REGISTRY[agent_id].prompt

    deps = AgentDeps(current_agent_id='jarvis_v1')

    result = await agent.run("Switch to alice_v1", deps=deps)

    # Should have ONE evaluation (at start)
    assert len(calls) == 1
    assert calls[0] == 'jarvis_v1'

    # Even though deps changed
    assert deps.current_agent_id == 'alice_v1'

    # The prompt wasn't re-evaluated during the run
    # (would show up as a second call with 'alice_v1')
```

This test would confirm dynamic prompts are evaluated once per run, not after tool execution.
