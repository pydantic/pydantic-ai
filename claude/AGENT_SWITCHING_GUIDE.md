# Dynamic Agent Switching with Pydantic AI

## Pydantic AI Full Code + Docs

All can be found at /Users/ericksonc/appdev/pydantic-ai/ - feel free to browse & study at your leisure. 

## Executive Summary

This guide explains how to implement **mid-conversation agent persona switching** using Pydantic AI's dynamic system prompt feature. The key insight: you can have a single agent that changes its "identity" during a conversation by combining:

1. **Dynamic system prompts** - Re-evaluated on each turn (including after tool calls)
2. **Instructions functions** - Provide ambient context about available agents
3. **Mutable deps** - Store current agent state
4. **Tool to switch** - Changes the active persona

## Architecture Overview

### YAML Agent Structure (from flutter-agent-api/agents/*)

```yaml
name: "Jarvis"
identifier: "jarvis_v1"
description: "Your own personal Jarvis."
voice_id: "0KYw5BqNtUJmEkwDENbP"
prompt: |
  You are Jarvis, a helpful and professional assistant...
```

### Key Components

1. **Agent Registry**: Load all YAML files at startup, indexed by `identifier`
2. **Dependencies Object**: Tracks current active agent identifier
3. **Dynamic System Prompt**: Returns prompt for current agent
4. **Instructions Function**: Provides list of available agents as ambient context
5. **change_agent Tool**: Switches active agent by updating deps

---

## Pydantic AI: Dynamic System Prompts Deep Dive

### The `dynamic=True` Parameter

**Location**: `pydantic_ai_slim/pydantic_ai/agent/__init__.py:196`

```python
@agent.system_prompt(dynamic=True)
async def get_current_agent_prompt(ctx: RunContext[AgentDeps]) -> str:
    """This function is called on EVERY turn, including after tool calls!"""
    return AGENT_REGISTRY[ctx.deps.current_agent_id].prompt
```

**How it works** (from `_agent_graph.py:311-334`):

1. System prompts marked with `dynamic=True` are stored in `system_prompt_dynamic_functions` dict
2. Each `SystemPromptPart` in the message history gets a `dynamic_ref` (the function's `__qualname__`)
3. Before **every** model request, `_reevaluate_dynamic_prompts()` is called
4. It looks up each `dynamic_ref` and re-runs the function
5. The `SystemPromptPart` content is updated with the new result

**Critical insight**: This happens in the `UserPromptNode.run()` method (lines 233-237) which runs:
- When processing existing messages with history
- After tool calls complete
- Before each new model request

This means your system prompt can change **mid-conversation** based on tool execution!

---

## Providing Ambient Context: Instructions Functions

### The Pattern

Use `@agent.instructions` (NOT `@agent.system_prompt`) to provide metadata about available agents:

```python
@agent.instructions
async def available_agents_context(ctx: RunContext[AgentDeps]) -> str:
    """Provides list of switchable agents - NOT re-evaluated unless dynamic=True"""
    agents_list = []
    for identifier, agent_config in AGENT_REGISTRY.items():
        if identifier != ctx.deps.current_agent_id:  # Don't list self
            agents_list.append(
                f"- **{agent_config.name}** (`{identifier}`): {agent_config.description}"
            )

    return f"""
# OTHER AVAILABLE AGENTS

You have the ability to transfer this conversation to another agent when appropriate.
Available agents you can switch to:

{chr(10).join(agents_list)}

Use the `change_agent` tool to switch. The switch happens immediately - the new agent
will handle the rest of the current message.
"""
```

### Instructions vs System Prompts

**From the Pydantic AI codebase** (`agent/__init__.py:582-598`):

- **Instructions**: Appended to the `instructions` field of `ModelRequest` (like OpenAI's separate instructions parameter)
- **System Prompts**: Become `SystemPromptPart` in message history
- **Key difference**: Instructions are evaluated once per run (unless you make them dynamic too)

**Recommendation for your use case**:
- Use `@agent.instructions(dynamic=True)` if you want the agent list to update after each tool call
- Use regular `@agent.instructions` if the list is static for the conversation

---

## The change_agent Tool Implementation

### Basic Pattern

```python
@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_identifier: str) -> str:
    """Switch to a different agent persona.

    Args:
        agent_identifier: The identifier of the agent to switch to (e.g., 'jarvis_v1', 'alice_v1')

    Returns:
        Confirmation message
    """
    if agent_identifier not in AGENT_REGISTRY:
        available = ', '.join(AGENT_REGISTRY.keys())
        return f"Unknown agent '{agent_identifier}'. Available: {available}"

    old_name = AGENT_REGISTRY[ctx.deps.current_agent_id].name
    new_name = AGENT_REGISTRY[agent_identifier].name

    # The magic happens here - just update the deps!
    ctx.deps.current_agent_id = agent_identifier

    # Note: The dynamic system prompt will automatically pick up this change
    # on the NEXT model request (which happens right after this tool completes)

    return f"Transferred from {old_name} to {new_name}. {new_name} is now handling the conversation."
```

### Execution Flow

Given user message: "Hey change to Alice please. Alice what do you think about Flutter?"

1. **Tool Call Phase** (`CallToolsNode`, line ~641-673 in `_agent_graph.py`):
   - Model calls `change_agent(agent_identifier='alice_v1')`
   - Tool executes: `ctx.deps.current_agent_id = 'alice_v1'`
   - Tool returns confirmation message
   - Tool return parts are collected in `output_parts`

2. **Next Model Request** (`ModelRequestNode`, line ~669-672):
   - New `ModelRequest` is created with the tool return parts
   - `instructions` are generated by calling `get_instructions(run_context)`

3. **Before Model Call** (`UserPromptNode.run()`, line ~233-237):
   - `_reevaluate_dynamic_prompts()` is called on all messages
   - Your dynamic system prompt function runs again
   - It sees `ctx.deps.current_agent_id == 'alice_v1'`
   - Returns Alice's prompt
   - The `SystemPromptPart` is updated in the message history

4. **Model Response**:
   - Model receives updated system prompt (now Alice's)
   - Model responds as Alice to "what do you think about Flutter?"

---

## Dependencies Object Design

### Recommended Structure

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentDeps:
    """Mutable state for agent execution."""

    current_agent_id: str  # e.g., 'jarvis_v1'

    # Optional: Track conversation metadata
    switch_count: int = 0
    agent_history: list[str] = None  # Track which agents have been used

    # Add any other runtime state here
    # e.g., user_id, session_id, etc.

    def __post_init__(self):
        if self.agent_history is None:
            self.agent_history = [self.current_agent_id]
```

### Why Mutable?

**Important**: The deps object is mutable and **shared across the entire conversation**. From `_agent_graph.py:713-729`:

```python
def build_run_context(ctx: GraphRunContext[...]) -> RunContext[DepsT]:
    return RunContext[DepsT](
        deps=ctx.deps.user_deps,  # Same object reference throughout!
        # ...
    )
```

This is **critical** for agent switching - when the tool modifies `ctx.deps.current_agent_id`, that change persists for subsequent turns.

---

## Agent Registry Pattern

### Loading from YAML

```python
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class AgentConfig:
    name: str
    identifier: str
    description: str
    voice_id: str
    prompt: str

    @classmethod
    def from_yaml(cls, path: Path) -> 'AgentConfig':
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

def load_agent_registry(agents_dir: Path) -> dict[str, AgentConfig]:
    """Load all agents from YAML files."""
    registry = {}
    for yaml_file in agents_dir.glob('*.yaml'):
        config = AgentConfig.from_yaml(yaml_file)
        registry[config.identifier] = config
    return registry

# At application startup
AGENT_REGISTRY = load_agent_registry(Path('agents'))
```

---

## Complete Implementation Pattern

### 1. Setup (Application Initialization)

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

# Load all agents
AGENT_REGISTRY = load_agent_registry(Path('agents'))

@dataclass
class AgentDeps:
    current_agent_id: str

# Create the meta-agent
agent = Agent('openai:gpt-4o', deps_type=AgentDeps)
```

### 2. Dynamic System Prompt

```python
@agent.system_prompt(dynamic=True)
def current_agent_prompt(ctx: RunContext[AgentDeps]) -> str:
    """Returns the prompt for whoever is currently 'active'."""
    return AGENT_REGISTRY[ctx.deps.current_agent_id].prompt
```

### 3. Ambient Context (Instructions)

```python
@agent.instructions
def available_agents_list(ctx: RunContext[AgentDeps]) -> str:
    """Lists other agents - shown to every agent."""
    other_agents = [
        f"- **{cfg.name}** (`{id}`): {cfg.description}"
        for id, cfg in AGENT_REGISTRY.items()
        if id != ctx.deps.current_agent_id
    ]

    return f"""
# OTHER AVAILABLE AGENTS

You can transfer to another agent using the `change_agent` tool:

{chr(10).join(other_agents)}

Transfer when the user explicitly requests it, or when another agent would be better suited.
"""
```

### 4. The Switching Tool

```python
@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_identifier: str) -> str:
    """Switch to a different agent.

    Args:
        agent_identifier: Agent ID to switch to (e.g., 'alice_v1')
    """
    if agent_identifier not in AGENT_REGISTRY:
        return f"Unknown agent. Available: {', '.join(AGENT_REGISTRY.keys())}"

    old = AGENT_REGISTRY[ctx.deps.current_agent_id].name
    new = AGENT_REGISTRY[agent_identifier].name

    ctx.deps.current_agent_id = agent_identifier

    return f"Transferred from {old} to {new}."
```

### 5. Running Conversations

```python
async def run_conversation(user_message: str, initial_agent: str = 'jarvis_v1'):
    deps = AgentDeps(current_agent_id=initial_agent)

    result = await agent.run(user_message, deps=deps)
    return result.output

# Example
await run_conversation(
    "Hey change to Alice. Alice, what's your favorite color?",
    initial_agent='jarvis_v1'
)
```

---

## Advanced: Message History Persistence

### Multi-Turn Conversations

If you're maintaining message history across multiple HTTP requests:

```python
from pydantic_ai.messages import ModelMessage

# Store this in your session/database
conversation_state = {
    'message_history': [],  # list[ModelMessage]
    'current_agent_id': 'jarvis_v1',
}

async def continue_conversation(user_input: str, state: dict):
    deps = AgentDeps(current_agent_id=state['current_agent_id'])

    result = await agent.run(
        user_input,
        message_history=state['message_history'],
        deps=deps
    )

    # Update state for next turn
    state['message_history'] = result.all_messages()
    state['current_agent_id'] = deps.current_agent_id  # May have changed!

    return result.output
```

**Key point**: The `deps.current_agent_id` might change during execution (via tool call), so save it back to your session state!

---

## Gotchas & Best Practices

### 1. Dynamic Prompt Timing

**Important**: Dynamic prompts are re-evaluated in `UserPromptNode`, which means:
- ✅ They update before the model sees the tool results
- ✅ Perfect for agent switching
- ❌ The model that made the tool call doesn't see the change (it already responded)

This is **exactly what you want** - Jarvis calls the tool, then Alice responds to the rest of the message.

### 2. Instructions vs System Prompts

For the "available agents" list, you have options:

| Approach | When to Use |
|----------|-------------|
| `@agent.instructions` | Static list, set once per conversation |
| `@agent.instructions(dynamic=True)` | List should update if agents change |
| Include in system prompt | Want it in message history (for debugging) |

**Recommendation**: Use regular `@agent.instructions` unless you're dynamically loading agents mid-conversation.

### 3. Tool Return Messages

The tool's return value is shown to the model. Consider:

```python
# Verbose (model sees this)
return f"Transferred from {old} to {new}. {new} is now handling the conversation."

# Minimal (if you don't want transition chatter)
return f"Switched to {new}."

# Silent (return empty or minimal message)
return "Done."
```

Choose based on whether you want the transition to be "visible" in the conversation flow.

### 4. Validation

Always validate the target agent exists:

```python
if agent_identifier not in AGENT_REGISTRY:
    # Don't crash - return error message
    # Model will see this and can handle it gracefully
    return f"Error: Unknown agent '{agent_identifier}'"
```

### 5. Deps Persistence Across HTTP Requests

If each user message is a separate HTTP request:

```python
# Create deps from session state
deps = AgentDeps(current_agent_id=session.get('agent_id', 'jarvis_v1'))

result = await agent.run(message, deps=deps)

# IMPORTANT: Save the potentially-modified state back!
session['agent_id'] = deps.current_agent_id
```

---

## Testing Strategies

### Test Agent Switching Logic

```python
async def test_agent_switch():
    deps = AgentDeps(current_agent_id='jarvis_v1')

    # Verify initial state
    assert deps.current_agent_id == 'jarvis_v1'

    # Run with switch command
    result = await agent.run(
        "Switch to alice_v1 please",
        deps=deps
    )

    # Verify switch happened
    assert deps.current_agent_id == 'alice_v1'

    # Verify new agent responds on next turn
    result2 = await agent.run(
        "Who are you?",
        message_history=result.all_messages(),
        deps=deps
    )
    assert 'Alice' in result2.output
```

### Test Dynamic Prompt Evaluation

```python
def test_dynamic_prompt():
    ctx = RunContext(deps=AgentDeps(current_agent_id='jarvis_v1'), ...)
    prompt1 = current_agent_prompt(ctx)
    assert 'Jarvis' in prompt1

    ctx.deps.current_agent_id = 'alice_v1'
    prompt2 = current_agent_prompt(ctx)
    assert 'Alice' in prompt2
    assert prompt1 != prompt2
```

---

## Integration with Voice/TTS (flutter-agent-api specific)

Since your agents have `voice_id` fields:

```python
@dataclass
class AgentDeps:
    current_agent_id: str

    @property
    def current_voice_id(self) -> str:
        """Get voice_id for TTS."""
        return AGENT_REGISTRY[self.current_agent_id].voice_id

# After agent responds
result = await agent.run(message, deps=deps)

# Use the correct voice for TTS
voice_id = deps.current_voice_id
tts_audio = await eleven_labs_tts(result.output, voice_id=voice_id)
```

This ensures that when you switch from Jarvis to Alice, the voice automatically changes too!

---

## Performance Considerations

### Dynamic Prompt Overhead

From `_agent_graph.py:311-334`, dynamic prompts are re-evaluated on **every turn**. This is negligible for:
- Simple lookups (`AGENT_REGISTRY[id].prompt`)
- In-memory operations

Avoid:
- Database queries in dynamic prompts
- Heavy computation
- External API calls

If you need those, cache them in `deps` or use a separate caching layer.

### Message History Growth

Each agent switch creates tool call + return messages. Long conversations with many switches = large history.

Consider:
- Periodic history summarization (use a `HistoryProcessor`)
- Trim old messages (keep only last N turns)
- Store full history in DB, send truncated version to model

---

## Debugging Tips

### 1. Log Agent Switches

```python
@agent.tool
def change_agent(ctx: RunContext[AgentDeps], agent_identifier: str) -> str:
    import logging
    logger = logging.getLogger(__name__)

    old_id = ctx.deps.current_agent_id
    logger.info(f"Agent switch: {old_id} -> {agent_identifier}")

    ctx.deps.current_agent_id = agent_identifier
    return f"Switched to {AGENT_REGISTRY[agent_identifier].name}"
```

### 2. Inspect Message History

```python
result = await agent.run(message, deps=deps)

# See what actually got sent to the model
for msg in result.all_messages():
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, SystemPromptPart):
                print(f"System prompt: {part.content[:100]}...")
```

### 3. Track Dynamic Prompt Calls

```python
@agent.system_prompt(dynamic=True)
def current_agent_prompt(ctx: RunContext[AgentDeps]) -> str:
    print(f"[DEBUG] Dynamic prompt evaluated, current agent: {ctx.deps.current_agent_id}")
    return AGENT_REGISTRY[ctx.deps.current_agent_id].prompt
```

---

## Summary: Key Pydantic AI Concepts

1. **`@agent.system_prompt(dynamic=True)`** - Re-evaluated every turn, including after tool calls
2. **`@agent.instructions`** - Adds context that's separate from message history
3. **`RunContext[DepsT].deps`** - Mutable state object shared across entire conversation
4. **Dynamic re-evaluation timing** - Happens in `UserPromptNode` before each model request
5. **Tool execution flow** - Tools run, then dynamic prompts update, then model responds

---

## Next Steps for Implementation

1. **Load agent registry** from YAML files at startup
2. **Define AgentDeps** with `current_agent_id` field
3. **Create meta-agent** with `deps_type=AgentDeps`
4. **Add dynamic system prompt** that looks up current agent's prompt
5. **Add instructions** listing available agents
6. **Add change_agent tool** that modifies `ctx.deps.current_agent_id`
7. **Test with multi-turn conversations** including agent switches
8. **Integrate with TTS** using agent's `voice_id`

The Pydantic AI framework handles all the plumbing - you just need to:
- Store current state in deps
- Look up agent config in dynamic prompt
- Update state in tool

That's it! The framework ensures the right prompt is used at the right time.
