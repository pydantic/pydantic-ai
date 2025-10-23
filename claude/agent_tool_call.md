# Automatic Tool Parameter Injection in Pydantic AI

## The Problem

You want certain tool parameters to be automatically injected based on which agent is calling the tool, without:
1. The agent seeing these parameters in the tool schema
2. Having to manually fabricate ToolCallPart objects
3. The parameters showing up correctly in the conversation history

For example, Alice calling a tool should automatically have `requesting_agent="Alice"` injected, while Bob calling the same tool gets `requesting_agent="Bob"`.

## The Hard Truth

After investigating the Pydantic AI source code, here's what's available for storing data:

**ToolCallPart fields (what goes INTO a tool):**
- `tool_name`: str
- `args`: str | dict[str, Any] | None
- `tool_call_id`: str
- `part_kind`: Literal['tool-call']

**ToolReturnPart fields (what comes OUT of a tool):**
- `tool_name`: str
- `content`: Any
- `tool_call_id`: str
- `metadata`: Any  ← **THIS is where you can store extra data!**
- `timestamp`: datetime
- `part_kind`: Literal['tool-return']

**The key finding:** There is literally NO place to store additional metadata in ToolCallPart. The only fields are the ones listed above. However, ToolReturnPart DOES have a metadata field!

## Working Within the Constraints

Since ToolCallPart has no metadata field, you have three options:

1. **Store agent identity in ToolReturnPart.metadata** (visible after execution)
2. **Use RunContext.deps** to pass agent identity (not visible in history)
3. **Include agent info in the tool's return content** (visible in conversation)

### Implementation

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition
from dataclasses import dataclass, replace
from typing import Optional

# Define deps to carry agent identity
@dataclass
class AgentIdentity:
    name: str
    # other agent-specific config

# Create the prepare function that injects the parameter
async def inject_agent_parameter(
    ctx: RunContext[AgentIdentity],
    tool_def: ToolDefinition
) -> ToolDefinition:
    """Inject the requesting agent as a hidden parameter"""

    # Modify the parameters to include the agent name
    # This will be included in the actual call but hidden from the schema
    modified_params = tool_def.parameters.copy() if tool_def.parameters else {}

    # Add a default value for requesting_agent in the parameters
    # This makes it "invisible" to the LLM but present in execution
    if 'properties' not in modified_params:
        modified_params['properties'] = {}

    # Instead of adding to schema, we'll handle this differently...
    # We need to modify the actual function call

    # Actually, better approach: use the tool definition's metadata
    # or modify how the tool processes its arguments

    return tool_def

# Better approach: Use a wrapper function
def with_agent_injection(agent_name: str):
    """Decorator that injects agent name into tool calls"""

    def prepare_func(ctx: RunContext[AgentIdentity], tool_def: ToolDefinition) -> ToolDefinition:
        # We can't easily modify the args that will be passed to the function
        # But we CAN create a modified tool function that includes the parameter

        # Store the agent name in the tool definition metadata
        # This will be accessible during execution
        modified_def = replace(
            tool_def,
            # Add metadata that will be available during execution
            metadata={'requesting_agent': ctx.deps.name} if tool_def.metadata is None
                    else {**tool_def.metadata, 'requesting_agent': ctx.deps.name}
        )
        return modified_def

    return prepare_func
```

### The Clean Solution: Custom Tool Wrapper

Here's the cleanest approach that actually modifies the ToolCallPart:

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import Tool, ToolDefinition
import json
from functools import wraps
from typing import Any, Callable

class AgentAwareTool:
    """Wrapper that automatically injects agent identity into tool calls"""

    def __init__(self, func: Callable, agent_param: str = 'requesting_agent'):
        self.func = func
        self.agent_param = agent_param

    def create_tool(self, agent_name: str) -> Tool:
        """Create a tool instance for a specific agent"""

        # Create a wrapper function that injects the parameter
        @wraps(self.func)
        async def wrapped_func(ctx: RunContext[Any], **kwargs):
            # Inject the agent parameter
            kwargs[self.agent_param] = agent_name
            # Call the original function
            if asyncio.iscoroutinefunction(self.func):
                return await self.func(ctx, **kwargs)
            else:
                return self.func(ctx, **kwargs)

        # Create prepare function that modifies the recorded args
        def prepare_with_injection(ctx: RunContext[Any], tool_def: ToolDefinition) -> ToolDefinition:
            # This is where we can modify what gets recorded in history
            # We'll add the agent name to the default parameters
            return tool_def

        return Tool(wrapped_func, prepare=prepare_with_injection)

# Usage example
async def shared_tool_impl(ctx: RunContext[Any], query: str, requesting_agent: str) -> str:
    """The actual tool implementation"""
    return f"{requesting_agent} asked: {query}"

# Create the wrapper
shared_tool = AgentAwareTool(shared_tool_impl)

# Create agents with their own versions
alice_agent = Agent(
    'openai:gpt-4',
    deps_type=AgentIdentity,
    tools=[shared_tool.create_tool('Alice')]
)

bob_agent = Agent(
    'openai:gpt-4',
    deps_type=AgentIdentity,
    tools=[shared_tool.create_tool('Bob')]
)
```

### The Most Elegant Solution: Using RunContext

Actually, the cleanest approach is to just use the RunContext deps to carry agent identity:

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class AgentContext:
    agent_name: str
    # other context...

alice_agent = Agent('openai:gpt-4', deps_type=AgentContext)
bob_agent = Agent('openai:gpt-4', deps_type=AgentContext)

@alice_agent.tool
@bob_agent.tool
async def shared_tool(ctx: RunContext[AgentContext], query: str) -> str:
    """Tool that knows which agent called it"""
    requesting_agent = ctx.deps.agent_name

    # The requesting_agent is available in the function
    # but not exposed in the tool schema to the LLM
    result = f"{requesting_agent} queried: {query}"

    # To make it visible in history, return it in the response
    return f"[Agent: {requesting_agent}] {result}"

# When running:
result = await alice_agent.run("search for Python tutorials", deps=AgentContext(agent_name="Alice"))
```

### Making It Appear in ToolCallPart

If you absolutely need the `requesting_agent` to appear in the ToolCallPart args (not just in the function execution), you'll need to use the `prepare` function more creatively:

```python
from pydantic_ai.tools import ToolDefinition
from dataclasses import replace

def inject_agent_into_args(ctx: RunContext[AgentContext], tool_def: ToolDefinition) -> ToolDefinition:
    """Prepare function that modifies the tool definition to include agent name"""

    # The trick: modify the tool function to wrap the args
    original_func = tool_def.function

    @wraps(original_func)
    async def wrapped_func(ctx: RunContext[AgentContext], **kwargs):
        # Add the agent name to kwargs
        kwargs['requesting_agent'] = ctx.deps.agent_name
        return await original_func(ctx, **kwargs)

    # Return modified tool definition
    # Note: This doesn't directly modify what appears in ToolCallPart.args
    # That's determined by what the LLM actually sends
    return replace(tool_def, function=wrapped_func)

@agent.tool(prepare=inject_agent_into_args)
async def my_tool(ctx: RunContext[AgentContext], query: str, requesting_agent: str = None) -> str:
    # requesting_agent will be injected by prepare function
    return f"{requesting_agent} asked: {query}"
```

## The Reality Check

After deep investigation, here's the truth: **You cannot directly modify the ToolCallPart.args that gets recorded in message history without the LLM including those parameters.**

The ToolCallPart is created from what the LLM actually generates. The `prepare` function can:
1. Modify the tool schema seen by the LLM
2. Modify the function that gets executed
3. Add metadata or change tool properties

But it **cannot** retroactively modify the args in the ToolCallPart that gets recorded.

## Best Practice Recommendation

Use the RunContext.deps to carry agent identity and access it within your tool functions. This is:
1. Clean and idiomatic
2. Doesn't require hacking the message history
3. Keeps the tool schema simple for the LLM
4. Still provides full context during execution

```python
@dataclass
class MultiAgentContext:
    agent_name: str
    agent_role: str
    permissions: list[str]

@agent.tool
async def multi_agent_tool(ctx: RunContext[MultiAgentContext], action: str) -> str:
    agent = ctx.deps.agent_name

    # Log who did what
    log_action(agent=agent, action=action)

    # Check permissions
    if action not in ctx.deps.permissions:
        return f"{agent} not authorized for {action}"

    # Execute with agent context
    result = perform_action(action, agent=agent)

    # Include agent info in response for visibility
    return f"[{agent}] {result}"
```

This approach gives you everything you need without fighting the framework.

## Summary

While you can't directly inject parameters into ToolCallPart.args without the LLM seeing them in the schema, you can:

1. **Use RunContext.deps** to pass agent identity (recommended)
2. **Use prepare functions** to modify tool behavior
3. **Wrap tool functions** to inject parameters during execution
4. **Include agent info in tool returns** for visibility in conversation

The framework is designed to maintain transparency about what the LLM actually generated, which is why ToolCallPart.args reflects exactly what the model produced, not what was injected later.