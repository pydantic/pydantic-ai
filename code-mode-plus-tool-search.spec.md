# Code Mode + Tool Search Integration Spec

## Current State (monty-code-mode branch)

### CodeModeToolset API

Wraps any toolset, exposing tools as Python functions callable via generated code.

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset, CodeModeToolset

# Define tools
toolset = FunctionToolset()

@toolset.tool
def get_user(user_id: str) -> dict:
    """Fetch user by ID."""
    return {'id': user_id, 'name': 'Alice'}

@toolset.tool
def get_orders(user_id: str) -> list[dict]:
    """Fetch orders for user."""
    return [{'id': '1', 'total': 100.0}]

# Wrap in code mode - ALL tools become code-callable
code_toolset = CodeModeToolset(wrapped=toolset)

agent = Agent('anthropic:claude-sonnet-4-5', toolsets=[code_toolset])
```

### How It Works

1. `get_tools()` returns single `run_code` tool with all wrapped tool signatures in description
2. LLM generates Python code that calls the tools
3. Code executes in Monty sandbox, tool calls intercepted and routed to real tools
4. Final expression becomes result

### Signature Generation (Public Functions)

```python
from pydantic_ai._signature_from_schema import (
    signature_from_function,  # For native Python tools
    signature_from_schema,    # For MCP/external tools (JSON schema input)
)

# Native function -> signature
result = signature_from_function(
    get_user,
    name='get_user',
    description='Fetch user by ID.',
    include_return_type=True,
)
# result.signature = "def get_user(user_id: str) -> dict:\n    """Fetch user by ID.""""

# JSON schema -> signature
result = signature_from_schema(
    name='mcp_tool',
    parameters_json_schema={'type': 'object', 'properties': {'query': {'type': 'string'}}},
    description='Search something.',
)
```

---

## Future: Tool Search + Code Mode Integration

### Problem

With ToolSearchToolset, tools are discovered dynamically. But:
- Code mode needs tool signatures upfront in the prompt
- Not all tools should be code-callable (some are simple, direct-call is fine)
- User needs control over which tools can be called via code vs direct

### Solution: `allowed_callers` Parameter

```python
from typing import Literal

AllowedCallers = Literal['direct', 'code', 'both']

@toolset.tool(allowed_callers='code')
def aggregate_data(items: list[dict]) -> dict:
    """Complex aggregation - should use code mode."""
    ...

@toolset.tool(allowed_callers='direct')  # default
def get_weather(city: str) -> str:
    """Simple lookup - direct call is fine."""
    ...

@toolset.tool(allowed_callers='both')
def search_docs(query: str) -> list[dict]:
    """Can be called either way."""
    ...
```

### Behavior

| `allowed_callers` | Direct tool call | Code mode call |
|-------------------|------------------|----------------|
| `'direct'` (default) | Yes | No |
| `'code'` | No | Yes |
| `'both'` | Yes | Yes |

### Integration with ToolSearchToolset

```python
# Tools explicitly in CodeModeToolset -> allowed_callers='code'
code_toolset = CodeModeToolset(wrapped=some_toolset)

# Tools in ToolSearchToolset with allowed_callers='code' ->
# added to code mode once discovered
search_toolset = ToolSearchToolset(
    search_fn=search_tools,
    # Tools marked allowed_callers='code' become code-callable when found
)

# Combined agent has both direct and code-callable tools
agent = Agent(
    'anthropic:claude-sonnet-4-5',
    toolsets=[
        direct_toolset,   # Direct-call tools
        code_toolset,     # Code-mode tools (known upfront)
        search_toolset,   # Discoverable tools (some may be code-callable)
    ]
)
```

### Smart Routing (Future Eval Goal)

LLM should:
1. Use direct call for simple, single-tool operations
2. Use code mode when chaining multiple tools or applying logic
3. NOT use code mode sequentially (one tool per call - defeats purpose)

This requires exposing both direct tools and code mode to the same agent, letting LLM choose.

---

## Open Questions

1. How does CodeModeToolset discover newly-found tools from ToolSearchToolset?
2. Should `allowed_callers` be on ToolDefinition or tool registration?
3. How to handle tools that appear in both direct and code mode (duplication)?
