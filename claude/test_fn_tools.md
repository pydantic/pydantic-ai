# Testing Function Tools in Pydantic AI

## Overview


In Pydantic AI, tools are functions that agents can call during execution. When writing tests, you often need to verify that an agent has specific tools registered and configured correctly. This document covers various approaches to test tool presence and configuration.

## Key Concepts

### Tool Storage

Tools in an agent are stored in different toolsets:
- `agent._function_toolset` - Contains tools added via `@agent.tool` or `@agent.tool_plain` decorators
- `agent._output_toolset` - Contains output/result tools 
- `agent._user_toolsets` - Contains additional toolsets passed during agent creation

### Direct Access (Internal API)

The most direct way to check if an agent has a tool is to access the internal `_function_toolset.tools` dictionary:

```python
# Check if a tool exists
assert 'my_tool' in agent._function_toolset.tools

# Access tool properties
tool = agent._function_toolset.tools['my_tool']
assert tool.takes_ctx is True  # or False
assert tool.max_retries == 3
```

**Note**: This uses private attributes (prefixed with `_`), which may change in future versions.

## Testing Patterns

### 1. Testing Tool Registration

```python
def test_tool_registration():
    agent = Agent('test')
    
    @agent.tool_plain
    def calculator(x: int, y: int) -> int:
        return x + y
    
    # Verify tool is registered
    assert 'calculator' in agent._function_toolset.tools
```

### 2. Testing Tool Configuration

```python
def test_tool_with_context():
    agent = Agent('test', deps_type=int, retries=7)
    
    @agent.tool
    def my_tool(ctx: RunContext[int], value: str) -> str:
        return f"{ctx.deps}: {value}"
    
    # Verify tool configuration
    tool = agent._function_toolset.tools['my_tool']
    assert tool.takes_ctx is True
    assert tool.max_retries == 7  # Inherits from agent
```

### 3. Testing Tools Added During Construction

```python
from pydantic_ai import Tool

def plain_tool(x: int) -> int:
    return x + 1

def ctx_tool(ctx: RunContext[int], x: int) -> int:
    return x + ctx.deps

def test_init_tools():
    # Using Tool wrapper for configuration
    agent = Agent(
        'test',
        tools=[
            Tool(plain_tool, max_retries=5),
            Tool(ctx_tool, takes_ctx=True, max_retries=3)
        ],
        deps_type=int
    )
    
    assert 'plain_tool' in agent._function_toolset.tools
    assert 'ctx_tool' in agent._function_toolset.tools
    
    assert agent._function_toolset.tools['plain_tool'].takes_ctx is False
    assert agent._function_toolset.tools['plain_tool'].max_retries == 5
    
    assert agent._function_toolset.tools['ctx_tool'].takes_ctx is True
    assert agent._function_toolset.tools['ctx_tool'].max_retries == 3
```

### 4. Testing Tool Name Conflicts

```python
def test_tool_name_conflicts():
    from pydantic_ai import UserError
    
    agent = Agent('test')
    
    @agent.tool_plain
    def my_tool() -> str:
        return "first"
    
    # Attempting to add another tool with same name raises error
    with pytest.raises(UserError, match="Tool name conflicts"):
        @agent.tool_plain
        def my_tool() -> str:  # Same name!
            return "second"
```

### 5. Testing Output Tools

```python
def test_output_tool():
    from pydantic_ai.output import ToolOutput
    
    agent = Agent(
        'test',
        output_type=ToolOutput(int, name='calculate_result')
    )
    
    # Output tools are in a different toolset
    # You'd need to run the agent to see the output tool in action
    result = agent.run_sync('calculate 2+2')
    
    # The output tool is used internally for structured output
```

### 6. Testing Tool Execution in Tests

```python
def test_tool_gets_called():
    call_count = {'calculator': 0}
    
    agent = Agent('test')
    
    @agent.tool_plain
    def calculator(x: int, y: int) -> int:
        call_count['calculator'] += 1
        return x + y
    
    result = agent.run_sync('add 5 and 3')
    
    # Verify tool was called
    assert call_count['calculator'] == 1
    assert 'calculator' in result.output  # Tool name appears in output
```

### 7. Using FunctionModel for Controlled Testing

```python
from pydantic_ai.models.function import FunctionModel, AgentInfo

def test_verify_tools_available():
    def check_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Inspect available tools during model execution
        assert info.function_tools is not None
        tool_names = [t.name for t in info.function_tools]
        assert 'my_tool' in tool_names
        
        # Return a response using the tool
        return ModelResponse(
            parts=[ToolCallPart('my_tool', '{"value": "test"}')]
        )
    
    agent = Agent(FunctionModel(check_tools))
    
    @agent.tool_plain
    def my_tool(value: str) -> str:
        return f"processed: {value}"
    
    result = agent.run_sync('test')
```

## Advanced Testing

### Testing Tool Schemas

```python
def test_tool_json_schema():
    from pydantic_ai.tools import Tool
    
    # Define custom JSON schema
    json_schema = {
        'type': 'object',
        'properties': {
            'query': {'type': 'string', 'description': 'Search query'},
            'limit': {'type': 'integer', 'default': 10}
        },
        'required': ['query']
    }
    
    def search_func(**kwargs) -> str:
        return f"Searching for: {kwargs['query']}"
    
    tool = Tool.from_schema(
        search_func,
        name='search',
        description='Search for items',
        json_schema=json_schema
    )
    
    agent = Agent('test', tools=[tool])
    assert 'search' in agent._function_toolset.tools
```

### Testing with Toolsets

```python
from pydantic_ai.toolsets.function import FunctionToolset

async def test_custom_toolset():
    toolset = FunctionToolset[None]()
    
    @toolset.tool
    def add(a: int, b: int) -> int:
        return a + b
    
    agent = Agent('test', toolsets=[toolset])
    
    # Tools from toolsets are available at runtime
    # They're combined with function tools
```

## Best Practices

1. **Avoid relying on private attributes in production code** - The `_function_toolset` attribute is private and may change. For production code, test behavior rather than internal state.

2. **Test tool behavior, not just presence** - Instead of just checking if a tool exists, test that it produces the expected results when called.

3. **Use descriptive tool names** - This makes tests more readable and debugging easier.

4. **Mock external dependencies** - If tools call external services, mock those calls in tests.

5. **Test error cases** - Verify tools handle invalid inputs gracefully.

## Common Pitfalls

1. **Tool name conflicts** - Each tool must have a unique name within an agent
2. **Context type mismatches** - Tools with `RunContext` must match the agent's `deps_type`
3. **Missing `takes_ctx` flag** - When using `RunContext`, you must set `takes_ctx=True` when using `Tool()`
4. **Output tool conflicts** - Output tool names can conflict with function tool names

## Summary

Testing tools in Pydantic AI primarily involves:
- Checking tool registration via `agent._function_toolset.tools`
- Verifying tool configuration (context usage, retry settings)
- Testing tool execution and results
- Handling edge cases like name conflicts

While internal APIs provide direct access for testing, prefer testing the observable behavior of your agents and tools when possible.