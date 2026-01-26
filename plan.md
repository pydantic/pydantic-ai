# Tool Search Toolset Implementation Plan

## Summary

Provider-agnostic `SearchableToolset` for all models. Anthropic uses native BM25 (default) or regex; others use custom `pai_tool_search_tool` with regex. Beta feature, CodeModeToolset-aware.

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single `SearchableToolset`, auto-injected by Agent | Only one search tool allowed; avoids user confusion |
| `tool_search_options: ToolSearchOptions` on Agent | Centralized config; dataclass is expandable |
| No nesting | Simplicity; algorithm applies to all deferred tools |
| `defer_loading` param on MCPServer | Consistency; avoid multiple ways to do same thing |
| Tool name: `pai_tool_search_tool` | Namespaced to avoid collisions; error if user defines same |
| No matches returns empty list | `{message: "No tools found for '{term}'", tools: []}` |
| Beta: `pydantic_ai.beta._tool_search` | Requires `beta=True` on Agent |
| BM25 fallback -> regex + warning | TODO: native BM25 via sqlite + optional dep |
| Track discovered tools via `discovered_tools: list[str]` on `ModelRequest` | Enables message history portability, CodeModeToolset integration |
| Follow builtin fallback pattern (#3212) | Anthropic native when supported, custom search tool otherwise |
| `tool_renderer` callable allowed | Warning if non-serializable in durable execution contexts |

## `ToolSearchOptions` Dataclass

```python
@dataclass
class ToolSearchOptions:
    algorithm: Literal['regex', 'bm25'] | None = None  # None = bm25 for Anthropic, regex otherwise
    max_results: int = 5
    tool_renderer: Callable[[ToolDefinition], DiscoveredToolInfo] | None = None
```

## Search Result Schema

```python
class DiscoveredToolInfo(TypedDict, total=False):
    name: Required[str]
    description: str | None  # from ToolDefinition.description
    # extensible for future fields
```

Return format: `{message: str, tools: list[DiscoveredToolInfo]}`

## API Examples

### Basic Usage - Decorator

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', beta=True)

@agent.tool(defer_loading=True)
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f'Weather in {city}: sunny'

@agent.tool(defer_loading=True)
def get_stock_price(symbol: str) -> float:
    """Get current stock price."""
    return 123.45

@agent.tool  # not deferred - always available
def get_time() -> str:
    """Get current time."""
    return '12:00 PM'

result = agent.run_sync('What is the weather in Paris?')
```

### With Custom Options

```python
from pydantic_ai import Agent
from pydantic_ai.beta import ToolSearchOptions

agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    beta=True,
    tool_search_options=ToolSearchOptions(
        algorithm='bm25',  # or 'regex', or None for provider default
        max_results=10,
    ),
)

@agent.tool(defer_loading=True)
def fetch_user_data(user_id: int) -> dict:
    """Fetch user data from database."""
    return {'id': user_id, 'name': 'John'}
```

### With Custom Tool Renderer

```python
from pydantic_ai import Agent
from pydantic_ai.beta import ToolSearchOptions, DiscoveredToolInfo
from pydantic_ai.tools import ToolDefinition

def custom_renderer(tool: ToolDefinition) -> DiscoveredToolInfo:
    return {
        'name': tool.name,
        'description': f'[TOOL] {tool.description}' if tool.description else None,
    }

agent = Agent(
    'openai:gpt-4o',
    beta=True,
    tool_search_options=ToolSearchOptions(tool_renderer=custom_renderer),
)
```

### With FunctionToolset

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

def db_query(query: str) -> list[dict]:
    """Execute a database query."""
    return [{'result': 'data'}]

def db_insert(table: str, data: dict) -> bool:
    """Insert data into a table."""
    return True

toolset = FunctionToolset(
    [db_query, db_insert],
    defer_loading=True,  # applies to all tools in this toolset
)

agent = Agent('openai:gpt-4o', toolsets=[toolset], beta=True)
```

### With MCP Server

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServer

mcp = MCPServer(
    'npx',
    args=['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'],
    defer_loading=True,  # all MCP tools are deferred
)

# Or defer only specific tools:
mcp_partial = MCPServer(
    'npx',
    args=['-y', '@modelcontextprotocol/server-filesystem', '/path/to/dir'],
    defer_loading=['read_file', 'write_file'],  # only these are deferred
)

agent = Agent('openai:gpt-4o', toolsets=[mcp], beta=True)
```

### With CodeModeToolset (Future)

```python
from pydantic_ai import Agent
from pydantic_ai.beta import ToolSearchOptions

agent = Agent(
    'anthropic:claude-sonnet-4-20250514',
    beta=True,
    tool_search_options=ToolSearchOptions(algorithm='bm25'),
)

@agent.tool(defer_loading=True, allowed_callers='code')
def complex_calculation(x: float, y: float, z: float) -> float:
    """Perform complex calculation. Best called from code."""
    return x * y + z

@agent.tool(defer_loading=True, allowed_callers='code')
def data_transform(data: list[dict]) -> list[dict]:
    """Transform data structure. Best called from code."""
    return [{'transformed': d} for d in data]

# Tools are:
# 1. Hidden initially (defer_loading=True)
# 2. Once discovered via search, added to CodeModeToolset
# 3. Model can then call them via code execution
```

### Message History Portability

```python
from pydantic_ai import Agent

agent1 = Agent('openai:gpt-4o', beta=True)

@agent1.tool(defer_loading=True)
def tool_a() -> str:
    return 'a'

@agent1.tool(defer_loading=True)
def tool_b() -> str:
    return 'b'

# Run agent1, which discovers tool_a
result1 = agent1.run_sync('Find and use tool_a')
history = result1.all_messages()

# Create agent2 with same tools
agent2 = Agent('anthropic:claude-sonnet-4-20250514', beta=True)

@agent2.tool(defer_loading=True)
def tool_a() -> str:
    return 'a'

@agent2.tool(defer_loading=True)
def tool_b() -> str:
    return 'b'

# agent2 automatically knows tool_a was discovered from history
# tool_b is still deferred
result2 = agent2.run_sync('Use tool_a again', message_history=history)
```

## Files to Create/Modify

1. **`pydantic_ai/beta/_tool_search.py`** (new)
   - `ToolSearchOptions`, `DiscoveredToolInfo`, `ToolSearchTool` builtin

2. **`pydantic_ai/toolsets/_searchable.py`** (new, private)
   - `SearchableToolset(WrapperToolset)` - per-`run_id` state, history hydration

3. **`pydantic_ai/tools.py`** - add `defer_loading: bool = False` to `ToolDefinition`

4. **`pydantic_ai/messages.py`** - add `discovered_tools: list[str] | None` to `ModelRequest`

5. **`pydantic_ai/agent/__init__.py`**
   - Add `tool_search_options: ToolSearchOptions | None`, `beta: bool = False`
   - Auto-inject `SearchableToolset` for `defer_loading=True` tools

6. **`pydantic_ai/toolsets/function.py`** - `defer_loading` on decorator + toolset

7. **`pydantic_ai/_mcp.py`** - `defer_loading: bool | list[str] = False` on MCPServer

8. **`pydantic_ai/models/__init__.py`** - `prepare_request` filters native vs custom search

9. **`pydantic_ai/models/anthropic.py`** - handle native responses, pass `defer_loading` tools + beta header

10. **`pydantic_ai/profiles/anthropic.py`** - add `supports_bm25_search: bool` (model-specific)

11. **`pydantic_ai/beta/_utils.py`** (new) - `require_beta(agent, feature_name)` helper

## Key Behaviors

- Tool name: `pai_tool_search_tool` (error on collision)
- No matches: `{message: "No tools found for '{term}'", tools: []}`
- Anthropic default: BM25 (better than regex), fallback to regex + warning if BM25 unsupported
- `discovered_tools` on `ModelRequest` enables history portability + CodeModeToolset integration
- `tool_renderer` callable allowed (warning in durable exec if non-serializable)

## CodeModeToolset Integration (Future)

- CodeModeToolset reads `discovered_tools` from message history
- Tools with `defer_loading=True, allowed_callers='code'` hidden until discovered
- Once discovered: tool added to CodeModeToolset registry, signature exposed in sandbox
- Search result includes full `ToolDefinition` serialization for reconstruction

## Open TODOs (Future PRs)

- Native BM25 implementation (sqlite + optional dep)
- Full CodeModeToolset implementation
- `allowed_callers` field on `ToolDefinition`

## Tests

- VCR: Anthropic native (BM25 + regex)
- VCR: OpenAI custom search tool
- Unit: `@agent.tool(defer_loading=True)`, state isolation, history hydration, beta gate, name collision

## Research Sources

- Issue #3590: Tool Search for All Models
- Issue #3666: Anthropic Advanced Tool Use Features (DouweM's main design comment)
- Issue #3212: Builtin Tool Fallbacks pattern
- PR #3680: First implementation attempt (closed, code/comments preserved)
- PR #3831: Rewritten implementation (closed, code/comments preserved)
- PR #3550: Original mega-PR with DouweM's review comments
