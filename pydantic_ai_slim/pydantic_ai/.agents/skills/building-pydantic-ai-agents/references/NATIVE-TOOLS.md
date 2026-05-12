# Native Tools

Read this file when the user wants provider-native tools such as web search, web fetch, code execution, memory, or file search.

Prefer provider-adaptive capabilities like `WebSearch()`, `WebFetch()`, `MCP()`, or `ImageGeneration()` when the user wants a provider-agnostic solution. Use native tools directly when they explicitly want provider-native behavior or provider-specific configuration.

## Give My Agent Web Search or Code Execution

Native tools are wrapped in [`NativeTool`][pydantic_ai.capabilities.NativeTool] and passed via `capabilities=[...]`.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool

agent = Agent('openai-responses:gpt-5.2', capabilities=[NativeTool(WebSearchTool())])
result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')
print(result.output)
```

For OpenAI web search, use the Responses API model prefix (`openai-responses:`), not `openai:`.

## Native Tool Defaults

Reach for these when the provider supports them:

- `WebSearchTool`
- `WebFetchTool`
- `CodeExecutionTool`
- `ImageGenerationTool`
- `MemoryTool`
- `MCPServerTool`
- `FileSearchTool`

## Dynamic Native Tool Configuration

Prepare native tools from `RunContext` when configuration depends on the current user or request. Wrap the prepare function in `NativeTool(...)`.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool


async def prepared_web_search(ctx: RunContext[dict]) -> WebSearchTool | None:
    if not ctx.deps.get('location'):
        return None
    return WebSearchTool(user_location={'city': ctx.deps['location']})


agent = Agent(
    'openai-responses:gpt-5.2',
    capabilities=[NativeTool(prepared_web_search)],
    deps_type=dict,
)
```

## When to Use Native Tools vs Provider-Adaptive Capabilities

Use provider-adaptive capabilities — `WebSearch()`, `WebFetch()`, `MCP()`, `ImageGeneration()` — when:

- the code should work across providers
- you want local fallback when native support is missing
- the user has not committed to a provider yet

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[WebSearch()])
```

Use native tools (`NativeTool(WebSearchTool(...))`) when:

- the user explicitly wants provider-native behavior
- provider-specific configuration matters (e.g. `WebSearchTool(user_location=...)`)
- the user already picked a provider that supports the tool
