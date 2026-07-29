# Tool Search

The [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] [capability](overview.md) handles discovery of tools marked with `defer_loading=True`, so agents with large toolsets only pay tokens for the tools the model needs. Like the [provider-adaptive tools](overview.md#provider-adaptive-tools) above, it picks the best path for the active model — native server-executed search on Anthropic and OpenAI Responses, a local `search_tools` function tool elsewhere — and is auto-injected into every agent with zero overhead when no deferred tools exist.

Pass an explicit [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] to pick a specific [`strategy`][pydantic_ai.capabilities.ToolSearch.strategy] (`'keywords'`, `'bm25'`, `'regex'`, or a custom callable) or tune the local fallback:

```python {title="tool_search_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ToolSearch(strategy='keywords')])
```

See [Tool Search](../tools-advanced.md#tool-search) for when to reach for it, the full strategy table, and provider support details.
