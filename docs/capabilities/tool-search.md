---
description: "Let the model discover deferred tools on demand with the Pydantic AI ToolSearch capability, so agents with many tools only pay tokens for the ones they use."
---

# Tool Search

The [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] [capability](overview.md) handles model-driven discovery of searchable tools marked with `defer_loading=True`, so agents with large toolsets only pay tokens for the tools the model needs. Like the [provider-adaptive tools](overview.md#provider-adaptive-tools) above, it picks the best path for the active model — native server-executed search on Anthropic and OpenAI Responses, a local `search_tools` function tool elsewhere — and is auto-injected into every agent when searchable deferred tools exist. Bundle-level disclosure is covered by [on-demand capabilities](on-demand.md).

It declares a default `id` of `'tool_search'`, so two instances merge into one instead of raising a duplicate-id error — see [building custom capabilities](custom.md) for the merge rules.

Pass an explicit [`ToolSearch`][pydantic_ai.capabilities.ToolSearch] to pick a specific [`strategy`][pydantic_ai.capabilities.ToolSearch.strategy] (`'keywords'`, `'bm25'`, `'regex'`, or a custom callable) or tune the local fallback:

```python {title="tool_search_capability.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ToolSearch

agent = Agent('{{ anthropic:medium }}', capabilities=[ToolSearch(strategy='keywords')])
```

A custom callable takes the run context, the model's queries and the searchable tool definitions, and returns the names of the matching tools. [`keywords_search`][pydantic_ai.capabilities.keywords_search] is the `'keywords'` algorithm as such a function, for a custom search to wrap or fall back to:

```python {title="tool_search_fallback.py"}
from collections.abc import Sequence

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ToolSearch, keywords_search
from pydantic_ai.tools import ToolDefinition


def search(ctx: RunContext, queries: Sequence[str], tools: Sequence[ToolDefinition]) -> list[str]:
    exact = [tool.name for tool in tools if tool.name in queries]
    return exact or keywords_search(ctx, queries, tools)


agent = Agent('{{ anthropic:medium }}', capabilities=[ToolSearch(strategy=search)])
```

When the local `search_tools` function tool is used, its retry budget follows the agent's tool budget — so `Agent(retries={'tools': N})` gives the model `N` attempts to correct a malformed `queries` argument, on the same [precedence ladder](../tools-advanced.md#which-retry-limit-wins) as any other tool. A search that finds no matches returns normally and never spends a retry.

See [Tool Search](../tools-advanced.md#tool-search) for when to reach for it, the full strategy table, and provider support details.
