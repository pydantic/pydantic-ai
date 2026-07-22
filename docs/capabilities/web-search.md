# Web Search

The [`WebSearch`][pydantic_ai.capabilities.WebSearch] [capability](overview.md) gives your agent web search. Like all [provider-adaptive tools](overview.md#provider-adaptive-tools), it uses the provider's native web search when the model supports it and can fall back to a local implementation on other models.

[`WebSearch`][pydantic_ai.capabilities.WebSearch] defaults to native-only. Backed by [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] on the native side (see [Web Search Tool](../native-tools.md#web-search-tool) for provider support and configuration) — pass `native=WebSearchTool(...)` directly when you need full control over the native instance.

For the local side, pass `local='duckduckgo'` (or `local=True`) for a [DuckDuckGo](../common-tools.md#duckduckgo-search-tool) fallback (requires the `duckduckgo` optional group); for other search providers, use a [Tavily][pydantic_ai.common_tools.tavily.tavily_search_tool] wrapper from [`common_tools`](../common-tools.md), the [Exa](https://pydantic.dev/docs/ai/harness/exa-search/) search toolset from the Pydantic AI Harness, or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset].

Native constraint fields: `search_context_size`, `user_location`, `blocked_domains`, `allowed_domains`, `max_uses`. The domain and `max_uses` constraints require native support (the shipped DuckDuckGo fallback doesn't enforce them).

```python {title="web_search.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import WebSearch

# Native-only — raises on models without native web search
WebSearch()

# Native preferred; DuckDuckGo fallback (needs `pydantic-ai-slim[duckduckgo]`)
WebSearch(local='duckduckgo')

# Native preferred; custom callable as fallback
def my_search(query: str) -> str: ...
WebSearch(local=my_search)
```
