---
description: "Give a Pydantic AI agent web search with the WebSearch capability, using the provider's native search when available and a local fallback such as DuckDuckGo."
---

# Web Search

The [`WebSearch`][pydantic_ai.capabilities.WebSearch] [capability](overview.md) gives your agent web search. Like all [provider-adaptive tools](overview.md#provider-adaptive-tools), it uses the provider's native web search when the model supports it and can fall back to a local implementation on other models.

[`WebSearch`][pydantic_ai.capabilities.WebSearch] defaults to native-only. Backed by [`WebSearchTool`][pydantic_ai.native_tools.WebSearchTool] on the native side (see [Web Search Tool](../native-tools.md#web-search-tool) for provider support and configuration) — pass `native=WebSearchTool(...)` directly when you need full control over the native instance.

For the local side, pass `local='duckduckgo'` (or `local=True`) for a [DuckDuckGo](../common-tools.md#duckduckgo-search-tool) fallback (requires the `duckduckgo` optional group); for other search providers, use a [Tavily][pydantic_ai.common_tools.tavily.tavily_search_tool] wrapper from [`common_tools`](../common-tools.md), the [`ExaSearchToolset`](https://pydantic.dev/docs/ai/harness/exa-search/) from the Pydantic AI Harness, or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset].

Native configuration fields: `search_context_size`, `user_location`, `blocked_domains`, `allowed_domains`,
`max_uses`, and OpenAI Responses' `external_web_access`. The domain and `max_uses` constraints require a native web
search that applies them, as does `external_web_access=False`, because a local fallback cannot guarantee cached or
indexed-only search. On a model whose native search can't apply one that is set (Gemini applies none of them; see
[restricting parameters](../native-tools.md#restricting-parameters)), `WebSearch` raises a
[`UserError`][pydantic_ai.exceptions.UserError] rather than searching without the restriction.

Setting these fields on `WebSearch` itself disables the `local` fallback, since the bundled local tools can't apply
them. To fall back to a local search that applies the restriction itself on such models, put the restriction on the
native tool instead:

```python {title="web_search_restricted_fallback.py"}
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.native_tools import WebSearchTool


def search_pydantic_docs(query: str) -> str:
    """Search pydantic.dev only."""
    return f'pydantic.dev results for {query}'


web_search = WebSearch(native=WebSearchTool(allowed_domains=['pydantic.dev']), local=search_pydantic_docs)
```

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
