# Web Fetch

The [`WebFetch`][pydantic_ai.capabilities.WebFetch] [capability](overview.md) lets your agent fetch the contents of URLs. Like all [provider-adaptive tools](overview.md#provider-adaptive-tools), it prefers the provider's native web fetch tool and can fall back to a local implementation on other models.

[`WebFetch`][pydantic_ai.capabilities.WebFetch] defaults to native-only. Backed by [`WebFetchTool`][pydantic_ai.native_tools.WebFetchTool] on the native side (see [Web Fetch Tool](../native-tools.md#web-fetch-tool) for provider support and configuration) — pass `native=WebFetchTool(...)` directly for full control.

For the local side, pass `local=True` for the bundled [markdownify-based fetch tool](../common-tools.md#web-fetch-tool) (requires the `web-fetch` optional group), or any callable, [`Tool`][pydantic_ai.tools.Tool], or [`AbstractToolset`][pydantic_ai.toolsets.AbstractToolset].

Native constraint fields: `allowed_domains`, `blocked_domains`, `max_uses`, `enable_citations`, `max_content_tokens`. Only `max_uses` requires native; domain filters are enforced locally when native isn't available.

```python {title="web_fetch.py" test="skip" lint="skip"}
from pydantic_ai.capabilities import WebFetch

# Native-only — raises on models without native web fetch
WebFetch()

# Native preferred; markdownify-based fallback (needs `pydantic-ai-slim[web-fetch]`)
WebFetch(local=True)

# Domain filters enforced locally when native isn't available
WebFetch(allowed_domains=['example.com'], local=True)
```
