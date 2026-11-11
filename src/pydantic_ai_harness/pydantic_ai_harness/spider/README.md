# Spider

Opt-in web research tools backed by the [Spider API](https://spider.cloud): bounded site crawling, search with optional page-content fetching, and JavaScript-rendered markdown scraping.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Spider

agent = Agent('openai:gpt-4o', capabilities=[Spider()])
```

Set `SPIDER_API_KEY` or pass `api_key` to `Spider`. The capability uses the harness's existing `httpx` dependency, so no extra install is required. Results are capped by page count and page-character budgets. Fetched web content is untrusted input.

The `scrape_page` tool explicitly uses Spider's browser request mode to render JavaScript before returning markdown. This differs from the core static `web_fetch_tool`.

- [Full documentation](https://ai.pydantic.dev/harness/spider/)
- [Source code](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/spider/)
