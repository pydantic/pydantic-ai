# Spider

Spider gives agents bounded site crawling, web search with optional inline page-content fetching, and browser-rendered markdown scraping through the Spider API.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Spider

agent = Agent('openai:gpt-4o', capabilities=[Spider()])
```

[Source code](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/spider/)

## Installation and authentication

`httpx` is already a Harness dependency; no optional extra is needed. Set `SPIDER_API_KEY` in the environment or pass `api_key` to `Spider`. An optional `base_url` can point to another HTTPS Spider-compatible service.

## Tools and limits

- `crawl_site(url)` crawls a site, asks Spider for at most `max_crawl_pages` pages, and caps the returned page count locally as well. Each page's content is limited by `max_page_chars`.
- `web_search(query, fetch_content=False)` searches Spider. Set `fetch_content=True` to request page content for each result; results are capped by `num_search_results` and content by `max_page_chars`.
- `scrape_page(url)` requests markdown using Spider's browser mode, which renders JavaScript. This is distinct from the core static `web_fetch_tool`.

All retrieved content is untrusted data. The capability's instructions tell the agent not to follow instructions found in pages or search results. The API client requires HTTPS and does not follow redirects, so the bearer token is not forwarded to another host.

## Example

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Spider

agent = Agent('openai:gpt-4o', capabilities=[Spider(max_crawl_pages=3, num_search_results=5)])
```

Telemetry: Spider does not add capability-specific telemetry; HTTP requests use the configured `httpx` transport.
