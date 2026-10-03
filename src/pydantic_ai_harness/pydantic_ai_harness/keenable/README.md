# Keenable Search

`KeenableSearch` gives an agent web research tools backed by the
[Keenable](https://keenable.ai) search API: search that returns a short excerpt
of each hit, and full-page retrieval for digging into a specific URL.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/keenable/)

**No API key and no extra install.** Keenable's endpoints are keyless by
default, and this capability talks to them over `httpx`, which
`pydantic-ai-harness` already depends on. `KeenableSearch()` works in a fresh
environment with no signup and no configuration.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import KeenableSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[KeenableSearch()])
```

The capability adds two tools:

- `web_search(query)`: returns the matching pages with title, URL, and a short
  excerpt of each.
- `get_page(url)`: retrieves one page as markdown.

Both return a `ToolReturn` whose `metadata['sources']` lists the URLs and
titles behind the result, so an application can render citations from
`ToolReturnPart.metadata` without parsing the text.

## Configuration

```python
from pydantic_ai import Agent
from pydantic_ai_harness import KeenableSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        KeenableSearch(
            num_results=10,
            max_snippet_chars=800,
            max_page_chars=20_000,
        )
    ],
)
```

| Field | Default | What it does |
|---|---|---|
| `num_results` | `5` | How many results `web_search` returns. |
| `max_snippet_chars` | `500` | Excerpt budget per search result. |
| `max_page_chars` | `10_000` | Page budget for `get_page`; longer pages are truncated and marked, with the marker counted against the budget. |
| `guidance` | `None` | Replaces the default research instructions, keeping the untrusted-content rule; `''` contributes none. |
| `client` | `None` | A `KeenableClient` to use instead of the default HTTP client. |

`max_snippet_chars` exists because Keenable returns whole-page text on every
search result, an order of magnitude more than the snippet a typical search API
returns. Left unbounded, a single `web_search` would spend the context window
on pages the agent has not chosen to read yet; `get_page` is how it opts into
one.

## Authentication

Keenable is keyless by default, so nothing is required. An API key raises rate
limits:

```bash
export KEENABLE_API_KEY='keen_...'
```

Set `KEENABLE_API_URL` to point at a different Keenable deployment. It must be
`https`, except against loopback for local development, and carry no query or
fragment: the endpoint path is appended to it.

To configure either explicitly rather than through the environment, pass a
client:

```python {title="keenable_client.py" test="skip" lint="skip"}
from pydantic_ai_harness.keenable import HttpKeenableClient, KeenableSearch

KeenableSearch(client=HttpKeenableClient(api_key='keen_...'))
```

Any object satisfying the `KeenableClient` protocol works, which is also how
tests substitute a fake.

For a proxy, a self-hosted deployment, or a `KeenableClient` of your own, this
is what the default client sends. `search` posts a JSON body `{"query": ...}`
to `/v1/search`, or to `/v1/search/public` when there is no key, and returns
the response's `results` list. `fetch` sends `GET /v1/fetch`, or
`/v1/fetch/public` when keyless, with the page address in the `url` query
parameter, and returns the response object; the toolset reads its `content`,
`title`, and `url`. Every request carries an `X-Keenable-Title: Pydantic AI
Harness` header, adds `X-API-Key` when a key is set, and times out after 30
seconds.

## Errors

Any HTTP error status other than `401`, `402`, and `403` becomes `ModelRetry`,
as do network failures and a `2xx` whose body is not JSON, so the model can
wait, rephrase, or try another URL rather than abort the run. A `get_page` call
that comes back with no readable content is a `ModelRetry` too. A `401`, `402`,
or `403` propagates as `httpx.HTTPStatusError`: a rejected key or an exhausted
account is configuration the model cannot fix.
