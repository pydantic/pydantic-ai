# You.com

Give an agent web research tools backed by the [You.com](https://you.com) APIs.
Two capabilities ship here. `YouSearch` adds `web_search` and `get_page`, for
surveying the web and reading one page in full. `YouResearch` adds `answer`,
`research`, and `finance_research`, for cited answers to questions a single
lookup cannot settle.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/youdotcom/)

## Installation

The examples below use an Anthropic model and `Agent.from_file`, so they also
pull in the `anthropic` provider and the `spec` YAML support:

```bash
uv add "pydantic-ai-harness[youdotcom,anthropic]" "pydantic-ai-slim[spec]"
```

Set the `YDC_API_KEY` environment variable (create a key at
<https://api.you.com>), or pass a configured client (see
[Custom client](#custom-client)). The legacy `YOU_API_KEY_AUTH` variable is
also honored.

## The problem

Some search tools return only a title and a snippet, so the agent has to fetch
the page before it can judge the source. Others return the whole page, which
fills the context with text the agent throws away. And some questions are not a
lookup at all: answering them takes many searches, reading across the results,
and writing up an answer with citations.

## The solution

`YouSearch` brings the search tools, a limit on how much text they return, and
short research guidance for the system prompt: survey cheaply with excerpts,
then read the pages that matter in full.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[YouSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

`YouResearch` adds the cited-answer and multi-step research tools, for
questions a single search cannot settle:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouResearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[YouResearch()])
```

The two compose in one agent: search and page reads for cheap surveying,
research for the hard questions.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouResearch, YouSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[YouSearch(), YouResearch()])
```

## Tools

| Tool | Capability | Purpose |
|---|---|---|
| `web_search` | `YouSearch` | Search the web and return the top `num_results` pages, each with title, URL, and its most relevant excerpts. |
| `get_page` | `YouSearch` | Retrieve the markdown of one specific URL -- a promising `web_search` hit, or a URL the user provided. |
| `answer` | `YouResearch` | Get a synthesized answer with citations, grounded in live web results, in one call. |
| `research` | `YouResearch` | Run multi-step research that reads across many sources and returns a thorough, cited answer. |
| `finance_research` | `YouResearch` | The finance-tuned counterpart of `research`, for companies, markets, and instruments. |

`web_search` returns short excerpts from each page rather than the whole page,
so looking over several sources stays cheap. The agent then reads a page it
picks with `get_page`. Set `extraction_mode='full_page'` to get each result's
full markdown instead.

`get_page` and full-page `web_search` text are capped at `max_text_chars`
characters, keeping the **head** (a page's lead carries the substance); when a
page exceeds the cap the output ends with a
`[... page text truncated at N characters]` marker. The number of results is
limited the same way: `num_results` is sent to You.com, and applied again to
the response that comes back.

A `web_search` query that matches nothing is a valid answer, not an error: the
tool returns `No results found for {query!r}.`, which the model can pass on to
the user. Other failures reach the model as a `ModelRetry`: a URL or question
that comes back empty, a rate limit, a parameter You.com rejected (422), or a
temporary API or network problem. The run keeps going, and the model can fix
the URL, reword the question, or try again. Authentication, billing, and
permission failures (401/402/403) are things you have to fix in your own setup,
so they stop the run.

## Research

Use `answer` for a direct question you expect one call to settle. Use
`research` when the question needs many searches and a write-up across the
sources. `finance_research` is `research` tuned for financial analysis.

`research` waits for its result rather than handing back a job to poll, and
deep and exhaustive research routinely take minutes, so `timeout_ms` defaults
to 10 minutes. How hard it works is set by `research_effort`: `lite`,
`standard`, `deep`, or `exhaustive`. You.com also has a `frontier` level, but
it only runs as a background job, so this capability does not offer it.
`finance_research` has its own `finance_effort`: `deep` or `exhaustive`.

Set `output_schema` to a JSON schema to have `research` return structured
output (written out as JSON) instead of prose. You.com rejects an
`output_schema` when `research_effort` is `'lite'`, so asking for both raises
an error when you create the capability.

## Structured citations

Every tool returns a
[`ToolReturn`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns).
Its `return_value` is the text the model sees, with a `Sources:` block added
when the tool has citations. Its `metadata['sources']` holds the same sources
as `YouSource` records (`{'url': ..., 'title': ...}`). `web_search` also puts
the response's `search_uuid` and `latency` in metadata, for tracing. The model
does not see metadata: your application reads it from the `ToolReturnPart` in
the message history, so you can show citations without parsing any text:

```python
from pydantic_ai.messages import ModelRequest, ToolReturnPart

for message in result.all_messages():
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.metadata is not None:
                for source in part.metadata.get('sources', []):
                    print(source['url'], source['title'])
```

## Instructions

`YouSearch` adds short research guidance to the system prompt: search wide
with `web_search` first, read the most promising pages in full with `get_page`
before drawing conclusions, prefer primary sources, and cite the URLs you used.
`YouResearch` adds guidance on when to use `answer`, `research`, and
`finance_research`. Both tell the model to treat the web content it fetches as
untrusted data, not as instructions to follow. On either capability, set
`guidance` to your own text to replace the default, or to `''` to add nothing.

## Configuration

Every field of `YouSearch` with its default:

```python
from pydantic_ai_harness import YouSearch

YouSearch(
    num_results=10,              # results per web_search call (1 to 20)
    extraction_mode='highlights',  # 'highlights' excerpts, or 'full_page' markdown
    max_text_chars=10_000,       # get_page / full-page text cap, in characters
    include_domains=[],          # only return results from these domains (allowlist)
    exclude_domains=[],          # never return results from these domains (denylist)
    boost_domains=[],            # re-rank these domains higher without excluding others
    freshness=None,              # 'day' | 'week' | 'month' | 'year' | 'YYYY-MM-DDtoYYYY-MM-DD'
    country=None,                # two-letter country code to focus results
    guidance=None,               # None = default instructions, '' = none, str = custom
    timeout_ms=60_000,           # per-request timeout for the default client
    client=None,                 # YouClient -- None builds youdotcom.You from YDC_API_KEY
)
```

Every field of `YouResearch` with its default:

```python
from pydantic_ai_harness import YouResearch

YouResearch(
    research_effort='standard',  # 'lite' | 'standard' | 'deep' | 'exhaustive'
    finance_effort='deep',       # 'deep' | 'exhaustive'
    include_domains=[],          # only draw from these domains (allowlist)
    exclude_domains=[],          # never draw from these domains (denylist)
    boost_domains=[],            # re-rank these domains higher
    freshness=None,              # 'day' | 'week' | 'month' | 'year' | 'YYYY-MM-DDtoYYYY-MM-DD'
    country=None,                # two-letter country code to focus results
    output_schema=None,          # JSON schema for research structured output
    guidance=None,               # None = default instructions, '' = none, str = custom
    timeout_ms=600_000,          # per-request timeout for the default client
    client=None,                 # YouClient -- None builds youdotcom.You from YDC_API_KEY
)
```

`include_domains` is an allowlist, and cannot be combined with
`exclude_domains` or `boost_domains`. Combining them raises an error when you
create the capability, as do out-of-range limits and an invalid `freshness`.
The domain, `freshness`, and `country` settings apply to `web_search`, `answer`
and `research`. `finance_research` takes only its input and `finance_effort`.

## Multiple instances

Two instances of the same capability register the same tool names, which is an
error. To run more than one setup in a single agent -- say one `YouSearch` over
the open web and one limited to a few domains -- wrap the extra ones in core's
`PrefixTools` capability. It puts a prefix in front of their tool names:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness import YouSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        YouSearch(),  # web_search, get_page
        PrefixTools(
            wrapped=YouSearch(include_domains=['sec.gov'], guidance=''),
            prefix='sec',
        ),  # sec_web_search, sec_get_page
    ],
)
```

Set `guidance=''` on the wrapped instance (or replace it with text that tells
the model when to use the prefixed tools), since each instance otherwise
contributes the same default research guidance.

## Custom client

The default client is `youdotcom.You`, built from the `YDC_API_KEY`
environment variable. When that variable is not set, creating the capability
fails with a message telling you how to fix it. To set the API key yourself,
point at a different host, or use a fake in tests, pass any object with the
methods listed in the `YouClient` protocol:

```python
from youdotcom import You

from pydantic_ai_harness import YouSearch

YouSearch(client=You(api_key_auth='...'))
```

## YouSearch vs core `WebSearch`

Core ships a [`WebSearch`](https://ai.pydantic.dev/capabilities/#provider-adaptive-tools)
capability that adapts to the model: it uses the provider's own search where
the model has one, and a local DuckDuckGo tool everywhere else. Use it when you
want search that follows whichever model you run. Use `YouSearch` when you want
the same search on every model: one vendor, excerpts with every result, page
reads you ask for, domain filters, and freshness controls.

Give an agent one web search capability: core `WebSearch`, harness
`ExaSearch`, or `YouSearch`. They all name their tools the same way --
`web_search`, plus `get_page` for the two harness ones -- and an agent cannot
have two tools with the same name, so it fails when you create it. If you want
two of them anyway, wrap one in `PrefixTools` to rename its tools, as shown in
[Multiple instances](#multiple-instances). There is one extra case: on
Anthropic models the built-in search is also called `web_search`, so
`WebSearch` clashes there even though the search runs on Anthropic's side. Pass
`WebSearch(native=False)` to switch it to the DuckDuckGo tool, which is called
`duckduckgo_search` and does not clash.

## Agent spec (YAML/JSON)

`YouSearch` and `YouResearch` work with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - YouSearch:
      num_results: 5
      freshness: month
  - YouResearch:
      research_effort: deep
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import YouResearch, YouSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[YouSearch, YouResearch])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate the
capabilities. The `client` field is not spec-serializable; spec-loaded
instances always build the default client from `YDC_API_KEY`. In specs,
`output_schema` takes the JSON-schema dict form.

## Further reading

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
- [You.com API documentation](https://documentation.you.com)
