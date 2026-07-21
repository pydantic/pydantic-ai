---
title: Exa Search
description: Give a Pydantic AI agent web research tools backed by the Exa search API -- search with relevant excerpts and optional synthesized text summaries, full-page retrieval, opt-in deep search, and deferred Exa agent runs.
---

# Exa Search

`ExaSearch` gives an agent web research tools backed by the
[Exa](https://exa.ai) search API: search that returns the most relevant
excerpts from each hit (with an optional synthesized text summary), full-page
retrieval for digging into a specific URL, and opt-in deep search that
synthesizes a cited answer in one call. The separate `ExaAgent` capability
delegates long-running research to the Exa Agent API as deferred tool calls.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/exa/)

## The problem

Search tools that return only titles and snippets force a second round of
fetching before the agent can judge a source, while search tools that return
full page text flood the context with pages the agent will discard. Wiring a
search API together with a page fetcher, budgeting what each tool returns, and
prompting the agent to research methodically is boilerplate every research
agent reinvents.

`ExaSearch` bundles that plumbing into a single
[capability](/ai/core-concepts/capabilities/): the research tools, per-tool
output budgets, and short research guidance in the system prompt.

## Usage

Install the `exa` extra and set the `EXA_API_KEY` environment variable (create
a key at <https://dashboard.exa.ai>):

```bash
uv add "pydantic-ai-harness[exa]"
```

Then pass `ExaSearch` to an `Agent` via the `capabilities` parameter:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

## Tools

`ExaSearch` contributes these tools to the agent:

| Tool | Purpose |
|---|---|
| `web_search` | Search the web and return the top `num_results` pages, each with title, URL, and its most relevant excerpts. |
| `get_page` | Retrieve the full text of one specific URL -- a promising `web_search` hit, or a URL the user provided. |
| `deep_search` | Run Exa's multi-step deep search and return a synthesized, cited answer. Opt-in via `include_deep_search=True`. |
| `exa_agent` | Delegate a research task to an asynchronous Exa agent run. Provided by the separate `ExaAgent` capability. |

`web_search` returns short excerpts (Exa highlights) rather than full page
text, following [Exa's own guidance for agents](https://exa.ai/docs/reference/search-api-guide-for-coding-agents),
so surveying several sources stays cheap; the agent reads a chosen page with
`get_page`.

`get_page` text is capped at `max_text_chars` characters, keeping the **head**
(a page's lead carries the substance). One character of headroom above the cap
is requested from Exa, so when a page exceeds the cap the output ends with a
`[... page text truncated at N characters]` marker; at the API ceiling of
10,000 characters no headroom exists, so the marker cannot appear there. The
result count is bounded the same way: `num_results` is requested from Exa and
re-applied to the response.

A URL or question that returns no content, a rate limit, or a transient API or
network failure surfaces to the model as a
[`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries) rather than a
hard error: the run continues and the model can correct the URL, rephrase, or
try again. Authentication failures (401/403) are configuration errors and
propagate.

## Deep search

`deep_search` calls Exa search with `type='deep'` and a plain-text output
schema: Exa expands the question into multiple queries, searches, and returns
an answer grounded in citations -- all in **one tool call**, with the cited
sources listed under the answer. Each call invests more time and search depth
than `web_search` (Exa's research-grade mode), and the model decides when to
invoke tools, so the tool is off by default -- enable it explicitly:

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(include_deep_search=True)
```

When enabled, the capability's instructions tell the model to treat it as an
escalation from `web_search`, not a replacement. The synthesized answer is
returned in full (it is Exa-generated and inherently bounded); `max_text_chars`
only applies to `get_page`.

## Text summary

Set `text_summary` to have every `web_search` call also request Exa's
plain-text output schema, so the response carries a short summary synthesized
from the results for question-style queries. Pass `True` for an unconstrained
summary, or a string describing the desired format (sent as the schema's
`description`):

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(text_summary='One concise sentence with the requested facts.')
```

The tool's return shape is unchanged and backward compatible: the result list
is returned as before, and when Exa returns a summary it is prepended as a
`Summary:` line.

## Structured citations

Every tool returns a
[`ToolReturn`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns):
`return_value` carries the readable text the model sees (unchanged from
previous releases, including the `Sources:` blocks), and `metadata` carries
the sources as structured `ExaSource` records (`{'url': ..., 'title': ...}`)
under the `'sources'` key. Metadata is never sent to the model; the
application reads it from the `ToolReturnPart` in the message history, so
rendering citations needs no text parsing:

```python
from pydantic_ai.messages import ModelRequest, ToolReturnPart

for message in result.all_messages():
    if isinstance(message, ModelRequest):
        for part in message.parts:
            if isinstance(part, ToolReturnPart) and part.metadata is not None:
                for source in part.metadata.get('sources', []):
                    print(source['url'], source['title'])
```

`exa_agent` results additionally carry the Exa run ID in metadata under
`RUN_ID_METADATA_KEY`.

## Instructions

`ExaSearch` contributes short research guidance to the system prompt: search
wide with `web_search` first, read the most promising pages in full with
`get_page` before drawing conclusions, prefer primary sources, and cite the
URLs relied on. With `include_deep_search=True`, the guidance also covers when
to escalate to `deep_search`. Set `guidance` to replace the default text, or
to `''` to contribute no instructions at all.

## Configuration

Every field of `ExaSearch` with its default:

```python
from pydantic_ai_harness.exa import ExaSearch

ExaSearch(
    num_results=5,              # results per web_search call (1 to 100)
    max_text_chars=10_000,      # get_page text cap, in characters (1 to 10,000)
    text_summary=False,         # web_search also returns a synthesized text summary
    include_deep_search=False,  # also expose the deep_search tool
    include_domains=[],         # only search these domains (allowlist)
    exclude_domains=[],         # never search these domains (denylist)
    guidance=None,              # None = default instructions, '' = none, str = custom
    client=None,                # ExaClient -- None builds exa_py.AsyncExa from EXA_API_KEY
)
```

`include_domains` and `exclude_domains` apply to `web_search` and `deep_search`,
and are mutually exclusive -- set one, not both. Out-of-range limits and
setting both domain lists raise at construction.

## Exa agent runs

The Exa [Agent API](https://exa.ai/docs/reference/agent-api-guide) runs open-ended research tasks
asynchronously: a run is created, moves through `queued -> running`, and
reaches a terminal status (`completed`, `failed`, or `cancelled`) after up to
an hour. The separate `ExaAgent` capability maps that lifecycle onto Pydantic
AI's [deferred tool calls](/ai/tools-toolsets/deferred-tools/): its
`exa_agent` tool creates the run and defers, carrying the Exa run ID in the
deferred call's metadata.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaAgent

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaAgent()])
```

By default (`execution='inline'`) the capability resolves its own deferred
calls within the agent run by polling the Exa run to completion, so the tool
behaves like a regular (if slow) tool. With `execution='external'` the calls
bubble up as `DeferredToolRequests` output for the host application to resolve
out of band -- including from a different process, since the Exa run ID
survives in the request metadata under `RUN_ID_METADATA_KEY`. The agent's
`output_type` must include `DeferredToolRequests`, otherwise the run raises
instead of returning the deferred requests:

```python
from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests

from pydantic_ai_harness.exa import ExaAgent

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    output_type=[str, DeferredToolRequests],
    capabilities=[ExaAgent(execution='external')],
)
```

Render a finished run with the `agent_run_result` helper, passing the same
`output_schema` the capability was constructed with so external resolution
applies the same validation and produces the same tool result shape as inline
execution, then feed the results and the original message history back into
the agent to resume the deferred run:

```python
from pydantic_ai.tools import DeferredToolResults

from pydantic_ai_harness.exa import RUN_ID_METADATA_KEY, agent_run_result


async def resolve(requests, runs, output_schema=None):  # e.g. in a worker process
    results = DeferredToolResults()
    for call in requests.calls:
        run_id = requests.metadata[call.tool_call_id][RUN_ID_METADATA_KEY]
        run = await runs.poll_until_finished(run_id)
        results.calls[call.tool_call_id] = agent_run_result(run, output_schema=output_schema)
    return results


async def resume(agent, messages, results):
    return await agent.run(message_history=messages, deferred_tool_results=results)
```

Every field of `ExaAgent` with its default:

```python
from pydantic_ai_harness.exa import ExaAgent

ExaAgent(
    effort=None,            # 'low' | 'medium' | 'high' | 'xhigh' | 'auto' -- None = API default
    execution='inline',     # 'inline' polls to completion; 'external' bubbles DeferredToolRequests
    output_schema=None,     # BaseModel class or dict schema for structured output
    system_prompt=None,     # forwarded to the Exa agent run
    poll_interval=1000,     # ms between polls when resolving inline
    timeout_ms=3_600_000,   # ms to wait for a run when resolving inline
    guidance=None,          # None = default instructions, '' = none, str = custom
    runs=None,              # ExaAgentRuns -- None builds AsyncExa().agent.runs from EXA_API_KEY
)
```

An `output_schema` model class is validated against
the completed run's structured output (mismatches surface as `ModelRetry`),
while a dict schema is forwarded without client-side validation and is the
agent-spec form. Terminal failures (`failed`, `cancelled`) are returned to the
model as a structured message rather than raised, so the agent can decide how
to proceed. Each result includes the run ID, which the model can pass back as
`previous_run_id` to ask follow-up questions in the context of a previous run.

## Multiple instances

Two instances of the same capability register the same tool names, which is an
error. To run several differently configured instances in one agent (for
example one open-web `ExaSearch` and one pinned to specific domains), wrap the
extra instances in core's `PrefixTools` capability, which prefixes their tool
names:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness.exa import ExaSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        ExaSearch(),  # web_search, get_page
        PrefixTools(
            wrapped=ExaSearch(include_domains=['crunchbase.com'], guidance=''),
            prefix='cb',
        ),  # cb_web_search, cb_get_page
    ],
)
```

Set `guidance=''` on the wrapped instance (or replace it with text that tells
the model when to use the prefixed tools), since each instance otherwise
contributes the same default research guidance.

This also works for `ExaAgent`: it identifies its deferred calls by metadata
it wrote when deferring, not by tool name, so a prefixed `exa_agent` still
resolves inline, and multiple `ExaAgent` instances never claim each other's
calls.

## Custom client

The default client is `exa_py.AsyncExa`, configured from the `EXA_API_KEY`
environment variable; when the variable is missing, construction fails with a
setup hint. Pass any object satisfying the `ExaClient` protocol -- the subset of
`AsyncExa` the toolset calls -- to configure authentication or the base URL
explicitly, or to substitute a fake in tests:

```python
from exa_py import AsyncExa

from pydantic_ai_harness.exa import ExaSearch

ExaSearch(client=AsyncExa(api_key='...'))
```

The API may change between releases while the capability settles; breaking
changes ship deprecation warnings where practical.

## ExaSearch vs core WebSearch

Pydantic AI core ships a provider-adaptive
[`WebSearch`](/ai/core-concepts/capabilities/#provider-adaptive-tools)
capability: on models with a native search tool it uses the provider's own
search, executed server-side; elsewhere it falls back to a local DuckDuckGo
tool. Reach for it when you want search that follows the model.

Reach for `ExaSearch` when you want the same search behavior on every model:
one vendor, excerpts with every hit, explicit page retrieval, domain filters,
and opt-in deep search.

One caveat when combining them: on Anthropic models the provider-native search
tool is also named `web_search` on the wire, so
`capabilities=[WebSearch(), ExaSearch()]` puts two tools with the same name in
the request. Use one search capability per agent on native-search models, or
force the local fallback with `WebSearch(native=False)` (its DuckDuckGo tool is
named `duckduckgo_search`, which does not collide).

## ExaSearch vs Exa's MCP server

Exa also ships an official hosted MCP server at `https://mcp.exa.ai/mcp`
([exa-labs/exa-mcp-server](https://github.com/exa-labs/exa-mcp-server)). By
default it exposes `web_search_exa` and `web_fetch_exa`; the full catalog adds
`web_search_advanced_exa` and an agent-run set (`agent_create_run`,
`agent_wait_for_run`, `agent_get_run_output`, `agent_cancel_run`).

`ExaSearch` is the curated, typed path: bounded output, a retry-on-empty
contract, bundled research instructions, and a client seam that makes it
testable offline. The MCP server is how you get Exa's full catalog with zero
wrapper code, via Pydantic AI core's MCP capability. Their agent runs are
create-then-poll (`agent_create_run` returns an ID immediately;
`agent_wait_for_run` polls it), where `deep_search` returns the answer in a
single call. The two compose in one `capabilities` list, and none of the MCP
tool names collide with `web_search`, `get_page`, or `deep_search`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai_harness.exa import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch(), MCP('https://mcp.exa.ai/mcp')])
```

## Agent spec (YAML/JSON)

`ExaSearch` works with Pydantic AI's
[agent spec](/ai/core-concepts/agent-spec/), so you can declare it in a config
file instead of Python:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - ExaSearch:
      num_results: 3
      include_deep_search: true
  - ExaAgent:
      effort: low
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.exa import ExaAgent, ExaSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[ExaSearch, ExaAgent])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate the
capabilities. The `client` and `runs` fields are not spec-serializable;
spec-loaded instances always build the default client from `EXA_API_KEY`. In
specs, `output_schema` takes the JSON-schema dict form; Pydantic model
classes are only available when constructing the capability in Python.

## Further reading

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Toolsets](/ai/tools-toolsets/toolsets/)
- [Exa API documentation](https://docs.exa.ai)

## API reference

::: pydantic_ai_harness.exa.ExaSearch

::: pydantic_ai_harness.exa.ExaAgent

::: pydantic_ai_harness.exa.agent_run_result

::: pydantic_ai_harness.exa.ExaSource
