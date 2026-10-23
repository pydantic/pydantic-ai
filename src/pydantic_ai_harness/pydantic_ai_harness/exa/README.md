# Exa Search

`ExaSearch` gives an agent web search backed by the [Exa](https://exa.ai) API.
`web_search` returns excerpts per hit, `get_page` reads one URL in full, and
opt-in `deep_search` returns a cited answer. `ExaAgent` hands long research
tasks to the Exa Agent API as deferred tool calls.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/exa/)

## Installation

uv:

```bash
uv add "pydantic-ai-harness[exa]"
```

pip:

```bash
pip install "pydantic-ai-harness[exa]"
```

Set the `EXA_API_KEY` environment variable (create a key at
<https://dashboard.exa.ai>), or pass a configured client (see
[Custom client](#custom-client)).

## The problem

Search tools that return only titles and snippets force a second round of
fetching before the agent can judge a source, while search tools that return
full page text flood the context with pages the agent will discard. Wiring a
search API together with a page fetcher, budgeting what each tool returns, and
prompting the agent to research methodically is boilerplate every research
agent reinvents.

## The solution

`ExaSearch` bundles the tools with output budgeting and short research
guidance in the system prompt: survey cheaply with excerpts, then read the
pages that matter in full.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch()])

result = agent.run_sync('What changed in the latest stable Python release?')
print(result.output)
```

## Tools

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

`get_page` returns the first `max_text_chars` characters of the page. A longer
page ends with `[... page text truncated at N characters]`. At the
10,000-character API maximum Exa cannot return the extra character that reveals
truncation, so no marker appears. `web_search` returns at most `num_results`
results, even if Exa returns more.

A `web_search` query with no matches returns `No results found for {query!r}.`,
which the model can relay to the user. An empty page or deep search answer, an
HTTP error response such as a rate limit, or a network failure is returned to
the model as a [retry prompt](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-retries), up to the tool's retry
limit; after that the run raises. Errors 401 and 403 (API key) and 402
(credits) are raised, and so is any other `exa_py` error, such as an option it
rejects or a response it can't parse.

## Untrusted content

Titles, excerpts, page text, and deep search answers come from third-party
pages, so they can carry instructions aimed at the model. Each title, URL,
author, date, and excerpt is rendered on one line, so those fields can't fake
another result's `Title:` or `URL:` line. Page text is passed through as
returned, and the `text_summary` summary and the `deep_search` answer are
Exa's own synthesized text, so any of them can still contain such lines.

`include_domains` and `exclude_domains` don't restrict `get_page`: it reads any
URL the model passes, including one a page told it to read, and the URL can
carry data from the conversation to that site's owner. To screen tool output,
add [`PromptInjectionDefender`](https://pydantic.dev/docs/ai/harness/prompt-injection-defender/); to vet each read,
require [approval](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval) for `get_page`.

## Deep search

`deep_search` calls Exa search with `type='deep'` and a plain-text output
schema: Exa expands the question into several queries, searches, and returns
one answer with its cited sources listed below it. A call takes longer and
costs more than `web_search`, and the model decides when to call it, so it is
off by default:

```python
from pydantic_ai_harness import ExaSearch

ExaSearch(include_deep_search=True)
```

When enabled, the capability's instructions tell the model to treat it as an
escalation from `web_search`, not a replacement. `deep_search` returns Exa's
answer untruncated; `max_text_chars` applies only to `get_page`.

## Text summary

Set `text_summary` to have every `web_search` call also request Exa's
plain-text output schema, so the response carries a short summary synthesized
from the results for question-style queries. Pass `True` for an unconstrained
summary, or a string describing the desired format (sent as the schema's
`description`):

```python
from pydantic_ai_harness import ExaSearch

ExaSearch(text_summary='One concise sentence with the requested facts.')
```

When Exa returns a summary, it is prepended to the results as a `Summary:`
line.

## Structured citations

Every tool returns a
[`ToolReturn`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns):
`return_value` carries the text the model sees, including any `Sources:`
block, and `metadata` carries the sources as structured `ExaSource` records
(`{'url': ..., 'title': ...}`, one per URL, with `title=None` when the page has
none) under the `'sources'` key. Metadata is never sent to the model; the
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
from pydantic_ai_harness import ExaSearch

ExaSearch(
    num_results=5,              # results per web_search call (1 to 100)
    max_text_chars=10_000,      # get_page text cap, in characters (1 to 10,000)
    text_summary=False,         # web_search also returns a synthesized text summary
    include_deep_search=False,  # also expose the deep_search tool
    include_domains=[],         # search only these domains (not get_page)
    exclude_domains=[],         # never search these domains (not get_page)
    native=False,               # prefer the model's native web search, Exa as fallback
    guidance=None,              # None = default instructions, '' = none, str = custom
    client=None,                # ExaClient -- None wraps exa_py.AsyncExa from EXA_API_KEY
)
```

`include_domains` and `exclude_domains` apply to `web_search` and
`deep_search`, not `get_page` (see [Untrusted content](#untrusted-content)),
and are mutually exclusive: set one, not both. Out-of-range limits and setting
both domain lists raise at construction.

## Exa agent runs

The Exa [Agent API](https://exa.ai/docs/reference/agent-api-guide) runs open-ended research tasks
asynchronously: a run is created, moves through `queued -> running`, and
reaches a terminal status (`completed`, `failed`, or `cancelled`) after up to
an hour. The separate `ExaAgent` capability maps that lifecycle onto Pydantic
AI's [deferred tool calls](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/): its
`exa_agent` tool creates the run and defers, carrying the Exa run ID in the
deferred call's metadata.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ExaAgent

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

from pydantic_ai_harness import ExaAgent

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    output_type=[str, DeferredToolRequests],
    capabilities=[ExaAgent(execution='external')],
)
```

Render a finished run with the `agent_run_result` helper, passing the
`output_schema` the capability was constructed with, so external resolution
validates and renders results the way inline execution does. A schema mismatch
raises `ModelRetry`; store it as the call's result so the model gets a retry
prompt, as it does inline. Skip deferred calls from other tools, which carry no
run ID. Then feed the results and the original message history back into the
agent to resume the deferred run:

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults

from pydantic_ai_harness.exa import RUN_ID_METADATA_KEY, ExaAgentRuns, agent_run_result


async def resolve(
    requests: DeferredToolRequests,
    runs: ExaAgentRuns,
    output_schema: type[BaseModel] | dict[str, object] | None = None,
) -> DeferredToolResults:  # e.g. in a worker process
    results = DeferredToolResults()
    for call in requests.calls:
        run_id = requests.metadata.get(call.tool_call_id, {}).get(RUN_ID_METADATA_KEY)
        if not isinstance(run_id, str):
            continue  # another tool's deferred call
        run = await runs.poll_until_finished(run_id)
        try:
            results.calls[call.tool_call_id] = agent_run_result(run, output_schema=output_schema)
        except ModelRetry as retry:
            results.calls[call.tool_call_id] = retry
    return results


async def resume(
    agent: Agent[None, str | DeferredToolRequests], messages: list[ModelMessage], results: DeferredToolResults
) -> str | DeferredToolRequests:
    result = await agent.run(message_history=messages, deferred_tool_results=results)
    return result.output
```

Every field of `ExaAgent` with its default:

```python
from pydantic_ai_harness import ExaAgent

ExaAgent(
    effort=None,            # an exa_py AgentEffort, e.g. 'low' or 'auto' -- None = API default
    execution='inline',     # 'inline' polls to completion; 'external' bubbles DeferredToolRequests
    output_schema=None,     # BaseModel class or dict schema for structured output
    system_prompt=None,     # forwarded to the Exa agent run
    poll_interval=1000,     # ms between polls when resolving inline (positive)
    timeout_ms=3_600_000,   # ms to wait for each run when resolving inline (positive)
    guidance=None,          # None = default instructions, '' = none, str = custom
    runs=None,              # ExaAgentRuns -- None wraps exa_py.AsyncExa from EXA_API_KEY
)
```

An `output_schema` model class is validated against
the completed run's structured output (mismatches surface as `ModelRetry`),
while a dict schema is forwarded without client-side validation and is the
agent-spec form. Terminal failures (`failed`, `cancelled`) are returned to the
model as a structured message rather than raised, so the agent can decide how
to proceed. Each result includes the run ID, which the model can pass back as
`previous_run_id` to ask follow-up questions in the context of a previous run.

Inline execution polls the runs one after another, and `timeout_ms` applies to
each, so a response that starts several runs can wait up to `timeout_ms` per
run. A run whose polling times out keeps running on Exa, and the retry prompt
gives the model its run ID to follow up on. When the agent run is cancelled
while it polls, or polling fails with an error the model can't fix, the
capability cancels the Exa runs nothing will read: the ones it hasn't finished
polling and the ones it was leaving for the model to follow up on. This needs a
`cancel` method on the `runs` client, as `exa_py`'s has, and holds the agent
run up to about 5 seconds while Exa accepts the cancellations. Under durable
execution the runs are left running, since the engine can resume the wait. If
creating a run fails in a way that Exa may have acted on (a 5xx response, or a
network failure after the request was sent), the error is raised instead of
retried, so the model can't start a second run.

## Multiple instances

Two instances of the same capability register the same tool names, and share
the default `id` (`exa_search` or `exa_agent`), so two that differ raise an
error. To run several differently configured instances in one agent (for
example one open-web `ExaSearch` and one pinned to specific domains), give each
extra instance a distinct `id` and wrap it in core's `PrefixTools` capability,
which prefixes their tool names:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness import ExaSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[
        ExaSearch(),  # web_search, get_page
        PrefixTools(
            wrapped=ExaSearch(include_domains=['crunchbase.com'], guidance='', id='crunchbase_search'),
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

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/), each Exa request is recorded, so a
recovered run reuses the result instead of making the request again. Temporal
and Prefect record each tool call in its own activity or task. DBOS runs
function tools in workflow code, so there the request runs as its own step.
`ExaAgent` records creating the Exa run and, with `execution='inline'`,
polling it to completion.

On Temporal an activity has 60 seconds by default, which `deep_search` can
exceed. Give the tool calls longer with
`TemporalDurability(toolset_activity_config={'exa_search': ActivityConfig(...)})`,
keyed by the capability's `id`,
as in [Temporal timeouts](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#harness-command-timeouts). `ExaAgent` polls its run in a capability
activity, which takes the base `activity_config`, so set its
`start_to_close_timeout` above `timeout_ms`.

The records are named after the capability's `id`, which defaults to
`exa_search` for `ExaSearch` and `exa_agent` for `ExaAgent`, so durable
execution needs no configuration. Changing an `id` renames the records, which
in-flight runs then cannot find.

## Custom client

The default client wraps an `exa_py.AsyncExa` configured from the
`EXA_API_KEY` environment variable, and offers only the calls the capabilities
make, not the rest of `AsyncExa`. When the variable is missing, building the
`Agent` fails with a setup hint; the capability itself constructs without it.
An `AsyncExa` connection pool belongs to the event loop that opened it, so the
default client builds a new `AsyncExa` when the agent runs on a new asyncio
event loop after the previous one has stopped, for example in a second
`asyncio.run`. Under Trio, which has no asyncio event loop to compare, it keeps
the one it has. A client you pass is used as is. Pass any object satisfying the
`ExaClient` protocol (the subset of `AsyncExa` the toolset calls) to configure
authentication or the base URL explicitly, or to substitute a fake in tests:

```python
from exa_py import AsyncExa

from pydantic_ai_harness import ExaSearch

ExaSearch(client=AsyncExa(api_key='...'))
```

## ExaSearch vs core WebSearch

Pydantic AI core ships a provider-adaptive
[`WebSearch`](https://pydantic.dev/docs/ai/capabilities/overview/#provider-adaptive-tools)
capability: on models with a native search tool it uses the provider's own
search, executed server-side. On other models it raises unless you pass a
`local=` fallback, such as `local=True` for DuckDuckGo. Reach for it when you
want search that follows the model.

Reach for `ExaSearch` when you want the same search behavior on every model.

To use the provider's search where there is one and Exa's elsewhere, either:

- Set `ExaSearch(native=True)`. The capability adds the native web search tool,
  and Exa's `web_search` is only sent to models without one. `get_page` and `deep_search` stay available on every model.
  `include_domains` and `exclude_domains` become the native tool's
  `allowed_domains` and `blocked_domains`.
  A model whose native search can't apply a domain filter you set gets
  Exa's `web_search` instead, so the filter still holds: on Gemini, whose
  native search takes no domain filters, setting either list means Exa
  searches.
- Or pass Exa's search as the fallback of core `WebSearch`, to configure the
  native search with `WebSearch`'s own options:

  ```python
  from pydantic_ai.capabilities import WebSearch
  from pydantic_ai_harness.exa import ExaSearch

  WebSearch(local=ExaSearch(num_results=10).web_search_tool())
  ```

Don't combine `WebSearch()` with the default `ExaSearch()` on native-search
models: on Anthropic models the native tool is also named `web_search` on the
wire, so the request would carry two tools with the same name.

## ExaSearch vs Exa's MCP server

Exa also ships an official hosted MCP server at `https://mcp.exa.ai/mcp`
([exa-labs/exa-mcp-server](https://github.com/exa-labs/exa-mcp-server)). By
default it exposes `web_search_exa` and `web_fetch_exa`; the full catalog adds
`web_search_advanced_exa` and an agent-run set (`agent_create_run`,
`agent_wait_for_run`, `agent_get_run_output`, `agent_cancel_run`).

`ExaSearch` caps output size, turns empty results into retry prompts, adds
research instructions, and accepts a fake client for tests. The MCP server
exposes Exa's full tool catalog through core's `MCP` capability. Its agent runs are
create-then-poll (`agent_create_run` returns an ID immediately;
`agent_wait_for_run` polls it), where `deep_search` returns the answer in a
single call. The two compose in one `capabilities` list, and none of the MCP
tool names collide with `web_search`, `get_page`, or `deep_search`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP
from pydantic_ai_harness import ExaSearch

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ExaSearch(), MCP('https://mcp.exa.ai/mcp')])
```

## Agent spec (YAML/JSON)

`ExaSearch` works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

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
from pydantic_ai_harness import ExaAgent, ExaSearch

agent = Agent.from_file('agent.yaml', custom_capability_types=[ExaSearch, ExaAgent])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate the
capabilities. The `client` and `runs` fields are not spec-serializable;
spec-loaded instances always build the default client from `EXA_API_KEY`. In
specs, `output_schema` takes the JSON-schema dict form; Pydantic model
classes are only available when constructing the capability in Python.

## Telemetry

`ExaSearch` and `ExaAgent` emit no spans of their own. Core's tool-call spans
cover each Exa search, page read, and run creation, and inline `exa_agent`
polling runs inside the agent run's span.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
- [Exa API documentation](https://docs.exa.ai)
