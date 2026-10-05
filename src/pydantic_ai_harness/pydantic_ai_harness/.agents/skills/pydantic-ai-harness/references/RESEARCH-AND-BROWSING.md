# Research and Browsing

Web research and browser capabilities: `Researcher` (a ready-made research stack), `ExaSearch` /
`ExaAgent` (Exa API), `YouSearch` / `YouResearch` (You.com API), `BrowserUse` (hand a goal to an
autonomous browser agent), and `PlaywrightBrowser` (the model drives Chromium itself). All are
capabilities passed in `Agent(capabilities=[...])`.

## Which one to use

| Need | Use |
| --- | --- |
| Search and read public pages; follow whatever the model's provider offers | Core `WebSearch(local=True)` + `WebFetch(local=True)` from `pydantic_ai.capabilities` |
| A research agent with cited answers, sub-question delegation, and bounded tool output, no API keys beyond the model | `Researcher` |
| Same search behaviour on every model, excerpts per hit, domain filters, one vendor | `ExaSearch` or `YouSearch` |
| Synthesized, cited answer or multi-step research in one tool call | `ExaSearch(include_deep_search=True)`, `ExaAgent`, or `YouResearch` |
| Fuzzy goal on unknown pages ("find the Pro plan price") | `BrowserUse` |
| Known, repeatable flows, logged-in pages, JS-rendered SPAs, deterministic actions | `PlaywrightBrowser` |

Core is enough when the model's provider has native search/fetch or DuckDuckGo plus markdownify
fetching is acceptable. `WebSearch()` and `WebFetch()` with no arguments are native-only and raise on
models without native support; pass `local=True` for the fallback (needs `pydantic-ai-slim[duckduckgo]`
and `pydantic-ai-slim[web-fetch]`).

Tool-name collisions: core `WebSearch` (its native tool on Anthropic models), `ExaSearch`, and `YouSearch` all expose a
tool named `web_search`; `ExaSearch` and `YouSearch` both expose `get_page`. Use one search capability
per agent, wrap extras in core `PrefixTools(wrapped=..., prefix='cb')` (a second instance of the same
class also needs its own `id=`, since each carries a default one), or use `WebSearch(native=False, local=True)`
(needs the `pydantic-ai-slim[duckduckgo]` extra) whose DuckDuckGo tool is `duckduckgo_search`. `Researcher` includes core `WebSearch`, so do not add
`ExaSearch`/`YouSearch` next to it on Anthropic models without `PrefixTools`.

To use the provider's native search where the model has one and Exa or You.com elsewhere, set
`ExaSearch(native=True)` / `YouSearch(native=True)` (their `web_search` is then only sent to models without
native search; `get_page` stays), or pass `WebSearch(local=ExaSearch().web_search_tool())` to keep core
`WebSearch`'s native options.

## Researcher

A combined capability: default research instructions, `WebSearch(local=True)`, `WebFetch(local=True)`,
`SubAgents` with one web `researcher` delegate, and `ToolOutputLimits` (spills oversized tool results
to the workspace).

```bash
uv add "pydantic-ai-harness[researcher]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import Researcher
from pydantic_ai_harness.researcher import DEFAULT_RESEARCHER_INSTRUCTIONS

agent = Agent('test', capabilities=[LocalWorkspace('.'), Researcher()])

custom = Researcher(instructions=DEFAULT_RESEARCHER_INSTRUCTIONS + 'Answer in French.\n')
```

`Researcher(*, instructions=DEFAULT_RESEARCHER_INSTRUCTIONS, subagents=None, store=None)`:

- `instructions`: replace the defaults with a string, or `None` for no instructions.
- `subagents`: `None` = the built-in `researcher` delegate; `[]` disables delegation; or your own
  `SubAgent` entries.
- `store`: overflow store for `ToolOutputLimits`. `None` spills into the run's workspace under
  `.pydantic-ai-harness/tool-output/`.

Gotchas:

- A run without a workspace fails at start with `UserError`. Attach `LocalWorkspace('.')` or a sandbox
  capability, or pass `store=LocalFileStore()` (`from pydantic_ai_harness.tool_output_limits import
  LocalFileStore`) to spill on the agent's host. Custom delegates need their own store.
- `Researcher` has no search-backend parameter. For Exa or You.com, compose the parts yourself:
  `Capability(instructions=...)`, the search capability, `SubAgents(agents=[...], agent_folders=None)`,
  `ToolOutputLimits()`. The search capability goes on the delegate that searches (see the Exa
  example below).
- `researcher_agent` (`pydantic_ai_harness.researcher`) is a model-less `Agent` for CLIs:
  `uvx --with 'pydantic-ai-harness[researcher]' clai -a pydantic_ai_harness.researcher:researcher_agent -m <model>`.
- For typed findings, give the agent an `output_type`; for parallel fan-out, add `DynamicWorkflow`.

## ExaSearch and ExaAgent

`ExaSearch` adds `web_search` (excerpts per hit), `get_page` (full text of one URL), and opt-in
`deep_search` (Exa's multi-query search returning a cited answer). `ExaAgent` adds `exa_agent`, which
runs an Exa Agent API research task as a deferred tool call.

```bash
uv add "pydantic-ai-harness[exa,anthropic]"   # and set EXA_API_KEY
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import ExaAgent, ExaSearch

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[ExaSearch(num_results=5, include_deep_search=True), ExaAgent(effort='low')],
)
```

`ExaSearch` fields: `num_results=5` (1 to 100), `max_text_chars=10_000` (1 to 10,000; `get_page` cap,
head kept), `text_summary=False` (`True` or a format string prepends a `Summary:` line),
`include_deep_search=False`, `include_domains=[]` / `exclude_domains=[]` (mutually exclusive),
`guidance=None` (`''` = no instructions), `client=None` (`ExaClient`; default `exa_py.AsyncExa` from
`EXA_API_KEY`).

`ExaAgent` fields: `effort=None` (`'low'|'medium'|'high'|'xhigh'|'auto'`), `execution='inline'` (polls
to completion inside the run) or `'external'`, `output_schema=None` (Pydantic model class, validated;
or dict schema, not validated), `system_prompt=None`, `poll_interval=1000` ms, `timeout_ms=3_600_000`,
`guidance=None`, `runs=None`.

Gotchas:

- A missing `EXA_API_KEY` raises `UserError` when the `Agent` is built (the capability itself constructs
  fine). Pass `client=AsyncExa(api_key=...)` / `runs=` to configure explicitly or inject a fake.
- `execution='external'` requires `output_type=[str, DeferredToolRequests]`; resolve with
  `agent_run_result(run, output_schema=...)` and the run id at
  `requests.metadata[call.tool_call_id][RUN_ID_METADATA_KEY]` (both from `pydantic_ai_harness.exa`).
- Empty results, rate limits, and transient errors become `ModelRetry`; 401/403 propagate.
- Citations: each tool returns a `ToolReturn` whose `metadata['sources']` is a list of `ExaSource`
  (`url`, `title`), read from `ToolReturnPart.metadata` (not sent to the model).
- Agent spec: `custom_capability_types=[ExaSearch, ExaAgent]`; `client`/`runs` are not serializable.
- Durable execution (Temporal, DBOS, Prefect) records each Exa request, so recovery does not repeat it.
  The records are named after the default `id` (`exa_search`, `exa_agent`); keep it stable once deployed.
- Tests: pass `client=` any object satisfying the exported `ExaClient` protocol
  (`pydantic_ai_harness.exa`), which needs two async methods, `search(query, *, contents, num_results,
  type, output_schema, include_domains, exclude_domains)` and `get_contents(urls, *, text)`, both
  returning an `exa_py` `SearchResponse`.

Exa-backed research orchestrator: `ExaSearch` sits on the researcher delegate, and the orchestrator
only delegates. A delegate `Agent` with no model runs on the parent run's model.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import ExaSearch, SubAgents, ToolOutputLimits
from pydantic_ai_harness.subagents import SubAgent

researcher = Agent(
    name='researcher',
    description='Research one sub-question on the web; report findings with source links',
    capabilities=[ExaSearch(num_results=5), ToolOutputLimits()],
)
orchestrator = Agent(
    'anthropic:claude-sonnet-5',
    instructions='Split the question, delegate each part, then write a cited answer.',
    capabilities=[
        LocalWorkspace('.'),  # ToolOutputLimits spills oversized results here
        SubAgents(agents=[SubAgent(researcher)], agent_folders=None),
    ],
)
```

## YouSearch and YouResearch

`YouSearch` adds `web_search` and `get_page`. `YouResearch` adds `answer` (one-call cited answer),
`research` (multi-step, minutes), and `finance_research`. All fields are keyword-only.

```bash
uv add "pydantic-ai-harness[youdotcom,anthropic]"   # and set YDC_API_KEY
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import YouResearch, YouSearch

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[YouSearch(freshness='month'), YouResearch(research_effort='deep')],
)
```

`YouSearch`: `num_results=10` (1 to 20), `extraction_mode='highlights'` or `'full_page'`,
`max_text_chars=10_000`, `include_domains` / `exclude_domains` / `boost_domains`, `freshness=None`
(`'day'|'week'|'month'|'year'|'YYYY-MM-DDtoYYYY-MM-DD'`), `country=None`, `guidance=None`,
`timeout_ms=60_000`, `client=None`.

`YouResearch`: `research_effort='standard'` (`'lite'|'standard'|'deep'|'exhaustive'`),
`finance_effort='deep'` (`'deep'|'exhaustive'`), the same domain/`freshness`/`country` filters,
`output_schema=None` (JSON-schema dict), `guidance=None`, `timeout_ms=600_000`, `client=None`.

Gotchas:

- `YDC_API_KEY` (or legacy `YOU_API_KEY_AUTH`) is checked when the `Agent` is built; missing raises
  `UserError`. Passing `client=` skips that check: for tests, pass a fake satisfying the exported
  `YouClient` protocol (the async `youdotcom.You` methods the toolsets call, such as `search_async`).
- `include_domains` cannot be combined with `exclude_domains` or `boost_domains`;
  `output_schema` with `research_effort='lite'` raises `ValueError`.
- `finance_research` ignores the domain/freshness/country filters.
- A no-match `web_search` returns `No results found for ...` (not an error); 401/402/403 stop the run.
- Citations under `metadata['sources']` as `YouSource`, like Exa.
- Durable execution records each You.com request under the default `id` (`you_search`, `you_research`),
  like Exa.

## BrowserUse

One `browse_web` tool that hands a self-contained goal to a browser-use agent driving real Chromium,
returning its final text (or validated JSON with `output_schema`).

```bash
uv add "pydantic-ai-harness[browser-use,anthropic]"   # Python 3.11+ only
```

```python {test="skip"}
from pydantic import BaseModel
from pydantic_ai import Agent

from pydantic_ai_harness import BrowserUse
from pydantic_ai_harness.browser_use import BrowserAgentSettings


class Product(BaseModel):
    name: str
    price_usd: float


agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[
        BrowserUse(
            llm='anthropic:claude-sonnet-5',
            allowed_domains=['example.com'],
            output_schema=Product,
            agent_settings=BrowserAgentSettings(use_judge=False),
        )
    ],
)
```

Key fields: `llm=None`, `allowed_domains=None`, `block_ip_addresses=True`, `headless=None`,
`max_steps=50`, `use_vision=True` (`'auto'`/`False`), `output_schema=None`, `sensitive_data=None`,
`extend_system_message=None` (steers the sub-agent), `guidance=None` (steers the host),
`session_scope='call'` or `'agent'`, `cdp_url=None`, `browser_profile=None`, `browser_agent=None`
(factory, for tests or custom setup).

Gotchas:

- Always pass `llm`. With `llm=None` browser-use falls back to its hosted `ChatBrowserUse` model, a
  separate account billed via `BROWSER_USE_API_KEY`, outside your observability.
- Costs: `use_vision=True` sends a screenshot every step; a judge call runs per task unless
  `BrowserAgentSettings(use_judge=False)`.
- Flat `sensitive_data` values require explicit-hostname `allowed_domains` (globs rejected); prefer
  the domain-scoped form `{'https://site': {'key': 'value'}}`.
- `session_scope='agent'` reuses one browser across calls (serialized); close with `aclose()` or
  `async with BrowserUse(...) as browser:`. Under `TemporalDurability`, `Agent(...)` raises
  `UserError` with either scope (its toolset has no stable id).
- Default `'call'` scope launches one Chromium per concurrent call.
- `file://`, `read_file`, and `upload_file` are disabled by the default factory.
- Set `ANONYMIZED_TELEMETRY=false` to disable browser-use telemetry.

## PlaywrightBrowser

The host model drives a stateful Chromium through 18 tools: `navigate`, `snapshot` (accessibility tree
with `aria-ref` handles), `click`, `type_text`, `press_key`, `select_option`, `hover`, `wait_for`,
`screenshot`, `get_text`, `scroll`, `go_back`, `go_forward`, `execute_js`, `console_messages`, `tabs`,
`handle_next_dialog`, `network_requests`.

```bash
uv add "pydantic-ai-harness[playwright]"
uv run playwright install chromium
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness.playwright import PlaywrightBrowser

agent = Agent(
    'test',
    capabilities=[PlaywrightBrowser(allowed_domains=['example.com'], headless=True)],
)
```

Fields: `headless=True`, `allowed_domains=None` (exact host plus subdomains), `policy=None` (full
`EgressPolicy`), `block_private_addresses=True`, `screenshot_on_navigate=False`,
`max_content_tokens=4000`, `action_timeout_ms=5000`, `navigation_timeout_ms=60000` (`0` disables
either), `chromium_sandbox=True`, `auto_install_chromium=False`, `storage_state=None` (Playwright
cookies + localStorage object, for logged-in runs), `cdp_url=None` (attach to a running Chromium).

`EgressPolicy` (from `pydantic_ai_harness.playwright`) fields: `allowed_domains=None`,
`blocked_domains=None`, `block_private_addresses=True`, `include_subdomains=True`,
`allowlist_reach=frozenset({'navigation', 'data'})`, `resolved_kinds` (all kinds). Request kinds are
`'navigation'`, `'subframe'`, `'data'` (fetch, XHR, WebSocket, beacon) and `'subresource'`
(images, scripts, fonts). `refuse(request)` checks in order: private IP literal, host that did not
resolve, private resolved address, `blocked_domains`, then `allowed_domains` only for kinds in
`allowlist_reach`. Add `'subframe'` or `'subresource'` to `allowlist_reach` to bound those too, or
subclass and override `refuse` (call `super().refuse(request)` for the default verdict).

```python {test="skip"}
from pydantic_ai_harness.playwright import EgressPolicy, EgressRequest

policy = EgressPolicy(
    allowed_domains=['example.com'],
    allowlist_reach=frozenset({'navigation', 'data', 'subframe'}),
)
public = ('93.184.215.14',)
print(policy.refuse(EgressRequest('https://evil.test/', 'subframe', resolved_addresses=public)))
#> domain not in allowed_domains
```

Gotchas:

- Without network, every hostname is refused as unresolved when `block_private_addresses=True`
  (the browser resolves names first). Test the allowlist by calling `policy.refuse(...)` with
  `resolved_addresses`, as above.
- `policy` together with `allowed_domains` or `block_private_addresses=False` raises `UserError`; put
  those on the `EgressPolicy` instead. A wildcard entry like `'*.example.com'` raises.
- Chromium starts lazily on the first browser tool call and closes at run end. A missing binary returns
  an install hint as the tool result and emits `BrowserUnavailableWarning`.
- Durable execution (Temporal, DBOS, Prefect capabilities) is rejected with `UserError` at agent
  construction.
- `storage_state` is credential material and is not accepted by agent-spec loading.
- No uploads or downloads. Tool errors come back as strings the model can act on.
- Use one browser capability per agent; `BrowserUse` and `PlaywrightBrowser` do not share sessions.

## See also

- https://pydantic.dev/docs/ai/harness/researcher/
- https://pydantic.dev/docs/ai/harness/exa-search/
- https://pydantic.dev/docs/ai/harness/youdotcom/
- https://pydantic.dev/docs/ai/harness/browser-use/
- https://pydantic.dev/docs/ai/harness/playwright/
- https://pydantic.dev/docs/ai/capabilities/web-search/
- https://pydantic.dev/docs/ai/capabilities/web-fetch/
