# Research and Browsing

Web research and browser capabilities: `Researcher` (a ready-made research stack), `ExaSearch` /
`ExaAgent` (Exa API), `YouSearch` / `YouResearch` (You.com API), `KeenableSearch` (Keenable API, keyless),
`BrowserUse` (hand a goal to an autonomous browser agent), `PlaywrightBrowser` (the model drives Chromium itself),
`CDPBrowser` (the model's code drives Chrome with raw DevTools Protocol commands), and `ComputerUse` (the model clicks
and types on the workspace's desktop from screenshots). All are
capabilities passed in `Agent(capabilities=[...])`.

## Which one to use

| Need | Use |
| --- | --- |
| Search and read public pages; follow whatever the model's provider offers | Core `WebSearch(local=True)` + `WebFetch(local=True)` from `pydantic_ai.capabilities` |
| A research agent with cited answers, sub-question delegation, and bounded tool output, no API keys beyond the model | `Researcher` |
| Same search behaviour on every model, excerpts per hit, domain filters, one vendor | `ExaSearch` or `YouSearch` |
| Same search behaviour on every model with no API key and no extra to install | `KeenableSearch` |
| Synthesized, cited answer or multi-step research in one tool call | `ExaSearch(include_deep_search=True)`, `ExaAgent`, or `YouResearch` |
| Fuzzy goal on unknown pages ("find the Pro plan price") | `BrowserUse` |
| Known, repeatable flows, logged-in pages, JS-rendered SPAs, deterministic actions | `PlaywrightBrowser` |
| Scraping and page automation from code (`CodeMode`), one low-level tool, fewest input tokens | `CDPBrowser` |
| Desktop applications, or a site exactly as a person sees it | `ComputerUse` |

Core is enough when the model's provider has native search/fetch or DuckDuckGo plus markdownify
fetching is acceptable. `WebSearch()` and `WebFetch()` with no arguments are native-only and raise on
models without native support; pass `local=True` for the fallback (needs `pydantic-ai-slim[duckduckgo]`
and `pydantic-ai-slim[web-fetch]`).

Tool-name collisions: core `WebSearch` (its native tool on Anthropic models), `ExaSearch`, `YouSearch`, and
`KeenableSearch` all expose a tool named `web_search`; `ExaSearch`, `YouSearch`, and `KeenableSearch` all expose
`get_page`. Use one search capability
per agent, wrap extras in core `PrefixTools(wrapped=..., prefix='cb')` (a second instance of the same
class also needs its own `id=`, since each carries a default one), or use `WebSearch(native=False, local=True)`
(needs the `pydantic-ai-slim[duckduckgo]` extra) whose DuckDuckGo tool is `duckduckgo_search`. `Researcher` includes core `WebSearch`, so do not add
`ExaSearch`/`YouSearch`/`KeenableSearch` next to it on Anthropic models without `PrefixTools`.

To use the provider's native search where the model has one and Exa or You.com elsewhere, set
`ExaSearch(native=True)` / `YouSearch(native=True)` (their `web_search` is then only sent to models without
native search, or whose native search can't apply the `include_domains`/`exclude_domains` filters, like Gemini's;
`get_page` stays), or pass `WebSearch(local=ExaSearch().web_search_tool())` to keep core
`WebSearch`'s native options. `KeenableSearch().web_search_tool()` works as that keyless fallback too, with `web_search` only (no
`get_page`); `KeenableSearch` has no `native` flag.

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
  capability, or pass `store=LocalFileOverflowStore()` (`from pydantic_ai_harness.tool_output_limits import
  LocalFileOverflowStore`) to spill on the agent's host. Custom delegates need their own store.
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
    capabilities=[ExaSearch(max_results=5, include_deep_search=True), ExaAgent(effort='low')],
)
```

`ExaSearch` fields: `max_results=5` (1 to 100), `max_text_chars=10_000` (1 to 10,000; `get_page` cap,
head kept), `text_summary=False` (`True` or a format string prepends a `Summary:` line),
`include_deep_search=False`, `include_domains=[]` / `exclude_domains=[]` (mutually exclusive; they filter
`web_search` and `deep_search`, not `get_page`),
`guidance=None` (`''` = no instructions), `client=None` (`ExaClient`; default wraps `exa_py.AsyncExa` from
`EXA_API_KEY`, rebuilt on a new asyncio event loop once the previous one has stopped).

`ExaAgent` fields: `effort=None` (an `exa_py` `AgentEffort`, e.g. `'low'` or `'auto'`), `execution='inline'`
(polls to completion inside the run) or `'external'`, `output_schema=None` (Pydantic model class, validated;
or dict schema, not validated), `system_prompt=None`, `poll_interval=1000` ms and `timeout_ms=3_600_000`
(both positive; the timeout applies to each run), `guidance=None`, `runs=None`.

Gotchas:

- A missing `EXA_API_KEY` raises `UserError` when the `Agent` is built (the capability itself constructs
  fine). Pass `client=AsyncExa(api_key=...)` / `runs=` to configure explicitly or inject a fake.
- `execution='external'` requires `output_type=[str, DeferredToolRequests]`; resolve with
  `agent_run_result(run, output_schema=...)` and the run id at
  `requests.metadata[call.tool_call_id][RUN_ID_METADATA_KEY]` (both from `pydantic_ai_harness.exa`).
  Skip calls without that key (other tools' deferred calls), and store a `ModelRetry` that
  `agent_run_result` raises on a schema mismatch as the call's result.
- Empty pages and answers, HTTP error responses such as rate limits, and network failures become
  `ModelRetry` (up to the tool's retry limit); 401/402/403 and other `exa_py` errors propagate.
  Creating an Exa run is not retried after a 5xx or a network failure past sending, since the run may exist.
- When a non-durable agent run is cancelled during inline polling, or polling fails with an error the model can't fix, the Exa runs nothing will read are cancelled (unfinished ones, and ones left for the model to follow up on), when `runs` has `cancel`.
- Page text, titles, and excerpts are third-party content; `get_page` reads any URL the model passes.
  Add `PromptInjectionDefender` or tool approval when that matters.
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
from pydantic_ai_harness.sub_agents import SubAgent

researcher = Agent(
    name='researcher',
    description='Research one sub-question on the web; report findings with source links',
    capabilities=[ExaSearch(max_results=5), ToolOutputLimits()],
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

`YouSearch`: `max_results=10` (1 to 20), `extraction_mode='highlights'` or `'full_page'`,
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

## KeenableSearch

`KeenableSearch` adds `web_search` (a short excerpt per hit) and `get_page` (one URL as markdown).
It needs no extra and no API key: Keenable's public endpoints are keyless, and the client uses `httpx2`,
which the harness already depends on.

```bash
uv add pydantic-ai-harness   # KEENABLE_API_KEY is optional and only lifts rate limits
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import KeenableSearch

agent = Agent('anthropic:claude-sonnet-5', capabilities=[KeenableSearch(num_results=5)])
```

Fields: `num_results=5` (1 to 50, sent as Keenable's `max_results`), `max_snippet_chars=500`
(per-hit excerpt cap), `max_page_chars=10_000` (`get_page` cap; the truncation marker counts
against it, and a budget too small to hold the marker ends the page with `…` instead),
`guidance=None`, `client=None`.

Gotchas:

- Keenable's snippets run to thousands of characters, so `max_snippet_chars` is what keeps
  `web_search` small; `get_page` is how the model opts into a full page.
- Budgets must be positive integers, and `num_results` at most 50; anything else raises `ValueError`
  when the capability is built.
- `KEENABLE_API_KEY` switches to the keyed endpoints; `KEENABLE_API_URL` overrides the base URL and must
  be `https://` (plain `http` only on loopback), or `UserError` is raised.
- Rate limits, 5xx, a 404, a non-JSON body, a search body without `results`, and an empty page
  become `ModelRetry`; 401/402/403 stop the run. `get_page` reads Keenable's indexed copy, and
  Keenable documents that an unindexed URL fails to fetch. For tests, pass `client=` a fake
  satisfying the exported `KeenableClient` protocol.
- `HttpKeenableClient(http_client=...)` sends every request through your `httpx2.AsyncClient` (connection
  reuse, proxy, TLS, retrying transport); you close it. Without it, each request uses a fresh client.
- A non-empty `guidance` replaces the default instructions, untrusted-content rule included, like
  `ExaSearch` and `YouSearch`.
- Citations under `metadata['sources']` as `KeenableSource`, like Exa.

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
(factory, for tests or custom setup), `browser_use_telemetry=False`.

Gotchas:

- `llm=None` runs the sub-agent on the host run's model (`RunContext.model`) and inherits its
  instrumentation settings, including content redaction. browser-use's hosted `ChatBrowserUse`
  model is opt-in via `llm=ChatBrowserUse()`: a separate account billed via
  `BROWSER_USE_API_KEY`, outside your observability. Pass `llm` for a cheaper sub-agent model.
- Costs: `use_vision=True` sends a screenshot every step; a judge call runs per task unless
  `BrowserAgentSettings(use_judge=False)`.
- Usage: browser calls on Pydantic AI models (host model, model `llm`, `*_llm` settings) count toward
  the host run's usage and `UsageLimits`; exhaustion raises `UsageLimitExceeded` from `browse_web`.
  Native browser-use chat models (`ChatAnthropic`, `ChatBrowserUse`) are not counted.
- Flat `sensitive_data` values require explicit-hostname `allowed_domains` (globs rejected) and are
  substituted only on allowlisted pages; prefer the domain-scoped form `{'https://site': {'key': 'value'}}`.
  Secret values, and their JSON-escaped and `quote_plus`-encoded forms, are replaced with
  their placeholder in the `browse_web` result. Screenshots are not redacted.
- `session_scope='agent'` reuses one browser across calls (serialized); close with `aclose()` or
  `async with BrowserUse(...) as browser:`. Under `TemporalDurability`, `Agent(...)` raises
  `UserError` with either scope (its toolset has no stable id).
- Default `'call'` scope launches one Chromium per concurrent call.
- `file://` navigation is always prohibited. Without an allowlist, a `browser_profile.prohibited_domains`
  that reaches 100 entries once the harness's `file://` rules (and, with `block_ip_addresses`, its
  localhost rules) are added raises `ValueError`.
  The default factory removes `read_file` and `upload_file`.
  With `sensitive_data` set, every factory's `settings.tools` drops `evaluate`; a factory that builds
  its own `Tools` must drop it too.
- browser-use's per-run telemetry event (task, visited URLs, result) is off for the capability's
  agents; `browser_use_telemetry=True` sends it. Set `ANONYMIZED_TELEMETRY=false` to also stop
  browser-use's process-wide PostHog client (which captures uncaught exceptions).

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
`cdp_url` also takes a sync or async function of the run context that returns the endpoint: the
first browser tool call of a run calls it and the result is reused for that run; a timeout is a
tool error and is retried, an exception ends the run, and the capability only disconnects, so the
app deletes what the function created (record it on `ctx.deps`). `cdp_context='default'` reuses
sign-ins and tabs instead of creating a `'new'` context. It requires a CDP endpoint and excludes
`storage_state`. The agent can act as every signed-in account, so attach it to a profile dedicated
to the agent, never an everyday browser, and set `allowed_domains` for untrusted pages. Only
`http(s)`, `about:blank` and error-page tabs are adopted: any other tab (the browser's start page,
extension pages, `file:`, `data:`, `blob:`) is left out whatever the policy, and so is a tab holding a
non-web frame other than an inline `srcdoc`/`data:`/`blob:` one, such as an extension's injected
`chrome-extension://` UI (use a profile whose extensions do not inject frames); for a tracked tab that
moves to one or gets such a frame, the call that finds it returns an error naming the page by its scheme and host only, and
the agent stops using that tab, which stays open in the browser. A frame that appears during a call can still be
captured by that call's `snapshot` or `screenshot_on_navigate` image; the next call drops the tab. Loaded tabs and frames must pass the egress policy
(each document once, at most five seconds per call; a call that runs out asks the model to retry), and new service-worker
registrations are refused; downloads follow the browser's settings.
Playwright routes service-worker requests through the guards unless
`PLAYWRIGHT_DISABLE_SERVICE_WORKER_NETWORK` is set. Connections opened before attachment bypass
them, as does a new WebSocket opened through the native `WebSocket` constructor that script running
before attachment kept, and a page loaded before attachment can register a worker until it navigates.
Under an enforced policy, pages lose WebRTC, WebTransport, `WebSocketStream` and FedCM, whose traffic no guard sees; workers, isolated sandboxed iframes and documents loaded before attachment keep some of them.

`EgressPolicy` (from `pydantic_ai_harness.playwright`) fields: `allowed_domains=None`,
`blocked_domains=None`, `block_private_addresses=True`, `include_subdomains=True`,
`allowlist_reach=frozenset({'navigation', 'data'})`, `resolved_kinds` (all kinds). Request kinds are
`'navigation'`, `'subframe'`, `'data'` (fetch, XHR, WebSocket, beacon) and `'subresource'`
(images, scripts, fonts). `refuse(request)` checks in order: private IP literal, host that did not
resolve, private resolved address, `blocked_domains`, then `allowed_domains` only for kinds in
`allowlist_reach`. A host the domain lists refuse is not looked up. Add `'subframe'` or
`'subresource'` to `allowlist_reach` to bound those too, or subclass and override `refuse` (call
`super().refuse(request)` for the default verdict).

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
  (the browser resolves names first), except one the domain lists refuse, which gets the
  domain-list reason without a lookup. Test the allowlist by calling `policy.refuse(...)` with
  `resolved_addresses`, as above.
- `policy` together with `allowed_domains` or `block_private_addresses=False` raises `UserError`; put
  those on the `EgressPolicy` instead. A wildcard entry like `'*.example.com'` raises.
- With `block_private_addresses=True`, a launched Chromium connects through a local proxy that applies the
  private-address block to the address it dials, so DNS rebinding cannot reach a private address and a `refuse`
  override cannot admit one. A browser reached over `cdp_url` gets no proxy.
- Chromium starts lazily on the first browser tool call. At run end a local browser closes and a CDP
  connection disconnects; runs using `cdp_context='default'` on one browser share its pages. A missing
  binary returns an install hint as the tool result and emits `BrowserUnavailableWarning`.
- Durable execution (Temporal, DBOS, Prefect capabilities) is rejected with `UserError` at agent
  construction.
- `storage_state` is credential material and is not accepted by agent-spec loading.
- No uploads or downloads. Tool errors come back as strings the model can act on.
- Use one browser capability per agent; `BrowserUse`, `PlaywrightBrowser` and `CDPBrowser` do not share sessions.

## CDPBrowser

One tool, `browser_cdp(method, params, wait_for_event, timeout)`, sends a raw Chrome DevTools Protocol command to
a throwaway Chrome tab. Pair it with `CodeMode`, so the model's code calls the tool in loops.

```bash
uv add "pydantic-ai-harness[cdp-browser]"
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import CDPBrowser, CodeMode

agent = Agent('test', capabilities=[CDPBrowser(allowed_domains=['example.com']), CodeMode()])
```

Fields: `headless=True`, `chrome_path=None` (`CHROME_PATH`, then the usual install locations),
`chrome_sandbox=True`, `allowed_methods=DEFAULT_ALLOWED_CDP_METHODS`, `allowed_domains=None`,
`block_private_addresses=True`, `policy=None` (the same `EgressPolicy` as `PlaywrightBrowser`, exported from
`pydantic_ai_harness.cdp_browser` too).

- Chrome launches when the run starts, even if the tool is never called, and closes when it ends; each run gets
  its own. A missing Chrome fails the run with `FileNotFoundError`.
- Commands outside `allowed_methods`, non-http(s) `Page.navigate` URLs, URLs the egress policy refuses,
  protocol errors and commands that time out (`timeout=30` covers the command and `wait_for_event`) are retries.
- The egress policy applies to every request of the tab, its frames, popups and workers through CDP `Fetch`
  interception, including navigations page scripts start. With `block_private_addresses=True`, `http` and `https`
  connections go through a local proxy that applies the private-address block to the address it dials, so DNS
  rebinding cannot reach a private address; the domain lists stay with the `Fetch` guard.
- `Page.captureScreenshot` returns image content (5 MB cap). Dialogs are dismissed; downloads are denied.
- WebSocket connections go through a local proxy that applies the egress policy to their host and port.
- Set `chrome_sandbox=False` only where Chrome's sandbox cannot start (it exits with "No usable sandbox!").
- Durable execution is rejected with `UserError` at agent construction.

## ComputerUse

One `computer` tool. The model calls it with a list of mouse and keyboard actions. The actions run on the screen of
`ctx.workspace`, and the tool returns a screenshot. It needs a workspace whose backend implements `SupportsDisplay`, and
a model that accepts images. Without a display, the run ends with `UserError`; a model that can't read images gets a
note in place of each screenshot.

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import ComputerUse, DockerContainer

agent = Agent('test', capabilities=[DockerContainer(desktop=True), ComputerUse()])
```

Fields:

- `settle_seconds=2`: how long, at most, to wait for the screen to stop changing after the actions run. It takes
  screenshots until two in a row match. `0` takes the screenshot at once.
- `native=True`: adds a `ComputerTool`, so a model whose adapter supports one gets its provider's own computer tool,
  and `ComputerUse` runs its calls. `False` sends the `computer` function tool to every model, for example so GPT can zoom.

Every request carries every earlier screenshot, and nothing drops them. A long run reaches the provider's image limit,
such as 100 images on Claude models with a 200k-token context.

Reading a run: each call is a `ComputerCallPart`, with its `actions`. Each result is a `ComputerReturnPart`, with
`result` (a `ComputerResult`) and `screenshot`. Both come from `pydantic_ai.computer_use`. Find them with `isinstance`,
not by the tool name. Each call that finishes with a screenshot sends an `ActionsPerformedEvent` in the `computer_use`
namespace. Its `result` is the call's `ComputerResult`.

A workspace backend with a screen implements `SupportsDisplay`: a `display` property that returns a `Display`, or
`None` without a screen. `Display` and the optional `SupportsZoom` and `SupportsCursorPosition` are in
`pydantic_ai_harness.computer_use`, not in core `pydantic_ai.workspaces`.

Gotchas:

- Calls run without asking. To approve each call that does more than look, raise `ApprovalRequired` from a
  `before_tool_execute` hook when all three hold: `tool_def.tool_kind == COMPUTER_KIND.id` (from
  `pydantic_ai.computer_use`), `ctx.tool_call_approved` is false, and an action in the validated `args['actions']`
  isn't `screenshot`, `zoom`, `wait` or `cursor_position`. Don't test `isinstance(call, ComputerCallPart)`: a call
  whose arguments a hook repaired arrives as a plain `ToolCallPart`.
- On the Gemini API, a call Gemini flags as needing confirmation raises `ApprovalRequired` in the tool itself. Resolve
  it with a `HandleDeferredToolCalls` capability, or add `DeferredToolRequests` to `output_type` and resume the run;
  with neither, the run fails at the first flagged call. Streamed runs (`run_stream()`, the UI adapters, an event
  stream handler) get no flags from Google, so they never pause.
- Screenshots are sent at the screen's full size. A 1280x800 screen suits every model.
- If the screen changes size, a call that gives positions is refused until the model takes a new screenshot.
- When one `computer` call in a model response fails, the response's later `computer` calls don't run, and the model
  is told so.
- Under durable execution, everything one `computer` call does on the screen is one durable step, so replay and
  recovery don't repeat a finished call. A call cut off before its step was recorded is retried under the engine's
  settings for capability steps. On Temporal, set `activity_config` with a longer `start_to_close_timeout` and
  `RetryPolicy(maximum_attempts=1)`, and keep model retries with
  `model_activity_config=ActivityConfig(retry_policy=RetryPolicy())`, since the base config reaches model requests too.
  When the engine stops retrying, the model is told to take a screenshot first.
- Under Temporal, screenshots pass the 2 MB payload limit after a few steps. Configure external storage.

## See also

- https://pydantic.dev/docs/ai/harness/researcher/
- https://pydantic.dev/docs/ai/harness/exa-search/
- https://pydantic.dev/docs/ai/harness/youdotcom/
- https://pydantic.dev/docs/ai/harness/keenable-search/
- https://pydantic.dev/docs/ai/harness/browser-use/
- https://pydantic.dev/docs/ai/harness/playwright/
- https://pydantic.dev/docs/ai/harness/cdp-browser/
- https://pydantic.dev/docs/ai/capabilities/web-search/
- https://pydantic.dev/docs/ai/capabilities/web-fetch/
- https://pydantic.dev/docs/ai/harness/computer-use/
