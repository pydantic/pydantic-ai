---
title: CDP Browser
description: "Give a Pydantic AI agent a throwaway Chrome tab it drives with raw Chrome DevTools Protocol commands, best paired with Code Mode for scraping and page automation."
---

# CDP Browser

`CDPBrowser` gives an agent a Chrome tab driven with raw
[Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/) (CDP) commands, through one
`browser_cdp` tool. Each run launches its own Chrome with a throwaway profile and closes it when the run ends.

Raw CDP is a low-level API: a click is several commands, and extracting data means evaluating JavaScript. Models do
best with it when they write code that calls `browser_cdp` in loops, so pair it with [Code Mode](code-mode.md). For
typed, one-action-per-call browsing, use [Playwright Browser](playwright.md) instead.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/cdp_browser/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Installation

The `cdp-browser` extra pulls in the websocket client. Chrome or Chromium is a separate install:

```bash
pip/uv-add "pydantic-ai-harness[cdp-browser]"
```

The capability finds Chrome in the usual install locations, or at `CHROME_PATH`. Pass `chrome_path` to choose the
binary yourself.

## Usage

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai_harness import CDPBrowser, CodeMode

agent = Agent('{{ anthropic:medium }}', capabilities=[CDPBrowser(), CodeMode()])
result = agent.run_sync('List the titles of the books in the Mystery category on books.toscrape.com.')
print(result.output)
```

`CDPBrowser` is a [capability](../capabilities/overview.md): it registers the `browser_cdp` tool, adds short
instructions on extracting data with `Runtime.evaluate`, and owns the Chrome process for the run.

## The tool

`browser_cdp(method, params, wait_for_event, timeout)` sends one CDP command to the tab and returns its result.
With `wait_for_event`, such as `'Page.loadEventFired'`, it waits for that event after the command and returns
`{'result': ..., 'event': ...}`. `timeout` (30 seconds by default) bounds the command and the event together.

- Only the commands in `allowed_methods` can be sent; anything else is a retry naming the allowed ones. The
  defaults, `DEFAULT_ALLOWED_CDP_METHODS`, read and interact with the page through `Page`, `Runtime`, `DOM` and
  `Input` commands. They leave out `Browser.*` and `Target.*`, which reach beyond the tab, `Fetch.*` and
  `Network.*`, which could lift the egress guard, and `DOM.setFileInputFiles` and `Input.dispatchDragEvent`,
  which take host file paths.
- `Page.navigate` only opens `http://` and `https://` URLs, and a URL the egress policy refuses is a retry that
  names the reason.
- `Page.captureScreenshot` returns the image as image content, so a vision model sees the page rather than a base64
  string. A capture over 5 MB comes back as an error asking for a smaller one.
- A command that does not answer in time is a retry. The page's running script is stopped with
  `Runtime.terminateExecution` first, so a page stuck in `while (true) {}` answers the next command.
- A protocol error, such as an unknown node id, is a retry carrying Chrome's message.
- Page dialogs (`alert`, `confirm`, `prompt`) are dismissed as they open, since an open dialog blocks every
  command. A `beforeunload` prompt is accepted, so the navigation that raised it goes through.
- Downloads are denied.
- Results come back whole. Under Code Mode they stay in the script; without it, add
  [Tool Output Limits](tool-output-limits.md) to cap what reaches the model.

The tool allows three retries in a row, since CDP mistakes are common and the model corrects them from the message.

## Options

| Option | Default | Purpose |
|---|---|---|
| `headless` | `True` | Run Chrome without a window. A headed run shares the desktop's clipboard, and the model can paste from it. |
| `chrome_path` | `None` | The Chrome or Chromium binary. `None` reads `CHROME_PATH`, then searches the usual install locations. |
| `chrome_sandbox` | `True` | Run Chrome with its renderer sandbox. Turn it off only where the sandbox cannot start (see [Limitations](#limitations)). |
| `allowed_methods` | `DEFAULT_ALLOWED_CDP_METHODS` | The CDP commands the model may send. |
| `allowed_domains` | `None` | Egress allowlist for navigation and data requests; `None` allows every public host (see [Egress](#egress-and-ssrf)). |
| `block_private_addresses` | `True` | Refuse private, loopback, link-local and other reserved addresses, written as an IP or reached through a hostname that resolves to one. |
| `policy` | `None` | Full `EgressPolicy`, for rules the two shorthands cannot express. Mutually exclusive with them. |

## Two of them

`CDPBrowser` declares the default `id` `'cdp_browser'`. Two `CDPBrowser`s on one agent that share it are one
browser stated twice: they become one when they agree, and the agent raises a `UserError` when it's built if they
differ, since merging them could widen the commands or the egress one of them was given. To keep two, give each its
own `id` and wrap them in [PrefixTools](../capabilities/prefix-tools.md), since both register `browser_cdp`.

## Egress and SSRF

`CDPBrowser` takes the same egress settings as [Playwright Browser](playwright.md#egress-and-ssrf), and the same
`EgressPolicy` object. By default every public host is reachable, and private, loopback, link-local and other
reserved addresses are refused, including `169.254.169.254`, the cloud metadata endpoint. A hostname is resolved
first, so a name pointing at one of those addresses is refused too, as is one whose lookup does not answer within
two seconds.

`allowed_domains` limits top-level navigation and the requests page scripts use to move data (`fetch`, XHR,
EventSource, `sendBeacon`, WebSocket) to the listed hosts and their subdomains. Passive subresources and sub-frame documents
stay reachable, so a permitted page keeps its assets and its identity-provider frames. The private-address block
applies to every request. For a denylist, apex-only matching or other kinds of request, pass an `EgressPolicy`:

```python
from pydantic_ai_harness.cdp_browser import CDPBrowser, EgressPolicy

browser = CDPBrowser(policy=EgressPolicy(allowed_domains=['example.com'], blocked_domains=['ads.example.com']))
```

The guard sits in Chrome itself. Every target Chrome opens, the tab and its out-of-process frames, popups and
workers, is held until CDP's `Fetch` interception is on for it, and each request it makes is paused until the
policy allows or refuses it. That covers navigations a page script starts and requests the page makes on its own,
not only `Page.navigate`. A refused request fails with `net::ERR_BLOCKED_BY_CLIENT`, and pages cannot register
service workers. A target whose requests cannot be intercepted is never resumed, so it makes none.

What the guard does not cover:

- DNS rebinding on HTTP requests: Chrome resolves a name again before it connects, so a record that changes between
  the two lookups gets past the block.
- Speculative connections: Chrome can open a TCP connection to a refused host before the request is refused,
  without sending it.
- Traffic other than HTTP requests and WebSockets: block it with an egress firewall.

Set `allowed_domains` when the agent acts on untrusted input, and run it on a network where these gaps are
acceptable.

### WebSockets

`Fetch` interception does not pause WebSocket handshakes, so Chrome is launched with a local proxy that carries
every WebSocket and nothing else. The proxy applies the same policy to the socket's host and port, and connects to
the addresses it checked. A socket it refuses fails to connect. A custom `EgressPolicy.refuse` sees the socket as
`ws://host:port/`, without its real scheme or path. Chrome's own proxy settings are replaced, so a system proxy is
not used.

## Lifecycle

Chrome launches when the run starts, before the first model request, and the run fails if it cannot start. Unlike
Playwright Browser, a run that never calls the tool still launches one. When the run ends, Chrome is asked to close,
its process group is killed, and its profile directory is removed once Chrome's processes release it. Concurrent runs each get their own Chrome.

An agent with a durability capability (such as [Temporal](../durable_execution/temporal.md)'s `TemporalDurability`) is
rejected at construction, and so is wrapping the agent in a deprecated wrapper agent such as `TemporalAgent`: a live
browser cannot survive activity replay or a worker restart.

`CDPBrowser` emits no spans of its own. Core's tool-call spans record each `browser_cdp` call, with its arguments and
result under the usual content settings. Requests the egress policy refuses get no span: Chrome reports them on its
event stream outside any tool call, and a refused navigation already shows in its tool result as
`net::ERR_BLOCKED_BY_CLIENT`.

## Choosing a browser capability

| Capability | What the model gets | Best with |
|---|---|---|
| CDP Browser | One `browser_cdp` tool that sends raw CDP commands | Code Mode, which calls the tool from Python |
| [Playwright Browser](playwright.md) | 18 typed tools: navigate, snapshot, click, type, screenshot, and more | Direct tool calls, one action per call |
| [Browser Use](browser-use.md) | One `browse_web` tool that hands a goal to a browser-use agent | Open-ended tasks the host model should not drive step by step |

A benchmark run on 2026-11-23 compared the raw-CDP approach, Playwright Browser, and browser-use's own agent loop:

| Harness | Score | Pass rate | Cost per run | Median time | Mean input tokens |
|---|---|---|---|---|---|
| Raw CDP under Code Mode | 0.99 | 97% | $0.036 | 39s | 116k |
| browser-use agent loop | 0.98 | 97% | $0.042 | 56s | 134k |
| Playwright Browser under Code Mode | 0.99 | 99% | $0.050 | 54s | 207k |

The three were equally accurate: the gap between them is within the 2 to 4 point spread between repeats. The
raw-CDP harness was the cheapest and the fastest, and read about 44% fewer input tokens than Playwright Browser.

How it was run: 270 runs, the three harnesses on 15 scraping tasks (5 easy, 10 hard ones that ask for full datasets
as files) with three models, DeepSeek V4.1 Flash, GPT-6.1 Terra and GPT-6.1 Sol, twice each. The raw-CDP harness
was the `browser_cdp` tool this capability was ported from, with the same instructions. Playwright Browser ran under
Code Mode too, with the same instructions minus the CDP advice, so only the browser tools differed. browser-use ran
its own loop, not this package's [Browser Use](browser-use.md) capability. Claude Opus 5.5 graded each run against
answers computed without a browser, giving a score from 0 to 1 with partial credit; the pass rate is the share of
runs it graded fully correct. Cost comes from each run's token counts priced with
[genai-prices](https://github.com/pydantic/genai-prices), the same way for every harness.

## Limitations

- `chrome_sandbox=True` needs a sandbox Chrome can start. Where it cannot, such as a container without the kernel
  privileges it needs, or a Linux host that restricts unprivileged user namespaces for a Chromium build without an
  AppArmor profile of its own, Chrome exits at launch with "No usable sandbox!". `chrome_sandbox=False` runs it
  without one, and a renderer exploit then runs with the agent's own access.
- There is one tab: the model cannot read a popup the page opens, though the egress guard covers it.
- Uploads and downloads are not available: downloads are denied, and the commands that take host file paths are
  left out of the defaults.
- On Windows, which has no process groups, teardown kills Chrome's own process only.
- The capability does not bound Chrome's memory or CPU use. Where that matters, run it under container or cgroup
  limits.
- Chrome's DevTools endpoint listens on a 127.0.0.1 port for the whole run and takes no authentication. Any local
  process can drive the browser through it, outside `allowed_methods`, the egress policy and the download denial.
  Don't run it on a host shared with untrusted users, and don't give the same agent a tool that reaches the host's
  loopback, such as `Shell` on `BubblewrapWorkspace(network=True)`. The pipe transport is tracked in
  [#14329](https://github.com/pydantic/pydantic-ai/issues/14329).
- It cannot attach to a browser that is already running; that is tracked in
  [#14246](https://github.com/pydantic/pydantic-ai/issues/14246).

## API reference

::: pydantic_ai_harness.cdp_browser.CDPBrowser

::: pydantic_ai_harness.cdp_browser.CDPBrowserToolset
