# Network Requests

This page lists every request Pydantic AI makes on its own initiative, what each one carries, and how to turn it off. It doesn't cover the requests you configure: to model providers, MCP servers, or from your own tools.

One of them, [the version check](#the-version-check), goes to a server run by Pydantic, which uses its `User-Agent` in aggregate to understand which versions and platforms Pydantic AI is used on. It only runs where a banner is displayed (at a terminal or for a coding agent, never in CI or under `pytest`), which in practice means a developer's machine, and it [can be turned off](#turning-the-version-check-off) with one environment variable.

## Model provider requests

Requests to model providers carry this header by default:

```text
User-Agent: pydantic-ai/{version}
```

For example, Pydantic AI 2.45.0 sends `User-Agent: pydantic-ai/2.45.0`. [Bedrock](models/bedrock.md) requests are the exception: `botocore` sets its own `User-Agent`.

## The version check

When the [first-run banner](logfire.md#the-first-run-banner) or the intro banner of [`clai`](cli.md) is displayed, Pydantic AI checks `https://info.pydantic.dev/versions.json` in a background thread for newer releases of Pydantic AI and Pydantic AI Harness. A deployed service with no terminal attached never displays them, so never makes the request. `clai` displays its banner under the same conditions as the first-run banner, except that configured instrumentation doesn't hide it.

The first banner only announces the check, and says how to turn it off. The request is made from the next banner on, at most once per 24 hours per user cache directory, and a newer release it finds appears in the banner after that. A container started with a fresh home directory every time therefore never makes the request.

The request has no query parameters or body. Its only explicitly configured header is a pip-style `User-Agent`, for example:

```text
User-Agent: pydantic-ai/2.45.0 (Python 3.13.5; Linux; x86_64) pydantic-ai-harness/0.8.0 genai-prices/0.1.6 agent/codex
```

It contains:

- the installed Pydantic AI version;
- the Python version, operating system, and machine architecture;
- the installed Pydantic AI Harness and `genai-prices` versions, when present; and
- the coding agent name when it is recognized, or the generic name `agent` otherwise.

It does not contain an identifier, model or provider names, prompts, tool or capability names, paths, hostnames, or usernames. Unrecognized coding-agent names provided through the environment are never sent.

The response is cached in `pydantic-ai/version-check.json` under the user cache directory: `$XDG_CACHE_HOME` or `~/.cache` on POSIX, and `%LOCALAPPDATA%` or `~/AppData/Local` on Windows. The file contains the time of the last check (`null` until the first) and the latest release versions:

```json
{"checked_at": 1750000000.0, "latest": {"pydantic-ai": "2.46.0", "pydantic-ai-harness": "0.8.0"}}
```

The cache contains no identifying information and is never sent to the server.

### Turning the version check off {#turning-the-version-check-off}

Set `PYDANTIC_AI_NO_VERSION_CHECK` to any value (for example, `PYDANTIC_AI_NO_VERSION_CHECK=1`) to turn the check off without hiding the banner. A non-empty `DO_NOT_TRACK` value other than `0` or `false` also turns it off. Anything that keeps the banner from being displayed, as described in [the first-run banner](logfire.md#the-first-run-banner), also keeps the check from running.

## Model price updates

Pydantic AI updates model prices only when you call [`pydantic_ai.prices.update_in_background()`][pydantic_ai.prices.update_in_background]. It fetches the `genai-prices` data file from `https://raw.githubusercontent.com/pydantic/genai-prices/refs/heads/main/prices/new_data/v2/data.json` immediately and then hourly in the background.

## Web chat UI assets

[`Agent.to_web()`][pydantic_ai.Agent.to_web] and `clai web` download the web chat UI HTML from jsDelivr and cache it under the user cache directory. The HTML can load its stylesheet and application chunks from the same CDN.

To avoid this download, save the offline HTML locally and pass its path as `html_source` to `Agent.to_web()`, or use `clai web --html-source PATH`. See [Custom HTML Source](web.md#custom-html-source) for the download command and other source options.
