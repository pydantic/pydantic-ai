# Network requests

This page lists every request Pydantic AI makes on its own initiative, what each one carries, and how to turn it off. It doesn't cover the requests you configure: to model providers, MCP servers, or from your own tools.

One of them, [the version check](#the-version-check), goes to a server run by Pydantic. Like any web server, it sees the IP address and `User-Agent` of each request, and Pydantic uses these in aggregate to understand which versions and platforms Pydantic AI is used on, and by what kind of organization. To turn it off, set `PYDANTIC_AI_NO_VERSION_CHECK=1`, or see [every way to turn it off](#turning-the-version-check-off).
<!-- TODO(DouweM): retention and IP handling wording, pending the proxylytics retention decision -->

## Model provider requests

Requests to model providers carry this header:

```text
User-Agent: pydantic-ai/{version}
```

For example, Pydantic AI 2.45.0 sends `User-Agent: pydantic-ai/2.45.0`.

## The version check

Whenever the banner is displayed, Pydantic AI checks `https://pydantic.dev/docs/api/versions` in a background thread. The endpoint lists the latest versions of Pydantic's packages, grouped by registry. Pydantic AI reads only `pydantic-ai` and `pydantic-ai-harness` from the `pypi` registry and ignores unknown registries, packages, and package fields. The result appears in the next banner, never the banner that starts the request.

The check runs at most once per 24 hours per machine. For an agent run, the banner is displayed on the first run in a process only when instrumentation is not configured and a terminal or coding agent is present. `clai` displays its intro banner when a terminal or coding agent is present, including when instrumentation is configured.

Neither banner is displayed:

- in CI;
- under `pytest`;
- when neither a terminal nor a coding agent is present; or
- when the banner is disabled.

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

The response is cached in `pydantic-ai/version-check.json` under the user cache directory: `$XDG_CACHE_HOME` or `~/.cache` on POSIX, and `%LOCALAPPDATA%` or `~/AppData/Local` on Windows. The file contains the time of the last check and the latest release versions:

```json
{"checked_at": 1750000000.0, "latest": {"pydantic-ai": "2.46.0", "pydantic-ai-harness": "0.8.0"}}
```

The cache contains no identifying information and is never sent to the server.

### Turning the version check off {#turning-the-version-check-off}

Set `PYDANTIC_AI_NO_VERSION_CHECK=1` to disable the check without hiding the banner. A non-empty `DO_NOT_TRACK` value other than `0` or `false` also disables it. Everything that prevents the relevant banner also prevents the check: `PYDANTIC_AI_NO_BANNER`, `pydantic_ai.BANNER_ENABLED = False`, CI, `pytest`, or the absence of both a terminal and a coding agent. Configured instrumentation prevents an agent run's banner and therefore its check, but does not prevent the `clai` intro banner.

## Model price updates

Pydantic AI updates model prices only when you call [`pydantic_ai.prices.update_in_background()`][pydantic_ai.prices.update_in_background]. It fetches the `genai-prices` data file from `https://raw.githubusercontent.com/pydantic/genai-prices/refs/heads/main/prices/new_data/v2/data.json` immediately and then hourly in the background.

## Web chat UI assets

[`Agent.to_web()`][pydantic_ai.Agent.to_web] and `clai web` download the web chat UI HTML from jsDelivr and cache it under the user cache directory. The HTML can load its stylesheet and application chunks from the same CDN.

To avoid this download, save the offline HTML locally and pass its path as `html_source` to `Agent.to_web()`, or use `clai web --html-source PATH`. See [Custom HTML Source](web.md#custom-html-source) for the download command and other source options.
