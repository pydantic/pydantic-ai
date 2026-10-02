---
title: Retry Policy
description: "Retry a tool's transient failures (rate limits, timeouts, provider errors) with exponential backoff; only tools you mark safe to re-run are retried."
---

# Retry Policy

`RetryPolicy` retries a tool's transient failures with exponential backoff: rate limits,
timeouts, connection drops, and provider errors reported with a retryable status code. A tool
that fails this way is re-run instead of failing the run, with a bounded sleep between attempts.

It retries nothing by default, because retrying re-runs the tool's function. A tool that already
wrote a file or sent a message must not simply run again; only tools you mark safe to re-run are
retried.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/retry_policy/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

`RetryPolicy` needs no extra beyond the base package:

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import RetryPolicy

agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[
        RetryPolicy(
            allow_idempotent_retries=True,
            idempotent_tools=frozenset({'web_search'}),  # retry only this tool
            max_retries=3,
            backoff_factor=0.5,  # delays of 0.5s, 1s, 2s, capped by max_backoff
        )
    ],
)
```

The example sets the two pieces that enable retries. With the defaults (`allow_idempotent_retries=False`
and an empty `idempotent_tools`), no tool call is retried: see [Why retries are opt-in](#why-retries-are-opt-in).

## Why retries are opt-in

Retrying a tool call re-runs its handler. For a tool with side effects (a file write, a charge, a
sent message), the second run can apply the side effect again. To retry a tool, mark it safe to
re-run:

1. Set `allow_idempotent_retries=True` on the capability.
2. List the tool's name in `idempotent_tools`, or set `idempotent: True` in its `tool_overrides`
   entry.

Only tools marked that way are retried, up to their `max_retries`. A transient failure on any
other tool is surfaced immediately, with `on_failure` called first if you set it.

A generic wrapper cannot know whether re-running a tool is safe; the tool author does. For a tool
whose flakiness is known and localized, retrying just the failing call inside the tool (with
[tenacity](https://tenacity.readthedocs.io/), for example) is often the better shape; see
[Tool retries](../retries.md#tool-retries) in the retries guide. `RetryPolicy` is for keeping that
policy out of the tool, in one place, across many tools.

## Which failures trigger a retry

A tool call is retried when the exception it raises matches any of:

- an instance of one of `retryable_exceptions` (default: `TimeoutError`, `ConnectionError`)
- an HTTP error whose status code is in `retryable_status_codes` (default: `429`, `500`, `502`,
  `503`, `504`); the code is read from the exception's `status_code` attribute, or from
  `status_code` on its `.response`
- an error whose `error_type` attribute is `rate_limit`, `timeout`, or `server_error`

Any other exception propagates immediately, unmodified. Model requests are not touched: retries
for those live at the transport and provider layers, see the
[retries guide](../retries.md).

## Backoff

Between attempts, the capability sleeps for `backoff_factor * 2**attempt` seconds (with the
default `backoff_factor=0.5`: 0.5s, 1s, 2s) plus jitter of up to 25% of the delay, in either
direction. Every delay is capped at `max_backoff` (default `30.0` seconds), with a minimum of
`0.01` seconds unless `max_backoff` is smaller, so the backoff is bounded no matter how many
attempts elapse. `backoff_factor` and `max_backoff` must be finite numbers greater than zero; a
zero, negative, or non-finite value raises `ValueError` at construction, in the top-level fields
and in per-tool `tool_overrides` entries alike.

Each retry logs a warning naming the tool, the attempt, and the delay.

## Options

| Field | Default | Purpose |
|---|---|---|
| `max_retries` | `3` | Retry attempts after the first; `0` disables retries for that tool |
| `backoff_factor` | `0.5` | Base delay in seconds; doubles with each attempt |
| `max_backoff` | `30.0` | Upper bound on the delay between attempts, in seconds |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | HTTP status codes that trigger a retry |
| `retryable_exceptions` | `(TimeoutError, ConnectionError)` | Exception types that trigger a retry |
| `allow_idempotent_retries` | `False` | The retry gate; see [Why retries are opt-in](#why-retries-are-opt-in) |
| `idempotent_tools` | `frozenset()` | Tool names safe to retry once the gate is open |
| `tool_overrides` | `{}` | Per-tool values for the fields above, plus `idempotent`, `on_retry`, `on_failure`; unknown keys raise `ValueError` |
| `on_retry` | `None` | Called as `on_retry(tool_name, attempt, exc)` before each retry |
| `on_failure` | `None` | Called as `on_failure(tool_name, exc)` when a transient failure is surfaced |

## Per-tool overrides

`tool_overrides` maps a tool name to a dict of field values that replace the top-level ones for
that tool alone:

```python
RetryPolicy(
    allow_idempotent_retries=True,
    tool_overrides={
        'web_search': {'max_retries': 5, 'backoff_factor': 1.0, 'idempotent': True},
        'shell': {'max_retries': 0},  # shell commands are rarely idempotent
    },
)
```

`idempotent: True` in an override marks that tool safe to retry without listing it in
`idempotent_tools`. Unknown override keys raise `ValueError` at construction, so a typo like
`max_retry` fails when the policy is built rather than doing nothing at run time.

## Composing

Multiple policies nest independently and can multiply retry attempts for overlapping tools.
Prefer one policy with per-tool overrides.

## API reference

::: pydantic_ai_harness.retry_policy.RetryPolicy
