---
title: Announce Current Time
description: "Tell a Pydantic AI agent today's date, or the current hour, in a time zone you pick, without breaking the prompt cache."
---

# Announce Current Time

Tell the model today's date, or the current hour, in a time zone you pick. Use it when an agent answers questions that depend on when they are asked, such as "what's due this week", which a model can't work out from its training data.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```bash
pip/uv-add pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import AnnounceCurrentTime

agent = Agent('{{ anthropic:large }}', capabilities=[AnnounceCurrentTime(timezone='Europe/Amsterdam')])
```

The model's instructions then include a line such as `Today's date is Friday, 2026-01-02 (Europe/Amsterdam).`

| Field | Default | Meaning |
|---|---|---|
| `timezone` | `datetime.UTC` | An IANA time zone name such as `'Europe/Amsterdam'`, or a `tzinfo`. A name is looked up in the system time zone database, or the `tzdata` package where there is none (Windows, minimal containers). An unknown name raises `ZoneInfoNotFoundError` when the capability is built |
| `precision` | `'date'` | `'date'` states the weekday and date; `'hour'` states the hour as well: `The current time is within the hour starting 2026-01-02T14:00+01:00 (Friday, Europe/Amsterdam).` |
| `id` | `'current_time'` | Keys the instruction block, so a change can be appended to the message history. `None` or `''` means the default |

In an [agent spec](../agent-spec.md), `timezone` takes a zone name: `- AnnounceCurrentTime: {timezone: Europe/Amsterdam}`, loaded with `custom_capability_types=[AnnounceCurrentTime]`.

## Prompt caching

The time is read from the timestamp the run gives the request it is about to send. When the date (or hour) moves on during a conversation, the new value is [appended](../agent.md#appending-instruction-changes) to the message history instead of rewriting the instructions, so the provider can keep serving the conversation so far from its prompt cache. With `precision='date'` that is at most one update per day; with `'hour'`, one per hour of conversation. There is no finer precision: text that changes on every request belongs in the user prompt or a tool.

## Durable execution

The capability reads no clock of its own, only the request's timestamp, so it is safe in workflow code under [durable execution](../durable_execution/overview.md), where Temporal's sandbox rejects clock reads.

## Telemetry

`AnnounceCurrentTime` emits no spans of its own. The text it adds is part of the request's instructions, which an [instrumented](../logfire.md) agent records on the model request span.

## Composition

Two `AnnounceCurrentTime` on one agent share the default `id` and merge into one, with the later one's settings. To state the time in two zones, give the second its own `id`.

## Renamed from `CurrentTime`

This capability was `CurrentTime`, in `pydantic_ai_harness.current_time`. It was renamed to follow the harness naming convention, which names a capability that acts on the run with a verb phrase. The old name still imports, from `pydantic_ai_harness` and `pydantic_ai_harness.current_time`, with a `HarnessDeprecationWarning`, until a future release removes it. Its default `id` is still `current_time`, so conversations recorded before the rename keep their instruction key.

Specs saved before the rename use the `CurrentTime` block name. To keep loading them, pass the deprecated `CurrentTime` class in `custom_capability_types`, alongside or instead of `AnnounceCurrentTime`: its class name is the block name. To migrate a spec, rename its `CurrentTime` block to `AnnounceCurrentTime`; the options are unchanged.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/announce_current_time/).

## API reference

::: pydantic_ai_harness.AnnounceCurrentTime
