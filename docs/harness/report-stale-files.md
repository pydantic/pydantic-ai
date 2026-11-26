---
title: Report Stale Files
description: "Tell a Pydantic AI agent when a workspace file it read or wrote has since changed or been deleted, before it reasons from or edits an out-of-date copy."
---

# Report Stale Files

Tell the model when a file it read or wrote has since changed or been deleted in the workspace, so it doesn't reason from, or edit against, an out-of-date copy.

A coding agent reads `src/config.py`, makes several more tool calls, and meanwhile the user, another agent, or a build step changes or deletes the file. The model can't see that, so it keeps working from what it read. [`FileSystem`](filesystem.md)'s `expected_hash` rejects a stale write, but only once the model writes, and only when it passes the hash. `ReportStaleFiles` tells the model before its next request instead.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem, ReportStaleFiles

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[LocalWorkspace('.'), FileSystem(), ReportStaleFiles()],
)
```

## How it works

- It records the content hash of every file a [`FileSystem`](filesystem.md) tool reads or writes, from the `FileReadEvent` and `FileWrittenEvent` it emits. A file tool of your own takes part by emitting the same events. The model's own writes and edits refresh the record, so they are not reported.
- Before each model request, it reads the recorded files through the run's [workspace](../workspace.md) (`ctx.workspace`), so it works the same on a local directory, a remote sandbox, or a filesystem-only workspace. A file whose bytes no longer match its record is named in a request-scoped `StaleFilesPart`, a [capability part](../capabilities/custom.md#capability-parts):

    ```text
    These files changed in the workspace after you last read or wrote them, so what you saw of them is out of date. Read them again before relying on them or editing them:
    - `src/config.py`
    ```

- A changed file is named on every request until the model reads it again. A deleted file is named once, then forgotten. A file that can't be read for another reason, such as a permission error or a workspace failure, is skipped and checked again on the next request. A file whose size the workspace doesn't report is not read, and one that grew past `max_checked_bytes` between its `stat` and its read is forgotten until the model reads it again. A file replaced by a directory counts as deleted.
- Paths are named the way the file tools take them: relative to the workspace's working directory, or absolute when outside it. The most recently read or written file comes first.
- The notice is request-scoped: the model sees it for the request it is sent with, and it stays in the message history without moving the cached prefix (see [request-scoped system prompts](../message-history.md#request-scoped-system-prompts)).
- The paths are named by whoever can write to the workspace, so the notice has the [authority](../workspace.md#workspace-authority) of the workspace's files: in the application's voice by default, and framed as data for a workspace with less, such as `LocalWorkspace('.', authority='none')`. A path that looks like a framework frame, such as one holding `<system>`, is counted rather than named. Find notices in history with `isinstance(part, StaleFilesPart)`; its `changed` and `deleted` fields hold the paths.

## Options

| Option | Default | What it does |
|---|---|---|
| `max_tracked_files` | `100` | How many files to record per conversation. Past it, the least recently read or written file is dropped. |
| `max_reported_files` | `20` | How many paths one notice names. The rest are counted (`- and 3 more`). |
| `max_checked_bytes` | `8 << 20` (8 MiB) | How many bytes of recorded files to read before one request, most recently read or written first. A file that doesn't fit in what is left is not checked on that request. |

Every check reads each recorded file in full, which detects a rewrite that keeps the size. Before every request, each recorded file costs a `realpath`, a `stat` and a read, one file after another, even when nothing changed. On a remote workspace each is a round trip, and on a backend without native `realpath` that one runs a shell command, so with the defaults a remote sandbox can pay up to 300 sequential round trips per request. `max_checked_bytes` bounds the bytes read; lower `max_tracked_files` if requests slow down.

## Scope

- **Observations are in-process memory, per conversation.** They are held on the instance the agent was built with and keyed by the run's `conversation_id`, so a run continued from an earlier run's `message_history` is told about changes made between the runs. A conversation continued through another instance, process, or worker starts with no records, and a run in another workspace than the one a file was read in doesn't check that file. An instance keeps the 1,024 most recently used conversations.
- **It sees only files reported through the events.** A file the model read with a shell command is not recorded.
- **A check doesn't follow a path that now leads elsewhere.** A recorded file whose real path changed since it was read, such as one replaced by a symlink to a file the file tools would refuse, is forgotten rather than read, until the model reads it again.
- **Durable execution is refused.** Under a durability capability such as `TemporalDurability`, the run fails at its start: the records are process memory, which a replay or another worker does not have.

## Telemetry

`ReportStaleFiles` emits no spans of its own. Each notice is part of the request it is sent with, so it appears in the model request span that Pydantic AI's [instrumentation](../capabilities/instrumentation.md) records.

## API reference

- [`pydantic_ai_harness.report_stale_files` source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_stale_files/)

::: pydantic_ai_harness.ReportStaleFiles

::: pydantic_ai_harness.report_stale_files.StaleFilesPart
