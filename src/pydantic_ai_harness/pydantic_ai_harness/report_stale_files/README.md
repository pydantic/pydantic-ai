# Report Stale Files

Tell the model when a file it read or wrote has since changed or been deleted in the workspace, so it doesn't
reason from, or edit against, an out-of-date copy.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/report_stale_files/)

A coding agent reads `src/config.py`, makes several more tool calls, and meanwhile the user, another agent, or a
build step changes or deletes the file. The model can't see that, so it keeps working from what it read.
`FileSystem`'s `expected_hash` rejects a stale write, but only once the model writes, and only when it passes the
hash. `ReportStaleFiles` tells the model before its next request instead.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import FileSystem, ReportStaleFiles

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[LocalWorkspace('.'), FileSystem(), ReportStaleFiles()],
)
```

## How it works

- It records the content hash of every file a [`FileSystem`](../file_system/) tool reads or writes, from the
  `FileReadEvent` and `FileWrittenEvent` it emits. A file tool of your own takes part by emitting the same events.
  The model's own writes and edits refresh the record, so they are not reported.
- Before each model request, it reads the recorded files through the run's workspace (`ctx.workspace`), so it works
  the same on a local directory, a remote sandbox, or a filesystem-only workspace. A file whose bytes no longer
  match its record is named in a request-scoped system prompt:

  ```text
  These files changed in the workspace after you last read them, so what you saw of them is out of date. Read them again before relying on them or editing them:
  - `src/config.py`
  ```

- A changed file is named on every request until the model reads it again. A deleted file is named once, then
  forgotten. A file that can't be read for another reason, such as a permission error or a workspace failure, is
  skipped and checked again on the next request.
- The notice is request-scoped: the model sees it for the request it is sent with, and it stays in the message
  history without moving the cached prefix (see
  [request-scoped system prompts](https://pydantic.dev/docs/ai/core-concepts/message-history/#request-scoped-system-prompts)).

## Options

| Option | Default | What it does |
|---|---|---|
| `max_tracked_files` | `100` | How many files to check before each request. Past it, the least recently read or written file is dropped. |
| `max_reported_files` | `20` | How many paths one notice names. The rest are counted (`- and 3 more`). |

Every check reads each recorded file in full, which detects a rewrite that keeps the size, but costs one read per
file per request. On a remote workspace each read is a round trip, so lower `max_tracked_files` if requests slow
down.

## Scope

- **Observations are in-process memory, per conversation.** They are held on the instance the agent was built
  with and keyed by the run's `conversation_id`, so a run continued from an earlier run's `message_history` is told
  about changes made between the runs. A conversation continued through another instance, process, or worker
  starts with no records. An instance keeps the 1,024 most recently used conversations.
- **Paths are named only when they are plain.** The notice is in system voice, so a path is named only when it
  holds letters, digits and `._/@+-`, up to 200 characters. Other paths, such as one with a space, are counted
  with the rest.
- **It sees only files reported through the events.** A file the model read with a shell command is not recorded.

## Telemetry

`ReportStaleFiles` emits no spans of its own. Each notice is a system prompt in the request it is sent with, so it
appears in the model request span that Pydantic AI's instrumentation records.
