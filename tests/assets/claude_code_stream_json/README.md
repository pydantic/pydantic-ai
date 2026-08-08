# Claude Code CLI stream-json golden fixtures

Captured from the real Claude Code CLI, version **2.1.222**, on 2026-08-06 (macOS). The
stream-json format is not a versioned public spec, so these fixtures ARE the spec for
`ClaudeCodeEventStream`: schema tests validate our emitted lines against the shapes observed here.

## Capture method

Each fixture is the verbatim stdout of a run of:

```
claude -p '<prompt>' --model <model> --settings '{"disableAllHooks": true}' \
  --output-format stream-json --verbose [--allowedTools ...] [--max-turns 1] [--include-partial-messages]
```

run in an empty scratch directory (containing only `readme-fixture.md`, a 3-line markdown file)
so no project instructions pollute the stream. `disableAllHooks` suppresses user-configured
hooks, which otherwise interleave `system`/`hook_started` + `hook_response` records.

| fixture | model | scenario |
|---|---|---|
| `01-plain-text.jsonl` | haiku | plain text response |
| `02-tool-use-read.jsonl` | haiku | one `Read` tool use + tool result |
| `03-multi-turn-tools.jsonl` | haiku | `Write` then `Read` tool loop (two tool results) |
| `04-error-max-turns.jsonl` | haiku | `--max-turns 1` exceeded -> `result` subtype `error_max_turns` |
| `05-thinking-sonnet.jsonl` | sonnet | "think hard" prompt (note: produced no thinking block; the haiku runs did) |
| `06-plain-partial.jsonl` | haiku | plain text with `--include-partial-messages` |
| `07-tool-use-partial.jsonl` | haiku | tool use with `--include-partial-messages` |
| `08-error-auth.jsonl` | haiku | API auth failure (run under an unauthenticated config dir) |

## Format observations (v2.1.222)

- One `assistant` line is emitted **per content block** (thinking, text, tool_use each get their
  own line), all sharing one `message.id`; `message.usage` is repeated on each line.
- Tool results arrive as `user` lines whose `message.content` holds `tool_result` blocks.
- `--include-partial-messages` is a **superset**: it interleaves `stream_event` lines (raw
  Anthropic SSE shapes: `message_start`, `content_block_start`/`delta`/`stop`, `message_delta`,
  `message_stop`) while still emitting the whole-block `assistant` lines and the `result` line.
- The CLI also emits record types beyond the classic four: `rate_limit_event`,
  `system`/`status`, `system`/`thinking_tokens` (and `system`/`hook_started` +
  `hook_response` when hooks are enabled).
- On API errors the `result` record keeps `subtype: 'success'` but sets `is_error: true` and
  `terminal_reason`; the accompanying synthetic `assistant` line has `model: '<synthetic>'` and
  an `error` field. Exceeding `--max-turns` does change the subtype (`error_max_turns`).

## Scrubbing applied

- Absolute capture paths normalized (`cwd`, `memory_paths`, tool inputs/outputs) to
  `/tmp/claude-fixtures` / `/home/user`.
- The `system`/`init` records' `tools`, `slash_commands`, `skills`, `agents`, `mcp_servers` and
  `plugins` lists were replaced with the defaults from a pristine (default-config) run, since the
  originals enumerated the capturing user's personal setup. Shapes are unchanged.
- Session ids, uuids, message ids, timestamps, usage numbers are as captured.
