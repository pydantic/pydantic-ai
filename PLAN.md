# TDD spec: Claude Code stream-json event stream for `pydantic_ai.ui`

Originally written 2026-08-06 in the harness `github-aw-pyai-engine` session as a handoff doc;
moved into this worktree (branch `claude-code-event-stream`) and committed for version control the
same day, with the review addendum at the bottom. Questions about intent go to David.

## Goal

A `UIEventStream` subclass in pydantic-ai core that maps a Pydantic AI agent's native run events to
**Claude Code CLI stream-json** lines (the JSONL that `claude -p --output-format stream-json
--verbose` writes to stdout). Output direction only. Open as a **DRAFT PR** -- it frames a design
conversation with Douwe (he proposed this seam; he has not yet blessed the exact public API).

## Why (context you'd otherwise lack)

- Pydantic is building a headless coding-agent CLI in pydantic-ai-harness to act as a GitHub
  Agentic Workflows (gh-aw) engine. Douwe wants the CLI "drop-in compatible with Claude code, or at
  least having such a mode".
- The compat mode has a concrete payoff in gh-aw: running as `engine: claude` with only `command:`
  overridden gives us gh-aw's Claude log parser, step-summary rendering, and token metrics for free
  -- third-party engines get none of that. Proven end-to-end by Bill's shim:
  https://github.com/strawgate/gh-aw/pull/4
- Douwe's design direction: implement "as a uiadapter, partially" -- the event mapping belongs in
  `pydantic_ai.ui` beside the AG-UI and Vercel AI adapters; the HTTP-shaped input half does not apply
  to a CLI. His earlier framing: "take any agent and with a couple lines of code make it look like
  Claude code."

## Read before designing

0. **Bill's shim in https://github.com/strawgate/gh-aw/pull/4** -- it already made most of the
   event-mapping decisions once and proved them against gh-aw end-to-end. Read it as a reference
   implementation, not just as evidence the approach works.
1. `pydantic_ai/ui/_event_stream.py` -- `UIEventStream`: ONE abstract method (`encode_event -> str`),
   ~25 optional `handle_*` hooks (text/thinking/tool-call starts/deltas/ends, `handle_run_result`,
   error-path part closing). It maps `AgentStreamEvent | AgentRunResultEvent` -> your `EventT`.
   `encode_event` returns a plain `str` -- SSE is NOT baked in; JSONL is legal.
2. `pydantic_ai/ui/vercel_ai/` and `pydantic_ai/ui/ag_ui/` -- the two merged exemplars (both
   web-chat-shaped; yours is the first stdout-shaped one).
3. `docs/ui/overview.md` -- note "Advanced Usage" documents the non-Starlette direct-call path.
4. PR https://github.com/pydantic/pydantic-ai/pull/5223 (`Agent.to_responses()`, David's, still
   OPEN) -- Douwe once cited it as the pattern. **Find out why it stalled before copying its shape**;
   report what you find to David before locking the API.
5. gh-aw's parser -- the compat oracle: `actions/setup/js/parse_claude_log.cjs` in
   https://github.com/github/gh-aw (pin the commit you vendor).

## Scope

**In:**
- `ClaudeCodeEventStream` (module placement + exact naming: propose, flag for Douwe) with the full
  event mapping: init/system record, assistant text (+ thinking if representable), `tool_use` /
  `tool_result` blocks, and the terminal `result` record carrying usage (and cost if available from
  `AgentRunResult`).
- JSONL encoding (`json.dumps(...) + '\n'`), transport left to the caller.
- Docs page section + tests per repo standards.

**Out (deliberate, do not drift into):**
- The CLI entry point, argv parsing, capability composition -- harness work, separate track.
- Session persistence / `--resume` semantics -- no counterpart in the seam yet.
- Capability auto-injection ("use the adapter, get claude-code capabilities") -- harness.
- The Anthropic-Messages-API HTTP shim -- future sibling reusing this event stream.
- Any `UIAdapter` input-half (`build_run_input`) implementation.

## TDD method -- fixtures first, oracle second

1. **Capture golden fixtures from the real Claude Code CLI** before writing any mapping code:
   `claude -p "<prompt>" --output-format stream-json --verbose` for at least: plain text response;
   response with tool use (e.g. a file read) + tool result; multi-turn tool loop; an error/refusal;
   check what a thinking-enabled run emits. Record the exact `claude --version` in the fixture dir;
   the format is not a versioned public spec, so fixtures ARE the spec. Store as committed test data.
2. **Schema tests**: our emitted lines validate against the shapes observed in fixtures (field
   names, nesting, the `type` discriminators, the `result` record's usage fields).
3. **Oracle test (the acceptance bar)**: vendor gh-aw's `parse_claude_log.cjs` (pinned commit) into
   test assets; run it (node) over a log produced by `ClaudeCodeEventStream` from a `TestModel` run;
   assert it extracts non-zero turns, token usage, and the tool-call list. If their parser reads our
   stream, compat is proven rather than claimed. Mark the test appropriately if node isn't
   available in some CI lane -- but it must run in at least one.
4. Unit tests for the mapping hooks per repo conventions (`TestModel`, no real API calls).

## Constraints

- pydantic-ai contribution standards apply (typing, coverage, docs); read the repo's CLAUDE.md and
  contributing docs in the worktree -- they, not this spec, are authoritative for mechanics.
- Never invoke the real Anthropic API in tests; the fixture-capture step is manual/David-assisted.
- Do not add model-version strings or token caps anywhere.
- Where our native events have no Claude equivalent, decide drop-vs-map explicitly and document
  each decision in the PR body.

## Acceptance

| # | criterion |
|---|---|
| 1 | Fixtures from real `claude` CLI committed, version recorded |
| 2 | `ClaudeCodeEventStream` maps text, tool_use/tool_result, and result-with-usage |
| 3 | gh-aw `parse_claude_log.cjs` (pinned) extracts turns + tokens + tool calls from our output |
| 4 | Docs updated; API surface flagged as draft pending Douwe review |
| 5 | Repo quality gates green (lint, typecheck, tests, coverage) |
| 6 | #5223 stall reason investigated and reported |

---

## Addendum — review round 2026-08-06 (pydantic-ai worktree session)

Reviewed against the actual `UIEventStream` surface in this worktree. Corrections and resolved
decisions below; the original text above is preserved verbatim except: Bill's shim promoted to
reading item 0, and the compaction example removed from Constraints (corrected here).

### Decisions resolved with David (2026-08-06)

1. **Granularity: BOTH modes in v1.** Claude Code stream-json is message-level by default — each
   line a complete `assistant`/`user` message with whole content blocks; per-token streaming only
   exists behind `--include-partial-messages`, which adds `stream_event` lines wrapping raw
   Anthropic-SSE-shaped events. The adapter therefore ships both: default message-level (buffer
   deltas, emit whole blocks at part-end — `PartEndEvent` carries the complete part) and an opt-in
   partial-messages mode emitting `stream_event` lines. Fixtures must cover both modes.
   **Do not copy the Vercel/AG-UI per-delta emission shape for the default mode.**
2. **Fixture capture runs in the pydantic-ai session** via nested `claude` CLI over Bash (not
   hand-run by David). Scrub before committing (see hygiene below).
3. **This spec is version-controlled here** (committed on the PR branch as `PLAN.md`, deletable
   before the PR leaves draft); the harness-worktree original is now a pointer stub.

### Corrections

- **Compaction: map, don't drop.** Claude Code HAS an equivalent — `{"type": "system", "subtype":
  "compact_boundary", "compact_metadata": {...}}`. `CompactionPart` maps to it via the existing
  `handle_compaction` hook. (The original spec wrongly listed compaction as having no counterpart.)
- **The gh-aw parser is a requirements document, not just an acceptance oracle.** Before locking
  the mapping, extract from `parse_claude_log.cjs` the load-bearing field list: which line types
  and fields it actually reads. Two known pressure points:
  - **Per-assistant-line `usage`**: the native event stream exposes no per-`ModelResponse` usage
    mid-run (usage arrives only with `AgentRunResultEvent`). If the parser sums usage off
    `assistant` lines, decide how to source or stub it; if it only reads the terminal `result`
    record, emit zeros/omit mid-stream and document that.
  - **`system`/`init` fields** (`session_id`, `model`, `cwd`, `tools`, `mcp_servers`,
    `permissionMode`, ...): none are knowable from events; they become constructor parameters with
    fabricated defaults. Which ones matter is a parser question.

### Additional requirements the original missed

- **Determinism for tests**: `session_id`, per-line `uuid`s, `msg_...` ids, `duration_ms`,
  `duration_api_ms`, `total_cost_usd` are all nondeterministic. Provide an injectable id
  factory/clock (or snapshot matchers) so schema tests snapshot cleanly.
- **`run_input` API wart**: `UIEventStream.run_input` is a required first field with no meaning for
  a stdout-shaped adapter. Whatever shape is chosen (`RunInputT = None` + default, or another
  solution), flag it for Douwe alongside naming/placement.
- **`content_type`** defaults to SSE; override for JSONL (e.g. `application/x-ndjson`).
- **Fixture hygiene**: scrub `cwd`, session ids, and machine-local tool output from captured
  fixtures. Also verify whether `-p` mode echoes the initial user prompt as a `user` line — if it
  does, the prompt must become a constructor param since it's not reconstructable from events.
- **Vendored parser mechanics**: confirm the pinned `parse_claude_log.cjs` runs standalone under
  plain node (check its imports), carry gh-aw's license attribution, and rely on GitHub's ubuntu
  runners having node preinstalled for the at-least-one-lane requirement; combined coverage across
  lanes keeps the 100% bar safe where the test skips.
- **Smaller decide-and-document items** (each goes in the PR-body decision log): thinking blocks'
  `signature` field for non-Anthropic models (omit vs empty); `result` subtype mapping
  (`UsageLimitExceeded` -> `error_max_turns`?, generic error path -> `error_during_execution` with
  `is_error: true`); `stop_reason` reconstruction from `finish_reason`; multimodal tool returns ->
  `tool_result` content arrays; `parent_tool_use_id: null` (no subagent counterpart in the seam).

### #5223 lessons — API-shape constraints (resolved 2026-08-06)

#5223 is still open/draft; it stalled because its load-bearing API question was never adjudicated
(the substantive review lived off-GitHub) and the diff grew to 40 files. Binding lessons here:

1. **No `Agent.to_claude_code()`, no `deps_factory`.** Maintainer review of #5223 preferred the
   construct-the-adapter shape over `Agent.to_*()` methods (`to_ag_ui()` predates that preference);
   the blessed shape is "construct the adapter/event-stream, call it in a couple of lines". We ship `ClaudeCodeEventStream` (+ docs showing the couple-of-lines usage); any
   one-liner convenience belongs harness-side, and would be a factory function, never an `Agent`
   method.
2. **The diff stays inside `pydantic_ai/ui/claude_code/` + its tests + docs.** Zero changes to
   `messages.py`, the shared `UIEventStream`/`UIAdapter`, or model modules. Wanting a new part
   type or a base-class change is the signal to open a sibling PR, not to widen this one.
3. **Design questions go to Douwe on the PR and stay open** — don't self-answer them a day later.
   PR stays draft per the framing agreed with Douwe, but small, with docs in the same PR.
4. **Cite both #5223 and #5949 in the PR body.** Kludex's #5949 (OpenResponses) deliberately
   rejects the UI-adapter seam for a *wire/server* protocol; ours uses it because Douwe
   explicitly routed this CLI-output seam through `pydantic_ai.ui`. Naming that distinction
   preempts the "why is this a UI adapter?" round.

### gh-aw parser requirements (resolved 2026-08-06)

Pinned oracle: `github/gh-aw` @ `fafef5837db7134eb1931954423f5c9d6e0bec3a`, entry point
`actions/setup/js/parse_claude_log.cjs`. **Vendoring needs the whole family** (that file is a
shell; logic lives in `log_parser_shared.cjs`, `log_parser_format.cjs`,
`log_parser_step_summary_builder.cjs`, plus `error_codes`/`error_helpers`/`markdown_unfencing`).
MIT (GitHub, Inc.) — retain the notice, header-comment each vendored file with upstream path +
sha. Test through the pure `parseClaudeLog(string)` export (plain node, zero deps); the `main`
export needs a github-script `core` global — don't use it. Assert on the **Information** section
of the returned markdown: it reads the literal last raw JSON line, making it the strictly
tighter oracle.

Conformance rules, ordered by risk (1–4 are silently-wrong-but-green):

1. **`result` is the final JSON line** — any parseable JSON after it (e.g. a trailing
   `stream_event`) collapses the Information section to "No information available".
2. **`message.content` is always a list of blocks**, never a bare string — an `Array.isArray`
   guard silently drops the whole entry (this exact bug swallows the final answer text in Bill's
   shim, unnoticed because his visible output came from the safe-outputs MCP tool).
3. **`usage` lives on the `result` record only** — per-`assistant` usage is never read. Exactly
   four names: `input_tokens`, `output_tokens`, `cache_creation_input_tokens`,
   `cache_read_input_tokens`; at least one of input/output must be truthy or the block is
   skipped. So the "per-response usage unavailable mid-run" gap is a non-issue: emit
   Anthropic-fidelity zeros (or nothing) on assistant lines, real totals on `result`.
4. **`is_error: true` on failed `tool_result` blocks** — omission renders failures as ✅ (Bill's
   shim inherits this; his unhandled retry-prompts also leave calls unpaired → ❓).
5. `tool_use.id` must exactly pair with `tool_result.tool_use_id`; unpaired renders ❓.
6. **The final answer must be an `assistant` `text` block** — `result.result` is never read.
7. `num_turns` honest (count of model responses): `>= GH_AW_MAX_TURNS` fails the build.
8. `mcp_servers[].status: 'failed'` in init can fail the build — emit `'connected'` or omit.
9. Emit at least one recognized entry, or gh-aw's guardrail `core.setFailed`s.
10. `total_cost_usd`: omit rather than emit `0` (falsy in JS — renders nothing either way).
11. Ignored by the oracle, emit only for CLI fidelity: per-line `uuid`/`session_id`/timestamps,
    `parent_tool_use_id`, `duration_api_ms`, `stop_reason`, `result.subtype`/`is_error`,
    `stream_event` lines, thinking `signature`.

Also confirmed: nothing throws — unknown types/malformed lines/missing fields all skip
gracefully, and gh-aw's real input is stream-json teed together with debug noise, so our clean
JSONL is a safe direction. Bill's shim is post-hoc replay of `result.all_messages()` (no
streaming, no text/thinking mapping) — a field-shape cheat-sheet only, not an ordering
reference.

### Fixture findings (captured 2026-08-06, CLI 2.1.222 — see `tests/assets/claude_code_stream_json/`)

- **One `assistant` line per content block**, sharing one `message.id`; `message.usage` repeated
  on each line. This settles the default-mode emission model: one line per part at part-end,
  one message id per model response.
- **Partial mode is a superset, not an alternative**: `--include-partial-messages` interleaves
  `stream_event` lines (raw Anthropic SSE shapes) while keeping the whole-block `assistant`
  lines and the terminal `result`. So "both modes" = one emission pipeline + an opt-in flag that
  additionally emits `stream_event` records.
- **Error semantics are two-axis**: API failures keep `result.subtype = 'success'` with
  `is_error: true` + `terminal_reason`, and the synthetic `assistant` line carries
  `model: '<synthetic>'` + an `error` field; only turn-limit exhaustion changes the subtype
  (`error_max_turns`). Don't assume `subtype` is the error channel.
- The live CLI emits record types beyond the classic four — `rate_limit_event`,
  `system`/`status`, `system`/`thinking_tokens`, hook records — which consumers evidently
  tolerate; our adapter emits none of them (documented drop).
- The `init` record is far richer than assumed (`agents`, `skills`, `capabilities`,
  `memory_paths`, `fast_mode_state`, ...); constructor params should cover only the load-bearing
  subset (per the gh-aw parser requirements), with the rest fabricated as static defaults.
