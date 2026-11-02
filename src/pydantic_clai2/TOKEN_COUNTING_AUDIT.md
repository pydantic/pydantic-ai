# Token counting audit

Symptom: CLAI seemed to reach its context limits early. This audit lists every place CLAI 2, and the
harness pieces it binds, counts, estimates, sums, or shows tokens, and checks each one for overcounts.

Method: a read of the code, plus a replay of the 25 most recent saved sessions in a local
`sessions.db` (Claude Opus, 1M window, the default plugins). For every request after the first, the
replay compares `estimate_context_tokens` of the history up to that request with the `input_tokens`
the provider then reported for it.

The audit and replay come from #9929. The tables and findings were last checked against the code on
`main` on 2026-10-16. Since #9929, #9969 removed `WarnNearLimits` from harness `Coder`. #9930 (open)
would replace `Coder`'s `ClearToolResults(max_fraction=0.7)` with a summarizing compaction chain,
and `clai2`'s `coder` plugin would bind `Coder(compaction=False)`; the notes below say where that
would change a finding.

## Where tokens come from

No code uses `tiktoken` or a provider count-tokens endpoint. Every figure is either provider-reported
`RequestUsage` or the harness heuristic of 4 characters per token. In Pydantic AI, `input_tokens`
already includes cache reads and writes; no CLAI sum adds them again.

| Surface | Source | What the number means |
| --- | --- | --- |
| Status row `context: X/Y`, before each request | `compaction` `_ContextGauge` on harness `ContextUsageEvent` (`ReportContextUsage`, `estimate_context_tokens`) | Latest response's `context_tokens` (input as `context_input_tokens` reads it, plus output), plus a heuristic for the messages and newly revealed tool schemas since, minus the reclaim of a compaction in this request. With no reported usage yet, the heuristic alone, without most tool definitions, shown with `~` |
| Status row `context`, during streaming | `Session._stream`, every event | Latest response's `context_tokens` (input as `context_input_tokens` reads it, plus output) |
| Status row `context`, end of turn | `_run_prompt` | Last response's `context_tokens` |
| Status row `context`, resumed conversation before a turn | `SessionRuntime.show_model` | Last retained response's `context_tokens` |
| Status row window `Y` | gauge window (honours the `compaction` override), else `Model.context_window`; before a request, the selected model's `infer_model_profile` window (a plugin model's `settings_from` provider's) | Model window |
| Status row `~N streamed tokens` / `N output tokens` | streamed characters / 4, then `result.usage.output_tokens` | Turn output, including a summary call and forwarded subagent usage |
| Status row cost, `/cost`, `/usage` | `usage_report.session_usage`: sum of `ModelResponse.usage` in retained history | Tokens processed and billed, per turn |
| Subagent status row | `tasks.py`: child's last `context_tokens`, child model window | Same as the main row |
| `host.tasks` and the web UI's Workers tree | `Tasks.info`: sum of `ModelResponse.usage` in the task's messages | Tokens the task processed |
| Web UI status, `/usage` and `/cost` cards | the terminal `Status` fields (`Remote`), `session_usage` (`protocol.usage_report`) | Same as the terminal rows |
| `herdr` pane: `tokens`, `context` | `session_usage(...).total.total_tokens`; gauge fraction | Tokens processed; context fraction |
| `/resume` browser `N tok` | harness `ConversationStore`: sum of `total_tokens` | Tokens processed |
| Automatic compaction (0.85), `coder` `ClearToolResults` (0.7; #9930 would drop it) | `estimate_context_tokens` | Trigger for each |
| Experimental background compaction (`background_start` of the 0.85 trigger), and whether its held summary still fits under the trigger | `estimate_context_tokens`, minus the `estimate_token_count` reclaim of the swap | Start and fit check |
| `protected_tokens` tail, `/compact` "tokens saved" | `estimate_message_tokens` (`find_token_cutoff` for the tail): per-message heuristic scaled to the reported `input_tokens` between responses, heuristic alone without usage | Estimate |
| Reclaim, `compact_messages` span `tokens_before`/`tokens_after` | `estimate_token_count` (heuristic only) | Estimate |
| Imported Claude Code and Codex sessions | per-request usage (`input + cache read + cache write`; Codex `last_token_usage`) | Set once per response, never summed |
| Session naming | naming run's `total_tokens` | Naming cost only |
| Speculative execution | none | `run_code` does not count tokens |

## Fixed in #9929 (harness `compaction/_shared.py`)

### 1. Changed instructions counted twice

When the latest instructions differ from those the usage anchor was measured with,
`estimate_context_tokens` added the whole new set on top of the anchor, which already contains the
old set. The next request sends one set, not both. The estimate now swaps the old set for the new one.

CLAI's instructions change whenever the opt-in `logfire_mcp` plugin's "current UTC time is within
the hour" line ticks over. In the replay, 10 of 25 sessions crossed an hour. Each crossing overcounted by
12,788 tokens, the full instructions. That pushed every trigger above (compaction, tool result
clearing, and the model-facing near-limit warning `Coder` bound until #9969) and the gauge earlier,
by about 6% of a 200K window.

| Session | Worst overestimate before | After |
| --- | ---: | ---: |
| `5285e527` | 49,998 | 37,210 |
| `9a1a8114` | 43,127 | 30,340 |
| `7e8c78ed` | 42,613 | 32,790 |
| `ac97a83c` | 26,966 | 14,178 |
| `d3a372d5` | 14,230 | 1,442 |

Sessions that did not cross an hour estimated within a few hundred tokens (median 300 under).
Finding 6 accounts for the remaining overestimates.

### 2. Files returned by tools counted as text

Tool results were counted as `str(part.content)`. When a tool returns an image or a document (MCP
image results, browser screenshots, any tool returning `BinaryContent`), that string spells out the
bytes: a 300 KB PNG estimated at about 215,000 tokens. A single screenshot could fire tool result
clearing, the near-limit warning, and a summarizing compaction on the next request. For
`ToolReturnPart`, #9632 fixed this: a returned file counts as the `See file <identifier>.`
reference a text-only tool result carries in its place (`tool_return_text`). #9929 did the same for
files returned by provider-side tools: `NativeToolReturnPart` is counted as `model_response_str()`,
which leaves files out, as `FilePart` and user-prompt images already were.

## Not fixed: recommendations

3. **Stale anchor after `/compact`.** The kept tail still holds responses whose usage measured the
   history before compaction. The next turn's estimate anchors on them, so it can trigger automatic
   compaction again at once, and the gauge stays high until a response arrives. In a run, the next
   response re-anchors and `ReportContextUsage` subtracts the reclaim; nothing carries that
   correction across turns. A fix needs a harness rule for anchors that predate a rewrite.
4. **Gauge readings from subagents and forks.** Plugins are bound into the stock agent, so a
   self-delegated child and a `/fork` session run the `compaction` gauge and the `herdr` reporter
   too. A replay with `SubAgents(include_self=True)` confirmed that the parent's capability
   instance receives the child's `ContextUsageEvent`. Both handlers write the foreground status, so
   a background child can repaint the main row, and its alert, while the user is at the prompt. The handlers need a signal for which run is the
   foreground one. `ctx.conversation_id` against `host.session_id` would cover subagents but not
   forks.
5. **The `compaction` `context_window` override does not reach `coder`.** `Coder` binds
   `ClearToolResults(max_fraction=0.7)` without a `context_window`, and the `coder` plugin cannot
   pass one, so it resolves the window from the registry. When the registry window is smaller than
   the real one, it clears tool results early. (`WarnNearLimits(0.9)`, which also told the model its
   context was almost full, left `Coder` in #9969.) #9930 would remove this: the `coder` plugin would
   bind `Coder(compaction=False)`, leaving the `compaction` plugin, which honours the override, as
   the only trigger.
6. **Hourly prompt cache rebuild.** While `logfire_mcp` is on, the same hour line invalidates the
   prompt cache once an hour. One replayed request rewrote 146,727 tokens to cache. On a rebuild, the provider
   also drops accumulated thinking: the reported input fell by 7K to 37K tokens, matching the
   thinking tokens since the last rebuild. The anchor reported what was billed, so this is not a
   counting bug, but it is the rest of each overestimate in finding 1.
7. **The status figure changes meaning within a request.** The gauge shows the next request's
   estimate, then the first streamed event replaces it with the previous response's total. The
   figure drops while streaming and jumps at the end of the request.
8. **Summary calls are missing from cost.** `/cost` and the footer sum the retained responses. An
   automatic summary call's usage only reaches the run's `RunUsage`. `/compact` sums into a
   `RunUsage` that is then discarded. A background summary (experimental) counts into the
   `RunUsage` of the turn that started it, like an automatic one, but when it finishes after that
   turn ended, the turn's output-token figure has already been shown without it. A `/fork` copies history, so its `/cost` includes the parent's
   retained responses.
9. **Totals of processed tokens are not context.** `/cost`, the `herdr` `tokens` field, and the
   browser's `tok` count every request's whole input again, so they grow roughly quadratically.
   They are correct as usage, but should not be compared with the window.
10. **Imported histories.** A resumed Claude Code or Codex session anchors on usage that included
    that agent's own system prompt and tools, so the first CLAI request overestimates until the
    response re-anchors.
11. **Request limit.** The default is 10,000 requests per prompt. The summary call and forwarded
    subagent requests count against it by design. This is not an early-limit cause unless
    `--request-limit` is lowered.
