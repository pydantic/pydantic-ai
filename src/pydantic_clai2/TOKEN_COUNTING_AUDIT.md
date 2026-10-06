# Token counting audit

Symptom: CLAI seemed to reach its context limits early. This audit lists every place CLAI 2, and the
harness pieces it binds, counts, estimates, sums, or shows tokens, and checks each one for overcounts.

Method: a read of the code, plus a replay of the 25 most recent saved sessions in a local
`sessions.db` (Claude Opus, 1M window, the default plugins). For every request after the first, the
replay compares `estimate_context_tokens` of the history up to that request with the `input_tokens`
the provider then reported for it.

## Where tokens come from

No code uses `tiktoken` or a provider count-tokens endpoint. Every figure is either provider-reported
`RequestUsage` or the harness heuristic of 4 characters per token. In Pydantic AI, `input_tokens`
already includes cache reads and writes; no CLAI sum adds them again.

| Surface | Source | What the number means |
| --- | --- | --- |
| Status row `context: X/Y`, before each request | `compaction` `_ContextGauge` on harness `ContextUsageEvent` (`ReportContextUsage`, `estimate_context_tokens`) | Latest response's `input + output`, plus a heuristic for the messages since, minus the reclaim of a compaction in this request |
| Status row `context`, during streaming | `Session._stream`, every event | Latest response's `usage.total_tokens` (`input + output`) |
| Status row `context`, end of turn | `_run_prompt` | Last response's `usage.total_tokens` |
| Status row window `Y` | gauge window (honours the `compaction` override), else `Model.context_window` | Model window |
| Status row `~N streamed tokens` / `N output tokens` | streamed characters / 4, then `result.usage.output_tokens` | Turn output, including a summary call and forwarded subagent usage |
| Status row cost, `/cost`, `/usage` | `usage_report.session_usage`: sum of `ModelResponse.usage` in retained history | Tokens processed and billed, per turn |
| Subagent status row | `tasks.py`: child's last `usage.total_tokens`, child model window | Same as the main row |
| `herdr` pane: `tokens`, `context` | `session_usage(...).total.total_tokens`; gauge fraction | Tokens processed; context fraction |
| `/resume` browser `N tok` | harness `ConversationStore`: sum of `total_tokens` | Tokens processed |
| Automatic compaction (0.85), `coder` `ClearToolResults` (0.7), `WarnNearLimits` (0.9) | `estimate_context_tokens` | Trigger for each |
| `protected_tokens` tail, reclaim, `/compact` "tokens saved" | `estimate_token_count` (heuristic only) | Estimate |
| Imported Claude Code and Codex sessions | per-request usage (`input + cache read + cache write`; Codex `last_token_usage`) | Set once per response, never summed |
| Session naming | naming run's `total_tokens` | Naming cost only |
| Speculative execution | none | `run_code` does not count tokens |

## Fixed in this PR (harness `compaction/_shared.py`)

### 1. Changed instructions counted twice

When the latest instructions differ from those the usage anchor was measured with,
`estimate_context_tokens` added the whole new set on top of the anchor, which already contains the
old set. The next request sends one set, not both. The estimate now swaps the old set for the new one.

CLAI's instructions change whenever the `logfire_mcp` plugin's "current UTC time is within the hour"
line ticks over. In the replay, 10 of 25 sessions crossed an hour. Each crossing overcounted by
12,788 tokens, the full instructions. That pushes every trigger above (compaction, tool result
clearing, the model-facing near-limit warning) and the gauge earlier, by about 6% of a 200K window.

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
clearing, the near-limit warning, and a summarizing compaction on the next request. Tool results are
now counted as `model_response_str()`, the text the provider receives. Files are left out, as
`FilePart` and user-prompt images already were, and structured results count as their JSON.

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
5. **The `compaction` `context_window` override does not reach `coder`.** `ClearToolResults(0.7)` and
   `WarnNearLimits(0.9)` resolve the window from the registry. When the registry window is smaller
   than the real one, they fire early, and the warning tells the model its context is almost full.
6. **Hourly prompt cache rebuild.** The same `logfire_mcp` hour line invalidates the prompt cache
   once an hour. One replayed request rewrote 146,727 tokens to cache. On a rebuild, the provider
   also drops accumulated thinking: the reported input fell by 7K to 37K tokens, matching the
   thinking tokens since the last rebuild. The anchor reported what was billed, so this is not a
   counting bug, but it is the rest of each overestimate in finding 1.
7. **The status figure changes meaning within a request.** The gauge shows the next request's
   estimate, then the first streamed event replaces it with the previous response's total. The
   figure drops while streaming and jumps at the end of the request.
8. **Summary calls are missing from cost.** `/cost` and the footer sum the retained responses. An
   automatic summary call's usage only reaches the run's `RunUsage`. `/compact` sums into a
   `RunUsage` that is then discarded. A `/fork` copies history, so its `/cost` includes the parent's
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
