# Running the fleet miner for the demo

The miner runs on Douwe's machine during the demo, not in Logfire. Logfire only holds two variables in the
`logfire/clai2` project on EU staging:

- `fleet_proposals__clai2`: what the miner found, which the Fleet UI shows and updates (accept, dismiss).
- `fleet_miner_control__clai2`: the UI's **Run now** button writes `requested_at`, and the miner writes back
  `started_at`, `finished_at`, `status` (`running`, `done` or `error`) and `message` (e.g. "2 new suggestions").

## Start it

From the `control-plane-miner` worktree:

```bash
uv run --env-file ~/pydantic-ai/.env python -m hackathon.fleet_miner --since 7d --min-users 2 --watch 10
```

It needs `LOGFIRE_CLAI2_API_KEY` (read OTLP, read and write variables on `logfire/clai2`) and
`PYDANTIC_AI_GATEWAY_API_KEY` (the models run on the Pydantic AI Gateway) in that `.env`.

With `--watch 10` it:

- runs right away, then every 10 minutes;
- polls the control variable every 30 seconds and runs immediately when **Run now** was pressed since its last run;
- prints one line per run: what it read, LLM requests and tokens, proposal counts, and whether it wrote;
- writes `fleet_proposals__clai2` only when something actually changed.

Leave it running in a terminal for the whole demo. Stop it with Ctrl-C.

## What it looks for

Two families of findings, both "never from one developer or one session":

1. **Behaviour gaps** (skills and instructions), braindump's pipeline ported to prompts:
   - **Extract** (`extract.py`, per prompt, cached by span id): is it actionable guidance (a `correction`, `preference`,
     `convention` or `procedure`), or rejected as a `question`, a one-off `task`, an `acknowledgment` or `unclear`?
     Actionable prompts yield generalized rules at braindump's "the pattern" level. Context per prompt: the previous
     prompt, the agent's last shell/MCP calls before it, and whether it was typed mid-run (clai2's `route`).
   - **Cluster** the rules (an LLM grouping: the gateway's Anthropic model has no embeddings, so no cosine clustering).
   - **Validate** each cluster (one cached call): one rule, which prompts actually support it (evidence verification),
     coherence and confidence, skill only for a spelled-out multi-step procedure, and whether a capable coding agent
     would already do it unprompted (`value_reason`). Tasks, generic good practice and questions about one subject
     never pass.
   - **Gate** over the verified prompts: `--min-users` (2+) developers, 3+ prompts, 2+ sessions, from 2+ days or 3+
     sessions, confidence 0.8+ (braindump's "high"), coherence 0.7+, and the value judgement.
   - **Rank** by confidence x spread over developers (braindump's 0.6/0.85/0.95/1.0), plus 0.05 per correction (up to
     3), then sessions, then recency. At most `--max-pending` (5) are pending.
2. **Risk and governance** (policy, `policy.py`): risky shell commands are found by pattern (force-push, push to main,
   hard reset, branch deletion, `rm -rf`, secrets files, `sudo`, remote hosts, skipped hooks/tests, ...), then each
   call is classified (cached per call): `requested` by the developer that turn or the one before, or `unprompted`;
   on a `protected` target, the developer's `own` work, or `harmless`. Requested on own work is normal and only counted.
   The rest is drafted as an advisor note (what agents do, how often, by how many, and the control: `ask` plus a
   "don't X unless the user asks" instruction for unprompted actions, `deny` for clearly destructive ones on protected
   targets), re-measured against the flagged calls, and gated on 2+ developers in `--min-policy-sessions` (2+)
   sessions. At most `--max-pending-policy` (3) are pending.

Rule globs are plain text and `*` only (no `?`, `[...]` or `{...}`), matched per command segment, e.g.
`git branch -D*`; a category that needs several patterns gets one rule per pattern. Every policy evidence item is a
call the final glob matches.

Evidence names people: each item carries `email` (when known) next to its `developer` number, and the document's
`developers` map names every number (`email`, `host`), so traces without `user.email` are labelled the same way.
Drafted text (what agents get) never names anyone, and nothing in the document carries tokens or credentials.

Statuses: a suggestion that passes but ranks below its cap is `stale` with `emerging: true` (the UI can show these
collapsed). An earlier pending one that no longer passes becomes `stale` with a `status_reason` ("didn't pass: only 1
developer", "didn't pass: its prompts are not guidance for the agent (task 3 of 3)", "replaced by ..."). Accepted and
dismissed ones are never touched.

## What a run costs

Mining is incremental. Each run queries only records newer than the last one it saw (with a 30-minute overlap for
late spans), extracts only prompts and classifies only risky calls it has never seen, places only new rules into the
existing groups, and reuses cached LLM output for anything whose input did not change. A full re-clustering happens
once a day, or with `--recluster`.

Measured on the clai2 project (164 typed prompts, 3 developers, 3,800 tool calls since 2026-10-06 21:00), all on
`gateway/anthropic:claude-sonnet-5-5`:

| Run | LLM requests | Tokens in / out |
|---|---|---|
| Re-validate, re-draft policy (extractions and classifications warm) | 7 | 31k / 5.8k |
| Scheduled run, nothing new | 0 | 0 |
| Scheduled run, one new risky tool call | 1 | 1.2k / 0.07k |
| One new actionable prompt (extract, place, policy refresh) | 3 | 5.3k / 0.3k |

## Useful flags

- `--recluster`: group everything from scratch now instead of incrementally.
- `--dry-run --out proposals.json`: mine and inspect without writing.
- `--dismiss ID=REASON`, `--drop ID`: curate the live document by hand.
- `--min-users N`: how many distinct developers a finding needs (2 for a small team, 3 by default).
- `--min-prompts`, `--min-sessions`, `--min-days`, `--min-sessions-one-day`, `--min-confidence`, `--min-coherence`,
  `--max-pending`, `--min-policy-sessions`, `--max-pending-policy`: the gates above.

## Caches

Everything lives in `hackathon/fleet_miner/.cache/` (gitignored), or `$FLEET_MINER_CACHE_DIR`:

- `prompts.json`, `tool_calls.json`: what was fetched, plus per-source watermarks.
- `extract-v<N>-<model>.json`: one extraction per prompt span. `call_classes.json`: one classification per risky call.
- `rule_clusters.json`, `rule_clustered.json`, `pattern_spans.json`: the groups of rules, which rules were already
  offered to them, and every item ever assigned to each proposal id (for stable ids and impact).
- `llm.json`: LLM outputs keyed by their exact input.

Deleting the directory makes the next run a full, cold one.
