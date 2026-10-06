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

## What a run costs

Mining is incremental. Each run queries only records newer than the last one it saw (with a 30-minute overlap for
late spans), facets only prompts it has never seen, places only new intents into the existing groups, and reuses
cached LLM output for anything whose input did not change. A full re-clustering happens once a day, or with
`--recluster`.

Measured on the clai2 project (about 170 typed prompts, 3 people, 6,700 tool calls over the week):

| Run | LLM requests | Tokens in / out |
|---|---|---|
| Full run (re-cluster, re-draft everything, policy), warm facets | 18 | 56k / 10.7k |
| Scheduled run, one new prompt | 1 | 1k / 0.05k |
| Run now, a few new prompts that formed new groups | 2 | 6k / 2.2k |

## Useful flags

- `--recluster`: group everything from scratch now instead of incrementally.
- `--dry-run --out proposals.json`: mine and inspect without writing.
- `--dismiss ID=REASON`, `--drop ID`: curate the live document by hand.
- `--min-users N`: how many distinct developers a pattern needs (2 for a small team, 3 by default).

## Caches

Everything lives in `hackathon/fleet_miner/.cache/` (gitignored), or `$FLEET_MINER_CACHE_DIR`:

- `prompts.json`, `tool_calls.json`: what was fetched, plus per-source watermarks.
- `facets-<model>.json`: one facet per prompt span.
- `clusters.json`, `clustered_spans.json`, `pattern_spans.json`: the groups, which intents were already grouped,
  and every span ever assigned to each proposal id (for stable ids and impact).
- `llm.json`: LLM outputs keyed by their exact input.

Deleting the directory makes the next run a full, cold one.
