# Mechanical review checks

Fast, **stdlib-only** mechanical scan for pydantic-ai worktrees. High-confidence rules only — designed to stay under ~10s on a full tree (worst case &lt;30s).

It runs as the `/review-branch` pre-pass. CI runs its self-test, not the scan.

## Run

From the repository root, with the **parent** of the `mechanical/` package on `PYTHONPATH`:

```bash
export PYTHONPATH=.agents/skills/review-branch
# full tree (silent-issues audit)
python3 -m mechanical --repo . --all
# or via the runner's path
python3 .agents/skills/review-branch/mechanical/runner.py --repo . --all

# PR / branch mode (changed files + added lines only)
python3 -m mechanical --repo . --diff "$(git merge-base origin/main HEAD)"

# subset + machine output
python3 -m mechanical --repo . --all --checks denylist,patterns --json
```

Exit code **1** if any `error` severity findings; **0** if clean or only warning/info.

## Checks

| Check | What it does |
|---|---|
| `denylist` | Removed APIs (the names in pydantic-ai's `_REMOVED_IN_V2` maps, `BuiltinToolCallPart`, `to_ag_ui`, `pydantic_ai.ag_ui`, durable `value_to_type`, stale restate paths). **error** for production/test *code use*; **info/warning** for docs/changelogs. |
| `patterns` | High-signal subset of review-patterns: bare type/pyright ignores, line-number refs in comments, `warnings.warn` without `stacklevel`, empty `snapshot()`, `pytest.importorskip` in function bodies, capped `Any` sample. |
| `mode_b` | Durable dual-path: `_agent.py` + `_durability.py` for temporal/dbos/prefect; sibling wrapper method gaps; high-signal Agent methods missing from wrappers/durability; restate dir presence. |
| `cassettes` | Cassette size &gt;1MB; likely secrets (`sk-…`, Bearer, AKIA, Authorization). Streams first 200KB. |
| `docs_refs` | Docs still naming removed APIs (`load_mcp_servers`, `BuiltinToolCallPart`, `to_ag_ui`, …). |
| `missing_tests` | **Diff-only**: new/changed public symbols with no name mention under `tests/`. Covers the package roots in `models.PACKAGE_ROOTS` (slim, harness, clai2); a harness/clai2 symbol counts when its package `__init__.py` lists it in `__all__`. Skipped on `--all`. |

Optional allowlist for Mode B: `mode_b_allowlist.txt` next to this package (one method name per line).

## Output

Human summary: counts by check, severity totals, top findings.

`--json`: `{repo, mode, checks, elapsed_s, counts, findings[]}` where each finding has `check`, `severity`, `path`, `line`, `message`, `rule_id`.

## Self-test

```bash
uv run pytest .agents/skills/review-branch/mechanical/test_checks.py
```

## Design notes

- Zero third-party deps (`argparse`, `ast`, `pathlib`, `re`, `json`, `dataclasses`, `subprocess`).
- `--diff BASE` uses `git diff --name-only` + unified-0 added-line map.
- Keep false positives low; prefer fewer high-signal rules over full review-patterns coverage.

## mode_b allowlist notes

- `run_stream_sync` — **not a dual-path gap**. Inherited from `AbstractAgent`; calls `self.run_stream` (wrapper overrides apply). Inside workflows, `SyncStreamBridge` rejects a running event loop. Documented under durable Streaming sections.
- `description` — property surface not mirrored on wrappers by design.
