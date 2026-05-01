# v2 deprecate-before — 1.x deprecation plan

This PR is a **draft / planning artifact** for the 1.x deprecation PRs in the `deprecate-before/` v2-card bucket. Each card describes a v2 breaking change that needs a `DeprecationWarning` to ship in 1.x first so users see warnings before removal.

After the [2026-04-30 PR comments](https://github.com/pydantic/pydantic-ai/pull/5263), only **cards 01 and 03** remain in scope here, **bundled into this single PR**:

- Card 02 deprecation is taken over by [#5075](https://github.com/pydantic/pydantic-ai/pull/5075) (commit cf8127ca1); cascade-removal deferred to a separate post-#5075 PR.
- Card 21 is fully covered by [#5188](https://github.com/pydantic/pydantic-ai/pull/5188) (@dfm88).
- This PR supersedes [#5076](https://github.com/pydantic/pydantic-ai/pull/5076) (closed) — actual code-side deprecation is authored here.

**Descriptor design — Approach B (per-type subclass) chosen.** `isinstance(result.usage, RunUsage)` correctness wins; one ~50-line private module with 4 subclasses is acceptable. Approach A (proxy with `__getattr__`) is rejected.

Cards live at `pydantic-ai-notes:david/v2-cards/deprecate-before/` (canonical; the working copy under `pydantic-ai-main/local-notes/v2-cards/` was retired on 2026-04-29).

## Coordination with in-flight PRs

| Card | Title | Status | In-flight PR | Plan for this draft |
|------|-------|--------|--------------|---------------------|
| 01 | result-class-consistency | no PR | — | **author here** |
| 02 | retries-split | covered | [#5075](https://github.com/pydantic/pydantic-ai/pull/5075) — adds `tool_retries=` kwarg + `DeprecationWarning` on `retries=` (commit cf8127ca1, pattern matches `mcp.py:1145` `sse_read_timeout`); 1.x cascade preserved; `TODO(v2)` left at resolution site | **out of scope here**; cascade-removal lands in a follow-up PR after #5075 merges |
| 03 | history-processors-deprecation | superseded PR | [#5076](https://github.com/pydantic/pydantic-ai/pull/5076) **closed**; this PR supersedes | **author here** — code-side ctor-kwarg deprecation |
| 21 | prepare-tools-none | covered | [#5188](https://github.com/pydantic/pydantic-ai/pull/5188) by @dfm88 — exact 1.x warning per card | **out of scope here**; v2 raise lands in v2-cut PR (per card, owned by @adtyavrdhn) |

Both cards land on `v2-changes` and ship as a single PR.

## Card 01 — result-class-consistency

**Goal**: in 1.x emit `DeprecationWarning` on call-style use of `usage`/`timestamp`/`get` on result classes; in v2, become plain `@property` (or removed in `get`'s case).

### Affected sites (verified against HEAD)

| File | Line | Symbol | Type today | Type v2 |
|------|------|--------|------------|---------|
| `pydantic_ai_slim/pydantic_ai/run.py` | 383 | `AgentRun.usage` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/run.py` | 522 | `AgentRunResult.usage` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/run.py` | 527 | `AgentRunResult.timestamp` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/result.py` | 153 | `AgentStream.get` | `def` | removed (use `.response`) |
| `pydantic_ai_slim/pydantic_ai/result.py` | 163 | `AgentStream.usage` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/result.py` | 172 | `AgentStream.timestamp` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/result.py` | 579 | `StreamedRunResult.usage` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/result.py` | 593 | `StreamedRunResult.timestamp` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/result.py` | 762 | `StreamedRunResultSync.usage` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/result.py` | 770 | `StreamedRunResultSync.timestamp` | `def` | `@property` |
| `pydantic_ai_slim/pydantic_ai/direct.py` | 404 | `StreamedResponseSync.get` | `def` | removed (use `.response`) |
| `pydantic_ai_slim/pydantic_ai/direct.py` | 414 | `StreamedResponseSync.usage` | `def` | `@property` |

12 sites total. 10 are method→property migrations; 2 are method removals (`get`).

### Design — `_DeprecatedCallable*` subclasses (Approach B)

Per-type subclass that adds `__call__` for the deprecation warning. `isinstance` and pyright stay happy because the wrapped value *is* an instance of the underlying type.

```python
class _DeprecatedCallableRunUsage(RunUsage):
    _message: str = PrivateAttr()

    def __call__(self) -> 'RunUsage':
        warnings.warn(self._message, DeprecationWarning, stacklevel=2)
        return self
```

Same shape for `_DeprecatedCallableRequestUsage(RequestUsage)`, `_DeprecatedCallableDatetime(datetime)`, `_DeprecatedCallableResponse(ModelResponse)`. All four live in `pydantic_ai_slim/pydantic_ai/_deprecated_callable.py` (private module) — import-only at the 12 migration sites.

### Migration commit shape (per file, after design blessed)

- `pydantic_ai/_deprecated_callable.py` — new file with `_DeprecatedCallableRunUsage`, `_DeprecatedCallableRequestUsage`, `_DeprecatedCallableDatetime`, `_DeprecatedCallableResponse` + helper `_make_deprecated_callable(value)` factory.
- One commit per result-classes file (`run.py`, `result.py`, `direct.py`) — converts `def usage` / `def timestamp` to `@property` returning the wrapped value.
- One commit for tests in `tests/` exercising both call-style (warns) and attribute-style (no warning) per site.

### Test strategy

- `tests/deprecations/test_result_class_consistency.py` (new file) — parametrized over the 12 sites:
  - access via attribute → no warning, returns expected value
  - access via call → emits `DeprecationWarning` with the documented message, returns expected value
  - `isinstance(value, ExpectedType)` true (validates approach B)
- Existing tests that already do `result.usage()` continue to pass; we add `pytest.warns(DeprecationWarning)` only in the new dedicated file.

## Card 03 — history_processors deprecation

**Goal**: deprecate `Agent(history_processors=...)` kwarg; remap to `capabilities=[ProcessHistory(p) for p in history_processors]`.

### Surface (verified against HEAD)

- `pydantic_ai_slim/pydantic_ai/agent/__init__.py:245` — `history_processors` ctor kwarg (and 5 more occurrences across overloads at lines 277, 307, 531, 565, 598)
- `pydantic_ai_slim/pydantic_ai/agent/__init__.py:402` — `self.history_processors: list[HistoryProcessor[AgentDepsT]] = list(history_processors or [])`
- `pydantic_ai_slim/pydantic_ai/agent/__init__.py:405` — capability registration loop

### Implementation

```python
def __init__(self, *, history_processors=None, capabilities=None, ...):
    if history_processors:
        warnings.warn(
            '`history_processors=` is deprecated, use `capabilities=[ProcessHistory(...)]` instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        from pydantic_ai.capabilities import ProcessHistory
        history_caps = [ProcessHistory(p) for p in history_processors]
        capabilities = history_caps + (list(capabilities) if capabilities else [])
    ...
```

The `self.history_processors` instance attribute can stay (callers iterate it for introspection); v2 removal is the separate concern in the v2-cut PR.

### Test strategy

- `tests/deprecations/test_history_processors_kwarg.py` — passes `history_processors=[fn]`, asserts warning fires, asserts `agent.capabilities` contains a `ProcessHistory` wrapping `fn`.
- Verify behavior parity with `capabilities=[ProcessHistory(fn)]`.

## Out of scope (already decided)

- **Card 02** — deprecation lands in [#5075](https://github.com/pydantic/pydantic-ai/pull/5075) (commit cf8127ca1): `tool_retries: int | None = None` kwarg added; `DeprecationWarning` on `retries=` (matches `mcp.py:1145` `sse_read_timeout` pattern); 1.x cascade preserved per [version policy](https://github.com/pydantic/pydantic-ai/blob/main/docs/version-policy.md); `TODO(v2)` at the cascade-resolution site. Cascade removal + `retries=` removal go to a follow-up PR after #5075 merges.
- **Card 21** — covered by [#5188](https://github.com/pydantic/pydantic-ai/pull/5188); v2 raise tracked separately by @adtyavrdhn.

## PR shape

Single bundled PR on `v2-changes`: cards 01 + 03 together. Conceptually both are 1.x deprecations whose v2 removal lands later.

## Documentation

- `CHANGELOG.md` — breaking-changes entry per deprecation, in this PR.
- `docs/version-policy.md` already covers the rule; no doc change needed.
- User-facing migration prose lands in the v2 release notes (separate doc, not in this PR).
