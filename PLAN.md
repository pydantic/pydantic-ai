# v2 deprecate-before — bundled 1.x deprecation plan

This PR is a **draft / planning artifact** for the four 1.x deprecation PRs in the `deprecate-before/` v2-card bucket. Each card describes a v2 breaking change that needs a `DeprecationWarning` to ship in 1.x first so users see warnings before removal.

The PR is filed as a single discussion artifact because the four items share two open questions:

1. **Bundle vs split** — should this land as 1 PR, 4 PRs, or 2 PRs (cards 01+03 ctor-side, cards 02+21 callback-side)?
2. **Card 01 descriptor design** — the `_DeprecatedCallableProperty` approach affects 12 migration sites; the descriptor's behavior under `isinstance`, `repr`, `==`, etc. needs sign-off before all 12 land.

The four cards live at `pydantic-ai-notes:david/v2-cards/deprecate-before/` (canonical; the working copy under `pydantic-ai-main/local-notes/v2-cards/` was retired on 2026-04-29).

## Coordination with in-flight PRs

| Card | Title | Status | In-flight PR | Plan for this draft |
|------|-------|--------|--------------|---------------------|
| 01 | result-class-consistency | no PR | — | author here |
| 02 | retries-split | adjacent PR | [#5075](https://github.com/pydantic/pydantic-ai/pull/5075) (`rename-retry-fields` — runtime override + ctx.max_retries fix + internal rename) | **don't duplicate**; verify #5075 also adds `tool_retries=` kwarg + `retries=` deprecation, or layer that on top |
| 03 | history-processors-deprecation | obsolete PR | [#5076](https://github.com/pydantic/pydantic-ai/pull/5076) (docs-only soft-deprecation in favor of `before_model_request` hooks — predates `ProcessHistory` direction) | close #5076; author actual code-side deprecation here |
| 21 | prepare-tools-none | covered | [#5188](https://github.com/pydantic/pydantic-ai/pull/5188) by @dfm88 (`warn-prepare-tools-returning-none`) — exact 1.x warning called out in card | **don't duplicate**; this draft references #5188; the v2 raise lands in v2-cut PR (per card, owned by @adtyavrdhn) |

After this plan is blessed, the implementation commits land on `v2-changes` and the draft converts to ready-for-review (or gets split — see "Bundle vs split" below).

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

### Design — `_DeprecatedCallableProperty` descriptor

Card sketch:

```python
class _DeprecatedCallableProperty:
    def __init__(self, fget, message):
        self.fget = fget
        self.message = message
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = self.fget(instance)
        return _CallableValue(value, self.message)
```

**Open question 1**: how does `_CallableValue(value, message)` make the underlying value transparent to attribute access, `repr`, `==`, `isinstance`, etc., while supporting `__call__` (with warning) that returns the underlying value?

Two candidate approaches — picking one is the primary blocker for landing card 01:

#### Approach A — generic proxy with `__getattr__` delegation

```python
class _DeprecatedCallableValue:
    """Proxy: forwards all attribute access to wrapped value, but supports __call__ with deprecation warning."""

    __slots__ = ('_value', '_message')

    def __init__(self, value, message: str):
        object.__setattr__(self, '_value', value)
        object.__setattr__(self, '_message', message)

    def __call__(self):
        warnings.warn(self._message, DeprecationWarning, stacklevel=2)
        return self._value

    def __getattr__(self, name):
        return getattr(self._value, name)

    def __repr__(self):
        return repr(self._value)

    def __eq__(self, other):
        return self._value == (other._value if isinstance(other, _DeprecatedCallableValue) else other)

    # ... possibly more dunders for hash, str, bool, etc.
```

**Pros**: one class covers all return types (`RunUsage`, `RequestUsage`, `datetime`, `ModelResponse`).
**Cons**: `isinstance(result.usage, RunUsage)` returns False — surprising. `type(result.usage)` is wrong. Type checkers (pyright) need `reveal_type`-friendly stubs. Some dunders won't proxy through `__getattr__` (e.g. `__iter__`, arithmetic operators).

#### Approach B — per-type subclass with `__call__`

For each concrete return type, subclass it and add `__call__`:

```python
class _DeprecatedCallableRunUsage(RunUsage):
    def __init__(self, *args, _message: str, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, '_message', _message)

    def __call__(self) -> 'RunUsage':
        warnings.warn(self._message, DeprecationWarning, stacklevel=2)
        return RunUsage(**self.model_dump())  # plain RunUsage
```

Repeat for `RequestUsage`, `datetime`, `ModelResponse`.

**Pros**: `isinstance(result.usage, RunUsage)` is True. Type checkers happy. No proxy fragility.
**Cons**: 4 subclasses to maintain. `RunUsage` is a Pydantic model — subclassing across `pydantic.BaseModel` + adding `_message` field needs validation off (`model_config = ConfigDict(extra='ignore')` or `PrivateAttr`). `datetime` is C-implemented and subclassing it is OK (Python stdlib does it; `datetime` is final-friendly).

#### Recommendation

Approach B. Pydantic AI's quality bar prefers `isinstance` correctness over code volume. The 4 subclasses can live in a single `pydantic_ai/_deprecated_callable.py` private module and be imported only at the migration sites.

If approach B turns out worse than expected (e.g. Pydantic ConfigDict noise), fall back to A — but document the tradeoff in the PR description.

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

## Card 02 — retries split (DEFERRED — coordinate with #5075)

**Status**: PR #5075 ("Add runtime `output_retries` override + fix `ctx.max_retries` on tool path + internal rename") is open by David SF on `rename-retry-fields`. The card calls for:

- Add `tool_retries=` kwarg (currently `retries=` controls both)
- Add explicit `output_retries=` default of 1 (currently `None` falls back to `retries`)
- Deprecate `retries=` kwarg with a warning that explicitly mentions the v2 behavior change (`retries=N` no longer also raises `output_retries`)

**Action**: confirm whether #5075 already covers all three. If not, coordinate (either expand #5075 or add a follow-up commit on this branch).

(No code added in this draft for card 02.)

## Card 21 — prepare_tools None warning (DEFERRED — covered by #5188)

**Status**: PR #5188 by @dfm88 implements the exact 1.x warning called out in the card. v2-cut PR (the actual `TypeError` raise) is owned by @adtyavrdhn per card metadata.

**Action**: review and merge #5188 ASAP so the warning ships before v2 cut. No code in this draft for card 21.

## Bundle vs split — open question

This draft PR currently bundles cards 01 + 03 + plan refs to 02 + 21. Three viable splits after the design questions resolve:

1. **One PR per card** (4 PRs) — matches `v2-todos-david.md`'s original plan; each is reviewable independently. Card 02 is partly subsumed by #5075; card 21 fully subsumed by #5188. So in practice it's 2 new PRs (01, 03).
2. **Two PRs** — card 01 (descriptor migration) on its own; card 03 alone or paired with anything else. Card 01 alone is large enough to deserve its own PR.
3. **One bundled PR** (this draft, expanded) — 01 + 03 land together. Useful only if the team wants one big "v2 deprecation prep" commit; downside is the descriptor design + the kwarg remap aren't conceptually related.

**Recommendation**: split (1) — one PR per actionable card. This draft converts to "card 01" once design is blessed; card 03 lands as a separate fast PR (small surface, no design Q). Cards 02 and 21 ride existing PRs.

## Documentation

- `CHANGELOG.md` — breaking-changes section gets an entry per deprecation, per PR (don't batch at release per project rule).
- `docs/version-policy.md` already covers the rule; no doc change needed.
- Per-card user-facing migration prose can live in the v2 release notes (separate doc, not in this PR).

## Open questions for review

1. Approach A (proxy) vs B (per-type subclass) for `_DeprecatedCallableValue`?
2. Should card 02's deprecation warning land in #5075 or in a follow-up PR on this branch?
3. Should we close #5076 in favor of authoring the actual code-side deprecation here?
4. Bundle vs split — final preference?

After these are settled, this draft PR shrinks to whichever cards make the cut, with implementation commits added.
