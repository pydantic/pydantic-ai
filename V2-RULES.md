# V2-RULES — conventions for v2 prep + exec PRs

Living rule set for the v2 migration. Updated as decisions land in pair reviews.

If you're a coding agent (Claude / Codex / Cursor / etc.) opening a v2 PR, **read this file end-to-end before drafting**, then re-read the rule any time you're about to do something that touches the topic.

## PR shapes

The v2 migration ships in two waves:

- **`v2:prep` PRs target `main`** and add `DeprecationWarning`s on the current symbols.
  - Title prefix: `v2 prep:`. Label: `v2:prep`.
  - They ship on the next 1.x minor release so users see warnings before v2 actually removes anything.
- **`v2:exec` PRs target a future v2 integration branch** (not yet created upstream).
  - Title prefix: `v2-exec:`. Label: `v2:exec`.
  - They do the breaking changes / removals / default flips.

### Branch naming

- `v2-prep-<theme>` for `v2:prep` PRs
- `v2-exec-<theme>` for `v2:exec` PRs
- `<theme>` is `card-NN-slug` for single-card PRs or a descriptive theme word for bundles (e.g. `agent-ctor-kwargs`)

### Card system

Per-change "cards" describing rationale, scope, and migration target live in [pydantic-ai-notes/david/v2-cards](https://github.com/pydantic/pydantic-ai-notes/tree/main/david/v2-cards). The [`00-index.md`](https://github.com/pydantic/pydantic-ai-notes/blob/main/david/v2-cards/00-index.md) is the master tracker; [`00-pr-tracker.md`](https://github.com/pydantic/pydantic-ai-notes/blob/main/david/v2-cards/00-pr-tracker.md) lists active PRs with branches and owners. Read the relevant card before drafting; append non-obvious implementation decisions to the card's `## Implementation log` section.

## Rules

### 1. No repo-wide `filterwarnings` for deprecations

**Don't** add `filterwarnings = ['ignore:...:DeprecationWarning']` entries to `pyproject.toml`'s `[tool.pytest.ini_options]` for the warnings this PR introduces.

Instead:

- **Migrate every internal caller** in `pydantic_ai_slim/`, `docs/**/*.md`, `examples/`, `clai/`, and any other live code path to the new API.
- For tests that **intentionally** exercise the old path (i.e. their job is coverage of the deprecated code), use **per-test** `@pytest.mark.filterwarnings('ignore::DeprecationWarning')` on the affected test functions. Or wrap the deprecated call in `pytest.warns(DeprecationWarning, match=...)` if asserting the warning fires.
- **Per-file** `pytestmark = pytest.mark.filterwarnings(...)` is acceptable **only** when the entire file is legacy-coverage (no new-API tests mixed in). Default to per-test.

**Why:** the suite-wide ignore hides the rest of the codebase emitting the new warning. A clean test run after your PR should mean both the lib and its examples have actually migrated. The user-facing signal of the warning only works if our own code doesn't trip it.

### 2. Change-log-ready PR titles

Titles end up in the [release changelog](https://github.com/pydantic/pydantic-ai/releases). Write them for users reading at merge time, not for us tracking work:

- **No card refs** in titles (`card-08`, `[card 23]`, `(cards 15, 19)`). Cards are internal bookkeeping.
- **No `v2 prep:` / `v2-exec:` prefix.** The PR label (`v2:prep` or `v2:exec`) does the grouping for the changelog tooling; the title doesn't need to repeat it. Practice as of the merged v2-exec wave ([#5396](https://github.com/pydantic/pydantic-ai/pull/5396), [#5434](https://github.com/pydantic/pydantic-ai/pull/5434), [#5444](https://github.com/pydantic/pydantic-ai/pull/5444), [#5458](https://github.com/pydantic/pydantic-ai/pull/5458), [#5459](https://github.com/pydantic/pydantic-ai/pull/5459), [#5460](https://github.com/pydantic/pydantic-ai/pull/5460)): titles read as plain change descriptions.
- **Wrap code symbols in backticks** — class / function / kwarg / prefix names get backticks so they render as code in the changelog. Good:

  > Drop `GrokProvider` in favor of `XaiProvider`

  Not:

  > Drop GrokProvider in favor of XaiProvider

- The title should say what the change *does for the user* — e.g. `Split GoogleProvider into GoogleProvider + GoogleCloudProvider` (with backticks around the class names) instead of `card 08 google-vertex split`.
- If the title claims multiple cards and the body defers some, **split the PR** or rewrite the body to explicitly state what's in scope vs deferred.

### 3. Code comment on every `provider.name` method

Every provider's `name` method (or property) **must** carry a comment explaining:

```
# Returned value flows into ModelMessage.provider_name on every part.
# Thinking-tag detection and built-in-tool detection check this value when
# the model class loads history, so silently renaming breaks replay of any
# message history captured against the old name.
```

Apply to every provider subclass, not just the one your PR touches. If you're renaming a provider's `name`, also update the corresponding **model** class to accept the old `provider_name` value(s) in every `self.system == ...` check — see rule 4.

### 4. Cross-history-replay tests

When a PR changes a `provider.name` value (or any other field that ends up serialized into `ModelMessage`), add at least one test that:

1. Constructs a `ModelMessage` history with `provider_name` set to the **old** literal value (hardcoded, e.g. `"google-vertex"`).
2. Replays that history through the new model class.
3. Asserts the new code still routes it correctly — thinking tags surface, built-in tools detect, system-string checks resolve.

We don't have these today across the suite. Adding one for the provider you're touching is a hard requirement; expanding coverage to other providers is a separate hardening pass.

### 5. Capability over hook in deprecation messages

When an `Agent.__init__` kwarg has both a capability replacement and a hook replacement with the same signature, the `DeprecationWarning` text should point users at the **capability**. Both work; the capability is the cleaner v2 path and matches the long-term composition layer.

### 6. Deprecation tests live under `tests/v2/`

New tests covering v2 `DeprecationWarning`s (and the symbols / prefixes / kwargs they steer users toward) go in `tests/v2/test_<theme>.py`, **not** at the top level of `tests/`.

The folder is the v2-specific bucket — when v2 is cut, the entire folder is reviewed and pruned together.

Existing tests that exercise the now-deprecated path **stay in their original location**. Migrate them to the new API if possible. If the test's purpose is to exercise the deprecated path for coverage, add `# pyright: ignore[reportDeprecated]` on the deprecated call site (specific code, not a blanket `# type: ignore`) and a per-test `@pytest.mark.filterwarnings` (per rule 1).

### 7. PR scope discipline

One topic per PR. If a PR title claims multiple cards, the body must explicitly state which are in scope vs deferred — or split the PR.

The card system is internal organization; the unit of review is the PR, and reviewers should be able to evaluate one coherent change at a time.

### 8. Deprecation-warning links point to `changelog.md` (only when verbose migration warrants it)

Only deprecations with a **verbose, multi-step migration path** need a `changelog.md` entry + link from the warning text (e.g. AG-UI's move from `Agent.to_ag_ui()` / `AGUIApp` to `AGUIAdapter.dispatch_request()` composition, per [#5345](https://github.com/pydantic/pydantic-ai/pull/5345)). Small, mechanical changes — prefix renames, kwarg removals, method-to-property migrations — don't need a full guide; an inline before/after snippet in the warning text (rule 9) suffices.

When you do link from a warning, link to [`changelog.md`](docs/changelog.md) — **not** to a per-page migration section that may be pruned after v2 (the docs site only shows `main`). If `changelog.md` doesn't yet have a v2 deprecations section, add one as part of your PR with just the entries your warnings link to.

**Removals (`v2:exec` leg):** the v2-exec PRs own the `changelog.md` updates for symbols being actually removed at the v2 cut. Don't pre-fill the v2-exec entries in `v2:prep` PRs.

### 9. Wordier warnings with embedded code snippets

`PydanticAIDeprecationWarning` text should embed the migration path inline — a literal "before / after" code shape — so a user reading the warning at runtime can fix their code from the warning text alone, only consulting `changelog.md` when the migration is too verbose for the warning (rule 8).

Shape:

```
'`X(...)` is deprecated and will be removed in v2.0. Replace with `Y(...)`. See <changelog link> for context.'
```

The deprecation warning copy **must** explicitly mention v2.0 and name the concrete migration target. Never just `'X is deprecated'`.

### 10. Forward-compat warnings can keep the `DeprecationWarning` framing

Some v2 changes are technically not deprecations — they're "behavior will change in v2, pick now to pin behavior" notices (e.g. OpenAI bare `openai:` prefix, MCP capability defaults). No better stdlib `Warning` subclass exists, so keeping `DeprecationWarning` framing is fine if the wording makes the forward-compat nature clear:

```
"'openai:' will resolve to the OpenAI Responses API in v2.0. Pick 'openai-chat:' (current behavior) or 'openai-responses:' (v2 default) to pin."
```

### 11. Factor deprecation-warning logic into reusable utilities (v2:prep)

When the same deprecation shape (warn + remap + delegate) repeats across two or more sites, extract it into a small helper in `pydantic_ai_slim/pydantic_ai/_utils.py` (or a module-local `_deprecation.py`) rather than copy-pasting the `warnings.warn(...)` + remap block at every call site.

**Why:** the warning text, the version target, and the migration snippet must stay in lockstep across sites. A helper centralizes that contract; copy-paste invites drift.

**How to apply:** if your PR introduces a second instance of the same warn-and-remap shape, extract before merging. Surfaced on AG-UI [#5345](https://github.com/pydantic/pydantic-ai/pull/5345); applies across all v2:prep PRs.

### 12. `**deprecated_kwargs` is the standard pattern for kwarg deprecations (v2:prep)

For deprecating an `Agent.__init__`-style kwarg (or any keyword on a public callable), **remove the deprecated key from the visible signature** and catch it via a trailing `**deprecated_kwargs: Unpack[...]` (or plain `**deprecated_kwargs`) parameter. Inside the body, pop the old key, emit the warning, and remap to the new kwarg.

```python
def __init__(self, *, new_kwarg: T | None = None, **deprecated_kwargs: Unpack[_DeprecatedKwargs]) -> None:
    new_kwarg = consume_deprecated_old_kwarg(deprecated_kwargs, new_kwarg)
    ...
```

**Why:** the deprecated key no longer surfaces in IDE autocomplete (the main win) while still working at runtime. The helper centralizes the warning text per rule 11.

**How to apply:** use on every v2:prep PR that touches an `Agent.__init__`-style kwarg. Pattern introduced in the instrumentation PR merged 2026-05-12.

### 13. Tests for the `**deprecated_kwargs` path are required (v2:prep)

Every `**deprecated_kwargs` consumer ships with a test that:

1. Calls the public API with the old kwarg name.
2. Asserts `pytest.warns(DeprecationWarning, match=...)` fires with the expected text.
3. Asserts the new kwarg / behavior is wired correctly (the remap actually took effect).

These tests live under `tests/v2/` per rule 6 — the whole directory gets reviewed and pruned at v2-cut, which makes "dropping all deprecation tests" a trivial `git rm tests/v2/`.

### 14. No verbose changelog paragraphs for niche-feature deprecations (v2:prep)

When a deprecation covers a niche feature (low user surface) or a path that has had a documented replacement for months, don't generate multi-paragraph changelog sections. An inline before/after snippet in the warning text (rule 9) plus a one-line changelog entry is enough.

**Why:** verbose changelog sections for niche features bloat the release notes and crowd out higher-impact changes. The warning carries the migration; the changelog needs only an anchor.

**Example:** the AG-UI shim deprec ([#5345](https://github.com/pydantic/pydantic-ai/pull/5345)) initially generated 5 paragraphs — Douwe cut the section because the new path has been the documented one since Oct 2025.

### 15. Deprecation-warning links target existing docs pages first, `changelog.md` second (both)

When a warning links out for migration context, prefer **an existing stable docs page** that already has working examples of the new path (e.g. the AG-UI page for the `AGUIAdapter` migration). Fall back to `changelog.md` only when no such page exists or the migration crosses multiple docs sections.

This sharpens rule 8: "link to `changelog.md`" was a stable-anchor rule; this refines it to "link to whatever stable docs page best teaches the migration, with `changelog.md` as the fallback." Both targets satisfy the no-stale-link constraint from rule 8.

(see also rule 8)

### 16. Repeat full symbol names in changelog entries (both)

When a changelog entry covers multiple renames or method-to-property migrations, write out **every full old → new pair**, not an abbreviated collective form.

```
# Good
Deprecate method-style accessors in favor of property style: `usage()` → `usage`, `timestamp()` → `timestamp`, `get()` → `response`.

# Bad
Deprecate method-style accessors in favor of property style.
```

**Why:** `get()` → `response` doesn't match the "property style" abbreviation — readers skim the changelog and an abbreviated form misleads them. Surfaced on [#5263](https://github.com/pydantic/pydantic-ai/pull/5263).

### 17. User-facing renames can ship ahead of upstream API renames (both)

When the user-facing string (provider prefix, capability name, etc.) needs to rename but the underlying upstream/internal value is owned by another team, **ship the user-facing rename now and keep the upstream value internally**. The mapping layer absorbs the gap until upstream catches up.

**Example:** `gateway/google-vertex` → `gateway/google-cloud` deprecates the old user-facing string in v1, makes `google-cloud` the v2 default, but still maps to the Gateway team's old API value internally until they rename their side.

**Why:** decouples our release cadence from upstream's. Users see a consistent name; the bridge is one line of internal mapping.

### 18. Provider-name constants live at file top-level (both)

Constants like provider names go at module top-level, not inside the provider class. Coding agents tend to inline these as class attributes because the immediate diff is smaller — push back and lift them out.

Consider making them **public** (exported via `__init__.py`) when users may need to import them for their own backward-compat shims around stored message history.

**Why:** top-of-file is the canonical repo pattern; class-internal hides them from reuse and from import paths.

### 19. Provider `name` is assigned explicitly on every provider (both)

Don't derive `name` via inheritance from a shared base or a decorator. Each provider sets its own `name` literal, with the rule 3 comment above it.

**Why:** the comment from rule 3 is the load-bearing piece — an inherited or decorated `name` strips the comment from the call site, which is exactly where future agents need to see it. Four lines of explicit assignment per provider is cheap insurance.

(reconfirms rule 3)

### 20. Module-level constant renames use `__init__.py` `__getattr__` (v2:prep)

When deprecating a renamed module-level constant (or any importable symbol), add a `__getattr__` to the relevant `__init__.py` that emits the warning and returns the new symbol. This is the canonical pattern used in the built-in-tools PR and the instrumentation PR.

```python
def __getattr__(name: str) -> Any:
    if name == 'OldConstant':
        warnings.warn('`OldConstant` is deprecated, use `NewConstant`...', DeprecationWarning, stacklevel=2)
        return NewConstant
    raise AttributeError(name)
```

**When to skip:** very obscure constants with near-zero user surface (e.g. the `VertexAI` location literal in [#5336](https://github.com/pydantic/pydantic-ai/pull/5336)) — a hard rename is acceptable. Document the skip on the card.

### 21. History-replay coverage will graduate to a dedicated test file (both)

Rule 4's cross-history-replay tests are currently added per-PR for the provider being touched. The end state is a dedicated test file (e.g. `tests/v2/test_history_replay.py`) that exercises every model class against hardcoded old `provider_name` values across the full provider matrix.

**Status:** **not in any current PR** — scoped as its own feature folder. Continue adding the per-PR tests required by rule 4; the consolidation pass comes later.

(see also rule 4)

### 22. Don't ship a rename and a yield-shape change in two separate PRs (both)

When redesigning a streaming yield from a scalar (e.g. `is_last: bool`) into a structured value (e.g. `state: Literal['incomplete', 'complete']`), the **shape change is breaking** — there's no `DeprecationWarning` path for "your callback now receives a different type." Hold the rename until the structured field is in place so the breaking change happens once.

**Why:** users who migrated to the renamed-but-still-scalar version would break a second time when the shape flips. The forcing function is to merge them together (or hold the rename).

**Example:** [#5296](https://github.com/pydantic/pydantic-ai/pull/5296) — `is_last` → `state=incomplete` held in one shot.

### 23. Keep legacy literal values in `Literal[...]` aliases for serialized fields (both)

Any field that ends up serialized into `ModelMessage` (or any other dataclass users persist) and is typed `Literal['a', 'b', ...]` **must retain old values** when renaming. Pydantic fails validation on deserialization otherwise — old stored histories no longer load.

```python
# Renaming 'google-vertex' → 'google-cloud':
ProviderName = Literal['google-cloud', 'google-vertex', ...]  # keep both
```

**Alternative (not adopted):** drop the old value from the alias and add a `model_validator(mode='before')` to coerce old → new. More invasive; avoid unless the alias bloat becomes unmanageable.

**Why:** rule 4 (cross-history-replay) tests catch the missing-literal case; this rule is the fix.

### 24. Conditional imports inside provider files (both)

In any provider file whose base SDK is an optional dep (e.g. anything importing `google.genai`, `anthropic`, `openai`), module-level imports of sibling provider classes must go inside the existing `try/except ImportError:` block — not at the top of the file.

```python
try:
    from google.genai import ...
    from .google import GoogleProvider  # also here, not at top
except ImportError as _import_error:
    raise ImportError(...) from _import_error
```

**Why:** users without the optional dep installed should hit our friendly "install `pydantic-ai-slim[google]`" error, not an opaque `ModuleNotFoundError` from a sibling-class import. Caught on [#5336](https://github.com/pydantic/pydantic-ai/pull/5336).

### 25. Docs anchor-link CI is an error (both)

Broken anchor links in docs CI must fail the build, not warn. Renames silently break deep-links otherwise.

**Status:** CI-tightening task on the harness side (David SF / harness). Listed here so PR authors who add anchors know to verify their links resolve locally.

### 27. v2:exec PRs don't touch `changelog.md` — a final sweep generates it (v2:exec)

Individual `v2:exec` PRs targeting `v2-main` do NOT add `changelog.md` entries. The final v2-cut integration PR (#5451 or its successor) runs a `changelog.md` sweep against the full `v2-main` diff before it merges into `main`, generating the v2.0 release notes from the actual landed changes.

Practice as of the merged wave on `v2-main`: [#5396](https://github.com/pydantic/pydantic-ai/pull/5396), [#5434](https://github.com/pydantic/pydantic-ai/pull/5434), [#5444](https://github.com/pydantic/pydantic-ai/pull/5444), [#5332](https://github.com/pydantic/pydantic-ai/pull/5332), [#5340](https://github.com/pydantic/pydantic-ai/pull/5340) all merged without `changelog.md` entries. The wave of David-SF v2-exec PRs (#5458–#5469) follows the same pattern.

**Why:** v2-exec PRs are mostly mechanical drops on a non-released branch. Per-PR changelog churn would (a) cause merge conflicts whenever two PRs touch the same section, (b) bloat individual PR diffs with prose changes orthogonal to the code change, (c) duplicate work because the integrator needs to do a final pass anyway to assemble the v2.0 release narrative. Defer the prose to the integration step where the full set of changes is visible at once.

**Exceptions** (rare): A `v2:exec` PR MAY add a `changelog.md` entry if the migration story is genuinely non-obvious AND the entry survives unedited through to release (i.e. won't be rewritten by the final sweep). Most v2-exec PRs won't qualify.

**`v2:prep` PRs are different**: prep PRs target `main` and ship to users on the next 1.x minor release, so they DO populate the `1.x.y` section of `changelog.md` when their warning links out (per rule 8). This rule only applies to `v2:exec`.

(Inverts the default reading of rules 8 / 14 / 16 for `v2:exec` PRs; rules 8 / 14 / 16 still govern entry SHAPE when an entry exists.)

### 26. Use `PydanticAIDeprecationWarning`, not stdlib `DeprecationWarning` (v2:prep)

Every `warnings.warn(...)` call introduced by a `v2:prep` PR uses `pydantic_ai._warnings.PydanticAIDeprecationWarning` (a `UserWarning` subclass), not the stdlib `DeprecationWarning`. The matching `pytest.warns(...)` calls and `pytest.mark.filterwarnings(...)` markers swap accordingly.

```python
from pydantic_ai._warnings import PydanticAIDeprecationWarning

warnings.warn('`X(...)` is deprecated...', PydanticAIDeprecationWarning, stacklevel=2)
```

**Why:** stdlib `DeprecationWarning` is silenced by default in user code (only shown when running under `pytest` or with `python -W`), so end users never see our warnings before v2 lands. `UserWarning` is shown by default. Per Seth Larson's [Deprecations via warnings don't work for Python libraries](https://sethmlarson.dev/deprecations-via-warnings-dont-work).

**`@deprecated` decorators stay as-is.** `typing_extensions.deprecated` only fires `DeprecationWarning` (no class kwarg). Migrating those is a library-wide change, separate from v2:prep scope. Leave them alone.

**How to apply:** swap in every v2:prep PR that introduces a `warnings.warn(...)`. Pattern surfaced 2026-05-12 by @DouweM in [#5338](https://github.com/pydantic/pydantic-ai/pull/5338#discussion_r3229851341); applied across all open v2:prep PRs in the same wave.

## v2:exec phase — what's different

The rules above are written from the perspective of a `v2:prep` PR (add deprecation, keep both paths). The `v2:exec` phase removes the deprecated paths and ships the breaking changes. A few rules invert or drop:

- **`v2:exec` PRs DELETE the prep scaffolding.** Every `consume_deprecated_*` helper (rule 11), every `**deprecated_kwargs` catch-all on a signature (rule 12), and every `@deprecated` decorator added during prep gets removed. The `_deprecation.py` utilities are 1.x-only.
- **`tests/v2/` gets DELETED at v2-cut.** Rule 6 puts deprecation tests under `tests/v2/` precisely so this is a one-command cleanup: `git rm -r tests/v2/`. Don't add anything to `tests/v2/` that's meant to survive v2 — it's a 1.x-only directory by construction.
- **Changelog entries describe REMOVALS, not deprecations.** Rule 8's "link to `changelog.md`" still applies; rule 14's "no verbose paragraphs for niche features" still applies; rule 16's "repeat full symbol names" still applies. What flips is the voice: from "X will be removed in v2.0, use Y" to "X has been removed; use Y".
- **Default-behavior flips happen here.** Examples queued for `v2:exec`: `openai:` prefix → Responses API default (deferred from [#5334](https://github.com/pydantic/pydantic-ai/pull/5334)), `End.data` → `End.output` (per Slack thread). Rule 10's forward-compat warnings stop firing because the default has actually changed.
- **Legacy `Literal[...]` values stay forever.** Rule 23 doesn't relax in `v2:exec` — old `provider_name` values in stored `ModelMessage` histories still need to validate. Removing them from the alias is a v3 conversation at the earliest.
- **Target branch is `v2-main` on upstream.** Created 2026-05-15 as part of [#5451 "Pydantic AI V2"](https://github.com/pydantic/pydantic-ai/pull/5451) (the integration PR that lands on `main` at v2-cut). All `v2:exec` PRs target `v2-main`, not `main`.

## Process

- Until the V2-RULES.md PR lands on `main`, the canonical copy lives in this file in the `v2-prep-rules` branch and is mirrored to `pydantic-ai-notes/david/v2-cards/V2-RULES.md` so all agents (David's + Douwe's + others') read from one source.
- Pair reviews surface new rules. When one lands, append a `### N. Rule name` block here with the rule, a one-paragraph "why," and a one-line action template if applicable. Reference the meeting/PR/Slack thread the rule came from.
- This is a working document, not a constitution. If a rule causes friction in practice, raise it on the next pair review rather than working around it silently.
