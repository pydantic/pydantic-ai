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
- The title should say what the change *does for the user* — e.g. `Split GoogleProvider into GoogleProvider + GoogleCloudProvider` instead of `v2 prep: card 08 google-vertex split`.
- The `v2 prep:` / `v2-exec:` prefix stays — it's how the changelog groups breaking-change prep work.
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

### 8. Deprecation-warning links point to `changelog.md`

`DeprecationWarning` messages that include a doc link should point at the project's [`changelog.md`](docs/changelog.md) breaking-changes section, **not** at a per-page migration section that may be pruned after v2.

**Why:** the docs site only shows `main`. Once v2 ships and the per-page migration content is removed, every v1 user's warning still links to a page section that no longer exists. `changelog.md` is the stable breaking-changes anchor.

If `changelog.md` doesn't yet have a v2 deprecations section, **add one as part of your PR** with the entries your warnings link to.

### 9. Wordier warnings with embedded code snippets

`DeprecationWarning` text should embed the migration path inline — a literal "before / after" code shape — instead of relying on a long external migration doc section. A user reading the warning at runtime should be able to fix their code from the warning text alone, only consulting `changelog.md` for context.

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

## Process

- Until the V2-RULES.md PR lands on `main`, the canonical copy lives in this file in the `v2-prep-rules` branch and is mirrored to `pydantic-ai-notes/david/v2-cards/V2-RULES.md` so all agents (David's + Douwe's + others') read from one source.
- Pair reviews surface new rules. When one lands, append a `### N. Rule name` block here with the rule, a one-paragraph "why," and a one-line action template if applicable. Reference the meeting/PR/Slack thread the rule came from.
- This is a working document, not a constitution. If a rule causes friction in practice, raise it on the next pair review rather than working around it silently.
