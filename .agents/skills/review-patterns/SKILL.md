---
name: review-patterns
description: INTERNAL — mechanical-pattern engine inside /review-branch. Use /review-branch as the user-facing entry. Kept user-invocable for a quick standalone mechanical pass before commit.
user-invocable: true
disable-model-invocation: true
allowed-tools: Bash(git diff:*), Bash(git status:*), Bash(git log:*), Bash(git show:*), Read, Glob, Grep
---

# Review Patterns

Scan added and modified lines for line-level mechanical violations: style, local use of types,
comments, imports and error handling, test mechanics, docstring and docs language. Other lanes
own the rest:

| Concern | Lane |
|---|---|
| Single-use helpers, module organization, new-file shape | `review-code-reuse` |
| Duplicate validation, provider parity, god methods | `auto-review` |
| Public-API signature changes | `review-public-api` |
| Behavior shift, perf signals | `review-runtime-behavior` |
| Docs coverage and `tests/test_examples.py` reminders | `review-docs` |

## 1. Get the diff

Under `/review-branch`, the dispatch prompt names `code_diff_path` and `merge_base`. Review only
that diff. Never run `git diff main`, `git diff origin/main`, `git diff upstream/main` or
`git merge-base` to recompute the base: in a worktree those refs can be stale or point at a fork.

Standalone, `$ARGUMENTS` picks the diff:

- none: `git diff HEAD` (uncommitted changes)
- `last`: `git diff HEAD~1`
- `branch`: the diff against the merge base with `origin/main`

## 2. Check added and modified lines against the repo rules

Apply every line-level rule in these files. Cite the file and section in each finding.

- `agent_docs/index.md` — Code Style, Type System, Error Handling, Naming, Imports, Testing
  (pragmas and what may follow them)
- `agent_docs/code-simplification.md`
- `agent_docs/documentation.md` — Docstrings, Code comments (redundant comments, history in
  comments, line-number references)
- `tests/AGENTS.md` — Test File Structure, Best Practices (snapshots, prefer `instructions=` over
  `system_prompt=`, `try_import` gating, no line numbers in tests, no whole-file `# pyright: ignore`)
- `docs/AGENTS.md` — docs conventions, including reference-style links for API elements
- `agent_docs/provider-adapters.md` — on lines added to `pydantic_ai_slim/pydantic_ai/models/*.py`:
  - a capability gate outside the model's `profile` property and `_instance_profile_defaults()`:
    an `isinstance` on the client, `self._is_google_cloud`, a model-name prefix check, a
    provider-name comparison or a `base_url` check that decides whether a feature is sent. A
    branch that sends the same feature in each transport's own wire form is not a gate.
  - a new part-type or native-tool-kind arm that maps the part inline in the mapping loop instead
    of calling the part's own method on the model class, an arm written as a nested `def`, or a
    kind branch that only swaps the wire name. An arm of one statement may stay inline.

  The fix covers the added lines only, an exception to the second rule in step 4: aligning
  existing adapter code is its own PR (see the guide's opening).
- root `AGENTS.md` → "Requirements of all contributions" — no unnecessary `cast` or `Any`

## 3. Check the review rules

The repo files do not state these. Check them on the same lines, and cite "review-patterns rule"
as the source.

**Types**

- An `Any` annotation is acceptable only when the upstream SDK types the value as `Any` and no
  narrower type exists.
- `# pyright: ignore[reportPrivateUsage]` is allowed in test files.
- Never `# pyright: ignore[reportArgumentType]` on `assert_never()`: it hides future unhandled cases.
- Prefer a `TypedDict` over `dict[str, Any]` for structured data.
- Annotate a dict or list literal bound to a name with its type
  (`settings: SomeModelSettings = {'tool_choice': 'auto'}`): pyright cannot infer a `TypedDict`
  from the literal.

**Structure**

- Avoid private methods in private modules. Do not re-export a private method from a public
  module, by alias or through `__all__`.
- Prefer an exhaustive `elif` chain or `match` with `assert_never` in the last branch over
  `# pragma: no branch`.
- Do not modify the base `TextPart` and `ThinkingPart` classes in `messages.py`.

**Comments and docstrings**

- In docs and docstrings, link a doc page relatively (`[text](../page.md#anchor)`), never by its
  live docs-site URL.
- Flag an early docstring on logic that is not final only on a large feature or new-abstraction PR,
  where the API shape still moves between review rounds. On every other PR, docs belong in the PR.
- Method-name vs docstring scope: a NEW override with a generic inherited name
  (`_map_tool_definition`, `_prepare_request`) whose docstring describes one branch ("Add X when Y")
  must describe the general purpose first and cite the branch as an example.

**Error handling**

- `warnings.warn()` without `stacklevel`.
- An unsupported input falls through silently instead of raising an explicit error.

**Tests**

- An empty `snapshot()` still needs `pytest --inline-snapshot=create`.
- `warnings.catch_warnings(record=True)` with manual filtering: use `pytest.warns(Category, match=...)`.
  For a warn / no-warn pair, parametrize `expected_warning: str | None`; `filterwarnings = ["error"]`
  fails a leaked warning.
- A NEW test that intercepts a provider request with a mock client (`patch.object(...client...)`,
  `AsyncMock`, `MagicMock`, a hand-rolled fake) while the diff sends a new field, kwarg or header on
  the wire, and records no cassette: a mock cannot show the provider accepts the field. Point to
  `tests/AGENTS.md` → "Asserting what goes out on the wire". When the feature cannot be recorded live
  (enterprise tier, private beta, destructive side effect), report WARNING "may be inaccessible to live
  recording; confirm with a maintainer that a mock is acceptable".

**Misc**

- A walrus where it simplifies.
- A bare string literal used as an identifier, key or status value belongs in a named constant.
  Log messages, docstrings and display text are exempt.

## 4. Assign a verdict

End every finding with its own line `Verdict: <do|defer|skip> — <one-line reason>`, assigned by the
net-value rules in `.agents/skills/review-branch/verdict-rubric.md`. Severity is not actionability:
a line-level nit can be a required `do`. Two rules for this lane:

- A hit on a documented repo rule is `do`. The fix is comment-, one-token- or deletion-only and
  clearly correct.
- Consistency with a pre-existing violation is never a skip. When the line copies a sibling that
  breaks a documented rule, the verdict is `do`, and the fix covers every instance, pre-existing
  ones included.

## 5. Report

```
## Pattern Review Results

### [Category] - N issues

- **file.py** · `symbol` - [Pattern description] ([rule source])
  Verdict: <do|defer|skip> — <reason>
  Suggestion: [How to fix]

### Summary
- Total issues: N
- Categories affected: [list]
- Recommendation: [pass / review before pushing]
```
