---
name: review-patterns
description: Scan diff for line-level mechanical code and style issues (comments, imports, types, tests, error handling, docs language). Run after coding to fix patterns before committing.
user-invocable: true
allowed-tools: Bash(git diff:*), Bash(git status:*), Bash(git log:*), Read, Glob, Grep
---

# Review Patterns

Scan the current diff for line-level mechanical violations. **Line-level only** — module shape, cross-file abstractions, and architecture belong to `/review-code-reuse` and `/auto-review`.

## Scope

This skill is the single source of truth for mechanical pattern rules.

What this skill flags:
- Style on an added/modified line
- Local use of types, comments, imports, error handling
- Test-file mechanics (snapshots, fixtures placement, try_import)
- Doc-string and doc-markdown language

What this skill does NOT flag (routed elsewhere):
- Single-use helpers, module organization, new-file shape → `/review-code-reuse`
- Duplicate validation, provider-parity, god methods → `/auto-review`
- Public-API signature changes → `/review-public-api`
- Behavior shift, perf signals → `/review-runtime-behavior`

## Steps

### 1. Get changes

- No args: `git diff HEAD` (uncommitted).
- `$ARGUMENTS = 'last'`: `git diff HEAD~1`.
- `$ARGUMENTS = 'branch'`: diff against merge-base of `origin/main`.

If a `/tmp/review-branch/code.diff` file was pre-computed by `/review-branch`, read that instead.

### 2. Scan for violations

Check **only added/modified lines**.

#### Imports

- **Inline imports** inside function bodies: only legitimate reason is circular-import resolution. Hoist to module scope. Exception: `if TYPE_CHECKING: from ... import ...` is always fine.
- **Optional-package imports in source files** (`pydantic_ai/models/<provider>.py`, etc.): wrap at module level with `try: ... except ImportError: raise ImportError('Please install …')`. No inline imports.
- **Optional-package imports in test files**: use `with try_import() as <provider>_imports:` at module top (from `tests/conftest.py`), paired with class-level `@pytest.mark.skipif(not <provider>_imports(), reason='…')`. Canonical examples: `tests/test_capabilities.py`, `tests/models/test_anthropic.py`. Red flags: `pytest.importorskip(...)` inside a test body or autouse fixture; `from pydantic_ai.models.<provider> import ...` inside a test function.

#### Comments

- **Redundant comments** that restate the code: e.g. `# Increment i by 1` on `i += 1`.
- **Past-state references**: "now supports…", "Original logic for…", "Previously this was…" — unless it documents a hard-won bug or footgun.
- **Pragma trailing comments**: `# pragma: no cover` stands alone. Exception: `# pyright: ignore[...]` after a pragma is a directive, not a comment.
- **Line-number references in comments**: "line 42", "L123", "lines 10-20".

#### Types

- **`Any` annotations** (`: Any`, `-> Any`, `list[Any]`): use specific types or `object`/`Unknown` with validation. Exception: when the upstream SDK defines the type as `Any` and no narrower type exists.
- **File-level `# pyright: ignore`** directives at top of file: use inline ignores on the specific offending lines.
- **Unspecific `# type: ignore`**: should be `# pyright: ignore[specificCode]`.
- **`# pyright: ignore[reportPrivateUsage]` in test files**: allowed — unit testing internals is valued.
- **Never `# pyright: ignore[reportArgumentType]` on `assert_never()`**: suppresses future unhandled cases. Find a different fix.
- **Unannotated dict/list literals** passed directly to methods: pyright can't infer, add explicit types.
- **`TypedDict` over `dict[str, Any]`** for structured data.
- **Stale `# type: ignore`** comments that no longer suppress real errors.

#### Branching

- **Exhaustive branches**: use `assert_never` in the final `else` / `case _:` for type-checkable unions/enums. For dynamic values (arbitrary strings) use `# pragma: no cover` + defensive `raise`. Never use `# pragma: no branch` to skip exhaustiveness.

#### Variables

- **No unnecessary intermediate variables** that alias an attribute (`x = self.attr`) just to conditionally reassign. Use the attribute directly, name the result for its role. Exception: an intermediate variable is justified when it solves a typing issue or normalizes a value (ternary, complex expression).

#### Error Handling

- Explicit errors for unsupported inputs — not silent fallthrough.
- Catch specific exceptions, not broad `Exception`.
- Missing `stacklevel` in `warnings.warn()` calls.

#### Tests

- **Multiple `assert`s** on similar data that could use `snapshot()`.
- **Fixtures defined far from their tests** — should be close or in conftest.
- **Empty `snapshot()` calls** that need `pytest --inline-snapshot=create` to populate.
- **Snapshot assertion failures** from changed logic: use `--inline-snapshot=fix` before manual debugging.
- **Prefer VCR over unit tests** for provider behavior: provider APIs are the ultimate judges. Remove unit tests when logic is covered by integration tests.
- **`warnings.catch_warnings(record=True)` + manual filtering** in tests: replace with `pytest.warns(Category, match=...)`. For parametrized "warn / no warn" pairs, parametrize `expected_warning: str | None` and branch on `None` (rely on `filterwarnings = ["error"]` in `pyproject.toml` to fail the no-warn case if a warning leaks). Drop the `import warnings`.

#### Documentation

- **Vague hedging** in docs/comments: "may want to", "might want to", "you could consider". Be definitive.
- **Hardcoded lists** that will go stale (e.g. "Supported models: gpt-4, gpt-5, …"). Link to the catalog instead.
- **Early docstrings** on functions whose logic isn't finalized yet. Write docs after `CHANGES_REQUESTED` reviews are addressed.
- **Backticks around code refs** in docstrings, e.g. `my_function`.
- **Inconsistent terminology** across related docstrings.
- If changed files include `docs/` markdown or source files with docstrings containing code examples, flag that `tests/test_examples.py` must be run.

#### Misc

- **For-loop just to check `isinstance`** on list items — use `any(isinstance(i, T) for i in items)`.
- **Tuple form for `isinstance(x, (A, B))`**.
- **List comprehensions over append loops**.
- **Walrus operator** where it simplifies.
- **Omit redundant name context** (e.g. `UserManager.get_user()` → `UserManager.get()`).
- **Magic strings** — bare string literals used as identifiers, keys, status values. Should be named constants (e.g. `STATUS_PENDING = 'pending'`). Excludes log messages, docstrings, user-facing display text.
- **Double quotes for strings** (pydantic-ai uses single quotes). Exception: prefer double quotes when the string contains single quotes to avoid backslash escaping.

### 3. Report findings

```
## Pattern Review Results

### [Category] - N issues

- **file.py** - [Pattern description]
  Suggestion: [How to fix]

### Summary
- Total issues: N
- Categories affected: [list]
- Recommendation: [pass / review before pushing]
```

For each issue: file path, specific pattern, concrete fix.

## Arguments

- No args: uncommitted changes
- `$ARGUMENTS = 'last'`: last commit
- `$ARGUMENTS = 'branch'`: all commits since main
