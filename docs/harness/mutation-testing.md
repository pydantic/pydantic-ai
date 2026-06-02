# Mutation Testing

Mutation testing complements the 100% branch-coverage requirement: coverage
proves every line and branch runs, mutation testing proves the assertions
actually pin the behavior down.

Covers `pydantic_ai_harness/filesystem/_toolset.py` and
`pydantic_ai_harness/shell/_toolset.py`.

Run with [mutmut](https://mutmut.readthedocs.io/) v3 via `scripts/run-mutmut.sh`,
which installs mutmut ephemerally with `uv run --with` — no dev dependency
required.

```bash
scripts/run-mutmut.sh run --max-children 1
scripts/run-mutmut.sh results
scripts/run-mutmut.sh show <mutant-name>
```

## Interpreting survivors

A surviving mutant is either a missing test or an equivalent mutant — a change
that produces behavior no test could distinguish from the original. Triage each
survivor; the recurring equivalent-mutant categories in this codebase are:

- **Trampoline default params** — mutmut v3 wraps functions, and the wrapper
  keeps the original defaults, so a mutated default is never observed.
- **Omitted `name=` in `add_function()`** — pydantic-ai falls back to
  `method.__name__`, which equals the explicit name being mutated away.
- **`'utf-8'` encoding mutations** — Python's codec lookup is case-insensitive
  and UTF-8 is the default text encoding, so case/omission changes are no-ops.
- **`errors='replace'` mutations** — exercised only by invalid bytes; valid
  UTF-8 test data never invokes the error handler.
- **Unreachable `except` blocks** (marked `pragma: no cover`) — paths that
  can't be triggered in the test environment.
- **`CancelScope(shield=True)` flips** — require an outer cancellation during
  the near-instant cleanup window.

Anything outside these categories should be treated as a real gap and killed
with a new test.

## Limitations

Trio-parametrized tests are excluded during mutation testing (`-k 'not trio'`
in `pyproject.toml [tool.mutmut]`) because trio segfaults in mutmut's
subprocess environment on Python 3.14 / macOS. The kill rate is unaffected —
the trio tests exercise the same code paths as the asyncio tests.
