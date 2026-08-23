# Review Checklist

Use this before opening a PR or reviewing a capability change.

## Product Fit

- The capability has a clear user or dogfooding need.
- The behavior belongs in harness, not Pydantic AI core.
- The public API is small and named around user concepts.
- The capability composes with relevant existing capabilities.

## Implementation

- Public exports are intentional.
- Every public capability class is lazily re-exported from the top-level
  `pydantic_ai_harness` package (`__all__` + `TYPE_CHECKING` + `__getattr__`;
  ACP/experimental excepted) — see `capability-authoring.md` "Capability
  Submodules And Exports".
- Private helpers stay private.
- Types are precise; new public signatures do not use `Any`.
- No casts are used to paper over type design.
- The implementation uses Pydantic AI hooks/toolsets instead of duplicating core
  runtime behavior.
- Capability ordering is justified when present.
- Dependency changes are required, linked to an issue, and made through `uv`;
  every PR touching `pyproject.toml` or `uv.lock` carries
  `dependencies:approved` for the current head.
- A capability that adds heavy CI machinery (a Docker image, an external service
  with a secret, a large system binary, live network calls) scopes its expensive
  job to its own paths and keeps the aggregate check green when that job is
  skipped. See `capability-authoring.md` "CI And Dependency Footprint".

## Executable Boundaries

Apply these checks when a change invokes a command/parser, process/container,
or network service, or changes CI:

- Trace user or model input through every transformation to the downstream
  parser. Verify guards against the syntax that parser accepts, including
  aliases, abbreviations, normalization, separators, and repeated options.
- Trace each created resource through readiness, use, and cleanup. Failed
  cleanup is reported, and tracked identity remains recoverable until cleanup
  succeeds.
- Trace each configurable address, endpoint, path, or credential with a
  non-default sentinel through provisioning, readiness, invocation, and
  teardown.
- Measure limits on the final value the caller receives, including framing,
  truncation markers, envelopes, and metadata.
- For each CI secret or write permission, trace event/ref -> checked-out code ->
  credential -> executable step. PR-controlled code must not receive repository
  or environment secrets; step-level scoping does not create that boundary.
  Check trusted and fork PR outcomes, including the aggregate required check.
- Starting from each conditional job's executed command, ensure its path filter
  includes the task-runner or script entry point and every dependency,
  configuration, image, and workflow input that can change execution.

Passing coverage alone is not evidence that these contracts hold. For
downstream-parser and external-runtime claims, run a focused reproduction and
retain the command and result as review evidence.

## Stale Or Pre-Merge PRs

Run these checks when adopting, rebasing, or re-reviewing a PR that was opened
well before now, or that was built against unreleased Pydantic AI changes.

- Temporary `[tool.uv.sources]` pins to a branch or git ref are removed once the
  upstream change they waited on has landed in a released `pydantic-ai-slim`.
- Each upstream Pydantic AI PR or branch the change rode on has merged. Link the
  upstream PR and its merge state.
- The touched surface has not drifted: re-check the capability, hook, and toolset
  signatures it depends on against current main, not against the state at fork
  time.
- Behavior the PR worked around because a primitive was missing is reconsidered
  if that primitive now exists in core.
- A flood of pyright or import errors right after merging main or rebasing is
  usually uninstalled extras, not a real regression. Re-sync (`make install`, or
  `uv sync --frozen --all-extras --group lint`) before treating the merge as
  broken; errors that name third-party types (`modal`, `openai`, ...) as unknown
  in files the PR did not touch are the tell.

## Issue References

Run these checks when a change adds a link to an open issue in a docs page, a
README, or source code.

- Comment on that issue in the same PR. A link from shipped text to an issue is
  one-directional: a reader who opens the issue later sees no sign that a
  released artifact documents it as forthcoming or depends on what it describes.
- State in the comment what now references the issue, and which constraint that
  reference imposes on whoever implements it. A bare backlink is not enough. The
  constraint is the part that reader would otherwise reconstruct from the docs
  page.
- Closing an issue in the same PR does not remove the requirement. The docs page
  outlives the close, and the qualifiers it carries are often recorded nowhere
  else.

## Tests

- Tests cover the public `Agent(..., capabilities=[...])` path where possible.
- Lower-level tests cover lifecycle, schemas, retries, and metadata when needed.
- Error paths and important option combinations are covered.
- For a stateful capability, or one that overrides `for_run`, require a public
  `Agent` durability-composition test for every supported wrapper or an
  explicit, tested incompatibility. Mocked lifecycle tests alone do not
  establish state continuity across activity, process, or replay boundaries.
- Relevant protocol-shaped output is snapshotted.
- **Harness cassettes are re-recorded on composition change.** Each packaged
  harness has a recorded end-to-end integration test (e.g.
  `tests/coder/test_coder_integration.py`) that runs it against a real task.
  Any change to that harness's composition, defaults, or instructions
  re-records the cassette in the same PR
  (`uv run --env-file .env --no-sync pytest -p no:cacheprovider <test> --record-mode=rewrite`) —
  a green replay of a stale cassette proves nothing about the new definition.
- Run the local verification commands in `AGENTS.md` before handoff.

## Docs

Every released capability ships two hand-maintained docs that must stay in sync
with the code and with each other:

- the **README** next to the implementation (`pydantic_ai_harness/<capability>/README.md`,
  or `pydantic_ai_harness/experimental/<capability>/README.md` for ACP), which
  serves GitHub and PyPI, and
- the **unified doc** on the docs site, flat under `docs/<capability>.md`. The
  sidebar is a flat list under "Pydantic AI Harness" -- no `capabilities/` or
  `experimental/` subdirectories.

Checks:

- Both the README and the unified doc are updated for any user-facing change
  (public class, params, defaults, tool names, extras, safety semantics). A
  change reflected in only one of them is a defect, not a follow-up.
- **Harness blown-out parity.** A packaged harness (`Coder`, `Researcher`, ...)
  has its composition written out in full — default instructions and allowlists
  included, not imported — in its docs page's "Blown-out equivalent" AND in its
  `examples/` counterpart. Any change to a harness's composition or defaults
  updates all three together (implementation, docs page, example) in the same
  PR; drift here is a defect, not a follow-up.
- The two do not contradict each other or the source on extras, option names,
  defaults, or safety caveats.
- Every snippet in both docs is runnable: all imports present, class/param names
  match the source, model ids unchanged from what the source uses. Imports use
  the canonical module path (never `pydantic_ai_harness.experimental.*` for a
  graduated capability).
- **Purpose-first lead.** The opening paragraph of each page and README states
  what the capability is for and when to reach for it -- no internal hook or
  class name (`before_model_request`, `after_tool_execute`, ...) before the
  purpose. Mechanism belongs lower down.
- **Name matches the capability.** The doc filename, its `# H1`, and the
  README's `# H1` all use the capability's descriptive name (e.g.
  "Tool Output Limits", not "Overflow"; "Runtime Capability Creation", not
  "Authoring").
- **Source link.** Each page links its source module
  (`https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/<module>/`)
  so a reading agent can verify behavior. Where the capability exposes a public
  class, the page may also end with a `## API reference` section of
  `::: pydantic_ai_harness...` autodoc blocks (auto-expanded from the docstring,
  not hand-written).
- **Stability framing.** Graduated capabilities carry the soft note "The API may
  change between releases..." mirrored from their README -- NOT a
  `HarnessExperimentalWarning` block or "removed in any release" wording. ACP is
  the only page that keeps an `!!! warning "Experimental"` (it may still be
  removed).
- Links: harness-internal links are relative `.md`; Pydantic AI docs use
  root-relative internal links `/ai/<section>/<page>/` (verify the route resolves
  on the live `pydantic.dev/docs` site before using it).
- Docs explain composition constraints and safety implications.
- The PR links an issue.

The mechanical half of these checks (README present + linked, flat page present,
source link present, name matches, no experimental strings on non-ACP pages, no
hook name in the lead) is enforced by `tests/test_docs_parity.py`. The semantic
half (does the prose match the code, are snippets truly runnable) is what the
reviewer below is for.

This is the last documentation gate before merge. Run the `docs-parity-reviewer`
subagent (`.agents/agents/docs-parity-reviewer.md`) on the change as the final
review step; treat its blocking findings as merge blockers.
