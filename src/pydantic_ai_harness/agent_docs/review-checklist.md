# Review Checklist

Use this before opening a PR or reviewing a capability change.

## Product Fit

- The capability has a clear user or dogfooding need.
- The behavior belongs in harness, not Pydantic AI core.
- The public API is small and named around user concepts.
- The capability composes with relevant existing capabilities.

## Implementation

- Public exports are intentional.
- Private helpers stay private.
- Types are precise; new public signatures do not use `Any`.
- No casts are used to paper over type design.
- The implementation uses Pydantic AI hooks/toolsets instead of duplicating core
  runtime behavior.
- Capability ordering is justified when present.
- Dependency changes were made through `uv` and have a clear reason.

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

## Tests

- Tests cover the public `Agent(..., capabilities=[...])` path where possible.
- Lower-level tests cover lifecycle, schemas, retries, and metadata when needed.
- Error paths and important option combinations are covered.
- Relevant protocol-shaped output is snapshotted.
- `make lint`, `make typecheck`, and `make test` pass before handoff.

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
  "Overflowing Tool Output", not "Overflow"; "Runtime Authoring", not
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
