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

- Capability README or root README is updated for user-facing behavior.
- Examples match declared extras.
- Docs explain composition constraints and safety implications.
- The PR links an issue.
