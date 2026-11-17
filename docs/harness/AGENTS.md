# `docs/harness/` guidelines

Rules for the Pydantic AI Harness pages. They add to [`docs/AGENTS.md`](../AGENTS.md). Where these rules or
`docs-conventions.md` differ from `docs/AGENTS.md`, such as the version-promise blockquote, they apply under
`docs/harness/`.

Read [`docs-conventions.md`](../../src/pydantic_ai_harness/agent_docs/docs-conventions.md) before you add, rename, or
edit a page.

- Each capability page has a twin: the `README.md` in the capability's package under
  `src/pydantic_ai_harness/pydantic_ai_harness/`. Update both for every user-facing change.
- Keep this directory flat. Do not add `capabilities/` or `experimental/` subdirectories.
- Follow "Writing style" in [`src/pydantic_ai_harness/AGENTS.md`](../../src/pydantic_ai_harness/AGENTS.md).
- Run the `docs-parity-reviewer` skill on a capability change as the last documentation check before merge. Treat
  its blocking findings as merge blockers.
