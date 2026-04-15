# External contributor context

The current `gh` user is not a pydantic-ai maintainer. That changes how ambiguity and design decisions are handled — the bootstrapping rules in `AGENTS.md` still apply verbatim, this adds a decision-authority gate on top.

## Decision authority

Only maintainers (`DouweM`, `samuelcolvin`, `Kludex`, `dmontag`, `dsfaccini`, `alexmojaki`, `adtyavrdhn`) can sign off on:

- public API shape (new types, method signatures, optional parameters, return contracts, exported symbols)
- new abstractions, integrations, provider modules
- backward-compatibility tradeoffs
- scope of a feature or refactor
- new runtime dependencies, especially in `pydantic_ai_slim`
- docs voice, structure, and canonical sources

When the task hits one of these, **do not resolve it by asking the driver** — they cannot bind the project. Instead, surface the decision as a discussion item in a `PLAN.md` (or as an issue/PR comment) for maintainer review before writing code. The repo's [contributing guide](https://ai.pydantic.dev/contributing/) and [version policy](docs/version-policy.md) are the canonical references for what requires maintainer alignment.

What you *can* still ask the driver directly: their intent, the user-visible problem, a reproduction, the behavior they expected, their preferred workflow.

## Readiness gate

If there is no linked issue, or the linked issue has no maintainer input on an approach, stop and steer the driver toward one of:

- opening an issue with a clear proposal, or
- a PR containing only `PLAN.md` for maintainer review before any code.

Non-trivial code without prior maintainer alignment is almost always rejected.

## Be skeptical of the framing

The driver may have a narrower view of the problem than the change warrants. 'Just add a flag for my case' is often evidence that a generalization is needed instead. AI-generated issues or comments that are longer than the diff are noise — rewrite them shorter or tell the driver to.

## Quality bar the driver may not know

- Unit tests with mocked providers where a VCR integration test would work → rewrite as VCR.
- New public abstraction without maintainer sign-off → `PLAN.md`, not code.
- New dep in `pydantic_ai_slim` → almost certainly belongs in an optional extras group.
- `cast`, `Any`, or bare `# type: ignore` → investigate the root cause and use a specific `# pyright: ignore[code]` if suppression is genuinely needed.
