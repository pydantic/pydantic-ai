# Pydantic AI Harness

## Repository purpose

`pydantic-ai-harness` is the first-party capability library for Pydantic AI.

Pydantic AI core owns the primitive runtime: agent loop semantics, normalized
messages, model/provider/profile behavior, tool execution semantics, durable
execution primitives, and generic capability hooks.

Harness owns optional, batteries-included compositions built from those
primitives: coding-agent tools, guardrails, memory, context management, repo
tools, verification loops, skills, planning, sub-agents, and other reusable
agent behaviors.

When a change needs new core semantics, stop and propose the Pydantic AI core
change instead of reimplementing core behavior in harness.

## Vocabulary

- **Capability**: an `AbstractCapability` subclass that bundles tools, hooks, instructions, and model settings into a reusable unit. This is the core abstraction of pydantic-ai-harness.
- **Hook**: a lifecycle method on `AbstractCapability` that intercepts agent graph execution (e.g. `before_model_request`, `wrap_run`, `after_tool_execute`)
- **Toolset**: a collection of tools that a capability can provide to the agent
- **Guard**: a type of capability that validates inputs/outputs or controls tool access (e.g. `InputGuard`, `OutputGuard`)
- **Harness**: this package -- a collection of pre-made capabilities for Pydantic AI.
- **AICA**: AI Code Assistant -- the automated agent that implements issues, reviews plans, and handles PR feedback
- **Ralph loop**: the state-machine-based workflow that drives AICA through phases (TRIAGE -> GOALS -> PLAN -> CODE -> VERIFY -> REVIEW -> PUBLISH)
- **DDD+ protocol**: classification system for PR review comments (do, dismiss, discuss, waiting, done)

## AICA preflight

Before implementing or reviewing a capability change:

1. Read `agent_docs/index.md`.
2. Read the linked `agent_docs/` guide for the task.
3. Read the public Pydantic AI docs for every integration point you touch:
   - capabilities: <https://pydantic.dev/docs/ai/capabilities/overview/>
   - hooks: <https://pydantic.dev/docs/ai/core-concepts/hooks/>
   - toolsets: <https://pydantic.dev/docs/ai/tools-toolsets/toolsets/>
   - advanced tools: <https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/>
   - agents: <https://pydantic.dev/docs/ai/core-concepts/agent/>
   - testing: <https://pydantic.dev/docs/ai/guides/testing/>
4. Inspect the installed `pydantic_ai` package source for exact hook/toolset
   signatures when needed. Do not assume a contributor's local checkout layout.
5. Use `pydantic_ai_harness.code_mode` as the exemplar for capability shape,
   docs, tests, and public exports until another capability becomes a better
   example. Capabilities live in their own top-level submodule
   `pydantic_ai_harness/<name>/` (module name = capability name; one module per
   capability or strategy) and are not re-exported from the root `__init__.py`,
   so each keeps its own optional dependencies. The `experimental` tier is
   retired; ACP is the sole remaining experimental capability (see
   `agent_docs/capability-authoring.md`, "Capability Submodules And Exports").

## Branch context (optional)

`.agents/skills/branch-context/` is opt-in per-branch state for work that spans
sessions: an issue brief, an append-only decisions log, and session handoffs.
The scaffolding (`SKILL.md`, scripts, `*.template.md`) is committed; the
instances are git-ignored and created per branch.

- If `.agents/skills/branch-context/issue-brief.md` exists, read it plus
  `pr-decisions.md` and the latest handoff before making design decisions.
- To adopt a branch, instantiate the surfaces from the templates or run
  `/adopt-pr`. Contributors who never instantiate them can ignore this section.

## Capabilities API reference

When implementing a new capability, reference these docs:

- <https://pydantic.dev/docs/ai/capabilities/overview/> -- main capabilities documentation, usage patterns, built-in capabilities
- <https://pydantic.dev/docs/ai/core-concepts/hooks/> -- lifecycle hooks reference, hook ordering, all hook categories
- <https://pydantic.dev/docs/ai/guides/extensibility/> -- publishing capabilities as packages, spec serialization
- <https://pydantic.dev/docs/ai/tools-toolsets/toolsets/> -- toolset abstraction, building tools for capabilities
- <https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/> -- tool hooks, prepare tools, tool validation
- <https://pydantic.dev/docs/ai/core-concepts/agent/> -- agent configuration, instructions, model settings
- Installed `pydantic_ai.capabilities` source -- `AbstractCapability`, hook signatures, and composition behavior
- Installed `pydantic_ai.toolsets` source -- `AbstractToolset`, `WrapperToolset`, and `ToolsetTool`

## Capability naming

Follow the naming convention in `agent_docs/capability-authoring.md` ("Naming
Capabilities"): a noun when the capability names a thing (a tool or faculty the
model uses, a subsystem, a named strategy); an imperative verb phrase when it
acts on the run and one verb phrase states its entire contract. Never invent a
nominalization for an action, and never name a capability after the problem it
solves.

## Telemetry

OpenTelemetry is part of a feature's design, not a follow-up. Every new
capability decides what it emits before it merges, from the point of view of
someone operating a run in production and asking what happened, why the run
changed course, and what it cost.

Emitting nothing is a valid answer when core's own spans already cover the work.
It is an answer to state in the docs, not a step to skip.

The house pattern (spans on `ctx.tracer`, attribute naming, content behind
`trace_include_content`) is in `agent_docs/capability-authoring.md`
"Telemetry".

## Coding standards

- Python 3.10+ (target version for pyright and ruff)
- **pyright strict** mode -- no `Any` types, full type annotations
- **ruff**: line-length=120, single quotes, max-complexity=15
- CI enforces 100% branch coverage from combined matrix data.
- Do not run coverage locally unless CI reports a specific coverage gap.
- Investigate a reported gap with focused coverage for the flagged file or test.
- docstrings use single backticks (markdown), not RST double backticks
- no typecasting (`as` in TypeScript, `cast()` in Python) -- use type narrowing instead
- prefer the most generic input types possible (reduce dependency chains)
- don't add comments that restate what the code does

## Writing style

Applies to docs, READMEs, docstrings, comments, commit messages, and PR text.

- No em-dashes (`—`). Use `--` for an aside or interruption, or split into two
  sentences. Em-dash-heavy prose reads as machine-generated.
- State facts, not sales copy. Cut marketing superlatives and hype ("blazingly
  fast", "battle-tested", "the single most expensive thing you can do",
  "footgun") and editorializing adjectives ("sprawling", "noisy", "silently").
- Avoid absolute claims ("never", "always", "guaranteed") unless they are
  literally true and load-bearing. Name the specific mechanism instead of the
  slogan.
- Use bold sparingly -- for the lead-in term of a list item, not to emphasize
  whole sentences.
- Document the why, the constraints, and the non-obvious. Don't restate what the
  code or signature already says.
- Prefer plain ASCII punctuation over decorative Unicode (arrows, fancy quotes)
  in prose and comments.

## Package management

- Change dependencies only when required. Use `uv` and link an issue.
- PRs touching `pyproject.toml` or `uv.lock` require the
  `dependencies:approved` label; pushes clear approval.

## Local verification

Run Ruff across the repository. Run Pyright and pytest for the paths that you modify.

```bash
uv run --no-sync ruff format --check .
uv run --no-sync ruff check .
PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run --no-sync pyright pydantic_ai_harness/<module>.py tests/<module>.py
uv run --no-sync pytest -p no:cacheprovider tests/<capability>
```

CI runs the repository-wide typecheck, test, and combined coverage gates.
Do not run repository-wide Pyright, pytest, or coverage locally.
If CI reports a coverage gap, run coverage only for the flagged file or focused test.

## File structure

The tree is discoverable by listing it; only the conventions that are not are
recorded here.

Each released capability is a self-contained package under
`pydantic_ai_harness/<capability>/` (naming and exports are covered in the
preflight above), with tests under `tests/<capability>/`. It ships **two**
hand-maintained docs that must stay in sync: the `README.md` next to the code
(GitHub/PyPI) and the `docs/<capability>.md` page (the docs site at
pydantic.dev/docs/ai/harness). The `docs/` folder is flat -- there are no
`capabilities/` or `experimental/` subdirectories. A user-facing change updates
both; `agent_docs/review-checklist.md` "Docs" and the `docs-parity-reviewer`
subagent enforce the parity before merge.

Do not add placeholder template files for new capabilities. Start from the
existing `CodeMode` package shape, then delete what the new capability does not
need.

## Testing patterns

- Import `TestModel` from `pydantic_ai.models.test` for model behavior.
- Keep real provider calls out of tests.
- `ALLOW_MODEL_REQUESTS = False` is set globally in `conftest.py`
- Tests use `pytest-anyio` for async support
- Each capability test class follows: `TestCapabilityName` with methods `test_<scenario>`
- Prefer tests through `Agent(..., capabilities=[...])` when that is the public
  behavior. Use direct `Toolset`/`RunContext` tests for lower-level lifecycle,
  schema, retry, or wrapper behavior that is hard to isolate through `Agent`.
- Don't import private (`_`-prefixed) helpers into tests. Exercise them through
  the capability's public surface so tests survive internal refactors: drive the
  behavior through `Agent(..., capabilities=[...])`, or import the public class
  re-exported from the capability package's `__init__.py` (e.g.
  `from pydantic_ai_harness.filesystem import FileSystemToolset`, not
  `from pydantic_ai_harness.filesystem._toolset import _content_hash`). When a
  branch is only reachable by calling a private helper directly, mark it
  `# pragma: no cover` rather than reaching into the helper from a test.

## Contributing rules for AICAs

- Always link sources for any claims made during research
- Commit messages should summarize the "why", not the "what"
- PR titles feed the release notes verbatim -- GitHub builds "What's Changed" from them. Write an
  imperative sentence naming the change, and wrap every code identifier (class names, keyword
  arguments, module paths, CLI flags, env vars, file paths) in backticks. No `feat:` / `fix:` /
  `docs:` prefix -- that belongs on the commit subject, not the title. Merged titles predating this
  rule carry prefixes; follow the rule, not the back catalogue.

## Pushing changes

**A restriction is a conclusion you earn from a real failure, not a field you read.** Never report an
operation as blocked, unavailable, or not-permitted based on a metadata flag, a config field, or a
docs claim — attempt it and quote the actual error. (`maintainerCanModify: false` on a PR does *not*
mean you cannot push: it governs the upstream-maintainer auto-grant, not your own access to the
fork.) If you genuinely cannot attempt it, say "not attempted", never "we can't".

**Pushing is not the end of the task.** After you push, do not go idle. The work is done when
**CI is green and there are no unresolved comments** — see the `pushing-commits-to-the-repo` skill
for the full loop.

**Do not leave work uncommitted.** Don't end a turn with unstaged or uncommitted local changes
unless the user's own instructions say otherwise.
