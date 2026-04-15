Pydantic AI is a provider-agnostic GenAI agent framework by the team behind [Pydantic Validation](https://docs.pydantic.dev/) and [Pydantic Logfire](https://pydantic.dev/logfire), used by hundreds of thousands of Python developers. People pick it because the code is simple to read and write, the same API works across every major LLM provider, and the abstractions are flexible enough to build anything from a single-tool chatbot to a multi-agent system without locking them into a particular shape. We optimize for modern idiomatic Python, end-to-end type safety, a tasteful public API, and a developer experience that doesn't fight you. [`README.md`](README.md) has the full 'why use it' pitch; docs live at <https://ai.pydantic.dev>.

# Repo invariants (assume true; never try to disprove)

- Zero pyright errors. Zero failing tests. 100% coverage on `main`. If you see a failure, your branch introduced it — do not waste turns checking whether it was 'already broken'.
- `pre-commit` is the gate: it runs `ruff format`, `ruff lint`, `pyright`, cassette integrity, and more on every commit.
- Commits are authored by the human user only. Never add a `Co-Authored-By: Claude` (or any AI) trailer.

# Non-negotiable rules

- Backward compatible per [version policy](docs/version-policy.md).
- Public API changes are hard to reverse — every new abstraction, type, or flag gets the corresponding amount of care.
- Type-safe public + internal API. No `cast`. No `Any`. Use `# pyright: ignore[code]` with a named error code, never bare `# type: ignore`.
- Prefer VCR integration tests over unit + mock. Tests should resemble how a user would use the public API.
- PRs use the [template](.github/pull_request_template.md) with `Closes #<issue>`. The 'AI generated code' checkbox is the human author's to tick.

# Repository layout (`uv` workspace)

- `pydantic_ai_slim/` — agent framework ([agents](docs/agent.md)). Slim core; optional extras: `openai`, `anthropic`, `google`, `mcp`, `temporal`, `logfire`.
- `pydantic_graph/` — graph library powering the agent loop ([graph](docs/graph.md)).
- `pydantic_evals/` — eval framework ([evals](docs/evals.md)).
- `clai/` — CLI + optional web UI ([cli](docs/cli.md), [web](docs/web.md)).
- Root `pyproject.toml` — umbrella package.

# Bootstrapping on a task

Universal to every driver — maintainer or external contributor — when picking up or resuming an issue/PR:

1. Read the linked issue/PR and every comment with `gh issue view`, `gh pr view`, or `gh api`. Walk cross-linked issues/PRs — maintainer decisions often live in a parent thread.
2. For provider/SDK work, check the provider's current API docs and the SDK's type definitions before assuming behavior. LLM provider APIs change fast.
3. For features that overlap with existing agent libraries, check how they solved it — tasteful API design is the bar.
4. Ask clarifying questions when scope, API shape, or intent are ambiguous. The driver's framing may be narrower than the change warrants.

# Local commands you run

- `make install` — install deps (`uv`, Python 3.10–3.13). One-time per venv.
- `make format && make lint` — fast; run before committing.
- `uv run pytest <targeted_paths>` — run only the tests that cover your change.
- `uv run pytest tests/test_examples.py -k '<docs_file>'` — whenever you add or modify a code snippet in `docs/` or in a docstring.

# Commands you do NOT run locally

- `make test` — CI runs the full suite on every push. Run targeted paths only (see above).
- `make typecheck` — pre-commit runs pyright on every commit; don't re-run manually.
- `make docs` / `make docs-serve` — the docs build is CI's job.

# Committing

Capture pre-commit output in one go so you don't feel the urge to re-run:

    git commit -m '<message>' 2>&1 | tee /tmp/commit-output.txt

If pre-commit fails, read `/tmp/commit-output.txt`, fix the root cause, and retry.

# Post-push flow

After `git push`, wait 10–15 minutes before taking the next step. In that window CI finishes and `devin-ai-integration[bot]` posts an auto-review. **Devin's review has equal weight to a maintainer's** — classify every Devin comment with DDD+ (do / discuss / dismiss / waiting / done) and address it before asking a human.

See the `address-feedback` skill at [`.claude/skills/address-feedback/SKILL.md`](.claude/skills/address-feedback/SKILL.md) for the wait → triage → respond loop.

# Design principles

Strong primitives and general extension points over narrow, opinionated features. Slim default, optional extras. If a proposed change solves one user's case and ignores N others, stop and generalize — or push back on the requester.

# Further reading

- [`agent_docs/index.md`](agent_docs/index.md) and its topic guides ([api-design](agent_docs/api-design.md), [code-simplification](agent_docs/code-simplification.md), [documentation](agent_docs/documentation.md)) — read the relevant sections when touching those areas.
