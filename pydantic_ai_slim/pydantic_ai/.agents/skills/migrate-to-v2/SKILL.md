---
name: migrate-to-v2
description: Migrate a Python codebase from Pydantic AI v1.x to v2.0. Use when the user mentions upgrading to v2, sees DeprecationWarnings from pydantic_ai / pydantic_graph / pydantic_evals, references v2.0.0b1, or imports symbols removed in v2 (MCPServerStdio, OpenAIModel, GeminiModel, Usage, AGUIApp, OutlinesModel, tool_from_aci, GrokProvider, GoogleGLAProvider, GoogleVertexProvider, pydantic_graph.beta, Agent.to_a2a, stream_responses, ...).
license: MIT
compatibility: Requires Python 3.10+
metadata:
  version: "1.0.0"
  author: pydantic
---

# Migrate to Pydantic AI v2.0

v2.0.0b1 was forked from v1.100.0. v1.100.0 emits `PydanticAIDeprecationWarning` (and the graph/evals equivalents — all `UserWarning` subclasses, shown by default) for nearly everything v2 removed. The path is: pin v1.100.0, clear every warning, bump to v2.0.0b1, review behavior changes that warnings can't surface.

Do not use this skill for greenfield v2 projects (use `building-pydantic-ai-agents`), Pydantic Validation v1→v2, or v1.x patch bumps.

## Procedure

### 1. Snapshot

- Clean working tree (`git status`).
- Find every pin: `grep -E 'pydantic-(ai|graph|evals)' pyproject.toml uv.lock requirements*.txt`. All Pydantic-AI-family packages share a release train and must move together.

### 2. Pin v1.100.0 and surface warnings

```bash
uv add 'pydantic-ai==1.100.0' && uv sync
PYTHONWARNINGS='default' uv run pytest -W default::DeprecationWarning -W default::UserWarning
```

If there is no test suite, run the actual entry point. Import-smoke is **not enough** — many deprecations only fire on construction or first call (`Agent(instrument=True)`, `result.usage()`, `stream.get()`). Collect every warning attributed to user code (not `site-packages/pydantic_ai*`), grouped by message.

### 3. Apply codemods

For each unique warning, look up the v1→v2 transformation in `references/DEPRECATIONS.md` (load on demand). Trust the warning's "use X instead" text over guesses. If a warning isn't in the table, `WebFetch` <https://ai.pydantic.dev/changelog/> before inventing a symbol.

Two things that bite repeatedly:

- **One file, many warnings.** `Agent(mcp_servers=[MCPServerStdio(...)])` fires for the kwarg *and* the class. After every edit, rescan the file for other deprecated symbols — import lines especially.
- **Merge capability kwargs into one list.** `instrument=`, `history_processors=`, `prepare_tools=`, `prepare_output_tools=`, `event_stream_handler=` all collapse into a single `capabilities=[...]` (a second `capabilities=` kwarg is a Python error). `tool_retries=` / `output_retries=` are different — they collapse into `retries={'tools': N, 'output': M}` on the same `retries=` kwarg, *not* into `capabilities=`. `Agent(retries=)` itself is not deprecated.

### 4. Re-run until clean

Repeat step 2's command until no user-code warning remains.

### 5. Bump to v2.0.0b1 and review behavior changes

```bash
uv add 'pydantic-ai==2.0.0b1' && uv sync
```

Walk `references/BEHAVIOR_CHANGES.md` with the user — six items the v1 warning system cannot surface (extras, `openai:` → Responses API, `WebSearch`/`WebFetch`/`MCP` defaults, instrumentation v5, `capture_run_messages` partials, `end_strategy='graceful'`).

### 6. Verify

```bash
uv run pyright
uv run pytest
PYTHONWARNINGS='error::pydantic_ai.exceptions.PydanticAIDeprecationWarning' uv run pytest   # CI gate
```

Type errors usually mean a signature change (`result.usage()` → `result.usage`, `stream_responses` tuple unpack). Test failures often come from the `end_strategy` change or the `openai:` → Responses flip.

## Guardrails

- Never `git commit --no-verify`. Never add `Co-Authored-By: Claude`.
- One concern per commit. Don't restructure unrelated code.
- Don't edit pins without re-running `uv sync` / `uv lock`.
- Users on much older v1 (e.g. 1.50): step through 1.100.0 first — older v1s don't warn for everything v2 removed.

## Supporting Files

- `references/DEPRECATIONS.md` — v1 → v2 codemod table indexed by warning-message substring. Load at step 3.
- `references/BEHAVIOR_CHANGES.md` — the six non-warning behavior changes. Load at step 5.
