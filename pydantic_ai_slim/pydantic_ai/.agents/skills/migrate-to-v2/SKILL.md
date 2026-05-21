---
name: migrate-to-v2
description: Migrate a Python codebase from Pydantic AI v1.x to v2.0. Use when the user mentions upgrading to v2, sees DeprecationWarnings from pydantic_ai or pydantic_graph, references v2.0.0b1, asks to clear v1 deprecations, or imports symbols this skill knows are removed in v2 (MCPServerStdio, Agent.to_a2a, stream_responses, OpenAIModel, GeminiModel, AGUIApp, OutlinesModel, tool_from_aci, GrokProvider, GoogleGLAProvider, GoogleVertexProvider, Usage, pydantic_graph.beta, etc.).
license: MIT
compatibility: Requires Python 3.10+
metadata:
  version: "1.0.0"
  author: pydantic
---

# Migrate to Pydantic AI v2.0

v2.0.0b1 was forked from v1.100.0. v1.100.0 emits `PydanticAIDeprecationWarning` / `PydanticGraphDeprecationWarning` / `PydanticEvalsDeprecationWarning` (all `UserWarning` subclasses, shown by default) for nearly everything v2 removed. The migration path is: bump to v1.100.0, clear every warning, then bump to v2.0.0b1 and review the behavior changes that warnings can't surface.

## When to Use This Skill

Invoke this skill when:
- User asks to upgrade to Pydantic AI v2, v2.0, or v2.0.0b1.
- User reports `DeprecationWarning` / `PydanticAIDeprecationWarning` from `pydantic_ai`, `pydantic_graph`, or `pydantic_evals`.
- User's code imports any symbol listed in `references/DEPRECATIONS.md` (MCPServerStdio, OpenAIModel, GeminiModel, Usage, AGUIApp, etc.).
- Install fails on v2.0.0b1 with a missing extra (e.g. `bedrock`, `groq`, `mistral`).

Do **not** use this skill for:
- Greenfield projects on v2 — use `building-pydantic-ai-agents` instead.
- Pydantic Validation (`pydantic`) v1→v2 — different library.
- v1.x patch upgrades — only triggers for the v1→v2 jump.

## Migration Procedure

Run these steps in order. Each step's output feeds the next.

### 1. Snapshot the starting state

- Confirm clean working tree (`git status`).
- Record current pinned version: `grep -E 'pydantic-ai|pydantic_ai' pyproject.toml uv.lock requirements*.txt 2>/dev/null`.
- Note whether the project pins `pydantic-ai` (the umbrella), `pydantic-ai-slim`, `pydantic-graph`, or `pydantic-evals` separately — all must move in lockstep.

### 2. Pin v1.100.0 and capture every warning

```bash
uv add 'pydantic-ai==1.100.0'   # or: pip install 'pydantic-ai==1.100.0'
uv sync                          # if pyproject + uv.lock are both pinned
```

Run the user's full test suite (not just imports) with warnings visible:

```bash
PYTHONWARNINGS='default' uv run pytest -W default::DeprecationWarning -W 'default::UserWarning'
```

If there is no test suite, run their actual entry point / smoke script. **Importing modules alone is not enough** — many deprecations only fire on construction (e.g. `Agent(instrument=True)`) or at runtime (e.g. inside an async tool, on the first call to `result.usage()`).

Capture every warning whose `filename` lives in the user's code, not under `site-packages/pydantic_ai*`. Group by warning message. Make a checklist.

### 3. Apply codemods, one warning class at a time

For each unique warning message:

1. Read `references/DEPRECATIONS.md` (load via `Read` on demand — do not pre-load).
2. Apply the v1 → v2 transformation listed there. Make one logical edit per commit if the user wants reviewable history.
3. If the warning text is not in the table, `WebFetch` <https://ai.pydantic.dev/changelog/> for the canonical fix before guessing.

**Do not stop at the first match.** A single user file often triggers multiple related warnings — e.g. `Agent(mcp_servers=[MCPServerStdio(...)])` fires *two* warnings (`mcp_servers=` kwarg AND `MCPServerStdio` class). Fixing the kwarg alone leaves the import broken in v2. After every edit, re-read the file and scan for other deprecated symbols listed in `DEPRECATIONS.md` — especially imports, since import-line deprecations are easy to miss when you're focused on a call site.

**Crucial**: when migrating multiple `Agent(...)` kwargs that all map to capabilities (`instrument=`, `history_processors=`, `prepare_tools=`, `prepare_output_tools=`, `event_stream_handler=`), MERGE them into a single `capabilities=[...]` list. Do not create multiple `capabilities=` kwargs — Python will reject the second one. **`tool_retries=` and `output_retries=` are different**: they collapse into a `retries={'tools': N, 'output': M}` dict on the same `retries=` kwarg, *not* into `capabilities=`. See A6 in `references/DEPRECATIONS.md`.

```text
# v1 — three deprecated kwargs
Agent(
    'openai:gpt-4o',
    instrument=True,
    history_processors=[strip_pii],
    prepare_tools=tweak_tools,
)
# v2 — one capabilities list
from pydantic_ai.capabilities import Instrumentation, ProcessHistory, PrepareTools
Agent(
    'openai-chat:gpt-4o',  # see step 5 — `openai:` now means Responses API
    capabilities=[
        Instrumentation(),
        ProcessHistory(strip_pii),
        PrepareTools(tweak_tools),
    ],
)
```

### 4. Re-run until zero warnings

Rerun step 2's command. Repeat step 3 for any remaining items. Do not move on while any user-code-attributed warning remains.

### 5. Bump to v2.0.0b1 and review behavior changes

```bash
uv add 'pydantic-ai==2.0.0b1'
uv sync
```

Read `references/BEHAVIOR_CHANGES.md` and walk through every applicable item with the user:
- Bare install ships fewer extras — add `bedrock`, `groq`, `mistral` explicitly if used.
- `openai:` prefix now → Responses API. Flip to `openai-chat:` to stay on Chat Completions.
- `WebSearch`/`WebFetch` capabilities default to native-only; `MCP(url=...)` defaults to local.
- Default instrumentation is v5 with aggregated token-usage span attributes — dashboards may need updates.
- `capture_run_messages()` now also captures partials from interrupted runs.
- `end_strategy='graceful'`: function tools requested alongside a successful output tool now run by default (v1 skipped them).

### 6. Verify

```bash
uv run pyright          # or: mypy
uv run pytest
```

Surface remaining failures grouped by category. Type errors usually indicate signature changes (`result.usage()` → `result.usage`, tuple unpacking in `stream_response`). Test failures often come from `end_strategy` change or the Responses-API flip.

## Edge Cases

- **User on much older v1 (e.g. v1.50)**: bump to `v1.100.0` first as an intermediate step. Do not skip — older v1s may not emit warnings for everything v2 removed.
- **Import-line deprecations** (e.g. `from pydantic_ai.mcp import MCPServerStdio`): the import itself triggers the warning via `__getattr__`. Both the import and the usage need to change.
- **Runtime-only deprecations**: `result.usage()`, `stream.get()`, and similar fire only when called. Import-smoke is insufficient; run the real flow.
- **`# type: ignore` suppressing the typing-level deprecation**: the runtime `warnings.warn` still fires. The codemod is still required.
- **Pinned in both `pyproject.toml` and `uv.lock`**: bump both, then `uv sync`. Skipping `uv sync` leaves stale resolution.
- **Mixed-package install** (`pydantic-ai-slim` extras pinned separately, or direct `pydantic-graph` / `pydantic-evals` pins): bump every Pydantic-AI-family package to the matching v1.100.0 / v2.0.0b1 version. They share a release train.
- **Capability-kwarg merge**: `instrument=`, `history_processors=`, `prepare_tools=`, `prepare_output_tools=`, `event_stream_handler=` all become entries in one `capabilities=[...]` list. Do not create separate `capabilities=` kwargs (Python will reject a duplicate kwarg).
- **Removed integrations**: `pydantic_ai.ext.aci` and `OutlinesModel`/`OutlinesProvider` have no in-tree replacement. The codemod is "remove the integration and tell the user where it lives now" (`Tool.from_schema` against `aci.ACI().functions.get_definition(...)` for ACI; an upstream issue for Outlines).
- **Tests that assert deprecation**: `pytest.warns(DeprecationWarning)` or `pytest.warns(PydanticAIDeprecationWarning)` blocks around deprecated APIs must be removed or rewritten to assert the new API instead.
- **`@deprecated_callable_property` traps**: in v1.100.0, `result.usage` / `result.timestamp` / `stream.response` are properties that return an object which both *is* the value and *is callable*. Calling it (`result.usage()`) emits a warning but still returns the right type, so tests pass while warnings fire — easy to miss in step 2. Grep for `\.usage()`, `\.timestamp()`, `\.get()` on stream/result objects.
- **`pydantic_graph` users**: `from pydantic_graph import Graph, BaseNode, End, ...` triggers a graph-runner deprecation. The builder API at the top level (`GraphBuilder`) is the v2 form. `pydantic_graph.beta.*` is also deprecated — drop the `.beta` segment.
- **`Agent(retries=)` is *not* deprecated** (it was un-deprecated in commit `55775fbe4`). Only `tool_retries=` and `output_retries=` ctor kwargs were dropped, and the v2 replacement is **a dict on `retries=`**, not a capability: `retries={'tools': N, 'output': M}` (or `retries=N` to set both budgets to the same value). The deprecation warning spells this out — read it carefully.
- **`DeferredToolRequests` lives in `pydantic_ai.tools` in v2, not `pydantic_ai.output`.** (It's also re-exported from top-level `pydantic_ai`.) The v1→v2 rename from `DeferredToolCalls` plus the field rename `.tool_calls` → `.calls` both apply.

## Validation

After the user has migrated their own code, fail CI on any lingering deprecation:

```bash
PYTHONWARNINGS='error::pydantic_ai.exceptions.PydanticAIDeprecationWarning' uv run pytest
```

For an ad-hoc check on an import, wrap it in `warnings.catch_warnings(record=True)` and filter to `PydanticAIDeprecationWarning` / `PydanticGraphDeprecationWarning` / `PydanticEvalsDeprecationWarning`.

## Guardrails

- Never pass `--no-verify` to `git commit`. If pre-commit fails, fix the cause and make a new commit.
- Do not restructure unrelated code while migrating. One concern per commit.
- Do not invent v2 symbol names. If a warning's "use X instead" text disagrees with what you'd guess, trust the warning.
- Do not add `Co-Authored-By: Claude` to migration commits.
- Do not edit the user's pinned versions in `pyproject.toml` without also running `uv sync` / `uv lock`.

## Supporting Files

- `references/DEPRECATIONS.md` — full v1 → v2 codemod table, indexed by warning-message substring. Load when applying step 3.
- `references/BEHAVIOR_CHANGES.md` — the six non-warning behavior changes from the v2.0.0b1 release notes, expanded with detection + remediation. Load at step 5.
