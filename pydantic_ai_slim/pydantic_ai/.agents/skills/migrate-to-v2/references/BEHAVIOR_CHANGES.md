# v2.0.0b1 Behavior Changes (Not Surfaced by Deprecation Warnings)

These six items are from the v2.0.0b1 release notes under "Behavior changes to review". They cannot fire a `DeprecationWarning` because the v1 API and the v2 API have the **same shape** — only the runtime behavior differs. Walk through each item with the user after they've cleared all warnings on v1.100.0 and bumped to v2.0.0b1.

---

## 1. Bare `pydantic-ai` install ships fewer extras

**What changed**: the umbrella `pydantic-ai` package no longer pulls every provider extra by default. Providers like `bedrock`, `groq`, and `mistral` must be added explicitly.

**How to detect**: after `uv sync` on v2, run the user's test suite or import-smoke. An `ImportError` or `ModuleNotFoundError` on `boto3`, `groq`, `mistralai`, etc., when the user uses those providers, indicates a missing extra.

**What to change**:

```toml
# pyproject.toml
dependencies = [
    "pydantic-ai[bedrock,groq,mistral]==2.0.0b1",   # add the extras you actually use
]
```

Or with `uv`:

```bash
uv add 'pydantic-ai[bedrock,groq,mistral]==2.0.0b1'
```

**When to leave alone**: user only uses providers still bundled in the slim default (e.g. `openai`, `anthropic`, `google`). Don't blanket-add every extra — that defeats the purpose.

---

## 2. `openai:` model names now use the Responses API

**What changed**: the string prefix `'openai:'` now resolves to `OpenAIResponsesModel`, not `OpenAIChatModel`. Use `'openai-chat:'` to keep the old behavior.

**How to detect**: grep for `'openai:` in the user's code (model strings passed to `Agent(...)`, `infer_model(...)`, YAML/JSON agent specs). Any code that depends on Chat-Completions-specific quirks (audio input, certain `model_settings` keys, response format) will break silently — same API surface, different request/response shapes.

**What to change** — two options, pick based on user intent:

```text
# Option A: stay on Chat Completions (lowest risk for an upgrade-only PR)
Agent('openai-chat:gpt-4o')

# Option B: opt into Responses API (the new default)
Agent('openai:gpt-4o')  # was Chat in v1, now Responses in v2
```

For an upgrade PR, **prefer Option A**. Moving to the Responses API is a separate decision that needs its own review (different streaming model, different tool-use ergonomics).

**When to leave alone**: user has explicitly typed `OpenAIResponsesModel(...)` everywhere (no string form) — the rename doesn't apply.

---

## 3. `WebSearch`/`WebFetch` native-only by default; `MCP(url=...)` local by default

**What changed**: the `WebSearch` and `WebFetch` capabilities used to fall back to a local (provider-adaptive) implementation when the active model didn't support the native tool. In v2 they are **native-only by default** — no silent fallback. Similarly, `MCP(url=...)` now defaults to running locally (proxying the remote MCP server through the agent process) rather than handing the URL to the model provider.

**How to detect**:
- Grep for `WebSearch(`, `WebFetch(`, `MCP(`.
- If the user runs the agent against a model that does *not* natively support web search (e.g. some Anthropic or Google models depending on version), the v1 code silently fell back; v2 will error or skip.

**What to change**: opt back into fallback explicitly with `local=`:

```text
# v1 (implicit fallback)
Agent(..., capabilities=[WebSearch()])

# v2 — same behavior as v1
Agent(..., capabilities=[WebSearch(local=...)])  # TODO(verify) exact signature; see PR #5331
```

Or accept the new default if native-only is what the user actually wants. For `MCP(url=...)`, similarly pass `local=False` if remote-execution semantics are needed.

**When to leave alone**: user only runs against models that support the native tool — no behavior change observable.

---

## 4. Default instrumentation is v5, with aggregated token-usage span attributes

**What changed**: the default `InstrumentationSettings` version moved from v4 (or earlier) to v5. Token-usage span attributes are now aggregated (per-run totals on the root span) rather than only per-request.

**How to detect**: user has Logfire / OTel dashboards that query specific span attribute names like `gen_ai.usage.input_tokens` per child span. After upgrading, panels relying on the old attribute layout may show gaps or wrong totals.

**What to change**:
- If the user wants the old layout: pin the instrumentation version explicitly.
  ```text
  from pydantic_ai.capabilities import Instrumentation
  from pydantic_ai.models.instrumented import InstrumentationSettings
  Agent(..., capabilities=[Instrumentation(InstrumentationSettings(version=4))])
  ```
- Otherwise, update dashboards / alerts to read the new aggregated attributes.

**When to leave alone**: user has no custom OTel/Logfire dashboards — the new defaults are an improvement.

---

## 5. `capture_run_messages()` now captures partial messages from interrupted runs

**What changed**: in v1, `capture_run_messages()` only yielded complete messages — if a run was cancelled, exception'd, or hit a usage limit mid-stream, the in-flight `ModelResponse` was dropped. In v2 it's included with `state='interrupted'`.

**How to detect**: grep for `capture_run_messages`. Any code that iterates the captured messages and assumes every `ModelResponse.state == 'complete'` will need to handle the `'interrupted'` case.

**What to change**:

```text
with capture_run_messages() as messages:
    try:
        await agent.run('...')
    except Exception:
        pass

for msg in messages:
    if isinstance(msg, ModelResponse) and msg.state != 'complete':
        # New in v2 — partial message from an interrupted run.
        continue   # or: log, or: handle
    ...
```

**When to leave alone**: user doesn't use `capture_run_messages()`, or already filters by `state`.

---

## 6. `end_strategy='graceful'`: function tools run alongside successful output tools

**What changed**: in v1, when the model returned an output tool call together with one or more function tool calls in the same response, the function tools were **skipped** (the run ended on the output tool). In v2 the default `end_strategy='graceful'` **runs the function tools first**, then ends the run with the output tool result.

**How to detect**:
- Grep for `end_strategy=` — if explicitly set to `'early'` (the old behavior), no behavior change applies.
- Look for function tools with side effects (writes, API calls, notifications) that the user *relied on being skipped* when the model also produced structured output. These will now execute.

**What to change** — two options:

```text
# Option A: preserve v1 behavior
Agent(..., end_strategy='early')

# Option B: accept the new default — make sure side-effect tools are idempotent.
Agent(...)   # defaults to 'graceful' in v2
```

**When to leave alone**: user's function tools have no side effects, or the user actually wanted them to run (the v1 behavior was widely reported as surprising — that's why it's changing). Document the behavior change in the PR description.

---

## Verification

After walking through all six items:

```bash
uv run pyright
uv run pytest -W error::DeprecationWarning -W error::UserWarning
```

`-W error` upgrades any residual deprecation/user-warning to a hard failure — useful as a regression gate, since v2 still uses `UserWarning` subclasses for its own deprecation infra.
