# v2.0.0b1 Behavior Changes (Not Surfaced by Deprecation Warnings)

Six items where the v1 and v2 APIs have the **same shape** but different runtime behavior, so no warning fires. Walk through each with the user after bumping to v2.0.0b1.

---

## 1. Bare `pydantic-ai` install ships fewer extras

The umbrella `pydantic-ai` no longer pulls every provider extra by default. `bedrock`, `groq`, `mistral` must be added explicitly.

**Detect**: `ImportError`/`ModuleNotFoundError` on `boto3`, `groq`, `mistralai`, etc.

```bash
uv add 'pydantic-ai[bedrock,groq,mistral]==2.0.0b1'
```

Don't blanket-add every extra — only what the user actually imports.

---

## 2. `openai:` model names now use the Responses API

`'openai:'` resolves to `OpenAIResponsesModel` in v2 (was `OpenAIChatModel`). Same API surface, different request/response shapes — code that depends on Chat-Completions quirks (audio input, certain `model_settings`, response format) breaks silently.

**Detect**: grep for `'openai:` in code, YAML, JSON specs.

For an upgrade-only PR, flip every `'openai:...'` to `'openai-chat:...'`. Moving to Responses is a separate decision.

---

## 3. `WebSearch`/`WebFetch` native-only by default; `MCP(url=...)` local by default

v1 silently fell back to a local implementation when the model didn't support the native tool. v2 is native-only — it errors or skips instead. `MCP(url=...)` flips the other way: v1 handed the URL to the provider; v2 proxies locally.

**Detect**: grep for `WebSearch(`, `WebFetch(`, `MCP(`. Behavior only diverges if the active model lacks native support.

Opt back into v1 semantics with `local=...` on `WebSearch`/`WebFetch`, or `local=False` on `MCP(url=...)`. Signature: see PR #5331.

---

## 4. Default instrumentation is v5, with aggregated token-usage span attributes

Token-usage attributes are now aggregated on the root span (per-run totals) rather than only per-request. Dashboards querying per-child-span `gen_ai.usage.*` may show gaps.

Pin the old layout if needed:

```python
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
Agent(..., capabilities=[Instrumentation(InstrumentationSettings(version=4))])
```

---

## 5. `capture_run_messages()` now captures partial messages from interrupted runs

v1 dropped the in-flight `ModelResponse` on cancellation/exception/usage-limit; v2 includes it with `state='interrupted'`. Code that assumes every captured `ModelResponse.state == 'complete'` needs to handle the new case.

```python
for msg in messages:
    if isinstance(msg, ModelResponse) and msg.state != 'complete':
        continue
    ...
```

---

## 6. `end_strategy='graceful'`: function tools run alongside successful output tools

When the model returns an output tool call together with function tool calls in the same response: v1 skipped the function tools, v2's default `'graceful'` runs them first.

**Detect**: grep `end_strategy=`. Look for function tools with side effects that previously relied on being skipped.

Set `end_strategy='early'` to preserve v1 behavior, or make side-effect tools idempotent and accept the new default.
