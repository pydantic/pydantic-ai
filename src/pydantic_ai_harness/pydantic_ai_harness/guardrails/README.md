# Input & Output Guardrails

Validate the user prompt before it reaches the model, and the model output before it reaches the caller.

> [!NOTE]
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/guardrails/)

## The problem

Agents take unstructured input from users and return unstructured output to callers. Without a validation layer, a prompt injection attempt, PII-laden message, or off-topic question goes to the model as-is, and any output the model produces is returned verbatim. The framework does not reason about "this is unsafe to send" or "this is unsafe to show".

## The solution

Two capabilities -- `InputGuard` and `OutputGuard` -- each backed by a `guard` callable you supply. The guard inspects a value (the prompt, or the output) and returns one of four outcomes:

| Outcome | `InputGuard` | `OutputGuard` |
|---|---|---|
| **allow** | send the prompt to the model | return the output to the caller |
| **block** | skip the model call; a refusal message becomes the response (`SkipModelRequest`) | raise `OutputBlocked` |
| **replace** | rewrite the prompt sent to the model (redaction) | substitute a sanitized output |
| **retry** | -- (not valid for input) | send the output back to the model to try again (`ModelRetry`) |

A guard that raises an exception instead propagates it as a hard failure. The asymmetry between input `block` and output `block` is intentional: blocking the input spends no tokens, so a graceful refusal is almost always right; blocking the output means the model already produced something you do not want exposed, so raising forces the caller to decide what to do next.

## Usage

A guard returns a bare `bool` (`True` = allow, `False` = block) for the simple case, or a `GuardResult` for the richer outcomes.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import GuardResult, InputGuard, OutputGuard


def no_secrets(prompt: str) -> bool:
    return 'api_key' not in prompt.lower()


def no_pii(output: object) -> GuardResult:
    if 'SSN' in str(output):
        return GuardResult.block('The response contained personal data.')
    return GuardResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuard(guard=no_secrets),
        OutputGuard(guard=no_pii),
    ],
)
```

`OutputGuard` receives the output unchanged -- no automatic stringification. For a string output the guard reads it directly; for a typed (Pydantic model) output the guard gets the model instance, so pick the serialization that fits the check (read a field, or call `output.model_dump_json()` for JSON text). This avoids the trap of `str(MyModel(...))` producing a `MyModel(field=...)` repr that hides field contents from regex-based checks.

Guards may also be async -- return an awaitable `bool`/`GuardResult`, e.g. to call a moderation API.

## `GuardResult`

Construct a `GuardResult` with its classmethods, not the raw fields:

```python
from pydantic_ai_harness import GuardResult

GuardResult.allow()                 # let the value through
GuardResult.block('reason')         # refuse; `reason` is optional (a default is used otherwise)
GuardResult.replace(cleaned_value)  # substitute a sanitized value and continue
GuardResult.retry('instruction')    # OutputGuard only: ask the model to redo the output
```

The block/retry message is produced at the moment the guard decides, so it can carry the guard's own reasoning rather than a string frozen at construction time.

## Redaction (`replace`)

Return `GuardResult.replace(value)` to sanitize rather than refuse. `InputGuard` rewrites the prompt sent to the model; `OutputGuard` substitutes the output returned to the caller.

```python
def scrub_emails(text: str) -> GuardResult:
    cleaned = EMAIL_RE.sub('[email]', text)
    return GuardResult.replace(cleaned) if cleaned != text else GuardResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuard(guard=scrub_emails),   # strip PII before it reaches the model
        OutputGuard(guard=scrub_emails),  # strip PII before it reaches the caller
    ],
)
```

Input redaction requires sequential mode -- it is incompatible with `parallel=True`, since a parallel guard runs alongside a model call that has already started with the original prompt.

## Retry (`retry`)

`OutputGuard` can send a bad output back to the model instead of blocking it. Return `GuardResult.retry(instruction)` -- the instruction is the retry prompt the model sees. This reuses pydantic-ai's normal retry machinery and counts against the run's output-retry budget.

```python
def must_cite_sources(output: object) -> GuardResult:
    if not has_citations(output):
        return GuardResult.retry('Include at least one source citation.')
    return GuardResult.allow()


OutputGuard(guard=must_cite_sources)
```

## Streaming

`OutputGuard` inspects the **final** output only -- during `run_stream()` partial chunks reach the caller before the guard runs, so a `block` or `replace` verdict cannot un-send content already streamed. Use `run()` / `run_sync()` when the output must be screened before any of it is exposed. `GuardResult.retry()` is **not** supported under `run_stream()` -- pydantic-ai does not retry output during streaming, and a `retry` verdict there surfaces as `UnexpectedModelBehavior`. `InputGuard` (including `parallel=True`) works the same in streamed and non-streamed runs.

## Tracing

`replace` and `block` are recorded as spans on the active OpenTelemetry tracer, so a redaction or refusal shows up in Logfire traces (`guardrail redacted input`, `guardrail blocked output`, etc.) with `guardrail.*` attributes. Content attributes -- the original/replacement values for a redaction and the refusal `message` for a block -- are attached **only** when `RunContext.trace_include_content` is enabled, since these can quote the very content the guard exists to keep out of traces. `retry` needs no special tracing: the retried model request appears in the trace on its own.

`OutputGuard` declares `position='outermost', wrapped_by=[Instrumentation]` so its block/redact spans are always captured by an enclosing `Instrumentation` span regardless of how the user orders capabilities. `InputGuard` declares `position='innermost'` so any capability that morphs messages (a prompt rewriter, a context manager) runs first and the guard sees the final prompt the model will receive.

## Parallel input guards

A slow guard (an LLM classifier, a network call) run sequentially adds its latency to every turn. Set `parallel=True` to run the guard concurrently with the model call instead, overlapping the two so the guard adds no latency on the pass path. The model call is cancelled the moment the guard reports a violation.

```python
InputGuard(guard=slow_async_classifier, parallel=True)
```

Parallel mode trades tokens for latency: sequential mode never calls the model when the guard blocks, but parallel mode has already started the model call -- if the guard trips only after the model has responded, those tokens were spent. For fast local checks (regex, keyword lookup) sequential is the better default. `replace` is not available under `parallel=True` (see [Redaction](#redaction-replace)).

## Accessing run context

A guard may take a `RunContext` as its first parameter when it needs run state -- `deps` for tenant- or role-aware policy, message history for conversation-aware checks. The parameter is detected from the signature, so prompt-only guards need not declare it:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness import InputGuard


def tenant_policy(ctx: RunContext[MyDeps], prompt: str) -> bool:
    return ctx.deps.tier == 'pro' or 'advanced-feature' not in prompt


InputGuard(guard=tenant_policy)
```

## Hard-fail path

`block` is the graceful path. To make the caller see an exception instead, raise from the guard:

```python
from pydantic_ai_harness import InputBlocked


def strict_guard(prompt: str) -> bool:
    if contains_credentials(prompt):
        raise InputBlocked('credentials detected')
    return True
```

Any exception raised by the guard propagates as-is -- use `InputBlocked` / `OutputBlocked` from this module, or your own exception types.

## API

```python {test="skip"}
@dataclass
class GuardResult:
    action: Literal['allow', 'block', 'replace', 'retry']
    message: str | None = None
    replacement: object | None = None
    # classmethods: allow(), block(message=None), replace(value), retry(message)


InputGuard(
    guard: Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]],
    parallel: bool = False,
)

OutputGuard(
    guard: Callable[..., bool | GuardResult | Awaitable[bool | GuardResult]],
)
```

The guard callable takes the inspected value -- the prompt for `InputGuard`, the output for `OutputGuard` -- optionally preceded by a `RunContext`.

## Relationship to `pydantic-ai-shields`

`pydantic-ai-shields` provides opinionated implementations on top of these primitives (prompt-injection detectors, PII scrubbers, keyword blocklists, etc.). Use the guardrails here when you want to plug in your own validation logic; reach for shields when you need a batteries-included detector.
