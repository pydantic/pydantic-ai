# Batch Processing for Pydantic AI

**Issue**: pydantic/pydantic-ai#1771

## The Problem

LLM APIs are expensive. OpenAI, Anthropic, and Google all offer batch APIs that process many requests as a single asynchronous job at roughly **50% cost**. The trade-off is latency: batches complete within a 24-hour window instead of returning immediately.

This matters to real users today. One user (bjorkbjork, via [Samuel on Slack](https://github.com/pydantic/pydantic-ai/issues/1771#issuecomment-3031440960)) is building RCA agents that ingest massive data from ClickHouse via MCP. They're getting rate-limited by Anthropic, who pointed them to batch processing. Another user (AlapinEnjoyer) runs agentic data generation and curation pipelines where cost and throughput matter far more than latency -- they want the same framework for both real-time and offline batch jobs, rather than maintaining separate codepaths.

Pydantic AI currently has no answer for these users. This plan adds one.

---

## What Users See

This is the primary deliverable -- if the user experience is wrong, nothing else matters.

### `model_request_batch()` — the direct API

Most users just want to submit a bunch of requests and get results:

```python
from pydantic_ai.direct import model_request_batch
from pydantic_ai.models import BatchRequest

results = await model_request_batch(
    'openai:gpt-5',  # use current frontier model at implementation time
    [
        BatchRequest('q1', [ModelRequest.user_text_prompt('What is 2+2?')]),
        BatchRequest('q2', [ModelRequest.user_text_prompt('What is 3+3?')]),
    ],
    poll_interval=60.0,
    timeout=3600.0,
)

for result in results:
    if result.is_successful:
        print(f'{result.custom_id}: {result.response.parts[0].content}')
```

One function handles the entire lifecycle: submit the batch, poll for completion, fetch results. Cancellation works via `asyncio.create_task(...).cancel()`. Timeout triggers automatic cancellation and raises `asyncio.TimeoutError`.

Each request is a `BatchRequest` with a `custom_id`, `messages`, and optional per-request `model_settings` (temperature, max_tokens, etc.). A dataclass makes the call site self-documenting and extensible without breaking changes -- tuples would be fragile and hard to read.

### `BatchModel` — the agent integration

`BatchModel` bridges batch processing and the existing agent system, wrapping a batch-capable model and intercepting `request()` calls -- queuing them instead of firing immediately:

```python
async with BatchModel('openai:gpt-5', batch_size=100) as batch_model:
    agent = Agent(model=batch_model)
    tasks = [agent.run(prompt) for prompt in prompts]
    await batch_model.submit()
    results = await asyncio.gather(*tasks)
```

**Important limitation: tool-using agents are incompatible with `BatchModel`.** The agent loop calls `request()`, gets tool calls, executes tools, then calls `request()` again. With `BatchModel`, the first `request()` blocks on a future that won't resolve until the batch completes -- the agent can't make progress. This is inherent to batch processing. `BatchModel` works for single-turn, no-tool workloads (data classification, content generation, translation, etc.).

Including `BatchModel` in this PR is deliberate -- per [Douwe's review of PR #3937](https://github.com/pydantic/pydantic-ai/pull/3937), it validates that the primitives are right. Shipping primitives without `BatchModel` risks discovering that abstractions need reshaping after they've already been merged.

---

## The Design Challenge

The three providers' batch APIs share the same concept but differ in almost every detail:

| Aspect | OpenAI | Anthropic | Google |
|--------|--------|-----------|--------|
| Submission | Upload a JSONL file | POST a JSON array | POST an inline array |
| Result ID matching | `custom_id` per result | `custom_id` per result | **Positional** -- responses in request order, no IDs |
| Cancel return | Updated batch object | Updated batch object | **`None`** -- SDK returns nothing |
| Result retrieval | Download separate output/error files | Stream from an endpoint | Results embedded on batch object |
| Status model | String statuses (`validating`, `in_progress`, ...) | 3 states + infer final from counts | `JobState` enum (`JOB_STATE_SUCCEEDED`, ...) |

Google is the most divergent. It has no `custom_id` concept at all -- we must track request order ourselves and match responses by position. Its cancel returns `None` instead of an updated batch. Its results aren't fetched separately but come back embedded in the batch object. If our abstraction works for Google, it works for anything.

---

## Architecture

This section describes what we're building. The rationale for each choice follows in [Design Decisions](#design-decisions).

### Core Types (`models/__init__.py`)

The shared vocabulary all other layers depend on.

**`BatchStatus`** -- a `str` enum with 5 states:

```
PENDING → IN_PROGRESS → COMPLETED
                       → FAILED
                       → CANCELLED
```

Terminal states: `COMPLETED`, `FAILED`, `CANCELLED`. Provider-specific sub-states (OpenAI's `validating`, `finalizing`; Anthropic's `canceling`) are preserved on provider `Batch` subclasses (e.g., `OpenAIBatch.raw_status`) for advanced users. `EXPIRED` maps to `FAILED`. `CANCELLING` maps to `IN_PROGRESS`.

**Type definitions** (following rule:124 -- `_: KW_ONLY` before optional fields):

```python
@dataclass
class BatchError:
    code: str       # e.g. 'rate_limit', 'validation_error'
    message: str

@dataclass
class Batch:
    id: str
    status: BatchStatus
    created_at: datetime
    _: KW_ONLY
    completed_at: datetime | None = None
    request_count: int = 0
    completed_count: int = 0
    failed_count: int = 0

    @property
    def is_complete(self) -> bool: ...  # status in terminal states

    @property
    def is_successful(self) -> bool: ...  # status == COMPLETED

@dataclass
class BatchResult:
    custom_id: str
    _: KW_ONLY
    response: ModelResponse | None = None
    error: BatchError | None = None

    @property
    def is_successful(self) -> bool: ...  # response is not None and error is None

@dataclass
class BatchRequest:
    custom_id: str
    messages: Sequence[ModelMessage]
    _: KW_ONLY
    model_settings: ModelSettings | None = None
```

`Batch` is a plain dataclass -- it carries data (`id`, `status`, timestamps, counts) but no reference to its `Model`. No `batch.wait()` or `batch.cancel()` methods. This is a deliberate choice explained in [Design Decisions](#batch-is-a-plain-dataclass-not-a-stateful-handle).

Each provider subclasses `Batch` with provider-specific fields (e.g., `OpenAIBatch` adds `input_file_id`, `output_file_id`; `GoogleBatch` adds `custom_ids: list[str]` to track positional matching).

**Batch method stubs on `Model`**: Four methods that raise `NotImplementedError`, matching the pattern of `count_tokens()` and `request_stream()`:

```python
async def batch_create(
    self,
    requests: Sequence[BatchRequest],
    model_settings: ModelSettings | None = None,
) -> Batch:
    raise NotImplementedError(f'Batch processing is not supported by {self.__class__.__name__}')

async def batch_status(self, batch: Batch) -> Batch:
    raise NotImplementedError(...)

async def batch_results(self, batch: Batch) -> list[BatchResult]:
    raise NotImplementedError(...)

async def batch_cancel(self, batch: Batch) -> Batch:
    raise NotImplementedError(...)
```

### `model_request_batch()` (`direct.py`)

```python
async def model_request_batch(
    model: Model | KnownModelName | str,
    requests: Sequence[BatchRequest],
    *,
    model_request_parameters: ModelRequestParameters | None = None,
    model_settings: ModelSettings | None = None,
    poll_interval: float = 60.0,
    timeout: float | None = None,
    instrument: InstrumentationSettings | bool | None = None,
) -> list[BatchResult]:
    """Submit batch, poll until complete, return results."""
```

The orchestration flow:

1. `_prepare_model(model, instrument)` -- resolves string to `Model`, wraps with `InstrumentedModel` if needed (same helper as `model_request()`)
2. `model.batch_create(requests, model_settings)` → `Batch` (catches `NotImplementedError`, wraps as `UserError`)
3. Poll loop: `await asyncio.sleep(poll_interval)` → `model.batch_status(batch)` → check `batch.is_complete`
4. On timeout: best-effort `model.batch_cancel(batch)`, raise `asyncio.TimeoutError`
5. On cancellation (`CancelledError`): best-effort cancel, re-raise
6. `model.batch_results(batch)` → return results

Also provides `model_request_batch_sync()` via `_get_event_loop().run_until_complete()` (same pattern as `model_request_sync`).

Settings merge order follows `merge_model_settings()`: model instance settings → batch-wide `model_settings` → per-request `model_settings`.

### `BatchModel` (`models/batch.py`)

```python
@dataclass(init=False)
class BatchModel(WrapperModel):
    batch_size: int | None = None
    should_submit: Callable | None = None
    poll_interval: float = 60.0
```

**Internal state**: `_queue: list[_QueuedRequest]` (each with `custom_id`, `messages`, settings, and an `asyncio.Future[ModelResponse]`), `_pending_batch: Batch | None`, `_batch_task: asyncio.Task | None`.

**`request()` flow**: Queues the request (creates a Future, generates a `custom_id`, appends to queue). If `batch_size` reached or `should_submit` returns True, auto-submits via `asyncio.create_task`. Returns `await future`. Always accepts new requests -- even if a batch is processing, new requests queue for the next batch.

**`submit()` flow**: Validates queue non-empty. Calls `self.wrapped.batch_create()`. On failure, propagates exception to all pending futures. On success, starts background `_wait_and_resolve` task.

**`_wait_and_resolve()` flow**: Polls until complete. Fetches results. Resolves each future by `custom_id` match: success → `set_result(response)`, error → `set_exception(...)`.

**Context manager**: `__aexit__` on clean exit auto-submits remaining queue and waits for pending batch. On exception exit, skips auto-submit but still waits for in-flight batches to avoid dangling tasks.

**Streaming**: `request_stream()` raises `NotImplementedError`. Inherent to batch processing.

### Shared Utilities (`models/_batch_utils.py`)

Cross-provider helpers for genuinely generic operations. Per `models/AGENTS.md`, provider-specific code stays in each provider's file.

- **`parse_batch_datetime(value)`** -- Handles Unix timestamps, ISO 8601 strings (with/without timezone), datetime objects, and None.
- **`validate_batch_complete(batch, operation)`** -- Guard that raises `ValueError` if you try to fetch results from a non-terminal batch.
- **`BatchResultBuilder`** -- Tracks processed `custom_id`s to prevent duplicates. Verify during implementation that 2+ providers need this (per rule:176 -- scope helpers to single usage site). If only OpenAI needs deduplication (output + error files), move it to `openai.py`.

Error normalization is **not** shared here. Each provider's error format is different enough that extracting a shared helper would be forced.

### Provider Implementations

Each provider's batch methods reuse the existing request-building machinery from `request()`. This is critical -- batch requests get the same tool handling, message mapping, and settings merging as real-time requests.

**Common pattern across all providers**: Every `batch_create` must:
1. Call `check_allow_model_requests()` (same guard as `request()`)
2. Call `self.prepare_request(model_settings, params)` per request for settings merging and parameter customization
3. Use the same request-building helpers as `request()` (e.g., `_build_request_params` for OpenAI, `_map_message` for Anthropic, `_build_content_and_config` for Google)
4. Map SDK errors to `ModelHTTPError` / `ModelAPIError`

#### OpenAI

OpenAI's batch API works through file uploads. You build a JSONL file where each line is a complete chat completion request, upload it, create a batch pointing to that file. Results come back as a separate downloadable JSONL file (with a third file for errors).

**`OpenAIBatch(Batch)`** adds: `input_file_id`, `output_file_id`, `error_file_id`, `raw_status`.

- `batch_create`: builds JSONL from the same `_build_request_params` code path as `request()`, uploads via `client.files.create()`, creates batch via `client.batches.create()`
- `batch_results`: downloads output/error files, parses JSONL lines, uses `ChatCompletion.model_validate()` → existing `_process_response`. Deduplicates between files. Best-effort file cleanup afterward.
- Status mapping: `validating` and `finalizing` → `IN_PROGRESS`, `expired` → `FAILED`

#### Anthropic

Anthropic's batch API is in beta (`client.beta.messages.batches`). You POST a JSON array of requests tagged with `custom_id`s. Results stream back.

**`AnthropicBatch(Batch)`** adds: `results_url`, `expires_at`, `cancel_initiated_at`, `processing_count`, `succeeded_count`.

- `batch_create`: reuses existing `_get_tools`, `_add_builtin_tools`, `_infer_tool_choice`, `_map_message` -- same code path as `request()`
- `batch_results`: streams via `client.beta.messages.batches.results(batch.id)`. Each entry has `result.type`: `succeeded`, `errored`, `canceled`, or `expired`
- Status inference: Anthropic has only 3 processing states (`in_progress`, `canceling`, `ended`). For `ended`, infer final status from request counts (all canceled → `CANCELLED`, all errored → `FAILED`, otherwise → `COMPLETED`)

#### Google

Google's batch API (via `google.genai` SDK) is the most divergent and the key test of whether our abstraction generalizes.

**`GoogleBatch(Batch)`** adds: `name` (full resource name), `display_name`, `state` (raw `JOB_STATE_*` string), `custom_ids: list[str]`.

Three challenges our abstraction must handle:

1. **No `custom_id`s**: Responses come back in request order without IDs. `GoogleBatch` stores `custom_ids: list[str]` to track the order; `batch_results` matches by index. Completely hidden from users.

2. **Cancel returns `None`**: `batch_cancel` returns an updated `Batch` with `status=IN_PROGRESS` (cancellation is in-flight). The next `batch_status` call gets the real state.

3. **Embedded results**: Results live on the batch object itself (in `dest.inlined_responses`), not in separate files or streams. `batch_results` re-fetches the batch and reads from there.

`batch_status` must preserve `custom_ids` from the original batch since Google doesn't store them.

### Wrapper Models

**`WrapperModel`** explicitly delegates every `Model` method. The four new batch stubs on `Model` will shadow `__getattr__` (Python resolves methods defined on the class before falling through to `__getattr__`), so `InstrumentedModel(OpenAIChatModel(...))` would hit the base stubs instead of reaching the wrapped model. Fix: add four explicit forwarding methods, consistent with how every other `Model` method is already forwarded. This automatically gives `ConcurrencyLimitedModel` and `InstrumentedModel` batch forwarding for free.

**`FallbackModel`** does not support batch. Batch + fallback semantics are unclear (do you fall back if `batch_create` fails? After 24 hours of polling?). The stubs on `Model` raise `NotImplementedError` as expected.

**Bedrock** uses the Anthropic API under the hood for Claude models. Whether Bedrock's endpoint supports the batch API needs verification. If it does, `WrapperModel` forwarding may handle it automatically. If not, Bedrock's stubs naturally raise `NotImplementedError`.

---

## Design Decisions

Now that the architecture is laid out, these are the non-obvious choices that shaped it and why.

### One high-level function, not four

**Decision**: Only `model_request_batch()` is exposed in `direct.py`. The four low-level methods (`batch_create`, `batch_status`, `batch_results`, `batch_cancel`) live only on `Model` instances.

**Why**: Douwe pushed back on exposing four functions as the primary API. The common case is "submit these requests and give me results." One function that handles create, poll, and results (with cancellation via `task.cancel()` and a `timeout` parameter) is the right default. Advanced users who need custom polling or multi-session workflows can call `model.batch_create(...)` directly.

### `Batch` is a plain dataclass, not a stateful handle

**Decision**: `Batch` carries data but no reference to its `Model`. No `batch.wait()` or `batch.cancel()` methods.

**Why**: A stateless `Batch` is serializable -- users can store it in a database and resume tomorrow with a fresh `Model` instance. No circular references, no hidden state. In the common case (`model_request_batch`), users never touch `Batch` directly.

**Trade-off**: Advanced users doing multi-session workflows must keep `Model` and `Batch` paired. This mirrors how provider SDKs work. `BatchModel` has its own stateful tracking on top of the stateless `Batch` -- this is intentional and keeps the concern separate from the data object. If friction surfaces during implementation, we adjust in this same PR.

### Per-request `model_settings` only

> **Deviation from maintainer feedback**: Douwe asked if we could "vary every aspect of the request (model name, messages, settings, model request param) for each batch job." This plan deliberately narrows per-request overrides to `model_settings` only -- varying tools or output schemas per request is unusual, adds significant complexity, and can be added later non-breakingly via `KW_ONLY`. Per-request model name variance (which Anthropic uniquely supports) is also deferred. **This narrowing needs explicit maintainer approval.**

**Decision**: Per-request overrides are limited to `model_settings` (temperature, max_tokens, etc.). The batch-wide `model_request_parameters` (tools, output schema, output mode) applies uniformly to all requests.

**Why**: `ModelRequestParameters` contains agent-level concerns -- function tools, output tools, output mode. These define *what kind of conversation* you're having. Varying them per request would mean each request could have different tool definitions, which adds complexity to settings merging and provider request building. What actually varies per request is `model_settings` -- different temperatures for creative vs. analytical prompts in the same batch.

### Detection via stubs, not `ModelProfile` or a protocol

**Decision**: The four batch method stubs on `Model` raise `NotImplementedError`, matching the established pattern for `count_tokens()` and `request_stream()`. No `supports_batch` flag on `ModelProfile`. No separate `BatchCapable` protocol.

**Why on `ModelProfile`**: The existing `supports_*` flags describe **intrinsic model capabilities** (tools, JSON output, image output). Batch processing is a **provider API feature** -- the same Claude Sonnet supports batch via Anthropic but not necessarily via Bedrock. Putting `supports_batch` on `ModelProfile` conflates these layers.

**Why not a `BatchCapable` protocol**: Since the stubs exist on `Model`, *every* `Model` instance would satisfy a `@runtime_checkable` `BatchCapable` protocol -- `isinstance(model, BatchCapable)` would always return `True`, making the check useless. The stubs-only approach is the established pattern: `count_tokens()` and `request_stream()` use the same "stub raises `NotImplementedError`" design without a separate protocol for detection.

`model_request_batch()` wraps the `NotImplementedError` from `batch_create()` into a `UserError` with a clear message, so users get a readable error rather than a raw traceback.

### `BatchModel` extends `WrapperModel`

**Decision**: `BatchModel` extends `WrapperModel`, not `Model` directly.

**Why**: `WrapperModel` is designed to be overridden -- `InstrumentedModel` already extends it and completely wraps `request()` with instrumentation spans. Same pattern. `BatchModel` overrides `request()` (to queue) and `request_stream()` (to raise `NotImplementedError`), delegating everything else (`model_name`, `profile`, `settings`, `system`, `prepare_request`) to the wrapped model for free.

### No framework-enforced minimum batch size

**Decision**: Let provider APIs reject invalid batches rather than enforcing `>= 2` at the framework level.

**Why**: Providers may have different minimums (or none). Adding our own constraint would be an opinion we'd have to maintain.

---

## Scope

This is a single PR containing everything needed to validate the abstraction end-to-end:

- Core types: `Batch`, `BatchStatus`, `BatchResult`, `BatchError`, `BatchRequest`
- Four batch method stubs on `Model`
- `WrapperModel` batch forwarding
- `BatchModel` wrapper (`models/batch.py`)
- Provider implementations: OpenAI + Anthropic + Google
- Shared utilities: `_batch_utils.py`
- `model_request_batch()` + `model_request_batch_sync()` in `direct.py`
- Tests (VCR cassettes for providers, unit tests for orchestration and `BatchModel`)
- Docs: `docs/batch-processing.md` + cross-references

Exports follow rule:29 (commonly-used types from top-level package):
- **`pydantic_ai/__init__.py`**: `Batch`, `BatchError`, `BatchModel`, `BatchRequest`, `BatchResult`, `BatchStatus`
- **`pydantic_ai/direct.py`**: `model_request_batch`, `model_request_batch_sync`
- **`pydantic_ai/models/__init__.py`**: `Batch`, `BatchError`, `BatchRequest`, `BatchResult`, `BatchStatus`

### Not in scope

- `agent.run_batch()` convenience method
- Per-request model variance (Anthropic allows mixing Sonnet and Haiku in one batch)
- Cost tracking, progress callbacks, state persistence/resumption
- Batch-specific OTel instrumentation (batch methods forward through `InstrumentedModel` without additional spans)

---

## Open Questions

These need maintainer input before or during implementation:

1. **Should `batch_cancel` be exposed in `direct.py`?**
   - *For*: users who submit a batch at 2pm and want to cancel at 5pm need `model_request_batch_cancel(model, batch)` without holding an `asyncio.Task`.
   - *Against*: `model.batch_cancel(batch)` already serves this need; a `direct.py` wrapper adds minimal value.
   - **Proposal**: don't expose it. Advanced users call `model.batch_cancel(batch)` directly.

2. **Per-request `model_request_parameters`?**
   - Douwe asked to "vary every aspect of the request" per batch job. This plan restricts per-request overrides to `model_settings` only, excluding `model_request_parameters` (tools, output schema) and per-request model name variance.
   - *Reasoning*: varying tools/output schemas per request is unusual, adds complexity to settings merging and provider request building, and can be added later non-breakingly via `KW_ONLY`.
   - **This is a deliberate deviation from maintainer feedback and needs explicit approval.** If Douwe wants per-request `model_request_parameters`, the `BatchRequest` dataclass and `batch_create` implementations need to be extended accordingly.

3. **Bedrock batch support?**
   - Does Bedrock's Anthropic endpoint support the batch API? Does `WrapperModel` forwarding handle it automatically?
   - **Proposal**: investigate but don't block the PR.

4. **Google scope?**
   - Google is the most divergent provider and the best test of whether the abstraction generalizes. But it also adds significant scope.
   - **Proposal**: include Google. If the abstraction works for Google, we have high confidence it generalizes. If it's taking too long, Google can slip to a fast-follow PR.

---

## Testing & Documentation

### Test strategy

Per `tests/AGENTS.md`, provider integration tests use `pytest-recording`/`vcrpy` cassettes recorded against real APIs. Batch tests follow the same pattern.

**Recording cassettes**: Batch APIs can take up to 24h, but VCR records the full HTTP conversation at recording time. Record with small batches (2-3 requests) and short `poll_interval` values. Cassettes capture the create → poll → complete → results sequence and replay instantly in CI. **Practical note**: even small batches typically take 10-30 minutes to complete, making cassettes expensive to re-record. The orchestration tests in `test_direct.py` should mock `asyncio.sleep` to avoid replaying the wait, while provider tests remain pure VCR to validate real API behavior.

**Assertion style**: `snapshot()` for structured outputs (`BatchResult` lists, `Batch` objects). `IsStr` for variable values (IDs, timestamps).

**Provider tests** (VCR cassettes) -- added to existing `tests/models/test_{provider}.py` files:
- `batch_create` builds correct provider-specific format, handles settings merging
- `batch_status` retrieves and parses updated status
- `batch_results` parses successes and errors
- `batch_cancel` calls the right API
- Status mapping covers every provider status → `BatchStatus` translation
- SDK errors map to `ModelHTTPError`/`ModelAPIError`

Provider-specific edge cases:
- *OpenAI*: JSONL parsing, output + error file deduplication, file cleanup
- *Anthropic*: streaming result iteration, `ended` status inference from counts
- *Google*: positional matching by index, `custom_ids` preservation, cancel returning `None`

**Framework tests** (unit tests / lightweight mocks):
- `test_batch_utils.py`: datetime parsing, batch completion validation, dedup logic
- `model_request_batch` integration: lifecycle orchestration with mocked model methods. Covers timeout, cancellation, sync variant, per-request settings, multiple status transitions.
- `test_batch_model.py`: queue mechanics, auto-submit at threshold, manual submit, context manager behavior (including exception-exit skipping auto-submit), error propagation to futures, `should_submit` callback, streaming raises `NotImplementedError`. Uses a mock `Model` with batch methods since we're testing wrapper logic, not providers.

### Documentation

Per `docs/AGENTS.md`: progressive disclosure, show recommended approach first, use admonitions for callouts, keep provider-specific details in provider docs.

**New page**: `docs/batch-processing.md`, added to `mkdocs.yml`:

1. **When to Use Batch** -- cost savings (~50%), rate limit relief, offline pipelines
2. **Quick Start** -- `model_request_batch()` with `BatchRequest`
3. **Cancellation and Timeouts** -- `timeout` param, `task.cancel()`, graceful cleanup
4. **Per-Request Settings** -- varying temperature/max_tokens per request
5. **Processing Results** -- `BatchResult`, `is_successful`, error handling
6. **Provider Support** -- table showing which providers support batch, with notes
7. **BatchModel for Agents** -- `BatchModel` + `asyncio.gather` + `agent.run()` pattern, with `!!! warning` that batch does NOT support multi-turn tool use
8. **Advanced: Low-Level Methods** -- `model.batch_create()` etc. for custom polling

**Existing page updates**:
- `docs/direct.md` -- cross-reference to batch processing page
- Provider docs (`docs/models/openai.md`, etc.) -- provider-specific batch notes
