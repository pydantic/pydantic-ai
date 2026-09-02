from __future__ import annotations as _annotations

from asyncio import Task
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import replace
from typing import Any

from .. import exceptions, messages as _messages, models, usage as _usage
from .._genai_prices import fill_response_cost
from .._instrumentation import capture_current_context
from .._run_context import set_current_run_context
from ..models import ModelRequestContext
from ..models._continuation import (
    MAX_BACKGROUND_POLLS,
    MAX_GENERATION_CONTINUATIONS,
    MergeMode,
    _ContinuationStreamedResponse,
    cancel_suspended_job,
    merge_mode,
    merge_responses,
)
from ..tools import RunContext
from .state import _agent_graph_sleep

__all__ = (
    'MAX_BACKGROUND_POLLS',
    'MAX_GENERATION_CONTINUATIONS',
    '_cancel_task',
    '_check_continuation_usage',
    '_resolve_interrupted_stream_state',
    'fill_response_cost',
    'model_request',
    'model_request_stream',
)


async def _cancel_task(task: Task[Any]) -> None:
    # `cancel()` is a documented no-op on an already-finished task, so there's no need to guard it.
    task.cancel()
    try:
        await task
    except BaseException:
        # Called while another stream error is already propagating; await only
        # to finish cleanup and retrieve the task exception, not replace it.
        pass


async def _resolve_interrupted_stream_state(
    model: models.Model,
    stream_error: BaseException,
    partial: _messages.ModelResponse,
) -> _messages.ModelResponseState:
    """State to record for a streamed turn the consumer stopped, cancelling a leaked job when appropriate.

    The composite treats every `aclose()` (which the handler teardown triggers) as a *detach*, so
    `partial.state` is `'suspended'` whenever the last segment is a still-pending job — regardless of
    *why* the consumer stopped. Only the graph knows why, from `stream_error`'s type:

    - `GeneratorExit` is a walk-away detach (the consumer broke out of `run_stream`/`stream_text`). Mirror
      the non-streaming detach: keep `'suspended'` so the run is resumable, and leave the job alive.
    - any other exception is a genuine downstream failure. Mirror the non-streaming cancel-on-error policy:
      force `'interrupted'` (non-resumable) and best-effort cancel the still-live job so it doesn't leak.
    """
    if isinstance(stream_error, GeneratorExit) and partial.state == 'suspended':
        return 'suspended'
    if partial.state == 'suspended':
        await cancel_suspended_job(model, partial)
    return 'interrupted'


def _split_resume_seed(
    messages: list[_messages.ModelMessage],
) -> tuple[list[_messages.ModelMessage], _messages.ModelResponse | None]:
    """Split a trailing suspended `ModelResponse` off `messages` as the continuation seed.

    A history ending in a `ModelResponse` with `state == 'suspended'` is the wire-truthful
    encoding of a paused turn to resume: the continuation loop echoes that response back to
    the provider itself, so it must not be part of the base history handed to the model.
    A normal request ends in a `ModelRequest`, so the seed is `None` and the messages pass
    through untouched.
    """
    if messages and isinstance(last := messages[-1], _messages.ModelResponse) and last.state == 'suspended':
        return messages[:-1], last
    return messages, None


def _check_continuation_usage(run_context: RunContext[Any], continuation_usage: _usage.RequestUsage) -> None:
    """Enforce token limits mid-turn against a provisional total during continuations.

    Continuation segments accumulate usage but aren't committed to the run usage until the
    final merged response is appended exactly once by `ModelRequestNode._append_response`
    (so a continuation isn't double-counted, nor counted as a separate request step). To
    still fail fast when a segment blows the token budget, check the limit against a
    throwaway copy of the run usage plus the accumulated continuation usage. Works both in
    the agent graph (where `run_context.usage` is the live run usage) and inside a durable
    boundary (where it's the serialized snapshot the activity/step/task received — the final
    workflow-side check still applies when the merged response is committed).
    """
    if run_context.usage_limits:
        provisional = deepcopy(run_context.usage)
        provisional.incr(continuation_usage)
        run_context.usage_limits.check_tokens(provisional)
        if continuation_usage.cost is not None:
            # Continuation usage is provisional, so only warn after the run successfully finishes.
            run_context.usage_limits.check_cost(provisional, warn_if_cost_unavailable=False)


async def _check_resume_seed_usage(
    model: models.Model, run_context: RunContext[Any], seed: _messages.ModelResponse | None
) -> None:
    """Check a suspended history seed before sending the continuation that resumes it."""
    usage_limits = run_context.usage_limits
    if seed is None or usage_limits is None or usage_limits.cost_limit is None:
        return
    try:
        fill_response_cost(seed)
        _check_continuation_usage(run_context, seed.usage)
    except BaseException:
        await cancel_suspended_job(model, seed)
        raise


async def model_request(
    model: models.Model,
    *,
    request_context: ModelRequestContext,
    run_context: RunContext[Any],
    on_progress: Callable[[_messages.ModelResponse], None],
) -> _messages.ModelResponse:
    """Run the innermost non-streaming model request, resolving any continuation chain.

    Loops over any suspended → complete continuation segments (Anthropic `pause_turn`,
    OpenAI background mode), echoing each suspended response back and merging the segments
    into one response. Only the final merged response is returned, so `wrap_model_request`
    spans the whole chain and `after_model_request` sees just the final response.
    Continuations are not separate request steps, so usage is committed exactly once when
    the merged response is appended to history. When `request_context.messages` ends in a
    suspended `ModelResponse` (a resumed run), that response seeds the loop.

    Under the bundled durable-execution capabilities (Temporal/DBOS/Prefect) this loop runs
    in workflow code: the capability swaps `request_context.model` for a wrapper that
    dispatches each segment's `model.request(...)` through its own activity/step/task, so a
    failed segment retries alone and each suspended response is checkpointed between
    segments.

    Args:
        model: The model to call.
        request_context: The merged request context (messages, settings, parameters).
        run_context: The current run context, made available via `get_current_run_context`.
        on_progress: Callback invoked with the merged response after each segment, so the
            caller can preserve partial progress when a later segment's error is converted
            to a retry.

    Returns:
        The (merged) model response.
    """
    base_messages, seed = _split_resume_seed(request_context.messages)
    await _check_resume_seed_usage(model, run_context, seed)

    # Two independent ceilings distinguished by the generic `merge_mode` signal, mirroring the
    # streamed composite in `_continuation`: every *fresh-generation* re-suspension (accumulate
    # `pause_turn`, a model change, or a `FallbackModel` replace directive) keeps the small
    # `MAX_GENERATION_CONTINUATIONS` cap against an unbounded model, while only a *same-id* re-suspension
    # re-polling one background job (OpenAI background mode, same `provider_response_id`) gets the
    # far more generous `MAX_BACKGROUND_POLLS` backstop so a legitimately long job isn't killed.
    accumulate_count = 0
    replace_count = 0
    # Mode of the merge that produced the current suspended `response`. A chain is homogeneous in
    # practice, so the previous merge's mode reliably classifies the next re-issue; the first
    # re-issue (`last_mode is None`) counts as strict, harmless since both ceilings allow ≥1.
    last_mode: MergeMode | None = None
    response = seed
    with set_current_run_context(run_context):
        while True:
            if response is None:
                messages = base_messages
            elif response.state == 'suspended':
                job_id = response.provider_response_id
                if last_mode == 'replace-same-id':
                    replace_count += 1
                    over_limit = replace_count > MAX_BACKGROUND_POLLS
                    limit_message = (
                        f'Model response for job {job_id!r} remained suspended after polling the maximum '
                        f'of {MAX_BACKGROUND_POLLS} times'
                    )
                else:
                    accumulate_count += 1
                    over_limit = accumulate_count > MAX_GENERATION_CONTINUATIONS
                    limit_message = (
                        f'Model response {job_id!r} was suspended more than the maximum of '
                        f'{MAX_GENERATION_CONTINUATIONS} times'
                    )
                if over_limit:
                    # Giving up on a still-suspended job: cancel it before raising so it doesn't leak.
                    await cancel_suspended_job(model, response)
                    raise exceptions.UnexpectedModelBehavior(limit_message)
                if delay := model.continuation_delay(response):
                    try:
                        await _agent_graph_sleep(delay)
                    except BaseException:
                        # A `CancelledError` (or any error) raised while parked in the inter-poll
                        # sleep sits outside the request's cancel guard below, so cancel the job
                        # here too before propagating.
                        await cancel_suspended_job(model, response)
                        raise
                messages = [*base_messages, response]
            else:
                return response

            try:
                new_response = await model.request(
                    messages, request_context.model_settings, request_context.model_request_parameters
                )
            except BaseException:
                # The broad catch is deliberate: `BaseException` also covers `CancelledError`,
                # `KeyboardInterrupt`, and `SystemExit`, and we must cancel the server-side
                # suspended/background job before letting any of them propagate so it doesn't leak.
                if response is not None:
                    await cancel_suspended_job(model, response)
                raise

            new_response = _narrow_tool_call_parts(new_response, request_context.model_request_parameters)
            if response is None:
                response = new_response
                if response.state == 'suspended':
                    fill_response_cost(response)
                    try:
                        _check_continuation_usage(run_context, response.usage)
                    except BaseException:
                        await cancel_suspended_job(model, response)
                        raise
            else:
                # Continuation segments are separately billed requests. Price them before merging so tiered
                # pricing is applied per request rather than once to their combined token counts.
                fill_response_cost(response)
                fill_response_cost(new_response)
                # Classify this transition (replace vs accumulate) so the next re-issue is
                # counted against the right ceiling.
                last_mode = merge_mode(response, new_response)
                response = merge_responses(response, new_response)
                # Enforce token limits early against a provisional total so a runaway
                # continuation can't blow the budget; the total is committed once later.
                try:
                    _check_continuation_usage(run_context, response.usage)
                except BaseException:
                    # The limit tripped on a still-suspended merge: cancel the live
                    # server-side job before propagating so it doesn't leak (mirrors the
                    # request-failure guard above and the streamed composite's check).
                    if response.state == 'suspended':
                        await cancel_suspended_job(model, response)
                    raise
            on_progress(response)


@asynccontextmanager
async def model_request_stream(
    model: models.Model,
    *,
    request_context: ModelRequestContext,
    run_context: RunContext[Any],
) -> AsyncGenerator[models.StreamedResponse]:
    """Open the innermost streaming model request, stitching any continuation chain.

    Under the bundled durable-execution capabilities (Temporal/DBOS/Prefect) this runs in
    workflow code: the capability swaps `request_context.model` for a wrapper whose
    `request_stream` drains one segment inside its own activity/step/task and replays its
    buffered events, so the composite below stitches per-segment replays.

    The yielded stream is a composite that stitches the (possibly suspended → complete)
    segments into one continuous stream: it opens a `model.request_stream(...)` per segment
    as it's iterated, so the whole chain is presented as a single stream and the
    model-request hooks wrap it once. When `request_context.messages` ends in a suspended
    `ModelResponse` (a resumed run), that response seeds the loop. On exit, the helper tears
    down any in-flight segment's connection (`aclose()`), which deliberately does *not*
    cancel a still-pending server-side job — cancellation stays on the
    `AgentStream.cancel()` → `close_stream()` path.

    Args:
        model: The model to call.
        request_context: The merged request context.
        run_context: The current run context.

    Yields:
        A `StreamedResponse` to iterate inside the durable boundary.
    """
    base_messages, seed = _split_resume_seed(request_context.messages)
    await _check_resume_seed_usage(model, run_context, seed)
    with set_current_run_context(run_context):
        sr = _ContinuationStreamedResponse(
            model_request_parameters=request_context.model_request_parameters,
            model=model,
            model_settings=request_context.model_settings,
            base_messages=base_messages,
            run_context=run_context,
            max_generation_continuations=MAX_GENERATION_CONTINUATIONS,
            max_background_polls=MAX_BACKGROUND_POLLS,
            sleep_func=_agent_graph_sleep,
            check_usage=lambda continuation_usage: _check_continuation_usage(run_context, continuation_usage),
            finalize_response=fill_response_cost,
            initial_suspended_response=seed,
            # The composite opens each segment lazily in the consumer task, which doesn't share
            # this task's OTel context (where `wrap_model_request` opened the `chat` span). Capture
            # it here so re-attaching it around each segment keeps `get_current_span()`-driven span
            # updates (e.g. `FallbackModel` recording the resolved inner model) on the right span.
            segment_context=capture_current_context(),
        )
        try:
            yield sr
        finally:
            # Deterministically tear down an in-flight segment's connection once the
            # consumer has stopped (mirrors the pre-stitching `async with request_stream`
            # teardown; a no-op after a fully-drained stream). Server-side cancellation
            # stays on the `AgentStream.cancel()` → `close_stream()` path.
            await sr.aclose()


def _narrow_tool_call_parts(
    response: _messages.ModelResponse, model_request_parameters: models.ModelRequestParameters
) -> _messages.ModelResponse:
    """Promote each base `ToolCallPart` in the response to its typed subclass via `ToolDefinition.tool_kind`.

    Lives here rather than in each model adapter so adapter authors emit base
    `ToolCallPart`s freely and the framework owns the typed-identity translation. Streaming
    parts are typed up-front by `ModelResponsePartsManager` via the same lookup; this
    function handles the non-streaming `Model.request()` return path. Either path produces
    the same typed end state — `isinstance(part, ToolSearchCallPart)` is true from the
    moment the call is emitted by the model.
    """
    tool_kind_by_name: dict[str, _messages.ToolPartKind] = {
        td.name: td.tool_kind for td in model_request_parameters.function_tools if td.tool_kind
    }
    if not tool_kind_by_name:
        return response

    changed = False
    new_parts: list[_messages.ModelResponsePart] = []
    for part in response.parts:
        if (
            isinstance(part, _messages.ToolCallPart)
            and part.tool_kind is None
            and (tool_kind := tool_kind_by_name.get(part.tool_name)) is not None
        ):
            promoted = _messages.ToolCallPart.narrow_type(part, tool_kind=tool_kind)
            new_parts.append(promoted)
            changed = True
        else:
            new_parts.append(part)
    return replace(response, parts=new_parts) if changed else response
