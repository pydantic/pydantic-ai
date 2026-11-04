"""Response metadata shared by `AnthropicModel._process_response` and `AnthropicStreamedResponse`."""

from __future__ import annotations as _annotations

from typing import Any, assert_never

import pydantic_core
from anthropic.types.beta import (
    BetaInputTransformation,
    BetaMessage,
    BetaMessageDeltaUsage,
    BetaRawMessageDeltaEvent,
    BetaRawMessageStartEvent,
    BetaStopReason,
    BetaUsage,
)
from opentelemetry.trace import get_current_span

from ... import usage
from ..._context_size import mark_context_input_tokens, marked_context_input_tokens
from ...messages import FinishReason, ModelMessage, ModelResponse
from .. import StreamedResponse

FINISH_REASON_MAP: dict[BetaStopReason, FinishReason | None] = {
    'compaction': 'stop',
    'end_turn': 'stop',
    'max_tokens': 'length',
    'model_context_window_exceeded': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_call',
    'pause_turn': None,
    'refusal': 'content_filter',
}


def report_input_transformations(
    transformations: list[BetaInputTransformation],
) -> list[dict[str, Any]]:
    """Report Anthropic's input transformations on the current span, and return them for `provider_details`.

    A dropped thinking block is otherwise silent: the model answered without the reasoning we
    replayed, and nothing in the response says so. `provider_details` keeps it inspectable after the
    run; the span event surfaces it in the trace, on the model request that lost the reasoning.
    """
    reported = [transformation.model_dump() for transformation in transformations]
    span = get_current_span()
    attributes = getattr(span, 'attributes', {})
    if span.is_recording() and attributes.get('gen_ai.operation.name') == 'chat':
        span.add_event(
            'anthropic.input_transformations',
            attributes={'anthropic.input_transformations': pydantic_core.to_json(reported).decode()},
        )
    return reported


_COMPACTION_TOKEN_KEYS = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')


def extract_usage_details(response_usage: BetaUsage | BetaMessageDeltaUsage) -> dict[str, int]:
    """Extract Anthropic usage into a flat dict, preserving the web search count and iteration totals.

    Anthropic's top-level `input_tokens`/`output_tokens` exclude both compaction and advisor iteration
    usage (see <https://docs.anthropic.com/en/docs/build-with-claude/compaction#understanding-usage>),
    so they're kept as-is here and the iteration totals are recorded under `compaction_*` / `advisor_*`
    keys. `map_usage` sums the compaction totals back into the request totals at extraction time (which
    also keeps streaming correct: the fixed totals set by the start event survive the merge with delta
    events that only carry the top-level values). Advisor totals are deliberately NOT summed back, since
    advisor tokens are billed at the advisor model's rates, not the executor's, and folding them into the
    request totals would misprice the request via genai-prices.
    """
    details: dict[str, int] = {}
    for key in _COMPACTION_TOKEN_KEYS:
        if isinstance((value := getattr(response_usage, key, None)), int):
            details[key] = value

    # One-hour cache writes are billed at a higher rate than five-minute ones (see
    # <https://platform.claude.com/docs/en/about-claude/pricing>). The rest of
    # `cache_creation_input_tokens` is priced at the five-minute rate, so the one-hour count is all pricing needs.
    # In streaming, only the start event carries the split, so it's kept in `details` to survive the merge.
    if (
        isinstance(response_usage, BetaUsage)
        and response_usage.cache_creation is not None
        and (ephemeral_1h_input_tokens := response_usage.cache_creation.ephemeral_1h_input_tokens)
    ):
        details['ephemeral_1h_input_tokens'] = ephemeral_1h_input_tokens

    # Anthropic bills thinking tokens inside `output_tokens`, so this is a readable subset of the
    # output total rather than an additive one, matching `reasoning_tokens` on OpenAI and
    # `thoughts_tokens` on Google.
    output_tokens_details = response_usage.output_tokens_details
    if output_tokens_details is not None and (thinking_tokens := output_tokens_details.thinking_tokens):
        details['thinking_tokens'] = thinking_tokens

    # Native web searches are billed per search (see
    # <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/web-search-tool#usage-and-pricing>)
    # and the count is only reported here, not as response parts, as searches can also run inside code execution.
    server_tool_use = response_usage.server_tool_use
    if server_tool_use is not None and (web_search_requests := server_tool_use.web_search_requests):
        details['web_search_requests'] = web_search_requests

    iterations = response_usage.iterations
    if not iterations:
        return details

    compaction_iterations = [it for it in iterations if it.type == 'compaction']
    advisor_iterations = [it for it in iterations if it.type == 'advisor_message']
    if not compaction_iterations and not advisor_iterations:
        return details

    # Both compaction and advisor iterations are separate from executor turns, so exclude them from
    # the `message_iterations` count.
    details['message_iterations'] = len(iterations) - len(compaction_iterations) - len(advisor_iterations)

    if compaction_iterations:
        details['compaction_iterations'] = len(compaction_iterations)
        for key in _COMPACTION_TOKEN_KEYS:
            if compaction_total := sum(getattr(it, key) for it in compaction_iterations):
                details[f'compaction_{key}'] = compaction_total
        if compaction_ephemeral_1h_input_tokens := sum(
            it.cache_creation.ephemeral_1h_input_tokens for it in compaction_iterations if it.cache_creation is not None
        ):
            details['compaction_ephemeral_1h_input_tokens'] = compaction_ephemeral_1h_input_tokens

    if advisor_iterations:
        details['advisor_iterations'] = len(advisor_iterations)
        # The advisor iteration token fields share the names in `_COMPACTION_TOKEN_KEYS`. These are
        # recorded for observability only; `map_usage` must not fold them into the request totals.
        for key in _COMPACTION_TOKEN_KEYS:
            if advisor_total := sum(getattr(it, key) for it in advisor_iterations):
                details[f'advisor_{key}'] = advisor_total

    return details


def last_pass_input_tokens(response_usage: BetaUsage | BetaMessageDeltaUsage) -> int | None:
    """The input tokens of a response's last sampling pass, cache included, or `None` when the passes aren't reported.

    A server-side tool loop samples the model once per tool round, and the top-level usage sums the passes: what they
    cost, but more than any pass's context. A beta that returns `iterations` (such as `compact-2026-01-12`) reports
    each pass, and the last `message` one (`fallback_message` when a fallback model served the turn) measures the
    context; a `compaction` entry measures the compaction, not the context. See `BetaUsage.iterations`.

    A single-pass response gets a value too, though its sum is already its context: each `pause_turn` segment needs a
    stamp of its own, as the continuation merges the segments' metadata shallowly.
    """
    for iteration in reversed(response_usage.iterations or []):
        if iteration.type in ('message', 'fallback_message'):
            return iteration.input_tokens + iteration.cache_read_input_tokens + iteration.cache_creation_input_tokens
    return None


def mark_continuation_context_input_tokens(
    messages: list[ModelMessage], response: ModelResponse | StreamedResponse
) -> None:
    """Give a `pause_turn` continuation segment that didn't report its passes the context size of its own usage.

    Otherwise the continuation's shallow metadata merge would keep an earlier segment's last pass as the context of
    the merged response. A continuation resends the same betas, so every segment is expected to report `iterations`
    if the first did; this guards against one that doesn't. A turn that never reported them keeps its summed usage as
    the context.
    """
    previous = messages[-1] if messages else None
    if (
        isinstance(previous, ModelResponse)
        and previous.state == 'suspended'
        and marked_context_input_tokens(previous) is not None
        and marked_context_input_tokens(response) is None
    ):
        mark_context_input_tokens(response, response.usage.input_tokens)


def map_usage(
    message: BetaMessage | BetaRawMessageStartEvent | BetaRawMessageDeltaEvent,
    provider: str,
    provider_url: str,
    model: str,
    existing_usage: usage.RequestUsage | None = None,
) -> usage.RequestUsage:
    if isinstance(message, BetaMessage):
        response_usage = message.usage
    elif isinstance(message, BetaRawMessageStartEvent):
        if message.message is None:  # pyright: ignore[reportUnnecessaryComparison]
            # On Bedrock the Anthropic SDK drops SSE event types, so Bedrock-only chunks
            # (e.g. `amazon-bedrock-invocationMetrics`) are non-validating `construct_type`d
            # into `BetaRawMessageStartEvent(message=None)`, violating the type annotation.
            # The metrics chunk's token counts duplicate the canonical `message_start` /
            # `message_delta` usage, so dropping them here avoids double-counting.
            return existing_usage or usage.RequestUsage()
        response_usage = message.message.usage
    elif isinstance(message, BetaRawMessageDeltaEvent):
        response_usage = message.usage
    else:
        assert_never(message)

    # In streaming, usage appears in different events.
    # The values are cumulative, meaning new values should replace existing ones entirely.
    details = (existing_usage.details if existing_usage else {}) | extract_usage_details(response_usage)
    # The one-hour count is capped at the total it's part of: after a compaction iteration's cache write, the final
    # event resets `cache_creation_input_tokens` without resending the split, so the start event's one-hour count
    # would otherwise survive, both in streamed usage and in a message accumulated from a stream. This assumes a
    # stale split can only overshoot the total.
    if 'ephemeral_1h_input_tokens' in details:
        if ephemeral_1h_input_tokens := min(
            details['ephemeral_1h_input_tokens'], details.get('cache_creation_input_tokens', 0)
        ):
            details['ephemeral_1h_input_tokens'] = ephemeral_1h_input_tokens
        else:
            del details['ephemeral_1h_input_tokens']

    # Anthropic reports top-level tokens excluding compaction iteration usage; add the
    # compaction totals back in so the extracted `RequestUsage` reflects the real request cost.
    usage_for_extraction: dict[str, Any] = dict(details)
    for key in _COMPACTION_TOKEN_KEYS:
        if compaction_value := details.get(f'compaction_{key}'):
            usage_for_extraction[key] = usage_for_extraction.get(key, 0) + compaction_value

    # genai-prices reads the web search count from Anthropic's nested wire shape and maps it to `web_searches`.
    if web_search_requests := details.get('web_search_requests'):
        usage_for_extraction['server_tool_use'] = {'web_search_requests': web_search_requests}
    # Likewise the one-hour cache write count, which it maps to `cache_write_1h_tokens`. Compaction iterations write
    # to the cache with the request's TTL, so their one-hour writes are summed back in like the totals above.
    if ephemeral_1h_input_tokens := details.get('ephemeral_1h_input_tokens', 0) + details.get(
        'compaction_ephemeral_1h_input_tokens', 0
    ):
        usage_for_extraction['cache_creation'] = {'ephemeral_1h_input_tokens': ephemeral_1h_input_tokens}

    # `extract_usage_details` keeps int fields only; carry the Gateway's cost objects through.
    for gateway_key in ('logfire_ai_gateway', 'pydantic_ai_gateway'):
        if gateway_usage := (response_usage.model_extra or {}).get(gateway_key):
            usage_for_extraction[gateway_key] = gateway_usage

    # Note: genai-prices already extracts cache_creation_input_tokens and cache_read_input_tokens
    # from the Anthropic response and maps them to cache_write_tokens and cache_read_tokens
    request_usage = usage.RequestUsage.extract(
        dict(model=model, usage=usage_for_extraction),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='anthropic',
        details=details,
    )
    # genai-prices doesn't map Anthropic's thinking token count, which is already priced inside `output_tokens`.
    if thinking_tokens := details.get('thinking_tokens'):
        request_usage.output_reasoning_tokens = thinking_tokens
    return request_usage


def compaction_provider_details(encrypted_content: str | None, signature: str | None = None) -> dict[str, Any] | None:
    """The `CompactionPart.provider_details` a compaction block is sent back with.

    A block from threshold compaction can carry `encrypted_content`; one from an on-demand compaction request carries
    a `signature`, which Anthropic checks against the summary when the block is sent back.
    """
    details = {
        key: value
        for key, value in (('encrypted_content', encrypted_content), ('signature', signature))
        if value is not None
    }
    return details or None
