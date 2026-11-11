"""Usage, Responses finish reasons and status, logprobs, compaction items and response-scoped tool call IDs."""

from __future__ import annotations as _annotations

from typing import Any, Literal

from openai.types import chat, responses
from openai.types.chat import ChatCompletionChunk, chat_completion_token_logprob
from openai.types.responses import ResponseCompactionItem
from openai.types.responses.response_status import ResponseStatus

from ... import usage
from ..._utils import is_str_dict as _is_str_dict
from ...messages import STANDING_PROMPT_PLANTED_KEY, CompactionPart, FinishReason, ModelResponseState

RESPONSES_FINISH_REASON_MAP: dict[
    Literal['max_output_tokens', 'max_messages', 'content_filter', 'steered'] | ResponseStatus, FinishReason
] = {
    'max_output_tokens': 'length',
    'content_filter': 'content_filter',
    'completed': 'stop',
    'cancelled': 'error',
    'failed': 'error',
}


def response_status_to_state(status: ResponseStatus | None, *, background: bool) -> ModelResponseState:
    """Map a Responses API status to a `ModelResponseState`.

    Only genuine background jobs (`response.background`) suspend on a pending status: they can be
    resumed server-side via `retrieve(..., starting_after=...)`. A foreground stream also emits
    `in_progress`/`queued` statuses while it runs, but it isn't resumable, so a pending status there
    means the response is merely `'incomplete'` (still streaming, or ended without a terminal event) —
    marking it `'suspended'` would send the continuation loop off to `retrieve` a non-background job.
    """
    if status in ('queued', 'in_progress'):
        return 'suspended' if background else 'incomplete'
    return 'complete'


# Convert logprobs to a serializable format
def map_logprobs(
    logprobs: list[chat_completion_token_logprob.ChatCompletionTokenLogprob]
    | list[responses.response_output_text.Logprob]
    | list[responses.response_text_done_event.Logprob],
) -> list[dict[str, Any]]:
    # The SDK's `output_text.done` logprob types don't declare the `bytes` the API sends, but keep it as an extra field.
    return [
        {
            'token': lp.token,
            'bytes': getattr(lp, 'bytes', None),
            'logprob': lp.logprob,
            'top_logprobs': [
                {
                    'token': tlp.token,
                    'bytes': getattr(tlp, 'bytes', None),
                    'logprob': tlp.logprob,
                }
                for tlp in (lp.top_logprobs or [])
            ],
        }
        for lp in logprobs
    ]


def map_compaction_item(
    item: ResponseCompactionItem, system: str, *, standing_prompt_planted: bool = False
) -> CompactionPart:
    """Convert an OpenAI `ResponseCompactionItem` to a `CompactionPart`.

    `standing_prompt_planted` stamps the part as minted by our own `responses.compact` call,
    whose input window explicitly plants the standing prompt — the provenance that lets the trim
    rely on the compaction item's retention instead of re-sending the standing prompt (see
    `_trim_messages_before_compaction`). Compaction items arriving in ordinary responses are left
    unstamped: their window may itself have relied on an earlier item's retention, and retention
    is only reliable for a single hop.
    """
    provider_details = item.model_dump()
    if standing_prompt_planted:
        provider_details[STANDING_PROMPT_PLANTED_KEY] = True
    return CompactionPart(
        content=None,
        id=item.id,
        provider_name=system,
        provider_details=provider_details,
    )


def billed_web_searches(
    response: chat.ChatCompletion | ChatCompletionChunk | responses.Response | responses.CompactedResponse,
) -> int:
    """Return the number of billed native web searches in a Responses API response.

    OpenAI bills native web search per search action (see
    <https://developers.openai.com/api/docs/guides/tools-web-search>) but doesn't report the count in `usage`.
    Newer responses carry it as `tool_usage.web_search.num_requests`, which the SDK doesn't type yet; older ones
    don't, so fall back to counting `search` actions. Reasoning models also emit `open_page` and `find_in_page`
    actions, which aren't billed as searches. A streamed `in_progress` or `queued` snapshot returns 0: its
    searches are counted again from the terminal event's response.
    """
    if not isinstance(response, responses.Response) or response.status in ('in_progress', 'queued'):
        return 0
    match (response.model_extra or {}).get('tool_usage'):
        case {'web_search': {'num_requests': int() as num_requests}}:
            return num_requests
        case _:
            return sum(
                1
                for item in response.output
                if isinstance(item, responses.ResponseFunctionWebSearch) and item.action.type == 'search'
            )


def map_usage(
    response: chat.ChatCompletion | ChatCompletionChunk | responses.Response | responses.CompactedResponse,
    provider: str,
    provider_url: str,
    model: str,
) -> usage.RequestUsage:
    response_usage = response.usage
    web_search_requests = billed_web_searches(response)
    if response_usage is None:
        if web_search_requests:
            return usage.RequestUsage(
                unmeasured_requests=1,
                web_searches=web_search_requests,
                details={'web_search_requests': web_search_requests},
            )
        return usage.RequestUsage(unmeasured_requests=1)

    usage_data = response_usage.model_dump(exclude_none=True)
    details = {
        k: v
        for k, v in usage_data.items()
        if k not in {'prompt_tokens', 'completion_tokens', 'input_tokens', 'output_tokens', 'total_tokens'}
        if isinstance(v, int)
    }
    response_data = dict(model=model, usage=usage_data)
    if isinstance(response_usage, responses.ResponseUsage):
        api_flavor = 'responses'
        input_tokens_details = usage_data.get('input_tokens_details')

        if getattr(response_usage, 'output_tokens_details', None) is not None:
            details['reasoning_tokens'] = getattr(response_usage.output_tokens_details, 'reasoning_tokens', 0)
        else:
            details['reasoning_tokens'] = 0

        if web_search_requests:
            details['web_search_requests'] = web_search_requests
    else:
        api_flavor = 'chat'
        input_tokens_details = usage_data.get('prompt_tokens_details')

        if response_usage.completion_tokens_details is not None:
            details.update(response_usage.completion_tokens_details.model_dump(exclude_none=True))

    request_usage = usage.RequestUsage.extract(
        response_data,
        provider=provider,
        provider_url=provider_url,
        provider_fallback='openai',
        api_flavor=api_flavor,
        details=details,
    )
    # genai-prices prices `web_searches` straight off the usage object (its OpenAI extractors have no mapping
    # for it since the count never appears in the wire `usage`), so lift the count found above.
    # It isn't a declared field, so it's set the way `RequestUsage.__init__` sets extra units.
    if web_search_requests:
        setattr(request_usage, 'web_searches', web_search_requests)
    # genai-prices maps OpenAI's nested `cache_write_tokens` on the `openai` extractors as of
    # https://github.com/pydantic/genai-prices/pull/463 (in 0.1.4), but not every OpenAI-compatible
    # provider's extractor does — Azure's still omits it — so lift it manually here.
    # TODO: Remove this block once those remaining extractors map it. Check `prices/providers/azure.yml`.
    if _is_str_dict(input_tokens_details):
        cache_write_tokens = input_tokens_details.get('cache_write_tokens')
        if isinstance(cache_write_tokens, int):
            request_usage.cache_write_tokens = cache_write_tokens
    return request_usage
