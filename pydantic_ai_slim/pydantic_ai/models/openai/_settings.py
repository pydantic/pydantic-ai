"""The settings of both models, and how a request resolves them: service tier, unsupported parameters, tool forcing and echoed storage."""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import Literal

from openai import Omit, omit as OMIT
from openai.types import responses
from openai.types.chat.chat_completion_prediction_content_param import ChatCompletionPredictionContentParam
from openai.types.chat.completion_create_params import Moderation
from openai.types.responses import ComputerToolParam, FileSearchToolParam, WebSearchToolParam
from openai.types.responses.response_create_params import ContextManagement
from openai.types.shared import ReasoningEffort
from typing_extensions import TypedDict

from ...profiles.openai import OpenAIModelProfile
from ...settings import ModelSettings
from .._tool_choice import support_tool_forcing, tool_forcing_unavailable_reason


def echoed_store(response: responses.Response, model_settings: OpenAIResponsesModelSettings) -> bool | None:
    # Socket responses echo storage, which determines whether their IDs can be used outside that socket.
    if model_settings.get('openai_transport') == 'websocket' and isinstance(
        store := (response.model_extra or {}).get('store'), bool
    ):
        return store
    return None


def with_profile_unsupported_settings(unsupported: frozenset[str], profile: OpenAIModelProfile) -> frozenset[str]:
    """Add what the profile says a model or OpenAI-compatible provider rejects, so a dropped guarantee still warns."""
    return unsupported | frozenset(profile.get('openai_unsupported_model_settings', ()))


def drop_unsupported_params(profile: OpenAIModelProfile, model_settings: OpenAIChatModelSettings) -> None:
    """Drop unsupported parameters based on model profile.

    Mutates `model_settings`.

    Used currently only by Cerebras
    """
    for setting in profile.get('openai_unsupported_model_settings', ()):
        model_settings.pop(setting, None)


class OpenAIPromptCacheOptions(TypedDict, total=False):
    """Options for OpenAI prompt caching on GPT-5.6 models."""

    mode: Literal['implicit', 'explicit']
    """Whether OpenAI may create an implicit cache breakpoint. Defaults to `implicit`."""

    ttl: Literal['30m', '12h']
    """The minimum lifetime for cache breakpoints. Defaults to `30m`.

    `12h` is accepted only by the Responses API; Chat Completions rejects it with a 400.
    """


class OpenAIChatModelSettings(ModelSettings, total=False):
    """Settings used for an OpenAI model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    openai_reasoning_effort: ReasoningEffort
    """Constrains effort on reasoning for [reasoning models](https://platform.openai.com/docs/guides/reasoning).

    Currently supported values are `low`, `medium`, and `high`. Reducing reasoning effort can
    result in faster responses and fewer tokens used on reasoning in a response.
    """

    openai_logprobs: bool
    """Include log probabilities in the response.

    For Chat models, these will be included in `ModelResponse.provider_details['logprobs']`.
    For Responses models, these will be included in the response output parts `TextPart.provider_details['logprobs']`.
    """

    openai_top_logprobs: int
    """Include log probabilities of the top n tokens in the response."""

    openai_store: bool | None
    """Whether or not to store the output of this request in OpenAI's systems.

    If `False`, OpenAI will not store the request for its own internal review or training.
    See [OpenAI API reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-store).

    When used with `OpenAIResponsesModel`, stored responses appear in OpenAI's dashboard and
    can be referenced via [`openai_previous_response_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_previous_response_id].
    Pair this with `openai_previous_response_id='auto'` to avoid storing duplicate copies of
    the conversation history across retries and subsequent requests within the same run.

    When set to `False` on `OpenAIResponsesModel`, image generation calls in the message history are
    not sent back to the model outside a live Responses WebSocket continuation, as the API can only
    look them up in a stored response or the connection-local cache. The model then doesn't see that
    it generated those images.
    """

    openai_user: str
    """A unique identifier representing the end-user, which can help OpenAI monitor and detect abuse.

    See [OpenAI's safety best practices](https://platform.openai.com/docs/guides/safety-best-practices#end-user-ids) for more details.
    """

    openai_moderation: Moderation
    """Run moderation on the input and output of the request, e.g. `{'model': 'omni-moderation-latest'}`.

    Supported by both the Chat Completions API and the Responses API. In both cases, the moderation
    results returned by the API are exposed in
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details]
    under the `'moderation'` key.

    See the [OpenAI moderation documentation](https://platform.openai.com/docs/guides/moderation)
    for more details.
    """

    openai_text_verbosity: Literal['low', 'medium', 'high']
    """Constrains the verbosity of the model's text response.

    Lower values will result in more concise responses, while higher values will
    result in more verbose responses.

    Supported by both the Chat Completions API (as `verbosity`) and the Responses API
    (as `text.verbosity`). Parameter support can differ depending on the model.
    """

    openai_service_tier: responses.ServiceTier
    """The service tier to use for the model request.

    `fast` is OpenAI's newer name for `priority`. OpenAI only accepts `ultrafast` on the Responses API.
    For more information, see [OpenAI's service tiers documentation](https://developers.openai.com/api/reference/resources/responses/methods/create).
    """

    openai_prediction: ChatCompletionPredictionContentParam
    """Enables [predictive outputs](https://platform.openai.com/docs/guides/predicted-outputs).

    This feature is currently only supported for some OpenAI models.
    """

    openai_prompt_cache_key: str
    """Used by OpenAI to cache responses for similar requests to optimize your cache hit rates.

    See the [OpenAI Prompt Caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching#how-caching-works) for more information.
    """

    openai_prompt_cache_retention: Literal['in_memory', '24h']
    """The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours.

    For GPT-5.6 and later models, OpenAI deprecates this field in favor of the `ttl` in
    `openai_prompt_cache_options`; earlier models keep using this field. The two are independent and do not
    interact: this field expresses a maximum retention policy, while `ttl` expresses a minimum cache lifetime.

    See the [OpenAI Prompt Caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching#how-caching-works) for more information.
    """

    openai_prompt_cache_options: OpenAIPromptCacheOptions
    """Controls implicit and explicit prompt cache breakpoints, supported by GPT-5.6 and later models.

    Explicit breakpoints are added to user content with [`CachePoint`][pydantic_ai.messages.CachePoint].
    OpenAI applies the request-wide `ttl` to every breakpoint and ignores `CachePoint.ttl`.
    The `ttl` here is independent of the `openai_prompt_cache_retention` setting, which OpenAI deprecates
    for GPT-5.6 and later models.

    See the [OpenAI prompt caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching)
    for more information.
    """

    openai_cache_instructions: bool
    """Whether to add a prompt cache breakpoint after the last static instruction.

    With no static instructions, the breakpoint goes on the last system prompt instead.

    Supported by GPT-5.6 and later models; other models ignore it. OpenAI applies the request-wide
    `ttl` from `openai_prompt_cache_options`. OpenAI writes at most four breakpoints per request and
    drops the earliest first, so if `CachePoint` markers push a request over that limit, the
    instruction breakpoint is the first one dropped.

    On the Responses API the top-level `instructions` field cannot carry a breakpoint, so the
    instructions are sent as leading input messages instead. That only happens on requests that
    don't continue server-side state: when `openai_previous_response_id` or `openai_conversation_id`
    is set, or the history has been compacted, this setting leaves the instructions where they would
    otherwise go (normally the top-level field) and adds no breakpoint. That includes
    `openai_previous_response_id='auto'` on the first request of a chain: a stored response keeps
    its input, so relocated instructions would be replayed alongside every later request's own.

    No breakpoint is added when a dynamic system prompt precedes the instructions either, since its
    per-request content would sit inside the cached prefix and miss the cache on every run, nor when
    the system prompt role is `'user'` or the model merges leading system messages.

    See the [OpenAI prompt caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching)
    for more information.
    """

    openai_continuous_usage_stats: bool
    """When True, enables continuous usage statistics in streaming responses.

    When enabled, the API returns cumulative usage data with each chunk rather than only at the end.
    This setting correctly handles the cumulative nature of these stats by using only the final
    usage values rather than summing all intermediate values.

    See [OpenAI's streaming documentation](https://platform.openai.com/docs/api-reference/chat/create#stream_options) for more information.
    """

    openai_include_raw_annotations: bool
    """Whether to include raw annotations in `TextPart.provider_details`.

    Annotations are normalized into `TextPart.citations` unless the text was changed after it arrived, for example by
    removing thinking tags. Enable this when the original annotation payload is also needed.
    """

    openai_include_unmapped_response_data: bool
    """Whether to keep Chat Completions response data Pydantic AI doesn't map, in `ModelResponse.provider_details['unmapped']`.

    OpenAI-compatible servers often return vendor fields the Chat Completions API doesn't define, or `content` as a list
    of parts where only the text parts are mapped (the others, like references, are otherwise dropped). With this on,
    that data is kept exactly as received, keyed by where it came from: `response`, `choice`, `message` and
    `content_parts`. A streamed response collects it per chunk, under `chunks`, with `delta` in place of `message`.

    It's response data for reading only: it's stored with the message history but never sent back to a provider.
    Defaults to `False`, since most users don't read it and it grows stored histories.
    """


class OpenAIResponsesModelSettings(OpenAIChatModelSettings, total=False):
    """Settings used for an OpenAI Responses model request.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_native_tools: Sequence[FileSearchToolParam | WebSearchToolParam | ComputerToolParam]
    """The provided OpenAI built-in tools to use.

    See [OpenAI's built-in tools](https://platform.openai.com/docs/guides/tools?api-mode=responses) for more details.
    """

    openai_reasoning_mode: Literal['standard', 'pro']
    """The reasoning mode to use, for models that support it (currently the GPT-5.6 family).

    `standard` is the default. `pro` performs more model work to improve reliability on difficult
    tasks, at the cost of higher latency and token usage. Reasoning mode is independent of
    [`openai_reasoning_effort`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_reasoning_effort],
    and the unified [`thinking`][pydantic_ai.settings.ModelSettings.thinking] setting only influences
    the effort, never the mode. This setting is ignored when the resolved model profile does not
    support reasoning mode
    ([`OpenAIModelProfile.openai_responses_supports_reasoning_mode`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_responses_supports_reasoning_mode]).

    See [OpenAI's reasoning mode documentation](https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode)
    for more details.
    """

    openai_reasoning_context: Literal['auto', 'current_turn', 'all_turns']
    """The reasoning context to use, for models that support it.

    Controls which prior-turn reasoning items the model can use when sampling: `auto` defers to the
    model's own default (OpenAI treats it exactly like not sending the field), `current_turn` makes
    only the active turn's reasoning available, and `all_turns` renders compatible reasoning items
    from earlier turns into the next sample (requires access to earlier response items via
    `previous_response_id`, a conversation, or replayed history).

    When this setting is omitted, Pydantic AI sends `all_turns` on models that support it, so that
    earlier-turn reasoning stays available by default. Set `auto` explicitly to defer to OpenAI's
    own per-model default instead.

    `auto` and `current_turn` are sent to any model that supports reasoning. `all_turns` is sent
    only to models whose profile sets
    [`OpenAIModelProfile.openai_responses_supports_reasoning_context`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_responses_supports_reasoning_context]
    (currently the GPT-5.4, GPT-5.5, and GPT-5.6 families). A value the resolved profile doesn't
    support is ignored.

    See [OpenAI's reasoning context documentation](https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-across-calls)
    for more details.
    """

    openai_reasoning_summary: Literal['detailed', 'concise', 'auto']
    """A summary of the reasoning performed by the model.

    This can be useful for debugging and understanding the model's reasoning process.
    One of `concise`, `detailed`, or `auto`.

    Check the [OpenAI Reasoning documentation](https://platform.openai.com/docs/guides/reasoning?api-mode=responses#reasoning-summaries)
    for more details.
    """

    # TODO(v3): rename to `openai_send_item_ids`; this gates far more than reasoning IDs (see docstring).
    openai_send_reasoning_ids: bool
    """Whether to send the unique IDs of reasoning, text, and function call parts from the message history to the model. Enabled by default for reasoning models.

    This can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
    if the message history you're sending does not match exactly what was received from the Responses API in a previous response,
    for example if you're using a [history processor](../../message-history.md#processing-message-history).
    In that case, you'll want to disable this.

    Most server-side tool items (web search, code interpreter, image generation) are replayed *by* their ID,
    so disabling this also stops them from being sent back entirely. Hosted tool-search items are the exception:
    they carry their state (the query and discovered tools) inline, so they are still replayed with the IDs
    omitted, and previously discovered tools stay callable.
    """

    openai_truncation: Literal['disabled', 'auto']
    """The truncation strategy to use for the model response.

    It can be either:
    - `disabled` (default): If a model response will exceed the context window size for a model, the
        request will fail with a 400 error.
    - `auto`: If the context of this response and previous ones exceeds the model's context window size,
        the model will truncate the response to fit the context window by dropping input items in the
        middle of the conversation.
    """

    openai_previous_response_id: Literal['auto'] | str
    """Reference a prior OpenAI response to continue a conversation server-side, omitting cached messages from the input.

    - `'auto'`: chain to the most recent usable `provider_response_id` in the message history.
      If the history contains no such response, no `previous_response_id` is sent.
    - A concrete response ID string: use it as the seed while the message history holds no
      usable response to chain to (e.g. to continue from a prior turn). Once it does, as on
      subsequent in-run requests (retries, tool-call continuations), the most recent
      `provider_response_id` from the message history takes precedence so the chain extends
      correctly without re-sending messages that are already server-side.

    In both cases, messages that precede the chosen response in the history are omitted
    from the input, since OpenAI reconstructs them from server-side state. A response is only
    chosen while it and those messages are unchanged: each response created with this setting, or with
    `openai_transport='websocket'`, records a digest of its parts and the history it was created from as
    [`ModelResponse.history_digest`][pydantic_ai.messages.ModelResponse.history_digest], and if a
    history processor or compaction has changed either since, the full history is sent instead. The same goes
    for a response without a digest, as there's no telling whether it has changed.
    A history missing its earlier messages counts as changed too, so pass
    [`all_messages()`][pydantic_ai.agent.AgentRunResult.all_messages] rather than
    [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages], or the model won't see what
    was left out.

    Either way, the message history is trusted for the response ID, which
    [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] keeps as sent, so don't use this setting with
    history from a client. With `openai_transport='websocket'`, where it defaults to `'auto'`, pass a concrete
    `openai_conversation_id` instead to turn it off; see
    [Loading untrusted history](../../message-history.md#loading-untrusted-history).

    Requires the referenced response to have been stored (see
    [`openai_store`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_store],
    which defaults to `True` on OpenAI's side), except on the Responses WebSocket that created it:
    a request on that socket can continue from a response it hasn't stored, while it's still the
    latest on its lane. A response the socket reports as unstored isn't continued from anywhere else, and a
    socket request with `openai_store=False` continues only from its lane's response.
    Defaults to `'auto'` for requests using `openai_transport='websocket'`, including HTTP overflow
    (see [`openai_transport`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_transport])
    unless `openai_conversation_id` is set.

    See the [OpenAI Responses API documentation](https://platform.openai.com/docs/guides/reasoning#keeping-reasoning-items-in-context)
    for more information.
    """

    openai_conversation_id: Literal['auto'] | str
    """Reference an OpenAI conversation to continue durable conversation state server-side.

    - `'auto'`: use the most recent OpenAI conversation ID from `ModelResponse.provider_details['conversation_id']`
      in the message history with the same Pydantic AI `conversation_id`, when available. If the history
      contains no such response, no `conversation` is sent.
    - A concrete conversation ID string: use it as the OpenAI Responses API `conversation` parameter.

    When a matching conversation ID is found in message history, messages that precede that response
    are omitted from the input, since OpenAI reconstructs them from the server-side conversation. So
    changes to those messages, by a history processor or compaction for instance, don't reach the
    model: it continues from the conversation as OpenAI stored it.

    `'auto'` trusts the message history for this ID, which
    [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] keeps as sent. With history from a client, pass a
    conversation ID you saved on your server instead; see
    [Loading untrusted history](../../message-history.md#loading-untrusted-history).

    Not compatible with
    [`openai_previous_response_id`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_previous_response_id].

    See the [OpenAI conversation state documentation](https://platform.openai.com/docs/guides/conversation-state)
    for more information.
    """

    openai_include_code_execution_outputs: bool
    """Whether to include the code execution results in the response.

    Corresponds to the `code_interpreter_call.outputs` value of the `include` parameter in the Responses API.
    """

    openai_include_web_search_sources: bool
    """Whether to include the web search results in the response.

    Corresponds to the `web_search_call.action.sources` value of the `include` parameter in the Responses API.
    """

    openai_include_file_search_results: bool
    """Whether to include the file search results in the response.

    Corresponds to the `file_search_call.results` value of the `include` parameter in the Responses API.
    """

    openai_context_management: list[ContextManagement]
    """Context management configuration for the request.

    This enables OpenAI's server-side automatic compaction inside the regular
    `/responses` call, as opposed to the standalone `/responses/compact` endpoint.
    See [OpenAI's compaction guide](https://developers.openai.com/api/docs/guides/compaction)
    for details.

    The [`OpenAICompaction`][pydantic_ai.models.openai.OpenAICompaction] capability
    sets this automatically in its default (stateful) mode.
    """

    openai_background: bool
    """Enable background mode for long-running requests.

    When enabled, this setting passes `background=True` to the Responses API and opts into
    automatic polling for completion. If the response is still pending (`'queued'` or
    `'in_progress'`), the agent automatically polls for completion using `retrieve()`.
    """

    openai_transport: Literal['http', 'websocket']
    """How to send requests to the Responses API. Defaults to `'http'`.

    With `'websocket'`, the model sends each request on one
    [Responses WebSocket](https://developers.openai.com/api/docs/guides/websocket-mode) it keeps open while an agent run,
    `async with model:` or `async with agent:` is using it, so the requests of a tool-calling run go over one
    connection and send only their new input. Requests the socket can't take use HTTP; see
    [Responses WebSockets](../../models/openai.md#responses-websockets) for which, and for the
    `pydantic-ai-slim[openai-realtime]` install. Other models ignore this setting.

    When neither `openai_previous_response_id` nor `openai_conversation_id` is set, this mode defaults
    `openai_previous_response_id` to `'auto'`. That default, history digests and echoed storage metadata also apply
    to HTTP requests, including setups that cannot use a socket.
    """

    openai_prompt_cache_diagnostics: bool
    """Whether to request [prompt cache diagnostics](https://developers.openai.com/api/docs/guides/prompt-caching/diagnostics). Defaults to `True`.

    On models that support them (GPT-5.6 and later on the OpenAI API), each request whose message history holds an
    earlier OpenAI response passes that response's ID as `prompt_cache_options.comparison_response_id`, and OpenAI
    reports whether the new request could reuse its cached prefix, and if not, why. The result is available in
    [`ModelResponse.provider_details`][pydantic_ai.messages.ModelResponse.provider_details] under the
    `'prompt_cache_diagnostics'` key, as OpenAI's `prompt_cache_diagnostics` object: `{'type': 'cache_hit'}`,
    `{'type': 'cache_miss', 'reason': ..., 'cache_missed_tokens': ..., 'comparison_reusable_tokens': ...}`,
    `{'type': 'comparison_response_not_found'}` or `{'type': 'unavailable'}`.

    Diagnostics are free, don't affect caching or latency, and work with `openai_store=False`.
    Set this to `False` to leave `comparison_response_id` off the request. The ID comes from the message history,
    which [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] keeps as sent, so set this to `False` with
    history from a client; see [Loading untrusted history](../../message-history.md#loading-untrusted-history).
    """


def resolve_openai_service_tier(
    model_settings: OpenAIChatModelSettings,
) -> responses.ServiceTier | Omit:
    """Resolve the value to send as `service_tier` on the OpenAI request.

    Per-provider [`openai_service_tier`][pydantic_ai.models.openai.OpenAIChatModelSettings.openai_service_tier]
    wins; otherwise the top-level [`service_tier`][pydantic_ai.settings.ModelSettings.service_tier]
    maps 1:1 to OpenAI's accepted values.
    """
    if openai_tier := model_settings.get('openai_service_tier'):
        return openai_tier
    if unified := model_settings.get('service_tier'):
        return unified
    return OMIT


def support_openai_tool_forcing(
    model_name: str,
    openai_profile: OpenAIModelProfile,
    model_settings: OpenAIChatModelSettings | OpenAIResponsesModelSettings,
    *,
    thinking: bool,
) -> bool:
    """Check if the model supports forced tool use, raising UserError if explicitly requested but unsupported."""
    return support_tool_forcing(
        model_name,
        model_settings,
        tool_forcing_unavailable_reason(
            openai_profile,
            thinking=thinking,
            thinking_remedy="Disable thinking with `thinking=False` or `openai_reasoning_effort='none'`",
        ),
        disables_thinking=thinking and openai_profile.get('forced_tool_choice_disables_thinking', False),
    )
