"""Every function that reads replay data from message history is classified, and each replay site uses its gate.

A replay site reads server-side state or a stateless signed or encrypted artifact from history into an outgoing
request. `agent_docs/provider-adapters.md` "Replay data" names the gate for each: `Model.accepts_server_state` for
state an endpoint looks up by ID, `Model.accepts_stateless_artifacts` for signed or encrypted data. A site that
continues a stored response also checks that the history before it is unchanged. Each site picks its gate by hand,
so `REPLAY_SITES` records the pick, and these tests hold the record to the code:

- `test_every_reading_function_is_classified` walks the source of `pydantic_ai/models/**` and `pydantic_ai/_mcp.py`
  with `ast`. It lists every function, method and nested function that reads a field only Pydantic AI history
  carries (`provider_response_id`, `history_digest`, `provider_url`, `canonical_system`, `provider_details`,
  `signature`) or uses a replay predicate (`accepts_server_state`, `accepts_stateless_artifacts`). A new one fails
  until `REPLAY_SITES` classifies it, and an entry that no longer names one fails as stale. The walk imports nothing,
  so it also reads the modules whose optional dependency isn't installed, and it's always complete.
- `test_replay_site_uses_its_gate` checks each `ReplaySite` against its function's body: it uses the predicate its
  gate names, it reaches the continuation check it names, or it compares `provider_name` for a `NameOnly` gate. A
  `NameOnly` gate quotes the doc line that keeps it name-only, or records that none does. A `GatedBy` gate names a
  registered site that gates it, and the site, or a function that calls the site, calls it by name, directly or
  through other functions. That check catches a record left behind by a rename or a moved call. It can't show that
  the gate's answer guards the call.

To classify a function the first test reports:

1. If it puts history data in a request, add a `ReplaySite`. Make its gate the predicate it uses, a `Continuation`
   when it continues a stored response, `GatedBy` when another site decides for it, or `NameOnly` when it compares
   `provider_name` alone. Follow "Replay data" before you add a `NameOnly`: it allows only the checks it names.
2. Otherwise add an `Exempt` with what the function does instead. Response parsing, bookkeeping, telemetry, and
   resuming or cancelling the suspended response a request continues aren't replay.

What the walk leaves out:

- Part `.id`, `.tool_call_id`, and `args`. SDK objects carry an `id` too, and most `.id` reads in the adapters parse
  a response. `tool_call_id` is the key that pairs a function call with its result, which every adapter sends.
  Neither can be told apart from a replay read without types. So the walk doesn't see a site that reads only these.
  Examples are the OpenAI Responses item IDs in `OpenAIResponsesModel._map_stateful_native_tool_call` and
  `_map_mcp_server_tool_call`, which reuse `tool_call_id` as the item ID, the `args['container_id']` it sends, and
  Bedrock's `_map_native_tool_call`. Their gates are recorded on the sites that gate them (`_map_model_response`,
  `_map_native_tool_return`).
- Reads off `self`: a streamed response building itself, or a model's own `canonical_system`, isn't history.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

import pytest

import pydantic_ai

from ..manifests import Exempt, assert_classified

PACKAGE = Path(pydantic_ai.__file__).parent
"""The package whose `models/**` and `_mcp.py` the walk reads."""

REPLAY_DATA_DOCS = Path(__file__).parents[2] / 'agent_docs' / 'provider-adapters.md'
"""The doc a `NameOnly` gate quotes."""

HISTORY_FIELDS = frozenset(
    {'provider_response_id', 'history_digest', 'provider_url', 'canonical_system', 'provider_details', 'signature'}
)
"""The fields the walk looks for. All but `signature` exist only on Pydantic AI history, and few SDK objects have one."""

PREDICATES = {'server_state': 'accepts_server_state', 'stateless': 'accepts_stateless_artifacts'}
"""The replay predicate each string gate names."""

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Continuation:
    """A site that continues a stored response: it uses `accepts_server_state`, then the history digest `check`."""

    check: str
    """The function, by name, that compares the history digest: the site reaches it by its calls."""


@dataclass(frozen=True)
class NameOnly:
    """A site that compares `provider_name` alone."""

    cites: str | None
    """The line of `agent_docs/provider-adapters.md` that keeps it name-only, verbatim.

    `None` when no doc line does: the entry records the check, it doesn't approve it.
    """
    reason: str


@dataclass(frozen=True)
class GatedBy:
    """A function whose replay registered sites decide, by the predicate's answer or the predicate they pass it."""

    sites: tuple[str, ...]
    reason: str


Gate = Literal['server_state', 'stateless'] | Continuation | NameOnly | GatedBy


@dataclass(frozen=True)
class ReplaySite:
    """A function that puts server-side state or a stateless artifact from history in a request, and its gate."""

    gate: Gate
    replays: tuple[str, ...]
    """What it sends back from history, or what the functions it gates send."""
    uncovered: str = ''
    """A read in the function that its gate doesn't cover, and what decides it instead."""

    @property
    def reason(self) -> str:
        """What it replays, so `assert_classified` reports an entry that names nothing as one without a reason."""
        return '; '.join(self.replays)


_PARSES = Exempt('builds a response from the API reply: it writes these fields, and replays none')
_STREAMS_A_RESPONSE = Exempt('serves a finished response as a stream')
_CONTINUATION_BOOKKEEPING = Exempt('merges or counts the continuations of a suspended response; it sends nothing')
_BATCH = Exempt(
    'batch bookkeeping: submits, checks, resubmits or prices the batch a suspended response waits on, which, like '
    'resuming a suspended response, acts on that one batch'
)
_OWN_IDENTITY = Exempt("reads a model's or provider's own `canonical_system`, not history")
_DELEGATES = Exempt('forwards a replay predicate to the model that decides')
_CHECK = Exempt('a continuation or reachability check a registered site applies after its gate; it sends nothing')
_RECORDS_DIGEST = Exempt('builds or stamps the history digest a later continuation compares')
_SHAPES_REQUEST = Exempt(
    'reads a marker in history to choose a parameter of this request, such as a beta or a thinking setting; it '
    'sends no stored ID or artifact'
)
_CONTINUES = Exempt(
    'resumes or cancels the suspended response this request continues, which `agent_docs/provider-adapters.md` '
    '"Replay data" says is not replay'
)

_TRIM = 'pydantic_ai.models::Model._trim_before_compaction'
_ANTHROPIC_NATIVE_TOOLS = 'pydantic_ai.models.anthropic::AnthropicModel._replays_native_tool_part'
_GOOGLE_MESSAGES = 'pydantic_ai.models.google::GoogleModel._map_messages'
_RESPONSES_PROMPT_CACHE = (
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._prompt_cache_comparison_response_id'
)
_RESPONSES_ITEM_IDS = 'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_model_response'
_RESPONSES_TOOL_SEARCH_CALL = (
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_response_native_tool_call_part'
)
_RESPONSES_TOOL_SEARCH_OUTPUT = (
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_response_native_tool_return_part'
)
_OPENROUTER_REASONING = (
    'pydantic_ai.models.openrouter::OpenRouterModel._MapModelResponseContext._map_response_thinking_part'
)
_SNOWFLAKE_REASONING = (
    'pydantic_ai.models.snowflake::SnowflakeModel._MapModelResponseContext._map_response_thinking_part'
)
_SNOWFLAKE_CARRIED_THINKING = 'pydantic_ai.models.snowflake::SnowflakeModel._carries_thinking_in_user_turn'
_XAI_PARTS = 'pydantic_ai.models.xai::XaiModel._map_response_parts'
_XAI_NATIVE_TOOL_CALLS = 'pydantic_ai.models.xai::XaiModel._append_builtin_tool_call_part'
_ITEM_IDS_UNCOVERED = (
    'item IDs, and native tool calls whose state lives server-side, go out only under `should_send_item_id`, which '
    '`_map_model_response` gates'
)

REPLAY_SITES: dict[str, ReplaySite | Exempt] = {
    # Base `Model`
    'pydantic_ai.models::Model.canonical_system': _OWN_IDENTITY,
    'pydantic_ai.models::Model.accepts_server_state': Exempt('the server-state gate itself'),
    'pydantic_ai.models::Model._restore_response_scoped_tool_call_ids': ReplaySite(
        'server_state', ('tool call IDs qualified by `provider_response_id`, restored to the ones the provider sent',)
    ),
    'pydantic_ai.models::Model._compaction_window_predicate': Exempt(
        'picks the predicate the execution-availability window judges compaction parts by; it sends nothing'
    ),
    _TRIM: ReplaySite('stateless', ('the history before a compaction part, which it drops',)),
    'pydantic_ai.models::Model.prepare_messages': ReplaySite(
        'stateless',
        (
            'signed parts, which `break_special_tokens_in_messages` leaves as they are',
            'native tool-search parts, which `synthesize_local_tool_search_messages` keeps',
        ),
    ),
    'pydantic_ai.models::Model._carries_thinking_in_user_turn': ReplaySite(
        'stateless', ('`signature`: thinking the model replays natively stays out of the user turn',)
    ),
    'pydantic_ai.models::CompletedStreamedResponse.__init__': _STREAMS_A_RESPONSE,
    'pydantic_ai.models::CompletedStreamedResponse.provider_url': _STREAMS_A_RESPONSE,
    'pydantic_ai.models::_trim_messages_before_compaction': ReplaySite(
        GatedBy((_TRIM,), 'judges compaction parts by the `accepts` predicate `_trim_before_compaction` passes'),
        ('the history before a compaction part',),
        uncovered='its `provider_details` read is the standing-prompt marker Pydantic AI plants, not provider data',
    ),
    'pydantic_ai.models._batch_codec::batch_priced': _BATCH,
    'pydantic_ai.models._batch_model::_pending_ticket': _BATCH,
    'pydantic_ai.models._batch_model::_checked_again': _BATCH,
    'pydantic_ai.models._batch_model::BatchedModel._submit': _BATCH,
    'pydantic_ai.models._batch_model::BatchedModel._check': _BATCH,
    'pydantic_ai.models._batch_model::BatchedModel._resubmit': _BATCH,
    'pydantic_ai.models._continuation::merge_mode': _CONTINUATION_BOOKKEEPING,
    'pydantic_ai.models._continuation::merge_responses': _CONTINUATION_BOOKKEEPING,
    'pydantic_ai.models._continuation::_ContinuationStreamedResponse._count_continuation': _CONTINUATION_BOOKKEEPING,
    'pydantic_ai.models._continuation::_ContinuationStreamedResponse.provider_url': _CONTINUATION_BOOKKEEPING,
    'pydantic_ai.models._google_batch::batch_status': _BATCH,
    'pydantic_ai.models._google_batch::batch_cancel': _BATCH,
    'pydantic_ai.models._google_batch::unlisted_status': _BATCH,
    'pydantic_ai.models._history_digest::call_as_sent': _RECORDS_DIGEST,
    'pydantic_ai.models._reasoning_details::from_reasoning_detail': _PARSES,
    'pydantic_ai.models._reasoning_details::into_reasoning_detail': ReplaySite(
        GatedBy(
            (_OPENROUTER_REASONING, _SNOWFLAKE_REASONING, _SNOWFLAKE_CARRIED_THINKING),
            'each caller asks `accepts_stateless_artifacts` before it sends the detail',
        ),
        ('`signature` and `provider_details` as a `reasoning_details` entry',),
    ),
    # Anthropic
    'pydantic_ai.models.anthropic::AnthropicModel.compact_messages': Exempt(
        'reads the stop reason of the compaction response it received, for its error message'
    ),
    'pydantic_ai.models.anthropic::AnthropicModel._cache_diagnostics': ReplaySite(
        'server_state', ('`provider_response_id` as `diagnostics.previous_message_id`',)
    ),
    'pydantic_ai.models.anthropic::AnthropicModel._get_container': ReplaySite(
        'server_state',
        ("`provider_details['container_id']` as `container`",),
        uncovered=(
            "a `pause_turn` continuation sends the suspended response's container ungated, as the docstring says: "
            'it continues that one response'
        ),
    ),
    'pydantic_ai.models.anthropic::AnthropicModel._process_response': _PARSES,
    'pydantic_ai.models.anthropic::AnthropicModel._map_text_part': ReplaySite(
        'stateless', ("`provider_details['citations']`",)
    ),
    'pydantic_ai.models.anthropic::AnthropicModel._map_compaction_part': ReplaySite(
        'stateless', ("`provider_details['encrypted_content']` and `['signature']`",)
    ),
    'pydantic_ai.models.anthropic::AnthropicModel._map_thinking_part': ReplaySite('stateless', ('`signature`',)),
    _ANTHROPIC_NATIVE_TOOLS: ReplaySite(
        NameOnly(
            cites=(
                'Anthropic code execution and MCP result blocks (`AnthropicModel._replays_native_tool_part`) are still '
                'judged by name, pending a live check against two endpoints'
            ),
            reason='code execution and MCP blocks refer to state kept by the account that ran them',
        ),
        ('code execution and MCP blocks',),
        uncovered='search, fetch and advisor blocks carry their state inline and go through `accepts_stateless_artifacts`',
    ),
    'pydantic_ai.models.anthropic::AnthropicModel._map_tool_search_call_input': ReplaySite(
        GatedBy((_ANTHROPIC_NATIVE_TOOLS,), 'reads a tool-search call `_replays_native_tool_part` sends'),
        ("`provider_details['strategy']` and `['extra_input']`",),
    ),
    'pydantic_ai.models.anthropic._compaction::is_signed_compaction': _SHAPES_REQUEST,
    'pydantic_ai.models.anthropic._native_tools::open_turn_has_code_execution': Exempt(
        "decides, by `provider_name` and a container's presence, whether the open turn already used code execution, "
        "to set this request's tools and citations; it sends no ID"
    ),
    'pydantic_ai.models.anthropic._native_tools::build_tool_search_replay_block': ReplaySite(
        GatedBy((_ANTHROPIC_NATIVE_TOOLS,), 'builds a tool-search result `_replays_native_tool_part` sends'),
        ('the error fields `provider_details` keeps for a tool-search result',),
    ),
    'pydantic_ai.models.anthropic._native_tools::add_anthropic_caller_param': ReplaySite(
        GatedBy((_ANTHROPIC_NATIVE_TOOLS,), 'adds the caller to a block `_replays_native_tool_part` sends'),
        ("`provider_details['anthropic_caller']`",),
    ),
    'pydantic_ai.models.anthropic._native_tools::get_anthropic_code_execution_tool_name': ReplaySite(
        GatedBy((_ANTHROPIC_NATIVE_TOOLS,), 'names a code execution block `_replays_native_tool_part` sends'),
        ('the code execution tool name `provider_details` keeps',),
    ),
    'pydantic_ai.models.anthropic._native_tools::finalize_streamed_tool_search_call_part': _PARSES,
    'pydantic_ai.models.anthropic._streaming::AnthropicStreamedResponse._map_content_block_start': _PARSES,
    'pydantic_ai.models.anthropic._streaming::AnthropicStreamedResponse._map_content_block_delta': _PARSES,
    'pydantic_ai.models.anthropic._thinking_binding::history_dropped_stale_thinking_blocks': _SHAPES_REQUEST,
    'pydantic_ai.models.anthropic._threads::ThreadRecord.stamp': _RECORDS_DIGEST,
    'pydantic_ai.models.anthropic._threads::thread_digest': _CHECK,
    'pydantic_ai.models.anthropic._threads::thread_unsupported': _SHAPES_REQUEST,
    'pydantic_ai.models.anthropic._threads::thread_continuation': ReplaySite(
        Continuation('thread_digest'), ('`provider_response_id` as `previous_message_id`',)
    ),
    # Bedrock
    'pydantic_ai.models.bedrock::BedrockConverseModel.canonical_system': _OWN_IDENTITY,
    'pydantic_ai.models.bedrock::BedrockConverseModel._map_thinking_part': ReplaySite('stateless', ('`signature`',)),
    'pydantic_ai.models.bedrock::BedrockConverseModel._map_native_tool_return': ReplaySite(
        NameOnly(
            cites=None,
            reason=(
                'Nova code interpreter blocks; `_map_native_tool_call` makes the same check for the call, which the '
                'walk misses because it reads only `tool_call_id`'
            ),
        ),
        ("Nova code interpreter results, with `provider_details['status']`",),
    ),
    # Cohere
    'pydantic_ai.models.cohere::CohereModel._is_own_tool_plan': Exempt(
        'decides whether a thinking part is the plan this provider wrote before its tool calls, which goes back as '
        'plain text in `tool_plan`: not server state or a signed artifact'
    ),
    # Decision, fallback, function and wrapper models
    'pydantic_ai.models.decision::DecisionStreamedResponse.__post_init__': _STREAMS_A_RESPONSE,
    'pydantic_ai.models.decision::DecisionStreamedResponse.provider_url': _STREAMS_A_RESPONSE,
    'pydantic_ai.models.decision::_decide_response_attributes': Exempt(
        'telemetry: records `provider_response_id` as a span attribute'
    ),
    'pydantic_ai.models.fallback::FallbackModel.canonical_system': _OWN_IDENTITY,
    'pydantic_ai.models.fallback::FallbackModel._model_for_response': Exempt(
        'picks the wrapped model that produced a response, whose own gates decide what it replays'
    ),
    'pydantic_ai.models.fallback::FallbackModel.accepts_server_state': _DELEGATES,
    'pydantic_ai.models.fallback::FallbackModel.accepts_stateless_artifacts': _DELEGATES,
    'pydantic_ai.models.function::FunctionStreamedResponse._map_thinking_delta': _PARSES,
    'pydantic_ai.models.wrapper::WrapperModel.canonical_system': _OWN_IDENTITY,
    'pydantic_ai.models.wrapper::WrapperModel.accepts_server_state': _DELEGATES,
    'pydantic_ai.models.wrapper::WrapperModel.accepts_stateless_artifacts': _DELEGATES,
    # Google
    _GOOGLE_MESSAGES: ReplaySite(
        'stateless',
        ('thought signatures and the native tool parts that carry them, through `_content_model_response`',),
    ),
    'pydantic_ai.models.google::_content_model_response': ReplaySite(
        GatedBy((_GOOGLE_MESSAGES,), 'judges each part by the `accepts` predicate `GoogleModel._map_messages` passes'),
        ('`signature`',),
    ),
    'pydantic_ai.models.google::_decode_inline_thought_signature': ReplaySite(
        GatedBy((_GOOGLE_MESSAGES,), 'judges each part by the `accepts` predicate `GoogleModel._map_messages` passes'),
        ("`provider_details['thought_signature']`",),
    ),
    'pydantic_ai.models.google::_process_part': _PARSES,
    'pydantic_ai.models.google::_set_grounding_citations': _PARSES,
    # OpenAI Chat Completions
    'pydantic_ai.models.openai._batch::batch_results': _BATCH,
    'pydantic_ai.models.openai._chat_model::OpenAIChatModel._MapModelResponseContext.accepts_stateless_artifacts': (
        _DELEGATES
    ),
    'pydantic_ai.models.openai._chat_model::OpenAIChatModel._MapModelResponseContext._map_response_tool_call_part': (
        ReplaySite('stateless', ("`provider_details['extra_content']`",))
    ),
    'pydantic_ai.models.openai._citations::attach_chat_annotations': _PARSES,
    'pydantic_ai.models.openai._model_response_streaming::ModelResponseStreamedResponse.__post_init__': (
        _STREAMS_A_RESPONSE
    ),
    'pydantic_ai.models.openai._model_response_streaming::ModelResponseStreamedResponse.provider_url': (
        _STREAMS_A_RESPONSE
    ),
    # OpenAI Responses
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._reserve_websocket_lane': Exempt(
        'picks, by `provider_name`, the socket lane that cached the latest response: the ID selects a lane and is '
        'not sent, and `_is_response_id_reachable` judges what the request continues from'
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel.cancel_suspended_response': _CONTINUES,
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel.continuation_delay': Exempt(
        "reads a suspended response's `background` marker to pick a poll delay; it sends nothing"
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._history_known_unchanged': _CHECK,
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._process_streamed_response': _PARSES,
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._get_continuation_info': _CONTINUES,
    _RESPONSES_PROMPT_CACHE: ReplaySite(
        'server_state',
        ('`provider_response_id` as `prompt_cache_options.comparison_response_id`, through `_comparable_response_id`',),
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._comparable_response_id': ReplaySite(
        GatedBy((_RESPONSES_PROMPT_CACHE,), 'called for the response `accepts_server_state` accepted'),
        ('`provider_response_id`',),
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._is_compaction_boundary': ReplaySite(
        'server_state',
        ("`provider_details['compaction']`, which marks where a continuation restarts",),
        uncovered="a response it doesn't accept is judged by its compaction parts, through `accepts_stateless_artifacts`",
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._is_response_id_reachable': _CHECK,
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._get_previous_response_id_and_new_messages': (
        ReplaySite(Continuation('_history_known_unchanged'), ('`provider_response_id` as `previous_response_id`',))
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._get_conversation_id_and_new_messages': (
        ReplaySite('server_state', ("`provider_details['conversation_id']` as `conversation`",))
    ),
    _RESPONSES_ITEM_IDS: ReplaySite(
        'server_state',
        ('item IDs, which its part mappers send only under `should_send_item_id`',),
        uncovered=(
            "a part from this provider in a response from another provider, or none, is judged by the part's "
            '`provider_name`, as "Replay data" says'
        ),
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_response_text_part': ReplaySite(
        GatedBy((_RESPONSES_ITEM_IDS,), 'sends citations only with the message item ID, under `should_send_item_id`'),
        ("`provider_details['citations']`",),
        uncovered="`provider_details['phase']`, a label, is sent by the part's `provider_name`",
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_function_call': ReplaySite(
        GatedBy((_RESPONSES_ITEM_IDS,), 'sends the function call item ID only under `should_send_item_id`'),
        ('the function call item ID',),
        uncovered=(
            "`provider_details['namespace']`, a label the API requires on a discovered tool's call, is sent by the "
            "part's `provider_name`"
        ),
    ),
    _RESPONSES_TOOL_SEARCH_CALL: ReplaySite('stateless', ('tool-search call bodies',), uncovered=_ITEM_IDS_UNCOVERED),
    _RESPONSES_TOOL_SEARCH_OUTPUT: ReplaySite(
        'stateless', ('tool-search output bodies',), uncovered=_ITEM_IDS_UNCOVERED
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_response_thinking_part': ReplaySite(
        'stateless',
        ('`signature` as `encrypted_content`',),
        uncovered=(
            "`provider_details['raw_content']`, plain text, is sent by the part's `provider_name`; the reasoning item "
            'ID goes with the signature, or under `should_send_item_id`'
        ),
    ),
    'pydantic_ai.models.openai._responses_model::OpenAIResponsesModel._map_response_compaction_part': ReplaySite(
        'stateless', ("`provider_details['encrypted_content']` and the compaction item ID",)
    ),
    'pydantic_ai.models.openai._responses_streaming::OpenAIResponsesStreamedResponse.get': _PARSES,
    'pydantic_ai.models.openai._responses_streaming::OpenAIResponsesStreamedResponse._handle_tool_search_call_done': (
        _PARSES
    ),
    'pydantic_ai.models.openai._tool_search::tool_search_replay_details': ReplaySite(
        GatedBy(
            (_RESPONSES_TOOL_SEARCH_CALL, _RESPONSES_TOOL_SEARCH_OUTPUT),
            'reads a tool-search part those sites decided to send',
        ),
        ("`provider_details['call_id']` and `['status']`",),
    ),
    'pydantic_ai.models.openai._tool_search::lacks_tool_search_output_identity': ReplaySite(
        GatedBy((_RESPONSES_TOOL_SEARCH_OUTPUT,), 'decides whether a tool-search output that site sends has its item'),
        ("`provider_details['id']`",),
    ),
    # OpenRouter and Snowflake
    'pydantic_ai.models.openrouter::OpenRouterModel._carries_thinking_in_user_turn': ReplaySite(
        'stateless', ('thinking the model replays as `reasoning_details`, kept out of the user turn',)
    ),
    _OPENROUTER_REASONING: ReplaySite(
        'stateless', ('thinking as `reasoning_details`, through `into_reasoning_detail`',)
    ),
    'pydantic_ai.models.openrouter::OpenRouterStreamedResponse._map_thinking_delta': _PARSES,
    _SNOWFLAKE_CARRIED_THINKING: ReplaySite(
        'stateless', ('thinking the model replays as `reasoning_details`, kept out of the user turn',)
    ),
    _SNOWFLAKE_REASONING: ReplaySite(
        'stateless', ('thinking as `reasoning_details`, through `into_reasoning_detail`',)
    ),
    'pydantic_ai.models.snowflake::SnowflakeStreamedResponse._map_thinking_delta': _PARSES,
    # xAI
    _XAI_PARTS: ReplaySite(
        'stateless',
        ('thinking and native tool results, through `_map_thinking_part` and `_map_builtin_tool_return_part`',),
    ),
    _XAI_NATIVE_TOOL_CALLS: ReplaySite('stateless', ('native tool calls, built by `_map_builtin_tool_call_part`',)),
    'pydantic_ai.models.xai::XaiModel._map_builtin_tool_call_part': ReplaySite(
        GatedBy((_XAI_NATIVE_TOOL_CALLS,), 'its call is sent only when `accepts_stateless_artifacts` accepts the part'),
        ("`provider_details['function_name']`",),
    ),
    'pydantic_ai.models.xai::XaiModel._map_builtin_tool_return_part': ReplaySite(
        GatedBy((_XAI_PARTS,), 'called only for a part `accepts_stateless_artifacts` accepts'),
        ("`provider_details['encrypted_content']`, `['status']` and `['error']`",),
    ),
    'pydantic_ai.models.xai::XaiModel._map_thinking_part': ReplaySite(
        GatedBy((_XAI_PARTS,), 'its `native` argument is the `accepts_stateless_artifacts` answer'),
        ('`signature` as `encrypted_content`',),
    ),
    'pydantic_ai.models.xai::_attach_encrypted_result': _PARSES,
}
"""Each function that reads replay data from history or uses a replay predicate, by `'<module>::<qualname>'`."""


def _body(functions: list[FunctionNode]) -> Iterator[ast.AST]:
    """The nodes of `functions`, without those of the functions and classes they define, which the walk keys apart."""
    pending: list[ast.AST] = [child for function in functions for child in ast.iter_child_nodes(function)]
    while pending:
        node = pending.pop()
        yield node
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            pending.extend(ast.iter_child_nodes(node))


def _loads(functions: list[FunctionNode]) -> set[tuple[str, bool]]:
    """The attributes `functions` read, each with whether it's read off `self`."""
    return {
        (node.attr, isinstance(node.value, ast.Name) and node.value.id == 'self')
        for node in _body(functions)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
    }


def _name(key: str) -> str:
    """The function's own name, the last part of its qualname."""
    return key.partition('::')[2].rpartition('.')[2]


def _reachable(call_graph: dict[str, set[str]], start: str) -> set[str]:
    """`start` and every function it calls, directly or through others."""
    seen: set[str] = set()
    pending = [start]
    while pending:
        key = pending.pop()
        if key not in seen:
            seen.add(key)
            pending.extend(call_graph.get(key, ()))
    return seen


@pytest.fixture(scope='module')
def functions() -> dict[str, list[FunctionNode]]:
    """Every function in the walked files, by key; a key defined twice, like a property and its setter, has both."""
    found: dict[str, list[FunctionNode]] = {}
    for path in sorted([*(PACKAGE / 'models').rglob('*.py'), PACKAGE / '_mcp.py']):
        module = '.'.join(('pydantic_ai', *path.relative_to(PACKAGE).with_suffix('').parts)).removesuffix('.__init__')
        scopes: list[tuple[ast.AST, tuple[str, ...]]] = [(ast.parse(path.read_text(encoding='utf-8')), ())]
        while scopes:
            node, scope = scopes.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    found.setdefault(f'{module}::{".".join((*scope, child.name))}', []).append(child)
                named = isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
                scopes.append((child, (*scope, child.name) if named else scope))
    return found


@pytest.fixture(scope='module')
def call_graph(functions: dict[str, list[FunctionNode]]) -> dict[str, set[str]]:
    """The walked functions each walked function calls, resolved by name: in its own module first, else anywhere."""
    by_name: dict[str, list[str]] = {}
    for key in functions:
        by_name.setdefault(_name(key), []).append(key)
    graph: dict[str, set[str]] = {}
    for key, nodes in functions.items():
        module = key.partition('::')[0]
        graph[key] = set()
        for node in _body(nodes):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute):
                candidates = by_name.get(node.func.id if isinstance(node.func, ast.Name) else node.func.attr, [])
                local = [candidate for candidate in candidates if candidate.startswith(f'{module}::')]
                graph[key].update(local or candidates)
    return graph


def test_every_reading_function_is_classified(functions: dict[str, list[FunctionNode]]):
    """A function that starts reading history fields or using a replay predicate fails until it's classified."""
    reading = {
        key
        for key, nodes in functions.items()
        if any(
            (attr in HISTORY_FIELDS and not on_self) or attr in PREDICATES.values() for attr, on_self in _loads(nodes)
        )
    }
    assert_classified(reading, REPLAY_SITES, complete_walk=True, table_name='REPLAY_SITES')


@pytest.mark.parametrize('key', [key for key, entry in REPLAY_SITES.items() if isinstance(entry, ReplaySite)])
def test_replay_site_uses_its_gate(key: str, functions: dict[str, list[FunctionNode]], call_graph: dict[str, set[str]]):
    """Each `ReplaySite` uses the gate it declares, so the record can't drift from the code."""
    site = REPLAY_SITES[key]
    assert isinstance(site, ReplaySite)
    attrs = {attr for attr, _ in _loads(functions[key])}
    match site.gate:
        case 'server_state' | 'stateless':
            predicate = PREDICATES[site.gate]
            assert predicate in attrs, f'`{key}` declares `{site.gate}`, but never uses `{predicate}`.'
        case Continuation(check=check):
            assert 'accepts_server_state' in attrs, f'`{key}` continues a response without `accepts_server_state`.'
            checks = [reached for reached in _reachable(call_graph, key) if _name(reached) == check]
            assert checks, f'`{key}` never reaches its continuation check `{check}`.'
            assert 'history_digest' in attrs | {
                attr for check_key in checks for attr, _ in _loads(functions[check_key])
            }
        case NameOnly(cites=cites):
            assert any(
                isinstance(node, ast.Compare)
                and any(
                    isinstance(operand, ast.Attribute) and operand.attr == 'provider_name'
                    for operand in (node.left, *node.comparators)
                )
                for node in _body(functions[key])
            ), f'`{key}` is recorded as name-only, but compares no `provider_name`.'
            if cites is not None:
                assert cites in REPLAY_DATA_DOCS.read_text(encoding='utf-8'), f'`{key}` cites a line the doc lacks.'
        case GatedBy(sites=sites):
            for gating in sites:
                gating_site = REPLAY_SITES.get(gating)
                assert isinstance(gating_site, ReplaySite) and not isinstance(gating_site.gate, GatedBy), (
                    f'`{key}` is gated by `{gating}`, which is not a registered site with a gate of its own.'
                )
                callers = {caller for caller, callees in call_graph.items() if gating in callees}
                assert any(key in _reachable(call_graph, origin) for origin in (gating, *callers)), (
                    f'Neither `{gating}` nor a function that calls it reaches `{key}`.'
                )
        case _:
            assert_never(site.gate)
