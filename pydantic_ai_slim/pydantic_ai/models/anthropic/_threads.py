"""Anthropic message threads: whether a request continues a thread, and what its create or continue request sends."""

from __future__ import annotations as _annotations

import hashlib
import hmac
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from anthropic import omit as OMIT
from anthropic.types.beta import BetaCompactionConfigParam, BetaMessageDeltaUsage, BetaMessageParam, BetaUsage

from ..._utils import is_str_mapping
from ...messages import ModelMessage, ModelResponse
from .._history_digest import Hasher, continuation_digest, history_hasher

if TYPE_CHECKING:
    from . import AnthropicModel, AnthropicModelSettings
    from ._request_params import MessagesCreateArgs

MESSAGE_THREADS_BETA = 'message-threads-2026-08-12'
_THREAD_FINGERPRINT_KEY = 'thread_fingerprint'
_THREAD_UNSUPPORTED_KEY = 'thread_unsupported'
_THREAD_REPLAYED_KEY = 'thread_replayed'

# The fields a thread keeps from its create request (`system`, `tools`) or requires unchanged on every request.
# The `skills` in `container`, whether `fallbacks` is sent, and the betas are fixed too: see `thread_fingerprint`.
_THREAD_FIXED_PARAMS = (
    'system',
    'tools',
    'model',
    'tool_choice',
    'thinking',
    'mcp_servers',
    'context_management',
    'output_config',
    'cache_control',
)


def thread_fingerprint(params: MessagesCreateArgs, betas: set[str], key: bytes | None) -> str:
    """A digest of the fields a request must share with the thread's create request to continue it.

    It's an HMAC with `key`, the model's `history_digest_key`, when there is one, like each response's
    `history_digest`. MCP authorization tokens may change between requests of a thread, so they're left out, also
    from servers that `extra_body` sets. `fallbacks`, which only `extra_body` sends, may change too: only whether
    it's sent is fixed.
    """
    fixed: dict[str, object] = {name: params[name] for name in _THREAD_FIXED_PARAMS}
    extra_body = params['extra_body']
    overrides: Mapping[str, object] = extra_body if is_str_mapping(extra_body) else {}
    # A field `extra_body` sets replaces the request's own on the wire, so the thread fixes that value instead.
    fixed.update({name: overrides[name] for name in _THREAD_FIXED_PARAMS if name in overrides})
    mcp_servers = fixed['mcp_servers']
    servers: list[object] = [*mcp_servers] if isinstance(mcp_servers, list | tuple) else []
    fixed['mcp_servers'] = [
        {key: value for key, value in server.items() if key != 'authorization_token'}
        if is_str_mapping(server)
        else server
        for server in servers
    ]
    container = overrides.get('container', params['container'])
    fixed['skills'] = container.get('skills') if is_str_mapping(container) else None
    fixed['fallbacks'] = 'fallbacks' in overrides
    fixed['betas'] = sorted(betas)
    serialized = json.dumps(fixed, sort_keys=True, default=lambda _omit: None)
    hasher: Hasher = hashlib.sha256() if key is None else hmac.new(key, digestmod=hashlib.sha256)
    hasher.update(serialized.encode())
    return hasher.hexdigest()


@dataclass(kw_only=True)
class Thread:
    """The `thread` of one request: a create request, or a continue request with only the new messages."""

    continuation: tuple[str, list[BetaMessageParam]] | None = None
    """For a continue request, the response ID it continues from and its new messages."""
    continues_replay: bool = False
    """For a continue request, whether the response it continues from was a replay (see `ThreadRecord`)."""
    cache_breakpoints: tuple[list[str], list[str]] | None = None
    """The cache breakpoints of the request, which for a continue request are only those of the new messages."""

    def apply(self, params: MessagesCreateArgs) -> MessagesCreateArgs:
        extra_body = params['extra_body']
        body: dict[str, object] = {**extra_body} if is_str_mapping(extra_body) else {}
        thread: dict[str, str] = {'type': 'create'}
        params = {**params}
        if self.continuation is not None:
            previous_message_id, messages = self.continuation
            thread = {'type': 'continue', 'previous_message_id': previous_message_id}
            # The thread keeps `system` and `tools`; sending either, even as `null` or through `extra_body`, is a 400.
            params.update(messages=messages, system=OMIT, tools=OMIT)
            body = {key: value for key, value in body.items() if key not in ('system', 'tools')}
        params['extra_body'] = {**body, 'thread': thread}
        return params


def may_use_thread(
    model: AnthropicModel, model_settings: AnthropicModelSettings, compaction: BetaCompactionConfigParam | None
) -> bool:
    """Whether `anthropic_previous_message_id` applies to a request.

    A compaction request samples no reply, so it never uses one. Nor does a request whose `extra_body` sets `messages`:
    that replaces the history the thread would be built from. Where the model's profile doesn't support threads
    (`anthropic_supports_message_threads`), the setting is ignored with a warning.
    """
    extra_body = model_settings.get('extra_body')
    if (
        model_settings.get('anthropic_previous_message_id') != 'auto'
        or compaction is not None
        or (is_str_mapping(extra_body) and 'messages' in extra_body)
    ):
        return False
    if model.profile.get('anthropic_supports_message_threads', False):
        return True
    warnings.warn(
        f'`anthropic_previous_message_id` is not supported on `{type(model.client).__name__}`; the setting will be '
        'ignored.',
        UserWarning,
        stacklevel=2,
    )
    return False


def with_thread(
    params: MessagesCreateArgs, betas: set[str], thread: Thread | None
) -> tuple[MessagesCreateArgs, set[str]]:
    """`params` and `betas` for a request sent in `thread`, or unchanged when it's sent without one."""
    if thread is None:
        return params, betas
    return thread.apply(params), betas | {MESSAGE_THREADS_BETA}


@dataclass
class ThreadRecord:
    """Set by `_first_attempt` when its request was sent in a thread, or without one that threads can't serve.

    The response is stamped with the thread's fingerprint in its `provider_details`. Sent in the thread, it also gets
    its `thread_digest` as its `history_digest`, unless a server-side fallback model wrote it: the thread doesn't
    store that turn, so it's left unstamped and the next request creates a new thread. Sent in a thread that replays
    the conversation after a continue request failed, it's also marked `'thread_replayed'`: if the API can't continue
    from it either, it may be unable to store the conversation, so that request is sent without a thread. Sent
    without it, because threads don't support the request or after that failed replay, it's marked
    `'thread_unsupported'`, so the next request with the same fingerprint doesn't try the thread again.
    """

    fingerprint: str | None = None
    history: Hasher | None = None
    """The history the request was sent in the thread with, or `None` when it was sent without a thread."""
    replayed: bool = False
    """Whether the request replayed the conversation in a new thread after a continue request failed."""

    def stamp(self, response: ModelResponse, *, from_fallback_model: bool) -> ModelResponse:
        """`response` with its thread recorded, if its request was sent in one or threads can't serve it.

        `from_fallback_model` says whether a server-side fallback model wrote it (see `served_by_fallback_model`).
        """
        if self.fingerprint is None:
            return response
        details = {**(response.provider_details or {}), _THREAD_FINGERPRINT_KEY: self.fingerprint}
        if self.history is None:
            return replace(response, provider_details={**details, _THREAD_UNSUPPORTED_KEY: True})
        if from_fallback_model:
            return response
        if self.replayed:
            details[_THREAD_REPLAYED_KEY] = True
        return replace(
            response, provider_details=details, history_digest=thread_digest(self.history, response, self.fingerprint)
        )


def served_by_fallback_model(usage: BetaUsage | BetaMessageDeltaUsage) -> bool:
    """Whether a server-side fallback model wrote a response: its `usage.iterations` then has a `fallback_message`.

    On a stream, the final `message_delta` carries it.
    """
    return any(iteration.type == 'fallback_message' for iteration in usage.iterations or [])


def thread_digest(history: Hasher, response: ModelResponse, fingerprint: str) -> str:
    """The `history_digest` of a thread response: its ID, its thread's fingerprint, and its parts after the history.

    A continue request names the response by its ID and relies on the thread's fixed fields, so the ID and the
    fingerprint are covered too. With `history_digest_key`, a client holding the history then can't swap in another
    conversation's response ID, or another configuration's fingerprint, to continue a thread.
    """
    bound = history.copy()
    bound.update(f'\n{fingerprint}\n{response.provider_response_id}'.encode())
    return continuation_digest(bound, response.parts)


def latest_response(messages: list[ModelMessage]) -> tuple[int, ModelResponse] | None:
    """The most recent response in `messages`, with its index."""
    return next(
        (
            (index, message)
            for index, message in reversed(list(enumerate(messages)))
            if isinstance(message, ModelResponse)
        ),
        None,
    )


def thread_unsupported(messages: list[ModelMessage], fingerprint: str) -> bool:
    """Whether the most recent response records that threads don't support a request with `fingerprint`."""
    latest = latest_response(messages)
    details = (latest[1].provider_details or {}) if latest is not None else {}
    return details.get(_THREAD_UNSUPPORTED_KEY) is True and details.get(_THREAD_FINGERPRINT_KEY) == fingerprint


def thread_continuation(
    model: AnthropicModel, messages: list[ModelMessage], fingerprint: str, history_digest_key: bytes | None
) -> tuple[str, list[ModelMessage], bool] | None:
    """The response ID to continue a thread from, the messages after it, and whether it was a replay, if allowed.

    Only the most recent response counts, as each response ID can be continued once: a later response from
    another model (say, through `FallbackModel`) wasn't stored in the thread. A refusal isn't continued, and
    neither is a paused turn, which pydantic-ai resumes by sending it back.

    The thread holds the response and the history before it as they were sent, so if a history processor or the
    application has changed either since (trimmed, redacted or rewritten them), the digest no longer matches and a
    new thread is started with the history as it is now. A response with no digest isn't continued: unlike
    `OpenAIResponsesModel`, there are no responses from before digests were recorded to stay compatible with.
    """
    latest = latest_response(messages)
    if latest is None:
        return None
    index, response = latest
    details = response.provider_details or {}
    new_messages = messages[index + 1 :]
    if (
        model.accepts_server_state(response)
        and response.provider_response_id
        and details.get(_THREAD_FINGERPRINT_KEY) == fingerprint
        and response.history_digest
        == thread_digest(history_hasher(messages[:index], history_digest_key), response, fingerprint)
        and response.finish_reason != 'content_filter'
        and response.state == 'complete'
        and new_messages
    ):
        return response.provider_response_id, new_messages, details.get(_THREAD_REPLAYED_KEY) is True
    return None
