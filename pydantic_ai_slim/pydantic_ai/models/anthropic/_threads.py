"""Anthropic message threads: whether a request continues a thread, and what its create or continue request sends."""

from __future__ import annotations as _annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from anthropic import omit as OMIT
from anthropic.types.beta import BetaCompactionConfigParam, BetaMessageParam

from ..._utils import is_str_mapping
from ...messages import ModelMessage, ModelResponse
from .._history_digest import Hasher, continuation_digest, history_hasher

if TYPE_CHECKING:
    from . import AnthropicModel, AnthropicModelSettings, _MessageCreateParams  # pyright: ignore[reportPrivateUsage]

MESSAGE_THREADS_BETA = 'message-threads-2026-08-12'

# The fields a thread keeps from its create request (`system`, `tools`) or requires unchanged on every request.
# `fallbacks` is fixed for a thread too; add it here once pydantic-ai sends it.
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


def thread_fingerprint(params: _MessageCreateParams, betas: set[str]) -> str:
    """A digest of the fields a request must share with the thread's create request to continue it.

    MCP authorization tokens may change between requests of a thread, so they're left out, unless `extra_body` sets
    the servers.
    """
    fixed: dict[str, object] = {name: params[name] for name in _THREAD_FIXED_PARAMS}
    fixed['mcp_servers'] = [
        {key: value for key, value in server.items() if key != 'authorization_token'}
        for server in params['mcp_servers'] or ()
    ]
    if is_str_mapping(extra_body := params['extra_body']):
        # A field `extra_body` sets replaces the request's own on the wire, so the thread fixes that value instead.
        fixed.update({name: extra_body[name] for name in _THREAD_FIXED_PARAMS if name in extra_body})
    container = params['container']
    fixed['skills'] = container.get('skills') if is_str_mapping(container) else None
    fixed['betas'] = sorted(betas)
    serialized = json.dumps(fixed, sort_keys=True, default=lambda _omit: None)
    return hashlib.sha256(serialized.encode()).hexdigest()


@dataclass(kw_only=True)
class Thread:
    """The `thread` of one request: a create request, or a continue request with only the new messages."""

    previous_message_id: str | None = None
    messages: list[BetaMessageParam] | None = None
    cache_breakpoints: tuple[list[str], list[str]] | None = None
    """The cache breakpoints of the request, which for a continue request are only those of the new messages."""

    def apply(self, params: _MessageCreateParams) -> _MessageCreateParams:
        extra_body = params['extra_body']
        body: dict[str, object] = {**extra_body} if is_str_mapping(extra_body) else {}
        thread: dict[str, str] = {'type': 'create'}
        params = {**params}
        if self.previous_message_id is not None and self.messages is not None:
            thread = {'type': 'continue', 'previous_message_id': self.previous_message_id}
            # The thread keeps `system` and `tools`; sending either, even as `null` or through `extra_body`, is a 400.
            params.update(messages=self.messages, system=OMIT, tools=OMIT)
            body = {key: value for key, value in body.items() if key not in ('system', 'tools')}
        params['extra_body'] = {**body, 'thread': thread}
        return params


def continue_cache_breakpoints(
    breakpoints: tuple[list[str], list[str]] | None,
) -> tuple[list[str], list[str]] | None:
    """The cache breakpoints of a continue request, from those of its new messages prepared on their own.

    A continue request leaves out `system` and `tools`, so their breakpoints aren't in it.
    """
    requested, applied = (
        [path for path in paths if not path.startswith(('system[', 'tools['))] for paths in breakpoints or ([], [])
    )
    return (requested, applied) if requested else None


def may_use_thread(model_settings: AnthropicModelSettings, compaction: BetaCompactionConfigParam | None) -> bool:
    """Whether `anthropic_thread` applies to a request. A compaction request samples no reply, so it never uses one."""
    return bool(model_settings.get('anthropic_thread')) and compaction is None


def with_thread(
    params: _MessageCreateParams, betas: set[str], thread: Thread | None
) -> tuple[_MessageCreateParams, set[str]]:
    """`params` and `betas` for a request sent in `thread`, or unchanged when it's sent without one."""
    if thread is None:
        return params, betas
    return thread.apply(params), betas | {MESSAGE_THREADS_BETA}


@dataclass
class ThreadRecord:
    """Set by `_messages_create` when its request was sent in a thread: the thread's fingerprint and the history.

    The response is stamped with both: the fingerprint in its `provider_details`, and the digest of the history
    with the response's own parts as its `history_digest`.
    """

    fingerprint: str | None = None
    history: Hasher | None = None

    def stamp(self, response: ModelResponse) -> ModelResponse:
        """`response` with its thread recorded, if its request was sent in one."""
        if self.fingerprint is None or self.history is None:
            return response
        details = {**(response.provider_details or {}), _THREAD_FINGERPRINT_KEY: self.fingerprint}
        return replace(
            response, provider_details=details, history_digest=continuation_digest(self.history, response.parts)
        )


_THREAD_FINGERPRINT_KEY = 'thread_fingerprint'


def thread_continuation(
    model: AnthropicModel, messages: list[ModelMessage], fingerprint: str, history_digest_key: bytes | None
) -> tuple[str, list[ModelMessage]] | None:
    """The response ID to continue a thread from, and the messages after it, if the latest response allows it.

    Only the most recent response counts, as each response ID can be continued once: a later response from
    another model (say, through `FallbackModel`) wasn't stored in the thread. A refusal isn't continued, and
    neither is a paused turn, which pydantic-ai resumes by sending it back.

    The thread holds the response and the history before it as they were sent, so if a history processor or the
    application has changed either since (trimmed, redacted or rewritten them), the digest no longer matches and a
    new thread is started with the history as it is now. A response with no digest isn't continued: unlike
    `OpenAIResponsesModel`, there are no responses from before digests were recorded to stay compatible with.
    """
    for index in range(len(messages) - 1, -1, -1):
        response = messages[index]
        if not isinstance(response, ModelResponse):
            continue
        details = response.provider_details or {}
        new_messages = messages[index + 1 :]
        if (
            model.accepts_server_state(response)
            and response.provider_response_id
            and details.get(_THREAD_FINGERPRINT_KEY) == fingerprint
            and response.history_digest is not None
            and response.history_digest
            == continuation_digest(history_hasher(messages[:index], history_digest_key), response.parts)
            and response.finish_reason != 'content_filter'
            and response.state == 'complete'
            and new_messages
        ):
            return response.provider_response_id, new_messages
        return None
    return None
