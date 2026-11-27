from __future__ import annotations as _annotations

import json
import warnings
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Literal, cast

from typing_extensions import override

from .. import usage
from .._framing import noting_transport_tags
from ..exceptions import UnsupportedSettingWarning
from ..messages import ModelMessage, ModelResponse
from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from anthropic.types.beta import BetaCompactionConfigParam

    from ..providers._claude_agent_sessions import (
        CONVERSATION_HEADER,
        INHERIT_MCP_SERVERS_HEADER,
        KEEP_TRANSCRIPTS_HEADER,
    )
    from ..providers.claude_agent import ClaudeAgentProvider
    from .anthropic import AnthropicModel, AnthropicModelName, AnthropicModelSettings
    from .anthropic._request_params import PreparedMessagesCreate
except ImportError as _import_error:
    raise ImportError(
        'Please install the `anthropic` and `claude-agent-sdk` packages to use the Claude Agent model, '
        'you can use the `claude-agent` optional group — `pip install "pydantic-ai-slim[claude-agent]"`'
    ) from _import_error

__all__ = ('ClaudeAgentModel',)


_WARNED_ANTHROPIC_SETTINGS = frozenset({'anthropic_betas', 'anthropic_metadata'})
"""The `anthropic_*` settings whose loss changes what the request guarantees, like `extra_headers` and `extra_body`.

Claude Code builds the request itself and drops the other `anthropic_*` settings it doesn't take silently, as tuning
knobs, apart from those `AnthropicModel` already reports or rejects because the Claude Code profile turns them off
(`anthropic_inference_geo`, `anthropic_speed`, `anthropic_task_budget`).
"""


_CLAUDE_CODE_TAGS = ('system-reminder',)
"""The tags Claude Code adds to a request in its own voice: copies in text from a user, a tool or a file get a note."""


class ClaudeAgentModelSettings(AnthropicModelSettings, total=False):
    """Settings used for a Claude Agent model request."""

    # ALL FIELDS MUST BE `claude_agent_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    claude_agent_inherit_mcp_servers: bool | list[str]
    """Let Claude Code offer the model the MCP servers and claude.ai connectors the user set up for it.

    `True` brings all of them, and a list of names, as `claude mcp list` shows them, brings those alone. Claude Code
    runs their tools itself, and each call and its result come back as a
    [`NativeToolCallPart`][pydantic_ai.messages.NativeToolCallPart] and a
    [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart], as with Anthropic's own MCP connector.
    Defaults to `False`: Claude Code offers only the agent's own tools.
    """

    claude_agent_keep_transcripts: bool
    """Let Claude Code keep each session's transcript, as it does for any Claude Code session.

    Claude Code then writes it in its configuration directory, where it shows up in `claude --resume`. Defaults to
    `False`: Pydantic AI sends each session the history it needs, so nothing reads the transcript back, and an agent
    that makes many requests leaves no files behind.
    """


class ClaudeAgentModel(AnthropicModel):
    """A model that runs on the user's logged-in Claude account, on any Claude subscription that includes Claude Code.

    Each request goes to a Claude Code process started through the Claude Agent SDK, which signs it with the user's own
    login; this process never sees the credentials. The agent's instructions replace Claude Code's own coding-agent
    system prompt, after the Agent SDK identity line and billing block Claude Code always puts first. Its function
    tools reach Claude Code as tools that Pydantic AI runs, and Claude Code's own tools, settings files, skills and MCP
    servers are turned off.

    A conversation keeps its Claude Code process for the length of an agent run, or for as long as the model,
    provider or agent is entered with `async with`, so Claude Code's prompt cache carries over between requests.

    Claude Code doesn't take sampling settings, so `temperature`, `top_p`, `top_k`, `seed`, the penalties and
    `logit_bias` are ignored, as are `stop_sequences`, `parallel_tool_calls=False`, `extra_headers`, `extra_body`,
    `service_tier` and `timeout`. Claude Code always caches, placing its own cache breakpoints, so `cache` has no
    effect. `max_tokens`, `max_retries`, `thinking`, `tool_choice`, `anthropic_thinking` and `anthropic_effort` are
    applied.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    unsupported_settings = frozenset(
        {
            'temperature',
            'top_p',
            'top_k',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'stop_sequences',
            'parallel_tool_calls',
            'extra_headers',
            'extra_body',
            'service_tier',
            'timeout',
        }
    )

    def __init__(
        self,
        model_name: AnthropicModelName,
        *,
        provider: Literal['claude-agent'] | ClaudeAgentProvider = 'claude-agent',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Claude Agent model.

        Args:
            model_name: The name of the Claude model to use, such as `claude-sonnet-5-5`, or one of Claude Code's
                aliases, such as `sonnet`.
            provider: The provider to use. Defaults to `'claude-agent'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        if provider == 'claude-agent':
            provider = ClaudeAgentProvider()
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def prepare_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        # Claude Code adds `<system-reminder>` blocks of its own after the request leaves us, and Claude reads them as
        # the client's. A copy in text from a user, a tool or a file gets the standard lookalike note naming its author.
        with noting_transport_tags(_CLAUDE_CODE_TAGS):
            return super().prepare_messages(messages, model_request_parameters)

    @override
    def _warn_unsupported_settings(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> None:
        super()._warn_unsupported_settings(model_settings, model_request_parameters)
        # `cache` is left alone: Claude Code always caches, placing its own cache breakpoints.
        for name in sorted(_WARNED_ANTHROPIC_SETTINGS.intersection(model_settings or {})):
            warnings.warn(
                f'The `{name}` setting is not supported by model {self.model_id!r} and will be ignored: '
                'Claude Code builds the request itself.',
                UnsupportedSettingWarning,
                stacklevel=3,
            )

    @override
    def _run_scope(self) -> AbstractAsyncContextManager[None]:
        provider = self._provider
        assert isinstance(provider, ClaudeAgentProvider)
        return provider._sessions.lease()  # pyright: ignore[reportPrivateUsage]

    @override
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        # Claude Code only streams.
        async with self.request_stream(messages, model_settings, model_request_parameters) as response:
            async for _ in response:
                pass
        return response.get()

    @override
    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        raise NotImplementedError('Claude Code does not count tokens ahead of a request.')

    @override
    async def _prepare_messages_create(
        self,
        messages: list[ModelMessage],
        model_settings: AnthropicModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        compaction: BetaCompactionConfigParam | None = None,
        continues_thread: bool = False,
        thread_history: Sequence[ModelMessage] = (),
        record_cache_breakpoints_on_send: bool = False,
        count_tokens: bool = False,
    ) -> PreparedMessagesCreate:
        prepared = await super()._prepare_messages_create(
            messages,
            model_settings,
            model_request_parameters,
            compaction=compaction,
            continues_thread=continues_thread,
            thread_history=thread_history,
            record_cache_breakpoints_on_send=record_cache_breakpoints_on_send,
            count_tokens=count_tokens,
        )
        # The conversation picks the Claude Code session that continues it. The header only reaches the provider's
        # in-memory transport.
        if conversation_id := next((m.conversation_id for m in reversed(messages) if m.conversation_id), None):
            prepared.extra_headers = {**prepared.extra_headers, CONVERSATION_HEADER: conversation_id}
        claude_agent_settings = cast(ClaudeAgentModelSettings, model_settings)
        if inherit := claude_agent_settings.get('claude_agent_inherit_mcp_servers'):
            prepared.extra_headers = {**prepared.extra_headers, INHERIT_MCP_SERVERS_HEADER: json.dumps(inherit)}
        if claude_agent_settings.get('claude_agent_keep_transcripts'):
            prepared.extra_headers = {**prepared.extra_headers, KEEP_TRANSCRIPTS_HEADER: '1'}
        return prepared
