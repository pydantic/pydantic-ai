from __future__ import annotations as _annotations

import warnings
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Literal

from typing_extensions import override

from .. import usage
from ..exceptions import UnsupportedSettingWarning
from ..messages import ModelMessage, ModelResponse
from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from anthropic.types.beta import BetaCompactionConfigParam

    from ..providers._claude_code_sessions import CONVERSATION_HEADER
    from ..providers.claude_code import ClaudeCodeProvider
    from .anthropic import AnthropicModel, AnthropicModelName, AnthropicModelSettings
    from .anthropic._request_params import PreparedMessagesCreate
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `anthropic` and `claude-agent-sdk` packages to use the Claude Code model, '
        'you can use the `claude-code` optional group — `pip install "pydantic-ai-slim[claude-code]"`'
    ) from _import_error

__all__ = ('ClaudeCodeModel',)


_HONORED_ANTHROPIC_SETTINGS = frozenset(
    {
        'anthropic_thinking',
        'anthropic_effort',
        # `AnthropicModel` itself warns about these, or rejects them, as the Claude Code profile turns them off.
        'anthropic_speed',
        'anthropic_task_budget',
    }
)
"""The `anthropic_*` settings Claude Code takes, or that `AnthropicModel` already reports as unsupported."""


class ClaudeCodeModel(AnthropicModel):
    """A model that uses Claude through Claude Code, with the Claude login the user set up for it.

    Each request goes to a Claude Code process started through the Claude Agent SDK, which signs it with the user's own
    login; this process never sees the credentials. The agent's instructions replace Claude Code's system prompt, its
    function tools reach Claude Code as tools that Pydantic AI runs, and Claude Code's own tools, settings files,
    skills and MCP servers are turned off.

    A conversation keeps its Claude Code process for the length of an agent run, or for as long as the model,
    provider or agent is entered with `async with`, so Claude Code's prompt cache carries over between requests.

    Claude Code doesn't take sampling settings, so `temperature`, `top_p`, `top_k`, `seed`, the penalties and
    `logit_bias` are ignored, as are `stop_sequences`, `parallel_tool_calls=False`, `extra_headers`, `extra_body`,
    `service_tier`, `timeout` and `max_retries`. `max_tokens`, `thinking` and `anthropic_effort` are applied.

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
        provider: Literal['claude-code'] | ClaudeCodeProvider = 'claude-code',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Claude Code model.

        Args:
            model_name: The name of the Claude model to use, such as `claude-sonnet-5-5`, or one of Claude Code's
                aliases, such as `sonnet`.
            provider: The provider to use. Defaults to `'claude-code'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Default model settings for this model instance.
        """
        if provider == 'claude-code':
            provider = ClaudeCodeProvider()
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def _warn_unsupported_settings(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> None:
        super()._warn_unsupported_settings(model_settings, model_request_parameters)
        for name in sorted(set(model_settings or {}) - _HONORED_ANTHROPIC_SETTINGS):
            if name.startswith('anthropic_'):
                warnings.warn(
                    f'The `{name}` setting is not supported by model {self.model_id!r} and will be ignored: '
                    'Claude Code builds the request itself.',
                    UnsupportedSettingWarning,
                    stacklevel=3,
                )

    @override
    def _run_scope(self) -> AbstractAsyncContextManager[None]:
        provider = self._provider
        assert isinstance(provider, ClaudeCodeProvider)
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
        return prepared
