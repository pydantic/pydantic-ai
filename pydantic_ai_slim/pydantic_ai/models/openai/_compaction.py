"""`OpenAICompaction`, the compaction capability for the Responses API."""

from __future__ import annotations as _annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

from openai.types.responses.response_create_params import ContextManagement

from ..._run_context import RunContext
from ...capabilities.abstract import AbstractCapability
from ...exceptions import UserError
from ...messages import ModelMessage
from ...settings import ModelSettings
from ...tools import AgentDepsT
from .. import ModelRequestContext
from ._responses_model import OpenAIResponsesModel


@dataclass(init=False)
class OpenAICompaction(AbstractCapability[AgentDepsT]):
    """Compaction capability for OpenAI Responses API.

    Automatically compacts conversation history to keep long-running agent
    runs within manageable context limits. Two modes are supported, selected
    by the `stateless` flag:

    - **Stateful mode** (default, `stateless=False`): configures
      [OpenAI's server-side auto-compaction](https://developers.openai.com/api/docs/guides/compaction)
      via the `context_management` field on the regular `/responses` request.
      The server triggers compaction when input tokens cross a threshold,
      and the compacted item is returned alongside the normal response. On
      subsequent requests, only that item and the content after it are sent.
      Compatible with [`openai_previous_response_id='auto'`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_previous_response_id]
      and server-side conversation state.

      Configurable with `token_threshold` (`compact_threshold` on the API).
      If omitted, OpenAI picks a server-side default.

    - **Stateless mode** (`stateless=True`): calls the stateless
      `/responses/compact` endpoint from a `before_model_request` hook when
      your trigger condition is met. Use this in
      [ZDR](https://openai.com/enterprise-privacy/) environments where
      OpenAI must not retain conversation data, when you set
      [`openai_store=False`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_store],
      or when you need explicit out-of-band control over when compaction runs.

      Requires either `message_count_threshold` or a custom `trigger` callable.

    If `stateless` is not set, it is inferred from which parameters you
    provide: passing any stateless-only parameter (`message_count_threshold`
    or `trigger`) implies `stateless=True`; otherwise stateful mode is used.

    Example usage:

    ```python {test="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAICompaction

    # Stateful mode with OpenAI's server-side default threshold:
    agent = Agent(
        'openai-responses:gpt-5.2',
        capabilities=[OpenAICompaction()],
    )

    # Stateful mode with a custom token threshold:
    agent = Agent(
        'openai-responses:gpt-5.2',
        capabilities=[OpenAICompaction(token_threshold=100_000)],
    )

    # Stateless mode for ZDR environments or explicit control:
    agent = Agent(
        'openai-responses:gpt-5.2',
        capabilities=[OpenAICompaction(message_count_threshold=20)],
    )
    ```
    """

    def __init__(
        self,
        *,
        stateless: bool | None = None,
        token_threshold: int | None = None,
        message_count_threshold: int | None = None,
        trigger: Callable[[list[ModelMessage]], bool] | None = None,
    ) -> None:
        """Initialize the OpenAI compaction capability.

        Args:
            stateless: Select the compaction mode explicitly. If `None` (the
                default), the mode is inferred from the other parameters:
                passing any stateless-only parameter (`message_count_threshold`
                or `trigger`) implies `stateless=True`; otherwise stateful
                mode is used.
            token_threshold: Stateful-mode only. Input token threshold at which
                OpenAI's server-side compaction is triggered. Corresponds to
                `compact_threshold` in the `context_management` API field. If
                `None`, OpenAI picks a server-side default.
            message_count_threshold: Stateless-mode only. Compact when the
                message count exceeds this threshold.
            trigger: Stateless-mode only. Custom callable that decides whether
                to compact based on the current messages. Takes precedence
                over `message_count_threshold`.
        """
        has_stateless_only = message_count_threshold is not None or trigger is not None
        has_stateful_only = token_threshold is not None

        if stateless is None:
            stateless = has_stateless_only

        if stateless:
            if has_stateful_only:
                raise UserError(
                    '`token_threshold` is only valid for stateful compaction (`stateless=False`). '
                    'For stateless `/compact` endpoint compaction, use `message_count_threshold` or `trigger`.'
                )
            if not has_stateless_only:
                raise UserError(
                    '`stateless=True` requires `message_count_threshold` or `trigger` '
                    'to determine when to invoke the `/compact` endpoint.'
                )
        else:
            if has_stateless_only:
                raise UserError(
                    '`message_count_threshold` and `trigger` are only valid for stateless compaction '
                    '(`stateless=True`). For stateful server-side compaction, use `token_threshold` '
                    '(or omit it to use the OpenAI-managed default).'
                )

        self.stateless = stateless
        self.token_threshold = token_threshold
        self.message_count_threshold = message_count_threshold
        self.trigger = trigger

    def get_model_settings(self) -> Callable[[RunContext[AgentDepsT]], ModelSettings] | None:
        if self.stateless:
            return None
        edit: ContextManagement = {'type': 'compaction'}
        if self.token_threshold is not None:
            edit['compact_threshold'] = self.token_threshold

        def resolve(ctx: RunContext[AgentDepsT]) -> ModelSettings:
            # If the user already set `openai_context_management` on their model settings,
            # defer to it entirely — we don't want to end up with two conflicting `compaction`
            # entries, since OpenAI's context_management list only meaningfully supports one.
            if ctx.model_settings:
                existing = cast(dict[str, Any], ctx.model_settings).get('openai_context_management')
                if existing:
                    return cast(ModelSettings, {})
            return cast(ModelSettings, {'openai_context_management': [edit]})

        return resolve

    def _should_compact(self, messages: Sequence[ModelMessage]) -> bool:
        if not self.stateless:
            return False
        if self.trigger is not None:
            return self.trigger(list(messages))
        if self.message_count_threshold is not None:
            return len(messages) > self.message_count_threshold
        return False  # pragma: no cover

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        if not self._should_compact(request_context.messages):
            return request_context

        from ..wrapper import WrapperModel

        model = request_context.model
        while isinstance(model, WrapperModel):
            model = model.wrapped
        if not isinstance(model, OpenAIResponsesModel):
            raise UserError(
                f'OpenAICompaction requires OpenAIResponsesModel, got {type(model).__name__}. '
                f'Use the provider-specific compaction capability for your model.'
            )

        # Need at least 2 messages (history + current request) to compact
        if len(request_context.messages) < 2:  # pragma: no cover
            return request_context

        # Compact all messages except the last (current) request
        compact_ctx = ModelRequestContext(
            model=request_context.model,
            messages=request_context.messages[:-1],
            model_settings=request_context.model_settings,
            model_request_parameters=request_context.model_request_parameters,
        )
        compacted_response = await request_context.model.compact_messages(compact_ctx)

        # Replace message history with compaction + last request
        messages = [compacted_response, request_context.messages[-1]]
        request_context.messages = messages.copy()
        ctx.messages[:] = messages
        return request_context

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'OpenAICompaction'
