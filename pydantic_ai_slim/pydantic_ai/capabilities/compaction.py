"""Unified compaction capability for managing conversation context size.

Provides [`ProviderCompaction`][pydantic_ai.capabilities.ProviderCompaction], a routed capability
that automatically selects the appropriate provider-specific compaction implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability
from .routed import RoutedCapability

if TYPE_CHECKING:
    from pydantic_ai.models import Model


@dataclass
class ProviderCompaction(RoutedCapability[AgentDepsT]):
    """Unified compaction capability that routes to provider-specific implementations.

    Supports:

    - **OpenAI Responses API**: Triggers explicit compaction via the ``responses.compact``
      endpoint when the configured trigger condition is met.
    - **Anthropic**: Configures automatic ``context_management`` so compaction triggers
      server-side when input tokens exceed the threshold.

    Errors at runtime on ``FallbackModel`` or unsupported model classes.

    Example usage::

        from pydantic_ai import Agent
        from pydantic_ai.capabilities import ProviderCompaction

        agent = Agent(
            'openai-responses:gpt-4o',
            capabilities=[ProviderCompaction(message_count_threshold=10)],
        )

    Args:
        instructions: Custom instructions for the compaction summarization. Used by both providers.
        token_threshold: Token threshold for Anthropic automatic compaction (default 150,000, min 50,000).
        pause_after_compaction: Anthropic-specific: if ``True``, the response stops after
            the compaction block with ``stop_reason='compaction'``.
        message_count_threshold: OpenAI-specific: compact when message count exceeds this.
        trigger: OpenAI-specific: custom callable that decides whether to compact.
            Takes precedence over ``message_count_threshold``.
    """

    instructions: str | None = None
    token_threshold: int = 150_000
    pause_after_compaction: bool = False
    message_count_threshold: int | None = None
    trigger: Callable[[list[ModelMessage]], bool] | None = field(default=None, repr=False)

    def get_capability_for_model(self, model: Model) -> AbstractCapability[AgentDepsT]:
        try:
            from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel

            if isinstance(model, OpenAIResponsesModel):
                return OpenAICompaction(
                    message_count_threshold=self.message_count_threshold,
                    trigger=self.trigger,
                    instructions=self.instructions,
                )
        except ImportError:
            pass

        try:
            from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel

            if isinstance(model, AnthropicModel):
                return AnthropicCompaction(
                    token_threshold=self.token_threshold,
                    instructions=self.instructions,
                    pause_after_compaction=self.pause_after_compaction,
                )
        except ImportError:
            pass

        raise UserError(f'Compaction is not supported for {model.model_name}')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return 'ProviderCompaction'
