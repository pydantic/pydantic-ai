"""Snowflake Cortex Inference model — unified auto-routing model."""
from __future__ import annotations as _annotations

from dataclasses import dataclass

from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from .wrapper import WrapperModel

_CLAUDE_PREFIX = 'claude-'


@dataclass(init=False)
class SnowflakeCortexModel(WrapperModel):
    """Unified model for Snowflake Cortex Inference.

    Auto-routes to the correct Cortex endpoint based on model name:

    * ``claude-*`` models use :class:`~pydantic_ai.models.anthropic.AnthropicModel`
      with :class:`~pydantic_ai.providers.snowflake.SnowflakeCortexAnthropicProvider`
      (``/api/v2/cortex/v1/messages`` — Anthropic-native path, supports extended
      thinking, cache control, and other Anthropic-specific features).
    * All other models use :class:`~pydantic_ai.models.openai.OpenAIChatModel`
      with :class:`~pydantic_ai.providers.snowflake.SnowflakeCortexProvider`
      (``/api/v2/cortex/v1/chat/completions`` — OpenAI-compatible path).

    Auth reads ``SNOWFLAKE_ACCOUNT`` and ``SNOWFLAKE_TOKEN`` from the environment,
    or pass them explicitly.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai.models.snowflake import SnowflakeCortexModel

        # Llama via OpenAI-compatible /chat/completions
        agent = Agent(SnowflakeCortexModel('llama4-maverick'))

        # Claude via Anthropic-native /messages
        agent = Agent(SnowflakeCortexModel('claude-sonnet-4-6'))

        # String shorthand (reads env vars)
        agent = Agent('snowflake-cortex:llama4-maverick')
    """

    def __init__(
        self,
        model_name: str,
        *,
        account: str | None = None,
        token: str | None = None,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """Create a Snowflake Cortex model.

        Args:
            model_name: Cortex model identifier, e.g. ``'llama4-maverick'`` or
                ``'claude-sonnet-4-6'``. See the Snowflake Cortex model catalog
                for the full list.
            account: Snowflake account identifier, e.g. ``myorg-myaccount``.
                Defaults to the ``SNOWFLAKE_ACCOUNT`` environment variable.
            token: PAT or OAuth token sent as ``Authorization: Bearer <token>``.
                Defaults to the ``SNOWFLAKE_TOKEN`` environment variable.
            settings: Model-level settings applied as defaults.
            profile: Override the model profile.
        """
        if model_name.lower().startswith(_CLAUDE_PREFIX):
            from ..providers.snowflake import SnowflakeCortexAnthropicProvider
            from .anthropic import AnthropicModel

            provider = SnowflakeCortexAnthropicProvider(account=account, token=token)
            inner = AnthropicModel(model_name, provider=provider, settings=settings, profile=profile)
        else:
            from ..providers.snowflake import SnowflakeCortexProvider
            from .openai import OpenAIChatModel

            provider = SnowflakeCortexProvider(account=account, token=token)
            inner = OpenAIChatModel(model_name, provider=provider, settings=settings, profile=profile)

        super().__init__(wrapped=inner)
