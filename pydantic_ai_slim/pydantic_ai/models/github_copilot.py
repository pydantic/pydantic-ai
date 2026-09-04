"""GitHub Copilot model implementation using OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass, replace
from typing import Literal, cast

from typing_extensions import override

from ..exceptions import UserError
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.github_copilot import GitHubCopilotModelProfile
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI
    from openai.types import chat

    from .openai import (
        OpenAIChatModel,
        _ChatCompletion,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the GitHub Copilot model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('GitHubCopilotModel', 'GitHubCopilotModelName')

GitHubCopilotModelName = str
"""Possible GitHub Copilot model names.

Copilot's catalog varies by subscription and changes often — an id one plan serves returns
`400 model_not_supported` on another — so no known-model list is shipped and any name is allowed.
List the ids your own plan serves with `GET https://api.githubcopilot.com/models`.
"""


@dataclass(init=False)
class GitHubCopilotModel(OpenAIChatModel):
    """A model that uses GitHub Copilot's OpenAI-compatible Chat Completions API.

    Copilot serves Anthropic, OpenAI, Google, xAI and MoonshotAI models behind one endpoint, so the
    model family — and with it the profile
    [`GitHubCopilotProvider`][pydantic_ai.providers.github_copilot.GitHubCopilotProvider] resolves —
    is derived from the prefix of the bare model id (`claude-`, `gpt-`, `gemini-`, …). Ids go out on
    the wire exactly as given.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: GitHubCopilotModelName,
        *,
        provider: Literal['github-copilot'] | Provider[AsyncOpenAI] = 'github-copilot',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a GitHub Copilot model.

        Args:
            model_name: The name of the Copilot model to use, e.g. `'claude-haiku-4.5'`.
            provider: The provider to use. Defaults to `'github-copilot'`.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        settings, params = super().prepare_request(model_settings, model_request_parameters)
        # The parent resolves the unified `thinking` setting onto the parameters, so this reads the
        # value that would actually be sent — including one that came from the model's own `settings`.
        profile = cast(GitHubCopilotModelProfile, self.profile)
        if not profile.get('github_copilot_supports_reasoning_effort', True):
            if params.thinking is False:
                # `thinking=False` asks for no reasoning, which this transport does anyway. Copilot
                # rejects `reasoning_effort='none'` on these models just as it rejects the enabling
                # levels, so the request is satisfied by sending nothing rather than by an error.
                params = replace(params, thinking=None)
            elif params.thinking is not None:
                raise UserError(
                    f'`thinking` is not supported with `GitHubCopilotModel` and model {self.model_name!r}: '
                    "GitHub Copilot's chat completions API rejects `reasoning_effort` for Anthropic models. "
                    'Use a model whose Copilot catalog entry lists `reasoning_effort`, or omit `thinking`.'
                )
        return settings, params

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> _ChatCompletion:
        # Copilot's Chat Completions envelope leaves out required OpenAI fields, and which ones
        # depends on the model: GPT ids omit `object` and `created`, Anthropic ids omit `object` and
        # each choice's `index`. The openai SDK builds responses without validating, so they arrive
        # as `None` and only fail here. Each is filled with the value the omitted field would have
        # carried, rather than widening the model and passing a hole downstream; `created` needs
        # nothing, as `OpenAIChatModel._process_response` has already filled it by this point.
        # The streamed path needs no counterpart, but not because chunks are complete: they omit
        # `object` too. It reads attributes directly instead of validating, and the two it reads —
        # a choice's `index` and each tool-call delta's `index`, which `_map_tool_call_delta` uses as
        # the part id — are present on both families, tool calls included. A missing delta index
        # would not raise here; it would silently merge parallel tool calls' argument fragments.
        payload = response.model_dump()
        # Unconditional: `object`'s type admits exactly one value, so there is nothing to overwrite.
        payload['object'] = 'chat.completion'
        for index, choice in enumerate(payload.get('choices') or []):
            if choice.get('index') is None:
                choice['index'] = index
        return _ChatCompletion.model_validate(payload)
