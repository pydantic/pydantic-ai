from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, Callable
from dataclasses import replace
from itertools import groupby
from typing import Literal, cast

from typing_extensions import TypedDict, override

from ..messages import ModelRequest, ModelRequestPart, RetryPromptPart, ToolAvailabilityDeltaPart, ToolReturnPart
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.moonshotai import MoonshotAIModelName
from ..settings import ModelSettings, ToolOrOutput
from ..toolsets._tool_search import discovered_tool_names_in_order
from . import ModelRequestParameters
from .openai import OpenAIChatModel, OpenAIChatModelSettings

try:
    from openai import AsyncOpenAI
    from openai.types import chat
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the MoonshotAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

__all__ = ('MoonshotAIModel',)


class _ToolAdditionMessage(TypedDict):
    role: Literal['system']
    tools: list[chat.ChatCompletionToolParam]


class MoonshotAIModel(OpenAIChatModel):
    """A model using Moonshot AI's Chat Completions API.

    On `kimi-k3`, deferred tools are declared in the conversation when revealed, keeping the
    top-level tool list unchanged. Other models use the ordinary Chat Completions mapping.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    supported_tool_addition_modes = frozenset({'with_definitions'})

    def __init__(
        self,
        model_name: MoonshotAIModelName | str,
        *,
        provider: Literal['moonshotai'] | Provider[AsyncOpenAI] = 'moonshotai',
        profile: ModelProfileSpec | None = None,
        settings: OpenAIChatModelSettings | None = None,
    ):
        """Initialize a Moonshot AI model.

        Args:
            model_name: The name of the Moonshot AI model.
            provider: The provider to use. Defaults to `'moonshotai'`.
            profile: Overrides for the model profile selected by the provider.
            settings: Default settings for requests to this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def _get_user_message_mapper(
        self, model_request_parameters: ModelRequestParameters, model_settings: ModelSettings
    ) -> Callable[[ModelRequest], AsyncIterable[chat.ChatCompletionMessageParam]]:
        if self.tool_addition_mode != 'with_definitions':
            return super()._get_user_message_mapper(model_request_parameters, model_settings)

        rendered: set[str] = set()
        # The base tool-choice mapper already validates names and warns once for partial matches.
        tool_choice = model_settings.get('tool_choice')
        allowed_names: list[str] | None = [] if tool_choice == 'none' else None
        if isinstance(tool_choice, ToolOrOutput):
            allowed_names = tool_choice.function_tools
        elif isinstance(tool_choice, list):
            allowed_names = tool_choice
        # Without output tools, the API's `none` choice disables all calls. Preserve historical
        # declarations in that case so toggling tool use does not rewrite the cached prefix.
        if allowed_names == [] and not model_request_parameters.output_tools:
            allowed_names = None
        tool_defs = {
            tool.name: tool
            for tool in model_request_parameters.function_tools
            if model_request_parameters.visibility_of(tool.name) == 'via_history'
            and (allowed_names is None or tool.name in allowed_names)
        }

        async def map_user_message(message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
            pending: list[ModelRequestPart] = []
            # Keep parallel results and their media together. A later user prompt is a stable
            # boundary even when consecutive ModelRequests are merged while resuming a run.
            for _, group in groupby(
                message.parts,
                key=lambda part: (
                    isinstance(part, (ToolReturnPart, ToolAvailabilityDeltaPart))
                    or (isinstance(part, RetryPromptPart) and part.tool_name is not None)
                ),
            ):
                parts = list(group)
                pending.extend(part for part in parts if not isinstance(part, ToolAvailabilityDeltaPart))
                additions: list[_ToolAdditionMessage] = []
                for part in parts:
                    # Use the same typed and legacy discovery contract as visibility resolution.
                    names = [
                        name
                        for name in discovered_tool_names_in_order([replace(message, parts=[part])])
                        if name not in rendered and name in tool_defs
                    ]
                    if names:
                        additions.append(
                            _ToolAdditionMessage(
                                role='system',
                                tools=[self._map_tool_definition(tool_defs[name], model_settings) for name in names],
                            )
                        )
                        rendered.update(names)
                if additions:
                    if pending:
                        async for item in self._map_user_message(replace(message, parts=pending)):
                            yield item
                        pending = []
                    # Keep separate reveal records separate, so appending another delta cannot
                    # rewrite the tools array of a declaration sent in an earlier request.
                    for addition in additions:
                        # The OpenAI SDK requires `content` on system messages; Kimi's extension
                        # requires its absence. Keep that wire-only mismatch at the SDK boundary.
                        yield cast(chat.ChatCompletionMessageParam, addition)
            if pending:
                async for item in self._map_user_message(replace(message, parts=pending)):
                    yield item

        return map_user_message
