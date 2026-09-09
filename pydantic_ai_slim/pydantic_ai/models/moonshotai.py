from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, Callable
from dataclasses import replace
from typing import Literal, cast

from typing_extensions import TypedDict, override

from ..messages import ModelRequest, ModelRequestPart, ToolAvailabilityDeltaPart, ToolSearchReturnPart
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.moonshotai import MoonshotAIModelName
from ..settings import ModelSettings, ToolOrOutput
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
        tool_defs = {
            tool.name: tool
            for tool in model_request_parameters.function_tools
            if model_request_parameters.visibility_of(tool.name) == 'via_history'
            and (allowed_names is None or tool.name in allowed_names)
        }

        async def map_user_message(message: ModelRequest) -> AsyncIterable[chat.ChatCompletionMessageParam]:
            pending: list[ModelRequestPart] = []
            for part in message.parts:
                if not isinstance(part, ToolAvailabilityDeltaPart):
                    pending.append(part)
                if isinstance(part, ToolAvailabilityDeltaPart | ToolSearchReturnPart):
                    names = (
                        part.tools_added
                        if isinstance(part, ToolAvailabilityDeltaPart)
                        else [tool['name'] for tool in part.discovered_tools]
                    )
                    tools: list[chat.ChatCompletionToolParam] = []
                    for name in names:
                        if name not in rendered and (tool := tool_defs.get(name)) is not None:
                            tools.append(self._map_tool_definition(tool, model_settings))
                            rendered.add(name)
                    if tools:
                        if pending:
                            async for item in self._map_user_message(replace(message, parts=pending)):
                                yield item
                            pending = []
                        addition = _ToolAdditionMessage(role='system', tools=tools)
                        # The OpenAI SDK requires `content` on system messages; Kimi's extension
                        # requires its absence. Keep that wire-only mismatch at the SDK boundary.
                        yield cast(chat.ChatCompletionMessageParam, addition)
            if pending:
                async for item in self._map_user_message(replace(message, parts=pending)):
                    yield item

        return map_user_message
