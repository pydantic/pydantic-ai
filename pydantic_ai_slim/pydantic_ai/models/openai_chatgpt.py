from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Literal

from typing_extensions import override

from ..exceptions import ModelAPIError
from ..messages import ModelMessage
from ..profiles import ModelProfileSpec
from ..profiles.openai import OpenAIModelProfile
from ..providers import Provider
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI, AsyncStream, Omit
    from openai.types import responses
    from openai.types.responses.response_input_item_param import AdditionalTools

    from .openai import (
        OpenAIModelName,
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
        OpenAIResponsesStreamedResponse,
        _ResponsesRequestParams,  # pyright: ignore[reportPrivateUsage]
    )
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai-chatgpt` optional group: `pip install "pydantic-ai-slim[openai-chatgpt]"`'
    ) from _import_error

__all__ = ('OpenAIChatGPTModel',)


class _CompletedStream(AsyncStream[responses.ResponseStreamEvent]):
    """Keep native Responses rendering, but require a successful terminal event."""

    def __init__(self, source: AsyncStream[responses.ResponseStreamEvent], model_name: str) -> None:
        self._source = source
        self._model_name = model_name

    @override
    async def __aiter__(self) -> AsyncIterator[responses.ResponseStreamEvent]:
        completed = False
        async for event in self._source:
            if isinstance(event, responses.ResponseErrorEvent):
                raise ModelAPIError(model_name=self._model_name, message=f'{event.code}: {event.message}')
            if isinstance(event, responses.ResponseFailedEvent):
                error = event.response.error
                raise ModelAPIError(
                    model_name=self._model_name,
                    message=f'{error.code}: {error.message}' if error else 'ChatGPT inference failed.',
                )
            if isinstance(event, responses.ResponseIncompleteEvent):
                raise ModelAPIError(model_name=self._model_name, message='ChatGPT inference was incomplete.')
            if isinstance(event, responses.ResponseCompletedEvent):
                completed = True
            yield event
        if not completed:
            raise ModelAPIError(
                model_name=self._model_name, message='ChatGPT stream ended without `response.completed`.'
            )

    @override
    async def close(self) -> None:
        await self._source.close()


class OpenAIChatGPTModel(OpenAIResponsesModel):
    """Responses model using the signed-in user's ChatGPT plan, not a Platform API key.

    Both ordinary and streaming requests use SSE with `store=false`. Generic `max_tokens`,
    `temperature`, and `top_p` are ignored by the provider profile. Explicit `openai_*` settings
    are forwarded so the API can report unsupported preview features. Local function/custom
    tools are supplied in `additional_tools` items; only web search is supported as a native tool.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openai-chatgpt'] | Provider[AsyncOpenAI] = 'openai-chatgpt',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ) -> None:
        """Initialize a ChatGPT plan-usage model.

        Args:
            model_name: A model slug from the signed-in account's model catalog.
            provider: A configured `OpenAIChatGPTProvider`; no implicit login or credential discovery.
            profile: Override the profile selected by the provider.
            settings: Default settings for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    async def _build_responses_request_params(
        self,
        messages: list[ModelMessage],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        profile: OpenAIModelProfile,
    ) -> _ResponsesRequestParams:
        params = await super()._build_responses_request_params(
            messages, model_settings, model_request_parameters, profile
        )
        if not isinstance(params.tools, Omit):
            functions: list[responses.ToolParam] = []
            native: list[responses.ToolParam] = []
            for tool in params.tools:
                if tool['type'] in ('function', 'custom'):
                    functions.append(tool)
                else:
                    native.append(tool)
            if functions:
                # Fixed leading position preserves the tool-prefix across a local tool roundtrip.
                params.input.insert(0, AdditionalTools(type='additional_tools', role='developer', tools=functions))
            params.tools = native or Omit()
        return params

    @override
    async def _process_streamed_response(
        self,
        response: AsyncStream[responses.ResponseStreamEvent],
        model_settings: OpenAIResponsesModelSettings,
        model_request_parameters: ModelRequestParameters,
        *,
        expected_model_name: OpenAIModelName | None = None,
        expected_response_id: str | None = None,
    ) -> OpenAIResponsesStreamedResponse:
        return await super()._process_streamed_response(
            _CompletedStream(response, self.model_name),
            model_settings,
            model_request_parameters,
            expected_model_name=expected_model_name,
            expected_response_id=expected_response_id,
        )
