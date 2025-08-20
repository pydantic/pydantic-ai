from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from .. import UnexpectedModelBehavior, _utils
from .._run_context import RunContext
from ..messages import (
    ModelMessage,
    ModelResponse,
    ModelResponseStreamEvent,
    TextPart,
)
from ..profiles import ModelProfileSpec
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
)

try:
    from outlines.inputs import Chat
    from outlines.models.base import AsyncModel as OutlinesAsyncBaseModel, Model as OutlinesBaseModel
    from outlines.models.llamacpp import from_llamacpp  # pyright: ignore[reportUnknownVariableType]
    from outlines.models.mlxlm import from_mlxlm  # pyright: ignore[reportUnknownVariableType]
    from outlines.models.sglang import from_sglang
    from outlines.models.tgi import from_tgi
    from outlines.models.transformers import from_transformers  # pyright: ignore[reportUnknownVariableType]
    from outlines.models.vllm import from_vllm
    from outlines.types.dsl import JsonSchema
except ImportError as _import_error:
    raise ImportError(
        'Please install `outlines` to use the Outlines model, '
        'you can use the `outlines` optional group — `pip install "pydantic-ai-slim[outlines]"`'
    ) from _import_error


@dataclass
class OutlinesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Outlines models."""

    _model_name: str
    _response: AsyncIterable[str]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self._response:
            event = self._parts_manager.handle_text_delta(vendor_part_id='content', content=event)
            if event is not None:  # pragma: no branch
                yield event

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


@dataclass(init=False)
class OutlinesModel(Model):
    """A model that relies on the Outlines library to run non API-based models."""

    _system: str = field(default='outlines', repr=False)

    def __init__(
        self,
        model: OutlinesBaseModel | OutlinesAsyncBaseModel,
        model_name: str | None = None,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Outlines model.

        Args:
            model: The Outlines model used for the model.
            model_name: The name of the model run by the provider.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        self.model = model
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @classmethod
    def transformers(
        cls,
        hf_model: Any,
        hf_tokenizer: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        outlines_model: OutlinesBaseModel = from_transformers(hf_model, hf_tokenizer)
        return cls(outlines_model, None, provider=provider, profile=profile, settings=settings)

    @classmethod
    def llama_cpp(
        cls,
        llama_model: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        outlines_model: OutlinesBaseModel = from_llamacpp(llama_model)
        return cls(outlines_model, None, provider=provider, profile=profile, settings=settings)

    @classmethod
    def mlxlm(
        cls,
        mlx_model: Any,
        mlx_tokenizer: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        outlines_model: OutlinesBaseModel = from_mlxlm(mlx_model, mlx_tokenizer)
        return cls(outlines_model, None, provider=provider, profile=profile, settings=settings)

    @classmethod
    def tgi(
        cls,
        client: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_tgi(client)
        return cls(outlines_model, None, provider=provider, profile=profile, settings=settings)

    @classmethod
    def sglang(
        cls,
        client: Any,
        model_name: str,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_sglang(client, model_name)
        return cls(outlines_model, None, provider=provider, profile=profile, settings=settings)

    @classmethod
    def vllm(
        cls,
        client: Any,
        model_name: str,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_vllm(client, model_name)
        return cls(outlines_model, None, provider=provider, profile=profile, settings=settings)

    @property
    def model_name(self) -> str:
        return self._model_name or ''

    @property
    def system(self) -> str:
        return self._system

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        prompt = self._format_prompt(messages)
        output_type = (
            JsonSchema(model_request_parameters.output_object.json_schema)
            if model_request_parameters.output_object
            else None
        )
        model_settings_dict = dict(model_settings) if model_settings else {}
        if isinstance(self.model, OutlinesAsyncBaseModel):
            response: str = await self.model(prompt, output_type, None, **model_settings_dict)
        else:
            response: str = self.model(prompt, output_type, None, **model_settings_dict)
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        prompt = self._format_prompt(messages)
        output_type = (
            JsonSchema(model_request_parameters.output_object.json_schema)
            if model_request_parameters.output_object
            else None
        )
        model_settings_dict = dict(model_settings) if model_settings else {}
        if isinstance(self.model, OutlinesAsyncBaseModel):
            response = self.model.stream(prompt, output_type, None, **model_settings_dict)
            async for chunk in response:
                yield chunk
        else:
            response = self.model.stream(prompt, output_type, None, **model_settings_dict)

            async def async_response():
                for chunk in response:
                    yield chunk

            yield await self._process_streamed_response(async_response(), model_request_parameters)

    def _format_prompt(self, messages: list[ModelMessage]) -> Chat:
        """Turn the model messages into an Outlines Chat instance."""
        chat = Chat()
        for message in messages:
            if message.kind == 'request':
                for part in message.parts:
                    if part.part_kind == 'system-prompt':
                        chat.add_system_message(part.content)
                    elif part.part_kind == 'user-prompt':
                        chat.add_user_message(str(part.content))
            elif message.kind == 'response':
                for part in message.parts:
                    if part.part_kind == 'text':
                        chat.add_assistant_message(str(part.content))
        return chat

    def _process_response(self, response: str) -> ModelResponse:
        """Turn the Outlines text response into a Pydantic AI model response instance."""
        return ModelResponse(parts=[TextPart(content=response)])

    async def _process_streamed_response(
        self, response: AsyncIterable[str], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Turn the Outlines text response into a Pydantic AI streamed response instance."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        timestamp = datetime.now(tz=timezone.utc)
        return OutlinesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self.model_name,
            _response=peekable_response,
            _timestamp=timestamp,
        )
