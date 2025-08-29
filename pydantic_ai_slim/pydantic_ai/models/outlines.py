from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, assert_never

import openai

from .. import UnexpectedModelBehavior, _utils
from .._run_context import RunContext
from ..messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
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
    from outlines.models.dottxt import from_dottxt  # pyright: ignore[reportUnknownVariableType]
    from outlines.models.llamacpp import from_llamacpp  # pyright: ignore[reportUnknownVariableType]
    from outlines.models.sglang import from_sglang
    from outlines.models.transformers import from_transformers  # pyright: ignore[reportUnknownVariableType]
    from outlines.models.vllm import from_vllm
    from outlines.types.dsl import JsonSchema
except ImportError as _import_error:
    raise ImportError(
        'Please install `outlines` to use the Outlines model, '
        'you can use the `outlines` optional group — `pip install "pydantic-ai-slim[outlines]"`'
    ) from _import_error


@dataclass(init=False)
class OutlinesModel(Model):
    """A model that relies on the Outlines library to run non API-based models."""

    def __init__(
        self,
        model: OutlinesBaseModel | OutlinesAsyncBaseModel,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize an Outlines model.

        Args:
            model: The Outlines model used for the model.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        self.model: OutlinesBaseModel | OutlinesAsyncBaseModel = model
        self._model_name: str = 'outlines-model'

        if isinstance(provider, str):
            provider = infer_provider(provider)

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    # TODO: Add support for MLXLM and TGI when the Chat input is supported
    # for them in Outlines.

    @classmethod
    def from_transformers(
        cls,
        hf_model: Any,
        hf_tokenizer: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a Hugging Face model and tokenizer.

        Args:
            hf_model: The Hugging Face PreTrainedModel or any model that is compatible with the
                `transformers` API.
            hf_tokenizer: The Hugging Face PreTrainedTokenizer or any tokenizer that is compatible with
                the `transformers` API.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel = from_transformers(hf_model, hf_tokenizer)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_llama_cpp(
        cls,
        llama_model: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a LlamaCPP model.

        Args:
            llama_model: The llama_cpp.Llama model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel = from_llamacpp(llama_model)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_sglang(
        cls,
        openai_client: openai.OpenAI | openai.AsyncOpenAI,
        model_name: str,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a OpenAI client to send requests to an SGLang server.

        Args:
            openai_client: The openai.OpenAI or openai.AsyncOpenAI client from which to create the Outlines SGLang model.
            model_name: The name of the model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_sglang(openai_client, model_name)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_vllm_offline(
        cls,
        vllm_model: Any,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a vLLM offline inference model.

        Args:
            vllm_model: The vllm.LLM local model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_vllm(vllm_model)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_dottxt(
        cls,
        dottxt_client: Any,
        model_name: str,
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a vLLM offline inference model.

        Args:
            dottxt_client: The dottxt.Dottxt client to use.
            model_name: The name of the model to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_dottxt(dottxt_client, model_name)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return 'outlines'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model."""
        if (
            model_request_parameters.function_tools
            or model_request_parameters.builtin_tools
            or model_request_parameters.output_tools
        ):
            raise NotImplementedError(
                'Outlines does not support function tools, builtin tools or ' + 'output tools yet.'
            )
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

    def _format_prompt(self, messages: list[ModelMessage]) -> Chat:  # noqa: C901
        """Turn the model messages into an Outlines Chat instance."""
        chat = Chat()
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        chat.add_system_message(part.content)
                    elif isinstance(part, UserPromptPart):
                        if isinstance(part.content, str):
                            chat.add_user_message(part.content)
                        elif isinstance(part.content, Sequence):
                            if not all(isinstance(item, str) for item in part.content):
                                # The expected format of the assets is not compatible
                                # between Pydantic-AI and Outlines yet. Outlines will
                                # widen support in the future.
                                raise ValueError('Outlines does not support multi-modal ' + 'user prompts yet.')
                            chat.add_user_message(str(part.content))
                        else:
                            assert_never(part.content)
                    elif isinstance(part, RetryPromptPart):
                        chat.add_user_message(str(part.content))
                    elif isinstance(part, ToolReturnPart):
                        raise NotImplementedError('Tool calls are not supported for Outlines models ' + 'yet.')
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, TextPart):
                        chat.add_assistant_message(str(part.content))
                    elif isinstance(part, ThinkingPart):
                        # NOTE: We don't send ThinkingPart to the providers yet.
                        pass
                    elif isinstance(
                        part,
                        (
                            ToolCallPart,
                            BuiltinToolCallPart,
                            BuiltinToolReturnPart,
                        ),
                    ):
                        raise NotImplementedError('Tool calls are not supported for Outlines models ' + 'yet.')
                    else:
                        assert_never(part)
            else:
                assert_never(message)
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
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=timestamp,
        )


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
