# There are linting and coverage escapes for MLXLM and VLLMOffline as the CI would not contain the right
# environment to be able to run the associated tests

from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils
from .._run_context import RunContext
from .._thinking_part import split_content_into_text_and_thinking
from ..exceptions import UserError
from ..messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..profiles import ModelProfile, ModelProfileSpec
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
    from outlines.models.llamacpp import LlamaCpp, from_llamacpp
    from outlines.models.mlxlm import MLXLM, from_mlxlm
    from outlines.models.sglang import AsyncSGLang, SGLang, from_sglang
    from outlines.models.transformers import (
        Transformers,
        from_transformers,
    )
    from outlines.models.vllm_offline import (
        VLLMOffline,
        from_vllm_offline,  # pyright: ignore[reportUnknownVariableType]
    )
    from outlines.types.dsl import JsonSchema
except ImportError as _import_error:
    raise ImportError(
        'Please install `outlines` to use the Outlines model, '
        'you can use the `outlines` optional group — `pip install "pydantic-ai-slim[outlines]"`'
    ) from _import_error

if TYPE_CHECKING:
    import llama_cpp
    import mlx.nn as nn
    import openai
    import transformers


class OutlinesModelSettings(ModelSettings, total=False):
    """Settings used for an Outlines model request.

    As Outlines is used to run models from various providers that do not all support the same settings, some fields of
    `ModelSettings` are available for some models, but not others.

    The docstring of the `ModelSettings` class doscuments the availability of each field. All additional
    inference arguments should be provided in the `extra_body` dictionary. They will be passed down to
    the model provider.

    """


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

    @classmethod
    def from_transformers(
        cls,
        hf_model: 'transformers.modeling_utils.PreTrainedModel',
        hf_tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer',
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
    def from_llamacpp(
        cls,
        llama_model: 'llama_cpp.Llama',
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a LlamaCpp model.

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
    def from_mlxlm(  # pragma: no cover
        cls,
        mlx_model: 'nn.Module',
        mlx_tokenizer: 'transformers.tokenization_utils.PreTrainedTokenizer',
        *,
        provider: Literal['outlines'] | Provider[OutlinesBaseModel] = 'outlines',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Create an Outlines model from a MLXLM model.

        Args:
            mlx_model: The nn.Module model to use.
            mlx_tokenizer: The PreTrainedTokenizer to use.
            provider: The provider to use for OutlinesModel. Can be either the string 'outlines' or an
                instance of `Provider[OutlinesBaseModel]`. If not provided, the other parameters will be used.
            profile: The model profile to use. Defaults to a profile picked by the provider.
            settings: Default model settings for this model instance.
        """
        outlines_model: OutlinesBaseModel = from_mlxlm(mlx_model, mlx_tokenizer)
        return cls(outlines_model, provider=provider, profile=profile, settings=settings)

    @classmethod
    def from_sglang(
        cls,
        openai_client: 'openai.OpenAI | openai.AsyncOpenAI',
        model_name: str | None = None,
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
    def from_vllm_offline(  # pragma: no cover
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
        outlines_model: OutlinesBaseModel | OutlinesAsyncBaseModel = from_vllm_offline(vllm_model)
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
            raise UserError('Outlines does not support function tools and builtin tools yet.')
        prompt = self._format_prompt(messages)
        output_type = (
            JsonSchema(model_request_parameters.output_object.json_schema)
            if model_request_parameters.output_object
            else None
        )
        inference_kwargs = self.format_inference_kwargs(model_settings)
        # Async is available for SgLang
        response: str
        if isinstance(self.model, OutlinesAsyncBaseModel):  # pragma: no cover
            response = await self.model(prompt, output_type, None, **inference_kwargs)
        else:
            response = self.model(prompt, output_type, None, **inference_kwargs)
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        if model_request_parameters.function_tools or model_request_parameters.builtin_tools:
            raise UserError('Outlines does not support function tools and builtin tools yet.')
        prompt = self._format_prompt(messages)
        output_type = (
            JsonSchema(model_request_parameters.output_object.json_schema)
            if model_request_parameters.output_object
            else None
        )
        inference_kwargs = dict(model_settings) if model_settings else {}
        # Async is available for SgLang
        if isinstance(self.model, OutlinesAsyncBaseModel):  # pragma: no cover
            response = self.model.stream(prompt, output_type, None, **inference_kwargs)
            async for chunk in response:
                yield chunk
        else:
            response = self.model.stream(prompt, output_type, None, **inference_kwargs)

            async def async_response():
                for chunk in response:
                    yield chunk

            yield await self._process_streamed_response(async_response(), model_request_parameters)

    def format_inference_kwargs(self, model_settings: ModelSettings | None) -> dict[str, Any]:
        """Format the model settings for the inference kwargs."""
        settings_dict: dict[str, Any] = dict(model_settings) if model_settings else {}
        extra_body = settings_dict.pop('extra_body', {})
        settings_dict.update(extra_body)

        if isinstance(self.model, Transformers):
            settings_dict = self._format_transformers_inference_kwargs(settings_dict)
        elif isinstance(self.model, LlamaCpp):
            settings_dict = self._format_llama_cpp_inference_kwargs(settings_dict)
        elif isinstance(self.model, MLXLM):  # pragma: no cover
            settings_dict = self._format_mlxlm_inference_kwargs(settings_dict)
        elif isinstance(self.model, SGLang | AsyncSGLang):
            settings_dict = self._format_sglang_inference_kwargs(settings_dict)
        elif isinstance(self.model, VLLMOffline):  # pragma: no cover
            settings_dict = self._format_vllm_offline_inference_kwargs(settings_dict)

        return settings_dict

    def _format_transformers_inference_kwargs(self, model_settings: dict[str, Any]) -> dict[str, Any]:
        """Map the model settings to the transformers inference kwargs."""
        for arg in [
            'timeout',
            'parallel_tool_calls',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'stop_sequences',
            'extra_headers',
        ]:
            model_settings.pop(arg, None)

        return model_settings

    def _format_llama_cpp_inference_kwargs(self, model_settings: dict[str, Any]) -> dict[str, Any]:
        """Map the model settings to the llama_cpp inference kwargs."""
        for arg in [
            'timeout',
            'parallel_tool_calls',
            'stop_sequences',
            'extra_headers',
        ]:
            model_settings.pop(arg, None)

        return model_settings

    def _format_mlxlm_inference_kwargs(  # pragma: no cover
        self, model_settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Map the model settings to the mlxlm inference kwargs."""
        for arg in [
            'temperature',
            'top_p',
            'timeout',
            'parallel_tool_calls',
            'seed',
            'presence_penalty',
            'frequency_penalty',
            'logit_bias',
            'stop_sequences',
            'extra_headers',
        ]:
            model_settings.pop(arg, None)

        return model_settings

    def _format_sglang_inference_kwargs(self, model_settings: dict[str, Any]) -> dict[str, Any]:
        """Map the model settings to the sglang inference kwargs."""
        for arg in [
            'timeout',
            'parallel_tool_calls',
            'seed',
            'logit_bias',
            'stop_sequences',
            'extra_headers',
        ]:
            model_settings.pop(arg, None)

        return model_settings

    def _format_vllm_offline_inference_kwargs(  # pragma: no cover
        self, model_settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Map the model settings to the vllm inference kwargs."""
        from vllm.sampling_params import SamplingParams  # pyright: ignore

        for arg in [
            'timeout',
            'parallel_tool_calls',
            'stop_sequences',
            'extra_headers',
        ]:
            model_settings.pop(arg, None)

        # The arguments that are part of the fields of `ModelSettings` must be put in a `SamplingParams` object and
        # provided through the `sampling_params` argument to vLLM
        sampling_params = model_settings.pop('sampling_params', SamplingParams())
        for key in model_settings.keys():
            if key in ['use_tqdm', 'lora_request', 'priority']:
                continue
            setattr(sampling_params, key, model_settings.pop(key, None))

        model_settings['sampling_params'] = sampling_params

        return model_settings

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
                                # between Pydantic AI and Outlines yet. Outlines will
                                # widen support in the future.
                                raise UserError('Outlines does not support multi-modal user prompts yet.')
                            chat.add_user_message(str(part.content))
                        else:
                            assert_never(part.content)
                    elif isinstance(part, RetryPromptPart):
                        chat.add_user_message(str(part.content))
                    elif isinstance(part, ToolReturnPart):
                        raise UserError('Tool calls are not supported for Outlines models yet.')
                    else:
                        assert_never(part)
            elif isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, TextPart):
                        chat.add_assistant_message(part.content)
                    elif isinstance(part, ThinkingPart):
                        # NOTE: We don't send ThinkingPart to the providers yet.
                        pass
                    elif isinstance(part, ToolCallPart | BuiltinToolCallPart | BuiltinToolReturnPart):
                        raise UserError('Tool calls are not supported for Outlines models yet.')
                    else:
                        assert_never(part)
            else:
                assert_never(message)
        return chat

    def _process_response(self, response: str) -> ModelResponse:
        """Turn the Outlines text response into a Pydantic AI model response instance."""
        return ModelResponse(
            parts=cast(
                list[ModelResponsePart], split_content_into_text_and_thinking(response, self.profile.thinking_tags)
            ),
        )

    async def _process_streamed_response(
        self, response: AsyncIterable[str], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Turn the Outlines text response into a Pydantic AI streamed response instance."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        timestamp = datetime.now(tz=timezone.utc)
        return OutlinesStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _model_profile=self.profile,
            _response=peekable_response,
            _timestamp=timestamp,
            _provider_name='outlines',
        )


@dataclass
class OutlinesStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Outlines models."""

    _model_name: str
    _model_profile: ModelProfile
    _response: AsyncIterable[str]
    _timestamp: datetime
    _provider_name: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for event in self._response:
            event = self._parts_manager.handle_text_delta(
                vendor_part_id='content',
                content=event,
                thinking_tags=self._model_profile.thinking_tags,
                ignore_leading_whitespace=self._model_profile.ignore_streamed_leading_whitespace,
            )
            if event is not None:  # pragma: no branch
                yield event

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp
