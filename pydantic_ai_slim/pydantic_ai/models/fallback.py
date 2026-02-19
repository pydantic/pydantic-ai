from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import get_current_span

from pydantic_ai._run_context import RunContext
from pydantic_ai.models.instrumented import InstrumentedModel

from ..exceptions import FallbackExceptionGroup, ModelAPIError
from ..messages import ModelRequest, ModelResponse
from ..profiles import ModelProfile
from . import KnownModelName, Model, ModelRequestParameters, StreamedResponse, infer_model

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings

_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
_FALLBACK_MODEL_NAME_KEY = 'fallback_model_name'


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _fallback_on: Callable[[Exception], bool]

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: Callable[[Exception], bool] | tuple[type[Exception], ...] = (ModelAPIError,),
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: A callable or tuple of exceptions that should trigger a fallback.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]

        if isinstance(fallback_on, tuple):
            self._fallback_on = _default_fallback_condition_factory(fallback_on)  # pyright: ignore[reportUnknownArgumentType]
        else:
            self._fallback_on = fallback_on

    @property
    def model_name(self) -> str:
        """The model name."""
        return f'fallback:{",".join(model.model_name for model in self.models)}'

    @property
    def model_id(self) -> str:
        """The fully qualified model identifier, combining the wrapped models' IDs."""
        return f'fallback:{",".join(model.model_id for model in self.models)}'

    @property
    def system(self) -> str:
        return f'fallback:{",".join(model.system for model in self.models)}'

    @property
    def base_url(self) -> str | None:
        return self.models[0].base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Try each model in sequence until one succeeds.

        In case of failure, raise a FallbackExceptionGroup with all exceptions.

        If a previous response set `state='suspended'`, the request is routed directly
        to the pinned continuation model, bypassing the fallback chain. If the pinned model
        raises a fallback-eligible error during continuation, the messages are rewound
        (stripping the suspended response and trailing continuation request) and the
        normal fallback chain is tried.
        """
        exceptions: list[Exception] = []

        if pinned := self._get_continuation_model(messages):
            try:
                _, prepared_parameters = pinned.prepare_request(model_settings, model_request_parameters)
                response = await pinned.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if not self._fallback_on(exc):
                    raise
                messages = _rewind_messages(messages)
                exceptions.append(exc)
                # Fall through to normal chain below
            else:
                if response.state == 'suspended':
                    _stamp_continuation(response, pinned)
                self._set_span_attributes(pinned, prepared_parameters)
                return response

        for model in self.models:
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await model.request(messages, model_settings, model_request_parameters)
            except Exception as exc:
                if self._fallback_on(exc):
                    exceptions.append(exc)
                    continue
                raise exc

            if response.state == 'suspended':
                _stamp_continuation(response, model)
            self._set_span_attributes(model, prepared_parameters)
            return response

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        """Try each model in sequence until one succeeds.

        If a previous response set `state='suspended'`, the request is routed directly
        to the pinned continuation model, bypassing the fallback chain. If the pinned model
        raises a fallback-eligible error while opening the stream, the messages are rewound
        and the normal fallback chain is tried. Mid-stream failures still propagate.
        """
        exceptions: list[Exception] = []

        if pinned := self._get_continuation_model(messages):
            async with AsyncExitStack() as stack:
                try:
                    _, prepared_parameters = pinned.prepare_request(model_settings, model_request_parameters)
                    streamed_response = await stack.enter_async_context(
                        pinned.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if not self._fallback_on(exc):
                        raise
                    messages = _rewind_messages(messages)
                    exceptions.append(exc)
                    # Fall through to normal chain below
                else:
                    self._set_span_attributes(pinned, prepared_parameters)
                    yield streamed_response
                    if streamed_response.state == 'suspended':
                        _stamp_continuation(streamed_response, pinned)
                    return

        for model in self.models:
            async with AsyncExitStack() as stack:
                try:
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                    streamed_response = await stack.enter_async_context(
                        model.request_stream(messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if self._fallback_on(exc):
                        exceptions.append(exc)
                        continue
                    raise exc  # pragma: no cover

                self._set_span_attributes(model, prepared_parameters)
                yield streamed_response
                if streamed_response.state == 'suspended':
                    _stamp_continuation(streamed_response, model)
                return

        raise FallbackExceptionGroup('All models from FallbackModel failed', exceptions)

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('FallbackModel does not have its own model profile.')

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters  # pragma: no cover

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    def _get_continuation_model(self, messages: list[ModelMessage]) -> Model | None:
        """Find the model that should handle continuation from message history.

        When a model returns `state='suspended'`, its `model_name` is stamped into
        the response's `metadata` (under a `__pydantic_ai__` key). On the next request,
        we extract that name and match it back to the correct model in `self.models`.
        """
        for message in reversed(messages):
            if isinstance(message, ModelResponse):
                if message.state != 'suspended':
                    return None
                pydantic_ai_meta = (message.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY, {})
                name = pydantic_ai_meta.get(_FALLBACK_MODEL_NAME_KEY)
                if name:
                    for model in self.models:
                        if model.model_name == name:
                            return model
                return None
        return None

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters):
        with suppress(Exception):
            span = get_current_span()
            if span.is_recording():
                attributes = getattr(span, 'attributes', {})
                if attributes.get('gen_ai.request.model') == self.model_name:  # pragma: no branch
                    span.set_attributes(
                        {
                            **InstrumentedModel.model_attributes(model),
                            **InstrumentedModel.model_request_parameters_attributes(model_request_parameters),
                        }
                    )


def _stamp_continuation(response: ModelResponse | StreamedResponse, model: Model) -> None:
    """Stamp the model's name into metadata for stateless continuation routing.

    Uses `metadata['__pydantic_ai__']` to avoid conflating framework-level routing state
    with provider-specific data in `provider_details`.
    """
    if response.metadata is None:
        response.metadata = {}
    pydantic_ai_meta = response.metadata.setdefault(_PYDANTIC_AI_METADATA_KEY, {})
    pydantic_ai_meta[_FALLBACK_MODEL_NAME_KEY] = model.model_name


def _rewind_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip suspended response and trailing continuation request from message end.

    When a pinned continuation model fails, the messages still contain the suspended
    response and the continuation request that triggered the failure. Before falling
    through to the normal chain, we need to remove these so models see a clean history.
    """
    rewound = list(messages)
    if rewound and isinstance(rewound[-1], ModelRequest):
        rewound.pop()
    if rewound and isinstance(rewound[-1], ModelResponse) and rewound[-1].state == 'suspended':
        rewound.pop()
    return rewound


def _default_fallback_condition_factory(exceptions: tuple[type[Exception], ...]) -> Callable[[Exception], bool]:
    """Create a default fallback condition for the given exceptions."""

    def fallback_condition(exception: Exception) -> bool:
        return isinstance(exception, exceptions)

    return fallback_condition
