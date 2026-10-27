"""Auto-injected capability that follows a Pydantic AI Gateway fallback to a model in another API format."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any

from pydantic_ai.exceptions import ModelHTTPError, RetryModelRequest, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, infer_model, parse_model_id
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.providers.gateway import (
    _FALLBACK_STATUS_CODE,  # pyright: ignore[reportPrivateUsage]
    _fallback_headers_scope,  # pyright: ignore[reportPrivateUsage]
    _fallback_provider,  # pyright: ignore[reportPrivateUsage]
    _fallback_target,  # pyright: ignore[reportPrivateUsage]
    is_gateway_provider,
)
from pydantic_ai.tools import RunContext

from .abstract import AbstractCapability, CapabilityOrdering
from .model_fallback import ModelFallback

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters, StreamedResponse
    from pydantic_ai.providers import Provider
    from pydantic_ai.settings import ModelSettings

__all__ = ('GatewayFallback',)

_MAX_HAND_BACKS = 5
"""How many times in a row the gateway's hand-back is followed before its error stands."""


class GatewayFallback(AbstractCapability[Any]):
    """Follow a Pydantic AI Gateway fallback to a model in another API format.

    A Gateway Endpoint can fall back to a provider that speaks a different API format than the request. The gateway
    can't forward the request there itself, so it hands it back: a `424` naming the `gateway/` model to send it to
    instead, which only clients that say they can follow it get. This capability says so on each attempt at a gateway
    model, and on a hand-back moves the attempt to the named model with
    [`RetryModelRequest`][pydantic_ai.exceptions.RetryModelRequest], so the request is rebuilt for it as it is for any
    fallback candidate. The new model's provider has the original one's credentials, base URL and route, and its
    requests carry the gateway's state token so the gateway resumes its chain where it left off.

    It's auto-injected into every agent and keeps no state. It only acts on requests to gateway models outside durable
    execution, so for any other request nothing changes, including when a stream is opened. It sits inside
    [`ModelFallback`][pydantic_ai.capabilities.ModelFallback], so the gateway's chain is followed before the agent's.
    """

    def get_ordering(self) -> CapabilityOrdering:
        # Innermost and inside `ModelFallback`, so a hand-back reaches this error hook before `ModelFallback` takes it
        # for a failure to move on from. Innermost also puts its model wrapper nearest the model, inside
        # `BatchModelRequests`'s, which sends a batched request through the batch API rather than calling it.
        return CapabilityOrdering(position='innermost', wrapped_by=(ModelFallback,))

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # not spec-constructible (internal, auto-injected)

    def _realtime_uncalled_hooks(self) -> tuple[str, ...]:
        # A realtime session makes no model requests for the gateway to hand back, so there's nothing to warn about.
        return ()

    def _handles_model_request_error(self, request_context: ModelRequestContext) -> bool:
        return _advertising_model(request_context.model) is not None

    def get_wrapper_model(self, model: Model) -> Model | None:
        if isinstance(model, _FallbackHeadersModel) or not _is_gateway_model(model) or _in_durable_unit():
            # A model the gateway handed a request to sends the headers itself. A durable unit wraps the model it
            # rebuilds, and the hand-back it gets couldn't be followed on the other side of the unit.
            return None
        return _FallbackHeadersModel(model)

    async def on_model_request_error(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        # This hook only runs for an attempt that told the gateway it follows a hand-back, as
        # `_handles_model_request_error` says.
        advertised = _advertising_model(request_context.model)
        assert advertised is not None
        if not (
            isinstance(error, ModelHTTPError)
            and error.status_code == _FALLBACK_STATUS_CODE
            and (target := _fallback_target(error.headers)) is not None
            and advertised.hand_backs < _MAX_HAND_BACKS
        ):
            raise error
        model_string, state = target
        provider = advertised.provider
        assert provider is not None  # a gateway provider, or the request wouldn't have advertised
        try:
            model = _build_target(provider, model_string)
        except (UserError, ImportError) as e:
            # A model this install can't build, like one of an unknown format: the hand-back is the error.
            raise error from e
        raise RetryModelRequest(_FallbackHeadersModel(model, state=state, hand_backs=advertised.hand_backs + 1))


def _build_target(provider: Provider[Any], model_string: str) -> Model:
    upstream, _ = parse_model_id(model_string)
    assert upstream is not None  # `_fallback_target` only returns `gateway/<provider>:<model>` strings
    target_provider = _fallback_provider(provider, upstream)
    return infer_model(model_string, provider_factory=lambda _: target_provider)


def _is_gateway_model(model: Model) -> bool:
    return (provider := model.provider) is not None and is_gateway_provider(provider)


def _advertising_model(model: Model) -> _FallbackHeadersModel | None:
    """The wrapper that tells the gateway the request to `model` follows a hand-back, if the request is sent here.

    A durable engine's model sends its request from a durable unit, which doesn't advertise: the model the next
    attempt needs couldn't be built there, so the gateway falls back within the request's API format only, as it does
    for any other client.
    """
    if not _is_gateway_model(model):
        return None
    from pydantic_ai.durable_exec._utils import DurableModel

    current = model
    while isinstance(current, WrapperModel) and not isinstance(current, DurableModel):
        if isinstance(current, _FallbackHeadersModel):
            return current
        current = current.wrapped
    return None


def _in_durable_unit() -> bool:
    from pydantic_ai.durable_exec._operation_backend import in_durable_unit

    return in_durable_unit()


class _FallbackHeadersModel(WrapperModel):
    """Tells the gateway, on the requests this model sends, that the client follows a hand-back.

    Wraps the model serving one attempt at a gateway model, and the model the gateway handed a request to, with the
    gateway's state token to send it.
    """

    def __init__(self, wrapped: Model, *, state: str | None = None, hand_backs: int = 0):
        super().__init__(wrapped)
        self._state = state
        self.hand_backs = hand_backs

    def _no_batch_api_message(self) -> str:
        return self.wrapped._no_batch_api_message()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        with _fallback_headers_scope(self._state):
            return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with AsyncExitStack() as stack:
            # The request goes out when the stream is opened; the events that follow arrive on that response.
            with _fallback_headers_scope(self._state):
                response_stream = await stack.enter_async_context(
                    self.wrapped.request_stream(messages, model_settings, model_request_parameters, run_context)
                )
            yield response_stream
