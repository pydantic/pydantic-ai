from __future__ import annotations

import sys
import warnings
from collections.abc import AsyncGenerator, Awaitable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import timedelta
from types import TracebackType
from typing import TYPE_CHECKING, Any, Self, TypeVar

import anyio

from .._run_context import RunContext
from .._warnings import PydanticAIDeprecationWarning
from ..messages import ModelMessage, ModelResponse, ModelResponsePart
from ..native_tools import AbstractNativeTool
from ..profiles import ModelProfile
from ..providers import Provider
from ..settings import ModelSettings, ThinkingEffort
from ..usage import RequestUsage
from . import (
    KnownModelName,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    StreamedResponse,
    _SettingsLayer,  # pyright: ignore[reportPrivateUsage]
    infer_model,
)
from ._request_timeout import RequestDeadline

if TYPE_CHECKING:
    from .batch import Batch, BatchRequest, BatchResult, BatchResultStatus, EncodedBatchRequest

__all__ = ['WrapperModel']

_T = TypeVar('_T')


@dataclass
class _Borrowed:
    """A model whose lifecycle its owner manages, which a wrapper being entered or exited around a request leaves alone.

    Only in the task entering or exiting the wrapper, and only while it is: a task the wrapper starts copies the
    context, and must enter and exit the model as usual.
    """

    model: Model
    task_id: int = field(default_factory=lambda: anyio.get_current_task().id)
    active: bool = True


_borrowed: ContextVar[_Borrowed | None] = ContextVar('_borrowed', default=None)


def _is_borrowed(model: Model) -> bool:
    return (
        (borrowed := _borrowed.get()) is not None
        and borrowed.active
        and borrowed.model is model
        and borrowed.task_id == anyio.get_current_task().id
    )


async def _leaving_borrowed_alone(borrowed: Model, call: Awaitable[_T]) -> _T:
    marker = _Borrowed(borrowed)
    token = _borrowed.set(marker)
    try:
        return await call
    finally:
        marker.active = False
        _borrowed.reset(token)


@asynccontextmanager
async def serving_wrapper_scope(wrapper: Model, borrowed: Model) -> AsyncGenerator[Model]:
    """Enter `wrapper`, a capability's wrapper around `borrowed`, around a request it serves.

    `borrowed` isn't entered or exited with it: its owner, like `async with agent:`, manages its lifecycle, and exiting
    it here could close an HTTP client another request is using. The exit is shielded from cancellation, so it
    finishes, and its return value is discarded, so it can't suppress the request's error. The cancel scope encloses
    the whole of it, so a cancel scope or task group the wrapper opens when entered is closed inside the same one.
    """
    with anyio.CancelScope() as scope:
        entered = await _leaving_borrowed_alone(borrowed, wrapper.__aenter__())
        try:
            yield entered
        except BaseException:
            scope.shield = True
            await _leaving_borrowed_alone(borrowed, wrapper.__aexit__(*sys.exc_info()))
            raise
        scope.shield = True
        await _leaving_borrowed_alone(borrowed, wrapper.__aexit__(None, None, None))


@dataclass(init=False)
class WrapperModel(Model):
    """Model which wraps another model.

    Does nothing on its own, used as a base class.
    """

    wrapped: Model
    """The underlying model being wrapped."""

    def __init__(self, wrapped: Model | KnownModelName):
        super().__init__()
        self.wrapped = infer_model(wrapped)

    async def __aenter__(self) -> Self:
        if not _is_borrowed(self.wrapped):
            await self.wrapped.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if _is_borrowed(self.wrapped):
            return None
        return await self.wrapped.__aexit__(exc_type, exc_val, exc_tb)

    def _refresh_http_client(self) -> None:
        self.wrapped._refresh_http_client()

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return await self.wrapped.request(messages, model_settings, model_request_parameters)

    @property
    def batch_max_requests(self) -> int | None:
        return self.wrapped.batch_max_requests

    @property
    def batch_max_bytes(self) -> int | None:
        return self.wrapped.batch_max_bytes

    def batch_request_size(self, request: EncodedBatchRequest) -> int:
        return self.wrapped.batch_request_size(request)

    def _batch_credential(self) -> str:
        return self.wrapped._batch_credential()

    def _batch_unlisted_status(self, batch: Batch) -> BatchResultStatus | None:
        return self.wrapped._batch_unlisted_status(batch)

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        return await self.wrapped.batch_encode(request)

    async def batch_create(self, requests: Sequence[EncodedBatchRequest]) -> Batch:
        return await self.wrapped.batch_create(requests)

    async def batch_status(self, batch: Batch) -> Batch:
        return await self.wrapped.batch_status(batch)

    async def batch_results(self, batch: Batch) -> list[BatchResult]:
        return await self.wrapped.batch_results(batch)

    async def batch_cancel(self, batch: Batch) -> Batch:
        return await self.wrapped.batch_cancel(batch)

    def batch_decode(self, result: BatchResult, request: BatchRequest) -> ModelResponse:
        return self.wrapped.batch_decode(result, request)

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        return await self.wrapped.cancel_suspended_response(response)

    def continuation_delay(self, response: ModelResponse) -> float | None:
        return self.wrapped.continuation_delay(response)

    def _limits_polls(self, response: ModelResponse) -> bool:
        return self.wrapped._limits_polls(response)

    def _release_suspended_response(self, response: ModelResponse) -> None:
        self.wrapped._release_suspended_response(response)

    def _withdraws_rejected_suspended_response(self, response: ModelResponse) -> bool:
        return self.wrapped._withdraws_rejected_suspended_response(response)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        return await self.wrapped.count_tokens(messages, model_settings, model_request_parameters)

    async def compact_messages(
        self,
        request_context: ModelRequestContext,
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        return await self.wrapped.compact_messages(request_context, instructions=instructions)  # pragma: no cover

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response_stream:
            yield response_stream

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return self.wrapped.customize_request_parameters(model_request_parameters)

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return self.wrapped.prepare_request(model_settings, model_request_parameters)

    def _warn_if_thinking_overridden(self, layers: Sequence[_SettingsLayer]) -> None:
        self.wrapped._warn_if_thinking_overridden(layers)

    def _unified_effort(self, model_settings: ModelSettings | None) -> ThinkingEffort | None:
        return self.wrapped._unified_effort(model_settings)

    def prepare_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        return self.wrapped.prepare_messages(messages, model_request_parameters)

    def unsupported_native_tool_settings(self, native_tool: AbstractNativeTool) -> frozenset[str]:
        return self.wrapped.unsupported_native_tool_settings(native_tool)

    @property
    def provider(self) -> Provider[Any] | None:
        return self.wrapped.provider  # pragma: no cover

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @property
    def canonical_system(self) -> str:
        return self.wrapped.canonical_system

    @property
    def compatible_systems(self) -> frozenset[str]:
        return self.wrapped.compatible_systems

    def accepts_stateless_artifacts(self, response: ModelResponse, part: ModelResponsePart | None = None) -> bool:
        # Forwarded rather than derived from the two properties above, so a wrapped model that overrides the
        # predicate keeps deciding.
        return self.wrapped.accepts_stateless_artifacts(response, part)

    def _model_for_response(self, response: ModelResponse) -> Model:
        return self.wrapped._model_for_response(response)

    @property
    def model_id(self) -> str:
        # `Model.model_id` derives from `system` and `model_name`, which are forwarded above, so for
        # most models this override is redundant. It matters for a wrapped model that computes its own
        # ID: `FallbackModel` joins its sub-models' `model_id`s, which recombining the two joined
        # strings can't reproduce. The ID names a Temporal activity and keys a Prefect cache, so a
        # mangled one isn't only a telemetry concern.
        return self.wrapped.model_id

    @property
    def profile(self) -> ModelProfile:  # type: ignore[override]
        return self.wrapped.profile

    @property
    def context_window(self) -> int | None:
        # Forwarded rather than read off `profile`: a wrapped `FallbackModel` has no profile but does
        # have a context window (the smallest among its candidates).
        return self.wrapped.context_window

    @property
    def settings(self) -> ModelSettings | None:
        """Get the settings from the wrapped model."""
        return self.wrapped.settings

    def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        # `Model.resolve_cache_retention` is defined on the base class, so without this override normal
        # attribute lookup succeeds and `__getattr__` never forwards.
        return self.wrapped.resolve_cache_retention(model_settings)

    def _start_request_deadline(self, model_settings: ModelSettings | None) -> RequestDeadline | None:
        return self.wrapped._start_request_deadline(model_settings)

    def _caching_not_enabled(self, model_settings: ModelSettings | None) -> bool:
        # Defined on the base class too, so it must forward explicitly, like `resolve_cache_retention`.
        return self.wrapped._caching_not_enabled(model_settings)

    @property
    def base_url(self) -> str | None:
        # `Model.base_url` defaults to `None`, so without this override normal attribute lookup
        # succeeds and `__getattr__` never forwards. Two consumers read it: the `server.*` span
        # attributes, and `best_effort_price(provider_api_url=...)` under
        # `UsageLimits.count_tokens_before_request` — which prefers the URL over the provider name,
        # so a wrapped model prices the same as an unwrapped one only once this forwards.
        return self.wrapped.base_url

    def __getattr__(self, item: str):
        if item == 'wrapped':
            raise AttributeError(item)
        return getattr(self.wrapped, item)


# TODO(v3): remove the `CompletedStreamedResponse` re-export shim
def __getattr__(name: str) -> Any:
    if name == 'CompletedStreamedResponse':
        warnings.warn(
            '`CompletedStreamedResponse` has moved from `pydantic_ai.models.wrapper` to `pydantic_ai.models`; '
            'import it from there instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        from . import CompletedStreamedResponse

        return CompletedStreamedResponse
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
