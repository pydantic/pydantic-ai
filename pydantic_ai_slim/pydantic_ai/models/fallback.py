from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from copy import copy
from dataclasses import dataclass, field, replace
from decimal import Decimal
from functools import cached_property
from types import TracebackType
from typing import TYPE_CHECKING, Any

import anyio
from opentelemetry.trace import get_current_span
from opentelemetry.util.types import AttributeValue

from pydantic_ai._instrumentation import (
    model_attributes,
    model_request_parameters_attributes,
    span_include_content,
)
from pydantic_ai._run_context import RunContext

from .._fallback import (
    ExceptionHandler,
    FallbackOn,
    FallbackPredicates,
    ResponseHandler,
    ResponseRejected,
    continuation_pin,
    raise_fallback_exception_group,
    stamp_continuation_pin,
)
from .._genai_prices import fill_response_cost
from ..exceptions import ModelAPIError
from ..messages import ModelResponse
from ..profiles import ModelProfile
from . import (
    KnownModelName,
    Model,
    ModelRequestParameters,
    StreamedResponse,
    infer_model,
)

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings

# Re-exported: these were public from this module before the fallback machinery was shared with the
# `Fallback` capability, and `from pydantic_ai.models.fallback import ...` must keep working.
__all__ = 'FallbackModel', 'FallbackOn', 'ExceptionHandler', 'ResponseHandler', 'ResponseRejected'

_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
# Must match `_continuation._REPLACE_PREVIOUS_RESPONSE_KEY`: the merge module reads this exact key
# (under `__pydantic_ai__`) to fold a post-rewind response as a replace. Duplicated as a literal rather
# than imported because that constant is module-private (importing it trips `reportPrivateUsage`).
_REPLACE_PREVIOUS_RESPONSE_KEY = 'replace_previous_response'


@dataclass(init=False)
class FallbackModel(Model):
    """A model that uses one or more fallback models upon failure.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    models: list[Model]

    _predicates: FallbackPredicates = field(repr=False)

    @cached_property
    def _enter_lock(self) -> anyio.Lock:
        # We use a cached_property for this because `anyio.Lock` binds to the event loop on which
        # it's first used; deferring creation until first access ensures it binds to the correct
        # running loop and avoids issues with Temporal's workflow sandbox.
        return anyio.Lock()

    def __init__(
        self,
        default_model: Model | KnownModelName | str,
        *fallback_models: Model | KnownModelName | str,
        fallback_on: FallbackOn = (ModelAPIError,),
    ):
        """Initialize a fallback model instance.

        Args:
            default_model: The name or instance of the default model to use.
            fallback_models: The names or instances of the fallback models to use upon failure.
            fallback_on: Conditions that trigger fallback to the next model. Accepts:

                - A tuple of exception types: `(ModelAPIError, RateLimitError)`
                - An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
                - A response handler (sync or async): `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers and lambdas), it's an exception handler.
        """
        super().__init__()
        self.models = [infer_model(default_model), *[infer_model(m) for m in fallback_models]]
        self._entered_count = 0

        self._predicates = FallbackPredicates.parse(fallback_on, owner='FallbackModel')

    async def __aenter__(self) -> FallbackModel:
        """Enter all sub-models so their providers can manage HTTP client lifecycle."""
        async with self._enter_lock:
            if self._entered_count == 0:
                async with AsyncExitStack() as exit_stack:
                    for model in self.models:
                        await exit_stack.enter_async_context(model)
                    self._exit_stack = exit_stack.pop_all()
            self._entered_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit all sub-models, closing their providers' HTTP clients."""
        async with self._enter_lock:
            self._entered_count -= 1
            if self._entered_count == 0:
                await self._exit_stack.aclose()

    @property
    def provider(self) -> None:
        return None  # pragma: no cover

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
        rejected_responses: list[ModelResponse] = []
        rejected_cost: Decimal | None = None
        # Set once a pinned continuation fails and we rewind to the chain: the first successful response
        # the chain then produces is fresh generation superseding the stale suspended turn, so it must
        # be stamped as a replace (see `_stamp_replace_previous`) rather than accumulated onto it.
        rewound = False

        if pinned := self._get_continuation_model(messages):
            # `_get_continuation_model` only returns a model when the last message is a suspended response.
            suspended_response = messages[-1]
            assert isinstance(suspended_response, ModelResponse)
            prepared_parameters = model_request_parameters
            try:
                _, prepared_parameters = pinned.prepare_request(model_settings, model_request_parameters)
                prepared_messages = pinned.prepare_messages(messages, model_request_parameters)
                response = await pinned.request(prepared_messages, model_settings, model_request_parameters)
            except Exception as exc:
                if not await self._predicates.should_fallback(exc):
                    self._set_span_attributes(pinned, prepared_parameters)
                    raise
                # Best-effort cancel the suspended server-side job we're abandoning before rewinding
                # and retrying the chain. `FallbackModel` swallows the error, so the graph's own
                # cancel path never sees it; without this an OpenAI background job would keep running
                # and billing while the chain issues a duplicate request.
                with suppress(Exception):
                    await pinned.cancel_suspended_response(suspended_response)
                messages = _rewind_messages(messages)
                rewound = True
                exceptions.append(exc)
                # Fall through to normal chain below
            else:
                if response.state == 'suspended':
                    _stamp_continuation(response, pinned)
                self._set_span_attributes(pinned, prepared_parameters)
                return response

        for model in self.models:
            prepared_parameters = model_request_parameters
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                # Each inner model has its own profile, so re-run `prepare_messages` per model.
                prepared_messages = model.prepare_messages(messages, model_request_parameters)
                response = await model.request(prepared_messages, model_settings, model_request_parameters)
            except Exception as exc:
                if await self._predicates.should_fallback(exc):
                    exceptions.append(exc)
                    continue
                self._set_span_attributes(model, prepared_parameters)
                raise exc

            if await self._predicates.should_fallback(response):
                fill_response_cost(response)
                if response.usage.cost is not None:
                    rejected_cost = (rejected_cost or Decimal()) + response.usage.cost
                rejected_responses.append(response)
                continue

            if rejected_cost is not None:
                fill_response_cost(response)
                usage = copy(response.usage)
                usage.cost = (usage.cost or Decimal()) + rejected_cost
                response = replace(response, usage=usage)

            # After a rewind, the first successful response is fresh generation that supersedes the
            # abandoned suspended turn (whether it ends complete or suspended), so mark it as a replace.
            if rewound:
                _stamp_replace_previous(response)
            if response.state == 'suspended':
                _stamp_continuation(response, model)
            self._set_span_attributes(model, prepared_parameters)
            return response

        raise_fallback_exception_group(exceptions, rejected_responses, owner='FallbackModel')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Try each model in sequence until one succeeds.

        If a previous response set `state='suspended'`, the request is routed directly
        to the pinned continuation model, bypassing the fallback chain. If the pinned model
        raises a fallback-eligible error while opening the stream, the messages are rewound
        and the normal fallback chain is tried. Mid-stream failures still propagate.
        """
        exceptions: list[Exception] = []
        # Set once a pinned continuation fails and we rewind to the chain: see the non-streaming `request`.
        rewound = False

        if pinned := self._get_continuation_model(messages):
            # `_get_continuation_model` only returns a model when the last message is a suspended response.
            suspended_response = messages[-1]
            assert isinstance(suspended_response, ModelResponse)
            async with AsyncExitStack() as stack:
                prepared_parameters = model_request_parameters
                try:
                    _, prepared_parameters = pinned.prepare_request(model_settings, model_request_parameters)
                    prepared_messages = pinned.prepare_messages(messages, model_request_parameters)
                    streamed_response = await stack.enter_async_context(
                        pinned.request_stream(prepared_messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if not await self._predicates.should_fallback(exc):
                        self._set_span_attributes(pinned, prepared_parameters)
                        raise
                    # Best-effort cancel the suspended server-side job we're abandoning before
                    # rewinding to the chain (see the non-streaming path above); `FallbackModel`
                    # swallows the error, so the graph's own cancel path never sees it.
                    with suppress(Exception):
                        await pinned.cancel_suspended_response(suspended_response)
                    messages = _rewind_messages(messages)
                    rewound = True
                    exceptions.append(exc)
                    # Fall through to normal chain below
                else:
                    self._set_span_attributes(pinned, prepared_parameters)
                    yield streamed_response
                    # Unlike `request()`, which stamps before returning, the streaming path stamps
                    # after `yield`: the final `state` is only known once the caller has consumed the
                    # stream. Callers must therefore call `get()` after the `async with` exits.
                    if streamed_response.state == 'suspended':
                        _stamp_continuation(streamed_response, pinned)
                    return

        for model in self.models:
            async with AsyncExitStack() as stack:
                prepared_parameters = model_request_parameters
                try:
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                    prepared_messages = model.prepare_messages(messages, model_request_parameters)
                    streamed_response = await stack.enter_async_context(
                        model.request_stream(prepared_messages, model_settings, model_request_parameters, run_context)
                    )
                except Exception as exc:
                    if await self._predicates.should_fallback(exc):
                        exceptions.append(exc)
                        continue
                    self._set_span_attributes(model, prepared_parameters)
                    raise exc

                # After a rewind, mark this fresh stream as replacing the abandoned suspended turn.
                # Unlike the continuation pin (stamped after `yield`, once the final `state` is known),
                # this must land on `metadata` *before* `yield`: the streamed composite resolves
                # `_segment_offset` (via `merge_mode`) on the first reindexable event, so a late stamp
                # would reindex against a stale `'accumulate'` verdict and misplace the parts. That this
                # stream supersedes the suspended turn is known the moment the rewound chain is entered.
                if rewound:
                    _stamp_replace_previous(streamed_response)
                self._set_span_attributes(model, prepared_parameters)
                yield streamed_response
                # Stamp after `yield` (see the pinned path above): `state` is only final once the
                # caller has consumed the stream, so callers must call `get()` after the context exits.
                if streamed_response.state == 'suspended':
                    _stamp_continuation(streamed_response, model)
                return

        raise_fallback_exception_group(exceptions, [], owner='FallbackModel')

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        """Cancel a suspended continuation on the underlying model holding the server-side job.

        When the response carries a continuation pin, resolve that model and delegate to it. Resolve
        the pin directly from metadata rather than via `_get_continuation_model`: the cancel path is
        driven by `_ContinuationStreamedResponse.get()`, whose `state` is already
        `'interrupted'`/`'incomplete'`/`'complete'` (never `'suspended'`) by the time cancellation
        unwinds, so gating on `state == 'suspended'` here would never find the pin.

        When no pin resolves, the response can still hold a live server-side job: the pin is only
        stamped when a segment *ends* suspended, so a streamed background job cancelled during its
        first segment (e.g. OpenAI background mode, marked by `provider_details['background']` +
        `provider_response_id`) has no pin yet. Best-effort delegate to every inner model so the job
        is torn down rather than leaked. This is safe because each model's own cancel guard is strict
        (OpenAI only acts on its own `background` marker with a matching `provider_name`; others
        no-op), and a raising model doesn't stop the rest.
        """
        if pinned := self._pinned_continuation_model(response):
            await pinned.cancel_suspended_response(response)
            return

        for model in self.models:
            with suppress(Exception):
                await model.cancel_suspended_response(response)

    def continuation_delay(self, response: ModelResponse) -> float | None:
        if pinned := self._pinned_continuation_model(response):
            return pinned.continuation_delay(response)
        for model in self.models:
            if (delay := model.continuation_delay(response)) is not None:
                return delay
        return None

    @cached_property
    def profile(self) -> ModelProfile:
        raise NotImplementedError('FallbackModel does not have its own model profile.')

    @property
    def context_window(self) -> int | None:
        """The smallest known context window among the candidate models, or `None` if none is known.

        Any candidate may end up answering, and history that fits the smallest window fits them all,
        so compacting against it errs towards compacting early rather than overflowing a fallback.
        Candidates with an unknown window don't constrain the result.
        """
        windows = [window for model in self.models if (window := model.context_window) is not None]
        return min(windows) if windows else None

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        return model_request_parameters  # pragma: no cover

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        return model_settings, model_request_parameters

    def prepare_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        # `FallbackModel` doesn't have its own profile; dispatch applies each inner model's profile instead.
        return messages

    def _get_continuation_model(self, messages: list[ModelMessage]) -> Model | None:
        """Find the model that should handle continuation from message history."""
        if not messages:  # pragma: lax no cover
            return None
        last = messages[-1]
        if not isinstance(last, ModelResponse) or last.state != 'suspended':
            return None
        return self._pinned_continuation_model(last)

    def _pinned_continuation_model(self, response: ModelResponse) -> Model | None:
        """Resolve the underlying model pinned to this continuation from its routing metadata."""
        if model_id := continuation_pin(response):
            return next((m for m in self.models if m.model_id == model_id), None)
        return None

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters) -> None:
        with suppress(Exception):
            span = get_current_span()
            if span.is_recording():
                attributes = getattr(span, 'attributes', {})
                if attributes.get('gen_ai.request.model') == self.model_name:  # pragma: no branch
                    span_attributes: dict[str, AttributeValue] = {**model_attributes(model)}
                    # Only refresh `model_request_parameters` if it was emitted at span open; its absence
                    # means `InstrumentationSettings.include_model_request_parameters` is off, and re-adding
                    # it here would leak the attribute the setting is meant to suppress.
                    if 'model_request_parameters' in attributes:
                        span_attributes.update(
                            model_request_parameters_attributes(
                                model_request_parameters,
                                # The settings aren't reachable from here, so the span carries its
                                # own `include_content` in a context variable, keyed by the span it
                                # was set for. This refresh serializes the *selected* model's
                                # parameters, whose instruction parts the outer request may not have
                                # had at all, so it cannot be inferred from what is already
                                # recorded. Fails closed on anything but this span's own policy.
                                include_content=span_include_content(span),
                            )
                        )
                    span.set_attributes(span_attributes)


def _stamp_continuation(response: ModelResponse | StreamedResponse, model: Model) -> None:
    """Stamp the model's identifier into metadata for stateless continuation routing."""
    stamp_continuation_pin(response, model.model_id)


def _stamp_replace_previous(response: ModelResponse | StreamedResponse) -> None:
    """Stamp the `replace_previous_response` marker so a fresh post-rewind turn supersedes the stale one.

    After a pinned continuation fails and `FallbackModel` rewinds and retries the chain, the first
    successful response is genuinely fresh generation, but may carry the same `model_name` as the
    abandoned suspended turn (only the `provider_response_id` differs). Without this marker
    `merge_mode` would classify the merge as an `accumulate` — same model, different id, indistinguishable
    from an Anthropic `pause_turn` — and duplicate the abandoned suspended parts ahead of the fresh turn.
    The marker (merged into the shared `__pydantic_ai__` namespace, alongside any continuation pin) tells
    the merge to `'replace-new'`; it's transient and popped after being honored so it can't persist into
    history. See `pydantic_ai.models._continuation`.
    """
    if response.metadata is None:
        response.metadata = {}
    pydantic_ai_meta = response.metadata.setdefault(_PYDANTIC_AI_METADATA_KEY, {})
    pydantic_ai_meta[_REPLACE_PREVIOUS_RESPONSE_KEY] = True


def _rewind_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Strip the suspended response from the end of message history.

    When a pinned continuation model fails, the messages still contain the suspended
    response. Before falling through to the normal chain, we remove it so models see
    clean history ending with the most recent ModelRequest.
    """
    rewound = list(messages)
    if rewound and isinstance(rewound[-1], ModelResponse) and rewound[-1].state == 'suspended':  # pragma: no branch
        rewound.pop()
    return rewound
