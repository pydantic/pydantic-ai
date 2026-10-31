from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import timedelta
from functools import cached_property
from types import TracebackType
from typing import TYPE_CHECKING, Any

import anyio
from opentelemetry.trace import Span, get_current_span
from opentelemetry.util.types import AttributeValue

from pydantic_ai._instrumentation import (
    model_attributes,
    model_request_parameters_attributes,
    open_request_policy,
    span_include_content,
)
from pydantic_ai._model_request_attempts import AttemptStart, carry_attempts
from pydantic_ai._run_context import RunContext

from .._canonical_system import stateless_artifact_system
from .._fallback import (
    FALLBACK_MODEL_PIN_KEY,
    ExceptionHandler,
    FallbackChain,
    FallbackOn,
    FallbackPredicates,
    ResponseHandler,
    ResponseRejected,
    continuation_pin,
    request_stream_until_first_event,
    stamp_continuation_pin,
)
from .._unsupported_settings import suppress_warnings as suppress_unsupported_setting_warnings
from ..exceptions import ModelAPIError
from ..messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    ModelResponse,
    ModelResponsePart,
)
from ..profiles import ModelProfile
from ..settings import merge_model_settings
from . import (
    KnownModelName,
    Model,
    ModelRequestParameters,
    StreamedResponse,
    _SettingsLayer,  # pyright: ignore[reportPrivateUsage]
    infer_model,
)
from ._continuation import cancel_suspended_job, mark_suspended_job_cancelled
from ._request_timeout import (
    RequestDeadline,
    carry_deadline,
    enforce_request_deadline,
    resume_deadline,
    start_request_deadline,
)

if TYPE_CHECKING:
    from ..messages import ModelMessage
    from ..settings import ModelSettings, ThinkingEffort

# Re-exported: these were public from this module before the fallback machinery was shared with the
# `ModelFallback` capability, and `from pydantic_ai.models.fallback import ...` must keep working.
__all__ = 'FallbackModel', 'FallbackOn', 'ExceptionHandler', 'ResponseHandler', 'ResponseRejected'

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

                - A tuple of exception types: `(ModelRateLimitError, ModelUnavailableError)`
                - An exception handler (sync or async): `lambda exc: isinstance(exc, MyError)`
                - A response handler (sync or async): `def check(r: ModelResponse) -> bool`
                - A sequence mixing all of the above: `[ModelAPIError, exc_handler, response_handler]`

                Handler type is auto-detected by inspecting type hints on the first parameter.
                If the first parameter is hinted as `ModelResponse`, it's a response handler.
                Otherwise (including untyped handlers and lambdas), it's an exception handler.

                Response handlers only apply to non-streamed requests. A streamed request accepts
                the response as is and emits a `UserWarning`. A request is streamed whenever
                something consumes the run's events, including `run()` with an
                `event_stream_handler` or an event listener.
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

    def _refresh_http_client(self) -> None:
        for model in self.models:
            model._refresh_http_client()

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

    @property
    def canonical_system(self) -> str:
        """The first model's: each model decides for itself which history data it sends when it's tried."""
        return self.models[0].canonical_system

    @property
    def compatible_systems(self) -> frozenset[str]:
        """The first model's, like `canonical_system`."""
        return self.models[0].compatible_systems

    def accepts_stateless_artifacts(self, response: ModelResponse, part: ModelResponsePart | None = None) -> bool:
        """The first model's answer, like `canonical_system`."""
        return self.models[0].accepts_stateless_artifacts(response, part)

    def accepts_server_state(self, response: ModelResponse) -> bool:
        """The first model's answer, like `canonical_system`."""
        return self.models[0].accepts_server_state(response)

    def _model_for_response(self, response: ModelResponse) -> Model:
        # A model with the response's provider name and canonical system: two models can share a name and differ
        # in system, like a `GoogleProvider` on the Gemini API and one on Google Cloud. The response doesn't say
        # which of two matching models produced it, such as an `OpenAIChatModel` and an `OpenAIResponsesModel` on
        # one provider, so one that trims before compaction parts is preferred: the window it gives tool dispatch
        # errs toward refusing a call rather than admitting a tool the request didn't show.
        candidates = [model._model_for_response(response) for model in self.models]
        provider_name = response.provider_name
        matching = [
            candidate
            for candidate in candidates
            if provider_name is not None
            and candidate.system == provider_name
            # A response recorded before it carried `canonical_system` is judged as its stateless data would be.
            and candidate.canonical_system
            == (response.canonical_system or stateless_artifact_system(response, provider_name)[0])
        ]
        return next(
            (candidate for candidate in matching if candidate._honors_compaction_boundaries),
            matching[0] if matching else candidates[0],
        )

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
        raises a fallback-eligible error during continuation, or its response is rejected by a
        response handler, the messages are rewound (stripping the suspended response and
        trailing continuation request) and the normal fallback chain is tried.
        """
        chain = self._chain()
        # Set once a pinned continuation fails and we rewind to the chain: the first successful response
        # the chain then produces is fresh generation superseding the stale suspended turn, so it must
        # be stamped as a replace (see `_stamp_replace_previous`) rather than accumulated onto it.
        rewound = False
        continued_job_cancelled = False

        if pinned := self._get_continuation_model(messages):
            # `_get_continuation_model` only returns a model when the last message is a suspended response.
            suspended_response = messages[-1]
            assert isinstance(suspended_response, ModelResponse)
            prepared_parameters = model_request_parameters
            start = AttemptStart()
            deadline = _pinned_request_deadline(suspended_response, pinned, model_settings, stream=False)
            try:
                with suppress_unsupported_setting_warnings():
                    _, prepared_parameters = pinned.prepare_request(model_settings, model_request_parameters)
                prepared_messages = pinned.prepare_messages(messages, model_request_parameters)
                with enforce_request_deadline(deadline):
                    response = await pinned.request(prepared_messages, model_settings, model_request_parameters)
            except Exception as exc:
                if not await chain.falls_back(exc):
                    self._set_span_attributes(pinned, prepared_parameters)
                    raise
                failure: Exception | ModelResponse = exc
                # The job the suspended response points at may still be running.
                abandoned_response: ModelResponse | None = suspended_response
            else:
                if not await chain.falls_back(response):
                    if response.state == 'suspended':
                        _stamp_continuation(response, pinned)
                        carry_deadline(response, deadline)
                    self._set_span_attributes(pinned, prepared_parameters)
                    return response
                failure = response
                # A rejected response that is still suspended points at a job that is still running;
                # a finished one has no job left to cancel.
                abandoned_response = response if response.state == 'suspended' else None
            # Best-effort cancel the suspended server-side job we're abandoning before rewinding
            # and retrying the chain. `FallbackModel` swallows the failure, so the graph's own
            # cancel path never sees it; without this an OpenAI background job would keep running
            # and billing while the chain issues a duplicate request.
            if abandoned_response is not None:
                await cancel_suspended_job(pinned, abandoned_response)
                # When it's the job of the response this request continues, the continuation loop that a failure of
                # the chain reaches mustn't cancel it again.
                continued_job_cancelled = abandoned_response is suspended_response
            messages = _rewind_messages(messages)
            rewound = True
            self._record_failed_attempt(chain, pinned, failure, start)
            # Fall through to normal chain below

        try:
            return await self._request_chain(chain, messages, model_settings, model_request_parameters, rewound=rewound)
        except BaseException as error:
            if continued_job_cancelled:
                mark_suspended_job_cancelled(error)
            carry_attempts(error, chain.log.attempts)
            raise

    async def _request_chain(
        self,
        chain: FallbackChain,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        *,
        rewound: bool,
    ) -> ModelResponse:
        """Try each model in the chain in turn, after any pinned continuation `request` gave up on."""
        for model in self.models:
            prepared_parameters = model_request_parameters
            start = AttemptStart()
            try:
                with suppress_unsupported_setting_warnings():
                    _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                # Each inner model has its own profile, so re-run `prepare_messages` per model.
                prepared_messages = model.prepare_messages(messages, model_request_parameters)
                deadline = start_request_deadline(model, model_settings, stream=False)
                with enforce_request_deadline(deadline):
                    response = await model.request(prepared_messages, model_settings, model_request_parameters)
            except Exception as exc:
                if await chain.falls_back(exc):
                    self._record_failed_attempt(chain, model, exc, start)
                    continue
                self._set_span_attributes(model, prepared_parameters)
                raise exc

            if await chain.falls_back(response):
                self._record_failed_attempt(chain, model, response, start)
                continue

            response = chain.log.attach(response)

            # After a rewind, the first successful response is fresh generation that supersedes the
            # abandoned suspended turn (whether it ends complete or suspended), so mark it as a replace.
            if rewound:
                _stamp_replace_previous(response)
            if response.state == 'suspended':
                _stamp_continuation(response, model)
                carry_deadline(response, deadline)
            self._set_span_attributes(model, prepared_parameters)
            return response

        chain.exhausted()

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
        and the normal fallback chain is tried.

        A stream is only handed over once it has produced its first event, so an error the stream
        raises before that, such as a provider reporting an overload after acknowledging the request,
        falls back like an error while opening it. Failures after the first event still propagate.

        Response handlers are not applied: the response has already been streamed by the time
        it could be judged, so a `UserWarning` is emitted and the response is accepted as is.
        """
        chain = self._chain()
        # Set once a pinned continuation fails and we rewind to the chain: see the non-streaming `request`.
        rewound = False
        # Set once a chain model's stream is handed over, after which an error is the stream's own.
        handed_over = False

        if pinned := self._get_continuation_model(messages):
            # `_get_continuation_model` only returns a model when the last message is a suspended response.
            suspended_response = messages[-1]
            assert isinstance(suspended_response, ModelResponse)
            async with AsyncExitStack() as stack:
                prepared_parameters = model_request_parameters
                start = AttemptStart()
                deadline = _pinned_request_deadline(suspended_response, pinned, model_settings, stream=True)
                try:
                    with suppress_unsupported_setting_warnings():
                        _, prepared_parameters = pinned.prepare_request(model_settings, model_request_parameters)
                    prepared_messages = pinned.prepare_messages(messages, model_request_parameters)
                    streamed_response = await stack.enter_async_context(
                        request_stream_until_first_event(
                            pinned, deadline, prepared_messages, model_settings, model_request_parameters, run_context
                        )
                    )
                except Exception as exc:
                    if not await chain.falls_back(exc):
                        self._set_span_attributes(pinned, prepared_parameters)
                        raise
                    # Best-effort cancel the suspended server-side job we're abandoning before
                    # rewinding to the chain (see the non-streaming path above); `FallbackModel`
                    # swallows the error, so the graph's own cancel path never sees it.
                    await cancel_suspended_job(pinned, suspended_response)
                    messages = _rewind_messages(messages)
                    rewound = True
                    self._record_failed_attempt(chain, pinned, exc, start)
                    # Fall through to normal chain below
                else:
                    self._set_span_attributes(pinned, prepared_parameters)
                    yield streamed_response
                    # Unlike `request()`, which stamps before returning, the streaming path stamps
                    # after `yield`: the final `state` is only known once the caller has consumed the
                    # stream. Callers must therefore call `get()` after the `async with` exits.
                    if streamed_response.state == 'suspended':
                        _stamp_continuation(streamed_response, pinned)
                        carry_deadline(streamed_response, deadline)
                    return

        try:
            chain.warn_if_streamed_responses_unjudged()

            for model in self.models:
                async with AsyncExitStack() as stack:
                    prepared_parameters = model_request_parameters
                    start = AttemptStart()
                    try:
                        with suppress_unsupported_setting_warnings():
                            _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                        prepared_messages = model.prepare_messages(messages, model_request_parameters)
                        deadline = start_request_deadline(model, model_settings, stream=True)
                        streamed_response = await stack.enter_async_context(
                            request_stream_until_first_event(
                                model,
                                deadline,
                                prepared_messages,
                                model_settings,
                                model_request_parameters,
                                run_context,
                            )
                        )
                    except Exception as exc:
                        if await chain.falls_back(exc):
                            self._record_failed_attempt(chain, model, exc, start)
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
                    chain.log.attach_to_stream(streamed_response)
                    self._set_span_attributes(model, prepared_parameters)
                    handed_over = True
                    yield streamed_response
                    # Stamp after `yield` (see the pinned path above): `state` is only final once the
                    # caller has consumed the stream, so callers must call `get()` after the context exits.
                    if streamed_response.state == 'suspended':
                        _stamp_continuation(streamed_response, model)
                        carry_deadline(streamed_response, deadline)
                    return

            chain.exhausted()
        except BaseException as error:
            if rewound:
                # The job of the response this request continued was cancelled when it was rewound, so the
                # continuation loop the error reaches mustn't cancel it again.
                mark_suspended_job_cancelled(error)
            if not handed_over:
                # Once the stream is handed over, the attempts it carries are counted with it.
                carry_attempts(error, chain.log.attempts)
            raise

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

    def _start_request_deadline(self, model_settings: ModelSettings | None, *, stream: bool) -> RequestDeadline | None:
        # Each model it tries gets fresh deadlines of its own instead.
        return None

    def resolve_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        """A fallback model can't know which model will serve the request, so no retention is claimed."""
        return None

    def _caching_not_enabled(self, model_settings: ModelSettings | None) -> bool:
        # Which model serves the request isn't known here, so nothing is claimed about its caching either.
        return False

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

    def _warn_if_thinking_overridden(self, layers: Sequence[_SettingsLayer]) -> None:
        # Each candidate resolves `thinking` itself, and checks its own `settings` in its `prepare_request`.
        for model in self.models:
            model._warn_if_thinking_overridden(layers)

    def _unified_effort(self, model_settings: ModelSettings | None) -> ThinkingEffort | None:
        # The request is recorded before a candidate is chosen, so it names an effort only if every candidate would
        # send the same one, each with its own `settings` merged in. Otherwise it's recorded as naming none, which
        # sends the effort of whichever candidate serves it as the request-level setting.
        merged = merge_model_settings(self.settings, model_settings)
        efforts: set[ThinkingEffort | None] = {model._unified_effort(merged) for model in self.models}
        return efforts.pop() if len(efforts) == 1 else None

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
        if model_id := continuation_pin(response, key=FALLBACK_MODEL_PIN_KEY):
            return next((m for m in self.models if m.model_id == model_id), None)
        return None

    def _fallback_span(self) -> Span | None:
        """The recording `chat` span instrumentation opened for this request, if any.

        Matching the span's request model to this `FallbackModel` keeps attempts off an unrelated
        ambient span, such as a user's own span around an uninstrumented model.
        """
        span = get_current_span()
        if span.is_recording() and getattr(span, 'attributes', {}).get('gen_ai.request.model') == self.model_name:
            return span
        return None

    def _set_span_attributes(self, model: Model, model_request_parameters: ModelRequestParameters) -> None:
        with suppress(Exception):
            if span := self._fallback_span():
                span_attributes: dict[str, AttributeValue] = {**model_attributes(model)}
                # Only refresh `model_request_parameters` if it was emitted at span open; its absence
                # means `InstrumentationSettings.include_model_request_parameters` is off, and re-adding
                # it here would leak the attribute the setting is meant to suppress.
                if 'model_request_parameters' in getattr(span, 'attributes', {}):
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

    def _chain(self) -> FallbackChain:
        return FallbackChain(self._predicates, owner='FallbackModel')

    def _record_failed_attempt(
        self, chain: FallbackChain, model: Model, failure: Exception | ModelResponse, start: AttemptStart
    ) -> None:
        """Record the attempt this request is falling back from, as a span under `chat` too."""
        # Only under the `chat` span instrumentation opened for this request, and on its tracer provider.
        span = self._fallback_span()
        policy = open_request_policy() if span else None
        chain.log.record(
            model,
            failure,
            start,
            parent=span,
            tracer=policy.tracer if policy else None,
            # Only a `fallback_on` response handler rejects a response here, unlike in the agent's attempt loop.
            rejected_status='Response rejected by a `fallback_on` response handler',
        )


def _pinned_request_deadline(
    suspended_response: ModelResponse, model: Model, model_settings: ModelSettings | None, *, stream: bool
) -> RequestDeadline | None:
    """The deadlines of a continuation segment for the model the suspended response is pinned to.

    What's left of the deadlines that model got when it was picked carries over, so the continuation stays one request
    to it. A nested `FallbackModel` gets none, as it carries over the deadlines of the model it picked itself.
    """
    return resume_deadline(start_request_deadline(model, model_settings, stream=stream), suspended_response)


def _stamp_continuation(response: ModelResponse | StreamedResponse, model: Model) -> None:
    """Stamp the model's identifier into metadata for stateless continuation routing."""
    stamp_continuation_pin(response, model.model_id, key=FALLBACK_MODEL_PIN_KEY)


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
