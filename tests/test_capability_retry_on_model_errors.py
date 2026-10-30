"""The `RetryOnModelErrors` capability.

These are offline tests with `FunctionModel` rather than recorded requests to providers: which errors are retried,
how long the agent waits between attempts and what each attempt records doesn't depend on the provider, and a
scripted model can fail on cue with the normalized errors every model raises. The waits are recorded through
`Agent.using_sleep` rather than slept.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Generator, Sequence
from contextlib import contextmanager
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import BatchModelRequests, Hooks, ModelFallback, RetryOnModelErrors
from pydantic_ai.exceptions import (
    FallbackExceptionGroup,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
    RetryModelRequest,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import ModelRequestAttempt
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.batch import BatchRequest, EncodedBatchRequest
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage, UsageLimits

from ._inline_snapshot import snapshot
from .batch_utils import FAST_BATCH_SETTINGS, SlowBatchTestModel


def _http_error(
    category: type[ModelRateLimitError | ModelQuotaExceededError | ModelUnavailableError | ModelServerError],
    model_name: str,
    *,
    status_code: int = 503,
    headers: dict[str, str] | None = None,
) -> ModelHTTPError:
    return ModelHTTPError.for_category(category, status_code=status_code, model_name=model_name, headers=headers)


def _flaky(name: str, errors: list[Exception], answered: list[str]) -> FunctionModel:
    """A model that raises each of `errors` in turn, then answers."""
    remaining = list(errors)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered.append(name)
        if remaining:
            raise remaining.pop(0)
        return ModelResponse(parts=[TextPart(f'{name} answered')])

    return FunctionModel(respond, model_name=name)


@contextmanager
def _recorded_sleeps() -> Generator[list[float]]:
    """Record the waits the agent makes between attempts instead of sleeping."""
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    with Agent.using_sleep(sleep):
        yield delays


def _kinds(attempts: Sequence[ModelRequestAttempt] | None) -> list[tuple[str, str, str | None, float | None]]:
    return [(a.model_name, a.kind, (a.error or '').split(':')[0] or None, a.retry_after) for a in attempts or []]


async def test_retries_a_transient_error_on_the_same_model():
    """A server error is retried on the same model, which the answering response lists as retried attempts."""
    answered: list[str] = []
    model = _flaky('primary', [_http_error(ModelServerError, 'primary', status_code=500)] * 2, answered)
    agent = Agent(model, capabilities=[RetryOnModelErrors()])

    with _recorded_sleeps() as delays:
        result = await agent.run('x')

    assert result.output == 'primary answered'
    assert answered == ['primary'] * 3
    assert _kinds(result.response.failed_attempts) == snapshot(
        [('primary', 'retry', 'ModelServerError', None), ('primary', 'retry', 'ModelServerError', None)]
    )
    # A step counts once, however many attempts it took.
    assert result.usage.requests == 1
    # Exponential backoff from `initial_delay`, with up to half of each wait taken off as jitter.
    assert len(delays) == 2
    assert 0.5 <= delays[0] <= 1
    assert 1 <= delays[1] <= 2


async def test_gives_up_after_max_attempts():
    """Once `max_attempts` attempts have failed, the last error is raised as is."""
    answered: list[str] = []
    errors: list[Exception] = [ModelConnectionError('primary', f'reset {i}') for i in range(5)]
    agent = Agent(_flaky('primary', errors, answered), capabilities=[RetryOnModelErrors(max_attempts=4)])

    with _recorded_sleeps() as delays, pytest.raises(ModelConnectionError, match='reset 3'):
        await agent.run('x')

    assert answered == ['primary'] * 4
    assert len(delays) == 3


async def test_counts_attempts_at_a_wrapped_model():
    """A model wrapped in another, as durable execution wraps it, is the same model on every attempt."""
    answered: list[str] = []
    errors: list[Exception] = [ModelConnectionError('primary', f'reset {i}') for i in range(3)]
    agent = Agent(WrapperModel(_flaky('primary', errors, answered)), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps(), pytest.raises(ModelConnectionError, match='reset 2'):
        await agent.run('x')
    assert answered == ['primary'] * 3


async def test_one_attempt_never_retries():
    answered: list[str] = []
    agent = Agent(
        _flaky('primary', [ModelTimeoutError('primary', 'timed out')], answered),
        capabilities=[RetryOnModelErrors(max_attempts=1)],
    )
    with _recorded_sleeps() as delays, pytest.raises(ModelTimeoutError):
        await agent.run('x')
    assert (answered, delays) == (['primary'], [])


@pytest.mark.parametrize(
    'error',
    [
        _http_error(ModelRateLimitError, 'primary', status_code=429),
        _http_error(ModelUnavailableError, 'primary'),
        _http_error(ModelOverloadedError, 'primary', status_code=529),
        _http_error(ModelServerError, 'primary', status_code=502),
        ModelConnectionError('primary', 'connection reset', phase='read'),
        ModelTimeoutError('primary', 'read timed out'),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_retries_transient_errors_by_default(error: ModelAPIError):
    answered: list[str] = []
    agent = Agent(_flaky('primary', [error], answered), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps():
        result = await agent.run('x')
    assert result.output == 'primary answered'
    assert answered == ['primary', 'primary']


@pytest.mark.parametrize(
    'error',
    [
        ModelHTTPError.for_category(ModelRequestRejectedError, status_code=400, model_name='primary'),
        ModelHTTPError.for_category(ModelAuthenticationError, status_code=401, model_name='primary'),
        ModelHTTPError.for_category(ModelNotFoundError, status_code=404, model_name='primary'),
        ModelHTTPError.for_category(ModelContextWindowExceededError, status_code=400, model_name='primary'),
        _http_error(ModelQuotaExceededError, 'primary', status_code=402),
        ModelAPIError('primary', 'uncategorized'),
        ValueError('not a model error'),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_does_not_retry_permanent_errors_by_default(error: Exception):
    """A rejected request or exhausted quota fails the same way when sent again, so it's raised at once."""
    answered: list[str] = []
    agent = Agent(_flaky('primary', [error], answered), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps() as delays, pytest.raises(type(error)):
        await agent.run('x')
    assert (answered, delays) == (['primary'], [])


async def test_retry_on_takes_exception_types_and_handlers():
    """`retry_on` takes the forms `fallback_on` does: exception types and sync or async exception handlers."""
    answered: list[str] = []

    async def is_flaky_rejection(error: Exception) -> bool:
        return isinstance(error, ModelRequestRejectedError) and 'flaky' in str(error)

    errors: list[Exception] = [
        ModelHTTPError.for_category(ModelRequestRejectedError, status_code=400, model_name='primary', body='flaky'),
        _http_error(ModelQuotaExceededError, 'primary', status_code=402),
    ]
    agent = Agent(
        _flaky('primary', errors, answered),
        capabilities=[RetryOnModelErrors(retry_on=[ModelQuotaExceededError, is_flaky_rejection])],
    )
    with _recorded_sleeps():
        result = await agent.run('x')
    assert result.output == 'primary answered'
    assert answered == ['primary'] * 3


async def test_does_not_retry_when_the_provider_says_it_would_not_help():
    """An `x-should-retry: false` header stops retries on the same model, whatever `retry_on` says."""
    answered: list[str] = []
    error = _http_error(ModelServerError, 'primary', status_code=500, headers={'x-should-retry': 'false'})
    agent = Agent(_flaky('primary', [error], answered), capabilities=[RetryOnModelErrors(retry_on=(Exception,))])
    with _recorded_sleeps() as delays, pytest.raises(ModelServerError):
        await agent.run('x')
    assert (answered, delays) == (['primary'], [])


async def test_x_should_retry_true_does_not_override_retry_on():
    """The application chooses what is retried: `x-should-retry: true` on an error it didn't choose isn't retried."""
    answered: list[str] = []
    error = ModelHTTPError.for_category(
        ModelRequestRejectedError, status_code=409, model_name='primary', headers={'x-should-retry': 'true'}
    )
    agent = Agent(_flaky('primary', [error], answered), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps(), pytest.raises(ModelRequestRejectedError):
        await agent.run('x')
    assert answered == ['primary']


async def test_waits_as_long_as_the_provider_asks():
    """The error's `retry_after` is waited exactly, and recorded on the attempt."""
    answered: list[str] = []
    errors: list[Exception] = [
        _http_error(ModelRateLimitError, 'primary', status_code=429, headers={'retry-after-ms': '2500'}),
        _http_error(ModelRateLimitError, 'primary', status_code=429, headers={'retry-after': '0'}),
    ]
    agent = Agent(_flaky('primary', errors, answered), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps() as delays:
        result = await agent.run('x')
    assert delays == [2.5, 0]
    assert _kinds(result.response.failed_attempts) == snapshot(
        [('primary', 'retry', 'ModelRateLimitError', 2.5), ('primary', 'retry', 'ModelRateLimitError', 0.0)]
    )


async def test_does_not_wait_longer_than_max_delay():
    """A provider asking for a longer wait than `max_delay` isn't retried, and the backoff is capped at it."""
    answered: list[str] = []
    errors: list[Exception] = [
        ModelConnectionError('primary', 'reset'),
        ModelConnectionError('primary', 'reset'),
        _http_error(ModelRateLimitError, 'primary', status_code=429, headers={'retry-after': '120'}),
    ]
    agent = Agent(
        _flaky('primary', errors, answered),
        capabilities=[RetryOnModelErrors(max_attempts=5, initial_delay=4, max_delay=6)],
    )
    with _recorded_sleeps() as delays, pytest.raises(ModelRateLimitError):
        await agent.run('x')
    assert answered == ['primary'] * 3
    assert len(delays) == 2
    assert 2 <= delays[0] <= 4
    # Doubled to 8, capped at 6.
    assert 3 <= delays[1] <= 6


async def test_does_not_wait_past_the_run_deadline():
    """A wait that would outlast the run's `timeout=` isn't made: the error is raised while the run still has time."""
    answered: list[str] = []
    agent = Agent(
        _flaky('primary', [ModelConnectionError('primary', 'reset')], answered),
        capabilities=[RetryOnModelErrors(initial_delay=30, max_delay=30)],
    )
    with _recorded_sleeps() as delays, pytest.raises(ModelConnectionError):
        await agent.run('x', timeout=10)
    assert (answered, delays) == (['primary'], [])


async def test_backoff_jitter_is_replay_stable():
    """The jitter is derived from the run, so the same run waits the same delays, and other runs differ."""

    async def delays_for(run_id: str) -> list[float]:
        answered: list[str] = []
        errors: list[Exception] = [ModelConnectionError('primary', 'reset')] * 2
        agent = Agent(_flaky('primary', errors, answered), capabilities=[RetryOnModelErrors()])
        with _recorded_sleeps() as delays:
            await agent.run('x', run_id=run_id)
        return delays

    first = await delays_for('run-1')
    assert await delays_for('run-1') == first
    assert await delays_for('run-2') != first


async def test_each_request_step_gets_its_own_budget():
    """The attempt count starts over for each request step of a run."""
    answered: list[str] = []
    steps = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal steps
        answered.append('primary')
        if len(answered) % 2:
            raise ModelConnectionError('primary', 'reset')
        steps += 1
        if steps == 1:
            return ModelResponse(parts=[ToolCallPart('get_value', {})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond, model_name='primary'), capabilities=[RetryOnModelErrors(max_attempts=2)])

    @agent.tool_plain
    def get_value() -> int:
        return 1

    with _recorded_sleeps():
        result = await agent.run('x')
    assert result.output == 'done'
    assert answered == ['primary'] * 4


async def test_two_instances_count_the_same_attempts():
    """Two instances with different `retry_on` each stop at their own `max_attempts` of the same attempts at a model."""
    answered: list[str] = []
    errors: list[Exception] = [
        _http_error(ModelServerError, 'primary', status_code=500),
        _http_error(ModelRateLimitError, 'primary', status_code=429),
        _http_error(ModelServerError, 'primary', status_code=500),
        _http_error(ModelRateLimitError, 'primary', status_code=429),
    ]
    agent = Agent(
        _flaky('primary', errors, answered),
        capabilities=[
            RetryOnModelErrors(retry_on=ModelServerError, max_attempts=3),
            RetryOnModelErrors(retry_on=ModelRateLimitError, max_attempts=3),
        ],
    )
    with _recorded_sleeps(), pytest.raises(ModelServerError):
        await agent.run('x')
    assert answered == ['primary'] * 3


async def test_a_usage_limit_is_not_retried():
    """A token limit the run's own check finds exceeded ends the run, even when `retry_on` matches every error."""
    calls: list[str] = []

    def expensive(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        calls.append('expensive')
        return ModelResponse(
            parts=[TextPart('partial')], state='suspended', usage=RequestUsage(input_tokens=100, output_tokens=1)
        )

    agent = Agent(
        FunctionModel(expensive, model_name='expensive'), capabilities=[RetryOnModelErrors(retry_on=(Exception,))]
    )
    with _recorded_sleeps() as delays, pytest.raises(UsageLimitExceeded):
        await agent.run('x', usage_limits=UsageLimits(input_tokens_limit=5))
    assert (calls, delays) == (['expensive'], [])


# --- Provider SDK retries and request deadlines ---


def _recording(
    name: str,
    errors: list[Exception],
    settings: list[ModelSettings | None],
    own: ModelSettings | None = None,
    model_class: type[FunctionModel] = FunctionModel,
) -> FunctionModel:
    """A model that records the settings each attempt is sent with, raises each of `errors` in turn, then answers."""
    remaining = list(errors)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        settings.append(info.model_settings)
        if remaining:
            raise remaining.pop(0)
        return ModelResponse(parts=[TextPart(f'{name} answered')])

    return model_class(respond, model_name=name, settings=own)


async def test_turns_off_provider_sdk_retries():
    """The capability is the one retry owner: every attempt is sent with `max_retries=0`."""
    settings: list[ModelSettings | None] = []
    agent = Agent(
        _recording('primary', [ModelConnectionError('primary', 'reset')], settings),
        capabilities=[RetryOnModelErrors()],
    )
    with _recorded_sleeps():
        await agent.run('x')
    assert [(s or {}).get('max_retries') for s in settings] == [0, 0]


async def test_an_explicit_max_retries_is_kept():
    """A `max_retries` the application set, on the agent or on a fallback model, is sent as is."""
    primary_settings: list[ModelSettings | None] = []
    backup_settings: list[ModelSettings | None] = []
    primary = _recording('primary', [ModelConnectionError('primary', 'reset')] * 2, primary_settings)
    backup = _recording('backup', [], backup_settings, own={'max_retries': 5})
    agent = Agent(
        primary,
        model_settings={'max_retries': 1},
        capabilities=[RetryOnModelErrors(max_attempts=2), ModelFallback(backup)],
    )
    with _recorded_sleeps():
        await agent.run('x')
    assert [(s or {}).get('max_retries') for s in primary_settings] == [1, 1]
    # The agent's setting is sent to the fallback model too, which applies its own on top, as without the capability.
    assert [(s or {}).get('max_retries') for s in backup_settings] == [1]

    backup_settings.clear()
    primary = _recording('primary', [ModelConnectionError('primary', 'reset')] * 2, [])
    agent = Agent(primary, capabilities=[RetryOnModelErrors(max_attempts=2), ModelFallback(backup)])
    with _recorded_sleeps():
        await agent.run('x')
    # The fallback model's own `max_retries` isn't overridden.
    assert [(s or {}).get('max_retries') for s in backup_settings] == [5]


class _NoSdkRetriesModel(FunctionModel):
    unsupported_settings = frozenset({'max_retries'})


async def test_a_model_without_sdk_retries_is_not_sent_max_retries():
    """A model that ignores `max_retries` isn't sent it, so it doesn't warn about it."""
    answering = _recording('primary', [], [], model_class=_NoSdkRetriesModel)
    with pytest.warns(UserWarning, match='max_retries'):
        await Agent(answering, model_settings={'max_retries': 0}).run('x')

    settings: list[ModelSettings | None] = []
    model = _recording('primary', [ModelConnectionError('primary', 'reset')], settings, model_class=_NoSdkRetriesModel)
    agent = Agent(model, capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps():
        await agent.run('x')
    assert [(s or {}).get('max_retries') for s in settings] == [None, None]


async def test_retries_share_the_request_deadline():
    """A retry gets what's left of the first attempt's `request_timeout`, so the deadline covers every retry."""
    settings: list[ModelSettings | None] = []
    agent = Agent(
        _recording('primary', [ModelConnectionError('primary', 'reset')], settings),
        model_settings={'request_timeout': 60},
        capabilities=[RetryOnModelErrors(initial_delay=0)],
    )
    with _recorded_sleeps():
        await agent.run('x')
    timeouts = [(s or {}).get('request_timeout') for s in settings]
    assert timeouts[0] == 60
    assert timeouts[1] is not None and 0 < timeouts[1] < 60


async def test_does_not_wait_past_the_request_deadline():
    """A wait that would outlast the request's `request_timeout` isn't made, so a `ModelFallback` can move on."""
    answered: list[str] = []
    primary = _flaky('primary', [ModelConnectionError('primary', 'reset')], answered)
    backup = _flaky('backup', [], answered)
    agent = Agent(
        primary,
        model_settings={'request_timeout': 5},
        capabilities=[RetryOnModelErrors(initial_delay=10, max_delay=10), ModelFallback(backup)],
    )
    with _recorded_sleeps() as delays:
        result = await agent.run('x')
    assert result.output == 'backup answered'
    assert (answered, delays) == (['primary', 'backup'], [])


class _FailingOnceBatchModel(SlowBatchTestModel):
    """Fails the request in its first batch with an error batching doesn't submit again, and records what it encodes."""

    def __init__(self) -> None:
        super().__init__(model_name='batched')
        self.encoded_settings: list[ModelSettings | None] = []
        self.failures = 1

    async def batch_encode(self, request: BatchRequest) -> EncodedBatchRequest:
        self.encoded_settings.append(request.model_settings)
        return await super().batch_encode(request)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        if self.failures:
            self.failures -= 1
            raise ModelHTTPError(400, self.model_name, body={'error': 'invalid request'})
        return await super().request(messages, model_settings, model_request_parameters)


async def test_retries_a_batched_request_in_a_new_batch():
    """With `BatchModelRequests`, a retry goes into a new batch on the same model, sent `max_retries=0` like any."""
    model = _FailingOnceBatchModel()
    agent = Agent(
        model,
        capabilities=[RetryOnModelErrors(retry_on=ModelHTTPError), BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )
    with _recorded_sleeps():
        result = await agent.run('x')
    assert result.output == 'success (no tool calls)'
    assert len(model.created) == 2
    assert [(s or {}).get('max_retries') for s in model.encoded_settings] == [0, 0]


# --- With `ModelFallback` ---


@pytest.mark.parametrize('retry_first', [True, False], ids=['retry-listed-first', 'fallback-listed-first'])
async def test_each_fallback_model_gets_its_own_budget(retry_first: bool):
    """Each model in a `ModelFallback` chain is retried before the chain moves on, whichever capability comes first.

    The answering response lists every attempt, retries and fallbacks, in the order they were made.
    """
    answered: list[str] = []
    primary = _flaky('primary', [_http_error(ModelUnavailableError, 'primary')] * 3, answered)
    backup = _flaky('backup', [_http_error(ModelServerError, 'backup', status_code=500)], answered)
    retry = RetryOnModelErrors(max_attempts=2)
    fallback = ModelFallback(backup)
    agent = Agent(primary, capabilities=[retry, fallback] if retry_first else [fallback, retry])

    with _recorded_sleeps() as delays:
        result = await agent.run('x')

    assert result.output == 'backup answered'
    assert answered == ['primary', 'primary', 'backup', 'backup']
    assert _kinds(result.response.failed_attempts) == snapshot(
        [
            ('primary', 'retry', 'ModelUnavailableError', None),
            ('primary', 'fallback', 'ModelUnavailableError', None),
            ('backup', 'retry', 'ModelServerError', None),
        ]
    )
    # No wait before falling back to another model.
    assert len(delays) == 2


async def test_an_error_not_retried_falls_back_at_once():
    """An error `retry_on` doesn't cover, or one the provider says not to retry, moves on to the next model."""
    answered: list[str] = []
    primary = _flaky('primary', [_http_error(ModelQuotaExceededError, 'primary', status_code=402)], answered)
    backup = _flaky('backup', [], answered)
    agent = Agent(primary, capabilities=[ModelFallback(backup), RetryOnModelErrors()])
    with _recorded_sleeps() as delays:
        result = await agent.run('x')
    assert result.output == 'backup answered'
    assert (answered, delays) == (['primary', 'backup'], [])
    assert _kinds(result.response.failed_attempts) == snapshot(
        [('primary', 'fallback', 'ModelQuotaExceededError', None)]
    )


async def test_an_exhausted_chain_lists_every_attempt():
    answered: list[str] = []
    primary = _flaky('primary', [ModelConnectionError('primary', 'reset')] * 2, answered)
    backup = _flaky('backup', [ModelConnectionError('backup', 'reset')] * 2, answered)
    agent = Agent(primary, capabilities=[ModelFallback(backup), RetryOnModelErrors(max_attempts=2)])
    with _recorded_sleeps(), pytest.raises(FallbackExceptionGroup) as exc_info:
        await agent.run('x')
    assert _kinds(exc_info.value.attempts) == snapshot(
        [
            ('primary', 'retry', 'ModelConnectionError', None),
            ('primary', 'fallback', 'ModelConnectionError', None),
            ('backup', 'retry', 'ModelConnectionError', None),
            ('backup', 'fallback', 'ModelConnectionError', None),
        ]
    )


async def test_a_hook_retrying_the_same_model_records_a_retry():
    """Any hook raising `RetryModelRequest` without a model records the attempt it moved on from as a retry."""
    answered: list[str] = []

    async def retry_once(
        ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        if request_context.attempt == 1:
            raise RetryModelRequest()
        raise error  # pragma: no cover

    hooks = Hooks()
    hooks.on.model_request_error(retry_once)
    agent = Agent(_flaky('primary', [ModelAPIError('primary', 'boom')], answered), capabilities=[hooks])
    result = await agent.run('x')
    assert _kinds(result.response.failed_attempts) == snapshot([('primary', 'retry', 'ModelAPIError', None)])


# --- Response handlers ---


def _answers(name: str, texts: list[str], answered: list[str]) -> FunctionModel:
    remaining = list(texts)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered.append(name)
        return ModelResponse(parts=[TextPart(remaining.pop(0))], usage=RequestUsage(input_tokens=10, output_tokens=5))

    return FunctionModel(respond, model_name=name)


def _is_refusal(response: ModelResponse) -> bool:
    return response.text == 'nope'


async def test_retries_a_rejected_response():
    """A response handler in `retry_on` rejects a response, and the same model is attempted again."""
    answered: list[str] = []
    agent = Agent(
        _answers('primary', ['nope', 'hello'], answered), capabilities=[RetryOnModelErrors(retry_on=_is_refusal)]
    )
    with _recorded_sleeps() as delays:
        result = await agent.run('x')
    assert result.output == 'hello'
    assert len(delays) == 1
    attempts = result.response.failed_attempts or []
    assert [(a.model_name, a.outcome, a.kind, a.usage) for a in attempts] == snapshot(
        [('primary', 'rejected', 'retry', RequestUsage(input_tokens=10, output_tokens=5))]
    )
    # The rejected response was billed, so it counts towards the run's usage.
    assert (result.usage.input_tokens, result.usage.output_tokens, result.usage.requests) == (20, 10, 1)


async def test_accepts_the_last_rejected_response():
    """Once its attempts are used up, a rejected response is accepted, so an outer `ModelFallback` can judge it."""
    answered: list[str] = []
    agent = Agent(
        _answers('primary', ['nope', 'nope'], answered),
        capabilities=[RetryOnModelErrors(retry_on=[ModelAPIError, _is_refusal], max_attempts=2)],
    )
    with _recorded_sleeps():
        result = await agent.run('x')
    assert result.output == 'nope'
    assert answered == ['primary', 'primary']


async def test_a_rejected_response_falls_back_once_retries_run_out():
    answered: list[str] = []
    agent = Agent(
        _answers('primary', ['nope', 'nope'], answered),
        capabilities=[
            RetryOnModelErrors(retry_on=_is_refusal, max_attempts=2),
            ModelFallback(_answers('backup', ['from backup'], answered), fallback_on=_is_refusal),
        ],
    )
    with _recorded_sleeps():
        result = await agent.run('x')
    assert result.output == 'from backup'
    assert [(a.model_name, a.kind) for a in result.response.failed_attempts or []] == snapshot(
        [('primary', 'retry'), ('primary', 'fallback')]
    )


async def test_a_rejected_response_is_not_retried_past_the_run_deadline():
    answered: list[str] = []
    agent = Agent(
        _answers('primary', ['nope'], answered),
        capabilities=[RetryOnModelErrors(retry_on=_is_refusal, initial_delay=30, max_delay=30)],
    )
    with _recorded_sleeps() as delays:
        result = await agent.run('x', timeout=10)
    assert (result.output, answered, delays) == ('nope', ['primary'], [])


async def test_response_handlers_are_not_applied_to_a_streamed_response():
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'nope'

    agent = Agent(FunctionModel(stream_function=stream), capabilities=[RetryOnModelErrors(retry_on=_is_refusal)])
    with pytest.warns(UserWarning, match='`RetryOnModelErrors` response handlers in `retry_on` are not applied'):
        async with agent.run_stream('x') as result:
            assert await result.get_output() == 'nope'


# --- Streaming ---


def _stream_model(fail_after_output: bool, answered: list[str]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        answered.append('primary')
        if len(answered) == 1:
            if fail_after_output:
                yield 'partial'
            raise _http_error(ModelOverloadedError, 'primary', status_code=529)
        yield 'streamed'

    return FunctionModel(stream_function=stream, model_name='primary')


async def test_retries_a_stream_that_fails_before_any_output():
    answered: list[str] = []
    agent = Agent(_stream_model(False, answered), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps():
        async with agent.run_stream('x') as result:
            assert await result.get_output() == 'streamed'
    assert answered == ['primary', 'primary']


async def test_does_not_retry_a_stream_after_output_reached_the_consumer():
    answered: list[str] = []
    agent = Agent(_stream_model(True, answered), capabilities=[RetryOnModelErrors()])
    with _recorded_sleeps(), pytest.raises(ModelOverloadedError):
        async with agent.run_stream('x') as result:
            await result.get_output()
    assert answered == ['primary']


# --- Configuration ---


@pytest.mark.parametrize(
    'kwargs, message',
    [
        ({'max_attempts': 0}, '`RetryOnModelErrors` `max_attempts` must be at least 1, got 0.'),
        ({'initial_delay': -1}, '`RetryOnModelErrors` `initial_delay` and `max_delay` must not be negative.'),
        ({'max_delay': -1}, '`RetryOnModelErrors` `initial_delay` and `max_delay` must not be negative.'),
        (
            {'retry_on': []},
            '`RetryOnModelErrors` created with an empty `retry_on`: all exceptions will propagate and all responses '
            'will be accepted. Use `retry_on=(ModelRateLimitError, ModelUnavailableError, ModelServerError, '
            'ModelConnectionError)` for the default behavior.',
        ),
    ],
)
def test_invalid_configuration(kwargs: dict[str, object], message: str):
    with pytest.raises(UserError) as exc_info:
        RetryOnModelErrors(**kwargs)  # pyright: ignore[reportArgumentType]
    assert str(exc_info.value) == message


async def test_from_spec_defaults_to_transient_errors():
    answered: list[str] = []
    errors: list[Exception] = [
        _http_error(ModelRateLimitError, 'primary', status_code=429),
        ModelHTTPError.for_category(ModelRequestRejectedError, status_code=400, model_name='primary'),
    ]
    agent = Agent.from_spec({'capabilities': ['RetryOnModelErrors']}, model=_flaky('primary', errors, answered))
    with _recorded_sleeps(), pytest.raises(ModelRequestRejectedError):
        await agent.run('x')
    assert answered == ['primary', 'primary']


async def test_from_spec_with_arguments():
    answered: list[str] = []
    spec = {
        'capabilities': [
            {'RetryOnModelErrors': {'retry_on': 'ModelRequestRejectedError', 'max_attempts': 3, 'initial_delay': 0}}
        ]
    }
    errors: list[Exception] = [
        ModelHTTPError.for_category(ModelRequestRejectedError, status_code=400, model_name='primary')
    ] * 2
    agent = Agent.from_spec(spec, model=_flaky('primary', errors, answered))
    with _recorded_sleeps() as delays:
        result = await agent.run('x')
    assert result.output == 'primary answered'
    assert delays == [0, 0]
    assert AgentSpec.model_validate(spec).capabilities[0].name == 'RetryOnModelErrors'


def test_from_spec_rejects_unknown_error_names():
    with pytest.raises(
        ValueError,
        match="`RetryOnModelErrors` `retry_on` in an agent spec takes names of `ModelAPIError` classes, got \\['UserError'\\]",
    ):
        Agent.from_spec({'capabilities': [{'RetryOnModelErrors': {'retry_on': ['ModelServerError', 'UserError']}}]})
