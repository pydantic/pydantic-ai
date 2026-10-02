"""Tests for the RetryPolicy capability."""

from __future__ import annotations

import math
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.retry_policy import RetryPolicy


class TestValidation:
    def test_defaults(self) -> None:
        policy = RetryPolicy()
        assert policy.max_retries == 3
        assert policy.backoff_factor == 0.5
        assert policy.max_backoff == 30.0
        assert policy.retryable_status_codes == (429, 500, 502, 503, 504)
        assert policy.retryable_exceptions == (TimeoutError, ConnectionError)
        assert policy.tool_overrides == {}
        assert policy.allow_idempotent_retries is False
        assert policy.idempotent_tools == frozenset()

    def test_custom_values(self) -> None:
        policy = RetryPolicy(
            max_retries=5,
            backoff_factor=1.0,
            max_backoff=60.0,
            retryable_status_codes=(429, 503),
            retryable_exceptions=(TimeoutError,),
        )
        assert policy.max_retries == 5
        assert policy.backoff_factor == 1.0
        assert policy.max_backoff == 60.0
        assert policy.retryable_status_codes == (429, 503)
        assert policy.retryable_exceptions == (TimeoutError,)

    @pytest.mark.parametrize('value', [-1, 1.5, True])
    def test_non_int_or_negative_max_retries_raises(self, value: Any) -> None:
        with pytest.raises(ValueError, match='max_retries must be an int >= 0'):
            RetryPolicy(max_retries=value)

    @pytest.mark.parametrize('field_name', ['backoff_factor', 'max_backoff'])
    @pytest.mark.parametrize('value', [0, -1.0, float('inf'), float('-inf'), float('nan')])
    def test_nonfinite_backoff_raises(self, field_name: str, value: float) -> None:
        kwargs: dict[str, Any] = {field_name: value}
        with pytest.raises(ValueError, match='must be a finite number > 0'):
            RetryPolicy(**kwargs)

    def test_tool_override_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match='unknown keys'):
            RetryPolicy(tool_overrides={'tool': {'max_retry': 0}})

    @pytest.mark.parametrize('value', [-1, 1.5, True])
    def test_tool_override_bad_max_retries_raises(self, value: Any) -> None:
        with pytest.raises(ValueError, match=r"tool_overrides\['tool'\]\['max_retries'\] must be an int >= 0"):
            RetryPolicy(tool_overrides={'tool': {'max_retries': value}})

    @pytest.mark.parametrize('field_name', ['backoff_factor', 'max_backoff'])
    @pytest.mark.parametrize('value', [0, float('inf'), float('nan')])
    def test_tool_override_nonfinite_backoff_raises(self, field_name: str, value: float) -> None:
        with pytest.raises(ValueError, match='must be a finite number > 0'):
            RetryPolicy(tool_overrides={'tool': {field_name: value}})

    def test_tool_override_non_tuple_retryable_raises(self) -> None:
        with pytest.raises(ValueError, match='must be a tuple'):
            RetryPolicy(tool_overrides={'tool': {'retryable_exceptions': [TimeoutError]}})

    def test_tool_override_non_callable_callback_raises(self) -> None:
        with pytest.raises(ValueError, match='must be callable'):
            RetryPolicy(tool_overrides={'tool': {'on_retry': 'nope'}})

    def test_tool_override_non_bool_idempotent_raises(self) -> None:
        with pytest.raises(ValueError, match='must be a bool'):
            RetryPolicy(tool_overrides={'tool': {'idempotent': 'yes'}})

    def test_max_backoff_below_backoff_factor_raises(self) -> None:
        with pytest.raises(ValueError, match=r'max_backoff.*>=.*backoff_factor'):
            RetryPolicy(backoff_factor=1.0, max_backoff=0.5)

    @pytest.mark.parametrize('value', ['false', 'true', 1, None])
    def test_non_bool_gate_raises(self, value: Any) -> None:
        with pytest.raises(ValueError, match='allow_idempotent_retries must be a bool'):
            kwargs: dict[str, Any] = {'allow_idempotent_retries': value}
            RetryPolicy(**kwargs)

    def test_non_exception_member_in_retryable_exceptions_raises(self) -> None:
        with pytest.raises(ValueError, match='retryable_exceptions must be a tuple of Exception types'):
            kwargs: dict[str, Any] = {'retryable_exceptions': (TimeoutError, 'TimeoutError')}
            RetryPolicy(**kwargs)

    def test_non_int_member_in_retryable_status_codes_raises(self) -> None:
        with pytest.raises(ValueError, match='retryable_status_codes must be a tuple of ints'):
            kwargs: dict[str, Any] = {'retryable_status_codes': ('429',)}
            RetryPolicy(**kwargs)

    def test_tool_override_non_exception_member_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"tool_overrides\['tool'\]\['retryable_exceptions'\] must be a tuple of Exception types"
        ):
            RetryPolicy(tool_overrides={'tool': {'retryable_exceptions': (ValueError, 123)}})

    def test_tool_override_non_int_status_code_raises(self) -> None:
        with pytest.raises(
            ValueError, match=r"tool_overrides\['tool'\]\['retryable_status_codes'\] must be a tuple of ints"
        ):
            RetryPolicy(tool_overrides={'tool': {'retryable_status_codes': (429.0,)}})


class TestResolution:
    def test_get_tool_config_default(self) -> None:
        assert RetryPolicy().get_tool_config('unknown_tool') == {}

    def test_get_tool_config_override(self) -> None:
        policy = RetryPolicy(tool_overrides={'web_search': {'max_retries': 2}})
        assert policy.get_tool_config('web_search')['max_retries'] == 2

    def test_get_max_retries_default(self) -> None:
        assert RetryPolicy(max_retries=5).get_max_retries('any_tool') == 5

    def test_get_max_retries_override(self) -> None:
        policy = RetryPolicy(max_retries=5, tool_overrides={'web_search': {'max_retries': 2}})
        assert policy.get_max_retries('web_search') == 2
        assert policy.get_max_retries('other_tool') == 5

    def test_per_tool_retryable_override(self) -> None:
        policy = RetryPolicy(tool_overrides={'safe_tool': {'retryable_exceptions': ()}})
        assert policy.should_retry(TimeoutError('timeout'), 'tool') is True
        assert policy.should_retry(TimeoutError('timeout'), 'safe_tool') is False


class TestShouldRetry:
    def test_timeout_error(self) -> None:
        assert RetryPolicy().should_retry(TimeoutError('timeout'), 'tool') is True

    def test_connection_error(self) -> None:
        assert RetryPolicy().should_retry(ConnectionError('connection'), 'tool') is True

    def test_permanent_os_error_is_not_retryable(self) -> None:
        assert RetryPolicy().should_retry(FileNotFoundError('gone'), 'tool') is False

    def test_non_retryable(self) -> None:
        assert RetryPolicy().should_retry(ValueError('bad value'), 'tool') is False

    def test_custom_exception(self) -> None:
        class MyError(Exception):
            pass

        policy = RetryPolicy(retryable_exceptions=(MyError,))
        assert policy.should_retry(MyError('custom'), 'tool') is True
        assert policy.should_retry(TimeoutError('timeout'), 'tool') is False

    @pytest.mark.parametrize('status', [429, 400, None])
    @pytest.mark.parametrize('nested', [False, True])
    def test_http_error_status(self, status: int | None, nested: bool) -> None:
        class Response:
            status_code = status

        class HttpError(Exception):
            status_code = None if nested else status
            response = Response() if nested else None

        assert RetryPolicy().should_retry(HttpError(), 'tool') is (status == 429)

    @pytest.mark.parametrize('error_type', ['rate_limit', 'timeout', 'server_error', 'invalid'])
    def test_provider_error_type(self, error_type: str) -> None:
        class ProviderError(Exception):
            def __init__(self) -> None:
                self.error_type = error_type

        assert RetryPolicy().should_retry(ProviderError(), 'tool') is (error_type != 'invalid')


class TestBackoff:
    def test_base_within_jitter_bounds(self) -> None:
        delay = RetryPolicy(backoff_factor=1.0).calculate_delay(0, 'tool')
        assert 0.75 <= delay <= 1.25

    def test_exponential_growth_within_jitter_bounds(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=100.0)
        for _ in range(10):
            delays = [policy.calculate_delay(attempt, 'tool') for attempt in range(3)]
            assert delays[1] > delays[0] * 0.8
            assert delays[2] > delays[1] * 0.8

    def test_capped_at_max_backoff_with_jitter_clamped_back_down(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=5.0)
        for _ in range(50):
            assert policy.calculate_delay(10, 'tool') <= 5.0

    def test_tool_override(self) -> None:
        policy = RetryPolicy(backoff_factor=0.5, tool_overrides={'fast_tool': {'backoff_factor': 0.1}})
        assert policy.calculate_delay(0, 'fast_tool') < policy.calculate_delay(0, 'tool')

    def test_large_attempt_no_overflow(self) -> None:
        delay = RetryPolicy().calculate_delay(10_000, 'tool')
        assert 0.01 <= delay <= 30.0

    def test_tiny_backoff_factor_floors_at_minimum(self) -> None:
        delay = RetryPolicy(backoff_factor=5e-324, max_backoff=30.0).calculate_delay(0, 'tool')
        assert delay == 0.01

    def test_upper_cap_edge_never_exceeds_max_backoff(self) -> None:
        policy = RetryPolicy(backoff_factor=5e-324, max_backoff=30.0)
        for _ in range(50):
            assert policy.calculate_delay(1024, 'tool') <= 30.0

    def test_log_space_boundary_is_safe(self) -> None:
        """An attempt exactly at the log-space boundary takes the cap branch, so `ldexp` never runs."""
        backoff_factor, max_backoff, attempt = 1.0, 8.0, 3
        policy = RetryPolicy(backoff_factor=backoff_factor, max_backoff=max_backoff)
        assert math.log2(backoff_factor) + attempt == math.log2(max_backoff)
        assert policy.calculate_delay(attempt, 'tool') <= max_backoff

    def test_just_below_boundary_scales_finite(self) -> None:
        policy = RetryPolicy(backoff_factor=1.0, max_backoff=1e308)
        assert policy.calculate_delay(1023, 'tool') <= 1e308


class TestCallbacks:
    def test_per_tool_on_retry_override(self) -> None:
        calls: list[str] = []

        def special(tool: str, attempt: int, exc: Exception) -> None:
            calls.append('special')

        policy = RetryPolicy(
            on_retry=lambda tool, attempt, exc: calls.append('default'),
            tool_overrides={'special': {'on_retry': special}},
        )
        default_callback = policy._get_on_retry('other_tool')  # pyright: ignore[reportPrivateUsage]
        special_callback = policy._get_on_retry('special')  # pyright: ignore[reportPrivateUsage]
        assert default_callback is not None
        assert special_callback is not None
        default_callback('other_tool', 1, Exception())
        special_callback('special', 1, Exception())
        assert calls == ['default', 'special']

    def test_per_tool_on_failure_override(self) -> None:
        calls: list[str] = []

        def special(tool: str, exc: Exception) -> None:
            calls.append('special')

        policy = RetryPolicy(
            on_failure=lambda tool, exc: calls.append('default'),
            tool_overrides={'special': {'on_failure': special}},
        )
        default_callback = policy._get_on_failure('other_tool')  # pyright: ignore[reportPrivateUsage]
        special_callback = policy._get_on_failure('special')  # pyright: ignore[reportPrivateUsage]
        assert default_callback is not None
        assert special_callback is not None
        default_callback('other_tool', Exception())
        special_callback('special', Exception())
        assert calls == ['default', 'special']


class TestIdempotency:
    def test_default_not_idempotent(self) -> None:
        assert RetryPolicy()._is_idempotent('any_tool') is False  # pyright: ignore[reportPrivateUsage]

    def test_marked_tools_with_gate_open(self) -> None:
        policy = RetryPolicy(allow_idempotent_retries=True, idempotent_tools=frozenset({'safe_read'}))
        assert policy._is_idempotent('safe_read') is True  # pyright: ignore[reportPrivateUsage]
        assert policy._is_idempotent('unsafe_write') is False  # pyright: ignore[reportPrivateUsage]

    def test_gate_closed_wins(self) -> None:
        policy = RetryPolicy(allow_idempotent_retries=False, idempotent_tools=frozenset({'safe_read'}))
        assert policy._is_idempotent('safe_read') is False  # pyright: ignore[reportPrivateUsage]

    def test_per_tool_idempotent_override(self) -> None:
        policy = RetryPolicy(allow_idempotent_retries=True, tool_overrides={'custom': {'idempotent': True}})
        assert policy._is_idempotent('custom') is True  # pyright: ignore[reportPrivateUsage]
        assert policy._is_idempotent('other') is False  # pyright: ignore[reportPrivateUsage]


class TestAgentIntegration:
    @pytest.mark.parametrize('with_callbacks', [False, True])
    @pytest.mark.parametrize('outcome', ['success', 'exhausted', 'nonretryable', 'unsafe'])
    async def test_tool_execution(self, with_callbacks: bool, outcome: str) -> None:
        """RetryPolicy through a real Agent: attempt counts, callbacks, and surfaced exceptions."""
        retries: list[int] = []
        failures: list[Exception] = []
        attempts = 0

        def on_retry(tool: str, attempt: int, exc: Exception) -> None:
            retries.append(attempt)

        def on_failure(tool: str, exc: Exception) -> None:
            failures.append(exc)

        policy = RetryPolicy(
            max_retries=1,
            backoff_factor=0.001,
            max_backoff=0.001,
            allow_idempotent_retries=outcome != 'unsafe',
            idempotent_tools=frozenset({'lookup'}),
            on_retry=on_retry if with_callbacks else None,
            on_failure=on_failure if with_callbacks else None,
        )
        agent = Agent(TestModel(), capabilities=[policy])

        @agent.tool_plain
        def lookup() -> str:
            nonlocal attempts
            attempts += 1
            if outcome == 'nonretryable':
                raise ValueError('invalid')
            if outcome == 'success' and attempts == 2:
                return 'found'
            raise TimeoutError('timeout')

        if outcome == 'success':
            await agent.run('lookup')
        else:
            with pytest.raises(ValueError if outcome == 'nonretryable' else TimeoutError):
                await agent.run('lookup')

        # `success` and `exhausted` retry once and run the tool twice; `nonretryable` and `unsafe`
        # surface on the first attempt. A non-retryable ValueError propagates unmodified.
        assert attempts == (2 if outcome in ('success', 'exhausted') else 1)
        assert retries == ([1] if with_callbacks and outcome in ('success', 'exhausted') else [])
        assert len(failures) == int(with_callbacks and outcome in ('exhausted', 'unsafe'))

    async def test_unsafe_tool_handler_runs_once(self) -> None:
        """A transient failure on a tool not marked idempotent surfaces immediately."""
        call_count = 0

        async def handler(args: dict[str, Any]) -> str:
            nonlocal call_count
            call_count += 1
            raise TimeoutError('timeout')

        policy = RetryPolicy(max_retries=3)
        assert policy._is_idempotent('write_tool') is False  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(TimeoutError):
            await policy.wrap_tool_execute(
                ctx=None,  # pyright: ignore[reportArgumentType]
                call=type('Call', (), {'tool_name': 'write_tool'})(),  # pyright: ignore[reportArgumentType]
                tool_def=None,  # pyright: ignore[reportArgumentType]
                args={},
                handler=handler,
            )
        assert call_count == 1

    def test_construction_failures_raise_before_any_run(self) -> None:
        with pytest.raises(ValueError, match='max_retries'):
            RetryPolicy(max_retries=-1)
        with pytest.raises(ValueError, match='unknown keys'):
            RetryPolicy(tool_overrides={'tool': {'max_retry': 1}})
