"""Tests for sync function online evaluation dispatched via background threads.

These tests are plain (non-async) functions — no running event loop — so they
exercise the background thread dispatch path in _wrap_sync that spawns a new
thread with asyncio.run(). This is the path used when sync decorated functions
are called from non-async code (CLI scripts, synchronous web frameworks, etc.).
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators import Evaluator, EvaluatorContext
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.online import (
        OnlineEvalConfig,
        OnlineEvaluator,
        _join_background_threads,  # pyright: ignore[reportPrivateUsage]
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')


if TYPE_CHECKING or imports_successful():
    from collections.abc import Sequence

    from pydantic_evals.evaluators import EvaluationResult, EvaluatorFailure

    @dataclass
    class AlwaysTrue(Evaluator):
        def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput:
            return True

    class SyncCollector:
        """Collects sink submissions — async callback that runs in background thread's event loop."""

        def __init__(self) -> None:
            self.calls: list[
                tuple[list[EvaluationResult[Any]], list[EvaluatorFailure], EvaluatorContext[Any, Any, Any]]
            ] = []

        async def __call__(
            self,
            results: Sequence[EvaluationResult[Any]],
            failures: Sequence[EvaluatorFailure],
            context: EvaluatorContext[Any, Any, Any],
        ) -> None:
            self.calls.append((list(results), list(failures), context))


def test_sync_background_thread_dispatch():
    """Sync decorated function dispatches evaluators via background thread when no event loop is running."""
    collector = SyncCollector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x * 2

    result = my_func(21)
    assert result == 42

    _join_background_threads()

    assert len(collector.calls) == 1
    results, _, ctx = collector.calls[0]
    assert len(results) == 1
    assert results[0].value is True
    assert ctx.output == 42
    assert ctx.inputs == {'x': 21}


def test_sync_background_thread_gate():
    """Sync gates work via background thread dispatch."""
    collector = SyncCollector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=lambda ctx: ctx.output > 10))
    def my_func(x: int) -> int:
        return x

    my_func(5)  # gate blocks
    _join_background_threads()
    assert len(collector.calls) == 0

    my_func(20)  # gate allows
    _join_background_threads()
    assert len(collector.calls) == 1


def test_sync_background_thread_async_gate():
    """Async gates work via background thread dispatch (thread runs its own event loop)."""
    collector = SyncCollector()
    config = OnlineEvalConfig(default_sink=collector)

    async def async_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        return ctx.output > 10

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=async_gate))
    def my_func(x: int) -> int:
        return x

    my_func(5)  # gate blocks
    _join_background_threads()
    assert len(collector.calls) == 0

    my_func(20)  # gate allows
    _join_background_threads()
    assert len(collector.calls) == 1


def test_sync_background_thread_disabled():
    """Disabled config doesn't dispatch any threads."""
    collector = SyncCollector()
    config = OnlineEvalConfig(default_sink=collector, enabled=False)

    @config.evaluate(AlwaysTrue())
    def my_func(x: int) -> int:
        return x

    result = my_func(42)
    assert result == 42

    _join_background_threads()
    assert len(collector.calls) == 0


def test_sync_background_thread_sample_rate_zero():
    """sample_rate=0 doesn't dispatch any threads."""
    collector = SyncCollector()
    config = OnlineEvalConfig(default_sink=collector)

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), sample_rate=0.0))
    def my_func(x: int) -> int:
        return x

    result = my_func(42)
    assert result == 42

    _join_background_threads()
    assert len(collector.calls) == 0


def test_sync_background_thread_gate_exception():
    """Gate exception in background thread skips evaluator gracefully."""
    collector = SyncCollector()
    config = OnlineEvalConfig(default_sink=collector)

    def bad_gate(ctx: EvaluatorContext[Any, Any, Any]) -> bool:
        raise ValueError('gate error')

    @config.evaluate(OnlineEvaluator(evaluator=AlwaysTrue(), gate=bad_gate))
    def my_func(x: int) -> int:
        return x

    result = my_func(42)
    assert result == 42

    _join_background_threads()
    assert len(collector.calls) == 0
