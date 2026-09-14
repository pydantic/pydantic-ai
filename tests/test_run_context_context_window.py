"""Tests for `RunContext.context_window_used`: the fraction of the model's context window in use.

These use `FunctionModel` with explicit `usage` on each response so the expected ratio is exact:
the property reads the most recent `ModelResponse`'s `total_tokens` against the model profile's
`context_window`.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models import AbstractModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage, RunUsage


def _model_with_usage(context_window: int | None) -> FunctionModel:
    """A model that first calls the `record` tool, then finishes; both responses carry explicit usage."""

    def func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            # Per the `RequestUsage` convention, `input_tokens` already includes the cache tokens.
            return ModelResponse(
                parts=[ToolCallPart(tool_name='record', args={})],
                usage=RequestUsage(input_tokens=175, cache_read_tokens=50, cache_write_tokens=25, output_tokens=25),
            )
        return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=300, output_tokens=100))

    profile = ModelProfile(context_window=context_window) if context_window is not None else None
    return FunctionModel(func, profile=profile)


def _recording_agent(model: FunctionModel, seen: list[float | None]) -> Agent[Any]:
    agent = Agent(model)

    @agent.tool
    def record(ctx: RunContext[Any]) -> str:
        seen.append(ctx.context_window_used)
        return 'done'

    return agent


def test_context_window_used_in_tool():
    """A tool sees the last response's `total_tokens` (cached input included) over the profile's window."""
    seen: list[float | None] = []
    agent = _recording_agent(_model_with_usage(context_window=1000), seen)

    agent.run_sync('hello')

    # (175 input, of which 75 are cache read/write + 25 output) / 1000
    assert seen == [0.2]


@pytest.mark.parametrize('context_window', [None, 0])
def test_context_window_used_none_when_window_unavailable(context_window: int | None):
    """`None` when the context window is unknown or non-positive."""
    seen: list[float | None] = []
    agent = _recording_agent(_model_with_usage(context_window=context_window), seen)

    agent.run_sync('hello')

    assert seen == [None]


def test_context_window_used_none_before_first_response():
    """`None` before any model response exists, e.g. in an instructions function."""
    seen: list[float | None] = []
    agent = _recording_agent(_model_with_usage(context_window=1000), seen)

    @agent.instructions
    def instructions(ctx: RunContext[Any]) -> str:
        seen.append(ctx.context_window_used)
        return 'Be nice.'

    agent.run_sync('hello')

    assert seen[0] is None


def test_context_window_used_none_when_usage_not_reported():
    """`None`, not a misleading `0.0`, when the provider reported no usage on the response."""
    ctx = RunContext(
        deps=None,
        model=FunctionModel(lambda messages, info: ModelResponse(parts=[]), profile=ModelProfile(context_window=1000)),
        usage=RunUsage(),
        messages=[ModelResponse(parts=[TextPart('done')], usage=RequestUsage())],
    )
    assert ctx.context_window_used is None


def test_context_window_used_uses_latest_response():
    """The most recent response's usage determines the ratio."""
    ctx = RunContext(
        deps=None,
        model=FunctionModel(lambda messages, info: ModelResponse(parts=[]), profile=ModelProfile(context_window=1000)),
        usage=RunUsage(),
        messages=[
            ModelResponse(parts=[TextPart('first')], usage=RequestUsage(input_tokens=100, output_tokens=50)),
            ModelResponse(parts=[TextPart('second')], usage=RequestUsage(input_tokens=300, output_tokens=100)),
        ],
    )
    assert ctx.context_window_used == 0.4


def test_context_window_used_measures_fallback_model_against_smallest_candidate_window():
    """A `FallbackModel` has no profile, but its `context_window` is the smallest known candidate window."""
    model = FallbackModel(
        FunctionModel(lambda messages, info: ModelResponse(parts=[]), profile=ModelProfile(context_window=1000)),
        FunctionModel(lambda messages, info: ModelResponse(parts=[]), profile=ModelProfile(context_window=500)),
        TestModel(),  # window unknown: doesn't constrain the result
    )
    assert model.context_window == 500

    ctx = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        messages=[ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=150, output_tokens=50))],
    )
    assert ctx.context_window_used == 0.4


class _WindowlessModel(AbstractModel):
    """A model outside the request-response hierarchy that doesn't override `context_window`."""

    @property
    def model_name(self) -> str:
        return 'windowless'

    @property
    def system(self) -> str:
        return 'test'


def test_context_window_used_none_for_model_without_context_window():
    """`None` for any `AbstractModel` whose `context_window` is unknown, e.g. a realtime model."""
    model = _WindowlessModel()
    assert model.model_id == 'test:windowless'
    assert model.context_window is None
    ctx = RunContext(
        deps=None,
        model=model,
        usage=RunUsage(),
        messages=[ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=150, output_tokens=50))],
    )
    assert ctx.context_window_used is None
