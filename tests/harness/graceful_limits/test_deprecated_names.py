"""The old `WrapUpAtLimits` name still imports and works, with a `HarnessDeprecationWarning`, and old specs still load."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.graceful_limits import (
    DEFAULT_WRAP_UP_PROMPT,
    GracefulLimits,
    WrapUpPart,
    wrapped_up_by,
)


def test_shim_warns_and_aliases_old_names() -> None:
    # Import once quietly so the deprecation fires on the reload we assert on, even if
    # another test already imported the shim.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shim = importlib.import_module('pydantic_ai_harness.wrap_up_at_limits')

    with pytest.warns(HarnessDeprecationWarning, match=r'renamed to `pydantic_ai_harness\.graceful_limits`'):
        importlib.reload(shim)

    assert issubclass(shim.WrapUpAtLimits, GracefulLimits)
    assert shim.WrapUpPart is WrapUpPart
    assert shim.wrapped_up_by is wrapped_up_by
    assert shim.DEFAULT_WRAP_UP_PROMPT == DEFAULT_WRAP_UP_PROMPT


def _import_old_class() -> type[GracefulLimits[None]]:
    """The shim warns once on import; these tests assert the class's own warnings instead."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from pydantic_ai_harness.wrap_up_at_limits import WrapUpAtLimits  # pyright: ignore[reportDeprecated]

    return WrapUpAtLimits  # pyright: ignore[reportDeprecated]


_CLASS_WARNING = r'`WrapUpAtLimits` is now `GracefulLimits`.*rename the `WrapUpAtLimits` block to `GracefulLimits`'


def test_root_old_name_resolves_to_shim() -> None:
    """The old name resolves from the root, but stays out of `__all__`, as other renamed names do.

    A star import resolves every name in `__all__`, so listing it would import the shim and warn for every user.
    """
    import pydantic_ai_harness

    assert 'WrapUpAtLimits' not in pydantic_ai_harness.__all__
    assert 'GracefulLimits' in pydantic_ai_harness.__all__
    old = _import_old_class()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert pydantic_ai_harness.WrapUpAtLimits is old  # pyright: ignore[reportDeprecated]


async def test_old_class_warns_and_still_wraps_up() -> None:
    """Every construction warns, and the old class behaves as `GracefulLimits` does."""
    old = _import_old_class()

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.model_settings and info.model_settings.get('tool_choice') == 'none':
            return ModelResponse(parts=[TextPart('what I found so far')])
        return ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'a'})])

    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        capability = old()
    assert isinstance(capability, GracefulLimits)
    agent = Agent(FunctionModel(model), deps_type=type(None), capabilities=[capability])

    @agent.tool_plain
    def look_up(topic: str) -> str:
        return topic

    # Each run takes its own copy of the capability, and so warns again.
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        result = await agent.run('Research', usage_limits=UsageLimits(request_limit=2))

    assert result.output == 'what I found so far'
    assert wrapped_up_by(result.new_messages()) == 'request_limit'


def test_subclass_of_old_name_warns_once_and_keeps_its_own_spec_name() -> None:
    """Defining a subclass warns, since its base class has to migrate; creating one doesn't warn again.

    As before the rename, the subclass's spec block name is its own class name.
    """
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):

        @dataclass(init=False)
        class MyWrapUp(old):
            pass

    assert MyWrapUp.get_serialization_name() == 'MyWrapUp'
    agent = Agent.from_spec(
        {'model': 'test', 'capabilities': [{'MyWrapUp': {'timeout_margin': 5}}]},
        custom_capability_types=[MyWrapUp],
    )
    assert [type(c) for c in agent.root_capability.capabilities if isinstance(c, GracefulLimits)] == [MyWrapUp]


def test_old_spec_block_name_loads() -> None:
    old = _import_old_class()
    with pytest.warns(HarnessDeprecationWarning, match=_CLASS_WARNING):
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'WrapUpAtLimits': {'prompt': 'Stop and summarize.'}}]},
            custom_capability_types=[old],
        )

    assert [c.prompt for c in agent.root_capability.capabilities if isinstance(c, GracefulLimits)] == [
        'Stop and summarize.'
    ]
