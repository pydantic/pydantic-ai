"""`UnsupportedSettingWarning`: when a model ignores a setting the request relies on, it says so.

These run on `FunctionModel` subclasses that declare `unsupported_settings`, the way a third-party model class
would, so they need no provider SDK or recording: the warning is emitted before any request leaves the process.
Which settings each built-in model ignores, and that it warns about exactly the right ones, is pinned against the
wire by `test_model_settings_support.py`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent, ModelHTTPError, UnsupportedSettingWarning
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.settings import ModelSettings


def _reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


def _fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=503, model_name='failing')


@dataclass(init=False)
class IgnoringModel(FunctionModel):
    """A model that ignores two guarantees and one tuning knob."""

    unsupported_settings = frozenset({'timeout', 'parallel_tool_calls', 'temperature'})


def _tool() -> str:
    return 'ok'  # pragma: no cover


@pytest.mark.parametrize('source', ['run', 'agent', 'model'])
async def test_warns_wherever_the_setting_comes_from(source: str):
    settings: ModelSettings = {'timeout': 5}
    model = IgnoringModel(_reply, model_name='ignoring', settings=settings if source == 'model' else None)
    agent = Agent(model, model_settings=settings if source == 'agent' else None)

    with pytest.warns(UnsupportedSettingWarning) as record:
        await agent.run('hello', model_settings=settings if source == 'run' else None)

    assert [str(w.message) for w in record] == [
        "The `timeout` setting is not supported by model 'function:ignoring' and will be ignored. "
        'Use `request_timeout` instead, which Pydantic AI enforces for every model.'
    ]


async def test_tuning_knobs_and_unset_or_supported_settings_stay_silent():
    """`temperature` is ignored silently, and `stop_sequences` isn't ignored by this model at all."""
    agent = Agent(IgnoringModel(_reply, model_name='ignoring'))

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnsupportedSettingWarning)
        result = await agent.run('hello', model_settings={'temperature': 0.5, 'stop_sequences': ['STOP']})

    assert result.output == 'done'


async def test_a_model_that_declares_nothing_never_warns():
    agent = Agent(FunctionModel(_reply))

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnsupportedSettingWarning)
        result = await agent.run('hello', model_settings={'timeout': 5, 'parallel_tool_calls': False})

    assert result.output == 'done'


@pytest.mark.parametrize(
    ('parallel_tool_calls', 'with_tools', 'warns'),
    [
        pytest.param(False, True, True, id='false-with-tools'),
        pytest.param(True, True, False, id='true-with-tools'),
        pytest.param(False, False, False, id='false-without-tools'),
    ],
)
async def test_parallel_tool_calls_warns_only_when_false_and_tools_are_offered(
    parallel_tool_calls: bool, with_tools: bool, warns: bool
):
    """Only `False` is a guarantee, and a request without function tools can't make parallel calls anyway."""
    agent = Agent(IgnoringModel(_reply, model_name='ignoring'), tools=[_tool] if with_tools else [])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always', UnsupportedSettingWarning)
        await agent.run('hello', model_settings={'parallel_tool_calls': parallel_tool_calls})

    assert [str(w.message) for w in record] == (
        ["The `parallel_tool_calls` setting is not supported by model 'function:ignoring' and will be ignored."]
        if warns
        else []
    )


async def test_fallback_model_warns_only_for_the_models_it_tries():
    """An agent-level setting reaches every model in a `FallbackModel`, but only one that's tried can ignore it."""
    honoring = FunctionModel(_reply)
    agent = Agent(FallbackModel(honoring, IgnoringModel(_reply, model_name='ignoring')), model_settings={'timeout': 5})

    with warnings.catch_warnings():
        warnings.simplefilter('error', UnsupportedSettingWarning)
        await agent.run('hello')

    agent = Agent(
        FallbackModel(FunctionModel(_fail), IgnoringModel(_reply, model_name='ignoring')), model_settings={'timeout': 5}
    )
    with pytest.warns(
        UnsupportedSettingWarning, match="^The `timeout` setting is not supported by model 'function:ignoring'"
    ):
        result = await agent.run('hello')
    assert result.output == 'done'
