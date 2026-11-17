"""`UnsupportedSettingWarning`: when a model ignores a setting the request relies on, it says so.

These run on `FunctionModel` subclasses that declare `unsupported_settings`, the way a third-party model class
would, so they need no provider SDK or recording: the warning is emitted before any request leaves the process.
Which settings each built-in model ignores, and that it warns about exactly the right ones, is pinned against the
wire by `test_model_settings_support.py`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import get_args, get_type_hints

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelHTTPError, UnsupportedSettingWarning
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.settings import ModelSettingName, ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage, UsageLimits

from ..conftest import try_import

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as mcp_available:
    from mcp import types as mcp_types

    from pydantic_ai.mcp import _build_sampling_handler  # pyright: ignore[reportPrivateUsage]


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


class Answer(BaseModel):
    answer: str


def _structured_reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'answer': 'done'})])


@dataclass(init=False)
class CountingIgnoringModel(IgnoringModel):
    """Prepares the request to count its tokens, like the provider models that count tokens through their API."""

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        self.prepare_request(model_settings, model_request_parameters)
        return RequestUsage(input_tokens=1)


@pytest.mark.parametrize('fallback', [False, True])
async def test_warns_once_per_request(fallback: bool):
    """Framework code prepares a request without sending it too, to resolve the output mode, count its tokens and
    describe it on a span; only the model that sends the request warns. (`FallbackModel` can't count tokens ahead.)
    """
    model = CountingIgnoringModel(_structured_reply, model_name='ignoring')
    agent = Agent(
        FallbackModel(FunctionModel(_fail), model) if fallback else model,
        output_type=Answer,
        capabilities=[Instrumentation(settings=InstrumentationSettings())],
        model_settings={'timeout': 5},
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always', UnsupportedSettingWarning)
        result = await agent.run('hello', usage_limits=UsageLimits(count_tokens_before_request=not fallback))

    assert result.output == Answer(answer='done')
    assert len(record) == 1


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize('profile_drops_it', [False, True])
def test_openai_profile_adds_ignored_settings(profile_drops_it: bool):
    """An OpenAI-compatible provider whose profile drops a setting warns about it like the model class would."""
    profile = OpenAIModelProfile(openai_unsupported_model_settings=('parallel_tool_calls',) if profile_drops_it else ())
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test'), profile=profile)
    params = ModelRequestParameters(function_tools=[ToolDefinition(name='tool')])

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always', UnsupportedSettingWarning)
        model.prepare_request({'parallel_tool_calls': False}, params)

    assert [str(w.message) for w in record] == (
        ["The `parallel_tool_calls` setting is not supported by model 'openai:gpt-4o' and will be ignored."]
        if profile_drops_it
        else []
    )


@pytest.mark.skipif(not mcp_available(), reason='mcp not installed')
@pytest.mark.parametrize('server_sends_stop_sequences', [True, False])
async def test_mcp_sampling_stays_silent_about_the_servers_stop_sequences(server_sends_stop_sequences: bool):
    """The server chose the `stop_sequences` of its sampling request, so only the user's own settings warn."""

    @dataclass(init=False)
    class StopIgnoringModel(FunctionModel):
        unsupported_settings = frozenset({'timeout', 'stop_sequences'})

    model = StopIgnoringModel(_reply, model_name='ignoring', settings={'timeout': 5, 'stop_sequences': ['END']})
    handler = _build_sampling_handler(model)
    params = mcp_types.CreateMessageRequestParams(
        messages=[mcp_types.SamplingMessage(role='user', content=mcp_types.TextContent(type='text', text='hi'))],
        maxTokens=42,
        stopSequences=['STOP'] if server_sends_stop_sequences else None,
    )

    with pytest.warns(UnsupportedSettingWarning) as record:
        await handler([], params, None)  # type: ignore[arg-type, misc]

    warned = [str(w.message).split('`')[1] for w in record]
    assert warned == (['timeout'] if server_sends_stop_sequences else ['timeout', 'stop_sequences'])


def test_model_setting_name_lists_every_model_settings_field():
    assert set(get_args(ModelSettingName)) == set(get_type_hints(ModelSettings))
