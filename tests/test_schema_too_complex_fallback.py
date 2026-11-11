"""A request whose inferred Native Output schema the provider can't compile is attempted again in Tool Output.

These are offline tests with `FunctionModel`, which can reject a schema on cue the way Anthropic does. The live
rejection and the Tool Output attempt that follows it are recorded in
`tests/models/anthropic/test_strict_schema_limits.py` and `tests/models/bedrock/test_strict_schema_limits.py`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessage, ModelResponse, NativeOutput, ToolCallPart
from pydantic_ai.capabilities import ModelFallback
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    ModelRequestRejectedError,
    ModelSchemaTooComplexError,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.output import OutputMode, StructuredOutputMode
from pydantic_ai.profiles import ModelProfile

FALLS_BACK_TO_TOOL_OUTPUT = pytest.mark.moves_cache_prefix(
    reason="the Tool Output attempt sends the output tool, which the rejected Native Output attempt, that nothing cached, didn't"
)


class Survey(BaseModel):
    answer: int


def too_complex(model_name: str = 'compiler') -> ModelAPIError:
    return ModelHTTPError.for_category(ModelSchemaTooComplexError, status_code=400, model_name=model_name)


def schema_model(
    modes: list[OutputMode],
    *,
    infers: StructuredOutputMode = 'native',
    rejects: tuple[OutputMode, ...] = ('native',),
    error: ModelAPIError | None = None,
    model_name: str = 'compiler',
    supports_tools: bool = True,
) -> FunctionModel:
    """A model that infers `infers`, raises `error` in the `rejects` modes, and calls the output tool otherwise.

    `modes` records the output mode of every attempt.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        mode = info.model_request_parameters.output_mode
        modes.append(mode)
        if mode in rejects:
            raise error or too_complex(model_name)
        [output_tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(output_tool.name, {'answer': 1})])

    profile = ModelProfile(
        supports_json_schema_output=True, default_structured_output_mode=infers, supports_tools=supports_tools
    )
    return FunctionModel(respond, model_name=model_name, profile=profile)


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_rejected_inferred_native_output_is_attempted_in_tool_output():
    """The model never sees the rejected attempt, which the response records as failed."""
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes), output_type=Survey)

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool']
    assert result.usage.requests == 1
    [failed] = result.response.failed_attempts or []
    assert failed.model_name == 'compiler'
    assert failed.outcome == 'error'
    assert failed.error is not None and failed.error.startswith('ModelSchemaTooComplexError: ')

    # Another run infers Native Output again: only the run keeps the switch.
    modes.clear()
    await agent.run('Answer it again.')
    assert modes == ['native', 'tool']


async def test_explicit_native_output_raises():
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes), output_type=NativeOutput(Survey))

    with pytest.raises(ModelSchemaTooComplexError):
        await agent.run('Answer the survey.')
    assert modes == ['native']


async def test_other_rejections_are_raised():
    modes: list[OutputMode] = []
    error = ModelHTTPError.for_category(ModelRequestRejectedError, status_code=400, model_name='compiler')
    agent = Agent(schema_model(modes, error=error), output_type=Survey)

    with pytest.raises(ModelRequestRejectedError) as exc_info:
        await agent.run('Answer the survey.')
    assert not isinstance(exc_info.value, ModelSchemaTooComplexError)
    assert modes == ['native']


async def test_rejection_in_inferred_tool_output_is_raised():
    """Strict tools can be too complex too, and a Tool Output attempt wouldn't change those."""
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes, infers='tool', rejects=('tool',)), output_type=Survey)

    with pytest.raises(ModelSchemaTooComplexError):
        await agent.run('Answer the survey.')
    assert modes == ['tool']


async def test_rejection_is_raised_when_tool_output_isnt_supported():
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes, supports_tools=False), output_type=Survey)

    with pytest.raises(ModelSchemaTooComplexError):
        await agent.run('Answer the survey.')
    assert modes == ['native']


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_same_model_is_attempted_in_tool_output_before_model_fallback():
    """`ModelFallback` falls back on any `ModelAPIError` by default, but the run switches modes before it's asked."""
    modes: list[OutputMode] = []
    fallback_modes: list[OutputMode] = []
    agent = Agent(
        schema_model(modes),
        output_type=Survey,
        capabilities=[ModelFallback(schema_model(fallback_modes, model_name='fallback'))],
    )

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool']
    assert fallback_modes == []
    assert result.response.model_name == 'compiler'


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_model_fallback_falls_back_when_tool_output_is_rejected_too():
    modes: list[OutputMode] = []
    fallback_modes: list[OutputMode] = []
    agent = Agent(
        schema_model(modes, rejects=('native', 'tool')),
        output_type=Survey,
        capabilities=[ModelFallback(schema_model(fallback_modes, model_name='fallback', infers='tool'))],
    )

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool']
    assert fallback_modes == ['tool']
    assert [attempt.model_name for attempt in result.response.failed_attempts or []] == ['compiler', 'compiler']


def survey_model(modes: list[OutputMode], reject: list[bool]) -> FunctionModel:
    """A model that calls `look_up` first and the output tool after.

    In Native Output, it raises `too_complex()` for each request whose turn in `reject` is `True`.
    """
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        mode = info.model_request_parameters.output_mode
        modes.append(mode)
        if mode == 'native' and reject[requests]:
            raise too_complex()
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart('look_up', {})])
        return ModelResponse(parts=[ToolCallPart('final_result', {'answer': 1})])

    profile = ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native')
    return FunctionModel(respond, model_name='compiler', profile=profile)


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_later_steps_of_the_run_use_tool_output():
    """From its second request on, the run sends the mode it keeps for the model, which is now Tool Output."""
    modes: list[OutputMode] = []
    agent = Agent(survey_model(modes, reject=[True, True]), output_type=Survey)

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool', 'tool']


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_rejection_of_the_native_output_the_run_kept_is_attempted_in_tool_output():
    """A later request sends the Native Output the run's first request inferred, which can still be rejected."""
    modes: list[OutputMode] = []
    agent = Agent(survey_model(modes, reject=[False, True]), output_type=Survey)

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'native', 'tool']


def streaming_schema_model(modes: list[OutputMode]) -> FunctionModel:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        mode = info.model_request_parameters.output_mode
        modes.append(mode)
        if mode == 'native':
            raise too_complex()
        [output_tool] = info.output_tools
        yield {0: DeltaToolCall(name=output_tool.name, json_args='{"answer": 1}')}

    profile = ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native')
    return FunctionModel(stream_function=stream, model_name='compiler', profile=profile)


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_streamed_request_is_attempted_in_tool_output():
    """The stream is opened before it's handed over, so its rejection can still be acted on."""
    modes: list[OutputMode] = []
    agent = Agent(streaming_schema_model(modes), output_type=Survey)

    async with agent.run_stream('Answer the survey.') as result:
        output = await result.get_output()

    assert output == Survey(answer=1)
    assert modes == ['native', 'tool']


async def test_streamed_explicit_native_output_raises():
    modes: list[OutputMode] = []
    agent = Agent(streaming_schema_model(modes), output_type=NativeOutput(Survey))

    with pytest.raises(ModelSchemaTooComplexError):
        async with agent.run_stream('Answer the survey.') as result:
            await result.get_output()  # pragma: no cover  # opening the stream raises
    assert modes == ['native']
