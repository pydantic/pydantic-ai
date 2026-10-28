"""The `OutputModeFallback` capability.

These are offline tests with `FunctionModel`, which can reject a schema on cue the way Anthropic does. The live
rejection and the Tool Output attempt that follows it are recorded in
`tests/models/anthropic/test_strict_schema_limits.py`.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelMessage, ModelResponse, NativeOutput, RunContext, ToolCallPart
from pydantic_ai.capabilities import ModelFallback, OutputModeFallback, output_mode_fallback
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import OutputMode, StructuredOutputMode
from pydantic_ai.profiles import ModelProfile

FALLS_BACK_TO_TOOL_OUTPUT = pytest.mark.moves_cache_prefix(
    reason="the Tool Output attempt sends the output tool, which the rejected Native Output attempt, that nothing cached, didn't"
)

TOO_COMPLEX = {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': 'Schema is too complex.'}}


class Survey(BaseModel):
    answer: int


@pytest.fixture(autouse=True)
def forget_rejected_schemas() -> Iterator[None]:
    """The schemas a process remembers are module state, which would carry over from one test to the next."""
    output_mode_fallback._too_complex_schemas.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    output_mode_fallback._too_complex_schemas.clear()  # pyright: ignore[reportPrivateUsage]


def schema_model(
    modes: list[OutputMode],
    *,
    infers: StructuredOutputMode = 'native',
    rejects: tuple[OutputMode, ...] = ('native',),
    error_body: object = TOO_COMPLEX,
    status_code: int = 400,
    model_name: str = 'compiler',
) -> FunctionModel:
    """A model that infers `infers`, answers with `error_body` in the `rejects` modes, and calls the output tool otherwise.

    `modes` records the output mode of every attempt.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        mode = info.model_request_parameters.output_mode
        modes.append(mode)
        if mode in rejects:
            raise ModelHTTPError(status_code, model_name, body=error_body)
        [output_tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(output_tool.name, {'answer': 1})])

    profile = ModelProfile(supports_json_schema_output=True, default_structured_output_mode=infers)
    return FunctionModel(respond, model_name=model_name, profile=profile)


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_rejected_inferred_native_output_is_attempted_in_tool_output():
    """The model never sees the rejected attempt, and the process uses Tool Output for the schema from then on."""
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes), output_type=Survey, capabilities=[OutputModeFallback()])

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool']
    assert result.usage.requests == 1

    modes.clear()
    result = await agent.run('Answer it again.')
    assert result.output == Survey(answer=1)
    assert modes == ['tool']


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_rejected_schema_is_remembered_per_model():
    """Another model may compile the schema, so it still gets Native Output first."""
    modes: list[OutputMode] = []
    capability = OutputModeFallback[None]()
    agent = Agent(schema_model(modes), deps_type=type(None), output_type=Survey, capabilities=[capability])
    await agent.run('Answer the survey.')

    modes.clear()
    other = schema_model(modes, model_name='other-compiler')
    await agent.run('Answer the survey.', model=other)
    assert modes == ['native', 'tool']


async def test_explicit_native_output_is_left_alone():
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes), output_type=NativeOutput(Survey), capabilities=[OutputModeFallback()])

    with pytest.raises(ModelHTTPError, match='status_code: 400'):
        await agent.run('Answer the survey.')
    assert modes == ['native']


@pytest.mark.parametrize(
    ('status_code', 'message'),
    [(400, 'messages: field required'), (500, 'Schema is too complex.')],
    ids=['other-400', 'not-a-400'],
)
async def test_other_errors_are_raised(status_code: int, message: str):
    modes: list[OutputMode] = []
    body = {'type': 'error', 'error': {'type': 'invalid_request_error', 'message': message}}
    model = schema_model(modes, error_body=body, status_code=status_code)
    agent = Agent(model, output_type=Survey, capabilities=[OutputModeFallback()])

    with pytest.raises(ModelHTTPError, match=f'status_code: {status_code}'):
        await agent.run('Answer the survey.')
    assert modes == ['native']


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_rejection_of_the_native_output_a_run_kept_is_attempted_in_tool_output():
    """A later request keeps the mode the run's first request inferred, and Anthropic's compile can still time out."""
    modes: list[OutputMode] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        mode = info.model_request_parameters.output_mode
        modes.append(mode)
        if len(modes) == 1:
            return ModelResponse(parts=[ToolCallPart('look_up', {})])
        if mode == 'native':
            raise ModelHTTPError(400, 'compiler', body=TOO_COMPLEX)
        return ModelResponse(parts=[ToolCallPart('final_result', {'answer': 1})])

    model = FunctionModel(
        respond,
        model_name='compiler',
        profile=ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native'),
    )
    agent = Agent(model, output_type=Survey, capabilities=[OutputModeFallback()])

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'native', 'tool']


async def test_rejection_in_inferred_tool_output_is_raised():
    """Strict tools can be too complex too, and a Tool Output attempt wouldn't change those."""
    modes: list[OutputMode] = []
    model = schema_model(modes, infers='tool', rejects=('tool',))
    agent = Agent(model, output_type=Survey, capabilities=[OutputModeFallback()])

    with pytest.raises(ModelHTTPError, match='status_code: 400'):
        await agent.run('Answer the survey.')
    assert modes == ['tool']


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_same_model_is_attempted_in_tool_output_before_model_fallback():
    modes: list[OutputMode] = []
    fallback_modes: list[OutputMode] = []
    agent = Agent(
        schema_model(modes),
        output_type=Survey,
        # Listed outside `ModelFallback`, it's still placed inside it.
        capabilities=[OutputModeFallback(), ModelFallback(schema_model(fallback_modes, model_name='fallback'))],
    )

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool']
    assert fallback_modes == []


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_later_steps_of_the_run_use_tool_output():
    """The mode a run's first request inferred is kept for later requests, but not once that mode was rejected."""
    modes: list[OutputMode] = []
    calls = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        mode = info.model_request_parameters.output_mode
        modes.append(mode)
        if mode == 'native':
            raise ModelHTTPError(400, 'compiler', body=TOO_COMPLEX)
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart('look_up', {})])
        return ModelResponse(parts=[ToolCallPart('final_result', {'answer': 1})])

    model = FunctionModel(
        respond,
        model_name='compiler',
        profile=ModelProfile(supports_json_schema_output=True, default_structured_output_mode='native'),
    )
    agent = Agent(model, output_type=Survey, capabilities=[OutputModeFallback()])

    @agent.tool_plain
    def look_up() -> str:
        # Forgotten by the process, as under durable execution: the run still remembers.
        output_mode_fallback._too_complex_schemas.clear()  # pyright: ignore[reportPrivateUsage]
        return 'found'

    result = await agent.run('Answer the survey.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool', 'tool']


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_durable_runs_dont_use_schemas_other_runs_rejected(monkeypatch: pytest.MonkeyPatch):
    """Workflow code has to make the same choices when it's replayed, by a process that may not remember them."""
    modes: list[OutputMode] = []
    agent = Agent(schema_model(modes), output_type=Survey, capabilities=[OutputModeFallback()])
    await agent.run('Answer the survey.')
    remembered = dict(output_mode_fallback._too_complex_schemas)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(RunContext, 'in_durable_context', property(lambda self: True))
    modes.clear()
    result = await agent.run('Answer it again.')

    assert result.output == Survey(answer=1)
    assert modes == ['native', 'tool']
    assert output_mode_fallback._too_complex_schemas == remembered  # pyright: ignore[reportPrivateUsage]


@FALLS_BACK_TO_TOOL_OUTPUT
async def test_remembered_schemas_are_bounded(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(output_mode_fallback, '_MAX_REMEMBERED_SCHEMAS', 1)
    modes: list[OutputMode] = []
    first = Agent(schema_model(modes), output_type=Survey, capabilities=[OutputModeFallback()])
    second = Agent(
        schema_model(modes, model_name='other-compiler'), output_type=Survey, capabilities=[OutputModeFallback()]
    )
    await first.run('Answer the survey.')
    await second.run('Answer the survey.')

    modes.clear()
    await first.run('Answer it again.')
    assert modes == ['native', 'tool']
