"""`ModelResponse.model_id`: the model the agent sent each request to, as it identifies it."""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.capabilities import AbstractCapability, ModelFallback, WrapModelRequestHandler
from pydantic_ai.exceptions import ModelAPIError, SkipModelRequest
from pydantic_ai.models import Model, ModelRequestContext, ModelResolutionContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext


def _answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if info.function_tools and any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
        return ModelResponse(parts=[ToolCallPart('noop')])
    return ModelResponse(parts=[TextPart('done')])


async def _stream_answer(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'done'


def _failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelAPIError(model_name='primary', message='down')


def _model_ids(messages: list[ModelMessage]) -> list[str | None]:
    return [message.model_id for message in messages if isinstance(message, ModelResponse)]


async def test_every_response_records_the_model_instance_it_came_from():
    agent = Agent(FunctionModel(_answer, model_name='primary'))

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    result = await agent.run('x')
    assert _model_ids(result.all_messages()) == ['function:primary', 'function:primary']


async def test_a_model_selected_by_name_is_recorded_by_that_name():
    """The string the model was selected by is recorded, here one only a capability can resolve."""

    class Alias(AbstractCapability[None]):
        async def resolve_model_id(self, ctx: ModelResolutionContext[None], *, model_id: str) -> Model | None:
            return FunctionModel(_answer, model_name='resolved') if model_id == 'my-alias' else None

    agent = Agent('my-alias', deps_type=type(None), capabilities=[Alias()])
    result = await agent.run('x')
    assert result.response.model_name == 'resolved'
    assert result.response.model_id == 'my-alias'


async def test_a_streamed_response_records_its_model():
    agent = Agent(FunctionModel(stream_function=_stream_answer, model_name='streamer'))
    async with agent.run_stream('x') as result:
        assert await result.get_output() == 'done'
    assert _model_ids(result.all_messages()) == ['function:streamer']


async def test_a_fallback_response_records_the_candidate_that_answered():
    agent = Agent(
        FunctionModel(_failure, model_name='primary'),
        capabilities=[ModelFallback(FunctionModel(_answer, model_name='backup'))],
    )
    result = await agent.run('x')
    assert result.response.model_id == 'function:backup'
    assert [attempt.model_name for attempt in result.response.failed_attempts or []] == ['primary']


async def test_a_response_skipped_before_any_request_records_no_model():
    class Skip(AbstractCapability[None]):
        async def wrap_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, handler: WrapModelRequestHandler
        ) -> ModelResponse:
            raise SkipModelRequest(ModelResponse(parts=[TextPart('skipped')]))

    agent = Agent(FunctionModel(_answer), deps_type=type(None), capabilities=[Skip()])
    result = await agent.run('x')
    assert result.output == 'skipped'
    assert result.response.model_id is None


async def test_a_model_id_the_response_already_has_is_kept():
    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')], model_id='upstream:model')

    result = await Agent(FunctionModel(answer)).run('x')
    assert result.response.model_id == 'upstream:model'


async def test_a_streamed_fallback_response_records_the_candidate_that_answered():
    async def failure_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise ModelAPIError(model_name='primary', message='down')
        yield ''  # pragma: no cover

    agent = Agent(
        FunctionModel(stream_function=failure_stream, model_name='primary'),
        capabilities=[ModelFallback(FunctionModel(stream_function=_stream_answer, model_name='backup'))],
    )
    async with agent.run_stream('x') as result:
        assert await result.get_output() == 'done'
    assert _model_ids(result.all_messages()) == ['function:backup']
