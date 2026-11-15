"""Synthetic streams reproduce code arriving in the initial item instead of only in deltas."""

import json

import pytest

from pydantic_ai.messages import ModelRequest, NativeToolCallPart, NativeToolReturnPart, PartStartEvent, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types import responses

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAIResponses, response_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@pytest.mark.parametrize(
    'seed, deltas',
    [
        pytest.param('print(1)', [], id='initial-code-only'),
        pytest.param('print(', ['1', ')'], id='initial-code-and-deltas'),
        pytest.param('print("a\\b")\n', ['print("café 🌍")\t#\r\n'], id='json-escaping'),
        pytest.param('', ['print(', '1)'], id='empty-seed'),
        pytest.param(None, ['print(1)'], id='null-seed'),
    ],
)
async def test_code_interpreter_stream_preserves_initial_code(
    allow_model_requests: None, seed: str | None, deltas: list[str]
):
    """Keep the seed and deltas exactly once when both done events repeat the completed code.

    Synthetic events make the provider's intermittent nonempty seed deterministic.
    """
    call_id = 'ci_1'
    container_id = 'cntr_1'
    code = (seed or '') + ''.join(deltas)
    added_item = responses.ResponseCodeInterpreterToolCall(
        id=call_id,
        container_id=container_id,
        code=seed,
        status='in_progress',
        type='code_interpreter_call',
    )
    done_item = added_item.model_copy(update={'code': code, 'status': 'completed', 'outputs': []})
    response = response_message([done_item])
    stream: list[responses.ResponseStreamEvent] = [
        responses.ResponseCreatedEvent(response=response_message([]), type='response.created', sequence_number=0),
        responses.ResponseOutputItemAddedEvent(
            item=added_item, output_index=0, type='response.output_item.added', sequence_number=1
        ),
        *(
            responses.ResponseCodeInterpreterCallCodeDeltaEvent(
                delta=delta,
                item_id=call_id,
                output_index=0,
                type='response.code_interpreter_call_code.delta',
                sequence_number=index,
            )
            for index, delta in enumerate(deltas, 2)
        ),
        responses.ResponseCodeInterpreterCallCodeDoneEvent(
            code=code,
            item_id=call_id,
            output_index=0,
            type='response.code_interpreter_call_code.done',
            sequence_number=len(deltas) + 2,
        ),
        responses.ResponseOutputItemDoneEvent(
            item=done_item,
            output_index=0,
            type='response.output_item.done',
            sequence_number=len(deltas) + 3,
        ),
        responses.ResponseCompletedEvent(
            response=response.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=len(deltas) + 4,
        ),
    ]
    model = OpenAIResponsesModel(
        'gpt-5', provider=OpenAIProvider(openai_client=MockOpenAIResponses.create_mock_stream(stream))
    )
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart('Run the code')])],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(native_tools=[CodeExecutionTool()]),
    ) as streamed:
        events = [event async for event in streamed]

    call_part, return_part = streamed.get().parts
    assert isinstance(call_part, NativeToolCallPart)
    assert isinstance(return_part, NativeToolReturnPart)
    assert isinstance(call_part.args, str)
    assert json.loads(call_part.args) == {'container_id': container_id, 'code': code}
    assert (
        sum(isinstance(event, PartStartEvent) and isinstance(event.part, NativeToolCallPart) for event in events) == 1
    )
