"""Record provider acceptance as well as the boolean root schema sent on the wire."""

from copy import deepcopy
from typing import Literal

import pytest

from pydantic_ai import Agent, Tool, ToolReturnPart
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.exceptions import UserError

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('definition', [True, False])
async def test_openai_boolean_root_tool(
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
    api: Literal['chat', 'responses'],
    definition: bool,
):
    def accept(x: int) -> str:
        return f'Accepted {x}'

    schema: JsonSchema = {
        '$ref': '#/$defs/X',
        '$defs': {'X': definition},
        'type': 'object',
        'properties': {'x': {'type': 'integer'}},
        'required': ['x'],
        'additionalProperties': False,
    }
    original = deepcopy(schema)
    tool = Tool.from_schema(accept, name='accept', description='Accept an integer', json_schema=schema)
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = (
        OpenAIChatModel('gpt-4.1-mini', provider=provider)
        if api == 'chat'
        else OpenAIResponsesModel('gpt-4.1-mini', provider=provider)
    )
    agent = Agent(model, tools=[tool])

    if not definition:
        with pytest.raises(UserError, match='does not support boolean `false` root schemas'):
            await agent.run('Call accept with x=7, then say done.')
        assert request_capture.raw_bodies == []
    else:
        result = await agent.run('Call accept with x=7, then say done.')
        tool_returns = [
            part.content
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert tool_returns == ['Accepted 7']
        expected = deepcopy(schema)
        expected.pop('$ref')
        function: JsonSchema = {
            'name': 'accept',
            'description': 'Accept an integer',
            'parameters': expected,
            'strict': True,
        }
        expected_tools: list[JsonSchema] = (
            [{'type': 'function', 'function': function}] if api == 'chat' else [{'type': 'function', **function}]
        )
        assert request_capture.body()['tools'] == expected_tools
    assert schema == original
