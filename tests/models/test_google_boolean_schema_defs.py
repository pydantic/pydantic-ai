"""Record Gemini acceptance of an inlined true definition with a constraining sibling."""

from copy import deepcopy

import pytest

from pydantic_ai import Agent, Tool, ToolReturnPart
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.google import GoogleOpenAPISchemaTransformer

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.vcr,
]


@pytest.mark.parametrize('definition', [True, False])
async def test_google_inlined_boolean_definition_tool(
    allow_model_requests: None, gemini_api_key: str, request_capture: RequestCapture, definition: bool
):
    def accept(x: int) -> str:
        return f'Accepted {x}'

    schema: JsonSchema = {
        'type': 'object',
        'properties': {'x': {'$ref': '#/$defs/X', 'type': 'integer', 'description': 'The integer to accept'}},
        'required': ['x'],
        '$defs': {'X': definition},
    }
    original = deepcopy(schema)
    tool = Tool.from_schema(accept, name='accept', description='Accept an integer', json_schema=schema)
    provider = GoogleProvider(api_key=gemini_api_key, http_client=request_capture.http_client(timeout=60))
    profile = ModelProfile(json_schema_transformer=GoogleOpenAPISchemaTransformer)
    model = GoogleModel('gemini-3.8-flash', provider=provider, profile=profile)
    agent = Agent(model, tools=[tool])

    if not definition:
        with pytest.raises(UserError, match='cannot represent a boolean `false` definition'):
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
        expected.pop('$defs')
        expected['properties']['x'].pop('$ref')
        assert request_capture.body()['tools'] == [
            {
                'functionDeclarations': [
                    {'name': 'accept', 'description': 'Accept an integer', 'parameters_json_schema': expected}
                ]
            }
        ]
    assert schema == original
