from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest
from httpx import AsyncClient
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import UserError
from pydantic_ai._utils import ObjectJsonSchema
from pydantic_ai.models.gemini import (
    GeminiModel,
    _GeminiFunction,  # pyright: ignore[reportPrivateUsage]
    _GeminiFunctionCallingConfig,  # pyright: ignore[reportPrivateUsage]
    _GeminiToolConfig,  # pyright: ignore[reportPrivateUsage]
    _GeminiTools,  # pyright: ignore[reportPrivateUsage]
)
from tests.conftest import TestEnv


def test_api_key_arg(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    assert m.api_key == 'via-arg'


def test_api_key_env_var(env: TestEnv):
    env.set('GEMINI_API_KEY', 'via-env-var')
    m = GeminiModel('gemini-1.5-flash')
    assert m.api_key == 'via-env-var'


def test_api_key_not_set(env: TestEnv):
    with pytest.raises(UserError, match='API key must be provided or set in the GEMINI_API_KEY environment variable'):
        GeminiModel('gemini-1.5-flash')


def test_api_key_empty(env: TestEnv):
    env.set('GEMINI_API_KEY', '')
    with pytest.raises(UserError, match='API key must be provided or set in the GEMINI_API_KEY environment variable'):
        GeminiModel('gemini-1.5-flash')


def test_agent_model_simple():
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    agent_model = m.agent_model({}, True, None)
    assert isinstance(agent_model.http_client, AsyncClient)
    assert agent_model.model_name == 'gemini-1.5-flash'
    assert agent_model.api_key == 'via-arg'
    assert agent_model.tools is None
    assert agent_model.tool_config is None


@dataclass
class TestToolDefinition:
    __test__ = False
    name: str
    description: str
    json_schema: ObjectJsonSchema


def test_agent_model_tools():
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    retrievers = {
        'foo': TestToolDefinition(
            'foo',
            'This is foo',
            {'type': 'object', 'title': 'Foo', 'properties': {'bar': {'type': 'number', 'title': 'Bar'}}},
        ),
        'apple': TestToolDefinition(
            'apple',
            'This is apple',
            {  # type: ignore
                'type': 'object',
                'properties': {
                    'banana': {'type': 'array', 'title': 'Banana', 'items': {'type': 'number', 'title': 'Bar'}}
                },
            },
        ),
    }
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}, 'required': ['spam']},
    )
    agent_model = m.agent_model(retrievers, True, result_tool)
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='foo',
                    description='This is foo',
                    parameters={'type': 'object', 'properties': {'bar': {'type': 'number'}}},
                ),
                _GeminiFunction(
                    name='apple',
                    description='This is apple',
                    parameters={
                        'type': 'object',
                        'properties': {'banana': {'type': 'array', 'items': {'type': 'number'}}},
                    },
                ),
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'type': 'object',
                        'properties': {'spam': {'type': 'number'}},
                        'required': ['spam'],
                    },
                ),
            ]
        )
    )
    assert agent_model.tool_config is None


def test_require_response_tool():
    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        {'type': 'object', 'title': 'Result', 'properties': {'spam': {'type': 'number'}}},
    )
    agent_model = m.agent_model({}, False, result_tool)
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'type': 'object',
                        'properties': {'spam': {'type': 'number'}},
                    },
                ),
            ]
        )
    )
    assert agent_model.tool_config == snapshot(
        _GeminiToolConfig(
            function_calling_config=_GeminiFunctionCallingConfig(mode='ANY', allowed_function_names=['result'])
        )
    )


def test_json_def_replaced():
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        locations: list[Location]

    json_schema = Locations.model_json_schema()
    assert json_schema == snapshot(
        {
            '$defs': {
                'Location': {
                    'properties': {
                        'lat': {'title': 'Lat', 'type': 'number'},
                        'lng': {'title': 'Lng', 'type': 'number'},
                    },
                    'required': ['lat', 'lng'],
                    'title': 'Location',
                    'type': 'object',
                }
            },
            'properties': {'locations': {'items': {'$ref': '#/$defs/Location'}, 'title': 'Locations', 'type': 'array'}},
            'required': ['locations'],
            'title': 'Locations',
            'type': 'object',
        }
    )

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,  # pyright: ignore[reportArgumentType]
    )
    agent_model = m.agent_model({}, True, result_tool)
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'properties': {
                            'locations': {
                                'items': {
                                    'properties': {
                                        'lat': {'type': 'number'},
                                        'lng': {'type': 'number'},
                                    },
                                    'required': ['lat', 'lng'],
                                    'type': 'object',
                                },
                                'type': 'array',
                            }
                        },
                        'required': ['locations'],
                        'type': 'object',
                    },
                )
            ]
        )
    )


def test_json_def_replaced_any_of():
    class Location(BaseModel):
        lat: float
        lng: float

    class Locations(BaseModel):
        op_location: Location | None = None

    json_schema = Locations.model_json_schema()

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,  # pyright: ignore[reportArgumentType]
    )
    agent_model = m.agent_model({}, True, result_tool)
    assert agent_model.tools == snapshot(
        _GeminiTools(
            function_declarations=[
                _GeminiFunction(
                    name='result',
                    description='This is the tool for the final Result',
                    parameters={
                        'properties': {
                            'op_location': {
                                'anyOf': [
                                    {
                                        'properties': {'lat': {'type': 'number'}, 'lng': {'type': 'number'}},
                                        'required': ['lat', 'lng'],
                                        'type': 'object',
                                    },
                                    {'type': 'null'},
                                ],
                                'default': None,
                            }
                        },
                        'type': 'object',
                    },
                )
            ]
        )
    )


def test_json_def_recursive():
    class Location(BaseModel):
        lat: float
        lng: float
        nested_locations: list[Location]

    json_schema = Location.model_json_schema()
    assert json_schema == snapshot(
        {
            '$defs': {
                'Location': {
                    'properties': {
                        'lat': {'title': 'Lat', 'type': 'number'},
                        'lng': {'title': 'Lng', 'type': 'number'},
                        'nested_locations': {
                            'items': {'$ref': '#/$defs/Location'},
                            'title': 'Nested Locations',
                            'type': 'array',
                        },
                    },
                    'required': ['lat', 'lng', 'nested_locations'],
                    'title': 'Location',
                    'type': 'object',
                }
            },
            '$ref': '#/$defs/Location',
        }
    )

    m = GeminiModel('gemini-1.5-flash', api_key='via-arg')
    result_tool = TestToolDefinition(
        'result',
        'This is the tool for the final Result',
        json_schema,  # pyright: ignore[reportArgumentType]
    )
    with pytest.raises(UserError, match=r'Recursive `\$ref`s in JSON Schema are not supported by Gemini'):
        m.agent_model({}, True, result_tool)
