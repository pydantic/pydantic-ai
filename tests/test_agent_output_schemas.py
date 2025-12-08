import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    BinaryImage,
    DeferredToolRequests,
    NativeOutput,
    PromptedOutput,
    StructuredDict,
    TextOutput,
    ToolOutput,
)

pytestmark = pytest.mark.anyio


class Bar(BaseModel):
    answer: str


class Foo(BaseModel):
    a: list[Bar]
    b: int


async def test_text_output_json_schema():
    agent = Agent('test')
    assert agent.output_json_schema() == snapshot({'type': 'string'})

    def func(x: str) -> str:
        return x  # pragma: no cover

    agent = Agent('test', output_type=TextOutput(func))
    assert agent.output_json_schema() == snapshot({'type': 'string'})


async def test_function_output_json_schema():
    def func(x: int) -> int:
        return x  # pragma: no cover

    agent = Agent('test', output_type=[func])
    assert agent.output_json_schema() == snapshot({'type': 'integer'})


async def test_auto_output_json_schema():
    # one output
    agent = Agent('test', output_type=bool)
    assert agent.output_json_schema() == snapshot({'type': 'boolean'})

    # multiple no str
    agent = Agent('test', output_type=bool | int)
    assert agent.output_json_schema() == snapshot({'anyOf': [{'type': 'boolean'}, {'type': 'integer'}]})

    # multiple outputs
    agent = Agent('test', output_type=str | bool | Foo)
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {'type': 'string'},
                {'type': 'boolean'},
                {
                    'properties': {
                        'a': {'items': {'$ref': '#/$defs/Bar'}, 'title': 'A', 'type': 'array'},
                        'b': {'title': 'B', 'type': 'integer'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
            ],
            '$defs': {
                'Bar': {
                    'properties': {'answer': {'title': 'Answer', 'type': 'string'}},
                    'required': ['answer'],
                    'title': 'Bar',
                    'type': 'object',
                }
            },
        }
    )


async def test_tool_output_json_schema():
    # one output
    agent = Agent(
        'test',
        output_type=[ToolOutput(bool)],
    )
    assert agent.output_json_schema() == snapshot({'type': 'boolean'})

    # multiple outputs
    agent = Agent(
        'test',
        output_type=[ToolOutput(str), ToolOutput(bool), ToolOutput(Foo)],
    )
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {'type': 'string'},
                {'type': 'boolean'},
                {
                    'properties': {
                        'a': {'items': {'$ref': '#/$defs/Bar'}, 'title': 'A', 'type': 'array'},
                        'b': {'title': 'B', 'type': 'integer'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
            ],
            '$defs': {
                'Bar': {
                    'properties': {'answer': {'title': 'Answer', 'type': 'string'}},
                    'required': ['answer'],
                    'title': 'Bar',
                    'type': 'object',
                }
            },
        }
    )

    # multiple duplicate output types
    agent = Agent(
        'test',
        output_type=[ToolOutput(bool), ToolOutput(bool), ToolOutput(bool)],
    )
    assert agent.output_json_schema() == snapshot({'type': 'boolean'})


async def test_native_output_json_schema():
    agent = Agent(
        'test',
        output_type=NativeOutput([bool]),
    )
    assert agent.output_json_schema() == snapshot({'type': 'boolean'})

    agent = Agent(
        'test',
        output_type=NativeOutput([bool, Foo]),
    )
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {'type': 'boolean'},
                {
                    'properties': {
                        'a': {'items': {'$ref': '#/$defs/Bar'}, 'title': 'A', 'type': 'array'},
                        'b': {'title': 'B', 'type': 'integer'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
            ],
            '$defs': {
                'Bar': {
                    'properties': {'answer': {'title': 'Answer', 'type': 'string'}},
                    'required': ['answer'],
                    'title': 'Bar',
                    'type': 'object',
                }
            },
        }
    )


async def test_prompted_output_json_schema():
    agent = Agent(
        'test',
        output_type=PromptedOutput([bool]),
    )
    assert agent.output_json_schema() == snapshot({'type': 'boolean'})

    agent = Agent(
        'test',
        output_type=PromptedOutput([bool, Foo]),
    )
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {'type': 'boolean'},
                {
                    'properties': {
                        'a': {'items': {'$ref': '#/$defs/Bar'}, 'title': 'A', 'type': 'array'},
                        'b': {'title': 'B', 'type': 'integer'},
                    },
                    'required': ['a', 'b'],
                    'title': 'Foo',
                    'type': 'object',
                },
            ],
            '$defs': {
                'Bar': {
                    'properties': {'answer': {'title': 'Answer', 'type': 'string'}},
                    'required': ['answer'],
                    'title': 'Bar',
                    'type': 'object',
                }
            },
        }
    )


async def test_custom_output_json_schema():
    HumanDict = StructuredDict(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
        },
        name='Human',
        description='A human with a name and age',
    )
    agent = Agent('test', output_type=HumanDict)
    assert agent.output_json_schema() == snapshot(
        {
            'description': 'A human with a name and age',
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'title': 'Human',
            'required': ['name', 'age'],
        }
    )


async def test_image_output_json_schema():
    # one output
    agent = Agent('test', output_type=BinaryImage)
    assert agent.output_json_schema() == snapshot(
        {
            'properties': {
                'data': {'format': 'binary', 'title': 'Data', 'type': 'string'},
                'media_type': {
                    'anyOf': [
                        {'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'], 'type': 'string'},
                        {'type': 'string'},
                    ],
                    'title': 'Media Type',
                },
                'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
            },
            'required': ['data', 'media_type'],
            'title': 'BinaryImage',
            'type': 'object',
        }
    )

    # multiple outputs
    agent = Agent('test', output_type=str | bool | BinaryImage)
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {'type': 'string'},
                {'type': 'boolean'},
                {
                    'properties': {
                        'data': {'format': 'binary', 'title': 'Data', 'type': 'string'},
                        'media_type': {
                            'anyOf': [
                                {'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'], 'type': 'string'},
                                {'type': 'string'},
                            ],
                            'title': 'Media Type',
                        },
                        'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
                    },
                    'required': ['data', 'media_type'],
                    'title': 'BinaryImage',
                    'type': 'object',
                },
            ]
        }
    )


async def test_override_output_json_schema():
    agent = Agent('test')
    assert agent.output_json_schema() == snapshot({'type': 'string'})
    output_type = [ToolOutput(bool)]
    assert agent.output_json_schema(output_type=output_type) == snapshot({'type': 'boolean'})


async def test_deferred_output_json_schema():
    agent = Agent('test', output_type=[str, DeferredToolRequests])
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {'type': 'string'},
                {
                    'properties': {
                        'calls': {'items': {'$ref': '#/$defs/ToolCallPart'}, 'title': 'Calls', 'type': 'array'},
                        'approvals': {'items': {'$ref': '#/$defs/ToolCallPart'}, 'title': 'Approvals', 'type': 'array'},
                        'metadata': {
                            'additionalProperties': {'additionalProperties': True, 'type': 'object'},
                            'title': 'Metadata',
                            'type': 'object',
                        },
                    },
                    'title': 'DeferredToolRequests',
                    'type': 'object',
                },
            ],
            '$defs': {
                'ToolCallPart': {
                    'description': 'A tool call from a model.',
                    'properties': {
                        'tool_name': {'title': 'Tool Name', 'type': 'string'},
                        'args': {
                            'anyOf': [
                                {'type': 'string'},
                                {'additionalProperties': True, 'type': 'object'},
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Args',
                        },
                        'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'},
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Id'},
                        'provider_details': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Provider Details',
                        },
                        'part_kind': {
                            'const': 'tool-call',
                            'default': 'tool-call',
                            'title': 'Part Kind',
                            'type': 'string',
                        },
                    },
                    'required': ['tool_name'],
                    'title': 'ToolCallPart',
                    'type': 'object',
                }
            },
        }
    )

    # special case of only BinaryImage and DeferredToolRequests
    agent = Agent('test', output_type=[BinaryImage, DeferredToolRequests])
    assert agent.output_json_schema() == snapshot(
        {
            'anyOf': [
                {
                    'properties': {
                        'data': {'format': 'binary', 'title': 'Data', 'type': 'string'},
                        'media_type': {
                            'anyOf': [
                                {'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'], 'type': 'string'},
                                {'type': 'string'},
                            ],
                            'title': 'Media Type',
                        },
                        'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
                    },
                    'required': ['data', 'media_type'],
                    'title': 'BinaryImage',
                    'type': 'object',
                },
                {
                    'properties': {
                        'calls': {'items': {'$ref': '#/$defs/ToolCallPart'}, 'title': 'Calls', 'type': 'array'},
                        'approvals': {'items': {'$ref': '#/$defs/ToolCallPart'}, 'title': 'Approvals', 'type': 'array'},
                        'metadata': {
                            'additionalProperties': {'additionalProperties': True, 'type': 'object'},
                            'title': 'Metadata',
                            'type': 'object',
                        },
                    },
                    'title': 'DeferredToolRequests',
                    'type': 'object',
                },
            ],
            '$defs': {
                'ToolCallPart': {
                    'description': 'A tool call from a model.',
                    'properties': {
                        'tool_name': {'title': 'Tool Name', 'type': 'string'},
                        'args': {
                            'anyOf': [
                                {'type': 'string'},
                                {'additionalProperties': True, 'type': 'object'},
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Args',
                        },
                        'tool_call_id': {'title': 'Tool Call Id', 'type': 'string'},
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Id'},
                        'provider_details': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Provider Details',
                        },
                        'part_kind': {
                            'const': 'tool-call',
                            'default': 'tool-call',
                            'title': 'Part Kind',
                            'type': 'string',
                        },
                    },
                    'required': ['tool_name'],
                    'title': 'ToolCallPart',
                    'type': 'object',
                }
            },
        }
    )
