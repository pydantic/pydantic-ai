import dataclasses
import json
import re

import pytest
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import (
    Agent,
    BinaryImage,
    DeferredToolRequests,
    NativeOutput,
    PromptedOutput,
    StructuredDict,
    TextOutput,
    ToolOutput,
    UserError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputObjectDefinition

from ._inline_snapshot import snapshot
from .conftest import remove_schema_descriptions

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

    def func_no_return_type_hint(x: int):
        return x  # pragma: no cover

    agent = Agent('test', output_type=[func_no_return_type_hint])
    assert agent.output_json_schema() == snapshot({'type': 'string'})


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


class Fruit(BaseModel):
    """A fruit"""

    name: str
    color: str


class Vehicle(BaseModel):
    """A vehicle"""

    name: str
    wheels: int


async def test_native_output_union_preserves_description():
    """A union `NativeOutput` keeps its own `name`/`description`, not the last member's title/docstring (issue #6262).

    Taps the internal `output_object` rather than being a VCR test because a cassette matcher isn't sensitive to the
    request-body schema `description` field, so a VCR test asserting only `result.output` would pass green even with the bug.
    """
    captured: OutputObjectDefinition | None = None

    async def capture(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal captured
        captured = info.model_request_parameters.output_object
        return ModelResponse(
            parts=[
                TextPart(
                    content=json.dumps({'result': {'kind': 'Fruit', 'data': {'name': 'banana', 'color': 'yellow'}}})
                )
            ]
        )

    agent = Agent(
        FunctionModel(function=capture),
        output_type=NativeOutput([Fruit, Vehicle], name='Fruit or vehicle', description='Return a fruit or vehicle.'),
    )
    result = await agent.run('What is a banana?')

    assert result.output == Fruit(name='banana', color='yellow')
    assert captured is not None
    assert captured.name == 'Fruit or vehicle'
    assert captured.description == 'Return a fruit or vehicle.'


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


async def test_structured_dict_non_recursive_defs():
    """Non-recursive `$defs` are inlined, and the schema is still exposed via the pydantic hook."""

    class Address(BaseModel):
        street: str

    class Person(BaseModel):
        name: str
        address: Address

    PersonDict = StructuredDict(Person.model_json_schema())

    # The pydantic hook returns the inlined schema, so `StructuredDict` can be used as a type elsewhere.
    assert TypeAdapter(PersonDict).json_schema() == snapshot(
        {
            'properties': {
                'name': {'title': 'Name', 'type': 'string'},
                'address': {
                    'properties': {'street': {'title': 'Street', 'type': 'string'}},
                    'required': ['street'],
                    'title': 'Address',
                    'type': 'object',
                },
            },
            'required': ['name', 'address'],
            'title': 'Person',
            'type': 'object',
        }
    )

    agent = Agent('test', output_type=PersonDict)
    assert agent.output_json_schema() == snapshot(
        {
            'properties': {
                'name': {'title': 'Name', 'type': 'string'},
                'address': {
                    'properties': {'street': {'title': 'Street', 'type': 'string'}},
                    'required': ['street'],
                    'title': 'Address',
                    'type': 'object',
                },
            },
            'required': ['name', 'address'],
            'title': 'Person',
            'type': 'object',
        }
    )


async def test_structured_dict_recursive_refs_sent_to_model():
    """A recursive `StructuredDict`'s `$defs` survive into the schema the model is actually sent (issue #4018).

    Not a VCR test: the tool schema sent to the model is built by a different code path than
    `agent.output_json_schema()`, and our cassette matchers aren't sensitive to the request body, so the
    recursive `$defs` could be dropped or emptied and a recorded run would still match and pass.
    """
    # The recursive union reported in the issue: `data` holds an arbitrarily nested JSON value.
    json_schema = {
        'type': 'object',
        'title': 'Output',
        'properties': {
            'name': {'type': 'string'},
            'data': {'$ref': '#/$defs/JSONValue'},
        },
        'required': ['name', 'data'],
        '$defs': {
            'JSONValue': {
                'anyOf': [
                    {'type': 'string'},
                    {'type': 'integer'},
                    {'type': 'boolean'},
                    {'type': 'null'},
                    {'type': 'array', 'items': {'$ref': '#/$defs/JSONValue'}},
                    {'$ref': '#/$defs/Map'},
                ],
            },
            'Map': {
                'type': 'object',
                'title': 'Map',
                'properties': {'entries': {'type': 'array', 'items': {'$ref': '#/$defs/JSONValue'}}},
                'required': ['entries'],
            },
        },
    }

    model = TestModel(custom_output_args={'name': 'test', 'data': ['hello', {'entries': [1]}]})
    agent = Agent(model, output_type=StructuredDict(json_schema))
    result = await agent.run('Return some data')
    assert result.output == snapshot({'name': 'test', 'data': ['hello', {'entries': [1]}]})

    # `Map` is inlined into the recursive `JSONValue` def, which keeps its full `anyOf` and stays in `$defs`.
    request_parameters = model.last_model_request_parameters
    assert request_parameters is not None
    assert request_parameters.output_tools[0].parameters_json_schema == snapshot(
        {
            'type': 'object',
            'title': 'Output',
            'properties': {
                'name': {'type': 'string'},
                'data': {
                    'anyOf': [
                        {'type': 'string'},
                        {'type': 'integer'},
                        {'type': 'boolean'},
                        {'type': 'null'},
                        {'type': 'array', 'items': {'$ref': '#/$defs/JSONValue'}},
                        {
                            'type': 'object',
                            'title': 'Map',
                            'properties': {'entries': {'type': 'array', 'items': {'$ref': '#/$defs/JSONValue'}}},
                            'required': ['entries'],
                        },
                    ]
                },
            },
            'required': ['name', 'data'],
            '$defs': {
                'JSONValue': {
                    'anyOf': [
                        {'type': 'string'},
                        {'type': 'integer'},
                        {'type': 'boolean'},
                        {'type': 'null'},
                        {'type': 'array', 'items': {'$ref': '#/$defs/JSONValue'}},
                        {
                            'type': 'object',
                            'title': 'Map',
                            'properties': {'entries': {'type': 'array', 'items': {'$ref': '#/$defs/JSONValue'}}},
                            'required': ['entries'],
                        },
                    ]
                }
            },
        }
    )


async def test_structured_dict_recursive_refs_nested_in_other_type():
    """A recursive `StructuredDict` can't go through `TypeAdapter`, so nesting it raises a helpful error.

    Not a VCR test: this is a pre-request guard, no model is ever called.
    """

    class Node(BaseModel):
        nodes: list['Node']

    NodeDict = StructuredDict(Node.model_json_schema())

    with pytest.raises(
        UserError,
        match=re.escape(
            'A `StructuredDict` with recursive `$ref`s and `$defs` can only be used as an `output_type` by itself, not nested inside another type.'
        ),
    ):
        TypeAdapter(NodeDict).json_schema()


async def test_structured_dict_recursive_root_key_collision():
    """A root whose title collides with a recursive `$defs` key resolves without clobbering that key.

    Not a VCR test: this pins the schema-building internals, which a cassette matcher wouldn't be sensitive to.
    """
    schema = StructuredDict(
        {
            'type': 'object',
            'title': 'Node',
            'properties': {'child': {'$ref': '#/$defs/Node'}},
            'required': ['child'],
            '$defs': {
                'Node': {
                    'type': 'object',
                    'title': 'Node',
                    'properties': {'child': {'$ref': '#/$defs/Node'}},
                    'required': ['child'],
                }
            },
        }
    )
    agent = Agent('test', output_type=schema)
    assert agent.output_json_schema() == snapshot(
        {
            '$defs': {
                'Node': {
                    'properties': {'child': {'$ref': '#/$defs/Node'}},
                    'required': ['child'],
                    'title': 'Node',
                    'type': 'object',
                }
            },
            'properties': {
                'child': {
                    'type': 'object',
                    'title': 'Node',
                    'properties': {'child': {'$ref': '#/$defs/Node'}},
                    'required': ['child'],
                }
            },
            'required': ['child'],
            'title': 'Node',
            'type': 'object',
        }
    )


async def test_image_output_json_schema():
    # one output
    agent = Agent('test', output_type=BinaryImage)
    assert agent.output_json_schema() == snapshot(
        {
            'description': "Binary content that's guaranteed to be an image.",
            'properties': {
                'data': {'format': 'base64url', 'title': 'Data', 'type': 'string'},
                'media_type': {
                    'anyOf': [
                        {
                            'enum': ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/aiff', 'audio/aac'],
                            'type': 'string',
                        },
                        {'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'], 'type': 'string'},
                        {
                            'enum': [
                                'application/pdf',
                                'text/plain',
                                'text/csv',
                                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                'text/html',
                                'text/markdown',
                                'application/msword',
                                'application/vnd.ms-excel',
                            ],
                            'type': 'string',
                        },
                        {'type': 'string'},
                    ],
                    'title': 'Media Type',
                },
                'vendor_metadata': {
                    'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Vendor Metadata',
                },
                'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
                'identifier': {
                    'description': """\
Identifier for the binary content, such as a unique ID.

This identifier can be provided to the model in a message to allow it to refer to this file in a tool call argument,
and the tool can look up the file in question by iterating over the message history and finding the matching `BinaryContent`.

This identifier is only automatically passed to the model when the `BinaryContent` is returned by a tool.
If you're passing the `BinaryContent` as a user message, it's up to you to include a separate text part with the identifier,
e.g. "This is file <identifier>:" preceding the `BinaryContent`.

It's also included in inline-text delimiters for providers that require inlining text documents, so the model can
distinguish multiple files.\
""",
                    'readOnly': True,
                    'title': 'Identifier',
                    'type': 'string',
                },
            },
            'required': ['data', 'media_type', 'identifier'],
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
                    'description': "Binary content that's guaranteed to be an image.",
                    'properties': {
                        'data': {'format': 'base64url', 'title': 'Data', 'type': 'string'},
                        'media_type': {
                            'anyOf': [
                                {
                                    'enum': [
                                        'audio/wav',
                                        'audio/mpeg',
                                        'audio/ogg',
                                        'audio/flac',
                                        'audio/aiff',
                                        'audio/aac',
                                    ],
                                    'type': 'string',
                                },
                                {'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'], 'type': 'string'},
                                {
                                    'enum': [
                                        'application/pdf',
                                        'text/plain',
                                        'text/csv',
                                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        'text/html',
                                        'text/markdown',
                                        'application/msword',
                                        'application/vnd.ms-excel',
                                    ],
                                    'type': 'string',
                                },
                                {'type': 'string'},
                            ],
                            'title': 'Media Type',
                        },
                        'vendor_metadata': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Vendor Metadata',
                        },
                        'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
                        'identifier': {
                            'description': """\
Identifier for the binary content, such as a unique ID.

This identifier can be provided to the model in a message to allow it to refer to this file in a tool call argument,
and the tool can look up the file in question by iterating over the message history and finding the matching `BinaryContent`.

This identifier is only automatically passed to the model when the `BinaryContent` is returned by a tool.
If you're passing the `BinaryContent` as a user message, it's up to you to include a separate text part with the identifier,
e.g. "This is file <identifier>:" preceding the `BinaryContent`.

It's also included in inline-text delimiters for providers that require inlining text documents, so the model can
distinguish multiple files.\
""",
                            'readOnly': True,
                            'title': 'Identifier',
                            'type': 'string',
                        },
                    },
                    'required': ['data', 'media_type', 'identifier'],
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
    assert remove_schema_descriptions(agent.output_json_schema()) == snapshot(
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
                        'tool_kind': {
                            'anyOf': [
                                {'enum': ['tool-search', 'capability-load'], 'type': 'string'},
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Tool Kind',
                        },
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Id'},
                        'provider_name': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Provider Name',
                        },
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
    assert remove_schema_descriptions(agent.output_json_schema()) == snapshot(
        {
            'anyOf': [
                {
                    'properties': {
                        'data': {'format': 'base64url', 'title': 'Data', 'type': 'string'},
                        'media_type': {
                            'anyOf': [
                                {
                                    'enum': [
                                        'audio/wav',
                                        'audio/mpeg',
                                        'audio/ogg',
                                        'audio/flac',
                                        'audio/aiff',
                                        'audio/aac',
                                    ],
                                    'type': 'string',
                                },
                                {'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'], 'type': 'string'},
                                {
                                    'enum': [
                                        'application/pdf',
                                        'text/plain',
                                        'text/csv',
                                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                                        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        'text/html',
                                        'text/markdown',
                                        'application/msword',
                                        'application/vnd.ms-excel',
                                    ],
                                    'type': 'string',
                                },
                                {'type': 'string'},
                            ],
                            'title': 'Media Type',
                        },
                        'vendor_metadata': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Vendor Metadata',
                        },
                        'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
                        'identifier': {
                            'readOnly': True,
                            'title': 'Identifier',
                            'type': 'string',
                        },
                    },
                    'required': ['data', 'media_type', 'identifier'],
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
                        'tool_kind': {
                            'anyOf': [
                                {'enum': ['tool-search', 'capability-load'], 'type': 'string'},
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Tool Kind',
                        },
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Id'},
                        'provider_name': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Provider Name',
                        },
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


# Pydantic suppresses stdlib dataclass docstrings from JSON schemas.
# These tests document the current behavior; see https://github.com/pydantic/pydantic/issues/12812
# regression test for https://github.com/pydantic/pydantic-ai/pull/4138#discussion_r2819140514


class BMWithDoc(BaseModel):
    """The result with name and score."""

    name: str
    score: int


@dataclasses.dataclass
class DCWithDoc:
    """The result with name and score."""

    name: str
    score: int = 0


class BMNested(BaseModel):
    """Nested filter criteria."""

    category: str = 'all'


@dataclasses.dataclass
class DCNested:
    """Nested filter criteria."""

    category: str = 'all'


class BMWithNestedField(BaseModel):
    """Output with nested model."""

    filters: BMNested


@dataclasses.dataclass
class DCWithNestedField:
    """Output with nested dataclass."""

    filters: DCNested


@pytest.mark.parametrize(
    'output_type, expected_schema',
    [
        pytest.param(
            BMWithDoc,
            snapshot(
                {
                    'properties': {
                        'name': {'title': 'Name', 'type': 'string'},
                        'score': {'title': 'Score', 'type': 'integer'},
                    },
                    'required': ['name', 'score'],
                    'title': 'BMWithDoc',
                    'type': 'object',
                }
            ),
            id='basemodel',
        ),
        pytest.param(
            DCWithDoc,
            snapshot(
                {
                    'properties': {
                        'name': {'title': 'Name', 'type': 'string'},
                        'score': {'default': 0, 'title': 'Score', 'type': 'integer'},
                    },
                    'required': ['name'],
                    'title': 'DCWithDoc',
                    'type': 'object',
                }
            ),
            id='dataclass',
        ),
    ],
)
async def test_output_type_description(output_type: type, expected_schema: dict[str, object]):
    agent: Agent[object, str] = Agent('test', output_type=output_type)
    assert remove_schema_descriptions(agent.output_json_schema()) == expected_schema


@pytest.mark.parametrize(
    'output_type, expected_schema',
    [
        pytest.param(
            BMWithNestedField,
            snapshot(
                {
                    '$defs': {
                        'BMNested': {
                            'properties': {'category': {'default': 'all', 'title': 'Category', 'type': 'string'}},
                            'title': 'BMNested',
                            'type': 'object',
                        }
                    },
                    'properties': {'filters': {'$ref': '#/$defs/BMNested'}},
                    'required': ['filters'],
                    'title': 'BMWithNestedField',
                    'type': 'object',
                }
            ),
            id='basemodel_nested',
        ),
        pytest.param(
            DCWithNestedField,
            snapshot(
                {
                    '$defs': {
                        'DCNested': {
                            'properties': {'category': {'default': 'all', 'title': 'Category', 'type': 'string'}},
                            'title': 'DCNested',
                            'type': 'object',
                        }
                    },
                    'properties': {'filters': {'$ref': '#/$defs/DCNested'}},
                    'required': ['filters'],
                    'title': 'DCWithNestedField',
                    'type': 'object',
                }
            ),
            id='dataclass_nested',
        ),
    ],
)
async def test_nested_output_type_description(output_type: type, expected_schema: dict[str, object]):
    agent: Agent[object, str] = Agent('test', output_type=output_type)
    assert remove_schema_descriptions(agent.output_json_schema()) == expected_schema
