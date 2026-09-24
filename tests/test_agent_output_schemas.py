import dataclasses
import json
from typing import Annotated, Any, Optional

import pytest
from pydantic import AfterValidator, BaseModel, Field
from typing_extensions import TypeAliasType

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
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.tools import ToolDefinition

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


class Ticket(BaseModel):
    """Triage a ticket."""

    urgent: bool


class Escalation(BaseModel):
    to: str


class Thread(BaseModel):
    """Reply to a thread."""

    ticket: Ticket


@pytest.mark.parametrize(
    'output_type, expected_tools',
    [
        pytest.param(
            [Annotated[Ticket, Field(description='An urgent ticket.')], Escalation],
            snapshot(
                [
                    ('final_result_Ticket', 'An urgent ticket.'),
                    ('final_result_Escalation', 'Escalation: The final response which ends this conversation'),
                ]
            ),
            id='field_description',
        ),
        pytest.param(
            [Annotated[Ticket, Field()], Escalation],
            snapshot(
                [
                    ('final_result_Ticket', 'Triage a ticket.'),
                    ('final_result_Escalation', 'Escalation: The final response which ends this conversation'),
                ]
            ),
            id='docstring',
        ),
        pytest.param(
            [Annotated[Ticket, Field(description='A ticket.')], Annotated[Escalation, Field(description='Escalate.')]],
            snapshot([('final_result_Ticket', 'A ticket.'), ('final_result_Escalation', 'Escalate.')]),
            id='two_annotated',
        ),
        pytest.param(
            [Annotated[Annotated[Ticket, Field(description='Inner.')], Field(description='Outer.')], Escalation],
            snapshot(
                [
                    ('final_result_Ticket', 'Outer.'),
                    ('final_result_Escalation', 'Escalation: The final response which ends this conversation'),
                ]
            ),
            id='nested_annotated',
        ),
        pytest.param(
            [Annotated[Ticket, Field(title='UrgentTicket')], Escalation],
            snapshot(
                [
                    ('final_result_UrgentTicket', 'Triage a ticket.'),
                    ('final_result_Escalation', 'Escalation: The final response which ends this conversation'),
                ]
            ),
            id='field_title',
        ),
        pytest.param(
            [Annotated[Thread, Field(description='A thread.')], Escalation],
            snapshot(
                [
                    ('final_result_Thread', 'A thread.'),
                    ('final_result_Escalation', 'Escalation: The final response which ends this conversation'),
                ]
            ),
            id='model_with_refs',
        ),
        pytest.param(
            [Ticket, Annotated[None, Field(description='Nothing needs doing.')]],
            snapshot(
                [
                    ('final_result_Ticket', 'Triage a ticket.'),
                    ('final_result_None', 'None: The final response which ends this conversation'),
                ]
            ),
            id='none',
        ),
    ],
)
async def test_annotated_output_tool_name_and_description(output_type: Any, expected_tools: list[tuple[str, str]]):
    """An `Annotated[X, ...]` union member is named and described after `X`, with a `Field(description=...)` winning.

    Not a VCR test: the output tool definitions are what's under test, and they're built before any request is sent.
    """
    model = TestModel()
    agent: Agent[None, Any] = Agent(model, output_type=output_type)
    await agent.run('Triage this ticket.')

    params = model.last_model_request_parameters
    assert params is not None
    assert [(tool.name, tool.description) for tool in params.output_tools] == expected_tools


async def test_annotated_output_tool_schema_and_validation():
    """An `Annotated` model is offered as the model's own schema with the annotation's keywords added, and its
    metadata still validates the output."""

    def require_urgent(ticket: Ticket) -> Ticket:
        if not ticket.urgent:
            raise ValueError('Only urgent tickets can be triaged.')
        return ticket

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert [tool.parameters_json_schema for tool in info.output_tools] == snapshot(
            [
                {
                    'properties': {'urgent': {'type': 'boolean'}},
                    'required': ['urgent'],
                    'title': 'Ticket',
                    'type': 'object',
                    'examples': [{'urgent': True}],
                },
                {
                    'properties': {'to': {'type': 'string'}},
                    'required': ['to'],
                    'title': 'Escalation',
                    'type': 'object',
                },
            ]
        )
        retried = any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts)
        return ModelResponse(parts=[ToolCallPart('final_result_Ticket', {'urgent': retried})])

    # Type checkers read an `Annotated[...]` expression as the `Annotated` special form rather than a type.
    output_type: Any = [
        Annotated[Ticket, AfterValidator(require_urgent), Field(examples=[{'urgent': True}])],
        Escalation,
    ]
    agent: Agent[None, Any] = Agent(FunctionModel(respond), output_type=output_type)
    result = await agent.run('Triage this ticket.')
    assert result.output == Ticket(urgent=True)


async def test_native_output_union_with_annotated_member():
    """A `NativeOutput` union member written as `Annotated[X, ...]` is named after `X` and resolves to it."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.output_object is not None
        [ticket_schema, _] = info.model_request_parameters.output_object.json_schema['properties']['result']['anyOf']
        assert ticket_schema['title'] == 'Ticket'
        assert ticket_schema['description'] == 'An urgent ticket.'
        return ModelResponse(parts=[TextPart(json.dumps({'result': {'kind': 'Ticket', 'data': {'urgent': True}}}))])

    # Type checkers read an `Annotated[...]` expression as the `Annotated` special form rather than a type.
    outputs: Any = [Annotated[Ticket, Field(description='An urgent ticket.')], Escalation]
    agent: Agent[None, Any] = Agent(FunctionModel(respond), output_type=NativeOutput(outputs))
    result = await agent.run('Triage this ticket.')
    assert result.output == Ticket(urgent=True)


def require_urgent(ticket: Ticket) -> Ticket:
    if not ticket.urgent:
        raise ValueError('Only urgent tickets can be triaged.')
    return ticket


UrgentTicket = Annotated[Ticket, AfterValidator(require_urgent), Field(description='An urgent ticket.')]
DescribedNone = Annotated[None, Field(description='Nothing needs doing.')]
TicketOrEscalation = TypeAliasType('TicketOrEscalation', UrgentTicket | Escalation)
Answer = Annotated[str, Field(description='A plain answer.', min_length=5)]


def urgent_on_retry(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call the `Ticket` output tool with a non-urgent ticket, and with an urgent one once the validator objected."""
    retried = any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts)
    return ModelResponse(parts=[ToolCallPart('final_result_Ticket', {'urgent': retried})])


# Type checkers read an `Annotated[...]` expression as the `Annotated` special form rather than a type, hence `Any`.
UNION_SPELLINGS: list[Any] = [
    pytest.param(
        [UrgentTicket, Escalation],
        UrgentTicket | Escalation,
        ['final_result_Ticket', 'final_result_Escalation'],
        id='union',
    ),
    pytest.param(
        [UrgentTicket, Escalation],
        TicketOrEscalation,
        ['final_result_Ticket', 'final_result_Escalation'],
        id='alias',
    ),
    pytest.param(
        [UrgentTicket, None],
        Optional[UrgentTicket],  # noqa: UP045
        ['final_result_Ticket', 'final_result_None'],
        id='optional',
    ),
    pytest.param(
        [Ticket, DescribedNone],
        Ticket | DescribedNone,
        ['final_result_Ticket', 'final_result_None'],
        id='described_none',
    ),
    # An annotated `str` is a value to validate, so it gets an output tool rather than the plain text a bare `str` allows.
    pytest.param([Ticket, Answer], Ticket | Answer, ['final_result_Ticket', 'final_result_str'], id='annotated_str'),
]


@pytest.mark.parametrize('listed, union, names', UNION_SPELLINGS)
async def test_annotated_union_member_matches_list(listed: Any, union: Any, names: list[str]):
    """`X | Y` offers the model exactly the output tools `[X, Y]` does, `Annotated` metadata included.

    Not a VCR test: the output tool definitions are built before any request is sent.
    """
    tools: list[list[ToolDefinition]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools.append(info.output_tools)
        return ModelResponse(parts=[ToolCallPart('final_result_Ticket', {'urgent': True})])

    for output_type in (listed, union):
        await Agent(FunctionModel(respond), output_type=output_type).run('Triage this ticket.')
    assert tools[0] == tools[1]
    assert [tool.name for tool in tools[1]] == names
    assert (
        Agent('test', output_type=listed).output_json_schema() == Agent('test', output_type=union).output_json_schema()
    )


@pytest.mark.parametrize(
    'output_type',
    [
        pytest.param([UrgentTicket, Escalation], id='list'),
        pytest.param(UrgentTicket | Escalation, id='union'),
        pytest.param(Optional[UrgentTicket], id='optional'),  # noqa: UP045
    ],
)
async def test_annotated_union_member_validates_and_describes(output_type: Any):
    """An `Annotated` member's validator runs and its description reaches the tool, however the union is spelled."""
    descriptions: list[str | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        descriptions.append(info.output_tools[0].description)
        return urgent_on_retry(messages, info)

    agent: Agent[None, Any] = Agent(FunctionModel(respond), output_type=output_type)
    result = await agent.run('Triage this ticket.')

    assert result.output == Ticket(urgent=True)
    assert descriptions == ['An urgent ticket.', 'An urgent ticket.']


@pytest.mark.parametrize('marker', [NativeOutput, PromptedOutput])
async def test_structured_output_union_annotated_member(marker: type[NativeOutput[Any] | PromptedOutput[Any]]):
    """A `NativeOutput` or `PromptedOutput` of `X | Y` keeps an `Annotated` member's validator and description."""
    descriptions: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.output_object is not None
        [ticket_schema, _] = info.model_request_parameters.output_object.json_schema['properties']['result']['anyOf']
        descriptions.append(ticket_schema['description'])
        retried = any(isinstance(part, RetryPromptPart) for message in messages for part in message.parts)
        return ModelResponse(parts=[TextPart(json.dumps({'result': {'kind': 'Ticket', 'data': {'urgent': retried}}}))])

    output_type: Any = UrgentTicket | Escalation
    agent: Agent[None, Any] = Agent(FunctionModel(respond), output_type=marker(output_type))
    result = await agent.run('Triage this ticket.')

    assert result.output == Ticket(urgent=True)
    assert descriptions == ['An urgent ticket.', 'An urgent ticket.']


@pytest.mark.parametrize(
    'output_type',
    [
        pytest.param([Ticket, DescribedNone], id='list'),
        pytest.param(Ticket | DescribedNone, id='union'),
    ],
)
async def test_described_none_output(output_type: Any):
    """A described `None` is still `None`: an empty response is its answer, and its description reaches its tool."""
    tools: list[ToolDefinition] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools.extend(info.output_tools)
        return ModelResponse(parts=[])

    agent: Agent[None, Any] = Agent(FunctionModel(respond), output_type=output_type)
    result = await agent.run('Triage this ticket.')

    assert result.output is None
    assert [(tool.name, tool.parameters_json_schema) for tool in tools] == snapshot(
        [
            (
                'final_result_Ticket',
                {
                    'properties': {'urgent': {'type': 'boolean'}},
                    'required': ['urgent'],
                    'title': 'Ticket',
                    'type': 'object',
                },
            ),
            (
                'final_result_None',
                {
                    'properties': {'response': {'description': 'Nothing needs doing.', 'type': 'null'}},
                    'required': ['response'],
                    'type': 'object',
                },
            ),
        ]
    )
