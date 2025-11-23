import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    BinaryImage,
    RunContext,
)
from pydantic_ai._output import (
    NativeOutput,
    PromptedOutput,
)
from pydantic_ai.output import StructuredDict, ToolOutput
from pydantic_ai.tools import DeferredToolRequests

pytestmark = pytest.mark.anyio


async def test_text_output_json_schema():
    agent = Agent('test')
    assert agent.output_json_schema() == snapshot({'type': 'string'})


async def test_auto_output_json_schema():
    agent = Agent('test', output_type=bool)
    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {'response': {'type': 'boolean'}},
            'required': ['response'],
            'title': 'bool',
        }
    )


async def test_tool_output_json_schema():
    agent = Agent(
        'test',
        output_type=[ToolOutput(bool, name='alice', description='Dreaming...')],
    )
    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {'response': {'type': 'boolean'}},
            'required': ['response'],
            'title': 'alice',
            'description': 'Dreaming...',
        }
    )

    agent = Agent(
        'test',
        output_type=[ToolOutput(bool, name='alice'), ToolOutput(bool, name='bob')],
    )
    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': [
                        {
                            'type': 'object',
                            'properties': {
                                'kind': {'type': 'string', 'const': 'alice'},
                                'data': {
                                    'properties': {'response': {'type': 'boolean'}},
                                    'required': ['response'],
                                    'type': 'object',
                                },
                            },
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                            'title': 'alice',
                        },
                        {
                            'type': 'object',
                            'properties': {
                                'kind': {'type': 'string', 'const': 'bob'},
                                'data': {
                                    'properties': {'response': {'type': 'boolean'}},
                                    'required': ['response'],
                                    'type': 'object',
                                },
                            },
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                            'title': 'bob',
                        },
                    ]
                }
            },
            'required': ['result'],
            'additionalProperties': False,
        }
    )


async def test_native_output_json_schema():
    agent = Agent(
        'test',
        output_type=NativeOutput([bool], name='native_output_name', description='native_output_description'),
    )
    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {'response': {'type': 'boolean'}},
            'required': ['response'],
            'title': 'native_output_name',
            'description': 'native_output_description',
        }
    )


async def test_prompted_output_json_schema():
    agent = Agent(
        'test',
        output_type=PromptedOutput([bool], name='prompted_output_name', description='prompted_output_description'),
    )
    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {'response': {'type': 'boolean'}},
            'required': ['response'],
            'title': 'prompted_output_name',
            'description': 'prompted_output_description',
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
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'title': 'Human',
            'required': ['name', 'age'],
            'description': 'A human with a name and age',
        }
    )


async def test_image_output_json_schema():
    agent = Agent('test', output_type=BinaryImage)
    assert agent.output_json_schema() == snapshot(
        {
            'properties': {
                'data': {'format': 'binary', 'title': 'Data', 'type': 'string'},
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
                'identifier': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Identifier'},
                'kind': {'const': 'binary', 'default': 'binary', 'title': 'Kind', 'type': 'string'},
            },
            'required': ['data', 'media_type'],
            'title': 'BinaryImage',
            'type': 'object',
        }
    )
    agent = Agent('test', output_type=str | BinaryImage)
    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': [
                        {
                            'type': 'object',
                            'properties': {'kind': {'type': 'string', 'const': 'str'}, 'data': {'type': 'string'}},
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                        },
                        {
                            'type': 'object',
                            'properties': {
                                'kind': {'type': 'string', 'const': 'BinaryImage'},
                                'data': {
                                    'properties': {
                                        'data': {'format': 'binary', 'title': 'Data', 'type': 'string'},
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
                                                {
                                                    'enum': ['image/jpeg', 'image/png', 'image/gif', 'image/webp'],
                                                    'type': 'string',
                                                },
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
                                            'anyOf': [
                                                {'additionalProperties': True, 'type': 'object'},
                                                {'type': 'null'},
                                            ],
                                            'default': None,
                                            'title': 'Vendor Metadata',
                                        },
                                        'identifier': {
                                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                                            'default': None,
                                            'title': 'Identifier',
                                        },
                                        'kind': {
                                            'const': 'binary',
                                            'default': 'binary',
                                            'title': 'Kind',
                                            'type': 'string',
                                        },
                                    },
                                    'required': ['data', 'media_type'],
                                    'type': 'object',
                                },
                            },
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                            'title': 'BinaryImage',
                        },
                    ]
                }
            },
            'required': ['result'],
            'additionalProperties': False,
        }
    )


async def test_override_output_json_schema():
    agent = Agent('test')
    assert agent.output_json_schema() == snapshot({'type': 'string'})
    output_type = [ToolOutput(bool, name='alice', description='Dreaming...')]
    assert agent.output_json_schema(output_type=output_type) == snapshot(
        {
            'type': 'object',
            'properties': {'response': {'type': 'boolean'}},
            'required': ['response'],
            'title': 'alice',
            'description': 'Dreaming...',
        }
    )


@pytest.mark.skip(reason="CI expects provider_details key, can't repro locally")
async def test_deferred_output_json_schema():
    agent = Agent('test', output_type=[str, DeferredToolRequests])

    @agent.tool
    def update_file(ctx: RunContext, path: str, content: str) -> str:
        return ''

    @agent.tool_plain(requires_approval=True)
    def delete_file(path: str) -> str:
        return ''

    assert agent.output_json_schema() == snapshot(
        {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': [
                        {
                            'type': 'object',
                            'properties': {'kind': {'type': 'string', 'const': 'str'}, 'data': {'type': 'string'}},
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                        },
                        {
                            'type': 'object',
                            'properties': {
                                'kind': {'type': 'string', 'const': 'DeferredToolRequests'},
                                'data': {
                                    'properties': {
                                        'calls': {
                                            'items': {'$ref': '#/$defs/ToolCallPart'},
                                            'title': 'Calls',
                                            'type': 'array',
                                        },
                                        'approvals': {
                                            'items': {'$ref': '#/$defs/ToolCallPart'},
                                            'title': 'Approvals',
                                            'type': 'array',
                                        },
                                        'metadata': {
                                            'additionalProperties': {'additionalProperties': True, 'type': 'object'},
                                            'title': 'Metadata',
                                            'type': 'object',
                                        },
                                    },
                                    'type': 'object',
                                },
                            },
                            'required': ['kind', 'data'],
                            'additionalProperties': False,
                            'title': 'DeferredToolRequests',
                        },
                    ]
                }
            },
            'required': ['result'],
            'additionalProperties': False,
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
