"""Tests for capability specifications, schemas, construction, and merging.

Split out of `test_capabilities.py` per #7304.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai import Capability as TopLevelCapability
from pydantic_ai._run_context import RunContext
from pydantic_ai._spec import CapabilitySpec
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    MCP,
    Capability,
    ImageGeneration,
    IncludeToolReturnSchemas,
    Instrumentation,
    NativeTool,
    PrefixTools,
    RaiseContentFilterError,
    ReinjectSystemPrompt,
    SetToolMetadata,
    Thinking,
    ToolSearch,
    WebFetch,
    WebSearch,
    XSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.native_tool import NativeTool as NativeToolCap
from pydantic_ai.exceptions import (
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
)
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.settings import ModelSettings as _ModelSettings

from ._inline_snapshot import snapshot
from .capability_models import (
    CustomCapability,
    ToolsetFuncCapability,
    make_text_response,
    noop_greet as _noop_greet,
    registered_capability_context as _registered_capability_context,
)
from .conftest import iter_message_parts, remove_schema_descriptions

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


def test_capability_top_level_export() -> None:
    assert TopLevelCapability is Capability


def test_capability_types() -> None:
    assert CAPABILITY_TYPES == snapshot(
        {
            'NativeTool': NativeTool,
            'RaiseContentFilterError': RaiseContentFilterError,
            'ImageGeneration': ImageGeneration,
            'IncludeToolReturnSchemas': IncludeToolReturnSchemas,
            'Instrumentation': Instrumentation,
            'MCP': MCP,
            'PrefixTools': PrefixTools,
            'ReinjectSystemPrompt': ReinjectSystemPrompt,
            'SetToolMetadata': SetToolMetadata,
            'Thinking': Thinking,
            'ToolSearch': ToolSearch,
            'WebFetch': WebFetch,
            'WebSearch': WebSearch,
            'XSearch': XSearch,
        }
    )


def test_instrumentation_default_settings() -> None:
    """`Instrumentation()` lazy-imports `InstrumentationSettings` and constructs default settings."""
    from pydantic_ai.models.instrumented import InstrumentationSettings

    instr = Instrumentation()
    assert isinstance(instr.settings, InstrumentationSettings)


def test_agent_from_spec_basic():
    """Test Agent.from_spec with basic capabilities."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'instructions': 'You are a helpful agent.',
            'model_settings': {'max_tokens': 4096},
            'capabilities': [
                {'WebSearch': {'local': 'duckduckgo'}},
            ],
        }
    )
    assert agent.model is not None


def test_agent_from_spec_no_capabilities():
    """Test Agent.from_spec with no capabilities."""
    agent = Agent.from_spec({'model': 'test'})
    assert agent.model is not None


def test_agent_from_spec_image_generation():
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [{'ImageGeneration': {'local': False}}],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, ImageGeneration))
    assert cap.local is False


def test_agent_from_spec_web_fetch():
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [{'WebFetch': {'allowed_domains': ['example.com'], 'max_uses': 5, 'local': True}}],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, WebFetch))
    assert cap.allowed_domains == ['example.com']
    assert cap.max_uses == 5


def test_agent_from_spec_mcp():
    pytest.importorskip('mcp', reason='mcp package not installed')
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {
                    'MCP': {
                        'url': 'https://mcp.example.com/sse',
                        'allowed_tools': ['search'],
                        'native': True,
                        'id': 'search-mcp',
                        'description': 'Search MCP server.',
                        'defer_loading': True,
                    }
                }
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, MCP))
    assert cap.url == 'https://mcp.example.com/sse'
    assert cap.allowed_tools == ['search']
    assert cap.id == 'search-mcp'
    assert cap.description == 'Search MCP server.'
    assert cap.defer_loading is True


def test_agent_from_spec_unknown_capability():
    """Test Agent.from_spec with an unknown capability name."""
    with pytest.raises(ValueError, match="Capability 'Unknown' is not in the provided"):
        Agent.from_spec(
            {
                'model': 'test',
                'capabilities': ['Unknown'],
            }
        )


def test_agent_from_spec_bad_args():
    """Test Agent.from_spec with bad arguments for a capability."""
    with pytest.raises(ValueError, match="Failed to instantiate capability 'WebSearch'"):
        Agent.from_spec(
            {
                'model': 'test',
                'capabilities': [
                    {'WebSearch': {'nonexistent_param': 'value'}},
                ],
            }
        )


@dataclass
class CapabilityWithCallbackParam(AbstractCapability):
    """Custom capability with a mix of serializable and non-serializable params."""

    max_retries: int = 3
    on_error: Callable[..., Any] = lambda: None  # purely Callable, filtered from schema
    verbose: Callable[..., Any] | bool = False  # Callable | bool, only bool survives in schema
    hooks: Callable[..., Any] | Callable[..., None] = lambda: None  # union of all non-serializable, entirely filtered


def test_agent_from_spec_custom_capability():
    """Test Agent.from_spec with a custom capability type."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'CustomCapability': 'world'},
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


def test_agent_from_spec_with_agent_spec_object():
    """Test Agent.from_spec with an AgentSpec instance."""
    spec = AgentSpec(
        model='test',
        instructions='You are helpful.',
        capabilities=[
            CapabilitySpec(name='WebSearch', arguments={'local': 'duckduckgo'}),
        ],
    )
    agent = Agent.from_spec(spec)
    assert agent.model is not None


def test_agent_from_spec_output_type():
    """Test Agent.from_spec with output_type parameter."""
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        name: str
        value: int

    agent = Agent.from_spec({'model': 'test'}, output_type=MyOutput)
    assert agent.output_type == MyOutput


def test_agent_from_spec_output_schema():
    """Test Agent.from_spec with output_schema in spec."""
    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'},
        },
        'required': ['name', 'age'],
    }
    agent = Agent.from_spec({'model': 'test', 'output_schema': schema})
    # output_type should be a StructuredDict subclass (dict subclass with JSON schema)
    assert agent.output_type is not str
    assert isinstance(agent.output_type, type) and issubclass(agent.output_type, dict)


def test_agent_from_spec_output_type_takes_precedence():
    """Test that output_type parameter takes precedence over output_schema in spec."""
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        name: str

    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'required': ['name'],
    }
    agent = Agent.from_spec({'model': 'test', 'output_schema': schema}, output_type=MyOutput)
    assert agent.output_type == MyOutput


def test_agent_from_spec_output_schema_invalid():
    """Test Agent.from_spec with a non-object output_schema raises UserError."""
    with pytest.raises(UserError, match='Schema must be an object'):
        Agent.from_spec({'model': 'test', 'output_schema': {'type': 'string'}})


async def test_agent_from_spec_output_schema_integration():
    """Test Agent.from_spec with output_schema produces dict output."""
    schema = {
        'type': 'object',
        'properties': {
            'city': {'type': 'string'},
            'country': {'type': 'string'},
        },
        'required': ['city', 'country'],
    }
    agent = Agent.from_spec({'model': 'test', 'output_schema': schema})
    result = await agent.run(
        'Tell me a city',
        model=TestModel(custom_output_args={'city': 'Paris', 'country': 'France'}),
    )
    assert result.output == {'city': 'Paris', 'country': 'France'}


def test_agent_from_spec_name():
    agent = Agent.from_spec({'model': 'test', 'name': 'my-agent'})
    assert agent.name == 'my-agent'


def test_agent_from_spec_name_override():
    agent = Agent.from_spec({'model': 'test', 'name': 'spec-name'}, name='override-name')
    assert agent.name == 'override-name'


def test_agent_from_spec_description():
    agent = Agent.from_spec({'model': 'test', 'description': 'A helpful agent'})
    assert agent.description == 'A helpful agent'


def test_agent_from_spec_description_override():
    agent = Agent.from_spec({'model': 'test', 'description': 'spec-desc'}, description='override-desc')
    assert agent.description == 'override-desc'


def test_agent_from_spec_instructions():
    agent = Agent.from_spec({'model': 'test', 'instructions': 'Be helpful.'})
    assert 'Be helpful.' in [sourced.instruction for sourced in agent._instructions]  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_list():
    agent = Agent.from_spec({'model': 'test', 'instructions': ['First.', 'Second.']})
    assert 'First.' in [sourced.instruction for sourced in agent._instructions]  # pyright: ignore[reportPrivateUsage]
    assert 'Second.' in [sourced.instruction for sourced in agent._instructions]  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_merged():
    agent = Agent.from_spec(
        {'model': 'test', 'instructions': 'From spec.'},
        instructions='From arg.',
    )
    assert 'From spec.' in [sourced.instruction for sourced in agent._instructions]  # pyright: ignore[reportPrivateUsage]
    assert 'From arg.' in [sourced.instruction for sourced in agent._instructions]  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_model_settings():
    agent = Agent.from_spec({'model': 'test', 'model_settings': {'temperature': 0.5, 'max_tokens': 100}})
    ms = agent.model_settings
    assert isinstance(ms, dict)
    assert ms.get('temperature') == 0.5  # pyright: ignore[reportUnknownMemberType]
    assert ms.get('max_tokens') == 100  # pyright: ignore[reportUnknownMemberType]


def test_agent_from_spec_model_settings_merged():
    agent = Agent.from_spec(
        {'model': 'test', 'model_settings': {'temperature': 0.5, 'max_tokens': 100}},
        model_settings={'temperature': 0.9},
    )
    ms = agent.model_settings
    assert isinstance(ms, dict)
    assert ms.get('temperature') == 0.9  # pyright: ignore[reportUnknownMemberType]
    assert ms.get('max_tokens') == 100  # pyright: ignore[reportUnknownMemberType]


def test_agent_from_spec_retries():
    agent = Agent.from_spec({'model': 'test', 'retries': 5})
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_retries_dict():
    agent = Agent.from_spec({'model': 'test', 'retries': {'tools': 2, 'output': 4}})
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 4  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_retries_override():
    agent = Agent.from_spec({'model': 'test', 'retries': 5}, retries=2)
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 2  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_no_retries_does_not_warn():
    """`from_spec` without an explicit retry budget uses the default budgets."""
    agent = Agent.from_spec({'model': 'test'})

    assert agent._max_tool_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 1  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_explicit_retries_does_not_warn():
    """`AgentSpec.retries` is canonical."""
    agent = Agent.from_spec({'model': 'test', 'retries': 5})
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_spec_retries_field():
    """`AgentSpec.retries` is the canonical field."""
    spec = AgentSpec(model='test', retries=5)
    assert spec.retries == 5


def test_agent_from_spec_end_strategy():
    agent = Agent.from_spec({'model': 'test', 'end_strategy': 'exhaustive'})
    assert agent.end_strategy == 'exhaustive'


def test_agent_from_spec_end_strategy_override():
    agent = Agent.from_spec({'model': 'test', 'end_strategy': 'exhaustive'}, end_strategy='early')
    assert agent.end_strategy == 'early'


def test_agent_from_spec_tool_timeout():
    agent = Agent.from_spec({'model': 'test', 'tool_timeout': 30.0})
    assert agent._tool_timeout == 30.0  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_tool_timeout_override():
    agent = Agent.from_spec({'model': 'test', 'tool_timeout': 30.0}, tool_timeout=5.0)
    assert agent._tool_timeout == 5.0  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_metadata():
    agent = Agent.from_spec({'model': 'test', 'metadata': {'env': 'prod', 'version': '1.0'}})
    assert agent._metadata == {'env': 'prod', 'version': '1.0'}  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_metadata_override():
    agent = Agent.from_spec(
        {'model': 'test', 'metadata': {'env': 'prod'}},
        metadata={'env': 'staging'},
    )
    assert agent._metadata == {'env': 'staging'}  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_model_override():
    agent = Agent.from_spec({'model': 'test'}, model='test')
    assert agent.model is not None


def test_agent_from_spec_capabilities_merged():
    @dataclass
    class ExtraCap(AbstractCapability):
        pass

    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [{'WebSearch': {'local': 'duckduckgo'}}],
        },
        capabilities=[ExtraCap()],
    )
    # Should have both the WebSearch capability from spec and ExtraCap from arg
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    assert any(isinstance(c, WebSearch) for c in children)
    assert any(isinstance(c, ExtraCap) for c in children)


def test_model_json_schema_with_capabilities():
    # Unit (not VCR): this pins the generated JSON-schema/capabilities mapping, which is built internally
    # from the known-model enum and never produced by any API response — no cassette could exercise it.
    pytest.importorskip('mcp', reason='schema varies without mcp package')
    schema = AgentSpec.model_json_schema_with_capabilities()
    assert remove_schema_descriptions(schema) == snapshot(
        {
            '$defs': {
                'AdvisorTool': {
                    'properties': {
                        'kind': {'default': 'advisor', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'model': {
                            'anyOf': [
                                {
                                    'enum': [
                                        'claude-fable-5',
                                        'claude-mythos-5',
                                        'claude-opus-5',
                                        'claude-opus-4-8',
                                        'claude-opus-4-7',
                                        'claude-opus-4-6',
                                        'claude-sonnet-4-6',
                                    ],
                                    'type': 'string',
                                },
                                {'type': 'string'},
                            ],
                            'title': 'Model',
                        },
                        'max_uses': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Max Uses',
                        },
                        'max_tokens': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Max Tokens',
                        },
                        'caching': {
                            'anyOf': [{'enum': ['5m', '1h'], 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Caching',
                        },
                    },
                    'required': ['model'],
                    'title': 'AdvisorTool',
                    'type': 'object',
                },
                'AgentRetries': {
                    'additionalProperties': False,
                    'properties': {
                        'tools': {'title': 'Tools', 'type': 'integer'},
                        'output': {'title': 'Output', 'type': 'integer'},
                    },
                    'title': 'AgentRetries',
                    'type': 'object',
                },
                'CodeExecutionTool': {
                    'properties': {
                        'kind': {'default': 'code_execution', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'files': {
                            'anyOf': [{'items': {'$ref': '#/$defs/UploadedFile'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Files',
                        },
                    },
                    'title': 'CodeExecutionTool',
                    'type': 'object',
                },
                'FileSearchTool': {
                    'properties': {
                        'kind': {'default': 'file_search', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'file_store_ids': {'items': {'type': 'string'}, 'title': 'File Store Ids', 'type': 'array'},
                        'max_num_results': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Max Num Results',
                        },
                        'instructions': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Instructions',
                        },
                        'retrieval_mode': {
                            'anyOf': [{'enum': ['hybrid', 'semantic', 'keyword'], 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Retrieval Mode',
                        },
                    },
                    'required': ['file_store_ids'],
                    'title': 'FileSearchTool',
                    'type': 'object',
                },
                'ImageGenerationTool': {
                    'properties': {
                        'kind': {'default': 'image_generation', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'action': {
                            'default': 'auto',
                            'enum': ['generate', 'edit', 'auto'],
                            'title': 'Action',
                            'type': 'string',
                        },
                        'background': {
                            'default': 'auto',
                            'enum': ['transparent', 'opaque', 'auto'],
                            'title': 'Background',
                            'type': 'string',
                        },
                        'input_fidelity': {
                            'anyOf': [{'enum': ['high', 'low'], 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Input Fidelity',
                        },
                        'moderation': {
                            'default': 'auto',
                            'enum': ['auto', 'low'],
                            'title': 'Moderation',
                            'type': 'string',
                        },
                        'model': {
                            'anyOf': [
                                {
                                    'enum': ['gpt-image-2', 'gpt-image-1.5', 'gpt-image-1', 'gpt-image-1-mini'],
                                    'type': 'string',
                                },
                                {'type': 'string'},
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Model',
                        },
                        'output_compression': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Output Compression',
                        },
                        'output_format': {
                            'anyOf': [{'enum': ['png', 'webp', 'jpeg'], 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Output Format',
                        },
                        'partial_images': {'default': 0, 'title': 'Partial Images', 'type': 'integer'},
                        'quality': {
                            'default': 'auto',
                            'enum': ['low', 'medium', 'high', 'auto'],
                            'title': 'Quality',
                            'type': 'string',
                        },
                        'size': {
                            'anyOf': [
                                {
                                    'enum': ['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'],
                                    'type': 'string',
                                },
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Size',
                        },
                        'aspect_ratio': {
                            'anyOf': [
                                {
                                    'enum': ['21:9', '16:9', '4:3', '3:2', '1:1', '9:16', '3:4', '2:3', '5:4', '4:5'],
                                    'type': 'string',
                                },
                                {'type': 'null'},
                            ],
                            'default': None,
                            'title': 'Aspect Ratio',
                        },
                    },
                    'title': 'ImageGenerationTool',
                    'type': 'object',
                },
                'KnownModelName': {
                    'enum': [
                        'anthropic:claude-fable-5',
                        'anthropic:claude-haiku-4-5',
                        'anthropic:claude-haiku-4-5-20251001',
                        'anthropic:claude-mythos-5',
                        'anthropic:claude-mythos-preview',
                        'anthropic:claude-opus-4-5',
                        'anthropic:claude-opus-4-5-20251101',
                        'anthropic:claude-opus-4-6',
                        'anthropic:claude-opus-4-7',
                        'anthropic:claude-opus-4-8',
                        'anthropic:claude-opus-5',
                        'anthropic:claude-sonnet-4-5',
                        'anthropic:claude-sonnet-4-5-20250929',
                        'anthropic:claude-sonnet-4-6',
                        'anthropic:claude-sonnet-5',
                        'bedrock-mantle:openai.gpt-5.4',
                        'bedrock-mantle:openai.gpt-5.4-2026-03-05',
                        'bedrock-mantle:openai.gpt-5.5',
                        'bedrock-mantle:openai.gpt-5.5-2026-04-23',
                        'bedrock-mantle:openai.gpt-5.6-luna',
                        'bedrock-mantle:openai.gpt-5.6-sol',
                        'bedrock-mantle:openai.gpt-5.6-terra',
                        'bedrock-mantle:openai.gpt-oss-120b',
                        'bedrock-mantle:openai.gpt-oss-20b',
                        'bedrock-mantle:openai.gpt-oss-safeguard-120b',
                        'bedrock-mantle:openai.gpt-oss-safeguard-20b',
                        'bedrock:amazon.titan-text-express-v1',
                        'bedrock:amazon.titan-text-lite-v1',
                        'bedrock:amazon.titan-tg1-large',
                        'bedrock:anthropic.claude-3-5-haiku-20241022-v1:0',
                        'bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0',
                        'bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0',
                        'bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0',
                        'bedrock:anthropic.claude-3-haiku-20240307-v1:0',
                        'bedrock:anthropic.claude-3-opus-20240229-v1:0',
                        'bedrock:anthropic.claude-3-sonnet-20240229-v1:0',
                        'bedrock:anthropic.claude-haiku-4-5-20251001-v1:0',
                        'bedrock:anthropic.claude-instant-v1',
                        'bedrock:anthropic.claude-opus-4-20250514-v1:0',
                        'bedrock:anthropic.claude-sonnet-4-20250514-v1:0',
                        'bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'bedrock:anthropic.claude-sonnet-4-6',
                        'bedrock:anthropic.claude-v2',
                        'bedrock:anthropic.claude-v2:1',
                        'bedrock:cohere.command-light-text-v14',
                        'bedrock:cohere.command-r-plus-v1:0',
                        'bedrock:cohere.command-r-v1:0',
                        'bedrock:cohere.command-text-v14',
                        'bedrock:deepseek.r1-v1:0',
                        'bedrock:deepseek.v3.2',
                        'bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0',
                        'bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0',
                        'bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'bedrock:eu.anthropic.claude-sonnet-4-6',
                        'bedrock:global.amazon.nova-2-lite-v1:0',
                        'bedrock:global.anthropic.claude-fable-5',
                        'bedrock:global.anthropic.claude-opus-4-5-20251101-v1:0',
                        'bedrock:global.anthropic.claude-opus-4-6-v1',
                        'bedrock:global.anthropic.claude-opus-4-7',
                        'bedrock:global.anthropic.claude-opus-4-8',
                        'bedrock:global.anthropic.claude-opus-5',
                        'bedrock:global.anthropic.claude-sonnet-5',
                        'bedrock:google.gemma-3-12b-it',
                        'bedrock:google.gemma-3-27b-it',
                        'bedrock:google.gemma-3-4b-it',
                        'bedrock:meta.llama3-1-405b-instruct-v1:0',
                        'bedrock:meta.llama3-1-70b-instruct-v1:0',
                        'bedrock:meta.llama3-1-8b-instruct-v1:0',
                        'bedrock:meta.llama3-70b-instruct-v1:0',
                        'bedrock:meta.llama3-8b-instruct-v1:0',
                        'bedrock:minimax.minimax-m2',
                        'bedrock:minimax.minimax-m2.1',
                        'bedrock:minimax.minimax-m2.5',
                        'bedrock:mistral.devstral-2-123b',
                        'bedrock:mistral.magistral-small-2509',
                        'bedrock:mistral.ministral-3-14b-instruct',
                        'bedrock:mistral.ministral-3-3b-instruct',
                        'bedrock:mistral.ministral-3-8b-instruct',
                        'bedrock:mistral.mistral-7b-instruct-v0:2',
                        'bedrock:mistral.mistral-large-2402-v1:0',
                        'bedrock:mistral.mistral-large-2407-v1:0',
                        'bedrock:mistral.mistral-large-3-675b-instruct',
                        'bedrock:mistral.mistral-small-2402-v1:0',
                        'bedrock:mistral.mixtral-8x7b-instruct-v0:1',
                        'bedrock:mistral.pixtral-large-2502-v1:0',
                        'bedrock:moonshot.kimi-k2-thinking',
                        'bedrock:moonshotai.kimi-k2.5',
                        'bedrock:nvidia.nemotron-nano-12b-v2',
                        'bedrock:nvidia.nemotron-nano-3-30b',
                        'bedrock:nvidia.nemotron-nano-9b-v2',
                        'bedrock:nvidia.nemotron-super-3-120b',
                        'bedrock:qwen.qwen3-32b-v1:0',
                        'bedrock:qwen.qwen3-coder-30b-a3b-v1:0',
                        'bedrock:qwen.qwen3-coder-next',
                        'bedrock:qwen.qwen3-next-80b-a3b',
                        'bedrock:qwen.qwen3-vl-235b-a22b',
                        'bedrock:us.amazon.nova-2-lite-v1:0',
                        'bedrock:us.amazon.nova-lite-v1:0',
                        'bedrock:us.amazon.nova-micro-v1:0',
                        'bedrock:us.amazon.nova-premier-v1:0',
                        'bedrock:us.amazon.nova-pro-v1:0',
                        'bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0',
                        'bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0',
                        'bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                        'bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                        'bedrock:us.anthropic.claude-3-haiku-20240307-v1:0',
                        'bedrock:us.anthropic.claude-3-opus-20240229-v1:0',
                        'bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0',
                        'bedrock:us.anthropic.claude-fable-5',
                        'bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0',
                        'bedrock:us.anthropic.claude-opus-4-1-20250805-v1:0',
                        'bedrock:us.anthropic.claude-opus-4-20250514-v1:0',
                        'bedrock:us.anthropic.claude-opus-4-5-20251101-v1:0',
                        'bedrock:us.anthropic.claude-opus-4-6-v1',
                        'bedrock:us.anthropic.claude-opus-4-7',
                        'bedrock:us.anthropic.claude-opus-4-8',
                        'bedrock:us.anthropic.claude-opus-5',
                        'bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0',
                        'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'bedrock:us.anthropic.claude-sonnet-4-6',
                        'bedrock:us.anthropic.claude-sonnet-5',
                        'bedrock:us.meta.llama3-1-70b-instruct-v1:0',
                        'bedrock:us.meta.llama3-1-8b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-11b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-1b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-3b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-90b-instruct-v1:0',
                        'bedrock:us.meta.llama3-3-70b-instruct-v1:0',
                        'bedrock:us.meta.llama4-maverick-17b-instruct-v1:0',
                        'bedrock:us.meta.llama4-scout-17b-instruct-v1:0',
                        'bedrock:us.mistral.pixtral-large-2502-v1:0',
                        'bedrock:us.writer.palmyra-x4-v1:0',
                        'bedrock:us.writer.palmyra-x5-v1:0',
                        'bedrock:zai.glm-4.7',
                        'bedrock:zai.glm-4.7-flash',
                        'bedrock:zai.glm-5',
                        'cerebras:gemma-4-31b',
                        'cerebras:gpt-oss-120b',
                        'cerebras:zai-glm-4.7',
                        'cohere:c4ai-aya-expanse-32b',
                        'cohere:c4ai-aya-expanse-8b',
                        'cohere:command-nightly',
                        'cohere:command-r-08-2024',
                        'cohere:command-r-plus-08-2024',
                        'cohere:command-r7b-12-2024',
                        'crusoe:Qwen/Qwen3-235B-A22B-Instruct-2507',
                        'crusoe:deepseek-ai/DeepSeek-V3-0324',
                        'crusoe:deepseek-ai/DeepSeek-V4-Pro',
                        'crusoe:deepseek-ai/Deepseek-V4-Flash',
                        'crusoe:google/gemma-4-31b-it',
                        'crusoe:meta-llama/Llama-3.3-70B-Instruct',
                        'crusoe:moonshotai/Kimi-K2.6',
                        'crusoe:nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B',
                        'crusoe:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B',
                        'crusoe:nvidia/Nemotron-3-Nano-Omni-Reasoning-30B-A3B',
                        'crusoe:nvidia/Nemotron-3.5-Lightning-30B-A3B',
                        'crusoe:openai/gpt-oss-120b',
                        'crusoe:yutori/n1.5',
                        'crusoe:zai/GLM-5.1',
                        'crusoe:zai/GLM-5.2',
                        'deepseek:deepseek-chat',
                        'deepseek:deepseek-reasoner',
                        'deepseek:deepseek-v4-flash',
                        'deepseek:deepseek-v4-pro',
                        'gateway/anthropic:claude-fable-5',
                        'gateway/anthropic:claude-haiku-4-5',
                        'gateway/anthropic:claude-haiku-4-5-20251001',
                        'gateway/anthropic:claude-opus-4-5',
                        'gateway/anthropic:claude-opus-4-5-20251101',
                        'gateway/anthropic:claude-opus-4-6',
                        'gateway/anthropic:claude-opus-4-7',
                        'gateway/anthropic:claude-opus-4-8',
                        'gateway/anthropic:claude-opus-5',
                        'gateway/anthropic:claude-sonnet-4-5',
                        'gateway/anthropic:claude-sonnet-4-5-20250929',
                        'gateway/anthropic:claude-sonnet-4-6',
                        'gateway/anthropic:claude-sonnet-5',
                        'gateway/bedrock:anthropic.claude-3-haiku-20240307-v1:0',
                        'gateway/bedrock:deepseek.r1-v1:0',
                        'gateway/bedrock:deepseek.v3.2',
                        'gateway/bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-sonnet-4-6',
                        'gateway/bedrock:global.amazon.nova-2-lite-v1:0',
                        'gateway/bedrock:global.anthropic.claude-fable-5',
                        'gateway/bedrock:global.anthropic.claude-opus-4-5-20251101-v1:0',
                        'gateway/bedrock:global.anthropic.claude-opus-4-6-v1',
                        'gateway/bedrock:global.anthropic.claude-opus-4-7',
                        'gateway/bedrock:global.anthropic.claude-opus-4-8',
                        'gateway/bedrock:global.anthropic.claude-opus-5',
                        'gateway/bedrock:global.anthropic.claude-sonnet-5',
                        'gateway/bedrock:google.gemma-3-12b-it',
                        'gateway/bedrock:google.gemma-3-27b-it',
                        'gateway/bedrock:google.gemma-3-4b-it',
                        'gateway/bedrock:minimax.minimax-m2',
                        'gateway/bedrock:minimax.minimax-m2.1',
                        'gateway/bedrock:minimax.minimax-m2.5',
                        'gateway/bedrock:mistral.devstral-2-123b',
                        'gateway/bedrock:mistral.magistral-small-2509',
                        'gateway/bedrock:mistral.ministral-3-14b-instruct',
                        'gateway/bedrock:mistral.ministral-3-3b-instruct',
                        'gateway/bedrock:mistral.ministral-3-8b-instruct',
                        'gateway/bedrock:mistral.mistral-large-3-675b-instruct',
                        'gateway/bedrock:mistral.mistral-small-2402-v1:0',
                        'gateway/bedrock:mistral.pixtral-large-2502-v1:0',
                        'gateway/bedrock:moonshot.kimi-k2-thinking',
                        'gateway/bedrock:moonshotai.kimi-k2.5',
                        'gateway/bedrock:nvidia.nemotron-nano-12b-v2',
                        'gateway/bedrock:nvidia.nemotron-nano-3-30b',
                        'gateway/bedrock:nvidia.nemotron-nano-9b-v2',
                        'gateway/bedrock:nvidia.nemotron-super-3-120b',
                        'gateway/bedrock:qwen.qwen3-32b-v1:0',
                        'gateway/bedrock:qwen.qwen3-coder-30b-a3b-v1:0',
                        'gateway/bedrock:qwen.qwen3-coder-next',
                        'gateway/bedrock:qwen.qwen3-next-80b-a3b',
                        'gateway/bedrock:qwen.qwen3-vl-235b-a22b',
                        'gateway/bedrock:us.amazon.nova-premier-v1:0',
                        'gateway/bedrock:us.anthropic.claude-fable-5',
                        'gateway/bedrock:us.anthropic.claude-opus-4-1-20250805-v1:0',
                        'gateway/bedrock:us.anthropic.claude-opus-4-5-20251101-v1:0',
                        'gateway/bedrock:us.anthropic.claude-opus-4-6-v1',
                        'gateway/bedrock:us.anthropic.claude-opus-4-7',
                        'gateway/bedrock:us.anthropic.claude-opus-4-8',
                        'gateway/bedrock:us.anthropic.claude-opus-5',
                        'gateway/bedrock:us.anthropic.claude-sonnet-5',
                        'gateway/bedrock:us.meta.llama4-maverick-17b-instruct-v1:0',
                        'gateway/bedrock:us.meta.llama4-scout-17b-instruct-v1:0',
                        'gateway/bedrock:us.mistral.pixtral-large-2502-v1:0',
                        'gateway/bedrock:us.writer.palmyra-x4-v1:0',
                        'gateway/bedrock:us.writer.palmyra-x5-v1:0',
                        'gateway/bedrock:zai.glm-4.7',
                        'gateway/bedrock:zai.glm-4.7-flash',
                        'gateway/bedrock:zai.glm-5',
                        'gateway/google-cloud:gemini-2.5-flash',
                        'gateway/google-cloud:gemini-2.5-flash-image',
                        'gateway/google-cloud:gemini-2.5-flash-lite',
                        'gateway/google-cloud:gemini-2.5-pro',
                        'gateway/google-cloud:gemini-3-flash-preview',
                        'gateway/google-cloud:gemini-3-pro-image',
                        'gateway/google-cloud:gemini-3.1-flash-image',
                        'gateway/google-cloud:gemini-3.1-flash-lite',
                        'gateway/google-cloud:gemini-3.1-pro-preview',
                        'gateway/google-cloud:gemini-3.5-flash',
                        'gateway/google-cloud:gemini-3.5-flash-lite',
                        'gateway/google-cloud:gemini-3.6-flash',
                        'gateway/google-cloud:gemini-3.7-flash',
                        'gateway/google:gemini-2.5-flash',
                        'gateway/google:gemini-2.5-flash-image',
                        'gateway/google:gemini-2.5-flash-lite',
                        'gateway/google:gemini-2.5-pro',
                        'gateway/google:gemini-3-flash-preview',
                        'gateway/google:gemini-3-pro-image',
                        'gateway/google:gemini-3.1-flash-image',
                        'gateway/google:gemini-3.1-flash-lite',
                        'gateway/google:gemini-3.1-pro-preview',
                        'gateway/google:gemini-3.5-flash',
                        'gateway/google:gemini-3.5-flash-lite',
                        'gateway/google:gemini-3.6-flash',
                        'gateway/google:gemini-3.7-flash',
                        'gateway/groq:llama-3.1-8b-instant',
                        'gateway/groq:llama-3.3-70b-versatile',
                        'gateway/groq:openai/gpt-oss-120b',
                        'gateway/groq:openai/gpt-oss-20b',
                        'gateway/groq:openai/gpt-oss-safeguard-20b',
                        'gateway/openai:gpt-3.5-turbo',
                        'gateway/openai:gpt-3.5-turbo-0125',
                        'gateway/openai:gpt-3.5-turbo-1106',
                        'gateway/openai:gpt-4',
                        'gateway/openai:gpt-4-0613',
                        'gateway/openai:gpt-4-turbo',
                        'gateway/openai:gpt-4-turbo-2024-04-09',
                        'gateway/openai:gpt-4.1',
                        'gateway/openai:gpt-4.1-2025-04-14',
                        'gateway/openai:gpt-4.1-mini',
                        'gateway/openai:gpt-4.1-mini-2025-04-14',
                        'gateway/openai:gpt-4.1-nano',
                        'gateway/openai:gpt-4.1-nano-2025-04-14',
                        'gateway/openai:gpt-4o',
                        'gateway/openai:gpt-4o-2024-05-13',
                        'gateway/openai:gpt-4o-2024-08-06',
                        'gateway/openai:gpt-4o-2024-11-20',
                        'gateway/openai:gpt-4o-mini',
                        'gateway/openai:gpt-4o-mini-2024-07-18',
                        'gateway/openai:gpt-5',
                        'gateway/openai:gpt-5-2025-08-07',
                        'gateway/openai:gpt-5-mini',
                        'gateway/openai:gpt-5-mini-2025-08-07',
                        'gateway/openai:gpt-5-nano',
                        'gateway/openai:gpt-5-nano-2025-08-07',
                        'gateway/openai:gpt-5-pro',
                        'gateway/openai:gpt-5-pro-2025-10-06',
                        'gateway/openai:gpt-5.1',
                        'gateway/openai:gpt-5.1-2025-11-13',
                        'gateway/openai:gpt-5.2',
                        'gateway/openai:gpt-5.2-2025-12-11',
                        'gateway/openai:gpt-5.2-chat-latest',
                        'gateway/openai:gpt-5.2-pro',
                        'gateway/openai:gpt-5.2-pro-2025-12-11',
                        'gateway/openai:gpt-5.3-chat-latest',
                        'gateway/openai:gpt-5.4',
                        'gateway/openai:gpt-5.4-mini',
                        'gateway/openai:gpt-5.4-mini-2026-03-17',
                        'gateway/openai:gpt-5.4-nano',
                        'gateway/openai:gpt-5.4-nano-2026-03-17',
                        'gateway/openai:gpt-5.5',
                        'gateway/openai:gpt-5.5-2026-04-23',
                        'gateway/openai:gpt-5.5-pro',
                        'gateway/openai:gpt-5.5-pro-2026-04-23',
                        'gateway/openai:gpt-5.6-cyber',
                        'gateway/openai:gpt-5.6-luna',
                        'gateway/openai:gpt-5.6-sol',
                        'gateway/openai:gpt-5.6-terra',
                        'gateway/openai:gpt-daybreak-blue-latest',
                        'gateway/openai:gpt-daybreak-red-latest',
                        'gateway/openai:o1',
                        'gateway/openai:o1-2024-12-17',
                        'gateway/openai:o1-pro',
                        'gateway/openai:o1-pro-2025-03-19',
                        'gateway/openai:o3',
                        'gateway/openai:o3-2025-04-16',
                        'gateway/openai:o3-mini',
                        'gateway/openai:o3-mini-2025-01-31',
                        'gateway/openai:o3-pro',
                        'gateway/openai:o3-pro-2025-06-10',
                        'gateway/openai:o4-mini',
                        'gateway/openai:o4-mini-2025-04-16',
                        'google-cloud:gemini-2.0-flash',
                        'google-cloud:gemini-2.0-flash-lite',
                        'google-cloud:gemini-2.5-flash',
                        'google-cloud:gemini-2.5-flash-image',
                        'google-cloud:gemini-2.5-flash-lite',
                        'google-cloud:gemini-2.5-flash-preview-09-2025',
                        'google-cloud:gemini-2.5-pro',
                        'google-cloud:gemini-3-flash-preview',
                        'google-cloud:gemini-3-pro-image',
                        'google-cloud:gemini-3-pro-image-preview',
                        'google-cloud:gemini-3-pro-preview',
                        'google-cloud:gemini-3.1-flash-image',
                        'google-cloud:gemini-3.1-flash-image-preview',
                        'google-cloud:gemini-3.1-flash-lite',
                        'google-cloud:gemini-3.1-pro-preview',
                        'google-cloud:gemini-3.5-flash',
                        'google-cloud:gemini-3.5-flash-lite',
                        'google-cloud:gemini-3.6-flash',
                        'google-cloud:gemini-3.7-flash',
                        'google-cloud:gemini-flash-latest',
                        'google-cloud:gemini-flash-lite-latest',
                        'google:gemini-2.0-flash',
                        'google:gemini-2.0-flash-lite',
                        'google:gemini-2.5-flash',
                        'google:gemini-2.5-flash-image',
                        'google:gemini-2.5-flash-lite',
                        'google:gemini-2.5-flash-preview-09-2025',
                        'google:gemini-2.5-pro',
                        'google:gemini-3-flash-preview',
                        'google:gemini-3-pro-image',
                        'google:gemini-3-pro-image-preview',
                        'google:gemini-3-pro-preview',
                        'google:gemini-3.1-flash-image',
                        'google:gemini-3.1-flash-image-preview',
                        'google:gemini-3.1-flash-lite',
                        'google:gemini-3.1-pro-preview',
                        'google:gemini-3.5-flash',
                        'google:gemini-3.5-flash-lite',
                        'google:gemini-3.6-flash',
                        'google:gemini-3.7-flash',
                        'google:gemini-flash-latest',
                        'google:gemini-flash-lite-latest',
                        'groq:llama-3.1-8b-instant',
                        'groq:llama-3.3-70b-versatile',
                        'groq:meta-llama/llama-4-maverick-17b-128e-instruct',
                        'groq:meta-llama/llama-guard-4-12b',
                        'groq:meta-llama/llama-prompt-guard-2-22m',
                        'groq:meta-llama/llama-prompt-guard-2-86m',
                        'groq:openai/gpt-oss-120b',
                        'groq:openai/gpt-oss-20b',
                        'groq:openai/gpt-oss-safeguard-20b',
                        'groq:playai-tts',
                        'groq:playai-tts-arabic',
                        'groq:whisper-large-v3',
                        'groq:whisper-large-v3-turbo',
                        'heroku:claude-3-5-haiku',
                        'heroku:claude-3-5-sonnet-latest',
                        'heroku:claude-3-7-sonnet',
                        'heroku:claude-3-haiku',
                        'heroku:claude-4-5-haiku',
                        'heroku:claude-4-5-sonnet',
                        'heroku:claude-4-6-sonnet',
                        'heroku:claude-4-sonnet',
                        'heroku:claude-opus-4-5',
                        'heroku:claude-opus-4-6',
                        'heroku:deepseek-v3-2',
                        'heroku:glm-4-7',
                        'heroku:glm-4-7-flash',
                        'heroku:gpt-oss-120b',
                        'heroku:kimi-k2-5',
                        'heroku:kimi-k2-thinking',
                        'heroku:minimax-m2',
                        'heroku:minimax-m2-1',
                        'heroku:nova-2-lite',
                        'heroku:nova-lite',
                        'heroku:nova-pro',
                        'heroku:qwen3-235b',
                        'heroku:qwen3-coder-480b',
                        'huggingface:Qwen/QwQ-32B',
                        'huggingface:Qwen/Qwen2.5-72B-Instruct',
                        'huggingface:Qwen/Qwen3-235B-A22B',
                        'huggingface:Qwen/Qwen3-32B',
                        'huggingface:deepseek-ai/DeepSeek-R1',
                        'huggingface:meta-llama/Llama-3.3-70B-Instruct',
                        'huggingface:meta-llama/Llama-4-Maverick-17B-128E-Instruct',
                        'huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct',
                        'mistral:codestral-latest',
                        'mistral:mistral-large-latest',
                        'mistral:mistral-moderation-latest',
                        'mistral:mistral-small-latest',
                        'moonshotai:kimi-k2-0711-preview',
                        'moonshotai:kimi-k2.5',
                        'moonshotai:kimi-k2.6',
                        'moonshotai:kimi-k2.7-code',
                        'moonshotai:kimi-k2.7-code-highspeed',
                        'moonshotai:kimi-k3',
                        'moonshotai:kimi-latest',
                        'moonshotai:kimi-thinking-preview',
                        'moonshotai:moonshot-v1-128k',
                        'moonshotai:moonshot-v1-128k-vision-preview',
                        'moonshotai:moonshot-v1-32k',
                        'moonshotai:moonshot-v1-32k-vision-preview',
                        'moonshotai:moonshot-v1-8k',
                        'moonshotai:moonshot-v1-8k-vision-preview',
                        'moonshotai:moonshot-v1-auto',
                        'openai-chat:computer-use-preview',
                        'openai-chat:computer-use-preview-2025-03-11',
                        'openai-chat:gpt-3.5-turbo',
                        'openai-chat:gpt-3.5-turbo-0125',
                        'openai-chat:gpt-3.5-turbo-0301',
                        'openai-chat:gpt-3.5-turbo-1106',
                        'openai-chat:gpt-3.5-turbo-16k',
                        'openai-chat:gpt-4',
                        'openai-chat:gpt-4-0314',
                        'openai-chat:gpt-4-0613',
                        'openai-chat:gpt-4-turbo',
                        'openai-chat:gpt-4-turbo-2024-04-09',
                        'openai-chat:gpt-4.1',
                        'openai-chat:gpt-4.1-2025-04-14',
                        'openai-chat:gpt-4.1-mini',
                        'openai-chat:gpt-4.1-mini-2025-04-14',
                        'openai-chat:gpt-4.1-nano',
                        'openai-chat:gpt-4.1-nano-2025-04-14',
                        'openai-chat:gpt-4o',
                        'openai-chat:gpt-4o-2024-05-13',
                        'openai-chat:gpt-4o-2024-08-06',
                        'openai-chat:gpt-4o-2024-11-20',
                        'openai-chat:gpt-4o-audio-preview',
                        'openai-chat:gpt-4o-audio-preview-2024-12-17',
                        'openai-chat:gpt-4o-audio-preview-2025-06-03',
                        'openai-chat:gpt-4o-mini',
                        'openai-chat:gpt-4o-mini-2024-07-18',
                        'openai-chat:gpt-4o-mini-audio-preview',
                        'openai-chat:gpt-4o-mini-audio-preview-2024-12-17',
                        'openai-chat:gpt-4o-mini-search-preview',
                        'openai-chat:gpt-4o-mini-search-preview-2025-03-11',
                        'openai-chat:gpt-4o-search-preview',
                        'openai-chat:gpt-4o-search-preview-2025-03-11',
                        'openai-chat:gpt-5',
                        'openai-chat:gpt-5-2025-08-07',
                        'openai-chat:gpt-5-chat-latest',
                        'openai-chat:gpt-5-codex',
                        'openai-chat:gpt-5-mini',
                        'openai-chat:gpt-5-mini-2025-08-07',
                        'openai-chat:gpt-5-nano',
                        'openai-chat:gpt-5-nano-2025-08-07',
                        'openai-chat:gpt-5-pro',
                        'openai-chat:gpt-5-pro-2025-10-06',
                        'openai-chat:gpt-5.1',
                        'openai-chat:gpt-5.1-2025-11-13',
                        'openai-chat:gpt-5.1-chat-latest',
                        'openai-chat:gpt-5.1-codex',
                        'openai-chat:gpt-5.1-codex-max',
                        'openai-chat:gpt-5.2',
                        'openai-chat:gpt-5.2-2025-12-11',
                        'openai-chat:gpt-5.2-chat-latest',
                        'openai-chat:gpt-5.2-pro',
                        'openai-chat:gpt-5.2-pro-2025-12-11',
                        'openai-chat:gpt-5.3-chat-latest',
                        'openai-chat:gpt-5.4',
                        'openai-chat:gpt-5.4-mini',
                        'openai-chat:gpt-5.4-mini-2026-03-17',
                        'openai-chat:gpt-5.4-nano',
                        'openai-chat:gpt-5.4-nano-2026-03-17',
                        'openai-chat:gpt-5.5',
                        'openai-chat:gpt-5.5-2026-04-23',
                        'openai-chat:gpt-5.5-pro',
                        'openai-chat:gpt-5.5-pro-2026-04-23',
                        'openai-chat:gpt-5.6-cyber',
                        'openai-chat:gpt-5.6-luna',
                        'openai-chat:gpt-5.6-sol',
                        'openai-chat:gpt-5.6-terra',
                        'openai-chat:gpt-daybreak-blue-latest',
                        'openai-chat:gpt-daybreak-red-latest',
                        'openai-chat:o1',
                        'openai-chat:o1-2024-12-17',
                        'openai-chat:o1-pro',
                        'openai-chat:o1-pro-2025-03-19',
                        'openai-chat:o3',
                        'openai-chat:o3-2025-04-16',
                        'openai-chat:o3-deep-research',
                        'openai-chat:o3-deep-research-2025-06-26',
                        'openai-chat:o3-mini',
                        'openai-chat:o3-mini-2025-01-31',
                        'openai-chat:o3-pro',
                        'openai-chat:o3-pro-2025-06-10',
                        'openai-chat:o4-mini',
                        'openai-chat:o4-mini-2025-04-16',
                        'openai-chat:o4-mini-deep-research',
                        'openai-chat:o4-mini-deep-research-2025-06-26',
                        'openai:computer-use-preview',
                        'openai:computer-use-preview-2025-03-11',
                        'openai:gpt-3.5-turbo',
                        'openai:gpt-3.5-turbo-0125',
                        'openai:gpt-3.5-turbo-0301',
                        'openai:gpt-3.5-turbo-1106',
                        'openai:gpt-4',
                        'openai:gpt-4-0314',
                        'openai:gpt-4-0613',
                        'openai:gpt-4-turbo',
                        'openai:gpt-4-turbo-2024-04-09',
                        'openai:gpt-4.1',
                        'openai:gpt-4.1-2025-04-14',
                        'openai:gpt-4.1-mini',
                        'openai:gpt-4.1-mini-2025-04-14',
                        'openai:gpt-4.1-nano',
                        'openai:gpt-4.1-nano-2025-04-14',
                        'openai:gpt-4o',
                        'openai:gpt-4o-2024-05-13',
                        'openai:gpt-4o-2024-08-06',
                        'openai:gpt-4o-2024-11-20',
                        'openai:gpt-4o-audio-preview',
                        'openai:gpt-4o-audio-preview-2024-12-17',
                        'openai:gpt-4o-audio-preview-2025-06-03',
                        'openai:gpt-4o-mini',
                        'openai:gpt-4o-mini-2024-07-18',
                        'openai:gpt-4o-mini-audio-preview',
                        'openai:gpt-4o-mini-audio-preview-2024-12-17',
                        'openai:gpt-5',
                        'openai:gpt-5-2025-08-07',
                        'openai:gpt-5-chat-latest',
                        'openai:gpt-5-codex',
                        'openai:gpt-5-mini',
                        'openai:gpt-5-mini-2025-08-07',
                        'openai:gpt-5-nano',
                        'openai:gpt-5-nano-2025-08-07',
                        'openai:gpt-5-pro',
                        'openai:gpt-5-pro-2025-10-06',
                        'openai:gpt-5.1',
                        'openai:gpt-5.1-2025-11-13',
                        'openai:gpt-5.1-chat-latest',
                        'openai:gpt-5.1-codex',
                        'openai:gpt-5.1-codex-max',
                        'openai:gpt-5.2',
                        'openai:gpt-5.2-2025-12-11',
                        'openai:gpt-5.2-chat-latest',
                        'openai:gpt-5.2-pro',
                        'openai:gpt-5.2-pro-2025-12-11',
                        'openai:gpt-5.3-chat-latest',
                        'openai:gpt-5.4',
                        'openai:gpt-5.4-mini',
                        'openai:gpt-5.4-mini-2026-03-17',
                        'openai:gpt-5.4-nano',
                        'openai:gpt-5.4-nano-2026-03-17',
                        'openai:gpt-5.5',
                        'openai:gpt-5.5-2026-04-23',
                        'openai:gpt-5.5-pro',
                        'openai:gpt-5.5-pro-2026-04-23',
                        'openai:gpt-5.6-cyber',
                        'openai:gpt-5.6-luna',
                        'openai:gpt-5.6-sol',
                        'openai:gpt-5.6-terra',
                        'openai:gpt-daybreak-blue-latest',
                        'openai:gpt-daybreak-red-latest',
                        'openai:o1',
                        'openai:o1-2024-12-17',
                        'openai:o1-pro',
                        'openai:o1-pro-2025-03-19',
                        'openai:o3',
                        'openai:o3-2025-04-16',
                        'openai:o3-deep-research',
                        'openai:o3-deep-research-2025-06-26',
                        'openai:o3-mini',
                        'openai:o3-mini-2025-01-31',
                        'openai:o3-pro',
                        'openai:o3-pro-2025-06-10',
                        'openai:o4-mini',
                        'openai:o4-mini-2025-04-16',
                        'openai:o4-mini-deep-research',
                        'openai:o4-mini-deep-research-2025-06-26',
                        'test',
                        'snowflake:claude-4-sonnet',
                        'snowflake:claude-fable-5',
                        'snowflake:claude-haiku-4-5',
                        'snowflake:claude-opus-4-5',
                        'snowflake:claude-opus-4-6',
                        'snowflake:claude-opus-4-7',
                        'snowflake:claude-opus-4-8',
                        'snowflake:claude-opus-5',
                        'snowflake:claude-sonnet-4-5',
                        'snowflake:claude-sonnet-4-6',
                        'snowflake:claude-sonnet-5',
                        'snowflake:deepseek-r1',
                        'snowflake:llama3.1-405b',
                        'snowflake:llama3.1-70b',
                        'snowflake:llama3.1-8b',
                        'snowflake:llama4-maverick',
                        'snowflake:mistral-7b',
                        'snowflake:mistral-large',
                        'snowflake:mistral-large2',
                        'snowflake:openai-gpt-4.1',
                        'snowflake:openai-gpt-5',
                        'snowflake:openai-gpt-5-6-luna',
                        'snowflake:openai-gpt-5-6-sol',
                        'snowflake:openai-gpt-5-6-terra',
                        'snowflake:openai-gpt-5-chat',
                        'snowflake:openai-gpt-5-mini',
                        'snowflake:openai-gpt-5-nano',
                        'snowflake:openai-gpt-5.1',
                        'snowflake:openai-gpt-5.2',
                        'snowflake:openai-gpt-5.4',
                        'snowflake:openai-gpt-5.5',
                        'snowflake:snowflake-llama-3.3-70b',
                        'xai:grok-3',
                        'xai:grok-3-fast',
                        'xai:grok-3-fast-latest',
                        'xai:grok-3-latest',
                        'xai:grok-3-mini',
                        'xai:grok-3-mini-fast',
                        'xai:grok-3-mini-fast-latest',
                        'xai:grok-4',
                        'xai:grok-4-0709',
                        'xai:grok-4-1-fast',
                        'xai:grok-4-1-fast-non-reasoning',
                        'xai:grok-4-1-fast-non-reasoning-latest',
                        'xai:grok-4-1-fast-reasoning',
                        'xai:grok-4-1-fast-reasoning-latest',
                        'xai:grok-4-fast',
                        'xai:grok-4-fast-non-reasoning',
                        'xai:grok-4-fast-non-reasoning-latest',
                        'xai:grok-4-fast-reasoning',
                        'xai:grok-4-fast-reasoning-latest',
                        'xai:grok-4-latest',
                        'xai:grok-4.20',
                        'xai:grok-4.20-0309',
                        'xai:grok-4.20-0309-non-reasoning',
                        'xai:grok-4.20-0309-reasoning',
                        'xai:grok-4.20-multi-agent',
                        'xai:grok-4.20-multi-agent-0309',
                        'xai:grok-4.20-multi-agent-latest',
                        'xai:grok-4.20-non-reasoning',
                        'xai:grok-4.20-non-reasoning-latest',
                        'xai:grok-4.20-reasoning-latest',
                        'xai:grok-4.3',
                        'xai:grok-4.3-latest',
                        'xai:grok-4.5',
                        'xai:grok-4.5-latest',
                        'xai:grok-4.6',
                        'xai:grok-build-0.1',
                        'xai:grok-code-fast-1',
                        'zai:autoglm-phone-multilingual',
                        'zai:glm-4-32b-0414-128k',
                        'zai:glm-4.5',
                        'zai:glm-4.5-air',
                        'zai:glm-4.5-airx',
                        'zai:glm-4.5-flash',
                        'zai:glm-4.5-x',
                        'zai:glm-4.5v',
                        'zai:glm-4.6',
                        'zai:glm-4.6v',
                        'zai:glm-4.6v-flash',
                        'zai:glm-4.6v-flashx',
                        'zai:glm-4.7',
                        'zai:glm-4.7-flash',
                        'zai:glm-4.7-flashx',
                        'zai:glm-5',
                        'zai:glm-5-turbo',
                        'zai:glm-5.1',
                        'zai:glm-5.2',
                        'zai:glm-5.3',
                        'zai:glm-5.3-flash',
                        'zai:glm-5v-turbo',
                    ],
                    'type': 'string',
                },
                'MCPServerTool': {
                    'properties': {
                        'kind': {'default': 'mcp_server', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'id': {'title': 'Id', 'type': 'string'},
                        'url': {'title': 'Url', 'type': 'string'},
                        'authorization_token': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Authorization Token',
                        },
                        'description': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Description',
                        },
                        'allowed_tools': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Allowed Tools',
                        },
                        'headers': {
                            'anyOf': [{'additionalProperties': {'type': 'string'}, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Headers',
                        },
                    },
                    'required': ['id', 'url'],
                    'title': 'MCPServerTool',
                    'type': 'object',
                },
                'MemoryTool': {
                    'properties': {
                        'kind': {'default': 'memory', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                    },
                    'title': 'MemoryTool',
                    'type': 'object',
                },
                'ModelSettings': {
                    'properties': {
                        'max_tokens': {'title': 'Max Tokens', 'type': 'integer'},
                        'temperature': {'title': 'Temperature', 'type': 'number'},
                        'top_p': {'title': 'Top P', 'type': 'number'},
                        'top_k': {'title': 'Top K', 'type': 'integer'},
                        'timeout': {'anyOf': [{'type': 'integer'}, {'type': 'number'}], 'title': 'Timeout'},
                        'parallel_tool_calls': {'title': 'Parallel Tool Calls', 'type': 'boolean'},
                        'tool_choice': {
                            'anyOf': [
                                {'enum': ['none', 'required', 'auto'], 'type': 'string'},
                                {'items': {'type': 'string'}, 'type': 'array'},
                                {'$ref': '#/$defs/ToolOrOutput'},
                                {'type': 'null'},
                            ],
                            'title': 'Tool Choice',
                        },
                        'seed': {'title': 'Seed', 'type': 'integer'},
                        'presence_penalty': {'title': 'Presence Penalty', 'type': 'number'},
                        'frequency_penalty': {'title': 'Frequency Penalty', 'type': 'number'},
                        'logit_bias': {
                            'additionalProperties': {'type': 'integer'},
                            'title': 'Logit Bias',
                            'type': 'object',
                        },
                        'stop_sequences': {'items': {'type': 'string'}, 'title': 'Stop Sequences', 'type': 'array'},
                        'extra_headers': {
                            'additionalProperties': {'type': 'string'},
                            'title': 'Extra Headers',
                            'type': 'object',
                        },
                        'thinking': {
                            'anyOf': [
                                {'type': 'boolean'},
                                {'enum': ['minimal', 'low', 'medium', 'high', 'xhigh'], 'type': 'string'},
                            ],
                            'title': 'Thinking',
                        },
                        'service_tier': {
                            'enum': ['auto', 'default', 'flex', 'priority'],
                            'title': 'Service Tier',
                            'type': 'string',
                        },
                        'extra_body': {'title': 'Extra Body'},
                    },
                    'title': 'ModelSettings',
                    'type': 'object',
                },
                'ToolOrOutput': {
                    'properties': {
                        'function_tools': {'items': {'type': 'string'}, 'title': 'Function Tools', 'type': 'array'}
                    },
                    'required': ['function_tools'],
                    'title': 'ToolOrOutput',
                    'type': 'object',
                },
                'ToolSearchTool': {
                    'properties': {
                        'kind': {'default': 'tool_search', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'strategy': {
                            'anyOf': [{'enum': ['bm25', 'regex', 'custom'], 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Strategy',
                        },
                    },
                    'title': 'ToolSearchTool',
                    'type': 'object',
                },
                'UploadedFile': {
                    'properties': {
                        'file_id': {'title': 'File Id', 'type': 'string'},
                        'provider_name': {
                            'enum': [
                                'anthropic',
                                'openai',
                                'google',
                                'google-cloud',
                                'google-gla',
                                'google-vertex',
                                'bedrock',
                                'xai',
                            ],
                            'title': 'Provider Name',
                            'type': 'string',
                        },
                        'vendor_metadata': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Vendor Metadata',
                        },
                        'media_type': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Media Type',
                        },
                        'identifier': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Identifier',
                        },
                        'kind': {
                            'const': 'uploaded-file',
                            'default': 'uploaded-file',
                            'title': 'Kind',
                            'type': 'string',
                        },
                    },
                    'required': ['file_id', 'provider_name'],
                    'title': 'UploadedFile',
                    'type': 'object',
                },
                'WebFetchTool': {
                    'properties': {
                        'kind': {'default': 'web_fetch', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'max_uses': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Max Uses',
                        },
                        'allowed_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Allowed Domains',
                        },
                        'blocked_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Blocked Domains',
                        },
                        'enable_citations': {'default': False, 'title': 'Enable Citations', 'type': 'boolean'},
                        'max_content_tokens': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Max Content Tokens',
                        },
                    },
                    'title': 'WebFetchTool',
                    'type': 'object',
                },
                'WebSearchTool': {
                    'properties': {
                        'kind': {'default': 'web_search', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'search_context_size': {
                            'default': 'medium',
                            'enum': ['low', 'medium', 'high'],
                            'title': 'Search Context Size',
                            'type': 'string',
                        },
                        'user_location': {
                            'anyOf': [{'$ref': '#/$defs/WebSearchUserLocation'}, {'type': 'null'}],
                            'default': None,
                        },
                        'blocked_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Blocked Domains',
                        },
                        'allowed_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Allowed Domains',
                        },
                        'max_uses': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Max Uses',
                        },
                        'external_web_access': {
                            'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
                            'default': None,
                            'title': 'External Web Access',
                        },
                    },
                    'title': 'WebSearchTool',
                    'type': 'object',
                },
                'WebSearchUserLocation': {
                    'additionalProperties': False,
                    'properties': {
                        'city': {'title': 'City', 'type': 'string'},
                        'country': {'title': 'Country', 'type': 'string'},
                        'region': {'title': 'Region', 'type': 'string'},
                        'timezone': {'title': 'Timezone', 'type': 'string'},
                    },
                    'title': 'WebSearchUserLocation',
                    'type': 'object',
                },
                'XSearchTool': {
                    'properties': {
                        'kind': {'default': 'x_search', 'title': 'Kind', 'type': 'string'},
                        'optional': {'default': False, 'title': 'Optional', 'type': 'boolean'},
                        'allowed_x_handles': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Allowed X Handles',
                        },
                        'excluded_x_handles': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'title': 'Excluded X Handles',
                        },
                        'from_date': {
                            'anyOf': [{'format': 'date-time', 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'From Date',
                        },
                        'to_date': {
                            'anyOf': [{'format': 'date-time', 'type': 'string'}, {'type': 'null'}],
                            'default': None,
                            'title': 'To Date',
                        },
                        'enable_image_understanding': {
                            'default': False,
                            'title': 'Enable Image Understanding',
                            'type': 'boolean',
                        },
                        'enable_video_understanding': {
                            'default': False,
                            'title': 'Enable Video Understanding',
                            'type': 'boolean',
                        },
                        'include_output': {
                            'default': False,
                            'title': 'Include Output',
                            'type': 'boolean',
                        },
                    },
                    'title': 'XSearchTool',
                    'type': 'object',
                },
                'short_spec_NativeTool': {
                    'additionalProperties': False,
                    'properties': {
                        'NativeTool': {
                            'anyOf': [
                                {
                                    'oneOf': [
                                        {'$ref': '#/$defs/WebSearchTool'},
                                        {'$ref': '#/$defs/XSearchTool'},
                                        {'$ref': '#/$defs/CodeExecutionTool'},
                                        {'$ref': '#/$defs/WebFetchTool'},
                                        {'$ref': '#/$defs/ImageGenerationTool'},
                                        {'$ref': '#/$defs/MemoryTool'},
                                        {'$ref': '#/$defs/MCPServerTool'},
                                        {'$ref': '#/$defs/FileSearchTool'},
                                        {'$ref': '#/$defs/AdvisorTool'},
                                        {'$ref': '#/$defs/ToolSearchTool'},
                                    ]
                                },
                                {'type': 'null'},
                            ],
                            'title': 'Nativetool',
                        }
                    },
                    'title': 'short_spec_NativeTool',
                    'type': 'object',
                },
                'short_spec_MCP': {
                    'additionalProperties': False,
                    'properties': {'MCP': {'title': 'Mcp', 'type': 'string'}},
                    'required': ['MCP'],
                    'title': 'short_spec_MCP',
                    'type': 'object',
                },
                'spec_IncludeToolReturnSchemas': {
                    'additionalProperties': False,
                    'properties': {
                        'IncludeToolReturnSchemas': {'$ref': '#/$defs/spec_params_IncludeToolReturnSchemas'}
                    },
                    'required': ['IncludeToolReturnSchemas'],
                    'title': 'spec_IncludeToolReturnSchemas',
                    'type': 'object',
                },
                'short_spec_SetToolMetadata': {
                    'additionalProperties': False,
                    'properties': {
                        'SetToolMetadata': {
                            'anyOf': [
                                {'const': 'all', 'type': 'string'},
                                {'items': {'type': 'string'}, 'type': 'array'},
                                {'additionalProperties': True, 'type': 'object'},
                            ],
                            'title': 'Settoolmetadata',
                        }
                    },
                    'title': 'short_spec_SetToolMetadata',
                    'type': 'object',
                },
                'spec_ReinjectSystemPrompt': {
                    'additionalProperties': False,
                    'properties': {'ReinjectSystemPrompt': {'$ref': '#/$defs/spec_params_ReinjectSystemPrompt'}},
                    'required': ['ReinjectSystemPrompt'],
                    'title': 'spec_ReinjectSystemPrompt',
                    'type': 'object',
                },
                'spec_Thinking': {
                    'additionalProperties': False,
                    'properties': {'Thinking': {'$ref': '#/$defs/spec_params_Thinking'}},
                    'required': ['Thinking'],
                    'title': 'spec_Thinking',
                    'type': 'object',
                },
                'spec_ImageGeneration': {
                    'additionalProperties': False,
                    'properties': {'ImageGeneration': {'$ref': '#/$defs/spec_params_ImageGeneration'}},
                    'required': ['ImageGeneration'],
                    'title': 'spec_ImageGeneration',
                    'type': 'object',
                },
                'spec_RaiseContentFilterError': {
                    'additionalProperties': False,
                    'properties': {'RaiseContentFilterError': {'$ref': '#/$defs/spec_params_RaiseContentFilterError'}},
                    'required': ['RaiseContentFilterError'],
                    'title': 'spec_RaiseContentFilterError',
                    'type': 'object',
                },
                'spec_MCP': {
                    'additionalProperties': False,
                    'properties': {'MCP': {'$ref': '#/$defs/spec_params_MCP'}},
                    'required': ['MCP'],
                    'title': 'spec_MCP',
                    'type': 'object',
                },
                'spec_PrefixTools': {
                    'additionalProperties': False,
                    'properties': {'PrefixTools': {'$ref': '#/$defs/spec_params_PrefixTools'}},
                    'required': ['PrefixTools'],
                    'title': 'spec_PrefixTools',
                    'type': 'object',
                },
                'spec_ToolSearch': {
                    'additionalProperties': False,
                    'properties': {'ToolSearch': {'$ref': '#/$defs/spec_params_ToolSearch'}},
                    'required': ['ToolSearch'],
                    'title': 'spec_ToolSearch',
                    'type': 'object',
                },
                'spec_params_IncludeToolReturnSchemas': {
                    'additionalProperties': False,
                    'properties': {
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'tools': {
                            'anyOf': [
                                {'const': 'all', 'type': 'string'},
                                {'items': {'type': 'string'}, 'type': 'array'},
                                {'additionalProperties': True, 'type': 'object'},
                            ],
                            'title': 'Tools',
                        },
                    },
                    'title': 'spec_params_IncludeToolReturnSchemas',
                    'type': 'object',
                },
                'spec_WebFetch': {
                    'additionalProperties': False,
                    'properties': {'WebFetch': {'$ref': '#/$defs/spec_params_WebFetch'}},
                    'required': ['WebFetch'],
                    'title': 'spec_WebFetch',
                    'type': 'object',
                },
                'spec_WebSearch': {
                    'additionalProperties': False,
                    'properties': {'WebSearch': {'$ref': '#/$defs/spec_params_WebSearch'}},
                    'required': ['WebSearch'],
                    'title': 'spec_WebSearch',
                    'type': 'object',
                },
                'spec_XSearch': {
                    'additionalProperties': False,
                    'properties': {'XSearch': {'$ref': '#/$defs/spec_params_XSearch'}},
                    'required': ['XSearch'],
                    'title': 'spec_XSearch',
                    'type': 'object',
                },
                'spec_params_ReinjectSystemPrompt': {
                    'additionalProperties': False,
                    'properties': {
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'replace_existing': {'title': 'Replace Existing', 'type': 'boolean'},
                    },
                    'title': 'spec_params_ReinjectSystemPrompt',
                    'type': 'object',
                },
                'spec_params_Thinking': {
                    'additionalProperties': False,
                    'properties': {
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'effort': {
                            'anyOf': [
                                {'type': 'boolean'},
                                {'enum': ['minimal', 'low', 'medium', 'high', 'xhigh'], 'type': 'string'},
                            ],
                            'title': 'Effort',
                        },
                    },
                    'title': 'spec_params_Thinking',
                    'type': 'object',
                },
                'spec_params_ImageGeneration': {
                    'additionalProperties': False,
                    'properties': {
                        'native': {
                            'anyOf': [{'$ref': '#/$defs/ImageGenerationTool'}, {'type': 'boolean'}],
                            'title': 'Native',
                        },
                        'local': {'anyOf': [{'const': False, 'type': 'boolean'}, {'type': 'null'}], 'title': 'Local'},
                        'fallback_model': {
                            'anyOf': [{'$ref': '#/$defs/KnownModelName'}, {'type': 'string'}, {'type': 'null'}],
                            'title': 'Fallback Model',
                        },
                        'action': {
                            'anyOf': [{'enum': ['generate', 'edit', 'auto'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Action',
                        },
                        'background': {
                            'anyOf': [{'enum': ['transparent', 'opaque', 'auto'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Background',
                        },
                        'input_fidelity': {
                            'anyOf': [{'enum': ['high', 'low'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Input Fidelity',
                        },
                        'moderation': {
                            'anyOf': [{'enum': ['auto', 'low'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Moderation',
                        },
                        'image_model': {
                            'anyOf': [
                                {
                                    'enum': ['gpt-image-2', 'gpt-image-1.5', 'gpt-image-1', 'gpt-image-1-mini'],
                                    'type': 'string',
                                },
                                {'type': 'string'},
                                {'type': 'null'},
                            ],
                            'title': 'Image Model',
                        },
                        'output_compression': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'title': 'Output Compression',
                        },
                        'output_format': {
                            'anyOf': [{'enum': ['png', 'webp', 'jpeg'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Output Format',
                        },
                        'quality': {
                            'anyOf': [{'enum': ['low', 'medium', 'high', 'auto'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Quality',
                        },
                        'size': {
                            'anyOf': [
                                {
                                    'enum': ['auto', '1024x1024', '1024x1536', '1536x1024', '512', '1K', '2K', '4K'],
                                    'type': 'string',
                                },
                                {'type': 'null'},
                            ],
                            'title': 'Size',
                        },
                        'aspect_ratio': {
                            'anyOf': [
                                {
                                    'enum': ['21:9', '16:9', '4:3', '3:2', '1:1', '9:16', '3:4', '2:3', '5:4', '4:5'],
                                    'type': 'string',
                                },
                                {'type': 'null'},
                            ],
                            'title': 'Aspect Ratio',
                        },
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                    },
                    'title': 'spec_params_ImageGeneration',
                    'type': 'object',
                },
                'spec_params_RaiseContentFilterError': {
                    'additionalProperties': False,
                    'properties': {
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                    },
                    'title': 'spec_params_RaiseContentFilterError',
                    'type': 'object',
                },
                'spec_params_MCP': {
                    'additionalProperties': False,
                    'properties': {
                        'url': {'title': 'Url', 'type': 'string'},
                        'native': {
                            'anyOf': [{'$ref': '#/$defs/MCPServerTool'}, {'type': 'boolean'}],
                            'title': 'Native',
                        },
                        'local': {
                            'anyOf': [{'type': 'string'}, {'type': 'boolean'}, {'type': 'null'}],
                            'title': 'Local',
                        },
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'authorization_token': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'title': 'Authorization Token',
                        },
                        'headers': {
                            'anyOf': [{'additionalProperties': {'type': 'string'}, 'type': 'object'}, {'type': 'null'}],
                            'title': 'Headers',
                        },
                        'allowed_tools': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Allowed Tools',
                        },
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                    },
                    'required': ['url'],
                    'title': 'spec_params_MCP',
                    'type': 'object',
                },
                'spec_params_PrefixTools': {
                    'additionalProperties': False,
                    'properties': {
                        'prefix': {'title': 'Prefix', 'type': 'string'},
                        'capability': {
                            'anyOf': [
                                {'const': 'NativeTool', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_NativeTool'},
                                {'const': 'RaiseContentFilterError', 'type': 'string'},
                                {'$ref': '#/$defs/spec_RaiseContentFilterError'},
                                {'const': 'ImageGeneration', 'type': 'string'},
                                {'$ref': '#/$defs/spec_ImageGeneration'},
                                {'const': 'IncludeToolReturnSchemas', 'type': 'string'},
                                {'$ref': '#/$defs/spec_IncludeToolReturnSchemas'},
                                {'const': 'Instrumentation', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_MCP'},
                                {'$ref': '#/$defs/spec_MCP'},
                                {'$ref': '#/$defs/spec_PrefixTools'},
                                {'const': 'ReinjectSystemPrompt', 'type': 'string'},
                                {'$ref': '#/$defs/spec_ReinjectSystemPrompt'},
                                {'const': 'SetToolMetadata', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_SetToolMetadata'},
                                {'const': 'Thinking', 'type': 'string'},
                                {'$ref': '#/$defs/spec_Thinking'},
                                {'const': 'ToolSearch', 'type': 'string'},
                                {'$ref': '#/$defs/spec_ToolSearch'},
                                {'const': 'WebFetch', 'type': 'string'},
                                {'$ref': '#/$defs/spec_WebFetch'},
                                {'const': 'WebSearch', 'type': 'string'},
                                {'$ref': '#/$defs/spec_WebSearch'},
                                {'const': 'XSearch', 'type': 'string'},
                                {'$ref': '#/$defs/spec_XSearch'},
                            ]
                        },
                    },
                    'required': ['prefix', 'capability'],
                    'title': 'spec_params_PrefixTools',
                    'type': 'object',
                },
                'spec_params_ToolSearch': {
                    'additionalProperties': False,
                    'properties': {
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'strategy': {
                            'anyOf': [
                                {'const': 'keywords', 'type': 'string'},
                                {'enum': ['bm25', 'regex'], 'type': 'string'},
                                {'type': 'null'},
                            ],
                            'title': 'Strategy',
                        },
                        'max_results': {'title': 'Max Results', 'type': 'integer'},
                        'tool_description': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'title': 'Tool Description',
                        },
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'parameter_description': {
                            'anyOf': [{'type': 'string'}, {'type': 'null'}],
                            'title': 'Parameter Description',
                        },
                    },
                    'title': 'spec_params_ToolSearch',
                    'type': 'object',
                },
                'spec_params_WebFetch': {
                    'additionalProperties': False,
                    'properties': {
                        'native': {
                            'anyOf': [{'$ref': '#/$defs/WebFetchTool'}, {'type': 'boolean'}],
                            'title': 'Native',
                        },
                        'local': {'anyOf': [{'type': 'boolean'}, {'type': 'null'}], 'title': 'Local'},
                        'allowed_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Allowed Domains',
                        },
                        'blocked_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Blocked Domains',
                        },
                        'max_uses': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'Max Uses'},
                        'enable_citations': {
                            'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
                            'title': 'Enable Citations',
                        },
                        'max_content_tokens': {
                            'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                            'title': 'Max Content Tokens',
                        },
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                    },
                    'title': 'spec_params_WebFetch',
                    'type': 'object',
                },
                'spec_params_WebSearch': {
                    'additionalProperties': False,
                    'properties': {
                        'native': {
                            'anyOf': [{'$ref': '#/$defs/WebSearchTool'}, {'type': 'boolean'}],
                            'title': 'Native',
                        },
                        'local': {
                            'anyOf': [{'const': 'duckduckgo', 'type': 'string'}, {'type': 'boolean'}, {'type': 'null'}],
                            'title': 'Local',
                        },
                        'search_context_size': {
                            'anyOf': [{'enum': ['low', 'medium', 'high'], 'type': 'string'}, {'type': 'null'}],
                            'title': 'Search Context Size',
                        },
                        'user_location': {'anyOf': [{'$ref': '#/$defs/WebSearchUserLocation'}, {'type': 'null'}]},
                        'blocked_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Blocked Domains',
                        },
                        'allowed_domains': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Allowed Domains',
                        },
                        'max_uses': {'anyOf': [{'type': 'integer'}, {'type': 'null'}], 'title': 'Max Uses'},
                        'external_web_access': {
                            'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
                            'title': 'External Web Access',
                        },
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                    },
                    'title': 'spec_params_WebSearch',
                    'type': 'object',
                },
                'spec_params_XSearch': {
                    'additionalProperties': False,
                    'properties': {
                        'native': {'anyOf': [{'$ref': '#/$defs/XSearchTool'}, {'type': 'boolean'}], 'title': 'Native'},
                        'local': {'anyOf': [{'const': False, 'type': 'boolean'}, {'type': 'null'}], 'title': 'Local'},
                        'fallback_model': {
                            'anyOf': [{'$ref': '#/$defs/KnownModelName'}, {'type': 'string'}, {'type': 'null'}],
                            'title': 'Fallback Model',
                        },
                        'allowed_x_handles': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Allowed X Handles',
                        },
                        'excluded_x_handles': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'title': 'Excluded X Handles',
                        },
                        'from_date': {
                            'anyOf': [{'format': 'date-time', 'type': 'string'}, {'type': 'null'}],
                            'title': 'From Date',
                        },
                        'to_date': {
                            'anyOf': [{'format': 'date-time', 'type': 'string'}, {'type': 'null'}],
                            'title': 'To Date',
                        },
                        'enable_image_understanding': {
                            'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
                            'title': 'Enable Image Understanding',
                        },
                        'enable_video_understanding': {
                            'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
                            'title': 'Enable Video Understanding',
                        },
                        'include_output': {'anyOf': [{'type': 'boolean'}, {'type': 'null'}], 'title': 'Include Output'},
                        'id': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Id'},
                        'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Description'},
                        'defer_loading': {'title': 'Defer Loading', 'type': 'boolean'},
                    },
                    'title': 'spec_params_XSearch',
                    'type': 'object',
                },
            },
            'additionalProperties': False,
            'properties': {
                'model': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Model'},
                'name': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, 'title': 'Name'},
                'description': {
                    'anyOf': [{'type': 'string'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Description',
                },
                'instructions': {
                    'anyOf': [{'type': 'string'}, {'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Instructions',
                },
                'deps_schema': {
                    'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Deps Schema',
                },
                'output_schema': {
                    'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Output Schema',
                },
                'model_settings': {'anyOf': [{'$ref': '#/$defs/ModelSettings'}, {'type': 'null'}], 'default': None},
                'retries': {
                    'anyOf': [{'type': 'integer'}, {'$ref': '#/$defs/AgentRetries'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Retries',
                },
                'end_strategy': {
                    'default': 'graceful',
                    'enum': ['early', 'graceful', 'exhaustive'],
                    'title': 'End Strategy',
                    'type': 'string',
                },
                'tool_timeout': {
                    'anyOf': [{'type': 'number'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Tool Timeout',
                },
                'metadata': {
                    'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Metadata',
                },
                'capabilities': {
                    'default': [],
                    'items': {
                        'anyOf': [
                            {'const': 'NativeTool', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_NativeTool'},
                            {'const': 'RaiseContentFilterError', 'type': 'string'},
                            {'$ref': '#/$defs/spec_RaiseContentFilterError'},
                            {'const': 'ImageGeneration', 'type': 'string'},
                            {'$ref': '#/$defs/spec_ImageGeneration'},
                            {'const': 'IncludeToolReturnSchemas', 'type': 'string'},
                            {'$ref': '#/$defs/spec_IncludeToolReturnSchemas'},
                            {'const': 'Instrumentation', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_MCP'},
                            {'$ref': '#/$defs/spec_MCP'},
                            {'$ref': '#/$defs/spec_PrefixTools'},
                            {'const': 'ReinjectSystemPrompt', 'type': 'string'},
                            {'$ref': '#/$defs/spec_ReinjectSystemPrompt'},
                            {'const': 'SetToolMetadata', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_SetToolMetadata'},
                            {'const': 'Thinking', 'type': 'string'},
                            {'$ref': '#/$defs/spec_Thinking'},
                            {'const': 'ToolSearch', 'type': 'string'},
                            {'$ref': '#/$defs/spec_ToolSearch'},
                            {'const': 'WebFetch', 'type': 'string'},
                            {'$ref': '#/$defs/spec_WebFetch'},
                            {'const': 'WebSearch', 'type': 'string'},
                            {'$ref': '#/$defs/spec_WebSearch'},
                            {'const': 'XSearch', 'type': 'string'},
                            {'$ref': '#/$defs/spec_XSearch'},
                        ]
                    },
                    'title': 'Capabilities',
                    'type': 'array',
                },
                '$schema': {'type': 'string'},
            },
            'title': 'AgentSpec',
            'type': 'object',
        }
    )


def test_model_json_schema_with_custom_capabilities():
    schema = AgentSpec.model_json_schema_with_capabilities(
        custom_capability_types=[CustomCapability],
    )

    any_of = schema['properties']['capabilities']['items']['anyOf']

    capability_names: set[str] = set()
    for entry in any_of:
        if 'const' in entry:
            capability_names.add(entry['const'])
        elif '$ref' in entry:  # pragma: no branch
            ref = entry['$ref']
            ref_name = ref.rsplit('/', 1)[-1]
            for prefix in ('spec_', 'short_spec_'):
                if ref_name.startswith(prefix):
                    capability_names.add(ref_name[len(prefix) :])

    assert 'CustomCapability' in capability_names
    # Default capabilities should still be present
    assert 'WebSearch' in capability_names


def test_model_json_schema_filters_non_serializable_params():
    """Custom capabilities with non-serializable __init__ params get filtered in schema."""
    schema = AgentSpec.model_json_schema_with_capabilities(
        custom_capability_types=[CapabilityWithCallbackParam],
    )
    any_of = schema['properties']['capabilities']['items']['anyOf']

    # String form: all remaining params are optional
    has_string_form = any(e.get('const') == 'CapabilityWithCallbackParam' for e in any_of)
    assert has_string_form

    # Long form: max_retries and verbose survive; on_error (purely Callable) is filtered out
    spec_ref = next(
        (e for e in any_of if '$ref' in e and 'spec_CapabilityWithCallbackParam' in e['$ref']),
        None,
    )
    assert spec_ref is not None
    params_def = schema['$defs']['spec_params_CapabilityWithCallbackParam']
    assert 'max_retries' in params_def['properties']
    assert 'verbose' in params_def['properties']
    # on_error should not appear — purely Callable, entirely filtered out
    assert 'on_error' not in params_def['properties']
    # hooks should not appear — union of only non-serializable types, entirely filtered out
    assert 'hooks' not in params_def['properties']
    # verbose should be boolean only (Callable member was stripped from the union)
    assert params_def['properties']['verbose'] == {'title': 'Verbose', 'type': 'boolean'}


def test_agent_spec_schema_field_parity():
    """Ensure the schema model's fields stay in sync with AgentSpec."""
    schema = AgentSpec.model_json_schema_with_capabilities()
    schema_fields = set(schema['properties'].keys())

    # Map AgentSpec field names to their JSON schema names (using aliases)
    spec_fields: set[str] = set()
    for name, field_info in AgentSpec.model_fields.items():
        alias = field_info.alias
        spec_fields.add(alias if isinstance(alias, str) else name)

    assert schema_fields == spec_fields


def test_native_tools_param_wrapped_as_capabilities():
    """`Agent(capabilities=[NativeTool(...)])` produces NativeTool capabilities."""
    agent = Agent('test', capabilities=[NativeTool(WebSearchTool()), NativeTool(CodeExecutionTool())])
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, NativeToolCap)]
    assert len(builtin_caps) == 2
    assert isinstance(builtin_caps[0].tool, WebSearchTool)
    assert isinstance(builtin_caps[1].tool, CodeExecutionTool)
    # Also available via _cap_native_tools (ToolSearchTool is auto-injected).
    cap_tools = [t for t in agent._cap_native_tools if not isinstance(t, ToolSearchTool)]  # pyright: ignore[reportPrivateUsage]
    assert len(cap_tools) == 2


def test_agent_from_spec_builtin_tool():
    """NativeTool capability can be constructed from spec."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'NativeTool': {'kind': 'web_search'}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, NativeToolCap)]
    assert len(builtin_caps) == 1
    assert isinstance(builtin_caps[0].tool, WebSearchTool)


def test_agent_from_spec_builtin_tool_with_options():
    """NativeTool spec supports builtin tool configuration options."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'NativeTool': {'kind': 'web_search', 'search_context_size': 'high'}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, NativeToolCap)]
    assert len(builtin_caps) == 1
    tool = builtin_caps[0].tool
    assert isinstance(tool, WebSearchTool)
    assert tool.search_context_size == 'high'


def test_agent_from_spec_builtin_tool_explicit_form():
    """NativeTool spec supports the explicit {tool: ...} form."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'NativeTool': {'tool': {'kind': 'code_execution'}}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, NativeToolCap)]
    assert len(builtin_caps) == 1
    assert isinstance(builtin_caps[0].tool, CodeExecutionTool)


def test_save_schema(tmp_path: str):
    schema_path = Path(tmp_path) / 'agent_spec.schema.json'
    AgentSpec._save_schema(schema_path)  # pyright: ignore[reportPrivateUsage]

    assert schema_path.exists()
    import json

    schema = json.loads(schema_path.read_text(encoding='utf-8'))
    assert schema['type'] == 'object'
    assert 'model' in schema['properties']
    assert 'capabilities' in schema['properties']

    # Calling again should not rewrite if content matches
    mtime = schema_path.stat().st_mtime
    AgentSpec._save_schema(schema_path)  # pyright: ignore[reportPrivateUsage]
    assert schema_path.stat().st_mtime == mtime


def test_from_file_yaml(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\nname: my-agent\ninstructions: Be helpful\n', encoding='utf-8')
    spec = AgentSpec.from_file(spec_path)
    assert spec.model == 'test'
    assert spec.name == 'my-agent'
    assert spec.instructions == 'Be helpful'


def test_from_file_json(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.json'
    spec_path.write_text('{"model": "test", "name": "my-agent"}', encoding='utf-8')
    spec = AgentSpec.from_file(spec_path)
    assert spec.model == 'test'
    assert spec.name == 'my-agent'


def test_from_file_with_schema_field(tmp_path: str):
    """$schema field in the file should be accepted and not cause validation errors."""
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\n', encoding='utf-8')

    # YAML with $schema comment (ignored by yaml parser)
    spec_with_schema = Path(tmp_path) / 'agent_with_schema.json'
    spec_with_schema.write_text('{"$schema": "./agent_schema.json", "model": "test"}', encoding='utf-8')
    spec = AgentSpec.from_file(spec_with_schema)
    assert spec.model == 'test'
    assert spec.json_schema_path == './agent_schema.json'


def test_from_file_empty_yaml_raises_user_error(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('', encoding='utf-8')

    with pytest.raises(UserError, match='Agent spec must parse to an object, got NoneType'):
        AgentSpec.from_file(spec_path)


def test_from_file_json_array_raises_user_error(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.json'
    spec_path.write_text('[{"model": "test"}]', encoding='utf-8')

    with pytest.raises(UserError, match='Agent spec must parse to an object, got list'):
        AgentSpec.from_file(spec_path)


def test_agent_from_file_yaml(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\nname: my-agent\ninstructions: Be helpful\n', encoding='utf-8')
    agent = Agent.from_file(spec_path)
    assert agent.name == 'my-agent'
    assert 'Be helpful' in [sourced.instruction for sourced in agent._instructions]  # pyright: ignore[reportPrivateUsage]


def test_agent_from_file_json(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.json'
    spec_path.write_text('{"model": "test", "name": "json-agent"}', encoding='utf-8')
    agent = Agent.from_file(spec_path)
    assert agent.name == 'json-agent'


def test_agent_from_file_with_overrides(tmp_path: str):
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec_path.write_text('model: test\nname: spec-name\nretries: 5\n', encoding='utf-8')
    agent = Agent.from_file(spec_path, name='override-name', retries=2)
    assert agent.name == 'override-name'
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]


def test_to_file_yaml(tmp_path: str):
    spec = AgentSpec(model='test', name='my-agent', instructions='Be helpful')
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec.to_file(spec_path)

    content = spec_path.read_text(encoding='utf-8')
    # Should start with yaml-language-server schema comment
    assert content.startswith('# yaml-language-server: $schema=')
    assert 'model: test' in content
    assert 'name: my-agent' in content

    # Schema file should be generated
    schema_path = Path(tmp_path) / 'agent_schema.json'
    assert schema_path.exists()


def test_to_file_json(tmp_path: str):
    import json

    spec = AgentSpec(model='test', name='my-agent')
    spec_path = Path(tmp_path) / 'agent.json'
    spec.to_file(spec_path)

    data = json.loads(spec_path.read_text(encoding='utf-8'))
    assert data['$schema'] == 'agent_schema.json'
    assert data['model'] == 'test'
    assert data['name'] == 'my-agent'

    # Schema file should be generated
    schema_path = Path(tmp_path) / 'agent_schema.json'
    assert schema_path.exists()


def test_to_file_json_with_absolute_schema_path(tmp_path: Path):
    import json

    spec = AgentSpec(model='test', name='my-agent')
    spec_path = Path(tmp_path) / 'agent.json'
    schema_path = Path(tmp_path) / 'agent_schema.json'

    spec.to_file(spec_path, schema_path=schema_path)

    data = json.loads(spec_path.read_text(encoding='utf-8'))
    assert data['$schema'] == 'agent_schema.json'
    assert schema_path.exists()


def test_to_file_yaml_with_absolute_schema_path(tmp_path: Path):
    spec = AgentSpec(model='test', name='my-agent')
    spec_path = Path(tmp_path) / 'agent.yaml'
    schema_path = Path(tmp_path) / 'agent_schema.json'

    spec.to_file(spec_path, schema_path=schema_path)

    content = spec_path.read_text(encoding='utf-8')
    assert content.startswith('# yaml-language-server: $schema=agent_schema.json')
    assert schema_path.exists()


def test_to_file_json_with_external_absolute_schema_path(tmp_path: Path):
    import json

    spec = AgentSpec(model='test', name='my-agent')
    spec_dir = tmp_path / 'specs'
    schema_dir = tmp_path / 'schemas'
    spec_dir.mkdir()
    schema_dir.mkdir()
    spec_path = spec_dir / 'agent.json'
    schema_path = schema_dir / 'agent_schema.json'

    spec.to_file(spec_path, schema_path=schema_path)

    data = json.loads(spec_path.read_text(encoding='utf-8'))
    assert data['$schema'] == str(schema_path)
    assert schema_path.exists()


def test_to_file_no_schema(tmp_path: str):
    spec = AgentSpec(model='test')
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec.to_file(spec_path, schema_path=None)

    content = spec_path.read_text(encoding='utf-8')
    assert '# yaml-language-server' not in content

    # No schema file should be generated
    schema_path = Path(tmp_path) / 'agent_schema.json'
    assert not schema_path.exists()


def test_to_file_roundtrip_yaml(tmp_path: str):
    spec = AgentSpec(model='test', name='roundtrip', instructions=['Be helpful', 'Be concise'])
    spec_path = Path(tmp_path) / 'agent.yaml'
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == 'test'
    assert loaded.name == 'roundtrip'
    assert loaded.instructions == ['Be helpful', 'Be concise']


def test_to_file_roundtrip_json(tmp_path: str):
    spec = AgentSpec(model='test', name='roundtrip', retries={'tools': 3})
    spec_path = Path(tmp_path) / 'agent.json'
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == 'test'
    assert loaded.name == 'roundtrip'
    assert loaded.retries == {'tools': 3}


async def test_capability_returning_toolset_func():
    """Test that a capability returning a ToolsetFunc works with an agent."""
    agent = Agent(
        TestModel(),
        capabilities=[ToolsetFuncCapability()],
    )
    result = await agent.run('Greet Alice')

    tool_calls = list(iter_message_parts(result.all_messages(), ModelResponse, ToolCallPart))
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == 'greet'

    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


def test_infer_fmt_explicit():
    """_infer_fmt returns the explicit fmt when provided."""
    from pydantic_ai.agent.spec import _infer_fmt  # pyright: ignore[reportPrivateUsage]

    assert _infer_fmt(Path('agent.txt'), 'json') == 'json'
    assert _infer_fmt(Path('agent.txt'), 'yaml') == 'yaml'


def test_infer_fmt_unknown_extension():
    """_infer_fmt raises ValueError for unknown extension without explicit fmt."""
    from pydantic_ai.agent.spec import _infer_fmt  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(ValueError, match=re.escape("Could not infer format for filename 'agent.txt'")):
        _infer_fmt(Path('agent.txt'), None)


def test_invalid_custom_capability_type():
    """Passing a non-AbstractCapability subclass to model_json_schema_with_capabilities raises ValueError."""
    with pytest.raises(ValueError, match='must be subclasses of AbstractCapability'):
        AgentSpec.model_json_schema_with_capabilities(
            custom_capability_types=[str],  # type: ignore[list-item]
        )


def test_to_file_with_path_schema_path(tmp_path: str):
    """to_file works when schema_path is passed as a relative Path (not str), triggering the non-str branch."""
    spec = AgentSpec(model='test', name='path-schema')
    spec_path = Path(tmp_path) / 'agent.yaml'
    # Pass a relative Path (not str) to exercise the isinstance(schema_path, str) == False branch
    schema_path = Path('custom_schema.json')
    spec.to_file(spec_path, schema_path=schema_path)

    resolved_schema = Path(tmp_path) / 'custom_schema.json'
    assert resolved_schema.exists()
    content = spec_path.read_text(encoding='utf-8')
    assert 'model: test' in content


# --- from_spec error cases ---


def test_from_spec_no_model_raises():
    """from_spec() without model raises UserError."""
    with pytest.raises(UserError, match='`model` must be provided'):
        Agent.from_spec({'instructions': 'hello'})


# --- run() with spec: additional merge scenarios ---


class TestRunWithSpecAdditional:
    async def test_run_with_spec_and_run_instructions_merged(self):
        """When run() passes both instructions and spec instructions, they merge."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run(
            'hello',
            spec={'instructions': 'spec instructions'},
            instructions='run instructions',
        )
        assert 'run instructions' in result.output
        assert 'spec instructions' in result.output

    async def test_run_with_spec_metadata_only(self):
        """Spec metadata is used when run() doesn't pass metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('ok')

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run('hello', spec={'metadata': {'from': 'spec'}})
        assert result.metadata == {'from': 'spec'}

    async def test_run_with_spec_metadata_callable_merged(self):
        """Callable metadata from run() merges with spec metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('ok')

        agent = Agent(FunctionModel(model_fn))

        def dynamic_metadata(ctx: RunContext) -> dict[str, Any]:
            return {'dynamic': 'value'}

        result = await agent.run(
            'hello',
            spec={'metadata': {'spec_key': 'spec_val'}},
            metadata=dynamic_metadata,
        )
        assert result.metadata is not None
        assert result.metadata['spec_key'] == 'spec_val'
        assert result.metadata['dynamic'] == 'value'

    async def test_run_with_spec_model_settings_callable_passthrough(self):
        """Callable model_settings from run() bypasses spec model_settings merge."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            temperature = info.model_settings.get('temperature') if info.model_settings else None
            max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
            return make_text_response(f'temperature={temperature} max_tokens={max_tokens}')

        agent = Agent(FunctionModel(model_fn))

        def dynamic_settings(ctx: RunContext) -> _ModelSettings:
            return {'temperature': 0.9}

        result = await agent.run(
            'hello',
            spec={'model_settings': {'max_tokens': 100}},
            model_settings=dynamic_settings,
        )
        # Callable model_settings bypass spec merge — spec model_settings are handled
        # via the capability layer instead
        assert 'temperature=0.9' in result.output


# --- override() with spec: additional field tests ---


class TestOverrideWithSpecAdditional:
    async def test_override_with_spec_name(self):
        """Override with spec providing agent name."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('ok')

        agent = Agent(FunctionModel(model_fn), name='original')

        with agent.override(spec={'name': 'spec-name'}):
            assert agent.name == 'spec-name'
            result = await agent.run('hello')
        assert result.output == 'ok'
        assert agent.name == 'original'

    async def test_override_with_spec_model(self):
        """Override with spec providing model."""
        agent = Agent('test', name='test-agent')

        with agent.override(spec={'model': 'test'}):
            result = await agent.run('hello')
        assert result.output == 'success (no tool calls)'

    async def test_override_with_spec_model_settings(self):
        """Override with spec providing model_settings."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
            return make_text_response(f'max_tokens={max_tokens}')

        agent = Agent(FunctionModel(model_fn))

        with agent.override(spec={'model_settings': {'max_tokens': 42}}):
            result = await agent.run('hello')
        assert 'max_tokens=42' in result.output

    async def test_override_with_spec_metadata(self):
        """Override with spec providing metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('ok')

        agent = Agent(FunctionModel(model_fn))

        with agent.override(spec={'metadata': {'env': 'test'}}):
            result = await agent.run('hello')
        assert result.metadata == {'env': 'test'}


# --- Capability construction tests ---


def test_web_fetch_with_constraints():
    """WebFetch capability populates native tool with all constraint kwargs."""
    cap = WebFetch(
        local=True,
        allowed_domains=['example.com'],
        blocked_domains=['bad.com'],
        max_uses=5,
        enable_citations=True,
        max_content_tokens=1000,
    )
    builtin_tools = cap.get_native_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, WebFetchTool)
    assert tool.allowed_domains == ['example.com']
    assert tool.blocked_domains == ['bad.com']
    assert tool.max_uses == 5
    assert tool.enable_citations is True
    assert tool.max_content_tokens == 1000
    # `max_uses` requires native support; domains are handled locally.
    assert cap._requires_native() is True  # pyright: ignore[reportPrivateUsage]


def test_web_fetch_unique_id():
    """WebFetch returns the correct native unique_id."""
    cap = WebFetch(local=True)
    assert cap._native_unique_id() == 'web_fetch'  # pyright: ignore[reportPrivateUsage]


def test_xsearch_unique_id():
    """XSearch returns the correct builtin unique_id."""
    cap = XSearch()
    assert cap._native_unique_id() == 'x_search'  # pyright: ignore[reportPrivateUsage]


def test_web_search_with_constraints():
    """WebSearch capability populates native tool with all constraint kwargs."""
    from pydantic_ai.native_tools import WebSearchUserLocation

    cap = WebSearch(
        local='duckduckgo',
        search_context_size='high',
        user_location=WebSearchUserLocation(city='NYC', country='US'),
        blocked_domains=['bad.com'],
        allowed_domains=['good.com'],
        max_uses=3,
        external_web_access=False,
    )
    builtin_tools = cap.get_native_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.search_context_size == 'high'
    assert tool.user_location is not None
    assert tool.blocked_domains == ['bad.com']
    assert tool.allowed_domains == ['good.com']
    assert tool.max_uses == 3
    assert tool.external_web_access is False
    assert cap._requires_native() is True  # pyright: ignore[reportPrivateUsage]


def test_web_search_external_access_constraint():
    """Disabling live access suppresses local fallback; allowing it does not."""
    without_access = WebSearch(local=_noop_greet, external_web_access=False)
    assert without_access._requires_native() is True  # pyright: ignore[reportPrivateUsage]
    assert without_access.get_toolset() is None

    with_access = WebSearch(local=_noop_greet, external_web_access=True)
    assert with_access._requires_native() is False  # pyright: ignore[reportPrivateUsage]
    assert with_access.get_toolset() is not None

    with pytest.raises(UserError, match='constraint fields require the native tool'):
        WebSearch(native=False, local=_noop_greet, external_web_access=False)


def test_web_search_duckduckgo_raises_without_extra(monkeypatch: pytest.MonkeyPatch):
    """WebSearch(local='duckduckgo') raises with install hint when [duckduckgo] extra is missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.common_tools.duckduckgo':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[duckduckgo\]'):
        WebSearch(local='duckduckgo')


def test_web_fetch_local_true_raises_without_extra(monkeypatch: pytest.MonkeyPatch):
    """WebFetch(local=True) raises with install hint when [web-fetch] extra is missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.common_tools.web_fetch':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[web-fetch\]'):
        WebFetch(local=True)


def test_mcp_default_local_only():
    """MCP(url=...) defaults to local-only via the MCP SDK — no native advertised."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    cap = MCP(url='http://example.com/mcp', id='my-mcp')
    assert cap.get_native_tools() == []
    assert cap.get_toolset() is not None


def test_mcp_native_true_default_construction():
    """MCP(url=..., native=True) constructs MCPServerTool with id from url."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    cap = MCP(url='http://example.com/mcp', id='my-mcp', native=True)
    native_tools = cap.get_native_tools()
    assert len(native_tools) == 1
    tool = native_tools[0]
    assert isinstance(tool, MCPServerTool)
    assert tool.url == 'http://example.com/mcp'
    assert tool.id == 'my-mcp'


def test_mcp_default_raises_user_error_when_mcp_extra_missing(monkeypatch: pytest.MonkeyPatch):
    """`MCP(url=...)` raises a `UserError` with install hint when the MCP extra is missing.

    MCP defaults to running the server locally, so the extra is required. To run without it,
    the user must opt into native-only (`native=True, local=False`).
    """
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.mcp':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[mcp\]'):
        MCP(url='http://example.com/mcp')


def test_mcp_native_only_constructs_without_mcp_extra():
    """`MCP(url=..., native=True, local=False)` constructs cleanly — local resolution is skipped."""
    # Note: no need to mock the import. `local=False` short-circuits before `_build_local()`,
    # so the test exercises the same path whether or not the MCP extra is installed.
    cap = MCP(url='http://example.com/mcp', native=True, local=False)
    assert cap.local is False
    assert len(cap.get_native_tools()) == 1


def test_mcp_local_true_raises_user_error_when_mcp_extra_missing(monkeypatch: pytest.MonkeyPatch):
    """`MCP(url=..., local=True)` raises a `UserError` with install hint when MCP extra is missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.mcp':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[mcp\]'):
        MCP(url='http://example.com/mcp', local=True, native=True)


def test_mcp_local_string_raises_user_error_when_mcp_extra_missing(monkeypatch: pytest.MonkeyPatch):
    """`MCP(url=..., local='https://override...')` raises a `UserError` when MCP extra is missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.mcp':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[mcp\]'):
        MCP(url='http://example.com/mcp', local='https://override.example.com/mcp', native=True)


def test_mcp_native_default_raises_user_error_when_mcp_extra_missing(monkeypatch: pytest.MonkeyPatch):
    """`MCP(url=..., native=True)` (default `local`) now raises when `[mcp]` is missing.

    Previously `_default_local` swallowed `ImportError` and returned None, so
    `MCP(url=..., native=True)` would silently work as native-only. Locking in the new
    construction-time error so users get a clear migration to `native=True, local=False`.
    """
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.mcp':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[mcp\]'):
        MCP(url='http://example.com/mcp', native=True)


def test_mcp_without_url_with_local_toolset():
    """`MCP(local=MCPToolset(...))` constructs without `url=` — the primary path for non-URL clients."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    from pydantic_ai.mcp import MCPToolset

    toolset = MCPToolset('http://example.com/mcp', include_instructions=True)
    cap = MCP(local=toolset)
    assert cap.url is None
    assert cap.local is toolset
    assert cap.get_native_tools() == []


def test_mcp_without_url_with_native_true_raises():
    """`MCP(native=True)` without `url=` raises — capability needs a URL to auto-construct an MCPServerTool."""
    with pytest.raises(UserError, match=r'MCP\(native=True\) requires `url=`'):
        MCP(native=True, local=False)


def test_mcp_without_url_with_explicit_native_instance():
    """`MCP(native=MCPServerTool(...))` constructs without capability `url=` — the instance carries the URL."""
    cap = MCP(
        native=MCPServerTool(id='my-mcp', url='http://example.com/mcp'),
        local=False,
    )
    assert cap.url is None
    natives = cap.get_native_tools()
    assert len(natives) == 1
    assert isinstance(natives[0], MCPServerTool)
    assert natives[0].url == 'http://example.com/mcp'


def test_mcp_without_url_local_true_raises():
    """`MCP(local=True)` without `url=` raises — no URL to derive the local transport from."""
    with pytest.raises(UserError, match=r'requires `url=`'):
        MCP(local=True)


def test_native_or_local_constraint_check_precedes_no_local_check():
    """`WebSearch(native=False, allowed_domains=...)` raises the constraint error, not the no-local error.

    Regression test for validation-order bug — the constraint case is unfixable by adding `local=`,
    so it must fire before the `requires an explicit local tool` check.
    """
    with pytest.raises(UserError, match='constraint fields require the native tool'):
        WebSearch(native=False, allowed_domains=['example.com'])


def test_web_search_local_string_strategy_silent():
    """WebSearch(local='duckduckgo') resolves silently to the DDG tool — no PydanticAIDeprecationWarning."""
    pytest.importorskip('duckduckgo_search', reason='duckduckgo extra not installed')
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        cap = WebSearch(local='duckduckgo')
    assert cap.local is not None and cap.local is not False


def test_web_search_local_true_silent():
    """WebSearch(local=True) resolves silently to the default strategy (DDG)."""
    pytest.importorskip('duckduckgo_search', reason='duckduckgo extra not installed')
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        cap = WebSearch(local=True)
    assert cap.local is not None and cap.local is not False


def test_web_fetch_local_true_silent():
    """WebFetch(local=True) resolves silently to the default markdownify-based tool."""
    pytest.importorskip('markdownify', reason='web-fetch extra not installed')
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        cap = WebFetch(local=True)
    assert cap.local is not None and cap.local is not False


def test_mcp_local_true_silent_with_explicit_native():
    """MCP(url=..., local=True, native=True) resolves silently — no PydanticAIDeprecationWarning."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticAIDeprecationWarning)
        cap = MCP(url='http://example.com/mcp', local=True, native=True)
    assert cap.local is not None and cap.local is not False
    assert len(cap.get_native_tools()) == 1


def test_native_or_local_base_no_default_native():
    """NativeOrLocalTool base class with native=True raises (no _default_native)."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    with pytest.raises(UserError, match='native=True requires a subclass'):
        NativeOrLocalTool()


def test_native_tool_from_spec_no_args():
    """NativeTool.from_spec() with no arguments raises TypeError."""
    from pydantic_ai.capabilities.native_tool import NativeTool as NativeToolCapDirect

    with pytest.raises(TypeError, match='requires either a `tool` argument'):
        NativeToolCapDirect.from_spec()


def test_native_or_local_no_default_local():
    """NativeOrLocalTool base class _default_local() returns None."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    cap = NativeOrLocalTool(native=WebSearchTool())
    # Base class _default_local() returns None — no local fallback
    assert cap.local is None
    assert cap.get_toolset() is None


def test_native_or_local_with_explicit_native():
    """NativeOrLocalTool used directly with an explicit native and local tool."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    def my_local_tool() -> str:
        """A local fallback tool."""
        return 'local result'  # pragma: no cover

    cap = NativeOrLocalTool(native=WebSearchTool(), local=my_local_tool)
    # get_native_tools returns the explicit native tool
    assert len(cap.get_native_tools()) == 1
    assert isinstance(cap.get_native_tools()[0], WebSearchTool)
    # get_toolset wraps local with unless_native from _native_unique_id()
    toolset = cap.get_toolset()
    assert toolset is not None


def test_native_or_local_native_unique_id_non_abstract():
    """_native_unique_id() raises when native is callable (not AbstractNativeTool)."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    cap = NativeOrLocalTool.__new__(NativeOrLocalTool)
    cap.native = lambda ctx: WebSearchTool()
    cap.local = False

    with pytest.raises(UserError, match='cannot derive native unique_id'):
        cap._native_unique_id()  # pyright: ignore[reportPrivateUsage]


def test_native_or_local_base_unknown_strategy_raises():
    """`NativeOrLocalTool(local='foo')` raises a UserError from the default `_resolve_local_strategy`."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool

    with pytest.raises(UserError, match=r"`local='foo'` is not supported"):
        NativeOrLocalTool(native=WebSearchTool(), local='foo')


def test_native_or_local_preserves_passed_tool_instance():
    """A pre-wrapped `Tool` passed as `local` is preserved (not re-wrapped or treated as a callable)."""
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool
    from pydantic_ai.tools import Tool as ToolDirect

    def my_search(query: str) -> str:
        return f'results for {query}'  # pragma: no cover

    tool = ToolDirect(my_search)
    cap = NativeOrLocalTool(native=WebSearchTool(), local=tool)
    assert cap.local is tool


def test_native_or_local_id_kwarg_overrides_default():
    """`id=` overrides the auto-derived capability id across `NativeOrLocalTool` subclasses.

    The id is the wire-side identifier (used in `ctx.capabilities` lookup and surfaced to the model
    in the deferred-capability catalog), so users need a way to disambiguate when they instantiate
    the same capability twice in one agent.
    """
    from pydantic_ai.capabilities.native_or_local import NativeOrLocalTool
    from pydantic_ai.tools import Tool as ToolDirect

    def _nop() -> None:
        return None  # pragma: no cover

    nop = ToolDirect(_nop)

    assert NativeOrLocalTool(native=WebSearchTool(), local=nop, id='custom').id == 'custom'
    assert WebFetch(local=nop, id='custom').id == 'custom'
    assert ImageGeneration(local=False, id='custom').id == 'custom'


def test_websearch_unknown_strategy_raises():
    """WebSearch(local='not_a_real_strategy') → UserError naming the unknown strategy."""
    with pytest.raises(UserError, match='not a known strategy'):
        WebSearch(local='not_a_real_strategy')  # type: ignore[arg-type]


def test_websearch_duckduckgo_missing_install_hint(monkeypatch: pytest.MonkeyPatch):
    """`WebSearch(local='duckduckgo')` raises a UserError with install hint when the extra is missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.common_tools.duckduckgo':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[duckduckgo\]'):
        WebSearch(local='duckduckgo')


def test_webfetch_unknown_strategy_raises():
    """WebFetch(local='not_a_real_strategy') → UserError naming the unknown strategy."""
    with pytest.raises(UserError, match='not a known strategy'):
        WebFetch(local='not_a_real_strategy')  # type: ignore[arg-type]


def test_webfetch_local_true_install_hint(monkeypatch: pytest.MonkeyPatch):
    """`WebFetch(local=True)` raises a UserError with install hint when the `web-fetch` extra is missing."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.common_tools.web_fetch':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.raises(UserError, match=r'pydantic-ai-slim\[web-fetch\]'):
        WebFetch(local=True)


def test_mcp_local_string_must_be_url_raises_user_error():
    """`MCP(url=..., local='not-a-url')` raises a `UserError` directing the user to `local=MCPToolset(...)`."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    with pytest.raises(UserError, match=r"MCP\(local='not_a_real_strategy'\) must be an `http\(s\)://` URL"):
        MCP(url='http://example.com/mcp', local='not_a_real_strategy', native=True)


def test_mcp_local_url_string_override_uses_provided_url():
    """`MCP(url=..., local='https://override...')` builds an `MCPToolset` from the override URL."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    pytest.importorskip('fastmcp', reason='fastmcp package not installed')
    from pydantic_ai.mcp import MCPToolset

    cap = MCP(
        url='http://primary.example.com/mcp',
        local='https://override.example.com/mcp',
        native=True,
    )
    assert isinstance(cap.local, MCPToolset)


def test_validate_capability_not_dataclass():
    """Custom capability type without @dataclass raises ValueError."""
    from pydantic_ai.agent.spec import get_capability_registry

    class NotADataclass(AbstractCapability[Any]):
        pass

    with pytest.raises(ValueError, match='must be decorated with `@dataclass`'):
        get_capability_registry(custom_types=(NotADataclass,))


async def test_deferred_capability_without_id_set_after_construction_raises_at_run() -> None:
    """`defer_loading` flipped on after construction escapes the eager check, so the run-time guard still fires."""

    @dataclass
    class DeferredCap(AbstractCapability):
        pass

    cap = DeferredCap()
    # Not deferred at construction, so the eager check passes; the run-time check is what catches it.
    agent = Agent(TestModel(), capabilities=[cap])
    cap.defer_loading = True
    assert cap.id is None

    with pytest.raises(UserError, match='stable explicit `id` values'):
        await agent.run('hi')

    assert DeferredCap(id='stable', defer_loading=True).id == 'stable'


async def test_plain_class_capability_can_use_class_metadata() -> None:
    """A plain class subclass can declare metadata without dataclass or super calls."""

    class DeferredCap(AbstractCapability):
        id = 'plain-deferred'
        description = 'Plain class deferred capability.'
        defer_loading = True

    cap = DeferredCap()
    capability_map, available_ids = await _registered_capability_context(cap)

    assert capability_map == {'plain-deferred': cap}
    assert 'plain-deferred' not in available_ids
    assert cap.defer_loading is True
    assert cap.get_description() == 'Plain class deferred capability.'


async def test_custom_init_capability_can_initialize_metadata_without_post_init() -> None:
    """Custom capability init can initialize metadata without a base-class ritual."""

    class DeferredCap(AbstractCapability):
        def __init__(self, *, id: str | None = None, defer_loading: bool = False) -> None:
            self.id = id
            self.description = None
            self.defer_loading = defer_loading

    cap = DeferredCap(id='stable', defer_loading=True)
    capability_map, available_ids = await _registered_capability_context(cap)

    assert cap.id == 'stable'
    assert cap.defer_loading is True
    assert capability_map == {'stable': cap}
    assert 'stable' not in available_ids

    non_deferred_cap = DeferredCap()
    non_deferred_capability_map, non_deferred_available_ids = await _registered_capability_context(non_deferred_cap)
    assert non_deferred_cap.id is None
    assert non_deferred_cap.description is None
    assert non_deferred_cap.defer_loading is False
    assert non_deferred_capability_map == {'deferred_cap': non_deferred_cap}
    assert 'deferred_cap' in non_deferred_available_ids


async def test_duplicate_explicit_capability_ids_set_after_construction_raise_at_run() -> None:
    """Ids that only collide after construction escape the eager check, so run registration still rejects them."""

    @dataclass
    class FirstCap(AbstractCapability):
        pass

    @dataclass
    class SecondCap(AbstractCapability):
        pass

    first = FirstCap(id='same')
    second = SecondCap()  # no id at construction, so the eager check passes
    agent = Agent(TestModel(), capabilities=[first, second])
    second.id = 'same'  # collision introduced after construction

    with pytest.raises(UserError, match="Capability id 'same' is used by multiple capabilities"):
        await agent.run('hi')


async def test_anonymous_non_deferred_capabilities_get_run_local_ids() -> None:
    """Anonymous non-deferred capabilities are still present in run context."""

    @dataclass
    class PlainCap(AbstractCapability):
        pass

    first = PlainCap()
    second = PlainCap()
    capability_map, available_ids = await _registered_capability_context(first, second)

    assert list(capability_map) == ['plain_cap', 'plain_cap_2']
    assert first.id is None
    assert second.id is None
    assert {'plain_cap', 'plain_cap_2'} <= available_ids
