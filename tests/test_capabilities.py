from __future__ import annotations

import asyncio
import inspect
import re
import threading
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from types import NoneType
from typing import Any, cast

import anyio
import pytest
from pydantic import BaseModel

from pydantic_ai._run_context import RunContext
from pydantic_ai._spec import CapabilitySpec, NamedSpec
from pydantic_ai._utils import Some
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import Agent
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    MCP,
    Capability,
    CapabilityOrdering,
    DynamicCapability,
    ImageGeneration,
    IncludeToolReturnSchemas,
    Instrumentation,
    NativeTool,
    PrefixTools,
    PrepareTools,
    RaiseContentFilterError,
    ReinjectSystemPrompt,
    ResolveModelId,
    SelectModel,
    SetToolMetadata,
    Thinking,
    ToolSearch,
    Toolset,
    UseThreadExecutor,
    WebFetch,
    WebSearch,
    WrapperCapability,
    XSearch,
)
from pydantic_ai.capabilities._dynamic import ResolvedDynamicCapability
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.capabilities.native_tool import NativeTool as NativeToolCap
from pydantic_ai.exceptions import (
    ModelRetry,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryImage,
    FilePart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    ModelSelectionContext,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    CodeExecutionTool,
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
    XSearchTool,
)
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.output import OutputContext, ToolOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.result import FinalResult
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetFunc, ToolsetTool, WrapperToolset
from pydantic_ai.toolsets._capability_owned import (
    resolve_capability_id,
    tool_defs_from_pre_definition_load_returns,
)
from pydantic_ai.toolsets._deferred_capability_loader import (
    LOAD_CAPABILITY_TOOL_NAME,
)
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .capability_models import (
    make_text_response,
    simple_model_function,
    simple_stream_function,
    tool_calling_model,
)
from .conftest import IsDatetime, IsInstance, IsStr, iter_message_parts, message, remove_schema_descriptions

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


class MyOutput(BaseModel):
    value: int


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
class CustomCapability(AbstractCapability):
    greeting: str = 'hello'


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
    assert 'Be helpful.' in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_list():
    agent = Agent.from_spec({'model': 'test', 'instructions': ['First.', 'Second.']})
    assert 'First.' in agent._instructions  # pyright: ignore[reportPrivateUsage]
    assert 'Second.' in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_merged():
    agent = Agent.from_spec(
        {'model': 'test', 'instructions': 'From spec.'},
        instructions='From arg.',
    )
    assert 'From spec.' in agent._instructions  # pyright: ignore[reportPrivateUsage]
    assert 'From arg.' in agent._instructions  # pyright: ignore[reportPrivateUsage]


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
                        'anthropic:claude-opus-4-1',
                        'anthropic:claude-opus-4-1-20250805',
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
                        'cerebras:gpt-oss-120b',
                        'cerebras:llama3.1-8b',
                        'cerebras:qwen-3-235b-a22b-instruct-2507',
                        'cerebras:zai-glm-4.7',
                        'cohere:c4ai-aya-expanse-32b',
                        'cohere:c4ai-aya-expanse-8b',
                        'cohere:command-nightly',
                        'cohere:command-r-08-2024',
                        'cohere:command-r-plus-08-2024',
                        'cohere:command-r7b-12-2024',
                        'deepseek:deepseek-chat',
                        'deepseek:deepseek-reasoner',
                        'deepseek:deepseek-v4-flash',
                        'deepseek:deepseek-v4-pro',
                        'gateway/anthropic:claude-fable-5',
                        'gateway/anthropic:claude-haiku-4-5',
                        'gateway/anthropic:claude-haiku-4-5-20251001',
                        'gateway/anthropic:claude-opus-4-1',
                        'gateway/anthropic:claude-opus-4-1-20250805',
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
                        'gateway/openai:gpt-5.6-luna',
                        'gateway/openai:gpt-5.6-sol',
                        'gateway/openai:gpt-5.6-terra',
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
                        'openai-chat:gpt-5.6-luna',
                        'openai-chat:gpt-5.6-sol',
                        'openai-chat:gpt-5.6-terra',
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
                        'openai:gpt-5.6-luna',
                        'openai:gpt-5.6-sol',
                        'openai:gpt-5.6-terra',
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
    assert 'Be helpful' in agent._instructions  # pyright: ignore[reportPrivateUsage]


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


@dataclass
class ToolsetFuncCapability(AbstractCapability):
    """A capability that returns a ToolsetFunc instead of an AbstractToolset."""

    def get_toolset(self) -> ToolsetFunc:
        def make_toolset(ctx: RunContext) -> AbstractToolset:
            toolset = FunctionToolset()

            @toolset.tool_plain
            def greet(name: str) -> str:
                """Greet someone by name."""
                return f'Hello, {name}!'

            return toolset

        return make_toolset


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


async def test_runtime_capability_contributions_applied():
    """Run-time `capabilities=` contributions (tools, instructions, etc.) must be applied.

    Regression guard: the `source_cap` selection previously only checked for `override()`
    or spec capabilities, so tool contributions from a capability passed only via
    `Agent.run(capabilities=[...])` were silently dropped.
    """
    agent = Agent(TestModel())
    result = await agent.run('Greet Alice', capabilities=[ToolsetFuncCapability()])

    tool_calls = list(iter_message_parts(result.all_messages(), ModelResponse, ToolCallPart))
    assert [c.tool_name for c in tool_calls] == ['greet']


async def test_capability_returning_toolset_func_combined():
    """Test that a ToolsetFunc capability works alongside other capabilities via CombinedCapability."""
    agent = Agent(
        TestModel(),
        instructions='You are a helpful greeter.',
        capabilities=[
            ToolsetFuncCapability(),
        ],
    )
    result = await agent.run('Greet Bob')

    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


def test_abstract_capability_get_model_settings_default():
    """AbstractCapability.get_model_settings() returns None by default."""

    @dataclass
    class PlainCap(AbstractCapability):
        pass

    cap = PlainCap()
    assert cap.get_model_settings() is None
    assert cap.get_description() is None


async def test_abstract_capability_description_field_is_optional_in_deferred_catalog() -> None:
    """Deferred capability catalog entries can include a description but do not require one."""

    @dataclass
    class AccountSecurityRunbook(AbstractCapability):
        id: str | None = 'account-security'
        description: str | None = 'Use for suspicious logins, account takeover, or session revocation.'
        defer_loading: bool = True

    @dataclass
    class RefundsRunbook(AbstractCapability):
        id: str | None = 'refunds'
        defer_loading: bool = True

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(model_fn), capabilities=[AccountSecurityRunbook(), RefundsRunbook()])
    result = await agent.run('hi')
    request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))

    assert request.instructions == snapshot(
        """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- account-security: Use for suspicious logins, account takeover, or session revocation.
- refunds\
"""
    )


async def test_deferred_capability_catalog_mentions_search_only_when_search_surface_exists() -> None:
    """The catalog steers away from tool search only in runs that actually offer a search surface.

    The surface exists exactly when `ToolSearch` (installed explicitly, or auto-injected by a
    searchable deferred tool) has a non-empty corpus — the run then carries the `search_tools`
    definition even when a native search surface will replace it on the wire. In a
    capability-only run there is nothing to search with, so mentioning searching would name an
    affordance that doesn't exist and invite hallucinated search calls.
    """

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    refunds = Capability[object](id='refunds', description='Refund tools.', defer_loading=True)

    async def first_request_instructions(agent: Agent[None, str]) -> str | None:
        result = await agent.run('hi')
        request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))
        return request.instructions

    assert await first_request_instructions(Agent(FunctionModel(model_fn), capabilities=[refunds])) == snapshot(
        "The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:\n"
        '- refunds: Refund tools.'
    )

    searchable_toolset = FunctionToolset()

    @searchable_toolset.tool_plain(defer_loading=True)
    def weather_forecast() -> str:  # pragma: no cover
        """Look up a weather forecast."""
        return 'sunny'

    assert await first_request_instructions(
        Agent(FunctionModel(model_fn), capabilities=[ToolSearch(), refunds], toolsets=[searchable_toolset])
    ) == snapshot(
        "The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded — load the capability first rather than searching for its tools:\n"
        '- refunds: Refund tools.'
    )

    # Without an explicit `ToolSearch`, a searchable deferred tool auto-injects one — the run
    # still offers a search surface, so the steering variant is still correct.
    assert await first_request_instructions(
        Agent(FunctionModel(model_fn), capabilities=[refunds], toolsets=[searchable_toolset])
    ) == snapshot(
        "The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded — load the capability first rather than searching for its tools:\n"
        '- refunds: Refund tools.'
    )

    # A named-native strategy registers no local `search_tools` fallback, but the run's search
    # surface is no less real for going native — the steering variant must still be picked.
    assert await first_request_instructions(
        Agent(
            FunctionModel(model_fn),
            capabilities=[ToolSearch(strategy='bm25'), refunds],
            toolsets=[searchable_toolset],
        )
    ) == snapshot(
        "The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded — load the capability first rather than searching for its tools:\n"
        '- refunds: Refund tools.'
    )


async def test_deferred_capability_catalog_bytes_stable_across_turns() -> None:
    """The catalog instruction is byte-identical on every request within a run.

    This is a multi-request property — a single-request snapshot proves correct variant
    selection, not stability. The run below searches, loads a capability, and finishes; a
    catalog that reacted to either event (variant flip, entry annotation) would change the
    instructions and bust the prompt-cache prefix at its very front.
    """
    instructions_seen: list[str | None] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        instructions_seen.append(request.instructions)
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not tool_returns:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=_SEARCH_TOOLS_NAME, args={'queries': ['weather']}, tool_call_id='s1')]
            )
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='l1')]
            )
        return ModelResponse(parts=[TextPart('done')])

    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain
    def lookup_refund_policy() -> str:  # pragma: no cover
        """Look up refund policy."""
        return 'ok'

    searchable_toolset = FunctionToolset()

    @searchable_toolset.tool_plain(defer_loading=True)
    def weather_forecast() -> str:  # pragma: no cover
        """Look up a weather forecast."""
        return 'sunny'

    refunds = Capability[object](
        id='refunds', description='Refund tools.', toolsets=[refunds_toolset], defer_loading=True
    )
    agent = Agent(FunctionModel(model_fn), capabilities=[refunds], toolsets=[searchable_toolset])
    result = await agent.run('search then load')

    assert result.output == 'done'
    assert len(instructions_seen) == 3
    assert len(set(instructions_seen)) == 1
    assert instructions_seen[0] is not None and 'rather than searching' in instructions_seen[0]


async def test_capability_description_can_be_dynamic() -> None:
    """The convenience Capability accepts a CapabilityDescription callable."""

    def describe(ctx: RunContext[str]) -> str:
        return f'Use for {ctx.deps} questions.'

    agent = Agent(
        FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('done')])),
        deps_type=str,
        capabilities=[Capability[str](id='dynamic-description', description=describe, defer_loading=True)],
    )

    result = await agent.run('hi', deps='billing')
    request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))

    assert request.instructions == snapshot(
        """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- dynamic-description: Use for billing questions.\
"""
    )


def test_combined_capability_get_model_settings_merge():
    """CombinedCapability.get_model_settings() merges settings from all sub-capabilities."""

    @dataclass
    class MaxTokensCap(AbstractCapability):
        def get_model_settings(self) -> _ModelSettings | None:
            return _ModelSettings(max_tokens=100)

    @dataclass
    class TemperatureCap(AbstractCapability):
        def get_model_settings(self) -> _ModelSettings | None:
            return _ModelSettings(temperature=0.5)

    caps = CombinedCapability(
        capabilities=[
            MaxTokensCap(),
            TemperatureCap(),
        ]
    )
    merged = caps.get_model_settings()
    assert merged is not None
    assert not callable(merged)
    assert merged.get('max_tokens') == 100
    assert merged.get('temperature') == 0.5


def test_combined_capability_get_model_settings_none():
    """CombinedCapability.get_model_settings() returns None when no capabilities provide settings."""

    @dataclass
    class PlainCap(AbstractCapability):
        pass

    caps = CombinedCapability(capabilities=[PlainCap()])
    assert caps.get_model_settings() is None


def test_combined_capability_get_model_settings_deferred():
    """Deferred capability model settings resolve only after the capability is loaded."""
    seen_dynamic_loaded: list[bool | None] = []

    @dataclass
    class StaticSettingsCap(AbstractCapability):
        def get_model_settings(self) -> _ModelSettings:
            return _ModelSettings(max_tokens=123)

    @dataclass
    class DynamicSettingsCap(AbstractCapability):
        def get_model_settings(self) -> Callable[[RunContext], _ModelSettings]:
            def settings(ctx: RunContext) -> _ModelSettings:
                seen_dynamic_loaded.append(ctx.capability_loaded)
                return _ModelSettings(temperature=0.2)

            return settings

    resolver = CombinedCapability(
        [
            StaticSettingsCap(id='static-settings', defer_loading=True),
            DynamicSettingsCap(id='dynamic-settings', defer_loading=True),
        ]
    ).get_model_settings()

    assert callable(resolver)

    def resolve(loaded_capability_ids: set[str]) -> _ModelSettings:
        return resolver(
            RunContext(
                deps=None,
                model=TestModel(),
                usage=RunUsage(),
                loaded_capability_ids=loaded_capability_ids,
            )
        )

    assert [
        resolve(set()),
        resolve({'static-settings'}),
        resolve({'static-settings', 'dynamic-settings'}),
    ] == snapshot(
        [
            {},
            {'max_tokens': 123},
            {'max_tokens': 123, 'temperature': 0.2},
        ]
    )
    assert seen_dynamic_loaded == [True]


async def test_deferred_hooks_do_not_fire_until_capability_is_loaded() -> None:
    """Hooks owned by a deferred capability are skipped until `load_capability` succeeds."""
    hooks = Hooks(id='audit', description='Audit request flow.', defer_loading=True)
    seen_loaded: list[bool | None] = []

    @hooks.on.before_model_request
    async def record(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext:
        seen_loaded.append(ctx.capability_loaded)
        return request_context

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        already_loaded = any(
            isinstance(part, LoadCapabilityReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if not already_loaded:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'audit'},
                        tool_call_id='load-audit',
                    )
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[hooks])
    result = await agent.run('hello')

    assert result.output == 'done'
    assert seen_loaded == [True]


def test_toolset_capability_get_toolset():
    """Toolset capability returns its toolset."""
    ts = FunctionToolset()
    cap = Toolset(toolset=ts)
    assert cap.get_toolset() is ts

    convenience_cap = Capability[object](toolsets=[ts])
    assert convenience_cap.get_toolset() is ts

    ts_b = FunctionToolset()
    combined_cap = Capability[object](toolsets=[ts, ts_b])
    from pydantic_ai.toolsets import CombinedToolset

    combined = cast(CombinedToolset, combined_cap.get_toolset())
    assert list(combined.toolsets) == [ts, ts_b]


def test_capability_stamps_id_on_contributed_function_toolset():
    """A capability's `id` is stamped on its contributed function toolset so it can be used with
    durable execution, which wraps leaf toolsets by `id` at construction time. User-provided
    toolsets keep their own ids and are never overwritten."""
    from pydantic_ai.toolsets import CombinedToolset

    def my_tool(x: int) -> int:
        return x + 1  # pragma: no cover

    stamped = Capability[object](id='billing', tools=[my_tool]).get_toolset()
    assert isinstance(stamped, FunctionToolset)
    assert stamped.id == 'billing'

    # No id → stays None (status quo; setting `id=` is what makes durable-exec errors actionable).
    unstamped = Capability[object](tools=[my_tool]).get_toolset()
    assert isinstance(unstamped, FunctionToolset)
    assert unstamped.id is None

    # An empty capability still returns its (live) function toolset carrying the id.
    empty = Capability[object](id='billing').get_toolset()
    assert isinstance(empty, FunctionToolset)
    assert empty.id == 'billing'

    # Combined with a user toolset: the function toolset gets the capability id; the user toolset
    # keeps its own id.
    user_toolset = FunctionToolset[object](id='user-ts')
    combined = cast(
        CombinedToolset, Capability[object](id='billing', tools=[my_tool], toolsets=[user_toolset]).get_toolset()
    )
    function_toolset, provided = combined.toolsets
    assert isinstance(function_toolset, FunctionToolset)
    assert function_toolset.id == 'billing'
    assert provided is user_toolset


def test_native_or_local_stamps_id_on_local_toolset():
    """`NativeOrLocalTool` stamps its `id` on the FunctionToolset wrapping a bare local callable, so
    the local fallback can be used with durable execution."""
    from pydantic_ai.capabilities import NativeOrLocalTool
    from pydantic_ai.toolsets import PreparedToolset

    def local_search(query: str) -> str:
        return 'result'  # pragma: no cover

    cap = NativeOrLocalTool[object](native=WebSearchTool(), local=local_search, id='search')
    toolset = cap.get_toolset()
    # native + local → the local FunctionToolset is wrapped in a PreparedToolset that tags it
    # `unless_native`; the leaf underneath carries the id.
    assert isinstance(toolset, PreparedToolset)
    leaf = toolset.wrapped
    assert isinstance(leaf, FunctionToolset)
    assert leaf.id == 'search'


def _noop_greet(name: str) -> str:
    return f'Hello, {name}!'  # pragma: no cover


def _noop_greet_with_context(_ctx: RunContext, name: str) -> str:
    return f'Hello, {name}!'  # pragma: no cover


def test_capability_combines_toolsets_and_tools_together():
    """`Capability[object](toolsets=..., tools=...)` mirrors `Agent` by combining both."""
    toolset = FunctionToolset()
    cap = Capability[object](toolsets=[toolset], tools=[_noop_greet])

    from pydantic_ai.toolsets import CombinedToolset

    combined = cast(CombinedToolset, cap.get_toolset())
    function_toolset, provided_toolset = combined.toolsets
    assert isinstance(function_toolset, FunctionToolset)
    assert function_toolset.tools.keys() == {'_noop_greet'}
    assert provided_toolset is toolset


def test_capability_tool_plain_combines_with_toolsets():
    """`Capability.tool_plain()` registers a function toolset alongside provided toolsets."""
    toolset = FunctionToolset()
    cap = Capability[object](toolsets=[toolset])
    cap.tool_plain(_noop_greet)

    from pydantic_ai.toolsets import CombinedToolset

    combined = cast(CombinedToolset, cap.get_toolset())
    function_toolset, provided_toolset = combined.toolsets
    assert isinstance(function_toolset, FunctionToolset)
    assert function_toolset.tools.keys() == {'_noop_greet'}
    assert provided_toolset is toolset


def test_capability_tool_combines_with_toolsets():
    """`Capability.tool()` registers a function toolset alongside provided toolsets."""
    toolset = FunctionToolset()
    cap = Capability[object](toolsets=[toolset])
    cap.tool(_noop_greet_with_context)

    from pydantic_ai.toolsets import CombinedToolset

    combined = cast(CombinedToolset, cap.get_toolset())
    function_toolset, provided_toolset = combined.toolsets
    assert isinstance(function_toolset, FunctionToolset)
    assert function_toolset.tools.keys() == {'_noop_greet_with_context'}
    assert provided_toolset is toolset


def test_capability_opts_out_of_spec_serialization():
    """`Capability` holds non-serializable state (function tools, instructions, callable
    descriptions), so it opts out of spec construction like the other non-serializable
    capabilities, and passing it as a custom capability type fails loudly."""
    from pydantic_ai.agent.spec import get_capability_registry

    assert Capability.get_serialization_name() is None
    with pytest.raises(ValueError, match='Capability has opted out of serialization'):
        get_capability_registry(custom_types=[Capability])


async def test_toolset_capability_in_agent():
    """A Toolset capability's tools are available to the agent."""
    ts = FunctionToolset()

    @ts.tool_plain
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f'Hello, {name}!'

    agent = Agent(TestModel(), capabilities=[Toolset(toolset=ts)])
    result = await agent.run('Greet Alice')

    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


async def test_capability_function_tools_shortcuts_in_agent():
    """A Capability can register function tools directly or with decorators."""

    def greet(name: str) -> str:
        """Greet someone by name."""
        return f'Hello, {name}!'

    cap = Capability[int](tools=[greet])

    @cap.tool_plain(name='wave')
    def wave(name: str) -> str:
        """Wave to someone by name."""
        return f'Waving to {name}!'

    @cap.tool
    def add_deps(ctx: RunContext[int], value: int) -> int:
        """Add the run dependency to a value."""
        return ctx.deps + value

    agent = Agent(TestModel(call_tools=['greet', 'wave', 'add_deps']), capabilities=[cap], deps_type=int)
    result = await agent.run('Use the capability tools', deps=10)

    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert [part.tool_name for part in tool_returns] == ['greet', 'wave', 'add_deps']


async def test_capability_instructions_decorator_without_parenthesis():
    """A Capability can register instructions with a bare decorator."""
    captured_messages: list[ModelMessage] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    cap = Capability[object]()

    @cap.instructions
    def instructions() -> str:
        return 'Use the capability runbook.'

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])
    result = await agent.run('Help me')

    assert result.output == 'done'
    assert [msg.instructions for msg in captured_messages if isinstance(msg, ModelRequest)] == snapshot(
        ['Use the capability runbook.']
    )


async def test_capability_instructions_decorator_with_parenthesis():
    """A Capability can register instructions with a called decorator."""
    captured_messages: list[ModelMessage] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    cap = Capability[object]()

    @cap.instructions()
    def instructions_2() -> str:
        return 'Use the capability runbook.'

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])
    result = await agent.run('Help me')

    assert result.output == 'done'
    assert [msg.instructions for msg in captured_messages if isinstance(msg, ModelRequest)] == snapshot(
        ['Use the capability runbook.']
    )


async def test_capability_instructions_decorator_combines_with_constructor_instructions():
    """Constructor instructions and decorator instructions are combined."""
    captured_messages: list[ModelMessage] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    cap = Capability[int](instructions='Use the capability runbook.')

    @cap.instructions
    def add_deps(ctx: RunContext[int]) -> str:
        return f'The current account id is {ctx.deps}.'

    agent = Agent(FunctionModel(model_fn), capabilities=[cap], deps_type=int)
    result = await agent.run('Help me', deps=123)

    assert result.output == 'done'
    assert [msg.instructions for msg in captured_messages if isinstance(msg, ModelRequest)] == snapshot(
        ['Use the capability runbook.\n\nThe current account id is 123.']
    )


async def test_deferred_capability_instructions_decorator_resolves_on_load() -> None:
    """A deferred capability returns decorator-registered instructions when loaded."""
    cap = Capability[int](
        id='account',
        description='Account-specific guidance.',
        defer_loading=True,
    )

    @cap.instructions
    def account_instructions(ctx: RunContext[int]) -> str:
        return f'Use account id {ctx.deps}.'

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        already_loaded = any(
            isinstance(part, LoadCapabilityReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        if not already_loaded:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'account'},
                        tool_call_id='load-account',
                    )
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap], deps_type=int)
    result = await agent.run('Help me', deps=123)

    assert result.output == 'done'
    [load_return] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert load_return.instructions == 'Use account id 123.'
    first_request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))
    assert first_request.instructions == snapshot(
        """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- account: Account-specific guidance.\
"""
    )


async def test_deferred_capability_partitions_native_tools() -> None:
    """Deferred native tools are kept out of the baseline request until loaded."""
    native_cap = NativeTool(
        tool=WebSearchTool(),
        id='web-search',
        defer_loading=True,
    )

    [native_tool_func] = CombinedCapability([native_cap]).get_native_tools()
    assert callable(native_tool_func)
    native_tool_ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        capabilities={'web-search': native_cap},
    )
    assert native_tool_func(native_tool_ctx) is None
    native_tool_ctx.loaded_capability_ids.add('web-search')
    assert native_tool_func(native_tool_ctx) == WebSearchTool()

    @dataclass
    class CallableNativeToolCap(AbstractCapability):
        id: str | None = 'callable-web-search'
        defer_loading: bool = True

        def get_native_tools(self) -> list[Callable[[RunContext], WebSearchTool]]:
            return [lambda ctx: WebSearchTool()]

    callable_native_cap = CallableNativeToolCap()
    [callable_native_tool_func] = CombinedCapability([callable_native_cap]).get_native_tools()
    assert callable(callable_native_tool_func)
    assert callable_native_tool_func(native_tool_ctx) is None
    native_tool_ctx.loaded_capability_ids.add('callable-web-search')
    assert callable_native_tool_func(native_tool_ctx) == WebSearchTool()

    seen_web_search_tools: list[list[WebSearchTool]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_web_search_tools.append(
            [tool for tool in info.model_request_parameters.native_tools if isinstance(tool, WebSearchTool)]
        )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[native_cap])
    await agent.run('before load')
    await agent.run(
        'after load',
        message_history=[
            ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'web-search'}, tool_call_id='load-web')]),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='load-web')]),
        ],
    )

    assert seen_web_search_tools == snapshot([[], [WebSearchTool()]])


async def test_load_capability_tool_name_conflict_raises() -> None:
    """The framework loader must not be shadowed by a user tool with the same name."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def load_capability() -> str:
        return 'user-defined loader'  # pragma: no cover

    hidden = Capability[object](
        id='hidden',
        description='Hidden instructions.',
        instructions='Hidden instructions.',
        defer_loading=True,
    )
    agent = Agent(TestModel(), toolsets=[toolset], capabilities=[hidden])

    with pytest.raises(UserError) as exc_info:
        await agent.run('hi')

    assert str(exc_info.value) == snapshot(
        "Tool name 'load_capability' is reserved for deferred capability loading. Rename your tool to avoid conflicts."
    )


def test_duplicate_capability_ids_raise() -> None:
    """Capability ids are used as a run registry, so duplicates must fail loudly — at construction."""
    with pytest.raises(UserError) as exc_info:
        Agent(
            TestModel(),
            capabilities=[
                Capability[object](id='dup', description='First capability.', instructions='First.'),
                Capability[object](id='dup', description='Second capability.', instructions='Second.'),
            ],
        )

    assert str(exc_info.value) == snapshot(
        "Capability id 'dup' is used by multiple capabilities. Capability ids must be unique within a run."
    )


def test_deferred_capability_without_id_raises_at_construction() -> None:
    """A statically-provided deferred capability without an `id` fails fast at construction."""
    with pytest.raises(UserError, match='stable explicit `id` values'):
        Agent(TestModel(), capabilities=[Capability[object](description='No id.', defer_loading=True)])


async def test_partial_load_capability_history_does_not_mark_loaded() -> None:
    """A partial/stale `load_capability` call in history must not load a capability on replay."""
    captured_messages: list[ModelMessage] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            Capability[object](
                id='reports',
                description='Report tools.',
                instructions='Report instructions.',
                defer_loading=True,
            )
        ],
    )

    result = await agent.run(
        'hi',
        message_history=[
            ModelResponse(parts=[LoadCapabilityCallPart(args='{"id":', tool_call_id='partial-load')]),
            ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='partial-load')]),
        ],
    )

    assert result.output == 'done'
    # `output == 'done'` alone would pass even if the stale partial load had wrongly marked
    # `reports` loaded, so assert the gating directly. The catalog lists `reports` whether or
    # not it is loaded (kept stable for prompt caching), so the real discriminator is the
    # capability's loaded-only instructions: they must be absent because it never loaded.
    final_instructions = next(
        msg.instructions for msg in reversed(captured_messages) if isinstance(msg, ModelRequest) and msg.instructions
    )
    assert 'Report instructions.' not in final_instructions
    assert 'reports: Report tools.' in final_instructions


async def test_load_capability_invalid_dict_args_recovers_via_retry() -> None:
    """Schema-violating dict args from the model must produce a retry, not crash the run.

    Providers like Anthropic (non-streaming) and Google deliver tool args as parsed
    dicts. A dict that doesn't match `LoadCapabilityArgs` fails the typed-subclass
    validation when the response is narrowed — promotion must be best-effort (leave
    the part plain) so the args validator at execution time can send the model a
    retry as designed. Reproduces a live crash with `claude-haiku-4-5` coerced into
    sending `{"name": ...}` instead of `{"id": ...}`.
    """
    calls = 0

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'name': 'refunds'})])
        if calls == 2:
            return ModelResponse(parts=[ToolCallPart(tool_name='load_capability', args={'id': 'refunds'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[
            Capability[object](
                id='refunds',
                description='Refund tools.',
                instructions='Refund instructions.',
                defer_loading=True,
            )
        ],
    )

    result = await agent.run('hi')
    assert result.output == 'done'

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hi', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='load_capability',
                        args={'name': 'refunds'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content=[
                            {'type': 'missing', 'loc': ('id',), 'msg': 'Field required', 'input': {'name': 'refunds'}}
                        ],
                        tool_name='load_capability',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=81, output_tokens=10),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    LoadCapabilityReturnPart(
                        content={'instructions': 'Refund instructions.'},
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=86, output_tokens=11),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@pytest.mark.parametrize(
    'args,expected_id',
    [
        pytest.param(None, None, id='partial-stream-no-args'),
        pytest.param({'id': 'refunds'}, 'refunds', id='validated-dict'),
        pytest.param('{"id": "billing"}', 'billing', id='complete-json-string'),
        pytest.param('{"id":', None, id='partial-stream-json'),
        pytest.param('[1, 2, 3]', None, id='non-dict-json'),
    ],
)
def test_load_capability_call_part_typed_args(args: Any, expected_id: str | None) -> None:
    """`typed_args` handles valid, partial, and invalid payloads."""
    part = LoadCapabilityCallPart(tool_call_id='c', args=args)
    assert part.capability_id == expected_id
    if expected_id is None:
        assert part.typed_args is None
    else:
        assert part.typed_args == {'id': expected_id}


def test_load_capability_return_part_accessors() -> None:
    """`instructions` reads the optional return payload field."""
    with_instructions = LoadCapabilityReturnPart(
        tool_call_id='c',
        content={'instructions': 'Use refunds carefully.'},
    )
    assert with_instructions.instructions == 'Use refunds carefully.'

    without_instructions = LoadCapabilityReturnPart(
        tool_call_id='c',
        content={},
    )
    assert without_instructions.instructions is None


def test_load_capability_narrow_type_promotes_and_is_idempotent() -> None:
    """Capability-load narrowing is idempotent."""
    base_call = ToolCallPart(
        tool_name='load_capability',
        tool_call_id='c',
        args={'id': 'refunds'},
        tool_kind='capability-load',
    )
    promoted_call = ToolCallPart.narrow_type(base_call)
    assert isinstance(promoted_call, LoadCapabilityCallPart)
    assert ToolCallPart.narrow_type(promoted_call) is promoted_call

    base_return = ToolReturnPart(
        tool_name='load_capability',
        tool_call_id='c',
        content={},
        tool_kind='capability-load',
    )
    promoted_return = ToolReturnPart.narrow_type(base_return)
    assert isinstance(promoted_return, LoadCapabilityReturnPart)
    assert ToolReturnPart.narrow_type(promoted_return) is promoted_return


def test_load_capability_parts_round_trip_through_message_history() -> None:
    """`capability-load` parts survive history (de)serialization as typed subclasses, and a
    user tool named `load_capability` without `tool_kind` is left as a plain `ToolCallPart`."""
    from pydantic_ai.messages import ModelRequest, ModelResponse

    raw: list[dict[str, Any]] = [
        {
            'kind': 'response',
            'parts': [
                {
                    'part_kind': 'tool-call',
                    'tool_name': 'load_capability',
                    'tool_kind': 'capability-load',
                    'args': {'id': 'refunds'},
                    'tool_call_id': 'c1',
                },
                # User tool colliding on the name but without `tool_kind`: must stay base.
                {
                    'part_kind': 'tool-call',
                    'tool_name': 'load_capability',
                    'args': {'foo': 'bar'},
                    'tool_call_id': 'c2',
                },
            ],
        },
        {
            'kind': 'request',
            'parts': [
                {
                    'part_kind': 'tool-return',
                    'tool_name': 'load_capability',
                    'tool_kind': 'capability-load',
                    'content': {'instructions': 'Confirm the order id.'},
                    'tool_call_id': 'c1',
                },
            ],
        },
    ]
    response, request = ModelMessagesTypeAdapter.validate_python(raw)
    assert isinstance(response, ModelResponse)
    assert isinstance(response.parts[0], LoadCapabilityCallPart)
    assert response.parts[0].capability_id == 'refunds'
    # Collision on `tool_name='load_capability'` without `tool_kind` stays a base part.
    assert type(response.parts[1]) is ToolCallPart
    assert response.parts[1].args == {'foo': 'bar'}
    assert isinstance(request, ModelRequest)
    assert isinstance(request.parts[0], LoadCapabilityReturnPart)
    assert request.parts[0].instructions == 'Confirm the order id.'

    # Full JSON dump -> load round-trip preserves the typed subclasses.
    rebuilt = ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json([response, request]))
    assert isinstance(rebuilt[0].parts[0], LoadCapabilityCallPart)
    assert isinstance(rebuilt[1].parts[0], LoadCapabilityReturnPart)



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


# --- for_run tests ---


def _build_run_context(deps: Any = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), run_step=0)


def test_resolve_capability_id_scans_run_context_capabilities() -> None:
    @dataclass
    class SimpleCap(AbstractCapability):
        pass

    target = SimpleCap()
    other = SimpleCap()
    ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        capabilities={'other': other, 'target': target},
    )

    assert resolve_capability_id(ctx, target) == 'target'


async def test_capability_for_run_default_returns_self():
    """Default for_run returns self."""

    @dataclass
    class SimpleCap(AbstractCapability):
        pass

    cap = SimpleCap()
    ctx = _build_run_context()
    assert await cap.for_run(ctx) is cap


async def test_run_context_available_tool_names_empty_before_tool_manager_is_ready() -> None:
    """Early capability hooks can ask for available tool names before the tool manager is populated."""
    seen_available_tool_names: list[set[str]] = []
    seen_tools: list[dict[str, ToolDefinition]] = []

    @dataclass
    class AvailableToolsCap(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            seen_available_tool_names.append(ctx.available_tool_names)
            seen_tools.append(ctx.tools)

    agent = Agent(TestModel(), capabilities=[AvailableToolsCap()])
    await agent.run('hello')

    assert seen_available_tool_names == [set()]
    # The `tools` empty-guard mirrors `available_tool_names`: no tool manager yet → empty dict.
    assert seen_tools == [{}]


def test_run_context_available_tool_names_includes_discovered_before_tool_manager() -> None:
    ctx = _build_run_context()
    ctx.discovered_tool_names = {'discovered_tool'}

    assert ctx.tools == {}
    assert ctx.available_tool_names == {'discovered_tool'}
    assert ctx.is_tool_available('discovered_tool')
    assert not ctx.is_tool_available('unknown_tool')


def test_run_context_is_tool_available_falls_back_while_tools_unresolved() -> None:
    """Mid-`get_tools` the manager exists but its tool set is `None`; the name form must take
    the same history fallback as `available_tool_names` instead of reporting `False`."""
    ctx = _build_run_context()
    ctx.tool_manager = ToolManager(FunctionToolset())
    ctx.discovered_tool_names = {'discovered_tool'}

    assert ctx.tool_manager.tools is None
    assert ctx.available_tool_names == {'discovered_tool'}
    assert ctx.is_tool_available('discovered_tool')
    assert not ctx.is_tool_available('unknown_tool')


async def test_run_context_available_tool_names_unions_discovered_current_tools() -> None:
    """Available tool names are always-visible current tools plus revealed corpus tools."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def always_tool() -> str:  # pragma: no cover
        return 'always'

    @toolset.tool_plain(defer_loading=True)
    def discovered_tool() -> str:  # pragma: no cover
        return 'discovered'

    @toolset.tool_plain(defer_loading=True)
    def pending_tool() -> str:  # pragma: no cover
        return 'pending'

    @toolset.tool_plain(defer_loading=True)
    def loaded_capability_tool() -> str:  # pragma: no cover
        return 'loaded'

    ctx = _build_run_context()
    ctx.capabilities = {
        'loaded_capability': Capability(id='loaded_capability', defer_loading=True),
    }
    ctx.discovered_tool_names = {'discovered_tool', 'removed_tool'}
    ctx.loaded_capability_ids = {'loaded_capability'}
    tools = await toolset.get_tools(ctx)
    tools['discovered_tool'] = replace(
        tools['discovered_tool'],
        tool_def=replace(tools['discovered_tool'].tool_def, with_native=ToolSearchTool.kind),
    )
    tools['pending_tool'] = replace(
        tools['pending_tool'],
        tool_def=replace(tools['pending_tool'].tool_def, with_native=ToolSearchTool.kind, defer_loading=True),
    )
    tools['loaded_capability_tool'] = replace(
        tools['loaded_capability_tool'],
        tool_def=replace(
            tools['loaded_capability_tool'].tool_def,
            with_native=ToolSearchTool.kind,
            capability_id='loaded_capability',
        ),
    )
    tool_manager = ToolManager(toolset=toolset, ctx=ctx, tools=tools)
    ctx.tool_manager = tool_manager

    assert ctx.available_tool_names == {'always_tool', 'discovered_tool'}


async def test_run_context_is_tool_available() -> None:
    """Exercise the predicate directly across every reveal path and both argument forms.

    Covers always-visible, history-revealed, still-hidden, and unknown-name
    outcomes for both the `str` and `ToolDefinition` forms; the end-to-end fold and stale-resume
    scenarios are covered by the integration tests below.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def plain_tool() -> str:  # pragma: no cover
        return 'plain'

    @toolset.tool_plain(defer_loading=True)
    def discovered_tool() -> str:  # pragma: no cover
        return 'discovered'

    @toolset.tool_plain(defer_loading=True)
    def pending_tool() -> str:  # pragma: no cover
        return 'pending'

    @toolset.tool_plain(defer_loading=True)
    def loaded_tool() -> str:  # pragma: no cover
        return 'loaded'

    @toolset.tool_plain(defer_loading=True)
    def unloaded_tool() -> str:  # pragma: no cover
        return 'unloaded'

    ctx = _build_run_context()
    ctx.capabilities = {
        'loaded': Capability(id='loaded', defer_loading=True),
        'unloaded': Capability(id='unloaded', defer_loading=True),
    }
    ctx.loaded_capability_ids = {'loaded'}
    ctx.discovered_tool_names = {'discovered_tool', 'loaded_tool'}
    tools = await toolset.get_tools(ctx)
    for name in ('discovered_tool', 'pending_tool', 'loaded_tool', 'unloaded_tool'):
        tools[name] = replace(
            tools[name],
            tool_def=replace(tools[name].tool_def, with_native=ToolSearchTool.kind),
        )
    tools['loaded_tool'] = replace(
        tools['loaded_tool'],
        tool_def=replace(tools['loaded_tool'].tool_def, capability_id='loaded'),
    )
    tools['unloaded_tool'] = replace(
        tools['unloaded_tool'],
        tool_def=replace(tools['unloaded_tool'].tool_def, capability_id='unloaded'),
    )
    ctx.tool_manager = ToolManager(toolset=toolset, ctx=ctx, tools=tools)

    assert ctx.is_tool_available('plain_tool')
    assert ctx.is_tool_available(tools['plain_tool'].tool_def)
    assert ctx.is_tool_available('discovered_tool')
    assert ctx.is_tool_available(tools['loaded_tool'].tool_def)
    assert not ctx.is_tool_available('pending_tool')
    assert not ctx.is_tool_available(tools['unloaded_tool'].tool_def)
    assert not ctx.is_tool_available('unknown_tool')


def test_stale_loaded_eager_capability_is_not_revealed() -> None:
    ctx = _build_run_context()
    ctx.capabilities = {'refunds': Capability(id='refunds')}
    ctx.loaded_capability_ids = {'refunds'}
    tool_def = ToolDefinition(
        name='lookup_refund',
        description='Look up a refund.',
        parameters_json_schema={'type': 'object', 'properties': {}},
        capability_id='refunds',
    )

    assert ctx.is_tool_available(tool_def)
    assert tool_defs_from_pre_definition_load_returns(ctx, [tool_def]) == {}


async def test_is_tool_available_definition_survives_aggregator_fold() -> None:
    """A caller-held definition stays available after an aggregator removes it from resolved tools."""
    capability_tools = FunctionToolset()

    @capability_tools.tool_plain
    def lookup_refund() -> str:  # pragma: no cover
        return 'refund available'

    @dataclass
    class FoldingToolset(WrapperToolset[Any]):
        availability: list[bool] = field(default_factory=list[bool])

        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            tools = await self.wrapped.get_tools(ctx)
            available = ctx.is_tool_available(tools['lookup_refund'].tool_def)
            self.availability.append(available)
            if available:
                tools = {name: value for name, value in tools.items() if name != 'lookup_refund'}
            return tools

    folding_toolset: FoldingToolset | None = None

    @dataclass
    class FoldAvailableTools(AbstractCapability[Any]):
        def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any]:
            nonlocal folding_toolset
            folding_toolset = FoldingToolset(toolset)
            return folding_toolset

    refunds = Capability[object](
        id='refunds', description='Refund tools.', toolsets=[capability_tools], defer_loading=True
    )

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        if not any(part.tool_name == 'ping' for part in tool_returns):
            return ModelResponse(parts=[ToolCallPart(tool_name='ping', args={}, tool_call_id='ping')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds, FoldAvailableTools()])

    @agent.tool_plain
    def ping() -> str:
        return 'pong'

    result = await agent.run('Load refunds, then ping.')

    assert result.output == 'done'
    assert folding_toolset is not None
    assert folding_toolset.availability == [False, True, True]


async def test_stale_loaded_eager_capability_tool_stays_hidden() -> None:
    """Resumed loaded state does not reveal a tool owned by a capability that is now eager."""
    toolset = FunctionToolset()

    @toolset.tool_plain(defer_loading=True)
    def searchable_tool() -> str:  # pragma: no cover
        return 'found'

    capability = Capability[object](id='x', toolsets=[toolset])
    visibility: list[tuple[bool, set[str]]] = []

    @dataclass
    class CaptureVisibility(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            visibility.append((ctx.is_tool_available('searchable_tool'), ctx.available_tool_names))
            return request_context

    revealed_names: list[set[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        revealed_names.append(info.model_request_parameters.revealed_tool_names)
        return make_text_response('done')

    history = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'x'}, tool_call_id='load-x')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='load-x')]),
    ]
    agent = Agent(FunctionModel(model_fn), capabilities=[capability, CaptureVisibility()])
    await agent.run('Resume.', message_history=history)
    discovered_history = [
        *history,
        ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['searchable']}, tool_call_id='search-searchable')]),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'searchable_tool'}]},
                    tool_call_id='search-searchable',
                )
            ]
        ),
    ]
    await agent.run('Resume after discovery.', message_history=discovered_history)

    [(is_available, available_names), (is_discovered, discovered_names)] = visibility
    assert not is_available
    assert 'searchable_tool' not in available_names
    assert is_discovered
    assert 'searchable_tool' in discovered_names
    assert revealed_names == [set(), {'searchable_tool'}]


_DEFERRED_HOOK_NAMES = {
    'prepare_output_tools',
    'wrap_run_event_stream',
    'on_model_request_error',
    'on_tool_validate_error',
    'on_tool_execute_error',
    'before_output_validate',
    'after_output_validate',
    'wrap_output_validate',
    'on_output_validate_error',
    'on_output_process_error',
    'handle_deferred_tool_calls',
}


@dataclass
class _FailIfDispatchedDeferredCap(AbstractCapability):
    id: str | None = 'deferred'
    defer_loading: bool = True

    def __getattribute__(self, name: str) -> Any:
        if name in _DEFERRED_HOOK_NAMES:  # pragma: no cover
            raise AssertionError(f'unloaded capability hook should be skipped: {name}')
        return super().__getattribute__(name)


@dataclass
class _NoopCap(AbstractCapability):
    pass


def _output_context() -> OutputContext:
    return OutputContext(mode='text', output_type=str, object_def=None, has_function=False)


async def _empty_event_stream() -> AsyncIterator[AgentStreamEvent]:
    if False:  # pragma: no cover
        yield cast(AgentStreamEvent, None)


async def _validate_output(output: str | dict[str, Any]) -> Any:
    return output


async def test_combined_capability_skips_unloaded_deferred_forward_hooks() -> None:
    """Forward-order hook dispatch skips unloaded deferred capabilities."""
    combined = CombinedCapability([_FailIfDispatchedDeferredCap(), _NoopCap()])
    ctx = _build_run_context()
    output_context = _output_context()
    tool_def = ToolDefinition(name='tool')

    assert await combined.prepare_output_tools(ctx, [tool_def]) == [tool_def]
    assert await combined.before_output_validate(ctx, output_context=output_context, output='raw') == 'raw'
    assert (
        await combined.handle_deferred_tool_calls(
            ctx, requests=DeferredToolRequests(calls=[ToolCallPart('tool', {}, tool_call_id='deferred-call')])
        )
        is None
    )


async def test_combined_capability_skips_unloaded_deferred_reverse_hooks() -> None:
    """Reverse-order hook dispatch skips unloaded deferred capabilities."""
    combined = CombinedCapability([_NoopCap(), _FailIfDispatchedDeferredCap()])
    ctx = _build_run_context()
    output_context = _output_context()
    tool_def = ToolDefinition(name='tool')
    call = ToolCallPart('tool', {}, tool_call_id='tool-call')
    request_context = ModelRequestContext(
        model=TestModel(),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )

    assert [event async for event in combined.wrap_run_event_stream(ctx, stream=_empty_event_stream())] == []
    assert await combined.after_output_validate(ctx, output_context=output_context, output='parsed') == 'parsed'
    assert (
        await combined.wrap_output_validate(ctx, output_context=output_context, output='raw', handler=_validate_output)
        == 'raw'
    )

    with pytest.raises(RuntimeError, match='model'):
        await combined.on_model_request_error(ctx, request_context=request_context, error=RuntimeError('model'))
    with pytest.raises(ModelRetry, match='tool validate'):
        await combined.on_tool_validate_error(
            ctx, call=call, tool_def=tool_def, args={}, error=ModelRetry('tool validate')
        )
    with pytest.raises(RuntimeError, match='tool execute'):
        await combined.on_tool_execute_error(
            ctx, call=call, tool_def=tool_def, args={}, error=RuntimeError('tool execute')
        )
    with pytest.raises(ModelRetry, match='output validate'):
        await combined.on_output_validate_error(
            ctx, output_context=output_context, output='raw', error=ModelRetry('output validate')
        )
    with pytest.raises(RuntimeError, match='output process'):
        await combined.on_output_process_error(
            ctx, output_context=output_context, output='parsed', error=RuntimeError('output process')
        )


async def test_combined_capability_for_run_propagates():
    """CombinedCapability propagates for_run to children."""

    @dataclass
    class SimpleCap(AbstractCapability):
        label: str = ''

    cap1 = SimpleCap(label='a')
    cap2 = SimpleCap(label='b')
    combined = CombinedCapability([cap1, cap2])
    ctx = _build_run_context()

    # No child changes → returns self
    result = await combined.for_run(ctx)
    assert result is combined


async def test_combined_capability_for_run_returns_new_when_child_changes():
    """CombinedCapability returns new instance when a child's for_run returns different."""

    @dataclass
    class PerRunCap(AbstractCapability):
        run_id: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return PerRunCap(run_id=self.run_id + 1)

    @dataclass
    class StaticCap(AbstractCapability):
        pass

    static_cap = StaticCap()
    per_run_cap = PerRunCap()
    combined = CombinedCapability([static_cap, per_run_cap])
    ctx = _build_run_context()

    result = await combined.for_run(ctx)
    assert result is not combined
    assert isinstance(result, CombinedCapability)
    assert result.capabilities[0] is static_cap  # unchanged
    new_per_run = result.capabilities[1]
    assert isinstance(new_per_run, PerRunCap)
    assert new_per_run.run_id == 1


async def test_combined_capability_for_run_cancels_siblings_on_failure():
    """When one child's for_run fails, siblings are cancelled instead of leaking as orphan tasks."""
    sibling_completed = False

    @dataclass
    class FailingCap(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            raise RuntimeError('boom')

    @dataclass
    class SlowCap(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            nonlocal sibling_completed
            await anyio.sleep(0.1)
            sibling_completed = True  # pragma: no cover
            return self  # pragma: no cover

    combined = CombinedCapability([FailingCap(), SlowCap()])
    ctx = _build_run_context()

    with pytest.raises(RuntimeError, match='boom'):
        await combined.for_run(ctx)

    await anyio.sleep(0.2)
    assert sibling_completed is False


def test_apply_single_capability():
    """AbstractCapability.apply() visits just the capability itself."""

    @dataclass
    class MyCap(AbstractCapability):
        pass

    cap = MyCap()
    visited: list[AbstractCapability] = []
    cap.apply(visited.append)
    assert visited == [cap]


def test_apply_combined_capability():
    """CombinedCapability.apply() recursively visits all leaf capabilities."""

    @dataclass
    class CapA(AbstractCapability):
        pass

    @dataclass
    class CapB(AbstractCapability):
        pass

    cap_a = CapA()
    cap_b = CapB()
    combined = CombinedCapability([cap_a, cap_b])

    visited: list[AbstractCapability] = []
    combined.apply(visited.append)
    assert visited == [cap_a, cap_b]


def test_apply_nested_combined_capability():
    """CombinedCapability.apply() flattens nested CombinedCapabilities."""

    @dataclass
    class CapA(AbstractCapability):
        pass

    @dataclass
    class CapB(AbstractCapability):
        pass

    @dataclass
    class CapC(AbstractCapability):
        pass

    cap_a = CapA()
    cap_b = CapB()
    cap_c = CapC()
    inner = CombinedCapability([cap_a, cap_b])
    outer = CombinedCapability([inner, cap_c])

    visited: list[AbstractCapability] = []
    outer.apply(visited.append)
    assert visited == [cap_a, cap_b, cap_c]


def test_apply_wrapper_capability():
    """WrapperCapability.apply() visits the wrapper registered for the wrapped capability."""
    inner = Thinking()
    wrapper = WrapperCapability(wrapped=inner)

    visited: list[AbstractCapability] = []
    wrapper.apply(visited.append)
    assert visited == [wrapper]


def test_apply_wrapper_over_combined_capability():
    """WrapperCapability.apply() also visits children when the wrapped capability is a container."""

    @dataclass
    class CapA(AbstractCapability):
        pass

    @dataclass
    class CapB(AbstractCapability):
        pass

    cap_a = CapA()
    cap_b = CapB()
    wrapper = WrapperCapability(wrapped=CombinedCapability([cap_a, cap_b]))

    visited: list[AbstractCapability] = []
    wrapper.apply(visited.append)
    assert visited == [wrapper, cap_a, cap_b]


async def test_wrapper_over_combined_capability_registers_child_tool_owners():
    """Child-owned toolsets still resolve capability ids when a wrapper contains a CombinedCapability."""
    toolset_a = FunctionToolset()

    @toolset_a.tool_plain
    def tool_a() -> str:
        return 'a'  # pragma: no cover

    toolset_b = FunctionToolset()

    @toolset_b.tool_plain
    def tool_b() -> str:
        return 'b'  # pragma: no cover

    wrapper = WrapperCapability(
        wrapped=CombinedCapability(
            [
                Toolset(toolset_a, id='a'),
                Toolset(toolset_b, id='b'),
            ]
        )
    )
    seen_capability_ids: list[str] = []

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for tool in info.function_tools:
            assert tool.capability_id is not None
            seen_capability_ids.append(tool.capability_id)
        return ModelResponse(parts=[TextPart(','.join(sorted(tool.name for tool in info.function_tools)))])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    result = await agent.run('list tools')

    assert result.output == 'tool_a,tool_b'
    assert sorted(seen_capability_ids) == ['a', 'b']


def test_apply_prefix_tools():
    """PrefixTools.apply() visits the wrapper registered for the wrapped capability."""
    thinking = Thinking()
    prefixed = PrefixTools(wrapped=thinking, prefix='ns')

    visited: list[AbstractCapability] = []
    prefixed.apply(visited.append)
    assert visited == [prefixed]


def test_apply_finds_capability_by_type():
    """Realistic usage: use apply() to check if a specific capability type is present."""
    thinking = Thinking()
    web_search = WebSearch(local='duckduckgo')
    combined = CombinedCapability([thinking, web_search])

    visited: list[AbstractCapability] = []
    combined.apply(visited.append)

    assert any(isinstance(c, Thinking) for c in visited)
    assert any(isinstance(c, WebSearch) for c in visited)
    assert not any(isinstance(c, WebFetch) for c in visited)


def test_apply_finds_wrapped_capability_by_type():
    """apply() registers wrappers themselves because wrapper behavior affects the loaded capability."""
    thinking = Thinking()
    prefixed = PrefixTools(wrapped=thinking, prefix='ns')
    combined = CombinedCapability([prefixed, WebSearch(local='duckduckgo')])

    visited: list[AbstractCapability] = []
    combined.apply(visited.append)

    assert not any(isinstance(c, Thinking) for c in visited)
    assert any(isinstance(c, WebSearch) for c in visited)
    assert any(isinstance(c, PrefixTools) for c in visited)


def test_apply_empty_combined():
    """CombinedCapability with no children visits nothing."""
    combined = CombinedCapability([])
    visited: list[AbstractCapability] = []
    combined.apply(visited.append)
    assert visited == []


async def test_for_run_with_different_toolset():
    """When for_run returns a capability with a different get_toolset(), the per-run toolset is used."""
    toolset_a = FunctionToolset(id='a')

    @toolset_a.tool_plain
    def tool_a() -> str:
        return 'a'  # pragma: no cover

    toolset_b = FunctionToolset(id='b')

    @toolset_b.tool_plain
    def tool_b() -> str:
        return 'b'  # pragma: no cover

    @dataclass
    class SwitchingCap(AbstractCapability):
        use_b: bool = False

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return SwitchingCap(use_b=True)

        def get_toolset(self) -> AbstractToolset:
            return toolset_b if self.use_b else toolset_a

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Check which tools are available
        tool_names = [t.name for t in info.function_tools]
        return ModelResponse(parts=[TextPart(f'tools: {",".join(sorted(tool_names))}')])

    agent = Agent(FunctionModel(respond), capabilities=[SwitchingCap()])

    # At run time, for_run switches to toolset_b
    result = await agent.run('Hello')
    assert 'tool_b' in result.output


async def test_for_run_with_different_instructions():
    """When for_run returns a capability with different get_instructions(), per-run instructions are used."""

    @dataclass
    class DynamicInstructionsCap(AbstractCapability):
        run_instructions: str = 'init-time'

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return DynamicInstructionsCap(run_instructions='per-run')

        def get_instructions(self) -> str:
            return self.run_instructions

    captured_messages: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[DynamicInstructionsCap()])
    await agent.run('Hello')

    # The per-run instructions should appear in the request's instructions field
    instructions_found = [
        msg.instructions for msg in captured_messages if isinstance(msg, ModelRequest) and msg.instructions
    ]
    assert any('per-run' in i for i in instructions_found), (
        f'Expected per-run instructions in messages, got: {captured_messages}'
    )


async def test_for_run_receives_populated_run_context():
    """`for_run` hooks receive a `RunContext` with run_id, conversation_id, and resolved metadata."""

    captured: dict[str, Any] = {}

    class CapturingCap(AbstractCapability):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            captured['run_id'] = ctx.run_id
            captured['conversation_id'] = ctx.conversation_id
            captured['metadata'] = ctx.metadata
            captured['instrumentation_version'] = ctx.instrumentation_version
            return self

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    def metadata_factory(ctx: RunContext) -> dict[str, Any]:
        # Factory should be able to read run_id/conversation_id from the early ctx.
        return {'run_id_seen': ctx.run_id, 'conversation_id_seen': ctx.conversation_id}

    agent = Agent(FunctionModel(respond), capabilities=[CapturingCap()])

    await agent.run('Hello', conversation_id='conv-123', metadata=metadata_factory)

    assert captured['run_id'] is not None
    assert captured['conversation_id'] == 'conv-123'
    assert captured['metadata'] == {'run_id_seen': captured['run_id'], 'conversation_id_seen': 'conv-123'}
    assert captured['instrumentation_version'] is not None


async def test_concurrent_runs_capability_isolation():
    """Multiple concurrent runs don't share state on stateful capabilities."""

    @dataclass
    class CountingCap(AbstractCapability):
        request_count: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return CountingCap()

        async def before_model_request(
            self,
            ctx: RunContext,
            request_context: ModelRequestContext,
        ) -> ModelRequestContext:
            self.request_count += 1
            assert self.request_count == 1, f'Expected 1, got {self.request_count} — state leaked between runs!'
            return request_context

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Done')])

    agent = Agent(FunctionModel(respond), capabilities=[CountingCap()])

    # Run two concurrent runs — each should get its own CountingCap with count=0
    results = await asyncio.gather(agent.run('A'), agent.run('B'))
    assert results[0].output == 'Done'
    assert results[1].output == 'Done'


@pytest.mark.parametrize(
    'forced_choice',
    [
        pytest.param('required', id='required'),
        pytest.param(['get_weather'], id='list'),
    ],
)
async def test_capability_can_inject_forcing_tool_choice_per_step(forced_choice: Any):
    """A capability returning a callable from get_model_settings() may inject `tool_choice='required'`
    or `list[str]` per step without tripping the agent.run baseline validator.

    Forces the tool on step 1, then steps aside so the agent can produce a final response.
    """

    class ForceFirstStep(AbstractCapability):
        def get_model_settings(self) -> Any:
            def settings(ctx: RunContext) -> _ModelSettings:
                tool_called = any(
                    isinstance(part, ToolReturnPart) and part.tool_name == 'get_weather'
                    for message in ctx.messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                )
                if tool_called:
                    return _ModelSettings()
                return _ModelSettings(tool_choice=forced_choice)

            return settings

    seen_tool_choices: list[Any] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_choices.append((info.model_settings or {}).get('tool_choice'))
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='sunny')])
        return ModelResponse(parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'})])

    agent = Agent(FunctionModel(respond), capabilities=[ForceFirstStep()])

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'Weather in {city}: sunny'

    result = await agent.run('Weather in Paris?')

    assert result.output == 'sunny'
    assert seen_tool_choices == [forced_choice, None]



# --- NativeOrLocalTool tests ---


class TestWebSearchCapability:
    def test_websearch_default_no_local(self):
        """WebSearch() defaults to builtin-only — no local fallback unless explicitly requested."""
        cap = WebSearch()
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], WebSearchTool)

        # No local fallback by default in v2
        assert cap.get_toolset() is None

    def test_websearch_default_with_nonsupporting_model_raises(self, allow_model_requests: None):
        """WebSearch() with a model that doesn't support builtin → UserError (no auto-fallback)."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
        agent = Agent(model, capabilities=[WebSearch()])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('search')

    def test_websearch_local_string_strategy(self, allow_model_requests: None):
        """WebSearch(local='duckduckgo') with non-supporting model → DuckDuckGo fallback used."""
        from unittest.mock import patch

        pytest.importorskip('duckduckgo_search', reason='duckduckgo extra not installed')
        from pydantic_ai.common_tools.duckduckgo import DDGS

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(parts=[TextPart(content=f'Tool result: {part.content}')])
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(tool_name=info.function_tools[0].name, args='{"query": "test"}', tool_call_id='c1')
                    ]
                )
            return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover

        model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(model, capabilities=[WebSearch(local='duckduckgo')])
        # `ddgs` calls Bing/DuckDuckGo via the Rust `primp` HTTP client, so VCR can't intercept it.
        # Mock the result at the library boundary to keep the test hermetic.
        fake_results = [{'title': 'Example', 'href': 'https://example.com', 'body': 'Example body'}]
        with patch.object(DDGS, 'text', return_value=fake_results):
            result = agent.run_sync('search for something')
        assert 'Tool result' in result.output

    def test_websearch_unknown_strategy_raises(self):
        """WebSearch(local='unknown_name') → UserError."""
        with pytest.raises(UserError, match='not a known strategy'):
            WebSearch(local='not_a_real_strategy')  # type: ignore[arg-type]

    def test_websearch_local_false_with_nonsupporting_model(self, allow_model_requests: None):
        """WebSearch(local=False) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
        agent = Agent(model, capabilities=[WebSearch(local=False)])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('search')

    def test_websearch_native_false_without_local_raises(self):
        """WebSearch(native=False) without an explicit local → UserError at construction."""
        with pytest.raises(UserError, match='requires an explicit local tool'):
            WebSearch(native=False)

    def test_websearch_native_false_with_local_string(self):
        """WebSearch(native=False, local='duckduckgo') → only local, no native registered."""
        cap = WebSearch(native=False, local='duckduckgo')
        assert cap.get_native_tools() == []
        toolset = cap.get_toolset()
        # Plain toolset (no PreparedToolset wrapping since native is disabled)
        assert toolset is not None

    def test_websearch_requires_native_with_constraints(self, allow_model_requests: None):
        """WebSearch(allowed_domains=...) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
        agent = Agent(model, capabilities=[WebSearch(allowed_domains=['example.com'], local='duckduckgo')])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('search')

    def test_websearch_both_false_raises(self):
        """WebSearch(native=False, local=False) → UserError at construction."""
        with pytest.raises(UserError, match='both `native` and `local` cannot be False'):
            WebSearch(native=False, local=False)

    def test_websearch_native_false_with_constraints_raises(self):
        """WebSearch(native=False, local='duckduckgo', allowed_domains=...) → UserError at construction."""
        with pytest.raises(UserError, match='constraint fields require the native tool'):
            WebSearch(native=False, local='duckduckgo', allowed_domains=['example.com'])

    def test_websearch_local_callable(self):
        """WebSearch(local=some_function) → bare callable wrapped in Tool."""
        from pydantic_ai.tools import Tool

        def my_search(query: str) -> str:
            return f'results for {query}'  # pragma: no cover

        cap = WebSearch(local=my_search)
        assert isinstance(cap.local, Tool)


class TestXSearchCapability:
    def test_xsearch_default(self):
        """XSearch() with defaults → native XSearchTool, no local."""
        cap = XSearch()
        assert cap.get_native_tools() == snapshot([XSearchTool()])
        assert cap.fallback_model is None
        assert cap.get_toolset() is None

    def test_xsearch_with_fallback_model(self):
        """XSearch(fallback_model=...) → native XSearchTool, local subagent fallback."""
        cap = XSearch(fallback_model='xai:grok-4-1-fast-non-reasoning')
        assert cap.get_native_tools() == snapshot([XSearchTool()])
        assert cap.get_toolset() is not None

    def test_xsearch_with_all_constraints(self):
        """XSearch with all constraint fields → XSearchTool configured."""
        cap = XSearch(
            allowed_x_handles=['handle1'],
            from_date=datetime(2024, 1, 1),
            to_date=datetime(2024, 12, 31),
            enable_image_understanding=True,
            enable_video_understanding=True,
            include_output=True,
        )
        assert cap.get_native_tools() == snapshot(
            [
                XSearchTool(
                    allowed_x_handles=['handle1'],
                    from_date=datetime(2024, 1, 1),
                    to_date=datetime(2024, 12, 31),
                    enable_image_understanding=True,
                    enable_video_understanding=True,
                    include_output=True,
                )
            ]
        )

    def test_xsearch_requires_native_with_handles(self):
        """XSearch with handle constraints requires builtin."""
        assert XSearch(allowed_x_handles=['h']).get_native_tools() == snapshot([XSearchTool(allowed_x_handles=['h'])])
        assert XSearch(excluded_x_handles=['h']).get_native_tools() == snapshot([XSearchTool(excluded_x_handles=['h'])])

    def test_xsearch_native_false_local_false_raises(self):
        """XSearch(native=False, local=False) → UserError."""
        with pytest.raises(UserError, match='both `native` and `local` cannot be False'):
            XSearch(native=False, local=False)

    def test_xsearch_native_false_with_constraints_raises(self):
        """XSearch(native=False, allowed_x_handles=...) without fallback_model → UserError."""
        with pytest.raises(UserError, match='constraint fields require the native tool'):
            XSearch(native=False, allowed_x_handles=['handle1'])

    def test_xsearch_resolved_native_merges_overrides(self):
        """Capability-level kwargs override fields on a passed native instance."""
        base = XSearchTool(allowed_x_handles=['a'], enable_image_understanding=True)
        cap = XSearch(native=base, from_date=datetime(2024, 1, 1), enable_image_understanding=False)
        resolved = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]
        assert resolved == snapshot(
            XSearchTool(
                allowed_x_handles=['a'],
                from_date=datetime(2024, 1, 1),
                enable_image_understanding=False,
            )
        )

    def test_xsearch_fallback_model_and_local_conflict(self):
        """XSearch(fallback_model=..., local=func) raises UserError."""

        def my_search(query: str) -> str:
            return 'result'  # pragma: no cover

        with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
            XSearch(fallback_model='xai:grok-4-1-fast-non-reasoning', local=my_search)

    def test_xsearch_fallback_model_with_local_false(self):
        """XSearch(fallback_model=..., local=False) raises UserError."""
        with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
            XSearch(fallback_model='xai:grok-4-1-fast-non-reasoning', local=False)

    def test_xsearch_callable_native_with_fallback(self):
        """Callable native with fallback_model still creates a local fallback tool."""
        from pydantic_ai.tools import Tool

        cap = XSearch(
            native=lambda ctx: XSearchTool(enable_image_understanding=True),
            fallback_model='xai:grok-4-1-fast-non-reasoning',
        )
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    async def test_xsearch_callable_fallback_model(self, allow_model_requests: None):
        """XSearch with callable fallback_model resolves the model per-run."""

        def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='summary of recent tweets')])

        inner_model = FunctionModel(
            inner_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool}))
        )

        async def model_factory(ctx: RunContext) -> FunctionModel:
            return inner_model

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "latest news"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[XSearch(fallback_model=model_factory)])
        result = await agent.run('What is happening on X?')
        assert result.output == 'done'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='What is happening on X?', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='x_search',
                            args='{"query": "latest news"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=55, output_tokens=6),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='x_search',
                            content='summary of recent tweets',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='done')],
                    usage=RequestUsage(input_tokens=59, output_tokens=7),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_xsearch_sync_callable_fallback_model(self, allow_model_requests: None):
        """XSearch with sync callable fallback_model resolves the model per-run."""

        def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='summary')])

        inner_model = FunctionModel(
            inner_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool}))
        )

        def model_factory(ctx: RunContext) -> FunctionModel:
            return inner_model

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "news"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[XSearch(fallback_model=model_factory)])
        result = await agent.run('search X')
        assert result.output == 'done'
        tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
        assert len(tool_returns) == 1
        assert tool_returns[0].content == 'summary'

    async def test_xsearch_subagent_error_becomes_model_retry(self, allow_model_requests: None):
        """UnexpectedModelBehavior from the subagent becomes a retry prompt to the outer model."""

        # Inner model returns an empty response → triggers UnexpectedModelBehavior in the subagent.
        def empty_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[])

        inner_model = FunctionModel(
            empty_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool}))
        )

        call_count = 0

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "test"}')])
            return ModelResponse(parts=[TextPart(content='gave up')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[XSearch(fallback_model=inner_model)])
        result = await agent.run('search X')
        assert result.output == 'gave up'
        retry_parts = list(iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart))
        assert len(retry_parts) == 1
        assert retry_parts[0].tool_name == 'x_search'

    def test_x_search_tool_unknown_kwarg_raises(self):
        """`x_search_tool(unknown=...)` raises TypeError naming the offending kwarg."""
        from pydantic_ai.common_tools.x_search import x_search_tool

        with pytest.raises(TypeError, match=r"unexpected keyword argument '?bogus'?"):
            x_search_tool('xai:grok-4-1-fast-non-reasoning', native_tool=XSearchTool(), bogus=1)  # type: ignore[call-arg]

    def test_x_search_tool_missing_native_tool_raises(self):
        """`x_search_tool()` without `native_tool=` raises TypeError."""
        from pydantic_ai.common_tools.x_search import x_search_tool

        with pytest.raises(TypeError, match=r"missing 1 required positional argument: 'native_tool'"):
            x_search_tool('xai:grok-4-1-fast-non-reasoning')  # type: ignore[call-arg]

    def test_xsearch_subagent_tool_unknown_attr_raises(self):
        """Unknown attribute access on `XSearchSubagentTool` raises AttributeError as usual."""
        from pydantic_ai.common_tools.x_search import XSearchSubagentTool

        subagent = XSearchSubagentTool(model='xai:grok-4-1-fast-non-reasoning', native_tool=XSearchTool())
        with pytest.raises(AttributeError, match='no_such_field'):
            subagent.no_such_field  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]


class TestWebFetchCapability:
    def test_webfetch_default_no_local(self):
        """WebFetch() defaults to builtin-only — no local fallback unless explicitly requested."""
        cap = WebFetch()
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], WebFetchTool)
        # No local fallback by default in v2
        assert cap.local is None
        assert cap.get_toolset() is None

    def test_webfetch_default_with_nonsupporting_model_raises(self, allow_model_requests: None):
        """WebFetch() with a model that doesn't support builtin → UserError (no auto-fallback)."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
        agent = Agent(model, capabilities=[WebFetch()])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('fetch')

    def test_webfetch_local_true_fallback(self, allow_model_requests: None):
        """WebFetch(local=True) with non-supporting model → markdownify fallback used."""
        from unittest.mock import AsyncMock, patch

        import httpx

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(parts=[TextPart(content=f'Tool result: {part.content}')])
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args='{"url": "https://example.com"}',
                            tool_call_id='c1',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover

        mock_response = httpx.Response(
            200,
            text='<html><head><title>Test</title></head><body><p>Hello</p></body></html>',
            headers={'content-type': 'text/html'},
            request=httpx.Request('GET', 'https://example.com'),
        )

        model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(model, capabilities=[WebFetch(local=True)])
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            result = agent.run_sync('fetch something')
        tool_calls = list(iter_message_parts(result.all_messages(), ModelResponse, ToolCallPart))
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == 'web_fetch'

    def test_webfetch_unknown_strategy_raises(self):
        """WebFetch(local='unknown_name') → UserError."""
        with pytest.raises(UserError, match='not a known strategy'):
            WebFetch(local='not_a_real_strategy')  # type: ignore[arg-type]

    def test_webfetch_local_false_with_nonsupporting_model(self, allow_model_requests: None):
        """WebFetch(local=False) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
        agent = Agent(model, capabilities=[WebFetch(local=False)])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('fetch')

    def test_webfetch_native_false_without_local_raises(self):
        """WebFetch(native=False) without explicit local → UserError at construction."""
        with pytest.raises(UserError, match='requires an explicit local tool'):
            WebFetch(native=False)

    def test_webfetch_native_false_with_local_string(self):
        """WebFetch(native=False, local=True) → only local, no native registered."""
        cap = WebFetch(native=False, local=True)
        assert cap.get_native_tools() == []
        toolset = cap.get_toolset()
        assert toolset is not None

    def test_webfetch_max_uses_requires_native(self, allow_model_requests: None):
        """WebFetch(max_uses=...) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_native_tools=frozenset()))  # pyright: ignore[reportArgumentType]
        agent = Agent(model, capabilities=[WebFetch(max_uses=5, local=True)])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('fetch')

    def test_webfetch_domains_forwarded_to_local(self, allow_model_requests: None):
        """WebFetch(allowed_domains=..., local=True) with non-supporting model → falls back to local with domain filtering."""
        from unittest.mock import AsyncMock, patch

        import httpx

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(parts=[TextPart(content=f'Tool result: {part.content}')])
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args='{"url": "https://example.com"}',
                            tool_call_id='c1',
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover

        mock_response = httpx.Response(
            200,
            text='<html><body><p>Hello</p></body></html>',
            headers={'content-type': 'text/html'},
            request=httpx.Request('GET', 'https://example.com'),
        )

        model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(model, capabilities=[WebFetch(allowed_domains=['example.com'], local=True)])
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            result = agent.run_sync('fetch example.com')
        tool_calls = list(iter_message_parts(result.all_messages(), ModelResponse, ToolCallPart))
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == 'web_fetch'

    def test_webfetch_both_false_raises(self):
        """WebFetch(native=False, local=False) → UserError at construction."""
        with pytest.raises(UserError, match='both `native` and `local` cannot be False'):
            WebFetch(native=False, local=False)

    def test_webfetch_native_false_with_max_uses_raises(self):
        """WebFetch(native=False, local=True, max_uses=...) → UserError at construction."""
        with pytest.raises(UserError, match='constraint fields require the native tool'):
            WebFetch(native=False, local=True, max_uses=5)

    def test_webfetch_local_callable(self):
        """WebFetch(local=some_function) → bare callable wrapped in Tool."""
        from pydantic_ai.tools import Tool

        def my_fetch(url: str) -> str:
            return f'fetched {url}'  # pragma: no cover

        cap = WebFetch(local=my_fetch)
        assert isinstance(cap.local, Tool)


class TestImageGenerationCapability:
    def test_image_gen_init_params_match_builtin_tool(self):
        """ImageGeneration.__init__ accepts all ImageGenerationTool configurable fields."""
        import dataclasses
        import inspect

        # partial_images is excluded — not useful for subagent fallback (no streaming).
        # optional is excluded — applies to wire-side dropping, not local-fallback config.
        builtin_fields = {
            f.name
            for f in dataclasses.fields(ImageGenerationTool)
            if f.name not in ('kind', 'optional', 'partial_images')
        }
        builtin_fields.remove('model')
        builtin_fields.add('image_model')
        # Subtract framework-inherited kw-only params from `AbstractCapability`
        # (forwarded so `dataclasses.replace` round-trips through the custom `__init__`).
        init_params = set(inspect.signature(ImageGeneration.__init__).parameters.keys()) - {
            'self',
            'native',
            'local',
            'fallback_model',
            'id',
            'defer_loading',
            'description',
        }
        assert init_params == builtin_fields

    def test_image_generation_default(self):
        """ImageGeneration() provides only builtin, no local fallback."""
        cap = ImageGeneration()
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], ImageGenerationTool)
        # No default local
        assert cap.local is None
        assert cap.get_toolset() is None

    def test_image_generation_with_custom_local(self):
        """ImageGeneration(local=custom) → provides custom local fallback."""
        from pydantic_ai.tools import Tool

        def my_gen(prompt: str) -> str:
            return 'image_url'  # pragma: no cover

        cap = ImageGeneration(local=my_gen)
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_with_fallback_model(self):
        """ImageGeneration(fallback_model=...) creates a local fallback tool."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(fallback_model='openai-responses:gpt-5.4')
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], ImageGenerationTool)

    def test_image_generation_forwards_config_to_builtin(self):
        """ImageGeneration config fields are forwarded to the ImageGenerationTool builtin."""
        cap = ImageGeneration(
            action='generate',
            background='opaque',
            input_fidelity='high',
            moderation='low',
            image_model='gpt-image-2',
            output_compression=80,
            output_format='jpeg',
            quality='high',
            size='1024x1024',
            aspect_ratio='16:9',
        )
        builtins = cap.get_native_tools()
        assert len(builtins) == 1
        tool = builtins[0]
        assert isinstance(tool, ImageGenerationTool)
        assert tool.action == 'generate'
        assert tool.background == 'opaque'
        assert tool.input_fidelity == 'high'
        assert tool.moderation == 'low'
        assert tool.model == 'gpt-image-2'
        assert tool.output_compression == 80
        assert tool.output_format == 'jpeg'
        assert tool.quality == 'high'
        assert tool.size == '1024x1024'
        assert tool.aspect_ratio == '16:9'

    def test_image_generation_fallback_merges_custom_native_with_overrides(self):
        """Custom native tool settings are merged with capability-level overrides for the fallback."""
        from pydantic_ai.tools import Tool

        custom_native = ImageGenerationTool(quality='high', size='1024x1024')
        cap = ImageGeneration(
            native=custom_native,
            fallback_model='openai-responses:gpt-5.4',
            output_format='jpeg',  # capability-level override
        )
        # The local fallback should exist and contain the merged config
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_callable_native_with_fallback(self):
        """When native is a callable, the fallback local tool still gets created."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(
            native=lambda ctx: ImageGenerationTool(quality='high'),
            fallback_model='openai-responses:gpt-5.4',
        )
        # Callable native can't be resolved at init time, but local fallback is still created
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_fallback_model_and_local_conflict(self):
        """ImageGeneration(fallback_model=..., local=func) raises UserError."""

        def my_gen(prompt: str) -> str:
            return 'image_url'  # pragma: no cover

        with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
            ImageGeneration(fallback_model='openai-responses:gpt-5.4', local=my_gen)

    def test_image_generation_fallback_model_with_local_false(self):
        """ImageGeneration(fallback_model=..., local=False) raises UserError."""
        with pytest.raises(UserError, match='cannot specify both `fallback_model` and `local`'):
            ImageGeneration(fallback_model='openai-responses:gpt-5.4', local=False)

    async def test_image_generation_callable_fallback_model(self, allow_model_requests: None):
        """ImageGeneration with async callable fallback_model resolves the model per-run."""

        image_data = b'\x89PNG\r\n\x1a\n'  # minimal PNG header

        def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=image_data, media_type='image/png'))])

        inner_model = FunctionModel(inner_model_fn, profile=ModelProfile(supports_image_output=True))

        async def model_factory(ctx: RunContext) -> FunctionModel:
            return inner_model

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)])
        result = await agent.run('Generate a test image')
        assert result.output == 'done'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Generate a test image', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "test"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=5),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='generate_image',
                            content=BinaryImage(data=b'\x89PNG\r\n\x1a\n', media_type='image/png'),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='done')],
                    usage=RequestUsage(input_tokens=54, output_tokens=6),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_image_generation_callable_returns_image_only_model(self, allow_model_requests: None):
        """Callable fallback_model returning an image-only model name is caught at call time."""

        def model_factory(ctx: RunContext) -> str:
            return 'openai-responses:gpt-image-1'

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)])
        with pytest.raises(UserError, match="'gpt-image-1' is a dedicated image generation model"):
            await agent.run('Generate a test image')

    async def test_image_generation_subagent_error_becomes_model_retry(self, allow_model_requests: None):
        """UnexpectedModelBehavior from subagent becomes a retry prompt to the outer model."""

        # FunctionModel that returns text but no image — triggers UnexpectedModelBehavior
        def no_image_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='No image generated.')])

        inner_model = FunctionModel(no_image_model_fn, profile=ModelProfile(supports_image_output=True))

        call_count = 0

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])
            return ModelResponse(parts=[TextPart(content='gave up')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)])
        result = await agent.run('Generate a test image')
        assert result.output == 'gave up'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Generate a test image', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "test"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=5),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Exceeded maximum output retries (1)',
                            tool_name='generate_image',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='gave up')],
                    usage=RequestUsage(input_tokens=66, output_tokens=7),
                    model_name='function:outer_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.parametrize(
        'provider, model_name, suggestion',
        [
            ('openai-responses', 'gpt-image-2', 'openai-responses:gpt-5.5'),
            ('openai-responses', 'gpt-image-1.5', 'openai-responses:gpt-5.5'),
            ('openai-responses', 'gpt-image-1', 'openai-responses:gpt-5.4'),
            ('openai-responses', 'gpt-image-1-mini', 'openai-responses:gpt-5.4'),
            ('google', 'imagen-3.0-generate-002', 'google:gemini-3-pro-image'),
            ('google', 'imagen-3.0-fast-generate-001', 'google:gemini-3-pro-image'),
        ],
    )
    def test_image_generation_rejects_image_only_model(self, provider: str, model_name: str, suggestion: str):
        """Using a dedicated image model raises a clear error with a conversational alternative."""
        with pytest.raises(
            UserError,
            match=re.escape(
                f'{model_name!r} is a dedicated image generation model that cannot be used as '
                f'`fallback_model` directly. Use a conversational model with image generation '
                f'support instead, e.g. {suggestion!r}.'
            ),
        ):
            ImageGeneration(fallback_model=f'{provider}:{model_name}')

    @pytest.mark.vcr()
    async def test_image_generation_local_fallback(self, allow_model_requests: None, openai_api_key: str):
        """ImageGeneration(fallback_model=...) with non-supporting outer model uses subagent fallback."""
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # If we see a tool return, the image was generated — return final text
            if any(
                isinstance(part, ToolReturnPart)
                for msg in messages
                if isinstance(msg, ModelRequest)
                for part in msg.parts
            ):
                return ModelResponse(parts=[TextPart(content='Here is the generated image.')])

            # First call: invoke the generate_image tool
            assert info.function_tools, 'Expected generate_image tool to be available'
            tool = info.function_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}')])

        inner_model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key=openai_api_key))
        outer_model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(
            outer_model,
            capabilities=[
                ImageGeneration(fallback_model=inner_model),
            ],
        )
        result = await agent.run('Generate an image of a cute baby sea otter')
        assert result.output == 'Here is the generated image.'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Generate an image of a cute baby sea otter', timestamp=IsDatetime())
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "A cute baby sea otter"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=59, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='generate_image',
                            content=IsInstance(BinaryImage),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Here is the generated image.')],
                    usage=RequestUsage(input_tokens=59, output_tokens=15),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.vcr()
    async def test_image_generation_local_fallback_google(self, allow_model_requests: None, gemini_api_key: str):
        """ImageGeneration fallback with Google image model."""
        pytest.importorskip('google.genai', reason='google extra not installed')
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='Here is the generated image.')])
            assert info.function_tools, 'Expected generate_image tool to be available'
            tool = info.function_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}')])

        inner_model = GoogleModel('gemini-3-pro-image', provider=GoogleProvider(api_key=gemini_api_key))
        outer_model = FunctionModel(model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)])
        result = await agent.run('Generate an image of a cute baby sea otter')
        assert result.output == 'Here is the generated image.'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(content='Generate an image of a cute baby sea otter', timestamp=IsDatetime())
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='generate_image',
                            args='{"prompt": "A cute baby sea otter"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=59, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='generate_image',
                            content=IsInstance(BinaryImage),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Here is the generated image.')],
                    usage=RequestUsage(input_tokens=59, output_tokens=15),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


has_mcp = find_spec('mcp') is not None


@pytest.mark.skipif(not has_mcp, reason='mcp is not installed')
class TestMCPCapability:
    def test_mcp_default_local_only(self):
        """MCP(url=...) defaults to local-only via the MCP SDK — no native advertised."""
        cap = MCP(url='https://mcp.example.com/api')
        assert cap.get_native_tools() == []
        assert cap.get_toolset() is not None

    def test_mcp_native_true_advertises_both(self):
        """MCP(url=..., native=True) advertises native + keeps local as fallback."""
        cap = MCP(url='https://mcp.example.com/api', native=True)
        native_tools = cap.get_native_tools()
        assert len(native_tools) == 1
        assert isinstance(native_tools[0], MCPServerTool)
        assert native_tools[0].url == 'https://mcp.example.com/api'
        assert cap.get_toolset() is not None

    def test_mcp_native_only(self):
        """MCP(url=..., native=True, local=False) advertises only the native tool."""
        cap = MCP(url='https://mcp.example.com/api', native=True, local=False)
        native_tools = cap.get_native_tools()
        assert len(native_tools) == 1
        assert isinstance(native_tools[0], MCPServerTool)
        assert cap.get_toolset() is None

    def test_mcp_id_from_url(self):
        """MCP auto-derives id from URL including hostname to avoid collisions."""
        cap = MCP(url='https://mcp.example.com/api', native=True)
        native = cap.get_native_tools()[0]
        assert isinstance(native, MCPServerTool)
        assert native.id == 'mcp.example.com-api'

        # SSE URLs include hostname to avoid collisions between different servers
        cap_sse = MCP(url='https://server1.example.com/sse', native=True)
        native_sse = cap_sse.get_native_tools()[0]
        assert isinstance(native_sse, MCPServerTool)
        assert native_sse.id == 'server1.example.com-sse'

    def test_mcp_local_toolset_id_derived(self):
        """MCP stamps a derived id on the local `MCPToolset` so it can be used with durable
        execution. Precedence: explicit `id` → native `MCPServerTool` id → host+slug from the URL,
        else `None` when there's nothing to derive from."""
        # `FastMCP` needs server deps; the `mcp` extra only pulls `fastmcp-slim[client]`.
        pytest.importorskip('fastmcp.server')
        from fastmcp import FastMCP

        from pydantic_ai.mcp import MCPToolset

        # (capability, expected local toolset id)
        cases: list[tuple[MCP[object], str | None]] = [
            # id derived from the URL (host + path slug)
            (MCP[object](url='https://mcp.example.com/api'), 'mcp.example.com-api'),
            # explicit id wins
            (MCP[object](url='https://mcp.example.com/api', id='docs'), 'docs'),
            # native MCPServerTool id is reused for the local fallback
            (
                MCP[object](
                    url='https://mcp.example.com/api',
                    native=MCPServerTool(id='custom-mcp', url='https://mcp.example.com/api'),
                    local=True,
                ),
                'custom-mcp',
            ),
            # `local='https://…'` override with no `url=`: id derived from the override URL,
            # exercising `_derive_id` deriving from the override URL even when `self.url` is `None`
            (MCP[object](local='https://other.example.com/sse'), 'other.example.com-sse'),
            # non-URL local input (in-process `FastMCP` server) wrapped into an `MCPToolset`,
            # inheriting the explicit id
            (MCP[object](id='local-mcp', local=FastMCP('test-server')), 'local-mcp'),
            # nothing to derive from — no id, no native tool, no URL → stays None
            (MCP[object](local=FastMCP('test-server')), None),
        ]
        for cap, expected_id in cases:
            local = cap.local
            assert isinstance(local, MCPToolset)
            assert local.id == expected_id

    def test_mcp_callable_native_without_url_or_id_errors(self):
        """A `native=<callable>` factory paired with a local fallback has nothing to derive the
        `unless_native` marker from (no `url=`, no `id=`, non-`MCPServerTool` native), so
        `get_toolset()` raises an actionable `UserError` rather than a bare `AssertionError`."""

        async def native_factory(ctx: RunContext[object]) -> MCPServerTool:
            return MCPServerTool(id='x', url='https://mcp.example.com/api')  # pragma: no cover

        def local_tool() -> str:
            return 'local'  # pragma: no cover

        cap = MCP[object](native=native_factory, local=local_tool)
        with pytest.raises(UserError, match='needs a stable `id` to tie the two together'):
            cap.get_toolset()

    async def test_mcp_explicit_native_id_marks_local_fallback(self):
        """An explicit native MCP tool keeps the local fallback tied to that server id."""

        def local_tool() -> str:
            return 'local result'  # pragma: no cover

        cap = MCP(
            url='https://mcp.example.com/api',
            native=MCPServerTool(id='custom-mcp', url='https://mcp.example.com/api'),
            local=local_tool,
        )
        toolset = cap.get_toolset()
        assert toolset is not None
        tools = await toolset.get_tools(_build_run_context())
        assert tools['local_tool'].tool_def.unless_native == 'mcp_server:custom-mcp'

    async def test_mcp_dynamic_native_id_marks_local_fallback(self):
        """A dynamic native MCP tool still marks the local fallback with the stable capability id."""

        def local_tool() -> str:
            return 'local result'  # pragma: no cover

        async def native_tool(ctx: RunContext) -> MCPServerTool:
            return MCPServerTool(id='dynamic-mcp', url='https://mcp.example.com/api')

        cap = MCP(url='https://mcp.example.com/api', id='dynamic-mcp', native=native_tool, local=local_tool)
        toolset = cap.get_toolset()
        assert toolset is not None
        tools = await toolset.get_tools(_build_run_context())
        assert tools['local_tool'].tool_def.unless_native == 'mcp_server:dynamic-mcp'

    def test_mcp_sse_transport(self):
        """MCP with /sse URL routes to an MCPToolset using FastMCP's SSE transport."""
        from fastmcp.client.transports import SSETransport

        from pydantic_ai.mcp import MCPToolset

        cap = MCP(url='https://mcp.example.com/sse', native=True)
        assert isinstance(cap.local, MCPToolset)
        assert isinstance(cap.local.client.transport, SSETransport)  # pyright: ignore[reportUnknownMemberType]

    def test_mcp_streamable_transport(self):
        """MCP with non-/sse URL routes to an MCPToolset using FastMCP's Streamable HTTP transport."""
        from fastmcp.client.transports import StreamableHttpTransport

        from pydantic_ai.mcp import MCPToolset

        cap = MCP(url='https://mcp.example.com/api', native=True)
        assert isinstance(cap.local, MCPToolset)
        assert isinstance(cap.local.client.transport, StreamableHttpTransport)  # pyright: ignore[reportUnknownMemberType]

    def test_mcp_authorization_token_in_local_headers(self):
        """MCP passes authorization_token as Authorization header through to the transport."""
        from fastmcp.client.transports import StreamableHttpTransport

        from pydantic_ai.mcp import MCPToolset

        cap = MCP(url='https://mcp.example.com/api', authorization_token='Bearer xyz', native=True)
        assert isinstance(cap.local, MCPToolset)
        transport = cap.local.client.transport  # pyright: ignore[reportUnknownMemberType]
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.headers == {'Authorization': 'Bearer xyz'}

    def test_mcp_allowed_tools_filters_local(self):
        """MCP(allowed_tools=...) applies FilteredToolset to the local toolset."""
        from pydantic_ai.toolsets.filtered import FilteredToolset

        cap = MCP(url='https://mcp.example.com/api', allowed_tools=['tool1'], native=True)
        toolset = cap.get_toolset()
        assert toolset is not None
        # The outer toolset should be a FilteredToolset wrapping the prepared toolset
        assert isinstance(toolset, FilteredToolset)

    def test_mcp_no_url_no_local_raises(self):
        """MCP() with neither `url=` nor `local=` raises — no way to construct a usable capability."""
        with pytest.raises(UserError, match='requires an explicit local tool'):
            MCP()

    def test_mcp_wraps_non_toolset_local_into_mcptoolset(self):
        """A bare `fastmcp.FastMCP` server passed as `local=` is wrapped in `MCPToolset` automatically."""
        # `FastMCP` needs server deps; the `mcp` extra only pulls `fastmcp-slim[client]`.
        pytest.importorskip('fastmcp.server')
        from fastmcp import FastMCP

        from pydantic_ai.mcp import MCPToolset

        cap = MCP(url='https://mcp.example.com/api', native=True, local=FastMCP(name='in_process'))
        assert isinstance(cap.local, MCPToolset)


class TestNamedSpecDictRoundTrip:
    """Test that NamedSpec correctly round-trips various argument forms."""

    def test_dict_positional_arg_uses_long_form(self):
        """A dict positional arg falls back to long form to avoid kwargs misinterpretation on round-trip."""
        spec = NamedSpec(name='CustomCap', arguments=({'key': 'value', 'other': 42},))
        serialized = spec.model_dump(context={'use_short_form': True})
        # Dict with string keys would be ambiguous in short form, so long form is used
        assert serialized['name'] == 'CustomCap'
        assert len(serialized['arguments']) == 1
        assert serialized['arguments'][0] == {'key': 'value', 'other': 42}
        # Round-trip preserves the dict as a positional arg
        deserialized = NamedSpec.model_validate(serialized)
        assert deserialized.args == ({'key': 'value', 'other': 42},)
        assert deserialized.kwargs == {}

    def test_non_dict_positional_arg_uses_short_form(self):
        """A non-dict positional arg still uses the compact short form."""
        spec = NamedSpec(name='WebSearch', arguments=(True,))
        serialized = spec.model_dump(context={'use_short_form': True})
        assert serialized == {'WebSearch': True}

    def test_kwargs_use_short_form(self):
        """Kwargs (dict arguments) use the short form correctly."""
        spec = NamedSpec(name='WebSearch', arguments={'local': True})
        serialized = spec.model_dump(context={'use_short_form': True})
        assert serialized == {'WebSearch': {'local': True}}


class TestPrepareToolsCapability:
    async def test_prepare_tools_filters(self):
        """PrepareTools capability filters tools using the provided callable."""
        from pydantic_ai.capabilities import PrepareTools

        async def hide_secret_tools(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if td.name != 'secret_tool']

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f'tools: {sorted(tool_names)}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(hide_secret_tools)])

        @agent.tool_plain
        def secret_tool() -> str:
            return 'secret'  # pragma: no cover

        @agent.tool_plain
        def public_tool() -> str:
            return 'public'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == "tools: ['public_tool']"

    async def test_prepare_tools_rejects_none(self):
        """PrepareTools rejects `None`; return [] to disable all tools explicitly."""
        from pydantic_ai.capabilities import PrepareTools

        async def invalid(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
            return None

        agent = Agent('test', capabilities=[PrepareTools(invalid)])  # pyright: ignore[reportArgumentType]

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        with pytest.raises(UserError, match="Prepare function 'invalid' returned `None`"):
            await agent.run('hello')

    async def test_prepare_tools_modifies_definitions(self):
        """PrepareTools can modify tool definitions (e.g. set strict mode)."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.capabilities import PrepareTools

        async def set_strict(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [dc_replace(td, strict=True) for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            strictness = [t.strict for t in info.function_tools]
            return make_text_response(f'strict: {strictness}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(set_strict)])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == 'strict: [True]'

    def test_prepare_tools_not_serializable(self):
        """PrepareTools opts out of spec serialization."""
        from pydantic_ai.capabilities import PrepareTools

        assert PrepareTools.get_serialization_name() is None

    async def test_prepare_tools_rejects_added_tools(self):
        """`prepare_func` may filter or modify tools but cannot add or rename."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.capabilities import PrepareTools
        from pydantic_ai.exceptions import UserError

        async def rename(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [dc_replace(td, name='renamed') for td in tool_defs]

        agent = Agent('test', capabilities=[PrepareTools(rename)])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        with pytest.raises(UserError, match='cannot add or rename'):
            await agent.run('hello')

    async def test_prepare_tools_filtering_blocks_hallucinated_calls(self):
        """A tool filtered out by `prepare_tools` must be unreachable, even if the model
        hallucinates a call to it. Regression test: the hook must affect `ToolManager.tools`,
        not just the model's `ModelRequestParameters` — otherwise the model could (re)call
        a filtered tool and `ToolManager` would happily execute it."""
        from pydantic_ai.capabilities import PrepareTools

        executed: list[str] = []

        async def hide_secret(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if td.name != 'secret_tool']

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            # First turn: hallucinate a call to the filtered tool. Even though the model
            # doesn't see `secret_tool` in `info.function_tools`, simulate it doing so anyway
            # (this can also happen via leftover history).
            if call_count == 1:
                return ModelResponse(parts=[ToolCallPart('secret_tool', {})])
            return make_text_response('done')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(hide_secret)])

        @agent.tool_plain
        def secret_tool() -> str:
            executed.append('secret')  # pragma: no cover
            return 'secret'  # pragma: no cover

        result = await agent.run('hello')

        # `secret_tool` was never executed — the hallucinated call resolved to "unknown tool"
        # because `prepare_tools` filtering also removed it from `ToolManager.tools`.
        assert executed == []
        # Snapshot the message flow: the hallucinated call should produce a "Unknown tool"
        # retry prompt referencing only the visible tools, and the second turn should succeed.
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='secret_tool', args={}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Unknown tool name: 'secret_tool'. No tools available.",
                            tool_name='secret_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='done')],
                    usage=RequestUsage(input_tokens=65, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestPrepareOutputToolsCapability:
    async def test_filters_output_tools(self):
        """`PrepareOutputTools` capability filters output tools using a callable."""
        from pydantic_ai.capabilities import PrepareOutputTools

        class Out(BaseModel):
            value: str

        async def disable_all(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response(f'output_tools: {len(info.output_tools)}')

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, ToolOutput(Out, name='out')],
            capabilities=[PrepareOutputTools(disable_all)],
        )

        result = await agent.run('hello')
        assert result.output == 'output_tools: 0'

    async def test_prepare_output_tools_rejects_none(self):
        """PrepareOutputTools rejects `None`; return [] to disable all output tools explicitly."""
        from pydantic_ai.capabilities import PrepareOutputTools

        class Out(BaseModel):
            value: str

        async def invalid(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
            return None

        agent = Agent(
            'test',
            output_type=[str, ToolOutput(Out, name='out')],
            capabilities=[PrepareOutputTools(invalid)],  # pyright: ignore[reportArgumentType]
        )

        with pytest.raises(UserError, match="Prepare function 'invalid' returned `None`"):
            await agent.run('hello')

    async def test_only_sees_output_tools(self):
        """`PrepareOutputTools` only receives output tools — function tools route to `PrepareTools`."""
        from pydantic_ai.capabilities import PrepareOutputTools

        seen_kinds: list[str] = []

        async def capture(ctx: RunContext, tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            seen_kinds.extend(td.kind for td in tool_defs)
            return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[PrepareOutputTools(capture)])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        await agent.run('hello')
        assert seen_kinds == ['output']

    def test_not_serializable(self):
        """`PrepareOutputTools` opts out of spec serialization."""
        from pydantic_ai.capabilities import PrepareOutputTools

        assert PrepareOutputTools.get_serialization_name() is None


class TestOverrideWithSpec:
    async def test_override_with_spec_instructions_and_model(self):
        """Spec instructions and model replace the agent's when used via override."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn), instructions='original')

        with agent.override(spec={'instructions': 'from spec'}):
            result = await agent.run('hello')

        assert 'from spec' in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions='from spec',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='instructions: from spec')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_override_with_spec_explicit_param_wins(self):
        """Explicit override param beats spec value."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn), instructions='original')

        with agent.override(spec={'instructions': 'from spec'}, instructions='explicit'):
            result = await agent.run('hello')

        assert 'explicit' in result.output
        assert 'from spec' not in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions='explicit',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='instructions: explicit')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_override_with_spec_instructions(self):
        """Override with spec instructions replaces agent's existing instructions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn), instructions='agent-instructions')

        with agent.override(spec={'instructions': 'from-spec-instructions'}):
            result = await agent.run('hello')
            # Override replaces: only spec instructions, not agent's
            assert 'from-spec-instructions' in result.output
            assert 'agent-instructions' not in result.output
            assert result.all_messages() == snapshot(
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                        timestamp=IsDatetime(),
                        instructions='from-spec-instructions',
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                    ModelResponse(
                        parts=[TextPart(content='instructions: from-spec-instructions')],
                        usage=RequestUsage(input_tokens=51, output_tokens=2),
                        model_name='function:model_fn:',
                        timestamp=IsDatetime(),
                        run_id=IsStr(),
                        conversation_id=IsStr(),
                    ),
                ]
            )

    async def test_override_with_spec_capabilities(self):
        """Override with spec providing capabilities uses them for the run."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('ok')

        agent = Agent(FunctionModel(model_fn))

        with agent.override(spec={'capabilities': [{'WebSearch': {'local': False}}]}):
            result = await agent.run('hello')
            assert result.output == 'ok'


class TestRunWithSpec:
    async def test_run_with_spec_instructions_added(self):
        """Spec instructions are added additively at run time."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn), instructions='original')

        result = await agent.run('hello', spec={'instructions': 'also from spec'})
        # Both original and spec instructions should be present
        assert 'original' in result.output
        assert 'also from spec' in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="""\
original
also from spec\
""",
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
instructions: original
also from spec\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_model_as_fallback(self):
        """Spec model is used as fallback when no run-time model is provided."""
        agent = Agent(None)  # No model set

        result = await agent.run('hello', spec={'model': 'test'})
        assert result.output == 'success (no tool calls)'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='success (no tool calls)')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_model_settings_merged(self):
        """Spec model_settings are merged with run model_settings."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            max_tokens = info.model_settings.get('max_tokens') if info.model_settings else None
            temperature = info.model_settings.get('temperature') if info.model_settings else None
            return make_text_response(f'max_tokens={max_tokens} temperature={temperature}')

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run(
            'hello',
            spec={'model_settings': {'max_tokens': 100}},
            model_settings={'temperature': 0.5},
        )
        assert 'max_tokens=100' in result.output
        assert 'temperature=0.5' in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='max_tokens=100 temperature=0.5')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_partial_no_model(self):
        """Partial spec without model works if agent has a model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run('hello', spec={'instructions': 'be helpful'})
        assert 'be helpful' in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions='be helpful',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='instructions: be helpful')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_capabilities(self):
        """Run with spec capabilities merges them with agent's root capability."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn), instructions='agent-level')

        result = await agent.run(
            'hello',
            spec={'capabilities': [{'WebSearch': {'local': False}}]},
        )
        # Agent-level instructions should be present; spec capabilities are merged additively
        assert 'agent-level' in result.output

    async def test_run_with_spec_instructions(self):
        """Run with spec instructions adds to agent's instructions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (m.instructions for m in messages if isinstance(m, ModelRequest) and m.instructions), None
            )
            return make_text_response(f'instructions: {instructions}')

        agent = Agent(FunctionModel(model_fn), instructions='agent-level')

        result = await agent.run(
            'hello',
            spec={
                'instructions': 'from-spec',
            },
        )
        # Both should be present (additive)
        assert 'agent-level' in result.output
        assert 'from-spec' in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="""\
agent-level
from-spec\
""",
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="""\
instructions: agent-level
from-spec\
"""
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_metadata_merged(self):
        """Spec metadata is merged with run metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('ok')

        agent = Agent(FunctionModel(model_fn), metadata={'agent_key': 'agent_val'})

        result = await agent.run(
            'hello',
            spec={'metadata': {'spec_key': 'spec_val'}},
            metadata={'run_key': 'run_val'},
        )
        assert result.output == 'ok'
        # Run metadata should take precedence, spec metadata should be present
        assert result.metadata is not None
        assert result.metadata == snapshot({'agent_key': 'agent_val', 'spec_key': 'spec_val', 'run_key': 'run_val'})
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='ok')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_spec_unsupported_fields_warns(self):
        """Non-default unsupported fields produce warnings."""
        agent = Agent('test')

        with pytest.warns(UserWarning, match='end_strategy'):
            await agent.run('hello', spec={'end_strategy': 'exhaustive'})

    async def test_spec_tool_retry_override(self):
        """A run-time spec's tool-retry budget replaces the agent default (3 here, not the agent's 1)."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('flaky', {})])

        agent = Agent(FunctionModel(model_fn), retries={'tools': 1})

        @agent.tool_plain
        def flaky() -> str:
            nonlocal call_count
            call_count += 1
            raise ModelRetry('again')

        with pytest.raises(UnexpectedModelBehavior, match=r"Tool 'flaky' exceeded max retries count of 3"):
            await agent.run('hello', spec={'retries': {'tools': 3}})

        # initial call + 3 retries, following the spec budget (3), not the agent default (1)
        assert call_count == 4


@dataclass
class _ModelCap(AbstractCapability):
    """Test capability that supplies a model via `get_model()`."""

    model: Model | KnownModelName | str | None = None

    def get_model(self) -> Model | KnownModelName | str | None:
        return self.model


def _text_model(text: str) -> FunctionModel:
    """A `FunctionModel` whose response text identifies which model handled the request."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return make_text_response(text)

    return FunctionModel(model_fn)


class TestGetModelHook:
    """Capabilities can supply the agent's model via `get_model()`."""

    async def test_model_less_agent_uses_capability_model(self):
        """A capability can supply the model for an agent that has none (the headline case)."""
        agent = Agent(None, capabilities=[_ModelCap(model='test')])

        result = await agent.run('hello')
        assert result.output == 'success (no tool calls)'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='success (no tool calls)')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_select_model_uses_first_step_dependencies(self):
        """The convenience capability's bootstrap selector needs live deps, which a provider cassette cannot prove."""
        small = _text_model('small')
        frontier = _text_model('frontier')
        seen_steps: list[int] = []

        def select(ctx: ModelSelectionContext[bool]) -> Model:
            seen_steps.append(ctx.run_step)
            assert ctx.model is None
            assert ctx.messages == []
            return frontier if ctx.deps else small

        agent = Agent(None, deps_type=bool, capabilities=[SelectModel(select)])

        assert SelectModel.get_serialization_name() is None
        assert (await agent.run('hello', deps=False)).output == 'small'
        assert (await agent.run('hello', deps=True)).output == 'frontier'
        assert seen_steps == [1, 1]

    async def test_model_less_agent_without_capability_model_raises(self):
        """With no model anywhere (capability returns None), the usual missing-model error is raised."""
        agent = Agent(None, capabilities=[_ModelCap(model=None)])

        with pytest.raises(UserError, match='`model` must either be set on the agent or included when calling it'):
            await agent.run('hello')

    async def test_run_model_arg_beats_capability_model(self):
        """A call-site `run(model=...)` wins over a capability-supplied model."""
        agent = Agent(None, capabilities=[_ModelCap(model='test')])

        result = await agent.run('hello', model=_text_model('from-run-arg'))
        assert result.output == 'from-run-arg'

    async def test_run_spec_model_beats_capability_model(self):
        """A run-level `spec=` model wins over a capability-supplied model."""
        agent = Agent(None, capabilities=[_ModelCap(model=_text_model('from-capability'))])

        result = await agent.run('hello', spec={'model': 'test'})
        assert result.output == 'success (no tool calls)'

    async def test_capability_model_beats_agent_constructor(self):
        """A capability-supplied model wins over the agent constructor's model."""
        agent = Agent(_text_model('from-constructor'), capabilities=[_ModelCap(model=_text_model('from-capability'))])

        result = await agent.run('hello')
        assert result.output == 'from-capability'

    async def test_callable_model_instance_is_static(self):
        """A callable `Model` instance is still a model, not a selector function."""
        from unittest.mock import Mock

        class CallableModel(FunctionModel):
            __call__ = Mock(side_effect=AssertionError('model must not be called as a selector'))

        selected = CallableModel(lambda messages, info: make_text_response('selected'))
        assert (await Agent(None, capabilities=[_ModelCap(model=selected)]).run('hello')).output == 'selected'
        selected.__call__.assert_not_called()

    async def test_agent_context_with_dynamic_capability_model(self):
        """The agent context leaves dynamic capability models to the runs that select them."""
        selected_model = _text_model('from-capability')

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return lambda ctx: selected_model

        agent = Agent(_text_model('from-constructor'), deps_type=NoneType, capabilities=[AdaptiveModel()])
        async with agent:
            assert (await agent.run('hello')).output == 'from-capability'

    async def test_agent_context_uses_model_override(self):
        """The agent context enters an override model instead of a capability model."""
        agent = Agent(None, capabilities=[_ModelCap(model=_text_model('from-capability'))])

        with agent.override(model=_text_model('from-override')):
            async with agent:
                assert (await agent.run('hello')).output == 'from-override'

    async def test_override_model_beats_capability_model(self):
        """`agent.override(model=...)` wins over a capability-supplied model, per its docs."""
        agent = Agent(None, capabilities=[_ModelCap(model='test')])

        with agent.override(model=_text_model('from-override')):
            result = await agent.run('hello')
        assert result.output == 'from-override'

    async def test_last_non_none_capability_wins(self):
        """Later capability contributions override earlier ones."""
        agent = Agent(
            None,
            capabilities=[
                _ModelCap(model=None),
                _ModelCap(model=_text_model('from-second')),
                _ModelCap(model=_text_model('from-third')),
            ],
        )

        result = await agent.run('hello')
        assert result.output == 'from-third'

    async def test_callable_selects_model_per_step(self):
        first = FunctionModel(lambda messages, info: ModelResponse(parts=[ToolCallPart('advance', '{}')]))

        def finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.model_settings == {'max_tokens': 123}
            return make_text_response('done')

        second = FunctionModel(finish, settings={'max_tokens': 123})
        selected_steps: list[int] = []
        selection_history_lengths: list[int] = []

        def select(ctx: ModelSelectionContext[int]) -> Model:
            selected_steps.append(ctx.run_step)
            selection_history_lengths.append(len(ctx.messages))
            ctx.messages.clear()  # The selection context must not expose mutable graph state.
            assert ctx.deps == 42
            return first if ctx.run_step == 1 else second

        @dataclass
        class AdaptiveModel(AbstractCapability[int]):
            def get_model(self) -> Callable[[ModelSelectionContext[int]], Model]:
                return select

        agent = Agent(None, deps_type=int, capabilities=[AdaptiveModel()])

        @agent.tool_plain
        def advance() -> str:
            return 'advanced'

        result = await agent.run('hello', deps=42)
        assert result.output == 'done'
        assert selected_steps == [1, 2]
        assert selection_history_lengths == [0, 2]

    async def test_explicit_run_model_skips_selector(self):
        from unittest.mock import Mock

        select = Mock(side_effect=AssertionError('selector should not run'))

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return select

        capability = AdaptiveModel()
        assert capability.get_model() is select
        select.reset_mock()

        result = await Agent(None, deps_type=NoneType, capabilities=[capability]).run(
            'hello', model=_text_model('explicit')
        )
        assert result.output == 'explicit'
        select.assert_not_called()

    async def test_selected_model_id_is_resolved_with_deps(self):
        target = _text_model('resolved')

        def select(ctx: ModelSelectionContext[str]) -> str:
            return 'alias'

        def resolve(ctx: ModelResolutionContext[str], model_id: str) -> Model | None:
            assert ctx.deps == 'tenant'
            return target if model_id == 'alias' else None

        @dataclass
        class SelectAlias(AbstractCapability[str]):
            def get_model(self) -> Callable[[ModelSelectionContext[str]], str]:
                return select

        agent = Agent(None, deps_type=str, capabilities=[SelectAlias(), ResolveModelId(resolve)])
        result = await agent.run('hello', deps='tenant')
        assert result.output == 'resolved'

    async def test_constructor_model_id_is_resolved_with_deps(self):
        target = _text_model('resolved')

        def resolve(ctx: ModelResolutionContext[str], model_id: str) -> Model | None:
            assert ctx.deps == 'tenant'
            return target if model_id == 'alias' else None

        agent = Agent('alias', deps_type=str, capabilities=[ResolveModelId(resolve)])
        assert (await agent.run('hello', deps='tenant')).output == 'resolved'

    async def test_static_model_id_is_resolved_once_per_run(self):
        requests = 0
        resolutions = 0

        def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal requests
            requests += 1
            if requests == 1:
                return ModelResponse(parts=[ToolCallPart('advance', '{}')])
            return make_text_response('done')

        selected = FunctionModel(request)

        def resolve(ctx: ModelResolutionContext[None], model_id: str) -> Model | None:
            nonlocal resolutions
            resolutions += 1
            return selected if model_id == 'alias' else None

        agent = Agent(None, deps_type=NoneType, capabilities=[_ModelCap(model='alias'), ResolveModelId(resolve)])

        @agent.tool_plain
        def advance() -> str:
            return 'advanced'

        assert (await agent.run('hello')).output == 'done'
        assert resolutions == 1

    async def test_dynamic_model_id_is_resolved_once_per_run(self):
        requests = 0
        selections = 0
        resolutions = 0

        def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal requests
            requests += 1
            if requests == 1:
                return ModelResponse(parts=[ToolCallPart('advance', '{}')])
            return make_text_response('done')

        selected = FunctionModel(request)

        def select(ctx: ModelSelectionContext[None]) -> str:
            nonlocal selections
            selections += 1
            return 'alias'

        def resolve(ctx: ModelResolutionContext[None], model_id: str) -> Model | None:
            nonlocal resolutions
            resolutions += 1
            return selected if model_id == 'alias' else None

        agent = Agent(None, deps_type=NoneType, capabilities=[SelectModel(select), ResolveModelId(resolve)])

        @agent.tool_plain
        def advance() -> str:
            return 'advanced'

        assert (await agent.run('hello')).output == 'done'
        assert selections == 2
        assert resolutions == 1

    async def test_unchanged_for_run_selector_is_not_repeated_on_first_step(self):
        selections = 0

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                # Deliberately return a fresh closure on every configuration read.
                def select(ctx: ModelSelectionContext[None]) -> Model:
                    nonlocal selections
                    selections += 1
                    return _text_model('selected')

                return select

        agent = Agent(None, deps_type=NoneType, capabilities=[AdaptiveModel()])
        assert (await agent.run('hello')).output == 'selected'
        assert selections == 1

    async def test_replaced_for_run_selector_reselects_first_step(self):
        selections: list[str] = []

        class LifecycleModel(FunctionModel):
            entered = 0
            exited = 0

            async def __aenter__(self):
                self.entered += 1
                return self

            async def __aexit__(self, *args: Any):
                self.exited += 1

        bootstrap_model = LifecycleModel(lambda messages, info: make_text_response('bootstrap'))
        replacement_model = LifecycleModel(lambda messages, info: make_text_response('replacement'))

        def selector(name: str) -> Callable[[ModelSelectionContext[None]], Model]:
            def select(ctx: ModelSelectionContext[None]) -> Model:
                selections.append(name)
                return bootstrap_model if name == 'bootstrap' else replacement_model

            return select

        @dataclass
        class Replacement(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return selector('replacement')

        @dataclass
        class Bootstrap(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return selector('bootstrap')

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return Replacement()

        agent = Agent(None, deps_type=NoneType, capabilities=[Bootstrap()])
        assert (await agent.run('hello')).output == 'replacement'
        assert selections == ['bootstrap', 'replacement']
        assert (bootstrap_model.entered, bootstrap_model.exited) == (1, 1)
        assert (replacement_model.entered, replacement_model.exited) == (1, 1)

    async def test_replaced_for_run_static_model_is_authoritative(self):
        @dataclass
        class Replacement(AbstractCapability[None]):
            def get_model(self) -> Model:
                return _text_model('replacement')

        @dataclass
        class Bootstrap(AbstractCapability[None]):
            def get_model(self) -> Model:
                return _text_model('bootstrap')

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return Replacement()

        assert (await Agent(None, deps_type=NoneType, capabilities=[Bootstrap()]).run('hello')).output == 'replacement'

    async def test_for_run_cannot_remove_only_bootstrap_model(self):
        @dataclass
        class Bootstrap(AbstractCapability[None]):
            def get_model(self) -> Model:
                return _text_model('bootstrap')

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return AbstractCapability()

        with pytest.raises(UserError, match='removed the bootstrap model'):
            await Agent(None, deps_type=NoneType, capabilities=[Bootstrap()]).run('hello')

    async def test_for_run_can_remove_capability_model_when_constructor_model_exists(self):
        @dataclass
        class Bootstrap(AbstractCapability[None]):
            def get_model(self) -> Model:
                return _text_model('bootstrap')

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return AbstractCapability()

        agent = Agent(_text_model('constructor'), deps_type=NoneType, capabilities=[Bootstrap()])
        assert (await agent.run('hello')).output == 'constructor'

    async def test_async_selector_and_repeated_model_lifecycle(self):
        requests = 0

        class LifecycleModel(FunctionModel):
            entered = 0

            async def __aenter__(self):
                self.entered += 1
                return self

        def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal requests
            requests += 1
            if requests == 1:
                return ModelResponse(parts=[ToolCallPart('advance', '{}')])
            return make_text_response('done')

        selected = LifecycleModel(request)

        async def select(ctx: ModelSelectionContext[None]) -> Model:
            return selected

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Awaitable[Model]]:
                return select

        agent = Agent(None, deps_type=NoneType, capabilities=[AdaptiveModel()])

        @agent.tool_plain
        def advance() -> str:
            return 'advanced'

        assert (await agent.run('hello')).output == 'done'
        assert selected.entered == 1

    async def test_run_spec_capability_can_bootstrap_model_less_agent(self, monkeypatch: pytest.MonkeyPatch):
        @dataclass
        class SpecModel(AbstractCapability[None]):
            @classmethod
            def get_serialization_name(cls) -> str:
                return 'SpecModel'

            def get_model(self) -> Model:
                return _text_model('from spec capability')

        monkeypatch.setitem(CAPABILITY_TYPES, 'SpecModel', SpecModel)
        agent = Agent(None)
        assert (await agent.run('hello', spec={'capabilities': ['SpecModel']})).output == 'from spec capability'

    async def test_first_model_id_resolver_wins(self):
        first = _text_model('first')
        second = _text_model('second')
        agent = Agent(
            'alias',
            capabilities=[
                ResolveModelId(lambda ctx, model_id: first),
                ResolveModelId(lambda ctx, model_id: second),
            ],
        )
        assert (await agent.run('hello')).output == 'first'

    async def test_model_id_resolver_delegates_to_registry_backstop(self):
        calls: list[str] = []
        registered = _text_model('registered')

        def user_resolver(ctx: ModelResolutionContext[None], model_id: str) -> Model | None:
            calls.append('user')
            return None

        def registry_resolver(ctx: ModelResolutionContext[None], model_id: str) -> Model | None:
            calls.append('registry')
            return registered if model_id == 'registered-id' else None

        agent = Agent(
            'registered-id',
            deps_type=NoneType,
            capabilities=[ResolveModelId(user_resolver), ResolveModelId(registry_resolver)],
        )
        assert (await agent.run('hello')).output == 'registered'
        assert calls == ['user', 'registry']

    async def test_async_model_id_resolver_and_deferred_resolver(self):
        from unittest.mock import AsyncMock

        calls: list[str] = []
        target = _text_model('resolved')

        deferred = AsyncMock(side_effect=AssertionError('deferred model resolver must not run'))

        async def eager(ctx: ModelResolutionContext[None], model_id: str) -> Model | None:
            calls.append(model_id)
            return target

        capability = CombinedCapability(
            [ResolveModelId(deferred, defer_loading=True, id='deferred-resolver'), ResolveModelId(eager)]
        )
        agent = Agent('alias', deps_type=NoneType, capabilities=[capability])
        assert (await agent.run('hello')).output == 'resolved'
        assert calls == ['alias']
        deferred.assert_not_awaited()
        assert ResolveModelId.get_serialization_name() is None

    async def test_override_spec_model_uses_spec_model_id_resolver(self, monkeypatch: pytest.MonkeyPatch):
        target = _text_model('resolved by spec')
        bound_agents: list[AbstractAgent[None, Any]] = []

        @dataclass
        class SpecResolver(AbstractCapability[None]):
            bound: bool = False

            @classmethod
            def get_serialization_name(cls) -> str:
                return 'SpecResolver'

            def for_agent(self, agent: AbstractAgent[None, Any]) -> SpecResolver:
                bound_agents.append(agent)
                return replace(self, bound=True)

            def get_model(self) -> Model | None:
                return target if self.bound else None

            async def resolve_model_id(
                self, ctx: ModelResolutionContext[None], *, model_id: KnownModelName | str
            ) -> Model | None:
                return target if self.bound and model_id == 'custom-id' else None

        monkeypatch.setitem(CAPABILITY_TYPES, 'SpecResolver', SpecResolver)
        agent = Agent('test')

        with agent.override(spec={'capabilities': ['SpecResolver']}, model='custom-id'):
            assert (await agent.run('hello')).output == 'resolved by spec'

        with agent.override(spec={'capabilities': ['SpecResolver']}):
            with agent.override(model='custom-id'):
                assert (await agent.run('hello')).output == 'resolved by spec'

        with agent.override(spec={'capabilities': ['SpecResolver']}):
            assert (await agent.run('hello')).output == 'resolved by spec'

        assert bound_agents == [agent, agent, agent]

    async def test_wrapper_subclass_model_id_resolver_is_detected(self):
        target = _text_model('resolved by wrapper')

        @dataclass
        class ResolvingWrapper(WrapperCapability[None]):
            async def resolve_model_id(
                self, ctx: ModelResolutionContext[None], *, model_id: KnownModelName | str
            ) -> Model | None:
                return target if model_id == 'custom-id' else None

        agent = Agent('test', deps_type=NoneType, capabilities=[ResolvingWrapper(wrapped=AbstractCapability[None]())])

        with agent.override(model='custom-id'):
            assert (await agent.run('hello')).output == 'resolved by wrapper'

    async def test_dynamic_models_are_entered_once_per_run(self):
        class LifecycleModel(FunctionModel):
            entered = 0
            exited = 0

            async def __aenter__(self):
                self.entered += 1
                return self

            async def __aexit__(self, *args: Any):
                self.exited += 1

        first = LifecycleModel(lambda messages, info: ModelResponse(parts=[ToolCallPart('advance', '{}')]))
        second = LifecycleModel(lambda messages, info: make_text_response('done'))

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return lambda ctx: first if ctx.run_step == 1 else second

        agent = Agent(None, deps_type=NoneType, capabilities=[AdaptiveModel()])

        @agent.tool_plain
        def advance() -> str:
            return 'advanced'

        assert (await agent.run('hello')).output == 'done'
        assert (first.entered, first.exited) == (1, 1)
        assert (second.entered, second.exited) == (1, 1)

    async def test_selector_can_return_fallback_model(self):
        def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('primary failed')

        fallback = FallbackModel(FunctionModel(fail), _text_model('fallback'), fallback_on=RuntimeError)

        @dataclass
        class SelectFallback(AbstractCapability[None]):
            def get_model(self) -> FallbackModel:
                return fallback

        agent = Agent(None, deps_type=NoneType, capabilities=[SelectFallback()])
        assert (await agent.run('hello')).output == 'fallback'

    async def test_cross_run_suspended_resume_rejects_dynamic_model(self):
        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return lambda ctx: _text_model('selected')

        history = [ModelResponse(parts=[], state='suspended')]
        with pytest.raises(UserError, match='cannot be reconstructed unambiguously'):
            agent = Agent(None, deps_type=NoneType, capabilities=[AdaptiveModel()])
            await agent.run(message_history=history)

    async def test_cross_run_suspended_resume_rejects_for_run_dynamic_model(self):
        @dataclass
        class DynamicModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return lambda ctx: _text_model('selected')

        @dataclass
        class BootstrapModel(AbstractCapability[None]):
            def get_model(self) -> Model:
                return _text_model('bootstrap')

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return DynamicModel()

        history = [ModelResponse(parts=[], state='suspended')]
        with pytest.raises(UserError, match='cannot be reconstructed unambiguously'):
            agent = Agent(None, deps_type=NoneType, capabilities=[BootstrapModel()])
            await agent.run(message_history=history)

    async def test_system_prompt_parts_uses_selector_when_model_is_omitted(self):
        selected = _text_model('selected')

        @dataclass
        class AdaptiveModel(AbstractCapability[str]):
            def get_model(self) -> Callable[[ModelSelectionContext[str]], Model]:
                return lambda ctx: selected

        agent = Agent(None, deps_type=str, capabilities=[AdaptiveModel()])

        @agent.system_prompt
        def prompt(ctx: RunContext[str]) -> str:
            assert ctx.model is selected
            assert ctx.deps == 'tenant'
            return 'system prompt'

        assert await agent.system_prompt_parts(deps='tenant') == snapshot(
            [SystemPromptPart(content='system prompt', timestamp=IsDatetime())]
        )

    async def test_callable_model_selection_streaming(self):
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'selected'

        selected = FunctionModel(stream_function=stream)

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return lambda ctx: selected

        agent = Agent(None, deps_type=NoneType, capabilities=[AdaptiveModel()])
        async with agent.run_stream('hello') as result:
            assert await result.get_output() == 'selected'

    async def test_agent_context_does_not_evaluate_dynamic_selector(self):
        calls = 0

        def select(ctx: ModelSelectionContext[None]) -> Model:
            nonlocal calls
            calls += 1
            return _text_model('selected')

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return select

        agent = Agent(None, deps_type=NoneType, capabilities=[AdaptiveModel()])
        async with agent:
            assert calls == 0

        assert (await agent.run('hello')).output == 'selected'
        assert calls == 1

    async def test_static_capability_model_is_entered_by_agent_context(self):
        class LifecycleModel(FunctionModel):
            entered = 0
            exited = 0

            async def __aenter__(self):
                self.entered += 1
                return self

            async def __aexit__(self, *args: Any):
                self.exited += 1

        selected = LifecycleModel(lambda messages, info: make_text_response('selected'))
        agent = Agent(None, capabilities=[_ModelCap(model=selected)])
        async with agent:
            assert selected.entered == 1
            assert (await agent.run('hello')).output == 'selected'
            assert (selected.entered, selected.exited) == (1, 0)
        assert selected.exited == 1

    async def test_static_capability_model_id_reuses_agent_context_model(self, monkeypatch: pytest.MonkeyPatch):
        class LifecycleModel(FunctionModel):
            entered = 0
            exited = 0

            async def __aenter__(self):
                self.entered += 1
                return self

            async def __aexit__(self, *args: Any):
                self.exited += 1

        inferred_models: list[LifecycleModel] = []

        def infer_model(model_id: str) -> Model:
            assert model_id == 'custom-model'
            model = LifecycleModel(lambda messages, info: make_text_response('selected'))
            inferred_models.append(model)
            return model

        monkeypatch.setattr('pydantic_ai.models.infer_model', infer_model)
        agent = Agent(None, capabilities=[_ModelCap(model='custom-model')])

        async with agent:
            assert (await agent.run('hello')).output == 'selected'
            assert len(inferred_models) == 1
            assert (inferred_models[0].entered, inferred_models[0].exited) == (1, 0)
        assert inferred_models[0].exited == 1

    async def test_system_prompt_parts_resolves_static_capability_model_id(self, monkeypatch: pytest.MonkeyPatch):
        inferred_models: list[Model] = []

        def infer_model(model_id: str) -> Model:
            assert model_id == 'custom-model'
            model = _text_model('selected')
            inferred_models.append(model)
            return model

        monkeypatch.setattr('pydantic_ai.models.infer_model', infer_model)
        agent = Agent(None, capabilities=[_ModelCap(model='custom-model')])

        assert await agent.system_prompt_parts() == []
        assert len(inferred_models) == 1

        async with agent:
            assert len(inferred_models) == 2
            assert await agent.system_prompt_parts() == []
            assert len(inferred_models) == 2

    async def test_system_prompt_parts_requires_a_model(self):
        agent = Agent(None)
        with pytest.raises(UserError, match='supplied by a capability'):
            await agent.system_prompt_parts()

    def test_mcp_sampling_rejects_dynamic_capability_model(self):
        selected = _text_model('selected')
        Agent(None, capabilities=[_ModelCap(model=selected)]).set_mcp_sampling_model()

        @dataclass
        class AdaptiveModel(AbstractCapability[None]):
            def get_model(self) -> Callable[[ModelSelectionContext[None]], Model]:
                return lambda ctx: selected

        agent = Agent(_text_model('constructor'), deps_type=NoneType, capabilities=[AdaptiveModel()])
        with pytest.raises(UserError, match='requires run dependencies'):
            agent.set_mcp_sampling_model()

        resolving_agent = Agent(
            'alias', capabilities=[ResolveModelId(lambda ctx, model_id: selected if model_id == 'alias' else None)]
        )
        with pytest.raises(UserError, match='requires run dependencies'):
            resolving_agent.set_mcp_sampling_model()

    async def test_wrapper_capability_delegates(self):
        """A `WrapperCapability` surfaces its wrapped leaf's model."""
        agent = Agent(None, capabilities=[WrapperCapability(wrapped=_ModelCap(model='test'))])

        result = await agent.run('hello')
        assert result.output == 'success (no tool calls)'

    async def test_combined_capability_uses_last_non_none_model(self):
        """A `CombinedCapability` uses the last non-`None` model contribution."""
        agent = Agent(
            None,
            capabilities=[
                CombinedCapability([_ModelCap(model=_text_model('first')), _ModelCap(model=_text_model('last'))])
            ],
        )

        result = await agent.run('hello')
        assert result.output == 'last'

    async def test_capability_returning_none_is_noop(self):
        """A capability whose `get_model()` returns None (the default) leaves the agent model in place."""
        agent = Agent(_text_model('from-agent'), capabilities=[_ModelCap(model=None)])

        result = await agent.run('hello')
        assert result.output == 'from-agent'


class TestGetWrapperToolsetHook:
    async def test_wrapper_prefixes_tools(self):
        """Capability can wrap the toolset to prefix tool names."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix='cap')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f'tools: {tool_names}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrefixCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == "tools: ['cap_my_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['cap_my_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_prefixes_tools_streaming(self):
        """Wrapper toolset works correctly with streaming runs."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix='cap')

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            tool_names = sorted(t.name for t in info.function_tools)
            yield f'tools: {tool_names}'

        agent = Agent(FunctionModel(stream_function=stream_fn), capabilities=[PrefixCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        async with agent.run_stream('hello') as result:
            output = await result.get_output()
        assert output == "tools: ['cap_my_tool']"

    async def test_wrapper_does_not_affect_output_tools(self):
        """Wrapper toolset does not wrap output tools."""
        from pydantic_ai.toolsets.wrapper import WrapperToolset

        seen_tool_names: list[list[str]] = []

        @dataclass
        class SpyWrapperToolset(WrapperToolset[Any]):
            async def get_tools(self, ctx: RunContext[Any]) -> dict[str, Any]:
                tools = await super().get_tools(ctx)
                seen_tool_names.append(sorted(tools.keys()))
                return tools

        @dataclass
        class SpyWrapperCap(AbstractCapability[Any]):
            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return SpyWrapperToolset(toolset)

        agent = Agent(
            TestModel(),
            output_type=int,
            capabilities=[SpyWrapperCap()],
        )

        @agent.tool_plain
        def add_one(x: int) -> int:
            """Add one to x."""
            return x + 1

        await agent.run('hello')
        # The wrapper should only see function tools, not output tools
        for tool_names in seen_tool_names:
            assert 'add_one' in tool_names
            # Output tool names should not appear in the wrapped toolset
            assert all(not name.startswith('final_result') for name in tool_names)

    async def test_wrapper_none_is_noop(self):
        """Returning None from get_wrapper_toolset leaves the toolset unchanged."""

        @dataclass
        class NoopCap(AbstractCapability[Any]):
            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return None

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f'tools: {tool_names}')

        agent = Agent(FunctionModel(model_fn), capabilities=[NoopCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == "tools: ['my_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['my_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_chaining_order(self):
        """Multiple capabilities' wrappers compose by nesting: first wraps outermost."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            prefix: str

            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix=self.prefix)

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f'tools: {tool_names}')

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[PrefixCap(prefix='a'), PrefixCap(prefix='b')],
        )

        @agent.tool_plain
        def tool() -> str:
            return 'r'  # pragma: no cover

        result = await agent.run('hello')
        # First cap wraps outermost (matching wrap_* hooks): a_b_tool
        assert result.output == "tools: ['a_b_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['a_b_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_with_per_run_capability(self):
        """Wrapper works correctly with capabilities returning new instances from for_run."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PerRunPrefixCap(AbstractCapability[Any]):
            prefix: str = 'default'

            async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
                return PerRunPrefixCap(prefix='runtime')

            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix=self.prefix)

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f'tools: {tool_names}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PerRunPrefixCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        # The per-run instance should use 'runtime' prefix, not 'default'
        assert result.output == "tools: ['runtime_my_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['runtime_my_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_with_agent_prepare_tools(self):
        """Agent-level prepare_tools is applied before capability wrapper."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            def get_wrapper_toolset(self, toolset: AbstractToolset[Any]) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix='cap')

        async def agent_prepare(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [dc_replace(td, description=f'[prepared] {td.description}') for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            descs = [t.description for t in info.function_tools]
            return make_text_response(f'tools: {tool_names}, descs: {descs}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(agent_prepare), PrefixCap()])

        @agent.tool_plain
        def my_tool() -> str:
            """Original."""
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        # Both agent prepare_tools (description) and capability wrapper (prefix) should apply
        assert result.output == "tools: ['cap_my_tool'], descs: ['[prepared] Original.']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['cap_my_tool'], descs: ['[prepared] Original.']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


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


async def _registered_capability_context(
    *capabilities: AbstractCapability,
) -> tuple[dict[str, AbstractCapability], set[str]]:
    captured_capabilities: dict[str, AbstractCapability] = {}
    captured_available_ids: set[str] = set()

    @dataclass
    class CaptureCapabilities(AbstractCapability):
        async def before_model_request(
            self, ctx: RunContext, request_context: ModelRequestContext
        ) -> ModelRequestContext:
            captured_capabilities.update(ctx.capabilities)
            captured_available_ids.update(ctx.available_capability_ids)
            return request_context

    agent = Agent(
        FunctionModel(lambda _messages, _info: make_text_response('done')),
        capabilities=[*capabilities, CaptureCapabilities()],
    )
    await agent.run('capture capabilities')
    capability_ids = {id(capability) for capability in capabilities}
    captured_capabilities = {
        capability_id: capability
        for capability_id, capability in captured_capabilities.items()
        if id(capability) in capability_ids
    }
    captured_available_ids &= set(captured_capabilities)
    return captured_capabilities, captured_available_ids


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



# --- WrapperCapability and PrefixTools tests ---


async def test_prefix_tools_prefixes_wrapped_capability_tools():
    """PrefixTools prefixes only the wrapped capability's tools, not other agent tools."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def inner_tool() -> str:
        return 'inner'  # pragma: no cover

    cap = PrefixTools(wrapped=Toolset(toolset), prefix='ns')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])

    @agent.tool_plain
    def outer_tool() -> str:
        return 'outer'  # pragma: no cover

    result = await agent.run('list tools')
    # inner_tool should be prefixed, outer_tool should not
    assert result.output == 'ns_inner_tool,outer_tool'


async def test_prefix_tools_from_spec():
    """PrefixTools from spec supports both dict-form and bare-name nested capabilities."""

    # Dict form (kwargs): nested capability with arguments
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {
                    'PrefixTools': {
                        'prefix': 'search',
                        'capability': {'NativeTool': {'kind': 'web_search'}},
                    }
                },
            ],
        },
    )
    assert agent.model is not None

    # Bare name form with custom_capability_types forwarded through contextvar
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {
                    'PrefixTools': {
                        'prefix': 'custom',
                        'capability': 'CustomCapability',
                    }
                },
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


async def test_prefix_tools_from_spec_direct():
    """PrefixTools.from_spec works outside Agent.from_spec (no contextvar), using default registry."""
    cap = PrefixTools.from_spec(prefix='ws', capability={'WebSearch': {'local': 'duckduckgo'}})  # pyright: ignore[reportArgumentType]
    assert isinstance(cap, PrefixTools)
    assert cap.prefix == 'ws'


async def test_prefix_tools_returns_none_when_no_toolset():
    """PrefixTools.get_toolset() returns None if the wrapped capability has no toolset."""
    cap = PrefixTools(wrapped=CustomCapability(), prefix='ns')
    assert cap.get_toolset() is None


async def test_prefix_tools_with_callable_toolset():
    """PrefixTools handles a wrapped capability that returns a callable toolset."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def dynamic_tool() -> str:
        return 'dynamic'  # pragma: no cover

    def toolset_func(ctx: RunContext) -> FunctionToolset:
        return toolset

    cap = PrefixTools(wrapped=Toolset(toolset_func), prefix='dyn')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('list tools')
    assert result.output == 'dyn_dynamic_tool'


async def test_prefix_tools_inherits_wrapped_metadata_for_registration():
    """A wrapper with no id of its own delegates identity to the capability it wraps.

    This is what lets a wrapper sit over a deferred capability without losing its deferral or
    its place in the load catalog: the wrapper registers under the wrapped capability's id and
    keeps `defer_loading` and `description`.
    """
    toolset = FunctionToolset()
    wrapped = Toolset(
        toolset,
        id='leaf-tools',
        description='Leaf tool bundle.',
        defer_loading=True,
    )
    cap = PrefixTools(wrapped=wrapped, prefix='leaf')

    visited: list[AbstractCapability] = []
    cap.apply(visited.append)
    capability_map, available_ids = await _registered_capability_context(cap)

    assert cap.id == 'leaf-tools'
    assert cap.defer_loading is True
    assert cap.get_description() == 'Leaf tool bundle.'
    assert capability_map == {'leaf-tools': cap}
    # Deferred and not yet loaded, so it is registered but not available this turn.
    assert 'leaf-tools' not in available_ids
    assert visited == [cap]


async def test_prefix_tools_can_override_metadata():
    """A wrapper with explicit metadata becomes its own registered capability."""
    wrapped = Toolset(FunctionToolset(), id='leaf-tools', description='Leaf tool bundle.', defer_loading=True)
    cap = PrefixTools(
        wrapped=wrapped,
        prefix='leaf',
        id='prefixed-leaf-tools',
        description='Prefixed leaf tools.',
        defer_loading=False,
    )

    visited: list[AbstractCapability] = []
    cap.apply(visited.append)
    capability_map, available_ids = await _registered_capability_context(cap)

    assert cap.id == 'prefixed-leaf-tools'
    assert cap.description == 'Prefixed leaf tools.'
    assert capability_map == {'prefixed-leaf-tools': cap}
    assert 'prefixed-leaf-tools' in available_ids
    assert cap.defer_loading is False
    assert visited == [cap]


async def test_prefix_tools_registration_inherits_or_overrides_wrapper_metadata():
    """A wrapper inherits the wrapped capability's identity, unless it sets its own id."""

    github = Capability[object](
        id='github',
        description='GitHub MCP server.',
        defer_loading=True,
    )

    # No id of its own: inherit the wrapped capability's id, deferral, and description, so the
    # deferred capability still shows up in the load catalog under its own id.
    prefixed = PrefixTools(github, prefix='github')

    registered, available_ids = await _registered_capability_context(prefixed)

    assert registered['github'] is prefixed
    assert 'github' not in available_ids
    assert prefixed.id == 'github'
    assert prefixed.defer_loading is True
    assert prefixed.get_description() == 'GitHub MCP server.'

    # An explicit id makes the wrapper its own capability: it no longer inherits the wrapped
    # capability's id or deferral, though it still falls back to its description.
    explicit_id = PrefixTools(github, prefix='github', id='github_prefixed')
    registered, available_ids = await _registered_capability_context(explicit_id)

    assert registered['github_prefixed'] is explicit_id
    assert 'github_prefixed' in available_ids
    assert explicit_id.defer_loading is False
    assert explicit_id.get_description() == 'GitHub MCP server.'


async def test_wrapper_over_deferred_capability_preserves_deferral_end_to_end() -> None:
    """Wrapping a deferred capability keeps it deferred through a full run.

    Regression guard for metadata delegation: a wrapper with no id of its own must surface the
    wrapped deferred capability in the load catalog and reveal its (prefixed) tools after
    `load_capability`, rather than silently becoming an always-available capability.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        toolsets=[toolset],
        defer_loading=True,
    )
    wrapped = PrefixTools(refunds, prefix='refunds')

    first_request_instructions: list[str | None] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            first_request = message(messages, ModelRequest)
            first_request_instructions.append(first_request.instructions)
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )

        if not any(part.tool_name == 'refunds_lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='refunds_lookup_refund_policy',
                        args={'order_id': 'order-1'},
                        tool_call_id='lookup',
                    )
                ]
            )

        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[wrapped])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done'
    # The deferred capability is surfaced in the catalog under the wrapped capability's id.
    assert first_request_instructions == [
        "The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:\n"
        '- refunds: Refund policy tools.'
    ]


async def test_prefix_tools_explicit_defer_loading_overrides_anonymous_wrapped() -> None:
    """`PrefixTools(..., id='github', defer_loading=True)` over an anonymous wrapped
    capability registers as deferred under the wrapper's own id, not the wrapped's."""
    explicit_deferred = PrefixTools(
        Capability[object](),
        prefix='github',
        id='github',
        defer_loading=True,
    )

    registered, available_ids = await _registered_capability_context(explicit_deferred)

    assert registered['github'] is explicit_deferred
    assert 'github' not in available_ids
    assert explicit_deferred.defer_loading is True


async def test_prefix_tools_can_be_deferred():
    """A deferred PrefixTools wrapper keeps its prefixed tools deferred until load."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        return f'{order_id}: refund allowed'

    cap = PrefixTools(
        wrapped=Toolset(
            toolset,
        ),
        prefix='billing',
        id='refunds',
        description='Refund policy tools.',
        defer_loading=True,
    )
    seen_tool_state: list[list[tuple[str, bool]]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_state.append([(t.name, bool(t.defer_loading)) for t in info.function_tools])
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(isinstance(part, LoadCapabilityReturnPart) for message in messages for part in message.parts):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ]
            )

        if not any(part.tool_name == 'billing_lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='billing_lookup_refund_policy',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )

        refund_result = next(part.content for part in tool_returns if part.tool_name == 'billing_lookup_refund_policy')
        return make_text_response(f'done: {refund_result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done: order-123: refund allowed'
    assert seen_tool_state == snapshot(
        [
            [('load_capability', False)],
            [('load_capability', False), ('billing_lookup_refund_policy', True)],
            [('load_capability', False), ('billing_lookup_refund_policy', True)],
        ]
    )


async def test_prefix_tools_convenience_method():
    """AbstractCapability.prefix_tools() returns a PrefixTools wrapping self."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def inner_tool() -> str:
        return 'inner'  # pragma: no cover

    cap = Toolset(toolset).prefix_tools('ns')
    assert isinstance(cap, PrefixTools)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('list tools')
    assert result.output == 'ns_inner_tool'


async def test_wrapper_capability_delegates_hooks():
    """WrapperCapability delegates lifecycle hooks to the wrapped capability."""
    hook_calls: list[str] = []

    @dataclass
    class HookCap(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            hook_calls.append('before_run')

        async def after_run(self, ctx: RunContext, *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            hook_calls.append('after_run')
            return result

    wrapper = WrapperCapability(wrapped=HookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run('Hello')

    assert 'before_run' in hook_calls
    assert 'after_run' in hook_calls


def test_wrapper_capability_for_agent_replaces():
    """WrapperCapability.for_agent replaces wrapped when its for_agent rebinds.

    Some capabilities (e.g. `TemporalDurability`) snapshot agent state in `for_agent`
    and return a new instance. The wrapper must propagate that.
    """

    @dataclass
    class RebindCap(AbstractCapability[None]):
        bound_to: str = ''

        def for_agent(self, agent: AbstractAgent[None, Any]) -> AbstractCapability[None]:
            return RebindCap(bound_to=agent.name or '')

    inner = RebindCap()
    wrapper = WrapperCapability(wrapped=inner)

    agent = Agent(FunctionModel(_resolve_dummy_model_fn), name='wrapper_for_agent_test')
    bound = wrapper.for_agent(agent)
    assert isinstance(bound, WrapperCapability)
    assert bound is not wrapper
    assert bound.wrapped is not inner
    assert cast(RebindCap, bound.wrapped).bound_to == 'wrapper_for_agent_test'


async def test_wrapper_capability_for_run_replaces():
    """WrapperCapability.for_run replaces wrapped when it changes."""
    toolset_a = FunctionToolset(id='a')

    @toolset_a.tool_plain
    def tool_a() -> str:
        return 'a'  # pragma: no cover

    toolset_b = FunctionToolset(id='b')

    @toolset_b.tool_plain
    def tool_b() -> str:
        return 'b'  # pragma: no cover

    @dataclass
    class SwitchCap(AbstractCapability):
        use_b: bool = False

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return SwitchCap(use_b=True)

        def get_toolset(self) -> AbstractToolset:
            return toolset_b if self.use_b else toolset_a

    wrapper = WrapperCapability(wrapped=SwitchCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    result = await agent.run('Hello')
    # for_run switches to toolset_b
    assert 'tool_b' in result.output


async def test_wrapper_capability_for_run_preserves_explicit_metadata() -> None:
    """WrapperCapability.for_run preserves explicit wrapper metadata."""

    @dataclass
    class SwitchCap(AbstractCapability):
        name: str = 'before'

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return SwitchCap(name='after')

    wrapper = WrapperCapability(
        wrapped=SwitchCap(),
        id='explicit-wrapper',
        description='Explicit wrapper metadata.',
        defer_loading=False,
    )

    result = await wrapper.for_run(_build_run_context())

    assert result is not wrapper
    assert isinstance(result, WrapperCapability)
    assert result.id == 'explicit-wrapper'
    assert result.description == 'Explicit wrapper metadata.'
    assert result.defer_loading is False
    assert isinstance(result.wrapped, SwitchCap)
    assert result.wrapped.name == 'after'


async def test_wrapper_capability_has_wrap_node_run():
    """WrapperCapability.has_wrap_node_run delegates to the wrapped capability."""
    plain = CustomCapability()
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert WrapperCapability(wrapped=plain).has_wrap_node_run is False  # type: ignore[reportDeprecated]

    @dataclass
    class NodeRunCap(AbstractCapability):
        async def wrap_node_run(self, ctx: RunContext, *, node: Any, handler: Any) -> Any:
            return await handler(node)  # pragma: no cover

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert WrapperCapability(wrapped=NodeRunCap()).has_wrap_node_run is True  # type: ignore[reportDeprecated]


async def test_combined_capability_has_wrap_node_run():
    """CombinedCapability.has_wrap_node_run reports whether any child overrides the hook.

    Nothing in the library branches on this anymore — the bare-iteration warning it used to gate
    is gone now that `async for node in agent_run` fires node hooks — but it stays available for
    capability authors introspecting a chain, alongside `has_wrap_run_event_stream`.
    """

    @dataclass
    class NodeRunCap(AbstractCapability):
        async def wrap_node_run(self, ctx: RunContext, *, node: Any, handler: Any) -> Any:
            return await handler(node)  # pragma: no cover

    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert CombinedCapability([CustomCapability()]).has_wrap_node_run is False  # type: ignore[reportDeprecated]
    with pytest.warns(PydanticAIDeprecationWarning, match=r'`has_wrap_node_run`.*`wrap_node_run`'):
        assert CombinedCapability([CustomCapability(), NodeRunCap()]).has_wrap_node_run is True  # type: ignore[reportDeprecated]


async def test_wrapper_capability_delegates_resolve_model_id():
    """WrapperCapability delegates `resolve_model_id` (and `has_resolve_model_id`) to the wrapped capability."""
    resolved = TestModel()

    @dataclass
    class ResolverCap(AbstractCapability[Any]):
        async def resolve_model_id(self, ctx: ModelResolutionContext[Any], *, model_id: str) -> Any:
            return resolved if model_id == 'magic' else None

    wrapper = WrapperCapability(wrapped=ResolverCap())
    assert wrapper.has_resolve_model_id is True

    agent = Agent('test', capabilities=[wrapper])
    resolution_ctx = ModelResolutionContext[Any](agent=agent, deps=None)
    assert await wrapper.resolve_model_id(resolution_ctx, model_id='magic') is resolved
    assert await wrapper.resolve_model_id(resolution_ctx, model_id='other') is None

    # Wrapping a capability without `resolve_model_id` is a no-op.
    plain_wrapper = WrapperCapability(wrapped=CustomCapability())
    assert plain_wrapper.has_resolve_model_id is False
    assert await plain_wrapper.resolve_model_id(resolution_ctx, model_id='any') is None


async def test_wrapper_capability_delegates_model_request_hooks():
    """WrapperCapability delegates before/after model request hooks."""
    hook_calls: list[str] = []

    @dataclass
    class ModelRequestHookCap(AbstractCapability):
        async def before_model_request(
            self, ctx: RunContext, request_context: ModelRequestContext
        ) -> ModelRequestContext:
            hook_calls.append('before_model_request')
            return request_context

        async def after_model_request(
            self, ctx: RunContext, *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            hook_calls.append('after_model_request')
            return response

    wrapper = WrapperCapability(wrapped=ModelRequestHookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run('Hello')

    assert 'before_model_request' in hook_calls
    assert 'after_model_request' in hook_calls


async def test_prefix_tools_tool_call_strips_prefix():
    """PrefixTools correctly strips the prefix when calling the underlying tool."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def greet(name: str) -> str:
        return f'hello {name}'

    cap = PrefixTools(wrapped=Toolset(toolset), prefix='ns')

    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('ns_greet', {'name': 'world'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('greet world')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='greet world', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='ns_greet',
                        args={'name': 'world'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ns_greet',
                        content='hello world',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=54, output_tokens=6),
                model_name='function:respond:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_wrapper_capability_get_serialization_name():
    """WrapperCapability.get_serialization_name returns None (abstract base)."""
    assert WrapperCapability.get_serialization_name() is None


async def test_wrapper_capability_delegates_on_run_error():
    """WrapperCapability delegates on_run_error to the wrapped capability."""

    @dataclass
    class RecoverCap(AbstractCapability[Any]):
        async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output='recovered')

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    agent = Agent(FunctionModel(failing_model), capabilities=[WrapperCapability(wrapped=RecoverCap())])
    result = await agent.run('hello')
    assert result.output == 'recovered'


async def test_wrapper_capability_delegates_on_node_run_error():
    """WrapperCapability delegates on_node_run_error to the wrapped capability."""
    from pydantic_graph import End

    @dataclass
    class NodeRecoverCap(AbstractCapability[Any]):
        async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            return End(FinalResult(output='node recovered'))

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model exploded')

    agent = Agent(FunctionModel(failing_model), capabilities=[WrapperCapability(wrapped=NodeRecoverCap())])
    async with agent.iter('hello') as agent_run:
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)
    assert isinstance(node, End)
    assert node.data.output == 'node recovered'


async def test_wrapper_capability_delegates_wrap_run_event_stream():
    """WrapperCapability delegates wrap_run_event_stream to the wrapped capability."""
    observed_events: list[AgentStreamEvent] = []

    @dataclass
    class StreamObserverCap(AbstractCapability[Any]):
        async def wrap_run_event_stream(
            self,
            ctx: RunContext[Any],
            *,
            stream: AsyncIterable[AgentStreamEvent],
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                observed_events.append(event)
                yield event

    agent = Agent(
        FunctionModel(simple_model_function, stream_function=simple_stream_function),
        capabilities=[WrapperCapability(wrapped=StreamObserverCap())],
    )

    async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for _ in stream:
            pass

    await agent.run('hello', event_stream_handler=handler)
    assert len(observed_events) > 0


async def test_wrapper_capability_delegates_on_model_request_error():
    """WrapperCapability delegates on_model_request_error to the wrapped capability."""

    @dataclass
    class ModelErrorRecoverCap(AbstractCapability[Any]):
        async def on_model_request_error(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='recovered from model error')])

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('model request failed')

    agent = Agent(FunctionModel(failing_model), capabilities=[WrapperCapability(wrapped=ModelErrorRecoverCap())])
    result = await agent.run('hello')
    assert result.output == 'recovered from model error'


async def test_wrapper_capability_delegates_on_tool_validate_error():
    """WrapperCapability delegates on_tool_validate_error to the wrapped capability."""

    @dataclass
    class ValidateErrorCap(AbstractCapability[Any]):
        async def on_tool_validate_error(
            self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            # Recover by providing valid args
            return {'x': 1}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(content='done')])
        if info.function_tools:
            return ModelResponse(parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='invalid json!!')])
        return ModelResponse(parts=[TextPart(content='no tools')])  # pragma: no cover

    agent = Agent(FunctionModel(model_fn), capabilities=[WrapperCapability(wrapped=ValidateErrorCap())])

    @agent.tool_plain
    def my_tool(x: int) -> str:
        return f'result: {x}'

    result = await agent.run('call tool')
    assert result.output == 'done'


async def test_wrapper_capability_delegates_on_tool_execute_error():
    """WrapperCapability delegates on_tool_execute_error to the wrapped capability."""

    @dataclass
    class ExecuteErrorCap(AbstractCapability[Any]):
        async def on_tool_execute_error(
            self,
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'recovered tool result'

    agent = Agent(
        FunctionModel(tool_calling_model),
        capabilities=[WrapperCapability(wrapped=ExecuteErrorCap())],
    )

    @agent.tool_plain
    def my_tool() -> str:
        raise ValueError('tool failed')

    result = await agent.run('call tool')
    assert result.output == 'final response'



# region --- Compaction capability tests ---


class TestCompaction:
    def test_compaction_part_serialization(self):
        """CompactionPart round-trips through Pydantic serialization."""
        from pydantic_ai.messages import CompactionPart, ModelResponse

        # Anthropic-style (text content)
        anthropic_part = CompactionPart(content='Summary of conversation', provider_name='anthropic')
        assert anthropic_part.has_content()
        assert anthropic_part.part_kind == 'compaction'

        # OpenAI-style (encrypted, no content)
        openai_part = CompactionPart(
            content=None,
            id='cmp_123',
            provider_name='openai',
            provider_details={'encrypted_content': 'abc123', 'type': 'compaction'},
        )
        assert not openai_part.has_content()
        assert openai_part.part_kind == 'compaction'

        # Round-trip through serialization
        response = ModelResponse(parts=[anthropic_part, openai_part])
        messages: list[ModelMessage] = [response]
        serialized = ModelMessagesTypeAdapter.dump_json(messages)
        deserialized = ModelMessagesTypeAdapter.validate_json(serialized)
        assert len(deserialized) == 1
        assert isinstance(deserialized[0], ModelResponse)
        parts = deserialized[0].parts
        assert len(parts) == 2
        assert isinstance(parts[0], CompactionPart)
        assert parts[0].content == 'Summary of conversation'
        assert parts[0].provider_name == 'anthropic'
        assert isinstance(parts[1], CompactionPart)
        assert parts[1].content is None
        assert parts[1].id == 'cmp_123'
        assert parts[1].provider_details == {'encrypted_content': 'abc123', 'type': 'compaction'}

    async def test_openai_compaction_with_wrong_model(self):
        """OpenAICompaction raises UserError when used with a non-OpenAI model."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[OpenAICompaction(message_count_threshold=0)],
        )
        with pytest.raises(UserError, match='OpenAICompaction requires OpenAIResponsesModel'):
            await agent.run('hello')

    async def test_openai_compaction_with_wrapped_wrong_model(self):
        """OpenAICompaction unwraps WrapperModel and raises for non-OpenAI model."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction
        from pydantic_ai.models.wrapper import WrapperModel

        wrapped = WrapperModel(FunctionModel(simple_model_function))
        agent = Agent(
            wrapped,
            capabilities=[OpenAICompaction(message_count_threshold=0)],
        )
        with pytest.raises(UserError, match='OpenAICompaction requires OpenAIResponsesModel'):
            await agent.run('hello')

    def test_openai_compaction_should_compact_with_trigger(self):
        """OpenAICompaction._should_compact delegates to custom trigger."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        cap = OpenAICompaction(trigger=lambda msgs: len(msgs) > 2)
        assert not cap._should_compact([ModelRequest(parts=[UserPromptPart(content='hi')])])  # pyright: ignore[reportPrivateUsage]
        assert cap._should_compact(  # pyright: ignore[reportPrivateUsage]
            [
                ModelRequest(parts=[UserPromptPart(content='1')]),
                ModelResponse(parts=[TextPart(content='r1')]),
                ModelRequest(parts=[UserPromptPart(content='2')]),
            ]
        )

    def test_openai_compaction_should_compact_no_config(self):
        """Bare `OpenAICompaction()` is stateful mode and never triggers the before_model_request hook."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        cap = OpenAICompaction()
        assert cap.stateless is False
        assert not cap._should_compact([ModelRequest(parts=[UserPromptPart(content='hi')])])  # pyright: ignore[reportPrivateUsage]

    def test_openai_compaction_mode_inference(self):
        """`stateless` is inferred from which mode-specific fields are passed."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        assert OpenAICompaction().stateless is False
        assert OpenAICompaction(token_threshold=1000).stateless is False
        assert OpenAICompaction(message_count_threshold=5).stateless is True
        assert OpenAICompaction(trigger=lambda _msgs: True).stateless is True

    def test_openai_compaction_stateful_model_settings(self):
        """Stateful mode returns `openai_context_management` via get_model_settings."""
        pytest.importorskip('openai')
        from types import SimpleNamespace
        from typing import cast

        from pydantic_ai.models.openai import OpenAICompaction

        def _resolve(cap: OpenAICompaction, model_settings: dict[str, Any] | None = None) -> dict[str, Any]:
            resolver = cap.get_model_settings()
            assert resolver is not None
            ctx = SimpleNamespace(model_settings=model_settings)
            return cast(dict[str, Any], resolver(cast(Any, ctx)))

        assert _resolve(OpenAICompaction()) == {'openai_context_management': [{'type': 'compaction'}]}
        assert _resolve(OpenAICompaction(token_threshold=50_000)) == {
            'openai_context_management': [{'type': 'compaction', 'compact_threshold': 50_000}]
        }
        # If the user already configured `openai_context_management` directly, we defer
        # to them entirely and don't append our own entry. OpenAI's context_management
        # list only meaningfully supports one `compaction` entry, so mixing the capability
        # with manual config would produce ambiguous/conflicting state.
        assert (
            _resolve(
                OpenAICompaction(token_threshold=50_000),
                model_settings={'openai_context_management': [{'type': 'compaction', 'compact_threshold': 200_000}]},
            )
            == {}
        )
        # When user has other model settings but no `openai_context_management`,
        # the capability's compaction entry is injected normally.
        assert _resolve(
            OpenAICompaction(token_threshold=50_000),
            model_settings={'temperature': 0.5},
        ) == {'openai_context_management': [{'type': 'compaction', 'compact_threshold': 50_000}]}
        # Stateless mode does not inject model settings
        assert OpenAICompaction(message_count_threshold=5).get_model_settings() is None

    def test_openai_compaction_rejects_mixed_fields(self):
        """Mixing stateful-only and stateless-only fields raises UserError."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        with pytest.raises(UserError, match='`token_threshold` is only valid for stateful compaction'):
            OpenAICompaction(stateless=True, token_threshold=1000, message_count_threshold=5)

        with pytest.raises(UserError, match='only valid for stateless compaction'):
            OpenAICompaction(stateless=False, message_count_threshold=5)

        with pytest.raises(UserError, match='only valid for stateless compaction'):
            OpenAICompaction(stateless=False, trigger=lambda _msgs: True)

    def test_openai_compaction_stateless_requires_trigger(self):
        """`stateless=True` without message_count_threshold or trigger raises UserError."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        with pytest.raises(UserError, match='requires `message_count_threshold` or `trigger`'):
            OpenAICompaction(stateless=True)

    def test_openai_compaction_serialization_name(self):
        """OpenAICompaction has the correct serialization name."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        assert OpenAICompaction.get_serialization_name() == 'OpenAICompaction'

    def test_anthropic_compaction_serialization_name(self):
        """AnthropicCompaction has the correct serialization name."""
        pytest.importorskip('anthropic')
        from pydantic_ai.models.anthropic import AnthropicCompaction

        assert AnthropicCompaction.get_serialization_name() == 'AnthropicCompaction'

    async def test_compaction_part_in_function_model_history(self):
        """FunctionModel handles message history containing CompactionPart."""
        from pydantic_ai.messages import CompactionPart

        compaction_response = ModelResponse(
            parts=[CompactionPart(content='Summary: user greeted.', provider_name='anthropic')],
            provider_name='anthropic',
        )
        history: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='Hello!')]),
            compaction_response,
            ModelRequest(parts=[UserPromptPart(content='How are you?')]),
        ]

        agent = Agent(FunctionModel(simple_model_function))
        result = await agent.run('Follow up', message_history=history)
        assert result.output == 'response from model'

    async def test_compaction_part_without_content_in_response(self):
        """CompactionPart with content=None (OpenAI-style) is handled alongside text."""
        from pydantic_ai.messages import CompactionPart

        def model_with_compaction(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    CompactionPart(content=None, id='cmp_123', provider_name='openai'),
                    TextPart(content='actual response'),
                ]
            )

        agent = Agent(FunctionModel(model_with_compaction))
        result = await agent.run('hello')
        assert result.output == 'actual response'


# endregion


def test_thread_executor_not_serializable() -> None:
    assert UseThreadExecutor.get_serialization_name() is None


def test_thread_executor_deprecated_alias() -> None:
    from pydantic_ai.exceptions import PydanticAIDeprecationWarning

    with pytest.warns(PydanticAIDeprecationWarning, match='renamed to `UseThreadExecutor`'):
        from pydantic_ai.capabilities import ThreadExecutor
    assert ThreadExecutor is UseThreadExecutor

    # The defining module resolves the old name too, so unpickling keeps working.
    with pytest.warns(PydanticAIDeprecationWarning, match='renamed to `UseThreadExecutor`'):
        from pydantic_ai.capabilities.thread_executor import ThreadExecutor as submodule_thread_executor
    assert submodule_thread_executor is UseThreadExecutor


async def test_thread_executor_capability() -> None:
    tool_threads: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='check_thread', args='{}')])

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='cap-pool')
    try:
        agent = Agent(FunctionModel(model_function), capabilities=[UseThreadExecutor(executor)])

        @agent.tool_plain
        def check_thread() -> str:
            tool_threads.append(threading.current_thread().name)
            return 'ok'

        result = await agent.run('test')
        assert result.output == 'done'
        assert len(tool_threads) == 1
        assert tool_threads[0].startswith('cap-pool')
    finally:
        executor.shutdown(wait=True)


async def test_thread_executor_static_method() -> None:
    tool_threads: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='check_thread', args='{}')])

    agent = Agent(FunctionModel(model_function))

    @agent.tool_plain
    def check_thread() -> str:
        tool_threads.append(threading.current_thread().name)
        return 'ok'

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='static-pool')
    try:
        with Agent.using_thread_executor(executor):
            result = await agent.run('test')
        assert result.output == 'done'
        assert len(tool_threads) == 1
        assert tool_threads[0].startswith('static-pool')
    finally:
        executor.shutdown(wait=True)


# --- Capability ordering tests ---


@dataclass
class OutermostCap(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')


@dataclass
class InnermostCap(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')


@dataclass
class PlainCapA(AbstractCapability[Any]):
    pass


@dataclass
class PlainCapB(AbstractCapability[Any]):
    pass


@dataclass
class WrapsACap(AbstractCapability[Any]):
    """Must wrap around PlainCapA."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=[PlainCapA])


@dataclass
class RequiresOutermostCap(AbstractCapability[Any]):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(requires=[OutermostCap])


def _cap_names(combined: CombinedCapability) -> list[str]:
    return [type(c).__name__ for c in combined.capabilities]


def test_ordering_outermost():
    """Capability declaring 'outermost' ends up at index 0."""
    combined = CombinedCapability([PlainCapA(), OutermostCap(), PlainCapB()])
    assert _cap_names(combined) == ['OutermostCap', 'PlainCapA', 'PlainCapB']


def test_ordering_innermost():
    """Capability declaring 'innermost' ends up last."""
    combined = CombinedCapability([InnermostCap(), PlainCapA(), PlainCapB()])
    assert _cap_names(combined) == ['PlainCapA', 'PlainCapB', 'InnermostCap']


def test_ordering_both_outermost_and_innermost():
    """Both outermost and innermost present."""
    combined = CombinedCapability([PlainCapA(), InnermostCap(), OutermostCap()])
    assert combined.capabilities[0].__class__ is OutermostCap
    assert combined.capabilities[-1].__class__ is InnermostCap


def test_ordering_multiple_outermost_tier():
    """Multiple outermost capabilities form a tier; original order breaks ties."""

    @dataclass
    class OutermostCap2(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost')

    combined = CombinedCapability([PlainCapA(), OutermostCap2(), OutermostCap()])
    # Both outermost caps before PlainCapA; original order (OutermostCap2 before OutermostCap) preserved
    assert _cap_names(combined) == ['OutermostCap2', 'OutermostCap', 'PlainCapA']


def test_ordering_multiple_innermost_tier():
    """Multiple innermost capabilities form a tier; original order breaks ties."""

    @dataclass
    class InnermostCap2(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

    combined = CombinedCapability([InnermostCap(), InnermostCap2(), PlainCapA()])
    # PlainCapA first, then both innermost in original order
    assert _cap_names(combined) == ['PlainCapA', 'InnermostCap', 'InnermostCap2']


def test_ordering_outermost_tier_with_wraps():
    """wraps/wrapped_by refines order within the outermost tier."""

    @dataclass
    class OuterA(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost')

    @dataclass
    class OuterB(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='outermost', wraps=[OuterA])

    # OuterB listed after OuterA, but wraps=[OuterA] overrides tiebreaker
    combined = CombinedCapability([OuterA(), PlainCapA(), OuterB()])
    assert _cap_names(combined) == ['OuterB', 'OuterA', 'PlainCapA']


def test_ordering_wraps():
    """Explicit 'wraps' edge is respected."""
    combined = CombinedCapability([PlainCapA(), WrapsACap()])
    assert _cap_names(combined) == ['WrapsACap', 'PlainCapA']


def test_ordering_wrapped_by():
    """Explicit 'wrapped_by' edge is respected."""

    @dataclass
    class WrappedByACap(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wrapped_by=[PlainCapA])

    combined = CombinedCapability([WrappedByACap(), PlainCapA()])
    assert _cap_names(combined) == ['PlainCapA', 'WrappedByACap']


def test_innermost_binds_after_capability_toolsets():
    """`innermost` capabilities bind after other capabilities' toolsets are extracted.

    Durability capabilities (the `innermost` tier) wrap `agent.toolsets` in their `for_agent`,
    so `Agent.__init__` binds them in a second phase, after toolsets contributed by other
    capabilities (e.g. `Capability(tools=...)`) have been extracted and are visible on the
    agent. Binding everything in one phase would leave those toolsets invisible to durability
    and running unwrapped (non-deterministically) inside durable workflows.
    """
    seen_tool_names: set[str] = set()

    @dataclass
    class RecordingInnermostCap(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> RecordingInnermostCap:
            for toolset in agent.toolsets:
                toolset.apply(
                    lambda leaf: seen_tool_names.update(leaf.tools) if isinstance(leaf, FunctionToolset) else None
                )
            # Return a bound copy, like durability capabilities do.
            return replace(self)

        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(position='innermost')

    def greet() -> str:
        return 'hi'  # pragma: no cover

    original = RecordingInnermostCap()
    agent = Agent('test', capabilities=[Capability(tools=[greet]), original])
    assert seen_tool_names == {'greet'}
    # The bound copy replaced the original in the agent's capability chain.
    assert not any(cap is original for cap in agent.root_capability.capabilities)
    assert any(isinstance(cap, RecordingInnermostCap) for cap in agent.root_capability.capabilities)


def test_combined_capability_for_agent_binds_children():
    """`CombinedCapability.for_agent` rebinds children that return new bound instances."""

    @dataclass
    class BindingCap(AbstractCapability[Any]):
        bound: bool = False

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> BindingCap:
            return replace(self, bound=True)

    combined = CombinedCapability([BindingCap(), PlainCapA()])
    agent = Agent('test')
    bound = combined.for_agent(agent)
    assert bound is not combined
    assert isinstance(bound.capabilities[0], BindingCap)
    assert bound.capabilities[0].bound is True


def test_ordering_requires_present():
    """No error when required capability is present."""
    combined = CombinedCapability([RequiresOutermostCap(), OutermostCap()])
    assert len(combined.capabilities) == 2


def test_ordering_requires_missing():
    with pytest.raises(UserError, match='`RequiresOutermostCap` requires `OutermostCap`'):
        CombinedCapability([RequiresOutermostCap(), PlainCapA()])


def test_ordering_preserves_user_order():
    """Capabilities without constraints keep their relative order."""
    a, b = PlainCapB(), PlainCapA()
    combined = CombinedCapability([a, b])
    assert list(combined.capabilities) == [a, b]


def test_ordering_nested_combined():
    """Leaves of a nested `CombinedCapability` participate as siblings in the outer sort.

    `CombinedCapability` auto-flattens nested instances so each leaf is sorted
    independently rather than as a group. Here `OutermostCap` (inside `inner`)
    sorts to the front; its former sibling `PlainCapB` is unconstrained.
    """
    inner = CombinedCapability([PlainCapB(), OutermostCap()])
    combined = CombinedCapability([PlainCapA(), inner])
    # `inner` is splatted; `OutermostCap` sorts first.
    assert [type(c) for c in combined.capabilities] == [OutermostCap, PlainCapA, PlainCapB]


def test_ordering_nested_combined_no_constraints():
    """A nested `CombinedCapability` with no ordering leaves is splatted as flat siblings."""
    inner = CombinedCapability([PlainCapA(), PlainCapB()])
    combined = CombinedCapability([inner, OutermostCap()])
    # `OutermostCap` first; `inner`'s leaves follow as flat siblings in their original order.
    assert [type(c) for c in combined.capabilities] == [OutermostCap, PlainCapA, PlainCapB]


def test_ordering_nested_combined_wraps_without_position():
    """A `wraps` constraint on a leaf inside a nested `CombinedCapability` applies to that leaf only."""
    inner = CombinedCapability([PlainCapB(), WrapsACap()])
    combined = CombinedCapability([PlainCapA(), inner])
    # `WrapsACap` is splatted and sorts before `PlainCapA`; `PlainCapB` is unconstrained
    # and keeps its insertion order (it sits between PlainCapA and WrapsACap in the
    # post-flatten input list, so the topo sort surfaces it first as ready-without-deps).
    assert [type(c) for c in combined.capabilities] == [PlainCapB, WrapsACap, PlainCapA]


def test_ordering_single_capability():
    """Single capability in CombinedCapability is unchanged."""
    cap = OutermostCap()
    combined = CombinedCapability([cap])
    assert list(combined.capabilities) == [cap]


def test_ordering_no_constraints_noop():
    """When no capability declares ordering, list is unchanged."""
    a, b = PlainCapA(), PlainCapB()
    combined = CombinedCapability([a, b])
    assert list(combined.capabilities) == [a, b]


def test_ordering_cycle_detection():
    @dataclass
    class CycleA(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[CycleB])

    @dataclass
    class CycleB(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[CycleA])

    with pytest.raises(UserError, match='Circular ordering constraints'):
        CombinedCapability([CycleA(), CycleB()])


def test_ordering_mixed_positions_in_nested():
    """Mixed positions in a nested `CombinedCapability` work — leaves are splatted into the outer sort."""
    inner = CombinedCapability([OutermostCap(), InnermostCap()])
    combined = CombinedCapability([inner, PlainCapA()])
    # `OutermostCap` first (outermost tier), `PlainCapA` middle, `InnermostCap` last (innermost tier).
    assert [type(c) for c in combined.capabilities] == [OutermostCap, PlainCapA, InnermostCap]


def test_ordering_conflicting_positions_in_custom_nested_capability():
    """A custom capability tree cannot collapse outermost and innermost leaves into one ordered group."""

    @dataclass
    class NestedCapabilityGroup(AbstractCapability[Any]):
        leaves: tuple[AbstractCapability[Any], ...]

        def apply(self, visitor: Callable[[AbstractCapability[Any]], None]) -> None:
            for leaf in self.leaves:
                leaf.apply(visitor)

    nested = NestedCapabilityGroup((OutermostCap(), InnermostCap()))

    with pytest.raises(UserError, match='Conflicting positions among nested leaves'):
        CombinedCapability([nested, PlainCapA()])


def test_ordering_hooks_ordering_parameter():
    """Hooks with ordering= are sorted according to those constraints."""
    hooks = Hooks(ordering=CapabilityOrdering(position='outermost'))
    combined = CombinedCapability([PlainCapA(), hooks, PlainCapB()])
    assert combined.capabilities[0] is hooks


def test_ordering_hooks_ordering_wraps():
    """Hooks with ordering wraps= are placed before the referenced type."""
    hooks = Hooks(ordering=CapabilityOrdering(wraps=[PlainCapA]))
    combined = CombinedCapability([PlainCapA(), hooks])
    assert combined.capabilities[0] is hooks


def test_ordering_hooks_ordering_wrapped_by():
    """Hooks with ordering wrapped_by= are placed after the referenced type."""
    hooks = Hooks(ordering=CapabilityOrdering(wrapped_by=[PlainCapA]))
    combined = CombinedCapability([hooks, PlainCapA()])
    assert combined.capabilities[0].__class__ is PlainCapA
    assert combined.capabilities[1] is hooks


def test_ordering_hooks_no_ordering():
    """Hooks without ordering= preserve their list position."""
    hooks = Hooks()
    combined = CombinedCapability([PlainCapA(), hooks, PlainCapB()])
    assert combined.capabilities[1] is hooks


def test_ordering_hooks_ordering_requires():
    """Hooks with ordering requires= validates that the required type is present."""
    hooks = Hooks(ordering=CapabilityOrdering(requires=[OutermostCap]))
    with pytest.raises(UserError, match='`Hooks` requires `OutermostCap`'):
        CombinedCapability([hooks, PlainCapA()])


def test_ordering_wraps_instance_ref():
    """wraps= with an instance ref only constrains the specific instance, not all instances of that type."""
    target = PlainCapA()
    other_a = PlainCapA()

    @dataclass
    class WrapsInstance(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[target])

    # Arrange so that instance ref vs type ref produces a distinguishable result:
    # - Instance ref wraps=[target] → only target must come after WrapsInstance
    # - A type ref wraps=[PlainCapA] would constrain both other_a and target
    combined = CombinedCapability([other_a, target, WrapsInstance()])
    # other_a stays before WrapsInstance (no constraint), WrapsInstance before target
    assert combined.capabilities[0] is other_a
    assert combined.capabilities[1].__class__ is WrapsInstance
    assert combined.capabilities[2] is target


def test_ordering_wrapped_by_instance_ref():
    """wrapped_by= can reference a specific capability instance."""
    wrapper = PlainCapA()

    @dataclass
    class WrappedByInstance(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wrapped_by=[wrapper])

    combined = CombinedCapability([WrappedByInstance(), wrapper])
    assert combined.capabilities[0] is wrapper
    assert combined.capabilities[1].__class__ is WrappedByInstance


def test_ordering_hooks_wraps_instance():
    """Hooks can order relative to a specific capability instance via wraps=."""
    target = PlainCapA()
    hooks = Hooks(ordering=CapabilityOrdering(wraps=[target]))
    combined = CombinedCapability([target, hooks])
    assert combined.capabilities[0] is hooks
    assert combined.capabilities[1] is target


def test_ordering_hooks_wrapped_by_instance():
    """Hooks can order relative to a specific capability instance via wrapped_by=."""
    outer = PlainCapA()
    hooks = Hooks(ordering=CapabilityOrdering(wrapped_by=[outer]))
    combined = CombinedCapability([hooks, outer])
    assert combined.capabilities[0] is outer
    assert combined.capabilities[1] is hooks


def test_ordering_instance_ref_not_present():
    """Instance ref in wraps= that isn't in the list has no effect (no edge added)."""
    absent = PlainCapA()
    hooks = Hooks(ordering=CapabilityOrdering(wraps=[absent]))
    # absent is NOT in the capabilities list — the wraps ref should be a no-op
    combined = CombinedCapability([PlainCapB(), hooks])
    # Order preserved since the instance ref doesn't match anything
    assert combined.capabilities[0].__class__ is PlainCapB
    assert combined.capabilities[1] is hooks


def test_ordering_mixed_type_and_instance_refs():
    """wraps= can mix type refs and instance refs."""
    target_instance = PlainCapB()

    @dataclass
    class MixedRefs(AbstractCapability[Any]):
        def get_ordering(self) -> CapabilityOrdering:
            return CapabilityOrdering(wraps=[PlainCapA, target_instance])

    combined = CombinedCapability([PlainCapA(), target_instance, MixedRefs()])
    assert combined.capabilities[0].__class__ is MixedRefs


async def test_ordering_survives_dynamic_capability_resolution():
    """A factory-returned capability's ordering constraints survive the per-run wrapper.

    `CombinedCapability.for_run` re-sorts the replaced capabilities, so the
    `ResolvedDynamicCapability` wrapper must delegate `get_ordering` to the resolved
    capability for its `outermost`/`innermost`/`wraps` declarations to be honored.
    """

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return OutermostCap()

    combined = CombinedCapability([PlainCapA(), DynamicCapability(factory)])
    # At construction, the unresolved wrapper has no ordering of its own.
    assert _cap_names(combined) == ['PlainCapA', 'DynamicCapability']

    ctx = _build_run_context()
    ctx.agent = Agent(TestModel())
    run_capability = await combined.for_run(ctx)
    assert isinstance(run_capability, CombinedCapability)
    assert _cap_names(run_capability) == ['ResolvedDynamicCapability', 'PlainCapA']
    assert isinstance(run_capability.capabilities[0], ResolvedDynamicCapability)
    assert isinstance(run_capability.capabilities[0].wrapped, OutermostCap)


async def test_runtime_capability_with_mixed_position_root():
    """Per-run capabilities can be added to an agent whose root mixes outermost and innermost.

    `Agent.iter()` builds the effective capability by merging per-run capabilities into the
    agent's `_root_capability`. If `_root_capability` is a `CombinedCapability` whose leaves
    span tiers (e.g. an outermost-tier cap and an innermost-tier cap), wrapping it in another
    `CombinedCapability` used to trigger "Conflicting positions in nested CombinedCapability"
    because the outer sort tried to compute a single effective ordering for the inner group.
    The fix splats the root container so each leaf participates as a sibling in the outer
    ordering pass.
    """
    agent = Agent(TestModel(), capabilities=[OutermostCap(), InnermostCap()])
    result = await agent.run('hi', capabilities=[Hooks()])
    assert result.output == snapshot('success (no tool calls)')


# --- Hook recovery tests (after_node_run End→node, ErrorMarker in next_node) ---


async def test_after_node_run_end_to_node_override():
    """after_node_run can convert an End result back to a node, continuing execution."""
    from pydantic_ai import ModelRequestNode

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[TextPart('first answer')])
        return ModelResponse(parts=[TextPart('second answer')])

    redirected = False

    @dataclass
    class RedirectOnFirstEnd(AbstractCapability[Any]):
        """Redirects the first End back to a ModelRequestNode to force a second model call."""

        _redirected: bool = field(default=False, init=False)

        async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            nonlocal redirected
            if isinstance(result, End) and not self._redirected:
                self._redirected = True
                redirected = True
                return ModelRequestNode(ModelRequest(parts=[UserPromptPart(content='try again')]))  # pyright: ignore[reportUnknownVariableType]
            return result  # pyright: ignore[reportUnknownVariableType]

    agent = Agent(FunctionModel(llm), capabilities=[RedirectOnFirstEnd()])
    result = await agent.run('hello')

    assert redirected
    assert call_count == 2
    assert result.output == 'second answer'


async def test_next_node_raises_on_error_marker():
    """Accessing next_node after a node error re-raises the original exception."""
    call_count = 0

    def failing_then_ok_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        raise ValueError('model failure')

    agent = Agent(FunctionModel(failing_then_ok_model))
    async with agent.iter('hello') as agent_run:
        node = agent_run.next_node
        node = cast(Any, await agent_run.next(cast(Any, node)))
        with pytest.raises(ValueError, match='model failure'):
            await agent_run.next(node)
        # After an unrecovered error, next_node should re-raise
        with pytest.raises(ValueError, match='model failure'):
            _ = agent_run.next_node


async def test_on_node_run_error_returns_end():
    """on_node_run_error can recover from an exception by returning End, completing the run."""

    def always_fails(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('model exploded')

    @dataclass
    class RecoverWithEnd(AbstractCapability[Any]):
        async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            return End(FinalResult('recovered output'))

    agent = Agent(FunctionModel(always_fails), capabilities=[RecoverWithEnd()])
    result = await agent.run('hello')
    assert result.output == 'recovered output'


async def test_on_node_run_error_returns_node():
    """on_node_run_error can recover by returning a retry node, continuing execution."""
    from pydantic_ai import ModelRequestNode

    call_count = 0

    def fails_then_succeeds(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError('transient failure')
        return ModelResponse(parts=[TextPart('recovered')])

    @dataclass
    class RetryOnError(AbstractCapability[Any]):
        async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            # Retry by returning a new ModelRequestNode with the same request
            return ModelRequestNode(request=node.request)  # pyright: ignore[reportUnknownVariableType]

    agent = Agent(FunctionModel(fails_then_succeeds), capabilities=[RetryOnError()])
    result = await agent.run('hello')
    assert call_count == 2
    assert result.output == 'recovered'


async def test_after_node_run_node_to_end():
    """after_node_run can short-circuit a run by converting a continuation node to End."""

    model_call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal model_call_count
        model_call_count += 1
        # Always request a tool call, producing a CallToolsNode (not End)
        return ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args='{}')])

    @dataclass
    class ShortCircuitAfterModelRequest(AbstractCapability[Any]):
        """Short-circuit after the first model request node by converting the continuation to End."""

        async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            from pydantic_ai import ModelRequestNode

            # The ModelRequestNode produces a CallToolsNode (not End); convert it to End.
            if isinstance(node, ModelRequestNode) and not isinstance(result, End):
                return End(FinalResult('short-circuited'))
            return result  # pyright: ignore[reportUnknownVariableType]

    agent = Agent(FunctionModel(model_fn), capabilities=[ShortCircuitAfterModelRequest()])

    @agent.tool_plain
    def my_tool() -> str:
        return 'tool result'  # pragma: no cover

    result = await agent.run('hello')
    assert result.output == 'short-circuited'
    assert model_call_count == 1


# --- resolve_model_id hook tests ---


def _resolve_dummy_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='ok')])


@dataclass
class _StringResolver(AbstractCapability[Any]):
    """Test capability that maps known strings to a fixed FunctionModel."""

    target: FunctionModel

    async def resolve_model_id(self, ctx: ModelResolutionContext[Any], *, model_id: Any) -> Any:
        if model_id == 'magic-model':
            return self.target
        return None


@dataclass
class _PassThroughResolver(AbstractCapability[Any]):
    """Test capability that always defers, recording what it saw."""

    seen: list[Any] = field(default_factory=list[Any])
    seen_deps: list[Any] = field(default_factory=list[Any])

    async def resolve_model_id(self, ctx: ModelResolutionContext[Any], *, model_id: Any) -> Any:
        self.seen.append(model_id)
        self.seen_deps.append(ctx.deps)
        return None


async def test_resolve_model_id_maps_string_to_model() -> None:
    """A capability's resolve_model_id maps a runtime string to a Model instance."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='resolved')
    agent = Agent(name='resolve_test', capabilities=[_StringResolver(target=target)])

    result = await agent.run('hi', model='magic-model')
    assert result.output == 'ok'


async def test_resolve_model_id_returns_none_falls_back_to_infer_model() -> None:
    """When all capabilities defer, _get_model uses the default infer_model path."""
    cap = _PassThroughResolver()
    agent = Agent(name='resolve_pass', capabilities=[cap], defer_model_check=True)

    # 'test' is the special string that infer_model maps to TestModel.
    result = await agent.run('hi', model='test')
    assert result.output is not None
    assert cap.seen == ['test']


async def test_resolve_model_id_returns_none_for_unknown_string() -> None:
    """A resolver that doesn't recognize the string returns None so the next layer can try."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='resolved')
    cap = _StringResolver(target=target)
    resolution_ctx = ModelResolutionContext(agent=cast(Any, None), deps=None)
    assert await cap.resolve_model_id(resolution_ctx, model_id='different-string') is None


async def test_resolve_model_id_first_non_none_wins() -> None:
    """When two capabilities declare resolve_model_id, the first one in the list wins.

    Composition is first-non-None-wins (not each-layer-wraps): only one capability
    can claim a given string. Per-request *wrapping* of a resolved Model lives in
    `before_model_request`, not here.
    """
    first_target = FunctionModel(_resolve_dummy_model_fn, model_name='first')
    second_target = FunctionModel(_resolve_dummy_model_fn, model_name='second')

    first = _StringResolver(target=first_target)
    second = _StringResolver(target=second_target)
    combined = CombinedCapability([first, second])

    agent = Agent(name='resolve_layered', capabilities=[first, second], defer_model_check=True)
    result = await combined.resolve_model_id(ModelResolutionContext(agent=agent, deps=None), model_id='magic-model')
    assert result is first_target


def test_resolve_model_id_skipped_for_model_instance() -> None:
    """The hook is never called when the user passes a Model instance directly."""
    cap = _PassThroughResolver()
    target = FunctionModel(_resolve_dummy_model_fn, model_name='direct')
    agent = Agent(target, name='resolve_skip_instance', capabilities=[cap])

    # No string ever flows through; cap.seen should stay empty.
    assert agent.model is target
    assert cap.seen == []


async def test_resolve_model_id_invoked_on_override() -> None:
    """`agent.override(model=string)` routes the string through resolve_model_id."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='override-resolved')
    cap = _StringResolver(target=target)

    initial_model = FunctionModel(_resolve_dummy_model_fn, model_name='initial')
    agent = Agent(initial_model, name='resolve_override', capabilities=[cap])

    with agent.override(model='magic-model'):
        result = await agent.run('hi')
    assert result.output == 'ok'


async def test_resolve_model_id_invoked_on_agent_default_string() -> None:
    """`Agent(model='string', capabilities=[cap])` routes the default through resolve_model_id at run setup.

    Capabilities with `resolve_model_id` need a shot at the default model string just
    like they do for runtime overrides. The hook is deps-aware and only fires at run
    setup, so the agent keeps the raw string at construction (like `defer_model_check`)
    and resolution happens per run — under different deps, potentially to different models.
    """
    target = FunctionModel(_resolve_dummy_model_fn, model_name='default-resolved')
    cap = _StringResolver(target=target)

    agent = Agent('magic-model', name='resolve_default_string', capabilities=[cap])

    # The default stays a string at construction; the hook can't run without deps.
    assert agent.model == 'magic-model'

    result = await agent.run('hi')
    assert result.output == 'ok'

    # No memoization: the raw string is kept so per-run resolution keeps firing.
    assert agent.model == 'magic-model'


async def test_resolve_model_id_receives_deps() -> None:
    """The hook receives the run's deps on `ctx.deps`, so resolution can be run-dependent."""
    cap = _PassThroughResolver()
    agent = Agent(name='resolve_deps', deps_type=str, capabilities=[cap], defer_model_check=True)

    await agent.run('hi', model='test', deps='user-credential')
    assert cap.seen == ['test']
    assert cap.seen_deps == ['user-credential']


async def test_override_model_string_deferral_considers_override_capabilities() -> None:
    """`override(model=str)`'s defer-vs-eager choice consults the effective root capability.

    Neither the spec capability nor the agent chain implements `resolve_model_id` here, so
    the string resolves eagerly via `infer_model` — checked against the spec-supplied root
    when set in the same call, and against an already-active root override when nested.
    """
    agent = Agent(name='override_deferral_effective_root')

    with agent.override(spec={'capabilities': [{'IncludeToolReturnSchemas': {}}]}, model='test'):
        result = await agent.run('hi')
        assert result.output is not None

    with agent.override(spec={'capabilities': [{'IncludeToolReturnSchemas': {}}]}):
        with agent.override(model='test'):
            result = await agent.run('hi')
            assert result.output is not None


async def test_resolve_model_id_uses_override_root_capability() -> None:
    """A root-capability override (as set by `override(spec=...)`) owns model-string resolution.

    Not a public-API test: no built-in spec-constructible capability implements
    `resolve_model_id` yet, so this drives the `_override_root_capability` contextvar —
    the exact seam `override(spec=...)` sets when a spec replaces the root — directly.
    Pins that resolution honors the effective (replaced) root, and that the resolved
    model doesn't get memoized onto `agent.model` past the override's scope.
    """
    chain_target = FunctionModel(_resolve_dummy_model_fn, model_name='agent-chain')
    override_target = FunctionModel(_resolve_dummy_model_fn, model_name='override-root')

    agent = Agent('magic-model', name='resolve_override_root', capabilities=[_StringResolver(target=chain_target)])

    override_root = CombinedCapability[Any]([_StringResolver(target=override_target)])
    token = agent._override_root_capability.set(Some(override_root))  # pyright: ignore[reportPrivateUsage]
    try:
        resolved = await agent._resolve_model_selection(  # pyright: ignore[reportPrivateUsage]
            agent._pick_raw_model(None),  # pyright: ignore[reportPrivateUsage]
            capability=agent._effective_root_capability(),  # pyright: ignore[reportPrivateUsage]
            deps=None,
        )
        assert resolved is override_target
        # No memoization under an override: the raw string default survives.
        assert agent.model == 'magic-model'
    finally:
        agent._override_root_capability.reset(token)  # pyright: ignore[reportPrivateUsage]

    resolved = await agent._resolve_model_selection(  # pyright: ignore[reportPrivateUsage]
        agent._pick_raw_model(None),  # pyright: ignore[reportPrivateUsage]
        capability=agent._effective_root_capability(),  # pyright: ignore[reportPrivateUsage]
        deps=None,
    )
    assert resolved is chain_target


async def test_resolve_model_id_alias_unusable_outside_run() -> None:
    """A capability-owned alias default resolves during runs, and says so clearly outside one.

    Sync entry points like `set_mcp_sampling_model` can't invoke the async, deps-aware
    hook, so an alias only a capability can resolve raises an explanation asking for a
    concrete model rather than attempting deps-blind resolution.
    """
    target = FunctionModel(_resolve_dummy_model_fn, model_name='aliased')

    def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> FunctionModel | None:
        return target if model_id == 'alias' else None

    agent = Agent('alias', name='alias_outside_run', capabilities=[ResolveModelId(resolver)])
    with pytest.raises(UserError, match='requires run dependencies and cannot be used for MCP sampling'):
        agent.set_mcp_sampling_model()

    # Inside a run, the alias resolves through the hook as usual.
    result = await agent.run('hi')
    assert result.output == 'ok'


# --- ResolveModelId capability tests ---


async def test_resolve_model_id_capability_sync_resolver() -> None:
    """`ResolveModelId` wraps a sync resolver function that maps strings to models using deps."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='sync-resolved')
    seen_deps: list[Any] = []

    def resolver(ctx: ModelResolutionContext[str], model_id: str) -> FunctionModel | None:
        seen_deps.append(ctx.deps)
        return target if model_id == 'alias' else None

    agent = Agent('alias', name='resolve_cap_sync', deps_type=str, capabilities=[ResolveModelId(resolver)])
    result = await agent.run('hi', deps='credential')
    assert result.output == 'ok'
    assert seen_deps == ['credential']


async def test_resolve_model_id_capability_async_resolver() -> None:
    """`ResolveModelId` also accepts an async resolver function."""
    target = FunctionModel(_resolve_dummy_model_fn, model_name='async-resolved')

    async def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> FunctionModel | None:
        return target if model_id == 'alias' else None

    agent = Agent(name='resolve_cap_async', capabilities=[ResolveModelId(resolver)])
    result = await agent.run('hi', model='alias')
    assert result.output == 'ok'


async def test_resolve_model_id_capability_defers_to_infer_model() -> None:
    """A `ResolveModelId` resolver returning None falls back to the default `infer_model` flow."""

    def resolver(ctx: ModelResolutionContext[Any], model_id: str) -> None:
        return None

    agent = Agent(name='resolve_cap_defer', capabilities=[ResolveModelId(resolver)])
    # 'test' is the special string that infer_model maps to TestModel.
    result = await agent.run('hi', model='test')
    assert result.output is not None





# --- Agent-bound capabilities ---


@dataclass
class _AgentBoundCapability(AbstractCapability[Any]):
    bound_name: str | None = None
    for_agent_calls: int = 0

    def for_agent(self, agent: AbstractAgent[Any, Any]) -> _AgentBoundCapability:
        return replace(self, bound_name=agent.name, for_agent_calls=self.for_agent_calls + 1)

    def get_instructions(self) -> str:
        return f'Bound to {self.bound_name}.'


async def test_for_agent_returns_bound_copy() -> None:
    capability = _AgentBoundCapability()

    first = Agent(TestModel(), name='first', capabilities=[capability])
    second = Agent(TestModel(), name='second', capabilities=[capability])

    first_bound = next(cap for cap in first.root_capability.capabilities if isinstance(cap, _AgentBoundCapability))
    second_bound = next(cap for cap in second.root_capability.capabilities if isinstance(cap, _AgentBoundCapability))
    assert capability.bound_name is None
    assert first_bound is not capability
    assert second_bound is not capability
    assert first_bound.bound_name == 'first'
    assert second_bound.bound_name == 'second'
    assert first_bound.for_agent_calls == second_bound.for_agent_calls == 1

    result = await first.run('hello')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to first.'


def test_wrapper_for_agent_replaces_wrapped_capability() -> None:
    capability = _AgentBoundCapability()
    wrapper = WrapperCapability(capability)

    agent = Agent(TestModel(), name='wrapped', capabilities=[wrapper])

    bound_wrapper = next(cap for cap in agent.root_capability.capabilities if isinstance(cap, WrapperCapability))
    assert bound_wrapper is not wrapper
    assert cast(_AgentBoundCapability, bound_wrapper.wrapped).bound_name == 'wrapped'


def test_wrapper_for_agent_preserves_identity_without_replacement() -> None:
    """Identity preservation is an internal binding contract that a request cassette cannot observe."""
    wrapper = WrapperCapability[Any](AbstractCapability[Any]())
    agent = Agent(TestModel())

    assert wrapper.for_agent(agent) is wrapper


async def test_for_agent_composes_with_model_selection_and_resolution() -> None:
    selected_model = TestModel(custom_output_text='selected')

    @dataclass
    class BoundModelCapability(AbstractCapability[Any]):
        model_id: str | None = None

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> BoundModelCapability:
            return replace(self, model_id=f'bound:{agent.name}')

        def get_model(self) -> str | None:
            return self.model_id

        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            assert ctx.agent.name == 'selector'
            return selected_model if model_id == self.model_id else None

    agent = Agent(name='selector', capabilities=[BoundModelCapability()])
    result = await agent.run('hello')
    assert result.output == 'selected'


async def test_for_agent_can_introduce_model_id_resolution() -> None:
    selected_model = TestModel(custom_output_text='selected')

    @dataclass
    class BoundResolver(AbstractCapability[Any]):
        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            return selected_model if model_id == 'custom-model' else None

    @dataclass
    class BindingCapability(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            assert agent.model == 'custom-model'
            return BoundResolver()

    agent = Agent('custom-model', capabilities=[BindingCapability()])
    assert (await agent.run('hello')).output == 'selected'


async def test_for_agent_can_introduce_resolution_for_known_model_id() -> None:
    selected_model = TestModel(custom_output_text='selected')

    @dataclass
    class BoundResolver(AbstractCapability[Any]):
        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            return selected_model if model_id == 'test' else None

    @dataclass
    class BindingCapability(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            assert agent.model == 'test'
            return BoundResolver()

    agent = Agent('test', capabilities=[BindingCapability()])
    assert agent.model == 'test'
    assert (await agent.run('hello')).output == 'selected'


def test_for_agent_without_resolver_preserves_unknown_model_error() -> None:
    with pytest.raises(UserError, match='Unknown model: custom-model'):
        Agent('custom-model', capabilities=[_AgentBoundCapability()])


async def test_for_agent_binds_per_run_capabilities() -> None:
    capability = _AgentBoundCapability()
    agent = Agent(TestModel(), name='runner')

    result = await agent.run('hello', capabilities=[capability])

    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to runner.'
    assert capability.for_agent_calls == 0


async def test_per_run_binding_can_supply_bootstrap_model_and_resolver() -> None:
    """Run binding precedes bootstrap selection and resolution, an ordering contract cassettes cannot isolate."""
    selected_model = TestModel(custom_output_text='run-bound')

    @dataclass
    class BoundRunModel(AbstractCapability[Any]):
        def get_model(self) -> str:
            return 'run-bound-id'

        async def resolve_model_id(
            self,
            ctx: ModelResolutionContext[Any],
            *,
            model_id: KnownModelName | str,
        ) -> Model | None:
            return selected_model if model_id == 'run-bound-id' else None

    @dataclass
    class BindAtRun(AbstractCapability[Any]):
        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            return BoundRunModel()

    agent = Agent(None)
    result = await agent.run('hello', capabilities=[BindAtRun()])

    assert result.output == 'run-bound'


# --- Dynamic capabilities ---


@dataclass
class _RecordingCapability(AbstractCapability[Any]):
    """Test capability that records every hook firing and contributes instructions."""

    label: str
    fired: list[str] = field(default_factory=list[str])

    def get_instructions(self) -> str:
        return f'Label is {self.label}.'

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.fired.append(f'{self.label}:before_run')

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.fired.append(f'{self.label}:before_model_request')
        return request_context


async def test_dynamic_capability_factory_called_with_run_context() -> None:
    """The factory receives the run's `RunContext` (with deps) once per run."""
    seen: list[Any] = []

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any] | None:
        seen.append(ctx.deps)
        return _RecordingCapability(label=ctx.deps)

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])
    await agent.run('hi', deps='admin')
    await agent.run('hi', deps='guest')
    assert seen == ['admin', 'guest']


async def test_dynamic_capability_factory_result_is_bound_to_agent() -> None:
    """A factory's standalone result is agent-bound before its run binding; a cassette cannot observe hook order."""

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return _AgentBoundCapability()

    agent = Agent(TestModel(), name='dynamic', capabilities=[factory])
    result = await agent.run('hi')

    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to dynamic.'


async def test_for_run_result_is_not_bound_again() -> None:
    """A specialized run-bound result skips agent binding; a provider cassette cannot observe that distinction."""

    @dataclass
    class BuildsRunCapability(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
            return _AgentBoundCapability()

    agent = Agent(TestModel(), name='static', capabilities=[BuildsRunCapability()])
    result = await agent.run('hi')

    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Bound to None.'


async def test_dynamic_capability_async_factory() -> None:
    """Async factories are awaited."""
    calls = 0

    async def factory(ctx: RunContext) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='async')

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert calls == 1


async def test_dynamic_capability_returning_none_contributes_nothing() -> None:
    """A factory returning None is a no-op for the run."""

    def factory(ctx: RunContext) -> AbstractCapability[Any] | None:
        return None

    agent = Agent(TestModel(), capabilities=[factory])
    result = await agent.run('hi')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions is None

    dynamic = DynamicCapability(factory)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    assert await dynamic.for_run(ctx) is dynamic

    # Direct toolset-factory call (unit-style): the standalone fallback — a context without the
    # run's capability registry, as inside a durable unit — re-resolves the factory, and an async
    # factory returning `None` still contributes nothing.
    async def async_none_factory(ctx: RunContext[Any]) -> AbstractCapability[Any] | None:
        return None

    async_dynamic = DynamicCapability(async_none_factory)
    resolved = async_dynamic.get_toolset().toolset_func(ctx)
    assert inspect.isawaitable(resolved)
    assert await resolved is None


def test_dynamic_capability_toolset_is_cached_and_inherits_id() -> None:
    dynamic = DynamicCapability(lambda ctx: None, id='x')
    toolset = dynamic.get_toolset()

    assert toolset.id == 'x'
    assert dynamic.get_toolset() is toolset


async def test_dynamic_capability_contributes_instructions_per_run() -> None:
    """Resolved capability's instructions flow through to the model request."""

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any] | None:
        if ctx.deps == 'admin':
            return _RecordingCapability(label='admin')
        return None

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])

    admin_result = await agent.run('hi', deps='admin')
    admin_request = next(m for m in admin_result.all_messages() if isinstance(m, ModelRequest))
    assert admin_request.instructions == 'Label is admin.'

    guest_result = await agent.run('hi', deps='guest')
    guest_request = next(m for m in guest_result.all_messages() if isinstance(m, ModelRequest))
    assert guest_request.instructions is None


async def test_dynamic_capability_contributes_toolset() -> None:
    """The resolved toolset is exposed once while instructions and settings still apply."""
    calls = 0
    toolset = FunctionToolset()

    @toolset.tool_plain
    def special() -> str:
        return 'used'

    @dataclass
    class ToolCap(AbstractCapability):
        def get_instructions(self) -> str:
            return 'Use the special tool.'

        def get_model_settings(self) -> _ModelSettings:
            return _ModelSettings(temperature=0.25)

        def get_toolset(self) -> AbstractToolset[Any]:
            return toolset

    def factory(ctx: RunContext[bool]) -> AbstractCapability[Any] | None:
        nonlocal calls
        calls += 1
        return ToolCap() if ctx.deps else None

    seen_tools: list[str] = []
    seen_temperatures: list[float | None] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append(','.join(sorted(t.name for t in info.function_tools)))
        seen_temperatures.append(info.model_settings.get('temperature') if info.model_settings else None)
        # On the first request call the tool if it's available; on the follow-up
        # request after the tool return, finish.
        already_called = any(
            isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts
        )
        if not already_called and any(t.name == 'special' for t in info.function_tools):
            return ModelResponse(parts=[ToolCallPart('special')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=bool, capabilities=[factory])

    with_tool = await agent.run('hi', deps=True)
    tool_returns = [
        p.content
        for m in with_tool.all_messages()
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart)
    ]
    assert tool_returns == ['used']
    first_request = next(m for m in with_tool.all_messages() if isinstance(m, ModelRequest))
    assert first_request.instructions == 'Use the special tool.'

    await agent.run('hi', deps=False)
    assert seen_tools == ['special', 'special', '']
    assert seen_temperatures == [0.25, 0.25, None]
    assert calls == 2


async def test_dynamic_capability_contributes_toolset_function() -> None:
    """A resolved capability may contribute a toolset *function*; it's evaluated with the run context."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def func_tool() -> str:
        # The tool listing is what's asserted.
        return 'from func'  # pragma: no cover

    @dataclass
    class AsyncToolFuncCap(AbstractCapability):
        def get_toolset(self):
            async def toolset_func(ctx: RunContext[Any]) -> AbstractToolset[Any] | None:
                return toolset if ctx.deps else None

            return toolset_func

    @dataclass
    class SyncToolFuncCap(AbstractCapability):
        def get_toolset(self):
            def toolset_func(ctx: RunContext[Any]) -> AbstractToolset[Any] | None:
                return toolset

            return toolset_func

    seen_tools: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append(','.join(sorted(t.name for t in info.function_tools)))
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(respond),
        deps_type=bool,
        capabilities=[DynamicCapability(lambda ctx: AsyncToolFuncCap())],
    )
    await agent.run('hi', deps=True)
    await agent.run('hi', deps=False)

    sync_agent = Agent(
        FunctionModel(respond),
        deps_type=bool,
        capabilities=[DynamicCapability(lambda ctx: SyncToolFuncCap())],
    )
    await sync_agent.run('hi', deps=True)
    assert seen_tools == ['func_tool', '', 'func_tool']


async def test_dynamic_capability_instructions_and_tools_share_resolved_state() -> None:
    """Instructions and tools observe the *same* resolved capability instance per run.

    The factory allocates fresh state on every call, so if the contributed toolset were
    resolved through a second factory invocation, the tool would see different state than
    the instructions.
    """
    resolution_count = 0

    @dataclass
    class StatefulCap(AbstractCapability):
        token: str = ''

        def get_instructions(self) -> str:
            return f'Token is {self.token}.'

        def get_toolset(self):
            toolset = FunctionToolset()

            @toolset.tool_plain
            def read_token() -> str:
                return self.token

            return toolset

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        nonlocal resolution_count
        resolution_count += 1
        return StatefulCap(token=f'run-{resolution_count}')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not tool_returns:
            return ModelResponse(parts=[ToolCallPart(tool_name='read_token', args={}, tool_call_id='read')])
        return make_text_response(str(tool_returns[0].content))

    agent = Agent(FunctionModel(respond), capabilities=[factory])
    result = await agent.run('hi')
    first_request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert first_request.instructions == 'Token is run-1.'
    assert result.output == 'run-1'
    assert resolution_count == 1


async def test_dynamic_capability_returning_deferred_capability() -> None:
    """A factory-returned deferred capability keeps its tools hidden until `load_capability`."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def hidden_tool() -> str:
        return 'now visible'

    def factory(ctx: RunContext[Any]) -> AbstractCapability[Any]:
        return Capability(
            id='skills',
            description='Deferred skills.',
            toolsets=[toolset],
            defer_loading=True,
        )

    seen_defer_flags: list[bool] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if hidden_def := next((t for t in info.function_tools if t.name == 'hidden_tool'), None):
            # Authored deferral remains stable after the capability is loaded.
            seen_defer_flags.append(hidden_def.defer_loading)
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'skills'}, tool_call_id='load')]
            )
        if not any(part.tool_name == 'hidden_tool' for part in tool_returns):
            return ModelResponse(parts=[ToolCallPart(tool_name='hidden_tool', args={}, tool_call_id='use')])
        return make_text_response('done')

    agent = Agent(FunctionModel(respond), capabilities=[factory])
    result = await agent.run('hi')
    assert result.output == 'done'
    assert seen_defer_flags == [True, True]


async def test_dynamic_capability_hooks_fire() -> None:
    """Hooks contributed by the resolved capability fire during the run."""
    cap = _RecordingCapability(label='dyn')

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return cap

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert 'dyn:before_run' in cap.fired
    assert 'dyn:before_model_request' in cap.fired


async def test_dynamic_capability_factory_called_once_per_run_not_per_step() -> None:
    """The factory is called once at for_run, not on every model request."""
    calls = 0

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='once')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Two-step run: first a tool call, then a final text response.
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('echo', {'text': 'hi'})])
        return ModelResponse(parts=[TextPart('done')])

    toolset = FunctionToolset()

    @toolset.tool_plain
    def echo(text: str) -> str:
        return text

    agent = Agent(FunctionModel(respond), toolsets=[toolset], capabilities=[factory])
    await agent.run('hi')
    assert calls == 1


async def test_dynamic_capability_returning_combined() -> None:
    """A factory may return a CombinedCapability; all child contributions flow through."""
    fired: list[str] = []

    @dataclass
    class A(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('A')

    @dataclass
    class B(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('B')

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return CombinedCapability([A(), B()])

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert fired == ['A', 'B']


async def test_dynamic_deferred_capability_returned_from_custom_init_requires_stable_id() -> None:
    """Deferred capability validation also catches custom init objects returned at run time."""

    @dataclass(init=False)
    class CustomInitDeferredCap(AbstractCapability):
        def __init__(self) -> None:
            self.defer_loading = True

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return CustomInitDeferredCap()

    agent = Agent(FunctionModel(lambda _messages, _info: make_text_response('done')), capabilities=[factory])

    with pytest.raises(UserError, match='stable explicit `id` values'):
        await agent.run('hi')


async def test_dynamic_deferred_capability_uses_resolved_capability_for_loaded_tools() -> None:
    """A loaded dynamic deferred capability exposes tools from the resolved capability."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed'

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return Capability[object](
            id='dynamic-refunds',
            description='Refund policy tools.',
            toolsets=[toolset],
            defer_loading=True,
        )

    seen_tool_state: list[list[tuple[str, bool]]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_state.append([(t.name, bool(t.defer_loading)) for t in info.function_tools])
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(
            isinstance(part, LoadCapabilityReturnPart)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'dynamic-refunds'},
                        tool_call_id='load-dynamic-refunds',
                    )
                ]
            )

        if not any(part.tool_name == 'lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )

        refund_result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund_policy')
        return make_text_response(f'done: {refund_result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[factory])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done: order-123: refund allowed'
    assert seen_tool_state == snapshot(
        [
            [('load_capability', False)],
            [('load_capability', False), ('lookup_refund_policy', True)],
            [('load_capability', False), ('lookup_refund_policy', True)],
        ]
    )


async def test_dynamic_capability_in_run_call() -> None:
    """`agent.run(capabilities=[factory])` accepts callables as well."""
    calls = 0

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='run-time')

    agent = Agent(TestModel())
    result = await agent.run('hi', capabilities=[factory])
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Label is run-time.'
    assert calls == 1


async def test_dynamic_capability_composes_with_static() -> None:
    """Static and dynamic capabilities both contribute, in order."""
    fired: list[str] = []

    @dataclass
    class Static(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('static')

    @dataclass
    class Dynamic(AbstractCapability):
        async def before_run(self, ctx: RunContext) -> None:
            fired.append('dynamic')

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return Dynamic()

    agent = Agent(TestModel(), capabilities=[Static(), factory])
    await agent.run('hi')
    assert fired == ['static', 'dynamic']


async def test_dynamic_capability_per_run_isolation() -> None:
    """Concurrent runs see independent factory calls and resolved capabilities."""
    seen_deps: list[str] = []

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any]:
        seen_deps.append(ctx.deps)
        return _RecordingCapability(label=ctx.deps)

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])
    results = await asyncio.gather(*(agent.run('hi', deps=f'user-{i}') for i in range(5)))

    assert sorted(seen_deps) == ['user-0', 'user-1', 'user-2', 'user-3', 'user-4']
    for i, result in enumerate(results):
        request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
        assert request.instructions == f'Label is user-{i}.'


async def test_dynamic_capability_wraps_func_in_constructor() -> None:
    """Constructor wraps a bare function into a `DynamicCapability`, and the factory runs at run time."""

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return _RecordingCapability(label='x')

    agent = Agent(TestModel(), capabilities=[factory])

    result = await agent.run('hi')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Label is x.'


def test_dynamic_capability_rejects_wrapper_fields() -> None:
    """`defer_loading` on the wrapper would otherwise be silently ignored — reject at construction."""

    def factory(ctx: RunContext) -> AbstractCapability[Any]:
        return _RecordingCapability(label='x')  # pragma: no cover

    with pytest.raises(UserError, match='not supported on `DynamicCapability`'):
        DynamicCapability(factory, defer_loading=True)


# endregion


async def test_combined_capability_subclass_custom_init_for_run() -> None:
    """`CombinedCapability` subclasses with a custom `__init__` don't crash in `for_run` when a child returns a fresh instance.

    Regression test for #6674: `dataclasses.replace` reconstructed through the subclass
    `__init__`, which does not accept the `capabilities` kwarg.
    """

    @dataclass
    class PerRunLeaf(AbstractCapability[Any]):
        n: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return PerRunLeaf(n=self.n + 1)

        def get_instructions(self) -> str:
            return f'leaf {self.n}'

    class CombinedSubclass(CombinedCapability[Any]):
        """Bundle a leaf behind a friendly constructor without exposing `capabilities`."""

        def __init__(self, *, size: int = 3) -> None:
            self.post_init_calls = 0
            super().__init__(capabilities=[PerRunLeaf(n=size)])

        def __post_init__(self) -> None:
            self.post_init_calls += 1
            super().__post_init__()

    combined = CombinedSubclass(size=5)
    ctx = _build_run_context()

    result = await combined.for_run(ctx)

    assert isinstance(result, CombinedSubclass)
    assert result is not combined
    assert result.post_init_calls == 1
    leaf = result.capabilities[0]
    assert isinstance(leaf, PerRunLeaf)
    assert leaf.n == 6
    # Exercising `get_instructions` also covers the leaf's instruction emission.
    assert leaf.get_instructions() == 'leaf 6'


def test_combined_capability_subclass_custom_init_for_agent() -> None:
    """`CombinedCapability` subclasses with a custom `__init__` don't crash in `for_agent` when a child returns a fresh instance.

    Regression test for #6674.
    """

    @dataclass
    class BindingLeaf(AbstractCapability[Any]):
        bound: bool = False

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            return replace(self, bound=True)

    class CombinedSubclass(CombinedCapability[Any]):
        def __init__(self) -> None:
            super().__init__(capabilities=[BindingLeaf()])

    combined = CombinedSubclass()
    agent = Agent('test')

    bound = combined.for_agent(agent)

    assert isinstance(bound, CombinedSubclass)
    assert bound is not combined
    bound_leaf = bound.capabilities[0]
    assert isinstance(bound_leaf, BindingLeaf)
    assert bound_leaf.bound is True


async def test_wrapper_capability_subclass_custom_init_rebinds_wrapped() -> None:
    """`WrapperCapability` subclasses with a custom `__init__` survive both binding paths.

    Same `dataclasses.replace`-through-subclass-`__init__` defect as #6674, on the sibling
    container: `WrapperCapability` rebuilt itself with `replace(self, wrapped=...)`, which the
    subclass constructor can't accept. Driven through `Agent` because — unlike
    `CombinedCapability`, whose `__post_init__` splats a nested subclass away — a wrapper
    reaches both `for_agent` (agent construction) and `for_run` (per-run) intact.
    """

    @dataclass
    class PerRunLeaf(AbstractCapability[Any]):
        n: int = 0
        bound: bool = False

        def for_agent(self, agent: AbstractAgent[Any, Any]) -> AbstractCapability[Any]:
            return replace(self, bound=True)

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return replace(self, n=self.n + 1)

        def get_instructions(self) -> str:
            return f'leaf {self.n}'

    class WrapperSubclass(WrapperCapability[Any]):
        """Bundle a leaf behind a friendly constructor without exposing `wrapped`."""

        def __init__(self, *, size: int = 3) -> None:
            self.post_init_calls = 0
            super().__init__(wrapped=PerRunLeaf(n=size))

        def __post_init__(self) -> None:
            self.post_init_calls += 1
            super().__post_init__()

    agent = Agent('test', capabilities=[WrapperSubclass(size=5)])
    result = await agent.run('hi')

    # `for_agent` bound the leaf at construction, then `for_run` incremented it for this run,
    # and the wrapper delegated the resulting instructions through both rebuilds.
    request = result.all_messages()[0]
    assert isinstance(request, ModelRequest)
    assert request.instructions == 'leaf 6'
    wrapper = next(cap for cap in agent.root_capability.capabilities if isinstance(cap, WrapperSubclass))
    assert wrapper.post_init_calls == 1


async def test_wrapper_capability_subclass_custom_init_preserves_type_and_id() -> None:
    """Rebuilding a `WrapperCapability` keeps the subclass type and re-resolves the adopted `id`.

    Pins transparent-wrapper identity re-resolution: a wrapper without an explicit `id` adopts
    the wrapped capability's `id`, which is only known after `for_run` has produced the new
    wrapped instance.
    """

    @dataclass
    class IdentifiedLeaf(AbstractCapability[Any]):
        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return IdentifiedLeaf(id='resolved-at-run-time')

    class WrapperSubclass(WrapperCapability[Any]):
        def __init__(self, *, size: int = 3) -> None:
            super().__init__(wrapped=IdentifiedLeaf())
            self.size = size

    wrapper = WrapperSubclass(size=5)
    assert wrapper.id is None

    rebuilt = await wrapper.for_run(_build_run_context())

    assert isinstance(rebuilt, WrapperSubclass)
    assert rebuilt is not wrapper
    assert rebuilt.size == 5, 'subclass-only attributes must survive the rebuild'
    assert rebuilt.id == 'resolved-at-run-time'
    assert wrapper.id is None, 'the original must not be mutated'


async def test_wrapper_capability_subclass_derived_state_contract() -> None:
    """Pins the documented rebind contract for subclass state.

    A rebind shallow-copies the wrapper without re-running `__init__`/`__post_init__`, so
    values derived from `wrapped` must be computed on access to stay fresh — an eager cache
    made at construction is carried over verbatim and reflects the pre-rebind wrapped.
    """

    @dataclass
    class PerRunLeaf(AbstractCapability[Any]):
        n: int = 0

        async def for_run(self, ctx: RunContext) -> AbstractCapability:
            return PerRunLeaf(n=self.n + 1)

    class SummarizingWrapper(WrapperCapability[Any]):
        def __init__(self, leaf: PerRunLeaf) -> None:
            super().__init__(wrapped=leaf)
            self.cached_summary = self.summary

        @property
        def summary(self) -> str:
            assert isinstance(self.wrapped, PerRunLeaf)
            return f'wrapping leaf {self.wrapped.n}'

    wrapper = SummarizingWrapper(PerRunLeaf(n=1))
    rebound = await wrapper.for_run(_build_run_context())

    assert isinstance(rebound, SummarizingWrapper)
    assert rebound.summary == 'wrapping leaf 2', 'computed-on-access state re-derives from the new wrapped'
    assert rebound.cached_summary == 'wrapping leaf 1', 'eagerly cached state is carried over verbatim'
    assert wrapper.summary == 'wrapping leaf 1', 'the original must not be mutated'


async def test_tool_return_cannot_reveal_capability_owned_tools_without_loading() -> None:
    """A bare-name reveal of a capability tool would skip the capability's hooks and instructions.

    `load_capability` activates the whole bundle; `ToolReturn.tools` naming a capability-owned tool
    while its capability is unloaded is rejected so the tool can never become callable with its
    capability's `before_tool_validate`/`before_tool_execute` hooks and instructions inactive.
    """
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain(name='capability_tool')
    def capability_tool() -> str:  # pragma: no cover
        return 'refund'

    refunds = Capability[object](id='refunds', toolsets=[refunds_toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if not list(iter_message_parts(messages, ModelRequest, ToolReturnPart)):
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_it', args={}, tool_call_id='reveal')])
        return make_text_response('done')  # pragma: no cover - the run raises before a second model call

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])

    @agent.tool_plain
    def reveal_it() -> ToolReturn[str]:
        return ToolReturn(return_value='revealed', tools=['capability_tool'])

    with pytest.raises(UserError, match=r"belongs to capability 'refunds', which must be loaded"):
        await agent.run('Reveal the capability tool directly.')


async def test_tool_return_can_reveal_capability_owned_tools_once_loaded() -> None:
    """After `load_capability`, naming a capability tool in `ToolReturn.tools` is a legal no-op-ish reveal."""
    refunds_toolset = FunctionToolset()

    @refunds_toolset.tool_plain(name='capability_tool')
    def capability_tool() -> str:  # pragma: no cover
        return 'refund'

    refunds = Capability[object](id='refunds', toolsets=[refunds_toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='l1')]
            )
        if not any(part.tool_name == 'reveal_it' for part in returns):
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_it', args={}, tool_call_id='r1')])
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])

    @agent.tool_plain
    def reveal_it() -> ToolReturn[str]:
        return ToolReturn(return_value='revealed', tools=['capability_tool'])

    result = await agent.run('Load, then reveal by name.')
    assert result.output == 'done'
