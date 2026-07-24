from __future__ import annotations

import asyncio
import contextvars
import inspect
import re
import threading
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from types import NoneType
from typing import Any, cast
from uuid import UUID

import anyio
import pytest
from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel, TypeAdapter, ValidationError

from pydantic_ai import _agent_graph
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai._run_context import RunContext
from pydantic_ai._spec import CapabilitySpec, NamedSpec
from pydantic_ai._tool_search import ToolSearchCallPart, ToolSearchReturnPart
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
    HandleDeferredToolCalls,
    ImageGeneration,
    IncludeToolReturnSchemas,
    Instrumentation,
    NativeTool,
    PrefixTools,
    PrepareTools,
    ProcessEventStream,
    ProcessHistory,
    RaiseContentFilterError,
    ReinjectSystemPrompt,
    ResolveModelId,
    SelectModel,
    SetToolMetadata,
    Thinking,
    ThreadExecutor,
    ToolSearch,
    Toolset,
    WebFetch,
    WebSearch,
    WrapperCapability,
    XSearch,
)
from pydantic_ai.capabilities._dynamic import ResolvedDynamicCapability
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks, HookTimeoutError
from pydantic_ai.capabilities.native_tool import NativeTool as NativeToolCap
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    ToolFailed,
    UndrainedPendingMessagesError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryImage,
    EnqueuedMessagesEvent,
    FilePart,
    ImageUrl,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
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
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    CodeExecutionTool,
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
    XSearchTool,
)
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.output import NativeOutput, OutputContext, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.result import AgentStream
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetFunc
from pydantic_ai.toolsets._capability_owned import resolve_capability_id
from pydantic_ai.toolsets._deferred_capability_loader import (
    LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE,
    LOAD_CAPABILITY_TOOL_NAME,
)
from pydantic_ai.toolsets._tool_search import _SEARCH_TOOLS_NAME  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInstance, IsStr, iter_message_parts, message, remove_schema_descriptions

pytestmark = [
    pytest.mark.anyio,
]


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
                        'anthropic:claude-sonnet-4-5',
                        'anthropic:claude-sonnet-4-5-20250929',
                        'anthropic:claude-sonnet-4-6',
                        'anthropic:claude-sonnet-5',
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
                        'gateway/google-cloud:gemini-3-pro-image-preview',
                        'gateway/google-cloud:gemini-3.1-flash-image-preview',
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
                        'gateway/google:gemini-3-pro-image-preview',
                        'gateway/google:gemini-3.1-flash-image-preview',
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
                        'gateway/openai:computer-use-preview',
                        'gateway/openai:computer-use-preview-2025-03-11',
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
                        'gateway/openai:gpt-5-chat-latest',
                        'gateway/openai:gpt-5-codex',
                        'gateway/openai:gpt-5-mini',
                        'gateway/openai:gpt-5-mini-2025-08-07',
                        'gateway/openai:gpt-5-nano',
                        'gateway/openai:gpt-5-nano-2025-08-07',
                        'gateway/openai:gpt-5-pro',
                        'gateway/openai:gpt-5-pro-2025-10-06',
                        'gateway/openai:gpt-5.1',
                        'gateway/openai:gpt-5.1-2025-11-13',
                        'gateway/openai:gpt-5.1-chat-latest',
                        'gateway/openai:gpt-5.1-codex',
                        'gateway/openai:gpt-5.1-codex-max',
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
                        'google-cloud:gemini-3-pro-image-preview',
                        'google-cloud:gemini-3-pro-preview',
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
                        'google:gemini-3-pro-image-preview',
                        'google:gemini-3-pro-preview',
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
        'The following capabilities are deferred and can be loaded using the `load_capability` tool:\n'
        '- account-security: Use for suspicious logins, account takeover, or session revocation.\n'
        '- refunds'
    )


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
        'The following capabilities are deferred and can be loaded using the `load_capability` tool:\n'
        '- dynamic-description: Use for billing questions.'
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
        'The following capabilities are deferred and can be loaded using the `load_capability` tool:\n'
        '- account: Account-specific guidance.'
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
The following capabilities are deferred and can be loaded using the `load_capability` tool:
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
The following capabilities are deferred and can be loaded using the `load_capability` tool:
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
The following capabilities are deferred and can be loaded using the `load_capability` tool:
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
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse

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


async def test_deferred_capability_loads_instructions_and_tools_e2e() -> None:
    """A deferred capability starts as a catalog entry and becomes usable after `load_capability`."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    def add_account_context(ctx: RunContext) -> str:
        return f'Load-time account context for run step {ctx.run_step}.'

    def empty_instruction(ctx: RunContext) -> None:
        return None

    always_on = Capability[object](
        id='always-on',
        description='Visible billing guidance.',
        instructions='Visible billing instructions.',
    )
    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        instructions=[
            'Use the refund policy before answering refund questions.',
            add_account_context,
            empty_instruction,
        ],
        toolsets=[toolset],
        defer_loading=True,
    )

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
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
        return make_text_response(f'final: {refund_result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[always_on, refunds])

    result = await agent.run('Can I get a refund?')

    assert result.output == snapshot('final: order-123: refund allowed for 30 days')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Can I get a refund?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool:
- refunds: Refund policy tools.""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    LoadCapabilityCallPart(
                        tool_name='load_capability',
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ],
                usage=RequestUsage(input_tokens=55, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    LoadCapabilityReturnPart(
                        tool_name='load_capability',
                        content={
                            'instructions': 'Use the refund policy before answering refund questions.\n\n'
                            'Load-time account context for run step 1.',
                        },
                        tool_call_id='load-refunds',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool:
- refunds: Refund policy tools.""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            # Synthesized by `ToolSearch.before_model_request` after the capability load.
            ModelResponse(
                parts=[
                    ToolSearchCallPart(
                        args={'queries': ['refunds']},
                        tool_call_id='auto_load_0f10f8b659c3c105',
                    )
                ],
                usage=RequestUsage(),
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'lookup_refund_policy'}]},
                        tool_call_id='auto_load_0f10f8b659c3c105',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy', args={'order_id': 'order-123'}, tool_call_id='lookup-refund'
                    )
                ],
                usage=RequestUsage(input_tokens=79, output_tokens=16),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='lookup_refund_policy',
                        content='order-123: refund allowed for 30 days',
                        tool_call_id='lookup-refund',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool:
- refunds: Refund policy tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final: order-123: refund allowed for 30 days')],
                usage=RequestUsage(input_tokens=85, output_tokens=23),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deferred_capability_tool_registered_after_construction_defers_until_load() -> None:
    """A tool registered via `@cap.tool` *after* construction defers like a constructor tool: hidden until load.

    Deferred tools stay in the toolset tagged `defer_loading=True` (the wire-level filter in
    `Model.prepare_request` is what hides them from a real provider), so the regression signal is the
    flag flipping `True` -> `False` once the capability loads, not the tool's mere presence.
    """
    refunds = Capability[object](id='refunds', description='Refund policy tools.', defer_loading=True)

    # Register on the deferred capability *after* construction (decorator path, not the `tools=` arg).
    @refunds.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    defer_flag_by_phase: dict[str, bool | None] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        loaded = any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns)
        refund_def = next((tool for tool in info.function_tools if tool.name == 'lookup_refund_policy'), None)
        defer_flag_by_phase['after_load' if loaded else 'before_load'] = (
            refund_def.defer_loading if refund_def else None
        )

        if not loaded:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        if not any(part.tool_name == 'lookup_refund_policy' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='lookup_refund_policy', args={'order_id': 'order-1'}, tool_call_id='look')
                ]
            )
        result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund_policy')
        return make_text_response(f'final: {result}')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('Can I get a refund?')

    assert result.output == snapshot('final: order-1: refund allowed for 30 days')
    # Deferred before the capability loads, revealed (and callable) afterward.
    assert defer_flag_by_phase == snapshot({'before_load': True, 'after_load': False})


async def test_deferred_capability_tool_stays_available_across_turns() -> None:
    """A capability-owned tool stays callable across every turn after `load_capability`.

    Regression guard: the `available_tool_names`/`discovered_tool_names` split must keep a
    loaded deferred tool non-deferred on the second (and later) post-load model request,
    not just on the turn immediately following the load.
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

    # Names of non-deferred function tools the model sees on each request.
    available_per_turn: list[set[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_per_turn.append({td.name for td in info.function_tools if not td.defer_loading})

        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))

        # Turn 1: load the capability.
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )

        lookup_calls = [part for part in tool_returns if part.tool_name == 'lookup_refund_policy']

        # Turns 2 and 3: call the loaded tool twice, so we exercise two post-load turns.
        if len(lookup_calls) < 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': f'order-{len(lookup_calls)}'},
                        tool_call_id=f'lookup-{len(lookup_calls)}',
                    )
                ]
            )

        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done'

    # First request: tool is still deferred (not yet loaded).
    assert 'lookup_refund_policy' not in available_per_turn[0]
    # Every request after the load must expose the loaded tool as non-deferred — including
    # the second post-load turn, which is what the regression broke.
    post_load_turns = available_per_turn[1:]
    assert len(post_load_turns) >= 2
    for turn_tools in post_load_turns:
        assert 'lookup_refund_policy' in turn_tools


async def test_run_context_tools_exposes_deferred_definitions_as_name_keyed_dict() -> None:
    """`ctx.tools` is the full name-keyed dict of `ToolDefinition`s, including entries
    that are still deferred (and therefore absent from `ctx.available_tool_names`)."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        return f'{order_id}: refund allowed'

    refunds = Capability[object](id='refunds', toolsets=[toolset], defer_loading=True)

    seen_tools: list[dict[str, ToolDefinition]] = []

    @dataclass
    class CaptureCtxToolsCap(AbstractCapability):
        async def before_model_request(
            self, ctx: RunContext, request_context: ModelRequestContext
        ) -> ModelRequestContext:
            seen_tools.append(ctx.tools)
            return request_context

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds, CaptureCtxToolsCap()])
    await agent.run('hi')

    [tools] = seen_tools
    # The deferred tool is keyed by its own name and carries `defer_loading=True`,
    # even though it's absent from `available_tool_names` until the capability loads.
    assert tools['lookup_refund_policy'].name == 'lookup_refund_policy'
    assert tools['lookup_refund_policy'].defer_loading is True


async def test_deferred_capability_synthetic_tool_search_persists_in_history() -> None:
    """The synthetic tool-search exchange injected after a capability load persists to
    the run's message history, and re-running with that history does not duplicate it."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds',
        description='Refund policy tools.',
        toolsets=[toolset],
        defer_loading=True,
    )

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('Can I get a refund?')

    def synthetic_pairs(messages: list[ModelMessage]) -> list[str]:
        call_ids: list[str] = []
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolSearchCallPart) and part.tool_call_id.startswith('auto_load_'):
                    call_ids.append(part.tool_call_id)
        return call_ids

    messages = result.all_messages()
    call_ids = synthetic_pairs(messages)
    # Exactly one synthetic call part, and its matching return part is present.
    assert len(call_ids) == 1
    return_ids = [
        part.tool_call_id
        for message in messages
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart) and part.tool_call_id == call_ids[0]
    ]
    assert return_ids == [call_ids[0]]

    # Idempotence: feeding the resulting history back in does not inject a duplicate pair
    # (the deterministic call_id means it's recognized as already discovered).
    result2 = await agent.run('And another refund?', message_history=messages)
    new_messages = result2.all_messages()[len(messages) :]
    assert synthetic_pairs(new_messages) == []


class _NoNativeToolSearchModel(FunctionModel):
    """`FunctionModel` that forces the local `search_tools` function path.

    `FunctionModel` reports support for every native tool (including native tool search),
    which would route deferred standalone tools through the provider rather than the
    synthetic `search_tools` function. Dropping `ToolSearchTool` mirrors a model without
    native tool-search support, exercising the function-tool discovery path.
    """

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset(super().supported_native_tools()) - {ToolSearchTool}


async def test_two_deferred_capabilities_loaded_sequentially_both_stay_available() -> None:
    """Loading a second deferred capability does not drop the first one's tool.

    Trajectory: load A and call A's tool, then on a later turn load B and call B's tool,
    then one more turn. Both capabilities' tools must be non-deferred on every turn after
    their respective loads, proving loads are additive and sticky.
    """
    toolset_a = FunctionToolset()

    @toolset_a.tool_plain
    def alpha_tool() -> str:
        """Capability A's tool."""
        return 'alpha-result'

    toolset_b = FunctionToolset()

    @toolset_b.tool_plain
    def beta_tool() -> str:
        """Capability B's tool."""
        return 'beta-result'

    cap_a = Capability[object](id='alpha', description='Alpha tools.', toolsets=[toolset_a], defer_loading=True)
    cap_b = Capability[object](id='beta', description='Beta tools.', toolsets=[toolset_b], defer_loading=True)

    available_per_turn: list[set[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_per_turn.append({td.name for td in info.function_tools if not td.defer_loading})

        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        names = {part.tool_name for part in tool_returns}

        # Turn 1: load A.
        if 'alpha' not in {part.capability_id for part in _load_calls(messages)}:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'alpha'}, tool_call_id='load-a')]
            )
        # Turn 2: use A's tool.
        if 'alpha_tool' not in names:
            return ModelResponse(parts=[ToolCallPart(tool_name='alpha_tool', args={}, tool_call_id='call-a')])
        # Turn 3: load B.
        if 'beta' not in {part.capability_id for part in _load_calls(messages)}:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'beta'}, tool_call_id='load-b')]
            )
        # Turn 4: use B's tool.
        if 'beta_tool' not in names:
            return ModelResponse(parts=[ToolCallPart(tool_name='beta_tool', args={}, tool_call_id='call-b')])
        # Turn 5+: just respond.
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap_a, cap_b])
    result = await agent.run('Use both capabilities.')

    assert result.output == 'done'
    # >= 5 turns: load A, use A, load B, use B, final.
    assert len(available_per_turn) >= 5

    # Identify the first turn on which each capability's tool became available.
    a_loaded_from = next(i for i, tools in enumerate(available_per_turn) if 'alpha_tool' in tools)
    b_loaded_from = next(i for i, tools in enumerate(available_per_turn) if 'beta_tool' in tools)
    assert a_loaded_from < b_loaded_from

    # Once loaded, each tool stays available on every later turn — loading B never drops A.
    for tools in available_per_turn[a_loaded_from:]:
        assert 'alpha_tool' in tools
    for tools in available_per_turn[b_loaded_from:]:
        assert 'beta_tool' in tools
    # Both present together on the final turn.
    assert {'alpha_tool', 'beta_tool'} <= available_per_turn[-1]


async def test_tool_search_discovery_and_capability_load_coexist() -> None:
    """A tool-search-discovered standalone tool and a load_capability tool coexist and persist.

    Trajectory: discover a standalone deferred tool via `search_tools`, load a deferred
    capability via `load_capability`, then continue for extra turns. Both the searched tool
    and the capability's tool must be available together and stay available afterwards.
    """
    standalone = FunctionToolset()

    @standalone.tool_plain(defer_loading=True)
    def searchable_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'{city}: sunny'

    cap_toolset = FunctionToolset()

    @cap_toolset.tool_plain
    def lookup_refund(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refundable'

    refunds = Capability[object](id='refunds', description='Refund tools.', toolsets=[cap_toolset], defer_loading=True)

    available_per_turn: list[set[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_per_turn.append({td.name for td in info.function_tools if not td.defer_loading})

        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        names = {part.tool_name for part in tool_returns}

        # Turn 1: search for the standalone deferred tool.
        if not any(part.tool_name == _SEARCH_TOOLS_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=_SEARCH_TOOLS_NAME, args={'queries': ['weather']}, tool_call_id='search')]
            )
        # Turn 2: load the deferred capability.
        if not _load_calls(messages):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        # Turn 3: use the discovered standalone tool.
        if 'searchable_weather' not in names:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='searchable_weather', args={'city': 'Paris'}, tool_call_id='call-w')]
            )
        # Turn 4: use the capability's tool.
        if 'lookup_refund' not in names:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='lookup_refund', args={'order_id': 'o1'}, tool_call_id='call-r')]
            )
        # Turn 5+: respond.
        return make_text_response('done')

    agent = Agent(_NoNativeToolSearchModel(model_fn), capabilities=[standalone_capability(standalone), refunds])
    result = await agent.run('Find weather and refund tools.')

    assert result.output == 'done'

    weather_from = next(i for i, tools in enumerate(available_per_turn) if 'searchable_weather' in tools)
    refund_from = next(i for i, tools in enumerate(available_per_turn) if 'lookup_refund' in tools)

    # Each reveal mechanism is sticky from the turn it first exposes its tool.
    for tools in available_per_turn[weather_from:]:
        assert 'searchable_weather' in tools
    for tools in available_per_turn[refund_from:]:
        assert 'lookup_refund' in tools
    # Both available together once both are revealed, including on the final turn.
    assert {'searchable_weather', 'lookup_refund'} <= available_per_turn[-1]


async def test_deferred_capability_synthetic_exchange_not_duplicated_over_long_trajectory() -> None:
    """The synthetic tool-search exchange for a loaded capability appears exactly once.

    Extends the persistence test to >= 3 model-request turns after the load: the deterministic
    `auto_load_*` call_id must keep the synthetic call/return pair singular across the whole
    trajectory, and the capability's tool stays available on every post-load turn.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds', description='Refund policy tools.', toolsets=[toolset], defer_loading=True
    )

    available_per_turn: list[set[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_per_turn.append({td.name for td in info.function_tools if not td.defer_loading})

        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )

        # Three post-load turns that each call the loaded tool, then respond.
        lookup_calls = [part for part in tool_returns if part.tool_name == 'lookup_refund_policy']
        if len(lookup_calls) < 3:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy',
                        args={'order_id': f'order-{len(lookup_calls)}'},
                        tool_call_id=f'lookup-{len(lookup_calls)}',
                    )
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('Refund please.')

    assert result.output == 'done'

    messages = result.all_messages()
    synthetic_call_ids = [
        part.tool_call_id
        for message in messages
        for part in message.parts
        if isinstance(part, ToolSearchCallPart) and part.tool_call_id.startswith('auto_load_')
    ]
    synthetic_return_ids = [
        part.tool_call_id
        for message in messages
        for part in message.parts
        if isinstance(part, ToolSearchReturnPart) and part.tool_call_id.startswith('auto_load_')
    ]
    # Exactly one synthetic exchange survives the long trajectory — no per-turn duplication.
    assert len(synthetic_call_ids) == 1
    assert synthetic_return_ids == synthetic_call_ids

    # The capability's tool was deferred on turn 1 and available on every post-load turn.
    assert 'lookup_refund_policy' not in available_per_turn[0]
    post_load_turns = available_per_turn[1:]
    assert len(post_load_turns) >= 3
    for tools in post_load_turns:
        assert 'lookup_refund_policy' in tools


async def test_deferred_capability_tool_available_on_turn_that_does_not_call_it() -> None:
    """A loaded capability's tool stays available on a turn that does not call it.

    After loading, the model calls an unrelated visible tool (not the capability's tool) and
    then responds. The capability's tool must remain non-deferred on those turns — loading is
    sticky, not gated on per-turn usage.
    """
    visible_toolset = FunctionToolset()

    @visible_toolset.tool_plain
    def ping() -> str:
        """An always-visible tool unrelated to the capability."""
        return 'pong'

    cap_toolset = FunctionToolset()

    @cap_toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:  # pragma: no cover
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds', description='Refund policy tools.', toolsets=[cap_toolset], defer_loading=True
    )

    available_per_turn: list[set[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_per_turn.append({td.name for td in info.function_tools if not td.defer_loading})

        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        names = {part.tool_name for part in tool_returns}

        # Turn 1: load the capability.
        if not any(part.tool_name == LOAD_CAPABILITY_TOOL_NAME for part in tool_returns):
            return ModelResponse(
                parts=[ToolCallPart(tool_name=LOAD_CAPABILITY_TOOL_NAME, args={'id': 'refunds'}, tool_call_id='load')]
            )
        # Turn 2: call an UNRELATED tool, never the capability's tool.
        if 'ping' not in names:
            return ModelResponse(parts=[ToolCallPart(tool_name='ping', args={}, tool_call_id='call-ping')])
        # Turn 3: respond without ever calling the capability's tool.
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), tools=[ping], capabilities=[refunds])
    # `ping` is registered via a function tool on the agent; ensure both paths see it.
    result = await agent.run('Load refunds but use ping.')

    assert result.output == 'done'
    assert len(available_per_turn) >= 3

    # Turn 1: capability tool still deferred.
    assert 'lookup_refund_policy' not in available_per_turn[0]
    # Every turn after the load: capability tool available even though it is never called.
    for tools in available_per_turn[1:]:
        assert 'lookup_refund_policy' in tools


def _load_calls(messages: list[ModelMessage]) -> list[LoadCapabilityCallPart]:
    """All `load_capability` call parts in the message history."""
    return [
        part
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, LoadCapabilityCallPart)
    ]


def standalone_capability(toolset: FunctionToolset) -> Capability:
    """Wrap a toolset of standalone deferred tools in an eager capability (tools keep their own defer flag)."""
    return Capability[object](id='standalone', description='Standalone searchable tools.', toolsets=[toolset])


async def test_deferred_capability_load_includes_toolset_instructions() -> None:
    """Instructions declared on a deferred capability's toolset surface via the `load_capability` return.

    The wrapping `CapabilityOwnedToolset` silences `get_instructions` for deferred-loading
    capabilities (so toolset hints don't leak into the prompt), then re-emits them on load
    alongside the capability's own instructions.
    """
    toolset = FunctionToolset(instructions='Use the refund tool with the order id, not the customer id.')

    @toolset.tool_plain
    def lookup_refund(order_id: str) -> str:
        return f'{order_id}: ok'

    refunds = Capability[object](
        id='refunds',
        description='Refund tools.',
        instructions='Quote the refund policy verbatim.',
        toolsets=[toolset],
        defer_loading=True,
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
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
                        args={'id': 'refunds'},
                        tool_call_id='load-refunds',
                    )
                ]
            )
        if not any(part.tool_name == 'lookup_refund' for part in tool_returns):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund',
                        args={'order_id': 'order-123'},
                        tool_call_id='lookup-refund',
                    )
                ]
            )
        refund_result = next(part.content for part in tool_returns if part.tool_name == 'lookup_refund')
        return make_text_response(str(refund_result))

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds])
    result = await agent.run('hi')

    assert result.output == 'order-123: ok'
    [load_return] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert load_return.instructions == snapshot("""\
Quote the refund policy verbatim.

Use the refund tool with the order id, not the customer id.\
""")
    first_request = next(message for message in result.all_messages() if isinstance(message, ModelRequest))
    assert first_request.instructions == snapshot(
        'The following capabilities are deferred and can be loaded using the `load_capability` tool:\n'
        '- refunds: Refund tools.'
    )
    assert first_request.instructions is not None
    assert 'Use the refund tool' not in first_request.instructions


async def test_deferred_capability_load_drops_empty_toolset_instructions() -> None:
    """Empty toolset instructions are filtered from load returns."""
    from dataclasses import dataclass

    from pydantic_ai.messages import InstructionPart
    from pydantic_ai.toolsets.wrapper import WrapperToolset

    @dataclass
    class _LiteralInstructionsToolset(WrapperToolset):
        raw: tuple[str | InstructionPart, ...] = ()

        async def get_instructions(self, ctx: RunContext) -> list[str | InstructionPart]:
            return list(self.raw)

    toolset = _LiteralInstructionsToolset(
        wrapped=FunctionToolset(),
        raw=(
            InstructionPart(content='   ', dynamic=False),
            InstructionPart(content='Real hint from toolset.', dynamic=False),
            '',
        ),
    )
    cap = Capability[object](id='cap', description='Custom-toolset cap.', toolsets=[toolset], defer_loading=True)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
                        args={'id': 'cap'},
                        tool_call_id='load',
                    )
                ]
            )
        return make_text_response('ok')

    agent = Agent(FunctionModel(model_fn), capabilities=[cap])
    result = await agent.run('hi')

    [load_return] = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert load_return.instructions == 'Real hint from toolset.'


async def test_unknown_deferred_capability_id_does_not_reveal_hidden_tools() -> None:
    toolset = FunctionToolset()

    @toolset.tool_plain
    def hidden_tool() -> str:
        return 'hidden'  # pragma: no cover

    hidden = Capability[object](
        id='hidden',
        description='Hidden tool access.',
        toolsets=[toolset],
        defer_loading=True,
    )
    seen_tool_state: list[list[tuple[str, bool]]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tool_state.append([(t.name, bool(t.defer_loading)) for t in info.function_tools])
        # Give up on the first signal of tool feedback — either a `ToolReturnPart`
        # (success, which can't happen here) or a `RetryPromptPart` (the framework
        # signaling the bad cap id). Without the retry branch, we'd loop past
        # `max_retries` and raise `UnexpectedModelBehavior` instead of giving up.
        if not any(
            isinstance(part, (ToolReturnPart, RetryPromptPart))
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        ):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=LOAD_CAPABILITY_TOOL_NAME,
                        args={'id': 'missing'},
                        tool_call_id='load-missing',
                    )
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn), capabilities=[hidden])
    result = await agent.run('load missing')

    assert result.output == snapshot('done')
    assert seen_tool_state == snapshot(
        [
            [('load_capability', False), ('hidden_tool', True), ('search_tools', False)],
            [('load_capability', False), ('hidden_tool', True), ('search_tools', False)],
        ]
    )
    history_parts = [part for message in result.all_messages() for part in message.parts]
    assert not any(isinstance(part, LoadCapabilityReturnPart) for part in history_parts)
    [retry] = [part for part in history_parts if isinstance(part, RetryPromptPart)]
    assert retry.content == snapshot("No capability found with id 'missing'.")


async def test_load_capability_retries_for_already_available_capability() -> None:
    always_on = Capability[object](
        id='always-on',
        description='Already visible.',
        instructions='Already visible instructions.',
    )
    deferred = Capability[object](
        id='deferred',
        description='Deferred.',
        instructions='Deferred instructions.',
        defer_loading=True,
    )
    expected_retry = LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE.format(capability_id='always-on')
    retry_messages: list[str] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        retries = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart) and isinstance(part.content, str)
        ]
        if retries:
            retry_messages.extend(retries)
            return make_text_response('done')

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=LOAD_CAPABILITY_TOOL_NAME,
                    args={'id': 'always-on'},
                    tool_call_id='load-always-on',
                )
            ]
        )

    agent = Agent(FunctionModel(model_fn), capabilities=[always_on, deferred])
    result = await agent.run('load always-on')

    assert result.output == 'done'
    assert retry_messages == [expected_retry]
    assert not any(
        isinstance(part, LoadCapabilityReturnPart) for message in result.all_messages() for part in message.parts
    )


async def test_load_capability_retries_when_capability_is_already_loaded() -> None:
    deferred = Capability[object](
        id='deferred',
        description='Deferred.',
        instructions='Deferred instructions.',
        defer_loading=True,
    )
    expected_retry = LOAD_CAPABILITY_ALREADY_AVAILABLE_MESSAGE_TEMPLATE.format(capability_id='deferred')
    retry_messages: list[str] = []

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        retries = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, RetryPromptPart) and isinstance(part.content, str)
        ]
        if retries:
            retry_messages.extend(retries)
            return make_text_response('done')

        load_returns = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, LoadCapabilityReturnPart)
        ]
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=LOAD_CAPABILITY_TOOL_NAME,
                    args={'id': 'deferred'},
                    tool_call_id=f'load-deferred-{len(load_returns)}',
                )
            ]
        )

    agent = Agent(FunctionModel(model_fn), capabilities=[deferred])
    result = await agent.run('load twice')

    assert result.output == 'done'
    assert retry_messages == [expected_retry]
    load_returns = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, LoadCapabilityReturnPart)
    ]
    assert len(load_returns) == 1
    assert load_returns[0].instructions == 'Deferred instructions.'


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
    ctx.discovered_tool_names = {'discovered_tool', 'removed_tool'}
    ctx.loaded_capability_ids = {'loaded_capability'}
    tools = await toolset.get_tools(ctx)
    tools['discovered_tool'] = replace(
        tools['discovered_tool'],
        tool_def=replace(tools['discovered_tool'].tool_def, with_native=ToolSearchTool.kind, defer_loading=False),
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
            defer_loading=False,
            capability_id='loaded_capability',
        ),
    )
    tool_manager = ToolManager(toolset=toolset, ctx=ctx, tools=tools)
    ctx.tool_manager = tool_manager

    assert ctx.available_tool_names == {'always_tool', 'discovered_tool', 'loaded_capability_tool'}


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


# --- Hooks test helpers ---


@dataclass
class _ReplacingCapability(AbstractCapability[Any]):
    """Capability that replaces ModelRequestNode with a fresh copy in before_node_run.

    Used to test that streaming + node replacement doesn't cause double model execution.
    """

    replaced: bool = field(default=False, init=False)

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        from pydantic_ai import ModelRequestNode

        if isinstance(node, ModelRequestNode) and not self.replaced:
            self.replaced = True
            return ModelRequestNode(request=node.request)  # pyright: ignore[reportUnknownVariableType]
        return node  # pyright: ignore[reportUnknownVariableType]


def make_text_response(text: str = 'hello') -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def simple_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return make_text_response('response from model')


async def simple_stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    yield 'streamed response'


async def tool_calling_stream_function(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """A streaming model that calls a tool on first request, then returns text."""
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                yield 'final response'
                return

    if info.function_tools:
        tool = info.function_tools[0]
        yield {0: DeltaToolCall(name=tool.name, json_args='{}', tool_call_id='call-1')}
        return

    yield 'no tools available'  # pragma: no cover


# Defined at module scope so pydantic-ai can resolve the annotation under `from __future__ import annotations`.
class SingleBaseModelArg(BaseModel):
    label: str = 'default'


def tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that calls a tool on first request, then returns text."""
    # Check if there's already a tool return in messages (i.e., tool was called)
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return make_text_response('final response')

    # First request: call the tool
    if info.function_tools:
        tool = info.function_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{}', tool_call_id='call-1')])

    return make_text_response('no tools available')  # pragma: no cover


# --- Logging capability for testing ---


@dataclass
class LoggingCapability(AbstractCapability[Any]):
    """A capability that logs all hook invocations for testing."""

    log: list[str] = field(default_factory=lambda: [])

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.log.append('before_run')

    async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        self.log.append('after_run')
        return result

    async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
        self.log.append('wrap_run:before')
        result = await handler()
        self.log.append('wrap_run:after')
        return result

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self.log.append('before_model_request')
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self.log.append('after_model_request')
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: Any,
        handler: Any,
    ) -> ModelResponse:
        self.log.append('wrap_model_request:before')
        response = await handler(request_context)
        self.log.append('wrap_model_request:after')
        return response

    async def before_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        self.log.append(f'before_tool_validate:{call.tool_name}')
        return args

    async def after_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.log.append(f'after_tool_validate:{call.tool_name}')
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: Any,
    ) -> dict[str, Any]:
        self.log.append(f'wrap_tool_validate:{call.tool_name}:before')
        result = await handler(args)
        self.log.append(f'wrap_tool_validate:{call.tool_name}:after')
        return result

    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        self.log.append(f'before_tool_execute:{call.tool_name}')
        return args

    async def after_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], result: Any
    ) -> Any:
        self.log.append(f'after_tool_execute:{call.tool_name}')
        return result

    async def wrap_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], handler: Any
    ) -> Any:
        self.log.append(f'wrap_tool_execute:{call.tool_name}:before')
        result = await handler(args)
        self.log.append(f'wrap_tool_execute:{call.tool_name}:after')
        return result

    async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
        self.log.append('on_run_error')
        raise error

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        self.log.append(f'before_node_run:{type(node).__name__}')
        return node

    async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
        self.log.append(f'after_node_run:{type(node).__name__}')
        return result

    async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
        self.log.append(f'on_node_run_error:{type(node).__name__}')
        raise error

    async def on_model_request_error(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        self.log.append('on_model_request_error')
        raise error

    async def on_tool_validate_error(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
    ) -> dict[str, Any]:
        self.log.append(f'on_tool_validate_error:{call.tool_name}')
        raise error

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        self.log.append(f'on_tool_execute_error:{call.tool_name}')
        raise error


# --- Tests ---


class TestRunHooks:
    async def test_before_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_run' in cap.log

    async def test_after_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'after_run' in cap.log

    async def test_wrap_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log

    async def test_run_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        # wrap_run wraps the run (which includes before_run inside iter),
        # then after_run fires at the end (outside wrap_run)
        assert cap.log.index('wrap_run:before') < cap.log.index('before_run')
        assert cap.log.index('before_run') < cap.log.index('wrap_run:after')
        assert cap.log.index('wrap_run:after') <= cap.log.index('after_run')

    async def test_after_run_can_modify_result(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                return AgentRunResult(output='modified output')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ModifyResultCap()])
        result = await agent.run('hello')
        assert result.output == 'modified output'

    async def test_wrap_run_can_short_circuit(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                # Don't call handler - short-circuit the run
                return AgentRunResult(output='short-circuited')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ShortCircuitRunCap()])
        result = await agent.run('hello')
        assert result.output == 'short-circuited'

    async def test_wrap_run_can_recover_from_error(self):
        """wrap_run can catch errors from handler() and return a recovery result."""

        @dataclass
        class ErrorRecoveryCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    return AgentRunResult(output='recovered from error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[ErrorRecoveryCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered from error'

    async def test_wrap_run_error_propagates_without_recovery(self):
        """Without recovery in wrap_run, errors propagate normally."""

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model))
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_wrap_run_recovery_via_iter(self):
        """wrap_run error recovery works when using agent.iter() too."""

        @dataclass
        class ErrorRecoveryCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    return AgentRunResult(output='recovered via iter')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[ErrorRecoveryCap()])
        async with agent.iter('hello') as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        assert agent_run.result.output == 'recovered via iter'


class TestModelRequestHooks:
    async def test_before_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_model_request' in cap.log

    @pytest.mark.parametrize(
        ('mode', 'streaming'),
        [('run', False), ('run_stream', True), ('event_stream_handler', True)],
    )
    async def test_before_model_request_sees_selection_context(self, mode: str, streaming: bool):
        """`before_model_request` sees the selected model ID and effective streaming mode."""
        contexts: list[ModelRequestContext] = []

        @dataclass
        class CaptureContext(AbstractCapability[None]):
            async def before_model_request(
                self,
                ctx: RunContext[None],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                contexts.append(request_context)
                return request_context

        agent = Agent('test', deps_type=type(None), capabilities=[CaptureContext()], defer_model_check=True)
        if mode == 'run_stream':
            async with agent.run_stream('hello') as result:
                await result.get_output()
        elif mode == 'event_stream_handler':

            async def handle_events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
                async for _ in stream:
                    pass

            await agent.run('hello', event_stream_handler=handle_events)
        else:
            await agent.run('hello')

        assert [(context.model_id, context.streaming) for context in contexts] == [('test', streaming)]

    async def test_withdrawn_bootstrap_model_id_does_not_leak_to_default(self):
        """A bootstrap model contribution withdrawn by `for_run` must not leak its selection string as provenance."""
        model_ids: list[str | None] = []

        @dataclass
        class BootstrapModel(AbstractCapability[None]):
            def get_model(self) -> str:
                return 'bootstrap-alias'

            async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
                return AbstractCapability()

            async def resolve_model_id(
                self, ctx: ModelResolutionContext[None], *, model_id: KnownModelName | str
            ) -> Model | None:
                return TestModel() if model_id == 'bootstrap-alias' else None

        @dataclass
        class CaptureModelId(AbstractCapability[None]):
            async def before_model_request(
                self,
                ctx: RunContext[None],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                model_ids.append(request_context.model_id)
                return request_context

        agent = Agent(TestModel(), deps_type=NoneType, capabilities=[BootstrapModel(), CaptureModelId()])
        await agent.run('hello')

        assert model_ids == [None]

    async def test_after_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'after_model_request' in cap.log

    async def test_wrap_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'wrap_model_request:before' in cap.log
        assert 'wrap_model_request:after' in cap.log

    async def test_model_request_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert cap.log.index('before_model_request') < cap.log.index('wrap_model_request:before')
        assert cap.log.index('wrap_model_request:before') < cap.log.index('wrap_model_request:after')
        assert cap.log.index('wrap_model_request:after') < cap.log.index('after_model_request')

    async def test_after_model_request_can_modify_response(self):
        @dataclass
        class ModifyResponseCap(AbstractCapability[Any]):
            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='modified by after hook')])

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ModifyResponseCap()])
        result = await agent.run('hello')
        assert result.output == 'modified by after hook'

    async def test_wrap_model_request_can_modify_response(self):
        @dataclass
        class WrapModifyCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                response = await handler(request_context)
                return ModelResponse(parts=[TextPart(content='wrapped: ' + response.parts[0].content)])

        agent = Agent(FunctionModel(simple_model_function), capabilities=[WrapModifyCap()])
        result = await agent.run('hello')
        assert result.output == 'wrapped: response from model'

    async def test_skip_model_request(self):
        @dataclass
        class SkipCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped model')]))

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SkipCap()])
        result = await agent.run('hello')
        assert result.output == 'skipped model'

    async def test_before_model_request_swaps_model(self):
        call_log: list[str] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_log.append('swap_model')
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapModelCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                request_context.model = swap_target
                return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SwapModelCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert call_log == ['swap_model']

    async def test_wrap_model_request_swaps_model(self):
        call_log: list[str] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_log.append('swap_model')
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapInWrapCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
            ) -> ModelResponse:
                request_context.model = swap_target
                return await handler(request_context)

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SwapInWrapCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert call_log == ['swap_model']

    async def test_before_model_request_swaps_model_streaming(self):
        call_log: list[str] = []

        async def swap_stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            call_log.append('swap_stream')
            yield 'from swap stream'

        swap_target = FunctionModel(stream_function=swap_stream_fn)

        @dataclass
        class SwapModelCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                request_context.model = swap_target
                return request_context

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[SwapModelCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'from swap stream'
        assert call_log == ['swap_stream']

    async def test_run_context_model_unchanged_after_swap(self):
        observed_models: list[Any] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('from swap model')

        original_model = FunctionModel(simple_model_function)
        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapAndObserveCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                observed_models.append(ctx.model)
                request_context.model = swap_target
                return request_context

        agent = Agent(original_model, capabilities=[SwapAndObserveCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert observed_models[0] is original_model

    async def test_hooks_before_model_request_swaps_model(self):
        call_log: list[str] = []
        hooks = Hooks()

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            call_log.append('swap_model')
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @hooks.on.before_model_request
        async def _(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            request_context.model = swap_target
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert call_log == ['swap_model']

    async def test_after_model_request_sees_wrap_swap(self):
        """after_model_request sees the model swapped during wrap_model_request."""
        after_models: list[Any] = []

        def swap_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('from swap model')

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapInWrapAndObserveCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
            ) -> ModelResponse:
                request_context.model = swap_target
                return await handler(request_context)

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                after_models.append(request_context.model)
                return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SwapInWrapAndObserveCap()])
        result = await agent.run('hello')
        assert result.output == 'from swap model'
        assert after_models[0] is swap_target


class TestToolValidateHooks:
    async def test_tool_validate_hooks_fire(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert 'before_tool_validate:my_tool' in cap.log
        assert 'after_tool_validate:my_tool' in cap.log
        assert 'wrap_tool_validate:my_tool:before' in cap.log
        assert 'wrap_tool_validate:my_tool:after' in cap.log

    async def test_before_tool_validate_can_modify_args(self):
        @dataclass
        class ModifyArgsCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                # Inject an argument
                if isinstance(args, dict):
                    return {**args, 'name': 'injected'}  # pragma: no cover
                return {'name': 'injected'}

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[ModifyArgsCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f'hello {name}'

        await agent.run('greet someone')
        assert received_name == 'injected'

    async def test_skip_tool_validation(self):
        @dataclass
        class SkipValidateCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                raise SkipToolValidation({'name': 'skip-validated'})

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[SkipValidateCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f'hello {name}'

        await agent.run('greet someone')
        assert received_name == 'skip-validated'

    async def test_tool_def_matches_called_tool(self):
        """Verify tool_def is the correct ToolDefinition for the tool being called."""
        received_tool_defs: list[ToolDefinition] = []

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                received_tool_defs.append(tool_def)
                return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[CaptureCap()])

        @agent.tool_plain(description='Say hello')
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert len(received_tool_defs) == 1
        td = received_tool_defs[0]
        assert td.name == 'my_tool'
        assert td.description == 'Say hello'
        assert td.kind == 'function'


class TestToolExecuteHooks:
    async def test_tool_execute_hooks_fire(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert 'before_tool_execute:my_tool' in cap.log
        assert 'after_tool_execute:my_tool' in cap.log
        assert 'wrap_tool_execute:my_tool:before' in cap.log
        assert 'wrap_tool_execute:my_tool:after' in cap.log

    async def test_after_tool_execute_can_modify_result(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
            ) -> Any:
                return f'modified: {result}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[ModifyResultCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return 'original'

        result = await agent.run('call tool')
        assert 'modified: original' in result.output

    async def test_skip_tool_execution(self):
        @dataclass
        class SkipExecCap(AbstractCapability[Any]):
            async def before_tool_execute(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
            ) -> dict[str, Any]:
                raise SkipToolExecution('denied')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[SkipExecCap()])

        tool_was_called = False

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_was_called
            tool_was_called = True  # pragma: no cover
            return 'should not be called'  # pragma: no cover

        result = await agent.run('call tool')
        assert not tool_was_called
        assert 'denied' in result.output

    async def test_wrap_tool_execute_with_error_handling(self):
        @dataclass
        class ErrorHandlingCap(AbstractCapability[Any]):
            caught_error: str | None = None

            async def wrap_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                handler: Any,
            ) -> Any:
                try:
                    return await handler(args)
                except Exception as e:
                    self.caught_error = str(e)
                    return 'recovered from error'

        cap = ErrorHandlingCap()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        await agent.run('call tool')
        assert cap.caught_error == 'tool failed'

    async def test_hooks_receive_dict_args_for_single_base_model_tool(self):
        """Validate and execute hooks receive dict-shaped args when the tool has a single BaseModel parameter.

        The JSON schema sent to the model unwraps the BaseModel, so the model generates its fields at the
        top level. Pydantic's validator returns a BaseModel instance directly, but the framework wraps it
        as `{param_name: model}` so hooks and `call_tool` always see a dict.
        """
        captured_args: list[tuple[str, dict[str, Any]]] = []

        @dataclass
        class CapturingCap(AbstractCapability[Any]):
            async def after_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                captured_args.append(('validate', args))
                return args

            async def wrap_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                handler: Any,
            ) -> Any:
                captured_args.append(('execute', args))
                return await handler(args)

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[CapturingCap()])

        @agent.tool_plain
        def my_tool(payload: SingleBaseModelArg) -> str:
            return f'got {payload.label}'

        await agent.run('call the tool')
        assert captured_args == [
            ('validate', {'payload': SingleBaseModelArg()}),
            ('execute', {'payload': SingleBaseModelArg()}),
        ]

    async def test_tool_hooks_skip_output_tools(self):
        """Tool hooks don't fire for internal output tools (#5111).

        Output tools deliver structured output to the user via `result.output`; they're not
        user-facing tool calls. Firing hooks on them lets e.g. `after_tool_execute` return a
        `ToolReturn` that leaks through to `result.output` instead of the typed value.
        """

        class MyOutput(BaseModel):
            answer: str

        hooks = Hooks()

        @hooks.on.after_tool_execute
        async def wrap_result(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            result: Any,
        ) -> ToolReturn:
            return ToolReturn(return_value=result, content='extra context')

        cap = LoggingCapability()
        agent = Agent(
            TestModel(custom_output_args={'answer': 'hi'}),
            output_type=MyOutput,
            capabilities=[cap, hooks],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        result = await agent.run('call tool and answer')

        # Function tool still fires every tool hook.
        assert 'before_tool_validate:my_tool' in cap.log
        assert 'after_tool_validate:my_tool' in cap.log
        assert 'wrap_tool_validate:my_tool:before' in cap.log
        assert 'wrap_tool_validate:my_tool:after' in cap.log
        assert 'before_tool_execute:my_tool' in cap.log
        assert 'after_tool_execute:my_tool' in cap.log
        assert 'wrap_tool_execute:my_tool:before' in cap.log
        assert 'wrap_tool_execute:my_tool:after' in cap.log
        # Output tool does not appear in any hook log entry.
        assert all('final_result' not in entry for entry in cap.log)
        # Regression for #5111: the ToolReturn from `after_tool_execute` would have corrupted
        # `result.output` if output tool hooks still fired.
        assert result.output == MyOutput(answer='hi')


class TestCompositionOrder:
    async def test_multiple_capabilities_model_request_order(self):
        """Test that multiple capabilities compose in the correct order."""
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append('cap1:before')
                return request_context

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append('cap1:after')
                return response

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                log.append('cap1:wrap:before')
                response = await handler(request_context)
                log.append('cap1:wrap:after')
                return response

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append('cap2:before')
                return request_context

            async def after_model_request(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
            ) -> ModelResponse:
                log.append('cap2:after')
                return response

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                log.append('cap2:wrap:before')
                response = await handler(request_context)
                log.append('cap2:wrap:after')
                return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Cap1(), Cap2()])
        await agent.run('hello')

        # before hooks: forward order (cap1 then cap2)
        assert log.index('cap1:before') < log.index('cap2:before')
        # wrap hooks: cap1 outermost, cap2 innermost
        assert log.index('cap1:wrap:before') < log.index('cap2:wrap:before')
        assert log.index('cap2:wrap:after') < log.index('cap1:wrap:after')
        # after hooks: reverse order (cap2 then cap1)
        assert log.index('cap2:after') < log.index('cap1:after')

    async def test_multiple_capabilities_run_hooks_order(self):
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                log.append('cap1:before_run')

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                log.append('cap1:after_run')
                return result

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                log.append('cap1:wrap_run:before')
                result = await handler()
                log.append('cap1:wrap_run:after')
                return result

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                log.append('cap2:before_run')

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                log.append('cap2:after_run')
                return result

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                log.append('cap2:wrap_run:before')
                result = await handler()
                log.append('cap2:wrap_run:after')
                return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Cap1(), Cap2()])
        await agent.run('hello')

        # before_run: forward order
        assert log.index('cap1:before_run') < log.index('cap2:before_run')
        # wrap_run: cap1 outermost
        assert log.index('cap1:wrap_run:before') < log.index('cap2:wrap_run:before')
        assert log.index('cap2:wrap_run:after') < log.index('cap1:wrap_run:after')
        # after_run: reverse order
        assert log.index('cap2:after_run') < log.index('cap1:after_run')


class TestCombinedBeforeWrapAfter:
    async def test_all_hook_types_on_same_capability(self):
        """Test before + wrap + after all fire correctly on a single capability."""
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')

        # Check run hooks
        assert 'before_run' in cap.log
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log
        assert 'after_run' in cap.log

        # Check model request hooks (should fire twice: once for tool call, once for final)
        model_request_before_count = cap.log.count('before_model_request')
        assert model_request_before_count == 2

        # Check tool hooks
        assert 'before_tool_validate:my_tool' in cap.log
        assert 'wrap_tool_validate:my_tool:before' in cap.log
        assert 'wrap_tool_validate:my_tool:after' in cap.log
        assert 'after_tool_validate:my_tool' in cap.log
        assert 'before_tool_execute:my_tool' in cap.log
        assert 'wrap_tool_execute:my_tool:before' in cap.log
        assert 'wrap_tool_execute:my_tool:after' in cap.log
        assert 'after_tool_execute:my_tool' in cap.log


class TestRunHooksRunStream:
    """Test that wrap_run and after_run fire for run_stream()."""

    async def test_wrap_run_fires_for_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log

    async def test_after_run_fires_for_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'after_run' in cap.log

    async def test_wrap_run_fires_for_iter(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        async with agent.iter('hello') as agent_run:
            async for _node in agent_run:
                pass
        assert 'wrap_run:before' in cap.log
        assert 'wrap_run:after' in cap.log
        assert 'after_run' in cap.log

    async def test_after_run_can_modify_result_via_iter(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                return AgentRunResult(output='modified by after_run')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ModifyResultCap()])
        async with agent.iter('hello') as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        assert agent_run.result.output == 'modified by after_run'

    async def test_run_hook_order_via_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert cap.log.index('wrap_run:before') < cap.log.index('before_run')
        assert cap.log.index('before_run') < cap.log.index('wrap_run:after')
        assert cap.log.index('wrap_run:after') <= cap.log.index('after_run')


class TestStreamingHooks:
    """Test that SkipModelRequest and wrap_model_request work in streaming paths."""

    async def test_skip_model_request_streaming(self):
        @dataclass
        class SkipCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped in stream')]))

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[SkipCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'skipped in stream'

    async def test_skip_model_request_from_wrap_model_request(self):
        """SkipModelRequest raised inside wrap_model_request is handled in non-streaming."""

        @dataclass
        class WrapSkipCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='wrap-skipped')]))

        agent = Agent(FunctionModel(simple_model_function), capabilities=[WrapSkipCap()])
        result = await agent.run('hello')
        assert result.output == 'wrap-skipped'

    async def test_skip_model_request_from_wrap_model_request_streaming(self):
        """SkipModelRequest raised inside wrap_model_request during streaming is handled."""

        @dataclass
        class WrapSkipCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='wrap-skipped in stream')]))

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[WrapSkipCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'wrap-skipped in stream'

    async def test_wrap_model_request_streaming(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'wrap_model_request:before' in cap.log
        assert 'wrap_model_request:after' in cap.log

    async def test_wrap_model_request_modifies_result_via_run_with_streaming(self):
        """wrap_model_request modification affects the final result when using run() with streaming."""

        @dataclass
        class WrapModifyCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                response = await handler(request_context)
                return ModelResponse(parts=[TextPart(content='wrapped: ' + response.parts[0].content)])

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[WrapModifyCap()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        result = await agent.run('hello', event_stream_handler=handler)
        assert result.output == 'wrapped: streamed response'

    async def test_after_model_request_fires_streaming(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[cap],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'after_model_request' in cap.log


class TestWrapRunEventStream:
    """Tests for the wrap_run_event_stream hook."""

    async def test_wrap_run_event_stream_observes(self):
        """Hook sees events from model streaming."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
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
            capabilities=[ObserverCap()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        await agent.run('hello', event_stream_handler=handler)
        assert len(observed_events) > 0

    async def test_wrap_run_event_stream_transforms(self):
        """Modifications by the hook are visible to event_stream_handler."""
        handler_events: list[AgentStreamEvent] = []

        @dataclass
        class TransformCap(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async for event in stream:
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[TransformCap()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        await agent.run('hello', event_stream_handler=handler)
        assert len(handler_events) > 0

    async def test_wrap_run_event_stream_composition(self):
        """Multiple capabilities compose in correct order (first = outermost)."""
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                log.append('cap1:enter')
                async for event in stream:
                    yield event
                log.append('cap1:exit')

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                log.append('cap2:enter')
                async for event in stream:
                    yield event
                log.append('cap2:exit')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[Cap1(), Cap2()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        await agent.run('hello', event_stream_handler=handler)

        # Cap1 is outermost, so enters first and exits last
        assert log.index('cap1:enter') < log.index('cap2:enter')
        assert log.index('cap2:exit') < log.index('cap1:exit')

    async def test_wrap_run_event_stream_tool_events(self):
        """HandleResponseEvents from CallToolsNode flow through the hook."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
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
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[ObserverCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        await agent.run('call tool', event_stream_handler=handler)
        # Should have observed events from both ModelRequestNode and CallToolsNode streams
        assert len(observed_events) > 0

    async def test_wrap_run_event_stream_fires_in_run_stream_without_handler(self):
        """wrap_run_event_stream fires in run_stream() even without an event_stream_handler."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
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
            capabilities=[ObserverCap()],
        )

        # No event_stream_handler — hook should still fire
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(observed_events) > 0

    async def test_wrap_run_event_stream_fires_in_run_without_handler(self):
        """wrap_run_event_stream fires in run() even without an event_stream_handler."""
        observed_events: list[AgentStreamEvent] = []

        @dataclass
        class ObserverCap(AbstractCapability[Any]):
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
            capabilities=[ObserverCap()],
        )

        # No event_stream_handler — hook should still fire via forced streaming
        result = await agent.run('hello')
        assert result.output is not None
        assert any(isinstance(e, PartStartEvent) for e in observed_events)


class TestProcessEventStream:
    """Tests for the ProcessEventStream capability."""

    async def test_handler_receives_events(self):
        """Handler registered via capability receives events from model streaming."""
        handler_events: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=handler)],
        )

        # No event_stream_handler arg — capability should drive streaming
        result = await agent.run('hello')
        assert result.output is not None
        assert any(isinstance(e, PartStartEvent) for e in handler_events)

    async def test_multiple_handlers_and_param_all_observe(self):
        """Multiple ProcessEventStream capabilities and an explicit event_stream_handler all see the same events."""
        cap1_events: list[AgentStreamEvent] = []
        cap2_events: list[AgentStreamEvent] = []
        param_events: list[AgentStreamEvent] = []

        async def cap1_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                cap1_events.append(event)

        async def cap2_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                cap2_events.append(event)

        async def param_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                param_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=cap1_handler), ProcessEventStream(handler=cap2_handler)],
        )

        await agent.run('hello', event_stream_handler=param_handler)
        assert len(cap1_events) > 0
        assert cap1_events == cap2_events == param_events

    async def test_handler_sees_events_after_inner_wrappers(self):
        """Events passed to the handler go through inner wrap_run_event_stream wrappers."""
        transformed_calls: list[AgentStreamEvent] = []
        handler_events: list[AgentStreamEvent] = []

        @dataclass
        class InnerWrapper(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                async for event in stream:
                    transformed_calls.append(event)
                    yield event

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                handler_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=handler), InnerWrapper()],
        )

        await agent.run('hello')
        assert handler_events == transformed_calls
        assert len(handler_events) > 0

    async def test_transformer_handler_replaces_stream(self):
        """An async-generator handler transforms the stream seen by downstream wrappers and the param handler."""
        downstream_events: list[AgentStreamEvent] = []

        async def transformer(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterator[AgentStreamEvent]:
            async for event in stream:
                if isinstance(event, PartStartEvent):
                    # Drop PartStart events — downstream should never see them.
                    continue
                yield event

        async def param_handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                downstream_events.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=transformer)],
        )

        await agent.run('hello', event_stream_handler=param_handler)
        assert len(downstream_events) > 0
        assert not any(isinstance(e, PartStartEvent) for e in downstream_events)

    async def test_callable_instance_processor(self):
        """A callable-class processor (not a plain async-generator function) is detected via its return type."""
        captured: list[AgentStreamEvent] = []

        class Transformer:
            async def __call__(
                self, _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
            ) -> AsyncIterator[AgentStreamEvent]:
                async for event in stream:
                    captured.append(event)
                    yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=Transformer())],
        )
        await agent.run('hello')
        assert any(isinstance(e, PartStartEvent) for e in captured)

    async def test_observer_bailout_does_not_break_downstream(self):
        """If an observer stops iterating early, downstream consumers still see all events."""
        received_by_observer: list[AgentStreamEvent] = []
        received_downstream: list[AgentStreamEvent] = []

        async def bail_after_first(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                received_by_observer.append(event)
                return

        async def downstream(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                received_downstream.append(event)

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ProcessEventStream(handler=bail_after_first)],
        )
        await agent.run('hello', event_stream_handler=downstream)
        assert len(received_by_observer) == 1
        assert len(received_downstream) > 1

    async def test_not_spec_serializable(self):
        """ProcessEventStream holds a callable so it cannot participate in spec-based construction."""
        assert ProcessEventStream.get_serialization_name() is None


class TestWrapRunShortCircuit:
    """Test short-circuiting wrap_run via iter() and run_stream()."""

    async def test_wrap_run_short_circuit_via_iter(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                return AgentRunResult(output='short-circuited')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[ShortCircuitRunCap()])
        async with agent.iter('hello') as agent_run:
            nodes: list[Any] = []
            async for node in agent_run:
                nodes.append(node)  # pragma: no cover
        # Iteration should stop immediately (no graph nodes executed)
        assert nodes == []
        assert agent_run.result is not None
        assert agent_run.result.output == 'short-circuited'

    async def test_wrap_run_short_circuit_via_run_stream(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                return AgentRunResult(output='short-circuited')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ShortCircuitRunCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'short-circuited'


class TestSkipModelRequestInteraction:
    """Test SkipModelRequest interaction with after_model_request."""

    async def test_skip_model_request_still_calls_after_model_request(self):
        log: list[str] = []

        @dataclass
        class SkipAndLogCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append('before_model_request')
                raise SkipModelRequest(ModelResponse(parts=[TextPart(content='skipped')]))

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append('after_model_request')
                return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SkipAndLogCap()])
        result = await agent.run('hello')
        assert result.output == 'skipped'
        # after_model_request should still fire via _finish_handling
        assert 'after_model_request' in log

    async def test_wrap_model_request_short_circuit_streaming(self):
        """wrap_model_request can return without calling handler in streaming path."""

        @dataclass
        class ShortCircuitModelCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                # Don't call handler — return a response directly
                return ModelResponse(parts=[TextPart(content='model short-circuited')])

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[ShortCircuitModelCap()],
        )
        async with agent.run_stream('hello') as stream:
            output = await stream.get_output()
        assert output == 'model short-circuited'


class TestPrepareToolsHook:
    async def test_filter_function_tools(self):
        """Capability can filter out function tools by name."""

        @dataclass
        class HideToolCap(AbstractCapability[Any]):
            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [td for td in tool_defs if td.name != 'hidden_tool']

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f'tools: {sorted(tool_names)}')

        agent = Agent(FunctionModel(model_fn), capabilities=[HideToolCap()])

        @agent.tool_plain
        def hidden_tool() -> str:
            return 'hidden'  # pragma: no cover

        @agent.tool_plain
        def visible_tool() -> str:
            return 'visible'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == "tools: ['visible_tool']"

    async def test_receives_function_tools_only(self):
        """`prepare_tools` receives **function** tools only. Output tools route to
        `prepare_output_tools` (with `ctx.max_retries` reflecting the output retry budget)."""

        @dataclass
        class CountKindsCap(AbstractCapability[Any]):
            seen_kinds: list[str] = field(default_factory=list[str])

            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                self.seen_kinds = sorted({td.kind for td in tool_defs})
                return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        cap = CountKindsCap()
        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        await agent.run('hello')
        assert cap.seen_kinds == ['function']

    async def test_modify_tool_description(self):
        """Capability can modify tool descriptions."""
        from dataclasses import replace as dc_replace

        @dataclass
        class PrefixDescriptionCap(AbstractCapability[Any]):
            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [dc_replace(td, description=f'[PREFIXED] {td.description}') for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs = [t.description for t in info.function_tools]
            return make_text_response(f'descriptions: {descs}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrefixDescriptionCap()])

        @agent.tool_plain
        def my_tool() -> str:
            """Original description."""
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        assert '[PREFIXED] Original description.' in result.output

    async def test_chaining_order(self):
        """Multiple capabilities chain prepare_tools in forward order."""

        @dataclass
        class AddSuffixCap(AbstractCapability[Any]):
            suffix: str

            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                from dataclasses import replace as dc_replace

                return [dc_replace(td, description=f'{td.description}{self.suffix}') for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs = [t.description for t in info.function_tools]
            return make_text_response(f'{descs}')

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[AddSuffixCap(suffix='_A'), AddSuffixCap(suffix='_B')],
        )

        @agent.tool_plain
        def tool() -> str:
            """desc"""
            return 'r'  # pragma: no cover

        result = await agent.run('hello')
        # A runs first, then B, so suffix order is _A_B
        assert 'desc_A_B' in result.output


class TestPrepareOutputToolsHook:
    async def test_only_receives_output_tools(self):
        """`prepare_output_tools` receives only output tools — function tools route to
        `prepare_tools`."""

        @dataclass
        class CountKindsCap(AbstractCapability[Any]):
            seen_kinds: list[str] = field(default_factory=list[str])

            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                self.seen_kinds = [td.kind for td in tool_defs]
                return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        cap = CountKindsCap()
        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        await agent.run('hello')
        assert cap.seen_kinds == ['output'], f'expected only output tools, got {cap.seen_kinds}'

    async def test_filter_output_tools(self):
        """Capability can hide output tools from the model."""

        class Out(BaseModel):
            value: str

        @dataclass
        class HideCap(AbstractCapability[Any]):
            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return []  # hide all output tools

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response(f'output_tools: {len(info.output_tools)}')

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, ToolOutput(Out, name='out')],
            capabilities=[HideCap()],
        )

        result = await agent.run('hello')
        assert result.output == 'output_tools: 0'

    async def test_run_context_carries_output_max_retries(self):
        """`prepare_output_tools` ctx.max_retries reflects the agent-level output retry budget,
        matching the contract of output hooks (and unlike `prepare_tools` which doesn't have
        a tool-specific retry budget at preparation time)."""
        seen: list[tuple[int, int]] = []

        @dataclass
        class CaptureCtxCap(AbstractCapability[Any]):
            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                seen.append((ctx.retry, ctx.max_retries))
                return tool_defs

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 7}', tool_call_id='c1')]
            )

        agent = Agent(
            FunctionModel(model_fn),
            output_type=MyOutput,
            retries={'tools': 4, 'output': 4},
            capabilities=[CaptureCtxCap()],
        )
        await agent.run('hello')
        assert seen == [(0, 4)]

    async def test_chaining_order(self):
        """Multiple capabilities chain `prepare_output_tools` in forward order."""
        from dataclasses import replace as dc_replace

        @dataclass
        class AddSuffixCap(AbstractCapability[Any]):
            suffix: str

            async def prepare_output_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [dc_replace(td, description=f'{td.description or ""}{self.suffix}') for td in tool_defs]

        descs: list[str | None] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs.extend(t.description for t in info.output_tools)
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.output_tools[0].name, args='{"value": 1}', tool_call_id='c1')]
            )

        agent = Agent(
            FunctionModel(model_fn),
            output_type=MyOutput,
            capabilities=[AddSuffixCap(suffix='_A'), AddSuffixCap(suffix='_B')],
        )
        await agent.run('hello')
        assert descs and descs[0] is not None and descs[0].endswith('_A_B')


class TestWrapNodeRunHook:
    async def test_observe_nodes(self):
        """wrap_node_run can observe all nodes in the agent run."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']

    async def test_observe_nodes_with_tools(self):
        """wrap_node_run fires for each node including tool call round-trips."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('hello')
        # UserPrompt -> ModelRequest (calls tool) -> CallTools (executes tool) ->
        # ModelRequest (gets final response) -> CallTools (produces End)
        assert cap.nodes == [
            'UserPromptNode',
            'ModelRequestNode',
            'CallToolsNode',
            'ModelRequestNode',
            'CallToolsNode',
        ]

    async def test_works_with_iter_next(self):
        """wrap_node_run fires when driving iter() with next()."""
        from pydantic_graph import End

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']

    async def test_bare_async_for_warns_with_wrap_node_run(self):
        """Using bare async for on iter() warns when a capability has wrap_node_run."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                return await handler(node)  # pragma: no cover — bare async for doesn't call this

        agent = Agent(FunctionModel(simple_model_function), capabilities=[NodeObserverCap()])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            async with agent.iter('hello') as agent_run:
                async for _node in agent_run:
                    pass
        assert len(w) == 1
        assert 'wrap_node_run' in str(w[0].message)

    async def test_works_with_manual_next(self):
        """wrap_node_run fires when using manual next() driving."""
        from pydantic_graph import End

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert cap.nodes == ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode']

    async def test_chaining_nests_correctly(self):
        """Multiple capabilities compose wrap_node_run as nested middleware."""
        log: list[str] = []

        @dataclass
        class OrderedCap(AbstractCapability[Any]):
            name: str

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'{self.name}:before:{type(node).__name__}')
                result = await handler(node)
                log.append(f'{self.name}:after:{type(result).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[OrderedCap(name='outer'), OrderedCap(name='inner')],
        )
        await agent.run('hello')
        # For UserPromptNode: outer wraps inner
        assert log[0] == 'outer:before:UserPromptNode'
        assert log[1] == 'inner:before:UserPromptNode'
        assert log[2] == 'inner:after:ModelRequestNode'
        assert log[3] == 'outer:after:ModelRequestNode'


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
        from pydantic_ai.messages import BinaryImage, FilePart

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

    @pytest.mark.parametrize('model_name', ['gpt-image-2', 'gpt-image-1.5', 'gpt-image-1', 'gpt-image-1-mini'])
    def test_image_generation_rejects_image_only_model(self, model_name: str):
        """Using a dedicated image model like gpt-image-2 raises a clear error at construction."""
        with pytest.raises(UserError, match=f'{model_name!r} is a dedicated image generation model'):
            ImageGeneration(fallback_model=f'openai-responses:{model_name}')

    @pytest.mark.vcr()
    async def test_image_generation_local_fallback(self, allow_model_requests: None, openai_api_key: str):
        """ImageGeneration(fallback_model=...) with non-supporting outer model uses subagent fallback."""
        from pydantic_ai.messages import BinaryImage
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
        from pydantic_ai.messages import BinaryImage
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='Here is the generated image.')])
            assert info.function_tools, 'Expected generate_image tool to be available'
            tool = info.function_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}')])

        inner_model = GoogleModel('gemini-3-pro-image-preview', provider=GoogleProvider(api_key=gemini_api_key))
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
    assert cap._requires_native() is True  # pyright: ignore[reportPrivateUsage]


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


# --- Node run lifecycle hook tests ---


class TestNodeRunHooks:
    async def test_before_node_run_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'before_node_run:UserPromptNode' in cap.log
        assert 'before_node_run:ModelRequestNode' in cap.log
        assert 'before_node_run:CallToolsNode' in cap.log

    async def test_after_node_run_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'after_node_run:UserPromptNode' in cap.log
        assert 'after_node_run:ModelRequestNode' in cap.log
        assert 'after_node_run:CallToolsNode' in cap.log

    async def test_node_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        # For each node, before fires before after
        for node_name in ('UserPromptNode', 'ModelRequestNode', 'CallToolsNode'):
            before_idx = cap.log.index(f'before_node_run:{node_name}')
            after_idx = cap.log.index(f'after_node_run:{node_name}')
            assert before_idx < after_idx


# --- Run error hook tests ---


class TestRunErrorHooks:
    async def test_on_run_error_fires_on_failure(self):
        cap = LoggingCapability()

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')
        assert 'on_run_error' in cap.log

    async def test_on_run_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'on_run_error' not in cap.log

    async def test_on_run_error_can_transform_error(self):
        @dataclass
        class TransformErrorCap(AbstractCapability[Any]):
            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                raise ValueError('transformed error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[TransformErrorCap()])
        with pytest.raises(ValueError, match='transformed error'):
            await agent.run('hello')

    async def test_on_run_error_can_recover(self):
        @dataclass
        class RecoverRunCap(AbstractCapability[Any]):
            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                return AgentRunResult(output='recovered')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[RecoverRunCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered'

    async def test_on_run_error_not_called_when_wrap_run_recovers(self):
        @dataclass
        class WrapRecoveryCap(AbstractCapability[Any]):
            log: list[str] = field(default_factory=lambda: [])

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    self.log.append('wrap_run:caught')
                    return AgentRunResult(output='wrap_recovered')

            async def on_run_error(  # pragma: no cover — verifying this is NOT called
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                self.log.append('on_run_error')
                raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = WrapRecoveryCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'wrap_recovered'
        assert 'wrap_run:caught' in cap.log
        assert 'on_run_error' not in cap.log

    async def test_on_run_error_fires_via_iter(self):
        from pydantic_graph import End

        @dataclass
        class RecoverRunCap(AbstractCapability[Any]):
            called: bool = False

            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                self.called = True
                return AgentRunResult(output='recovered via iter')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = RecoverRunCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):  # pragma: no branch
                node = await agent_run.next(node)
        assert cap.called
        assert agent_run.result is not None
        assert agent_run.result.output == 'recovered via iter'


# --- Node run error hook tests ---


class TestNodeRunErrorHooks:
    async def test_on_node_run_error_fires(self):
        cap = LoggingCapability()

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')
        assert 'on_node_run_error:ModelRequestNode' in cap.log

    async def test_on_node_run_error_can_recover_with_end(self):
        from pydantic_ai.result import FinalResult
        from pydantic_graph import End

        @dataclass
        class RecoverNodeCap(AbstractCapability[Any]):
            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: BaseException) -> Any:
                return End(FinalResult(output='recovered'))

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = RecoverNodeCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)
        assert isinstance(node, End)
        assert node.data.output == 'recovered'

    async def test_on_node_run_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert all('on_node_run_error' not in entry for entry in cap.log)


# --- Model request error hook tests ---


class TestModelRequestErrorHooks:
    async def test_on_model_request_error_fires(self):
        cap = LoggingCapability()

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')
        assert 'on_model_request_error' in cap.log

    async def test_on_model_request_error_can_recover(self):
        @dataclass
        class RecoverModelCap(AbstractCapability[Any]):
            async def on_model_request_error(
                self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content='recovered response')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[RecoverModelCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered response'

    async def test_on_model_request_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run('hello')
        assert 'on_model_request_error' not in cap.log

    async def test_default_on_model_request_error_reraises(self):
        """Default on_model_request_error re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            def get_instructions(self):
                return 'Be helpful.'

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[MinimalCap()])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_default_on_model_request_error_reraises_streaming(self):
        """Default on_model_request_error re-raises in streaming path (wrap_task error after stream consumed)."""

        @dataclass
        class PostProcessFailCap(AbstractCapability[Any]):
            """wrap_model_request that fails AFTER handler returns (post-processing error)."""

            def get_instructions(self):
                return 'Be helpful.'

            async def wrap_model_request(self, ctx: RunContext[Any], *, request_context: Any, handler: Any) -> Any:
                await handler(request_context)
                raise RuntimeError('post-processing exploded')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[PostProcessFailCap()],
        )
        with pytest.raises(RuntimeError, match='post-processing exploded'):
            async with agent.run_stream('hello') as stream:
                await stream.get_output()


# --- Tool validate error hook tests ---


class TestToolValidateErrorHooks:
    async def test_on_tool_validate_error_fires_on_validation_failure(self):
        cap = LoggingCapability()

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "correct"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[cap])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        await agent.run('greet someone')
        assert 'on_tool_validate_error:greet' in cap.log

    async def test_on_tool_validate_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert all('on_tool_validate_error' not in entry for entry in cap.log)

    async def test_on_tool_validate_error_can_recover(self):
        @dataclass
        class RecoverValidateCap(AbstractCapability[Any]):
            async def on_tool_validate_error(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
            ) -> dict[str, Any]:
                return {'name': 'recovered-name'}

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[RecoverValidateCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert received_name == 'recovered-name'
        assert 'hello recovered-name' in result.output

    async def test_default_on_tool_validate_error_reraises(self):
        """The default on_tool_validate_error re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            def get_instructions(self):
                return 'Be helpful.'

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "correct"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert 'hello correct' in result.output


# --- Tool execute error hook tests ---


class TestToolExecuteErrorHooks:
    async def test_on_tool_execute_error_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call the tool')
        assert 'on_tool_execute_error:my_tool' in cap.log

    async def test_on_tool_execute_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        await agent.run('call the tool')
        assert all('on_tool_execute_error' not in entry for entry in cap.log)

    async def test_on_tool_execute_error_can_recover(self):
        @dataclass
        class RecoverExecCap(AbstractCapability[Any]):
            async def on_tool_execute_error(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'fallback result'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[RecoverExecCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert 'fallback result' in result.output


# --- Hooks capability tests ---


class TestHooksCapability:
    """Tests for the Hooks decorator-based capability."""

    async def test_decorator_registration(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        @hooks.on.after_model_request
        async def log_response(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
        ) -> ModelResponse:
            call_log.append('after_model_request')
            return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['before_model_request', 'after_model_request']

    async def test_constructor_form(self):
        call_log: list[str] = []

        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[Hooks(before_model_request=log_request)])
        await agent.run('hello')
        assert call_log == ['before_model_request']

    async def test_multiple_hooks_same_event(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def first(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('first')
            return request_context

        @hooks.on.before_model_request
        async def second(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('second')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['first', 'second']

    async def test_tool_names_filtering(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_tool_execute(tools=['target_tool'])
        async def filtered(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            call_log.append(f'filtered:{call.tool_name}')
            return args

        @hooks.on.after_tool_execute
        async def unfiltered(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], result: Any
        ) -> Any:
            call_log.append(f'unfiltered:{call.tool_name}')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def target_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert 'filtered:target_tool' in call_log
        assert 'unfiltered:target_tool' in call_log

    async def test_wrap_model_request(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.model_request
        async def wrap(ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any) -> ModelResponse:
            call_log.append('wrap_start')
            result = await handler(request_context)
            call_log.append('wrap_end')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['wrap_start', 'wrap_end']

    async def test_wrap_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.run
        async def wrap(ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            call_log.append('wrap_run_start')
            result = await handler()
            call_log.append('wrap_run_end')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['wrap_run_start', 'wrap_run_end']

    async def test_on_error_recovery(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def recover(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='recovered')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'recovered'

    async def test_sync_function_auto_wrapping(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        def sync_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('sync_hook')
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['sync_hook']

    async def test_timeout(self):
        hooks = Hooks()

        @hooks.on.before_model_request(timeout=0.01)
        async def slow_hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            await asyncio.sleep(10)
            return request_context  # pragma: no cover

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        with pytest.raises(HookTimeoutError) as exc_info:
            await agent.run('hello')
        assert exc_info.value.hook_name == 'before_model_request'
        assert exc_info.value.func_name == 'slow_hook'
        assert exc_info.value.timeout == 0.01

    async def test_has_wrap_node_run(self):
        hooks = Hooks()
        assert hooks.has_wrap_node_run is False

        nodes_seen: list[str] = []

        @hooks.on.node_run
        async def wrap(ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
            nodes_seen.append(type(node).__name__)
            return await handler(node)

        assert hooks.has_wrap_node_run is True

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert len(nodes_seen) > 0

    async def test_composition_with_other_capabilities(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def hooks_before(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('hooks_before')
            return request_context

        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks, cap])
        await agent.run('hello')
        assert 'hooks_before' in call_log
        assert 'before_model_request' in cap.log

    async def test_before_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_run
        async def on_start(ctx: RunContext[Any]) -> None:
            call_log.append('before_run')

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert call_log == ['before_run']

    async def test_after_run(self):
        hooks = Hooks()
        outputs: list[str] = []

        @hooks.on.after_run
        async def on_end(ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            outputs.append(result.output)
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run('hello')
        assert outputs == [result.output]

    async def test_repr(self):
        hooks = Hooks()
        assert repr(hooks) == 'Hooks({})'

        @hooks.on.before_model_request
        async def hook(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            return request_context

        assert repr(hooks) == "Hooks({'before_model_request': 1})"

        # Verify the registered hook actually works
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')

    async def test_on_model_request_error_reraise(self):
        """Error hooks that re-raise propagate the error to the caller."""

        hooks = Hooks()

        @hooks.on.model_request_error
        async def log_and_reraise(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_on_run_error_reraise(self):
        """on_run_error hooks that re-raise propagate the error."""

        hooks = Hooks()

        @hooks.on.run_error
        async def log_and_reraise(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='model exploded'):
            await agent.run('hello')

    async def test_on_run_error_recovery(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def recover(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output='recovered from run error')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'recovered from run error'

    async def test_on_run_error_chaining(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def first_handler(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            raise ValueError('transformed by first')

        @hooks.on.run_error
        async def second_handler(ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
            return AgentRunResult(output=f'caught: {error}')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('original error')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert 'transformed by first' in result.output

    async def test_error_hook_chaining(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def first(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise ValueError('transformed')

        @hooks.on.model_request_error
        async def second(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=f'recovered: {error}')])

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('original')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run('hello')
        assert 'transformed' in result.output

    async def test_wrap_run_event_stream(self):
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.run_event_stream
        async def observe_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                events_seen.append(type(event).__name__)
                yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_hooks_with_streaming_run(self):
        """Hooks capability used during a streaming run exercises the default wrap_run_event_stream path."""

        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(ctx: RunContext[Any], request_context: ModelRequestContext) -> ModelRequestContext:
            call_log.append('before_model_request')
            return request_context

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert 'before_model_request' in call_log

    async def test_node_run_hooks(self):
        """Exercise before_node_run, after_node_run, and node_run (wrap) via .on namespace."""
        hooks = Hooks()
        nodes_seen: list[str] = []

        @hooks.on.before_node_run
        async def before(ctx: RunContext[Any], *, node: Any) -> Any:
            nodes_seen.append(f'before:{type(node).__name__}')
            return node

        @hooks.on.after_node_run
        async def after(ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            nodes_seen.append(f'after:{type(node).__name__}')
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run('hello')
        assert any('before:' in n for n in nodes_seen)
        assert any('after:' in n for n in nodes_seen)

    async def test_node_run_error_hook(self):
        """on.node_run_error fires when a node fails."""
        hooks = Hooks()
        error_log: list[str] = []

        @hooks.on.node_run_error
        async def handle(ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            error_log.append(f'error:{type(error).__name__}')
            raise error

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('node exploded')

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match='node exploded'):
            await agent.run('hello')
        assert any('error:RuntimeError' in e for e in error_log)

    async def test_on_event_hook(self):
        """on.event fires for each stream event and can modify events."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_on_event_hook_fires_in_run(self):
        """on.event fires in run() even without an event_stream_handler."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert result.output is not None
        assert 'PartStartEvent' in events_seen

    async def test_wrap_run_event_stream_fires_in_run(self):
        """on.run_event_stream fires in run() even without an event_stream_handler."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.run_event_stream
        async def observe_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            async for event in stream:
                events_seen.append(type(event).__name__)
                yield event

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert result.output is not None
        assert 'PartStartEvent' in events_seen

    async def test_on_event_with_run_event_stream(self):
        """on.event and on.run_event_stream can be used together."""
        hooks = Hooks()
        event_log: list[str] = []
        stream_log: list[str] = []

        @hooks.on.event
        async def per_event(ctx: RunContext[Any], event: AgentStreamEvent) -> AgentStreamEvent:
            event_log.append(type(event).__name__)
            return event

        @hooks.on.run_event_stream
        async def wrap_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            stream_log.append('started')
            async for event in stream:
                yield event
            stream_log.append('finished')

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[hooks],
        )
        async with agent.run_stream('hello') as stream:
            await stream.get_output()
        assert len(event_log) > 0
        assert stream_log == ['started', 'finished']

    async def test_prepare_tools_hook(self):
        """on.prepare_tools filters tool definitions."""
        hooks = Hooks()

        @hooks.on.prepare_tools
        async def hide_tools(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if not td.name.startswith('hidden_')]

        tool_called = False

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def visible_tool() -> str:
            nonlocal tool_called
            tool_called = True
            return 'visible'

        @agent.tool_plain
        def hidden_tool() -> str:
            return 'hidden'  # pragma: no cover

        await agent.run('call tool')
        assert tool_called

    async def test_prepare_output_tools_hook(self):
        """`on.prepare_output_tools` filters output tool definitions — model only sees the
        non-filtered ones."""
        hooks = Hooks()

        @hooks.on.prepare_output_tools
        async def hide_secret(ctx: RunContext[Any], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
            return [td for td in tool_defs if td.name != 'secret_output']

        seen_output_tools: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            seen_output_tools.extend(td.name for td in info.output_tools)
            # Call the only remaining (non-filtered) output tool
            return ModelResponse(parts=[ToolCallPart('public_output', {'value': 'ok'})])

        class SecretOutput(BaseModel):
            value: str

        class PublicOutput(BaseModel):
            value: str

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[
                ToolOutput(SecretOutput, name='secret_output'),
                ToolOutput(PublicOutput, name='public_output'),
            ],
            capabilities=[hooks],
        )
        result = await agent.run('hello')
        assert isinstance(result.output, PublicOutput)
        assert seen_output_tools == ['public_output']

    async def test_tool_validate_hooks(self):
        """Exercise before/after/wrap tool_validate and on_tool_validate_error."""
        hooks = Hooks()
        validate_log: list[str] = []

        @hooks.on.before_tool_validate
        async def before_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any
        ) -> Any:
            validate_log.append('before_validate')
            return args

        @hooks.on.after_tool_validate
        async def after_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
        ) -> dict[str, Any]:
            validate_log.append('after_validate')
            return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert 'before_validate' in validate_log
        assert 'after_validate' in validate_log

    async def test_wrap_tool_validate_hook(self):
        """Exercise on.tool_validate (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_validate
        async def wrap_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, handler: Any
        ) -> dict[str, Any]:
            wrap_log.append('wrap_start')
            result = await handler(args)
            wrap_log.append('wrap_end')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert wrap_log == ['wrap_start', 'wrap_end']

    async def test_tool_validate_error_hook(self):
        """on.tool_validate_error can recover from validation failures."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def recover_validate(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            return {'name': 'recovered'}

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        result = await agent.run('greet someone')
        assert 'hello recovered' in result.output

    async def test_wrap_tool_execute_hook(self):
        """Exercise on.tool_execute (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_execute
        async def wrap_exec(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any], handler: Any
        ) -> Any:
            wrap_log.append('exec_start')
            result = await handler(args)
            wrap_log.append('exec_end')
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'

        await agent.run('call tool')
        assert wrap_log == ['exec_start', 'exec_end']

    async def test_tool_execute_error_hook(self):
        """on.tool_execute_error can recover from tool execution failures."""
        hooks = Hooks()

        @hooks.on.tool_execute_error
        async def recover_exec(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'fallback result'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert 'fallback result' in result.output

    async def test_tool_validate_error_reraise(self):
        """on.tool_validate_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def reraise(
            ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: Any, error: Any
        ) -> dict[str, Any]:
            raise error

        call_count = 0

        def bad_args_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f'got: {part.content}')
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"wrong": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"name": "ok"}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f'hello {name}'

        await agent.run('greet someone')

    async def test_tool_execute_error_reraise(self):
        """on.tool_execute_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_execute_error
        async def reraise(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            error: Exception,
        ) -> Any:
            raise error

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call tool')

    async def test_get_serialization_name(self):
        assert Hooks.get_serialization_name() is None

    async def test_default_on_tool_execute_error_reraises(self):
        """The default on_tool_execute_error just re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            """Capability that doesn't override error hooks."""

            def get_instructions(self):
                return 'Be helpful.'

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        with pytest.raises(ValueError, match='tool failed'):
            await agent.run('call the tool')


# --- Context var propagation tests ---

_test_cv: contextvars.ContextVar[str] = contextvars.ContextVar('_test_cv')


class TestContextVarPropagation:
    """Context vars set in wrap_run propagate to all hooks in the outer task."""

    async def test_wrap_run_contextvar_visible_in_node_hooks(self):
        """A capability that sets a contextvar in wrap_run should have it
        visible in another capability's node-level hooks via agent.run()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('from-wrap-run')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                self.seen.append(('wrap_node_run', _test_cv.get(None)))
                return await handler(node)

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                self.seen.append(('after_node_run', _test_cv.get(None)))
                return result

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                self.seen.append(('after_run', _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run('hello')

        for hook_name, value in reader.seen:
            assert value == 'from-wrap-run', f'{hook_name} did not see contextvar'

    async def test_wrap_run_contextvar_visible_via_iter_next(self):
        """Context vars set in wrap_run are visible when using agent.iter() + next()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('from-iter')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

            async def after_run(self, ctx: RunContext[Any], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
                self.seen.append(('after_run', _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])

        async with agent.iter('hello') as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        for hook_name, value in reader.seen:
            assert value == 'from-iter', f'{hook_name} did not see contextvar'

    async def test_contextvar_cleaned_up_after_run(self):
        """Context vars set in wrap_run are restored after the run completes."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('temporary')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        await agent.run('hello')

        # After the run, the contextvar should be cleaned up
        assert _test_cv.get(None) is None

    async def test_contextvar_cleaned_up_on_early_iter_exit(self):
        """Context vars are restored even when the caller exits iter() early."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('early-exit')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        async with agent.iter('hello') as agent_run:
            # Exit immediately without driving any nodes
            _ = agent_run.next_node

        # Context var must be cleaned up even though we abandoned the run
        assert _test_cv.get(None) is None

    async def test_before_run_contextvar_propagates(self):
        """Context vars set in before_run (not wrap_run) also propagate."""

        @dataclass
        class Setter(AbstractCapability):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                _test_cv.set('from-before-run')

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(('before_node_run', _test_cv.get(None)))
                return node

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run('hello')

        for hook_name, value in reader.seen:
            assert value == 'from-before-run', f'{hook_name} did not see contextvar'

    async def test_contextvar_visible_in_on_run_error(self):
        """Context vars set in wrap_run are visible in on_run_error."""

        @dataclass
        class SetterWithRecovery(AbstractCapability):
            seen_in_error: str | None = None

            async def wrap_run(self, ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
                token = _test_cv.set('error-path')
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

            async def on_run_error(self, ctx: RunContext[Any], *, error: BaseException) -> AgentRunResult[Any]:
                self.seen_in_error = _test_cv.get(None)
                return AgentRunResult(output='recovered')

        def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model exploded')

        cap = SetterWithRecovery()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run('hello')

        assert result.output == 'recovered'
        assert cap.seen_in_error == 'error-path'


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
    available_per_turn: list[set[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        available_per_turn.append({td.name for td in info.function_tools if not td.defer_loading})

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
        'The following capabilities are deferred and can be loaded using the `load_capability` tool:\n'
        '- refunds: Refund policy tools.'
    ]
    # The prefixed tool is hidden until the capability is loaded, then becomes callable.
    assert 'refunds_lookup_refund_policy' not in available_per_turn[0]
    assert 'refunds_lookup_refund_policy' in available_per_turn[-1]


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
            [('load_capability', False), ('billing_lookup_refund_policy', True), ('search_tools', False)],
            [('load_capability', False), ('billing_lookup_refund_policy', False), ('search_tools', False)],
            [('load_capability', False), ('billing_lookup_refund_policy', False), ('search_tools', False)],
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
    assert WrapperCapability(wrapped=plain).has_wrap_node_run is False

    @dataclass
    class NodeRunCap(AbstractCapability):
        async def wrap_node_run(self, ctx: RunContext, *, node: Any, handler: Any) -> Any:
            return await handler(node)  # pragma: no cover

    assert WrapperCapability(wrapped=NodeRunCap()).has_wrap_node_run is True


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
    from pydantic_ai.result import FinalResult
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


# --- Tests for double-execution bug fix (streaming + before_node_run replacement) ---


class TestNodeStreamingWithHooks:
    """Tests that node streaming with event_stream_handler doesn't cause double model execution
    when before_node_run replaces a node."""

    async def test_before_node_run_replacement_no_double_execution(self):
        """When before_node_run replaces a ModelRequestNode and event_stream_handler is set,
        the model should be called exactly once (not twice)."""
        model_call_count = 0

        async def counting_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield 'streamed response'

        cap = _ReplacingCapability()
        agent = Agent(FunctionModel(simple_model_function, stream_function=counting_stream), capabilities=[cap])

        events_received: list[AgentStreamEvent] = []

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events_received.append(event)

        result = await agent.run('hello', event_stream_handler=handler)
        assert result.output == 'streamed response'
        assert model_call_count == 1, f'Model was called {model_call_count} times, expected 1'
        assert len(events_received) > 0

    async def test_hook_ordering_with_event_stream_handler(self):
        """before_node_run fires BEFORE streaming events, wrap_node_run wraps the streaming,
        and after_node_run fires after graph advancement."""
        log: list[str] = []

        @dataclass
        class OrderTrackingCapability(AbstractCapability[Any]):
            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                log.append(f'before:{type(node).__name__}')
                return node

            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                log.append(f'wrap:enter:{type(node).__name__}')
                result = await handler(node)
                log.append(f'wrap:exit:{type(node).__name__}')
                return result

            async def after_node_run(self, ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
                log.append(f'after:{type(node).__name__}')
                return result

        agent = Agent(
            FunctionModel(simple_model_function, stream_function=simple_stream_function),
            capabilities=[OrderTrackingCapability()],
        )

        async def handler(_ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass
            log.append('stream:consumed')

        await agent.run('hello', event_stream_handler=handler)

        # For ModelRequestNode: before → wrap:enter → stream:consumed → wrap:exit → after
        mr_before = log.index('before:ModelRequestNode')
        mr_wrap_enter = log.index('wrap:enter:ModelRequestNode')
        stream_consumed_idx = log.index('stream:consumed')
        mr_wrap_exit = log.index('wrap:exit:ModelRequestNode')
        mr_after = log.index('after:ModelRequestNode')
        assert mr_before < mr_wrap_enter < stream_consumed_idx < mr_wrap_exit < mr_after

    async def test_run_stream_before_node_run_replacement_no_double_execution(self):
        """Same as the run() test but for run_stream(): before_node_run replacement
        should not cause double model execution."""
        model_call_count = 0

        async def counting_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield 'streamed response'

        cap = _ReplacingCapability()
        agent = Agent(FunctionModel(simple_model_function, stream_function=counting_stream), capabilities=[cap])

        async with agent.run_stream('hello') as streamed:
            output = await streamed.get_output()

        assert output == 'streamed response'
        assert model_call_count == 1, f'Model was called {model_call_count} times, expected 1'

    async def test_on_node_run_error_fires_in_run_stream(self):
        """on_node_run_error in run_stream() fires when wrap_node_run raises during graph advancement."""
        error_log: list[str] = []

        @dataclass
        class WrapErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(self, ctx: RunContext[Any], *, node: Any, handler: Any) -> Any:
                # Raise on CallToolsNode — after UserPromptNode and ModelRequestNode pass through.
                # ModelRequestNode with tool calls doesn't produce a FinalResultEvent in run_stream(),
                # so it falls through to wrap_node_run; CallToolsNode is next and triggers the error.
                from pydantic_ai._agent_graph import CallToolsNode

                if isinstance(node, CallToolsNode):
                    raise RuntimeError('wrap error')
                return await handler(node)

            async def on_node_run_error(self, ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
                error_log.append(type(node).__name__)
                raise error

        agent = Agent(
            FunctionModel(tool_calling_model, stream_function=tool_calling_stream_function),
            capabilities=[WrapErrorCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        with pytest.raises(RuntimeError, match='wrap error'):
            async with agent.run_stream('hello') as _streamed:
                pass

        assert error_log == ['CallToolsNode']


# --- ToolFailed and ModelRetry from hooks tests ---


class _BeforeToolFailedCap(AbstractCapability[Any]):
    async def before_tool_execute(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
    ) -> dict[str, Any]:
        raise ToolFailed('failed before execution')


class _WrapToolFailedCap(AbstractCapability[Any]):
    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        try:
            return await handler(args)
        except RuntimeError as e:
            raise ToolFailed('failed during wrapper') from e


class _AfterToolFailedCap(AbstractCapability[Any]):
    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        raise ToolFailed('failed after execution')


class _OnToolExecuteErrorFailedCap(AbstractCapability[Any]):
    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        raise ToolFailed('failed while handling error')


class _BeforeToolValidateFailedCap(AbstractCapability[Any]):
    async def before_tool_validate(
        self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: str | dict[str, Any]
    ) -> str | dict[str, Any]:
        raise ToolFailed('failed before validation')


class _OnToolValidateErrorFailedCap(AbstractCapability[Any]):
    async def on_tool_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        error: ValidationError | ModelRetry,
    ) -> dict[str, Any]:
        raise ToolFailed('failed while handling validation error')


class _WrapToolValidateFailedCap(AbstractCapability[Any]):
    async def wrap_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
        handler: Any,
    ) -> dict[str, Any]:
        raise ToolFailed('failed during validate wrapper')


def _tool_failed_roundtrip_model(tool_args: str) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return make_text_response(f'got: {part.outcome}:{part.content}')
        if info.function_tools:
            return ModelResponse(
                parts=[ToolCallPart(tool_name=info.function_tools[0].name, args=tool_args, tool_call_id='call-1')]
            )
        return make_text_response('no tools')  # pragma: no cover

    return model_fn


def _assert_failed_tool_result(result: AgentRunResult[Any], expected_message: str) -> None:
    assert result.output == f'got: failed:{expected_message}'

    parts = [part for msg in result.all_messages() for part in msg.parts]
    tool_return = next(part for part in parts if isinstance(part, ToolReturnPart))
    assert tool_return.outcome == 'failed'
    assert tool_return.content == expected_message
    assert not any(isinstance(part, RetryPromptPart) for part in parts)


class TestToolFailedFromHooks:
    """Tests for raising ToolFailed from capability tool hooks."""

    @pytest.mark.parametrize('hook_name', ['before', 'wrap', 'after', 'on_error'])
    async def test_tool_execute_hook_tool_failed(self, hook_name: str):
        tool_call_count = 0
        cap_type, expected_message, tool_should_run = {
            'before': (_BeforeToolFailedCap, 'failed before execution', False),
            'wrap': (_WrapToolFailedCap, 'failed during wrapper', True),
            'after': (_AfterToolFailedCap, 'failed after execution', True),
            'on_error': (_OnToolExecuteErrorFailedCap, 'failed while handling error', True),
        }[hook_name]

        agent = Agent(FunctionModel(_tool_failed_roundtrip_model('{}')), capabilities=[cap_type()])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            if hook_name in {'wrap', 'on_error'}:
                raise RuntimeError('tool failed')
            return 'tool result'

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, expected_message)
        assert tool_call_count == int(tool_should_run)

    async def test_deferred_tool_validate_hook_tool_failed(self):
        """Deferred tool validation can return a failed tool result instead of a deferred request."""
        tool_call_count = 0

        agent = Agent(
            FunctionModel(_tool_failed_roundtrip_model('{}')),
            capabilities=[_BeforeToolValidateFailedCap()],
            output_type=[str, DeferredToolRequests],
            retries={'tools': 0, 'output': 2},
        )

        @agent.tool_plain(requires_approval=True)
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, 'failed before validation')
        assert tool_call_count == 0

    @pytest.mark.parametrize(
        ('capability', 'tool_args', 'expected_message'),
        [
            pytest.param(
                _BeforeToolValidateFailedCap(), '{"x":1}', 'failed before validation', id='before_tool_validate'
            ),
            pytest.param(
                _OnToolValidateErrorFailedCap(),
                '{"x":"bad"}',
                'failed while handling validation error',
                id='on_tool_validate_error',
            ),
            pytest.param(
                _WrapToolValidateFailedCap(), '{"x":1}', 'failed during validate wrapper', id='wrap_tool_validate'
            ),
        ],
    )
    async def test_tool_validate_hook_tool_failed(
        self, capability: AbstractCapability[Any], tool_args: str, expected_message: str
    ):
        """Non-deferred tool validation hooks can report a failed tool result instead of retrying."""
        tool_call_count = 0

        agent = Agent(
            FunctionModel(_tool_failed_roundtrip_model(tool_args)),
            capabilities=[capability],
            retries={'tools': 0, 'output': 2},
        )

        @agent.tool_plain
        def my_tool(x: int) -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return f'tool result: {x}'  # pragma: no cover

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, expected_message)
        assert tool_call_count == 0

    async def test_args_validator_tool_failed(self):
        """An `args_validator` raising `ToolFailed` reports a failed tool result instead of retrying."""
        tool_call_count = 0
        expected_message = 'failed in args validator'

        def validate_args(ctx: RunContext[Any]) -> None:
            raise ToolFailed(expected_message)

        agent = Agent(
            FunctionModel(_tool_failed_roundtrip_model('{}')),
            retries={'tools': 0, 'output': 2},
        )

        @agent.tool_plain(args_validator=validate_args)
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1  # pragma: no cover
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')

        _assert_failed_tool_result(result, expected_message)
        assert tool_call_count == 0


class TestModelRetryFromHooks:
    """Tests for raising ModelRetry from capability hooks."""

    async def test_after_model_request_model_retry(self):
        """after_model_request raises ModelRetry — model is called again with retry prompt."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_text_response('bad response')
            return make_text_response('good response')

        @dataclass
        class RetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Response was bad, please try again')
                return response

        cap = RetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'good response'
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='bad response')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Response was bad, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=66, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_model_request_model_retry_max_retries(self):
        """after_model_request raises ModelRetry repeatedly — hits output_retries."""

        @dataclass
        class AlwaysRetryCap(AbstractCapability[Any]):
            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                raise ModelRetry('always bad')

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[AlwaysRetryCap()],
            retries={'output': 2},
        )
        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
            await agent.run('hello')

    async def test_after_model_request_model_retry_streaming(self):
        """after_model_request raises ModelRetry during streaming with tool calls — model is called again."""
        call_count = 0

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return a tool call that after_model_request will reject
                yield {0: DeltaToolCall(name='my_tool', json_args='{}', tool_call_id='call-1')}
            elif call_count == 2:
                # Second call (after retry): return text
                yield 'good response'
            else:
                yield 'unexpected'  # pragma: no cover

        @dataclass
        class RetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Response was bad, please try again')
                return response

        cap = RetryCap()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=stream_fn),
            capabilities=[cap],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        async with agent.run_stream('hello') as streamed:
            result = await streamed.get_output()
        assert result == 'good response'
        assert call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Response was bad, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_streaming_short_circuit(self):
        """wrap_model_request raises ModelRetry without calling handler during streaming."""

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield 'good response'

        @dataclass
        class ShortCircuitRetryCap(AbstractCapability[Any]):
            call_count: int = 0

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                self.call_count += 1
                if self.call_count == 1:
                    # Short-circuit: don't call handler, raise ModelRetry
                    raise ModelRetry('Short-circuit retry')
                return await handler(request_context)

        cap = ShortCircuitRetryCap()
        agent = Agent(FunctionModel(simple_model_function, stream_function=stream_fn), capabilities=[cap])
        async with agent.run_stream('hello') as streamed:
            result = await streamed.get_output()
        assert result == 'good response'
        assert cap.call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Short-circuit retry',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_streaming_after_handler(self):
        """wrap_model_request raises ModelRetry after calling handler during streaming (tool call scenario)."""
        call_count = 0

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tool call that wrap hook will reject
                yield {0: DeltaToolCall(name='my_tool', json_args='{}', tool_call_id='call-1')}
            else:
                yield 'good response'

        @dataclass
        class AfterHandlerRetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                response = await handler(request_context)
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Post-handler retry')
                return response

        cap = AfterHandlerRetryCap()
        agent = Agent(FunctionModel(simple_model_function, stream_function=stream_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        async with agent.run_stream('hello') as streamed:
            result = await streamed.get_output()
        assert result == 'good response'
        assert call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Post-handler retry',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good response')],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name='function:simple_model_function:stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry(self):
        """wrap_model_request raises ModelRetry after calling handler — triggers retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_text_response('first attempt')
            return make_text_response('second attempt')

        @dataclass
        class WrapRetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                response = await handler(request_context)
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Wrap says retry')
                return response

        cap = WrapRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'second attempt'
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='first attempt')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Wrap says retry',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='second attempt')],
                    usage=RequestUsage(input_tokens=63, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_skips_on_error(self):
        """wrap_model_request raising ModelRetry should NOT call on_model_request_error."""
        on_error_called = False

        @dataclass
        class WrapRetrySkipErrorCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                raise ModelRetry('retry please')

            async def on_model_request_error(  # pragma: no cover — verifying this is NOT called
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                error: Exception,
            ) -> ModelResponse:
                nonlocal on_error_called
                on_error_called = True
                raise error

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[WrapRetrySkipErrorCap()], retries={'output': 1}
        )
        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum output retries'):
            await agent.run('hello')
        assert not on_error_called

    async def test_on_model_request_error_model_retry(self):
        """on_model_request_error raises ModelRetry to recover via retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError('model failed')
            return make_text_response('recovered response')

        @dataclass
        class ErrorRetryCap(AbstractCapability[Any]):
            async def on_model_request_error(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                error: Exception,
            ) -> ModelResponse:
                raise ModelRetry('Model failed, please try again')

        cap = ErrorRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run('hello')
        assert result.output == 'recovered response'
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Model failed, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='recovered response')],
                    usage=RequestUsage(input_tokens=65, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_execute_model_retry(self):
        """after_tool_execute raises ModelRetry — tool retry prompt sent to model, tool retried on success."""
        tool_call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Always call the tool — after retry, the hook won't raise again
            if info.function_tools:
                # Check if we already got a tool return (second call succeeded)
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class AfterExecRetryCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
            ) -> Any:
                if not self.retried:
                    self.retried = True
                    raise ModelRetry('Tool result is bad, try again')
                return result

        cap = AfterExecRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        assert tool_call_count == 2  # Tool called twice: first rejected by hook, second succeeds
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Tool result is bad, try again',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=65, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_execute_model_retry(self):
        """before_tool_execute raises ModelRetry — tool execution is skipped, then succeeds on retry."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Always call the tool — after retry, the hook won't raise again
            if info.function_tools:
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        hooks = Hooks[Any]()
        hook_called = False

        @hooks.on.before_tool_execute
        async def reject_first(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            nonlocal hook_called
            if not hook_called:
                hook_called = True
                raise ModelRetry('Not ready to execute, try again')
            return args

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Not ready to execute, try again',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=65, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_execute_validation_error(self):
        """after_tool_execute raises ValidationError — converted to ToolRetryError for retry."""
        from pydantic import TypeAdapter

        tool_call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.function_tools:
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class ValErrCap(AbstractCapability[Any]):
            retried: bool = False

            async def after_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                result: Any,
            ) -> Any:
                if not self.retried:
                    self.retried = True
                    # Simulate a user hook doing additional Pydantic validation
                    TypeAdapter(int).validate_python('not_an_int')
                return result

        cap = ValErrCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        assert tool_call_count == 2  # Retried after ValidationError
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'int_parsing',
                                    'loc': (),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'not_an_int',
                                }
                            ],
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=88, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=90, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_execute_validation_error(self):
        """before_tool_execute raises ValidationError — converted to ToolRetryError for retry."""
        from pydantic import TypeAdapter

        tool_call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.function_tools:
                for msg in messages:
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            return make_text_response(f'got: {part.content}')
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class ValErrCap(AbstractCapability[Any]):
            retried: bool = False

            async def before_tool_execute(
                self, ctx: RunContext[Any], *, call: ToolCallPart, tool_def: ToolDefinition, args: dict[str, Any]
            ) -> dict[str, Any]:
                if not self.retried:
                    self.retried = True
                    TypeAdapter(int).validate_python('not_an_int')
                return args

        cap = ValErrCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return 'tool result'

        result = await agent.run('call tool')
        assert result.output == 'got: tool result'
        # Tool only called once — before_tool_execute ValidationError prevented first call
        assert tool_call_count == 1
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'int_parsing',
                                    'loc': (),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'not_an_int',
                                }
                            ],
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=88, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool', content='tool result', tool_call_id='call-1', timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got: tool result')],
                    usage=RequestUsage(input_tokens=90, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_tool_execute_model_retry_skips_on_error(self):
        """wrap_tool_execute raising ModelRetry should NOT call on_tool_execute_error."""
        on_error_called = False

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got retry')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class WrapExecRetryCap(AbstractCapability[Any]):
            async def wrap_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                handler: Any,
            ) -> Any:
                raise ModelRetry('Wrap says retry tool')

            async def on_tool_execute_error(  # pragma: no cover — verifying this is NOT called
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                error: Exception,
            ) -> Any:
                nonlocal on_error_called
                on_error_called = True
                raise error

        agent = Agent(FunctionModel(model_fn), capabilities=[WrapExecRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')
        assert result.output == 'got retry'
        assert not on_error_called
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Wrap says retry tool',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got retry')],
                    usage=RequestUsage(input_tokens=63, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_on_tool_execute_error_model_retry(self):
        """on_tool_execute_error raises ModelRetry to recover via retry."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got retry after error')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class ErrorRetryCap(AbstractCapability[Any]):
            async def on_tool_execute_error(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
                error: Exception,
            ) -> Any:
                raise ModelRetry('Tool errored, please retry')

        agent = Agent(FunctionModel(model_fn), capabilities=[ErrorRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError('tool failed')

        result = await agent.run('call tool')
        assert result.output == 'got retry after error'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Tool errored, please retry',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got retry after error')],
                    usage=RequestUsage(input_tokens=63, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_validate_model_retry(self):
        """after_tool_validate raises ModelRetry — validation retry sent to model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got validation retry')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class AfterValRetryCap(AbstractCapability[Any]):
            async def after_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                raise ModelRetry('Validated args are bad')

        agent = Agent(FunctionModel(model_fn), capabilities=[AfterValRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')
        assert result.output == 'got validation retry'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Validated args are bad',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got validation retry')],
                    usage=RequestUsage(input_tokens=63, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_validate_model_retry(self):
        """before_tool_validate raises ModelRetry — validation retry sent to model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response('got pre-validation retry')
            if info.function_tools:
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=info.function_tools[0].name, args='{}', tool_call_id='call-1')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class BeforeValRetryCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                raise ModelRetry('Args look bad before validation')

        agent = Agent(FunctionModel(model_fn), capabilities=[BeforeValRetryCap()], retries={'tools': 2, 'output': 2})

        @agent.tool_plain
        def my_tool() -> str:
            return 'tool result'  # pragma: no cover

        result = await agent.run('call tool')
        assert result.output == 'got pre-validation retry'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='call tool', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Args look bad before validation',
                            tool_name='my_tool',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='got pre-validation retry')],
                    usage=RequestUsage(input_tokens=64, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestCtxAgentInCapability:
    """Test that ctx.agent is available in capability hooks."""

    async def test_ctx_agent_in_hooks(self):
        hook_agent_names: list[str | None] = []

        @dataclass
        class AgentTrackingCap(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                assert ctx.agent is not None
                hook_agent_names.append(ctx.agent.name)

            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                assert ctx.agent is not None
                hook_agent_names.append(ctx.agent.name)
                return request_context

        agent = Agent(FunctionModel(simple_model_function), name='hook_test_agent', capabilities=[AgentTrackingCap()])
        await agent.run('hello')
        assert hook_agent_names == ['hook_test_agent', 'hook_test_agent']


# region --- Compaction capability tests ---


class TestCompaction:
    def test_compaction_part_serialization(self):
        """CompactionPart round-trips through Pydantic serialization."""
        from pydantic_ai.messages import CompactionPart, ModelMessagesTypeAdapter, ModelResponse

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
    assert ThreadExecutor.get_serialization_name() is None


async def test_thread_executor_capability() -> None:
    tool_threads: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='check_thread', args='{}')])

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='cap-pool')
    try:
        agent = Agent(FunctionModel(model_function), capabilities=[ThreadExecutor(executor)])

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
    from pydantic_ai.result import FinalResult

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
    from pydantic_ai.result import FinalResult

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


# ===== Pending Message Queue Tests =====


async def test_enqueue_asap_message_from_tool():
    """`'asap'` messages enqueued from a tool are injected before the next model request."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('Injected asap message')
        return 'ok'

    result = await agent.run('Hello')
    assert result.output == 'done'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_msg', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_msg',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Injected asap message', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_asap_delivery_event_from_tool():
    """An `EnqueuedMessagesEvent` is emitted when an `'asap'` message is delivered, before the next model response."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msg', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        enqueue_id = ctx.enqueue('Injected asap message')
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(result.all_messages()[3],))]
    # The delivery event precedes the model response that can depend on the delivered message.
    delivery_index = events.index(delivery_events[0])
    done_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content == 'done'
    )
    assert delivery_index < done_index


async def test_enqueue_when_idle_delivery_event_during_iter_streaming():
    """A `'when_idle'` delivery surfaces as an `EnqueuedMessagesEvent` during `agent.iter` streaming."""
    call_count = 0

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_count
        call_count += 1
        yield f'response {call_count}'

    agent = Agent(FunctionModel(stream_function=stream_fn))
    events: list[AgentStreamEvent] = []

    async with agent.iter('Hello') as agent_run:
        enqueue_id = agent_run.enqueue('External follow-up', priority='when_idle')
        # Drive with `next()` (not bare `async for`) so `when_idle` messages drain, while
        # streaming each model-request node to observe its events.
        node = agent_run.next_node
        while not isinstance(node, End):
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
            node = await agent_run.next(node)

    assert enqueue_id is not None
    assert agent_run.result is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [
        EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(agent_run.result.all_messages()[2],))
    ]
    delivery_index = events.index(delivery_events[0])
    response_2_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and event.part.content == 'response 2'
    )
    assert delivery_index < response_2_index


async def test_multiple_enqueue_delivery_events_keep_order():
    """Multiple `enqueue` calls each emit one `EnqueuedMessagesEvent`, in enqueue order, via `run_stream_events`."""
    enqueue_ids: list[str] = []

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msgs', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_msgs(ctx: RunContext[Any]) -> str:
        first = ctx.enqueue('First injected message')
        second = ctx.enqueue('Second injected message')
        assert first is not None and second is not None
        enqueue_ids.extend([first, second])
        return 'ok'

    delivery_events: list[EnqueuedMessagesEvent] = []
    result: AgentRunResult[Any] | None = None
    async with agent.run_stream_events('Hello') as stream:
        async for event in stream:
            if isinstance(event, EnqueuedMessagesEvent):
                delivery_events.append(event)
            elif isinstance(event, AgentRunResultEvent):
                result = event.result

    assert result is not None
    messages = result.all_messages()
    assert delivery_events == [
        EnqueuedMessagesEvent(enqueue_id=enqueue_ids[0], messages=(messages[3],)),
        EnqueuedMessagesEvent(enqueue_id=enqueue_ids[1], messages=(messages[4],)),
    ]


async def test_enqueue_delivery_event_survives_history_processor_rebuild():
    """The delivery event still matches final history when a history processor rebuilds message objects."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msg', json_args='{}')}

    def rebuild_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Round-trip through JSON so every message is a fresh, equal-but-not-identical object.
        return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))

    agent = Agent(FunctionModel(stream_function=stream_fn), capabilities=[ProcessHistory(rebuild_messages)])

    @agent.tool
    def inject_msg(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        enqueue_id = ctx.enqueue('Injected asap message')
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(result.all_messages()[3],))]


async def test_empty_enqueue_emits_no_delivery_event():
    """An empty `enqueue()` call delivers nothing and emits no `EnqueuedMessagesEvent`."""

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='noop_enqueue', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def noop_enqueue(ctx: RunContext[Any]) -> str:
        assert ctx.enqueue() is None
        return 'ok'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert [event for event in events if isinstance(event, EnqueuedMessagesEvent)] == []


def test_enqueued_messages_event_serialization_roundtrip():
    """`EnqueuedMessagesEvent` round-trips through the `AgentStreamEvent` union as JSON.

    Durable execution (e.g. Temporal's per-event `event_stream_handler` wrapping) serializes
    events to JSON across the activity boundary, so JSON mode is the actual constraint.
    """
    event = EnqueuedMessagesEvent(
        enqueue_id='enq-1',
        messages=(ModelRequest(parts=[UserPromptPart(content='hi')]),),
    )
    adapter = TypeAdapter[AgentStreamEvent](AgentStreamEvent)
    dumped = adapter.dump_python(event)
    assert dumped['event_kind'] == 'enqueued_messages'
    assert adapter.validate_python(dumped) == event
    assert adapter.validate_json(adapter.dump_json(event)) == event


def test_pending_message_positional_construction_keeps_priority_second():
    """`PendingMessage(messages, priority)` positional construction still sets `priority`.

    Guards the field order: `enqueue_id` (which has a generated default) must stay after
    `priority`, or positional callers would silently assign their priority to `enqueue_id`.
    """
    pending = PendingMessage([ModelRequest(parts=[UserPromptPart(content='hi')])], 'when_idle')
    assert pending.priority == 'when_idle'
    assert pending.enqueue_id != 'when_idle'
    assert UUID(pending.enqueue_id).version == 7


async def test_single_enqueue_with_multiple_messages_emits_one_event():
    """One `enqueue` call carrying multiple messages emits a single event with all delivered messages."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_exchange', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_exchange(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        # A synthetic prior turn (a complete response) followed by a fresh user request:
        # one enqueue call, two delivered messages.
        enqueue_id = ctx.enqueue(
            ModelResponse(parts=[TextPart(content='synthetic recap')]),
            'Follow up on the recap',
        )
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=tuple(result.all_messages()[3:5]))]
    assert isinstance(delivery_events[0].messages[0], ModelResponse)
    assert isinstance(delivery_events[0].messages[1], ModelRequest)


async def test_enqueue_delivery_event_via_run_stream():
    """The delivery event surfaces through `agent.run_stream`'s `event_stream_handler`."""
    events: list[AgentStreamEvent] = []
    enqueue_id: str | None = None

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='inject_msg', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[Any]) -> str:
        nonlocal enqueue_id
        enqueue_id = ctx.enqueue('Injected asap message')
        return 'ok'

    async def event_stream_handler(_: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    async with agent.run_stream('Hello', event_stream_handler=event_stream_handler) as result:
        output = await result.get_output()

    assert output == 'done'
    assert enqueue_id is not None
    delivery_events = [event for event in events if isinstance(event, EnqueuedMessagesEvent)]
    assert delivery_events == [EnqueuedMessagesEvent(enqueue_id=enqueue_id, messages=(result.all_messages()[3],))]


async def test_with_event_stream_buffer_drains_around_node_stream():
    """`_with_event_stream_buffer` yields buffered events before, between, and after node events."""
    buffer: list[AgentStreamEvent] = []
    during = EnqueuedMessagesEvent(enqueue_id='during', messages=())
    after = EnqueuedMessagesEvent(enqueue_id='after', messages=())
    model_event = PartStartEvent(index=0, part=TextPart(content='done'))

    async def stream() -> AsyncIterator[AgentStreamEvent]:
        buffer.append(during)
        yield model_event
        buffer.append(after)

    drained = [event async for event in _agent_graph._with_event_stream_buffer(stream(), buffer)]  # pyright: ignore[reportPrivateUsage]
    assert drained == [during, model_event, after]


async def test_agent_stream_events_iter_drains_buffer_before_each_pull():
    """`AgentStream._events_iter` drains buffered run events before each pull from the model stream.

    Events buffered while a pull is in flight surface on the next pull; events buffered after the
    last model event are not drained here — they flow through the response-handling node's stream
    (`_with_event_stream_buffer`'s trailing drain) once this stream is exhausted.
    """
    initial = EnqueuedMessagesEvent(enqueue_id='initial', messages=())
    during = EnqueuedMessagesEvent(enqueue_id='during', messages=())
    after = EnqueuedMessagesEvent(enqueue_id='after', messages=())
    model_event = PartStartEvent(index=0, part=TextPart(content='done'))
    buffer: list[AgentStreamEvent] = [initial]

    async def base_iter() -> AsyncIterator[ModelResponseStreamEvent]:
        buffer.append(during)
        yield model_event
        buffer.append(after)

    stream = cast(AgentStream[Any, str], object.__new__(AgentStream))
    stream._event_stream_buffer_getter = lambda: buffer  # pyright: ignore[reportPrivateUsage]
    stream._anext_lock = anyio.Lock()  # pyright: ignore[reportPrivateUsage]

    drained = [event async for event in stream._events_iter(base_iter())]  # pyright: ignore[reportPrivateUsage]
    assert drained == [initial, model_event, during]
    # `after` stays buffered for the response-handling node's stream to deliver.
    assert buffer == [after]


class _FixedEventsAgentStream(AgentStream[Any, str]):
    """An `AgentStream` whose event stream is a fixed list, for testing the event filters."""

    def __init__(self, events: list[AgentStreamEvent]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncIterator[AgentStreamEvent]:
        return self._iter_events()

    async def _iter_events(self) -> AsyncIterator[AgentStreamEvent]:
        for event in self._events:
            yield event


async def test_agent_stream_model_response_events_skips_buffered_events():
    """`AgentStream._model_response_events` filters buffered run events out of the model response stream."""
    buffered = EnqueuedMessagesEvent(enqueue_id='buffered', messages=())
    model_event = PartStartEvent(index=0, part=TextPart(content='done'))
    stream = _FixedEventsAgentStream([buffered, model_event])

    drained = [event async for event in stream._model_response_events()]  # pyright: ignore[reportPrivateUsage]
    assert drained == [model_event]


async def test_enqueue_when_idle_message_prevents_end():
    """`'when_idle'` messages prevent the agent from ending and are drained into a new ModelRequest."""
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        elif call_count == 2:
            # Agent produces final result, but follow-up is pending
            return ModelResponse(
                parts=[TextPart(content='premature end')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        else:
            # After follow-up is drained, agent produces real final result
            return ModelResponse(
                parts=[TextPart(content='final answer after follow-up')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_follow_up(ctx: RunContext[object]) -> str:
        ctx.enqueue('Follow-up context', priority='when_idle')
        return 'ok'

    result = await agent.run('Hello')
    assert result.output == 'final answer after follow-up'
    assert call_count == 3
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_follow_up',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='premature end')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Follow-up context', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final answer after follow-up')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_when_idle_redirects_after_output_tool_end():
    """A `when_idle` message redirects the run even when it would end via an output tool.

    The run terminates when the model calls an output tool (`ToolOutput` mode), which produces
    an `End` from `CallToolsNode`. The drain's `after_node_run` still sees that `End` and
    redirects into a fresh request, so the agent gets another turn after the structured output —
    and the final `result.output` comes from that later turn.
    """

    class Answer(BaseModel):
        value: int

    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        output_tool = info.output_tools[0].name
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        if call_count == 2:
            # Would end the run via the output tool, but a `when_idle` message is pending.
            return ModelResponse(
                parts=[ToolCallPart(tool_name=output_tool, args='{"value": 1}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        # After the follow-up is drained, the model produces the real final output.
        return ModelResponse(
            parts=[ToolCallPart(tool_name=output_tool, args='{"value": 2}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn), output_type=Answer)

    @agent.tool
    def inject_follow_up(ctx: RunContext[object]) -> str:
        ctx.enqueue('Follow-up context', priority='when_idle')
        return 'ok'

    result = await agent.run('Hello')

    assert result.output == Answer(value=2)
    assert call_count == 3
    # The `when_idle` follow-up lands as its own request after the first (superseded) output-tool
    # call, redirecting the run so the second output-tool call produces the real output.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_follow_up', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_follow_up',
                        content='ok',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"value": 1}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='Follow-up context', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"value": 2}',
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_from_agent_run():
    """Messages can be enqueued from external code via AgentRun.enqueue."""
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        return ModelResponse(
            parts=[TextPart(content=f'response {call_count}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    async with agent.iter('Hello') as agent_run:
        assert agent_run.pending_messages == []
        # Enqueue a when_idle message from external code before iteration
        agent_run.enqueue('External follow-up', priority='when_idle')
        assert len(agent_run.pending_messages) == 1
        # Use next() to drive iteration so after_node_run fires
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)

    assert agent_run.result is not None
    assert call_count == 2  # First response triggers End, follow-up prevents it, second response is final
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response 1')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='External follow-up', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='response 2')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_bare_async_for_raises_with_undrained_pending_messages():
    """Bare `async for` reaching End with undrained `when_idle` messages raises rather than stranding them.

    `when_idle` (and end-of-step `asap` leftovers) drain in `after_node_run`, which bare
    iteration skips — so they'd be silently lost. `__anext__` raises
    `UndrainedPendingMessagesError` when it would yield the `End` node with a non-empty queue,
    pointing the user at `next()` driving.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart(content='done')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    async with agent.iter('hi') as agent_run:
        agent_run.enqueue('stranded follow-up', priority='when_idle')
        with pytest.raises(UndrainedPendingMessagesError, match='undrained pending messages'):
            async for _ in agent_run:
                pass

        # The message was never delivered: it's still queued.
        assert len(agent_run.pending_messages) == 1


async def test_pending_messages_accessible_on_run_context():
    """RunContext.pending_messages is accessible and initially empty."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='check_queue', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def check_queue(ctx: RunContext[object]) -> str:
        # The queue must be live (mutations from inside a tool reach the drain).
        assert ctx.pending_messages is not None
        assert len(ctx.pending_messages) == 0
        ctx.enqueue('observed', priority='asap')
        assert len(ctx.pending_messages) == 1
        return 'done'

    result = await agent.run('Test')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Test', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='check_queue', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='check_queue',
                        content='done',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='observed', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_with_no_args_is_a_noop():
    """`ctx.enqueue()` and `agent_run.enqueue()` with no content are silent no-ops.

    Producers that conditionally enqueue (e.g. "announce if new tools were discovered")
    can call `enqueue(*maybe_items)` without guarding for the empty case — `enqueue`
    simply doesn't append a `PendingMessage` when there's nothing to send.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='from_tool', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def from_tool(ctx: RunContext[Any]) -> str:
        assert ctx.enqueue() is None  # no-op, no exception, no id
        assert ctx.pending_messages == []
        return 'ok'

    async with agent.iter('hi') as agent_run:
        assert agent_run.enqueue() is None  # no-op, no exception, no id
        assert agent_run.pending_messages == []
        async for _ in agent_run:
            pass


async def test_enqueue_coerces_string_to_user_prompt():
    """A bare string passed to `enqueue` is wrapped in a `UserPromptPart`."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('steering as plain string')
        return 'ok'

    result = await agent.run('Hello')
    injected = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if part.content == 'steering as plain string'
    ]
    assert len(injected) == 1, 'string-coerced enqueue did not land as a UserPromptPart'


async def test_enqueue_accepts_multimodal_user_content():
    """Adjacent user-content args (text + multi-modal) are gathered into one `UserPromptPart`."""
    image = ImageUrl(url='https://example.com/image.png')

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('look at this', image)
        return 'ok'

    result = await agent.run('Hello')
    injected = [
        part
        for part in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if part.content == ['look at this', image]
    ]
    assert len(injected) == 1


async def test_enqueue_accepts_model_request_passthrough():
    """A full `ModelRequest` is enqueued verbatim, preserving `instructions`/`metadata`.

    Two passthroughs cover both branches of the fill-in-if-unset stamping logic:
    one with `timestamp`/`run_id` unset (drain stamps them); one with both set
    (drain leaves them alone).
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    unstamped = ModelRequest(
        parts=[UserPromptPart(content='wire-level payload')],
        instructions='do this carefully',
        metadata={'origin': 'webhook-42'},
    )
    preset_timestamp = datetime(2024, 1, 1, tzinfo=timezone.utc)
    prestamped = ModelRequest(
        parts=[UserPromptPart(content='already stamped')],
        instructions='preserve me',
        timestamp=preset_timestamp,
        run_id='caller-run-id',
        conversation_id='caller-conv-id',
    )

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue(unstamped)
        ctx.enqueue(prestamped)
        return 'ok'

    result = await agent.run('Hello')

    injected_unstamped = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest) and msg.instructions == 'do this carefully'
    )
    assert injected_unstamped.metadata == {'origin': 'webhook-42'}
    # Drain should have stamped timestamp/run_id/conversation_id since the user didn't set them.
    assert injected_unstamped.timestamp is not None
    assert injected_unstamped.run_id is not None
    assert injected_unstamped.conversation_id is not None

    injected_prestamped = next(
        msg for msg in result.all_messages() if isinstance(msg, ModelRequest) and msg.instructions == 'preserve me'
    )
    # Producer-supplied timestamp/run_id/conversation_id are preserved (drain doesn't overwrite).
    assert injected_prestamped.timestamp == preset_timestamp
    assert injected_prestamped.run_id == 'caller-run-id'
    assert injected_prestamped.conversation_id == 'caller-conv-id'


def test_pending_message_drain_capability_is_not_spec_constructible():
    """`PendingMessageDrainCapability` is auto-injected only; can't be in an `AgentSpec`."""
    from pydantic_ai.capabilities._pending_messages import PendingMessageDrainCapability

    assert PendingMessageDrainCapability.get_serialization_name() is None


def test_pending_message_allows_empty_request():
    """`PendingMessage` doesn't validate its `messages`; empty-parts requests are tolerated.

    `enqueue()` already filters out the no-args case (no `PendingMessage` is appended).
    An empty `ModelRequest` reaching the queue is harmless — the drain stamps and forwards
    it, and downstream wire-merging absorbs zero-part messages as a natural no-op.
    """
    msg = PendingMessage(messages=[ModelRequest(parts=[])])
    assert msg.priority == 'asap'
    assert msg.messages[0].parts == []


def test_enqueue_without_live_queue_raises():
    """`ctx.enqueue` raises when the `RunContext` isn't backed by a running agent's queue.

    Synthetic contexts (e.g. the one `Agent.system_prompt_parts` builds to resolve system
    prompts outside a run) have no queue to drain to, so enqueue fails loudly instead of
    silently dropping the message.
    """
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[])
    assert ctx.pending_messages is None
    with pytest.raises(UserError, match='only available during an agent run'):
        ctx.enqueue('this has nowhere to go')


async def test_enqueue_parts_style_calls_produce_one_request_per_call():
    """Each `enqueue` call produces its own `ModelRequest` in history.

    Each `enqueue` call pre-packages its content into a `ModelRequest` at enqueue time,
    so two calls produce two `PendingMessage`s with two separate `ModelRequest`s. The
    history reflects per-call structure; wire-level `_clean_message_history` still merges
    adjacent compatible `ModelRequest`s so the model sees one turn. Producers wanting a
    single message should pass a single `ModelRequest(parts=[...])` themselves.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('first hint')
        ctx.enqueue('second hint')
        return 'ok'

    result = await agent.run('Hello')
    drained = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content in ('first hint', 'second hint') for p in msg.parts)
    ]
    assert len(drained) == 2, 'expected one ModelRequest per enqueue call'
    assert [p.content for p in iter_message_parts(drained, ModelRequest, UserPromptPart)] == [
        'first hint',
        'second hint',
    ]


async def test_enqueue_passthrough_stays_separate_from_parts_style():
    """A passthrough `ModelRequest` stays its own message even when surrounded by parts-style enqueues."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_msg', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject_msg(ctx: RunContext[object]) -> str:
        ctx.enqueue('before')
        ctx.enqueue(
            ModelRequest(parts=[UserPromptPart(content='passthrough')], instructions='careful'),
        )
        ctx.enqueue('after')
        return 'ok'

    result = await agent.run('Hello')
    # Three drained requests: synthesized(["before"]), passthrough, synthesized(["after"]).
    drained = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content in ('before', 'passthrough', 'after') for p in msg.parts)
    ]
    assert len(drained) == 3
    contents = [
        next(
            p.content
            for p in r.parts
            if isinstance(p, UserPromptPart) and p.content in ('before', 'passthrough', 'after')
        )
        for r in drained
    ]
    assert contents == ['before', 'passthrough', 'after']
    # Passthrough preserved its instructions.
    assert drained[1].instructions == 'careful'
    assert drained[0].instructions is None
    assert drained[2].instructions is None


async def test_enqueue_system_prompt_part():
    """A bare `SystemPromptPart` is coalesced into a `ModelRequest` and delivered.

    Now that mid-conversation `SystemPromptPart`s are rendered inline (not hoisted) on all
    providers, `enqueue` accepts request parts directly — no `ModelRequest` wrapper needed.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='announce', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def announce(ctx: RunContext[object]) -> str:
        ctx.enqueue(SystemPromptPart(content='New tools are now available.'))
        return 'ok'

    result = await agent.run('Hello')
    injected = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, SystemPromptPart) and p.content == 'New tools are now available.' for p in msg.parts)
    )
    assert injected is not None


async def test_enqueue_interleaved_response_and_request():
    """One `enqueue` call can inject an interleaved `ModelResponse` + `ModelRequest` exchange.

    This is the synthetic "tool-search call + result" shape (a `ModelResponse` carrying the call
    followed by a `ModelRequest` carrying the return). Both land in history in order, and the
    trailing request is what the agent responds to next.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject_exchange', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    synthetic_response = ModelResponse(
        parts=[TextPart(content='synthetic prior turn')],
        usage=RequestUsage(input_tokens=1, output_tokens=1),
    )

    @agent.tool
    def inject_exchange(ctx: RunContext[object]) -> str:
        ctx.enqueue(
            synthetic_response,
            ModelRequest(parts=[UserPromptPart(content='follow-up after synthetic turn')]),
            priority='when_idle',
        )
        return 'ok'

    result = await agent.run('Hello')
    # The synthetic response is appended to history immediately before its paired request.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='inject_exchange', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inject_exchange',
                        content='ok',
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
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='synthetic prior turn')],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='follow-up after synthetic turn', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_rejects_content_not_ending_in_request():
    """Enqueued content must end in a `ModelRequest`; a lone `ModelResponse` is rejected.

    The agent needs a request to respond to — content that ends in a `ModelResponse` (with no
    trailing request/part-style items) would leave nothing for the model to react to.
    """

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(msg, ModelResponse) for msg in messages):
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='from_tool', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    lone_response = ModelResponse(
        parts=[TextPart(content='synthetic')], usage=RequestUsage(input_tokens=1, output_tokens=1)
    )

    @agent.tool
    def from_tool(ctx: RunContext[object]) -> str:
        with pytest.raises(UserError, match='must end with a `ModelRequest`'):
            ctx.enqueue(lone_response)
        return 'ok'

    async with agent.iter('hi') as agent_run:
        with pytest.raises(UserError, match='must end with a `ModelRequest`'):
            agent_run.enqueue(lone_response)
        async for _ in agent_run:
            pass


async def test_drain_rejects_directly_queued_content_not_ending_in_request():
    """Directly appending a malformed `PendingMessage` raises a `UserError` at end-of-run drain.

    `enqueue` enforces the "ends in a `ModelRequest`" rule up front, but `RunContext.pending_messages`
    is public, so a producer can append a `PendingMessage` directly. The end-of-run drain catches a
    request-less message with a helpful `UserError` rather than a bare assertion.
    """
    lone_response = ModelResponse(
        parts=[TextPart(content='synthetic')], usage=RequestUsage(input_tokens=1, output_tokens=1)
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')], usage=RequestUsage(input_tokens=10, output_tokens=5))
        return ModelResponse(
            parts=[ToolCallPart(tool_name='queue_bad', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def queue_bad(ctx: RunContext[object]) -> str:
        assert ctx.pending_messages is not None
        ctx.pending_messages.append(PendingMessage(messages=[lone_response], priority='when_idle'))
        return 'ok'

    with pytest.raises(UserError, match='must end with a `ModelRequest`'):
        await agent.run('hi')


async def test_enqueue_asap_with_rich_message_history_tail():
    """`'asap'` enqueue lands as its own `ModelRequest` in history *and* gets wire-merged into the rich tail.

    The history keeps the un-merged view (drain's request is a separate `ModelRequest`
    after the rich tail) so `all_messages()` reflects per-call structure. On the wire,
    `_clean_message_history` merges the two adjacent `ModelRequest`s and sorts
    `ToolReturnPart`/`RetryPromptPart` first — non-tool parts keep arrival order, so the
    enqueued content lands at the *end* of the merged turn (not interleaved between
    existing parts). Captures the `messages` arg `FunctionModel` actually received to
    validate the wire-level merge through the public path.
    """
    captured_wire_messages: list[list[ModelMessage]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_wire_messages.append(messages)
        return ModelResponse(
            parts=[TextPart(content='done')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='original prompt')]),
        ModelResponse(
            parts=[ToolCallPart(tool_name='hint', args='{}', tool_call_id='call-1')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
            model_name='function:model_fn:',
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='hint', content='ok', tool_call_id='call-1'),
                UserPromptPart(content='follow-up question'),
            ],
        ),
    ]

    async with agent.iter(message_history=history) as agent_run:
        agent_run.enqueue('injected after rich tail')
        async for _ in agent_run:
            pass

    assert agent_run.result is not None
    # `all_messages()` keeps the un-merged view (drain's request is a separate
    # `ModelRequest` after the rich tail).
    assert agent_run.result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='original prompt', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='hint', args='{}', tool_call_id='call-1')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='hint', content='ok', tool_call_id='call-1', timestamp=IsDatetime()),
                    UserPromptPart(content='follow-up question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='injected after rich tail', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # And the wire-level view: the rich tail and the drained request merged into one
    # `ModelRequest`, with `ToolReturnPart` first and the user-prompt parts in arrival
    # order (so the enqueued content lands at the end, not interleaved).
    assert len(captured_wire_messages) == 1
    assert captured_wire_messages[0] == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='original prompt', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='hint', args='{}', tool_call_id='call-1')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='hint', content='ok', tool_call_id='call-1', timestamp=IsDatetime()),
                    UserPromptPart(content='follow-up question', timestamp=IsDatetime()),
                    UserPromptPart(content='injected after rich tail', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_enqueue_asap_drains_at_end_if_arrived_during_final_step():
    """`'asap'` arriving during the final step (after its `before_model_request` drain) still gets delivered.

    Simulates the background-tools pattern: a long-running task completes *during* what
    would have been the model's final response. The enqueue happens after the step's
    `before_model_request` drain has already fired, so the message can only be picked up
    by the end-of-run drain (matching pi-mono's drain-on-end). Without this fallback the
    message would be lost. `'asap'` semantically means "deliver at the earliest opportunity"
    — including redirecting if the agent would otherwise terminate before another call.
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[TextPart(content='would-have-ended')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[TextPart(content='final after late asap')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    @dataclass
    class BackgroundTaskCap(AbstractCapability[Any]):
        """Simulates a background task that completes mid-model-response on the first call only."""

        fired: bool = False

        async def after_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            if not self.fired:
                ctx.enqueue('background task result', priority='asap')
                self.fired = True
            return response

    agent = Agent(FunctionModel(model_fn), capabilities=[BackgroundTaskCap()])

    result = await agent.run('Hello')
    assert result.output == 'final after late asap'
    assert call_count == 2
    # The 'asap' message landed in its own ModelRequest before the final response,
    # not lost despite the agent producing a no-tool-call response.
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='would-have-ended')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='background task result', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final after late asap')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_enqueue_when_idle_drains_after_leftover_asap():
    """If both `'asap'` and `'when_idle'` are queued at end-of-run, `'asap'` drains first."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Only fire enqueues once.
        already_enqueued = any(
            isinstance(p, UserPromptPart) and p.content in ('A', 'B')
            for msg in messages
            if isinstance(msg, ModelRequest)
            for p in msg.parts
        )
        # If we've already seen our injected messages, just terminate.
        if already_enqueued:
            return ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[ToolCallPart(tool_name='inject', args='{}')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    agent = Agent(FunctionModel(model_fn))

    @agent.tool
    def inject(ctx: RunContext[object]) -> str:
        ctx.enqueue('B', priority='when_idle')
        ctx.enqueue('A', priority='asap')
        return 'ok'

    result = await agent.run('Hello')
    # Both A and B should appear in history. `'asap'` (A) drains in `before_model_request`
    # before the second call. `'when_idle'` (B) drains at end-of-run when the second
    # response has no tool calls.
    requests_with_injected = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) and p.content in ('A', 'B') for p in msg.parts)
    ]
    contents = [
        [p.content for p in r.parts if isinstance(p, UserPromptPart) and p.content in ('A', 'B')]
        for r in requests_with_injected
    ]
    assert contents == [['A'], ['B']], f'expected A before B in separate requests, got {contents}'


async def test_enqueue_priorities_stay_separate_when_both_drain_at_end_of_run():
    """When both `'asap'` and `'when_idle'` parts-style payloads drain together at end-of-run,
    they land in separate `ModelRequest`s — the priority split stays visible in history.

    Reaches the case Devin flagged: a tool enqueues `'when_idle'` (which sits until
    end-of-run), and a capability `after_model_request` hook enqueues `'asap'` during the
    final step (after that step's `before_model_request` drain has already fired). Both
    arrive at `after_node_run`. Without the per-priority split they'd merge into one
    synthesized request, blurring the priority distinction in the persisted history.
    On the wire `_clean_message_history` still merges them for the model.
    """
    call_count = 0

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='inject', args='{}')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        if call_count == 2:
            return ModelResponse(
                parts=[TextPart(content='would-have-ended')],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            )
        return ModelResponse(
            parts=[TextPart(content='final')],
            usage=RequestUsage(input_tokens=10, output_tokens=5),
        )

    @dataclass
    class LateAsapCap(AbstractCapability[Any]):
        """Enqueues an `'asap'` message during `after_model_request` of the no-tool-call step.

        Fires after the step's `before_model_request` drain, so the message can only be
        delivered via the end-of-run drain in `after_node_run`.
        """

        fired: bool = False

        async def after_model_request(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            if not self.fired and any(
                isinstance(p, TextPart) and p.content == 'would-have-ended' for p in response.parts
            ):
                ctx.enqueue('asap-from-cap')
                self.fired = True
            return response

    agent = Agent(FunctionModel(model_fn), capabilities=[LateAsapCap()])

    @agent.tool
    def inject(ctx: RunContext[object]) -> str:
        ctx.enqueue('when-idle-from-tool', priority='when_idle')
        return 'ok'

    result = await agent.run('Hello')
    assert result.output == 'final'

    # Find the two end-of-run drained requests: one with the 'asap' content, one with 'when_idle'.
    drained = [
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(
            isinstance(p, UserPromptPart) and p.content in ('asap-from-cap', 'when-idle-from-tool') for p in msg.parts
        )
    ]
    contents = [
        next(
            p.content
            for p in r.parts
            if isinstance(p, UserPromptPart) and p.content in ('asap-from-cap', 'when-idle-from-tool')
        )
        for r in drained
    ]
    assert contents == ['asap-from-cap', 'when-idle-from-tool'], (
        f'asap and when_idle should land in separate ModelRequests with asap first, got {contents}'
    )
    # Each priority bucket got its own ModelRequest (not merged into one).
    assert all(len([p for p in r.parts if isinstance(p, UserPromptPart)]) == 1 for r in drained)


# --- Output hook tests ---


class MyOutput(BaseModel):
    value: int


class TestBeforeOutputValidate:
    """before_output_validate can transform raw output before parsing."""

    async def test_structured_prompted_output(self):
        """before_output_validate transforms raw text before Pydantic validation for PromptedOutput."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "not_a_number"}')])

        @dataclass
        class FixJsonCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                if isinstance(output, str):
                    return output.replace('"not_a_number"', '42')
                return output  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[FixJsonCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)

    async def test_plain_str_output(self):
        """For plain str output, validate hooks are skipped; process hooks fire instead."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello world')

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('validate')  # pragma: no cover — should NOT fire for plain text
                return output  # pragma: no cover

            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append(f'process:{output}')
                assert output_context.mode == 'text'
                assert output_context.output_type is str
                assert output_context.has_function is False
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == 'hello world'
        # Validate hooks do NOT fire for plain text; only process hooks fire
        assert log == ['process:hello world']

    async def test_text_output_function(self):
        """For TextOutput, validate hooks are skipped; process hooks fire and call the function."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append(f'before:{output}')
                assert output_context.has_function is True
                return output

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == 'WORLD'
        assert log == ['before:world']

    async def test_can_transform_text_before_function(self):
        """before_output_process can modify text before the TextOutput function runs."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class PrependCap(AbstractCapability[Any]):
            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                assert isinstance(output, str)
                return f'hello {output}'

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[PrependCap()])
        result = await agent.run('greet')
        assert result.output == 'HELLO WORLD'


class TestOnOutputValidateError:
    """on_output_validate_error can recover from validation errors."""

    async def test_recover_from_invalid_json(self):
        """on_output_validate_error can fix raw output and return corrected data."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])

        @dataclass
        class RecoverCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                # Recovery replaces the validation result; for structured output
                # the execute step (call()) returns this as-is when there's no function.
                return {'value': 99}

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RecoverCap()])
        result = await agent.run('hello')
        # The error hook bypasses Pydantic validation, so the output is the raw dict
        assert result.output == {'value': 99}
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": "bad"}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_default_reraises(self):
        """Without an error hook, validation errors propagate normally as retries."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput))
        result = await agent.run('hello')
        # Model retries and eventually gets it right
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": "bad"}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'int_parsing',
                                    'loc': ('value',),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'bad',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=87, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestOnOutputValidateErrorModelRetry:
    """on_output_validate_error can raise ModelRetry to trigger a retry with a custom message."""

    async def test_error_hook_raises_model_retry(self):
        """on_output_validate_error raises ModelRetry, which becomes a retry prompt."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class RetryHookCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                raise ModelRetry('Please return a valid integer for value')

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RetryHookCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": "bad"}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Please return a valid integer for value',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestModelRetryFromOutputHooks:
    """Hooks can raise ModelRetry to trigger a model retry."""

    async def test_before_output_validate_raises_model_retry(self):
        """before_output_validate can raise ModelRetry to skip validation and retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class RejectNegativeCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                if isinstance(output, str) and '-1' in output:
                    raise ModelRetry('Negative values are not allowed')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectNegativeCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": -1}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Negative values are not allowed',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=65, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_output_validate_raises_model_retry(self):
        """after_output_validate can raise ModelRetry to reject validated output."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": 0}')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class RejectZeroCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                # Validated output is a MyOutput instance (Pydantic returns model instances)
                if isinstance(output, MyOutput) and output.value == 0:
                    raise ModelRetry('Zero is not a valid value')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectZeroCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 0}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Zero is not a valid value',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=66, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_after_output_process_raises_model_retry(self):
        """after_output_process can raise ModelRetry to reject the execution result."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='short')])
            return ModelResponse(parts=[TextPart(content='this is long enough')])

        @dataclass
        class MinLengthCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, str) and len(output) < 10:
                    raise ModelRetry('Output too short, please elaborate')
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[MinLengthCap()])
        result = await agent.run('hello')
        assert result.output == 'this is long enough'
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='short')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Output too short, please elaborate',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='this is long enough')],
                    usage=RequestUsage(input_tokens=65, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_output_process_model_retry_skips_error_hook(self):
        """ModelRetry from wrap_output_process bypasses on_output_process_error."""
        error_hook_called = False
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='bad')])
            return ModelResponse(parts=[TextPart(content='good')])

        @dataclass
        class WrapRetryCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, handler: Any
            ) -> Any:
                result = await handler(output)
                if result == 'bad':
                    raise ModelRetry('Bad output, please try again')
                return result

            async def on_output_process_error(  # pragma: no cover — verifying this is NOT called
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, error: Exception
            ) -> Any:
                nonlocal error_hook_called
                error_hook_called = True
                raise error

        agent = Agent(FunctionModel(model_fn), capabilities=[WrapRetryCap()])
        result = await agent.run('hello')
        assert result.output == 'good'
        assert call_count == 2
        assert not error_hook_called  # ModelRetry skips on_output_process_error
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='bad')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Bad output, please try again',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='good')],
                    usage=RequestUsage(input_tokens=65, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_before_output_process_raises_model_retry(self):
        """before_output_process can raise ModelRetry to skip execution."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": 0}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class RejectBeforeExecCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value == 0:
                    raise ModelRetry('Cannot execute with zero value')
                return output

        agent = Agent(
            FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectBeforeExecCap()]
        )
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 0}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Cannot execute with zero value',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 5}')],
                    usage=RequestUsage(input_tokens=65, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_before_validate_raises_model_retry(self):
        """ModelRetry from before_output_validate on a tool output includes tool_call_id."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": -1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class RejectNegativeCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                if (
                    isinstance(output, str)
                    and '-1' in output
                    or isinstance(output, dict)
                    and output.get('value', 0) < 0
                ):
                    raise ModelRetry('Negative values not allowed')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[RejectNegativeCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": -1}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Negative values not allowed',
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 42}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=62, output_tokens=8),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_after_execute_raises_model_retry(self):
        """ModelRetry from after_output_process on a tool output triggers retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": 0}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 10}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class RejectZeroCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value == 0:
                    raise ModelRetry('Zero not allowed')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[RejectZeroCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 0}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Zero not allowed',
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 10}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=61, output_tokens=8),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_validation_failure(self):
        """Invalid output tool args trigger retry through output validate hooks."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": "bad"}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput)
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": "bad"}', tool_call_id='call-1')],
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
                                {
                                    'type': 'int_parsing',
                                    'loc': ('value',),
                                    'msg': 'Input should be a valid integer, unable to parse string as an integer',
                                    'input': 'bad',
                                }
                            ],
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 42}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=89, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_output_tool_error_hook_raises_model_retry(self):
        """on_output_validate_error raises ModelRetry for output tool, includes tool_call_id."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": "bad"}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        @dataclass
        class RetryOnErrorCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                raise ModelRetry('Please provide a valid integer')

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[RetryOnErrorCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": "bad"}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='Please provide a valid integer',
                            tool_name='final_result',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 42}', tool_call_id='call-2')],
                    usage=RequestUsage(input_tokens=63, output_tokens=9),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-2',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestOutputToolWithOutputFunction:
    """Output tools with output functions that raise ModelRetry."""

    async def test_output_function_model_retry(self):
        """An output function on a tool output type that raises ModelRetry triggers a retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 10}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        def my_output_fn(output: MyOutput) -> MyOutput:
            if output.value < 5:
                raise ModelRetry('Value must be >= 5')
            return output

        agent = Agent(FunctionModel(model_fn), output_type=my_output_fn)
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert call_count == 2

    async def test_output_function_model_retry_with_hooks(self):
        """Output function ModelRetry works correctly when output hooks are present."""
        log: list[str] = []
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if info.output_tools:
                tool = info.output_tools[0]
                if call_count == 1:
                    return ModelResponse(
                        parts=[ToolCallPart(tool_name=tool.name, args='{"value": 1}', tool_call_id='call-1')]
                    )
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 10}', tool_call_id='call-2')]
                )
            return make_text_response('no tools')  # pragma: no cover

        def my_output_fn(output: MyOutput) -> MyOutput:
            if output.value < 5:
                raise ModelRetry('Value must be >= 5')
            return output

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(f'execute:{output}')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=my_output_fn, capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert call_count == 2
        # Execute hook fires for both attempts (retry + success)
        assert len(log) == 2


class TestWrapOutputValidate:
    """wrap_output_validate provides full middleware control around validation."""

    async def test_wrap_can_observe(self):
        """wrap_output_validate can observe without modifying."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 10}')])

        @dataclass
        class WrapCap(AbstractCapability[Any]):
            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('before')
                result = await handler(output)
                log.append('after')
                return result

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[WrapCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)
        assert log == ['before', 'after']

    async def test_wrap_can_transform_input(self):
        """wrap_output_validate can transform the output before passing to handler."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "oops"}')])

        @dataclass
        class TransformCap(AbstractCapability[Any]):
            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                # Fix the input before validation
                fixed = '{"value": 7}' if isinstance(output, str) else output
                return await handler(fixed)

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[TransformCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=7)

    async def test_wrap_can_catch_and_recover(self):
        """wrap_output_validate can catch validation errors and return a fallback."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='not json at all')])

        @dataclass
        class RecoverWrapCap(AbstractCapability[Any]):
            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                try:
                    return await handler(output)
                except (ValidationError, ModelRetry):
                    return {'value': 0}

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RecoverWrapCap()])
        result = await agent.run('hello')
        # The wrap recovery bypasses Pydantic validation, so the output is the raw dict
        assert result.output == {'value': 0}


class TestAfterOutputProcess:
    """after_output_process can transform the final result after execution."""

    async def test_transform_structured_result(self):
        """after_output_process transforms the result of structured output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class DoubleResultCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                assert isinstance(output, MyOutput)
                return MyOutput(value=output.value * 2)

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[DoubleResultCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)

    async def test_transform_plain_text_result(self):
        """after_output_process can transform plain text output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class UpperCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return output.upper() if isinstance(output, str) else output

        agent = Agent(FunctionModel(model_fn), capabilities=[UpperCap()])
        result = await agent.run('hello')
        assert result.output == 'HELLO'

    async def test_transform_text_function_result(self):
        """after_output_process fires after TextOutput function has executed."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('world')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class WrapResultCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                # output is already 'WORLD' from upcase
                return f'[{output}]'

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[WrapResultCap()])
        result = await agent.run('hello')
        assert result.output == '[WORLD]'


class TestToolOutputWithOutputHooks:
    """Output hooks fire for tool-based output, nested inside tool hooks."""

    async def test_output_hooks_fire_for_tool_output(self):
        """Output hooks fire when the output type uses tool mode."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class OutputLogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'before_output_validate:{output_context.mode}')
                return output

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_output_validate')
                return output

            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('before_output_process')
                return output

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_output_process')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[OutputLogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert 'before_output_validate:tool' in log
        assert 'after_output_validate' in log
        assert 'before_output_process' in log
        assert 'after_output_process' in log

    async def test_output_hooks_fire_without_tool_hooks(self):
        """Output tools use output hooks only — tool hooks do NOT fire."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 42}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class BothHooksCap(AbstractCapability[Any]):
            async def before_tool_validate(  # pragma: no cover — verifying this is NOT called
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append(f'tool_validate:{call.tool_name}')
                return args

            async def before_tool_execute(  # pragma: no cover — verifying this is NOT called
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                log.append(f'tool_execute:{call.tool_name}')
                return args

            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('output_validate')
                return output

            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('output_process')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[BothHooksCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        # Only output hooks fire for output tools — tool hooks are skipped
        assert 'tool_validate:final_result' not in log
        assert 'tool_execute:final_result' not in log
        assert 'output_validate' in log
        assert 'output_process' in log

    async def test_after_output_process_transforms_tool_output(self):
        """after_output_process can transform the result of tool-based output."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 5}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class DoubleOutputCap(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                if isinstance(output, MyOutput):
                    return MyOutput(value=output.value * 2)
                return output  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[DoubleOutputCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=10)


class TestHookComposition:
    """Multiple capabilities with output hooks compose correctly."""

    async def test_multiple_before_output_validate(self):
        """Multiple capabilities' before_output_validate hooks chain in order."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('cap1')
                return output

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('cap2')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[Cap1(), Cap2()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=1)
        assert log == ['cap1', 'cap2']

    async def test_chained_transformations(self):
        """Multiple capabilities can chain transformations in before_output_validate."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class AddExclamation(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return f'{output}!' if isinstance(output, str) else output

        @dataclass
        class AddQuestion(AbstractCapability[Any]):
            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                return f'{output}?' if isinstance(output, str) else output

        agent = Agent(FunctionModel(model_fn), capabilities=[AddExclamation(), AddQuestion()])
        result = await agent.run('hello')
        # after hooks run in reversed order: AddQuestion first, then AddExclamation
        assert result.output == 'hello?!'


class TestHooksClassOutputDecorators:
    """Test decorator registration for output hooks with Hooks class."""

    async def test_before_output_validate_decorator(self):
        """Hooks.on.before_output_validate registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_validate
        def fix_output(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_output_validate')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 3}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=3)
        assert log == ['before_output_validate']

    async def test_after_output_validate_decorator(self):
        """Hooks.on.after_output_validate registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.after_output_validate
        async def after_validate(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: Any,
        ) -> Any:
            log.append('after_output_validate')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 4}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=4)
        assert log == ['after_output_validate']

    async def test_wrap_output_validate_decorator(self):
        """Hooks.on.output_validate (wrap) registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.output_validate
        async def wrap_validate(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            handler: Any,
        ) -> Any:
            log.append('wrap_start')
            result = await handler(output)
            log.append('wrap_end')
            return result

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert log == ['wrap_start', 'wrap_end']

    async def test_on_output_validate_error_decorator(self):
        """Hooks.on.output_validate_error can recover from validation failures."""
        hooks = Hooks()

        @hooks.on.output_validate_error
        async def recover(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            error: ValidationError | ModelRetry,
        ) -> Any:
            return {'value': 999}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='not valid json')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        # Error recovery bypasses Pydantic validation, so the output is the raw dict
        assert result.output == {'value': 999}

    async def test_before_output_process_decorator(self):
        """Hooks.on.before_output_process registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_process
        async def before_exec(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_output_process')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 6}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=6)
        assert log == ['before_output_process']

    async def test_after_output_process_decorator(self):
        """Hooks.on.after_output_process transforms the final result."""
        hooks = Hooks()

        @hooks.on.after_output_process
        async def double_output(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: Any,
        ) -> Any:
            if isinstance(output, MyOutput):
                return MyOutput(value=output.value * 2)
            return output  # pragma: no cover

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 7}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=14)

    async def test_wrap_output_process_decorator(self):
        """Hooks.on.output_process (wrap) registers correctly."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.output_process
        async def wrap_exec(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            handler: Any,
        ) -> Any:
            log.append('exec_start')
            result = await handler(output)
            log.append('exec_end')
            return result

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 8}')])

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=8)
        assert log == ['exec_start', 'exec_end']

    async def test_sync_hook_auto_wrapping(self):
        """Sync output hook functions are auto-wrapped to async."""
        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_process
        def sync_hook(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: Any,
        ) -> Any:
            log.append('sync_before')
            return output

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks])
        result = await agent.run('hello')
        assert result.output == 'hello'
        assert log == ['sync_before']


class TestOutputHookFullLifecycle:
    """Test the full output hook lifecycle fires in the correct order."""

    async def test_full_validate_and_execute_order(self):
        """All output hooks fire in the expected order for structured text output."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        @dataclass
        class FullLifecycleCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_validate')
                return output

            async def wrap_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('wrap_validate:before')
                result = await handler(output)
                log.append('wrap_validate:after')
                return result

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_validate')
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_execute')
                return output

            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('wrap_execute:before')
                result = await handler(output)
                log.append('wrap_execute:after')
                return result

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_execute')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[FullLifecycleCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=1)
        assert log == [
            'before_validate',
            'wrap_validate:before',
            'wrap_validate:after',
            'after_validate',
            'before_execute',
            'wrap_execute:before',
            'wrap_execute:after',
            'after_execute',
        ]
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 1}')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_full_lifecycle_with_tool_output(self):
        """All output hooks fire in order for tool-based output."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 100}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class FullLifecycleCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_validate')
                assert output_context.mode == 'tool'
                assert output_context.tool_call is not None
                assert output_context.tool_def is not None
                return output

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_validate')
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_execute')
                return output

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_execute')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[FullLifecycleCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=100)
        assert log == [
            'before_validate',
            'after_validate',
            'before_execute',
            'after_execute',
        ]
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='final_result', args='{"value": 100}', tool_call_id='call-1')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id='call-1',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )


class TestOutputContext:
    """OutputContext is populated correctly for different output modes."""

    async def test_output_context_for_prompted_output(self):
        """OutputContext has correct fields for prompted text output."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'prompted'
        assert oc.output_type is MyOutput
        assert oc.object_def is not None
        assert oc.has_function is False
        assert oc.tool_call is None
        assert oc.tool_def is None

    async def test_output_context_for_plain_text(self):
        """OutputContext has correct fields for plain text output (via process hooks)."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'text'
        assert oc.output_type is str
        assert oc.object_def is None
        assert oc.has_function is False

    async def test_output_context_for_text_function(self):
        """OutputContext has correct fields for TextOutput function (via process hooks)."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('hello')

        def upcase(text: str) -> str:
            return text.upper()

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(upcase), capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'text'
        assert oc.output_type is str
        assert oc.has_function is True

    async def test_output_context_for_tool_output(self):
        """OutputContext has correct fields for tool-based output, including tool_call and tool_def."""
        captured: list[OutputContext] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 1}', tool_call_id='call-1')]
                )
            return make_text_response('no output tools')  # pragma: no cover

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                captured.append(output_context)
                return output

        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[CaptureCap()])
        await agent.run('hello')
        assert len(captured) == 1
        oc = captured[0]
        assert oc.mode == 'tool'
        assert oc.output_type is MyOutput
        assert oc.object_def is not None
        assert oc.has_function is False
        assert oc.tool_call is not None
        assert oc.tool_call.tool_name == 'final_result'
        assert oc.tool_def is not None
        assert oc.tool_def.name == 'final_result'
        assert oc.tool_def.kind == 'output'


class TestWrapOutputProcess:
    """wrap_output_process provides full middleware control around execution."""

    async def test_wrap_can_observe(self):
        """wrap_output_process can observe without modifying."""
        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class WrapCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                log.append('before')
                result = await handler(output)
                log.append('after')
                return result

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[WrapCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)
        assert log == ['before', 'after']

    async def test_wrap_can_replace_result(self):
        """wrap_output_process can replace the result entirely."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        @dataclass
        class ReplaceCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                handler: Any,
            ) -> Any:
                await handler(output)  # Call handler but ignore result
                return MyOutput(value=0)

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[ReplaceCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=0)


class TestOnOutputProcessError:
    """on_output_process_error can recover from execution failures."""

    async def test_recover_from_output_function_error(self):
        """on_output_process_error catches errors from output functions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('trigger error')

        def failing_func(text: str) -> str:
            raise ValueError('output function failed')

        @dataclass
        class RecoverCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'recovered'

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(failing_func), capabilities=[RecoverCap()])
        result = await agent.run('hello')
        assert result.output == 'recovered'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='trigger error')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_default_reraises(self):
        """Without a recovery hook, output execution errors propagate."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response('trigger error')

        def failing_func(text: str) -> str:
            raise ValueError('output function failed')

        agent = Agent(FunctionModel(model_fn), output_type=TextOutput(failing_func))
        with pytest.raises(ValueError, match='output function failed'):
            await agent.run('hello')


class TestRunSync:
    """Output hooks work with run_sync as well as run."""

    def test_before_output_validate_with_run_sync(self):
        """Output hooks fire correctly with agent.run_sync."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 77}')])

        hooks = Hooks()
        log: list[str] = []

        @hooks.on.before_output_validate
        def log_hook(
            ctx: RunContext[Any],
            /,
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
        ) -> str | dict[str, Any]:
            log.append('before_validate')
            return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=77)
        assert log == ['before_validate']


class TestOutputHookErrorPaths:
    """Test error paths to ensure correct error wrapping and hook firing."""

    def test_on_output_validate_error_reraise_wraps_in_tool_retry(self):
        """When on_output_validate_error re-raises ValidationError, it's wrapped in ToolRetryError causing retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not valid json')])
            return ModelResponse(parts=[TextPart(content='{"value": 42}')])

        error_log: list[str] = []

        @dataclass
        class ErrorLogCapability(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append(f'validate_error: {type(error).__name__}')
                raise error  # Re-raise — should cause retry

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[ErrorLogCapability()],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=42)
        assert call_count == 2
        assert len(error_log) == 1
        assert error_log[0] == 'validate_error: ValidationError'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='not valid json')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected ident at line 1 column 2',
                                    'input': 'not valid json',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 42}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=6),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_on_output_process_error_recovery(self):
        """on_output_process_error can recover from output function failure."""

        def bad_function(value: int) -> str:
            raise ValueError('value too small')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        @dataclass
        class RecoverCapability(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'recovered value'

        agent = Agent(
            FunctionModel(model_fn),
            output_type=bad_function,
            capabilities=[RecoverCapability()],
        )
        result = agent.run_sync('hello')
        assert result.output == 'recovered value'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 42}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_composed_on_output_validate_error_chain(self):
        """Multiple capabilities' on_output_validate_error hooks chain correctly."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[TextPart(content='invalid')])
            return ModelResponse(parts=[TextPart(content='{"value": 1}')])

        error_log: list[str] = []

        @dataclass
        class FirstCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('first_error')
                raise error

        @dataclass
        class SecondCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('second_error')
                raise error

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[FirstCap(), SecondCap()],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=1)
        # Both error hooks should have been called (reverse order per composition)
        assert 'second_error' in error_log
        assert 'first_error' in error_log
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='invalid')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected value at line 1 column 1',
                                    'input': 'invalid',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 1}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_composed_on_output_process_error_chain(self):
        """Multiple capabilities' on_output_process_error hooks chain correctly."""

        def failing_func(value: int) -> str:
            raise ValueError('intentional')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        @dataclass
        class FirstCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'recovered_by_first'

        @dataclass
        class SecondCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                raise error  # Don't recover, pass to next cap

        agent = Agent(
            FunctionModel(model_fn),
            output_type=failing_func,
            capabilities=[FirstCap(), SecondCap()],
        )
        result = agent.run_sync('hello')
        assert result.output == 'recovered_by_first'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 42}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_hooks_output_validate_error_decorator(self):
        """Test on_output_validate_error via Hooks decorator API."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) == 1:
                return ModelResponse(parts=[TextPart(content='bad json')])
            return ModelResponse(parts=[TextPart(content='{"value": 99}')])

        hooks = Hooks()

        @hooks.on.output_validate_error
        async def handle_error(
            ctx: RunContext[Any],
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            error: ValidationError | ModelRetry,
        ) -> Any:
            raise error  # Re-raise to trigger retry

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=99)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='bad json')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected value at line 1 column 1',
                                    'input': 'bad json',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 99}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_hooks_output_process_error_decorator(self):
        """Test on_output_process_error via Hooks decorator API."""

        def bad_function(value: int) -> str:
            raise ValueError('intentional failure')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 10}')])

        hooks = Hooks()

        @hooks.on.output_process_error
        async def handle_error(
            ctx: RunContext[Any],
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            error: Exception,
        ) -> Any:
            return 'fallback result'

        agent = Agent(
            FunctionModel(model_fn),
            output_type=bad_function,
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == 'fallback result'
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 10}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_tool_output_validate_error_hook_not_triggered_on_valid_data(self):
        """For tool output with valid data, on_output_validate_error does not fire."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 42}')])

        hooks = Hooks()
        error_log: list[str] = []

        @hooks.on.before_output_validate
        def log_validate(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            error_log.append('before_validate')
            return output

        agent = Agent(
            FunctionModel(model_fn),
            output_type=MyOutput,
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=42)
        assert error_log == ['before_validate']  # Validate fires but no error

    def test_wrapper_capability_output_hooks_delegate(self):
        """WrapperCapability delegates output hooks to wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        log: list[str] = []

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                log.append('inner_before_validate')
                return output

            async def after_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('inner_after_execute')
                return output

        @dataclass
        class OuterCap(WrapperCapability[Any]):
            pass

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[OuterCap(wrapped=InnerCap())],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=5)
        assert 'inner_before_validate' in log
        assert 'inner_after_execute' in log


class TestDefaultOutputErrorHooks:
    """Test that default (no override) error hooks work correctly via retry."""

    def test_default_on_output_validate_error_causes_retry(self):
        """Default on_output_validate_error re-raises, triggering model retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not json')])
            return ModelResponse(parts=[TextPart(content='{"value": 7}')])

        # Hooks with only a before_output_validate hook (no error hook override).
        # Default on_output_validate_error re-raises → ToolRetryError → model retry.
        hooks = Hooks()

        @hooks.on.before_output_validate
        def noop(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[hooks])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=7)
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='not json')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                {
                                    'type': 'json_invalid',
                                    'loc': (),
                                    'msg': 'Invalid JSON: expected ident at line 1 column 2',
                                    'input': 'not json',
                                }
                            ],
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"value": 7}')],
                    usage=RequestUsage(input_tokens=81, output_tokens=5),
                    model_name='function:model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    def test_default_on_output_process_error_reraises(self):
        """Default on_output_process_error re-raises the error."""

        def failing_func(value: int) -> str:
            raise ValueError('intentional')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        # Hooks with only a before_output_process hook (no error hook override).
        hooks = Hooks()

        @hooks.on.before_output_process
        def noop(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
        ) -> str | dict[str, Any]:
            return output

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[hooks])
        with pytest.raises(ValueError, match='intentional'):
            agent.run_sync('hello')


class TestStreamingOutputHooks:
    """Output hooks fire during streaming (partial and final validation)."""

    async def test_output_hooks_fire_during_streaming(self):
        """Validate hooks fire on partial attempts; execute hooks fire only when partial validation succeeds."""

        hook_calls: list[tuple[str, bool]] = []

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            # Stream the JSON response in chunks
            yield {0: DeltaToolCall(name='final_result', json_args='{"val')}
            yield {0: DeltaToolCall(json_args='ue": 42}')}

        @dataclass
        class StreamLogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                hook_calls.append(('before_validate', ctx.partial_output))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                hook_calls.append(('after_execute', ctx.partial_output))
                return output

        agent = Agent(FunctionModel(stream_function=stream_fn), output_type=MyOutput, capabilities=[StreamLogCap()])
        async with agent.run_stream('hello') as stream:
            outputs = [o async for o in stream.stream_output(debounce_by=None)]
        assert outputs[-1] == MyOutput(value=42)
        # Validate hooks fire on partial attempts AND the final result
        validate_calls = [(phase, partial) for phase, partial in hook_calls if phase == 'before_validate']
        assert any(partial for _, partial in validate_calls), 'Expected at least one partial validation call'
        assert any(not partial for _, partial in validate_calls), 'Expected at least one final validation call'
        # Execute hooks fire only when validation succeeds (partial or final)
        execute_calls = [(phase, partial) for phase, partial in hook_calls if phase == 'after_execute']
        assert any(not partial for _, partial in execute_calls), 'Expected at least one final execute call'
        assert stream.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": 42}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=4),
                    model_name='function::stream_fn',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )

    async def test_union_output_hooks_fire_during_streaming(self):
        """Union output types: hooks fire during partial and final validation, with the kind
        resolved per-invocation so concurrent streams can't clobber each other."""

        class TypeA(BaseModel):
            value: int

        class TypeB(BaseModel):
            name: str

        hook_calls: list[tuple[str, bool]] = []

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            yield {0: DeltaToolCall(name='final_result_TypeA', json_args='{"va')}
            yield {0: DeltaToolCall(json_args='lue": 7}')}

        @dataclass
        class StreamLogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                hook_calls.append(('before_validate', ctx.partial_output))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                hook_calls.append(('after_execute', ctx.partial_output))
                return output

        agent = Agent(
            FunctionModel(stream_function=stream_fn),
            output_type=[TypeA, TypeB],
            capabilities=[StreamLogCap()],
        )
        async with agent.run_stream('hello') as stream:
            outputs = [o async for o in stream.stream_output(debounce_by=None)]
        assert isinstance(outputs[-1], TypeA)
        assert outputs[-1].value == 7
        # Validate hooks fire on partial attempts AND final
        assert any(partial for phase, partial in hook_calls if phase == 'before_validate')
        assert any(not partial for phase, partial in hook_calls if phase == 'before_validate')
        # Execute hooks fire on final at minimum
        assert any(not partial for phase, partial in hook_calls if phase == 'after_execute')


class TestOutputHookEdgeCases:
    """Tests for edge cases to ensure full coverage of output hook code paths."""

    def test_before_output_validate_transforms_text_to_dict(self):
        """before_output_validate can transform raw text to a pre-parsed dict."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='ignored raw text')])

        @dataclass
        class PreParseCapability(AbstractCapability[Any]):
            async def before_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                # Transform text to a pre-parsed dict
                return {'value': 99}

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[PreParseCapability()],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=99)

    def test_streaming_output_hooks_fire_on_partial(self):
        """Process hooks fire for plain text output (validate hooks are skipped)."""
        from pydantic_ai.models.function import FunctionModel

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='hello world')])

        @dataclass
        class StreamLogCapability(AbstractCapability[Any]):
            async def before_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append(f'before_process partial={ctx.partial_output}')
                return output

        agent = Agent(FunctionModel(model_fn), capabilities=[StreamLogCapability()])
        result = agent.run_sync('hello')
        assert result.output == 'hello world'
        assert any('before_process' in entry for entry in log)

    def test_no_capability_fast_path_structured_raw_validation_error(self):
        """`ObjectOutputProcessor.hook_validate` — used by streaming paths without retries —
        must let `ValidationError` propagate unwrapped.
        """
        from pydantic_ai._output import ObjectOutputProcessor

        processor = ObjectOutputProcessor(output=MyOutput)

        ctx = RunContext(
            deps=None,
            model=None,  # pyright: ignore[reportArgumentType]
            usage=None,  # pyright: ignore[reportArgumentType]
            prompt='test',
            run_step=0,
            retry=0,
            max_retries=3,
            trace_include_content=False,
            tracer=NoOpTracer(),
            instrumentation_version=0,
        )
        with pytest.raises(ValidationError):
            processor.hook_validate('not valid json', run_context=ctx)

    def test_no_capability_fast_path_union_raw_validation_error(self):
        """Same as above but for `UnionOutputProcessor.hook_validate`."""
        from pydantic_ai._output import UnionOutputProcessor

        processor = UnionOutputProcessor(outputs=[MyOutput])

        ctx = RunContext(
            deps=None,
            model=None,  # pyright: ignore[reportArgumentType]
            usage=None,  # pyright: ignore[reportArgumentType]
            prompt='test',
            run_step=0,
            retry=0,
            max_retries=3,
            trace_include_content=False,
            tracer=NoOpTracer(),
            instrumentation_version=0,
        )
        with pytest.raises(ValidationError):
            processor.hook_validate('not valid json', run_context=ctx)

    def test_output_toolset_call_tool_raises(self):
        """`OutputToolset.call_tool` exists only to satisfy `AbstractToolset` — output tools go
        through `ToolManager.validate_output_tool_call` / `execute_output_tool_call`, never
        through the normal toolset path. Calling `call_tool` directly must raise.
        """
        import asyncio

        from pydantic_ai._output import OutputToolset

        toolset = OutputToolset.build([MyOutput])
        assert toolset is not None
        toolset.max_retries = 1  # Agent normally sets this; required by `get_tools`

        async def run():
            ctx = RunContext(
                deps=None,
                model=None,  # pyright: ignore[reportArgumentType]
                usage=None,  # pyright: ignore[reportArgumentType]
                prompt='test',
                run_step=0,
                retry=0,
                max_retries=3,
                trace_include_content=False,
                tracer=NoOpTracer(),
                instrumentation_version=0,
            )
            tools = await toolset.get_tools(ctx)
            tool_name = next(iter(tools))
            tool = tools[tool_name]
            await toolset.call_tool(tool_name, {}, ctx, tool)

        with pytest.raises(NotImplementedError, match='validate_output_tool_call'):
            asyncio.run(run())

    def test_hooks_on_output_process_via_hooks_class(self):
        """Test wrap_output_process via Hooks decorator API."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 10}')])

        hooks = Hooks()
        execute_log: list[str] = []

        @hooks.on.output_process
        async def wrap_exec(
            ctx: RunContext[Any],
            *,
            output_context: OutputContext,
            output: str | dict[str, Any],
            handler: Any,
        ) -> Any:
            execute_log.append('wrap_execute_before')
            result = await handler(output)
            execute_log.append('wrap_execute_after')
            return result

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput(MyOutput),
            capabilities=[hooks],
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=10)
        assert execute_log == ['wrap_execute_before', 'wrap_execute_after']


class TestErrorHookCoveragePaths:
    """Tests to exercise error hook delegation paths (abstract defaults, wrapper, hooks chaining)."""

    def test_bare_capability_default_on_output_validate_error(self):
        """A bare AbstractCapability subclass with no error hook override exercises default `raise error`."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not json')])
            return ModelResponse(parts=[TextPart(content='{"value": 3}')])

        @dataclass
        class BareCap(AbstractCapability[Any]):
            """Has no hook overrides — uses all defaults."""

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[BareCap()])
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=3)
        assert call_count == 2  # First attempt failed, retried

    def test_bare_capability_default_on_output_process_error(self):
        """A bare AbstractCapability subclass with no error hook override lets execute errors propagate."""

        def failing_func(value: int) -> str:
            raise ValueError('execute fail')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        @dataclass
        class BareCap(AbstractCapability[Any]):
            pass

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[BareCap()])
        with pytest.raises(ValueError, match='execute fail'):
            agent.run_sync('hello')

    def test_wrapper_on_output_validate_error_delegates(self):
        """WrapperCapability delegates on_output_validate_error to the wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='invalid')])
            return ModelResponse(parts=[TextPart(content='{"value": 8}')])

        error_log: list[str] = []

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('inner_error')
                raise error

        @dataclass
        class OuterWrap(WrapperCapability[Any]):
            pass

        agent = Agent(
            FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[OuterWrap(wrapped=InnerCap())]
        )
        result = agent.run_sync('hello')
        assert result.output == MyOutput(value=8)
        assert 'inner_error' in error_log

    def test_wrapper_on_output_process_error_delegates(self):
        """WrapperCapability delegates on_output_process_error to the wrapped capability."""
        from pydantic_ai.capabilities.wrapper import WrapperCapability

        def failing_func(value: int) -> str:
            raise ValueError('exec fail')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        @dataclass
        class InnerCap(AbstractCapability[Any]):
            async def on_output_process_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: Exception,
            ) -> Any:
                return 'wrapper_recovered'

        @dataclass
        class OuterWrap(WrapperCapability[Any]):
            pass

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[OuterWrap(wrapped=InnerCap())])
        result = agent.run_sync('hello')
        assert result.output == 'wrapper_recovered'

    def test_hooks_on_output_process_error_chaining(self):
        """Hooks class on_output_process_error re-raises, chaining errors."""

        def failing_func(value: int) -> str:
            raise ValueError('original')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')])

        hooks = Hooks()

        @hooks.on.output_process_error
        async def first_handler(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any], error: Exception
        ) -> Any:
            raise ValueError('chained')  # Re-raise different error

        @hooks.on.output_process_error
        async def second_handler(
            ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any], error: Exception
        ) -> Any:
            return 'recovered'  # This one recovers

        agent = Agent(FunctionModel(model_fn), output_type=failing_func, capabilities=[hooks])
        result = agent.run_sync('hello')
        assert result.output == 'recovered'


class TestUnionOutputWithHooks:
    """Tests for UnionOutputProcessor with output hooks — verifying clean validate/call decomposition."""

    def test_union_output_hooks_fire_for_both_phases(self):
        """Union output types properly split into validate (Pydantic) and execute (function call) phases."""

        class TypeA(BaseModel):
            kind: str = 'a'
            value: int

        class TypeB(BaseModel):
            kind: str = 'b'
            name: str

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "TypeA", "data": {"value": 42}}}')])

        @dataclass
        class LogCapability(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('before_validate')
                return output

            async def after_output_validate(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
            ) -> Any:
                log.append('after_validate')
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('before_execute')
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('after_execute')
                return output

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[LogCapability()],
        )
        result = agent.run_sync('hello')
        assert isinstance(result.output, TypeA)
        assert result.output.value == 42
        # Both validate and execute hooks should fire
        assert 'before_validate' in log
        assert 'after_validate' in log
        assert 'before_execute' in log
        assert 'after_execute' in log

    def test_union_output_process_hook_transforms_result(self):
        """Execute hooks can transform the result for union output types."""

        class TypeA(BaseModel):
            value: int

        class TypeB(BaseModel):
            name: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "TypeA", "data": {"value": 5}}}')])

        @dataclass
        class DoubleCapability(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                assert isinstance(output, TypeA)
                output.value *= 2
                return output

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[DoubleCapability()],
        )
        result = agent.run_sync('hello')
        assert isinstance(result.output, TypeA)
        assert result.output.value == 10

    def test_union_with_multi_arg_output_function_runs(self):
        """A multi-arg output function in a union must actually execute.

        Regression: `UnionOutputProcessor.hook_execute` previously isinstance-checked the
        validated dict against the function's first-arg type, which always failed for
        multi-arg functions, so the function was silently bypassed.
        """
        executed: list[tuple[int, str]] = []

        def combine(x: int, y: str) -> str:
            executed.append((x, y))
            return f'{x}:{y}'

        class Other(BaseModel):
            value: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Emit the discriminated union shape that PromptedOutput expects, selecting the
            # `combine` branch with the dict the multi-arg function will receive.
            return ModelResponse(
                parts=[TextPart(content='{"result": {"kind": "combine", "data": {"x": 7, "y": "ok"}}}')]
            )

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput([combine, Other]))
        result = agent.run_sync('hello')
        assert result.output == '7:ok'
        assert executed == [(7, 'ok')]

    def test_union_resolve_by_type_skips_multi_arg_inners(self):
        """When a process hook swaps the semantic value to a different type, `hook_execute`
        falls through to `_resolve_inner_for_value`. That fallback can't pick a multi-arg
        function inner because its `output_type` is just the first arg's type — it should
        skip multi-arg inners and only consider single-value inners (BaseModel, primitives).
        """

        def combine(x: int, y: str) -> str:  # pragma: no cover
            return f'{x}:{y}'

        class Single(BaseModel):
            value: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "Single", "data": {"value": 1}}}')])

        @dataclass
        class SwapToInt(AbstractCapability[Any]):
            """Swap the validated `Single` instance for a bare `int` during the process
            phase, so the value no longer matches `Single`'s type. The fallthrough resolver
            should iterate inners — skip `combine` (multi-arg, can't isinstance-check),
            and not find any matching single-value inner for `int` since `Single` is the
            only single-value inner and the int isn't a `Single`."""

            async def wrap_output_process(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: Any,
                handler: Callable[[Any], Awaitable[Any]],
            ) -> Any:
                return await handler(99)

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([combine, Single]),
            capabilities=[SwapToInt()],
        )
        # No matching inner found → semantic returned unmodified.
        result = agent.run_sync('hello')
        assert result.output == 99

    def test_union_on_output_validate_error_fires(self):
        """on_output_validate_error fires for union output when validation fails."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='not json')])
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "MyOutput", "data": {"value": 1}}}')])

        error_log: list[str] = []

        @dataclass
        class ErrorLogCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                error_log.append('validate_error')
                raise error

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([MyOutput, MyOutput]),
            capabilities=[ErrorLogCap()],
        )
        result = agent.run_sync('hello')
        assert isinstance(result.output, MyOutput)
        assert call_count == 2
        assert 'validate_error' in error_log

    async def test_union_error_hook_recovery(self):
        """on_output_validate_error can recover for union types without crashing."""

        class TypeA(BaseModel):
            a_val: int

        class TypeB(BaseModel):
            b_val: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Return invalid union JSON — missing 'result' envelope
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverUnionCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                # Recover with a pre-built result
                return TypeA(a_val=42)

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[RecoverUnionCap()],
        )
        result = await agent.run('hello')
        assert result.output == TypeA(a_val=42)

    async def test_union_error_hook_recovery_second_type(self):
        """Error recovery matching the second union type exercises the isinstance loop."""

        class TypeA(BaseModel):
            a_val: int

        class TypeB(BaseModel):
            b_val: str

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverUnionCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                # Recover with TypeB — the second union member — so isinstance(output, TypeA)
                # fails first, then isinstance(output, TypeB) succeeds
                return TypeB(b_val='recovered')

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([TypeA, TypeB]),
            capabilities=[RecoverUnionCap()],
        )
        result = await agent.run('hello')
        assert result.output == TypeB(b_val='recovered')

    async def test_union_error_hook_recovery_with_primitive(self):
        """Union mixing a BaseModel with a primitive (`Foo | bool | None`).

        `bool` gets an `outer_typed_dict_key='response'` wrapper; recovery must rewrap the
        primitive into the inner processor's dict shape before calling.
        """

        class Foo(BaseModel):
            x: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverPrimitiveCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                return True  # recover with a bool, matching the second union member

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, bool]),
            capabilities=[RecoverPrimitiveCap()],
        )
        result = await agent.run('hello')
        assert result.output is True

    async def test_union_error_hook_recovery_with_generic(self):
        """Union mixing a BaseModel with a generic (`Foo | list[Bar]`).

        `isinstance(x, list[Bar])` raises `TypeError`; resolution must fall back to the
        generic origin (`list`) so the recovered list-valued output still maps to its
        inner processor.
        """

        class Foo(BaseModel):
            x: int

        class Bar(BaseModel):
            y: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverListCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                return [Bar(y=1), Bar(y=2)]

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, list[Bar]]),
            capabilities=[RecoverListCap()],
        )
        result = await agent.run('hello')
        assert result.output == [Bar(y=1), Bar(y=2)]

    async def test_union_after_validate_hook_swaps_union_member(self):
        """`after_output_validate` can return a value of a different union member.

        If the validated kind was `Foo` but a hook returned a `Bar`, `hook_execute` must
        fall through to type-based resolution instead of passing a `Bar` to `Foo`'s inner
        processor.
        """

        class Foo(BaseModel):
            kind: str = 'Foo'
            x: int

        class Bar(BaseModel):
            kind: str = 'Bar'
            y: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"result": {"kind": "Foo", "data": {"x": 1}}}')])

        @dataclass
        class SwapUnionCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                # Model said "Foo", hook swaps to "Bar" — execute must route to Bar's processor.
                return Bar(y=42)

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, Bar]),
            capabilities=[SwapUnionCap()],
        )
        result = await agent.run('hello')
        assert result.output == Bar(y=42)

    async def test_union_hook_returns_unknown_type_passes_through(self):
        """If a hook returns a value matching NO union member, `hook_execute` passes it through.

        The output function (if any) doesn't run, and the value reaches the user as-is —
        better than silently dropping to `None`.
        """

        class Foo(BaseModel):
            x: int

        class Bar(BaseModel):
            y: int

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"bad": "data"}')])

        @dataclass
        class RecoverUnknownCap(AbstractCapability[Any]):
            async def on_output_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                output_context: OutputContext,
                output: str | dict[str, Any],
                error: ValidationError | ModelRetry,
            ) -> Any:
                return 'not in union'  # str isn't Foo or Bar

        agent = Agent(
            FunctionModel(model_fn),
            output_type=PromptedOutput([Foo, Bar]),
            capabilities=[RecoverUnknownCap()],
        )
        result = await agent.run('hello')
        assert result.output == 'not in union'


class TestTextFunctionOutputCallHook:
    """Tests that TextFunctionOutputProcessor.call() is exercised through execute hooks."""

    def test_text_function_execute_hook_wraps_call(self):
        """Execute hooks wrap the text function call (processor.call)."""

        def uppercase(text: str) -> str:
            return text.upper()

        log: list[str] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='hello world')])

        @dataclass
        class ExecLogCap(AbstractCapability[Any]):
            async def wrap_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any, handler: Any
            ) -> Any:
                log.append(f'input: {output}')
                result = await handler(output)
                log.append(f'output: {result}')
                return result

        agent = Agent(
            FunctionModel(model_fn),
            output_type=TextOutput(uppercase),
            capabilities=[ExecLogCap()],
        )
        result = agent.run_sync('hello')
        assert result.output == 'HELLO WORLD'
        assert log == ['input: hello world', 'output: HELLO WORLD']


class TestNativeOutputWithHooks:
    """Output hooks fire for native structured output mode."""

    async def test_hooks_fire_for_native_output(self):
        """Output hooks fire with mode='native' for NativeOutput."""
        log: list[tuple[str, str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 7}')])

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append(('before_validate', output_context.mode))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('after_execute', output_context.mode))
                return output

        agent = Agent(FunctionModel(model_fn), output_type=NativeOutput(MyOutput), capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=7)
        assert log == [('before_validate', 'native'), ('after_execute', 'native')]

    async def test_before_validate_transforms_native_output(self):
        """before_output_validate can transform raw text before native output parsing."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": "bad"}')])

        @dataclass
        class FixCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                if isinstance(output, str):
                    return output.replace('"bad"', '42')
                return output  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), output_type=NativeOutput(MyOutput), capabilities=[FixCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=42)

    async def test_model_retry_from_native_output_hook(self):
        """ModelRetry from output hooks triggers retry for native output."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class RejectCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value < 0:
                    raise ModelRetry('Value must be non-negative')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=NativeOutput(MyOutput), capabilities=[RejectCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2


class TestImageOutputWithHooks:
    """Image output fires process hooks (not validate hooks, since there's no parsing)."""

    async def test_process_hooks_fire_for_image_output(self):
        """Process hooks fire for image output; validate hooks are skipped."""
        log: list[str] = []

        def return_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'test-png', media_type='image/png'))])

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append('validate')  # pragma: no cover — should NOT fire for images
                return output  # pragma: no cover

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(f'process:{output_context.mode}')
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append('after_process')
                assert isinstance(output, BinaryImage)
                return output

        image_profile = ModelProfile(supports_image_output=True)
        agent = Agent(
            FunctionModel(return_image, profile=image_profile), output_type=BinaryImage, capabilities=[LogCap()]
        )
        result = await agent.run('hello')
        assert isinstance(result.output, BinaryImage)
        assert result.output.data == b'test-png'
        # Process hooks fire; validate hooks do NOT (no parsing for images)
        assert log == ['process:image', 'after_process']

    async def test_image_process_hook_can_transform(self):
        """Process hooks can transform image output."""

        def return_image(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'original', media_type='image/png'))])

        @dataclass
        class TransformCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, BinaryImage):
                    return BinaryImage(data=b'transformed', media_type=output.media_type)
                return output  # pragma: no cover

        image_profile = ModelProfile(supports_image_output=True)
        agent = Agent(
            FunctionModel(return_image, profile=image_profile), output_type=BinaryImage, capabilities=[TransformCap()]
        )
        result = await agent.run('hello')
        assert isinstance(result.output, BinaryImage)
        assert result.output.data == b'transformed'


class TestAutoModeOutputWithHooks:
    """Output hooks fire for auto mode (which delegates to tool or text based on model)."""

    async def test_hooks_fire_for_auto_mode_tool_path(self):
        """Auto mode that resolves to tool output fires output hooks."""
        log: list[tuple[str, str]] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Auto mode with default tool profile — model uses output tools
            if info.output_tools:
                tool = info.output_tools[0]
                return ModelResponse(
                    parts=[ToolCallPart(tool_name=tool.name, args='{"value": 99}', tool_call_id='call-1')]
                )
            return ModelResponse(parts=[TextPart(content='{"value": 99}')])  # pragma: no cover

        @dataclass
        class LogCap(AbstractCapability[Any]):
            async def before_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: str | dict[str, Any]
            ) -> str | dict[str, Any]:
                log.append(('before_validate', output_context.mode))
                return output

            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('after_execute', output_context.mode))
                return output

        # Default auto mode — FunctionModel defaults to tool mode
        agent = Agent(FunctionModel(model_fn), output_type=MyOutput, capabilities=[LogCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=99)
        assert log == [('before_validate', 'tool'), ('after_execute', 'tool')]


class TestHookSemanticValue:
    """Output hooks see the **semantic value** (what the model was asked to produce), not the
    internal dict-wrapped form used by the validator pipeline.

    This is intentionally different from *tool* call hooks, which always see `dict[str, Any]`
    (matching the tool schema the model satisfies). For outputs, users think of
    `Agent(output_type=T)` as "the model produces a T", so hooks should see T.
    """

    async def _run_and_capture(
        self,
        *,
        output_type: Any,
        model_fn: Any,
    ) -> tuple[Any, list[tuple[str, Any]]]:
        log: list[tuple[str, Any]] = []

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('after_validate', output))
                return output

            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                log.append(('before_process', output))
                return output

        agent = Agent(FunctionModel(model_fn), output_type=output_type, capabilities=[CaptureCap()])
        result = await agent.run('hello')
        return result.output, log

    async def test_case_a_bare_basemodel_tool_output(self):
        """Case A: `Agent(output_type=MyOutput)` — hooks see the BaseModel instance."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"value": 42}')])

        output, log = await self._run_and_capture(output_type=MyOutput, model_fn=model_fn)
        assert output == MyOutput(value=42)
        assert log == [('after_validate', MyOutput(value=42)), ('before_process', MyOutput(value=42))]

    async def test_case_b_bare_int_tool_output(self):
        """Case B: `Agent(output_type=int)` — hooks see `42`, not `{'response': 42}`."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"response": 42}')])

        output, log = await self._run_and_capture(output_type=int, model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_case_c_function_basemodel_arg(self):
        """Case C: `def f(data: MyOutput) -> int` — hooks see `MyOutput(...)`, not `{'data': MyOutput(...)}`."""

        def double(data: MyOutput) -> int:
            return data.value * 2

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"value": 21}')])

        output, log = await self._run_and_capture(output_type=double, model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', MyOutput(value=21)), ('before_process', MyOutput(value=21))]

    async def test_case_d_function_primitive_arg(self):
        """Case D: `def f(data: int) -> str` — hooks see `42`, not `{'data': 42}`."""

        def stringify(data: int) -> str:
            return f'got {data}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"data": 42}')])

        output, log = await self._run_and_capture(output_type=stringify, model_fn=model_fn)
        assert output == 'got 42'
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_case_e_function_multiple_args(self):
        """Case E: multi-arg function — hooks see the dict (genuine multi-value input)."""

        def combine(data: MyOutput, other: str) -> str:
            return f'{data.value}:{other}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"data": {"value": 7}, "other": "x"}')])

        output, log = await self._run_and_capture(output_type=combine, model_fn=model_fn)
        assert output == '7:x'
        # Multi-arg: hooks see the dict
        assert log == [
            ('after_validate', {'data': MyOutput(value=7), 'other': 'x'}),
            ('before_process', {'data': MyOutput(value=7), 'other': 'x'}),
        ]

    async def test_native_output_unwraps_primitive(self):
        """NativeOutput(int) — hooks see `42`, not `{'response': 42}`."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"response": 42}')])

        output, log = await self._run_and_capture(output_type=NativeOutput(int), model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_native_output_unwraps_function_basemodel(self):
        """NativeOutput(func-with-basemodel-arg) — hooks see the BaseModel, not the wrap dict."""

        def double(data: MyOutput) -> int:
            return data.value * 2

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"value": 21}')])

        output, log = await self._run_and_capture(output_type=NativeOutput(double), model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', MyOutput(value=21)), ('before_process', MyOutput(value=21))]

    async def test_prompted_output_unwraps_primitive(self):
        """PromptedOutput(int) — hooks see `42`, not `{'response': 42}`."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"response": 42}')])

        output, log = await self._run_and_capture(output_type=PromptedOutput(int), model_fn=model_fn)
        assert output == 42
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_prompted_output_unwraps_function_primitive(self):
        """PromptedOutput(func-with-primitive-arg) — hooks see the primitive value."""

        def stringify(data: int) -> str:
            return f'got {data}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content='{"data": 42}')])

        output, log = await self._run_and_capture(output_type=PromptedOutput(stringify), model_fn=model_fn)
        assert output == 'got 42'
        assert log == [('after_validate', 42), ('before_process', 42)]

    async def test_output_validator_sees_final_processed_value(self):
        """Output validators see the final value (after function call), not the wrapped form."""

        def double(data: MyOutput) -> int:
            return data.value * 2

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"value": 21}')])

        seen: list[Any] = []
        agent = Agent(FunctionModel(model_fn), output_type=double)

        @agent.output_validator
        def validate(v: int) -> int:
            seen.append(v)
            return v

        result = await agent.run('hello')
        assert result.output == 42
        # Validator sees the post-process value (function's return), an int
        assert seen == [42]

    async def test_hook_transform_at_semantic_boundary(self):
        """A hook can transform the semantic value and the transformed value flows through correctly."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"response": 10}')])

        @dataclass
        class DoubleCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                return output * 2  # transform the semantic int value

        agent = Agent(FunctionModel(model_fn), output_type=int, capabilities=[DoubleCap()])
        result = await agent.run('hello')
        assert result.output == 20

    async def test_dict_output_type_contains_unwrap_key(self):
        """Regression: `output_type=dict[str, Any]` where the dict contains the unwrap key
        ('response') must not be mistaken for an already-wrapped value during re-wrap.
        """

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            assert info.output_tools is not None
            tool = info.output_tools[0]
            # The dict itself contains a 'response' key — the same key used as the outer wrapper
            return ModelResponse(parts=[ToolCallPart(tool.name, '{"response": {"response": "yes", "other": "stuff"}}')])

        output, log = await self._run_and_capture(output_type=dict[str, Any], model_fn=model_fn)
        # Hook sees the inner dict (unwrapped)
        assert log == [
            ('after_validate', {'response': 'yes', 'other': 'stuff'}),
            ('before_process', {'response': 'yes', 'other': 'stuff'}),
        ]
        # Final output is the full inner dict — NOT just "yes" (which would happen if re-wrap
        # was skipped due to the buggy "already wrapped" check)
        assert output == {'response': 'yes', 'other': 'stuff'}


class TestHookExceptionHandling:
    """ValidationError/ModelRetry raised from before_* and after_* hooks should trigger retry,
    matching the behavior when raised from wrap_output_validate/wrap_output_process.
    """

    async def test_validation_error_from_after_output_validate_triggers_retry(self):
        """ValidationError from after_output_validate should be caught and trigger model retry."""
        from pydantic import TypeAdapter

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class StricterCap(AbstractCapability[Any]):
            async def after_output_validate(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                # Additional Pydantic validation: reject negative values
                if isinstance(output, MyOutput) and output.value < 0:
                    # Simulate Pydantic validation failing
                    TypeAdapter(int).validate_python('not_an_int')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[StricterCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2  # retry happened

    async def test_validation_error_from_after_output_process_triggers_retry(self):
        """ValidationError from after_output_process should be caught and trigger model retry."""
        from pydantic import TypeAdapter

        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class StricterCap(AbstractCapability[Any]):
            async def after_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value < 0:
                    TypeAdapter(int).validate_python('not_an_int')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[StricterCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2

    async def test_model_retry_from_before_output_process_triggers_retry(self):
        """ModelRetry from before_output_process should trigger model retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(parts=[TextPart(content='{"value": -1}')])
            return ModelResponse(parts=[TextPart(content='{"value": 5}')])

        @dataclass
        class RejectCap(AbstractCapability[Any]):
            async def before_output_process(
                self, ctx: RunContext[Any], *, output_context: OutputContext, output: Any
            ) -> Any:
                if isinstance(output, MyOutput) and output.value < 0:
                    raise ModelRetry('Value must be non-negative')
                return output

        agent = Agent(FunctionModel(model_fn), output_type=PromptedOutput(MyOutput), capabilities=[RejectCap()])
        result = await agent.run('hello')
        assert result.output == MyOutput(value=5)
        assert call_count == 2


# region HandleDeferredToolCalls


async def test_deferred_tool_handler_approve():
    """HandleDeferredToolCalls capability auto-approves a requires_approval tool inline."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done!')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done!'
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={'x': 5}, tool_call_id='call1')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content=50,
                        tool_call_id='call1',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deferred_tool_handler_deny():
    """HandleDeferredToolCalls capability denies a requires_approval tool inline, producing a `ToolReturnPart(outcome='denied')`."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Understood, denied.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Not allowed.') for call in requests.approvals}
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood, denied.'
    # The denial must surface in message history as outcome='denied', not a successful return.
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].tool_call_id == 'call1'
    assert tool_returns[0].outcome == 'denied'
    assert tool_returns[0].content == 'Not allowed.'


async def test_deferred_tool_handler_no_output_type_needed():
    """When handler resolves all deferred calls, DeferredToolRequests is not needed in output type."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 3}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Result received.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # Note: output_type is just str, no DeferredToolRequests
    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 100

    result = await agent.run('Hello')
    assert result.output == 'Result received.'


async def test_deferred_tool_handler_none_fallback():
    """When no handler is present, deferred tools bubble up as DeferredToolRequests output."""

    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise ApprovalRequired

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1


async def test_deferred_tool_handler_partial_resolution():
    """Handler resolves some calls, remaining bubble up as DeferredToolRequests output."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('tool_a', {}, tool_call_id='a1'),
                ToolCallPart('tool_b', {}, tool_call_id='b1'),
            ]
        )

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Only approve tool_a, leave tool_b unresolved
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a done'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b done'  # pragma: no cover

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.approvals[0].tool_name == 'tool_b'


async def test_deferred_tool_handler_sync_handler():
    """HandleDeferredToolCalls works with a sync handler function."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('OK')])

    def handle_deferred_sync(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred_sync)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'OK'


async def test_deferred_tool_handler_accumulation():
    """Two capabilities each resolve different deferred calls."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, tool_call_id='a1'),
                    ToolCallPart('tool_b', {}, tool_call_id='b1'),
                ]
            )
        return ModelResponse(parts=[TextPart('Both done.')])

    def handler_a(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    def handler_b(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # handler_a resolved tool_a, so we only see tool_b
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[
            HandleDeferredToolCalls(handler=handler_a),
            HandleDeferredToolCalls(handler=handler_b),
        ],
    )

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a result'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b result'

    result = await agent.run('Hello')
    assert result.output == 'Both done.'


async def test_deferred_tool_handler_unresolved_no_output_type_error():
    """Unresolved deferred calls without DeferredToolRequests in output type raises UserError."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    # Handler returns None → does not resolve
    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults()  # Empty results → nothing resolved

    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    with pytest.raises(UserError, match='DeferredToolRequests'):
        await agent.run('Hello')


async def test_deferred_tool_handler_external_call():
    """HandleDeferredToolCalls capability resolves an externally-executed tool."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 3}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Got it.')])

    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.messages import ToolReturn

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Simulate external execution: return a ToolReturn with metadata
        return DeferredToolResults(
            calls={
                call.tool_call_id: ToolReturn(return_value='external result', metadata={'source': 'ext'})
                for call in requests.calls
            }
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        raise CallDeferred

    result = await agent.run('Hello')
    assert result.output == 'Got it.'


async def test_deferred_tool_handler_via_handle_call():
    """handle_call(resolve_deferred=True) resolves deferred tools inline via ToolManager."""

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('All done.')])

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        result = await ctx.tool_manager.handle_call(inner_call)
        return f'inner returned: {result}'

    @agent.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_handle_call_wrap_validation_errors_false():
    """`wrap_validation_errors=False` propagates through deferred-tool resolution.

    Regression for the case where a sandboxed caller (`handle_call(wrap_validation_errors=False)`)
    invokes a tool that requires approval: after the handler approves, the re-execution must
    keep the raw-error contract — `ModelRetry` from the approved tool body should propagate
    as-is, not wrapped as `ToolRetryError`.
    """

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('Done.')])

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            await ctx.tool_manager.handle_call(inner_call, wrap_validation_errors=False)
        except ModelRetry as e:
            return f'raw ModelRetry: {e}'
        return 'no error'  # pragma: no cover

    @agent.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('post-approval retry')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # outer_tool caught the raw ModelRetry from the approved inner_tool body and surfaced it
    # in its return value; if wrap_validation_errors hadn't been forwarded through
    # _resolve_single_deferred, outer_tool would have seen a ToolRetryError instead.
    inner_message = next(
        msg
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_name == 'outer_tool' for part in msg.parts)
    )
    outer_return = next(
        part for part in inner_message.parts if isinstance(part, ToolReturnPart) and part.tool_name == 'outer_tool'
    )
    assert outer_return.content == 'raw ModelRetry: post-approval retry'


async def test_deferred_tool_handler_via_handle_call_no_handler():
    """handle_call(resolve_deferred=True) re-raises when no handler is available."""
    from pydantic_ai.toolsets import FunctionToolset

    # inner_tool is only available via ToolManager, not as a top-level agent tool
    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'  # pragma: no cover

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('OK')])

    agent = Agent(FunctionModel(llm), toolsets=[inner_toolset])

    @agent.tool
    async def outer_tool(ctx: RunContext) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            result = await ctx.tool_manager.handle_call(inner_call)
            return f'inner returned: {result}'  # pragma: no cover
        except ApprovalRequired:
            return 'inner needs approval'

    result = await agent.run('Hello')
    assert result.output == 'OK'


async def test_deferred_tool_handler_build_results_helper():
    """DeferredToolRequests.build_results() creates a DeferredToolResults."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'Done.'


def test_deferred_tool_requests_build_results_validates_ids():
    """build_results rejects result IDs that don't match a pending request of the right kind."""
    requests = DeferredToolRequests(
        approvals=[ToolCallPart('a', {}, tool_call_id='approval_1')],
        calls=[ToolCallPart('b', {}, tool_call_id='call_1')],
    )

    # Mis-routed ID: tool-result provided for something in the approvals list.
    with pytest.raises(ValueError, match=r'calls.*not in.*DeferredToolRequests.calls'):
        requests.build_results(calls={'approval_1': 'oops'})

    # Unknown ID entirely.
    with pytest.raises(ValueError, match=r'approvals.*not in.*DeferredToolRequests.approvals'):
        requests.build_results(approvals={'unknown_id': True})

    # Happy path still works.
    results = requests.build_results(approvals={'approval_1': True}, calls={'call_1': 'result'})
    assert results.approvals == {'approval_1': True}
    assert results.calls == {'call_1': 'result'}


def test_deferred_tool_requests_build_results_approve_all():
    """approve_all=True approves every pending approval not explicitly specified."""
    requests = DeferredToolRequests(
        approvals=[
            ToolCallPart('a', {}, tool_call_id='approval_1'),
            ToolCallPart('b', {}, tool_call_id='approval_2'),
            ToolCallPart('c', {}, tool_call_id='approval_3'),
        ],
    )

    # Explicit deny wins; the other two get auto-approved.
    results = requests.build_results(
        approvals={'approval_1': False},
        approve_all=True,
    )
    assert results.approvals['approval_1'] is False
    assert isinstance(results.approvals['approval_2'], ToolApproved)
    assert isinstance(results.approvals['approval_3'], ToolApproved)


async def test_deferred_tool_handler_wrapper_capability():
    """HandleDeferredToolCalls works through WrapperCapability delegation."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # PrefixTools wraps HandleDeferredToolCalls — tests WrapperCapability delegation
    inner = HandleDeferredToolCalls(handler=handle_deferred)
    agent = Agent(
        FunctionModel(llm),
        capabilities=[inner.prefix_tools('ns')],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'

    result = await agent.run('Hello')
    assert result.output == 'Done.'


async def test_deferred_tool_handler_external_call_plain_value():
    """HandleDeferredToolCalls resolves an external call with a plain value (not ToolReturn)."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Got it.')])

    from pydantic_ai.exceptions import CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: 'plain string result' for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool_plain
    def my_tool() -> str:
        raise CallDeferred

    result = await agent.run('Hello')
    assert result.output == 'Got it.'


async def test_deferred_tool_handler_re_deferred_with_metadata():
    """When an approved tool re-raises ApprovalRequired, it stays unresolved with metadata."""

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        # Always requires approval — even when approved, raises again with metadata
        raise ApprovalRequired(metadata={'attempt': call_count})

    result = await agent.run('Hello')
    # Tool re-raised after approval → goes to remaining → becomes output
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    assert result.output.metadata.get('call1') == {'attempt': 2}


async def test_deferred_tool_handler_denied_via_batch():
    """Batch path deny via handler produces a `ToolReturnPart(outcome='denied')` in message history."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Understood.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Policy denied.') for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood.'
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].outcome == 'denied'
    assert tool_returns[0].content == 'Policy denied.'


async def test_deferred_tool_handler_batch_deny_via_bool_and_default():
    """Batch path: covers `approvals[id] = False` AND default `ToolDenied()` as separate calls."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('needs_approval', {'x': 1}, tool_call_id='bool_false'),
                    ToolCallPart('needs_approval', {'x': 2}, tool_call_id='default_denied'),
                ]
            )
        return ModelResponse(parts=[TextPart('ok')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={
                'bool_false': False,
                'default_denied': ToolDenied(),  # no custom message
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x  # pragma: no cover

    result = await agent.run('go')
    assert result.output == 'ok'
    tool_returns = {p.tool_call_id: p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)}
    assert tool_returns['bool_false'].outcome == 'denied'
    assert tool_returns['bool_false'].content == ToolDenied().message
    assert tool_returns['default_denied'].outcome == 'denied'
    assert tool_returns['default_denied'].content == ToolDenied().message


async def test_deferred_tool_handler_batch_approve_via_tool_approved_default():
    """Batch path: covers `approvals[id] = ToolApproved()` (default, no override_args)."""
    from pydantic_ai.tools import ToolApproved

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('needs_approval', {'x': 7}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 2

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].outcome != 'denied'
    assert tool_returns[0].content == 14


async def test_deferred_tool_handler_batch_external_tool_return_metadata():
    """Batch path: handler-supplied external `ToolReturn(value, metadata)` lands on the return part."""
    from pydantic_ai.messages import ToolReturn as _ToolReturn

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: _ToolReturn(
                    return_value='computed', metadata={'source': 'external'}, content='user extra'
                )
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'done'
    messages = result.all_messages()
    tool_returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'computed'
    assert tool_returns[0].metadata == {'source': 'external'}
    # The `content` field on ToolReturn becomes a UserPromptPart.
    from pydantic_ai.messages import UserPromptPart

    user_extras = [p for p in iter_message_parts(messages, ModelRequest, UserPromptPart) if p.content == 'user extra']
    assert len(user_extras) == 1


async def test_deferred_tool_handler_batch_external_model_retry():
    """Batch path: handler-supplied `ModelRetry` in `calls` surfaces as a `RetryPromptPart`, not a tool return."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('retried')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('try again') for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    messages = result.all_messages()
    retry_parts = list(iter_message_parts(messages, ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].content == 'try again'
    tool_returns = [p for p in iter_message_parts(messages, ModelRequest, ToolReturnPart) if p.tool_call_id == 'c1']
    assert tool_returns == []


async def test_deferred_tool_handler_batch_external_retry_prompt_part():
    """Batch path: handler-supplied `RetryPromptPart` in `calls` surfaces as a retry (names stamped from the deferred call)."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart('external_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('retried')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    retry_parts = list(iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart))
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].tool_name == 'external_tool'
    assert retry_parts[0].content == 'retry via part'


async def test_deferred_tool_handler_via_handle_call_external_tool_return():
    """Per-call path: handler-supplied external `ToolReturn(value, metadata)` is returned verbatim from handle_call."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.messages import ToolReturn as _ToolReturn
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={call.tool_call_id: _ToolReturn(return_value='ext', metadata={'k': 'v'}) for call in requests.calls}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    # Per-call path returns whatever the handler supplied verbatim — full ToolReturn wrapper preserved.
    assert isinstance(captured_result, _ToolReturn)
    assert captured_result.return_value == 'ext'
    assert captured_result.metadata == {'k': 'v'}


async def test_deferred_tool_handler_via_handle_call_tool_failed():
    """Per-call path: handler-supplied `ToolFailed` raises `ToolFailedError`, matching a tool that raises `ToolFailed` in-process."""
    from pydantic_ai.exceptions import CallDeferred, ToolFailed, ToolFailedError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext[object], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={call.tool_call_id: ToolFailed('backend unavailable') for call in requests.calls}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_error: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext[object]) -> str:
        nonlocal captured_error
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
        except ToolFailedError as e:
            captured_error = e
        return 'done'

    await agent.run('go')
    assert captured_error is not None
    assert captured_error.tool_failed.tool_name == 'inner_tool'
    assert captured_error.tool_failed.tool_call_id == 'inner_1'
    assert captured_error.tool_failed.content == 'backend unavailable'
    assert captured_error.tool_failed.outcome == 'failed'


def test_deferred_tool_handler_serialization_name():
    """HandleDeferredToolCalls is not spec-constructible."""
    assert HandleDeferredToolCalls.get_serialization_name() is None


async def test_deferred_tool_handler_via_handle_call_with_resolve():
    """handle_call(resolve_deferred=True) goes through _resolve_single_deferred happy path.

    This exercises the per-call resolution path used by CodeMode-style callers.
    """
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved result'

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        # Call inner_tool via handle_call — exercises _resolve_single_deferred
        result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        # _resolve_single_deferred returns result_part.content
        assert result == 'approved result'
        return f'got: {result}'

    result = await agent.run('go')
    assert result.output == 'final'
    # Verify the inner tool was called (tool return visible in messages)
    tool_returns = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
        if p.tool_name == 'caller_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'got: approved result'


async def test_deferred_tool_handler_approved_tool_returns_tool_return():
    """Approved tool returning a ToolReturn preserves metadata and user content."""
    from pydantic_ai.messages import ToolReturn as _ToolReturn, UserPromptPart

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='result', metadata={'source': 'tool'}, content='user prompt extra')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # Verify ToolReturn.metadata preserved
    tool_returns = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart) if p.tool_name == 'my_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].metadata == {'source': 'tool'}
    # Verify ToolReturn.content appears as UserPromptPart
    user_parts = [
        p
        for p in iter_message_parts(result.all_messages(), ModelRequest, UserPromptPart)
        if p.content == 'user prompt extra'
    ]
    assert len(user_parts) == 1


async def test_deferred_tool_handler_approved_tool_raises_model_retry():
    """Approved tool that raises ModelRetry produces a RetryPromptPart."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Retried and done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('try again')

    result = await agent.run('Hello')
    assert result.output == 'Retried and done.'
    # Verify the retry happened
    retry_parts = [
        p for p in iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart) if p.tool_name == 'my_tool'
    ]
    assert len(retry_parts) == 1


async def test_deferred_tool_handler_approved_tool_override_args():
    """Approved tool with ToolApproved(override_args=...) uses the override."""
    from pydantic_ai.tools import ToolApproved

    received_x = None

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done.')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Override the args: replace x=5 with x=42
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        nonlocal received_x
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        received_x = x
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    assert received_x == 42  # Override was applied


async def test_deferred_tool_handler_via_handle_call_retry():
    """handle_call path: approved tool raising ModelRetry propagates ToolRetryError."""
    from pydantic_ai.exceptions import ToolRetryError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()
    retry_count = 0

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        nonlocal retry_count
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        retry_count += 1
        raise ModelRetry('try again')

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no retry'  # pragma: no cover
        except ToolRetryError:
            return 'got retry'

    result = await agent.run('go')
    assert result.output == 'final'
    assert retry_count == 1


async def test_deferred_tool_handler_re_deferred_without_metadata():
    """Approved tool that re-raises without metadata — no metadata added to remaining."""
    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        # No metadata
        raise ApprovalRequired

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    # No metadata set (tool raised without metadata both times)
    assert 'call1' not in result.output.metadata


async def test_deferred_tool_handler_mixed_unresolved_and_re_deferred():
    """Handler resolves some, another call is re-deferred — both end up in remaining."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('re_raising_tool', {}, tool_call_id='re1'),
                ToolCallPart('unhandled_tool', {}, tool_call_id='un1'),
            ]
        )

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Only approve the re-raising one; leave unhandled_tool unresolved
        return DeferredToolResults(
            approvals={call.tool_call_id: True for call in requests.approvals if call.tool_name == 're_raising_tool'}
        )

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def re_raising_tool(ctx: RunContext) -> str:
        # Always raises — even after approval
        raise ApprovalRequired

    @agent.tool
    def unhandled_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    # Both calls in remaining: unhandled_tool (never resolved) + re_raising_tool (re-deferred after approval)
    approval_ids = {call.tool_call_id for call in result.output.approvals}
    assert 're1' in approval_ids
    assert 'un1' in approval_ids


async def test_deferred_tool_handler_re_deferred_as_call_deferred():
    """Approved tool that re-raises CallDeferred (not ApprovalRequired) stays in remaining.calls."""
    from pydantic_ai.exceptions import CallDeferred

    call_count = 0

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise CallDeferred (external execution needed)
        raise CallDeferred(metadata={'reason': 'external'})

    result = await agent.run('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    # Should be in calls (external), not approvals
    assert len(result.output.calls) == 1
    assert len(result.output.approvals) == 0
    assert result.output.metadata == {'call1': {'reason': 'external'}}


async def test_deferred_tool_handler_via_handle_call_preserves_tool_return():
    """handle_call(resolve_deferred=True) preserves `ToolReturn` wrapper (metadata, user content).

    The non-deferred `handle_call` path returns whatever the tool returned verbatim.
    The deferred path should do the same — critical for CodeMode-style callers that
    check `isinstance(result, ToolReturn)` to preserve metadata on nested return parts.
    """
    from pydantic_ai.messages import ToolReturn as _ToolReturn
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='actual result', metadata={'source': 'inner'}, content='user extra')

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        captured_result = result
        return 'done'

    await agent.run('go')
    # handle_call returned the ToolReturn wrapper verbatim, not the unwrapped content
    assert isinstance(captured_result, _ToolReturn)
    assert captured_result.return_value == 'actual result'
    assert captured_result.metadata == {'source': 'inner'}
    assert captured_result.content == 'user extra'


async def test_deferred_tool_handler_via_handle_call_denied_via_bool():
    """When a handler denies via `approvals[id] = False`, handle_call returns `ToolDenied()` with the default denial message."""
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: False for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured
        assert ctx.tool_manager is not None
        captured = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'caught' if isinstance(captured, ToolDenied) else 'no denial'

    await agent.run('go')
    assert isinstance(captured, ToolDenied)
    assert captured == ToolDenied()


async def test_deferred_tool_handler_via_handle_call_override_args():
    """When a handler approves with override_args, handle_call executes the tool with those args."""
    from pydantic_ai.tools import ToolApproved
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext, x: int) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return f'x={x}'

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={'x': 1}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    assert captured_result == 'x=42'


async def test_deferred_tool_handler_via_handle_call_external_plain_value():
    """When a handler supplies an external-call plain value, handle_call returns it verbatim."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: 'external value' for call in requests.calls})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured_result: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured_result
        assert ctx.tool_manager is not None
        captured_result = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'done'

    await agent.run('go')
    assert captured_result == 'external value'


async def test_deferred_tool_handler_via_handle_call_external_model_retry():
    """When a handler supplies a `ModelRetry` external-call result, handle_call raises `ToolRetryError`."""
    from pydantic_ai.exceptions import CallDeferred, ModelRetry, ToolRetryError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('retry please') for call in requests.calls})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught: ToolRetryError | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ToolRetryError as e:
            caught = e
            return 'caught'

    await agent.run('go')
    assert caught is not None
    assert caught.tool_retry.content == 'retry please'
    assert caught.tool_retry.tool_name == 'inner_tool'
    assert caught.tool_retry.tool_call_id == 'inner_1'


async def test_deferred_tool_handler_via_handle_call_external_retry_prompt_part():
    """When a handler supplies a `RetryPromptPart` external-call result, handle_call raises `ToolRetryError` with the part."""
    from pydantic_ai.exceptions import CallDeferred, ToolRetryError
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool_plain
    def inner_tool() -> str:
        raise CallDeferred

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught: ToolRetryError | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ToolRetryError as e:
            caught = e
            return 'caught'

    await agent.run('go')
    assert caught is not None
    assert caught.tool_retry.content == 'retry via part'
    # The helper stamps the real tool name / id onto the prompt part.
    assert caught.tool_retry.tool_name == 'inner_tool'
    assert caught.tool_retry.tool_call_id == 'inner_1'


async def test_deferred_tool_handler_via_handle_call_denied_returns_message():
    """When a handler denies a deferred call, handle_call returns the custom `ToolDenied` value verbatim."""
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied(message='not today') for call in requests.approvals}
        )

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    captured: Any = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal captured
        assert ctx.tool_manager is not None
        captured = await ctx.tool_manager.handle_call(
            ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
        )
        return 'caught' if isinstance(captured, ToolDenied) else 'no denial'

    await agent.run('go')
    assert isinstance(captured, ToolDenied)
    assert captured == ToolDenied(message='not today')


async def test_deferred_tool_handler_via_handle_call_re_raises_new_exception():
    """After approval, if tool re-raises CallDeferred (not ApprovalRequired), the new exception type is propagated."""
    from pydantic_ai.exceptions import CallDeferred
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()
    call_count = 0

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise a *different* deferral type with new metadata
        raise CallDeferred(metadata={'reason': 'external-after-approval'})

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    caught_exc_type: type | None = None
    caught_metadata: dict[str, Any] | None = None

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        nonlocal caught_exc_type, caught_metadata
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except (CallDeferred, ApprovalRequired) as e:
            caught_exc_type = type(e)
            caught_metadata = e.metadata
            return 'caught'

    result = await agent.run('go')
    assert result.output == 'final'
    # The new CallDeferred exception should surface, not the original ApprovalRequired
    assert caught_exc_type is CallDeferred
    assert caught_metadata == {'reason': 'external-after-approval'}


async def test_deferred_tool_handler_via_handle_call_handler_resolves_wrong_id():
    """handle_call path: handler returns results for wrong ID → remaining non-empty → raises original exc."""
    from pydantic_ai.toolsets import FunctionToolset

    inner_toolset = FunctionToolset()

    @inner_toolset.tool
    def inner_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    async def handle_deferred(ctx: RunContext, requests: DeferredToolRequests) -> DeferredToolResults:
        # Resolve a non-existent ID — our tool's ID stays in remaining
        return DeferredToolResults(approvals={'wrong_id': True})

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('caller_tool', {}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('final')])

    agent = Agent(
        FunctionModel(llm),
        toolsets=[inner_toolset],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    async def caller_tool(ctx: RunContext) -> str:
        assert ctx.tool_manager is not None
        try:
            await ctx.tool_manager.handle_call(
                ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner_1'),
            )
            return 'no raise'  # pragma: no cover
        except ApprovalRequired:
            return 'caught'

    result = await agent.run('go')
    assert result.output == 'final'


async def test_deferred_tool_handler_via_hooks_decorator():
    """`@hooks.on.deferred_tool_calls` resolves deferred calls inline."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {'x': 5}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Done!')])

    hooks = Hooks()

    @hooks.on.deferred_tool_calls
    async def handler(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def my_tool(ctx: RunContext, x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10

    result = await agent.run('Hello')
    assert result.output == 'Done!'


async def test_deferred_tool_handler_via_hooks_constructor_kwarg_and_accumulation():
    """`Hooks(deferred_tool_calls=...)` accumulates results across registered handlers."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {}, tool_call_id='a1'),
                    ToolCallPart('tool_b', {}, tool_call_id='b1'),
                    ToolCallPart('tool_c', {}, tool_call_id='c1'),
                ]
            )
        return ModelResponse(parts=[TextPart('All done.')])

    def handle_a(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    hooks = Hooks(deferred_tool_calls=handle_a)

    @hooks.on.deferred_tool_calls
    async def handle_rest(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # tool_a was already resolved by handle_a; this handler sees only tool_b and tool_c
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    @hooks.on.deferred_tool_calls
    async def never_called(  # pragma: no cover
        ctx: RunContext, *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        # All calls should already be resolved by the previous handler — this is the early-break path
        raise AssertionError('Should not be called: all requests already resolved')

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def tool_a(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a'

    @agent.tool
    def tool_b(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b'

    @agent.tool
    def tool_c(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'c'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_hooks_returns_none_when_unhandled():
    """`Hooks` returns None from the deferred-tool-calls hook when no registered handler resolves anything."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    hooks = Hooks()

    @hooks.on.deferred_tool_calls
    async def declines(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        return None

    @hooks.on.deferred_tool_calls
    async def empty(ctx: RunContext, *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # Empty results count as "didn't handle"
        return DeferredToolResults()

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[hooks],
    )

    @agent.tool
    def my_tool(ctx: RunContext) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    # Falls through to bubble-up since no handler resolved anything
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1


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
        return 'from func'  # pragma: no cover — the tool listing is what's asserted

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
        hidden_def = next(t for t in info.function_tools if t.name == 'hidden_tool')
        # `defer_loading=True` is what keeps the tool off the provider wire until loaded.
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
    assert seen_defer_flags == [True, False, False]


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
            [('load_capability', False), ('lookup_refund_policy', True), ('search_tools', False)],
            [('load_capability', False), ('lookup_refund_policy', False), ('search_tools', False)],
            [('load_capability', False), ('lookup_refund_policy', False), ('search_tools', False)],
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
