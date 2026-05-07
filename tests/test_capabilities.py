from __future__ import annotations

import asyncio
import contextvars
import threading
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import pytest
from opentelemetry.trace import NoOpTracer
from pydantic import BaseModel, ValidationError

from pydantic_ai._run_context import RunContext
from pydantic_ai._spec import CapabilitySpec, NamedSpec
from pydantic_ai.agent import Agent
from pydantic_ai.agent.spec import AgentSpec
from pydantic_ai.builtin_tools import (
    CodeExecutionTool,
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
    XSearchTool,
)
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    MCP,
    BuiltinTool,
    CapabilityOrdering,
    HandleDeferredToolCalls,
    ImageGeneration,
    IncludeToolReturnSchemas,
    PrefixTools,
    ProcessEventStream,
    ReinjectSystemPrompt,
    SetToolMetadata,
    Thinking,
    ThreadExecutor,
    Toolset,
    WebFetch,
    WebSearch,
    WrapperCapability,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.builtin_tool import BuiltinTool as BuiltinToolCap
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks, HookTimeoutError
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryImage,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import NativeOutput, OutputContext, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.toolsets._dynamic import ToolsetFunc
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInstance, IsStr, try_import

with try_import() as xai_imports:
    from pydantic_ai.models.xai import XSearch

pytestmark = [
    pytest.mark.anyio,
]


def test_capability_types() -> None:
    assert CAPABILITY_TYPES == snapshot(
        {
            'BuiltinTool': BuiltinTool,
            'ImageGeneration': ImageGeneration,
            'IncludeToolReturnSchemas': IncludeToolReturnSchemas,
            'MCP': MCP,
            'PrefixTools': PrefixTools,
            'ReinjectSystemPrompt': ReinjectSystemPrompt,
            'SetToolMetadata': SetToolMetadata,
            'Thinking': Thinking,
            'WebFetch': WebFetch,
            'WebSearch': WebSearch,
        }
    )


def test_agent_from_spec_basic():
    """Test Agent.from_spec with basic capabilities."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'instructions': 'You are a helpful agent.',
            'model_settings': {'max_tokens': 4096},
            'capabilities': [
                'WebSearch',
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
            'capabilities': [{'WebFetch': {'allowed_domains': ['example.com'], 'max_uses': 5}}],
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
            'capabilities': [{'MCP': {'url': 'https://mcp.example.com/sse', 'allowed_tools': ['search']}}],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, MCP))
    assert cap.url == 'https://mcp.example.com/sse'
    assert cap.allowed_tools == ['search']


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
class CustomCapability(AbstractCapability[None]):
    greeting: str = 'hello'


@dataclass
class CapabilityWithCallbackParam(AbstractCapability[None]):
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
            CapabilitySpec(name='WebSearch', arguments=None),
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
    with pytest.warns(DeprecationWarning, match=r'`retries` is deprecated'):
        agent = Agent.from_spec({'model': 'test', 'retries': 5})
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_retries_override():
    with pytest.warns(DeprecationWarning, match=r'`retries` is deprecated'):
        agent = Agent.from_spec({'model': 'test', 'retries': 5}, retries=2)
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 2  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_output_retries():
    with pytest.warns(DeprecationWarning, match=r'`retries` is deprecated'):
        agent = Agent.from_spec({'model': 'test', 'retries': 3, 'output_retries': 10})
    assert agent._max_tool_retries == 3  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 10  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_no_retries_does_not_warn():
    """`from_spec` without an explicit `retries` must not emit the deprecation warning.

    The default for `AgentSpec.retries` is `None` so it can be distinguished from a
    user-set value; only an explicit value is forwarded to the deprecated
    `Agent(retries=...)` kwarg.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        agent = Agent.from_spec({'model': 'test'})

    assert agent._max_tool_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 1  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_explicit_retries_warns():
    """An explicit `retries` in the spec still triggers the deprecation warning."""
    with pytest.warns(DeprecationWarning, match=r'`retries` is deprecated.*Use `tool_retries`'):
        agent = Agent.from_spec({'model': 'test', 'retries': 5})
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_tool_retries():
    """`tool_retries` on the spec sets the tool budget without cascading to output and without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        agent = Agent.from_spec({'model': 'test', 'tool_retries': 5})

    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_output_retries == 1  # pyright: ignore[reportPrivateUsage]


def test_agent_spec_retries_deprecated_getter():
    """Reading `AgentSpec.retries` warns; the value is still returned for backward compatibility."""
    spec = AgentSpec(model='test', retries=5)
    with pytest.warns(DeprecationWarning, match=r'`retries` is deprecated'):
        assert spec.retries == 5


def test_agent_spec_tool_retries_field():
    """`AgentSpec.tool_retries` is the canonical field and does not warn on access."""
    spec = AgentSpec(model='test', tool_retries=5)
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        assert spec.tool_retries == 5


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


def test_agent_from_spec_instrument():
    agent = Agent.from_spec({'model': 'test', 'instrument': True})
    assert agent.instrument is True


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
    class ExtraCap(AbstractCapability[None]):
        pass

    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': ['WebSearch'],
        },
        capabilities=[ExtraCap()],
    )
    # Should have both the WebSearch capability from spec and ExtraCap from arg
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    assert any(isinstance(c, WebSearch) for c in children)
    assert any(isinstance(c, ExtraCap) for c in children)


def test_model_json_schema_with_capabilities():
    pytest.importorskip('mcp', reason='schema varies without mcp package')
    schema = AgentSpec.model_json_schema_with_capabilities()
    assert schema == snapshot(
        {
            '$defs': {
                'CodeExecutionTool': {
                    'properties': {'kind': {'default': 'code_execution', 'title': 'Kind', 'type': 'string'}},
                    'title': 'CodeExecutionTool',
                    'type': 'object',
                },
                'FileSearchTool': {
                    'properties': {
                        'kind': {'default': 'file_search', 'title': 'Kind', 'type': 'string'},
                        'file_store_ids': {'items': {'type': 'string'}, 'title': 'File Store Ids', 'type': 'array'},
                    },
                    'required': ['file_store_ids'],
                    'title': 'FileSearchTool',
                    'type': 'object',
                },
                'ImageGenerationTool': {
                    'properties': {
                        'kind': {'default': 'image_generation', 'title': 'Kind', 'type': 'string'},
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
                        'anthropic:claude-3-haiku-20240307',
                        'anthropic:claude-haiku-4-5-20251001',
                        'anthropic:claude-mythos-preview',
                        'anthropic:claude-haiku-4-5',
                        'anthropic:claude-opus-4-0',
                        'anthropic:claude-opus-4-1',
                        'anthropic:claude-opus-4-1-20250805',
                        'anthropic:claude-opus-4-20250514',
                        'anthropic:claude-opus-4-5-20251101',
                        'anthropic:claude-opus-4-5',
                        'anthropic:claude-opus-4-6',
                        'anthropic:claude-opus-4-7',
                        'anthropic:claude-sonnet-4-0',
                        'anthropic:claude-sonnet-4-20250514',
                        'anthropic:claude-sonnet-4-5-20250929',
                        'anthropic:claude-sonnet-4-5',
                        'anthropic:claude-sonnet-4-6',
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
                        'bedrock:anthropic.claude-v2:1',
                        'bedrock:anthropic.claude-v2',
                        'bedrock:cohere.command-light-text-v14',
                        'bedrock:cohere.command-r-plus-v1:0',
                        'bedrock:cohere.command-r-v1:0',
                        'bedrock:cohere.command-text-v14',
                        'bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0',
                        'bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0',
                        'bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'bedrock:eu.anthropic.claude-sonnet-4-6',
                        'bedrock:global.anthropic.claude-opus-4-5-20251101-v1:0',
                        'bedrock:meta.llama3-1-405b-instruct-v1:0',
                        'bedrock:meta.llama3-1-70b-instruct-v1:0',
                        'bedrock:meta.llama3-1-8b-instruct-v1:0',
                        'bedrock:meta.llama3-70b-instruct-v1:0',
                        'bedrock:meta.llama3-8b-instruct-v1:0',
                        'bedrock:mistral.mistral-7b-instruct-v0:2',
                        'bedrock:mistral.mistral-large-2402-v1:0',
                        'bedrock:mistral.mistral-large-2407-v1:0',
                        'bedrock:mistral.mixtral-8x7b-instruct-v0:1',
                        'bedrock:us.amazon.nova-2-lite-v1:0',
                        'bedrock:us.amazon.nova-lite-v1:0',
                        'bedrock:us.amazon.nova-micro-v1:0',
                        'bedrock:us.amazon.nova-pro-v1:0',
                        'bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0',
                        'bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0',
                        'bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                        'bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0',
                        'bedrock:us.anthropic.claude-3-haiku-20240307-v1:0',
                        'bedrock:us.anthropic.claude-3-opus-20240229-v1:0',
                        'bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0',
                        'bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0',
                        'bedrock:us.anthropic.claude-opus-4-20250514-v1:0',
                        'bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0',
                        'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'bedrock:us.anthropic.claude-sonnet-4-6',
                        'bedrock:us.meta.llama3-1-70b-instruct-v1:0',
                        'bedrock:us.meta.llama3-1-8b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-11b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-1b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-3b-instruct-v1:0',
                        'bedrock:us.meta.llama3-2-90b-instruct-v1:0',
                        'bedrock:us.meta.llama3-3-70b-instruct-v1:0',
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
                        'gateway/anthropic:claude-3-haiku-20240307',
                        'gateway/anthropic:claude-haiku-4-5-20251001',
                        'gateway/anthropic:claude-mythos-preview',
                        'gateway/anthropic:claude-haiku-4-5',
                        'gateway/anthropic:claude-opus-4-0',
                        'gateway/anthropic:claude-opus-4-1',
                        'gateway/anthropic:claude-opus-4-1-20250805',
                        'gateway/anthropic:claude-opus-4-20250514',
                        'gateway/anthropic:claude-opus-4-5-20251101',
                        'gateway/anthropic:claude-opus-4-5',
                        'gateway/anthropic:claude-opus-4-6',
                        'gateway/anthropic:claude-opus-4-7',
                        'gateway/anthropic:claude-sonnet-4-0',
                        'gateway/anthropic:claude-sonnet-4-20250514',
                        'gateway/anthropic:claude-sonnet-4-5-20250929',
                        'gateway/anthropic:claude-sonnet-4-5',
                        'gateway/anthropic:claude-sonnet-4-6',
                        'gateway/bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0',
                        'gateway/bedrock:anthropic.claude-3-haiku-20240307-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0',
                        'gateway/bedrock:eu.anthropic.claude-sonnet-4-6',
                        'gateway/bedrock:global.anthropic.claude-opus-4-5-20251101-v1:0',
                        'gateway/google-vertex:gemini-2.5-flash-image',
                        'gateway/google-vertex:gemini-2.5-flash-lite-preview-09-2025',
                        'gateway/google-vertex:gemini-2.5-flash-lite',
                        'gateway/google-vertex:gemini-2.5-flash',
                        'gateway/google-vertex:gemini-2.5-pro',
                        'gateway/google-vertex:gemini-3-flash-preview',
                        'gateway/google-vertex:gemini-3-pro-image-preview',
                        'gateway/google-vertex:gemini-3.1-flash-image-preview',
                        'gateway/google-vertex:gemini-3.1-flash-lite-preview',
                        'gateway/google-vertex:gemini-3.1-pro-preview',
                        'gateway/groq:llama-3.1-8b-instant',
                        'gateway/groq:llama-3.3-70b-versatile',
                        'gateway/groq:meta-llama/llama-4-scout-17b-16e-instruct',
                        'gateway/groq:moonshotai/kimi-k2-instruct-0905',
                        'gateway/groq:openai/gpt-oss-120b',
                        'gateway/groq:openai/gpt-oss-20b',
                        'gateway/groq:openai/gpt-oss-safeguard-20b',
                        'gateway/openai:gpt-3.5-turbo-0125',
                        'gateway/openai:gpt-3.5-turbo-1106',
                        'gateway/openai:gpt-3.5-turbo-16k',
                        'gateway/openai:gpt-3.5-turbo',
                        'gateway/openai:gpt-4-0613',
                        'gateway/openai:gpt-4-turbo-2024-04-09',
                        'gateway/openai:gpt-4-turbo',
                        'gateway/openai:gpt-4.1-2025-04-14',
                        'gateway/openai:gpt-4.1-mini-2025-04-14',
                        'gateway/openai:gpt-4.1-mini',
                        'gateway/openai:gpt-4.1-nano-2025-04-14',
                        'gateway/openai:gpt-4.1-nano',
                        'gateway/openai:gpt-4.1',
                        'gateway/openai:gpt-4',
                        'gateway/openai:gpt-4o-2024-05-13',
                        'gateway/openai:gpt-4o-2024-08-06',
                        'gateway/openai:gpt-4o-2024-11-20',
                        'gateway/openai:gpt-4o-mini-2024-07-18',
                        'gateway/openai:gpt-4o-mini-search-preview-2025-03-11',
                        'gateway/openai:gpt-4o-mini-search-preview',
                        'gateway/openai:gpt-4o-mini',
                        'gateway/openai:gpt-4o-search-preview-2025-03-11',
                        'gateway/openai:gpt-4o-search-preview',
                        'gateway/openai:gpt-4o',
                        'gateway/openai:gpt-5-2025-08-07',
                        'gateway/openai:gpt-5-chat-latest',
                        'gateway/openai:gpt-5-mini-2025-08-07',
                        'gateway/openai:gpt-5-mini',
                        'gateway/openai:gpt-5-nano-2025-08-07',
                        'gateway/openai:gpt-5-nano',
                        'gateway/openai:gpt-5.1-2025-11-13',
                        'gateway/openai:gpt-5.1-chat-latest',
                        'gateway/openai:gpt-5.1',
                        'gateway/openai:gpt-5.2-2025-12-11',
                        'gateway/openai:gpt-5.2-chat-latest',
                        'gateway/openai:gpt-5.2',
                        'gateway/openai:gpt-5.4-mini-2026-03-17',
                        'gateway/openai:gpt-5.4-mini',
                        'gateway/openai:gpt-5.4-nano-2026-03-17',
                        'gateway/openai:gpt-5.4-nano',
                        'gateway/openai:gpt-5.4',
                        'gateway/openai:gpt-5',
                        'gateway/openai:o1-2024-12-17',
                        'gateway/openai:o1',
                        'gateway/openai:o3-2025-04-16',
                        'gateway/openai:o3-mini-2025-01-31',
                        'gateway/openai:o3-mini',
                        'gateway/openai:o3',
                        'gateway/openai:o4-mini-2025-04-16',
                        'gateway/openai:o4-mini',
                        'google-gla:gemini-2.0-flash-lite',
                        'google-gla:gemini-2.0-flash',
                        'google-gla:gemini-2.5-flash-image',
                        'google-gla:gemini-2.5-flash-lite-preview-09-2025',
                        'google-gla:gemini-2.5-flash-lite',
                        'google-gla:gemini-2.5-flash-preview-09-2025',
                        'google-gla:gemini-2.5-flash',
                        'google-gla:gemini-2.5-pro',
                        'google-gla:gemini-3-flash-preview',
                        'google-gla:gemini-3-pro-image-preview',
                        'google-gla:gemini-3-pro-preview',
                        'google-gla:gemini-3.1-flash-image-preview',
                        'google-gla:gemini-3.1-flash-lite-preview',
                        'google-gla:gemini-3.1-pro-preview',
                        'google-gla:gemini-flash-latest',
                        'google-gla:gemini-flash-lite-latest',
                        'google-vertex:gemini-2.0-flash-lite',
                        'google-vertex:gemini-2.0-flash',
                        'google-vertex:gemini-2.5-flash-image',
                        'google-vertex:gemini-2.5-flash-lite-preview-09-2025',
                        'google-vertex:gemini-2.5-flash-lite',
                        'google-vertex:gemini-2.5-flash-preview-09-2025',
                        'google-vertex:gemini-2.5-flash',
                        'google-vertex:gemini-2.5-pro',
                        'google-vertex:gemini-3-flash-preview',
                        'google-vertex:gemini-3-pro-image-preview',
                        'google-vertex:gemini-3-pro-preview',
                        'google-vertex:gemini-3.1-flash-image-preview',
                        'google-vertex:gemini-3.1-flash-lite-preview',
                        'google-vertex:gemini-3.1-pro-preview',
                        'google-vertex:gemini-flash-latest',
                        'google-vertex:gemini-flash-lite-latest',
                        'grok:grok-2-image-1212',
                        'grok:grok-2-vision-1212',
                        'grok:grok-3-fast',
                        'grok:grok-3-mini-fast',
                        'grok:grok-3-mini',
                        'grok:grok-3',
                        'grok:grok-4-0709',
                        'grok:grok-4-latest',
                        'grok:grok-4-1-fast-non-reasoning',
                        'grok:grok-4-1-fast-reasoning',
                        'grok:grok-4-1-fast',
                        'grok:grok-4-fast-non-reasoning',
                        'grok:grok-4-fast-reasoning',
                        'grok:grok-4-fast',
                        'grok:grok-4',
                        'grok:grok-code-fast-1',
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
                        'xai:grok-code-fast-1',
                        'groq:llama-3.1-8b-instant',
                        'groq:llama-3.3-70b-versatile',
                        'groq:meta-llama/llama-guard-4-12b',
                        'groq:openai/gpt-oss-120b',
                        'groq:openai/gpt-oss-20b',
                        'groq:whisper-large-v3',
                        'groq:whisper-large-v3-turbo',
                        'groq:meta-llama/llama-4-maverick-17b-128e-instruct',
                        'groq:meta-llama/llama-4-scout-17b-16e-instruct',
                        'groq:meta-llama/llama-prompt-guard-2-22m',
                        'groq:meta-llama/llama-prompt-guard-2-86m',
                        'groq:moonshotai/kimi-k2-instruct-0905',
                        'groq:openai/gpt-oss-safeguard-20b',
                        'groq:playai-tts',
                        'groq:playai-tts-arabic',
                        'groq:qwen/qwen-3-32b',
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
                        'heroku:qwen3-235b',
                        'heroku:qwen3-coder-480b',
                        'heroku:nova-2-lite',
                        'heroku:nova-lite',
                        'heroku:nova-pro',
                        'huggingface:deepseek-ai/DeepSeek-R1',
                        'huggingface:meta-llama/Llama-3.3-70B-Instruct',
                        'huggingface:meta-llama/Llama-4-Maverick-17B-128E-Instruct',
                        'huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct',
                        'huggingface:Qwen/Qwen2.5-72B-Instruct',
                        'huggingface:Qwen/Qwen3-235B-A22B',
                        'huggingface:Qwen/Qwen3-32B',
                        'huggingface:Qwen/QwQ-32B',
                        'mistral:codestral-latest',
                        'mistral:mistral-large-latest',
                        'mistral:mistral-moderation-latest',
                        'mistral:mistral-small-latest',
                        'moonshotai:kimi-k2-0711-preview',
                        'moonshotai:kimi-latest',
                        'moonshotai:kimi-thinking-preview',
                        'moonshotai:moonshot-v1-128k-vision-preview',
                        'moonshotai:moonshot-v1-128k',
                        'moonshotai:moonshot-v1-32k-vision-preview',
                        'moonshotai:moonshot-v1-32k',
                        'moonshotai:moonshot-v1-8k-vision-preview',
                        'moonshotai:moonshot-v1-8k',
                        'openai:computer-use-preview-2025-03-11',
                        'openai:computer-use-preview',
                        'openai:gpt-3.5-turbo-0125',
                        'openai:gpt-3.5-turbo-0301',
                        'openai:gpt-3.5-turbo-0613',
                        'openai:gpt-3.5-turbo-1106',
                        'openai:gpt-3.5-turbo-16k-0613',
                        'openai:gpt-3.5-turbo-16k',
                        'openai:gpt-3.5-turbo',
                        'openai:gpt-4-0314',
                        'openai:gpt-4-0613',
                        'openai:gpt-4-turbo-2024-04-09',
                        'openai:gpt-4-turbo',
                        'openai:gpt-4.1-2025-04-14',
                        'openai:gpt-4.1-mini-2025-04-14',
                        'openai:gpt-4.1-mini',
                        'openai:gpt-4.1-nano-2025-04-14',
                        'openai:gpt-4.1-nano',
                        'openai:gpt-4.1',
                        'openai:gpt-4',
                        'openai:gpt-4o-2024-05-13',
                        'openai:gpt-4o-2024-08-06',
                        'openai:gpt-4o-2024-11-20',
                        'openai:gpt-4o-audio-preview-2024-12-17',
                        'openai:gpt-4o-audio-preview-2025-06-03',
                        'openai:gpt-4o-audio-preview',
                        'openai:gpt-4o-mini-2024-07-18',
                        'openai:gpt-4o-mini-audio-preview-2024-12-17',
                        'openai:gpt-4o-mini-audio-preview',
                        'openai:gpt-4o-mini-search-preview-2025-03-11',
                        'openai:gpt-4o-mini-search-preview',
                        'openai:gpt-4o-mini',
                        'openai:gpt-4o-search-preview-2025-03-11',
                        'openai:gpt-4o-search-preview',
                        'openai:gpt-4o',
                        'openai:gpt-5-2025-08-07',
                        'openai:gpt-5-chat-latest',
                        'openai:gpt-5-codex',
                        'openai:gpt-5-mini-2025-08-07',
                        'openai:gpt-5-mini',
                        'openai:gpt-5-nano-2025-08-07',
                        'openai:gpt-5-nano',
                        'openai:gpt-5-pro-2025-10-06',
                        'openai:gpt-5-pro',
                        'openai:gpt-5.1-2025-11-13',
                        'openai:gpt-5.1-chat-latest',
                        'openai:gpt-5.1-codex-max',
                        'openai:gpt-5.1-codex',
                        'openai:gpt-5.1',
                        'openai:gpt-5.2-2025-12-11',
                        'openai:gpt-5.2-chat-latest',
                        'openai:gpt-5.2-pro-2025-12-11',
                        'openai:gpt-5.2-pro',
                        'openai:gpt-5.2',
                        'openai:gpt-5.3-chat-latest',
                        'openai:gpt-5.4-mini-2026-03-17',
                        'openai:gpt-5.4-mini',
                        'openai:gpt-5.4-nano-2026-03-17',
                        'openai:gpt-5.4-nano',
                        'openai:gpt-5.4',
                        'openai:gpt-5',
                        'openai:o1-2024-12-17',
                        'openai:o1-pro-2025-03-19',
                        'openai:o1-pro',
                        'openai:o1',
                        'openai:o3-2025-04-16',
                        'openai:o3-deep-research-2025-06-26',
                        'openai:o3-deep-research',
                        'openai:o3-mini-2025-01-31',
                        'openai:o3-mini',
                        'openai:o3-pro-2025-06-10',
                        'openai:o3-pro',
                        'openai:o3',
                        'openai:o4-mini-2025-04-16',
                        'openai:o4-mini-deep-research-2025-06-26',
                        'openai:o4-mini-deep-research',
                        'openai:o4-mini',
                        'test',
                    ],
                    'type': 'string',
                },
                'MCPServerTool': {
                    'properties': {
                        'kind': {'default': 'mcp_server', 'title': 'Kind', 'type': 'string'},
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
                    'properties': {'kind': {'default': 'memory', 'title': 'Kind', 'type': 'string'}},
                    'title': 'MemoryTool',
                    'type': 'object',
                },
                'ModelSettings': {
                    'description': """\
Settings to configure an LLM.

Here we include only settings which apply to multiple models / model providers,
though not all of these settings are supported by all models.\
""",
                    'properties': {
                        'max_tokens': {'title': 'Max Tokens', 'type': 'integer'},
                        'temperature': {'title': 'Temperature', 'type': 'number'},
                        'top_p': {'title': 'Top P', 'type': 'number'},
                        'timeout': {'title': 'Timeout', 'type': 'number'},
                        'parallel_tool_calls': {'title': 'Parallel Tool Calls', 'type': 'boolean'},
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
                'UrlContextTool': {
                    'deprecated': True,
                    'properties': {
                        'kind': {'default': 'url_context', 'title': 'Kind', 'type': 'string'},
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
                    'title': 'UrlContextTool',
                    'type': 'object',
                },
                'WebFetchTool': {
                    'properties': {
                        'kind': {'default': 'web_fetch', 'title': 'Kind', 'type': 'string'},
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
                    'description': """\
Allows you to localize search results based on a user's location.

Supported by:

* Anthropic
* OpenAI Responses\
""",
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
                'short_spec_BuiltinTool': {
                    'additionalProperties': False,
                    'properties': {
                        'BuiltinTool': {
                            'anyOf': [
                                {
                                    'oneOf': [
                                        {'$ref': '#/$defs/WebSearchTool'},
                                        {'$ref': '#/$defs/XSearchTool'},
                                        {'$ref': '#/$defs/CodeExecutionTool'},
                                        {'$ref': '#/$defs/WebFetchTool'},
                                        {'$ref': '#/$defs/UrlContextTool'},
                                        {'$ref': '#/$defs/ImageGenerationTool'},
                                        {'$ref': '#/$defs/MemoryTool'},
                                        {'$ref': '#/$defs/MCPServerTool'},
                                        {'$ref': '#/$defs/FileSearchTool'},
                                    ]
                                },
                                {'type': 'null'},
                            ],
                            'title': 'Builtintool',
                        }
                    },
                    'title': 'short_spec_BuiltinTool',
                    'type': 'object',
                },
                'short_spec_IncludeToolReturnSchemas': {
                    'additionalProperties': False,
                    'properties': {
                        'IncludeToolReturnSchemas': {
                            'anyOf': [
                                {'const': 'all', 'type': 'string'},
                                {'items': {'type': 'string'}, 'type': 'array'},
                                {'additionalProperties': True, 'type': 'object'},
                            ],
                            'title': 'Includetoolreturnschemas',
                        }
                    },
                    'title': 'short_spec_IncludeToolReturnSchemas',
                    'type': 'object',
                },
                'short_spec_MCP': {
                    'additionalProperties': False,
                    'properties': {'MCP': {'title': 'Mcp', 'type': 'string'}},
                    'required': ['MCP'],
                    'title': 'short_spec_MCP',
                    'type': 'object',
                },
                'short_spec_ReinjectSystemPrompt': {
                    'additionalProperties': False,
                    'properties': {'ReinjectSystemPrompt': {'title': 'Reinjectsystemprompt', 'type': 'boolean'}},
                    'title': 'short_spec_ReinjectSystemPrompt',
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
                'short_spec_Thinking': {
                    'additionalProperties': False,
                    'properties': {
                        'Thinking': {
                            'anyOf': [
                                {'type': 'boolean'},
                                {'enum': ['minimal', 'low', 'medium', 'high', 'xhigh'], 'type': 'string'},
                            ],
                            'title': 'Thinking',
                        }
                    },
                    'title': 'short_spec_Thinking',
                    'type': 'object',
                },
                'spec_ImageGeneration': {
                    'additionalProperties': False,
                    'properties': {'ImageGeneration': {'$ref': '#/$defs/spec_params_ImageGeneration'}},
                    'required': ['ImageGeneration'],
                    'title': 'spec_ImageGeneration',
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
                'spec_params_ImageGeneration': {
                    'additionalProperties': False,
                    'properties': {
                        'builtin': {
                            'anyOf': [
                                {'$ref': '#/$defs/ImageGenerationTool'},
                                {'type': 'boolean'},
                            ],
                            'title': 'Builtin',
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
                    },
                    'title': 'spec_params_ImageGeneration',
                    'type': 'object',
                },
                'spec_params_MCP': {
                    'additionalProperties': False,
                    'properties': {
                        'url': {'title': 'Url', 'type': 'string'},
                        'builtin': {
                            'anyOf': [{'$ref': '#/$defs/MCPServerTool'}, {'type': 'boolean'}],
                            'title': 'Builtin',
                        },
                        'local': {'anyOf': [{'const': False, 'type': 'boolean'}, {'type': 'null'}], 'title': 'Local'},
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
                                {'const': 'BuiltinTool', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_BuiltinTool'},
                                {'const': 'ImageGeneration', 'type': 'string'},
                                {'$ref': '#/$defs/spec_ImageGeneration'},
                                {'const': 'IncludeToolReturnSchemas', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_IncludeToolReturnSchemas'},
                                {'$ref': '#/$defs/short_spec_MCP'},
                                {'$ref': '#/$defs/spec_MCP'},
                                {'$ref': '#/$defs/spec_PrefixTools'},
                                {'const': 'ReinjectSystemPrompt', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_ReinjectSystemPrompt'},
                                {'const': 'SetToolMetadata', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_SetToolMetadata'},
                                {'const': 'Thinking', 'type': 'string'},
                                {'$ref': '#/$defs/short_spec_Thinking'},
                                {'const': 'WebFetch', 'type': 'string'},
                                {'$ref': '#/$defs/spec_WebFetch'},
                                {'const': 'WebSearch', 'type': 'string'},
                                {'$ref': '#/$defs/spec_WebSearch'},
                            ]
                        },
                    },
                    'required': ['prefix', 'capability'],
                    'title': 'spec_params_PrefixTools',
                    'type': 'object',
                },
                'spec_params_WebFetch': {
                    'additionalProperties': False,
                    'properties': {
                        'builtin': {
                            'anyOf': [
                                {'$ref': '#/$defs/WebFetchTool'},
                                {'type': 'boolean'},
                            ],
                            'title': 'Builtin',
                        },
                        'local': {'anyOf': [{'const': False, 'type': 'boolean'}, {'type': 'null'}], 'title': 'Local'},
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
                    },
                    'title': 'spec_params_WebFetch',
                    'type': 'object',
                },
                'spec_params_WebSearch': {
                    'additionalProperties': False,
                    'properties': {
                        'builtin': {
                            'anyOf': [
                                {'$ref': '#/$defs/WebSearchTool'},
                                {'type': 'boolean'},
                            ],
                            'title': 'Builtin',
                        },
                        'local': {'anyOf': [{'const': False, 'type': 'boolean'}, {'type': 'null'}], 'title': 'Local'},
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
                    },
                    'title': 'spec_params_WebSearch',
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
                'tool_retries': {
                    'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Tool Retries',
                },
                'retries': {
                    'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                    'default': None,
                    'deprecated': True,
                    'title': 'Retries',
                },
                'output_retries': {
                    'anyOf': [{'type': 'integer'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Output Retries',
                },
                'end_strategy': {
                    'default': 'early',
                    'enum': ['early', 'graceful', 'exhaustive'],
                    'title': 'End Strategy',
                    'type': 'string',
                },
                'tool_timeout': {
                    'anyOf': [{'type': 'number'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Tool Timeout',
                },
                'instrument': {
                    'anyOf': [{'type': 'boolean'}, {'type': 'null'}],
                    'default': None,
                    'title': 'Instrument',
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
                            {'const': 'BuiltinTool', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_BuiltinTool'},
                            {'const': 'ImageGeneration', 'type': 'string'},
                            {'$ref': '#/$defs/spec_ImageGeneration'},
                            {'const': 'IncludeToolReturnSchemas', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_IncludeToolReturnSchemas'},
                            {'$ref': '#/$defs/short_spec_MCP'},
                            {'$ref': '#/$defs/spec_MCP'},
                            {'$ref': '#/$defs/spec_PrefixTools'},
                            {'const': 'ReinjectSystemPrompt', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_ReinjectSystemPrompt'},
                            {'const': 'SetToolMetadata', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_SetToolMetadata'},
                            {'const': 'Thinking', 'type': 'string'},
                            {'$ref': '#/$defs/short_spec_Thinking'},
                            {'const': 'WebFetch', 'type': 'string'},
                            {'$ref': '#/$defs/spec_WebFetch'},
                            {'const': 'WebSearch', 'type': 'string'},
                            {'$ref': '#/$defs/spec_WebSearch'},
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


def test_builtin_tools_param_wrapped_as_capabilities():
    """The builtin_tools parameter items are wrapped in BuiltinTool capabilities."""
    agent = Agent('test', builtin_tools=[WebSearchTool(), CodeExecutionTool()])
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, BuiltinToolCap)]
    assert len(builtin_caps) == 2
    assert isinstance(builtin_caps[0].tool, WebSearchTool)
    assert isinstance(builtin_caps[1].tool, CodeExecutionTool)
    # Also available via _cap_builtin_tools
    assert len(agent._cap_builtin_tools) == 2  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_builtin_tool():
    """BuiltinTool capability can be constructed from spec."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'BuiltinTool': {'kind': 'web_search'}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, BuiltinToolCap)]
    assert len(builtin_caps) == 1
    assert isinstance(builtin_caps[0].tool, WebSearchTool)


def test_agent_from_spec_builtin_tool_with_options():
    """BuiltinTool spec supports builtin tool configuration options."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'BuiltinTool': {'kind': 'web_search', 'search_context_size': 'high'}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, BuiltinToolCap)]
    assert len(builtin_caps) == 1
    tool = builtin_caps[0].tool
    assert isinstance(tool, WebSearchTool)
    assert tool.search_context_size == 'high'


def test_agent_from_spec_builtin_tool_explicit_form():
    """BuiltinTool spec supports the explicit {tool: ...} form."""
    agent = Agent.from_spec(
        {
            'model': 'test',
            'capabilities': [
                {'BuiltinTool': {'tool': {'kind': 'code_execution'}}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, BuiltinToolCap)]
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
    with pytest.warns(DeprecationWarning, match=r'`retries` is deprecated'):
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
    spec = AgentSpec(model='test', name='roundtrip', tool_retries=3)
    spec_path = Path(tmp_path) / 'agent.json'
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == 'test'
    assert loaded.name == 'roundtrip'
    assert loaded.tool_retries == 3


@dataclass
class ToolsetFuncCapability(AbstractCapability[None]):
    """A capability that returns a ToolsetFunc instead of an AbstractToolset."""

    def get_toolset(self) -> ToolsetFunc[None]:
        def make_toolset(ctx: RunContext[None]) -> AbstractToolset[None]:
            toolset = FunctionToolset[None]()

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

    tool_calls = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == 'greet'

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
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

    tool_calls = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
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

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith('Hello, ')


def test_abstract_capability_get_model_settings_default():
    """AbstractCapability.get_model_settings() returns None by default."""

    @dataclass
    class PlainCap(AbstractCapability[None]):
        pass

    cap = PlainCap()
    assert cap.get_model_settings() is None


def test_combined_capability_get_model_settings_merge():
    """CombinedCapability.get_model_settings() merges settings from all sub-capabilities."""

    @dataclass
    class MaxTokensCap(AbstractCapability[None]):
        def get_model_settings(self) -> _ModelSettings | None:
            return _ModelSettings(max_tokens=100)

    @dataclass
    class TemperatureCap(AbstractCapability[None]):
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
    class PlainCap(AbstractCapability[None]):
        pass

    caps = CombinedCapability(capabilities=[PlainCap()])
    assert caps.get_model_settings() is None


def test_toolset_capability_get_toolset():
    """Toolset capability returns its toolset."""
    ts = FunctionToolset[None]()
    cap = Toolset(toolset=ts)
    assert cap.get_toolset() is ts


async def test_toolset_capability_in_agent():
    """A Toolset capability's tools are available to the agent."""
    ts = FunctionToolset[None]()

    @ts.tool_plain
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f'Hello, {name}!'

    agent = Agent(TestModel(), capabilities=[Toolset(toolset=ts)])
    result = await agent.run('Greet Alice')

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
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

    with pytest.raises(ValueError, match="Could not infer format for filename 'agent.txt'"):
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


async def test_capability_for_run_default_returns_self():
    """Default for_run returns self."""

    @dataclass
    class SimpleCap(AbstractCapability[None]):
        pass

    cap = SimpleCap()
    ctx = _build_run_context()
    assert await cap.for_run(ctx) is cap


async def test_combined_capability_for_run_propagates():
    """CombinedCapability propagates for_run to children."""

    @dataclass
    class SimpleCap(AbstractCapability[None]):
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

    class PerRunCap(AbstractCapability[None]):
        def __init__(self, run_id: int = 0):
            self.run_id = run_id

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return PerRunCap(run_id=self.run_id + 1)

    @dataclass
    class StaticCap(AbstractCapability[None]):
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
    class FailingCap(AbstractCapability[None]):
        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            raise RuntimeError('boom')

    @dataclass
    class SlowCap(AbstractCapability[None]):
        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
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
    class MyCap(AbstractCapability[None]):
        pass

    cap = MyCap()
    visited: list[AbstractCapability[None]] = []
    cap.apply(visited.append)
    assert visited == [cap]


def test_apply_combined_capability():
    """CombinedCapability.apply() recursively visits all leaf capabilities."""

    @dataclass
    class CapA(AbstractCapability[None]):
        pass

    @dataclass
    class CapB(AbstractCapability[None]):
        pass

    cap_a = CapA()
    cap_b = CapB()
    combined = CombinedCapability([cap_a, cap_b])

    visited: list[AbstractCapability[None]] = []
    combined.apply(visited.append)
    assert visited == [cap_a, cap_b]


def test_apply_nested_combined_capability():
    """CombinedCapability.apply() flattens nested CombinedCapabilities."""

    @dataclass
    class CapA(AbstractCapability[None]):
        pass

    @dataclass
    class CapB(AbstractCapability[None]):
        pass

    @dataclass
    class CapC(AbstractCapability[None]):
        pass

    cap_a = CapA()
    cap_b = CapB()
    cap_c = CapC()
    inner = CombinedCapability([cap_a, cap_b])
    outer = CombinedCapability([inner, cap_c])

    visited: list[AbstractCapability[None]] = []
    outer.apply(visited.append)
    assert visited == [cap_a, cap_b, cap_c]


def test_apply_wrapper_capability():
    """WrapperCapability.apply() delegates to the wrapped capability."""
    inner = Thinking()
    wrapper = WrapperCapability(wrapped=inner)

    visited: list[AbstractCapability[None]] = []
    wrapper.apply(visited.append)
    assert visited == [inner]


def test_apply_prefix_tools():
    """PrefixTools (a WrapperCapability) delegates apply() to the wrapped capability."""
    thinking = Thinking()
    prefixed = PrefixTools(wrapped=thinking, prefix='ns')

    visited: list[AbstractCapability[None]] = []
    prefixed.apply(visited.append)
    assert visited == [thinking]


def test_apply_finds_capability_by_type():
    """Realistic usage: use apply() to check if a specific capability type is present."""
    thinking = Thinking()
    web_search = WebSearch()
    combined = CombinedCapability([thinking, web_search])

    visited: list[AbstractCapability[None]] = []
    combined.apply(visited.append)

    assert any(isinstance(c, Thinking) for c in visited)
    assert any(isinstance(c, WebSearch) for c in visited)
    assert not any(isinstance(c, WebFetch) for c in visited)


def test_apply_finds_wrapped_capability_by_type():
    """apply() traverses through wrappers, so wrapped capabilities are discoverable by type."""
    thinking = Thinking()
    prefixed = PrefixTools(wrapped=thinking, prefix='ns')
    combined = CombinedCapability([prefixed, WebSearch()])

    visited: list[AbstractCapability[None]] = []
    combined.apply(visited.append)

    assert any(isinstance(c, Thinking) for c in visited)
    assert any(isinstance(c, WebSearch) for c in visited)
    assert not any(isinstance(c, PrefixTools) for c in visited)


def test_apply_empty_combined():
    """CombinedCapability with no children visits nothing."""
    combined = CombinedCapability[None]([])
    visited: list[AbstractCapability[None]] = []
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

    class SwitchingCap(AbstractCapability[None]):
        def __init__(self, use_b: bool = False):
            self.use_b = use_b

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return SwitchingCap(use_b=True)

        def get_toolset(self) -> AbstractToolset[None]:
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

    class DynamicInstructionsCap(AbstractCapability[None]):
        def __init__(self, run_instructions: str = 'init-time'):
            self._run_instructions = run_instructions

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return DynamicInstructionsCap(run_instructions='per-run')

        def get_instructions(self) -> str:
            return self._run_instructions

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


async def test_concurrent_runs_capability_isolation():
    """Multiple concurrent runs don't share state on stateful capabilities."""

    class CountingCap(AbstractCapability[None]):
        def __init__(self) -> None:
            self.request_count = 0

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return CountingCap()

        async def before_model_request(
            self,
            ctx: RunContext[None],
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
            tool_retries=4,
            output_retries=4,
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


# --- BuiltinOrLocalTool tests ---


class TestWebSearchCapability:
    def test_websearch_default_with_supporting_model(self):
        """WebSearch() with a model that supports builtin web search → builtin used, local removed."""
        cap = WebSearch()
        builtins = cap.get_builtin_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], WebSearchTool)

        toolset = cap.get_toolset()
        # Should have a toolset (for the DuckDuckGo fallback wrapped with PreparedToolset)
        assert toolset is not None

    def test_websearch_default_with_nonsupporting_model(self, allow_model_requests: None):
        """WebSearch() with a model that doesn't support builtin → DuckDuckGo fallback used."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # When called with tools, call the first one
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

        model = FunctionModel(model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
        agent = Agent(model, capabilities=[WebSearch()])
        result = agent.run_sync('search for something')
        # Should have used the DuckDuckGo fallback tool
        assert 'Tool result' in result.output

    def test_websearch_local_false_with_nonsupporting_model(self, allow_model_requests: None):
        """WebSearch(local=False) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset()))  # type: ignore
        agent = Agent(model, capabilities=[WebSearch(local=False)])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('search')

    def test_websearch_builtin_false(self):
        """WebSearch(builtin=False) → only local, no builtin registered."""
        cap = WebSearch(builtin=False)
        assert cap.get_builtin_tools() == []
        toolset = cap.get_toolset()
        # Should have a plain toolset (no PreparedToolset wrapping)
        assert toolset is not None

    def test_websearch_requires_builtin_with_constraints(self, allow_model_requests: None):
        """WebSearch(allowed_domains=...) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset()))  # type: ignore
        agent = Agent(model, capabilities=[WebSearch(allowed_domains=['example.com'])])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('search')

    def test_websearch_both_false_raises(self):
        """WebSearch(builtin=False, local=False) → UserError at construction."""
        with pytest.raises(UserError, match='both builtin and local cannot be False'):
            WebSearch(builtin=False, local=False)

    def test_websearch_builtin_false_with_constraints_raises(self):
        """WebSearch(builtin=False, allowed_domains=...) → UserError at construction."""
        with pytest.raises(UserError, match='constraint fields require the builtin tool'):
            WebSearch(builtin=False, allowed_domains=['example.com'])

    def test_websearch_local_callable(self):
        """WebSearch(local=some_function) → bare callable wrapped in Tool."""
        from pydantic_ai.tools import Tool

        def my_search(query: str) -> str:
            return f'results for {query}'  # pragma: no cover

        cap = WebSearch(local=my_search)
        assert isinstance(cap.local, Tool)


@pytest.mark.skipif(not xai_imports(), reason='xai_sdk not installed')
class TestXSearchCapability:
    def test_xsearch_default(self):
        """XSearch() with defaults → builtin XSearchTool, no local."""
        cap = XSearch()
        assert cap.get_builtin_tools() == snapshot([XSearchTool()])
        assert cap.get_toolset() is None

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
        assert cap.get_builtin_tools() == snapshot(
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

    def test_xsearch_requires_builtin_with_handles(self):
        """XSearch with handle constraints requires builtin."""
        assert XSearch(allowed_x_handles=['h']).get_builtin_tools() == snapshot([XSearchTool(allowed_x_handles=['h'])])
        assert XSearch(excluded_x_handles=['h']).get_builtin_tools() == snapshot(
            [XSearchTool(excluded_x_handles=['h'])]
        )

    def test_xsearch_builtin_false_local_false_raises(self):
        """XSearch(builtin=False, local=False) → UserError."""
        with pytest.raises(UserError, match='both builtin and local cannot be False'):
            XSearch(builtin=False, local=False)

    def test_xsearch_builtin_false_with_constraints_raises(self):
        """XSearch(builtin=False, allowed_x_handles=...) → UserError."""
        with pytest.raises(UserError, match='constraint fields require the builtin tool'):
            XSearch(builtin=False, allowed_x_handles=['handle1'])


class TestWebFetchCapability:
    def test_webfetch_default(self):
        """WebFetch() provides builtin and default local fallback."""
        cap = WebFetch()
        builtins = cap.get_builtin_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], WebFetchTool)
        # Default local fallback is auto-detected (markdownify-based)
        assert cap.local is not None
        assert cap.get_toolset() is not None

    def test_webfetch_default_with_nonsupporting_model(self, allow_model_requests: None):
        """WebFetch() with a model that doesn't support builtin → markdownify fallback used."""
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

        model = FunctionModel(model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
        agent = Agent(model, capabilities=[WebFetch()])
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            result = agent.run_sync('fetch something')
        # Verify the web_fetch fallback tool was actually called
        tool_calls = [
            part
            for msg in result.all_messages()
            if isinstance(msg, ModelResponse)
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == 'web_fetch'

    def test_webfetch_local_false_with_nonsupporting_model(self, allow_model_requests: None):
        """WebFetch(local=False) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset()))  # type: ignore
        agent = Agent(model, capabilities=[WebFetch(local=False)])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('fetch')

    def test_webfetch_builtin_false(self):
        """WebFetch(builtin=False) → only local, no builtin registered."""
        cap = WebFetch(builtin=False)
        assert cap.get_builtin_tools() == []
        toolset = cap.get_toolset()
        assert toolset is not None

    def test_webfetch_max_uses_requires_builtin(self, allow_model_requests: None):
        """WebFetch(max_uses=...) with non-supporting model → UserError."""
        model = FunctionModel(lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset()))  # type: ignore
        agent = Agent(model, capabilities=[WebFetch(max_uses=5)])
        with pytest.raises(UserError, match='not supported'):
            agent.run_sync('fetch')

    def test_webfetch_domains_forwarded_to_local(self, allow_model_requests: None):
        """WebFetch(allowed_domains=...) with non-supporting model → falls back to local with domain filtering."""
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

        model = FunctionModel(model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
        agent = Agent(model, capabilities=[WebFetch(allowed_domains=['example.com'])])
        with patch(
            'pydantic_ai.common_tools.web_fetch.safe_download', new_callable=AsyncMock, return_value=mock_response
        ):
            result = agent.run_sync('fetch example.com')
        # Verify the web_fetch fallback tool was actually called with domain filtering
        tool_calls = [
            part
            for msg in result.all_messages()
            if isinstance(msg, ModelResponse)
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == 'web_fetch'

    def test_webfetch_both_false_raises(self):
        """WebFetch(builtin=False, local=False) → UserError at construction."""
        with pytest.raises(UserError, match='both builtin and local cannot be False'):
            WebFetch(builtin=False, local=False)

    def test_webfetch_builtin_false_with_max_uses_raises(self):
        """WebFetch(builtin=False, max_uses=...) → UserError at construction."""
        with pytest.raises(UserError, match='constraint fields require the builtin tool'):
            WebFetch(builtin=False, max_uses=5)

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

        # partial_images is excluded — not useful for subagent fallback (no streaming)
        builtin_fields = {
            f.name for f in dataclasses.fields(ImageGenerationTool) if f.name not in ('kind', 'partial_images')
        }
        builtin_fields.remove('model')
        builtin_fields.add('image_model')
        init_params = set(inspect.signature(ImageGeneration.__init__).parameters.keys()) - {
            'self',
            'builtin',
            'local',
            'fallback_model',
        }
        assert init_params == builtin_fields

    def test_image_generation_default(self):
        """ImageGeneration() provides only builtin, no local fallback."""
        cap = ImageGeneration()
        builtins = cap.get_builtin_tools()
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
        builtins = cap.get_builtin_tools()
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
        builtins = cap.get_builtin_tools()
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

    def test_image_generation_fallback_merges_custom_builtin_with_overrides(self):
        """Custom builtin settings are merged with capability-level overrides for the fallback."""
        from pydantic_ai.tools import Tool

        custom_builtin = ImageGenerationTool(quality='high', size='1024x1024')
        cap = ImageGeneration(
            builtin=custom_builtin,
            fallback_model='openai-responses:gpt-5.4',
            output_format='jpeg',  # capability-level override
        )
        # The local fallback should exist and contain the merged config
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_callable_builtin_with_fallback(self):
        """When builtin is a callable, the fallback local tool still gets created."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(
            builtin=lambda ctx: ImageGenerationTool(quality='high'),
            fallback_model='openai-responses:gpt-5.4',
        )
        # Callable builtin can't be resolved at init time, but local fallback is still created
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

        async def model_factory(ctx: RunContext[None]) -> FunctionModel:
            return inner_model

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
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

        def model_factory(ctx: RunContext[None]) -> str:
            return 'openai-responses:gpt-image-1'

        def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
        agent = Agent(outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)])  # pyright: ignore[reportArgumentType]
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

        outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
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
    @pytest.mark.filterwarnings('ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning')
    @pytest.mark.filterwarnings('ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning')
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
        outer_model = FunctionModel(model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
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
    @pytest.mark.filterwarnings('ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning')
    @pytest.mark.filterwarnings('ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning')
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
        outer_model = FunctionModel(model_fn, profile=ModelProfile(supported_builtin_tools=frozenset()))
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


try:
    import mcp as _mcp

    has_mcp = True
    del _mcp
except ImportError:
    has_mcp = False


@pytest.mark.skipif(not has_mcp, reason='mcp is not installed')
class TestMCPCapability:
    def test_mcp_default(self):
        """MCP(url=...) provides builtin + local fallback."""
        cap = MCP(url='https://mcp.example.com/api')
        builtins = cap.get_builtin_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], MCPServerTool)
        assert builtins[0].url == 'https://mcp.example.com/api'
        assert cap.get_toolset() is not None

    def test_mcp_id_from_url(self):
        """MCP auto-derives id from URL including hostname to avoid collisions."""
        cap = MCP(url='https://mcp.example.com/api')
        builtin = cap.get_builtin_tools()[0]
        assert isinstance(builtin, MCPServerTool)
        assert builtin.id == 'mcp.example.com-api'

        # SSE URLs include hostname to avoid collisions between different servers
        cap_sse = MCP(url='https://server1.example.com/sse')
        builtin_sse = cap_sse.get_builtin_tools()[0]
        assert isinstance(builtin_sse, MCPServerTool)
        assert builtin_sse.id == 'server1.example.com-sse'

    def test_mcp_sse_transport(self):
        """MCP with /sse URL uses MCPServerSSE for local."""
        from pydantic_ai.mcp import MCPServerSSE

        cap = MCP(url='https://mcp.example.com/sse')
        assert isinstance(cap.local, MCPServerSSE)

    def test_mcp_streamable_transport(self):
        """MCP with non-/sse URL uses MCPServerStreamableHTTP for local."""
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        cap = MCP(url='https://mcp.example.com/api')
        assert isinstance(cap.local, MCPServerStreamableHTTP)

    def test_mcp_authorization_token_in_local_headers(self):
        """MCP passes authorization_token as Authorization header to local."""
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        cap = MCP(url='https://mcp.example.com/api', authorization_token='Bearer xyz')
        assert isinstance(cap.local, MCPServerStreamableHTTP)
        assert cap.local.headers == {'Authorization': 'Bearer xyz'}

    def test_mcp_allowed_tools_filters_local(self):
        """MCP(allowed_tools=...) applies FilteredToolset to the local toolset."""
        from pydantic_ai.toolsets.filtered import FilteredToolset

        cap = MCP(url='https://mcp.example.com/api', allowed_tools=['tool1'])
        toolset = cap.get_toolset()
        assert toolset is not None
        # The outer toolset should be a FilteredToolset wrapping the prepared toolset
        assert isinstance(toolset, FilteredToolset)

    def test_mcp_url_required(self):
        """MCP without url raises TypeError."""
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'url'"):
            MCP()  # type: ignore[call-arg]


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

        async def hide_secret_tools(
            ctx: RunContext[None], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition] | None:
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

    async def test_prepare_tools_none_disables_all(self):
        """PrepareTools treats None return as 'disable all tools', consistent with ToolsPrepareFunc docs."""
        from pydantic_ai.capabilities import PrepareTools

        async def disable_all(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
            return None

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f'tools: {sorted(tool_names)}')

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(disable_all)])

        @agent.tool_plain
        def my_tool() -> str:
            return 'result'  # pragma: no cover

        result = await agent.run('hello')
        assert result.output == 'tools: []'

    async def test_prepare_tools_modifies_definitions(self):
        """PrepareTools can modify tool definitions (e.g. set strict mode)."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.capabilities import PrepareTools

        async def set_strict(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
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

        async def rename(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
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

        async def hide_secret(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
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

        async def disable_all(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
            return None

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response(f'output_tools: {len(info.output_tools)}')

        agent = Agent(
            FunctionModel(model_fn),
            output_type=[str, ToolOutput(Out, name='out')],
            capabilities=[PrepareOutputTools(disable_all)],
        )

        result = await agent.run('hello')
        assert result.output == 'output_tools: 0'

    async def test_only_sees_output_tools(self):
        """`PrepareOutputTools` only receives output tools — function tools route to `PrepareTools`."""
        from pydantic_ai.capabilities import PrepareOutputTools

        seen_kinds: list[str] = []

        async def capture(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
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


class TestAgentPrepareArgInjection:
    """The Agent `prepare_tools` / `prepare_output_tools` constructor args are
    sugar for `PrepareTools` / `PrepareOutputTools` capabilities — verify they
    show up in `root_capability` and apply the same way."""

    def test_prepare_tools_arg_injects_capability(self):
        from pydantic_ai.capabilities import PrepareTools

        async def noop(
            ctx: RunContext[None], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:  # pragma: no cover
            return tool_defs

        agent = Agent('test', prepare_tools=noop)
        injected = [c for c in agent.root_capability.capabilities if isinstance(c, PrepareTools)]
        assert len(injected) == 1
        assert injected[0].prepare_func is noop

    def test_prepare_output_tools_arg_injects_capability(self):
        from pydantic_ai.capabilities import PrepareOutputTools

        async def noop(
            ctx: RunContext[None], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:  # pragma: no cover
            return tool_defs

        agent = Agent('test', output_type=str, prepare_output_tools=noop)
        injected = [c for c in agent.root_capability.capabilities if isinstance(c, PrepareOutputTools)]
        assert len(injected) == 1
        assert injected[0].prepare_func is noop


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

        with pytest.warns(UserWarning, match='retries'):
            await agent.run('hello', spec={'retries': 5})


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

        agent = Agent(FunctionModel(model_fn), prepare_tools=agent_prepare, capabilities=[PrefixCap()])

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

        def dynamic_metadata(ctx: RunContext[None]) -> dict[str, Any]:
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

        def dynamic_settings(ctx: RunContext[None]) -> _ModelSettings:
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
    """WebFetch capability populates builtin tool with all constraint kwargs."""
    cap = WebFetch(
        allowed_domains=['example.com'],
        blocked_domains=['bad.com'],
        max_uses=5,
        enable_citations=True,
        max_content_tokens=1000,
    )
    builtin_tools = cap.get_builtin_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, WebFetchTool)
    assert tool.allowed_domains == ['example.com']
    assert tool.blocked_domains == ['bad.com']
    assert tool.max_uses == 5
    assert tool.enable_citations is True
    assert tool.max_content_tokens == 1000
    # Only max_uses requires builtin (domains are handled locally)
    assert cap._requires_builtin() is True  # pyright: ignore[reportPrivateUsage]


def test_web_fetch_unique_id():
    """WebFetch returns the correct builtin unique_id."""
    cap = WebFetch()
    assert cap._builtin_unique_id() == 'web_fetch'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(not xai_imports(), reason='xai_sdk not installed')
def test_xsearch_unique_id():
    """XSearch returns the correct builtin unique_id."""
    cap = XSearch()
    assert cap._builtin_unique_id() == 'x_search'  # pyright: ignore[reportPrivateUsage]


def test_web_search_with_constraints():
    """WebSearch capability populates builtin tool with all constraint kwargs."""
    from pydantic_ai.builtin_tools import WebSearchUserLocation

    cap = WebSearch(
        search_context_size='high',
        user_location=WebSearchUserLocation(city='NYC', country='US'),
        blocked_domains=['bad.com'],
        allowed_domains=['good.com'],
        max_uses=3,
    )
    builtin_tools = cap.get_builtin_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.search_context_size == 'high'
    assert tool.user_location is not None
    assert tool.blocked_domains == ['bad.com']
    assert tool.allowed_domains == ['good.com']
    assert tool.max_uses == 3
    assert cap._requires_builtin() is True  # pyright: ignore[reportPrivateUsage]


def test_web_search_default_local_import_error(monkeypatch: pytest.MonkeyPatch):
    """WebSearch._default_local() warns and returns None when duckduckgo is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.common_tools.duckduckgo':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.warns(UserWarning, match='duckduckgo'):
        cap = WebSearch(builtin=False)
    # With builtin disabled and no duckduckgo, local is None
    assert cap.local is None


def test_web_fetch_default_local_import_error(monkeypatch: pytest.MonkeyPatch):
    """WebFetch._default_local() warns and returns None when markdownify is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'pydantic_ai.common_tools.web_fetch':
            raise ImportError('mocked')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', mock_import)
    with pytest.warns(UserWarning, match='web-fetch'):
        cap = WebFetch(builtin=False)
    # With builtin disabled and no markdownify, local is None
    assert cap.local is None


def test_mcp_default_builtin():
    """MCP capability constructs the default builtin MCPServerTool."""
    pytest.importorskip('mcp', reason='mcp package not installed')
    cap = MCP(url='http://example.com/mcp', id='my-mcp')
    builtin_tools = cap.get_builtin_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, MCPServerTool)
    assert tool.url == 'http://example.com/mcp'
    assert tool.id == 'my-mcp'


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_builtin_or_local_base_no_default_builtin():
    """BuiltinOrLocalTool base class with builtin=True raises (no _default_builtin)."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    with pytest.raises(UserError, match='builtin=True requires a subclass'):
        BuiltinOrLocalTool()


def test_builtin_tool_from_spec_no_args():
    """BuiltinTool.from_spec() with no arguments raises TypeError."""
    from pydantic_ai.capabilities.builtin_tool import BuiltinTool as BuiltinToolCapDirect

    with pytest.raises(TypeError, match='requires either a `tool` argument'):
        BuiltinToolCapDirect.from_spec()


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_builtin_or_local_no_default_local():
    """BuiltinOrLocalTool base class _default_local() returns None."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    cap = BuiltinOrLocalTool(builtin=WebSearchTool())
    # Base class _default_local() returns None — no local fallback
    assert cap.local is None
    assert cap.get_toolset() is None


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_builtin_or_local_with_explicit_builtin():
    """BuiltinOrLocalTool used directly with an explicit builtin and local tool."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    def my_local_tool() -> str:
        """A local fallback tool."""
        return 'local result'  # pragma: no cover

    cap = BuiltinOrLocalTool(builtin=WebSearchTool(), local=my_local_tool)
    # get_builtin_tools returns the explicit builtin
    assert len(cap.get_builtin_tools()) == 1
    assert isinstance(cap.get_builtin_tools()[0], WebSearchTool)
    # get_toolset wraps local with prefer_builtin from _builtin_unique_id()
    toolset = cap.get_toolset()
    assert toolset is not None


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_builtin_or_local_builtin_unique_id_non_abstract():
    """_builtin_unique_id() raises when builtin is callable (not AbstractBuiltinTool)."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    cap = BuiltinOrLocalTool.__new__(BuiltinOrLocalTool)
    cap.builtin = lambda ctx: WebSearchTool()
    cap.local = False

    with pytest.raises(UserError, match='cannot derive builtin_unique_id'):
        cap._builtin_unique_id()  # pyright: ignore[reportPrivateUsage]


def test_validate_capability_not_dataclass():
    """Custom capability type without @dataclass raises ValueError."""
    from pydantic_ai.agent.spec import get_capability_registry

    class NotADataclass(AbstractCapability[Any]):
        pass

    with pytest.raises(ValueError, match='must be decorated with `@dataclass`'):
        get_capability_registry(custom_types=(NotADataclass,))


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
                        'capability': {'BuiltinTool': {'kind': 'web_search'}},
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
    cap = PrefixTools.from_spec(prefix='ws', capability='WebSearch')  # pyright: ignore[reportArgumentType]
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

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        return toolset

    cap = PrefixTools(wrapped=Toolset(toolset_func), prefix='dyn')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run('list tools')
    assert result.output == 'dyn_dynamic_tool'


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
    class HookCap(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            hook_calls.append('before_run')

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            hook_calls.append('after_run')
            return result

    wrapper = WrapperCapability(wrapped=HookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run('Hello')

    assert 'before_run' in hook_calls
    assert 'after_run' in hook_calls


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
    class SwitchCap(AbstractCapability[None]):
        use_b: bool = False

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return SwitchCap(use_b=True)

        def get_toolset(self) -> AbstractToolset[None]:
            return toolset_b if self.use_b else toolset_a

    wrapper = WrapperCapability(wrapped=SwitchCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(','.join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    result = await agent.run('Hello')
    # for_run switches to toolset_b
    assert 'tool_b' in result.output


async def test_wrapper_capability_has_wrap_node_run():
    """WrapperCapability.has_wrap_node_run delegates to the wrapped capability."""
    plain = CustomCapability()
    assert WrapperCapability(wrapped=plain).has_wrap_node_run is False

    @dataclass
    class NodeRunCap(AbstractCapability[None]):
        async def wrap_node_run(self, ctx: RunContext[None], *, node: Any, handler: Any) -> Any:
            return await handler(node)  # pragma: no cover

    assert WrapperCapability(wrapped=NodeRunCap()).has_wrap_node_run is True


async def test_wrapper_capability_delegates_model_request_hooks():
    """WrapperCapability delegates before/after model request hooks."""
    hook_calls: list[str] = []

    @dataclass
    class ModelRequestHookCap(AbstractCapability[None]):
        async def before_model_request(
            self, ctx: RunContext[None], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            hook_calls.append('before_model_request')
            return request_context

        async def after_model_request(
            self, ctx: RunContext[None], *, request_context: ModelRequestContext, response: ModelResponse
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


# --- ModelRetry from hooks tests ---


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
            output_retries=2,
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

        agent = Agent(FunctionModel(simple_model_function), capabilities=[WrapRetrySkipErrorCap()], output_retries=1)
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

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks], tool_retries=2, output_retries=2)

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

        agent = Agent(FunctionModel(model_fn), capabilities=[WrapExecRetryCap()], tool_retries=2, output_retries=2)

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

        agent = Agent(FunctionModel(model_fn), capabilities=[ErrorRetryCap()], tool_retries=2, output_retries=2)

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

        agent = Agent(FunctionModel(model_fn), capabilities=[AfterValRetryCap()], tool_retries=2, output_retries=2)

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

        agent = Agent(FunctionModel(model_fn), capabilities=[BeforeValRetryCap()], tool_retries=2, output_retries=2)

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

        def _resolve(cap: OpenAICompaction[None], model_settings: dict[str, Any] | None = None) -> dict[str, Any]:
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

    def test_openai_compaction_instructions_deprecated(self):
        """Passing `instructions` emits a DeprecationWarning because OpenAI semantics differ from Anthropic."""
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAICompaction

        with pytest.warns(DeprecationWarning, match='OpenAICompaction\\(instructions='):
            OpenAICompaction(message_count_threshold=5, instructions='Summarize briefly')

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
    """Ordering from leaves inside a nested CombinedCapability is respected.

    When a CombinedCapability is nested inside another, its leaves' ordering
    constraints are merged and applied to the outer sort. Leaves without
    ordering constraints are skipped during the merge.
    """
    # OutermostCap declares position='outermost'; PlainCapB has no ordering.
    # The merged effective ordering for 'inner' should be position='outermost',
    # placing it before PlainCapA despite being listed second.
    inner = CombinedCapability([PlainCapB(), OutermostCap()])
    combined = CombinedCapability([PlainCapA(), inner])
    assert combined.capabilities[0] is inner


def test_ordering_nested_combined_no_constraints():
    """A nested CombinedCapability with no ordering leaves is treated as unconstrained."""
    inner = CombinedCapability([PlainCapA(), PlainCapB()])
    combined = CombinedCapability([inner, OutermostCap()])
    # OutermostCap first (has ordering), inner second (no constraints → unconstrained)
    assert combined.capabilities[0].__class__ is OutermostCap
    assert combined.capabilities[1] is inner


def test_ordering_nested_combined_wraps_without_position():
    """A nested CombinedCapability with wraps constraints (but no position) merges correctly."""
    inner = CombinedCapability([PlainCapB(), WrapsACap()])
    # WrapsACap has wraps=[PlainCapA] but no position.
    # The nested group inherits that constraint, so it sorts before PlainCapA.
    combined = CombinedCapability([PlainCapA(), inner])
    assert combined.capabilities[0] is inner
    assert combined.capabilities[1].__class__ is PlainCapA


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


def test_ordering_conflicting_positions_in_nested():
    """Conflicting positions in a nested CombinedCapability raise UserError."""
    inner = CombinedCapability([OutermostCap(), InnermostCap()])
    with pytest.raises(UserError, match='Conflicting positions in nested CombinedCapability'):
        CombinedCapability([inner, PlainCapA()])


def test_ordering_wrapper_capability_recurses():
    """Ordering constraints on capabilities inside a WrapperCapability are preserved."""
    wrapped = WrapperCapability(wrapped=OutermostCap())
    # The WrapperCapability wraps an OutermostCap; ordering sees through via apply()
    # and picks up OutermostCap's position='outermost' constraint.
    combined = CombinedCapability([PlainCapA(), wrapped])
    assert combined.capabilities[0] is wrapped


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
        while not isinstance(node, End):
            try:
                node = await agent_run.next(node)
            except ValueError:
                # After an unrecovered error, next_node should re-raise
                with pytest.raises(ValueError, match='model failure'):
                    _ = agent_run.next_node
                break


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
            model=None,  # type: ignore
            usage=None,  # type: ignore
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
            model=None,  # type: ignore
            usage=None,  # type: ignore
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
                model=None,  # type: ignore
                usage=None,  # type: ignore
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
            asyncio.get_event_loop().run_until_complete(run())

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

            pass

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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Not allowed.') for call in requests.approvals}
        )

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 10  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood, denied.'
    # The denial must surface in message history as outcome='denied', not a successful return.
    tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # Note: output_type is just str, no DeferredToolRequests
    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    def tool_a(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a done'

    @agent.tool
    def tool_b(ctx: RunContext[None]) -> str:
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

    def handle_deferred_sync(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred_sync)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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

    def handler_a(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    def handler_b(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    def tool_a(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a result'

    @agent.tool
    def tool_b(ctx: RunContext[None]) -> str:
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
    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults()  # Empty results → nothing resolved

    agent = Agent(
        FunctionModel(llm),
        output_type=str,
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def outer_tool(ctx: RunContext[None]) -> str:
        """A tool that internally calls another tool via ToolManager.handle_call."""
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        result = await ctx.tool_manager.handle_call(inner_call)
        return f'inner returned: {result}'

    @agent.tool
    def inner_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def outer_tool(ctx: RunContext[None]) -> str:
        assert ctx.tool_manager is not None
        inner_call = ToolCallPart(tool_name='inner_tool', args={}, tool_call_id='inner1')
        try:
            await ctx.tool_manager.handle_call(inner_call, wrap_validation_errors=False)
        except ModelRetry as e:
            return f'raw ModelRetry: {e}'
        return 'no error'  # pragma: no cover

    @agent.tool
    def inner_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved inner result'  # pragma: no cover

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('outer_tool', {}, tool_call_id='outer1')])
        return ModelResponse(parts=[TextPart('OK')])

    agent = Agent(FunctionModel(llm), toolsets=[inner_toolset])

    @agent.tool
    async def outer_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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
    with pytest.raises(ValueError, match='calls.*not in.*DeferredToolRequests.calls'):
        requests.build_results(calls={'approval_1': 'oops'})

    # Unknown ID entirely.
    with pytest.raises(ValueError, match='approvals.*not in.*DeferredToolRequests.approvals'):
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    # PrefixTools wraps HandleDeferredToolCalls — tests WrapperCapability delegation
    inner = HandleDeferredToolCalls(handler=handle_deferred)
    agent = Agent(
        FunctionModel(llm),
        capabilities=[inner.prefix_tools('ns')],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolDenied('Policy denied.') for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    assert result.output == 'Understood.'
    tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            approvals={
                'bool_false': False,
                'default_denied': ToolDenied(),  # no custom message
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x  # pragma: no cover

    result = await agent.run('go')
    assert result.output == 'ok'
    tool_returns = {p.tool_call_id: p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)}
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: ToolApproved() for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def needs_approval(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 2

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    def external_tool(ctx: RunContext[None]) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'done'
    messages = result.all_messages()
    tool_returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    assert len(tool_returns) == 1
    assert tool_returns[0].content == 'computed'
    assert tool_returns[0].metadata == {'source': 'external'}
    # The `content` field on ToolReturn becomes a UserPromptPart.
    from pydantic_ai.messages import UserPromptPart

    user_extras = [p for m in messages for p in m.parts if isinstance(p, UserPromptPart) and p.content == 'user extra']
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(calls={call.tool_call_id: ModelRetry('try again') for call in requests.calls})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext[None]) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    messages = result.all_messages()
    retry_parts = [p for m in messages for p in m.parts if isinstance(p, RetryPromptPart)]
    assert len(retry_parts) == 1
    assert retry_parts[0].tool_call_id == 'c1'
    assert retry_parts[0].content == 'try again'
    tool_returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 'c1']
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(
            calls={
                call.tool_call_id: RetryPromptPart(content='retry via part', tool_name='', tool_call_id='')
                for call in requests.calls
            }
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def external_tool(ctx: RunContext[None]) -> str:
        raise CallDeferred

    result = await agent.run('go')
    assert result.output == 'retried'
    retry_parts = [p for m in result.all_messages() for p in m.parts if isinstance(p, RetryPromptPart)]
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'approved result'

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'caller_tool'
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext[None]):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='result', metadata={'source': 'tool'}, content='user prompt extra')

    result = await agent.run('Hello')
    assert result.output == 'Done.'
    # Verify ToolReturn.metadata preserved
    tool_returns = [
        p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_name == 'my_tool'
    ]
    assert len(tool_returns) == 1
    assert tool_returns[0].metadata == {'source': 'tool'}
    # Verify ToolReturn.content appears as UserPromptPart
    user_parts = [
        p
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, UserPromptPart) and p.content == 'user prompt extra'
    ]
    assert len(user_parts) == 1


async def test_deferred_tool_handler_approved_tool_raises_model_retry():
    """Approved tool that raises ModelRetry produces a RetryPromptPart."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])
        return ModelResponse(parts=[TextPart('Retried and done.')])

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        raise ModelRetry('try again')

    result = await agent.run('Hello')
    assert result.output == 'Retried and done.'
    # Verify the retry happened
    retry_parts = [
        p for m in result.all_messages() for p in m.parts if isinstance(p, RetryPromptPart) and p.tool_name == 'my_tool'
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        # Override the args: replace x=5 with x=42
        return DeferredToolResults(
            approvals={call.tool_call_id: ToolApproved(override_args={'x': 42}) for call in requests.approvals}
        )

    agent = Agent(FunctionModel(llm), capabilities=[HandleDeferredToolCalls(handler=handle_deferred)])

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        nonlocal retry_count
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        retry_count += 1
        raise ModelRetry('try again')

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    def re_raising_tool(ctx: RunContext[None]) -> str:
        # Always raises — even after approval
        raise ApprovalRequired

    @agent.tool
    def unhandled_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[HandleDeferredToolCalls(handler=handle_deferred)],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]):
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return _ToolReturn(return_value='actual result', metadata={'source': 'inner'}, content='user extra')

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None], x: int) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return f'x={x}'

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'never'  # pragma: no cover

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ApprovalRequired
        # After approval, raise a *different* deferral type with new metadata
        raise CallDeferred(metadata={'reason': 'external-after-approval'})

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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
    def inner_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    async def handle_deferred(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
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
    async def caller_tool(ctx: RunContext[None]) -> str:
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

    hooks = Hooks[None]()

    @hooks.on.deferred_tool_calls
    async def handler(ctx: RunContext[None], *, requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
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

    def handle_a(ctx: RunContext[None], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        results = DeferredToolResults()
        for call in requests.approvals:
            if call.tool_name == 'tool_a':
                results.approvals[call.tool_call_id] = True
        return results

    hooks = Hooks[None](deferred_tool_calls=handle_a)

    @hooks.on.deferred_tool_calls
    async def handle_rest(ctx: RunContext[None], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # tool_a was already resolved by handle_a; this handler sees only tool_b and tool_c
        return DeferredToolResults(approvals={call.tool_call_id: True for call in requests.approvals})

    @hooks.on.deferred_tool_calls
    async def never_called(  # pragma: no cover
        ctx: RunContext[None], *, requests: DeferredToolRequests
    ) -> DeferredToolResults | None:
        # All calls should already be resolved by the previous handler — this is the early-break path
        raise AssertionError('Should not be called: all requests already resolved')

    agent = Agent(FunctionModel(llm), capabilities=[hooks])

    @agent.tool
    def tool_a(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'a'

    @agent.tool
    def tool_b(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'b'

    @agent.tool
    def tool_c(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'c'

    result = await agent.run('Hello')
    assert result.output == 'All done.'


async def test_deferred_tool_handler_via_hooks_returns_none_when_unhandled():
    """`Hooks` returns None from the deferred-tool-calls hook when no registered handler resolves anything."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('my_tool', {}, tool_call_id='call1')])

    hooks = Hooks[None]()

    @hooks.on.deferred_tool_calls
    async def declines(ctx: RunContext[None], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        return None

    @hooks.on.deferred_tool_calls
    async def empty(ctx: RunContext[None], *, requests: DeferredToolRequests) -> DeferredToolResults | None:
        # Empty results count as "didn't handle"
        return DeferredToolResults()

    agent = Agent(
        FunctionModel(llm),
        output_type=[str, DeferredToolRequests],
        capabilities=[hooks],
    )

    @agent.tool
    def my_tool(ctx: RunContext[None]) -> str:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return 'done'  # pragma: no cover

    result = await agent.run('Hello')
    # Falls through to bubble-up since no handler resolved anything
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1


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
    """The factory receives the run's RunContext (with deps) once per run."""
    seen: list[Any] = []

    def factory(ctx: RunContext[str]) -> AbstractCapability[Any] | None:
        seen.append(ctx.deps)
        return _RecordingCapability(label=ctx.deps)

    agent = Agent(TestModel(), deps_type=str, capabilities=[factory])
    await agent.run('hi', deps='admin')
    await agent.run('hi', deps='guest')
    assert seen == ['admin', 'guest']


async def test_dynamic_capability_async_factory() -> None:
    """Async factories are awaited."""
    calls = 0

    async def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='async')

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert calls == 1


async def test_dynamic_capability_returning_none_contributes_nothing() -> None:
    """A factory returning None is a no-op for the run."""

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any] | None:
        return None

    agent = Agent(TestModel(), capabilities=[factory])
    result = await agent.run('hi')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions is None


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
    """Resolved capability's toolset is exposed to the model and its tools execute."""
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def special() -> str:
        return 'used'

    @dataclass
    class ToolCap(AbstractCapability[None]):
        def get_toolset(self):
            return toolset

    def factory(ctx: RunContext[bool]) -> AbstractCapability[Any] | None:
        return ToolCap() if ctx.deps else None

    seen_tools: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_tools.append(','.join(sorted(t.name for t in info.function_tools)))
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

    await agent.run('hi', deps=False)
    assert seen_tools == ['special', 'special', '']


async def test_dynamic_capability_hooks_fire() -> None:
    """Hooks contributed by the resolved capability fire during the run."""
    cap = _RecordingCapability(label='dyn')

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
        return cap

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert 'dyn:before_run' in cap.fired
    assert 'dyn:before_model_request' in cap.fired


async def test_dynamic_capability_factory_called_once_per_run_not_per_step() -> None:
    """The factory is called once at for_run, not on every model request."""
    calls = 0

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
        nonlocal calls
        calls += 1
        return _RecordingCapability(label='once')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Two-step run: first a tool call, then a final text response.
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('echo', {'text': 'hi'})])
        return ModelResponse(parts=[TextPart('done')])

    toolset = FunctionToolset[None]()

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
    class A(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            fired.append('A')

    @dataclass
    class B(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            fired.append('B')

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
        return CombinedCapability([A(), B()])

    agent = Agent(TestModel(), capabilities=[factory])
    await agent.run('hi')
    assert fired == ['A', 'B']


async def test_dynamic_capability_in_run_call() -> None:
    """`agent.run(capabilities=[factory])` accepts callables as well."""
    calls = 0

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
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
    class Static(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            fired.append('static')

    @dataclass
    class Dynamic(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            fired.append('dynamic')

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
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

    def factory(ctx: RunContext[None]) -> AbstractCapability[Any]:
        return _RecordingCapability(label='x')

    agent = Agent(TestModel(), capabilities=[factory])

    result = await agent.run('hi')
    request = next(m for m in result.all_messages() if isinstance(m, ModelRequest))
    assert request.instructions == 'Label is x.'


# endregion
