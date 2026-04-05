from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterable, AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

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
)
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    MCP,
    BuiltinTool,
    ImageGeneration,
    PrefixTools,
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
    ModelRetry,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.toolsets._dynamic import ToolsetFunc
from pydantic_ai.usage import RequestUsage, RunUsage
from pydantic_graph import End

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsInstance, IsStr

pytestmark = [
    pytest.mark.anyio,
]


def test_capability_types() -> None:
    assert CAPABILITY_TYPES == snapshot(
        {
            "BuiltinTool": BuiltinTool,
            "ImageGeneration": ImageGeneration,
            "MCP": MCP,
            "PrefixTools": PrefixTools,
            "Thinking": Thinking,
            "WebFetch": WebFetch,
            "WebSearch": WebSearch,
        }
    )


def test_agent_from_spec_basic():
    """Test Agent.from_spec with basic capabilities."""
    agent = Agent.from_spec(
        {
            "model": "test",
            "instructions": "You are a helpful agent.",
            "model_settings": {"max_tokens": 4096},
            "capabilities": [
                "WebSearch",
            ],
        }
    )
    assert agent.model is not None


def test_agent_from_spec_no_capabilities():
    """Test Agent.from_spec with no capabilities."""
    agent = Agent.from_spec({"model": "test"})
    assert agent.model is not None


def test_agent_from_spec_image_generation():
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [{"ImageGeneration": {"local": False}}],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, ImageGeneration))
    assert cap.local is False


def test_agent_from_spec_web_fetch():
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [
                {"WebFetch": {"allowed_domains": ["example.com"], "max_uses": 5}}
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, WebFetch))
    assert cap.allowed_domains == ["example.com"]
    assert cap.max_uses == 5


def test_agent_from_spec_mcp():
    pytest.importorskip("mcp", reason="mcp package not installed")
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [
                {
                    "MCP": {
                        "url": "https://mcp.example.com/sse",
                        "allowed_tools": ["search"],
                    }
                }
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    cap = next(c for c in children if isinstance(c, MCP))
    assert cap.url == "https://mcp.example.com/sse"
    assert cap.allowed_tools == ["search"]


def test_agent_from_spec_unknown_capability():
    """Test Agent.from_spec with an unknown capability name."""
    with pytest.raises(ValueError, match="Capability 'Unknown' is not in the provided"):
        Agent.from_spec(
            {
                "model": "test",
                "capabilities": ["Unknown"],
            }
        )


def test_agent_from_spec_bad_args():
    """Test Agent.from_spec with bad arguments for a capability."""
    with pytest.raises(
        ValueError, match="Failed to instantiate capability 'WebSearch'"
    ):
        Agent.from_spec(
            {
                "model": "test",
                "capabilities": [
                    {"WebSearch": {"nonexistent_param": "value"}},
                ],
            }
        )


@dataclass
class CustomCapability(AbstractCapability[None]):
    greeting: str = "hello"


@dataclass
class CapabilityWithCallbackParam(AbstractCapability[None]):
    """Custom capability with a mix of serializable and non-serializable params."""

    max_retries: int = 3
    on_error: Callable[..., Any] = lambda: None  # purely Callable, filtered from schema
    verbose: Callable[..., Any] | bool = (
        False  # Callable | bool, only bool survives in schema
    )
    hooks: Callable[..., Any] | Callable[..., None] = (
        lambda: None
    )  # union of all non-serializable, entirely filtered


def test_agent_from_spec_custom_capability():
    """Test Agent.from_spec with a custom capability type."""
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [
                {"CustomCapability": "world"},
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


def test_agent_from_spec_with_agent_spec_object():
    """Test Agent.from_spec with an AgentSpec instance."""
    spec = AgentSpec(
        model="test",
        instructions="You are helpful.",
        capabilities=[
            CapabilitySpec(name="WebSearch", arguments=None),
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

    agent = Agent.from_spec({"model": "test"}, output_type=MyOutput)
    assert agent.output_type == MyOutput


def test_agent_from_spec_output_schema():
    """Test Agent.from_spec with output_schema in spec."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }
    agent = Agent.from_spec({"model": "test", "output_schema": schema})
    # output_type should be a StructuredDict subclass (dict subclass with JSON schema)
    assert agent.output_type is not str
    assert isinstance(agent.output_type, type) and issubclass(agent.output_type, dict)


def test_agent_from_spec_output_type_takes_precedence():
    """Test that output_type parameter takes precedence over output_schema in spec."""
    from pydantic import BaseModel

    class MyOutput(BaseModel):
        name: str

    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    agent = Agent.from_spec(
        {"model": "test", "output_schema": schema}, output_type=MyOutput
    )
    assert agent.output_type == MyOutput


def test_agent_from_spec_output_schema_invalid():
    """Test Agent.from_spec with a non-object output_schema raises UserError."""
    with pytest.raises(UserError, match="Schema must be an object"):
        Agent.from_spec({"model": "test", "output_schema": {"type": "string"}})


async def test_agent_from_spec_output_schema_integration():
    """Test Agent.from_spec with output_schema produces dict output."""
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "country": {"type": "string"},
        },
        "required": ["city", "country"],
    }
    agent = Agent.from_spec({"model": "test", "output_schema": schema})
    result = await agent.run(
        "Tell me a city",
        model=TestModel(custom_output_args={"city": "Paris", "country": "France"}),
    )
    assert result.output == {"city": "Paris", "country": "France"}


def test_agent_from_spec_name():
    agent = Agent.from_spec({"model": "test", "name": "my-agent"})
    assert agent.name == "my-agent"


def test_agent_from_spec_name_override():
    agent = Agent.from_spec(
        {"model": "test", "name": "spec-name"}, name="override-name"
    )
    assert agent.name == "override-name"


def test_agent_from_spec_description():
    agent = Agent.from_spec({"model": "test", "description": "A helpful agent"})
    assert agent.description == "A helpful agent"


def test_agent_from_spec_description_override():
    agent = Agent.from_spec(
        {"model": "test", "description": "spec-desc"}, description="override-desc"
    )
    assert agent.description == "override-desc"


def test_agent_from_spec_instructions():
    agent = Agent.from_spec({"model": "test", "instructions": "Be helpful."})
    assert "Be helpful." in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_list():
    agent = Agent.from_spec({"model": "test", "instructions": ["First.", "Second."]})
    assert "First." in agent._instructions  # pyright: ignore[reportPrivateUsage]
    assert "Second." in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instructions_merged():
    agent = Agent.from_spec(
        {"model": "test", "instructions": "From spec."},
        instructions="From arg.",
    )
    assert "From spec." in agent._instructions  # pyright: ignore[reportPrivateUsage]
    assert "From arg." in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_model_settings():
    agent = Agent.from_spec(
        {"model": "test", "model_settings": {"temperature": 0.5, "max_tokens": 100}}
    )
    ms = agent.model_settings
    assert isinstance(ms, dict)
    assert ms.get("temperature") == 0.5  # pyright: ignore[reportUnknownMemberType]
    assert ms.get("max_tokens") == 100  # pyright: ignore[reportUnknownMemberType]


def test_agent_from_spec_model_settings_merged():
    agent = Agent.from_spec(
        {"model": "test", "model_settings": {"temperature": 0.5, "max_tokens": 100}},
        model_settings={"temperature": 0.9},
    )
    ms = agent.model_settings
    assert isinstance(ms, dict)
    assert ms.get("temperature") == 0.9  # pyright: ignore[reportUnknownMemberType]
    assert ms.get("max_tokens") == 100  # pyright: ignore[reportUnknownMemberType]


def test_agent_from_spec_retries():
    agent = Agent.from_spec({"model": "test", "retries": 5})
    assert agent._max_tool_retries == 5  # pyright: ignore[reportPrivateUsage]
    assert agent._max_result_retries == 5  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_retries_override():
    agent = Agent.from_spec({"model": "test", "retries": 5}, retries=2)
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]
    assert agent._max_result_retries == 2  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_output_retries():
    agent = Agent.from_spec({"model": "test", "retries": 3, "output_retries": 10})
    assert agent._max_tool_retries == 3  # pyright: ignore[reportPrivateUsage]
    assert agent._max_result_retries == 10  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_end_strategy():
    agent = Agent.from_spec({"model": "test", "end_strategy": "exhaustive"})
    assert agent.end_strategy == "exhaustive"


def test_agent_from_spec_end_strategy_override():
    agent = Agent.from_spec(
        {"model": "test", "end_strategy": "exhaustive"}, end_strategy="early"
    )
    assert agent.end_strategy == "early"


def test_agent_from_spec_tool_timeout():
    agent = Agent.from_spec({"model": "test", "tool_timeout": 30.0})
    assert agent._tool_timeout == 30.0  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_tool_timeout_override():
    agent = Agent.from_spec({"model": "test", "tool_timeout": 30.0}, tool_timeout=5.0)
    assert agent._tool_timeout == 5.0  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_instrument():
    agent = Agent.from_spec({"model": "test", "instrument": True})
    assert agent.instrument is True


def test_agent_from_spec_metadata():
    agent = Agent.from_spec(
        {"model": "test", "metadata": {"env": "prod", "version": "1.0"}}
    )
    assert agent._metadata == {"env": "prod", "version": "1.0"}  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_metadata_override():
    agent = Agent.from_spec(
        {"model": "test", "metadata": {"env": "prod"}},
        metadata={"env": "staging"},
    )
    assert agent._metadata == {"env": "staging"}  # pyright: ignore[reportPrivateUsage]


def test_agent_from_spec_model_override():
    agent = Agent.from_spec({"model": "test"}, model="test")
    assert agent.model is not None


def test_agent_from_spec_capabilities_merged():
    @dataclass
    class ExtraCap(AbstractCapability[None]):
        pass

    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": ["WebSearch"],
        },
        capabilities=[ExtraCap()],
    )
    # Should have both the WebSearch capability from spec and ExtraCap from arg
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    assert any(isinstance(c, WebSearch) for c in children)
    assert any(isinstance(c, ExtraCap) for c in children)


def test_model_json_schema_with_capabilities():
    pytest.importorskip("mcp", reason="schema varies without mcp package")
    schema = AgentSpec.model_json_schema_with_capabilities()
    assert schema == snapshot(
        {
            "$defs": {
                "CodeExecutionTool": {
                    "properties": {
                        "kind": {
                            "default": "code_execution",
                            "title": "Kind",
                            "type": "string",
                        },
                        "file_ids": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "File Ids",
                        },
                    },
                    "title": "CodeExecutionTool",
                    "type": "object",
                },
                "FileSearchTool": {
                    "properties": {
                        "kind": {
                            "default": "file_search",
                            "title": "Kind",
                            "type": "string",
                        },
                        "file_store_ids": {
                            "items": {"type": "string"},
                            "title": "File Store Ids",
                            "type": "array",
                        },
                    },
                    "required": ["file_store_ids"],
                    "title": "FileSearchTool",
                    "type": "object",
                },
                "ImageGenerationTool": {
                    "properties": {
                        "kind": {
                            "default": "image_generation",
                            "title": "Kind",
                            "type": "string",
                        },
                        "background": {
                            "default": "auto",
                            "enum": ["transparent", "opaque", "auto"],
                            "title": "Background",
                            "type": "string",
                        },
                        "input_fidelity": {
                            "anyOf": [
                                {"enum": ["high", "low"], "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Input Fidelity",
                        },
                        "moderation": {
                            "default": "auto",
                            "enum": ["auto", "low"],
                            "title": "Moderation",
                            "type": "string",
                        },
                        "output_compression": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Output Compression",
                        },
                        "output_format": {
                            "anyOf": [
                                {"enum": ["png", "webp", "jpeg"], "type": "string"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Output Format",
                        },
                        "partial_images": {
                            "default": 0,
                            "title": "Partial Images",
                            "type": "integer",
                        },
                        "quality": {
                            "default": "auto",
                            "enum": ["low", "medium", "high", "auto"],
                            "title": "Quality",
                            "type": "string",
                        },
                        "size": {
                            "anyOf": [
                                {
                                    "enum": [
                                        "auto",
                                        "1024x1024",
                                        "1024x1536",
                                        "1536x1024",
                                        "512",
                                        "1K",
                                        "2K",
                                        "4K",
                                    ],
                                    "type": "string",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Size",
                        },
                        "aspect_ratio": {
                            "anyOf": [
                                {
                                    "enum": [
                                        "21:9",
                                        "16:9",
                                        "4:3",
                                        "3:2",
                                        "1:1",
                                        "9:16",
                                        "3:4",
                                        "2:3",
                                        "5:4",
                                        "4:5",
                                    ],
                                    "type": "string",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Aspect Ratio",
                        },
                    },
                    "title": "ImageGenerationTool",
                    "type": "object",
                },
                "KnownModelName": {
                    "enum": [
                        "anthropic:claude-3-5-haiku-20241022",
                        "anthropic:claude-3-5-haiku-latest",
                        "anthropic:claude-3-7-sonnet-20250219",
                        "anthropic:claude-3-7-sonnet-latest",
                        "anthropic:claude-3-haiku-20240307",
                        "anthropic:claude-3-opus-20240229",
                        "anthropic:claude-3-opus-latest",
                        "anthropic:claude-4-opus-20250514",
                        "anthropic:claude-4-sonnet-20250514",
                        "anthropic:claude-haiku-4-5-20251001",
                        "anthropic:claude-haiku-4-5",
                        "anthropic:claude-opus-4-0",
                        "anthropic:claude-opus-4-1-20250805",
                        "anthropic:claude-opus-4-20250514",
                        "anthropic:claude-opus-4-5-20251101",
                        "anthropic:claude-opus-4-5",
                        "anthropic:claude-opus-4-6",
                        "anthropic:claude-sonnet-4-0",
                        "anthropic:claude-sonnet-4-20250514",
                        "anthropic:claude-sonnet-4-5-20250929",
                        "anthropic:claude-sonnet-4-5",
                        "anthropic:claude-sonnet-4-6",
                        "bedrock:amazon.titan-text-express-v1",
                        "bedrock:amazon.titan-text-lite-v1",
                        "bedrock:amazon.titan-tg1-large",
                        "bedrock:anthropic.claude-3-5-haiku-20241022-v1:0",
                        "bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "bedrock:anthropic.claude-3-haiku-20240307-v1:0",
                        "bedrock:anthropic.claude-3-opus-20240229-v1:0",
                        "bedrock:anthropic.claude-3-sonnet-20240229-v1:0",
                        "bedrock:anthropic.claude-haiku-4-5-20251001-v1:0",
                        "bedrock:anthropic.claude-instant-v1",
                        "bedrock:anthropic.claude-opus-4-20250514-v1:0",
                        "bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
                        "bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "bedrock:anthropic.claude-sonnet-4-6",
                        "bedrock:anthropic.claude-v2:1",
                        "bedrock:anthropic.claude-v2",
                        "bedrock:cohere.command-light-text-v14",
                        "bedrock:cohere.command-r-plus-v1:0",
                        "bedrock:cohere.command-r-v1:0",
                        "bedrock:cohere.command-text-v14",
                        "bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0",
                        "bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "bedrock:eu.anthropic.claude-sonnet-4-6",
                        "bedrock:global.anthropic.claude-opus-4-5-20251101-v1:0",
                        "bedrock:meta.llama3-1-405b-instruct-v1:0",
                        "bedrock:meta.llama3-1-70b-instruct-v1:0",
                        "bedrock:meta.llama3-1-8b-instruct-v1:0",
                        "bedrock:meta.llama3-70b-instruct-v1:0",
                        "bedrock:meta.llama3-8b-instruct-v1:0",
                        "bedrock:mistral.mistral-7b-instruct-v0:2",
                        "bedrock:mistral.mistral-large-2402-v1:0",
                        "bedrock:mistral.mistral-large-2407-v1:0",
                        "bedrock:mistral.mixtral-8x7b-instruct-v0:1",
                        "bedrock:us.amazon.nova-2-lite-v1:0",
                        "bedrock:us.amazon.nova-lite-v1:0",
                        "bedrock:us.amazon.nova-micro-v1:0",
                        "bedrock:us.amazon.nova-pro-v1:0",
                        "bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0",
                        "bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "bedrock:us.anthropic.claude-3-haiku-20240307-v1:0",
                        "bedrock:us.anthropic.claude-3-opus-20240229-v1:0",
                        "bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0",
                        "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "bedrock:us.anthropic.claude-opus-4-20250514-v1:0",
                        "bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0",
                        "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "bedrock:us.anthropic.claude-sonnet-4-6",
                        "bedrock:us.meta.llama3-1-70b-instruct-v1:0",
                        "bedrock:us.meta.llama3-1-8b-instruct-v1:0",
                        "bedrock:us.meta.llama3-2-11b-instruct-v1:0",
                        "bedrock:us.meta.llama3-2-1b-instruct-v1:0",
                        "bedrock:us.meta.llama3-2-3b-instruct-v1:0",
                        "bedrock:us.meta.llama3-2-90b-instruct-v1:0",
                        "bedrock:us.meta.llama3-3-70b-instruct-v1:0",
                        "cerebras:gpt-oss-120b",
                        "cerebras:llama-3.3-70b",
                        "cerebras:llama3.1-8b",
                        "cerebras:qwen-3-235b-a22b-instruct-2507",
                        "cerebras:qwen-3-32b",
                        "cerebras:qwen-3-coder-480b",
                        "cerebras:zai-glm-4.6",
                        "cerebras:zai-glm-4.7",
                        "cohere:c4ai-aya-expanse-32b",
                        "cohere:c4ai-aya-expanse-8b",
                        "cohere:command-nightly",
                        "cohere:command-r-08-2024",
                        "cohere:command-r-plus-08-2024",
                        "cohere:command-r7b-12-2024",
                        "deepseek:deepseek-chat",
                        "deepseek:deepseek-reasoner",
                        "gateway/anthropic:claude-3-5-haiku-20241022",
                        "gateway/anthropic:claude-3-5-haiku-latest",
                        "gateway/anthropic:claude-3-7-sonnet-20250219",
                        "gateway/anthropic:claude-3-7-sonnet-latest",
                        "gateway/anthropic:claude-3-haiku-20240307",
                        "gateway/anthropic:claude-3-opus-20240229",
                        "gateway/anthropic:claude-3-opus-latest",
                        "gateway/anthropic:claude-4-opus-20250514",
                        "gateway/anthropic:claude-4-sonnet-20250514",
                        "gateway/anthropic:claude-haiku-4-5-20251001",
                        "gateway/anthropic:claude-haiku-4-5",
                        "gateway/anthropic:claude-opus-4-0",
                        "gateway/anthropic:claude-opus-4-1-20250805",
                        "gateway/anthropic:claude-opus-4-20250514",
                        "gateway/anthropic:claude-opus-4-5-20251101",
                        "gateway/anthropic:claude-opus-4-5",
                        "gateway/anthropic:claude-opus-4-6",
                        "gateway/anthropic:claude-sonnet-4-0",
                        "gateway/anthropic:claude-sonnet-4-20250514",
                        "gateway/anthropic:claude-sonnet-4-5-20250929",
                        "gateway/anthropic:claude-sonnet-4-5",
                        "gateway/anthropic:claude-sonnet-4-6",
                        "gateway/bedrock:amazon.titan-text-express-v1",
                        "gateway/bedrock:amazon.titan-text-lite-v1",
                        "gateway/bedrock:amazon.titan-tg1-large",
                        "gateway/bedrock:anthropic.claude-3-5-haiku-20241022-v1:0",
                        "gateway/bedrock:anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "gateway/bedrock:anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "gateway/bedrock:anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "gateway/bedrock:anthropic.claude-3-haiku-20240307-v1:0",
                        "gateway/bedrock:anthropic.claude-3-opus-20240229-v1:0",
                        "gateway/bedrock:anthropic.claude-3-sonnet-20240229-v1:0",
                        "gateway/bedrock:anthropic.claude-haiku-4-5-20251001-v1:0",
                        "gateway/bedrock:anthropic.claude-instant-v1",
                        "gateway/bedrock:anthropic.claude-opus-4-20250514-v1:0",
                        "gateway/bedrock:anthropic.claude-sonnet-4-20250514-v1:0",
                        "gateway/bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "gateway/bedrock:anthropic.claude-sonnet-4-6",
                        "gateway/bedrock:anthropic.claude-v2:1",
                        "gateway/bedrock:anthropic.claude-v2",
                        "gateway/bedrock:cohere.command-light-text-v14",
                        "gateway/bedrock:cohere.command-r-plus-v1:0",
                        "gateway/bedrock:cohere.command-r-v1:0",
                        "gateway/bedrock:cohere.command-text-v14",
                        "gateway/bedrock:eu.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "gateway/bedrock:eu.anthropic.claude-sonnet-4-20250514-v1:0",
                        "gateway/bedrock:eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "gateway/bedrock:eu.anthropic.claude-sonnet-4-6",
                        "gateway/bedrock:global.anthropic.claude-opus-4-5-20251101-v1:0",
                        "gateway/bedrock:meta.llama3-1-405b-instruct-v1:0",
                        "gateway/bedrock:meta.llama3-1-70b-instruct-v1:0",
                        "gateway/bedrock:meta.llama3-1-8b-instruct-v1:0",
                        "gateway/bedrock:meta.llama3-70b-instruct-v1:0",
                        "gateway/bedrock:meta.llama3-8b-instruct-v1:0",
                        "gateway/bedrock:mistral.mistral-7b-instruct-v0:2",
                        "gateway/bedrock:mistral.mistral-large-2402-v1:0",
                        "gateway/bedrock:mistral.mistral-large-2407-v1:0",
                        "gateway/bedrock:mistral.mixtral-8x7b-instruct-v0:1",
                        "gateway/bedrock:us.amazon.nova-2-lite-v1:0",
                        "gateway/bedrock:us.amazon.nova-lite-v1:0",
                        "gateway/bedrock:us.amazon.nova-micro-v1:0",
                        "gateway/bedrock:us.amazon.nova-pro-v1:0",
                        "gateway/bedrock:us.anthropic.claude-3-5-haiku-20241022-v1:0",
                        "gateway/bedrock:us.anthropic.claude-3-5-sonnet-20240620-v1:0",
                        "gateway/bedrock:us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                        "gateway/bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                        "gateway/bedrock:us.anthropic.claude-3-haiku-20240307-v1:0",
                        "gateway/bedrock:us.anthropic.claude-3-opus-20240229-v1:0",
                        "gateway/bedrock:us.anthropic.claude-3-sonnet-20240229-v1:0",
                        "gateway/bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "gateway/bedrock:us.anthropic.claude-opus-4-20250514-v1:0",
                        "gateway/bedrock:us.anthropic.claude-sonnet-4-20250514-v1:0",
                        "gateway/bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        "gateway/bedrock:us.anthropic.claude-sonnet-4-6",
                        "gateway/bedrock:us.meta.llama3-1-70b-instruct-v1:0",
                        "gateway/bedrock:us.meta.llama3-1-8b-instruct-v1:0",
                        "gateway/bedrock:us.meta.llama3-2-11b-instruct-v1:0",
                        "gateway/bedrock:us.meta.llama3-2-1b-instruct-v1:0",
                        "gateway/bedrock:us.meta.llama3-2-3b-instruct-v1:0",
                        "gateway/bedrock:us.meta.llama3-2-90b-instruct-v1:0",
                        "gateway/bedrock:us.meta.llama3-3-70b-instruct-v1:0",
                        "gateway/google-vertex:gemini-2.0-flash-lite",
                        "gateway/google-vertex:gemini-2.0-flash",
                        "gateway/google-vertex:gemini-2.5-flash-image",
                        "gateway/google-vertex:gemini-2.5-flash-lite-preview-09-2025",
                        "gateway/google-vertex:gemini-2.5-flash-lite",
                        "gateway/google-vertex:gemini-2.5-flash-preview-09-2025",
                        "gateway/google-vertex:gemini-2.5-flash",
                        "gateway/google-vertex:gemini-2.5-pro",
                        "gateway/google-vertex:gemini-3-flash-preview",
                        "gateway/google-vertex:gemini-3-pro-image-preview",
                        "gateway/google-vertex:gemini-3-pro-preview",
                        "gateway/google-vertex:gemini-3.1-flash-image-preview",
                        "gateway/google-vertex:gemini-3.1-flash-lite-preview",
                        "gateway/google-vertex:gemini-3.1-pro-preview",
                        "gateway/google-vertex:gemini-flash-latest",
                        "gateway/google-vertex:gemini-flash-lite-latest",
                        "gateway/groq:llama-3.1-8b-instant",
                        "gateway/groq:llama-3.3-70b-versatile",
                        "gateway/groq:meta-llama/llama-guard-4-12b",
                        "gateway/groq:openai/gpt-oss-120b",
                        "gateway/groq:openai/gpt-oss-20b",
                        "gateway/groq:whisper-large-v3",
                        "gateway/groq:whisper-large-v3-turbo",
                        "gateway/groq:meta-llama/llama-4-maverick-17b-128e-instruct",
                        "gateway/groq:meta-llama/llama-4-scout-17b-16e-instruct",
                        "gateway/groq:meta-llama/llama-prompt-guard-2-22m",
                        "gateway/groq:meta-llama/llama-prompt-guard-2-86m",
                        "gateway/groq:moonshotai/kimi-k2-instruct-0905",
                        "gateway/groq:openai/gpt-oss-safeguard-20b",
                        "gateway/groq:playai-tts",
                        "gateway/groq:playai-tts-arabic",
                        "gateway/groq:qwen/qwen-3-32b",
                        "gateway/openai:computer-use-preview-2025-03-11",
                        "gateway/openai:computer-use-preview",
                        "gateway/openai:gpt-3.5-turbo-0125",
                        "gateway/openai:gpt-3.5-turbo-0301",
                        "gateway/openai:gpt-3.5-turbo-0613",
                        "gateway/openai:gpt-3.5-turbo-1106",
                        "gateway/openai:gpt-3.5-turbo-16k-0613",
                        "gateway/openai:gpt-3.5-turbo-16k",
                        "gateway/openai:gpt-3.5-turbo",
                        "gateway/openai:gpt-4-0314",
                        "gateway/openai:gpt-4-0613",
                        "gateway/openai:gpt-4-turbo-2024-04-09",
                        "gateway/openai:gpt-4-turbo",
                        "gateway/openai:gpt-4.1-2025-04-14",
                        "gateway/openai:gpt-4.1-mini-2025-04-14",
                        "gateway/openai:gpt-4.1-mini",
                        "gateway/openai:gpt-4.1-nano-2025-04-14",
                        "gateway/openai:gpt-4.1-nano",
                        "gateway/openai:gpt-4.1",
                        "gateway/openai:gpt-4",
                        "gateway/openai:gpt-4o-2024-05-13",
                        "gateway/openai:gpt-4o-2024-08-06",
                        "gateway/openai:gpt-4o-2024-11-20",
                        "gateway/openai:gpt-4o-audio-preview-2024-12-17",
                        "gateway/openai:gpt-4o-audio-preview-2025-06-03",
                        "gateway/openai:gpt-4o-audio-preview",
                        "gateway/openai:gpt-4o-mini-2024-07-18",
                        "gateway/openai:gpt-4o-mini-audio-preview-2024-12-17",
                        "gateway/openai:gpt-4o-mini-audio-preview",
                        "gateway/openai:gpt-4o-mini-search-preview-2025-03-11",
                        "gateway/openai:gpt-4o-mini-search-preview",
                        "gateway/openai:gpt-4o-mini",
                        "gateway/openai:gpt-4o-search-preview-2025-03-11",
                        "gateway/openai:gpt-4o-search-preview",
                        "gateway/openai:gpt-4o",
                        "gateway/openai:gpt-5-2025-08-07",
                        "gateway/openai:gpt-5-chat-latest",
                        "gateway/openai:gpt-5-codex",
                        "gateway/openai:gpt-5-mini-2025-08-07",
                        "gateway/openai:gpt-5-mini",
                        "gateway/openai:gpt-5-nano-2025-08-07",
                        "gateway/openai:gpt-5-nano",
                        "gateway/openai:gpt-5-pro-2025-10-06",
                        "gateway/openai:gpt-5-pro",
                        "gateway/openai:gpt-5.1-2025-11-13",
                        "gateway/openai:gpt-5.1-chat-latest",
                        "gateway/openai:gpt-5.1-codex-max",
                        "gateway/openai:gpt-5.1-codex",
                        "gateway/openai:gpt-5.1",
                        "gateway/openai:gpt-5.2-2025-12-11",
                        "gateway/openai:gpt-5.2-chat-latest",
                        "gateway/openai:gpt-5.2-pro-2025-12-11",
                        "gateway/openai:gpt-5.2-pro",
                        "gateway/openai:gpt-5.2",
                        "gateway/openai:gpt-5.3-chat-latest",
                        "gateway/openai:gpt-5.4-mini-2026-03-17",
                        "gateway/openai:gpt-5.4-mini",
                        "gateway/openai:gpt-5.4-nano-2026-03-17",
                        "gateway/openai:gpt-5.4-nano",
                        "gateway/openai:gpt-5.4",
                        "gateway/openai:gpt-5",
                        "gateway/openai:o1-2024-12-17",
                        "gateway/openai:o1-pro-2025-03-19",
                        "gateway/openai:o1-pro",
                        "gateway/openai:o1",
                        "gateway/openai:o3-2025-04-16",
                        "gateway/openai:o3-deep-research-2025-06-26",
                        "gateway/openai:o3-deep-research",
                        "gateway/openai:o3-mini-2025-01-31",
                        "gateway/openai:o3-mini",
                        "gateway/openai:o3-pro-2025-06-10",
                        "gateway/openai:o3-pro",
                        "gateway/openai:o3",
                        "gateway/openai:o4-mini-2025-04-16",
                        "gateway/openai:o4-mini-deep-research-2025-06-26",
                        "gateway/openai:o4-mini-deep-research",
                        "gateway/openai:o4-mini",
                        "google-gla:gemini-2.0-flash-lite",
                        "google-gla:gemini-2.0-flash",
                        "google-gla:gemini-2.5-flash-image",
                        "google-gla:gemini-2.5-flash-lite-preview-09-2025",
                        "google-gla:gemini-2.5-flash-lite",
                        "google-gla:gemini-2.5-flash-preview-09-2025",
                        "google-gla:gemini-2.5-flash",
                        "google-gla:gemini-2.5-pro",
                        "google-gla:gemini-3-flash-preview",
                        "google-gla:gemini-3-pro-image-preview",
                        "google-gla:gemini-3-pro-preview",
                        "google-gla:gemini-3.1-flash-image-preview",
                        "google-gla:gemini-3.1-flash-lite-preview",
                        "google-gla:gemini-3.1-pro-preview",
                        "google-gla:gemini-flash-latest",
                        "google-gla:gemini-flash-lite-latest",
                        "google-vertex:gemini-2.0-flash-lite",
                        "google-vertex:gemini-2.0-flash",
                        "google-vertex:gemini-2.5-flash-image",
                        "google-vertex:gemini-2.5-flash-lite-preview-09-2025",
                        "google-vertex:gemini-2.5-flash-lite",
                        "google-vertex:gemini-2.5-flash-preview-09-2025",
                        "google-vertex:gemini-2.5-flash",
                        "google-vertex:gemini-2.5-pro",
                        "google-vertex:gemini-3-flash-preview",
                        "google-vertex:gemini-3-pro-image-preview",
                        "google-vertex:gemini-3-pro-preview",
                        "google-vertex:gemini-3.1-flash-image-preview",
                        "google-vertex:gemini-3.1-flash-lite-preview",
                        "google-vertex:gemini-3.1-pro-preview",
                        "google-vertex:gemini-flash-latest",
                        "google-vertex:gemini-flash-lite-latest",
                        "grok:grok-2-image-1212",
                        "grok:grok-2-vision-1212",
                        "grok:grok-3-fast",
                        "grok:grok-3-mini-fast",
                        "grok:grok-3-mini",
                        "grok:grok-3",
                        "grok:grok-4-0709",
                        "grok:grok-4-latest",
                        "grok:grok-4-1-fast-non-reasoning",
                        "grok:grok-4-1-fast-reasoning",
                        "grok:grok-4-1-fast",
                        "grok:grok-4-fast-non-reasoning",
                        "grok:grok-4-fast-reasoning",
                        "grok:grok-4-fast",
                        "grok:grok-4",
                        "grok:grok-code-fast-1",
                        "xai:grok-3",
                        "xai:grok-3-fast",
                        "xai:grok-3-fast-latest",
                        "xai:grok-3-latest",
                        "xai:grok-3-mini",
                        "xai:grok-3-mini-fast",
                        "xai:grok-3-mini-fast-latest",
                        "xai:grok-4",
                        "xai:grok-4-0709",
                        "xai:grok-4-1-fast",
                        "xai:grok-4-1-fast-non-reasoning",
                        "xai:grok-4-1-fast-non-reasoning-latest",
                        "xai:grok-4-1-fast-reasoning",
                        "xai:grok-4-1-fast-reasoning-latest",
                        "xai:grok-4-fast",
                        "xai:grok-4-fast-non-reasoning",
                        "xai:grok-4-fast-non-reasoning-latest",
                        "xai:grok-4-fast-reasoning",
                        "xai:grok-4-fast-reasoning-latest",
                        "xai:grok-4-latest",
                        "xai:grok-code-fast-1",
                        "groq:llama-3.1-8b-instant",
                        "groq:llama-3.3-70b-versatile",
                        "groq:meta-llama/llama-guard-4-12b",
                        "groq:openai/gpt-oss-120b",
                        "groq:openai/gpt-oss-20b",
                        "groq:whisper-large-v3",
                        "groq:whisper-large-v3-turbo",
                        "groq:meta-llama/llama-4-maverick-17b-128e-instruct",
                        "groq:meta-llama/llama-4-scout-17b-16e-instruct",
                        "groq:meta-llama/llama-prompt-guard-2-22m",
                        "groq:meta-llama/llama-prompt-guard-2-86m",
                        "groq:moonshotai/kimi-k2-instruct-0905",
                        "groq:openai/gpt-oss-safeguard-20b",
                        "groq:playai-tts",
                        "groq:playai-tts-arabic",
                        "groq:qwen/qwen-3-32b",
                        "heroku:claude-3-5-haiku",
                        "heroku:claude-3-5-sonnet-latest",
                        "heroku:claude-3-7-sonnet",
                        "heroku:claude-3-haiku",
                        "heroku:claude-4-5-haiku",
                        "heroku:claude-4-5-sonnet",
                        "heroku:claude-4-sonnet",
                        "heroku:claude-opus-4-5",
                        "heroku:gpt-oss-120b",
                        "heroku:kimi-k2-thinking",
                        "heroku:minimax-m2",
                        "heroku:qwen3-235b",
                        "heroku:qwen3-coder-480b",
                        "heroku:nova-2-lite",
                        "heroku:nova-lite",
                        "heroku:nova-pro",
                        "huggingface:deepseek-ai/DeepSeek-R1",
                        "huggingface:meta-llama/Llama-3.3-70B-Instruct",
                        "huggingface:meta-llama/Llama-4-Maverick-17B-128E-Instruct",
                        "huggingface:meta-llama/Llama-4-Scout-17B-16E-Instruct",
                        "huggingface:Qwen/Qwen2.5-72B-Instruct",
                        "huggingface:Qwen/Qwen3-235B-A22B",
                        "huggingface:Qwen/Qwen3-32B",
                        "huggingface:Qwen/QwQ-32B",
                        "mistral:codestral-latest",
                        "mistral:mistral-large-latest",
                        "mistral:mistral-moderation-latest",
                        "mistral:mistral-small-latest",
                        "moonshotai:kimi-k2-0711-preview",
                        "moonshotai:kimi-latest",
                        "moonshotai:kimi-thinking-preview",
                        "moonshotai:moonshot-v1-128k-vision-preview",
                        "moonshotai:moonshot-v1-128k",
                        "moonshotai:moonshot-v1-32k-vision-preview",
                        "moonshotai:moonshot-v1-32k",
                        "moonshotai:moonshot-v1-8k-vision-preview",
                        "moonshotai:moonshot-v1-8k",
                        "openai:computer-use-preview-2025-03-11",
                        "openai:computer-use-preview",
                        "openai:gpt-3.5-turbo-0125",
                        "openai:gpt-3.5-turbo-0301",
                        "openai:gpt-3.5-turbo-0613",
                        "openai:gpt-3.5-turbo-1106",
                        "openai:gpt-3.5-turbo-16k-0613",
                        "openai:gpt-3.5-turbo-16k",
                        "openai:gpt-3.5-turbo",
                        "openai:gpt-4-0314",
                        "openai:gpt-4-0613",
                        "openai:gpt-4-turbo-2024-04-09",
                        "openai:gpt-4-turbo",
                        "openai:gpt-4.1-2025-04-14",
                        "openai:gpt-4.1-mini-2025-04-14",
                        "openai:gpt-4.1-mini",
                        "openai:gpt-4.1-nano-2025-04-14",
                        "openai:gpt-4.1-nano",
                        "openai:gpt-4.1",
                        "openai:gpt-4",
                        "openai:gpt-4o-2024-05-13",
                        "openai:gpt-4o-2024-08-06",
                        "openai:gpt-4o-2024-11-20",
                        "openai:gpt-4o-audio-preview-2024-12-17",
                        "openai:gpt-4o-audio-preview-2025-06-03",
                        "openai:gpt-4o-audio-preview",
                        "openai:gpt-4o-mini-2024-07-18",
                        "openai:gpt-4o-mini-audio-preview-2024-12-17",
                        "openai:gpt-4o-mini-audio-preview",
                        "openai:gpt-4o-mini-search-preview-2025-03-11",
                        "openai:gpt-4o-mini-search-preview",
                        "openai:gpt-4o-mini",
                        "openai:gpt-4o-search-preview-2025-03-11",
                        "openai:gpt-4o-search-preview",
                        "openai:gpt-4o",
                        "openai:gpt-5-2025-08-07",
                        "openai:gpt-5-chat-latest",
                        "openai:gpt-5-codex",
                        "openai:gpt-5-mini-2025-08-07",
                        "openai:gpt-5-mini",
                        "openai:gpt-5-nano-2025-08-07",
                        "openai:gpt-5-nano",
                        "openai:gpt-5-pro-2025-10-06",
                        "openai:gpt-5-pro",
                        "openai:gpt-5.1-2025-11-13",
                        "openai:gpt-5.1-chat-latest",
                        "openai:gpt-5.1-codex-max",
                        "openai:gpt-5.1-codex",
                        "openai:gpt-5.1",
                        "openai:gpt-5.2-2025-12-11",
                        "openai:gpt-5.2-chat-latest",
                        "openai:gpt-5.2-pro-2025-12-11",
                        "openai:gpt-5.2-pro",
                        "openai:gpt-5.2",
                        "openai:gpt-5.3-chat-latest",
                        "openai:gpt-5.4-mini-2026-03-17",
                        "openai:gpt-5.4-mini",
                        "openai:gpt-5.4-nano-2026-03-17",
                        "openai:gpt-5.4-nano",
                        "openai:gpt-5.4",
                        "openai:gpt-5",
                        "openai:o1-2024-12-17",
                        "openai:o1-pro-2025-03-19",
                        "openai:o1-pro",
                        "openai:o1",
                        "openai:o3-2025-04-16",
                        "openai:o3-deep-research-2025-06-26",
                        "openai:o3-deep-research",
                        "openai:o3-mini-2025-01-31",
                        "openai:o3-mini",
                        "openai:o3-pro-2025-06-10",
                        "openai:o3-pro",
                        "openai:o3",
                        "openai:o4-mini-2025-04-16",
                        "openai:o4-mini-deep-research-2025-06-26",
                        "openai:o4-mini-deep-research",
                        "openai:o4-mini",
                        "test",
                    ],
                    "type": "string",
                },
                "MCPServerTool": {
                    "properties": {
                        "kind": {
                            "default": "mcp_server",
                            "title": "Kind",
                            "type": "string",
                        },
                        "id": {"title": "Id", "type": "string"},
                        "url": {"title": "Url", "type": "string"},
                        "authorization_token": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Authorization Token",
                        },
                        "description": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "default": None,
                            "title": "Description",
                        },
                        "allowed_tools": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Allowed Tools",
                        },
                        "headers": {
                            "anyOf": [
                                {
                                    "additionalProperties": {"type": "string"},
                                    "type": "object",
                                },
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Headers",
                        },
                    },
                    "required": ["id", "url"],
                    "title": "MCPServerTool",
                    "type": "object",
                },
                "MemoryTool": {
                    "properties": {
                        "kind": {"default": "memory", "title": "Kind", "type": "string"}
                    },
                    "title": "MemoryTool",
                    "type": "object",
                },
                "ModelSettings": {
                    "description": """\
Settings to configure an LLM.

Here we include only settings which apply to multiple models / model providers,
though not all of these settings are supported by all models.\
""",
                    "properties": {
                        "max_tokens": {"title": "Max Tokens", "type": "integer"},
                        "temperature": {"title": "Temperature", "type": "number"},
                        "top_p": {"title": "Top P", "type": "number"},
                        "timeout": {"title": "Timeout", "type": "number"},
                        "parallel_tool_calls": {
                            "title": "Parallel Tool Calls",
                            "type": "boolean",
                        },
                        "seed": {"title": "Seed", "type": "integer"},
                        "presence_penalty": {
                            "title": "Presence Penalty",
                            "type": "number",
                        },
                        "frequency_penalty": {
                            "title": "Frequency Penalty",
                            "type": "number",
                        },
                        "logit_bias": {
                            "additionalProperties": {"type": "integer"},
                            "title": "Logit Bias",
                            "type": "object",
                        },
                        "stop_sequences": {
                            "items": {"type": "string"},
                            "title": "Stop Sequences",
                            "type": "array",
                        },
                        "extra_headers": {
                            "additionalProperties": {"type": "string"},
                            "title": "Extra Headers",
                            "type": "object",
                        },
                        "thinking": {
                            "anyOf": [
                                {"type": "boolean"},
                                {
                                    "enum": [
                                        "minimal",
                                        "low",
                                        "medium",
                                        "high",
                                        "xhigh",
                                    ],
                                    "type": "string",
                                },
                            ],
                            "title": "Thinking",
                        },
                        "extra_body": {"title": "Extra Body"},
                    },
                    "title": "ModelSettings",
                    "type": "object",
                },
                "UrlContextTool": {
                    "deprecated": True,
                    "properties": {
                        "kind": {
                            "default": "url_context",
                            "title": "Kind",
                            "type": "string",
                        },
                        "max_uses": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Max Uses",
                        },
                        "allowed_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Allowed Domains",
                        },
                        "blocked_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Blocked Domains",
                        },
                        "enable_citations": {
                            "default": False,
                            "title": "Enable Citations",
                            "type": "boolean",
                        },
                        "max_content_tokens": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Max Content Tokens",
                        },
                    },
                    "title": "UrlContextTool",
                    "type": "object",
                },
                "WebFetchTool": {
                    "properties": {
                        "kind": {
                            "default": "web_fetch",
                            "title": "Kind",
                            "type": "string",
                        },
                        "max_uses": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Max Uses",
                        },
                        "allowed_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Allowed Domains",
                        },
                        "blocked_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Blocked Domains",
                        },
                        "enable_citations": {
                            "default": False,
                            "title": "Enable Citations",
                            "type": "boolean",
                        },
                        "max_content_tokens": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Max Content Tokens",
                        },
                    },
                    "title": "WebFetchTool",
                    "type": "object",
                },
                "WebSearchTool": {
                    "properties": {
                        "kind": {
                            "default": "web_search",
                            "title": "Kind",
                            "type": "string",
                        },
                        "search_context_size": {
                            "default": "medium",
                            "enum": ["low", "medium", "high"],
                            "title": "Search Context Size",
                            "type": "string",
                        },
                        "user_location": {
                            "anyOf": [
                                {"$ref": "#/$defs/WebSearchUserLocation"},
                                {"type": "null"},
                            ],
                            "default": None,
                        },
                        "blocked_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Blocked Domains",
                        },
                        "allowed_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "default": None,
                            "title": "Allowed Domains",
                        },
                        "max_uses": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "default": None,
                            "title": "Max Uses",
                        },
                    },
                    "title": "WebSearchTool",
                    "type": "object",
                },
                "WebSearchUserLocation": {
                    "additionalProperties": False,
                    "description": """\
Allows you to localize search results based on a user's location.

Supported by:

* Anthropic
* OpenAI Responses\
""",
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                        "region": {"title": "Region", "type": "string"},
                        "timezone": {"title": "Timezone", "type": "string"},
                    },
                    "title": "WebSearchUserLocation",
                    "type": "object",
                },
                "short_spec_BuiltinTool": {
                    "additionalProperties": False,
                    "properties": {
                        "BuiltinTool": {
                            "anyOf": [
                                {
                                    "oneOf": [
                                        {"$ref": "#/$defs/WebSearchTool"},
                                        {"$ref": "#/$defs/CodeExecutionTool"},
                                        {"$ref": "#/$defs/WebFetchTool"},
                                        {"$ref": "#/$defs/UrlContextTool"},
                                        {"$ref": "#/$defs/ImageGenerationTool"},
                                        {"$ref": "#/$defs/MemoryTool"},
                                        {"$ref": "#/$defs/MCPServerTool"},
                                        {"$ref": "#/$defs/FileSearchTool"},
                                    ]
                                },
                                {"type": "null"},
                            ],
                            "title": "Builtintool",
                        }
                    },
                    "title": "short_spec_BuiltinTool",
                    "type": "object",
                },
                "short_spec_MCP": {
                    "additionalProperties": False,
                    "properties": {"MCP": {"title": "Mcp", "type": "string"}},
                    "required": ["MCP"],
                    "title": "short_spec_MCP",
                    "type": "object",
                },
                "short_spec_Thinking": {
                    "additionalProperties": False,
                    "properties": {
                        "Thinking": {
                            "anyOf": [
                                {"type": "boolean"},
                                {
                                    "enum": [
                                        "minimal",
                                        "low",
                                        "medium",
                                        "high",
                                        "xhigh",
                                    ],
                                    "type": "string",
                                },
                            ],
                            "title": "Thinking",
                        }
                    },
                    "title": "short_spec_Thinking",
                    "type": "object",
                },
                "spec_ImageGeneration": {
                    "additionalProperties": False,
                    "properties": {
                        "ImageGeneration": {
                            "$ref": "#/$defs/spec_params_ImageGeneration"
                        }
                    },
                    "required": ["ImageGeneration"],
                    "title": "spec_ImageGeneration",
                    "type": "object",
                },
                "spec_MCP": {
                    "additionalProperties": False,
                    "properties": {"MCP": {"$ref": "#/$defs/spec_params_MCP"}},
                    "required": ["MCP"],
                    "title": "spec_MCP",
                    "type": "object",
                },
                "spec_PrefixTools": {
                    "additionalProperties": False,
                    "properties": {
                        "PrefixTools": {"$ref": "#/$defs/spec_params_PrefixTools"}
                    },
                    "required": ["PrefixTools"],
                    "title": "spec_PrefixTools",
                    "type": "object",
                },
                "spec_WebFetch": {
                    "additionalProperties": False,
                    "properties": {
                        "WebFetch": {"$ref": "#/$defs/spec_params_WebFetch"}
                    },
                    "required": ["WebFetch"],
                    "title": "spec_WebFetch",
                    "type": "object",
                },
                "spec_WebSearch": {
                    "additionalProperties": False,
                    "properties": {
                        "WebSearch": {"$ref": "#/$defs/spec_params_WebSearch"}
                    },
                    "required": ["WebSearch"],
                    "title": "spec_WebSearch",
                    "type": "object",
                },
                "spec_params_ImageGeneration": {
                    "additionalProperties": False,
                    "properties": {
                        "builtin": {
                            "anyOf": [
                                {"$ref": "#/$defs/ImageGenerationTool"},
                                {"type": "boolean"},
                            ],
                            "title": "Builtin",
                        },
                        "local": {
                            "anyOf": [
                                {"const": False, "type": "boolean"},
                                {"type": "null"},
                            ],
                            "title": "Local",
                        },
                        "fallback_model": {
                            "anyOf": [
                                {"$ref": "#/$defs/KnownModelName"},
                                {"type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Fallback Model",
                        },
                        "background": {
                            "anyOf": [
                                {
                                    "enum": ["transparent", "opaque", "auto"],
                                    "type": "string",
                                },
                                {"type": "null"},
                            ],
                            "title": "Background",
                        },
                        "input_fidelity": {
                            "anyOf": [
                                {"enum": ["high", "low"], "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Input Fidelity",
                        },
                        "moderation": {
                            "anyOf": [
                                {"enum": ["auto", "low"], "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Moderation",
                        },
                        "output_compression": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "title": "Output Compression",
                        },
                        "output_format": {
                            "anyOf": [
                                {"enum": ["png", "webp", "jpeg"], "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Output Format",
                        },
                        "quality": {
                            "anyOf": [
                                {
                                    "enum": ["low", "medium", "high", "auto"],
                                    "type": "string",
                                },
                                {"type": "null"},
                            ],
                            "title": "Quality",
                        },
                        "size": {
                            "anyOf": [
                                {
                                    "enum": [
                                        "auto",
                                        "1024x1024",
                                        "1024x1536",
                                        "1536x1024",
                                        "512",
                                        "1K",
                                        "2K",
                                        "4K",
                                    ],
                                    "type": "string",
                                },
                                {"type": "null"},
                            ],
                            "title": "Size",
                        },
                        "aspect_ratio": {
                            "anyOf": [
                                {
                                    "enum": [
                                        "21:9",
                                        "16:9",
                                        "4:3",
                                        "3:2",
                                        "1:1",
                                        "9:16",
                                        "3:4",
                                        "2:3",
                                        "5:4",
                                        "4:5",
                                    ],
                                    "type": "string",
                                },
                                {"type": "null"},
                            ],
                            "title": "Aspect Ratio",
                        },
                    },
                    "title": "spec_params_ImageGeneration",
                    "type": "object",
                },
                "spec_params_MCP": {
                    "additionalProperties": False,
                    "properties": {
                        "url": {"title": "Url", "type": "string"},
                        "builtin": {
                            "anyOf": [
                                {"$ref": "#/$defs/MCPServerTool"},
                                {"type": "boolean"},
                            ],
                            "title": "Builtin",
                        },
                        "local": {
                            "anyOf": [
                                {"const": False, "type": "boolean"},
                                {"type": "null"},
                            ],
                            "title": "Local",
                        },
                        "id": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Id",
                        },
                        "authorization_token": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Authorization Token",
                        },
                        "headers": {
                            "anyOf": [
                                {
                                    "additionalProperties": {"type": "string"},
                                    "type": "object",
                                },
                                {"type": "null"},
                            ],
                            "title": "Headers",
                        },
                        "allowed_tools": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "title": "Allowed Tools",
                        },
                        "description": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "title": "Description",
                        },
                    },
                    "required": ["url"],
                    "title": "spec_params_MCP",
                    "type": "object",
                },
                "spec_params_PrefixTools": {
                    "additionalProperties": False,
                    "properties": {
                        "prefix": {"title": "Prefix", "type": "string"},
                        "capability": {
                            "anyOf": [
                                {"const": "BuiltinTool", "type": "string"},
                                {"$ref": "#/$defs/short_spec_BuiltinTool"},
                                {"const": "ImageGeneration", "type": "string"},
                                {"$ref": "#/$defs/spec_ImageGeneration"},
                                {"$ref": "#/$defs/short_spec_MCP"},
                                {"$ref": "#/$defs/spec_MCP"},
                                {"$ref": "#/$defs/spec_PrefixTools"},
                                {"const": "Thinking", "type": "string"},
                                {"$ref": "#/$defs/short_spec_Thinking"},
                                {"const": "WebFetch", "type": "string"},
                                {"$ref": "#/$defs/spec_WebFetch"},
                                {"const": "WebSearch", "type": "string"},
                                {"$ref": "#/$defs/spec_WebSearch"},
                            ]
                        },
                    },
                    "required": ["prefix", "capability"],
                    "title": "spec_params_PrefixTools",
                    "type": "object",
                },
                "spec_params_WebFetch": {
                    "additionalProperties": False,
                    "properties": {
                        "builtin": {
                            "anyOf": [
                                {"$ref": "#/$defs/WebFetchTool"},
                                {"type": "boolean"},
                            ],
                            "title": "Builtin",
                        },
                        "local": {
                            "anyOf": [
                                {"const": False, "type": "boolean"},
                                {"type": "null"},
                            ],
                            "title": "Local",
                        },
                        "allowed_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "title": "Allowed Domains",
                        },
                        "blocked_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "title": "Blocked Domains",
                        },
                        "max_uses": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "title": "Max Uses",
                        },
                        "enable_citations": {
                            "anyOf": [{"type": "boolean"}, {"type": "null"}],
                            "title": "Enable Citations",
                        },
                        "max_content_tokens": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "title": "Max Content Tokens",
                        },
                    },
                    "title": "spec_params_WebFetch",
                    "type": "object",
                },
                "spec_params_WebSearch": {
                    "additionalProperties": False,
                    "properties": {
                        "builtin": {
                            "anyOf": [
                                {"$ref": "#/$defs/WebSearchTool"},
                                {"type": "boolean"},
                            ],
                            "title": "Builtin",
                        },
                        "local": {
                            "anyOf": [
                                {"const": False, "type": "boolean"},
                                {"type": "null"},
                            ],
                            "title": "Local",
                        },
                        "search_context_size": {
                            "anyOf": [
                                {"enum": ["low", "medium", "high"], "type": "string"},
                                {"type": "null"},
                            ],
                            "title": "Search Context Size",
                        },
                        "user_location": {
                            "anyOf": [
                                {"$ref": "#/$defs/WebSearchUserLocation"},
                                {"type": "null"},
                            ]
                        },
                        "blocked_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "title": "Blocked Domains",
                        },
                        "allowed_domains": {
                            "anyOf": [
                                {"items": {"type": "string"}, "type": "array"},
                                {"type": "null"},
                            ],
                            "title": "Allowed Domains",
                        },
                        "max_uses": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}],
                            "title": "Max Uses",
                        },
                    },
                    "title": "spec_params_WebSearch",
                    "type": "object",
                },
            },
            "additionalProperties": False,
            "properties": {
                "model": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Model",
                },
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Name",
                },
                "description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Description",
                },
                "instructions": {
                    "anyOf": [
                        {"type": "string"},
                        {"items": {"type": "string"}, "type": "array"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Instructions",
                },
                "deps_schema": {
                    "anyOf": [
                        {"additionalProperties": True, "type": "object"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Deps Schema",
                },
                "output_schema": {
                    "anyOf": [
                        {"additionalProperties": True, "type": "object"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Output Schema",
                },
                "model_settings": {
                    "anyOf": [{"$ref": "#/$defs/ModelSettings"}, {"type": "null"}],
                    "default": None,
                },
                "retries": {"default": 1, "title": "Retries", "type": "integer"},
                "output_retries": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "title": "Output Retries",
                },
                "end_strategy": {
                    "default": "early",
                    "enum": ["early", "exhaustive"],
                    "title": "End Strategy",
                    "type": "string",
                },
                "tool_timeout": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "default": None,
                    "title": "Tool Timeout",
                },
                "instrument": {
                    "anyOf": [{"type": "boolean"}, {"type": "null"}],
                    "default": None,
                    "title": "Instrument",
                },
                "metadata": {
                    "anyOf": [
                        {"additionalProperties": True, "type": "object"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Metadata",
                },
                "capabilities": {
                    "default": [],
                    "items": {
                        "anyOf": [
                            {"const": "BuiltinTool", "type": "string"},
                            {"$ref": "#/$defs/short_spec_BuiltinTool"},
                            {"const": "ImageGeneration", "type": "string"},
                            {"$ref": "#/$defs/spec_ImageGeneration"},
                            {"$ref": "#/$defs/short_spec_MCP"},
                            {"$ref": "#/$defs/spec_MCP"},
                            {"$ref": "#/$defs/spec_PrefixTools"},
                            {"const": "Thinking", "type": "string"},
                            {"$ref": "#/$defs/short_spec_Thinking"},
                            {"const": "WebFetch", "type": "string"},
                            {"$ref": "#/$defs/spec_WebFetch"},
                            {"const": "WebSearch", "type": "string"},
                            {"$ref": "#/$defs/spec_WebSearch"},
                        ]
                    },
                    "title": "Capabilities",
                    "type": "array",
                },
                "$schema": {"type": "string"},
            },
            "title": "AgentSpec",
            "type": "object",
        }
    )


def test_model_json_schema_with_custom_capabilities():
    schema = AgentSpec.model_json_schema_with_capabilities(
        custom_capability_types=[CustomCapability],
    )

    any_of = schema["properties"]["capabilities"]["items"]["anyOf"]

    capability_names: set[str] = set()
    for entry in any_of:
        if "const" in entry:
            capability_names.add(entry["const"])
        elif "$ref" in entry:  # pragma: no branch
            ref = entry["$ref"]
            ref_name = ref.rsplit("/", 1)[-1]
            for prefix in ("spec_", "short_spec_"):
                if ref_name.startswith(prefix):
                    capability_names.add(ref_name[len(prefix) :])

    assert "CustomCapability" in capability_names
    # Default capabilities should still be present
    assert "WebSearch" in capability_names


def test_model_json_schema_filters_non_serializable_params():
    """Custom capabilities with non-serializable __init__ params get filtered in schema."""
    schema = AgentSpec.model_json_schema_with_capabilities(
        custom_capability_types=[CapabilityWithCallbackParam],
    )
    any_of = schema["properties"]["capabilities"]["items"]["anyOf"]

    # String form: all remaining params are optional
    has_string_form = any(
        e.get("const") == "CapabilityWithCallbackParam" for e in any_of
    )
    assert has_string_form

    # Long form: max_retries and verbose survive; on_error (purely Callable) is filtered out
    spec_ref = next(
        (
            e
            for e in any_of
            if "$ref" in e and "spec_CapabilityWithCallbackParam" in e["$ref"]
        ),
        None,
    )
    assert spec_ref is not None
    params_def = schema["$defs"]["spec_params_CapabilityWithCallbackParam"]
    assert "max_retries" in params_def["properties"]
    assert "verbose" in params_def["properties"]
    # on_error should not appear — purely Callable, entirely filtered out
    assert "on_error" not in params_def["properties"]
    # hooks should not appear — union of only non-serializable types, entirely filtered out
    assert "hooks" not in params_def["properties"]
    # verbose should be boolean only (Callable member was stripped from the union)
    assert params_def["properties"]["verbose"] == {
        "title": "Verbose",
        "type": "boolean",
    }


def test_agent_spec_schema_field_parity():
    """Ensure the schema model's fields stay in sync with AgentSpec."""
    schema = AgentSpec.model_json_schema_with_capabilities()
    schema_fields = set(schema["properties"].keys())

    # Map AgentSpec field names to their JSON schema names (using aliases)
    spec_fields: set[str] = set()
    for name, field_info in AgentSpec.model_fields.items():
        alias = field_info.alias
        spec_fields.add(alias if isinstance(alias, str) else name)

    assert schema_fields == spec_fields


def test_builtin_tools_param_wrapped_as_capabilities():
    """The builtin_tools parameter items are wrapped in BuiltinTool capabilities."""
    agent = Agent("test", builtin_tools=[WebSearchTool(), CodeExecutionTool()])
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
            "model": "test",
            "capabilities": [
                {"BuiltinTool": {"kind": "web_search"}},
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
            "model": "test",
            "capabilities": [
                {"BuiltinTool": {"kind": "web_search", "search_context_size": "high"}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, BuiltinToolCap)]
    assert len(builtin_caps) == 1
    tool = builtin_caps[0].tool
    assert isinstance(tool, WebSearchTool)
    assert tool.search_context_size == "high"


def test_agent_from_spec_builtin_tool_explicit_form():
    """BuiltinTool spec supports the explicit {tool: ...} form."""
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [
                {"BuiltinTool": {"tool": {"kind": "code_execution"}}},
            ],
        }
    )
    children = agent._root_capability.capabilities  # pyright: ignore[reportPrivateUsage]
    builtin_caps = [c for c in children if isinstance(c, BuiltinToolCap)]
    assert len(builtin_caps) == 1
    assert isinstance(builtin_caps[0].tool, CodeExecutionTool)


def test_save_schema(tmp_path: str):
    schema_path = Path(tmp_path) / "agent_spec.schema.json"
    AgentSpec._save_schema(schema_path)  # pyright: ignore[reportPrivateUsage]

    assert schema_path.exists()
    import json

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["type"] == "object"
    assert "model" in schema["properties"]
    assert "capabilities" in schema["properties"]

    # Calling again should not rewrite if content matches
    mtime = schema_path.stat().st_mtime
    AgentSpec._save_schema(schema_path)  # pyright: ignore[reportPrivateUsage]
    assert schema_path.stat().st_mtime == mtime


def test_from_file_yaml(tmp_path: str):
    spec_path = Path(tmp_path) / "agent.yaml"
    spec_path.write_text(
        "model: test\nname: my-agent\ninstructions: Be helpful\n", encoding="utf-8"
    )
    spec = AgentSpec.from_file(spec_path)
    assert spec.model == "test"
    assert spec.name == "my-agent"
    assert spec.instructions == "Be helpful"


def test_from_file_json(tmp_path: str):
    spec_path = Path(tmp_path) / "agent.json"
    spec_path.write_text('{"model": "test", "name": "my-agent"}', encoding="utf-8")
    spec = AgentSpec.from_file(spec_path)
    assert spec.model == "test"
    assert spec.name == "my-agent"


def test_from_file_with_schema_field(tmp_path: str):
    """$schema field in the file should be accepted and not cause validation errors."""
    spec_path = Path(tmp_path) / "agent.yaml"
    spec_path.write_text("model: test\n", encoding="utf-8")

    # YAML with $schema comment (ignored by yaml parser)
    spec_with_schema = Path(tmp_path) / "agent_with_schema.json"
    spec_with_schema.write_text(
        '{"$schema": "./agent_schema.json", "model": "test"}', encoding="utf-8"
    )
    spec = AgentSpec.from_file(spec_with_schema)
    assert spec.model == "test"
    assert spec.json_schema_path == "./agent_schema.json"


def test_agent_from_file_yaml(tmp_path: str):
    spec_path = Path(tmp_path) / "agent.yaml"
    spec_path.write_text(
        "model: test\nname: my-agent\ninstructions: Be helpful\n", encoding="utf-8"
    )
    agent = Agent.from_file(spec_path)
    assert agent.name == "my-agent"
    assert "Be helpful" in agent._instructions  # pyright: ignore[reportPrivateUsage]


def test_agent_from_file_json(tmp_path: str):
    spec_path = Path(tmp_path) / "agent.json"
    spec_path.write_text('{"model": "test", "name": "json-agent"}', encoding="utf-8")
    agent = Agent.from_file(spec_path)
    assert agent.name == "json-agent"


def test_agent_from_file_with_overrides(tmp_path: str):
    spec_path = Path(tmp_path) / "agent.yaml"
    spec_path.write_text("model: test\nname: spec-name\nretries: 5\n", encoding="utf-8")
    agent = Agent.from_file(spec_path, name="override-name", retries=2)
    assert agent.name == "override-name"
    assert agent._max_tool_retries == 2  # pyright: ignore[reportPrivateUsage]


def test_to_file_yaml(tmp_path: str):
    spec = AgentSpec(model="test", name="my-agent", instructions="Be helpful")
    spec_path = Path(tmp_path) / "agent.yaml"
    spec.to_file(spec_path)

    content = spec_path.read_text(encoding="utf-8")
    # Should start with yaml-language-server schema comment
    assert content.startswith("# yaml-language-server: $schema=")
    assert "model: test" in content
    assert "name: my-agent" in content

    # Schema file should be generated
    schema_path = Path(tmp_path) / "agent_schema.json"
    assert schema_path.exists()


def test_to_file_json(tmp_path: str):
    import json

    spec = AgentSpec(model="test", name="my-agent")
    spec_path = Path(tmp_path) / "agent.json"
    spec.to_file(spec_path)

    data = json.loads(spec_path.read_text(encoding="utf-8"))
    assert data["$schema"] == "agent_schema.json"
    assert data["model"] == "test"
    assert data["name"] == "my-agent"

    # Schema file should be generated
    schema_path = Path(tmp_path) / "agent_schema.json"
    assert schema_path.exists()


def test_to_file_no_schema(tmp_path: str):
    spec = AgentSpec(model="test")
    spec_path = Path(tmp_path) / "agent.yaml"
    spec.to_file(spec_path, schema_path=None)

    content = spec_path.read_text(encoding="utf-8")
    assert "# yaml-language-server" not in content

    # No schema file should be generated
    schema_path = Path(tmp_path) / "agent_schema.json"
    assert not schema_path.exists()


def test_to_file_roundtrip_yaml(tmp_path: str):
    spec = AgentSpec(
        model="test", name="roundtrip", instructions=["Be helpful", "Be concise"]
    )
    spec_path = Path(tmp_path) / "agent.yaml"
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == "test"
    assert loaded.name == "roundtrip"
    assert loaded.instructions == ["Be helpful", "Be concise"]


def test_to_file_roundtrip_json(tmp_path: str):
    spec = AgentSpec(model="test", name="roundtrip", retries=3)
    spec_path = Path(tmp_path) / "agent.json"
    spec.to_file(spec_path)

    loaded = AgentSpec.from_file(spec_path)
    assert loaded.model == "test"
    assert loaded.name == "roundtrip"
    assert loaded.retries == 3


@dataclass
class ToolsetFuncCapability(AbstractCapability[None]):
    """A capability that returns a ToolsetFunc instead of an AbstractToolset."""

    def get_toolset(self) -> ToolsetFunc[None]:
        def make_toolset(ctx: RunContext[None]) -> AbstractToolset[None]:
            toolset = FunctionToolset[None]()

            @toolset.tool_plain
            def greet(name: str) -> str:
                """Greet someone by name."""
                return f"Hello, {name}!"

            return toolset

        return make_toolset


async def test_capability_returning_toolset_func():
    """Test that a capability returning a ToolsetFunc works with an agent."""
    agent = Agent(
        TestModel(),
        capabilities=[ToolsetFuncCapability()],
    )
    result = await agent.run("Greet Alice")

    tool_calls = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == "greet"

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith("Hello, ")


async def test_capability_returning_toolset_func_combined():
    """Test that a ToolsetFunc capability works alongside other capabilities via CombinedCapability."""
    agent = Agent(
        TestModel(),
        instructions="You are a helpful greeter.",
        capabilities=[
            ToolsetFuncCapability(),
        ],
    )
    result = await agent.run("Greet Bob")

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith("Hello, ")


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
    assert merged.get("max_tokens") == 100
    assert merged.get("temperature") == 0.5


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
        return f"Hello, {name}!"

    agent = Agent(TestModel(), capabilities=[Toolset(toolset=ts)])
    result = await agent.run("Greet Alice")

    tool_returns = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    assert len(tool_returns) == 1
    assert isinstance(tool_returns[0].content, str)
    assert tool_returns[0].content.startswith("Hello, ")


def test_infer_fmt_explicit():
    """_infer_fmt returns the explicit fmt when provided."""
    from pydantic_ai.agent.spec import _infer_fmt  # pyright: ignore[reportPrivateUsage]

    assert _infer_fmt(Path("agent.txt"), "json") == "json"
    assert _infer_fmt(Path("agent.txt"), "yaml") == "yaml"


def test_infer_fmt_unknown_extension():
    """_infer_fmt raises ValueError for unknown extension without explicit fmt."""
    from pydantic_ai.agent.spec import _infer_fmt  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(
        ValueError, match="Could not infer format for filename 'agent.txt'"
    ):
        _infer_fmt(Path("agent.txt"), None)


def test_invalid_custom_capability_type():
    """Passing a non-AbstractCapability subclass to model_json_schema_with_capabilities raises ValueError."""
    with pytest.raises(ValueError, match="must be subclasses of AbstractCapability"):
        AgentSpec.model_json_schema_with_capabilities(
            custom_capability_types=[str],  # type: ignore[list-item]
        )


def test_to_file_with_path_schema_path(tmp_path: str):
    """to_file works when schema_path is passed as a relative Path (not str), triggering the non-str branch."""
    spec = AgentSpec(model="test", name="path-schema")
    spec_path = Path(tmp_path) / "agent.yaml"
    # Pass a relative Path (not str) to exercise the isinstance(schema_path, str) == False branch
    schema_path = Path("custom_schema.json")
    spec.to_file(spec_path, schema_path=schema_path)

    resolved_schema = Path(tmp_path) / "custom_schema.json"
    assert resolved_schema.exists()
    content = spec_path.read_text(encoding="utf-8")
    assert "model: test" in content


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
        label: str = ""

    cap1 = SimpleCap(label="a")
    cap2 = SimpleCap(label="b")
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


async def test_for_run_with_different_toolset():
    """When for_run returns a capability with a different get_toolset(), the per-run toolset is used."""
    toolset_a = FunctionToolset(id="a")

    @toolset_a.tool_plain
    def tool_a() -> str:
        return "a"  # pragma: no cover

    toolset_b = FunctionToolset(id="b")

    @toolset_b.tool_plain
    def tool_b() -> str:
        return "b"  # pragma: no cover

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
        return ModelResponse(parts=[TextPart(f"tools: {','.join(sorted(tool_names))}")])

    agent = Agent(FunctionModel(respond), capabilities=[SwitchingCap()])

    # At run time, for_run switches to toolset_b
    result = await agent.run("Hello")
    assert "tool_b" in result.output


async def test_for_run_with_different_instructions():
    """When for_run returns a capability with different get_instructions(), per-run instructions are used."""

    class DynamicInstructionsCap(AbstractCapability[None]):
        def __init__(self, run_instructions: str = "init-time"):
            self._run_instructions = run_instructions

        async def for_run(self, ctx: RunContext[None]) -> AbstractCapability[None]:
            return DynamicInstructionsCap(run_instructions="per-run")

        def get_instructions(self) -> str:
            return self._run_instructions

    captured_messages: list[ModelMessage] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured_messages.extend(messages)
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), capabilities=[DynamicInstructionsCap()])
    await agent.run("Hello")

    # The per-run instructions should appear in the request's instructions field
    instructions_found = [
        msg.instructions
        for msg in captured_messages
        if isinstance(msg, ModelRequest) and msg.instructions
    ]
    assert any("per-run" in i for i in instructions_found), (
        f"Expected per-run instructions in messages, got: {captured_messages}"
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
            assert self.request_count == 1, (
                f"Expected 1, got {self.request_count} — state leaked between runs!"
            )
            return request_context

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Done")])

    agent = Agent(FunctionModel(respond), capabilities=[CountingCap()])

    # Run two concurrent runs — each should get its own CountingCap with count=0
    results = await asyncio.gather(agent.run("A"), agent.run("B"))
    assert results[0].output == "Done"
    assert results[1].output == "Done"


# --- Hooks test helpers ---


@dataclass
class _ReplacingCapability(AbstractCapability[Any]):
    """Capability that replaces ModelRequestNode with a fresh copy in before_node_run.

    Used to test that streaming + node replacement doesn't cause double model execution.
    """

    replaced: bool = field(default=False, init=False)

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        from pydantic_ai._agent_graph import ModelRequestNode

        if isinstance(node, ModelRequestNode) and not self.replaced:
            self.replaced = True
            return ModelRequestNode(request=node.request)  # pyright: ignore[reportUnknownVariableType]
        return node  # pyright: ignore[reportUnknownVariableType]


def make_text_response(text: str = "hello") -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=text)])


def simple_model_function(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    return make_text_response("response from model")


async def simple_stream_function(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    yield "streamed response"


async def tool_calling_stream_function(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str | DeltaToolCalls]:
    """A streaming model that calls a tool on first request, then returns text."""
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                yield "final response"
                return

    if info.function_tools:
        tool = info.function_tools[0]
        yield {0: DeltaToolCall(name=tool.name, json_args="{}", tool_call_id="call-1")}
        return

    yield "no tools available"  # pragma: no cover


def tool_calling_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A model that calls a tool on first request, then returns text."""
    # Check if there's already a tool return in messages (i.e., tool was called)
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, ToolReturnPart):
                return make_text_response("final response")

    # First request: call the tool
    if info.function_tools:
        tool = info.function_tools[0]
        return ModelResponse(
            parts=[ToolCallPart(tool_name=tool.name, args="{}", tool_call_id="call-1")]
        )

    return make_text_response("no tools available")  # pragma: no cover


# --- Logging capability for testing ---


@dataclass
class LoggingCapability(AbstractCapability[Any]):
    """A capability that logs all hook invocations for testing."""

    log: list[str] = field(default_factory=lambda: [])

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.log.append("before_run")

    async def after_run(
        self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
    ) -> AgentRunResult[Any]:
        self.log.append("after_run")
        return result

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: Any
    ) -> AgentRunResult[Any]:
        self.log.append("wrap_run:before")
        result = await handler()
        self.log.append("wrap_run:after")
        return result

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        self.log.append("before_model_request")
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self.log.append("after_model_request")
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: Any,
        handler: Any,
    ) -> ModelResponse:
        self.log.append("wrap_model_request:before")
        response = await handler(request_context)
        self.log.append("wrap_model_request:after")
        return response

    async def before_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        self.log.append(f"before_tool_validate:{call.tool_name}")
        return args

    async def after_tool_validate(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self.log.append(f"after_tool_validate:{call.tool_name}")
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
        self.log.append(f"wrap_tool_validate:{call.tool_name}:before")
        result = await handler(args)
        self.log.append(f"wrap_tool_validate:{call.tool_name}:after")
        return result

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        self.log.append(f"before_tool_execute:{call.tool_name}")
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        self.log.append(f"after_tool_execute:{call.tool_name}")
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        self.log.append(f"wrap_tool_execute:{call.tool_name}:before")
        result = await handler(args)
        self.log.append(f"wrap_tool_execute:{call.tool_name}:after")
        return result

    async def on_run_error(
        self, ctx: RunContext[Any], *, error: BaseException
    ) -> AgentRunResult[Any]:
        self.log.append("on_run_error")
        raise error

    async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
        self.log.append(f"before_node_run:{type(node).__name__}")
        return node

    async def after_node_run(
        self, ctx: RunContext[Any], *, node: Any, result: Any
    ) -> Any:
        self.log.append(f"after_node_run:{type(node).__name__}")
        return result

    async def on_node_run_error(
        self, ctx: RunContext[Any], *, node: Any, error: Exception
    ) -> Any:
        self.log.append(f"on_node_run_error:{type(node).__name__}")
        raise error

    async def on_model_request_error(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        self.log.append("on_model_request_error")
        raise error

    async def on_tool_validate_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: Any,
        error: Any,
    ) -> dict[str, Any]:
        self.log.append(f"on_tool_validate_error:{call.tool_name}")
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
        self.log.append(f"on_tool_execute_error:{call.tool_name}")
        raise error


# --- Tests ---


class TestRunHooks:
    async def test_before_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "before_run" in cap.log

    async def test_after_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "after_run" in cap.log

    async def test_wrap_run(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "wrap_run:before" in cap.log
        assert "wrap_run:after" in cap.log

    async def test_run_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        # wrap_run wraps the run (which includes before_run inside iter),
        # then after_run fires at the end (outside wrap_run)
        assert cap.log.index("wrap_run:before") < cap.log.index("before_run")
        assert cap.log.index("before_run") < cap.log.index("wrap_run:after")
        assert cap.log.index("wrap_run:after") <= cap.log.index("after_run")

    async def test_after_run_can_modify_result(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_run(
                self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
            ) -> AgentRunResult[Any]:
                return AgentRunResult(output="modified output")

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[ModifyResultCap()]
        )
        result = await agent.run("hello")
        assert result.output == "modified output"

    async def test_wrap_run_can_short_circuit(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                # Don't call handler - short-circuit the run
                return AgentRunResult(output="short-circuited")

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[ShortCircuitRunCap()]
        )
        result = await agent.run("hello")
        assert result.output == "short-circuited"

    async def test_wrap_run_can_recover_from_error(self):
        """wrap_run can catch errors from handler() and return a recovery result."""

        @dataclass
        class ErrorRecoveryCap(AbstractCapability[Any]):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    return AgentRunResult(output="recovered from error")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[ErrorRecoveryCap()])
        result = await agent.run("hello")
        assert result.output == "recovered from error"

    async def test_wrap_run_error_propagates_without_recovery(self):
        """Without recovery in wrap_run, errors propagate normally."""

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model))
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")

    async def test_wrap_run_recovery_via_iter(self):
        """wrap_run error recovery works when using agent.iter() too."""

        @dataclass
        class ErrorRecoveryCap(AbstractCapability[Any]):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    return AgentRunResult(output="recovered via iter")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[ErrorRecoveryCap()])
        async with agent.iter("hello") as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        assert agent_run.result.output == "recovered via iter"


class TestModelRequestHooks:
    async def test_before_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "before_model_request" in cap.log

    async def test_after_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "after_model_request" in cap.log

    async def test_wrap_model_request(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "wrap_model_request:before" in cap.log
        assert "wrap_model_request:after" in cap.log

    async def test_model_request_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert cap.log.index("before_model_request") < cap.log.index(
            "wrap_model_request:before"
        )
        assert cap.log.index("wrap_model_request:before") < cap.log.index(
            "wrap_model_request:after"
        )
        assert cap.log.index("wrap_model_request:after") < cap.log.index(
            "after_model_request"
        )

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
                return ModelResponse(parts=[TextPart(content="modified by after hook")])

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[ModifyResponseCap()]
        )
        result = await agent.run("hello")
        assert result.output == "modified by after hook"

    async def test_wrap_model_request_can_modify_response(self):
        @dataclass
        class WrapModifyCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                response = await handler(request_context)
                return ModelResponse(
                    parts=[TextPart(content="wrapped: " + response.parts[0].content)]
                )

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[WrapModifyCap()]
        )
        result = await agent.run("hello")
        assert result.output == "wrapped: response from model"

    async def test_skip_model_request(self):
        @dataclass
        class SkipCap(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                raise SkipModelRequest(
                    ModelResponse(parts=[TextPart(content="skipped model")])
                )

        agent = Agent(FunctionModel(simple_model_function), capabilities=[SkipCap()])
        result = await agent.run("hello")
        assert result.output == "skipped model"

    async def test_before_model_request_swaps_model(self):
        call_log: list[str] = []

        def swap_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            call_log.append("swap_model")
            return make_text_response("from swap model")

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapModelCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                request_context.model = swap_target
                return request_context

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[SwapModelCap()]
        )
        result = await agent.run("hello")
        assert result.output == "from swap model"
        assert call_log == ["swap_model"]

    async def test_wrap_model_request_swaps_model(self):
        call_log: list[str] = []

        def swap_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            call_log.append("swap_model")
            return make_text_response("from swap model")

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapInWrapCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
            ) -> ModelResponse:
                request_context.model = swap_target
                return await handler(request_context)

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[SwapInWrapCap()]
        )
        result = await agent.run("hello")
        assert result.output == "from swap model"
        assert call_log == ["swap_model"]

    async def test_before_model_request_swaps_model_streaming(self):
        call_log: list[str] = []

        async def swap_stream_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            call_log.append("swap_stream")
            yield "from swap stream"

        swap_target = FunctionModel(stream_function=swap_stream_fn)

        @dataclass
        class SwapModelCap(AbstractCapability[Any]):
            async def before_model_request(
                self, ctx: RunContext[Any], request_context: ModelRequestContext
            ) -> ModelRequestContext:
                request_context.model = swap_target
                return request_context

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[SwapModelCap()],
        )
        async with agent.run_stream("hello") as stream:
            output = await stream.get_output()
        assert output == "from swap stream"
        assert call_log == ["swap_stream"]

    async def test_run_context_model_unchanged_after_swap(self):
        observed_models: list[Any] = []

        def swap_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return make_text_response("from swap model")

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
        result = await agent.run("hello")
        assert result.output == "from swap model"
        assert observed_models[0] is original_model

    async def test_hooks_before_model_request_swaps_model(self):
        call_log: list[str] = []
        hooks = Hooks()

        def swap_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            call_log.append("swap_model")
            return make_text_response("from swap model")

        swap_target = FunctionModel(swap_model_fn)

        @hooks.on.before_model_request
        async def _(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            request_context.model = swap_target
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run("hello")
        assert result.output == "from swap model"
        assert call_log == ["swap_model"]

    async def test_after_model_request_sees_wrap_swap(self):
        """after_model_request sees the model swapped during wrap_model_request."""
        after_models: list[Any] = []

        def swap_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return make_text_response("from swap model")

        swap_target = FunctionModel(swap_model_fn)

        @dataclass
        class SwapInWrapAndObserveCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                handler: Any,
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

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[SwapInWrapAndObserveCap()],
        )
        result = await agent.run("hello")
        assert result.output == "from swap model"
        assert after_models[0] is swap_target


class TestToolValidateHooks:
    async def test_tool_validate_hooks_fire(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        await agent.run("call the tool")
        assert "before_tool_validate:my_tool" in cap.log
        assert "after_tool_validate:my_tool" in cap.log
        assert "wrap_tool_validate:my_tool:before" in cap.log
        assert "wrap_tool_validate:my_tool:after" in cap.log

    async def test_before_tool_validate_can_modify_args(self):
        @dataclass
        class ModifyArgsCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                # Inject an argument
                if isinstance(args, dict):
                    return {**args, "name": "injected"}  # pragma: no cover
                return {"name": "injected"}

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[ModifyArgsCap()])

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f"hello {name}"

        await agent.run("greet someone")
        assert received_name == "injected"

    async def test_skip_tool_validation(self):
        @dataclass
        class SkipValidateCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                raise SkipToolValidation({"name": "skip-validated"})

        agent = Agent(
            FunctionModel(tool_calling_model), capabilities=[SkipValidateCap()]
        )

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f"hello {name}"

        await agent.run("greet someone")
        assert received_name == "skip-validated"

    async def test_tool_def_matches_called_tool(self):
        """Verify tool_def is the correct ToolDefinition for the tool being called."""
        received_tool_defs: list[ToolDefinition] = []

        @dataclass
        class CaptureCap(AbstractCapability[Any]):
            async def before_tool_validate(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: str | dict[str, Any],
            ) -> str | dict[str, Any]:
                received_tool_defs.append(tool_def)
                return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[CaptureCap()])

        @agent.tool_plain(description="Say hello")
        def my_tool() -> str:
            return "tool result"

        await agent.run("call the tool")
        assert len(received_tool_defs) == 1
        td = received_tool_defs[0]
        assert td.name == "my_tool"
        assert td.description == "Say hello"
        assert td.kind == "function"


class TestToolExecuteHooks:
    async def test_tool_execute_hooks_fire(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        await agent.run("call the tool")
        assert "before_tool_execute:my_tool" in cap.log
        assert "after_tool_execute:my_tool" in cap.log
        assert "wrap_tool_execute:my_tool:before" in cap.log
        assert "wrap_tool_execute:my_tool:after" in cap.log

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
                return f"modified: {result}"

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[ModifyResultCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return "original"

        result = await agent.run("call tool")
        assert "modified: original" in result.output

    async def test_skip_tool_execution(self):
        @dataclass
        class SkipExecCap(AbstractCapability[Any]):
            async def before_tool_execute(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: dict[str, Any],
            ) -> dict[str, Any]:
                raise SkipToolExecution("denied")

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[SkipExecCap()])

        tool_was_called = False

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_was_called
            tool_was_called = True  # pragma: no cover
            return "should not be called"  # pragma: no cover

        result = await agent.run("call tool")
        assert not tool_was_called
        assert "denied" in result.output

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
                    return "recovered from error"

        cap = ErrorHandlingCap()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError("tool failed")

        await agent.run("call tool")
        assert cap.caught_error == "tool failed"


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
                log.append("cap1:before")
                return request_context

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append("cap1:after")
                return response

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                log.append("cap1:wrap:before")
                response = await handler(request_context)
                log.append("cap1:wrap:after")
                return response

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_model_request(
                self,
                ctx: RunContext[Any],
                request_context: ModelRequestContext,
            ) -> ModelRequestContext:
                log.append("cap2:before")
                return request_context

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append("cap2:after")
                return response

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                log.append("cap2:wrap:before")
                response = await handler(request_context)
                log.append("cap2:wrap:after")
                return response

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[Cap1(), Cap2()]
        )
        await agent.run("hello")

        # before hooks: forward order (cap1 then cap2)
        assert log.index("cap1:before") < log.index("cap2:before")
        # wrap hooks: cap1 outermost, cap2 innermost
        assert log.index("cap1:wrap:before") < log.index("cap2:wrap:before")
        assert log.index("cap2:wrap:after") < log.index("cap1:wrap:after")
        # after hooks: reverse order (cap2 then cap1)
        assert log.index("cap2:after") < log.index("cap1:after")

    async def test_multiple_capabilities_run_hooks_order(self):
        log: list[str] = []

        @dataclass
        class Cap1(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                log.append("cap1:before_run")

            async def after_run(
                self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
            ) -> AgentRunResult[Any]:
                log.append("cap1:after_run")
                return result

            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                log.append("cap1:wrap_run:before")
                result = await handler()
                log.append("cap1:wrap_run:after")
                return result

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                log.append("cap2:before_run")

            async def after_run(
                self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
            ) -> AgentRunResult[Any]:
                log.append("cap2:after_run")
                return result

            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                log.append("cap2:wrap_run:before")
                result = await handler()
                log.append("cap2:wrap_run:after")
                return result

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[Cap1(), Cap2()]
        )
        await agent.run("hello")

        # before_run: forward order
        assert log.index("cap1:before_run") < log.index("cap2:before_run")
        # wrap_run: cap1 outermost
        assert log.index("cap1:wrap_run:before") < log.index("cap2:wrap_run:before")
        assert log.index("cap2:wrap_run:after") < log.index("cap1:wrap_run:after")
        # after_run: reverse order
        assert log.index("cap2:after_run") < log.index("cap1:after_run")


class TestCombinedBeforeWrapAfter:
    async def test_all_hook_types_on_same_capability(self):
        """Test before + wrap + after all fire correctly on a single capability."""
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"

        await agent.run("call tool")

        # Check run hooks
        assert "before_run" in cap.log
        assert "wrap_run:before" in cap.log
        assert "wrap_run:after" in cap.log
        assert "after_run" in cap.log

        # Check model request hooks (should fire twice: once for tool call, once for final)
        model_request_before_count = cap.log.count("before_model_request")
        assert model_request_before_count == 2

        # Check tool hooks
        assert "before_tool_validate:my_tool" in cap.log
        assert "wrap_tool_validate:my_tool:before" in cap.log
        assert "wrap_tool_validate:my_tool:after" in cap.log
        assert "after_tool_validate:my_tool" in cap.log
        assert "before_tool_execute:my_tool" in cap.log
        assert "wrap_tool_execute:my_tool:before" in cap.log
        assert "wrap_tool_execute:my_tool:after" in cap.log
        assert "after_tool_execute:my_tool" in cap.log


class TestRunHooksRunStream:
    """Test that wrap_run and after_run fire for run_stream()."""

    async def test_wrap_run_fires_for_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[cap],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert "wrap_run:before" in cap.log
        assert "wrap_run:after" in cap.log

    async def test_after_run_fires_for_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[cap],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert "after_run" in cap.log

    async def test_wrap_run_fires_for_iter(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        async with agent.iter("hello") as agent_run:
            async for _node in agent_run:
                pass
        assert "wrap_run:before" in cap.log
        assert "wrap_run:after" in cap.log
        assert "after_run" in cap.log

    async def test_after_run_can_modify_result_via_iter(self):
        @dataclass
        class ModifyResultCap(AbstractCapability[Any]):
            async def after_run(
                self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
            ) -> AgentRunResult[Any]:
                return AgentRunResult(output="modified by after_run")

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[ModifyResultCap()]
        )
        async with agent.iter("hello") as agent_run:
            async for _node in agent_run:
                pass
        assert agent_run.result is not None
        assert agent_run.result.output == "modified by after_run"

    async def test_run_hook_order_via_run_stream(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[cap],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert cap.log.index("wrap_run:before") < cap.log.index("before_run")
        assert cap.log.index("before_run") < cap.log.index("wrap_run:after")
        assert cap.log.index("wrap_run:after") <= cap.log.index("after_run")


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
                raise SkipModelRequest(
                    ModelResponse(parts=[TextPart(content="skipped in stream")])
                )

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[SkipCap()],
        )
        async with agent.run_stream("hello") as stream:
            output = await stream.get_output()
        assert output == "skipped in stream"

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
                raise SkipModelRequest(
                    ModelResponse(parts=[TextPart(content="wrap-skipped")])
                )

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[WrapSkipCap()]
        )
        result = await agent.run("hello")
        assert result.output == "wrap-skipped"

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
                raise SkipModelRequest(
                    ModelResponse(parts=[TextPart(content="wrap-skipped in stream")])
                )

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[WrapSkipCap()],
        )
        async with agent.run_stream("hello") as stream:
            output = await stream.get_output()
        assert output == "wrap-skipped in stream"

    async def test_wrap_model_request_streaming(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[cap],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert "wrap_model_request:before" in cap.log
        assert "wrap_model_request:after" in cap.log

    async def test_wrap_model_request_modifies_result_via_run_with_streaming(self):
        """wrap_model_request modification affects the final result when using run() with streaming."""

        @dataclass
        class WrapModifyCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                response = await handler(request_context)
                return ModelResponse(
                    parts=[TextPart(content="wrapped: " + response.parts[0].content)]
                )

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[WrapModifyCap()],
        )

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for _ in stream:
                pass

        result = await agent.run("hello", event_stream_handler=handler)
        assert result.output == "wrapped: streamed response"

    async def test_after_model_request_fires_streaming(self):
        cap = LoggingCapability()
        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[cap],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert "after_model_request" in cap.log


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
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[ObserverCap()],
        )

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for _ in stream:
                pass

        await agent.run("hello", event_stream_handler=handler)
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
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[TransformCap()],
        )

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for event in stream:
                handler_events.append(event)

        await agent.run("hello", event_stream_handler=handler)
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
                log.append("cap1:enter")
                async for event in stream:
                    yield event
                log.append("cap1:exit")

        @dataclass
        class Cap2(AbstractCapability[Any]):
            async def wrap_run_event_stream(
                self,
                ctx: RunContext[Any],
                *,
                stream: AsyncIterable[AgentStreamEvent],
            ) -> AsyncIterable[AgentStreamEvent]:
                log.append("cap2:enter")
                async for event in stream:
                    yield event
                log.append("cap2:exit")

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[Cap1(), Cap2()],
        )

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for _ in stream:
                pass

        await agent.run("hello", event_stream_handler=handler)

        # Cap1 is outermost, so enters first and exits last
        assert log.index("cap1:enter") < log.index("cap2:enter")
        assert log.index("cap2:exit") < log.index("cap1:exit")

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
            FunctionModel(
                tool_calling_model, stream_function=tool_calling_stream_function
            ),
            capabilities=[ObserverCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for _ in stream:
                pass

        await agent.run("call tool", event_stream_handler=handler)
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
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[ObserverCap()],
        )

        # No event_stream_handler — hook should still fire
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert len(observed_events) > 0


class TestWrapRunShortCircuit:
    """Test short-circuiting wrap_run via iter() and run_stream()."""

    async def test_wrap_run_short_circuit_via_iter(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                return AgentRunResult(output="short-circuited")

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[ShortCircuitRunCap()]
        )
        async with agent.iter("hello") as agent_run:
            nodes: list[Any] = []
            async for node in agent_run:
                nodes.append(node)  # pragma: no cover
        # Iteration should stop immediately (no graph nodes executed)
        assert nodes == []
        assert agent_run.result is not None
        assert agent_run.result.output == "short-circuited"

    async def test_wrap_run_short_circuit_via_run_stream(self):
        @dataclass
        class ShortCircuitRunCap(AbstractCapability[Any]):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                return AgentRunResult(output="short-circuited")

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[ShortCircuitRunCap()],
        )
        async with agent.run_stream("hello") as stream:
            output = await stream.get_output()
        assert output == "short-circuited"


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
                log.append("before_model_request")
                raise SkipModelRequest(
                    ModelResponse(parts=[TextPart(content="skipped")])
                )

            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                log.append("after_model_request")
                return response

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[SkipAndLogCap()]
        )
        result = await agent.run("hello")
        assert result.output == "skipped"
        # after_model_request should still fire via _finish_handling
        assert "after_model_request" in log

    async def test_wrap_model_request_short_circuit_streaming(self):
        """wrap_model_request can return without calling handler in streaming path."""

        @dataclass
        class ShortCircuitModelCap(AbstractCapability[Any]):
            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> ModelResponse:
                # Don't call handler — return a response directly
                return ModelResponse(parts=[TextPart(content="model short-circuited")])

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[ShortCircuitModelCap()],
        )
        async with agent.run_stream("hello") as stream:
            output = await stream.get_output()
        assert output == "model short-circuited"


class TestPrepareToolsHook:
    async def test_filter_function_tools(self):
        """Capability can filter out function tools by name."""

        @dataclass
        class HideToolCap(AbstractCapability[Any]):
            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [td for td in tool_defs if td.name != "hidden_tool"]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f"tools: {sorted(tool_names)}")

        agent = Agent(FunctionModel(model_fn), capabilities=[HideToolCap()])

        @agent.tool_plain
        def hidden_tool() -> str:
            return "hidden"  # pragma: no cover

        @agent.tool_plain
        def visible_tool() -> str:
            return "visible"  # pragma: no cover

        result = await agent.run("hello")
        assert result.output == "tools: ['visible_tool']"

    async def test_filter_output_tools(self):
        """Capability can filter output tools (kind='output')."""

        @dataclass
        class RemoveOutputToolsCap(AbstractCapability[Any]):
            seen_output_tool_count: int = 0

            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                self.seen_output_tool_count = len(
                    [td for td in tool_defs if td.kind == "output"]
                )
                return [td for td in tool_defs if td.kind != "output"]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            has_output_tools = len(info.output_tools) > 0
            return make_text_response(f"has output tools: {has_output_tools}")

        cap = RemoveOutputToolsCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        await agent.run("hello")
        # The capability should have seen 0 output tools (no output_type set),
        # but the hook itself was called
        assert cap.seen_output_tool_count == 0

    async def test_modify_tool_description(self):
        """Capability can modify tool descriptions."""
        from dataclasses import replace as dc_replace

        @dataclass
        class PrefixDescriptionCap(AbstractCapability[Any]):
            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [
                    dc_replace(td, description=f"[PREFIXED] {td.description}")
                    if td.kind == "function"
                    else td
                    for td in tool_defs
                ]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs = [t.description for t in info.function_tools]
            return make_text_response(f"descriptions: {descs}")

        agent = Agent(FunctionModel(model_fn), capabilities=[PrefixDescriptionCap()])

        @agent.tool_plain
        def my_tool() -> str:
            """Original description."""
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        assert "[PREFIXED] Original description." in result.output

    async def test_chaining_order(self):
        """Multiple capabilities chain prepare_tools in forward order."""

        @dataclass
        class AddSuffixCap(AbstractCapability[Any]):
            suffix: str

            async def prepare_tools(
                self, ctx: RunContext[Any], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                from dataclasses import replace as dc_replace

                return [
                    dc_replace(td, description=f"{td.description}{self.suffix}")
                    for td in tool_defs
                ]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            descs = [t.description for t in info.function_tools]
            return make_text_response(f"{descs}")

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[AddSuffixCap(suffix="_A"), AddSuffixCap(suffix="_B")],
        )

        @agent.tool_plain
        def tool() -> str:
            """desc"""
            return "r"  # pragma: no cover

        result = await agent.run("hello")
        # A runs first, then B, so suffix order is _A_B
        assert "desc_A_B" in result.output


class TestWrapNodeRunHook:
    async def test_observe_nodes(self):
        """wrap_node_run can observe all nodes in the agent run."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert cap.nodes == ["UserPromptNode", "ModelRequestNode", "CallToolsNode"]

    async def test_observe_nodes_with_tools(self):
        """wrap_node_run fires for each node including tool call round-trips."""

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        await agent.run("hello")
        # UserPrompt -> ModelRequest (calls tool) -> CallTools (executes tool) ->
        # ModelRequest (gets final response) -> CallTools (produces End)
        assert cap.nodes == [
            "UserPromptNode",
            "ModelRequestNode",
            "CallToolsNode",
            "ModelRequestNode",
            "CallToolsNode",
        ]

    async def test_works_with_iter_next(self):
        """wrap_node_run fires when driving iter() with next()."""
        from pydantic_graph import End

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter("hello") as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert cap.nodes == ["UserPromptNode", "ModelRequestNode", "CallToolsNode"]

    async def test_bare_async_for_warns_with_wrap_node_run(self):
        """Using bare async for on iter() warns when a capability has wrap_node_run."""
        import warnings

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                return await handler(
                    node
                )  # pragma: no cover — bare async for doesn't call this

        agent = Agent(
            FunctionModel(simple_model_function), capabilities=[NodeObserverCap()]
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            async with agent.iter("hello") as agent_run:
                async for _node in agent_run:
                    pass
        assert len(w) == 1
        assert "wrap_node_run" in str(w[0].message)

    async def test_works_with_manual_next(self):
        """wrap_node_run fires when using manual next() driving."""
        from pydantic_graph import End

        @dataclass
        class NodeObserverCap(AbstractCapability[Any]):
            nodes: list[str] = field(default_factory=lambda: [])

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                self.nodes.append(type(node).__name__)
                return await handler(node)

        cap = NodeObserverCap()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])

        async with agent.iter("hello") as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        assert cap.nodes == ["UserPromptNode", "ModelRequestNode", "CallToolsNode"]

    async def test_chaining_nests_correctly(self):
        """Multiple capabilities compose wrap_node_run as nested middleware."""
        log: list[str] = []

        @dataclass
        class OrderedCap(AbstractCapability[Any]):
            name: str

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                log.append(f"{self.name}:before:{type(node).__name__}")
                result = await handler(node)
                log.append(f"{self.name}:after:{type(result).__name__}")
                return result

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[OrderedCap(name="outer"), OrderedCap(name="inner")],
        )
        await agent.run("hello")
        # For UserPromptNode: outer wraps inner
        assert log[0] == "outer:before:UserPromptNode"
        assert log[1] == "inner:before:UserPromptNode"
        assert log[2] == "inner:after:ModelRequestNode"
        assert log[3] == "outer:after:ModelRequestNode"


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

    def test_websearch_default_with_nonsupporting_model(
        self, allow_model_requests: None
    ):
        """WebSearch() with a model that doesn't support builtin → DuckDuckGo fallback used."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # When called with tools, call the first one
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(
                            parts=[TextPart(content=f"Tool result: {part.content}")]
                        )
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args='{"query": "test"}',
                            tool_call_id="c1",
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart(content="no tools")]
            )  # pragma: no cover

        model = FunctionModel(
            model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(model, capabilities=[WebSearch()])
        result = agent.run_sync("search for something")
        # Should have used the DuckDuckGo fallback tool
        assert "Tool result" in result.output

    def test_websearch_local_false_with_nonsupporting_model(
        self, allow_model_requests: None
    ):
        """WebSearch(local=False) with non-supporting model → UserError."""
        model = FunctionModel(
            lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset())
        )  # type: ignore
        agent = Agent(model, capabilities=[WebSearch(local=False)])
        with pytest.raises(UserError, match="not supported"):
            agent.run_sync("search")

    def test_websearch_builtin_false(self):
        """WebSearch(builtin=False) → only local, no builtin registered."""
        cap = WebSearch(builtin=False)
        assert cap.get_builtin_tools() == []
        toolset = cap.get_toolset()
        # Should have a plain toolset (no PreparedToolset wrapping)
        assert toolset is not None

    def test_websearch_requires_builtin_with_constraints(
        self, allow_model_requests: None
    ):
        """WebSearch(allowed_domains=...) with non-supporting model → UserError."""
        model = FunctionModel(
            lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset())
        )  # type: ignore
        agent = Agent(model, capabilities=[WebSearch(allowed_domains=["example.com"])])
        with pytest.raises(UserError, match="not supported"):
            agent.run_sync("search")

    def test_websearch_both_false_raises(self):
        """WebSearch(builtin=False, local=False) → UserError at construction."""
        with pytest.raises(UserError, match="both builtin and local cannot be False"):
            WebSearch(builtin=False, local=False)

    def test_websearch_builtin_false_with_constraints_raises(self):
        """WebSearch(builtin=False, allowed_domains=...) → UserError at construction."""
        with pytest.raises(
            UserError, match="constraint fields require the builtin tool"
        ):
            WebSearch(builtin=False, allowed_domains=["example.com"])

    def test_websearch_local_callable(self):
        """WebSearch(local=some_function) → bare callable wrapped in Tool."""
        from pydantic_ai.tools import Tool

        def my_search(query: str) -> str:
            return f"results for {query}"  # pragma: no cover

        cap = WebSearch(local=my_search)
        assert isinstance(cap.local, Tool)


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

    def test_webfetch_default_with_nonsupporting_model(
        self, allow_model_requests: None
    ):
        """WebFetch() with a model that doesn't support builtin → markdownify fallback used."""
        from unittest.mock import AsyncMock, patch

        import httpx

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(
                            parts=[TextPart(content=f"Tool result: {part.content}")]
                        )
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args='{"url": "https://example.com"}',
                            tool_call_id="c1",
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart(content="no tools")]
            )  # pragma: no cover

        mock_response = httpx.Response(
            200,
            text="<html><head><title>Test</title></head><body><p>Hello</p></body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", "https://example.com"),
        )

        model = FunctionModel(
            model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(model, capabilities=[WebFetch()])
        with patch(
            "pydantic_ai.common_tools.web_fetch.safe_download",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = agent.run_sync("fetch something")
        # Verify the web_fetch fallback tool was actually called
        tool_calls = [
            part
            for msg in result.all_messages()
            if isinstance(msg, ModelResponse)
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "web_fetch"

    def test_webfetch_local_false_with_nonsupporting_model(
        self, allow_model_requests: None
    ):
        """WebFetch(local=False) with non-supporting model → UserError."""
        model = FunctionModel(
            lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset())
        )  # type: ignore
        agent = Agent(model, capabilities=[WebFetch(local=False)])
        with pytest.raises(UserError, match="not supported"):
            agent.run_sync("fetch")

    def test_webfetch_builtin_false(self):
        """WebFetch(builtin=False) → only local, no builtin registered."""
        cap = WebFetch(builtin=False)
        assert cap.get_builtin_tools() == []
        toolset = cap.get_toolset()
        assert toolset is not None

    def test_webfetch_max_uses_requires_builtin(self, allow_model_requests: None):
        """WebFetch(max_uses=...) with non-supporting model → UserError."""
        model = FunctionModel(
            lambda m, i: None, profile=ModelProfile(supported_builtin_tools=frozenset())
        )  # type: ignore
        agent = Agent(model, capabilities=[WebFetch(max_uses=5)])
        with pytest.raises(UserError, match="not supported"):
            agent.run_sync("fetch")

    def test_webfetch_domains_forwarded_to_local(self, allow_model_requests: None):
        """WebFetch(allowed_domains=...) with non-supporting model → falls back to local with domain filtering."""
        from unittest.mock import AsyncMock, patch

        import httpx

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return ModelResponse(
                            parts=[TextPart(content=f"Tool result: {part.content}")]
                        )
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args='{"url": "https://example.com"}',
                            tool_call_id="c1",
                        )
                    ]
                )
            return ModelResponse(
                parts=[TextPart(content="no tools")]
            )  # pragma: no cover

        mock_response = httpx.Response(
            200,
            text="<html><body><p>Hello</p></body></html>",
            headers={"content-type": "text/html"},
            request=httpx.Request("GET", "https://example.com"),
        )

        model = FunctionModel(
            model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(model, capabilities=[WebFetch(allowed_domains=["example.com"])])
        with patch(
            "pydantic_ai.common_tools.web_fetch.safe_download",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = agent.run_sync("fetch example.com")
        # Verify the web_fetch fallback tool was actually called with domain filtering
        tool_calls = [
            part
            for msg in result.all_messages()
            if isinstance(msg, ModelResponse)
            for part in msg.parts
            if isinstance(part, ToolCallPart)
        ]
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "web_fetch"

    def test_webfetch_both_false_raises(self):
        """WebFetch(builtin=False, local=False) → UserError at construction."""
        with pytest.raises(UserError, match="both builtin and local cannot be False"):
            WebFetch(builtin=False, local=False)

    def test_webfetch_builtin_false_with_max_uses_raises(self):
        """WebFetch(builtin=False, max_uses=...) → UserError at construction."""
        with pytest.raises(
            UserError, match="constraint fields require the builtin tool"
        ):
            WebFetch(builtin=False, max_uses=5)

    def test_webfetch_local_callable(self):
        """WebFetch(local=some_function) → bare callable wrapped in Tool."""
        from pydantic_ai.tools import Tool

        def my_fetch(url: str) -> str:
            return f"fetched {url}"  # pragma: no cover

        cap = WebFetch(local=my_fetch)
        assert isinstance(cap.local, Tool)


class TestImageGenerationCapability:
    def test_image_gen_init_params_match_builtin_tool(self):
        """ImageGeneration.__init__ accepts all ImageGenerationTool configurable fields."""
        import dataclasses
        import inspect

        # partial_images is excluded — not useful for subagent fallback (no streaming)
        builtin_fields = {
            f.name
            for f in dataclasses.fields(ImageGenerationTool)
            if f.name not in ("kind", "partial_images")
        }
        init_params = set(
            inspect.signature(ImageGeneration.__init__).parameters.keys()
        ) - {
            "self",
            "builtin",
            "local",
            "fallback_model",
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
            return "image_url"  # pragma: no cover

        cap = ImageGeneration(local=my_gen)
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_with_fallback_model(self):
        """ImageGeneration(fallback_model=...) creates a local fallback tool."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(fallback_model="openai-responses:gpt-5.4")
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None
        builtins = cap.get_builtin_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], ImageGenerationTool)

    def test_image_generation_forwards_config_to_builtin(self):
        """ImageGeneration config fields are forwarded to the ImageGenerationTool builtin."""
        cap = ImageGeneration(
            background="opaque",
            input_fidelity="high",
            moderation="low",
            output_compression=80,
            output_format="jpeg",
            quality="high",
            size="1024x1024",
            aspect_ratio="16:9",
        )
        builtins = cap.get_builtin_tools()
        assert len(builtins) == 1
        tool = builtins[0]
        assert isinstance(tool, ImageGenerationTool)
        assert tool.background == "opaque"
        assert tool.input_fidelity == "high"
        assert tool.moderation == "low"
        assert tool.output_compression == 80
        assert tool.output_format == "jpeg"
        assert tool.quality == "high"
        assert tool.size == "1024x1024"
        assert tool.aspect_ratio == "16:9"

    def test_image_generation_fallback_merges_custom_builtin_with_overrides(self):
        """Custom builtin settings are merged with capability-level overrides for the fallback."""
        from pydantic_ai.tools import Tool

        custom_builtin = ImageGenerationTool(quality="high", size="1024x1024")
        cap = ImageGeneration(
            builtin=custom_builtin,
            fallback_model="openai-responses:gpt-5.4",
            output_format="jpeg",  # capability-level override
        )
        # The local fallback should exist and contain the merged config
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_callable_builtin_with_fallback(self):
        """When builtin is a callable, the fallback local tool still gets created."""
        from pydantic_ai.tools import Tool

        cap = ImageGeneration(
            builtin=lambda ctx: ImageGenerationTool(quality="high"),
            fallback_model="openai-responses:gpt-5.4",
        )
        # Callable builtin can't be resolved at init time, but local fallback is still created
        assert isinstance(cap.local, Tool)
        assert cap.get_toolset() is not None

    def test_image_generation_fallback_model_and_local_conflict(self):
        """ImageGeneration(fallback_model=..., local=func) raises UserError."""

        def my_gen(prompt: str) -> str:
            return "image_url"  # pragma: no cover

        with pytest.raises(
            UserError, match="cannot specify both `fallback_model` and `local`"
        ):
            ImageGeneration(fallback_model="openai-responses:gpt-5.4", local=my_gen)

    def test_image_generation_fallback_model_with_local_false(self):
        """ImageGeneration(fallback_model=..., local=False) raises UserError."""
        with pytest.raises(
            UserError, match="cannot specify both `fallback_model` and `local`"
        ):
            ImageGeneration(fallback_model="openai-responses:gpt-5.4", local=False)

    async def test_image_generation_callable_fallback_model(
        self, allow_model_requests: None
    ):
        """ImageGeneration with async callable fallback_model resolves the model per-run."""
        from pydantic_ai.messages import BinaryImage, FilePart

        image_data = b"\x89PNG\r\n\x1a\n"  # minimal PNG header

        def inner_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    FilePart(
                        content=BinaryImage(data=image_data, media_type="image/png")
                    )
                ]
            )

        inner_model = FunctionModel(
            inner_model_fn, profile=ModelProfile(supports_image_output=True)
        )

        async def model_factory(ctx: RunContext[None]) -> FunctionModel:
            return inner_model

        def outer_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            if any(
                isinstance(p, ToolReturnPart)
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            ):
                return ModelResponse(parts=[TextPart(content="done")])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="generate_image", args='{"prompt": "test"}')
                ]
            )

        outer_model = FunctionModel(
            outer_model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(
            outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)]
        )
        result = await agent.run("Generate a test image")
        assert result.output == "done"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="Generate a test image", timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="generate_image",
                            args='{"prompt": "test"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=5),
                    model_name="function:outer_model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="generate_image",
                            content=BinaryImage(
                                data=b"\x89PNG\r\n\x1a\n", media_type="image/png"
                            ),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="done")],
                    usage=RequestUsage(input_tokens=54, output_tokens=6),
                    model_name="function:outer_model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_image_generation_callable_returns_image_only_model(
        self, allow_model_requests: None
    ):
        """Callable fallback_model returning an image-only model name is caught at call time."""

        def model_factory(ctx: RunContext[None]) -> str:
            return "openai-responses:gpt-image-1"

        def outer_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name="generate_image", args='{"prompt": "test"}')
                ]
            )

        outer_model = FunctionModel(
            outer_model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(
            outer_model, capabilities=[ImageGeneration(fallback_model=model_factory)]
        )  # pyright: ignore[reportArgumentType]
        with pytest.raises(
            UserError, match="'gpt-image-1' is a dedicated image generation model"
        ):
            await agent.run("Generate a test image")

    async def test_image_generation_subagent_error_becomes_model_retry(
        self, allow_model_requests: None
    ):
        """UnexpectedModelBehavior from subagent becomes a retry prompt to the outer model."""

        # FunctionModel that returns text but no image — triggers UnexpectedModelBehavior
        def no_image_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="No image generated.")])

        inner_model = FunctionModel(
            no_image_model_fn, profile=ModelProfile(supports_image_output=True)
        )

        call_count = 0

        def outer_model_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="generate_image", args='{"prompt": "test"}'
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="gave up")])

        outer_model = FunctionModel(
            outer_model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(
            outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)]
        )
        result = await agent.run("Generate a test image")
        assert result.output == "gave up"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="Generate a test image", timestamp=IsDatetime()
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="generate_image",
                            args='{"prompt": "test"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=54, output_tokens=5),
                    model_name="function:outer_model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Exceeded maximum retries (1) for output validation",
                            tool_name="generate_image",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="gave up")],
                    usage=RequestUsage(input_tokens=68, output_tokens=7),
                    model_name="function:outer_model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    def test_image_generation_rejects_image_only_model(self):
        """Using a dedicated image model like gpt-image-1 raises a clear error at construction."""
        with pytest.raises(
            UserError, match="'gpt-image-1' is a dedicated image generation model"
        ):
            ImageGeneration(fallback_model="openai-responses:gpt-image-1")

    @pytest.mark.vcr()
    @pytest.mark.filterwarnings(
        "ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning"
    )
    async def test_image_generation_local_fallback(
        self, allow_model_requests: None, openai_api_key: str
    ):
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
                return ModelResponse(
                    parts=[TextPart(content="Here is the generated image.")]
                )

            # First call: invoke the generate_image tool
            assert info.function_tools, "Expected generate_image tool to be available"
            tool = info.function_tools[0]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}'
                    )
                ]
            )

        inner_model = OpenAIResponsesModel(
            "gpt-5.4", provider=OpenAIProvider(api_key=openai_api_key)
        )
        outer_model = FunctionModel(
            model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(
            outer_model,
            capabilities=[
                ImageGeneration(fallback_model=inner_model),
            ],
        )
        result = await agent.run("Generate an image of a cute baby sea otter")
        assert result.output == "Here is the generated image."
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="Generate an image of a cute baby sea otter",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="generate_image",
                            args='{"prompt": "A cute baby sea otter"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=59, output_tokens=9),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="generate_image",
                            content=IsInstance(BinaryImage),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="Here is the generated image.")],
                    usage=RequestUsage(input_tokens=59, output_tokens=15),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.vcr()
    @pytest.mark.filterwarnings(
        "ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning"
    )
    @pytest.mark.filterwarnings(
        "ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning"
    )
    async def test_image_generation_local_fallback_google(
        self, allow_model_requests: None, gemini_api_key: str
    ):
        """ImageGeneration fallback with Google image model."""
        pytest.importorskip("google.genai", reason="google extra not installed")
        from pydantic_ai.messages import BinaryImage
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(
                isinstance(p, ToolReturnPart)
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            ):
                return ModelResponse(
                    parts=[TextPart(content="Here is the generated image.")]
                )
            assert info.function_tools, "Expected generate_image tool to be available"
            tool = info.function_tools[0]
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=tool.name, args='{"prompt": "A cute baby sea otter"}'
                    )
                ]
            )

        inner_model = GoogleModel(
            "gemini-3-pro-image-preview",
            provider=GoogleProvider(api_key=gemini_api_key),
        )
        outer_model = FunctionModel(
            model_fn, profile=ModelProfile(supported_builtin_tools=frozenset())
        )
        agent = Agent(
            outer_model, capabilities=[ImageGeneration(fallback_model=inner_model)]
        )
        result = await agent.run("Generate an image of a cute baby sea otter")
        assert result.output == "Here is the generated image."
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="Generate an image of a cute baby sea otter",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="generate_image",
                            args='{"prompt": "A cute baby sea otter"}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=59, output_tokens=9),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="generate_image",
                            content=IsInstance(BinaryImage),
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="Here is the generated image.")],
                    usage=RequestUsage(input_tokens=59, output_tokens=15),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )


try:
    import mcp as _mcp

    has_mcp = True
    del _mcp
except ImportError:
    has_mcp = False


@pytest.mark.skipif(not has_mcp, reason="mcp is not installed")
class TestMCPCapability:
    def test_mcp_default(self):
        """MCP(url=...) provides builtin + local fallback."""
        cap = MCP(url="https://mcp.example.com/api")
        builtins = cap.get_builtin_tools()
        assert len(builtins) == 1
        assert isinstance(builtins[0], MCPServerTool)
        assert builtins[0].url == "https://mcp.example.com/api"
        assert cap.get_toolset() is not None

    def test_mcp_id_from_url(self):
        """MCP auto-derives id from URL including hostname to avoid collisions."""
        cap = MCP(url="https://mcp.example.com/api")
        builtin = cap.get_builtin_tools()[0]
        assert isinstance(builtin, MCPServerTool)
        assert builtin.id == "mcp.example.com-api"

        # SSE URLs include hostname to avoid collisions between different servers
        cap_sse = MCP(url="https://server1.example.com/sse")
        builtin_sse = cap_sse.get_builtin_tools()[0]
        assert isinstance(builtin_sse, MCPServerTool)
        assert builtin_sse.id == "server1.example.com-sse"

    def test_mcp_sse_transport(self):
        """MCP with /sse URL uses MCPServerSSE for local."""
        from pydantic_ai.mcp import MCPServerSSE

        cap = MCP(url="https://mcp.example.com/sse")
        assert isinstance(cap.local, MCPServerSSE)

    def test_mcp_streamable_transport(self):
        """MCP with non-/sse URL uses MCPServerStreamableHTTP for local."""
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        cap = MCP(url="https://mcp.example.com/api")
        assert isinstance(cap.local, MCPServerStreamableHTTP)

    def test_mcp_authorization_token_in_local_headers(self):
        """MCP passes authorization_token as Authorization header to local."""
        from pydantic_ai.mcp import MCPServerStreamableHTTP

        cap = MCP(url="https://mcp.example.com/api", authorization_token="Bearer xyz")
        assert isinstance(cap.local, MCPServerStreamableHTTP)
        assert cap.local.headers == {"Authorization": "Bearer xyz"}

    def test_mcp_allowed_tools_filters_local(self):
        """MCP(allowed_tools=...) applies FilteredToolset to the local toolset."""
        from pydantic_ai.toolsets.filtered import FilteredToolset

        cap = MCP(url="https://mcp.example.com/api", allowed_tools=["tool1"])
        toolset = cap.get_toolset()
        assert toolset is not None
        # The outer toolset should be a FilteredToolset wrapping the prepared toolset
        assert isinstance(toolset, FilteredToolset)

    def test_mcp_url_required(self):
        """MCP without url raises TypeError."""
        with pytest.raises(
            TypeError, match="missing 1 required positional argument: 'url'"
        ):
            MCP()  # type: ignore[call-arg]


class TestNamedSpecDictRoundTrip:
    """Test that NamedSpec correctly round-trips various argument forms."""

    def test_dict_positional_arg_uses_long_form(self):
        """A dict positional arg falls back to long form to avoid kwargs misinterpretation on round-trip."""
        spec = NamedSpec(name="CustomCap", arguments=({"key": "value", "other": 42},))
        serialized = spec.model_dump(context={"use_short_form": True})
        # Dict with string keys would be ambiguous in short form, so long form is used
        assert serialized["name"] == "CustomCap"
        assert len(serialized["arguments"]) == 1
        assert serialized["arguments"][0] == {"key": "value", "other": 42}
        # Round-trip preserves the dict as a positional arg
        deserialized = NamedSpec.model_validate(serialized)
        assert deserialized.args == ({"key": "value", "other": 42},)
        assert deserialized.kwargs == {}

    def test_non_dict_positional_arg_uses_short_form(self):
        """A non-dict positional arg still uses the compact short form."""
        spec = NamedSpec(name="WebSearch", arguments=(True,))
        serialized = spec.model_dump(context={"use_short_form": True})
        assert serialized == {"WebSearch": True}

    def test_kwargs_use_short_form(self):
        """Kwargs (dict arguments) use the short form correctly."""
        spec = NamedSpec(name="WebSearch", arguments={"local": True})
        serialized = spec.model_dump(context={"use_short_form": True})
        assert serialized == {"WebSearch": {"local": True}}


class TestPrepareToolsCapability:
    async def test_prepare_tools_filters(self):
        """PrepareTools capability filters tools using the provided callable."""
        from pydantic_ai.capabilities import PrepareTools

        async def hide_secret_tools(
            ctx: RunContext[None], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition] | None:
            return [td for td in tool_defs if td.name != "secret_tool"]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f"tools: {sorted(tool_names)}")

        agent = Agent(
            FunctionModel(model_fn), capabilities=[PrepareTools(hide_secret_tools)]
        )

        @agent.tool_plain
        def secret_tool() -> str:
            return "secret"  # pragma: no cover

        @agent.tool_plain
        def public_tool() -> str:
            return "public"  # pragma: no cover

        result = await agent.run("hello")
        assert result.output == "tools: ['public_tool']"

    async def test_prepare_tools_none_disables_all(self):
        """PrepareTools treats None return as 'disable all tools', consistent with ToolsPrepareFunc docs."""
        from pydantic_ai.capabilities import PrepareTools

        async def disable_all(
            ctx: RunContext[None], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition] | None:
            return None

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = [t.name for t in info.function_tools]
            return make_text_response(f"tools: {sorted(tool_names)}")

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(disable_all)])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        assert result.output == "tools: []"

    async def test_prepare_tools_modifies_definitions(self):
        """PrepareTools can modify tool definitions (e.g. set strict mode)."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.capabilities import PrepareTools

        async def set_strict(
            ctx: RunContext[None], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition] | None:
            return [dc_replace(td, strict=True) for td in tool_defs]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            strictness = [t.strict for t in info.function_tools]
            return make_text_response(f"strict: {strictness}")

        agent = Agent(FunctionModel(model_fn), capabilities=[PrepareTools(set_strict)])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        assert result.output == "strict: [True]"

    def test_prepare_tools_not_serializable(self):
        """PrepareTools opts out of spec serialization."""
        from pydantic_ai.capabilities import PrepareTools

        assert PrepareTools.get_serialization_name() is None


class TestOverrideWithSpec:
    async def test_override_with_spec_instructions_and_model(self):
        """Spec instructions and model replace the agent's when used via override."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn), instructions="original")

        with agent.override(spec={"instructions": "from spec"}):
            result = await agent.run("hello")

        assert "from spec" in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="from spec",
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="instructions: from spec")],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_override_with_spec_explicit_param_wins(self):
        """Explicit override param beats spec value."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn), instructions="original")

        with agent.override(
            spec={"instructions": "from spec"}, instructions="explicit"
        ):
            result = await agent.run("hello")

        assert "explicit" in result.output
        assert "from spec" not in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="explicit",
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="instructions: explicit")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_override_with_spec_instructions(self):
        """Override with spec instructions replaces agent's existing instructions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn), instructions="agent-instructions")

        with agent.override(spec={"instructions": "from-spec-instructions"}):
            result = await agent.run("hello")
            # Override replaces: only spec instructions, not agent's
            assert "from-spec-instructions" in result.output
            assert "agent-instructions" not in result.output
            assert result.all_messages() == snapshot(
                [
                    ModelRequest(
                        parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                        timestamp=IsDatetime(),
                        instructions="from-spec-instructions",
                        run_id=IsStr(),
                    ),
                    ModelResponse(
                        parts=[
                            TextPart(content="instructions: from-spec-instructions")
                        ],
                        usage=RequestUsage(input_tokens=51, output_tokens=2),
                        model_name="function:model_fn:",
                        timestamp=IsDatetime(),
                        run_id=IsStr(),
                    ),
                ]
            )

    async def test_override_with_spec_capabilities(self):
        """Override with spec providing capabilities uses them for the run."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response("ok")

        agent = Agent(FunctionModel(model_fn))

        with agent.override(spec={"capabilities": [{"WebSearch": {"local": False}}]}):
            result = await agent.run("hello")
            assert result.output == "ok"


class TestRunWithSpec:
    async def test_run_with_spec_instructions_added(self):
        """Spec instructions are added additively at run time."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn), instructions="original")

        result = await agent.run("hello", spec={"instructions": "also from spec"})
        # Both original and spec instructions should be present
        assert "original" in result.output
        assert "also from spec" in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="""\
original
also from spec\
""",
                    run_id=IsStr(),
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
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_model_as_fallback(self):
        """Spec model is used as fallback when no run-time model is provided."""
        agent = Agent(None)  # No model set

        result = await agent.run("hello", spec={"model": "test"})
        assert result.output == "success (no tool calls)"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="success (no tool calls)")],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name="test",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_model_settings_merged(self):
        """Spec model_settings are merged with run model_settings."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            max_tokens = (
                info.model_settings.get("max_tokens") if info.model_settings else None
            )
            temperature = (
                info.model_settings.get("temperature") if info.model_settings else None
            )
            return make_text_response(
                f"max_tokens={max_tokens} temperature={temperature}"
            )

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run(
            "hello",
            spec={"model_settings": {"max_tokens": 100}},
            model_settings={"temperature": 0.5},
        )
        assert "max_tokens=100" in result.output
        assert "temperature=0.5" in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="max_tokens=100 temperature=0.5")],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_partial_no_model(self):
        """Partial spec without model works if agent has a model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run("hello", spec={"instructions": "be helpful"})
        assert "be helpful" in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="be helpful",
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="instructions: be helpful")],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_capabilities(self):
        """Run with spec capabilities merges them with agent's root capability."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn), instructions="agent-level")

        result = await agent.run(
            "hello",
            spec={"capabilities": [{"WebSearch": {"local": False}}]},
        )
        # Agent-level instructions should be present; spec capabilities are merged additively
        assert "agent-level" in result.output

    async def test_run_with_spec_instructions(self):
        """Run with spec instructions adds to agent's instructions."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn), instructions="agent-level")

        result = await agent.run(
            "hello",
            spec={
                "instructions": "from-spec",
            },
        )
        # Both should be present (additive)
        assert "agent-level" in result.output
        assert "from-spec" in result.output
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    instructions="""\
agent-level
from-spec\
""",
                    run_id=IsStr(),
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
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_run_with_spec_metadata_merged(self):
        """Spec metadata is merged with run metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response("ok")

        agent = Agent(FunctionModel(model_fn), metadata={"agent_key": "agent_val"})

        result = await agent.run(
            "hello",
            spec={"metadata": {"spec_key": "spec_val"}},
            metadata={"run_key": "run_val"},
        )
        assert result.output == "ok"
        # Run metadata should take precedence, spec metadata should be present
        assert result.metadata is not None
        assert result.metadata == snapshot(
            {"agent_key": "agent_val", "spec_key": "spec_val", "run_key": "run_val"}
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="ok")],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_spec_unsupported_fields_warns(self):
        """Non-default unsupported fields produce warnings."""
        agent = Agent("test")

        with pytest.warns(UserWarning, match="retries"):
            await agent.run("hello", spec={"retries": 5})


class TestGetWrapperToolsetHook:
    async def test_wrapper_prefixes_tools(self):
        """Capability can wrap the toolset to prefix tool names."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix="cap")

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f"tools: {tool_names}")

        agent = Agent(FunctionModel(model_fn), capabilities=[PrefixCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        assert result.output == "tools: ['cap_my_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['cap_my_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_prefixes_tools_streaming(self):
        """Wrapper toolset works correctly with streaming runs."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix="cap")

        async def stream_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            tool_names = sorted(t.name for t in info.function_tools)
            yield f"tools: {tool_names}"

        agent = Agent(
            FunctionModel(stream_function=stream_fn), capabilities=[PrefixCap()]
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        async with agent.run_stream("hello") as result:
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
            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
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

        await agent.run("hello")
        # The wrapper should only see function tools, not output tools
        for tool_names in seen_tool_names:
            assert "add_one" in tool_names
            # Output tool names should not appear in the wrapped toolset
            assert all(not name.startswith("final_result") for name in tool_names)

    async def test_wrapper_none_is_noop(self):
        """Returning None from get_wrapper_toolset leaves the toolset unchanged."""

        @dataclass
        class NoopCap(AbstractCapability[Any]):
            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
                return None

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f"tools: {tool_names}")

        agent = Agent(FunctionModel(model_fn), capabilities=[NoopCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        assert result.output == "tools: ['my_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['my_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_chaining_order(self):
        """Multiple capabilities' wrappers compose by nesting: first wraps innermost."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            prefix: str

            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix=self.prefix)

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f"tools: {tool_names}")

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[PrefixCap(prefix="a"), PrefixCap(prefix="b")],
        )

        @agent.tool_plain
        def tool() -> str:
            return "r"  # pragma: no cover

        result = await agent.run("hello")
        # First cap wraps innermost (a_tool), then second wraps that (b_a_tool)
        assert result.output == "tools: ['b_a_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['b_a_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_with_per_run_capability(self):
        """Wrapper works correctly with capabilities returning new instances from for_run."""
        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PerRunPrefixCap(AbstractCapability[Any]):
            prefix: str = "default"

            async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
                return PerRunPrefixCap(prefix="runtime")

            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix=self.prefix)

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            return make_text_response(f"tools: {tool_names}")

        agent = Agent(FunctionModel(model_fn), capabilities=[PerRunPrefixCap()])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        # The per-run instance should use 'runtime' prefix, not 'default'
        assert result.output == "tools: ['runtime_my_tool']"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="tools: ['runtime_my_tool']")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_wrapper_with_agent_prepare_tools(self):
        """Agent-level prepare_tools is applied before capability wrapper."""
        from dataclasses import replace as dc_replace

        from pydantic_ai.toolsets.prefixed import PrefixedToolset

        @dataclass
        class PrefixCap(AbstractCapability[Any]):
            def get_wrapper_toolset(
                self, toolset: AbstractToolset[Any]
            ) -> AbstractToolset[Any] | None:
                return PrefixedToolset(toolset, prefix="cap")

        async def agent_prepare(
            ctx: RunContext[Any], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            return [
                dc_replace(td, description=f"[prepared] {td.description}")
                for td in tool_defs
            ]

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            tool_names = sorted(t.name for t in info.function_tools)
            descs = [t.description for t in info.function_tools]
            return make_text_response(f"tools: {tool_names}, descs: {descs}")

        agent = Agent(
            FunctionModel(model_fn),
            prepare_tools=agent_prepare,
            capabilities=[PrefixCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            """Original."""
            return "result"  # pragma: no cover

        result = await agent.run("hello")
        # Both agent prepare_tools (description) and capability wrapper (prefix) should apply
        assert (
            result.output == "tools: ['cap_my_tool'], descs: ['[prepared] Original.']"
        )
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content="tools: ['cap_my_tool'], descs: ['[prepared] Original.']"
                        )
                    ],
                    usage=RequestUsage(input_tokens=51, output_tokens=6),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )


# --- from_spec error cases ---


def test_from_spec_no_model_raises():
    """from_spec() without model raises UserError."""
    with pytest.raises(UserError, match="`model` must be provided"):
        Agent.from_spec({"instructions": "hello"})


# --- run() with spec: additional merge scenarios ---


class TestRunWithSpecAdditional:
    async def test_run_with_spec_and_run_instructions_merged(self):
        """When run() passes both instructions and spec instructions, they merge."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            instructions = next(
                (
                    m.instructions
                    for m in messages
                    if isinstance(m, ModelRequest) and m.instructions
                ),
                None,
            )
            return make_text_response(f"instructions: {instructions}")

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run(
            "hello",
            spec={"instructions": "spec instructions"},
            instructions="run instructions",
        )
        assert "run instructions" in result.output
        assert "spec instructions" in result.output

    async def test_run_with_spec_metadata_only(self):
        """Spec metadata is used when run() doesn't pass metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response("ok")

        agent = Agent(FunctionModel(model_fn))

        result = await agent.run("hello", spec={"metadata": {"from": "spec"}})
        assert result.metadata == {"from": "spec"}

    async def test_run_with_spec_metadata_callable_merged(self):
        """Callable metadata from run() merges with spec metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response("ok")

        agent = Agent(FunctionModel(model_fn))

        def dynamic_metadata(ctx: RunContext[None]) -> dict[str, Any]:
            return {"dynamic": "value"}

        result = await agent.run(
            "hello",
            spec={"metadata": {"spec_key": "spec_val"}},
            metadata=dynamic_metadata,
        )
        assert result.metadata is not None
        assert result.metadata["spec_key"] == "spec_val"
        assert result.metadata["dynamic"] == "value"

    async def test_run_with_spec_model_settings_callable_passthrough(self):
        """Callable model_settings from run() bypasses spec model_settings merge."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            temperature = (
                info.model_settings.get("temperature") if info.model_settings else None
            )
            max_tokens = (
                info.model_settings.get("max_tokens") if info.model_settings else None
            )
            return make_text_response(
                f"temperature={temperature} max_tokens={max_tokens}"
            )

        agent = Agent(FunctionModel(model_fn))

        def dynamic_settings(ctx: RunContext[None]) -> _ModelSettings:
            return {"temperature": 0.9}

        result = await agent.run(
            "hello",
            spec={"model_settings": {"max_tokens": 100}},
            model_settings=dynamic_settings,
        )
        # Callable model_settings bypass spec merge — spec model_settings are handled
        # via the capability layer instead
        assert "temperature=0.9" in result.output


# --- override() with spec: additional field tests ---


class TestOverrideWithSpecAdditional:
    async def test_override_with_spec_name(self):
        """Override with spec providing agent name."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response("ok")

        agent = Agent(FunctionModel(model_fn), name="original")

        with agent.override(spec={"name": "spec-name"}):
            assert agent.name == "spec-name"
            result = await agent.run("hello")
        assert result.output == "ok"
        assert agent.name == "original"

    async def test_override_with_spec_model(self):
        """Override with spec providing model."""
        agent = Agent("test", name="test-agent")

        with agent.override(spec={"model": "test"}):
            result = await agent.run("hello")
        assert result.output == "success (no tool calls)"

    async def test_override_with_spec_model_settings(self):
        """Override with spec providing model_settings."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            max_tokens = (
                info.model_settings.get("max_tokens") if info.model_settings else None
            )
            return make_text_response(f"max_tokens={max_tokens}")

        agent = Agent(FunctionModel(model_fn))

        with agent.override(spec={"model_settings": {"max_tokens": 42}}):
            result = await agent.run("hello")
        assert "max_tokens=42" in result.output

    async def test_override_with_spec_metadata(self):
        """Override with spec providing metadata."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return make_text_response("ok")

        agent = Agent(FunctionModel(model_fn))

        with agent.override(spec={"metadata": {"env": "test"}}):
            result = await agent.run("hello")
        assert result.metadata == {"env": "test"}


# --- Capability construction tests ---


def test_web_fetch_with_constraints():
    """WebFetch capability populates builtin tool with all constraint kwargs."""
    cap = WebFetch(
        allowed_domains=["example.com"],
        blocked_domains=["bad.com"],
        max_uses=5,
        enable_citations=True,
        max_content_tokens=1000,
    )
    builtin_tools = cap.get_builtin_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, WebFetchTool)
    assert tool.allowed_domains == ["example.com"]
    assert tool.blocked_domains == ["bad.com"]
    assert tool.max_uses == 5
    assert tool.enable_citations is True
    assert tool.max_content_tokens == 1000
    # Only max_uses requires builtin (domains are handled locally)
    assert cap._requires_builtin() is True  # pyright: ignore[reportPrivateUsage]


def test_web_fetch_unique_id():
    """WebFetch returns the correct builtin unique_id."""
    cap = WebFetch()
    assert cap._builtin_unique_id() == "web_fetch"  # pyright: ignore[reportPrivateUsage]


def test_web_search_with_constraints():
    """WebSearch capability populates builtin tool with all constraint kwargs."""
    from pydantic_ai.builtin_tools import WebSearchUserLocation

    cap = WebSearch(
        search_context_size="high",
        user_location=WebSearchUserLocation(city="NYC", country="US"),
        blocked_domains=["bad.com"],
        allowed_domains=["good.com"],
        max_uses=3,
    )
    builtin_tools = cap.get_builtin_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, WebSearchTool)
    assert tool.search_context_size == "high"
    assert tool.user_location is not None
    assert tool.blocked_domains == ["bad.com"]
    assert tool.allowed_domains == ["good.com"]
    assert tool.max_uses == 3
    assert cap._requires_builtin() is True  # pyright: ignore[reportPrivateUsage]


def test_web_search_default_local_import_error(monkeypatch: pytest.MonkeyPatch):
    """WebSearch._default_local() warns and returns None when duckduckgo is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic_ai.common_tools.duckduckgo":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.warns(UserWarning, match="duckduckgo"):
        cap = WebSearch(builtin=False)
    # With builtin disabled and no duckduckgo, local is None
    assert cap.local is None


def test_web_fetch_default_local_import_error(monkeypatch: pytest.MonkeyPatch):
    """WebFetch._default_local() warns and returns None when markdownify is not installed."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic_ai.common_tools.web_fetch":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    with pytest.warns(UserWarning, match="web-fetch"):
        cap = WebFetch(builtin=False)
    # With builtin disabled and no markdownify, local is None
    assert cap.local is None


def test_mcp_default_builtin():
    """MCP capability constructs the default builtin MCPServerTool."""
    pytest.importorskip("mcp", reason="mcp package not installed")
    cap = MCP(url="http://example.com/mcp", id="my-mcp")
    builtin_tools = cap.get_builtin_tools()
    assert len(builtin_tools) == 1
    tool = builtin_tools[0]
    assert isinstance(tool, MCPServerTool)
    assert tool.url == "http://example.com/mcp"
    assert tool.id == "my-mcp"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_builtin_or_local_base_no_default_builtin():
    """BuiltinOrLocalTool base class with builtin=True raises (no _default_builtin)."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    with pytest.raises(UserError, match="builtin=True requires a subclass"):
        BuiltinOrLocalTool()


def test_builtin_tool_from_spec_no_args():
    """BuiltinTool.from_spec() with no arguments raises TypeError."""
    from pydantic_ai.capabilities.builtin_tool import (
        BuiltinTool as BuiltinToolCapDirect,
    )

    with pytest.raises(TypeError, match="requires either a `tool` argument"):
        BuiltinToolCapDirect.from_spec()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_builtin_or_local_no_default_local():
    """BuiltinOrLocalTool base class _default_local() returns None."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    cap = BuiltinOrLocalTool(builtin=WebSearchTool())
    # Base class _default_local() returns None — no local fallback
    assert cap.local is None
    assert cap.get_toolset() is None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_builtin_or_local_with_explicit_builtin():
    """BuiltinOrLocalTool used directly with an explicit builtin and local tool."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    def my_local_tool() -> str:
        """A local fallback tool."""
        return "local result"  # pragma: no cover

    cap = BuiltinOrLocalTool(builtin=WebSearchTool(), local=my_local_tool)
    # get_builtin_tools returns the explicit builtin
    assert len(cap.get_builtin_tools()) == 1
    assert isinstance(cap.get_builtin_tools()[0], WebSearchTool)
    # get_toolset wraps local with prefer_builtin from _builtin_unique_id()
    toolset = cap.get_toolset()
    assert toolset is not None


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_builtin_or_local_builtin_unique_id_non_abstract():
    """_builtin_unique_id() raises when builtin is callable (not AbstractBuiltinTool)."""
    from pydantic_ai.capabilities.builtin_or_local import BuiltinOrLocalTool

    cap = BuiltinOrLocalTool.__new__(BuiltinOrLocalTool)
    cap.builtin = lambda ctx: WebSearchTool()
    cap.local = False

    with pytest.raises(UserError, match="cannot derive builtin_unique_id"):
        cap._builtin_unique_id()  # pyright: ignore[reportPrivateUsage]


def test_validate_capability_not_dataclass():
    """Custom capability type without @dataclass raises ValueError."""
    from pydantic_ai.agent.spec import get_capability_registry

    class NotADataclass(AbstractCapability[Any]):
        pass

    with pytest.raises(ValueError, match="must be decorated with `@dataclass`"):
        get_capability_registry(custom_types=(NotADataclass,))


# --- Node run lifecycle hook tests ---


class TestNodeRunHooks:
    async def test_before_node_run_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "before_node_run:UserPromptNode" in cap.log
        assert "before_node_run:ModelRequestNode" in cap.log
        assert "before_node_run:CallToolsNode" in cap.log

    async def test_after_node_run_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "after_node_run:UserPromptNode" in cap.log
        assert "after_node_run:ModelRequestNode" in cap.log
        assert "after_node_run:CallToolsNode" in cap.log

    async def test_node_hook_order(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        # For each node, before fires before after
        for node_name in ("UserPromptNode", "ModelRequestNode", "CallToolsNode"):
            before_idx = cap.log.index(f"before_node_run:{node_name}")
            after_idx = cap.log.index(f"after_node_run:{node_name}")
            assert before_idx < after_idx


# --- Run error hook tests ---


class TestRunErrorHooks:
    async def test_on_run_error_fires_on_failure(self):
        cap = LoggingCapability()

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")
        assert "on_run_error" in cap.log

    async def test_on_run_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "on_run_error" not in cap.log

    async def test_on_run_error_can_transform_error(self):
        @dataclass
        class TransformErrorCap(AbstractCapability[Any]):
            async def on_run_error(
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                raise ValueError("transformed error")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[TransformErrorCap()])
        with pytest.raises(ValueError, match="transformed error"):
            await agent.run("hello")

    async def test_on_run_error_can_recover(self):
        @dataclass
        class RecoverRunCap(AbstractCapability[Any]):
            async def on_run_error(
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                return AgentRunResult(output="recovered")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[RecoverRunCap()])
        result = await agent.run("hello")
        assert result.output == "recovered"

    async def test_on_run_error_not_called_when_wrap_run_recovers(self):
        @dataclass
        class WrapRecoveryCap(AbstractCapability[Any]):
            log: list[str] = field(default_factory=lambda: [])

            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                try:
                    return await handler()
                except RuntimeError:
                    self.log.append("wrap_run:caught")
                    return AgentRunResult(output="wrap_recovered")

            async def on_run_error(  # pragma: no cover — verifying this is NOT called
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                self.log.append("on_run_error")
                raise error

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        cap = WrapRecoveryCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run("hello")
        assert result.output == "wrap_recovered"
        assert "wrap_run:caught" in cap.log
        assert "on_run_error" not in cap.log

    async def test_on_run_error_fires_via_iter(self):
        from pydantic_graph import End

        @dataclass
        class RecoverRunCap(AbstractCapability[Any]):
            called: bool = False

            async def on_run_error(
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                self.called = True
                return AgentRunResult(output="recovered via iter")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        cap = RecoverRunCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        async with agent.iter("hello") as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):  # pragma: no branch
                node = await agent_run.next(node)
        assert cap.called
        assert agent_run.result is not None
        assert agent_run.result.output == "recovered via iter"


# --- Node run error hook tests ---


class TestNodeRunErrorHooks:
    async def test_on_node_run_error_fires(self):
        cap = LoggingCapability()

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")
        assert "on_node_run_error:ModelRequestNode" in cap.log

    async def test_on_node_run_error_can_recover_with_end(self):
        from pydantic_ai.result import FinalResult
        from pydantic_graph import End

        @dataclass
        class RecoverNodeCap(AbstractCapability[Any]):
            async def on_node_run_error(
                self, ctx: RunContext[Any], *, node: Any, error: BaseException
            ) -> Any:
                return End(FinalResult(output="recovered"))

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        cap = RecoverNodeCap()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        async with agent.iter("hello") as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)
        assert isinstance(node, End)
        assert node.data.output == "recovered"

    async def test_on_node_run_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert all("on_node_run_error" not in entry for entry in cap.log)


# --- Model request error hook tests ---


class TestModelRequestErrorHooks:
    async def test_on_model_request_error_fires(self):
        cap = LoggingCapability()

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")
        assert "on_model_request_error" in cap.log

    async def test_on_model_request_error_can_recover(self):
        @dataclass
        class RecoverModelCap(AbstractCapability[Any]):
            async def on_model_request_error(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                error: Exception,
            ) -> ModelResponse:
                return ModelResponse(parts=[TextPart(content="recovered response")])

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[RecoverModelCap()])
        result = await agent.run("hello")
        assert result.output == "recovered response"

    async def test_on_model_request_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[cap])
        await agent.run("hello")
        assert "on_model_request_error" not in cap.log

    async def test_default_on_model_request_error_reraises(self):
        """Default on_model_request_error re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            def get_instructions(self):
                return "Be helpful."

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[MinimalCap()])
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")

    async def test_default_on_model_request_error_reraises_streaming(self):
        """Default on_model_request_error re-raises in streaming path (wrap_task error after stream consumed)."""

        @dataclass
        class PostProcessFailCap(AbstractCapability[Any]):
            """wrap_model_request that fails AFTER handler returns (post-processing error)."""

            def get_instructions(self):
                return "Be helpful."

            async def wrap_model_request(
                self, ctx: RunContext[Any], *, request_context: Any, handler: Any
            ) -> Any:
                await handler(request_context)
                raise RuntimeError("post-processing exploded")

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[PostProcessFailCap()],
        )
        with pytest.raises(RuntimeError, match="post-processing exploded"):
            async with agent.run_stream("hello") as stream:
                await stream.get_output()


# --- Tool validate error hook tests ---


class TestToolValidateErrorHooks:
    async def test_on_tool_validate_error_fires_on_validation_failure(self):
        cap = LoggingCapability()

        call_count = 0

        def bad_args_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name=tool.name,
                                args='{"wrong": 1}',
                                tool_call_id="call-1",
                            )
                        ]
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=tool.name,
                            args='{"name": "correct"}',
                            tool_call_id="call-2",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[cap])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f"hello {name}"

        await agent.run("greet someone")
        assert "on_tool_validate_error:greet" in cap.log

    async def test_on_tool_validate_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        await agent.run("call the tool")
        assert all("on_tool_validate_error" not in entry for entry in cap.log)

    async def test_on_tool_validate_error_can_recover(self):
        @dataclass
        class RecoverValidateCap(AbstractCapability[Any]):
            async def on_tool_validate_error(
                self,
                ctx: RunContext[Any],
                *,
                call: ToolCallPart,
                tool_def: ToolDefinition,
                args: Any,
                error: Any,
            ) -> dict[str, Any]:
                return {"name": "recovered-name"}

        def bad_args_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=tool.name,
                            args='{"wrong": 1}',
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(
            FunctionModel(bad_args_model), capabilities=[RecoverValidateCap()]
        )

        received_name = None

        @agent.tool_plain
        def greet(name: str) -> str:
            nonlocal received_name
            received_name = name
            return f"hello {name}"

        result = await agent.run("greet someone")
        assert received_name == "recovered-name"
        assert "hello recovered-name" in result.output

    async def test_default_on_tool_validate_error_reraises(self):
        """The default on_tool_validate_error re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            def get_instructions(self):
                return "Be helpful."

        call_count = 0

        def bad_args_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name=tool.name,
                                args='{"wrong": 1}',
                                tool_call_id="call-1",
                            )
                        ]
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=tool.name,
                            args='{"name": "correct"}',
                            tool_call_id="call-2",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f"hello {name}"

        result = await agent.run("greet someone")
        assert "hello correct" in result.output


# --- Tool execute error hook tests ---


class TestToolExecuteErrorHooks:
    async def test_on_tool_execute_error_fires(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError("tool failed")

        with pytest.raises(ValueError, match="tool failed"):
            await agent.run("call the tool")
        assert "on_tool_execute_error:my_tool" in cap.log

    async def test_on_tool_execute_error_not_called_on_success(self):
        cap = LoggingCapability()
        agent = Agent(FunctionModel(tool_calling_model), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        await agent.run("call the tool")
        assert all("on_tool_execute_error" not in entry for entry in cap.log)

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
                return "fallback result"

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[RecoverExecCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError("tool failed")

        result = await agent.run("call tool")
        assert "fallback result" in result.output


# --- Hooks capability tests ---


class TestHooksCapability:
    """Tests for the Hooks decorator-based capability."""

    async def test_decorator_registration(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("before_model_request")
            return request_context

        @hooks.on.after_model_request
        async def log_response(
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            call_log.append("after_model_request")
            return response

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert call_log == ["before_model_request", "after_model_request"]

    async def test_constructor_form(self):
        call_log: list[str] = []

        async def log_request(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("before_model_request")
            return request_context

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[Hooks(before_model_request=log_request)],
        )
        await agent.run("hello")
        assert call_log == ["before_model_request"]

    async def test_multiple_hooks_same_event(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def first(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("first")
            return request_context

        @hooks.on.before_model_request
        async def second(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("second")
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert call_log == ["first", "second"]

    async def test_tool_names_filtering(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_tool_execute(tools=["target_tool"])
        async def filtered(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            call_log.append(f"filtered:{call.tool_name}")
            return args

        @hooks.on.after_tool_execute
        async def unfiltered(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            result: Any,
        ) -> Any:
            call_log.append(f"unfiltered:{call.tool_name}")
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def target_tool() -> str:
            return "result"

        await agent.run("call tool")
        assert "filtered:target_tool" in call_log
        assert "unfiltered:target_tool" in call_log

    async def test_wrap_model_request(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.model_request
        async def wrap(
            ctx: RunContext[Any], *, request_context: ModelRequestContext, handler: Any
        ) -> ModelResponse:
            call_log.append("wrap_start")
            result = await handler(request_context)
            call_log.append("wrap_end")
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert call_log == ["wrap_start", "wrap_end"]

    async def test_wrap_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.run
        async def wrap(ctx: RunContext[Any], *, handler: Any) -> AgentRunResult[Any]:
            call_log.append("wrap_run_start")
            result = await handler()
            call_log.append("wrap_run_end")
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert call_log == ["wrap_run_start", "wrap_run_end"]

    async def test_on_error_recovery(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def recover(
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            error: Exception,
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="recovered")])

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run("hello")
        assert result.output == "recovered"

    async def test_sync_function_auto_wrapping(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        def sync_hook(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("sync_hook")
            return request_context

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert call_log == ["sync_hook"]

    async def test_timeout(self):
        hooks = Hooks()

        @hooks.on.before_model_request(timeout=0.01)
        async def slow_hook(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            await asyncio.sleep(10)
            return request_context  # pragma: no cover

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        with pytest.raises(HookTimeoutError) as exc_info:
            await agent.run("hello")
        assert exc_info.value.hook_name == "before_model_request"
        assert exc_info.value.func_name == "slow_hook"
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
        await agent.run("hello")
        assert len(nodes_seen) > 0

    async def test_composition_with_other_capabilities(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def hooks_before(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("hooks_before")
            return request_context

        cap = LoggingCapability()
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks, cap])
        await agent.run("hello")
        assert "hooks_before" in call_log
        assert "before_model_request" in cap.log

    async def test_before_run(self):
        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_run
        async def on_start(ctx: RunContext[Any]) -> None:
            call_log.append("before_run")

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert call_log == ["before_run"]

    async def test_after_run(self):
        hooks = Hooks()
        outputs: list[str] = []

        @hooks.on.after_run
        async def on_end(
            ctx: RunContext[Any], *, result: AgentRunResult[Any]
        ) -> AgentRunResult[Any]:
            outputs.append(result.output)
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        result = await agent.run("hello")
        assert outputs == [result.output]

    async def test_repr(self):
        hooks = Hooks()
        assert repr(hooks) == "Hooks({})"

        @hooks.on.before_model_request
        async def hook(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            return request_context

        assert repr(hooks) == "Hooks({'before_model_request': 1})"

        # Verify the registered hook actually works
        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")

    async def test_on_model_request_error_reraise(self):
        """Error hooks that re-raise propagate the error to the caller."""

        hooks = Hooks()

        @hooks.on.model_request_error
        async def log_and_reraise(
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            error: Exception,
        ) -> ModelResponse:
            raise error

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")

    async def test_on_run_error_reraise(self):
        """on_run_error hooks that re-raise propagate the error."""

        hooks = Hooks()

        @hooks.on.run_error
        async def log_and_reraise(
            ctx: RunContext[Any], *, error: BaseException
        ) -> AgentRunResult[Any]:
            raise error

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match="model exploded"):
            await agent.run("hello")

    async def test_on_run_error_recovery(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def recover(
            ctx: RunContext[Any], *, error: BaseException
        ) -> AgentRunResult[Any]:
            return AgentRunResult(output="recovered from run error")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run("hello")
        assert result.output == "recovered from run error"

    async def test_on_run_error_chaining(self):
        hooks = Hooks()

        @hooks.on.run_error
        async def first_handler(
            ctx: RunContext[Any], *, error: BaseException
        ) -> AgentRunResult[Any]:
            raise ValueError("transformed by first")

        @hooks.on.run_error
        async def second_handler(
            ctx: RunContext[Any], *, error: BaseException
        ) -> AgentRunResult[Any]:
            return AgentRunResult(output=f"caught: {error}")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("original error")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run("hello")
        assert "transformed by first" in result.output

    async def test_error_hook_chaining(self):
        hooks = Hooks()

        @hooks.on.model_request_error
        async def first(
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            error: Exception,
        ) -> ModelResponse:
            raise ValueError("transformed")

        @hooks.on.model_request_error
        async def second(
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            error: Exception,
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content=f"recovered: {error}")])

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("original")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        result = await agent.run("hello")
        assert "transformed" in result.output

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
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[hooks],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_hooks_with_streaming_run(self):
        """Hooks capability used during a streaming run exercises the default wrap_run_event_stream path."""

        hooks = Hooks()
        call_log: list[str] = []

        @hooks.on.before_model_request
        async def log_request(
            ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            call_log.append("before_model_request")
            return request_context

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[hooks],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert "before_model_request" in call_log

    async def test_node_run_hooks(self):
        """Exercise before_node_run, after_node_run, and node_run (wrap) via .on namespace."""
        hooks = Hooks()
        nodes_seen: list[str] = []

        @hooks.on.before_node_run
        async def before(ctx: RunContext[Any], *, node: Any) -> Any:
            nodes_seen.append(f"before:{type(node).__name__}")
            return node

        @hooks.on.after_node_run
        async def after(ctx: RunContext[Any], *, node: Any, result: Any) -> Any:
            nodes_seen.append(f"after:{type(node).__name__}")
            return result

        agent = Agent(FunctionModel(simple_model_function), capabilities=[hooks])
        await agent.run("hello")
        assert any("before:" in n for n in nodes_seen)
        assert any("after:" in n for n in nodes_seen)

    async def test_node_run_error_hook(self):
        """on.node_run_error fires when a node fails."""
        hooks = Hooks()
        error_log: list[str] = []

        @hooks.on.node_run_error
        async def handle(ctx: RunContext[Any], *, node: Any, error: Exception) -> Any:
            error_log.append(f"error:{type(error).__name__}")
            raise error

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("node exploded")

        agent = Agent(FunctionModel(failing_model), capabilities=[hooks])
        with pytest.raises(RuntimeError, match="node exploded"):
            await agent.run("hello")
        assert any("error:RuntimeError" in e for e in error_log)

    async def test_on_event_hook(self):
        """on.event fires for each stream event and can modify events."""
        hooks = Hooks()
        events_seen: list[str] = []

        @hooks.on.event
        async def observe(
            ctx: RunContext[Any], event: AgentStreamEvent
        ) -> AgentStreamEvent:
            events_seen.append(type(event).__name__)
            return event

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[hooks],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert len(events_seen) > 0

    async def test_on_event_with_run_event_stream(self):
        """on.event and on.run_event_stream can be used together."""
        hooks = Hooks()
        event_log: list[str] = []
        stream_log: list[str] = []

        @hooks.on.event
        async def per_event(
            ctx: RunContext[Any], event: AgentStreamEvent
        ) -> AgentStreamEvent:
            event_log.append(type(event).__name__)
            return event

        @hooks.on.run_event_stream
        async def wrap_stream(
            ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
        ) -> AsyncIterable[AgentStreamEvent]:
            stream_log.append("started")
            async for event in stream:
                yield event
            stream_log.append("finished")

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[hooks],
        )
        async with agent.run_stream("hello") as stream:
            await stream.get_output()
        assert len(event_log) > 0
        assert stream_log == ["started", "finished"]

    async def test_prepare_tools_hook(self):
        """on.prepare_tools filters tool definitions."""
        hooks = Hooks()

        @hooks.on.prepare_tools
        async def hide_tools(
            ctx: RunContext[Any], tool_defs: list[ToolDefinition]
        ) -> list[ToolDefinition]:
            return [td for td in tool_defs if not td.name.startswith("hidden_")]

        tool_called = False

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def visible_tool() -> str:
            nonlocal tool_called
            tool_called = True
            return "visible"

        @agent.tool_plain
        def hidden_tool() -> str:
            return "hidden"  # pragma: no cover

        await agent.run("call tool")
        assert tool_called

    async def test_tool_validate_hooks(self):
        """Exercise before/after/wrap tool_validate and on_tool_validate_error."""
        hooks = Hooks()
        validate_log: list[str] = []

        @hooks.on.before_tool_validate
        async def before_validate(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: Any,
        ) -> Any:
            validate_log.append("before_validate")
            return args

        @hooks.on.after_tool_validate
        async def after_validate(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
        ) -> dict[str, Any]:
            validate_log.append("after_validate")
            return args

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"

        await agent.run("call tool")
        assert "before_validate" in validate_log
        assert "after_validate" in validate_log

    async def test_wrap_tool_validate_hook(self):
        """Exercise on.tool_validate (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_validate
        async def wrap_validate(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: Any,
            handler: Any,
        ) -> dict[str, Any]:
            wrap_log.append("wrap_start")
            result = await handler(args)
            wrap_log.append("wrap_end")
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"

        await agent.run("call tool")
        assert wrap_log == ["wrap_start", "wrap_end"]

    async def test_tool_validate_error_hook(self):
        """on.tool_validate_error can recover from validation failures."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def recover_validate(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: Any,
            error: Any,
        ) -> dict[str, Any]:
            return {"name": "recovered"}

        def bad_args_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                tool = info.function_tools[0]
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=tool.name,
                            args='{"wrong": 1}',
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f"hello {name}"

        result = await agent.run("greet someone")
        assert "hello recovered" in result.output

    async def test_wrap_tool_execute_hook(self):
        """Exercise on.tool_execute (wrap) via decorator."""
        hooks = Hooks()
        wrap_log: list[str] = []

        @hooks.on.tool_execute
        async def wrap_exec(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: dict[str, Any],
            handler: Any,
        ) -> Any:
            wrap_log.append("exec_start")
            result = await handler(args)
            wrap_log.append("exec_end")
            return result

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            return "result"

        await agent.run("call tool")
        assert wrap_log == ["exec_start", "exec_end"]

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
            return "fallback result"

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError("tool failed")

        result = await agent.run("call tool")
        assert "fallback result" in result.output

    async def test_tool_validate_error_reraise(self):
        """on.tool_validate_error that re-raises propagates the error."""
        hooks = Hooks()

        @hooks.on.tool_validate_error
        async def reraise(
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: Any,
            error: Any,
        ) -> dict[str, Any]:
            raise error

        call_count = 0

        def bad_args_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        return make_text_response(f"got: {part.content}")
            if info.function_tools:
                tool = info.function_tools[0]
                if call_count <= 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name=tool.name,
                                args='{"wrong": 1}',
                                tool_call_id="call-1",
                            )
                        ]
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=tool.name,
                            args='{"name": "ok"}',
                            tool_call_id="call-2",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

        agent = Agent(FunctionModel(bad_args_model), capabilities=[hooks])

        @agent.tool_plain
        def greet(name: str) -> str:
            return f"hello {name}"

        await agent.run("greet someone")

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
            raise ValueError("tool failed")

        with pytest.raises(ValueError, match="tool failed"):
            await agent.run("call tool")

    async def test_get_serialization_name(self):
        assert Hooks.get_serialization_name() is None

    async def test_default_on_tool_execute_error_reraises(self):
        """The default on_tool_execute_error just re-raises, exercised with a minimal capability."""

        @dataclass
        class MinimalCap(AbstractCapability[Any]):
            """Capability that doesn't override error hooks."""

            def get_instructions(self):
                return "Be helpful."

        agent = Agent(FunctionModel(tool_calling_model), capabilities=[MinimalCap()])

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError("tool failed")

        with pytest.raises(ValueError, match="tool failed"):
            await agent.run("call the tool")


# --- Context var propagation tests ---

_test_cv: contextvars.ContextVar[str] = contextvars.ContextVar("_test_cv")


class TestContextVarPropagation:
    """Context vars set in wrap_run propagate to all hooks in the outer task."""

    async def test_wrap_run_contextvar_visible_in_node_hooks(self):
        """A capability that sets a contextvar in wrap_run should have it
        visible in another capability's node-level hooks via agent.run()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                token = _test_cv.set("from-wrap-run")
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(("before_node_run", _test_cv.get(None)))
                return node

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                self.seen.append(("wrap_node_run", _test_cv.get(None)))
                return await handler(node)

            async def after_node_run(
                self, ctx: RunContext[Any], *, node: Any, result: Any
            ) -> Any:
                self.seen.append(("after_node_run", _test_cv.get(None)))
                return result

            async def after_run(
                self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
            ) -> AgentRunResult[Any]:
                self.seen.append(("after_run", _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run("hello")

        for hook_name, value in reader.seen:
            assert value == "from-wrap-run", f"{hook_name} did not see contextvar"

    async def test_wrap_run_contextvar_visible_via_iter_next(self):
        """Context vars set in wrap_run are visible when using agent.iter() + next()."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                token = _test_cv.set("from-iter")
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(("before_node_run", _test_cv.get(None)))
                return node

            async def after_run(
                self, ctx: RunContext[Any], *, result: AgentRunResult[Any]
            ) -> AgentRunResult[Any]:
                self.seen.append(("after_run", _test_cv.get(None)))
                return result

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])

        async with agent.iter("hello") as agent_run:
            node = agent_run.next_node
            while not isinstance(node, End):
                node = await agent_run.next(node)

        for hook_name, value in reader.seen:
            assert value == "from-iter", f"{hook_name} did not see contextvar"

    async def test_contextvar_cleaned_up_after_run(self):
        """Context vars set in wrap_run are restored after the run completes."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                token = _test_cv.set("temporary")
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        await agent.run("hello")

        # After the run, the contextvar should be cleaned up
        assert _test_cv.get(None) is None

    async def test_contextvar_cleaned_up_on_early_iter_exit(self):
        """Context vars are restored even when the caller exits iter() early."""

        @dataclass
        class Setter(AbstractCapability):
            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                token = _test_cv.set("early-exit")
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

        agent = Agent(TestModel(), capabilities=[Setter()])
        assert _test_cv.get(None) is None

        async with agent.iter("hello") as agent_run:
            # Exit immediately without driving any nodes
            _ = agent_run.next_node

        # Context var must be cleaned up even though we abandoned the run
        assert _test_cv.get(None) is None

    async def test_before_run_contextvar_propagates(self):
        """Context vars set in before_run (not wrap_run) also propagate."""

        @dataclass
        class Setter(AbstractCapability):
            async def before_run(self, ctx: RunContext[Any]) -> None:
                _test_cv.set("from-before-run")

        @dataclass
        class Reader(AbstractCapability):
            seen: list[tuple[str, str | None]] = field(default_factory=lambda: [])

            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                self.seen.append(("before_node_run", _test_cv.get(None)))
                return node

        reader = Reader()
        agent = Agent(TestModel(), capabilities=[Setter(), reader])
        await agent.run("hello")

        for hook_name, value in reader.seen:
            assert value == "from-before-run", f"{hook_name} did not see contextvar"

    async def test_contextvar_visible_in_on_run_error(self):
        """Context vars set in wrap_run are visible in on_run_error."""

        @dataclass
        class SetterWithRecovery(AbstractCapability):
            seen_in_error: str | None = None

            async def wrap_run(
                self, ctx: RunContext[Any], *, handler: Any
            ) -> AgentRunResult[Any]:
                token = _test_cv.set("error-path")
                try:
                    return await handler()
                finally:
                    _test_cv.reset(token)

            async def on_run_error(
                self, ctx: RunContext[Any], *, error: BaseException
            ) -> AgentRunResult[Any]:
                self.seen_in_error = _test_cv.get(None)
                return AgentRunResult(output="recovered")

        def failing_model(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("model exploded")

        cap = SetterWithRecovery()
        agent = Agent(FunctionModel(failing_model), capabilities=[cap])
        result = await agent.run("hello")

        assert result.output == "recovered"
        assert cap.seen_in_error == "error-path"


# --- WrapperCapability and PrefixTools tests ---


async def test_prefix_tools_prefixes_wrapped_capability_tools():
    """PrefixTools prefixes only the wrapped capability's tools, not other agent tools."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def inner_tool() -> str:
        return "inner"  # pragma: no cover

    cap = PrefixTools(wrapped=Toolset(toolset), prefix="ns")

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(",".join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])

    @agent.tool_plain
    def outer_tool() -> str:
        return "outer"  # pragma: no cover

    result = await agent.run("list tools")
    # inner_tool should be prefixed, outer_tool should not
    assert result.output == "ns_inner_tool,outer_tool"


async def test_prefix_tools_from_spec():
    """PrefixTools from spec supports both dict-form and bare-name nested capabilities."""

    # Dict form (kwargs): nested capability with arguments
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [
                {
                    "PrefixTools": {
                        "prefix": "search",
                        "capability": {"BuiltinTool": {"kind": "web_search"}},
                    }
                },
            ],
        },
    )
    assert agent.model is not None

    # Bare name form with custom_capability_types forwarded through contextvar
    agent = Agent.from_spec(
        {
            "model": "test",
            "capabilities": [
                {
                    "PrefixTools": {
                        "prefix": "custom",
                        "capability": "CustomCapability",
                    }
                },
            ],
        },
        custom_capability_types=[CustomCapability],
    )
    assert agent.model is not None


async def test_prefix_tools_from_spec_direct():
    """PrefixTools.from_spec works outside Agent.from_spec (no contextvar), using default registry."""
    cap = PrefixTools.from_spec(prefix="ws", capability="WebSearch")  # pyright: ignore[reportArgumentType]
    assert isinstance(cap, PrefixTools)
    assert cap.prefix == "ws"


async def test_prefix_tools_returns_none_when_no_toolset():
    """PrefixTools.get_toolset() returns None if the wrapped capability has no toolset."""
    cap = PrefixTools(wrapped=CustomCapability(), prefix="ns")
    assert cap.get_toolset() is None


async def test_prefix_tools_with_callable_toolset():
    """PrefixTools handles a wrapped capability that returns a callable toolset."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def dynamic_tool() -> str:
        return "dynamic"  # pragma: no cover

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        return toolset

    cap = PrefixTools(wrapped=Toolset(toolset_func), prefix="dyn")

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(",".join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run("list tools")
    assert result.output == "dyn_dynamic_tool"


async def test_prefix_tools_convenience_method():
    """AbstractCapability.prefix_tools() returns a PrefixTools wrapping self."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def inner_tool() -> str:
        return "inner"  # pragma: no cover

    cap = Toolset(toolset).prefix_tools("ns")
    assert isinstance(cap, PrefixTools)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_names = sorted(t.name for t in info.function_tools)
        return ModelResponse(parts=[TextPart(",".join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run("list tools")
    assert result.output == "ns_inner_tool"


async def test_wrapper_capability_delegates_hooks():
    """WrapperCapability delegates lifecycle hooks to the wrapped capability."""
    hook_calls: list[str] = []

    @dataclass
    class HookCap(AbstractCapability[None]):
        async def before_run(self, ctx: RunContext[None]) -> None:
            hook_calls.append("before_run")

        async def after_run(
            self, ctx: RunContext[None], *, result: AgentRunResult[Any]
        ) -> AgentRunResult[Any]:
            hook_calls.append("after_run")
            return result

    wrapper = WrapperCapability(wrapped=HookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run("Hello")

    assert "before_run" in hook_calls
    assert "after_run" in hook_calls


async def test_wrapper_capability_for_run_replaces():
    """WrapperCapability.for_run replaces wrapped when it changes."""
    toolset_a = FunctionToolset(id="a")

    @toolset_a.tool_plain
    def tool_a() -> str:
        return "a"  # pragma: no cover

    toolset_b = FunctionToolset(id="b")

    @toolset_b.tool_plain
    def tool_b() -> str:
        return "b"  # pragma: no cover

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
        return ModelResponse(parts=[TextPart(",".join(tool_names))])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    result = await agent.run("Hello")
    # for_run switches to toolset_b
    assert "tool_b" in result.output


async def test_wrapper_capability_has_wrap_node_run():
    """WrapperCapability.has_wrap_node_run delegates to the wrapped capability."""
    plain = CustomCapability()
    assert WrapperCapability(wrapped=plain).has_wrap_node_run is False

    @dataclass
    class NodeRunCap(AbstractCapability[None]):
        async def wrap_node_run(
            self, ctx: RunContext[None], *, node: Any, handler: Any
        ) -> Any:
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
            hook_calls.append("before_model_request")
            return request_context

        async def after_model_request(
            self,
            ctx: RunContext[None],
            *,
            request_context: ModelRequestContext,
            response: ModelResponse,
        ) -> ModelResponse:
            hook_calls.append("after_model_request")
            return response

    wrapper = WrapperCapability(wrapped=ModelRequestHookCap())

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), capabilities=[wrapper])
    await agent.run("Hello")

    assert "before_model_request" in hook_calls
    assert "after_model_request" in hook_calls


async def test_prefix_tools_tool_call_strips_prefix():
    """PrefixTools correctly strips the prefix when calling the underlying tool."""
    toolset = FunctionToolset()

    @toolset.tool_plain
    def greet(name: str) -> str:
        return f"hello {name}"

    cap = PrefixTools(wrapped=Toolset(toolset), prefix="ns")

    call_count = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart("ns_greet", {"name": "world"})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(respond), capabilities=[cap])
    result = await agent.run("greet world")
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content="greet world", timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="ns_greet",
                        args={"name": "world"},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name="function:respond:",
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="ns_greet",
                        content="hello world",
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content="done")],
                usage=RequestUsage(input_tokens=54, output_tokens=6),
                model_name="function:respond:",
                timestamp=IsDatetime(),
                run_id=IsStr(),
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
        async def on_run_error(
            self, ctx: RunContext[Any], *, error: BaseException
        ) -> AgentRunResult[Any]:
            return AgentRunResult(output="recovered")

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    agent = Agent(
        FunctionModel(failing_model),
        capabilities=[WrapperCapability(wrapped=RecoverCap())],
    )
    result = await agent.run("hello")
    assert result.output == "recovered"


async def test_wrapper_capability_delegates_on_node_run_error():
    """WrapperCapability delegates on_node_run_error to the wrapped capability."""
    from pydantic_ai.result import FinalResult
    from pydantic_graph import End

    @dataclass
    class NodeRecoverCap(AbstractCapability[Any]):
        async def on_node_run_error(
            self, ctx: RunContext[Any], *, node: Any, error: Exception
        ) -> Any:
            return End(FinalResult(output="node recovered"))

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model exploded")

    agent = Agent(
        FunctionModel(failing_model),
        capabilities=[WrapperCapability(wrapped=NodeRecoverCap())],
    )
    async with agent.iter("hello") as agent_run:
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)
    assert isinstance(node, End)
    assert node.data.output == "node recovered"


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

    async def handler(
        _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
    ) -> None:
        async for _ in stream:
            pass

    await agent.run("hello", event_stream_handler=handler)
    assert len(observed_events) > 0


async def test_wrapper_capability_delegates_on_model_request_error():
    """WrapperCapability delegates on_model_request_error to the wrapped capability."""

    @dataclass
    class ModelErrorRecoverCap(AbstractCapability[Any]):
        async def on_model_request_error(
            self,
            ctx: RunContext[Any],
            *,
            request_context: ModelRequestContext,
            error: Exception,
        ) -> ModelResponse:
            return ModelResponse(parts=[TextPart(content="recovered from model error")])

    def failing_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError("model request failed")

    agent = Agent(
        FunctionModel(failing_model),
        capabilities=[WrapperCapability(wrapped=ModelErrorRecoverCap())],
    )
    result = await agent.run("hello")
    assert result.output == "recovered from model error"


async def test_wrapper_capability_delegates_on_tool_validate_error():
    """WrapperCapability delegates on_tool_validate_error to the wrapped capability."""

    @dataclass
    class ValidateErrorCap(AbstractCapability[Any]):
        async def on_tool_validate_error(
            self,
            ctx: RunContext[Any],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: Any,
            error: Any,
        ) -> dict[str, Any]:
            # Recover by providing valid args
            return {"x": 1}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    return ModelResponse(parts=[TextPart(content="done")])
        if info.function_tools:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=info.function_tools[0].name, args="invalid json!!"
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="no tools")])  # pragma: no cover

    agent = Agent(
        FunctionModel(model_fn),
        capabilities=[WrapperCapability(wrapped=ValidateErrorCap())],
    )

    @agent.tool_plain
    def my_tool(x: int) -> str:
        return f"result: {x}"

    result = await agent.run("call tool")
    assert result.output == "done"


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
            return "recovered tool result"

    agent = Agent(
        FunctionModel(tool_calling_model),
        capabilities=[WrapperCapability(wrapped=ExecuteErrorCap())],
    )

    @agent.tool_plain
    def my_tool() -> str:
        raise ValueError("tool failed")

    result = await agent.run("call tool")
    assert result.output == "final response"


# --- Tests for double-execution bug fix (streaming + before_node_run replacement) ---


class TestNodeStreamingWithHooks:
    """Tests that node streaming with event_stream_handler doesn't cause double model execution
    when before_node_run replaces a node."""

    async def test_before_node_run_replacement_no_double_execution(self):
        """When before_node_run replaces a ModelRequestNode and event_stream_handler is set,
        the model should be called exactly once (not twice)."""
        model_call_count = 0

        async def counting_stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield "streamed response"

        cap = _ReplacingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=counting_stream),
            capabilities=[cap],
        )

        events_received: list[AgentStreamEvent] = []

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for event in stream:
                events_received.append(event)

        result = await agent.run("hello", event_stream_handler=handler)
        assert result.output == "streamed response"
        assert model_call_count == 1, (
            f"Model was called {model_call_count} times, expected 1"
        )
        assert len(events_received) > 0

    async def test_hook_ordering_with_event_stream_handler(self):
        """before_node_run fires BEFORE streaming events, wrap_node_run wraps the streaming,
        and after_node_run fires after graph advancement."""
        log: list[str] = []

        @dataclass
        class OrderTrackingCapability(AbstractCapability[Any]):
            async def before_node_run(self, ctx: RunContext[Any], *, node: Any) -> Any:
                log.append(f"before:{type(node).__name__}")
                return node

            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                log.append(f"wrap:enter:{type(node).__name__}")
                result = await handler(node)
                log.append(f"wrap:exit:{type(node).__name__}")
                return result

            async def after_node_run(
                self, ctx: RunContext[Any], *, node: Any, result: Any
            ) -> Any:
                log.append(f"after:{type(node).__name__}")
                return result

        agent = Agent(
            FunctionModel(
                simple_model_function, stream_function=simple_stream_function
            ),
            capabilities=[OrderTrackingCapability()],
        )

        async def handler(
            _ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]
        ) -> None:
            async for _ in stream:
                pass
            log.append("stream:consumed")

        await agent.run("hello", event_stream_handler=handler)

        # For ModelRequestNode: before → wrap:enter → stream:consumed → wrap:exit → after
        mr_before = log.index("before:ModelRequestNode")
        mr_wrap_enter = log.index("wrap:enter:ModelRequestNode")
        stream_consumed_idx = log.index("stream:consumed")
        mr_wrap_exit = log.index("wrap:exit:ModelRequestNode")
        mr_after = log.index("after:ModelRequestNode")
        assert mr_before < mr_wrap_enter < stream_consumed_idx < mr_wrap_exit < mr_after

    async def test_run_stream_before_node_run_replacement_no_double_execution(self):
        """Same as the run() test but for run_stream(): before_node_run replacement
        should not cause double model execution."""
        model_call_count = 0

        async def counting_stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            nonlocal model_call_count
            model_call_count += 1
            yield "streamed response"

        cap = _ReplacingCapability()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=counting_stream),
            capabilities=[cap],
        )

        async with agent.run_stream("hello") as streamed:
            output = await streamed.get_output()

        assert output == "streamed response"
        assert model_call_count == 1, (
            f"Model was called {model_call_count} times, expected 1"
        )

    async def test_on_node_run_error_fires_in_run_stream(self):
        """on_node_run_error in run_stream() fires when wrap_node_run raises during graph advancement."""
        error_log: list[str] = []

        @dataclass
        class WrapErrorCap(AbstractCapability[Any]):
            async def wrap_node_run(
                self, ctx: RunContext[Any], *, node: Any, handler: Any
            ) -> Any:
                # Raise on CallToolsNode — after UserPromptNode and ModelRequestNode pass through.
                # ModelRequestNode with tool calls doesn't produce a FinalResultEvent in run_stream(),
                # so it falls through to wrap_node_run; CallToolsNode is next and triggers the error.
                from pydantic_ai._agent_graph import CallToolsNode

                if isinstance(node, CallToolsNode):
                    raise RuntimeError("wrap error")
                return await handler(node)

            async def on_node_run_error(
                self, ctx: RunContext[Any], *, node: Any, error: Exception
            ) -> Any:
                error_log.append(type(node).__name__)
                raise error

        agent = Agent(
            FunctionModel(
                tool_calling_model, stream_function=tool_calling_stream_function
            ),
            capabilities=[WrapErrorCap()],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        with pytest.raises(RuntimeError, match="wrap error"):
            async with agent.run_stream("hello") as _streamed:
                pass

        assert error_log == ["CallToolsNode"]


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
                return make_text_response("bad response")
            return make_text_response("good response")

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
                    raise ModelRetry("Response was bad, please try again")
                return response

        cap = RetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run("hello")
        assert result.output == "good response"
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="bad response")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Response was bad, please try again",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="good response")],
                    usage=RequestUsage(input_tokens=66, output_tokens=4),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_after_model_request_model_retry_max_retries(self):
        """after_model_request raises ModelRetry repeatedly — hits max_result_retries."""

        @dataclass
        class AlwaysRetryCap(AbstractCapability[Any]):
            async def after_model_request(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                response: ModelResponse,
            ) -> ModelResponse:
                raise ModelRetry("always bad")

        agent = Agent(
            FunctionModel(simple_model_function),
            capabilities=[AlwaysRetryCap()],
            output_retries=2,
        )
        with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum retries"):
            await agent.run("hello")

    async def test_after_model_request_model_retry_streaming(self):
        """after_model_request raises ModelRetry during streaming with tool calls — model is called again."""
        call_count = 0

        async def stream_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | DeltaToolCalls]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: return a tool call that after_model_request will reject
                yield {
                    0: DeltaToolCall(
                        name="my_tool", json_args="{}", tool_call_id="call-1"
                    )
                }
            elif call_count == 2:
                # Second call (after retry): return text
                yield "good response"
            else:
                yield "unexpected"  # pragma: no cover

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
                    raise ModelRetry("Response was bad, please try again")
                return response

        cap = RetryCap()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=stream_fn),
            capabilities=[cap],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"  # pragma: no cover

        async with agent.run_stream("hello") as streamed:
            result = await streamed.get_output()
        assert result == "good response"
        assert call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name="function:simple_model_function:stream_fn",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Response was bad, please try again",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="good response")],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name="function:simple_model_function:stream_fn",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_streaming_short_circuit(self):
        """wrap_model_request raises ModelRetry without calling handler during streaming."""

        async def stream_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            yield "good response"

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
                    raise ModelRetry("Short-circuit retry")
                return await handler(request_context)

        cap = ShortCircuitRetryCap()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=stream_fn),
            capabilities=[cap],
        )
        async with agent.run_stream("hello") as streamed:
            result = await streamed.get_output()
        assert result == "good response"
        assert cap.call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Short-circuit retry",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="good response")],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name="function:simple_model_function:stream_fn",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_wrap_model_request_model_retry_streaming_after_handler(self):
        """wrap_model_request raises ModelRetry after calling handler during streaming (tool call scenario)."""
        call_count = 0

        async def stream_fn(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | DeltaToolCalls]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: tool call that wrap hook will reject
                yield {
                    0: DeltaToolCall(
                        name="my_tool", json_args="{}", tool_call_id="call-1"
                    )
                }
            else:
                yield "good response"

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
                    raise ModelRetry("Post-handler retry")
                return response

        cap = AfterHandlerRetryCap()
        agent = Agent(
            FunctionModel(simple_model_function, stream_function=stream_fn),
            capabilities=[cap],
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"  # pragma: no cover

        async with agent.run_stream("hello") as streamed:
            result = await streamed.get_output()
        assert result == "good response"
        assert call_count == 2
        assert streamed.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name="function:simple_model_function:stream_fn",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Post-handler retry",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="good response")],
                    usage=RequestUsage(input_tokens=50, output_tokens=2),
                    model_name="function:simple_model_function:stream_fn",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
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
                return make_text_response("first attempt")
            return make_text_response("second attempt")

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
                    raise ModelRetry("Wrap says retry")
                return response

        cap = WrapRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run("hello")
        assert result.output == "second attempt"
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="first attempt")],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Wrap says retry",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="second attempt")],
                    usage=RequestUsage(input_tokens=63, output_tokens=4),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
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
                raise ModelRetry("retry please")

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
            FunctionModel(simple_model_function),
            capabilities=[WrapRetrySkipErrorCap()],
            output_retries=1,
        )
        with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum retries"):
            await agent.run("hello")
        assert not on_error_called

    async def test_on_model_request_error_model_retry(self):
        """on_model_request_error raises ModelRetry to recover via retry."""
        call_count = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("model failed")
            return make_text_response("recovered response")

        @dataclass
        class ErrorRetryCap(AbstractCapability[Any]):
            async def on_model_request_error(
                self,
                ctx: RunContext[Any],
                *,
                request_context: ModelRequestContext,
                error: Exception,
            ) -> ModelResponse:
                raise ModelRetry("Model failed, please try again")

        cap = ErrorRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])
        result = await agent.run("hello")
        assert result.output == "recovered response"
        assert call_count == 2
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="hello", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Model failed, please try again",
                            tool_call_id=IsStr(),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="recovered response")],
                    usage=RequestUsage(input_tokens=65, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
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
                            return make_text_response(f"got: {part.content}")
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

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
                    raise ModelRetry("Tool result is bad, try again")
                return result

        cap = AfterExecRetryCap()
        agent = Agent(FunctionModel(model_fn), capabilities=[cap])

        @agent.tool_plain
        def my_tool() -> str:
            nonlocal tool_call_count
            tool_call_count += 1
            return "tool result"

        result = await agent.run("call tool")
        assert result.output == "got: tool result"
        assert (
            tool_call_count == 2
        )  # Tool called twice: first rejected by hook, second succeeds
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="call tool", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Tool result is bad, try again",
                            tool_name="my_tool",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=65, output_tokens=4),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="my_tool",
                            content="tool result",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="got: tool result")],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
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
                            return make_text_response(f"got: {part.content}")
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

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
                raise ModelRetry("Not ready to execute, try again")
            return args

        agent = Agent(FunctionModel(model_fn), capabilities=[hooks], retries=2)

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"

        result = await agent.run("call tool")
        assert result.output == "got: tool result"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="call tool", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Not ready to execute, try again",
                            tool_name="my_tool",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=65, output_tokens=4),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="my_tool",
                            content="tool result",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="got: tool result")],
                    usage=RequestUsage(input_tokens=67, output_tokens=7),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
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
                        return make_text_response("got retry")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

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
                raise ModelRetry("Wrap says retry tool")

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

        agent = Agent(
            FunctionModel(model_fn), capabilities=[WrapExecRetryCap()], retries=2
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"  # pragma: no cover

        result = await agent.run("call tool")
        assert result.output == "got retry"
        assert not on_error_called
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="call tool", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Wrap says retry tool",
                            tool_name="my_tool",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="got retry")],
                    usage=RequestUsage(input_tokens=63, output_tokens=4),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_on_tool_execute_error_model_retry(self):
        """on_tool_execute_error raises ModelRetry to recover via retry."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response("got retry after error")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

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
                raise ModelRetry("Tool errored, please retry")

        agent = Agent(
            FunctionModel(model_fn), capabilities=[ErrorRetryCap()], retries=2
        )

        @agent.tool_plain
        def my_tool() -> str:
            raise ValueError("tool failed")

        result = await agent.run("call tool")
        assert result.output == "got retry after error"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="call tool", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Tool errored, please retry",
                            tool_name="my_tool",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="got retry after error")],
                    usage=RequestUsage(input_tokens=63, output_tokens=6),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_after_tool_validate_model_retry(self):
        """after_tool_validate raises ModelRetry — validation retry sent to model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response("got validation retry")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

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
                raise ModelRetry("Validated args are bad")

        agent = Agent(
            FunctionModel(model_fn), capabilities=[AfterValRetryCap()], retries=2
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"  # pragma: no cover

        result = await agent.run("call tool")
        assert result.output == "got validation retry"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="call tool", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Validated args are bad",
                            tool_name="my_tool",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="got validation retry")],
                    usage=RequestUsage(input_tokens=63, output_tokens=5),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_before_tool_validate_model_retry(self):
        """before_tool_validate raises ModelRetry — validation retry sent to model."""

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                for part in msg.parts:
                    if isinstance(part, RetryPromptPart):
                        return make_text_response("got pre-validation retry")
            if info.function_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.function_tools[0].name,
                            args="{}",
                            tool_call_id="call-1",
                        )
                    ]
                )
            return make_text_response("no tools")  # pragma: no cover

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
                raise ModelRetry("Args look bad before validation")

        agent = Agent(
            FunctionModel(model_fn), capabilities=[BeforeValRetryCap()], retries=2
        )

        @agent.tool_plain
        def my_tool() -> str:
            return "tool result"  # pragma: no cover

        result = await agent.run("call tool")
        assert result.output == "got pre-validation retry"
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content="call tool", timestamp=IsDatetime())],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_tool", args="{}", tool_call_id="call-1"
                        )
                    ],
                    usage=RequestUsage(input_tokens=52, output_tokens=2),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content="Args look bad before validation",
                            tool_name="my_tool",
                            tool_call_id="call-1",
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content="got pre-validation retry")],
                    usage=RequestUsage(input_tokens=64, output_tokens=5),
                    model_name="function:model_fn:",
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
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

        agent = Agent(
            FunctionModel(simple_model_function),
            name="hook_test_agent",
            capabilities=[AgentTrackingCap()],
        )
        await agent.run("hello")
        assert hook_agent_names == ["hook_test_agent", "hook_test_agent"]


def test_thread_executor_not_serializable() -> None:
    assert ThreadExecutor.get_serialization_name() is None


async def test_thread_executor_capability() -> None:
    tool_threads: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name="check_thread", args="{}")])

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cap-pool")
    try:
        agent = Agent(
            FunctionModel(model_function), capabilities=[ThreadExecutor(executor)]
        )

        @agent.tool_plain
        def check_thread() -> str:
            tool_threads.append(threading.current_thread().name)
            return "ok"

        result = await agent.run("test")
        assert result.output == "done"
        assert len(tool_threads) == 1
        assert tool_threads[0].startswith("cap-pool")
    finally:
        executor.shutdown(wait=True)


async def test_thread_executor_static_method() -> None:
    tool_threads: list[str] = []

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content="done")])
        return ModelResponse(parts=[ToolCallPart(tool_name="check_thread", args="{}")])

    agent = Agent(FunctionModel(model_function))

    @agent.tool_plain
    def check_thread() -> str:
        tool_threads.append(threading.current_thread().name)
        return "ok"

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="static-pool")
    try:
        with Agent.using_thread_executor(executor):
            result = await agent.run("test")
        assert result.output == "done"
        assert len(tool_threads) == 1
        assert tool_threads[0].startswith("static-pool")
    finally:
        executor.shutdown(wait=True)
