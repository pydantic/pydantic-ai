"""Fallback subagent resolution of dynamic `native=` factories."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent, BinaryImage
from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import await_maybe
from pydantic_ai.capabilities import ImageGeneration, XSearch
from pydantic_ai.common_tools.image_generation import ImageGenerationSubagentTool
from pydantic_ai.common_tools.x_search import XSearchSubagentTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import AbstractNativeTool, ImageGenerationTool, XSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import Tool
from pydantic_ai.usage import RunUsage

pytestmark = [pytest.mark.anyio]


def _run_context(deps: Any = None) -> RunContext[Any]:
    return RunContext(deps=deps, model=TestModel(), usage=RunUsage(), run_step=0)


async def test_xsearch_callable_native_config_is_used_by_fallback(allow_model_requests: None):
    """The fallback subagent resolves callable native config with the outer run context."""
    seen_native_tools: list[list[AbstractNativeTool]] = []

    def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_native_tools.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[TextPart(content='summary of recent tweets')])

    inner_model = FunctionModel(inner_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool})))

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "latest news"}')])

    def native_factory(ctx: RunContext[str]) -> XSearchTool:
        return XSearchTool(allowed_x_handles=[ctx.deps])

    outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
    agent = Agent[str, str](
        outer_model,
        deps_type=str,
        capabilities=[XSearch(native=native_factory, fallback_model=inner_model, include_output=True)],
    )

    result = await agent.run('What is happening on X?', deps='pydantic')

    assert result.output == 'done'
    assert seen_native_tools == [[XSearchTool(allowed_x_handles=['pydantic'], include_output=True)]]


async def test_xsearch_callable_native_factory_invoked_once_on_fallback(allow_model_requests: None):
    """The outer native-path resolution is reused by the fallback subagent."""
    factory_calls: list[str] = []

    def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content='summary of recent tweets')])

    inner_model = FunctionModel(inner_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool})))

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "latest news"}')])

    def native_factory(ctx: RunContext[str]) -> XSearchTool:
        factory_calls.append(ctx.deps)
        return XSearchTool(allowed_x_handles=[ctx.deps])

    outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
    agent = Agent[str, str](
        outer_model,
        deps_type=str,
        capabilities=[XSearch(native=native_factory, fallback_model=inner_model)],
    )

    result = await agent.run('What is happening on X?', deps='pydantic')

    assert result.output == 'done'
    assert factory_calls == ['pydantic']


async def test_xsearch_callable_native_none_then_tool_still_raises(allow_model_requests: None):
    """A factory that omits on the outer resolution cannot later enable the fallback tool."""
    n = 0

    def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError(
            'fallback model should not run when the native factory omitted the tool'
        )  # pragma: no cover

    inner_model = FunctionModel(inner_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool})))

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "latest news"}')])

    def native_factory(ctx: RunContext[Any]) -> XSearchTool | None:
        nonlocal n
        n += 1
        if n == 1:
            return None
        return XSearchTool()  # pragma: no cover

    outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
    agent = Agent(
        outer_model,
        capabilities=[XSearch(native=native_factory, fallback_model=inner_model)],
    )

    with pytest.raises(UserError, match='Native tool factory returned `None`'):
        await agent.run('What is happening on X?')

    assert n == 1


async def test_xsearch_callable_native_pass_through_without_overrides():
    """A factory result is unchanged when the capability has no override fields."""
    factory_tool = XSearchTool(enable_image_understanding=True)

    def native_factory(ctx: RunContext[Any]) -> XSearchTool:
        return factory_tool

    cap = XSearch(native=native_factory, fallback_model='xai:grok-4-1-fast-non-reasoning')
    assert isinstance(cap.local, Tool)
    resolved = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]
    assert callable(resolved)
    assert await await_maybe(resolved(_run_context())) is factory_tool


def test_xsearch_incompatible_native_tool_raises():
    """Invalid static native configuration raises at capability construction."""
    with pytest.raises(UserError, match=r'`native` must be a `XSearchTool` instance'):
        XSearch(
            native=ImageGenerationTool(),  # pyright: ignore[reportArgumentType]
            fallback_model='xai:grok-4-1-fast-non-reasoning',
        )


async def test_xsearch_callable_native_wrong_tool_type_raises():
    """The shared resolver validates dynamic factory results before applying overrides."""

    def native_factory(ctx: RunContext[Any]) -> ImageGenerationTool:
        return ImageGenerationTool()

    cap = XSearch(
        native=native_factory,  # pyright: ignore[reportArgumentType]
        fallback_model='xai:grok-4-1-fast-non-reasoning',
    )
    resolved = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]
    assert callable(resolved)
    with pytest.raises(UserError, match=r'must resolve to a `XSearchTool` instance'):
        await await_maybe(resolved(_run_context()))


async def test_xsearch_callable_native_none_raises(allow_model_requests: None):
    """A callable native factory returning None raises rather than enabling default X search."""

    def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('fallback model should not run when the native factory returns None')  # pragma: no cover

    inner_model = FunctionModel(inner_model_fn, profile=ModelProfile(supported_native_tools=frozenset({XSearchTool})))

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='x_search', args='{"query": "latest news"}')])

    def native_factory(ctx: RunContext[Any]) -> None:
        return None

    outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
    agent = Agent(
        outer_model,
        capabilities=[XSearch(native=native_factory, fallback_model=inner_model, include_output=True)],
    )
    with pytest.raises(UserError, match='returned `None`'):
        await agent.run('What is happening on X?')


def test_xsearch_native_false_keeps_fallback_overrides():
    """Disabling the outer native tool retains fallback-native configuration."""
    cap = XSearch(native=False, fallback_model='xai:grok-4-1-fast-non-reasoning', include_output=True)

    assert cap.get_native_tools() == []
    assert cap._resolved_native() == XSearchTool(include_output=True)  # pyright: ignore[reportPrivateUsage]


async def test_xsearch_subagent_dynamic_native_none_raises():
    """The subagent raises when its dynamic factory returns None instead of enabling default X search."""

    def native_factory(ctx: RunContext[Any]) -> None:
        return None

    subagent = XSearchSubagentTool(model='xai:grok-4-1-fast-non-reasoning', native_tool=native_factory)
    with pytest.raises(UserError, match='returned `None`'):
        await subagent(_run_context(), 'latest news')


async def test_image_generation_callable_native_config_is_used_by_fallback(allow_model_requests: None):
    """The fallback subagent resolves callable native config with the outer run context."""
    seen_native_tools: list[list[AbstractNativeTool]] = []

    def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_native_tools.append(info.model_request_parameters.native_tools)
        return ModelResponse(parts=[FilePart(content=BinaryImage(data=b'png', media_type='image/png'))])

    inner_model = FunctionModel(
        inner_model_fn,
        profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool}), supports_image_output=True),
    )

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

    async def native_factory(ctx: RunContext[str]) -> ImageGenerationTool:
        return ImageGenerationTool(model=ctx.deps, quality='high')

    outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
    agent = Agent[str, str](
        outer_model,
        deps_type=str,
        capabilities=[ImageGeneration(native=native_factory, fallback_model=inner_model, output_format='jpeg')],
    )

    result = await agent.run('Generate an image', deps='gpt-image-2')

    assert result.output == 'done'
    assert seen_native_tools == [[ImageGenerationTool(model='gpt-image-2', quality='high', output_format='jpeg')]]


async def test_image_generation_callable_native_pass_through_without_overrides():
    """A factory result is unchanged when the capability has no override fields."""
    factory_tool = ImageGenerationTool(quality='high')

    def native_factory(ctx: RunContext[Any]) -> ImageGenerationTool:
        return factory_tool

    cap = ImageGeneration(native=native_factory, fallback_model='openai-responses:gpt-5.4')
    assert isinstance(cap.local, Tool)
    resolved = cap._resolved_native()  # pyright: ignore[reportPrivateUsage]
    assert callable(resolved)
    assert await await_maybe(resolved(_run_context())) is factory_tool


async def test_image_generation_callable_native_none_raises(allow_model_requests: None):
    """A callable native factory returning None raises rather than enabling default image generation."""

    def inner_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise AssertionError('fallback model should not run when the native factory returns None')  # pragma: no cover

    inner_model = FunctionModel(
        inner_model_fn,
        profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool}), supports_image_output=True),
    )

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(tool_name='generate_image', args='{"prompt": "test"}')])

    def native_factory(ctx: RunContext[Any]) -> None:
        return None

    outer_model = FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=frozenset()))
    agent = Agent(
        outer_model,
        capabilities=[ImageGeneration(native=native_factory, fallback_model=inner_model, output_format='jpeg')],
    )
    with pytest.raises(UserError, match='returned `None`'):
        await agent.run('Generate an image')


def test_image_generation_native_false_keeps_fallback_overrides():
    """Disabling the outer native tool retains fallback-native configuration."""
    cap = ImageGeneration(
        native=False,
        fallback_model='openai-responses:gpt-5.4',
        output_format='jpeg',
    )

    assert cap.get_native_tools() == []
    assert cap._resolved_native() == ImageGenerationTool(output_format='jpeg')  # pyright: ignore[reportPrivateUsage]


async def test_image_generation_subagent_dynamic_native_none_raises():
    """The subagent raises when its dynamic factory returns None instead of enabling default image generation."""

    def native_factory(ctx: RunContext[Any]) -> None:
        return None

    subagent = ImageGenerationSubagentTool(model='openai-responses:gpt-5.4', native_tool=native_factory)
    with pytest.raises(UserError, match='returned `None`'):
        await subagent(_run_context(), 'test')
