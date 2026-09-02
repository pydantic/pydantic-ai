"""Fallback subagent resolution of dynamic `native=` factories.

These are unit tests rather than VCR tests because what they assert —
`info.model_request_parameters.native_tools`, the native-tool objects the subagent hands its model —
is internal to the request build and never reaches the wire, so a cassette could not pin it. The
end-to-end wire proof for this feature is
`tests/test_capability_native_or_local.py::TestImageGenerationCapability::test_image_generation_local_fallback`,
which records a real OpenAI image-generation call and snapshots the outgoing `tools` payload.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, BinaryImage
from pydantic_ai.capabilities import ImageGeneration, NativeOrLocalTool, XSearch
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
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.native_tools import AbstractNativeTool, ImageGenerationTool, XSearchTool
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import RunContext

from .capability_models import build_run_context

pytestmark = [pytest.mark.anyio]


def _none_native_factory(ctx: RunContext[str]) -> None:
    """Omits the native tool: legal on the native path, but the invoked fallback cannot honor it."""
    return None


def _xsearch_from_deps(ctx: RunContext[str]) -> XSearchTool:
    return XSearchTool(allowed_x_handles=[ctx.deps])


_XSEARCH_PASS_THROUGH = XSearchTool(enable_image_understanding=True)


def _xsearch_pass_through(ctx: RunContext[str]) -> XSearchTool:
    return _XSEARCH_PASS_THROUGH


async def _image_generation_from_deps(ctx: RunContext[str]) -> ImageGenerationTool:
    return ImageGenerationTool(model=ctx.deps, quality='high')


_IMAGE_GENERATION_PASS_THROUGH = ImageGenerationTool(quality='high')


def _image_generation_pass_through(ctx: RunContext[str]) -> ImageGenerationTool:
    return _IMAGE_GENERATION_PASS_THROUGH


@dataclass(frozen=True, kw_only=True)
class Case:
    """A `NativeOrLocalTool` subclass that routes native-tool configuration into a fallback subagent.

    The builder fields each take the fallback model and return the capability under test, so a case
    reads as the table of `native=` shapes the capability has to handle.
    """

    id: str
    prompt: str
    deps: str
    """Outer-run dependency the dynamic native factory reads."""
    tool_name: str
    tool_args: str
    """JSON arguments the outer model calls the local fallback tool with."""
    fallback_profile: ModelProfile
    """Profile of the subagent's model: it supports the native tool the outer model lacks."""
    make_fallback_response: Callable[[], ModelResponse]
    with_deps_factory: Callable[[Model], NativeOrLocalTool[str]]
    expected_fallback_native_tools: list[AbstractNativeTool]
    """What the subagent's model is given: the factory result plus the capability-level override."""
    with_pass_through_factory: Callable[[Model], NativeOrLocalTool[str]]
    pass_through_tool: AbstractNativeTool
    """The exact instance `with_pass_through_factory`'s factory returns."""
    with_none_factory: Callable[[Model], NativeOrLocalTool[str]]
    with_native_false: Callable[[Model], NativeOrLocalTool[str]]
    expected_override_only_native_tools: list[AbstractNativeTool]
    """What the subagent's model is given when the capability's own fields are the only config."""
    with_instance_and_overrides: Callable[[Model], NativeOrLocalTool[str]]
    expected_instance_native_tools: list[AbstractNativeTool]
    """What the subagent's model is given for a static `native=` instance plus capability overrides."""
    native_tool_type: type[AbstractNativeTool]
    """The native tool this capability configures, for building an outer model that supports it."""
    subagent: Callable[[RunContext[str], str], Awaitable[object]]
    """The capability's subagent tool, built directly with a `None`-returning factory."""
    subagent_input: str = 'a query'


XSEARCH_CASE = Case(
    id='x_search',
    prompt='What is happening on X?',
    deps='pydantic',
    tool_name='x_search',
    tool_args='{"query": "latest news"}',
    fallback_profile=ModelProfile(supported_native_tools=frozenset({XSearchTool})),
    make_fallback_response=lambda: ModelResponse(parts=[TextPart(content='summary of recent tweets')]),
    with_deps_factory=lambda fallback_model: XSearch[str](
        native=_xsearch_from_deps, fallback_model=fallback_model, include_output=True
    ),
    expected_fallback_native_tools=snapshot([XSearchTool(allowed_x_handles=['pydantic'], include_output=True)]),
    with_pass_through_factory=lambda fallback_model: XSearch[str](
        native=_xsearch_pass_through, fallback_model=fallback_model
    ),
    pass_through_tool=_XSEARCH_PASS_THROUGH,
    with_none_factory=lambda fallback_model: XSearch[str](
        native=_none_native_factory, fallback_model=fallback_model, include_output=True
    ),
    with_native_false=lambda fallback_model: XSearch[str](
        native=False, fallback_model=fallback_model, include_output=True
    ),
    expected_override_only_native_tools=snapshot([XSearchTool(include_output=True)]),
    with_instance_and_overrides=lambda fallback_model: XSearch[str](
        native=XSearchTool(allowed_x_handles=['a'], enable_image_understanding=True),
        fallback_model=fallback_model,
        include_output=True,
    ),
    expected_instance_native_tools=snapshot(
        [XSearchTool(allowed_x_handles=['a'], enable_image_understanding=True, include_output=True)]
    ),
    native_tool_type=XSearchTool,
    # The narrowed `native_tool` type rejects a `None`-returning factory; this case exists to
    # prove the runtime `UserError` still fires for callers who bypass the type checker.
    subagent=XSearchSubagentTool(
        model='xai:grok-4-1-fast-non-reasoning',
        native_tool=_none_native_factory,  # pyright: ignore[reportArgumentType]
    ),
)

IMAGE_GENERATION_CASE = Case(
    id='image_generation',
    prompt='Generate an image',
    deps='gpt-image-2',
    tool_name='generate_image',
    tool_args='{"prompt": "test"}',
    fallback_profile=ModelProfile(supported_native_tools=frozenset({ImageGenerationTool}), supports_image_output=True),
    make_fallback_response=lambda: ModelResponse(
        parts=[FilePart(content=BinaryImage(data=b'png', media_type='image/png'))]
    ),
    with_deps_factory=lambda fallback_model: ImageGeneration[str](
        native=_image_generation_from_deps, fallback_model=fallback_model, output_format='jpeg'
    ),
    expected_fallback_native_tools=snapshot(
        [ImageGenerationTool(model='gpt-image-2', quality='high', output_format='jpeg')]
    ),
    with_pass_through_factory=lambda fallback_model: ImageGeneration[str](
        native=_image_generation_pass_through, fallback_model=fallback_model
    ),
    pass_through_tool=_IMAGE_GENERATION_PASS_THROUGH,
    with_none_factory=lambda fallback_model: ImageGeneration[str](
        native=_none_native_factory, fallback_model=fallback_model, output_format='jpeg'
    ),
    with_native_false=lambda fallback_model: ImageGeneration[str](
        native=False, fallback_model=fallback_model, output_format='jpeg'
    ),
    expected_override_only_native_tools=snapshot([ImageGenerationTool(output_format='jpeg')]),
    with_instance_and_overrides=lambda fallback_model: ImageGeneration[str](
        native=ImageGenerationTool(quality='high', size='1024x1024'),
        fallback_model=fallback_model,
        output_format='jpeg',
    ),
    expected_instance_native_tools=snapshot(
        [ImageGenerationTool(quality='high', size='1024x1024', output_format='jpeg')]
    ),
    native_tool_type=ImageGenerationTool,
    # The narrowed `native_tool` type rejects a `None`-returning factory; this case exists to
    # prove the runtime `UserError` still fires for callers who bypass the type checker.
    subagent=ImageGenerationSubagentTool(
        model='openai-responses:gpt-5.4',
        native_tool=_none_native_factory,  # pyright: ignore[reportArgumentType]
    ),
)

CASES = [XSEARCH_CASE, IMAGE_GENERATION_CASE]

case_param = pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in CASES])


def _outer_model(
    case: Case,
    *,
    supported_native_tools: frozenset[type[AbstractNativeTool]] = frozenset(),
    seen_function_tools: list[list[str]] | None = None,
) -> FunctionModel:
    """A model that calls the capability's local fallback tool once, then answers."""

    def outer_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if seen_function_tools is not None:
            seen_function_tools.append([t.name for t in info.function_tools])
        if any(isinstance(p, ToolReturnPart) for m in messages if isinstance(m, ModelRequest) for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=[ToolCallPart(tool_name=case.tool_name, args=case.tool_args)])

    return FunctionModel(outer_model_fn, profile=ModelProfile(supported_native_tools=supported_native_tools))


def _recording_fallback_model(case: Case, seen_native_tools: list[AbstractNativeTool]) -> FunctionModel:
    """The subagent's model: it supports the native tool and records the ones it is handed."""

    def fallback_model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen_native_tools.extend(info.model_request_parameters.native_tools)
        return case.make_fallback_response()

    return FunctionModel(fallback_model_fn, profile=case.fallback_profile)


@case_param
async def test_callable_native_config_is_used_by_fallback(case: Case, allow_model_requests: None):
    """The fallback subagent resolves the callable native config with the outer run context."""
    seen_native_tools: list[AbstractNativeTool] = []
    capability = case.with_deps_factory(_recording_fallback_model(case, seen_native_tools))
    agent = Agent[str, str](_outer_model(case), deps_type=str, capabilities=[capability])

    result = await agent.run(case.prompt, deps=case.deps)

    assert result.output == 'done'
    assert seen_native_tools == case.expected_fallback_native_tools


@case_param
async def test_callable_native_pass_through_without_overrides(case: Case, allow_model_requests: None):
    """A factory result reaches the subagent unchanged when the capability sets no override fields."""
    seen_native_tools: list[AbstractNativeTool] = []
    capability = case.with_pass_through_factory(_recording_fallback_model(case, seen_native_tools))
    agent = Agent[str, str](_outer_model(case), deps_type=str, capabilities=[capability])

    result = await agent.run(case.prompt, deps=case.deps)

    assert result.output == 'done'
    assert seen_native_tools == [case.pass_through_tool]
    assert seen_native_tools[0] is case.pass_through_tool


@case_param
async def test_callable_native_none_raises(case: Case, allow_model_requests: None):
    """A callable native factory returning `None` raises rather than enabling the default native tool."""
    seen_native_tools: list[AbstractNativeTool] = []
    capability = case.with_none_factory(_recording_fallback_model(case, seen_native_tools))
    agent = Agent[str, str](_outer_model(case), deps_type=str, capabilities=[capability])

    with pytest.raises(UserError, match=r'returned `None`.*drop `fallback_model`'):
        await agent.run(case.prompt, deps=case.deps)

    assert seen_native_tools == []


@case_param
async def test_native_false_keeps_fallback_overrides(case: Case, allow_model_requests: None):
    """Disabling the outer native tool retains fallback-native configuration."""
    seen_native_tools: list[AbstractNativeTool] = []
    capability = case.with_native_false(_recording_fallback_model(case, seen_native_tools))
    agent = Agent[str, str](_outer_model(case), deps_type=str, capabilities=[capability])

    result = await agent.run(case.prompt, deps=case.deps)

    assert capability.get_native_tools() == []
    assert result.output == 'done'
    assert seen_native_tools == case.expected_override_only_native_tools


@case_param
async def test_instance_native_config_is_merged_for_fallback(case: Case, allow_model_requests: None):
    """A static `native=` instance reaches the subagent with capability-level fields layered over it."""
    seen_native_tools: list[AbstractNativeTool] = []
    capability = case.with_instance_and_overrides(_recording_fallback_model(case, seen_native_tools))
    agent = Agent[str, str](_outer_model(case), deps_type=str, capabilities=[capability])

    result = await agent.run(case.prompt, deps=case.deps)

    assert result.output == 'done'
    assert seen_native_tools == case.expected_instance_native_tools


@case_param
async def test_callable_native_none_raises_on_natively_supporting_model(case: Case, allow_model_requests: None):
    """`None` does not omit the fallback tool even when the outer model supports the native tool.

    Native-tool support is recomputed from the tools the factory actually resolved, so a `None`
    return leaves the subagent tool on the wire for a model that would otherwise have dropped it.
    """
    seen_native_tools: list[AbstractNativeTool] = []
    seen_function_tools: list[list[str]] = []
    capability = case.with_none_factory(_recording_fallback_model(case, seen_native_tools))
    agent = Agent[str, str](
        _outer_model(
            case,
            supported_native_tools=frozenset({case.native_tool_type}),
            seen_function_tools=seen_function_tools,
        ),
        deps_type=str,
        capabilities=[capability],
    )

    with pytest.raises(UserError, match=r'returned `None`.*drop `fallback_model`'):
        await agent.run(case.prompt, deps=case.deps)

    assert seen_function_tools == [[case.tool_name]]
    assert seen_native_tools == []


@case_param
async def test_subagent_dynamic_native_none_raises(case: Case):
    """The subagent raises when its dynamic factory returns `None` instead of enabling the default tool."""
    with pytest.raises(UserError, match=r'returned `None`.*drop `fallback_model`'):
        await case.subagent(build_run_context(), case.subagent_input)


def test_xsearch_incompatible_native_tool_raises():
    """Invalid static native configuration raises at capability construction."""
    with pytest.raises(
        UserError, match=r'`native` must be `True`, `False`, a callable, or an instance of `XSearchTool`'
    ):
        XSearch(
            native=ImageGenerationTool(),  # pyright: ignore[reportArgumentType]
            fallback_model='xai:grok-4-1-fast-non-reasoning',
        )


async def test_xsearch_callable_native_wrong_tool_type_raises(allow_model_requests: None):
    """The shared resolver validates dynamic factory results before applying overrides.

    The outer model supports the tool type the factory wrongly returns, so it passes the native
    path's own support check and the mismatch surfaces where the subagent resolves it.
    """

    def native_factory(ctx: RunContext[str]) -> ImageGenerationTool:
        return ImageGenerationTool()

    seen_native_tools: list[AbstractNativeTool] = []
    capability = XSearch[str](
        native=native_factory,  # pyright: ignore[reportArgumentType]
        fallback_model=_recording_fallback_model(XSEARCH_CASE, seen_native_tools),
        include_output=True,
    )
    agent = Agent[str, str](
        _outer_model(XSEARCH_CASE, supported_native_tools=frozenset({ImageGenerationTool})),
        deps_type=str,
        capabilities=[capability],
    )

    with pytest.raises(UserError, match=r'must resolve to an instance of `XSearchTool`'):
        await agent.run(XSEARCH_CASE.prompt, deps=XSEARCH_CASE.deps)

    assert seen_native_tools == []
