"""Tests for deferred capability loading, reveal deltas, and the capability catalog.

Split out of `test_capabilities.py` per #7304.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.capabilities import (
    Capability,
    NativeTool,
    ProcessHistory,
    ToolSearch,
    Toolset,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import (
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolAvailabilityDeltaEvent,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    ModelRequestContext,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    AbstractNativeTool,
    WebSearchTool,
)
from pydantic_ai.native_tools._tool_search import ToolSearchTool
from pydantic_ai.settings import ModelSettings as _ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets._deferred_capability_loader import (
    LOAD_CAPABILITY_ALREADY_ACTIVE_MESSAGE_TEMPLATE,
    LOAD_CAPABILITY_TOOL_NAME,
)
from pydantic_ai.usage import RequestUsage, RunUsage

from ._inline_snapshot import snapshot
from .capability_models import (
    make_text_response,
    noop_greet as _noop_greet,
)
from .conftest import IsDatetime, IsStr, iter_message_parts

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


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
                seen_dynamic_loaded.append(ctx.capability_active)
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
        seen_loaded.append(ctx.capability_active)
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

The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy tools.\
""",
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
                        content={
                            'instructions': """\
Use the refund policy before answering refund questions.

Load-time account context for run step 1.\
"""
                        },
                        tool_call_id='load-refunds',
                        timestamp=IsDatetime(),
                    ),
                    ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='load-refunds'),
                ],
                timestamp=IsDatetime(),
                instructions="""\
Visible billing instructions.

The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='lookup_refund_policy', args={'order_id': 'order-123'}, tool_call_id='lookup-refund'
                    )
                ],
                usage=RequestUsage(input_tokens=80, output_tokens=10),
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

The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund policy tools.\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='final: order-123: refund allowed for 30 days')],
                usage=RequestUsage(input_tokens=86, output_tokens=17),
                model_name='function:model_fn:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_tool_return_reveals_deferred_tool_without_capability() -> None:
    """A user tool can reveal a deferred tool and records the delta beside its return."""

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [
            part
            for part in iter_message_parts(messages, ModelRequest, ToolReturnPart)
            if part.tool_name in {'reveal_weather', 'get_weather'}
        ]
        if not returns:
            assert info.model_request_parameters.revealed_tool_names == set()
            return ModelResponse(parts=[ToolCallPart(tool_name='reveal_weather', args={}, tool_call_id='reveal')])
        if len(returns) == 1:
            assert info.model_request_parameters.revealed_tool_names == {'get_weather'}
            return ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args={'city': 'Paris'}, tool_call_id='weather')]
            )
        return make_text_response(str(returns[-1].content))

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def reveal_weather() -> ToolReturn[str]:
        return ToolReturn(return_value='Weather tools are ready.', tools=['get_weather'])

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:
        return f'Sunny in {city}'

    result = await agent.run('What is the weather?')

    assert result.output == 'Sunny in Paris'
    reveal_request = next(
        message
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_call_id == 'reveal' for part in message.parts)
    )
    assert reveal_request.parts == snapshot(
        [
            ToolReturnPart(
                tool_name='reveal_weather',
                content='Weather tools are ready.',
                tool_call_id='reveal',
                timestamp=IsDatetime(),
            ),
            ToolAvailabilityDeltaPart(tools_added=['get_weather'], tool_call_id='reveal'),
        ]
    )


async def test_processed_history_determines_request_reveal_state() -> None:
    """Removing a reveal from outgoing history also removes it from request parameters."""
    seen: list[set[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.revealed_tool_names)
        assert 'hidden_tool' not in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[TextPart(content='done')])

    def strip_deltas(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            replace(message, parts=[part for part in message.parts if not isinstance(part, ToolAvailabilityDeltaPart)])
            if isinstance(message, ModelRequest)
            else message
            for message in messages
        ]

    agent = Agent(FunctionModel(model_fn), capabilities=[ProcessHistory(strip_deltas)])

    @agent.tool_plain(defer_loading=True)
    def hidden_tool() -> str:  # pragma: no cover
        return 'hidden'

    await agent.run(
        'continue',
        message_history=[ModelRequest(parts=[ToolAvailabilityDeltaPart(tools_added=['hidden_tool'])])],
    )

    assert seen == [set()]


async def test_orphaned_reveal_evidence_stripped_by_cleanup_does_not_count_as_revealed() -> None:
    """Evidence orphaned by a history processor is stripped before reveal derivation.

    A processor that drops the response carrying a `ToolSearchCallPart` leaves an orphaned
    `ToolSearchReturnPart`; history cleanup removes the orphan before the request ships, so the
    derived reveal state must not count it — otherwise the request would declare a revealed tool
    with zero reveal evidence on the outgoing wire.
    """
    seen: list[set[str]] = []

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info.model_request_parameters.revealed_tool_names)
        assert 'hidden_tool' not in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[TextPart(content='done')])

    def drop_search_calls(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            message
            for message in messages
            if not (
                isinstance(message, ModelResponse)
                and any(isinstance(part, ToolSearchCallPart) for part in message.parts)
            )
        ]

    agent = Agent(FunctionModel(model_fn), capabilities=[ProcessHistory(drop_search_calls)])

    @agent.tool_plain(defer_loading=True)
    def hidden_tool() -> str:  # pragma: no cover
        return 'hidden'

    await agent.run(
        'continue',
        message_history=[
            ModelRequest(parts=[UserPromptPart(content='find tools')]),
            ModelResponse(parts=[ToolSearchCallPart(args={'queries': ['hidden']}, tool_call_id='search-1')]),
            ModelRequest(
                parts=[
                    ToolSearchReturnPart(
                        content={'discovered_tools': [{'name': 'hidden_tool'}]},
                        tool_call_id='search-1',
                    )
                ]
            ),
        ],
    )

    assert seen == [set()]


async def test_model_calling_a_withheld_tool_is_refused_and_reveals_nothing() -> None:
    """Calling a hidden tool by (guessed) name is refused, and authors no reveal.

    Hiding is now an availability gate, not just prompt engineering: a tool the model was never
    shown cannot be executed by guessing its name. The refusal is a retry pointing at search, and
    a refused call is not a discovery, so the tool stays off the wire afterwards.
    """
    wire_tools: list[list[str]] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        wire_tools.append(sorted(tool.name for tool in info.function_tools))
        if list(iter_message_parts(messages, ModelRequest, RetryPromptPart)):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool_name='hidden_tool', args={}, tool_call_id='guess')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain(defer_loading=True)
    def hidden_tool() -> str:
        return 'secret'  # pragma: no cover

    result = await agent.run('guess the hidden tool')

    assert result.output == 'done'
    returns = list(iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart))
    assert returns == []
    retries = list(iter_message_parts(result.all_messages(), ModelRequest, RetryPromptPart))
    assert [str(part.content) for part in retries] == snapshot(
        [
            "Tool 'hidden_tool' is not available yet: search for it first, then call it again once you've seen its schema."
        ]
    )
    deltas = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert deltas == []
    assert all('hidden_tool' not in tools for tools in wire_tools)


async def test_tool_return_deduplicates_new_reveals() -> None:
    """Duplicate names and repeated reveals author one ordered availability delta.

    A fully repeated reveal drops out entirely; a partial overlap keeps only the genuinely new
    names, in order.
    """

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(parts=[ToolCallPart('revealer', {}, tool_call_id='first')])
        if len(returns) == 1:
            return ModelResponse(parts=[ToolCallPart('revealer', {}, tool_call_id='second')])
        if len(returns) == 2:
            return ModelResponse(parts=[ToolCallPart('partial_revealer', {}, tool_call_id='third')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def revealer() -> ToolReturn[str]:
        return ToolReturn(return_value='ready', tools=['tool_b', 'tool_a', 'tool_b'])

    @agent.tool_plain
    def partial_revealer() -> ToolReturn[str]:
        return ToolReturn(return_value='partially new', tools=['tool_a', 'tool_c'])

    result = await agent.run('reveal')
    deltas = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert deltas == [
        ToolAvailabilityDeltaPart(tools_added=['tool_b', 'tool_a'], tool_call_id='first'),
        ToolAvailabilityDeltaPart(tools_added=['tool_c'], tool_call_id='third'),
    ]


@pytest.mark.parametrize(
    'tools',
    ['get_weather', 1, [1], [[]]],
    ids=['bare-string', 'non-sequence', 'non-string-element', 'unhashable-element'],
)
async def test_tool_return_rejects_invalid_tools(tools: object) -> None:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if list(iter_message_parts(messages, ModelRequest, ToolReturnPart)):  # pragma: no cover
            return make_text_response('done')
        return ModelResponse(parts=[ToolCallPart(tool_name='reveal_weather', args={}, tool_call_id='reveal')])

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    def reveal_weather() -> ToolReturn[str]:
        return ToolReturn(return_value='Weather tools are ready.', tools=cast(Any, tools))

    with pytest.raises(UserError, match=r'`ToolReturn\.tools` must be a list of tool names'):
        await agent.run('Reveal the weather tool.')


async def test_parallel_tool_returns_keep_each_availability_delta_adjacent() -> None:
    """Parallel execution reorders each return together with its own sibling delta."""

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='reveal_b', args={}, tool_call_id='b'),
                    ToolCallPart(tool_name='reveal_a', args={}, tool_call_id='a'),
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    async def reveal_a() -> ToolReturn[str]:
        await asyncio.sleep(0)
        return ToolReturn(return_value='a', tools=['tool_a'])

    @agent.tool_plain
    async def reveal_b() -> ToolReturn[str]:
        await asyncio.sleep(0.01)
        return ToolReturn(return_value='b', tools=['tool_b'])

    result = await agent.run('reveal both')
    request = next(
        message
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        and any(isinstance(part, ToolReturnPart) and part.tool_call_id == 'b' for part in message.parts)
    )
    assert [(type(part).__name__, getattr(part, 'tool_call_id', None)) for part in request.parts] == snapshot(
        [
            ('ToolReturnPart', 'b'),
            ('ToolAvailabilityDeltaPart', 'b'),
            ('ToolReturnPart', 'a'),
            ('ToolAvailabilityDeltaPart', 'a'),
        ]
    )


async def test_parallel_tool_returns_dedupe_same_reveal_in_history_order() -> None:
    """When parallel calls reveal the same tool, the first call in emitted history owns the delta.

    Deduplication must not depend on task completion order: the first-emitted call finishes
    LAST here, and must still be the one that carries the availability delta — otherwise the
    durable history (and the reveal's wire anchor) would vary with scheduling.
    """

    def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        returns = list(iter_message_parts(messages, ModelRequest, ToolReturnPart))
        if not returns:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='slow_first', args={}, tool_call_id='first'),
                    ToolCallPart(tool_name='fast_second', args={}, tool_call_id='second'),
                ]
            )
        return make_text_response('done')

    agent = Agent(FunctionModel(model_fn))

    @agent.tool_plain
    async def slow_first() -> ToolReturn[str]:
        await asyncio.sleep(0.01)
        return ToolReturn(return_value='slow', tools=['revealed'])

    @agent.tool_plain
    async def fast_second() -> ToolReturn[str]:
        return ToolReturn(return_value='fast', tools=['revealed'])

    events: list[AgentStreamEvent] = []
    async with agent.iter('reveal in parallel') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    events.extend([event async for event in stream])

    assert agent_run.result is not None
    result = agent_run.result
    deltas = [
        part
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert deltas == [ToolAvailabilityDeltaPart(tools_added=['revealed'], tool_call_id='first')]
    assert [event for event in events if isinstance(event, ToolAvailabilityDeltaEvent)] == [
        ToolAvailabilityDeltaEvent(part=deltas[0])
    ]


async def test_deferred_capability_tool_registered_after_construction_defers_until_load() -> None:
    """A tool registered via `@cap.tool` *after* construction defers like a constructor tool: hidden until load.

    Deferred tools stay tagged `defer_loading=True`; current visibility is tracked separately.
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
    assert defer_flag_by_phase == snapshot({'before_load': None, 'after_load': True})


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
    hooks = Hooks()
    available_per_turn: list[set[str]] = []

    @hooks.on.before_model_request
    async def record_available_tools(ctx: RunContext, request_context: ModelRequestContext) -> ModelRequestContext:
        available_per_turn.append(ctx.available_tool_names)
        return request_context

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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

    agent = Agent(FunctionModel(model_fn), capabilities=[refunds, hooks])
    result = await agent.run('Can I get a refund?')

    assert result.output == 'done'
    assert 'lookup_refund_policy' not in available_per_turn[0]
    assert len(available_per_turn[1:]) >= 2
    assert all('lookup_refund_policy' in names for names in available_per_turn[1:])


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


async def test_deferred_capability_tool_delta_persists_in_history() -> None:
    """The tool delta after a capability load persists, without duplication on resume."""
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
    events: list[AgentStreamEvent] = []
    async with agent.iter('Can I get a refund?') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    events.extend([event async for event in stream])

    assert agent_run.result is not None
    result = agent_run.result

    def availability_deltas(messages: list[ModelMessage]) -> list[ToolAvailabilityDeltaPart]:
        return [part for message in messages for part in message.parts if isinstance(part, ToolAvailabilityDeltaPart)]

    messages = result.all_messages()
    assert availability_deltas(messages) == [
        ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='load')
    ]
    assert [event for event in events if isinstance(event, ToolAvailabilityDeltaEvent)] == [
        ToolAvailabilityDeltaEvent(part=availability_deltas(messages)[0])
    ]

    # Idempotence: feeding the resulting history back in does not inject a duplicate pair
    # (the deterministic call_id means it's recognized as already discovered).
    result2 = await agent.run('And another refund?', message_history=messages)
    new_messages = result2.all_messages()[len(messages) :]
    assert availability_deltas(new_messages) == []


async def test_capability_load_history_without_delta_is_backfilled() -> None:
    """An ID-only capability load history gains one delta before the resumed model request."""
    refunds = Capability[object](id='refunds', defer_loading=True)
    visibility: list[tuple[bool, set[str]]] = []

    @refunds.tool_plain
    def lookup_refund_policy() -> str:  # pragma: no cover
        return 'refund allowed'

    @dataclass
    class CaptureVisibility(AbstractCapability[Any]):
        async def before_model_request(
            self, ctx: RunContext[Any], request_context: ModelRequestContext
        ) -> ModelRequestContext:
            visibility.append((ctx.is_tool_available('lookup_refund_policy'), ctx.available_tool_names))
            return request_context

    history: list[ModelMessage] = [
        ModelResponse(parts=[LoadCapabilityCallPart(args={'id': 'refunds'}, tool_call_id='old-load')]),
        ModelRequest(parts=[LoadCapabilityReturnPart(content={}, tool_call_id='old-load')]),
    ]

    def model_fn(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.model_request_parameters.revealed_tool_names == {'lookup_refund_policy'}
        return make_text_response('done')

    result = await Agent(FunctionModel(model_fn), capabilities=[refunds, CaptureVisibility()]).run(
        'Continue.', message_history=history
    )

    assert visibility == [(True, {'load_capability', 'lookup_refund_policy'})]
    new_deltas = [
        part
        for message in result.new_messages()
        for part in message.parts
        if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert new_deltas == [ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'])]


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

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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


async def test_deferred_capability_tool_delta_not_duplicated_over_long_trajectory() -> None:
    """The tool availability delta for a loaded capability appears exactly once.

    Extends the persistence test to >= 3 model-request turns after the load: the delta must
    remain singular across the whole trajectory, and the capability's tool stays available
    on every post-load turn.
    """
    toolset = FunctionToolset()

    @toolset.tool_plain
    def lookup_refund_policy(order_id: str) -> str:
        """Look up the refund policy for an order."""
        return f'{order_id}: refund allowed for 30 days'

    refunds = Capability[object](
        id='refunds', description='Refund policy tools.', toolsets=[toolset], defer_loading=True
    )

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
    tool_deltas = [
        part for message in messages for part in message.parts if isinstance(part, ToolAvailabilityDeltaPart)
    ]
    assert tool_deltas == [ToolAvailabilityDeltaPart(tools_added=['lookup_refund_policy'], tool_call_id='load')]


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

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
        """\
The following capabilities are deferred and can be loaded using the `load_capability` tool. A capability's tools stay hidden until it is loaded:
- refunds: Refund tools.\
"""
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
            [('load_capability', False)],
            [('load_capability', False)],
        ]
    )
    history_parts = [part for message in result.all_messages() for part in message.parts]
    assert not any(isinstance(part, LoadCapabilityReturnPart) for part in history_parts)
    [retry] = [part for part in history_parts if isinstance(part, RetryPromptPart)]
    assert retry.content == snapshot("No capability found with id 'missing'.")


async def test_load_capability_inherits_agent_tool_retries() -> None:
    """`load_capability` honors the agent's tool retry budget."""
    deferred = Capability[object](
        id='deferred',
        description='Deferred.',
        defer_loading=True,
    )
    calls = 0

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=LOAD_CAPABILITY_TOOL_NAME,
                    args={'id': 'missing'},
                    tool_call_id=f'load-missing-{calls}',
                )
            ]
        )

    agent = Agent(FunctionModel(model_fn), capabilities=[deferred], retries={'tools': 3})

    with pytest.raises(UnexpectedModelBehavior):
        await agent.run('load missing')

    assert calls == 4


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
    expected_retry = LOAD_CAPABILITY_ALREADY_ACTIVE_MESSAGE_TEMPLATE.format(capability_id='always-on')
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
    expected_retry = LOAD_CAPABILITY_ALREADY_ACTIVE_MESSAGE_TEMPLATE.format(capability_id='deferred')
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
