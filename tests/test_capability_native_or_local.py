"""Tests for native-or-local capability tools.

Split out of `test_capabilities.py` per #7304.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from importlib.util import find_spec
from types import NoneType
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai._run_context import RunContext
from pydantic_ai._spec import NamedSpec
from pydantic_ai.agent import Agent
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import (
    CAPABILITY_TYPES,
    MCP,
    ImageGeneration,
    PrepareTools,
    ResolveModelId,
    SelectModel,
    ToolSearch,
    WebFetch,
    WebSearch,
    WrapperCapability,
    XSearch,
)
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.capabilities.combined import CombinedCapability
from pydantic_ai.exceptions import (
    ModelRetry,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import (
    KnownModelName,
    Model,
    ModelResolutionContext,
    ModelSelectionContext,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import (
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
    XSearchTool,
)
from pydantic_ai.output import ToolOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .capability_models import (
    MyOutput,
    build_run_context as _build_run_context,
    make_text_response,
)
from .conftest import IsDatetime, IsInstance, IsStr, iter_message_parts

_SEARCH_TOOLS_NAME = ToolSearch.function_tool_name

pytestmark = [
    pytest.mark.anyio,
]


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
