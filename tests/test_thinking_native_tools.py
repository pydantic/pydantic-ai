"""Wire-contract tests: unified `thinking` and a native web-search tool both land on the request body.

Parametrized across the providers that support reasoning and native web search at once. Each case asserts
on the actual request wire body (`vcr.requests[0].body`) — the cassette matcher isn't sensitive to the
body, so asserting it directly is what proves both the thinking config and the tool survive together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.settings import ModelSettings

from .conftest import try_import

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as google_imports:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


@dataclass(frozen=True)
class WireCase:
    id: str
    provider: str
    model_name: str
    present: tuple[tuple[tuple[str, ...], object], ...]
    """(path, value) pairs: each nested key path in the request body must resolve to exactly this value."""
    tool_marker: str
    """Substring that must appear in the compact-JSON-serialized `tools` array (the native web-search tool)."""
    max_tokens: int | None = None
    """Set when the model needs `max_tokens` above the thinking budget (Anthropic rejects budget >= max_tokens)."""
    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    WireCase(
        id='anthropic',
        provider='anthropic',
        model_name='claude-sonnet-4-5',
        present=((('thinking', 'type'), 'enabled'), (('thinking', 'budget_tokens'), 16384)),
        tool_marker='"type":"web_search_',
        max_tokens=20000,
        marks=(pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),),
    ),
    WireCase(
        id='openai-responses',
        provider='openai-responses',
        model_name='gpt-5',
        present=((('reasoning', 'effort'), 'high'),),
        tool_marker='"type":"web_search"',
        marks=(pytest.mark.skipif(not openai_imports(), reason='openai not installed'),),
    ),
    WireCase(
        id='google',
        provider='google',
        # Gemini 2.5 rejects a native-tool-only request (`functionCallingConfig` without
        # `functionDeclarations`); Gemini 3 accepts it, so this uses a Gemini 3 model.
        model_name='gemini-3-flash-preview',
        present=(
            (('generationConfig', 'thinkingConfig', 'include_thoughts'), True),
            (('generationConfig', 'thinkingConfig', 'thinking_level'), 'HIGH'),
        ),
        tool_marker='"googleSearch":',
        marks=(pytest.mark.skipif(not google_imports(), reason='google-genai not installed'),),
    ),
]


def _build_model(case: WireCase, *, anthropic_api_key: str, openai_api_key: str, gemini_api_key: str) -> Model:
    if case.provider == 'anthropic':
        return AnthropicModel(case.model_name, provider=AnthropicProvider(api_key=anthropic_api_key))
    if case.provider == 'openai-responses':
        return OpenAIResponsesModel(case.model_name, provider=OpenAIProvider(api_key=openai_api_key))
    if case.provider == 'google':
        return GoogleModel(case.model_name, provider=GoogleProvider(api_key=gemini_api_key))
    raise ValueError(f'unknown provider {case.provider!r}')  # pragma: no cover


# When the model actually invokes the native tool, the agent yields the legacy (v2-deprecated) builtin-tool
# events. These tests assert the request wire body (built before any response), so the event deprecation is
# irrelevant here and would otherwise turn into an error under `filterwarnings = ['error']`.
@pytest.mark.filterwarnings('ignore:`BuiltinToolCallEvent` is deprecated:DeprecationWarning')
@pytest.mark.filterwarnings('ignore:`BuiltinToolResultEvent` is deprecated:DeprecationWarning')
@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_thinking_with_native_tool_wire_contract(
    case: WireCase,
    allow_model_requests: None,
    anthropic_api_key: str,
    openai_api_key: str,
    gemini_api_key: str,
    vcr: Cassette,
):
    """`thinking='high'` and a native web-search tool both land on the request wire body."""
    model = _build_model(
        case, anthropic_api_key=anthropic_api_key, openai_api_key=openai_api_key, gemini_api_key=gemini_api_key
    )
    settings = ModelSettings(thinking='high')
    if case.max_tokens is not None:
        settings['max_tokens'] = case.max_tokens
    agent = Agent(model, model_settings=settings, capabilities=[NativeTool(WebSearchTool())])
    await agent.run('What is the top news story today? Use web search.')

    body = json.loads(vcr.requests[0].body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    for path, value in case.present:
        node = body
        for key in path:
            node = node[key]
        assert node == value, f'{path!r}={node!r}, expected {value!r}'
    tools = json.dumps(body['tools'], separators=(',', ':'))
    assert case.tool_marker in tools, f'{case.tool_marker!r} not in tools {tools!r}'
