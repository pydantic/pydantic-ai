"""Regression test for the harness README's Quick start example.

The README ships a Hacker News + web-search agent wrapped in `CodeMode` and
asks it to find the most-discussed HN story across three feeds, then pull
the comment thread, the submitter's profile, and follow-up coverage. We
fake everything that talks to the network so the test runs in CI without
`ddgs`, an MCP package, or any HTTP traffic:

- A `FunctionModel` drives the conversation through the same shape the
  example produces in production -- two `run_code` calls (parallel feed
  fetches + dedupe + filter, then parallel follow-ups) and a final summary.
- The Hacker News MCP toolset is replaced with a `FunctionToolset` of
  fake functions whose return values come from the public Logfire trace
  linked in the README.
- `WebSearch(native=False, local=...)` skips the default DuckDuckGo
  fallback so the test doesn't pull `ddgs` and the harness doesn't depend
  on it in CI.

`CodeMode` itself is real -- the `FunctionModel`'s emitted Python code
runs through the Monty sandbox, dispatches the calls back through
pydantic-ai's tool machinery to our fakes, and the return values flow
back into the model loop. Any future change in pydantic-ai or the harness
that breaks how these capabilities compose makes this test fail.
"""

from __future__ import annotations

import textwrap
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import MCP, WebSearch
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RequestUsage

from pydantic_ai_harness import CodeMode

from .conftest import IsDatetime, IsPartialDict, IsStr

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (pydantic-ai uses asyncio.create_task internally)."""
    return 'asyncio'


# ---------------------------------------------------------------------------
# Canned tool responses -- shapes mirror what the cyanheads HN MCP server
# actually returns, with values from the run captured in the README's
# linked public trace.
# ---------------------------------------------------------------------------

_WINNER_ID = 48037128
_WINNER_USER = 'e12e'

_TOP_FEED = {
    'stories': [
        {
            'id': 48037555,
            'type': 'story',
            'title': 'Valve releases Steam Controller CAD files under Creative Commons license',
            'score': 1687,
            'by': 'haunter',
            'time': 1778082253,
            'descendants': 572,
            'url': 'https://www.digitalfoundry.net/news/2026/05/valve-releases-steam-controller-cad-files',
        },
        {
            'id': 48050499,
            'type': 'story',
            'title': 'I want to live like Costco people',
            'score': 235,
            'by': 'speckx',
            'time': 1778167167,
            'descendants': 495,
            'url': 'https://tastecooking.com/i-want-to-live-like-costco-people/',
        },
    ],
}

_BEST_FEED = {
    'stories': [
        {
            'id': _WINNER_ID,
            'type': 'story',
            'title': "Vibe coding and agentic engineering are getting closer than I'd like",
            'score': 748,
            'by': _WINNER_USER,
            'time': 1778079997,
            'descendants': 853,
            'url': 'https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering/',
        },
        {
            'id': 48038001,
            'type': 'story',
            'title': 'Appearing productive in the workplace',
            'score': 1534,
            'by': 'diebillionaires',
            'time': 1778084309,
            'descendants': 629,
            'url': 'https://nooneshappy.com/article/appearing-productive-in-the-workplace/',
        },
        {
            'id': 48037555,
            'type': 'story',
            'title': 'Valve releases Steam Controller CAD files under Creative Commons license',
            'score': 1687,
            'by': 'haunter',
            'time': 1778082253,
            'descendants': 572,
            'url': 'https://www.digitalfoundry.net/news/2026/05/valve-releases-steam-controller-cad-files',
        },
    ],
}

_SHOW_FEED: dict[str, list[dict[str, Any]]] = {'stories': []}

_THREAD = {
    'item': {
        'id': _WINNER_ID,
        'title': "Vibe coding and agentic engineering are getting closer than I'd like",
        'url': 'https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering/',
        'score': 748,
        'by': _WINNER_USER,
        'descendants': 853,
    },
    'comments': [
        {'by': 'etothet', 'depth': 0, 'text': 'LLMs exposed sloppy practices, not created them.'},
        {'by': 'kelnos', 'depth': 0, 'text': 'Normalization of deviance as engineers stop reviewing diffs.'},
    ],
    'totalLoaded': 2,
    'totalAvailable': 853,
}

_USER_PROFILE: dict[str, Any] = {
    'user': {
        'id': _WINNER_USER,
        'created': 1331059200,
        'karma': 15024,
        'submitted': 9700,
        'about': 'perpetual student and sometimes developer based in Tromsø, Norway',
    },
    'submissions': [],
}

_WEB_RESULTS: dict[str, list[dict[str, Any]]] = {
    'results': [
        {
            'title': 'GLM-5: From Vibe Coding to Agentic Engineering',
            'url': 'https://simonwillison.net/2026/Feb/11/glm-5/',
            'snippet': 'Earlier piece by the same author tracing the same arc.',
        },
    ],
}


# ---------------------------------------------------------------------------
# Fake tool implementations. The model's tool dispatch is captured in each
# `run_code` ToolReturnPart's metadata, so the snapshot assertion below
# already records every call -- no separate recorder needed.
# ---------------------------------------------------------------------------


def _make_fake_hn_toolset() -> FunctionToolset[None]:
    feeds: dict[str, dict[str, list[dict[str, Any]]]] = {
        'top': _TOP_FEED,
        'best': _BEST_FEED,
        'show': _SHOW_FEED,
    }

    def hn_get_stories(*, feed: str, count: int = 50) -> dict[str, Any]:
        """Fetch a Hacker News feed (top, best, or show)."""
        return feeds[feed]

    def hn_get_thread(*, itemId: int, depth: int = 2, maxComments: int = 60) -> dict[str, Any]:
        """Fetch the comment thread for a story id."""
        return _THREAD

    def hn_get_user(
        *,
        username: str,
        includeSubmissions: bool = False,
        submissionCount: int = 5,
    ) -> dict[str, Any]:
        """Fetch a Hacker News user's profile."""
        return _USER_PROFILE

    return FunctionToolset[None](
        tools=[
            Tool(hn_get_stories),
            Tool(hn_get_thread),
            Tool(hn_get_user),
        ]
    )


def _fake_web_search(*, query: str) -> dict[str, Any]:
    """Stand-in for the WebSearch capability's local DDG fallback."""
    return _WEB_RESULTS


# ---------------------------------------------------------------------------
# FunctionModel state machine -- two run_code calls, then a final synthesis,
# matching the trace in the README.
# ---------------------------------------------------------------------------

# First run_code: parallel feed fetches, dedupe by id, score filter, rank by descendants.
_FIRST_RUN_CODE = textwrap.dedent(
    """
    import asyncio
    top, best, show = await asyncio.gather(
        hn_get_stories(feed='top', count=50),
        hn_get_stories(feed='best', count=50),
        hn_get_stories(feed='show', count=50),
    )
    seen = {}
    for feed_name, data in [('top', top), ('best', best), ('show', show)]:
        for s in data['stories']:
            if s.get('score', 0) >= 100:
                if s['id'] not in seen:
                    entry = dict(s)
                    entry['feeds'] = []
                    seen[s['id']] = entry
                seen[s['id']]['feeds'].append(feed_name)
    ranked = sorted(seen.values(), key=lambda x: x.get('descendants', 0), reverse=True)
    ranked[:5]
    """
).strip()

# Second run_code: parallel follow-up calls on the winner. Uses the WebSearch
# capability's local fallback (`web_search`) and the MCP-served HN tools side by
# side, mirroring the README example's "HN tools + web search" composition.
_SECOND_RUN_CODE = textwrap.dedent(
    f"""
    import asyncio
    thread, user, coverage = await asyncio.gather(
        hn_get_thread(itemId={_WINNER_ID}, depth=2, maxComments=60),
        hn_get_user(username='{_WINNER_USER}', includeSubmissions=True, submissionCount=5),
        web_search(query='vibe coding agentic engineering simonwillison'),
    )
    (thread['item'], user['user'], coverage['results'][:5])
    """
).strip()

_FINAL_SYNTHESIS = (
    'The most-discussed HN story across top/best/show clearing 100 points is '
    '"Vibe coding and agentic engineering are getting closer than I\'d like" '
    'by Simon Willison (748 points, 853 comments), submitted by e12e.'
)


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    completed_run_codes = [
        p
        for m in messages
        if isinstance(m, ModelRequest)
        for p in m.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
    ]
    if not completed_run_codes:
        return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': _FIRST_RUN_CODE})])
    if len(completed_run_codes) == 1:
        return ModelResponse(
            parts=[
                TextPart('The winner is the Simon Willison post; pulling thread, user, and coverage in parallel.'),
                ToolCallPart(tool_name='run_code', args={'code': _SECOND_RUN_CODE}),
            ]
        )
    return ModelResponse(parts=[TextPart(_FINAL_SYNTHESIS)])


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


class TestReadmeQuickStart:
    """End-to-end check that the README's Quick start example still works."""

    async def test_quick_start_runs_through_codemode_with_faked_io(self) -> None:
        agent: Agent[None, str] = Agent(
            FunctionModel(_model_fn),
            capabilities=[
                # Wire the fake HN tools through the `MCP` capability the same way
                # the README does -- `local=` overrides the default MCP HTTP
                # toolset with our in-process fake, so the test exercises the same
                # capability composition path as production without any network.
                # MCP's `__init__` narrows `local` to MCP-specific types, but the
                # parent `NativeOrLocalTool` accepts any `AbstractToolset` at runtime.
                MCP[None](
                    'https://hn.caseyjhand.com/mcp',
                    native=False,
                    local=_make_fake_hn_toolset(),  # pyright: ignore[reportArgumentType]
                ),
                # The auto-wrapped Tool would take its name from the function
                # (`_fake_web_search`); pass `name='web_search'` so the sandbox
                # exposes it under the same name the model uses.
                WebSearch[None](native=False, local=Tool(_fake_web_search, name='web_search')),
                CodeMode[None](),
            ],
        )

        result = await agent.run(
            "Across the top, best, and 'show HN' Hacker News feeds, find the most-discussed "
            'story with at least 100 points. Pull its comment thread, its submitter profile, '
            'and any web coverage. Summarize what you find in one paragraph.'
        )

        # The full message tree -- two `run_code` calls, each with parallel tool
        # dispatches captured in the return metadata, plus the final synthesis.
        # Run with `--inline-snapshot=fix` to update if the example legitimately
        # changes shape.
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content="Across the top, best, and 'show HN' Hacker News feeds, find the most-discussed story with at least 100 points. Pull its comment thread, its submitter profile, and any web coverage. Summarize what you find in one paragraph.",
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
                            tool_name='run_code',
                            args={
                                'code': """\
import asyncio
top, best, show = await asyncio.gather(
    hn_get_stories(feed='top', count=50),
    hn_get_stories(feed='best', count=50),
    hn_get_stories(feed='show', count=50),
)
seen = {}
for feed_name, data in [('top', top), ('best', best), ('show', show)]:
    for s in data['stories']:
        if s.get('score', 0) >= 100:
            if s['id'] not in seen:
                entry = dict(s)
                entry['feeds'] = []
                seen[s['id']] = entry
            seen[s['id']]['feeds'].append(feed_name)
ranked = sorted(seen.values(), key=lambda x: x.get('descendants', 0), reverse=True)
ranked[:5]\
"""
                            },
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=88, output_tokens=72),
                    model_name='function:_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='run_code',
                            content=[
                                {
                                    'id': 48037128,
                                    'type': 'story',
                                    'title': "Vibe coding and agentic engineering are getting closer than I'd like",
                                    'score': 748,
                                    'by': 'e12e',
                                    'time': 1778079997,
                                    'descendants': 853,
                                    'url': 'https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering/',
                                    'feeds': ['best'],
                                },
                                {
                                    'id': 48038001,
                                    'type': 'story',
                                    'title': 'Appearing productive in the workplace',
                                    'score': 1534,
                                    'by': 'diebillionaires',
                                    'time': 1778084309,
                                    'descendants': 629,
                                    'url': 'https://nooneshappy.com/article/appearing-productive-in-the-workplace/',
                                    'feeds': ['best'],
                                },
                                {
                                    'id': 48037555,
                                    'type': 'story',
                                    'title': 'Valve releases Steam Controller CAD files under Creative Commons license',
                                    'score': 1687,
                                    'by': 'haunter',
                                    'time': 1778082253,
                                    'descendants': 572,
                                    'url': 'https://www.digitalfoundry.net/news/2026/05/valve-releases-steam-controller-cad-files',
                                    'feeds': ['top', 'best'],
                                },
                                {
                                    'id': 48050499,
                                    'type': 'story',
                                    'title': 'I want to live like Costco people',
                                    'score': 235,
                                    'by': 'speckx',
                                    'time': 1778167167,
                                    'descendants': 495,
                                    'url': 'https://tastecooking.com/i-want-to-live-like-costco-people/',
                                    'feeds': ['top'],
                                },
                            ],
                            tool_call_id=IsStr(),
                            metadata=IsPartialDict({'code_mode': True}),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The winner is the Simon Willison post; pulling thread, user, and coverage in parallel.'
                        ),
                        ToolCallPart(
                            tool_name='run_code',
                            args={
                                'code': """\
import asyncio
thread, user, coverage = await asyncio.gather(
    hn_get_thread(itemId=48037128, depth=2, maxComments=60),
    hn_get_user(username='e12e', includeSubmissions=True, submissionCount=5),
    web_search(query='vibe coding agentic engineering simonwillison'),
)
(thread['item'], user['user'], coverage['results'][:5])\
"""
                            },
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=211, output_tokens=116),
                    model_name='function:_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='run_code',
                            content=(
                                {
                                    'id': 48037128,
                                    'title': "Vibe coding and agentic engineering are getting closer than I'd like",
                                    'url': 'https://simonwillison.net/2026/May/6/vibe-coding-and-agentic-engineering/',
                                    'score': 748,
                                    'by': 'e12e',
                                    'descendants': 853,
                                },
                                {
                                    'id': 'e12e',
                                    'created': 1331059200,
                                    'karma': 15024,
                                    'submitted': 9700,
                                    'about': 'perpetual student and sometimes developer based in Tromsø, Norway',
                                },
                                [
                                    {
                                        'title': 'GLM-5: From Vibe Coding to Agentic Engineering',
                                        'url': 'https://simonwillison.net/2026/Feb/11/glm-5/',
                                        'snippet': 'Earlier piece by the same author tracing the same arc.',
                                    }
                                ],
                            ),
                            tool_call_id=IsStr(),
                            metadata=IsPartialDict({'code_mode': True}),
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        TextPart(
                            content='The most-discussed HN story across top/best/show clearing 100 points is "Vibe coding and agentic engineering are getting closer than I\'d like" by Simon Willison (748 points, 853 comments), submitted by e12e.'
                        )
                    ],
                    usage=RequestUsage(input_tokens=281, output_tokens=148),
                    model_name='function:_model_fn:',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                    conversation_id=IsStr(),
                ),
            ]
        )
