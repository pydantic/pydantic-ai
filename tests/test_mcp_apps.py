"""MCP Apps from a real MCP server through an agent run to the messages a frontend receives.

Each test runs the whole chain with no hand-built app data: an in-process FastMCP server, `MCPToolset`,
an agent whose model calls the tools, and a UI adapter's encoded stream and dumped history.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from ._inline_snapshot import snapshot
from .conftest import IsInt, IsStr, try_import

with try_import() as imports_successful:
    import httpx
    from ag_ui.core import ActivityMessage, RunAgentInput, UserMessage
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from fastmcp.tools import ToolResult as FastMCPToolResult
    from mcp.types import TextContent
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response
    from starlette.routing import Route

    from pydantic_ai._mcp_compat import is_mcp_sdk_v2
    from pydantic_ai.mcp import MCPToolset
    from pydantic_ai.ui.ag_ui import AGUIAdapter
    from pydantic_ai.ui.ag_ui._utils import ACTIVITY_EVENTS_VERSION, detect_ag_ui_version, parse_ag_ui_version
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, UIMessage

pytestmark = pytest.mark.skipif(not imports_successful(), reason='fastmcp, ag-ui-protocol or starlette not installed')

# FastMCP 4 servers stamp every result's `_meta` with their identity, which the app data passes through.
SERVER_INFO: dict[str, Any] = (
    {'io.modelcontextprotocol/serverInfo': {'name': 'weather_server', 'version': IsStr()}}
    if imports_successful() and is_mcp_sdk_v2()
    else {}
)
READ_FIELDS: dict[str, Any] = (
    {'resultType': 'complete', 'cacheScope': 'private', 'ttlMs': 0, '_meta': SERVER_INFO} if SERVER_INFO else {}
)
"""The fields FastMCP 4 adds to every `resources/read` result."""


def weather_server() -> FastMCP[None]:
    """A tool with an MCP Apps UI, a tool without one, and the UI resource."""
    server: FastMCP[None] = FastMCP('weather_server')

    @server.resource('ui://weather/forecast', mime_type='text/html;profile=mcp-app')
    def forecast_view() -> str:
        return '<!doctype html><html><body>forecast</body></html>'

    @server.tool(meta={'ui': {'resourceUri': 'ui://weather/forecast'}})
    def get_forecast(city: str) -> FastMCPToolResult:
        """Text for the model; structured content and per-call `_meta` for the app."""
        return FastMCPToolResult(
            content=[TextContent(type='text', text=f'{city}: 14C')],
            structured_content={'city': city, 'temp_c': 14},
            meta={'selectedDay': 'tuesday'},
        )

    @server.tool
    def get_time(city: str) -> str:
        """Has no UI."""
        return f'{city}: 09:00'

    @server.tool(meta={'ui': {'visibility': ['app']}})
    def select_day(day: str) -> str:
        """Only the app calls this; it has no forecast for weekends."""
        if day in ('saturday', 'sunday'):
            raise ToolError(f'No forecast for {day}')
        return f'Selected {day}'

    @server.tool(meta={'ui': {'visibility': ['model']}})
    def model_only() -> str:
        """Only the model may call this."""
        return 'model only'  # pragma: no cover

    return server


@pytest.fixture
def agent() -> Agent[None, str]:
    """An agent whose model calls both server tools in one step, then answers.

    A sync fixture, because the first `MCPToolset` on FastMCP 4 imports its task extension from disk.
    """

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(name='get_forecast', json_args='{"city": "Den Haag"}', tool_call_id='forecast-1'),
                1: DeltaToolCall(name='get_time', json_args='{"city": "Den Haag"}', tool_call_id='time-1'),
            }
        else:
            yield 'Sunny-ish.'

    return Agent(FunctionModel(stream_function=stream_function), toolsets=[MCPToolset(weather_server(), id='weather')])


FORECAST_RESULT = {
    'content': [{'type': 'text', 'text': 'Den Haag: 14C'}],
    'structuredContent': {'city': 'Den Haag', 'temp_c': 14},
    '_meta': {'selectedDay': 'tuesday', **SERVER_INFO},
    'isError': False,
}
"""The `CallToolResult` the server returned for `get_forecast`, as the app expects to receive it."""


def model_view(messages: list[ModelMessage]) -> Any:
    """The loaded history as the model sees it, without the load-time timestamps that differ between loads."""

    def drop_timestamps(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: drop_timestamps(v) for k, v in value.items() if k != 'timestamp'}  # pyright: ignore[reportUnknownVariableType]
        if isinstance(value, list):
            return [drop_timestamps(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
        return value

    return drop_timestamps(ModelMessagesTypeAdapter.dump_python(messages, mode='json'))


async def test_vercel_ai_frontend_receives_app_after_tool_result(agent: Agent[None, str]):
    """The UI tool's call produces one `data-mcp-apps` chunk, after its tool output, carrying the server's full result."""
    request = SubmitMessage(id='chat', messages=[UIMessage(id='u1', role='user', parts=[TextUIPart(text='Weather?')])])
    adapter = VercelAIAdapter(agent, request)
    chunks = [
        json.loads(encoded.removeprefix('data: '))
        async for encoded in adapter.encode_stream(adapter.run_stream())
        if '[DONE]' not in encoded
    ]

    app_chunks = [chunk for chunk in chunks if chunk['type'] == 'data-mcp-apps']
    assert app_chunks == snapshot(
        [
            {
                'type': 'data-mcp-apps',
                'id': 'forecast-1',
                'data': {
                    'resourceUri': 'ui://weather/forecast',
                    'toolInput': {'city': 'Den Haag'},
                    'result': FORECAST_RESULT,
                    'serverId': 'weather',
                    'serverHash': 'weather',
                },
            }
        ]
    )
    # A frontend mounts the app when it sees the chunk, and hands it the result it already has.
    tool_output = next(
        i for i, c in enumerate(chunks) if c['type'] == 'tool-output-available' and c['toolCallId'] == 'forecast-1'
    )
    assert chunks.index(app_chunks[0]) > tool_output

    # Dumped history carries the same chunk, so a reloaded conversation can mount the app again.
    async with agent.run_stream('Weather?') as result:
        await result.get_output()
    dumped = VercelAIAdapter.dump_messages(result.all_messages())
    dumped_app_parts = [
        part.model_dump(by_alias=True, exclude_none=True)
        for message in dumped
        for part in message.parts
        if part.type == 'data-mcp-apps'
    ]
    assert dumped_app_parts == app_chunks

    # Loading that history back gives the model exactly what it had without the app part.
    without_app = [
        message.model_copy(update={'parts': [part for part in message.parts if part.type != 'data-mcp-apps']})
        for message in dumped
    ]
    assert model_view(VercelAIAdapter.load_messages(dumped)) == model_view(VercelAIAdapter.load_messages(without_app))


@pytest.mark.skipif(
    not imports_successful() or parse_ag_ui_version(detect_ag_ui_version()) < ACTIVITY_EVENTS_VERSION,
    reason='requires ag-ui-protocol with activity events',
)
async def test_ag_ui_frontend_receives_app_activity_after_tool_result(agent: Agent[None, str]):
    """The UI tool's call produces one `mcp-apps` activity, after its tool result, in the shape the AG-UI MCP Apps middleware emits.

    The middleware is the reference producer that frontends such as CopilotKit validate against: its activity content
    has `result`, `resourceUri`, `serverHash`, `serverId` and `toolInput`, and CopilotKit requires `serverHash`.
    See `middlewares/mcp-apps-middleware/src/index.ts` in https://github.com/ag-ui-protocol/ag-ui.
    """
    run_input = RunAgentInput(
        thread_id='thread',
        run_id='run',
        messages=[UserMessage(id='u1', content='Weather?')],
        state={},
        context=[],
        tools=[],
        forwarded_props=None,
    )
    adapter = AGUIAdapter(agent=agent, run_input=run_input, ag_ui_version='0.1.19')
    events = [
        json.loads(encoded.removeprefix('data: ')) async for encoded in adapter.encode_stream(adapter.run_stream())
    ]

    activities = [event for event in events if event['type'] == 'ACTIVITY_SNAPSHOT']
    assert activities == snapshot(
        [
            {
                'type': 'ACTIVITY_SNAPSHOT',
                'timestamp': IsInt(),
                'messageId': IsStr(),
                'activityType': 'mcp-apps',
                'content': {
                    'resourceUri': 'ui://weather/forecast',
                    'toolInput': {'city': 'Den Haag'},
                    'result': FORECAST_RESULT,
                    'serverId': 'weather',
                    'serverHash': 'weather',
                },
                'replace': True,
            }
        ]
    )
    tool_result = next(
        i for i, e in enumerate(events) if e['type'] == 'TOOL_CALL_RESULT' and e['toolCallId'] == 'forecast-1'
    )
    assert events.index(activities[0]) > tool_result

    # Dumped history carries the same activity, and loading it back adds nothing the model sees.
    async with agent.run_stream('Weather?') as result:
        await result.get_output()
    dumped = AGUIAdapter.dump_messages(result.all_messages())
    [dumped_activity] = [message for message in dumped if isinstance(message, ActivityMessage)]
    assert dumped_activity.content == activities[0]['content']
    without_activity = [message for message in dumped if not isinstance(message, ActivityMessage)]
    assert model_view(AGUIAdapter.load_messages(dumped)) == model_view(AGUIAdapter.load_messages(without_activity))


@pytest.fixture
def weather() -> MCPToolset[None]:
    """The toolset the page's requests are relayed to, built outside the event loop like `agent`."""
    return MCPToolset(weather_server(), id='weather')


async def post_run(
    app: Starlette, forwarded_props: dict[str, Any], messages: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """POST an AG-UI run to `app` the way `@ag-ui/client` does, and return the decoded events."""
    body = {
        'threadId': 'thread',
        'runId': 'run',
        'messages': messages or [],
        'state': {},
        'tools': [],
        'context': [],
        'forwardedProps': forwarded_props,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://host') as client:
        response = await client.post('/agui', json=body, headers={'accept': 'text/event-stream'})
    return [json.loads(line.removeprefix('data: ')) for line in response.text.splitlines() if line.startswith('data: ')]


def agui_app(agent: Agent[None, str], mcp_app_toolsets: list[MCPToolset[None]]) -> Starlette:
    async def agui(request: Request) -> Response:
        return await AGUIAdapter.dispatch_request(request, agent=agent, mcp_app_toolsets=mcp_app_toolsets)

    return Starlette(routes=[Route('/agui', agui, methods=['POST'])])


@dataclass(frozen=True)
class ProxiedRequestCase:
    id: str
    request: dict[str, Any]
    result: Any


PROXIED_REQUEST_CASES = [
    ProxiedRequestCase(
        id='read-ui-resource',
        request={
            'serverHash': 'weather',
            'serverId': 'weather',
            'method': 'resources/read',
            'params': {'uri': 'ui://weather/forecast'},
        },
        result={
            'contents': [
                {
                    'uri': 'ui://weather/forecast',
                    'mimeType': 'text/html;profile=mcp-app',
                    'text': '<!doctype html><html><body>forecast</body></html>',
                }
            ],
            **READ_FIELDS,
        },
    ),
    ProxiedRequestCase(
        id='call-app-only-tool',
        request={
            'serverHash': 'weather',
            'serverId': 'weather',
            'method': 'tools/call',
            'params': {'name': 'select_day', 'arguments': {'day': 'friday'}},
        },
        result={
            'content': [{'type': 'text', 'text': 'Selected friday'}],
            'structuredContent': {'result': 'Selected friday'},
            '_meta': {'fastmcp': {'wrap_result': True}, **SERVER_INFO},
            'isError': False,
        },
    ),
    ProxiedRequestCase(
        id='tool-error-is-a-result',
        request={
            'serverHash': 'weather',
            'serverId': 'weather',
            'method': 'tools/call',
            'params': {'name': 'select_day', 'arguments': {'day': 'sunday'}},
        },
        result={
            'content': [{'type': 'text', 'text': 'No forecast for sunday'}],
            **({'_meta': SERVER_INFO} if SERVER_INFO else {}),
            'isError': True,
        },
    ),
    ProxiedRequestCase(
        id='server-hash-only',
        request={
            'serverHash': 'weather',
            'method': 'tools/call',
            'params': {'name': 'select_day', 'arguments': {'day': 'monday'}},
        },
        result={
            'content': [{'type': 'text', 'text': 'Selected monday'}],
            'structuredContent': {'result': 'Selected monday'},
            '_meta': {'fastmcp': {'wrap_result': True}, **SERVER_INFO},
            'isError': False,
        },
    ),
    ProxiedRequestCase(
        id='model-only-tool-refused',
        request={
            'serverHash': 'weather',
            'serverId': 'weather',
            'method': 'tools/call',
            'params': {'name': 'model_only'},
        },
        result=snapshot({'error': "Tool 'model_only' is not available to apps"}),
    ),
    ProxiedRequestCase(
        id='unknown-tool-refused',
        request={'serverHash': 'weather', 'serverId': 'weather', 'method': 'tools/call', 'params': {'name': 'nope'}},
        result=snapshot({'error': "Tool 'nope' is not available to apps"}),
    ),
    ProxiedRequestCase(
        id='unknown-server',
        request={
            'serverHash': 'other',
            'serverId': 'other',
            'method': 'resources/read',
            'params': {'uri': 'ui://weather/forecast'},
        },
        result=snapshot({'error': 'Unknown MCP server'}),
    ),
    ProxiedRequestCase(
        id='unsupported-method',
        request={'serverHash': 'weather', 'serverId': 'weather', 'method': 'prompts/list', 'params': {}},
        result=snapshot({'error': 'MCP method not supported: prompts/list'}),
    ),
    ProxiedRequestCase(
        id='invalid-params',
        request={'serverHash': 'weather', 'serverId': 'weather', 'method': 'tools/call', 'params': {'arguments': {}}},
        result=snapshot({'error': 'Invalid MCP params'}),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id) for c in PROXIED_REQUEST_CASES])
async def test_ag_ui_answers_proxied_mcp_app_requests(case: ProxiedRequestCase, weather: MCPToolset[None]):
    """A page's MCP request sent the way CopilotKit sends it is answered from the server, without running the model.

    CopilotKit's renderer fetches the app's HTML and relays the app's tool calls by starting an agent run with
    `forwardedProps.__proxiedMCPRequest`, and reads the answer from `RUN_FINISHED.result`.
    """

    async def no_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:  # pragma: no cover
        raise AssertionError('the model must not run for a proxied MCP request')
        yield

    agent = Agent(FunctionModel(stream_function=no_model))
    events = await post_run(agui_app(agent, [weather]), {'__proxiedMCPRequest': case.request})
    assert [event['type'] for event in events] == ['RUN_STARTED', 'RUN_FINISHED']
    assert events[1]['result'] == case.result


async def test_ag_ui_runs_the_agent_for_proxied_requests_without_mcp_app_toolsets(agent: Agent[None, str]):
    """Without `mcp_app_toolsets`, a run carrying a proxied MCP request runs the agent as before."""
    request = {'serverId': 'weather', 'method': 'resources/read', 'params': {'uri': 'ui://weather/forecast'}}
    events = await post_run(
        agui_app(agent, []),
        {'__proxiedMCPRequest': request},
        messages=[{'id': 'u1', 'role': 'user', 'content': 'Weather?'}],
    )
    finished = events[-1]
    assert finished['type'] == 'RUN_FINISHED'
    assert 'result' not in finished
    assert any(event['type'] == 'TOOL_CALL_START' for event in events)
