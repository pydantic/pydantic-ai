"""MCP Apps from a real MCP server through an agent run to the messages a frontend receives.

The tests use an in-process FastMCP server and `MCPToolset`, and the frontend tests check the UI adapters' encoded
stream and dumped history. Only the malformed-record test builds app data by hand.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ToolReturn, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from ._inline_snapshot import snapshot
from .conftest import IsInt, IsStr, iter_message_parts, try_import

with try_import() as imports_successful:
    from ag_ui.core import RunAgentInput, ToolMessage, UserMessage
    from fastmcp import Client, Context, FastMCP
    from fastmcp.exceptions import McpError, ToolError
    from fastmcp.tools import ToolResult as FastMCPToolResult
    from mcp import types as mcp_types
    from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent

    from pydantic_ai._mcp_compat import is_mcp_sdk_v2
    from pydantic_ai.mcp import CallToolFunc, MCPError, MCPToolset, ToolResult
    from pydantic_ai.ui.ag_ui import AGUIAdapter
    from pydantic_ai.ui.ag_ui._utils import ACTIVITY_EVENTS_VERSION, detect_ag_ui_version, parse_ag_ui_version
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter
    from pydantic_ai.ui.vercel_ai._utils import mcp_app_model_output
    from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage, TextUIPart, ToolOutputAvailablePart, UIMessage

with try_import() as logfire_imports_successful:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.skipif(not imports_successful(), reason='fastmcp, ag-ui-protocol or starlette not installed')

MCP_SDK_V2 = imports_successful() and is_mcp_sdk_v2()

# FastMCP 4 servers stamp every result's `_meta` with their identity, which the app data passes through.
SERVER_INFO: dict[str, Any] = (
    {'io.modelcontextprotocol/serverInfo': {'name': 'weather_server', 'version': IsStr()}} if MCP_SDK_V2 else {}
)
READ_FIELDS: dict[str, Any] = (
    {'resultType': 'complete', 'cacheScope': 'private', 'ttlMs': 0, '_meta': SERVER_INFO} if SERVER_INFO else {}
)
"""The fields FastMCP 4 adds to every `resources/read` result."""

TIME_ANNOTATIONS = {
    'title': None,
    'readOnlyHint': True,
    'destructiveHint': None,
    'idempotentHint': None,
    'openWorldHint': None,
}
"""`get_time`'s annotations as `MCPToolset` records them: every spec key, unset ones `None`."""


def weather_server() -> FastMCP[None]:
    """Tools with an MCP Apps UI (under the current and the deprecated `_meta` key), a tool without one, tools only
    the app or only the model may call, and the UI resource. Titles are explicit because FastMCP 4 generates one
    from the function name and FastMCP 3 doesn't.
    """
    server: FastMCP[None] = FastMCP('weather_server')

    @server.resource(
        'ui://weather/forecast',
        mime_type='text/html;profile=mcp-app',
        meta={'ui': {'csp': {'connectDomains': ['https://api.weather.example']}}},
    )
    def forecast_view() -> str:
        return '<!doctype html><html><body>forecast</body></html>'

    @server.tool(title='Forecast', meta={'ui': {'resourceUri': 'ui://weather/forecast'}})
    def get_forecast(city: str, api_key: str | None = None) -> FastMCPToolResult:
        """Text for the model; structured content and per-call `_meta` for the app. `api_key` is for a hook to add."""
        return FastMCPToolResult(
            content=[TextContent(type='text', text=f'{city}: 14C')],
            structured_content={'city': city, 'temp_c': 14},
            meta={'selectedDay': 'tuesday'},
        )

    @server.tool(title='Legacy forecast', meta={'ui/resourceUri': 'ui://weather/forecast'})
    def legacy_forecast(city: str) -> FastMCPToolResult:
        return FastMCPToolResult(content=[TextContent(type='text', text=f'{city}: 14C')])

    @server.tool(title='Local time', annotations={'readOnlyHint': True})
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

    @server.tool
    def supports_mcp_apps(ctx: Context) -> bool:
        """Whether the client told the server it supports MCP Apps."""
        return ctx.client_supports_extension('io.modelcontextprotocol/ui')

    return server


@pytest.fixture
def weather() -> MCPToolset[Any]:
    """The toolset for `weather_server`, built in a sync fixture so FastMCP 4's task-extension import runs outside the event loop."""
    return MCPToolset(weather_server(), id='weather')


@pytest.fixture
def agent(weather: MCPToolset[Any]) -> Agent[None, str]:
    """An agent whose model calls the UI tool and the plain tool in one step, then answers."""

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(name='get_forecast', json_args='{"city": "Den Haag"}', tool_call_id='forecast-1'),
                1: DeltaToolCall(name='get_time', json_args='{"city": "Den Haag"}', tool_call_id='time-1'),
            }
        else:
            yield 'Sunny-ish.'

    return Agent(FunctionModel(stream_function=stream_function), toolsets=[weather])


def forecast_result(city: str) -> dict[str, Any]:
    """The `CallToolResult` the server returns for `get_forecast`, as the app expects to receive it."""
    return {
        'content': [{'type': 'text', 'text': f'{city}: 14C'}],
        'structuredContent': {'city': city, 'temp_c': 14},
        '_meta': {'selectedDay': 'tuesday', **SERVER_INFO},
        'isError': False,
    }


def model_view(messages: list[ModelMessage]) -> Any:
    """The loaded history as the model sees it, without the load-time timestamps that differ between loads."""

    def drop_timestamps(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: drop_timestamps(v) for k, v in value.items() if k != 'timestamp'}  # pyright: ignore[reportUnknownVariableType]
        if isinstance(value, list):
            return [drop_timestamps(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
        return value

    return drop_timestamps(ModelMessagesTypeAdapter.dump_python(messages, mode='json'))


async def test_server_declares_mcp_apps_extension(weather: MCPToolset[Any]):
    async with weather:
        assert weather.capabilities.extensions == snapshot({'io.modelcontextprotocol/ui': {}})


async def test_tool_returns_carry_tool_info_and_app_data(weather: MCPToolset[Any]):
    """Every call records which server and tool it went to; calls to UI tools also keep the full result for the app.

    Neither changes what the model sees.
    """
    agent = Agent(TestModel(call_tools=['get_forecast', 'legacy_forecast', 'get_time']), toolsets=[weather])
    result = await agent.run('Forecast?')
    returns = {
        part.tool_name: (part.content, part.metadata)
        for part in iter_message_parts(result.all_messages(), ModelRequest, ToolReturnPart)
    }
    assert returns == {
        'get_forecast': (
            {'city': 'a', 'temp_c': 14},
            {
                'mcp_tool': {'toolset_id': 'weather', 'title': 'Forecast', 'annotations': None},
                'mcp_app': {
                    'resource_uri': 'ui://weather/forecast',
                    'tool_input': {'city': 'a'},
                    'result': forecast_result('a'),
                    'toolset_id': 'weather',
                },
            },
        ),
        'legacy_forecast': (
            'a: 14C',
            {
                'mcp_tool': {'toolset_id': 'weather', 'title': 'Legacy forecast', 'annotations': None},
                'mcp_app': {
                    'resource_uri': 'ui://weather/forecast',
                    'tool_input': {'city': 'a'},
                    'result': {
                        'content': [{'type': 'text', 'text': 'a: 14C'}],
                        **({'_meta': SERVER_INFO} if SERVER_INFO else {}),
                        'isError': False,
                    },
                    'toolset_id': 'weather',
                },
            },
        ),
        'get_time': (
            'a: 09:00',
            {'mcp_tool': {'toolset_id': 'weather', 'title': 'Local time', 'annotations': TIME_ANNOTATIONS}},
        ),
    }


HookKind = Literal[
    'passthrough',
    'metadata',
    'added-key',
    'rewrite',
    'rewrite-in-place',
    'other-tool',
    'other-arguments',
    'arguments-in-place',
    'tool-return',
]


@pytest.fixture
def hooked_weather(request: pytest.FixtureRequest) -> MCPToolset[Any]:
    """The toolset for `weather_server` with a `process_tool_call` hook of the kind in `request.param`."""
    hook_kind: HookKind = request.param

    async def hook(ctx: RunContext[Any], call_tool: CallToolFunc, name: str, args: dict[str, Any]) -> ToolResult:
        if hook_kind == 'other-tool':
            return await call_tool('get_time', args)
        if hook_kind == 'other-arguments':
            return await call_tool(name, {'city': 'b'})
        if hook_kind == 'arguments-in-place':
            args['city'] = 'b'
        if hook_kind == 'metadata':
            return await call_tool(name, args, metadata={'deps': 42})
        if hook_kind == 'added-key':
            return await call_tool(name, {**args, 'api_key': 'server-only'})
        result = await call_tool(name, args)
        if hook_kind == 'rewrite':
            assert isinstance(result, dict)
            return {'temp_c': result['temp_c']}
        if hook_kind == 'rewrite-in-place':
            assert isinstance(result, dict)
            del result['city']
        if hook_kind == 'tool-return':
            return cast(Any, ToolReturn(return_value=result, metadata={'own': True}))
        return result

    return MCPToolset(weather_server(), process_tool_call=hook)


@pytest.mark.parametrize(
    ('hooked_weather', 'expected', 'app_city'),
    [
        pytest.param('passthrough', {'city': 'a', 'temp_c': 14}, 'a', id='passthrough'),
        pytest.param('metadata', {'city': 'a', 'temp_c': 14}, 'a', id='metadata'),
        pytest.param('added-key', {'city': 'a', 'temp_c': 14}, 'a', id='added-key'),
        pytest.param('rewrite', {'temp_c': 14}, None, id='rewritten-result'),
        pytest.param('rewrite-in-place', {'temp_c': 14}, None, id='result-rewritten-in-place'),
        pytest.param('other-tool', 'a: 09:00', None, id='other-tool'),
        pytest.param('other-arguments', {'city': 'b', 'temp_c': 14}, 'b', id='other-arguments'),
        pytest.param('arguments-in-place', {'city': 'b', 'temp_c': 14}, 'b', id='arguments-rewritten-in-place'),
    ],
    indirect=['hooked_weather'],
)
async def test_app_data_with_process_tool_call(hooked_weather: MCPToolset[Any], expected: Any, app_city: str | None):
    """App data is kept when a `process_tool_call` hook returns the result of its call to the tool unchanged.

    The app gets the model's arguments, not ones the hook added, such as a key. A hook that changes the result,
    including by editing it in place, or doesn't call the tool, decides what leaves the toolset, so no raw result
    is forwarded. The tool record is always attached.
    """
    run_context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    async with hooked_weather:
        tools = await hooked_weather.get_tools(run_context)
        returned = await hooked_weather.call_tool('get_forecast', {'city': 'a'}, run_context, tools['get_forecast'])
    assert returned.return_value == expected
    metadata: dict[str, Any] = {'mcp_tool': {'toolset_id': None, 'title': 'Forecast', 'annotations': None}}
    if app_city is not None:
        metadata['mcp_app'] = {
            'resource_uri': 'ui://weather/forecast',
            'tool_input': {'city': 'a'},
            'result': forecast_result(app_city),
            'toolset_id': None,
        }
    assert returned.metadata == metadata


@pytest.mark.skipif(not logfire_imports_successful(), reason='logfire not installed')
async def test_tool_span_omits_app_binary_without_binary_content(capfire: CaptureLogfire):
    """`include_binary_content=False` also drops the base64 data the app record keeps in wire form."""
    server: FastMCP[None] = FastMCP('charts')

    @server.tool(meta={'ui': {'resourceUri': 'ui://charts/view'}})
    def chart() -> FastMCPToolResult:
        return FastMCPToolResult(
            content=[
                TextContent(type='text', text='A chart'),
                ImageContent(type='image', data='aGk=', mimeType='image/png'),
                EmbeddedResource(
                    type='resource',
                    resource=BlobResourceContents.model_validate(
                        {'uri': 'file:///chart.csv', 'blob': 'aGk=', 'mimeType': 'text/csv'}
                    ),
                ),
            ]
        )

    agent = Agent(
        TestModel(call_tools=['chart']),
        toolsets=[MCPToolset(server, id='charts')],
        capabilities=[Instrumentation(settings=InstrumentationSettings(include_binary_content=False))],
    )
    await agent.run('Chart?')

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    tool_result = next(span for span in spans if span['name'] == 'execute_tool chart')['attributes'][
        'gen_ai.tool.call.result'
    ]
    assert tool_result['metadata']['mcp_app']['result']['content'] == snapshot(
        [
            {'type': 'text', 'text': 'A chart'},
            {'type': 'image', 'mimeType': 'image/png'},
            {'type': 'resource', 'resource': {'uri': 'file:///chart.csv', 'mimeType': 'text/csv'}},
        ]
    )


UI_RESOURCE_READ = {
    'contents': [
        {
            'uri': 'ui://weather/forecast',
            'mimeType': 'text/html;profile=mcp-app',
            '_meta': {'ui': {'csp': {'connectDomains': ['https://api.weather.example']}}},
            'text': '<!doctype html><html><body>forecast</body></html>',
        }
    ],
    **READ_FIELDS,
}
"""The `ReadResourceResult` the server returns for the UI resource, as the page expects to receive it."""


async def test_read_resource_for_app(weather: MCPToolset[Any]):
    """Reading the app's page keeps the MIME type and `_meta.ui` that `read_resource` drops, in wire form."""
    assert await weather.read_resource_for_app('ui://weather/forecast') == UI_RESOURCE_READ
    with pytest.raises(MCPError, match='ui://missing'):
        await weather.read_resource_for_app('ui://missing')


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
                    'result': forecast_result('Den Haag'),
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


async def test_vercel_ai_v7_tool_outputs_carry_tool_metadata(agent: Agent[None, str]):
    """On AI SDK v7, every MCP tool output carries `toolMetadata` in the shape AI SDK's own MCP client sets.

    A tool without a page gets `clientName`, `toolName`, `title` and `annotations`; `ToolDefinition.metadata` is
    never sent as is. A tool with a page also gets `app`, and its `output` is the server's full result, which
    `experimental_MCPAppRenderer` hands the page. What the model saw is kept in the provider metadata, so history the
    frontend sends back loads as it does on v6.
    """
    request = SubmitMessage(id='chat', messages=[UIMessage(id='u1', role='user', parts=[TextUIPart(text='Weather?')])])
    adapter = VercelAIAdapter(agent, request, sdk_version=7)
    chunks = [
        json.loads(encoded.removeprefix('data: '))
        async for encoded in adapter.encode_stream(adapter.run_stream())
        if '[DONE]' not in encoded
    ]
    outputs = {c['toolCallId']: c for c in chunks if c['type'] == 'tool-output-available'}
    assert outputs['time-1'] == snapshot(
        {
            'type': 'tool-output-available',
            'toolCallId': 'time-1',
            'output': 'Den Haag: 09:00',
            'toolMetadata': {
                'clientName': 'weather',
                'toolName': 'get_time',
                'title': 'Local time',
                'annotations': {'readOnlyHint': True},
            },
        }
    )
    assert outputs['forecast-1'] == snapshot(
        {
            'type': 'tool-output-available',
            'toolCallId': 'forecast-1',
            'output': forecast_result('Den Haag'),
            'providerMetadata': {'pydantic_ai': {'model_output': {'city': 'Den Haag', 'temp_c': 14}}},
            'toolMetadata': {
                'clientName': 'weather',
                'toolName': 'get_forecast',
                'title': 'Forecast',
                'app': {'resourceUri': 'ui://weather/forecast', 'mimeType': 'text/html;profile=mcp-app'},
            },
        }
    )

    async with agent.run_stream('Weather?') as result:
        await result.get_output()
    sent_back = [
        UIMessage.model_validate_json(message.model_dump_json(by_alias=True, exclude_none=True))
        for message in VercelAIAdapter.dump_messages(result.all_messages(), sdk_version=7)
    ]
    v6 = VercelAIAdapter.dump_messages(result.all_messages(), sdk_version=6)
    assert model_view(VercelAIAdapter.load_messages(sent_back)) == model_view(VercelAIAdapter.load_messages(v6))


@pytest.mark.parametrize('hooked_weather', ['tool-return'], indirect=True)
async def test_process_tool_call_returning_a_tool_return_owns_the_result(hooked_weather: MCPToolset[Any]):
    """A hook that builds its own `ToolReturn` decides what leaves the toolset; no record is added to it."""
    run_context = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    async with hooked_weather:
        tools = await hooked_weather.get_tools(run_context)
        returned = await hooked_weather.call_tool('get_forecast', {'city': 'a'}, run_context, tools['get_forecast'])
    assert returned == ToolReturn(return_value={'city': 'a', 'temp_c': 14}, metadata={'own': True})


async def test_vercel_ai_v7_tool_metadata_from_hand_built_records():
    """Records built by hand: an app record alone, a tool record alone, and malformed or missing records.

    `MCPToolset` always attaches both records, so these shapes only come from other code.
    """

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(name='app_only', json_args='{}', tool_call_id='call_1'),
                1: DeltaToolCall(name='tool_only', json_args='{}', tool_call_id='call_2'),
                2: DeltaToolCall(name='malformed', json_args='{}', tool_call_id='call_3'),
                3: DeltaToolCall(name='other_metadata', json_args='{}', tool_call_id='call_4'),
            }
        else:
            yield 'Done'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    async def app_only() -> ToolReturn:
        app: dict[str, Any] = {
            'resource_uri': 'ui://weather/forecast',
            'tool_input': {},
            'result': {},
            'toolset_id': None,
        }
        return ToolReturn(return_value='app', metadata={'mcp_app': app})

    @agent.tool_plain
    async def tool_only() -> ToolReturn:
        return ToolReturn(
            return_value='tool', metadata={'mcp_tool': {'toolset_id': 'srv', 'title': None, 'annotations': None}}
        )

    @agent.tool_plain
    async def malformed() -> ToolReturn:
        return ToolReturn(return_value='bad', metadata={'mcp_app': {'resource_uri': 1}, 'mcp_tool': {'toolset_id': 1}})

    @agent.tool_plain
    async def other_metadata() -> ToolReturn:
        return ToolReturn(return_value='other', metadata={'theirs': True})

    request = SubmitMessage(id='chat', messages=[UIMessage(id='u1', role='user', parts=[TextUIPart(text='Hi')])])
    adapter = VercelAIAdapter(agent, request, sdk_version=7)
    chunks = [
        json.loads(encoded.removeprefix('data: '))
        async for encoded in adapter.encode_stream(adapter.run_stream())
        if '[DONE]' not in encoded
    ]
    outputs = {c['toolCallId']: c for c in chunks if c['type'] == 'tool-output-available'}
    assert {call_id: c.get('toolMetadata') for call_id, c in outputs.items()} == snapshot(
        {
            'call_1': {
                'toolName': 'app_only',
                'app': {'resourceUri': 'ui://weather/forecast', 'mimeType': 'text/html;profile=mcp-app'},
            },
            'call_2': {'clientName': 'srv', 'toolName': 'tool_only'},
            'call_3': None,
            'call_4': None,
        }
    )
    assert outputs['call_1']['output'] == {}
    assert outputs['call_2']['output'] == 'tool'


@pytest.mark.parametrize(
    ('output', 'expected'),
    [
        pytest.param({'content': [{'type': 'text', 'text': 'a: 14C'}]}, 'a: 14C', id='one-text-block'),
        pytest.param(
            {'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}]}, ['a', 'b'], id='text-blocks'
        ),
        pytest.param(
            {'content': [{'type': 'text', 'text': 'a'}, {'type': 'image', 'data': 'AA==', 'mimeType': 'image/png'}]},
            [{'type': 'text', 'text': 'a'}, {'type': 'image', 'data': 'AA==', 'mimeType': 'image/png'}],
            id='mixed-blocks',
        ),
        pytest.param({'content': 'not a list'}, {'content': 'not a list'}, id='content-not-a-list'),
        pytest.param('plain', 'plain', id='output-not-a-result'),
    ],
)
def test_mcp_app_model_output_rebuilt_from_result(output: Any, expected: Any):
    """Without `structuredContent` and without the model's view, the model gets the result's text blocks."""
    part = ToolOutputAvailablePart(
        type='tool-get_forecast',
        tool_call_id='forecast-1',
        output=output,
        tool_metadata={'app': {'resourceUri': 'ui://weather/forecast', 'mimeType': 'text/html;profile=mcp-app'}},
    )
    assert mcp_app_model_output(part) == expected


async def test_vercel_ai_v7_history_without_model_view(agent: Agent[None, str]):
    """A frontend that sends tool parts back without `resultProviderMetadata` still gives the model its own view.

    The model's view is rebuilt from the result's `structuredContent`; the page's `_meta` never reaches the model.
    """
    async with agent.run_stream('Weather?') as result:
        await result.get_output()
    sent_back: list[UIMessage] = []
    for message in VercelAIAdapter.dump_messages(result.all_messages(), sdk_version=7):
        data = message.model_dump(by_alias=True, exclude_none=True)
        for part in data['parts']:
            part.pop('resultProviderMetadata', None)
        sent_back.append(UIMessage.model_validate(data))
    loaded = VercelAIAdapter.load_messages(sent_back)
    [forecast] = [p for p in iter_message_parts(loaded, ModelRequest, ToolReturnPart) if p.tool_call_id == 'forecast-1']
    assert forecast.content == snapshot({'city': 'Den Haag', 'temp_c': 14})


async def test_vercel_ai_app_chunk_without_toolset_id_or_with_malformed_record():
    """A record from a toolset without an `id` is sent without `serverId`/`serverHash`; a malformed one is not sent.

    The records are built by hand: a malformed one can't come from `MCPToolset`.
    """

    async def stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(name='no_toolset_id', json_args='{}', tool_call_id='call_1'),
                1: DeltaToolCall(name='malformed', json_args='{}', tool_call_id='call_2'),
            }
        else:
            yield 'Done'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    async def no_toolset_id() -> ToolReturn:
        app: dict[str, Any] = {
            'resource_uri': 'ui://weather/forecast',
            'tool_input': {},
            'result': {},
            'toolset_id': None,
        }
        return ToolReturn(return_value='app', metadata={'mcp_app': app})

    @agent.tool_plain
    async def malformed() -> ToolReturn:
        return ToolReturn(return_value='no app', metadata={'mcp_app': {'resource_uri': 1}})

    request = SubmitMessage(id='chat', messages=[UIMessage(id='u1', role='user', parts=[TextUIPart(text='Hi')])])
    adapter = VercelAIAdapter(agent, request)
    chunks = [
        json.loads(encoded.removeprefix('data: '))
        async for encoded in adapter.encode_stream(adapter.run_stream())
        if '[DONE]' not in encoded
    ]
    assert [chunk for chunk in chunks if chunk['type'] == 'data-mcp-apps'] == snapshot(
        [
            {
                'type': 'data-mcp-apps',
                'id': 'call_1',
                'data': {'resourceUri': 'ui://weather/forecast', 'toolInput': {}, 'result': {}},
            }
        ]
    )


def ag_ui_run_input() -> RunAgentInput:
    return RunAgentInput(
        thread_id='thread',
        run_id='run',
        messages=[UserMessage(id='u1', content='Weather?')],
        state={},
        context=[],
        tools=[],
        forwarded_props=None,
    )


@pytest.mark.skipif(
    not imports_successful() or parse_ag_ui_version(detect_ag_ui_version()) < ACTIVITY_EVENTS_VERSION,
    reason='requires ag-ui-protocol with activity events',
)
async def test_ag_ui_frontend_receives_app_activity_after_tool_result(agent: Agent[None, str]):
    """The UI tool's call produces one `mcp-apps` activity, after its tool result, in the shape the AG-UI MCP Apps middleware emits.

    The middleware is the reference producer that frontends such as CopilotKit validate against: its activity content
    has `result`, `resourceUri`, `serverHash`, `serverId` and `toolInput`, and CopilotKit requires `serverHash`.
    """
    from ag_ui.core import ActivityMessage

    adapter = AGUIAdapter(agent=agent, run_input=ag_ui_run_input(), ag_ui_version='0.1.19')
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
                    'result': forecast_result('Den Haag'),
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

    # Dumped history carries the same activity after the tool result, and loading it back adds nothing the model sees.
    async with agent.run_stream('Weather?') as result:
        await result.get_output()
    dumped = AGUIAdapter.dump_messages(result.all_messages())
    [activity_index] = [i for i, message in enumerate(dumped) if isinstance(message, ActivityMessage)]
    [tool_message_index] = [
        i for i, m in enumerate(dumped) if isinstance(m, ToolMessage) and m.tool_call_id == 'forecast-1'
    ]
    assert activity_index > tool_message_index
    assert dumped[activity_index].content == activities[0]['content']
    without_activity = [message for message in dumped if not isinstance(message, ActivityMessage)]
    assert model_view(AGUIAdapter.load_messages(dumped)) == model_view(AGUIAdapter.load_messages(without_activity))


async def test_ag_ui_frontend_without_activity_events_receives_no_app(agent: Agent[None, str]):
    """Frontends on an AG-UI version from before activity events get the tool result and no activity."""
    adapter = AGUIAdapter(agent=agent, run_input=ag_ui_run_input(), ag_ui_version='0.1.10')
    event_types = [
        json.loads(encoded.removeprefix('data: '))['type']
        async for encoded in adapter.encode_stream(adapter.run_stream())
    ]
    assert 'TOOL_CALL_RESULT' in event_types
    assert 'ACTIVITY_SNAPSHOT' not in event_types


def fastmcp_text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    """The `CallToolResult` FastMCP returns for a tool that returns `text`, or raises `ToolError(text)`."""
    if is_error:
        return {
            'content': [{'type': 'text', 'text': text}],
            **({'_meta': SERVER_INFO} if SERVER_INFO else {}),
            'isError': True,
        }
    return {
        'content': [{'type': 'text', 'text': text}],
        'structuredContent': {'result': text},
        '_meta': {'fastmcp': {'wrap_result': True}, **SERVER_INFO},
        'isError': False,
    }


@pytest.mark.parametrize(
    ('name', 'arguments', 'expected'),
    [
        pytest.param('select_day', {'day': 'friday'}, fastmcp_text_result('Selected friday'), id='app-only-tool'),
        pytest.param(
            'select_day',
            {'day': 'sunday'},
            fastmcp_text_result('No forecast for sunday', is_error=True),
            id='error-result',
        ),
        pytest.param('get_time', {'city': 'a'}, fastmcp_text_result('a: 09:00'), id='tool-without-visibility'),
    ],
)
async def test_call_tool_for_app(
    weather: MCPToolset[Any], name: str, arguments: dict[str, Any], expected: dict[str, Any]
):
    """A page's tool calls get the server's full result in wire form, `isError` results included."""
    assert await weather.call_tool_for_app(name, arguments) == expected


@pytest.mark.parametrize('name', ['model_only', 'missing'])
async def test_call_tool_for_app_refuses_tools_hidden_from_apps(weather: MCPToolset[Any], name: str):
    """A page can't call tools hidden from apps or missing from this server; both get the same error."""
    with pytest.raises(MCPError) as exc_info:
        await weather.call_tool_for_app(name)
    assert str(exc_info.value) == f'Tool {name!r} is not available to apps (code: -32602)'


@pytest.mark.parametrize('arguments', ['day=friday', []])
async def test_call_tool_for_app_refuses_arguments_that_are_not_an_object(weather: MCPToolset[Any], arguments: Any):
    """A page's arguments come from its JSON, so ones that aren't an object get an `MCPError` instead of a call."""
    with pytest.raises(MCPError) as exc_info:
        await weather.call_tool_for_app('select_day', arguments)
    assert str(exc_info.value) == 'Tool arguments must be an object (code: -32602)'


async def test_call_tool_for_app_skips_process_tool_call():
    """A page's tool calls don't go through `process_tool_call`, which handles the model's calls."""

    async def hook(
        ctx: RunContext[Any], call_tool: CallToolFunc, name: str, args: dict[str, Any]
    ) -> ToolResult:  # pragma: no cover
        raise AssertionError("a page's tool call must not reach process_tool_call")

    hooked = MCPToolset(weather_server(), process_tool_call=hook)
    assert await hooked.call_tool_for_app('select_day', {'day': 'friday'}) == fastmcp_text_result('Selected friday')


async def test_call_tool_for_app_raises_protocol_errors(weather: MCPToolset[Any], monkeypatch: pytest.MonkeyPatch):
    """A JSON-RPC error from the server's `tools/call` is raised as `MCPError`.

    The error is injected because in-process FastMCP servers turn every tool failure into an `isError` result.
    """
    # SDK v1 wraps an `ErrorData`; v2 takes the fields directly, and CI type-checks against v1.
    error = (
        cast(Any, McpError)(code=-32001, message='call refused')
        if MCP_SDK_V2
        else McpError(mcp_types.ErrorData(code=-32001, message='call refused'))
    )

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(weather.client, 'call_tool', refuse)
    with pytest.raises(MCPError, match='call refused'):
        await weather.call_tool_for_app('select_day', {'day': 'friday'})


@pytest.mark.skipif(not MCP_SDK_V2, reason='FastMCP 3 clients have no client extensions')
async def test_client_extension_announces_mcp_apps_support(weather: MCPToolset[Any]):
    """The documented FastMCP 4 client extension tells servers the client supports MCP Apps; a plain one doesn't."""
    # Only the MCP SDK v2 that FastMCP 4 uses has `advertise`, and CI type-checks against FastMCP 3.
    mcp_client: Any = importlib.import_module('mcp.client')
    ui = mcp_client.advertise('io.modelcontextprotocol/ui', {'mimeTypes': ['text/html;profile=mcp-app']})
    announced = MCPToolset(Client(weather_server(), extensions=[ui]))
    async with weather, announced:
        assert await weather.direct_call_tool('supports_mcp_apps', {}) is False
        assert await announced.direct_call_tool('supports_mcp_apps', {}) is True
