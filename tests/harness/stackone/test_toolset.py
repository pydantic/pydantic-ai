"""Tests for `StackOneToolset` wire construction, filtering, and tool execution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip('fastmcp')

from fastmcp.client.transports import FastMCPTransport, PythonStdioTransport, StreamableHttpTransport
from pydantic import AnyUrl
from pydantic_ai.exceptions import UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.stackone import StackOneToolset, ToolMode

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

pytestmark = pytest.mark.anyio


def http_transport(toolset: StackOneToolset[None]) -> StreamableHttpTransport:
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


class TestStackOneToolset:
    def test_is_mcp_toolset(self, stackone_server: FastMCP):
        assert isinstance(StackOneToolset(account_id='45320', client=stackone_server), MCPToolset)

    def test_default_url_headers_and_id(self):
        toolset = StackOneToolset[None](account_id='45320', api_key='key')
        transport = http_transport(toolset)
        assert transport.url == 'https://api.stackone.com/mcp?tool-mode=search_execute'
        assert transport.headers['Authorization'].startswith('Basic ')
        assert transport.headers['x-account-id'] == '45320'
        assert toolset.id == 'stackone'

    def test_default_mode_resolution(self):
        with_actions = StackOneToolset[None](account_id='1', api_key='key', actions=['*_list_*'])
        assert http_transport(with_actions).url == 'https://api.stackone.com/mcp'
        explicit = StackOneToolset[None](account_id='1', api_key='key', tool_mode='individual')
        assert http_transport(explicit).url == 'https://api.stackone.com/mcp'

    def test_custom_id_reaches_the_connection(self):
        toolset = StackOneToolset(account_id='45320', api_key='key', id='stackone_eu')
        assert toolset.id == 'stackone_eu'

    def test_custom_base_url(self):
        toolset = StackOneToolset[None](
            account_id='45320', api_key='key', base_url='https://api.eu1.stackone.com/', tool_mode='individual'
        )
        assert http_transport(toolset).url == 'https://api.eu1.stackone.com/mcp'

    @pytest.mark.parametrize('base_url', ['ftp://api.stackone.com', 'localhost:9999', 'https://'])
    def test_rejects_invalid_base_url(self, base_url: str):
        with pytest.raises(UserError, match='`base_url` must be an absolute HTTPS URL'):
            StackOneToolset(account_id='45320', api_key='key', base_url=base_url)

    @pytest.mark.parametrize('suffix', ['?region=eu', '#region-eu'])
    def test_rejects_base_url_query_and_fragment(self, suffix: str):
        with pytest.raises(UserError, match='`base_url` must not contain a query or fragment'):
            StackOneToolset(account_id='45320', api_key='key', base_url=f'https://proxy.example{suffix}')

    @pytest.mark.parametrize('client', ['http://api.stackone.com/mcp', AnyUrl('http://api.stackone.com/mcp')])
    def test_rejects_insecure_http_client(self, client: str | AnyUrl):
        with pytest.raises(UserError, match='`client` must be an absolute HTTPS URL'):
            StackOneToolset(account_id='45320', api_key='key', client=client)

    def test_non_url_client_ignores_base_url(self, tmp_path: Path):
        script = tmp_path / 'server.py'
        script.write_text('', encoding='utf-8')
        toolset = StackOneToolset(account_id='45320', client=str(script), base_url='not-used')
        transport = toolset.client.transport
        assert isinstance(transport, PythonStdioTransport)
        assert transport.script_path == script

    def test_search_execute_url(self):
        toolset = StackOneToolset[None](account_id='45320', api_key='key', tool_mode='search_execute')
        assert http_transport(toolset).url == 'https://api.stackone.com/mcp?tool-mode=search_execute'

    def test_search_execute_param_appended_to_custom_urls(self):
        plain = StackOneToolset[None](
            account_id='1', api_key='key', tool_mode='search_execute', client='https://proxy.example/mcp'
        )
        assert http_transport(plain).url == 'https://proxy.example/mcp?tool-mode=search_execute'
        with_query = StackOneToolset[None](
            account_id='1', api_key='key', tool_mode='search_execute', client='https://proxy.example/mcp?region=eu'
        )
        assert http_transport(with_query).url == 'https://proxy.example/mcp?region=eu&tool-mode=search_execute'

    def test_any_url_gets_headers_and_tool_mode(self):
        toolset = StackOneToolset[None](
            account_id='1',
            api_key='key',
            tool_mode='search_execute',
            client=AnyUrl('https://proxy.example/mcp?region=eu'),
        )
        transport = http_transport(toolset)
        assert transport.url == 'https://proxy.example/mcp?region=eu&tool-mode=search_execute'
        assert transport.headers['Authorization'].startswith('Basic ')
        assert transport.headers['x-account-id'] == '1'

    def test_url_scheme_is_case_insensitive(self):
        toolset = StackOneToolset[None](account_id='1', api_key='key', client='HTTPS://proxy.example/mcp')
        assert http_transport(toolset).headers['Authorization'].startswith('Basic ')

    def test_custom_url_tool_mode_matches_configuration(self):
        search_execute_url = 'https://proxy.example/mcp?tool%2Dmode=search%5Fexecute&signature=a%2fb%20c&flag#fragment'
        search_execute = StackOneToolset[None](
            account_id='1',
            api_key='key',
            tool_mode='search_execute',
            client=search_execute_url,
        )
        assert http_transport(search_execute).url == search_execute_url
        individual_url = 'https://proxy.example/mcp?signature=a%2fb%20c&tool-mode=individual&flag#fragment'
        individual = StackOneToolset[None](
            account_id='1',
            api_key='key',
            tool_mode='individual',
            client=individual_url,
        )
        assert http_transport(individual).url == individual_url

    @pytest.mark.parametrize(
        ('tool_mode', 'client'),
        [
            (
                'search_execute',
                'https://proxy.example/mcp?opaque=a%2fb%20c&flag&not-tool-mode=individual'
                '&tool%2Dmode=individual#fragment',
            ),
            ('individual', 'https://proxy.example/mcp?region=eu&tool-mode=search_execute'),
            (
                'search_execute',
                'https://proxy.example/mcp?tool-mode=search_execute&tool%2Dmode=search%5Fexecute',
            ),
        ],
    )
    def test_custom_url_conflicting_tool_mode_is_rejected(self, tool_mode: ToolMode, client: str):
        with pytest.raises(
            UserError,
            match='conflicts with the configured `tool_mode`.*rewriting would invalidate signed URLs',
        ):
            StackOneToolset(
                account_id='1',
                api_key='key',
                tool_mode=tool_mode,
                client=client,
            )

    def test_no_headers_for_non_url_clients(self, stackone_server: FastMCP):
        toolset = StackOneToolset(account_id='45320', api_key='key', client=stackone_server)
        transport = toolset.client.transport
        assert isinstance(transport, FastMCPTransport)
        assert transport.server is stackone_server

    def test_prebuilt_http_client_keeps_its_configuration(self):
        from fastmcp import Client

        client = Client('http://proxy.example/mcp')
        toolset = StackOneToolset(account_id='45320', api_key='key', client=client)
        assert toolset.client is client
        transport = client.transport
        assert isinstance(transport, StreamableHttpTransport)
        assert transport.url == 'http://proxy.example/mcp'
        assert transport.headers == {}

    def test_missing_api_key_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='STACKONE_API_KEY'):
            StackOneToolset(account_id='45320')

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        toolset = StackOneToolset[None](account_id='45320')
        assert http_transport(toolset).headers['Authorization'].startswith('Basic ')

    def test_rejects_actions_in_search_execute(self):
        with pytest.raises(UserError, match='cannot apply in `search_execute` mode'):
            StackOneToolset(account_id='1', api_key='key', tool_mode='search_execute', actions=['*_list_*'])

    async def test_actions_filter_is_case_insensitive(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(
            account_id='1',
            api_key='key',
            client=stackone_server,
            actions=['*_LIST_*'],
            # `task` collides with a server-provided key: user metadata must win,
            # matching `.with_metadata()` merge order.
            metadata={'meta': 'user', 'task': True},
        )
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert set(tools) == {'bamboohr_list_employees'}
        assert tools['bamboohr_list_employees'].tool_def.metadata == {
            'meta': 'user',
            'annotations': None,
            'task': True,
        }

    async def test_call_tool_executes_via_the_connection(self, stackone_server: FastMCP, run_context: RunContext[None]):
        toolset = StackOneToolset(account_id='1', api_key='key', client=stackone_server)
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_create_employee', {'name': 'Grace'}, run_context, tools['bamboohr_create_employee']
            )
        assert 'Grace' in str(result)

    async def test_visit_and_replace_returns_stackone_toolset(
        self, stackone_server: FastMCP, run_context: RunContext[None]
    ):
        toolset = StackOneToolset(account_id='1', client=stackone_server)
        rewritten = toolset.visit_and_replace(lambda inner: inner)
        assert rewritten is toolset
        async with toolset:
            tools = await toolset.get_tools(run_context)
            result = await toolset.call_tool(
                'bamboohr_create_employee',
                {'name': 'Grace'},
                run_context,
                tools['bamboohr_create_employee'],
            )
        assert 'Grace' in str(result)

    async def test_prebuilt_fastmcp_client(self, stackone_server: FastMCP, run_context: RunContext[None]):
        from fastmcp import Client

        toolset = StackOneToolset(account_id='1', api_key='key', client=Client(stackone_server))
        async with toolset:
            tools = await toolset.get_tools(run_context)
        assert 'bamboohr_list_employees' in tools
