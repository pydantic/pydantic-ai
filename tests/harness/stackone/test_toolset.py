"""Tests for `StackOneToolset` wire construction."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

pytest.importorskip('fastmcp')

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import AnyUrl
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness.stackone import StackOneToolset, ToolMode


def http_transport(toolset: StackOneToolset[None]) -> StreamableHttpTransport:
    transport = toolset.client.transport
    assert isinstance(transport, StreamableHttpTransport)
    return transport


def basic(api_key: str) -> str:
    return 'Basic ' + base64.b64encode(f'{api_key}:'.encode()).decode()


class TestStackOneToolset:
    def test_default_connection(self):
        toolset = StackOneToolset[None](account_id='45320', api_key='key')
        transport = http_transport(toolset)
        assert (transport.url, transport.headers, toolset.id) == (
            'https://api.stackone.com/mcp?tool-mode=search_execute',
            {'Authorization': basic('key'), 'x-account-id': '45320'},
            'stackone',
        )

    @pytest.mark.parametrize(
        ('toolset', 'url'),
        [
            (
                StackOneToolset[None](account_id='1', api_key='key', actions=['*_list_*']),
                'https://api.stackone.com/mcp',
            ),
            (
                StackOneToolset[None](account_id='1', api_key='key', tool_mode='individual'),
                'https://api.stackone.com/mcp',
            ),
            (
                StackOneToolset[None](
                    account_id='1', api_key='key', base_url='https://api.eu1.stackone.com/', tool_mode='individual'
                ),
                'https://api.eu1.stackone.com/mcp',
            ),
            (
                StackOneToolset[None](account_id='1', api_key='key', client='https://proxy.example/mcp'),
                'https://proxy.example/mcp?tool-mode=search_execute',
            ),
            (
                StackOneToolset[None](account_id='1', api_key='key', client='HTTPS://proxy.example/mcp'),
                'https://proxy.example/mcp?tool-mode=search_execute',
            ),
            (
                StackOneToolset[None](account_id='1', api_key='key', client='https://proxy.example/mcp?region=eu'),
                'https://proxy.example/mcp?region=eu&tool-mode=search_execute',
            ),
            (
                StackOneToolset[None](
                    account_id='1', api_key='key', client=AnyUrl('https://proxy.example/mcp?region=eu')
                ),
                'https://proxy.example/mcp?region=eu&tool-mode=search_execute',
            ),
            (
                StackOneToolset[None](
                    account_id='1',
                    api_key='key',
                    client='https://proxy.example/mcp?tool%2Dmode=search%5Fexecute&signature=a%2fb%20c&flag#fragment',
                ),
                'https://proxy.example/mcp?tool%2Dmode=search%5Fexecute&signature=a%2fb%20c&flag#fragment',
            ),
            (
                StackOneToolset[None](
                    account_id='1',
                    api_key='key',
                    client='https://proxy.example/mcp?signature=a%2fb%20c&tool-mode=individual&flag#fragment',
                    tool_mode='individual',
                ),
                'https://proxy.example/mcp?signature=a%2fb%20c&tool-mode=individual&flag#fragment',
            ),
        ],
    )
    def test_connection_url(self, toolset: StackOneToolset[None], url: str):
        assert http_transport(toolset).url == url

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
            StackOneToolset(account_id='1', api_key='key', tool_mode=tool_mode, client=client)

    def test_non_url_client_needs_no_api_key_or_base_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        script = tmp_path / 'server.py'
        script.write_text('', encoding='utf-8')
        StackOneToolset(account_id='45320', client=str(script), base_url='not-used')

    def test_prebuilt_client_gets_no_stackone_headers(self):
        toolset = StackOneToolset[None](account_id='45320', api_key='key', client=Client('http://proxy.example/mcp'))
        assert http_transport(toolset).headers == {}

    def test_missing_api_key_fails_at_construction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv('STACKONE_API_KEY', raising=False)
        with pytest.raises(UserError, match='STACKONE_API_KEY'):
            StackOneToolset(account_id='45320')

    def test_api_key_from_environment(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv('STACKONE_API_KEY', 'env-key')
        assert http_transport(StackOneToolset[None](account_id='45320')).headers['Authorization'] == basic('env-key')

    def test_rejects_actions_in_search_execute(self):
        with pytest.raises(UserError, match='cannot apply in `search_execute` mode'):
            StackOneToolset(account_id='1', api_key='key', tool_mode='search_execute', actions=['*_list_*'])
