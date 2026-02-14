import asyncio
import logging

import pytest

from pydantic_ai import Agent

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

try:
    from starlette.testclient import TestClient

    from pydantic_ai._acp import FastACP
except ImportError:
    FastACP = None


@pytest.mark.skipif(FastACP is None, reason='ACP dependencies not installed')
async def test_acp_initialize_refined():
    agent = Agent('test')
    app = agent.to_acp()

    client = TestClient(app)
    with client.websocket_connect('/acp') as websocket:
        websocket.send_json(
            {
                'jsonrpc': '2.0',
                'method': 'initialize',
                'id': 1,
                'params': {
                    'client_name': 'test',
                    'client_version': '1.0',
                    'protocol_version': '2024-11-05',
                    '_meta': {'foo': 'bar'},  # Test meta
                },
            }
        )
        response = websocket.receive_json()
        caps = response['result']['capabilities']
        assert caps['promptCapabilities']['text'] is True
        assert caps['promptCapabilities']['image'] is True
        assert caps['terminalCapabilities']['poll'] is True
        assert caps['notificationCapabilities']['terminal_output'] is True
        assert caps['mcpCapabilities']['http'] is False


@pytest.mark.skipif(FastACP is None, reason='ACP dependencies not installed')
async def test_acp_terminal_buffering_refined(tmp_path):
    agent = Agent('test')
    app = agent.to_acp(root_dir=tmp_path)
    client = TestClient(app)

    with client.websocket_connect('/acp') as websocket:
        # Create Session
        websocket.send_json({'jsonrpc': '2.0', 'method': 'session/new', 'id': 1})
        sid = websocket.receive_json()['result']['sessionId']

        # Create Terminal that echoes
        websocket.send_json(
            {
                'jsonrpc': '2.0',
                'method': 'terminal/create',
                'params': {'sessionId': sid, 'command': 'echo', 'args': ['buffered_output']},
                'id': 2,
            }
        )
        term_id = websocket.receive_json()['result']['terminalId']

        # Wait a bit for execution
        await asyncio.sleep(0.1)

        # Poll Output
        websocket.send_json(
            {
                'jsonrpc': '2.0',
                'method': 'terminal/output',
                'params': {'sessionId': sid, 'terminalId': term_id},
                'id': 3,
            }
        )
        resp = websocket.receive_json()
        result = resp['result']
        assert 'buffered_output' in result['output']
        assert result['truncated'] is False
        # exitStatus might be set if echo finished quickly


@pytest.mark.skipif(FastACP is None, reason='ACP dependencies not installed')
async def test_acp_session_cancel_notification_refined():
    agent = Agent('test')
    app = agent.to_acp()
    client = TestClient(app)
    with client.websocket_connect('/acp') as websocket:
        websocket.send_json({'jsonrpc': '2.0', 'method': 'session/new', 'id': 1})
        sid = websocket.receive_json()['result']['sessionId']

        # Send cancel as notification (no id)
        websocket.send_json({'jsonrpc': '2.0', 'method': 'session/cancel', 'params': {'sessionId': sid, '_meta': {}}})

        # Verify checking existence fails
        websocket.send_json({'jsonrpc': '2.0', 'method': 'session/load', 'id': 2, 'params': {'sessionId': sid}})
        resp = websocket.receive_json()
        assert 'error' in resp
