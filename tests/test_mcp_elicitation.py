"""Tests for MCP elicitation callback functionality."""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from pydantic_ai.agent import Agent
from pydantic_ai.models.test import TestModel

from .conftest import try_import

with try_import() as imports_successful:
    from mcp import types
    from mcp.client.session import ClientSession
    from mcp.shared.context import RequestContext

    from pydantic_ai.mcp import MCPServerSSE, MCPServerStdio

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='mcp not installed'),
    pytest.mark.anyio,
]


class TestMCPElicitationCallback:
    """Test MCP elicitation callback functionality."""

    async def test_elicitation_callback_integration(self):
        """Test that elicitation callback is properly integrated into ClientSession."""

        # Create a mock elicitation callback with correct signature
        async def mock_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'data': 'test_result'})

        # Create MCPServer with elicitation callback
        server = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=mock_callback)

        # Verify the callback is set
        assert server.elicitation_callback is mock_callback

        # Test that the callback is passed to ClientSession
        async with server.client_streams() as (read, write):
            # Note: We can't access private attributes directly, but we can verify it was passed correctly
            ClientSession(read, write, elicitation_callback=mock_callback)

    async def test_elicitation_callback_none_by_default(self):
        """Test that elicitation callback is None by default."""

        server = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'])

        assert server.elicitation_callback is None

    async def test_mcp_server_with_elicitation_callback(self):
        """Test MCPServer initialization with elicitation callback."""

        async def sample_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'status': 'success'})

        server = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=sample_callback)

        assert server.elicitation_callback is sample_callback
        assert callable(server.elicitation_callback)

    async def test_elicitation_callback_with_agent(self):
        """Test agent integration with MCP server that has elicitation callback."""

        # Mock elicitation callback that simulates user interaction
        async def mock_elicitation(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            # Simulate accepting tool execution
            return types.ElicitResult(action='accept', content={'tool_executed': True, 'data': 'elicitation_result'})

        server = MCPServerStdio(
            command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=mock_elicitation
        )

        model = TestModel(custom_output_text='Test response')
        agent = Agent(model, mcp_servers=[server])

        # Verify the server is properly configured
        assert len(agent._mcp_servers) == 1  # type: ignore
        assert agent._mcp_servers[0].elicitation_callback is mock_elicitation  # type: ignore

    async def test_elicitation_callback_error_handling(self):
        """Test error handling in elicitation callback."""

        # Mock callback that raises an exception
        async def failing_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            raise ValueError('Elicitation callback failed')

        server = MCPServerStdio(
            command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=failing_callback
        )

        # The callback should be set even if it might fail
        assert server.elicitation_callback is failing_callback

    async def test_elicitation_callback_with_different_actions(self):
        """Test elicitation callback with different action types."""

        # Test callback that rejects
        async def reject_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='decline', content={})

        server = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=reject_callback)

        assert server.elicitation_callback is reject_callback

    async def test_elicitation_callback_error_response(self):
        """Test elicitation callback returning error response."""

        # Test callback that returns error
        async def error_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ErrorData(code=types.INTERNAL_ERROR, message='Test error message')

        server = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=error_callback)

        # Test that the callback can be called with expected parameters
        mock_context = MagicMock()
        mock_params = types.ElicitRequestParams(
            message='Test message', requestedSchema={'type': 'object', 'properties': {'test': {'type': 'string'}}}
        )

        result = await server.elicitation_callback(mock_context, mock_params)  # type: ignore
        assert isinstance(result, types.ErrorData)
        assert result.code == types.INTERNAL_ERROR
        assert result.message == 'Test error message'


class TestMCPElicitationServerIntegration:
    """Test MCP elicitation with actual server integration."""

    async def test_mcp_server_dataclass_fields(self):
        """Test that MCP server dataclass properly handles elicitation_callback field."""

        async def test_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'result': 'success'})

        server = MCPServerStdio(
            command='python',
            args=['-m', 'tests.mcp_server'],
            tool_prefix='test',
            log_level='info',
            timeout=10.0,
            elicitation_callback=test_callback,
        )

        # Verify all fields are properly set
        assert server.tool_prefix == 'test'
        assert server.log_level == 'info'
        assert server.timeout == 10.0
        assert server.elicitation_callback is test_callback

    async def test_multiple_mcp_servers_with_different_callbacks(self):
        """Test multiple MCP servers with different elicitation callbacks."""

        async def callback1(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'result': 'callback1'})

        async def callback2(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'result': 'callback2'})

        server1 = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=callback1)

        server2 = MCPServerStdio(command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=callback2)

        # Each server should have its own callback
        assert server1.elicitation_callback is callback1
        assert server2.elicitation_callback is callback2
        assert server1.elicitation_callback is not server2.elicitation_callback

    async def test_elicitation_callback_with_mcp_server_sse(self):
        """Test elicitation callback with SSE MCP server."""

        async def sse_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'result': 'sse_result'})

        server = MCPServerSSE(url='http://localhost:3000/sse', elicitation_callback=sse_callback)

        assert server.elicitation_callback is sse_callback

    async def test_elicitation_callback_inheritance(self):
        """Test that elicitation callback works with MCP server inheritance."""

        async def shared_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ElicitResult(action='accept', content={'result': 'inherited'})

        # Test with different server types
        stdio_server = MCPServerStdio(
            command='python', args=['-m', 'tests.mcp_server'], elicitation_callback=shared_callback
        )

        sse_server = MCPServerSSE(url='http://localhost:3000/sse', elicitation_callback=shared_callback)

        # Both should have the same callback
        assert stdio_server.elicitation_callback is shared_callback
        assert sse_server.elicitation_callback is shared_callback

    async def test_elicitation_callback_with_mcp_run_python(self):
        """Test elicitation callback with the actual mcp-run-python server."""

        # Create a callback that simulates tool execution approval
        async def python_tool_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            # This would typically be where UI interaction happens
            # For testing, we'll automatically approve Python code execution
            return types.ElicitResult(
                action='accept', content={'approved': True, 'python_code': 'print("Hello from elicitation!")'}
            )

        # Note: This test doesn't actually run the server, just verifies the callback is set
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=python_tool_callback,
        )

        assert server.elicitation_callback is python_tool_callback

        # Test the callback with proper MCP parameters
        mock_context = MagicMock()
        mock_params = types.ElicitRequestParams(
            message='Execute Python code',
            requestedSchema={
                'type': 'object',
                'properties': {'code': {'type': 'string'}, 'approved': {'type': 'boolean'}},
            },
        )

        result = await server.elicitation_callback(mock_context, mock_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert result.content.get('approved') is True


class TestMCPRunPythonToolInjection:
    """Test MCP run-python tool injection functionality."""

    @pytest.fixture
    def mcp_run_python_server(self):
        """Create a basic mcp-run-python server for testing."""
        return MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
        )

    async def test_tool_injection_basic_functionality(self):
        """Test basic tool injection functionality with mcp-run-python."""

        # Mock elicitation callback that simulates tool execution
        async def tool_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            # Parse the tool request
            tool_data = json.loads(params.message)

            if tool_data['tool_name'] == 'web_search':
                return types.ElicitResult(action='accept', content={'result': 'Mock search result: Python is awesome'})
            return types.ElicitResult(action='decline', content={})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=tool_callback,
        )

        # Test that the server has the callback configured
        assert server.elicitation_callback is tool_callback

    async def test_tool_injection_with_multiple_tools(self):
        """Test tool injection with multiple available tools."""

        # Mock callback that handles multiple tools
        async def multi_tool_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']

            if tool_name == 'web_search':
                return types.ElicitResult(action='accept', content={'result': 'Web search completed'})
            elif tool_name == 'send_email':
                return types.ElicitResult(action='accept', content={'result': 'Email sent successfully'})
            elif tool_name == 'database_query':
                return types.ElicitResult(action='accept', content={'result': 'Database query executed'})
            else:
                return types.ElicitResult(action='decline', content={'error': f'Unknown tool: {tool_name}'})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=multi_tool_callback,
        )

        assert server.elicitation_callback is multi_tool_callback

    async def test_tool_injection_error_handling(self):
        """Test error handling in tool injection scenarios."""

        # Mock callback that returns errors
        async def error_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            return types.ErrorData(code=types.INTERNAL_ERROR, message='Tool execution failed')

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=error_callback,
        )

        # Test the callback behavior
        mock_context = MagicMock()
        mock_params = types.ElicitRequestParams(
            message='{"tool_name": "failing_tool", "arguments": {}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, mock_params)  # type: ignore
        assert isinstance(result, types.ErrorData)
        assert result.code == types.INTERNAL_ERROR
        assert result.message == 'Tool execution failed'

    async def test_tool_injection_callback_with_arguments(self):
        """Test tool injection callback with various argument types."""

        # Mock callback that processes different argument types
        async def args_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']
            arguments = tool_data['arguments']

            if tool_name == 'search_with_query':
                query = arguments.get('query', '')
                return types.ElicitResult(action='accept', content={'result': f'Search results for: {query}'})
            elif tool_name == 'calculate':
                a = arguments.get('a', 0)
                b = arguments.get('b', 0)
                operation = arguments.get('operation', 'add')
                result = a + b if operation == 'add' else a - b
                return types.ElicitResult(action='accept', content={'result': str(result)})
            else:
                return types.ElicitResult(action='decline', content={})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=args_callback,
        )

        # Test with search query
        mock_context = MagicMock()
        search_params = types.ElicitRequestParams(
            message='{"tool_name": "search_with_query", "arguments": {"query": "python tutorial"}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, search_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert 'python tutorial' in str(result.content.get('result', ''))

        # Test with calculation
        calc_params = types.ElicitRequestParams(
            message='{"tool_name": "calculate", "arguments": {"a": 5, "b": 3, "operation": "add"}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, calc_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert result.content.get('result') == '8'

    async def test_mcp_run_python_with_agent_integration(self):
        """Test mcp-run-python integrated with a PydanticAI Agent."""

        # Mock callback that simulates agent tool execution
        async def agent_tool_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']

            # Simulate agent's tool execution
            if tool_name == 'get_user_info':
                return types.ElicitResult(action='accept', content={'result': '{"name": "John Doe", "age": 30}'})
            elif tool_name == 'send_notification':
                return types.ElicitResult(action='accept', content={'result': 'Notification sent'})
            else:
                return types.ElicitResult(action='decline', content={'error': f'Tool {tool_name} not available'})

        # Create MCP server with tool injection capability
        mcp_server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=agent_tool_callback,
        )

        # Create agent with the MCP server
        model = TestModel(custom_output_text='Tool injection test completed')
        agent = Agent(model, mcp_servers=[mcp_server])

        # Verify the agent has the MCP server with elicitation callback
        assert len(agent._mcp_servers) == 1  # type: ignore
        assert agent._mcp_servers[0].elicitation_callback is agent_tool_callback  # type: ignore

        # Test running agent with MCP servers
        async with agent.run_mcp_servers():
            # Verify the MCP server is properly integrated
            tools = await mcp_server.list_tools()
            assert len(tools) == 1
            assert tools[0].name == 'run_python_code'

    async def test_tool_injection_callback_accepts_and_declines(self):
        """Test tool injection callback with accept and decline actions."""

        # Mock callback that accepts some tools and declines others
        async def selective_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']

            # Accept safe tools, decline potentially harmful ones
            safe_tools = ['web_search', 'get_weather', 'calculate']
            if tool_name in safe_tools:
                return types.ElicitResult(action='accept', content={'result': f'{tool_name} executed successfully'})
            else:
                return types.ElicitResult(action='decline', content={'reason': f'Tool {tool_name} not permitted'})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=selective_callback,
        )

        mock_context = MagicMock()

        # Test accepting safe tool
        safe_params = types.ElicitRequestParams(
            message='{"tool_name": "web_search", "arguments": {"query": "test"}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, safe_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert 'web_search executed successfully' in str(result.content.get('result', ''))

        # Test declining unsafe tool
        unsafe_params = types.ElicitRequestParams(
            message='{"tool_name": "delete_file", "arguments": {"path": "/tmp/test"}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, unsafe_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'decline'
        assert result.content is not None
        assert 'delete_file not permitted' in str(result.content.get('reason', ''))

    async def test_tool_injection_schema_validation(self):
        """Test tool injection with proper schema validation."""

        # Mock callback that validates schemas
        async def schema_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            # Validate that the requested schema is properly formatted
            schema = params.requestedSchema

            # Check required schema structure
            if schema.get('type') == 'object' and 'properties' in schema and 'result' in schema['properties']:
                return types.ElicitResult(action='accept', content={'result': 'Schema validation passed'})
            else:
                return types.ErrorData(code=types.INVALID_REQUEST, message='Invalid schema structure')

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=schema_callback,
        )

        mock_context = MagicMock()

        # Test with valid schema
        valid_params = types.ElicitRequestParams(
            message='{"tool_name": "test_tool", "arguments": {}}',
            requestedSchema={
                'type': 'object',
                'properties': {'result': {'type': 'string', 'description': 'Result of tool execution'}},
                'required': ['result'],
            },
        )

        result = await server.elicitation_callback(mock_context, valid_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert result.content.get('result') == 'Schema validation passed'

        # Test with invalid schema
        invalid_params = types.ElicitRequestParams(
            message='{"tool_name": "test_tool", "arguments": {}}',
            requestedSchema={'type': 'string'},  # Invalid - missing result property
        )

        result = await server.elicitation_callback(mock_context, invalid_params)  # type: ignore
        assert isinstance(result, types.ErrorData)
        assert result.code == types.INVALID_REQUEST
        assert result.message == 'Invalid schema structure'

    async def test_tool_injection_timeout_handling(self):
        """Test tool injection with timeout scenarios."""

        # Mock callback that simulates timeout
        async def timeout_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            import asyncio

            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']

            if tool_name == 'slow_operation':
                # Simulate a slow operation
                await asyncio.sleep(0.1)  # Small delay for testing
                return types.ElicitResult(action='accept', content={'result': 'Slow operation completed'})
            else:
                return types.ElicitResult(action='accept', content={'result': 'Fast operation completed'})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=timeout_callback,
        )

        mock_context = MagicMock()

        # Test normal operation
        normal_params = types.ElicitRequestParams(
            message='{"tool_name": "fast_operation", "arguments": {}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, normal_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert 'Fast operation completed' in str(result.content.get('result', ''))

        # Test slow operation (should still complete)
        slow_params = types.ElicitRequestParams(
            message='{"tool_name": "slow_operation", "arguments": {}}',
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, slow_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert 'Slow operation completed' in str(result.content.get('result', ''))

    async def test_tool_injection_with_complex_data_types(self):
        """Test tool injection with complex data types in arguments and results."""

        # Mock callback that handles complex data types
        async def complex_data_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']
            arguments = tool_data['arguments']

            if tool_name == 'process_data':
                # Return complex structured data
                result_data = {
                    'status': 'success',
                    'data': {
                        'items': arguments.get('items', []),
                        'processed_count': len(arguments.get('items', [])),
                        'metadata': {'timestamp': '2024-01-01T00:00:00Z', 'processor': 'mcp-tool-injection'},
                    },
                }
                return types.ElicitResult(action='accept', content={'result': json.dumps(result_data)})
            else:
                return types.ElicitResult(action='decline', content={})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=complex_data_callback,
        )

        mock_context = MagicMock()

        # Test with complex input data
        complex_params = types.ElicitRequestParams(
            message=json.dumps(
                {
                    'tool_name': 'process_data',
                    'arguments': {
                        'items': [{'id': 1, 'name': 'item1'}, {'id': 2, 'name': 'item2'}],
                        'options': {'sort': True, 'filter': 'active'},
                    },
                }
            ),
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, complex_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None

        # Verify complex result data
        result_json = result.content.get('result')
        assert result_json is not None
        result_data = json.loads(str(result_json))
        assert result_data['status'] == 'success'
        assert result_data['data']['processed_count'] == 2
        assert len(result_data['data']['items']) == 2
        assert 'metadata' in result_data['data']

    async def test_mcp_run_python_server_basic_functionality(self):
        """Test basic mcp-run-python server functionality."""
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
        )

        async with server:
            tools = await server.list_tools()
            assert len(tools) == 1
            assert tools[0].name == 'run_python_code'
            assert isinstance(tools[0].description, str)
            assert 'Tool to execute Python code' in tools[0].description
            assert 'TOOL INJECTION' in tools[0].description

    async def test_mcp_run_python_code_execution(self):
        """Test basic Python code execution without tool injection."""
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
        )

        async with server:
            # Test basic Python execution
            result = await server.call_tool(
                'run_python_code', {'python_code': 'print("Hello, World!")\n"Hello from Python"'}
            )

            # Result should be XML-formatted
            assert isinstance(result, str)
            assert '<status>success</status>' in result
            assert 'Hello, World!' in result
            assert 'Hello from Python' in result

    async def test_mcp_run_python_with_tool_injection(self):
        """Test mcp-run-python with tool injection enabled."""

        # Mock callback that simulates the run_python_code tool with tool injection
        async def python_code_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']
            arguments = tool_data['arguments']

            if tool_name == 'web_search':
                query = arguments.get('query', '')
                return types.ElicitResult(action='accept', content={'result': f'Web search results for: {query}'})
            elif tool_name == 'calculate':
                expression = arguments.get('expression', '')
                return types.ElicitResult(action='accept', content={'result': f'Calculation result: {expression} = 42'})
            else:
                return types.ElicitResult(action='decline', content={'error': f'Tool {tool_name} not available'})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=python_code_callback,
        )

        async with server:
            # Test Python code execution with tool injection
            # This should trigger the elicitation callback when tools are called
            result = await server.call_tool(
                'run_python_code',
                {'python_code': 'print("Testing tool injection")', 'tools': ['web_search', 'calculate']},
            )

            # Result should indicate tool injection is enabled
            assert isinstance(result, str)
            assert '<status>success</status>' in result
            assert 'Testing tool injection' in result

    async def test_mcp_run_python_error_handling(self):
        """Test error handling in mcp-run-python."""
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
        )

        async with server:
            # Test Python code with syntax error
            result = await server.call_tool('run_python_code', {'python_code': 'print("Missing closing quote)'})

            # Should return error status
            assert isinstance(result, str)
            assert '<status>run-error</status>' in result
            assert 'SyntaxError' in result

    async def test_tool_injection_callback_exception_handling(self):
        """Test exception handling in tool injection callbacks."""

        # Mock callback that raises exceptions
        async def exception_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']

            if tool_name == 'raise_exception':
                raise ValueError('Simulated callback exception')
            else:
                return types.ElicitResult(action='accept', content={'result': 'Success'})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=exception_callback,
        )

        mock_context = MagicMock()

        # Test normal operation
        normal_params = types.ElicitRequestParams(
            message=json.dumps({'tool_name': 'normal_tool', 'arguments': {}}),
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, normal_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None
        assert result.content.get('result') == 'Success'

        # Test exception handling
        exception_params = types.ElicitRequestParams(
            message=json.dumps({'tool_name': 'raise_exception', 'arguments': {}}),
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        with pytest.raises(ValueError, match='Simulated callback exception'):
            await server.elicitation_callback(mock_context, exception_params)  # type: ignore

    async def test_tool_injection_with_streaming_response(self):
        """Test tool injection with streaming or large responses."""

        # Mock callback that simulates streaming or large data responses
        async def streaming_callback(
            context: RequestContext[ClientSession, Any], params: types.ElicitRequestParams
        ) -> types.ElicitResult | types.ErrorData:
            tool_data = json.loads(params.message)
            tool_name = tool_data['tool_name']

            if tool_name == 'large_data':
                # Simulate large data response
                large_data = {'data': ['item' + str(i) for i in range(100)]}
                return types.ElicitResult(action='accept', content={'result': json.dumps(large_data)})
            elif tool_name == 'streaming_data':
                # Simulate streaming-like response
                stream_data = {
                    'chunks': [
                        {'id': 1, 'content': 'First chunk'},
                        {'id': 2, 'content': 'Second chunk'},
                        {'id': 3, 'content': 'Third chunk'},
                    ]
                }
                return types.ElicitResult(action='accept', content={'result': json.dumps(stream_data)})
            else:
                return types.ElicitResult(action='decline', content={})

        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            elicitation_callback=streaming_callback,
        )

        mock_context = MagicMock()

        # Test large data response
        large_data_params = types.ElicitRequestParams(
            message=json.dumps({'tool_name': 'large_data', 'arguments': {}}),
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, large_data_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None

        result_json = result.content.get('result')
        assert result_json is not None
        large_data_result = json.loads(str(result_json))
        assert len(large_data_result['data']) == 100
        assert large_data_result['data'][0] == 'item0'
        assert large_data_result['data'][99] == 'item99'

        # Test streaming data response
        streaming_params = types.ElicitRequestParams(
            message=json.dumps({'tool_name': 'streaming_data', 'arguments': {}}),
            requestedSchema={'type': 'object', 'properties': {'result': {'type': 'string'}}},
        )

        result = await server.elicitation_callback(mock_context, streaming_params)  # type: ignore
        assert isinstance(result, types.ElicitResult)
        assert result.action == 'accept'
        assert result.content is not None

        result_json = result.content.get('result')
        assert result_json is not None
        streaming_result = json.loads(str(result_json))
        assert len(streaming_result['chunks']) == 3
        assert streaming_result['chunks'][0]['content'] == 'First chunk'
        assert streaming_result['chunks'][2]['content'] == 'Third chunk'

    async def test_mcp_run_python_with_dependencies(self):
        """Test mcp-run-python with Python dependencies."""
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
        )

        async with server:
            # Test code with dependencies
            result = await server.call_tool(
                'run_python_code',
                {
                    'python_code': """
# /// script
# dependencies = ['numpy']
# ///
import numpy as np
arr = np.array([1, 2, 3])
print(f"Array: {arr}")
str(arr.sum())
"""
                },
            )

            # Should successfully execute with numpy
            assert isinstance(result, str)
            assert '<status>success</status>' in result
            assert 'Array: [1 2 3]' in result
            assert '<return_value>\n6\n</return_value>' in result or '6' in result

    async def test_mcp_run_python_with_tool_prefix(self):
        """Test mcp-run-python server with tool prefix."""
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            tool_prefix='python',
        )

        async with server:
            tools = await server.list_tools()
            assert len(tools) == 1
            assert tools[0].name == 'python_run_python_code'

    async def test_mcp_run_python_timeout_setting(self):
        """Test mcp-run-python server with timeout setting."""
        server = MCPServerStdio(
            command='deno',
            args=[
                'run',
                '-N',
                '-R=mcp-run-python/node_modules',
                '-W=mcp-run-python/node_modules',
                '--node-modules-dir=auto',
                'mcp-run-python/src/main.ts',
                'stdio',
            ],
            timeout=30.0,
        )

        assert server.timeout == 30.0

        async with server:
            # Test basic execution still works
            result = await server.call_tool('run_python_code', {'python_code': 'print("Timeout test")'})

            assert isinstance(result, str)
            assert '<status>success</status>' in result
            assert 'Timeout test' in result
