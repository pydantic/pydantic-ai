"""Unit tests for _build_tool_error_message — no VCR or external deps needed."""

from mcp.shared.exceptions import McpError
from mcp.types import ErrorData
from fastmcp.exceptions import ToolError

# Import the helper under test — assumes repo root is on PYTHONPATH or we add it
import sys
sys.path.insert(0, 'pydantic_ai_slim')
from pydantic_ai.mcp import _build_tool_error_message


class TestBuildToolErrorMessage:
    def test_mcp_error_includes_code_in_message(self):
        """Bare McpError → message includes the error code."""
        err = McpError(ErrorData(code=-32603, message='internal error'))
        msg = _build_tool_error_message(err)
        assert 'internal error' in msg
        assert 'code: -32603' in msg

    def test_tool_error_wrapping_mcp_error(self):
        """ToolError whose __cause__ is an McpError → message includes code."""
        mcpe = McpError(ErrorData(code=-32002, message='not found'))
        tool_err = ToolError('tool failed')
        tool_err.__cause__ = mcpe
        msg = _build_tool_error_message(tool_err)
        assert 'not found' in msg
        assert 'code: -32002' in msg

    def test_mcp_error_with_data(self):
        """McpError carrying structured data → includes data in message."""
        err = McpError(ErrorData(code=408, message='timeout', data={'waited': '5s'}))
        msg = _build_tool_error_message(err)
        assert 'timeout' in msg
        assert 'code: 408' in msg
        assert 'waited' in msg

    def test_non_mcp_error_falls_back_to_str(self):
        """Non-MCP exception → plain str(error) fallback."""
        err = ValueError('something went wrong')
        msg = _build_tool_error_message(err)
        assert msg == 'something went wrong'
