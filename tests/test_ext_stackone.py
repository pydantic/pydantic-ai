"""Tests for StackOne integration with Pydantic AI."""

import os
from unittest.mock import Mock, patch

import pytest

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset


# Test helper to skip tests if stackone-ai is not available
def requires_stackone():
    """Decorator to skip tests if stackone-ai is not available."""
    try:
        import stackone_ai  # noqa: F401

        return lambda func: func
    except ImportError:
        return pytest.mark.skip(reason='stackone-ai not installed')


class TestStackOneImportError:
    """Test import error handling."""

    def test_import_error_without_stackone(self):
        """Test that ImportError is raised when stackone-ai is not available."""
        # Test that importing the module raises the expected error when stackone_ai is not available
        with patch.dict('sys.modules', {'stackone_ai': None}):
            with pytest.raises(ImportError, match='Please install `stackone-ai`'):
                # Force reimport by using importlib
                import importlib

                import pydantic_ai.ext.stackone

                importlib.reload(pydantic_ai.ext.stackone)


@requires_stackone()
class TestToolFromStackOne:
    """Test the tool_from_stackone function."""

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_creation(self, mock_stackone_toolset_class):
        """Test creating a single tool from StackOne."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the StackOne tool
        mock_tool = Mock()
        mock_tool.call.return_value = 'Employee list result'

        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.get_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create the tool
        tool = tool_from_stackone('hris_list_employees', account_id='test-account', api_key='test-key')

        # Verify tool creation
        assert isinstance(tool, Tool)
        assert tool.name == 'hris_list_employees'
        assert tool.description == 'StackOne tool: hris_list_employees'

        # Verify StackOneToolSet was called with correct parameters
        mock_stackone_toolset_class.assert_called_once_with(api_key='test-key', account_id='test-account')

        # Verify the tool was retrieved correctly
        mock_stackone_toolset.get_tools.assert_called_once_with(['hris_list_employees'])
        mock_tools.get_tool.assert_called_once_with('hris_list_employees')

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_execution(self, mock_stackone_toolset_class):
        """Test executing a StackOne tool."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the StackOne tool
        mock_tool = Mock()
        mock_tool.call.return_value = {'employees': [{'id': 1, 'name': 'John Doe'}]}

        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.get_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create and execute the tool
        tool = tool_from_stackone('hris_list_employees')
        result = tool.function(limit=10)

        # Verify execution
        mock_tool.call.assert_called_once_with(limit=10)
        assert '"employees"' in result  # Should be JSON string

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_execution_error(self, mock_stackone_toolset_class):
        """Test error handling in tool execution."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the StackOne tool to raise an error
        mock_tool = Mock()
        mock_tool.call.side_effect = Exception('StackOne API error')

        mock_tools = Mock()
        mock_tools.get_tool.return_value = mock_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.get_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create and execute the tool
        tool = tool_from_stackone('hris_list_employees')
        result = tool.function()

        # Verify error handling
        assert 'Error executing StackOne tool' in result
        assert 'StackOne API error' in result

    @patch.dict(os.environ, {'STACKONE_API_KEY': 'env-key', 'STACKONE_ACCOUNT_ID': 'env-account'})
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_environment_variables(self, mock_stackone_toolset_class):
        """Test using environment variables for configuration."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create tool without explicit parameters
        tool_from_stackone('hris_list_employees')

        # Verify environment variables were used
        mock_stackone_toolset_class.assert_called_once_with(api_key='env-key', account_id='env-account')


@requires_stackone()
class TestStackOneToolset:
    """Test the StackOneToolset class."""

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_creation(self, mock_stackone_toolset_class):
        """Test creating a StackOneToolset."""
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock the meta tools and discovery
        mock_filter_tool = Mock()
        mock_filter_tool.call.return_value = {
            'tools': [
                {'name': 'hris_list_employees'},
                {'name': 'hris_get_employee'},
            ]
        }

        mock_meta_tools = Mock()
        mock_meta_tools.get_tool.return_value = mock_filter_tool

        mock_tools = Mock()
        mock_tools.meta_tools.return_value = mock_meta_tools
        mock_tools.get_tool.return_value = Mock()  # Mock individual tools

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.get_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create the toolset
        toolset = StackOneToolset(['hris_*'], account_id='test-account', api_key='test-key')

        # Verify it's a FunctionToolset
        assert isinstance(toolset, FunctionToolset)

        # Verify StackOneToolSet was initialized correctly
        mock_stackone_toolset_class.assert_called_once_with(api_key='test-key', account_id='test-account')

        # Verify tools were retrieved with patterns
        mock_stackone_toolset.get_tools.assert_called_once_with(['hris_*'])

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_with_single_pattern(self, mock_stackone_toolset_class):
        """Test creating a StackOneToolset with a single pattern string."""
        from pydantic_ai.ext.stackone import StackOneToolset

        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tools to avoid meta tool discovery
        mock_tools = Mock()
        mock_tools.meta_tools.side_effect = Exception('No meta tools')
        mock_stackone_toolset.get_tools.return_value = mock_tools

        # Create toolset with single pattern
        StackOneToolset('hris_*', account_id='test-account')

        # Verify single pattern was converted to list
        mock_stackone_toolset.get_tools.assert_called_once_with(['hris_*'])

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_no_patterns(self, mock_stackone_toolset_class):
        """Test creating a StackOneToolset with no patterns (all tools)."""
        from pydantic_ai.ext.stackone import StackOneToolset

        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tools to avoid meta tool discovery
        mock_tools = Mock()
        mock_tools.meta_tools.side_effect = Exception('No meta tools')
        mock_stackone_toolset.get_tools.return_value = mock_tools

        # Create toolset without patterns
        StackOneToolset(account_id='test-account')

        # Verify default pattern was used
        mock_stackone_toolset.get_tools.assert_called_once_with(['*'])

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_fallback_tools(self, mock_stackone_toolset_class):
        """Test fallback to common tool names when meta discovery fails."""
        from pydantic_ai.ext.stackone import StackOneToolset

        # Mock a tool that works
        mock_individual_tool = Mock()
        mock_individual_tool.call.return_value = 'test result'

        mock_tools = Mock()
        mock_tools.meta_tools.side_effect = Exception('No meta tools')
        mock_tools.get_tool.return_value = mock_individual_tool

        mock_stackone_toolset = Mock()
        mock_stackone_toolset.get_tools.return_value = mock_tools
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Create toolset
        toolset = StackOneToolset(account_id='test-account')

        # Verify that fallback tools were attempted
        # The toolset should try to create tools for common HRIS operations
        assert len(toolset._tools) > 0  # Some tools should be created

    @patch.dict(os.environ, {'STACKONE_API_KEY': 'env-key', 'STACKONE_ACCOUNT_ID': 'env-account'})
    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_environment_variables(self, mock_stackone_toolset_class):
        """Test using environment variables in StackOneToolset."""
        from pydantic_ai.ext.stackone import StackOneToolset

        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tools to avoid meta tool discovery
        mock_tools = Mock()
        mock_tools.meta_tools.side_effect = Exception('No meta tools')
        mock_stackone_toolset.get_tools.return_value = mock_tools

        # Create toolset without explicit parameters
        StackOneToolset()

        # Verify environment variables were used
        mock_stackone_toolset_class.assert_called_once_with(api_key='env-key', account_id='env-account')

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_toolset_with_base_url(self, mock_stackone_toolset_class):
        """Test creating a StackOneToolset with custom base URL."""
        from pydantic_ai.ext.stackone import StackOneToolset

        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tools to avoid meta tool discovery
        mock_tools = Mock()
        mock_tools.meta_tools.side_effect = Exception('No meta tools')
        mock_stackone_toolset.get_tools.return_value = mock_tools

        # Create toolset with custom base URL
        StackOneToolset(account_id='test-account', base_url='https://custom-api.stackone.co')

        # Verify base URL was passed
        mock_stackone_toolset_class.assert_called_once_with(
            api_key=None, account_id='test-account', base_url='https://custom-api.stackone.co'
        )


@requires_stackone()
class TestStackOneIntegration:
    """Integration tests for StackOne functionality."""

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_tool_json_schema_structure(self, mock_stackone_toolset_class):
        """Test that tools have proper JSON schema structure."""
        from pydantic_ai.ext.stackone import tool_from_stackone

        # Mock the StackOne setup
        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        mock_tools = Mock()
        mock_tools.get_tool.return_value = Mock()
        mock_stackone_toolset.get_tools.return_value = mock_tools

        # Create tool
        tool = tool_from_stackone('hris_list_employees')

        # Verify JSON schema structure
        schema = tool.json_schema
        assert schema['type'] == 'object'
        assert 'properties' in schema
        assert 'additionalProperties' in schema
        assert 'required' in schema

    @patch('pydantic_ai.ext.stackone.StackOneToolSet')
    def test_multiple_toolsets_different_accounts(self, mock_stackone_toolset_class):
        """Test creating multiple toolsets with different accounts."""
        from pydantic_ai.ext.stackone import StackOneToolset

        mock_stackone_toolset = Mock()
        mock_stackone_toolset_class.return_value = mock_stackone_toolset

        # Mock tools to avoid meta tool discovery
        mock_tools = Mock()
        mock_tools.meta_tools.side_effect = Exception('No meta tools')
        mock_stackone_toolset.get_tools.return_value = mock_tools

        # Create toolsets with different accounts
        StackOneToolset('hris_*', account_id='hris-account', api_key='test-key')
        StackOneToolset('ats_*', account_id='ats-account', api_key='test-key')

        # Verify separate StackOne instances were created
        assert mock_stackone_toolset_class.call_count == 2

        # Verify different account IDs were used
        calls = mock_stackone_toolset_class.call_args_list
        assert calls[0][1]['account_id'] == 'hris-account'
        assert calls[1][1]['account_id'] == 'ats-account'
