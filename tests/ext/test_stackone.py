from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock, patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.tools import Tool

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.ext.stackone import (
        StackOneToolset,
        tool_from_stackone,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='stackone-ai not installed'),
]


@dataclass
class SimulatedStackOneTool:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=lambda: {'type': 'object', 'properties': {}})

    def to_openai_function(self) -> dict[str, Any]:
        return {
            'type': 'function',
            'function': {
                'name': self.name,
                'description': self.description,
                'parameters': self.parameters,
            },
        }

    def execute(self, arguments: dict[str, Any]) -> Any:
        return f'executed {self.name} with {arguments}'


employee_tool = SimulatedStackOneTool(
    name='bamboohr_list_employees',
    description='List all employees from BambooHR',
    parameters={
        'type': 'object',
        'properties': {
            'limit': {'type': 'integer', 'description': 'Max results to return'},
        },
    },
)


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_conversion(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool = tool_from_stackone('bamboohr_list_employees', api_key='test-key', account_ids='test-account')
    assert isinstance(tool, Tool)
    assert tool.name == 'bamboohr_list_employees'
    assert tool.description == 'List all employees from BambooHR'


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_conversion_with_agent(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool = tool_from_stackone('bamboohr_list_employees', api_key='test-key', account_ids='test-account')
    agent = Agent('test', tools=[tool])
    result = agent.run_sync('foobar')
    assert result.output == snapshot('{"bamboohr_list_employees":"executed bamboohr_list_employees with {}"}')


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_execution(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool = tool_from_stackone('bamboohr_list_employees', api_key='test-key', account_ids='test-account')
    result = tool.function(limit=10)  # type: ignore
    assert result == snapshot("executed bamboohr_list_employees with {'limit': 10}")


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_none_description(mock_toolset_cls: Any):
    tool_with_none_desc = SimulatedStackOneTool(name='test_tool', description=None)  # type: ignore
    mock_tools = Mock()
    mock_tools.get_tool.return_value = tool_with_none_desc
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool = tool_from_stackone('test_tool', api_key='test-key', account_ids='test-account')
    assert tool.description == ''


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_schema(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool = tool_from_stackone('bamboohr_list_employees', api_key='test-key', account_ids='test-account')
    assert tool.function_schema.json_schema == employee_tool.to_openai_function()['function']['parameters']


def test_tool_from_stackone_missing_account_id_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('STACKONE_ACCOUNT_ID', raising=False)
    with pytest.raises(ValueError, match='StackOne account ID'):
        tool_from_stackone('workday_list_workers', api_key='test-key')


def test_stackone_toolset_missing_account_id_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('STACKONE_ACCOUNT_ID', raising=False)
    with pytest.raises(ValueError, match='StackOne account ID'):
        StackOneToolset(api_key='test-key')


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_from_stackone(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool = tool_from_stackone('bamboohr_list_employees', api_key='test-key', account_ids='test-account')

    assert tool.name == 'bamboohr_list_employees'
    mock_toolset_cls.assert_called_once_with(api_key='test-key', base_url=None)
    mock_toolset_cls.return_value.set_accounts.assert_called_once_with(['test-account'])
    mock_toolset_cls.return_value.fetch_tools.assert_called_once_with(actions=['bamboohr_list_employees'])


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_from_stackone_with_multiple_account_ids(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool_from_stackone('bamboohr_list_employees', api_key='test-key', account_ids=['acct-1', 'acct-2'])
    mock_toolset_cls.return_value.set_accounts.assert_called_once_with(['acct-1', 'acct-2'])


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_from_stackone_not_found(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = None
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    with pytest.raises(ValueError, match="Tool 'nonexistent' not found"):
        tool_from_stackone('nonexistent', api_key='test-key', account_ids='test-account')


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_tool_from_stackone_with_base_url(mock_toolset_cls: Any):
    mock_tools = Mock()
    mock_tools.get_tool.return_value = employee_tool
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_tools

    tool_from_stackone(
        'bamboohr_list_employees', api_key='k', account_ids='acct', base_url='https://custom.api.stackone.com'
    )
    mock_toolset_cls.assert_called_once_with(api_key='k', base_url='https://custom.api.stackone.com')
    mock_toolset_cls.return_value.set_accounts.assert_called_once_with(['acct'])


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_stackone_toolset_with_tools(mock_toolset_cls: Any):
    tool2 = SimulatedStackOneTool(name='bamboohr_get_employee', description='Get an employee')
    mock_fetched = Mock()
    mock_fetched.__iter__ = Mock(return_value=iter([employee_tool, tool2]))
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_fetched

    toolset = StackOneToolset(
        tools=['bamboohr_list_employees', 'bamboohr_get_employee'],
        api_key='test-key',
        account_ids='test-account',
    )

    mock_toolset_cls.return_value.set_accounts.assert_called_once_with(['test-account'])
    mock_toolset_cls.return_value.fetch_tools.assert_called_once_with(
        actions=['bamboohr_list_employees', 'bamboohr_get_employee']
    )
    agent = Agent('test', toolsets=[toolset])
    result = agent.run_sync('foobar')
    assert 'bamboohr_list_employees' in result.output or 'bamboohr_get_employee' in result.output


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_stackone_toolset_with_multiple_account_ids(mock_toolset_cls: Any):
    mock_fetched = Mock()
    mock_fetched.__iter__ = Mock(return_value=iter([employee_tool]))
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_fetched

    StackOneToolset(tools=['bamboohr_list_employees'], api_key='test-key', account_ids=['acct-1', 'acct-2'])
    mock_toolset_cls.return_value.set_accounts.assert_called_once_with(['acct-1', 'acct-2'])


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_stackone_toolset_with_filter_pattern(mock_toolset_cls: Any):
    mock_fetched = Mock()
    mock_fetched.__iter__ = Mock(return_value=iter([employee_tool]))
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_fetched

    StackOneToolset(filter_pattern='bamboohr_*', api_key='test-key', account_ids='test-account')
    mock_toolset_cls.return_value.fetch_tools.assert_called_once_with(actions=['bamboohr_*'])


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_stackone_toolset_with_list_filter_pattern(mock_toolset_cls: Any):
    mock_fetched = Mock()
    mock_fetched.__iter__ = Mock(return_value=iter([employee_tool]))
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_fetched

    StackOneToolset(filter_pattern=['bamboohr_*', 'workday_*'], api_key='test-key', account_ids='test-account')
    mock_toolset_cls.return_value.fetch_tools.assert_called_once_with(actions=['bamboohr_*', 'workday_*'])


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_stackone_toolset_no_filter(mock_toolset_cls: Any):
    mock_fetched = Mock()
    mock_fetched.__iter__ = Mock(return_value=iter([employee_tool]))
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_fetched

    StackOneToolset(api_key='test-key', account_ids='test-account')
    mock_toolset_cls.return_value.fetch_tools.assert_called_once_with(actions=None)


@patch('pydantic_ai.ext.stackone.StackOneToolSet')
def test_stackone_toolset_with_base_url(mock_toolset_cls: Any):
    mock_fetched = Mock()
    mock_fetched.__iter__ = Mock(return_value=iter([employee_tool]))
    mock_toolset_cls.return_value.fetch_tools.return_value = mock_fetched

    StackOneToolset(
        tools=['bamboohr_list_employees'], api_key='k', account_ids='acct', base_url='https://custom.stackone.com'
    )
    mock_toolset_cls.assert_called_once_with(api_key='k', base_url='https://custom.stackone.com')


def test_stackone_toolset_tools_and_filter_pattern_mutual_exclusion():
    with pytest.raises(ValueError, match="Cannot specify both 'tools' and 'filter_pattern'"):
        StackOneToolset(
            tools=['bamboohr_list_employees'],
            filter_pattern='bamboohr_*',
            api_key='test-key',
            account_ids='test-account',
        )


def test_stackone_toolset_search_config_requires_mode():
    with pytest.raises(ValueError, match="require mode='search_and_execute'"):
        StackOneToolset(
            tools=['bamboohr_list_employees'],
            search_config={'method': 'semantic'},
            api_key='test-key',
            account_ids='test-account',
        )

    with pytest.raises(ValueError, match="require mode='search_and_execute'"):
        StackOneToolset(
            execute_config={'account_ids': ['acct-1']},
            api_key='test-key',
            account_ids='test-account',
        )
