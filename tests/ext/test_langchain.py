from dataclasses import dataclass
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic.json_schema import JsonSchemaValue

from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset, tool_from_langchain


@dataclass
class SimulatedLangChainTool:
    name: str
    description: str
    args: dict[str, dict[str, str]]
    additional_properties_missing: bool = False

    def run(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,
        start_color: str | None = 'green',
        color: str | None = 'green',
        callbacks: Any = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: Any | None = None,
        config: Any | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(tool_input, dict):
            tool_input = dict(sorted(tool_input.items()))
        return f'I was called with {tool_input}'

    def get_input_jsonschema(self) -> JsonSchemaValue:
        if self.additional_properties_missing:
            return {
                'type': 'object',
                'properties': self.args,
            }
        return {
            'type': 'object',
            'properties': self.args,
            'additionalProperties': False,
        }


langchain_tool = SimulatedLangChainTool(
    name='file_search',
    description='Recursively search for files in a subdirectory that match the regex pattern',
    args={
        'dir_path': {
            'default': '.',
            'description': 'Subdirectory to search in.',
            'title': 'Dir Path',
            'type': 'string',
        },
        'pattern': {
            'description': 'Unix shell regex, where * matches everything.',
            'title': 'Pattern',
            'type': 'string',
        },
    },
)


def test_langchain_tool_conversion():
    pydantic_tool = tool_from_langchain(langchain_tool)

    agent = Agent('test', tools=[pydantic_tool], retries=7)
    result = agent.run_sync('foobar')
    assert result.output == snapshot("{\"file_search\":\"I was called with {'dir_path': '.', 'pattern': 'a'}\"}")


def test_langchain_toolset():
    toolset = LangChainToolset([langchain_tool])
    agent = Agent('test', toolsets=[toolset], retries=7)
    result = agent.run_sync('foobar')
    assert result.output == snapshot("{\"file_search\":\"I was called with {'dir_path': '.', 'pattern': 'a'}\"}")


def test_langchain_tool_no_additional_properties():
    langchain_tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'dir_path': {
                'default': '.',
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
            'pattern': {
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
        },
        additional_properties_missing=True,
    )
    pydantic_tool = tool_from_langchain(langchain_tool)

    agent = Agent('test', tools=[pydantic_tool], retries=7)
    result = agent.run_sync('foobar')
    assert result.output == snapshot("{\"file_search\":\"I was called with {'dir_path': '.', 'pattern': 'a'}\"}")


def test_langchain_tool_conversion_no_defaults():
    langchain_tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'dir_path': {
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
            'pattern': {
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
        },
    )
    pydantic_tool = tool_from_langchain(langchain_tool)

    agent = Agent('test', tools=[pydantic_tool], retries=7)
    result = agent.run_sync('foobar')
    assert result.output == snapshot("{\"file_search\":\"I was called with {'dir_path': 'a', 'pattern': 'a'}\"}")


def test_langchain_tool_conversion_no_required():
    langchain_tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'dir_path': {
                'default': '.',
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
            'pattern': {
                'default': '*',
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
        },
    )
    pydantic_tool = tool_from_langchain(langchain_tool)

    agent = Agent('test', tools=[pydantic_tool], retries=7)
    result = agent.run_sync('foobar')
    assert result.output == snapshot("{\"file_search\":\"I was called with {'dir_path': '.', 'pattern': '*'}\"}")


def test_langchain_tool_defaults():
    langchain_tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'dir_path': {
                'default': '.',
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
            'pattern': {
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
        },
    )
    pydantic_tool = tool_from_langchain(langchain_tool)

    result = pydantic_tool.function(pattern='something')  # type: ignore
    assert result == snapshot("I was called with {'dir_path': '.', 'pattern': 'something'}")


def test_langchain_tool_positional():
    langchain_tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'pattern': {
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
            'dir_path': {
                'default': '.',
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
        },
    )
    pydantic_tool = tool_from_langchain(langchain_tool)

    with pytest.raises(AssertionError, match='This should always be called with kwargs'):
        pydantic_tool.function('something')  # type: ignore


def test_langchain_tool_default_override():
    langchain_tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'dir_path': {
                'default': '.',
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
            'pattern': {
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
        },
    )
    pydantic_tool = tool_from_langchain(langchain_tool)

    result = pydantic_tool.function(pattern='something', dir_path='somewhere')  # type: ignore
    assert result == snapshot("I was called with {'dir_path': 'somewhere', 'pattern': 'something'}")


def test_simulated_tool_string_input():
    tool = SimulatedLangChainTool(
        name='file_search',
        description='Recursively search for files in a subdirectory that match the regex pattern',
        args={
            'dir_path': {
                'default': '.',
                'description': 'Subdirectory to search in.',
                'title': 'Dir Path',
                'type': 'string',
            },
            'pattern': {
                'description': 'Unix shell regex, where * matches everything.',
                'title': 'Pattern',
                'type': 'string',
            },
        },
    )
    result = tool.run('this string argument')
    assert result == snapshot('I was called with this string argument')
