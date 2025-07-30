from typing import Any, Protocol

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool
from pydantic_ai.toolsets.function import FunctionToolset


class LangChainTool(Protocol):
    # args are like
    # {'dir_path': {'default': '.', 'description': 'Subdirectory to search in.', 'title': 'Dir Path', 'type': 'string'},
    #  'pattern': {'description': 'Unix shell regex, where * matches everything.', 'title': 'Pattern', 'type': 'string'}}
    @property
    def args(self) -> dict[str, JsonSchemaValue]: ...

    def get_input_jsonschema(self) -> JsonSchemaValue: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, *args: Any, **kwargs: Any) -> str: ...


__all__ = ('tool_from_langchain', 'LangChainToolset')


def tool_from_langchain(langchain_tool: LangChainTool) -> Tool:
    """Creates a Pydantic AI tool proxy from a LangChain tool.

    Args:
        langchain_tool: The LangChain tool to wrap.

    Returns:
        A Pydantic AI tool that corresponds to the LangChain tool.
    """
    function_name = langchain_tool.name
    function_description = langchain_tool.description
    inputs = langchain_tool.args.copy()
    required: list[str] = []
    defaults: dict[str, Any] = {}
    for name, detail in inputs.items():
        if 'default' in detail:
            defaults[name] = detail['default']
        else:
            required.append(name)
    if required:
        required.sort()
    schema: JsonSchemaValue = langchain_tool.get_input_jsonschema()
    if required:
        schema['required'] = required
    if 'additionalProperties' not in schema:
        schema['additionalProperties'] = False

    # restructures the arguments to match langchain tool run
    def proxy(*args: Any, **kwargs: Any) -> str:
        assert not args, 'This should always be called with kwargs'
        kwargs = defaults | kwargs
        return langchain_tool.run(kwargs)

    return Tool.from_schema(
        function=proxy,
        name=function_name,
        description=function_description,
        json_schema=schema,
    )


class LangChainToolset(FunctionToolset):
    """A toolset that wraps LangChain tools."""

    def __init__(self, tools: list[LangChainTool]):
        super().__init__([tool_from_langchain(tool) for tool in tools])
