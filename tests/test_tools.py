import json
from dataclasses import dataclass, replace
from typing import Annotated, Any, Callable, Literal, Union

import pydantic_core
import pytest
from _pytest.logging import LogCaptureFixture
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, WithJsonSchema
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticSerializationError, core_schema
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, Tool, ToolOutput, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


def test_tool_no_ctx():
    agent = Agent(TestModel())

    with pytest.raises(UserError) as exc_info:

        @agent.tool  # pyright: ignore[reportArgumentType]
        def invalid_tool(x: int) -> str:  # pragma: no cover
            return 'Hello'

    assert str(exc_info.value) == snapshot(
        'Error generating schema for test_tool_no_ctx.<locals>.invalid_tool:\n'
        '  First parameter of tools that take context must be annotated with RunContext[...]'
    )


def test_tool_plain_with_ctx():
    agent = Agent(TestModel())

    with pytest.raises(UserError) as exc_info:

        @agent.tool_plain
        async def invalid_tool(ctx: RunContext[None]) -> str:  # pragma: no cover
            return 'Hello'

    assert str(exc_info.value) == snapshot(
        'Error generating schema for test_tool_plain_with_ctx.<locals>.invalid_tool:\n'
        '  RunContext annotations can only be used with tools that take context'
    )


def test_tool_ctx_second():
    agent = Agent(TestModel())

    with pytest.raises(UserError) as exc_info:

        @agent.tool  # pyright: ignore[reportArgumentType]
        def invalid_tool(x: int, ctx: RunContext[None]) -> str:  # pragma: no cover
            return 'Hello'

    assert str(exc_info.value) == snapshot(
        'Error generating schema for test_tool_ctx_second.<locals>.invalid_tool:\n'
        '  First parameter of tools that take context must be annotated with RunContext[...]\n'
        '  RunContext annotations can only be used as the first argument'
    )


async def google_style_docstring(foo: int, bar: str) -> str:  # pragma: no cover
    """Do foobar stuff, a lot.

    Args:
        foo: The foo thing.
        bar: The bar thing.
    """
    return f'{foo} {bar}'


async def get_json_schema(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(info.function_tools) == 1:
        r = info.function_tools[0]
        return ModelResponse(parts=[TextPart(pydantic_core.to_json(r).decode())])
    else:
        return ModelResponse(parts=[TextPart(pydantic_core.to_json(info.function_tools).decode())])


@pytest.mark.parametrize('docstring_format', ['google', 'auto'])
def test_docstring_google(docstring_format: Literal['google', 'auto']):
    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(docstring_format=docstring_format)(google_style_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'google_style_docstring',
            'description': 'Do foobar stuff, a lot.',
            'parameters_json_schema': {
                'properties': {
                    'foo': {'description': 'The foo thing.', 'type': 'integer'},
                    'bar': {'description': 'The bar thing.', 'type': 'string'},
                },
                'required': ['foo', 'bar'],
                'type': 'object',
                'additionalProperties': False,
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )
    keys = list(json_schema.keys())
    # name should be the first key
    assert keys[0] == 'name'
    # description should be the second key
    assert keys[1] == 'description'


def sphinx_style_docstring(foo: int, /) -> str:  # pragma: no cover
    """Sphinx style docstring.

    :param foo: The foo thing.
    """
    return str(foo)


@pytest.mark.parametrize('docstring_format', ['sphinx', 'auto'])
def test_docstring_sphinx(docstring_format: Literal['sphinx', 'auto']):
    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(docstring_format=docstring_format)(sphinx_style_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'sphinx_style_docstring',
            'description': 'Sphinx style docstring.',
            'parameters_json_schema': {
                'properties': {'foo': {'description': 'The foo thing.', 'type': 'integer'}},
                'required': ['foo'],
                'type': 'object',
                'additionalProperties': False,
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def numpy_style_docstring(*, foo: int, bar: str) -> str:  # pragma: no cover
    """Numpy style docstring.

    Parameters
    ----------
    foo : int
        The foo thing.
    bar : str
        The bar thing.
    """
    return f'{foo} {bar}'


@pytest.mark.parametrize('docstring_format', ['numpy', 'auto'])
def test_docstring_numpy(docstring_format: Literal['numpy', 'auto']):
    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(docstring_format=docstring_format)(numpy_style_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'numpy_style_docstring',
            'description': 'Numpy style docstring.',
            'parameters_json_schema': {
                'properties': {
                    'foo': {'description': 'The foo thing.', 'type': 'integer'},
                    'bar': {'description': 'The bar thing.', 'type': 'string'},
                },
                'required': ['foo', 'bar'],
                'type': 'object',
                'additionalProperties': False,
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def test_google_style_with_returns():
    agent = Agent(FunctionModel(get_json_schema))

    def my_tool(x: int) -> str:  # pragma: no cover
        """A function that does something.

        Args:
            x: The input value.

        Returns:
            str: The result as a string.
        """
        return str(x)

    agent.tool_plain(my_tool)
    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'my_tool',
            'description': """\
<summary>A function that does something.</summary>
<returns>
<type>str</type>
<description>The result as a string.</description>
</returns>\
""",
            'parameters_json_schema': {
                'additionalProperties': False,
                'properties': {'x': {'description': 'The input value.', 'type': 'integer'}},
                'required': ['x'],
                'type': 'object',
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def test_sphinx_style_with_returns():
    agent = Agent(FunctionModel(get_json_schema))

    def my_tool(x: int) -> str:  # pragma: no cover
        """A sphinx function with returns.

        :param x: The input value.
        :rtype: str
        :return: The result as a string with type.
        """
        return str(x)

    agent.tool_plain(docstring_format='sphinx')(my_tool)
    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'my_tool',
            'description': """\
<summary>A sphinx function with returns.</summary>
<returns>
<type>str</type>
<description>The result as a string with type.</description>
</returns>\
""",
            'parameters_json_schema': {
                'additionalProperties': False,
                'properties': {'x': {'description': 'The input value.', 'type': 'integer'}},
                'required': ['x'],
                'type': 'object',
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def test_numpy_style_with_returns():
    agent = Agent(FunctionModel(get_json_schema))

    def my_tool(x: int) -> str:  # pragma: no cover
        """A numpy function with returns.

        Parameters
        ----------
        x : int
            The input value.

        Returns
        -------
        str
            The result as a string with type.
        """
        return str(x)

    agent.tool_plain(docstring_format='numpy')(my_tool)
    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'my_tool',
            'description': """\
<summary>A numpy function with returns.</summary>
<returns>
<type>str</type>
<description>The result as a string with type.</description>
</returns>\
""",
            'parameters_json_schema': {
                'additionalProperties': False,
                'properties': {'x': {'description': 'The input value.', 'type': 'integer'}},
                'required': ['x'],
                'type': 'object',
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def only_returns_type() -> str:  # pragma: no cover
    """

    Returns:
        str: The result as a string.
    """
    return 'foo'


def test_only_returns_type():
    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(only_returns_type)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'only_returns_type',
            'description': """\
<returns>
<type>str</type>
<description>The result as a string.</description>
</returns>\
""",
            'parameters_json_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def unknown_docstring(**kwargs: int) -> str:  # pragma: no cover
    """Unknown style docstring."""
    return str(kwargs)


def test_docstring_unknown():
    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(unknown_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'unknown_docstring',
            'description': 'Unknown style docstring.',
            'parameters_json_schema': {'properties': {}, 'type': 'object'},
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


# fmt: off
async def google_style_docstring_no_body(
    foo: int, bar: Annotated[str, Field(description='from fields')]
) -> str:  # pragma: no cover
    """
    Args:
        foo: The foo thing.
        bar: The bar thing.
    """

    return f'{foo} {bar}'
# fmt: on


@pytest.mark.parametrize('docstring_format', ['google', 'auto'])
def test_docstring_google_no_body(docstring_format: Literal['google', 'auto']):
    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(docstring_format=docstring_format)(google_style_docstring_no_body)

    result = agent.run_sync('')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'google_style_docstring_no_body',
            'description': '',
            'parameters_json_schema': {
                'properties': {
                    'foo': {'description': 'The foo thing.', 'type': 'integer'},
                    'bar': {'description': 'from fields', 'type': 'string'},
                },
                'required': ['foo', 'bar'],
                'type': 'object',
                'additionalProperties': False,
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


class Foo(BaseModel):
    x: int
    y: str


def test_takes_just_model():
    agent = Agent()

    @agent.tool_plain
    def takes_just_model(model: Foo) -> str:
        return f'{model.x} {model.y}'

    result = agent.run_sync('', model=FunctionModel(get_json_schema))
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'takes_just_model',
            'description': None,
            'parameters_json_schema': {
                'properties': {
                    'x': {'type': 'integer'},
                    'y': {'type': 'string'},
                },
                'required': ['x', 'y'],
                'title': 'Foo',
                'type': 'object',
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )

    result = agent.run_sync('', model=TestModel())
    assert result.output == snapshot('{"takes_just_model":"0 a"}')


def test_takes_model_and_int():
    agent = Agent()

    @agent.tool_plain
    def takes_just_model(model: Foo, z: int) -> str:
        return f'{model.x} {model.y} {z}'

    result = agent.run_sync('', model=FunctionModel(get_json_schema))
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'takes_just_model',
            'description': '',
            'parameters_json_schema': {
                '$defs': {
                    'Foo': {
                        'properties': {
                            'x': {'type': 'integer'},
                            'y': {'type': 'string'},
                        },
                        'required': ['x', 'y'],
                        'title': 'Foo',
                        'type': 'object',
                    }
                },
                'properties': {
                    'model': {'$ref': '#/$defs/Foo'},
                    'z': {'type': 'integer'},
                },
                'required': ['model', 'z'],
                'type': 'object',
                'additionalProperties': False,
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )

    result = agent.run_sync('', model=TestModel())
    assert result.output == snapshot('{"takes_just_model":"0 a 0"}')


# pyright: reportPrivateUsage=false
def test_init_tool_plain():
    call_args: list[int] = []

    def plain_tool(x: int) -> int:
        call_args.append(x)
        return x + 1

    agent = Agent('test', tools=[Tool(plain_tool)], retries=7)
    result = agent.run_sync('foobar')
    assert result.output == snapshot('{"plain_tool":1}')
    assert call_args == snapshot([0])
    assert agent._function_tools['plain_tool'].takes_ctx is False
    assert agent._function_tools['plain_tool'].max_retries == 7

    agent_infer = Agent('test', tools=[plain_tool], retries=7)
    result = agent_infer.run_sync('foobar')
    assert result.output == snapshot('{"plain_tool":1}')
    assert call_args == snapshot([0, 0])
    assert agent_infer._function_tools['plain_tool'].takes_ctx is False
    assert agent_infer._function_tools['plain_tool'].max_retries == 7


def ctx_tool(ctx: RunContext[int], x: int) -> int:
    return x + ctx.deps


# pyright: reportPrivateUsage=false
def test_init_tool_ctx():
    agent = Agent('test', tools=[Tool(ctx_tool, takes_ctx=True, max_retries=3)], deps_type=int, retries=7)
    result = agent.run_sync('foobar', deps=5)
    assert result.output == snapshot('{"ctx_tool":5}')
    assert agent._function_tools['ctx_tool'].takes_ctx is True
    assert agent._function_tools['ctx_tool'].max_retries == 3

    agent_infer = Agent('test', tools=[ctx_tool], deps_type=int)
    result = agent_infer.run_sync('foobar', deps=6)
    assert result.output == snapshot('{"ctx_tool":6}')
    assert agent_infer._function_tools['ctx_tool'].takes_ctx is True


def test_repeat_tool_by_rename():
    """
    1. add tool `bar`
    2. add tool `foo` then rename it to `bar`, causing a conflict with `bar`
    """

    with pytest.raises(UserError, match="Tool name conflicts with existing tool: 'ctx_tool'"):
        Agent('test', tools=[Tool(ctx_tool), ctx_tool], deps_type=int)

    agent = Agent('test')

    async def change_tool_name(ctx: RunContext[None], tool_def: ToolDefinition) -> Union[ToolDefinition, None]:
        tool_def.name = 'bar'
        return tool_def

    @agent.tool_plain
    def bar(x: int, y: str) -> str:  # pragma: no cover
        return f'{x} {y}'

    @agent.tool_plain(prepare=change_tool_name)
    def foo(x: int, y: str) -> str:  # pragma: no cover
        return f'{x} {y}'

    with pytest.raises(UserError, match=r"Renaming tool 'foo' to 'bar' conflicts with existing tool."):
        agent.run_sync('')


def test_repeat_tool():
    """
    1. add tool `foo`, then rename it to `bar`
    2. add tool `bar`, causing a conflict with `bar`
    """

    agent = Agent('test')

    async def change_tool_name(ctx: RunContext[None], tool_def: ToolDefinition) -> Union[ToolDefinition, None]:
        tool_def.name = 'bar'
        return tool_def

    @agent.tool_plain(prepare=change_tool_name)
    def foo(x: int, y: str) -> str:  # pragma: no cover
        return f'{x} {y}'

    @agent.tool_plain
    def bar(x: int, y: str) -> str:  # pragma: no cover
        return f'{x} {y}'

    with pytest.raises(UserError, match=r"Tool name conflicts with existing tool: 'bar'."):
        agent.run_sync('')


def test_tool_return_conflict():
    # this is okay
    Agent('test', tools=[ctx_tool], deps_type=int)
    # this is also okay
    Agent('test', tools=[ctx_tool], deps_type=int, output_type=int)
    # this raises an error
    with pytest.raises(UserError, match="Tool name conflicts with result schema name: 'ctx_tool'"):
        Agent('test', tools=[ctx_tool], deps_type=int, output_type=ToolOutput(int, name='ctx_tool'))


def test_init_ctx_tool_invalid():
    def plain_tool(x: int) -> int:  # pragma: no cover
        return x + 1

    m = r'First parameter of tools that take context must be annotated with RunContext\[\.\.\.\]'
    with pytest.raises(UserError, match=m):
        Tool(plain_tool, takes_ctx=True)


def test_init_plain_tool_invalid():
    with pytest.raises(UserError, match='RunContext annotations can only be used with tools that take context'):
        Tool(ctx_tool, takes_ctx=False)


@pytest.mark.parametrize(
    'args, expected',
    [
        ('', {}),
        ({'x': 42, 'y': 'value'}, {'x': 42, 'y': 'value'}),
        ('{"a": 1, "b": "c"}', {'a': 1, 'b': 'c'}),
    ],
)
def test_tool_call_part_args_as_dict(args: Union[str, dict[str, Any]], expected: dict[str, Any]):
    part = ToolCallPart(tool_name='foo', args=args)
    result = part.args_as_dict()
    assert result == expected


def test_return_pydantic_model():
    agent = Agent('test')

    @agent.tool_plain
    def return_pydantic_model(x: int) -> Foo:
        return Foo(x=x, y='a')

    result = agent.run_sync('')
    assert result.output == snapshot('{"return_pydantic_model":{"x":0,"y":"a"}}')


def test_return_bytes():
    agent = Agent('test')

    @agent.tool_plain
    def return_pydantic_model() -> bytes:
        return '🐈 Hello'.encode()

    result = agent.run_sync('')
    assert result.output == snapshot('{"return_pydantic_model":"🐈 Hello"}')


def test_return_bytes_invalid():
    agent = Agent('test')

    @agent.tool_plain
    def return_pydantic_model() -> bytes:
        return b'\00 \x81'

    with pytest.raises(PydanticSerializationError, match='invalid utf-8 sequence of 1 bytes from index 2'):
        agent.run_sync('')


def test_return_unknown():
    agent = Agent('test')

    class Foobar:
        pass

    @agent.tool_plain
    def return_pydantic_model() -> Foobar:
        return Foobar()

    with pytest.raises(PydanticSerializationError, match='Unable to serialize unknown type:'):
        agent.run_sync('')


def test_dynamic_cls_tool():
    @dataclass
    class MyTool(Tool[int]):
        spam: int

        def __init__(self, spam: int = 0, **kwargs: Any):
            self.spam = spam
            kwargs.update(function=self.tool_function, takes_ctx=False)
            super().__init__(**kwargs)

        def tool_function(self, x: int, y: str) -> str:
            return f'{self.spam} {x} {y}'

        async def prepare_tool_def(self, ctx: RunContext[int]) -> Union[ToolDefinition, None]:
            if ctx.deps != 42:
                return await super().prepare_tool_def(ctx)

    agent = Agent('test', tools=[MyTool(spam=777)], deps_type=int)
    r = agent.run_sync('', deps=1)
    assert r.output == snapshot('{"tool_function":"777 0 a"}')

    r = agent.run_sync('', deps=42)
    assert r.output == snapshot('success (no tool calls)')


def test_dynamic_plain_tool_decorator():
    agent = Agent('test', deps_type=int)

    async def prepare_tool_def(ctx: RunContext[int], tool_def: ToolDefinition) -> Union[ToolDefinition, None]:
        if ctx.deps != 42:
            return tool_def

    @agent.tool_plain(prepare=prepare_tool_def)
    def foobar(x: int, y: str) -> str:
        return f'{x} {y}'

    r = agent.run_sync('', deps=1)
    assert r.output == snapshot('{"foobar":"0 a"}')

    r = agent.run_sync('', deps=42)
    assert r.output == snapshot('success (no tool calls)')


def test_dynamic_tool_decorator():
    agent = Agent('test', deps_type=int)

    async def prepare_tool_def(ctx: RunContext[int], tool_def: ToolDefinition) -> Union[ToolDefinition, None]:
        if ctx.deps != 42:
            return tool_def

    @agent.tool(prepare=prepare_tool_def)
    def foobar(ctx: RunContext[int], x: int, y: str) -> str:
        return f'{ctx.deps} {x} {y}'

    r = agent.run_sync('', deps=1)
    assert r.output == snapshot('{"foobar":"1 0 a"}')

    r = agent.run_sync('', deps=42)
    assert r.output == snapshot('success (no tool calls)')


def test_plain_tool_name():
    agent = Agent(FunctionModel(get_json_schema))

    def my_tool(arg: str) -> str: ...  # pragma: no branch

    agent.tool_plain(name='foo_tool')(my_tool)
    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema['name'] == 'foo_tool'


def test_tool_name():
    agent = Agent(FunctionModel(get_json_schema))

    def my_tool(ctx: RunContext, arg: str) -> str: ...  # pragma: no branch

    agent.tool(name='foo_tool')(my_tool)
    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema['name'] == 'foo_tool'


def test_dynamic_tool_use_messages():
    async def repeat_call_foobar(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.function_tools:
            tool = info.function_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, {'x': 42, 'y': 'a'})])
        else:
            return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(repeat_call_foobar), deps_type=int)

    async def prepare_tool_def(ctx: RunContext[int], tool_def: ToolDefinition) -> Union[ToolDefinition, None]:
        if len(ctx.messages) < 5:
            return tool_def

    @agent.tool(prepare=prepare_tool_def)
    def foobar(ctx: RunContext[int], x: int, y: str) -> str:
        return f'{ctx.deps} {x} {y}'

    r = agent.run_sync('', deps=1)
    assert r.output == snapshot('done')
    message_part_kinds = [(m.kind, [p.part_kind for p in m.parts]) for m in r.all_messages()]
    assert message_part_kinds == snapshot(
        [
            ('request', ['user-prompt']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('response', ['tool-call']),
            ('request', ['tool-return']),
            ('response', ['text']),
        ]
    )


def test_future_run_context(create_module: Callable[[str], Any]):
    mod = create_module("""
from __future__ import annotations

from pydantic_ai import Agent, RunContext

def ctx_tool(ctx: RunContext[int], x: int) -> int:
    return x + ctx.deps

agent = Agent('test', tools=[ctx_tool], deps_type=int)
    """)
    result = mod.agent.run_sync('foobar', deps=5)
    assert result.output == snapshot('{"ctx_tool":5}')


async def tool_without_return_annotation_in_docstring() -> str:  # pragma: no cover
    """A tool that documents what it returns but doesn't have a return annotation in the docstring."""

    return ''


def test_suppress_griffe_logging(caplog: LogCaptureFixture):
    # This would cause griffe to emit a warning log if we didn't suppress the griffe logging.

    agent = Agent(FunctionModel(get_json_schema))
    agent.tool_plain(tool_without_return_annotation_in_docstring)

    result = agent.run_sync('')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'description': "A tool that documents what it returns but doesn't have a return annotation in the docstring.",
            'name': 'tool_without_return_annotation_in_docstring',
            'outer_typed_dict_key': None,
            'parameters_json_schema': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
            'strict': None,
        }
    )

    # Without suppressing griffe logging, we get:
    # assert caplog.messages == snapshot(['<module>:4: No type or annotation for returned value 1'])
    assert caplog.messages == snapshot([])


async def missing_parameter_descriptions_docstring(foo: int, bar: str) -> str:  # pragma: no cover
    """Describes function ops, but missing parameter descriptions."""
    return f'{foo} {bar}'


def test_enforce_parameter_descriptions() -> None:
    agent = Agent(FunctionModel(get_json_schema))

    with pytest.raises(UserError) as exc_info:
        agent.tool_plain(require_parameter_descriptions=True)(missing_parameter_descriptions_docstring)

    error_reason = exc_info.value.args[0]
    error_parts = [
        'Error generating schema for missing_parameter_descriptions_docstring',
        'Missing parameter descriptions for ',
        'foo',
        'bar',
    ]
    assert all(err_part in error_reason for err_part in error_parts)


def test_enforce_parameter_descriptions_noraise() -> None:
    async def complete_parameter_descriptions_docstring(ctx: RunContext, foo: int) -> str:  # pragma: no cover
        """Describes function ops, but missing ctx description and contains non-existent parameter description.

        :param foo: The foo thing.
        :param bar: The bar thing.
        """
        return f'{foo}'

    agent = Agent(FunctionModel(get_json_schema))

    agent.tool(require_parameter_descriptions=True)(complete_parameter_descriptions_docstring)


def test_json_schema_required_parameters(set_event_loop: None):
    agent = Agent(FunctionModel(get_json_schema))

    @agent.tool
    def my_tool(ctx: RunContext[None], a: int, b: int = 1) -> int:
        raise NotImplementedError

    @agent.tool_plain
    def my_tool_plain(*, a: int = 1, b: int) -> int:
        raise NotImplementedError

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        [
            {
                'description': '',
                'name': 'my_tool',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a'],
                    'type': 'object',
                },
                'strict': None,
            },
            {
                'description': '',
                'name': 'my_tool_plain',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['b'],
                    'type': 'object',
                },
                'strict': None,
            },
        ]
    )


def test_call_tool_without_unrequired_parameters(set_event_loop: None):
    async def call_tools_first(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='my_tool', args={'a': 13}),
                    ToolCallPart(tool_name='my_tool', args={'a': 13, 'b': 4}),
                    ToolCallPart(tool_name='my_tool_plain', args={'b': 17}),
                    ToolCallPart(tool_name='my_tool_plain', args={'a': 4, 'b': 17}),
                    ToolCallPart(tool_name='no_args_tool', args=''),
                ]
            )
        else:
            return ModelResponse(parts=[TextPart('finished')])

    agent = Agent(FunctionModel(call_tools_first))

    @agent.tool_plain
    def no_args_tool() -> None:
        return None

    @agent.tool
    def my_tool(ctx: RunContext[None], a: int, b: int = 2) -> int:
        return a + b

    @agent.tool_plain
    def my_tool_plain(*, a: int = 3, b: int) -> int:
        return a * b

    result = agent.run_sync('Hello')
    all_messages = result.all_messages()
    first_response = all_messages[1]
    second_request = all_messages[2]
    assert isinstance(first_response, ModelResponse)
    assert isinstance(second_request, ModelRequest)
    tool_call_args = [p.args for p in first_response.parts if isinstance(p, ToolCallPart)]
    tool_returns = [p.content for p in second_request.parts if isinstance(p, ToolReturnPart)]
    assert tool_call_args == snapshot(
        [
            {'a': 13},
            {'a': 13, 'b': 4},
            {'b': 17},
            {'a': 4, 'b': 17},
            '',
        ]
    )
    assert tool_returns == snapshot([15, 17, 51, 68, None])


def test_schema_generator():
    class MyGenerateJsonSchema(GenerateJsonSchema):
        def typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> JsonSchemaValue:
            # Add useless property titles just to show we can
            s = super().typed_dict_schema(schema)
            for p in s.get('properties', {}):
                s['properties'][p]['title'] = f'{s["properties"][p].get("title")} title'
            return s

    agent = Agent(FunctionModel(get_json_schema))

    def my_tool(x: Annotated[Union[str, None], WithJsonSchema({'type': 'string'})] = None, **kwargs: Any):
        return x  # pragma: no cover

    agent.tool_plain(name='my_tool_1')(my_tool)
    agent.tool_plain(name='my_tool_2', schema_generator=MyGenerateJsonSchema)(my_tool)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        [
            {
                'description': '',
                'name': 'my_tool_1',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'properties': {'x': {'type': 'string'}},
                    'type': 'object',
                },
                'strict': None,
            },
            {
                'description': '',
                'name': 'my_tool_2',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'properties': {'x': {'type': 'string', 'title': 'X title'}},
                    'type': 'object',
                },
                'strict': None,
            },
        ]
    )


def test_tool_parameters_with_attribute_docstrings():
    agent = Agent(FunctionModel(get_json_schema))

    class Data(TypedDict):
        a: int
        """The first parameter"""
        b: int
        """The second parameter"""

    @agent.tool_plain
    def get_score(data: Data) -> int: ...  # pragma: no branch

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        {
            'name': 'get_score',
            'description': None,
            'parameters_json_schema': {
                'additionalProperties': False,
                'properties': {
                    'a': {'description': 'The first parameter', 'type': 'integer'},
                    'b': {'description': 'The second parameter', 'type': 'integer'},
                },
                'required': ['a', 'b'],
                'title': 'Data',
                'type': 'object',
            },
            'outer_typed_dict_key': None,
            'strict': None,
        }
    )


def test_dynamic_tools_agent_wide():
    async def prepare_tool_defs(
        ctx: RunContext[int], tool_defs: list[ToolDefinition]
    ) -> Union[list[ToolDefinition], None]:
        if ctx.deps == 42:
            return []
        elif ctx.deps == 43:
            return None
        elif ctx.deps == 21:
            return [replace(tool_def, strict=True) for tool_def in tool_defs]
        return tool_defs

    agent = Agent('test', deps_type=int, prepare_tools=prepare_tool_defs)

    @agent.tool
    def foobar(ctx: RunContext[int], x: int, y: str) -> str:
        return f'{ctx.deps} {x} {y}'

    result = agent.run_sync('', deps=42)
    assert result.output == snapshot('success (no tool calls)')

    result = agent.run_sync('', deps=43)
    assert result.output == snapshot('success (no tool calls)')

    with agent.override(model=FunctionModel(get_json_schema)):
        result = agent.run_sync('', deps=21)
        json_schema = json.loads(result.output)
        assert agent._function_tools['foobar'].strict is None
        assert json_schema['strict'] is True

    result = agent.run_sync('', deps=1)
    assert result.output == snapshot('{"foobar":"1 0 a"}')
