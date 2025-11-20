import json
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

import pydantic_core
import pytest
from _pytest.logging import LogCaptureFixture
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, TypeAdapter, WithJsonSchema
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import PydanticSerializationError, core_schema
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    ExternalToolset,
    FunctionToolset,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PrefixedToolset,
    RetryPromptPart,
    RunContext,
    TextPart,
    Tool,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserError,
    UserPromptPart,
)
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UnexpectedModelBehavior
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.output import ToolOutput
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsStr


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


def test_builtin_tool_registration():
    """
    Test that built-in functions can't be registered as tools.
    """

    with pytest.raises(
        UserError,
        match='Error generating schema for min:\n  no signature found for builtin <built-in function min>',
    ):
        agent = Agent(TestModel())
        agent.tool_plain(min)

    with pytest.raises(
        UserError,
        match='Error generating schema for max:\n  no signature found for builtin <built-in function max>',
    ):
        agent = Agent(TestModel())
        agent.tool_plain(max)


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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
        }
    )


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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'parameters_json_schema': {'additionalProperties': {'type': 'integer'}, 'properties': {}, 'type': 'object'},
            'outer_typed_dict_key': None,
            'strict': None,
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
            'description': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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
    assert agent._function_toolset.tools['plain_tool'].takes_ctx is False
    assert agent._function_toolset.tools['plain_tool'].max_retries == 7

    agent_infer = Agent('test', tools=[plain_tool], retries=7)
    result = agent_infer.run_sync('foobar')
    assert result.output == snapshot('{"plain_tool":1}')
    assert call_args == snapshot([0, 0])
    assert agent_infer._function_toolset.tools['plain_tool'].takes_ctx is False
    assert agent_infer._function_toolset.tools['plain_tool'].max_retries == 7


def ctx_tool(ctx: RunContext[int], x: int) -> int:
    return x + ctx.deps


# pyright: reportPrivateUsage=false
def test_init_tool_ctx():
    agent = Agent('test', tools=[Tool(ctx_tool, takes_ctx=True, max_retries=3)], deps_type=int, retries=7)
    result = agent.run_sync('foobar', deps=5)
    assert result.output == snapshot('{"ctx_tool":5}')
    assert agent._function_toolset.tools['ctx_tool'].takes_ctx is True
    assert agent._function_toolset.tools['ctx_tool'].max_retries == 3

    agent_infer = Agent('test', tools=[ctx_tool], deps_type=int)
    result = agent_infer.run_sync('foobar', deps=6)
    assert result.output == snapshot('{"ctx_tool":6}')
    assert agent_infer._function_toolset.tools['ctx_tool'].takes_ctx is True


def test_repeat_tool_by_rename():
    """
    1. add tool `bar`
    2. add tool `foo` then rename it to `bar`, causing a conflict with `bar`
    """

    with pytest.raises(UserError, match="Tool name conflicts with existing tool: 'ctx_tool'"):
        Agent('test', tools=[Tool(ctx_tool), ctx_tool], deps_type=int)

    agent = Agent('test')

    async def change_tool_name(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
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

    async def change_tool_name(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
        tool_def.name = 'bar'
        return tool_def

    @agent.tool_plain(prepare=change_tool_name)
    def foo(x: int, y: str) -> str:  # pragma: no cover
        return f'{x} {y}'

    @agent.tool_plain
    def bar(x: int, y: str) -> str:  # pragma: no cover
        return f'{x} {y}'

    with pytest.raises(UserError, match="Tool name conflicts with previously renamed tool: 'bar'."):
        agent.run_sync('')


def test_tool_return_conflict():
    # this is okay
    Agent('test', tools=[ctx_tool], deps_type=int).run_sync('', deps=0)
    # this is also okay
    Agent('test', tools=[ctx_tool], deps_type=int, output_type=int).run_sync('', deps=0)
    # this raises an error
    with pytest.raises(
        UserError,
        match=re.escape(
            "The agent defines a tool whose name conflicts with existing tool from the agent's output tools: 'ctx_tool'. Rename the tool or wrap the toolset in a `PrefixedToolset` to avoid name conflicts."
        ),
    ):
        Agent('test', tools=[ctx_tool], deps_type=int, output_type=ToolOutput(int, name='ctx_tool')).run_sync(
            '', deps=0
        )


def test_tool_name_conflict_hint():
    with pytest.raises(
        UserError,
        match=re.escape(
            "PrefixedToolset(FunctionToolset 'tool') defines a tool whose name conflicts with existing tool from the agent: 'foo_tool'. Change the `prefix` attribute to avoid name conflicts."
        ),
    ):

        def tool(x: int) -> int:
            return x + 1  # pragma: no cover

        def foo_tool(x: str) -> str:
            return x + 'foo'  # pragma: no cover

        function_toolset = FunctionToolset([tool], id='tool')
        prefixed_toolset = PrefixedToolset(function_toolset, 'foo')
        Agent('test', tools=[foo_tool], toolsets=[prefixed_toolset]).run_sync('')


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
def test_tool_call_part_args_as_dict(args: str | dict[str, Any], expected: dict[str, Any]):
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

        async def prepare_tool_def(self, ctx: RunContext[int]) -> ToolDefinition | None:
            if ctx.deps != 42:
                return await super().prepare_tool_def(ctx)

    agent = Agent('test', tools=[MyTool(spam=777)], deps_type=int)
    r = agent.run_sync('', deps=1)
    assert r.output == snapshot('{"tool_function":"777 0 a"}')

    r = agent.run_sync('', deps=42)
    assert r.output == snapshot('success (no tool calls)')


def test_dynamic_plain_tool_decorator():
    agent = Agent('test', deps_type=int)

    async def prepare_tool_def(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition | None:
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

    async def prepare_tool_def(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition | None:
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

    async def prepare_tool_def(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition | None:
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
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


def test_json_schema_required_parameters():
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
                'description': None,
                'name': 'my_tool',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'default': 1, 'type': 'integer'}},
                    'required': ['a'],
                    'type': 'object',
                },
                'strict': None,
                'kind': 'function',
                'sequential': False,
                'metadata': None,
            },
            {
                'description': None,
                'name': 'my_tool_plain',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'additionalProperties': False,
                    'properties': {'a': {'default': 1, 'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['b'],
                    'type': 'object',
                },
                'strict': None,
                'kind': 'function',
                'sequential': False,
                'metadata': None,
            },
        ]
    )


def test_call_tool_without_unrequired_parameters():
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

    def my_tool(x: Annotated[str | None, WithJsonSchema({'type': 'string'})] = None, **kwargs: Any):
        return x  # pragma: no cover

    agent.tool_plain(name='my_tool_1')(my_tool)
    agent.tool_plain(name='my_tool_2', schema_generator=MyGenerateJsonSchema)(my_tool)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.output)
    assert json_schema == snapshot(
        [
            {
                'description': None,
                'name': 'my_tool_1',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'additionalProperties': True,
                    'properties': {'x': {'default': None, 'type': 'string'}},
                    'type': 'object',
                },
                'strict': None,
                'kind': 'function',
                'sequential': False,
                'metadata': None,
            },
            {
                'description': None,
                'name': 'my_tool_2',
                'outer_typed_dict_key': None,
                'parameters_json_schema': {
                    'properties': {'x': {'default': None, 'type': 'string', 'title': 'X title'}},
                    'type': 'object',
                },
                'strict': None,
                'kind': 'function',
                'sequential': False,
                'metadata': None,
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
            'kind': 'function',
            'sequential': False,
            'metadata': None,
        }
    )


def test_dynamic_tools_agent_wide():
    async def prepare_tool_defs(ctx: RunContext[int], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
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
        assert agent._function_toolset.tools['foobar'].strict is None
        assert json_schema['strict'] is True

    result = agent.run_sync('', deps=1)
    assert result.output == snapshot('{"foobar":"1 0 a"}')


def test_function_tool_consistent_with_schema():
    def function(*args: Any, **kwargs: Any) -> str:
        assert len(args) == 0
        assert set(kwargs) == {'one', 'two'}
        return 'I like being called like this'

    json_schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'one': {'description': 'first argument', 'type': 'string'},
            'two': {'description': 'second argument', 'type': 'object'},
        },
        'required': ['one', 'two'],
    }
    pydantic_tool = Tool.from_schema(function, name='foobar', description='does foobar stuff', json_schema=json_schema)

    agent = Agent('test', tools=[pydantic_tool], retries=0)
    result = agent.run_sync('foobar')
    assert result.output == snapshot('{"foobar":"I like being called like this"}')
    assert agent._function_toolset.tools['foobar'].takes_ctx is False
    assert agent._function_toolset.tools['foobar'].max_retries == 0


def test_function_tool_from_schema_with_ctx():
    def function(ctx: RunContext[str], *args: Any, **kwargs: Any) -> str:
        assert len(args) == 0
        assert set(kwargs) == {'one', 'two'}
        return ctx.deps + 'I like being called like this'

    json_schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'one': {'description': 'first argument', 'type': 'string'},
            'two': {'description': 'second argument', 'type': 'object'},
        },
        'required': ['one', 'two'],
    }
    pydantic_tool = Tool[str].from_schema(
        function, name='foobar', description='does foobar stuff', json_schema=json_schema, takes_ctx=True
    )

    assert pydantic_tool.takes_ctx is True
    assert pydantic_tool.function_schema.takes_ctx is True

    agent = Agent('test', tools=[pydantic_tool], retries=0, deps_type=str)
    result = agent.run_sync('foobar', deps='Hello, ')
    assert result.output == snapshot('{"foobar":"Hello, I like being called like this"}')
    assert agent._function_toolset.tools['foobar'].takes_ctx is True
    assert agent._function_toolset.tools['foobar'].max_retries == 0


def test_function_tool_inconsistent_with_schema():
    def function(three: str, four: int) -> str:
        return 'Coverage made me call this'

    json_schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'one': {'description': 'first argument', 'type': 'string'},
            'two': {'description': 'second argument', 'type': 'object'},
        },
        'required': ['one', 'two'],
    }
    pydantic_tool = Tool.from_schema(function, name='foobar', description='does foobar stuff', json_schema=json_schema)

    agent = Agent('test', tools=[pydantic_tool], retries=0)
    with pytest.raises(TypeError, match=".* got an unexpected keyword argument 'one'"):
        agent.run_sync('foobar')

    result = function('three', 4)
    assert result == 'Coverage made me call this'


def test_async_function_tool_consistent_with_schema():
    async def function(*args: Any, **kwargs: Any) -> str:
        assert len(args) == 0
        assert set(kwargs) == {'one', 'two'}
        return 'I like being called like this'

    json_schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'one': {'description': 'first argument', 'type': 'string'},
            'two': {'description': 'second argument', 'type': 'object'},
        },
        'required': ['one', 'two'],
    }
    pydantic_tool = Tool.from_schema(function, name='foobar', description='does foobar stuff', json_schema=json_schema)

    agent = Agent('test', tools=[pydantic_tool], retries=0)
    result = agent.run_sync('foobar')
    assert result.output == snapshot('{"foobar":"I like being called like this"}')
    assert agent._function_toolset.tools['foobar'].takes_ctx is False
    assert agent._function_toolset.tools['foobar'].max_retries == 0


def test_tool_retries():
    prepare_tools_retries: list[int] = []
    prepare_retries: list[int] = []
    prepare_max_retries: list[int] = []
    prepare_last_attempt: list[bool] = []
    call_retries: list[int] = []
    call_max_retries: list[int] = []
    call_last_attempt: list[bool] = []

    async def prepare_tool_defs(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition] | None:
        nonlocal prepare_tools_retries
        retry = ctx.retries.get('infinite_retry_tool', 0)
        prepare_tools_retries.append(retry)
        return tool_defs

    agent = Agent(TestModel(), retries=3, prepare_tools=prepare_tool_defs)

    async def prepare_tool_def(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
        nonlocal prepare_retries
        prepare_retries.append(ctx.retry)
        prepare_max_retries.append(ctx.max_retries)
        prepare_last_attempt.append(ctx.last_attempt)
        return tool_def

    @agent.tool(retries=5, prepare=prepare_tool_def)
    def infinite_retry_tool(ctx: RunContext[None]) -> int:
        nonlocal call_retries
        call_retries.append(ctx.retry)
        call_max_retries.append(ctx.max_retries)
        call_last_attempt.append(ctx.last_attempt)
        raise ModelRetry('Please try again.')

    with pytest.raises(UnexpectedModelBehavior, match="Tool 'infinite_retry_tool' exceeded max retries count of 5"):
        agent.run_sync('Begin infinite retry loop!')

    assert prepare_tools_retries == snapshot([0, 1, 2, 3, 4, 5])

    assert prepare_retries == snapshot([0, 1, 2, 3, 4, 5])
    assert prepare_max_retries == snapshot([5, 5, 5, 5, 5, 5])
    assert prepare_last_attempt == snapshot([False, False, False, False, False, True])

    assert call_retries == snapshot([0, 1, 2, 3, 4, 5])
    assert call_max_retries == snapshot([5, 5, 5, 5, 5, 5])
    assert call_last_attempt == snapshot([False, False, False, False, False, True])


def test_tool_raises_call_deferred():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred

    result = agent.run_sync('Hello')
    assert result.output == snapshot(
        DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])
    )


def test_tool_raises_approval_required():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('my_tool', {'x': 1}, tool_call_id='my_tool'),
                ]
            )
        else:
            return ModelResponse(
                parts=[
                    TextPart('Done!'),
                ]
            )

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 42

    result = agent.run_sync('Hello')
    messages = result.all_messages()
    assert result.output == snapshot(
        DeferredToolRequests(approvals=[ToolCallPart(tool_name='my_tool', args={'x': 1}, tool_call_id='my_tool')])
    )

    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={'my_tool': ToolApproved(override_args={'x': 2})}),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={'x': 1}, tool_call_id='my_tool')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content=84,
                        tool_call_id='my_tool',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=52, output_tokens=5),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot('Done!')


def test_call_deferred_with_metadata():
    """Test that CallDeferred exception can carry metadata."""
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred(metadata={'task_id': 'task-123', 'estimated_cost': 25.50})

    result = agent.run_sync('Hello')
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
            metadata={'pyd_ai_tool_call_id__my_tool': {'task_id': 'task-123', 'estimated_cost': 25.5}},
        )
    )


def test_approval_required_with_metadata():
    """Test that ApprovalRequired exception can carry metadata."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('my_tool', {'x': 1}, tool_call_id='my_tool'),
                ]
            )
        else:
            return ModelResponse(
                parts=[
                    TextPart('Done!'),
                ]
            )

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired(
                metadata={
                    'reason': 'High compute cost',
                    'estimated_time': '5 minutes',
                    'cost_usd': 100.0,
                }
            )
        return x * 42

    result = agent.run_sync('Hello')
    assert result.output == snapshot(
        DeferredToolRequests(
            approvals=[ToolCallPart(tool_name='my_tool', args={'x': 1}, tool_call_id=IsStr())],
            metadata={'my_tool': {'reason': 'High compute cost', 'estimated_time': '5 minutes', 'cost_usd': 100.0}},
        )
    )

    # Continue with approval
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={'my_tool': ToolApproved()}),
    )
    assert result.output == 'Done!'


def test_call_deferred_without_metadata():
    """Test backward compatibility: CallDeferred without metadata still works."""
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred  # No metadata

    result = agent.run_sync('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.calls) == 1

    tool_call_id = result.output.calls[0].tool_call_id
    # Should have an empty metadata dict for this tool
    assert result.output.metadata.get(tool_call_id, {}) == {}


def test_approval_required_without_metadata():
    """Test backward compatibility: ApprovalRequired without metadata still works."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('my_tool', {'x': 1}, tool_call_id='my_tool'),
                ]
            )
        else:
            return ModelResponse(
                parts=[
                    TextPart('Done!'),
                ]
            )

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired  # No metadata
        return x * 42

    result = agent.run_sync('Hello')
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1

    # Should have an empty metadata dict for this tool
    assert result.output.metadata.get('my_tool', {}) == {}

    # Continue with approval
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={'my_tool': ToolApproved()}),
    )
    assert result.output == 'Done!'


def test_mixed_deferred_tools_with_metadata():
    """Test multiple deferred tools with different metadata."""

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('tool_a', {'x': 1}, tool_call_id='call_a'),
                    ToolCallPart('tool_b', {'y': 2}, tool_call_id='call_b'),
                    ToolCallPart('tool_c', {'z': 3}, tool_call_id='call_c'),
                ]
            )
        else:
            return ModelResponse(parts=[TextPart('Done!')])

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def tool_a(ctx: RunContext[None], x: int) -> int:
        raise CallDeferred(metadata={'type': 'external', 'priority': 'high'})

    @agent.tool
    def tool_b(ctx: RunContext[None], y: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired(metadata={'reason': 'Needs approval', 'level': 'manager'})
        return y * 10

    @agent.tool
    def tool_c(ctx: RunContext[None], z: int) -> int:
        raise CallDeferred  # No metadata

    result = agent.run_sync('Hello')
    assert isinstance(result.output, DeferredToolRequests)

    # Check that we have the right tools deferred
    assert len(result.output.calls) == 2  # tool_a and tool_c
    assert len(result.output.approvals) == 1  # tool_b

    # Check metadata
    assert result.output.metadata['call_a'] == {'type': 'external', 'priority': 'high'}
    assert result.output.metadata['call_b'] == {'reason': 'Needs approval', 'level': 'manager'}
    assert result.output.metadata.get('call_c', {}) == {}

    # Continue with results for all three tools
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            calls={'call_a': 10, 'call_c': 30},
            approvals={'call_b': ToolApproved()},
        ),
    )
    assert result.output == 'Done!'


def test_deferred_tool_with_output_type():
    class MyModel(BaseModel):
        foo: str

    deferred_toolset = ExternalToolset(
        [
            ToolDefinition(
                name='my_tool',
                description='',
                parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']},
            ),
        ]
    )
    agent = Agent(TestModel(call_tools=[]), output_type=[MyModel, DeferredToolRequests], toolsets=[deferred_toolset])

    result = agent.run_sync('Hello')
    assert result.output == snapshot(MyModel(foo='a'))


def test_deferred_tool_with_tool_output_type():
    class MyModel(BaseModel):
        foo: str

    deferred_toolset = ExternalToolset(
        [
            ToolDefinition(
                name='my_tool',
                description='',
                parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']},
            ),
        ]
    )
    agent = Agent(
        TestModel(call_tools=[]),
        output_type=[[ToolOutput(MyModel), ToolOutput(MyModel)], DeferredToolRequests],
        toolsets=[deferred_toolset],
    )

    result = agent.run_sync('Hello')
    assert result.output == snapshot(MyModel(foo='a'))


async def test_deferred_tool_without_output_type():
    deferred_toolset = ExternalToolset(
        [
            ToolDefinition(
                name='my_tool',
                description='',
                parameters_json_schema={'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']},
            ),
        ]
    )
    agent = Agent(TestModel(), toolsets=[deferred_toolset])

    msg = 'A deferred tool call was present, but `DeferredToolRequests` is not among output types. To resolve this, add `DeferredToolRequests` to the list of output types for this agent.'

    with pytest.raises(UserError, match=msg):
        await agent.run('Hello')

    with pytest.raises(UserError, match=msg):
        async with agent.run_stream('Hello') as result:
            await result.get_output()


def test_output_type_deferred_tool_requests_by_itself():
    with pytest.raises(UserError, match='At least one output type must be provided other than `DeferredToolRequests`.'):
        Agent(TestModel(), output_type=DeferredToolRequests)


def test_output_type_empty():
    with pytest.raises(UserError, match='At least one output type must be provided.'):
        Agent(TestModel(), output_type=[])


def test_parallel_tool_return_with_deferred():
    final_received_messages: list[ModelMessage] | None = None

    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('get_price', {'fruit': 'apple'}, tool_call_id='get_price_apple'),
                    ToolCallPart('get_price', {'fruit': 'banana'}, tool_call_id='get_price_banana'),
                    ToolCallPart('get_price', {'fruit': 'pear'}, tool_call_id='get_price_pear'),
                    ToolCallPart('get_price', {'fruit': 'grape'}, tool_call_id='get_price_grape'),
                    ToolCallPart('buy', {'fruit': 'apple'}, tool_call_id='buy_apple'),
                    ToolCallPart('buy', {'fruit': 'banana'}, tool_call_id='buy_banana'),
                    ToolCallPart('buy', {'fruit': 'pear'}, tool_call_id='buy_pear'),
                ]
            )
        else:
            nonlocal final_received_messages
            final_received_messages = messages
            return ModelResponse(
                parts=[
                    TextPart('Done!'),
                ]
            )

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def get_price(fruit: str) -> ToolReturn:
        if fruit in ['apple', 'pear']:
            return ToolReturn(
                return_value=10.0,
                content=f'The price of {fruit} is 10.0.',
                metadata={'fruit': fruit, 'price': 10.0},
            )
        else:
            raise ModelRetry(f'Unknown fruit: {fruit}')

    @agent.tool_plain
    def buy(fruit: str):
        raise CallDeferred

    result = agent.run_sync('What do an apple, a banana, a pear and a grape cost? Also buy me a pear.')

    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What do an apple, a banana, a pear and a grape cost? Also buy me a pear.',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_price', args={'fruit': 'apple'}, tool_call_id='get_price_apple'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'banana'}, tool_call_id='get_price_banana'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'pear'}, tool_call_id='get_price_pear'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'grape'}, tool_call_id='get_price_grape'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'apple'}, tool_call_id='buy_apple'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'banana'}, tool_call_id='buy_banana'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'pear'}, tool_call_id='buy_pear'),
                ],
                usage=RequestUsage(input_tokens=68, output_tokens=35),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_price',
                        content=10.0,
                        tool_call_id='get_price_apple',
                        metadata={'fruit': 'apple', 'price': 10.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Unknown fruit: banana',
                        tool_name='get_price',
                        tool_call_id='get_price_banana',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_price',
                        content=10.0,
                        tool_call_id='get_price_pear',
                        metadata={'fruit': 'pear', 'price': 10.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Unknown fruit: grape',
                        tool_name='get_price',
                        tool_call_id='get_price_grape',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='The price of apple is 10.0.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='The price of pear is 10.0.',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot(
        DeferredToolRequests(
            calls=[
                ToolCallPart(tool_name='buy', args={'fruit': 'apple'}, tool_call_id='buy_apple'),
                ToolCallPart(tool_name='buy', args={'fruit': 'banana'}, tool_call_id='buy_banana'),
                ToolCallPart(tool_name='buy', args={'fruit': 'pear'}, tool_call_id='buy_pear'),
            ]
        )
    )

    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            calls={
                'buy_apple': ModelRetry('Apples are not available'),
                'buy_banana': ToolReturn(
                    return_value=True,
                    content='I bought a banana',
                    metadata={'fruit': 'banana', 'price': 100.0},
                ),
                'buy_pear': RetryPromptPart(
                    content='The purchase of pears was denied.',
                ),
            },
        ),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What do an apple, a banana, a pear and a grape cost? Also buy me a pear.',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_price', args={'fruit': 'apple'}, tool_call_id='get_price_apple'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'banana'}, tool_call_id='get_price_banana'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'pear'}, tool_call_id='get_price_pear'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'grape'}, tool_call_id='get_price_grape'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'apple'}, tool_call_id='buy_apple'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'banana'}, tool_call_id='buy_banana'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'pear'}, tool_call_id='buy_pear'),
                ],
                usage=RequestUsage(input_tokens=68, output_tokens=35),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_price',
                        content=10.0,
                        tool_call_id='get_price_apple',
                        metadata={'fruit': 'apple', 'price': 10.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Unknown fruit: banana',
                        tool_name='get_price',
                        tool_call_id='get_price_banana',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_price',
                        content=10.0,
                        tool_call_id='get_price_pear',
                        metadata={'fruit': 'pear', 'price': 10.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Unknown fruit: grape',
                        tool_name='get_price',
                        tool_call_id='get_price_grape',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='The price of apple is 10.0.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='The price of pear is 10.0.',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Apples are not available',
                        tool_name='buy',
                        tool_call_id='buy_apple',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='buy',
                        content=True,
                        tool_call_id='buy_banana',
                        metadata={'fruit': 'banana', 'price': 100.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='The purchase of pears was denied.',
                        tool_name='buy',
                        tool_call_id='buy_pear',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='I bought a banana',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=137, output_tokens=36),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Apples are not available',
                        tool_name='buy',
                        tool_call_id='buy_apple',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='buy',
                        content=True,
                        tool_call_id='buy_banana',
                        metadata={'fruit': 'banana', 'price': 100.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='The purchase of pears was denied.',
                        tool_name='buy',
                        tool_call_id='buy_pear',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='I bought a banana',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=137, output_tokens=36),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )

    assert final_received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What do an apple, a banana, a pear and a grape cost? Also buy me a pear.',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_price', args={'fruit': 'apple'}, tool_call_id='get_price_apple'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'banana'}, tool_call_id='get_price_banana'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'pear'}, tool_call_id='get_price_pear'),
                    ToolCallPart(tool_name='get_price', args={'fruit': 'grape'}, tool_call_id='get_price_grape'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'apple'}, tool_call_id='buy_apple'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'banana'}, tool_call_id='buy_banana'),
                    ToolCallPart(tool_name='buy', args={'fruit': 'pear'}, tool_call_id='buy_pear'),
                ],
                usage=RequestUsage(input_tokens=68, output_tokens=35),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_price',
                        content=10.0,
                        tool_call_id='get_price_apple',
                        metadata={'fruit': 'apple', 'price': 10.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Unknown fruit: banana',
                        tool_name='get_price',
                        tool_call_id='get_price_banana',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='get_price',
                        content=10.0,
                        tool_call_id='get_price_pear',
                        metadata={'fruit': 'pear', 'price': 10.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Unknown fruit: grape',
                        tool_name='get_price',
                        tool_call_id='get_price_grape',
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='Apples are not available',
                        tool_name='buy',
                        tool_call_id='buy_apple',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='buy',
                        content=True,
                        tool_call_id='buy_banana',
                        metadata={'fruit': 'banana', 'price': 100.0},
                        timestamp=IsDatetime(),
                    ),
                    RetryPromptPart(
                        content='The purchase of pears was denied.',
                        tool_name='buy',
                        tool_call_id='buy_pear',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='The price of apple is 10.0.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='The price of pear is 10.0.',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='I bought a banana',
                        timestamp=IsDatetime(),
                    ),
                ]
            ),
        ]
    )


def test_deferred_tool_call_approved_fails():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('foo', {'x': 0}, tool_call_id='foo'),
            ]
        )

    agent = Agent(FunctionModel(llm), output_type=[str, DeferredToolRequests])

    async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
        return replace(tool_def, kind='external')

    @agent.tool_plain(prepare=defer)
    def foo(x: int) -> int:
        return x + 1  # pragma: no cover

    result = agent.run_sync('foo')
    assert result.output == snapshot(
        DeferredToolRequests(calls=[ToolCallPart(tool_name='foo', args={'x': 0}, tool_call_id='foo')])
    )

    with pytest.raises(RuntimeError, match='External tools cannot be called'):
        agent.run_sync(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(
                approvals={
                    'foo': True,
                },
            ),
        )


async def test_approval_required_toolset():
    def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('foo', {'x': 1}, tool_call_id='foo1'),
                    ToolCallPart('foo', {'x': 2}, tool_call_id='foo2'),
                    ToolCallPart('bar', {'x': 3}, tool_call_id='bar'),
                ]
            )
        else:
            return ModelResponse(
                parts=[
                    TextPart('Done!'),
                ]
            )

    toolset = FunctionToolset[None]()

    @toolset.tool
    def foo(x: int) -> int:
        return x * 2

    @toolset.tool
    def bar(x: int) -> int:
        return x * 3

    toolset = toolset.approval_required(lambda ctx, tool_def, tool_args: tool_def.name == 'foo')

    agent = Agent(FunctionModel(llm), toolsets=[toolset], output_type=[str, DeferredToolRequests])

    result = await agent.run('foo')
    messages = result.all_messages()
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='foo',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='foo', args={'x': 1}, tool_call_id='foo1'),
                    ToolCallPart(tool_name='foo', args={'x': 2}, tool_call_id='foo2'),
                    ToolCallPart(tool_name='bar', args={'x': 3}, tool_call_id='bar'),
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=12),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='bar',
                        content=9,
                        tool_call_id='bar',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot(
        DeferredToolRequests(
            approvals=[
                ToolCallPart(tool_name='foo', args={'x': 1}, tool_call_id='foo1'),
                ToolCallPart(tool_name='foo', args={'x': 2}, tool_call_id='foo2'),
            ]
        )
    )

    result = await agent.run(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(
            approvals={
                'foo1': True,
                'foo2': False,
            },
        ),
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='foo',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='foo', args={'x': 1}, tool_call_id='foo1'),
                    ToolCallPart(tool_name='foo', args={'x': 2}, tool_call_id='foo2'),
                    ToolCallPart(tool_name='bar', args={'x': 3}, tool_call_id='bar'),
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=12),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='bar',
                        content=9,
                        tool_call_id='bar',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='foo',
                        content=2,
                        tool_call_id='foo1',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='foo',
                        content='The tool call was denied.',
                        tool_call_id='foo2',
                        timestamp=IsDatetime(),
                    ),
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Done!')],
                usage=RequestUsage(input_tokens=59, output_tokens=13),
                model_name='function:llm:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.output == snapshot('Done!')


def test_deferred_tool_results_serializable():
    results = DeferredToolResults(
        calls={
            'tool-return': ToolReturn(
                return_value=1,
                content='The tool call was approved.',
                metadata={'foo': 'bar'},
            ),
            'model-retry': ModelRetry('The tool call was denied.'),
            'retry-prompt-part': RetryPromptPart(
                content='The tool call was denied.',
                tool_name='foo',
                tool_call_id='foo',
            ),
            'any': {'foo': 'bar'},
        },
        approvals={
            'true': True,
            'false': False,
            'tool-approved': ToolApproved(override_args={'foo': 'bar'}),
            'tool-denied': ToolDenied('The tool call was denied.'),
        },
    )
    results_ta = TypeAdapter(DeferredToolResults)
    serialized = results_ta.dump_python(results)
    assert serialized == snapshot(
        {
            'calls': {
                'tool-return': {
                    'return_value': 1,
                    'content': 'The tool call was approved.',
                    'metadata': {'foo': 'bar'},
                    'kind': 'tool-return',
                },
                'model-retry': {'message': 'The tool call was denied.', 'kind': 'model-retry'},
                'retry-prompt-part': {
                    'content': 'The tool call was denied.',
                    'tool_name': 'foo',
                    'tool_call_id': 'foo',
                    'timestamp': IsDatetime(),
                    'part_kind': 'retry-prompt',
                },
                'any': {'foo': 'bar'},
            },
            'approvals': {
                'true': True,
                'false': False,
                'tool-approved': {'override_args': {'foo': 'bar'}, 'kind': 'tool-approved'},
                'tool-denied': {'message': 'The tool call was denied.', 'kind': 'tool-denied'},
            },
        }
    )
    deserialized = results_ta.validate_python(serialized)
    assert deserialized == results


def test_tool_metadata():
    """Test that metadata is properly set on tools."""
    metadata = {'category': 'test', 'version': '1.0'}

    def simple_tool(ctx: RunContext[None], x: int) -> int:
        return x * 2  # pragma: no cover

    tool = Tool(simple_tool, metadata=metadata)
    assert tool.metadata == metadata
    assert tool.tool_def.metadata == metadata

    # Test with agent decorator
    agent = Agent('test')

    @agent.tool(metadata={'source': 'agent'})
    def agent_tool(ctx: RunContext[None], y: int) -> int:
        return y + 1  # pragma: no cover

    agent_tool_def = agent._function_toolset.tools['agent_tool']
    assert agent_tool_def.metadata == {'source': 'agent'}

    # Test with agent.tool_plain decorator
    @agent.tool_plain(metadata={'type': 'plain'})
    def plain_tool(z: int) -> int:
        return z * 3  # pragma: no cover

    plain_tool_def = agent._function_toolset.tools['plain_tool']
    assert plain_tool_def.metadata == {'type': 'plain'}

    # Test with FunctionToolset.tool decorator
    toolset = FunctionToolset(metadata={'foo': 'bar'})

    @toolset.tool
    def toolset_plain_tool(a: str) -> str:
        return a.upper()  # pragma: no cover

    toolset_plain_tool_def = toolset.tools['toolset_plain_tool']
    assert toolset_plain_tool_def.metadata == {'foo': 'bar'}

    @toolset.tool(metadata={'toolset': 'function'})
    def toolset_tool(ctx: RunContext[None], a: str) -> str:
        return a.upper()  # pragma: no cover

    toolset_tool_def = toolset.tools['toolset_tool']
    assert toolset_tool_def.metadata == {'foo': 'bar', 'toolset': 'function'}

    # Test with FunctionToolset.add_function
    def standalone_func(ctx: RunContext[None], b: float) -> float:
        return b / 2  # pragma: no cover

    toolset.add_function(standalone_func, metadata={'method': 'add_function'})
    standalone_tool_def = toolset.tools['standalone_func']
    assert standalone_tool_def.metadata == {'foo': 'bar', 'method': 'add_function'}


def test_retry_tool_until_last_attempt():
    model = TestModel()
    agent = Agent(model, retries=2)

    @agent.tool
    def always_fail(ctx: RunContext[None]) -> str:
        if ctx.last_attempt:
            return 'I guess you never learn'
        else:
            raise ModelRetry('Please try again.')

    result = agent.run_sync('Always fail!')
    assert result.output == snapshot('{"always_fail":"I guess you never learn"}')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Always fail!',
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='always_fail', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='always_fail',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='always_fail', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=62, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='always_fail',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='always_fail', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=72, output_tokens=6),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='always_fail',
                        content='I guess you never learn',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"always_fail":"I guess you never learn"}')],
                usage=RequestUsage(input_tokens=77, output_tokens=14),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
