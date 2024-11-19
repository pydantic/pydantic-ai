import json
from typing import Annotated

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from pydantic_ai import Agent, CallContext, UserError
from pydantic_ai.messages import Message, ModelAnyResponse, ModelTextResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel


def test_retriever_no_ctx():
    agent = Agent(TestModel())

    with pytest.raises(UserError) as exc_info:

        @agent.retriever  # pyright: ignore[reportArgumentType]
        def invalid_retriever(x: int) -> str:  # pragma: no cover
            return 'Hello'

    assert str(exc_info.value) == snapshot(
        'Error generating schema for test_retriever_no_ctx.<locals>.invalid_retriever:\n'
        '  First argument must be a CallContext instance when using `.retriever`'
    )


def test_retriever_plain_with_ctx():
    agent = Agent(TestModel())

    with pytest.raises(UserError) as exc_info:

        @agent.retriever_plain
        async def invalid_retriever(ctx: CallContext[None]) -> str:  # pragma: no cover
            return 'Hello'

    assert str(exc_info.value) == snapshot(
        'Error generating schema for test_retriever_plain_with_ctx.<locals>.invalid_retriever:\n'
        '  CallContext instance can only be used with `.retriever`'
    )


def test_retriever_ctx_second():
    agent = Agent(TestModel())

    with pytest.raises(UserError) as exc_info:

        @agent.retriever  # pyright: ignore[reportArgumentType]
        def invalid_retriever(x: int, ctx: CallContext[None]) -> str:  # pragma: no cover
            return 'Hello'

    assert str(exc_info.value) == snapshot(
        'Error generating schema for test_retriever_ctx_second.<locals>.invalid_retriever:\n'
        '  First argument must be a CallContext instance when using `.retriever`\n'
        '  CallContext instance can only be used as the first argument'
    )


async def google_style_docstring(foo: int, bar: str) -> str:  # pragma: no cover
    """Do foobar stuff, a lot.

    Args:
        foo: The foo thing.
        bar: The bar thing.
    """
    return f'{foo} {bar}'


async def get_json_schema(_messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
    assert len(info.retrievers) == 1
    r = next(iter(info.retrievers.values()))
    return ModelTextResponse(json.dumps(r.json_schema))


def test_docstring_google():
    agent = Agent(FunctionModel(get_json_schema))
    agent.retriever_plain(google_style_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            'description': 'Do foobar stuff, a lot.',
            'additionalProperties': False,
            'properties': {
                'foo': {'description': 'The foo thing.', 'title': 'Foo', 'type': 'integer'},
                'bar': {'description': 'The bar thing.', 'title': 'Bar', 'type': 'string'},
            },
            'required': ['foo', 'bar'],
            'type': 'object',
        }
    )
    # description should be the first key
    assert next(iter(json_schema)) == 'description'


def sphinx_style_docstring(foo: int, /) -> str:  # pragma: no cover
    """Sphinx style docstring.

    :param foo: The foo thing.
    :return: The result.
    """
    return str(foo)


def test_docstring_sphinx():
    agent = Agent(FunctionModel(get_json_schema))
    agent.retriever_plain(sphinx_style_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            'description': 'Sphinx style docstring.',
            'additionalProperties': False,
            'properties': {
                'foo': {'description': 'The foo thing.', 'title': 'Foo', 'type': 'integer'},
            },
            'required': ['foo'],
            'type': 'object',
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


def test_docstring_numpy():
    agent = Agent(FunctionModel(get_json_schema))
    agent.retriever_plain(numpy_style_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            'description': 'Numpy style docstring.',
            'additionalProperties': False,
            'properties': {
                'foo': {'description': 'The foo thing.', 'title': 'Foo', 'type': 'integer'},
                'bar': {'description': 'The bar thing.', 'title': 'Bar', 'type': 'string'},
            },
            'required': ['foo', 'bar'],
            'type': 'object',
        }
    )


def unknown_docstring(**kwargs: int) -> str:  # pragma: no cover
    """Unknown style docstring."""
    return str(kwargs)


def test_docstring_unknown():
    agent = Agent(FunctionModel(get_json_schema))
    agent.retriever_plain(unknown_docstring)

    result = agent.run_sync('Hello')
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            'description': 'Unknown style docstring.',
            'additionalProperties': True,
            'properties': {},
            'type': 'object',
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
    # fmt: on
    return f'{foo} {bar}'


def test_docstring_google_no_body():
    agent = Agent(FunctionModel(get_json_schema))
    agent.retriever_plain(google_style_docstring_no_body)

    result = agent.run_sync('')
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            'additionalProperties': False,
            'properties': {
                'foo': {'description': 'The foo thing.', 'title': 'Foo', 'type': 'integer'},
                'bar': {'description': 'from fields', 'title': 'Bar', 'type': 'string'},
            },
            'required': ['foo', 'bar'],
            'type': 'object',
        }
    )


def test_takes_just_model():
    agent = Agent()

    class Foo(BaseModel):
        x: int
        y: str

    @agent.retriever_plain
    def takes_just_model(model: Foo) -> str:
        return f'{model.x} {model.y}'

    result = agent.run_sync('', model=FunctionModel(get_json_schema))
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            'title': 'Foo',
            'properties': {'x': {'title': 'X', 'type': 'integer'}, 'y': {'title': 'Y', 'type': 'string'}},
            'required': ['x', 'y'],
            'type': 'object',
        }
    )

    result = agent.run_sync('', model=TestModel())
    assert result.data == snapshot('{"takes_just_model":"0 a"}')


def test_takes_model_and_int():
    agent = Agent()

    class Foo(BaseModel):
        x: int
        y: str

    @agent.retriever_plain
    def takes_just_model(model: Foo, z: int) -> str:
        return f'{model.x} {model.y} {z}'

    result = agent.run_sync('', model=FunctionModel(get_json_schema))
    json_schema = json.loads(result.data)
    assert json_schema == snapshot(
        {
            '$defs': {
                'Foo': {
                    'properties': {
                        'x': {'title': 'X', 'type': 'integer'},
                        'y': {'title': 'Y', 'type': 'string'},
                    },
                    'required': ['x', 'y'],
                    'title': 'Foo',
                    'type': 'object',
                }
            },
            'additionalProperties': False,
            'properties': {
                'model': {'$ref': '#/$defs/Foo'},
                'z': {'title': 'Z', 'type': 'integer'},
            },
            'required': ['model', 'z'],
            'type': 'object',
        }
    )

    result = agent.run_sync('', model=TestModel())
    assert result.data == snapshot('{"takes_just_model":"0 a 0"}')
