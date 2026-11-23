"""Output functions are called with their arguments under their Python parameter names, in every output mode.

An output function's JSON schema can name a parameter differently from the function: an aliased parameter is named
after its alias, a `SkipJsonSchema` parameter is left out, `**kwargs` stays open, and a single model-like parameter
takes the model's own schema. Every output mode turns what the model sent back into a call. Each test here crosses
signature shapes with output modes, and asserts what the function received. The cells that are invalid by design
assert the documented behavior instead.

These aren't VCR tests: which argument a value reaches is in-memory framework behavior that no provider response
changes. `FunctionModel` replies with exactly the arguments each schema advertises.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.output import NativeOutput, OutputSpec, PromptedOutput, TextOutput, ToolOutput


@dataclass(frozen=True)
class Mode:
    id: str
    output_type: Callable[[Callable[..., str]], OutputSpec[str]]
    reply: Literal['tool_call', 'json', 'union_json', 'text']
    """What the model replies: a call of the function's output tool, its arguments as JSON text, those arguments in
    the envelope of a `NativeOutput` or `PromptedOutput` union, or the plain text `hello`."""
    stream: bool = False


def other(n: int) -> str:
    """The other member of each union, which the model never picks."""
    return f'other n={n}'


STRUCTURED_MODES = [
    Mode('tool', lambda f: f, 'tool_call'),
    Mode('tool-output', lambda f: ToolOutput(f), 'tool_call'),
    Mode('native', lambda f: NativeOutput(f), 'json'),
    Mode('prompted', lambda f: PromptedOutput(f), 'json'),
    Mode('tool-union', lambda f: [f, other], 'tool_call'),
    Mode('native-union', lambda f: NativeOutput([f, other]), 'union_json'),
    Mode('prompted-union', lambda f: PromptedOutput([f, other]), 'union_json'),
    Mode('tool-or-text', lambda f: [f, str], 'tool_call'),
    Mode('tool-stream', lambda f: f, 'tool_call', stream=True),
    Mode('native-stream', lambda f: NativeOutput(f), 'json', stream=True),
]

TEXT_MODES = [
    Mode('text', lambda f: TextOutput(f), 'text'),
    Mode('text-stream', lambda f: TextOutput(f), 'text', stream=True),
    Mode('text-or-tool', lambda f: [TextOutput(f), other], 'text'),
]


def model_for(mode: Mode, args: dict[str, object]) -> FunctionModel:
    """A model that replies as `mode` says, sending `args` as the function's arguments."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if mode.reply == 'tool_call':
            name = next(tool.name for tool in info.output_tools if tool.name != 'final_result_other')
            return ModelResponse(parts=[ToolCallPart(name, args)])
        if mode.reply == 'union_json':
            output_object = info.model_request_parameters.output_object
            assert output_object is not None
            members = output_object.json_schema['properties']['result']['anyOf']
            kind = next(kind for member in members if (kind := member['properties']['kind']['const']) != 'other')
            return ModelResponse(parts=[TextPart(json.dumps({'result': {'kind': kind, 'data': args}}))])
        return ModelResponse(parts=[TextPart('hello' if mode.reply == 'text' else json.dumps(args))])

    async def stream_respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if mode.reply == 'tool_call':
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args=json.dumps(args))}
        else:
            yield 'hello' if mode.reply == 'text' else json.dumps(args)

    return FunctionModel(respond, stream_function=stream_respond)


async def run_output(agent: Agent[None, str], *, stream: bool) -> str:
    if stream:
        async with agent.run_stream('hi') as result:
            return await result.get_output()
    return (await agent.run('hi')).output


class Point(BaseModel):
    value: int


class AliasedPoint(BaseModel):
    value: int = Field(alias='v')


class PointDict(TypedDict):
    value: int


@dataclass
class PointDataclass:
    value: int


def plain(city: str) -> str:
    return f'city={city}'


def alias(city: Annotated[str, Field(alias='cityName')]) -> str:
    return f'city={city}'


def validation_alias(city: Annotated[str, Field(validation_alias='cityName')]) -> str:
    return f'city={city}'


def alias_choices(city: Annotated[str, Field(validation_alias=AliasChoices('cityName', 'c'))]) -> str:
    return f'city={city}'


def alias_and_skip_json_schema(city: Annotated[str, Field(alias='cityName')], b: SkipJsonSchema[int] = 1) -> str:
    return f'city={city} b={b}'


def alias_and_var_kwargs(city: Annotated[str, Field(alias='cityName')], **kwargs: int) -> str:
    return f'city={city} kwargs={kwargs}'


def two_aliases(a: Annotated[int, Field(alias='A')], b: Annotated[int, Field(alias='B')]) -> str:
    return f'a={a} b={b}'


def skip_json_schema_second(a: int, b: SkipJsonSchema[int] = 1) -> str:
    return f'a={a} b={b}'


def skip_json_schema_only(a: SkipJsonSchema[int] = 1) -> str:
    return f'a={a}'


def var_kwargs(a: int, **kwargs: int) -> str:
    return f'a={a} kwargs={kwargs}'


def var_kwargs_only(**kwargs: int) -> str:
    return f'kwargs={kwargs}'


def basemodel(point: Point) -> str:
    return f'point={point!r}'


def aliased_basemodel_param(point: Annotated[Point, Field(alias='payload')]) -> str:
    return f'point={point!r}'


def basemodel_field_alias(point: AliasedPoint) -> str:
    return f'value={point.value}'


def dataclass_param(point: PointDataclass) -> str:
    return f'point={point!r}'


def typeddict(point: PointDict) -> str:
    return f'point={point!r}'


def typeddict_and_int(point: PointDict, n: int) -> str:
    return f'point={point!r} n={n}'


def ctx_only(ctx: RunContext[None]) -> str:
    return 'ctx only'


def ctx_and_alias(ctx: RunContext[None], city: Annotated[str, Field(alias='cityName')]) -> str:
    return f'city={city}'


def positional_only_alias(city: Annotated[str, Field(alias='cityName')], /) -> str:
    return f'city={city}'


def positional_only_alias_and_skip_json_schema(
    city: Annotated[str, Field(alias='cityName')], /, b: SkipJsonSchema[int] = 1
) -> str:
    return f'city={city} b={b}'


def var_args(*args: int) -> str:
    return f'args={args}'


def alias_names_hidden_param(a: Annotated[int, Field(alias='b')] = 0, b: SkipJsonSchema[int] = 9) -> str:
    return f'a={a} b={b}'


def literal_param(x: Literal['a', 'b']) -> str:
    return f'x={x}'


# `Any` is the parameter type under test here, not a shortcut.
def any_param(x: Any) -> str:
    return f'x={x}'


@dataclass(frozen=True)
class StructuredCase:
    id: str
    function: Callable[..., str]
    args: dict[str, object]
    """The arguments the model sends, named as the function's JSON schema names them."""
    expected: str
    union_json_xfail: pytest.MarkDecorator | None = None
    """The known failure when the model picks the function from a `NativeOutput` or `PromptedOutput` union."""


STRUCTURED_CASES = [
    StructuredCase('plain', plain, {'city': 'Paris'}, 'city=Paris'),
    StructuredCase('alias', alias, {'cityName': 'Paris'}, 'city=Paris'),
    StructuredCase('validation-alias', validation_alias, {'cityName': 'Paris'}, 'city=Paris'),
    StructuredCase('alias-choices', alias_choices, {'cityName': 'Paris'}, 'city=Paris'),
    StructuredCase('alias-and-skip-json-schema', alias_and_skip_json_schema, {'cityName': 'Paris'}, 'city=Paris b=1'),
    StructuredCase(
        'alias-and-var-kwargs', alias_and_var_kwargs, {'cityName': 'Paris', 'x': 2}, "city=Paris kwargs={'x': 2}"
    ),
    StructuredCase('two-aliases', two_aliases, {'A': 1, 'B': 2}, 'a=1 b=2'),
    StructuredCase('skip-json-schema-second', skip_json_schema_second, {'a': 1}, 'a=1 b=1'),
    StructuredCase('skip-json-schema-only', skip_json_schema_only, {}, 'a=1'),
    StructuredCase('var-kwargs', var_kwargs, {'a': 1, 'x': 2}, "a=1 kwargs={'x': 2}"),
    StructuredCase('var-kwargs-only', var_kwargs_only, {'x': 2}, "kwargs={'x': 2}"),
    StructuredCase('basemodel', basemodel, {'value': 21}, 'point=Point(value=21)'),
    # An aliased model parameter isn't unwrapped: its alias names the schema's one property.
    StructuredCase(
        'aliased-basemodel-param', aliased_basemodel_param, {'payload': {'value': 21}}, 'point=Point(value=21)'
    ),
    StructuredCase('basemodel-field-alias', basemodel_field_alias, {'v': 21}, 'value=21'),
    StructuredCase('dataclass', dataclass_param, {'value': 21}, 'point=PointDataclass(value=21)'),
    StructuredCase(
        'typeddict',
        typeddict,
        {'value': 21},
        "point={'value': 21}",
        union_json_xfail=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=(
                'A union checks the validated value against the picked function with `isinstance`, which a `TypedDict` '
                "can't be checked with, so the function isn't called and the raw dict is the output."
            ),
        ),
    ),
    StructuredCase('typeddict-and-int', typeddict_and_int, {'point': {'value': 21}, 'n': 3}, "point={'value': 21} n=3"),
    StructuredCase('ctx-only', ctx_only, {}, 'ctx only'),
    StructuredCase('ctx-and-alias', ctx_and_alias, {'cityName': 'Paris'}, 'city=Paris'),
    StructuredCase('positional-only-alias', positional_only_alias, {'cityName': 'Paris'}, 'city=Paris'),
    StructuredCase(
        'positional-only-alias-and-skip-json-schema',
        positional_only_alias_and_skip_json_schema,
        {'cityName': 'Paris'},
        'city=Paris b=1',
    ),
    StructuredCase(
        'var-args',
        var_args,
        {'args': [1, 2]},
        'args=(1, 2)',
        union_json_xfail=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=(
                'A union checks the validated value, the list of positional arguments, against their item type with '
                "`isinstance`, so the function isn't called and the raw list is the output."
            ),
        ),
    ),
    # Pydantic reads the `b` key for both parameters, `a` by its alias and `b` by its name, so both get the value.
    StructuredCase('alias-names-hidden-param', alias_names_hidden_param, {'b': 5}, 'a=5 b=5'),
    StructuredCase(
        'literal',
        literal_param,
        {'x': 'a'},
        'x=a',
        union_json_xfail=pytest.mark.xfail(
            strict=True,
            raises=TypeError,
            reason=(
                "A union checks the validated value against the picked function with `isinstance`, which can't check "
                'a `Literal`, so the run raises `TypeError`: https://github.com/pydantic/pydantic-ai/issues/12640, '
                'fixed by https://github.com/pydantic/pydantic-ai/pull/12641'
            ),
        ),
    ),
    StructuredCase(
        'any',
        any_param,
        {'x': 1},
        'x=1',
        union_json_xfail=pytest.mark.xfail(
            strict=True,
            raises=AssertionError,
            reason=(
                "A union checks the validated value against the picked function with `isinstance`, which can't check "
                '`Any`, so it calls the first member whose type the value matches: `other` runs instead.'
            ),
        ),
    ),
]


@pytest.mark.parametrize(
    ('case', 'mode'),
    [
        pytest.param(
            case,
            mode,
            id=f'{case.id}-in-{mode.id}',
            marks=case.union_json_xfail if case.union_json_xfail and mode.reply == 'union_json' else (),
        )
        for case in STRUCTURED_CASES
        for mode in STRUCTURED_MODES
    ],
)
async def test_structured_output_function(case: StructuredCase, mode: Mode):
    """The function gets each argument the model sends, under the parameter's Python name."""
    agent = Agent(model_for(mode, case.args), output_type=mode.output_type(case.function))
    assert await run_output(agent, stream=mode.stream) == case.expected


def text_plain(text: str) -> str:
    return f'text={text}'


def text_alias(text: Annotated[str, Field(alias='theText')]) -> str:
    return f'text={text}'


def text_alias_and_skip_json_schema(text: Annotated[str, Field(alias='theText')], b: SkipJsonSchema[int] = 1) -> str:
    return f'text={text}'


def text_alias_and_var_kwargs(text: Annotated[str, Field(alias='theText')] = 'default', **kwargs: int) -> str:
    return f'text={text}'


def text_skip_json_schema(text: str, b: SkipJsonSchema[int] = 1) -> str:
    return f'text={text}'


def text_var_kwargs(text: str, **kwargs: int) -> str:
    return f'text={text}'


def text_ctx_and_alias(ctx: RunContext[None], text: Annotated[str, Field(alias='theText')]) -> str:
    return f'text={text}'


def text_positional_only_alias_and_skip_json_schema(
    text: Annotated[str, Field(alias='theText')], /, b: SkipJsonSchema[int] = 1
) -> str:
    return f'text={text}'


def text_alias_path_and_skip_json_schema(
    text: Annotated[str, Field(validation_alias=AliasPath('payload', 'text'))], b: SkipJsonSchema[int] = 1
) -> str:
    return f'text={text}'


class TextModel(BaseModel):
    text: str


class TextDict(TypedDict):
    text: str


def text_basemodel(model: TextModel) -> str:
    return f'model={model!r}'  # pragma: no cover


def text_basemodel_named_like_field(text: TextModel) -> str:
    return f'text={text!r}'  # pragma: no cover


def text_typeddict(model: TextDict) -> str:
    return f'model={model!r}'  # pragma: no cover


def text_ctx_only(ctx: RunContext[None]) -> str:
    return 'ctx only'  # pragma: no cover


def text_alias_names_hidden_param(
    text: Annotated[str, Field(alias='b')] = 'default', b: SkipJsonSchema[str] = 'x'
) -> str:
    return f'text={text} b={b}'


@dataclass(frozen=True)
class TextCase:
    id: str
    function: Callable[..., str]
    expected: str | type[UserError]
    """The output for the text `hello`, or `UserError` when the agent rejects the function."""


TEXT_CASES = [
    TextCase('plain', text_plain, 'text=hello'),
    TextCase('alias', text_alias, 'text=hello'),
    TextCase('alias-and-skip-json-schema', text_alias_and_skip_json_schema, 'text=hello'),
    TextCase('alias-and-var-kwargs', text_alias_and_var_kwargs, 'text=hello'),
    TextCase('skip-json-schema', text_skip_json_schema, 'text=hello'),
    TextCase('var-kwargs', text_var_kwargs, 'text=hello'),
    TextCase('ctx-and-alias', text_ctx_and_alias, 'text=hello'),
    TextCase(
        'positional-only-alias-and-skip-json-schema', text_positional_only_alias_and_skip_json_schema, 'text=hello'
    ),
    TextCase('alias-path-and-skip-json-schema', text_alias_path_and_skip_json_schema, 'text=hello'),
    # The text is passed as a `str`, so a model-like parameter is rejected, even one whose model has one `str` field.
    TextCase('basemodel', text_basemodel, UserError),
    TextCase('basemodel-named-like-field', text_basemodel_named_like_field, UserError),
    TextCase('typeddict', text_typeddict, UserError),
    TextCase('ctx-only', text_ctx_only, UserError),
    # A property named after a parameter's Python name names only that parameter, so the text goes to `b`, and `text`
    # keeps its default: https://github.com/pydantic/pydantic-ai/issues/13438
    TextCase('alias-names-hidden-param', text_alias_names_hidden_param, 'text=default b=hello'),
]


@pytest.mark.parametrize(
    ('case', 'mode'),
    [pytest.param(case, mode, id=f'{case.id}-in-{mode.id}') for case in TEXT_CASES for mode in TEXT_MODES],
)
async def test_text_output_function(case: TextCase, mode: Mode):
    """A `TextOutput` function gets the text under its `str` parameter's Python name, or is rejected up front."""
    if isinstance(case.expected, str):
        agent = Agent(model_for(mode, {}), output_type=mode.output_type(case.function))
        assert await run_output(agent, stream=mode.stream) == case.expected
    else:
        with pytest.raises(case.expected, match='TextOutput must take a function taking a single `str` argument'):
            Agent(model_for(mode, {}), output_type=mode.output_type(case.function))
