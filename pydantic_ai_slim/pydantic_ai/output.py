from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Literal, Protocol, cast, overload

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import TypeAliasType, TypeForm, TypeVar

from . import _utils, exceptions
from ._json_schema import InlineDefsJsonSchemaTransformer
from ._run_context import RunContext
from .messages import ToolCallPart
from .tools import ObjectJsonSchema, ToolDefinition

__all__ = (
    # classes
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    'Choice',
    'Choices',
    'BoolCriteria',
    'OutputObjectDefinition',
    'OutputContext',
    # types
    'OutputDataT',
    'OutputMode',
    'StructuredOutputMode',
    'OutputSpec',
    'OutputTypeOrFunction',
    'TextOutputFunc',
)

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
ChoiceValueT = TypeVar('ChoiceValueT')
"""Function-scoped type variable for what a `Choice` stands for, so `Choice.__init__` can specialize `self`."""

OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the output data type of a run."""

# TODO(v3): remove the `tool_or_text` output mode
OutputMode = Literal['text', 'tool', 'native', 'prompted', 'tool_or_text', 'image', 'auto']
"""All output modes.

- `tool_or_text` is deprecated and no longer in use.
- `auto` means the model will automatically choose a structured output mode based on the model's `ModelProfile.default_structured_output_mode`.
"""
StructuredOutputMode = Literal['tool', 'native', 'prompted']
"""Output modes that can be used for structured output. Used by ModelProfile.default_structured_output_mode"""


OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction',
    type[T_co] | TypeForm[T_co] | Callable[..., Awaitable[T_co] | T_co],
    type_params=(T_co,),
)
"""Definition of an output type or function.

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""


class _NoneOutput(Protocol[T_co]):
    """The type of a bare `None` output type, as seen by a type checker.

    `None` in `output_type=[Foo, None]` or `ToolOutput(None)` is a value, not a type, so `type[T]` cannot match it
    and would not add `None` to the output type if it did. Matching `None` structurally binds `T` to `None` through
    `__class__`, while `__bool__` returning `Literal[False]` keeps other values out (a class annotating its own
    `__bool__` that way would also match). A list of only `None` still type-checks, though it is refused at run time.

    Pyright 1.1.412 and later also matches `None` as a `TypeForm`, so this is for older Pyright versions.
    """

    @property
    def __class__(self) -> type[T_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]

    def __bool__(self) -> Literal[False]: ...


TextOutputFunc = TypeAliasType(
    'TextOutputFunc',
    Callable[[RunContext[Any], str], Awaitable[T_co] | T_co] | Callable[[str], Awaitable[T_co] | T_co],
    type_params=(T_co,),
)
"""Definition of a function that will be called to process the model's plain text output. The function must take a single string argument.

You should not need to import or use this type directly.

See [text output docs](../output.md#text-output) for more information.
"""


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use a tool for output and optionally customize the tool.

    Example:
    ```python {title="tool_output.py"}
    from pydantic import BaseModel

    from pydantic_ai import Agent, ToolOutput


    class Fruit(BaseModel):
        name: str
        color: str


    class Vehicle(BaseModel):
        name: str
        wheels: int


    agent = Agent(
        'openai:gpt-5.2',
        output_type=[
            ToolOutput(Fruit, name='return_fruit'),
            ToolOutput(Vehicle, name='return_vehicle'),
        ],
    )
    result = agent.run_sync('What is a banana?')
    print(repr(result.output))
    #> Fruit(name='banana', color='yellow')
    ```
    """

    output: OutputTypeOrFunction[OutputDataT]
    """An output type or function."""
    name: str | None
    """The name of the tool that will be passed to the model. If not specified and only one output is provided, `final_result` will be used. If multiple outputs are provided, the name of the output type or function will be added to the tool name."""
    description: str | None
    """The description of the tool that will be passed to the model. If not specified, the docstring of the output type or function will be used."""
    max_retries: int | None
    """Per-tool retry limit for this output tool.

    Overrides the output side of the agent's retry budget, which itself acts as the per-tool default
    for output tools that do not specify their own limit. If not set, the agent-level value is used.
    """
    strict: bool | None
    """Whether to use strict mode for the tool."""
    sequential: bool
    """Whether this output tool must run as a barrier, not overlapping with other tool calls.

    Only meaningful under `end_strategy='exhaustive'`, where tools otherwise run in parallel: a
    `sequential=True` output tool runs alone, so function tools the model emitted before it complete
    first. Under `'early'`/`'graceful'` output tools already run sequentially, so this has no effect.
    """

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | _NoneOutput[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
        sequential: bool = False,
    ):
        if max_retries is not None and max_retries < 0:
            raise exceptions.UserError(f'max_retries must be >= 0, got {max_retries}')
        # A bare `None` is kept as is: the output schema treats it as the `None` output type.
        self.output = cast(OutputTypeOrFunction[OutputDataT], type_)
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict
        self.sequential = sequential


@dataclass(init=False)
class NativeOutput(Generic[OutputDataT]):
    """Marker class to use the model's native structured outputs functionality for outputs and optionally customize the name and description.

    Example:
    ```python {title="native_output.py" requires="tool_output.py"}
    from pydantic_ai import Agent, NativeOutput

    from tool_output import Fruit, Vehicle

    agent = Agent(
        'openai:gpt-5.2',
        output_type=NativeOutput(
            [Fruit, Vehicle],
            name='Fruit or vehicle',
            description='Return a fruit or vehicle.'
        ),
    )
    result = agent.run_sync('What is a Ford Explorer?')
    print(repr(result.output))
    #> Vehicle(name='Ford Explorer', wheels=4)
    ```
    """

    outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]]
    """The output types or functions."""
    name: str | None
    """The name of the structured output that will be passed to the model. If not specified and only one output is provided, the name of the output type or function will be used."""
    description: str | None
    """The description of the structured output that will be passed to the model. If not specified and only one output is provided, the docstring of the output type or function will be used."""
    strict: bool | None
    """Whether to use strict mode for the output, if the model supports it."""
    template: str | Literal[False] | None
    """Template for the prompt passed to the model.
    The '{schema}' placeholder will be replaced with the output JSON schema.
    If no template is specified but the model's profile indicates that it requires the schema to be sent as a prompt, the default template specified on the profile will be used.
    Set to `False` to disable the schema prompt entirely.
    """

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT]
        | Sequence[OutputTypeOrFunction[OutputDataT] | _NoneOutput[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
        template: str | Literal[False] | None = None,
    ):
        # A bare `None` item stays in the list: the output schema treats it as the `None` output type.
        self.outputs = cast(OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]], outputs)
        self.name = name
        self.description = description
        self.strict = strict
        self.template = template


@dataclass(init=False)
class PromptedOutput(Generic[OutputDataT]):
    """Marker class to use a prompt to tell the model what to output and optionally customize the prompt.

    Example:
    ```python {title="prompted_output.py" requires="tool_output.py"}
    from pydantic import BaseModel

    from pydantic_ai import Agent, PromptedOutput

    from tool_output import Vehicle


    class Device(BaseModel):
        name: str
        kind: str


    agent = Agent(
        'openai:gpt-5.2',
        output_type=PromptedOutput(
            [Vehicle, Device],
            name='Vehicle or device',
            description='Return a vehicle or device.'
        ),
    )
    result = agent.run_sync('What is a MacBook?')
    print(repr(result.output))
    #> Device(name='MacBook', kind='laptop')

    agent = Agent(
        'openai:gpt-5.2',
        output_type=PromptedOutput(
            [Vehicle, Device],
            template='Gimme some JSON: {schema}'
        ),
    )
    result = agent.run_sync('What is a Ford Explorer?')
    print(repr(result.output))
    #> Vehicle(name='Ford Explorer', wheels=4)
    ```
    """

    outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]]
    """The output types or functions."""
    name: str | None
    """The name of the structured output that will be passed to the model. If not specified and only one output is provided, the name of the output type or function will be used."""
    description: str | None
    """The description that will be passed to the model. If not specified and only one output is provided, the docstring of the output type or function will be used."""
    template: str | Literal[False] | None
    """Template for the prompt passed to the model.
    The '{schema}' placeholder will be replaced with the output JSON schema.
    If not specified, the default template specified on the model's profile will be used.
    Set to `False` to disable the schema prompt entirely.
    """

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT]
        | Sequence[OutputTypeOrFunction[OutputDataT] | _NoneOutput[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        template: str | Literal[False] | None = None,
    ):
        # A bare `None` item stays in the list: the output schema treats it as the `None` output type.
        self.outputs = cast(OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]], outputs)
        self.name = name
        self.description = description
        self.template = template


@dataclass
class OutputObjectDefinition:
    """Definition of an output object used for structured output generation."""

    json_schema: ObjectJsonSchema
    name: str | None = None
    description: str | None = None
    strict: bool | None = None


@dataclass
class OutputContext:
    """Context about the output being processed, passed to output hooks."""

    mode: OutputMode
    """The schema's output mode ('text', 'native', 'prompted', 'tool', 'image', 'auto').

    This reflects the configured schema, not the format of this particular response. For
    example, a `ToolOutputSchema` with a `text_processor` (hybrid mode) reports `'tool'`
    even if the model returned text — check [`tool_call`][pydantic_ai.output.OutputContext.tool_call]
    to distinguish."""
    output_type: type[Any] | None
    """The resolved output type (e.g. MyModel, str). For output functions, the function's input type (what the model produces)."""
    object_def: OutputObjectDefinition | None
    """The output object definition (schema, name, description), if structured output."""
    has_function: bool
    """Whether there's an output function to call in the execute step."""
    function_name: str | None = None
    """Name of the output function that will run, when known. `None` for union processors that dispatch
    by output subtype, or when the schema has no function."""
    tool_call: ToolCallPart | None = None
    """The tool call part, for tool-based output. `None` when the current output did not arrive via a tool call (text or image)."""
    tool_def: ToolDefinition | None = None
    """The tool definition, for tool-based output. `None` when the current output did not arrive via a tool call."""
    allows_text: bool = False
    """Whether the schema accepts text output (including via a `text_processor` on a `ToolOutputSchema`)."""
    allows_image: bool = False
    """Whether the schema accepts image output."""
    allows_deferred_tools: bool = False
    """Whether the schema accepts deferred tool requests as output."""


@dataclass
class TextOutput(Generic[OutputDataT]):
    """Marker class to use text output for an output function taking a string argument.

    Example:
    ```python
    from pydantic_ai import Agent, TextOutput


    def split_into_words(text: str) -> list[str]:
        return text.split()


    agent = Agent(
        'openai:gpt-5.2',
        output_type=TextOutput(split_into_words),
    )
    result = agent.run_sync('Who was Albert Einstein?')
    print(result.output)
    #> ['Albert', 'Einstein', 'was', 'a', 'German-born', 'theoretical', 'physicist.']
    ```

    !!! note
        When streaming, [`stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text] does not apply the
        wrapped function. Use [`stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output] to stream
        the value it produces.
    """

    output_function: TextOutputFunc[OutputDataT]
    """The function that will be called to process the model's plain text output. The function must take a single string argument."""


def StructuredDict(
    json_schema: JsonSchemaValue, name: str | None = None, description: str | None = None
) -> type[JsonSchemaValue]:
    """Returns a `dict[str, Any]` subclass with a JSON schema attached that will be used for structured output.

    Args:
        json_schema: A JSON schema of type `object` defining the structure of the dictionary content.
        name: Optional name of the structured output. If not provided, the `title` field of the JSON schema will be used if it's present.
        description: Optional description of the structured output. If not provided, the `description` field of the JSON schema will be used if it's present.

    Example:
    ```python {title="structured_dict.py"}
    from pydantic_ai import Agent, StructuredDict

    schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'}
        },
        'required': ['name', 'age']
    }

    agent = Agent('openai:gpt-5.2', output_type=StructuredDict(schema))
    result = agent.run_sync('Create a person')
    print(result.output)
    #> {'name': 'John Doe', 'age': 30}
    ```
    """
    json_schema = _utils.check_object_json_schema(json_schema)

    # Pydantic `TypeAdapter` fails when `object.__get_pydantic_json_schema__` has `$defs`, so we inline them
    # See https://github.com/pydantic/pydantic/issues/12145
    if '$defs' in json_schema:
        json_schema = InlineDefsJsonSchemaTransformer(json_schema).walk()
        if '$defs' in json_schema:
            raise exceptions.UserError(
                '`StructuredDict` does not currently support recursive `$ref`s and `$defs`. See https://github.com/pydantic/pydantic/issues/12145 for more information.'
            )

    if name:
        json_schema['title'] = name

    if description:
        json_schema['description'] = description

    class _StructuredDict(JsonSchemaValue):
        __is_model_like__ = True

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.dict_schema(
                keys_schema=core_schema.str_schema(),
                values_schema=core_schema.any_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            return json_schema

    return _StructuredDict


@dataclass(init=False)
class Choice(Generic[T_co]):
    """One choice in a [`Choices`][pydantic_ai.output.Choices] set: what it means, and what it stands for.

    Building a set from descriptions alone yields the key the model picked, so `Choice` is only reached for
    when a choice stands for something other than its key:

    ```python {title="choice.py"}
    from pydantic_ai import Agent, Choice, Choices

    Card = Choices(
        {
            'visa': Choice('Any card starting with a 4.', value=4),
            'amex': Choice('Any card starting with a 3.', value=3),
        }
    )

    agent = Agent('openai:gpt-5.2', output_type=Card)
    result = agent.run_sync('4111 1111 1111 1111')
    print(result.output)
    #> 4
    ```
    """

    description: str | None
    """What this choice means, shown to the model beside the choice itself."""

    value: T_co
    """What the picked choice resolves to. Defaults to the choice's own key.

    A callable value is *called* when the model picks this choice, the way an
    [output function](../output.md#output-functions) is, so the run's output is what the action returned.
    It is called with no arguments, so bind what it needs with `functools.partial` or a closure, and it can
    be `async`. A `Choices` set with a callable value can only be used as an agent's `output_type`.
    """

    @overload
    def __init__(self: Choice[str], description: str | None = None) -> None: ...

    @overload
    def __init__(
        self: Choice[ChoiceValueT], description: str | None = None, *, value: Callable[[], Awaitable[ChoiceValueT]]
    ) -> None: ...

    @overload
    def __init__(
        self: Choice[ChoiceValueT], description: str | None = None, *, value: Callable[[], ChoiceValueT]
    ) -> None: ...

    @overload
    def __init__(self: Choice[ChoiceValueT], description: str | None = None, *, value: ChoiceValueT) -> None: ...

    def __init__(self, description: str | None = None, *, value: Any = _utils.UNSET) -> None:
        if _requires_arguments(value):
            raise exceptions.UserError(
                'A callable `Choice` value is called with no arguments when the model picks it, '
                f'but {value!r} requires some. Bind them with `functools.partial` or a closure.'
            )
        self.description = description
        self.value = value


def _requires_arguments(value: Any) -> bool:
    """Whether `value` is a callable that would fail when called with no arguments."""
    if not callable(value):
        return False
    try:
        parameters = inspect.signature(value).parameters.values()
    except (TypeError, ValueError):
        # Some C callables have no introspectable signature; assume the best and let the call speak for itself.
        return False
    return any(
        parameter.default is parameter.empty
        and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        for parameter in parameters
    )


class _ChoicesActions:
    """Base class of the type [`Choices`][pydantic_ai.output.Choices] returns when a choice value is callable.

    Such a set is only meaningful as an agent's `output_type`, where `_output.py` validates the pick against
    `keys_type` and resolves and calls the value itself. Anywhere else there is nothing to call the action, so
    the core schema refuses to be built at all.
    """

    keys_type: ClassVar[type[Any]]
    """The same set of keys with no values attached, which is what validates the model's pick."""

    values: ClassVar[Mapping[str, Any]]
    """What each key stands for, resolved once the pick is final."""

    @staticmethod
    def of(output: Any) -> type[_ChoicesActions] | None:
        """`output` if it is a `Choices` set whose values include callables, else `None`."""
        if isinstance(output, type) and issubclass(output, _ChoicesActions):
            return output
        return None


@overload
def Choices(
    choices: Sequence[str] | Mapping[str, str], *, name: str | None = None, description: str | None = None
) -> type[str]: ...


@overload
def Choices(
    choices: Mapping[str, Choice[T_co]], *, name: str | None = None, description: str | None = None
) -> type[T_co]: ...


@overload
def Choices(
    choices: Mapping[str, str | Choice[T_co]], *, name: str | None = None, description: str | None = None
) -> type[str | T_co]: ...


def Choices(
    choices: Sequence[str] | Mapping[str, str | Choice[Any]],
    *,
    name: str | None = None,
    description: str | None = None,
) -> type[Any]:
    """Returns a type the model can only fill with one of `choices`, each described where it is built.

    Use it when the set is only known once the run is under way -- the actions available on the screen in front
    of an agent, the records a search returned. For a set you know when you write the code, use a `Literal` or
    an `Enum` (with [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings] to describe its
    members), which give you exhaustiveness checking that a run-time set cannot.

    Like [`StructuredDict`][pydantic_ai.output.StructuredDict] it returns a type, so it works as an
    `output_type`, as a field of a Pydantic model, and as a tool parameter.

    Args:
        choices: The choices, as a sequence of keys, a mapping from key to its description, or a mapping from
            key to a [`Choice`][pydantic_ai.output.Choice] carrying both a description and what the key stands
            for.
        name: Name of the output tool or structured output. Defaults to `'Choices'`.
        description: What the model is being asked to pick, e.g. `'Which action to take next.'`.

    Example:
    ```python {title="choices.py"}
    from pydantic_ai import Agent, Choices

    Intent = Choices(
        {
            'refund': 'The customer wants their money back.',
            'replace': 'The customer wants a working unit instead.',
            'escalate': 'Nobody on this tier can resolve it.',
        },
        name='customer_intent',
        description='What the customer is asking for.',
    )

    agent = Agent('openai:gpt-5.2', output_type=Intent)
    result = agent.run_sync('The blender arrived smashed. Just send me another one.')
    print(result.output)
    #> replace
    ```
    """
    if isinstance(choices, str):
        raise exceptions.UserError('`Choices` takes a sequence or mapping of choices, not a single string.')

    resolved: dict[str, Choice[Any]] = (
        {key: Choice(choice) if isinstance(choice, str) else choice for key, choice in choices.items()}
        if isinstance(choices, Mapping)
        else {key: Choice() for key in choices}
    )
    if not resolved:
        raise exceptions.UserError('`Choices` requires at least one choice.')

    choice_values: dict[str, Any] = {
        key: choice.value if _utils.is_set(choice.value) else key for key, choice in resolved.items()
    }
    # The values are only worth resolving when at least one of them isn't the key itself.
    resolves_values = any(value is not key for key, value in choice_values.items())
    has_actions = any(callable(value) for value in choice_values.values())

    descriptions = {key: choice.description for key, choice in resolved.items() if choice.description}
    if descriptions:
        # The same `anyOf`-of-`const`s shape `GenerateToolJsonSchema.enum_schema` emits for an enum whose
        # members carry docstrings, so that "pick one of these, and here is what each means" has one shape on
        # the wire however it was authored.
        options: list[JsonSchemaValue] = []
        for key in choice_values:
            option: JsonSchemaValue = {'const': key}
            if choice_description := descriptions.get(key):
                option['description'] = choice_description
            options.append(option)
        json_schema: JsonSchemaValue = {'type': 'string', 'anyOf': options}
    else:
        json_schema = {'type': 'string', 'enum': list(choice_values)}
    if description:
        json_schema['description'] = description

    class _Choices:
        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            schema = core_schema.literal_schema(list(choice_values))
            # An action's value is resolved and called once the pick is final, not here: a validator has
            # nowhere to await, and streaming re-validates a completed value on every later chunk.
            if resolves_values and not has_actions:
                schema = core_schema.no_info_after_validator_function(choice_values.__getitem__, schema)
            return schema

        @classmethod
        def __get_pydantic_json_schema__(
            cls, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
        ) -> JsonSchemaValue:
            return dict(json_schema)

    _Choices.__name__ = _Choices.__qualname__ = name or 'Choices'
    if not has_actions:
        return _Choices

    class _ChoicesWithActions(_ChoicesActions):
        keys_type = _Choices
        values = choice_values

        @classmethod
        def __get_pydantic_core_schema__(
            cls, source_type: Any, handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            raise exceptions.UserError(
                "A `Choices` set with a callable `Choice` value can only be used as an agent's `output_type`, "
                'as that is the only place Pydantic AI can call the action the model picked. '
                'To pick one as a tool parameter or a model field, give the choices plain values and call the '
                'action yourself.'
            )

    _ChoicesWithActions.__name__ = _ChoicesWithActions.__qualname__ = name or 'Choices'
    return _ChoicesWithActions


@dataclass(frozen=True, kw_only=True)
class BoolCriteria:
    """What a yes and what a no would each mean, for a `bool` field or parameter to carry in `Annotated`.

    A `bool` field's description says what is being asked; these two say what either answer amounts to,
    which is what a set of options gets from [`Choices()`][pydantic_ai.output.Choices] and an `Enum` gets
    from [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings]. Both descriptions reach the model
    in the schema, as a description on each of the two constants a boolean can be.

    It is a marker rather than a type, so the field stays a plain `bool` to every type checker, and the
    value you get back is a plain `True` or `False`:

    ```python {title="bool_criteria.py"}
    from typing import Annotated

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent, BoolCriteria


    class Settled(BaseModel):
        refunded: Annotated[
            bool,
            BoolCriteria(true='Money was returned to the customer.', false='No refund was issued.'),
        ] = Field(description='Was a refund issued?')


    agent = Agent('openai:gpt-5.2', output_type=Settled)
    result = agent.run_sync('We have sent the 40 pounds back to your card.')
    print(result.output.refunded)
    #> True
    ```

    On [TypeSafe's Jev](../models/typesafe.md), which asks a yes/no as its own primitive, the two land in
    that question's `criteria` as `true` and `false`, sent verbatim.
    """

    true: str
    """What it means for the answer to be `True`."""

    false: str
    """What it means for the answer to be `False`."""

    def __get_pydantic_json_schema__(
        self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(core_schema)
        # A `Literal` of `True` and/or `False` renders as a `boolean` too, but already pins its values in a
        # `const` or `enum` that the two meanings below would contradict or be shadowed by. The rendered schema
        # rather than the core schema is checked, so a `bool` wrapped in a validator is still accepted.
        if json_schema.get('type') != 'boolean' or 'const' in json_schema or 'enum' in json_schema:
            raise exceptions.UserError(
                '`BoolCriteria` says what each answer of a `bool` means, so it can only annotate a plain `bool`, '
                'not a `Literal` of `True` and/or `False` or any other type.'
            )
        # The same `anyOf`-of-`const`s shape `Choices()` and a described `Enum` produce, on the two constants a
        # boolean can be.
        return {
            **json_schema,
            'anyOf': [{'const': True, 'description': self.true}, {'const': False, 'description': self.false}],
        }


_OutputSpecItem = TypeAliasType(
    '_OutputSpecItem',
    OutputTypeOrFunction[T_co] | ToolOutput[T_co] | NativeOutput[T_co] | PromptedOutput[T_co] | TextOutput[T_co],
    type_params=(T_co,),
)


OutputSpec = TypeAliasType(
    'OutputSpec',
    _OutputSpecItem[T_co] | Sequence['OutputSpec[T_co] | _NoneOutput[T_co]'],
    type_params=(T_co,),
)
"""Specification of the agent's output data.

This can be a single type, a function, a sequence of types and/or functions (which can include `None` to allow no output), or an instance of one of the output mode marker classes:
- [`ToolOutput`][pydantic_ai.output.ToolOutput]
- [`NativeOutput`][pydantic_ai.output.NativeOutput]
- [`PromptedOutput`][pydantic_ai.output.PromptedOutput]
- [`TextOutput`][pydantic_ai.output.TextOutput]

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""
