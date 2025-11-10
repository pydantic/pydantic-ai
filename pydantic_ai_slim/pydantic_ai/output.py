from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema
from typing_extensions import TypeAliasType, TypeVar, deprecated

from . import _utils, exceptions
from ._json_schema import InlineDefsJsonSchemaTransformer
from .messages import ToolCallPart
from .tools import DeferredToolRequests, ObjectJsonSchema, RunContext, ToolDefinition

__all__ = (
    # classes
    'ToolOutput',
    'NativeOutput',
    'PromptedOutput',
    'TextOutput',
    'StructuredDict',
    'OutputObjectDefinition',
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

OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the output data type of a run."""

OutputMode = Literal['text', 'tool', 'native', 'prompted', 'tool_or_text', 'image', 'auto']
"""All output modes.

- `tool_or_text` is deprecated and no longer in use.
- `auto` means the model will automatically choose a structured output mode based on the model's `ModelProfile.default_structured_output_mode`.
"""
StructuredOutputMode = Literal['tool', 'native', 'prompted']
"""Output modes that can be used for structured output. Used by ModelProfile.default_structured_output_mode"""


OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction', type[T_co] | Callable[..., Awaitable[T_co] | T_co], type_params=(T_co,)
)
"""Definition of an output type or function.

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""


TextOutputFunc = TypeAliasType(
    'TextOutputFunc',
    Callable[[RunContext, str], Awaitable[T_co] | T_co] | Callable[[str], Awaitable[T_co] | T_co],
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
        'openai:gpt-4o',
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
    """The maximum number of retries for the tool."""
    strict: bool | None
    """Whether to use strict mode for the tool."""

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        max_retries: int | None = None,
        strict: bool | None = None,
    ):
        self.output = type_
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.strict = strict


@dataclass(init=False)
class NativeOutput(Generic[OutputDataT]):
    """Marker class to use the model's native structured outputs functionality for outputs and optionally customize the name and description.

    Example:
    ```python {title="native_output.py" requires="tool_output.py"}
    from pydantic_ai import Agent, NativeOutput

    from tool_output import Fruit, Vehicle

    agent = Agent(
        'openai:gpt-4o',
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

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self.outputs = outputs
        self.name = name
        self.description = description
        self.strict = strict


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
        'openai:gpt-4o',
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
        'openai:gpt-4o',
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
    template: str | None
    """Template for the prompt passed to the model.
    The '{schema}' placeholder will be replaced with the output JSON schema.
    If not specified, the default template specified on the model's profile will be used.
    """

    def __init__(
        self,
        outputs: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        template: str | None = None,
    ):
        self.outputs = outputs
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
class TextOutput(Generic[OutputDataT]):
    """Marker class to use text output for an output function taking a string argument.

    Example:
    ```python
    from pydantic_ai import Agent, TextOutput


    def split_into_words(text: str) -> list[str]:
        return text.split()


    agent = Agent(
        'openai:gpt-4o',
        output_type=TextOutput(split_into_words),
    )
    result = agent.run_sync('Who was Albert Einstein?')
    print(result.output)
    #> ['Albert', 'Einstein', 'was', 'a', 'German-born', 'theoretical', 'physicist.']
    ```
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

    agent = Agent('openai:gpt-4o', output_type=StructuredDict(schema))
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


_OutputSpecItem = TypeAliasType(
    '_OutputSpecItem',
    OutputTypeOrFunction[T_co] | ToolOutput[T_co] | NativeOutput[T_co] | PromptedOutput[T_co] | TextOutput[T_co],
    type_params=(T_co,),
)

OutputSpec = TypeAliasType(
    'OutputSpec',
    _OutputSpecItem[T_co] | Sequence['OutputSpec[T_co]'],
    type_params=(T_co,),
)
"""Specification of the agent's output data.

This can be a single type, a function, a sequence of types and/or functions, or an instance of one of the output mode marker classes:
- [`ToolOutput`][pydantic_ai.output.ToolOutput]
- [`NativeOutput`][pydantic_ai.output.NativeOutput]
- [`PromptedOutput`][pydantic_ai.output.PromptedOutput]
- [`TextOutput`][pydantic_ai.output.TextOutput]

You should not need to import or use this type directly.

See [output docs](../output.md) for more information.
"""


@deprecated('`DeferredToolCalls` is deprecated, use `DeferredToolRequests` instead')
class DeferredToolCalls(DeferredToolRequests):  # pragma: no cover
    @property
    @deprecated('`DeferredToolCalls.tool_calls` is deprecated, use `DeferredToolRequests.calls` instead')
    def tool_calls(self) -> list[ToolCallPart]:
        return self.calls

    @property
    @deprecated('`DeferredToolCalls.tool_defs` is deprecated')
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {}
