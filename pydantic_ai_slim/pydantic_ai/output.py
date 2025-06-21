from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Generic, Literal, Union

from typing_extensions import TypeAliasType, TypeVar, get_args, get_origin
from typing_inspection import typing_objects
from typing_inspection.introspection import is_union_origin

from .messages import RetryPromptPart, ToolCallPart
from .tools import RunContext, ToolDefinition

OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
"""Covariant type variable for the result data type of a run."""

T = TypeVar('T')

OutputMode = Literal['text', 'tool', 'model_structured', 'prompted_structured', 'tool_or_text']
"""All output modes."""
StructuredOutputMode = Literal['tool', 'model_structured', 'prompted_structured']
"""Output modes that can be used for structured output. Used by ModelProfile.default_structured_output_mode"""


class ToolRetryError(Exception):
    """Internal exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__()


@dataclass(init=False)
class ToolOutput(Generic[OutputDataT]):
    """Marker class to use a tool for output and optionally customize the tool.

    Example:
    ```python
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
    name: str | None
    description: str | None
    max_retries: int | None
    strict: bool | None

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
class ModelStructuredOutput(Generic[OutputDataT]):
    """Marker class to use the model's built-in structured outputs functionality for outputs and optionally customize the name and description.

    Example:
    ```python
    from pydantic import BaseModel

    from pydantic_ai import Agent, ModelStructuredOutput


    class Fruit(BaseModel):
        name: str
        color: str


    class Vehicle(BaseModel):
        name: str
        wheels: int


    agent = Agent(
        'openai:gpt-4o',
        output_type=ModelStructuredOutput(
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

    outputs: Sequence[OutputTypeOrFunction[OutputDataT]]
    name: str | None
    description: str | None

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        self.outputs = _flatten_output_spec(type_)
        self.name = name
        self.description = description


@dataclass(init=False)
class PromptedStructuredOutput(Generic[OutputDataT]):
    """Marker class to use a prompt to tell the model what to output and optionally customize the prompt.

    Example:
    ```python
    from pydantic import BaseModel

    from pydantic_ai import Agent, PromptedStructuredOutput


    class Vehicle(BaseModel):
        name: str
        wheels: int


    class Device(BaseModel):
        name: str
        kind: str


    agent = Agent(
        'openai:gpt-4o',
        output_type=PromptedStructuredOutput(
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
        output_type=PromptedStructuredOutput(
            [Vehicle, Device],
            template='Gimme some JSON: {schema}'
        ),
    )
    result = agent.run_sync('What is a Ford Explorer?')
    print(repr(result.output))
    #> Vehicle(name='Ford Explorer', wheels=4)
    ```
    """

    outputs: Sequence[OutputTypeOrFunction[OutputDataT]]
    name: str | None
    description: str | None
    template: str | None
    """Template for the prompt passed to the model.
    The '{schema}' placeholder will be replaced with the output JSON schema.
    If not specified, the default template specified on the model's profile will be used.
    """

    def __init__(
        self,
        type_: OutputTypeOrFunction[OutputDataT] | Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        template: str | None = None,
    ):
        self.outputs = _flatten_output_spec(type_)
        self.name = name
        self.description = description
        self.template = template


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

    output_function: TextOutputFunction[OutputDataT]


@dataclass
class PendingToolCalls:
    """Output type for calls to tools defined as pending."""

    tool_calls: list[ToolCallPart]
    tool_defs: dict[str, ToolDefinition]


def _get_union_args(tp: Any) -> tuple[Any, ...]:
    """Extract the arguments of a Union type if `output_type` is a union, otherwise return an empty tuple."""
    if typing_objects.is_typealiastype(tp):
        tp = tp.__value__

    origin = get_origin(tp)
    if is_union_origin(origin):
        return get_args(tp)
    else:
        return ()


def _flatten_output_spec(output_spec: T | Sequence[T]) -> list[T]:
    outputs: Sequence[T]
    if isinstance(output_spec, Sequence):
        outputs = output_spec
    else:
        outputs = (output_spec,)

    outputs_flat: list[T] = []
    for output in outputs:
        if union_types := _get_union_args(output):
            outputs_flat.extend(union_types)
        else:
            outputs_flat.append(output)
    return outputs_flat


T_co = TypeVar('T_co', covariant=True)

OutputTypeOrFunction = TypeAliasType(
    'OutputTypeOrFunction', Union[type[T_co], Callable[..., Union[Awaitable[T_co], T_co]]], type_params=(T_co,)
)

OutputSpecItem = TypeAliasType(
    'OutputSpecItem',
    Union[
        OutputTypeOrFunction[T_co],
        ToolOutput[T_co],
        ModelStructuredOutput[T_co],
        PromptedStructuredOutput[T_co],
        TextOutput[T_co],
    ],
    type_params=(T_co,),
)

OutputSpec = TypeAliasType(
    'OutputSpec',
    Union[
        OutputSpecItem[T_co],
        Sequence[OutputSpecItem[T_co]],
    ],
    type_params=(T_co,),
)

TextOutputFunction = TypeAliasType(
    'TextOutputFunction',
    Union[
        Callable[[RunContext, str], Union[Awaitable[T_co], T_co]],
        Callable[[str], Union[Awaitable[T_co], T_co]],
    ],
    type_params=(T_co,),
)
