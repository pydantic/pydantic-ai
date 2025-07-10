from __future__ import annotations as _annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union, cast, overload

from pydantic import TypeAdapter, ValidationError
from pydantic_core import SchemaValidator
from typing_extensions import TypedDict, TypeVar, assert_never

from . import _function_schema, _utils, messages as _messages
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, UserError
from .output import (
    NativeOutput,
    OutputDataT,
    OutputMode,
    OutputSpec,
    OutputTypeOrFunction,
    PromptedOutput,
    StructuredOutputMode,
    TextOutput,
    TextOutputFunc,
    ToolOutput,
)
from .tools import GenerateToolJsonSchema, ObjectJsonSchema, ToolDefinition

if TYPE_CHECKING:
    from .profiles import ModelProfile

T = TypeVar('T')
"""An invariant TypeVar."""
OutputDataT_inv = TypeVar('OutputDataT_inv', default=str)
"""
An invariant type variable for the result data of a model.

We need to use an invariant typevar for `OutputValidator` and `OutputValidatorFunc` because the output data type is used
in both the input and output of a `OutputValidatorFunc`. This can theoretically lead to some issues assuming that types
possessing OutputValidator's are covariant in the result data type, but in practice this is rarely an issue, and
changing it would have negative consequences for the ergonomics of the library.

At some point, it may make sense to change the input to OutputValidatorFunc to be `Any` or `object` as doing that would
resolve these potential variance issues.
"""

OutputValidatorFunc = Union[
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], OutputDataT_inv],
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], Awaitable[OutputDataT_inv]],
    Callable[[OutputDataT_inv], OutputDataT_inv],
    Callable[[OutputDataT_inv], Awaitable[OutputDataT_inv]],
]
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `OutputValidatorFunc[AgentDepsT, T]`.
"""


DEFAULT_OUTPUT_TOOL_NAME = 'final_result'
DEFAULT_OUTPUT_TOOL_DESCRIPTION = 'The final response which ends this conversation'


class ToolRetryError(Exception):
    """Exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: _messages.RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__()


@dataclass
class OutputValidator(Generic[AgentDepsT, OutputDataT_inv]):
    function: OutputValidatorFunc[AgentDepsT, OutputDataT_inv]
    _takes_ctx: bool = field(init=False)
    _is_async: bool = field(init=False)

    def __post_init__(self):
        self._takes_ctx = len(inspect.signature(self.function).parameters) > 1
        self._is_async = _utils.is_async_callable(self.function)

    async def validate(
        self,
        result: T,
        tool_call: _messages.ToolCallPart | None,
        run_context: RunContext[AgentDepsT],
    ) -> T:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            tool_call: The original tool call message, `None` if there was no tool call.
            run_context: The current run context.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            ctx = run_context.replace_with(tool_name=tool_call.tool_name if tool_call else None)
            args = ctx, result
        else:
            args = (result,)

        try:
            if self._is_async:
                function = cast(Callable[[Any], Awaitable[T]], self.function)
                result_data = await function(*args)
            else:
                function = cast(Callable[[Any], T], self.function)
                result_data = await _utils.run_in_executor(function, *args)
        except ModelRetry as r:
            m = _messages.RetryPromptPart(content=r.message)
            if tool_call is not None:
                m.tool_name = tool_call.tool_name
                m.tool_call_id = tool_call.tool_call_id
            raise ToolRetryError(m) from r
        else:
            return result_data


class BaseOutputSchema(ABC, Generic[OutputDataT]):
    @abstractmethod
    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        raise NotImplementedError()

    @property
    def tools(self) -> dict[str, OutputTool[OutputDataT]]:
        """Get the tools for this output schema."""
        return {}


@dataclass(init=False)
class OutputSchema(BaseOutputSchema[OutputDataT], ABC):
    """Model the final output from an agent run."""

    @classmethod
    @overload
    def build(
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: StructuredOutputMode,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputSchema[OutputDataT]: ...

    @classmethod
    @overload
    def build(
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: None = None,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> BaseOutputSchema[OutputDataT]: ...

    @classmethod
    def build(
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: StructuredOutputMode | None = None,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> BaseOutputSchema[OutputDataT]:
        """Build an OutputSchema dataclass from an output type."""
        if output_spec is str:
            return PlainTextOutputSchema()

        if isinstance(output_spec, NativeOutput):
            return NativeOutputSchema(
                cls._build_processor(
                    _flatten_output_spec(output_spec.outputs),
                    name=output_spec.name,
                    description=output_spec.description,
                    strict=output_spec.strict,
                )
            )
        elif isinstance(output_spec, PromptedOutput):
            return PromptedOutputSchema(
                cls._build_processor(
                    _flatten_output_spec(output_spec.outputs),
                    name=output_spec.name,
                    description=output_spec.description,
                ),
                template=output_spec.template,
            )

        text_outputs: Sequence[type[str] | TextOutput[OutputDataT]] = []
        tool_outputs: Sequence[ToolOutput[OutputDataT]] = []
        other_outputs: Sequence[OutputTypeOrFunction[OutputDataT]] = []
        for output in _flatten_output_spec(output_spec):
            if output is str:
                text_outputs.append(cast(type[str], output))
            elif isinstance(output, TextOutput):
                text_outputs.append(output)
            elif isinstance(output, ToolOutput):
                tool_outputs.append(output)
            else:
                other_outputs.append(output)

        tools = cls._build_tools(tool_outputs + other_outputs, name=name, description=description, strict=strict)

        if len(text_outputs) > 0:
            if len(text_outputs) > 1:
                raise UserError('Only one text output is allowed.')
            text_output = text_outputs[0]

            text_output_schema = None
            if isinstance(text_output, TextOutput):
                text_output_schema = PlainTextOutputProcessor(text_output.output_function)

            if len(tools) == 0:
                return PlainTextOutputSchema(text_output_schema)
            else:
                return ToolOrTextOutputSchema(processor=text_output_schema, tools=tools)

        if len(tool_outputs) > 0:
            return ToolOutputSchema(tools)

        if len(other_outputs) > 0:
            schema = OutputSchemaWithoutMode(
                processor=cls._build_processor(other_outputs, name=name, description=description, strict=strict),
                tools=tools,
            )
            if default_mode:
                schema = schema.with_default_mode(default_mode)
            return schema

        raise UserError('No output type provided.')  # pragma: no cover

    @staticmethod
    def _build_tools(
        outputs: list[OutputTypeOrFunction[OutputDataT] | ToolOutput[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> dict[str, OutputTool[OutputDataT]]:
        tools: dict[str, OutputTool[OutputDataT]] = {}

        default_name = name or DEFAULT_OUTPUT_TOOL_NAME
        default_description = description
        default_strict = strict

        multiple = len(outputs) > 1
        for output in outputs:
            name = None
            description = None
            strict = None
            if isinstance(output, ToolOutput):
                # do we need to error on conflicts here? (DavidM): If this is internal maybe doesn't matter, if public, use overloads
                name = output.name
                description = output.description
                strict = output.strict

                output = output.output

            description = description or default_description
            if strict is None:
                strict = default_strict

            processor = ObjectOutputProcessor(output=output, description=description, strict=strict)

            if name is None:
                name = default_name
                if multiple:
                    name += f'_{processor.object_def.name}'

            i = 1
            original_name = name
            while name in tools:
                i += 1
                name = f'{original_name}_{i}'

            tools[name] = OutputTool(name=name, processor=processor, multiple=multiple)

        return tools

    @staticmethod
    def _build_processor(
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT]:
        outputs = _flatten_output_spec(outputs)
        if len(outputs) == 1:
            return ObjectOutputProcessor(output=outputs[0], name=name, description=description, strict=strict)

        return UnionOutputProcessor(outputs=outputs, strict=strict, name=name, description=description)

    @property
    @abstractmethod
    def mode(self) -> OutputMode:
        raise NotImplementedError()

    @abstractmethod
    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        raise NotImplementedError()

    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        return self


@dataclass(init=False)
class OutputSchemaWithoutMode(BaseOutputSchema[OutputDataT]):
    processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT]
    _tools: dict[str, OutputTool[OutputDataT]] = field(default_factory=dict)

    def __init__(
        self,
        processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT],
        tools: dict[str, OutputTool[OutputDataT]],
    ):
        self.processor = processor
        self._tools = tools

    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        if mode == 'native':
            return NativeOutputSchema(self.processor)
        elif mode == 'prompted':
            return PromptedOutputSchema(self.processor)
        elif mode == 'tool':
            return ToolOutputSchema(self.tools)
        else:
            assert_never(mode)

    @property
    def tools(self) -> dict[str, OutputTool[OutputDataT]]:
        """Get the tools for this output schema."""
        # We return tools here as they're checked in Agent._register_tool.
        # At that point we may don't know yet what output mode we're going to use if no model was provided or it was deferred until agent.run time.
        return self._tools


class TextOutputSchema(OutputSchema[OutputDataT], ABC):
    @abstractmethod
    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        raise NotImplementedError()


@dataclass
class PlainTextOutputSchema(TextOutputSchema[OutputDataT]):
    processor: PlainTextOutputProcessor[OutputDataT] | None = None

    @property
    def mode(self) -> OutputMode:
        return 'text'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        pass

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        if self.processor is None:
            return cast(OutputDataT, text)

        return await self.processor.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass
class StructuredTextOutputSchema(TextOutputSchema[OutputDataT], ABC):
    processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT]

    @property
    def object_def(self) -> OutputObjectDefinition:
        return self.processor.object_def


@dataclass
class NativeOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    @property
    def mode(self) -> OutputMode:
        return 'native'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        if not profile.supports_json_schema_output:
            raise UserError('Structured output is not supported by the model.')

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        return await self.processor.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass
class PromptedOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    template: str | None = None

    @property
    def mode(self) -> OutputMode:
        return 'prompted'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        pass

    def instructions(self, default_template: str) -> str:
        """Get instructions to tell model to output JSON matching the schema."""
        template = self.template or default_template

        if '{schema}' not in template:
            template = '\n\n'.join([template, '{schema}'])

        object_def = self.object_def
        schema = object_def.json_schema.copy()
        if object_def.name:
            schema['title'] = object_def.name
        if object_def.description:
            schema['description'] = object_def.description

        return template.format(schema=json.dumps(schema))

    async def process(
        self,
        text: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Validate an output message.

        Args:
            text: The output text to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        text = _utils.strip_markdown_fences(text)

        return await self.processor.process(
            text, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class ToolOutputSchema(OutputSchema[OutputDataT]):
    _tools: dict[str, OutputTool[OutputDataT]] = field(default_factory=dict)

    def __init__(self, tools: dict[str, OutputTool[OutputDataT]]):
        self._tools = tools

    @property
    def mode(self) -> OutputMode:
        return 'tool'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        if not profile.supports_tools:
            raise UserError('Output tools are not supported by the model.')

    @property
    def tools(self) -> dict[str, OutputTool[OutputDataT]]:
        """Get the tools for this output schema."""
        return self._tools

    def tool_names(self) -> list[str]:
        """Return the names of the tools."""
        return list(self.tools.keys())

    def tool_defs(self) -> list[ToolDefinition]:
        """Get tool definitions to register with the model."""
        return [t.tool_def for t in self.tools.values()]

    def find_named_tool(
        self, parts: Iterable[_messages.ModelResponsePart], tool_name: str
    ) -> tuple[_messages.ToolCallPart, OutputTool[OutputDataT]] | None:
        """Find a tool that matches one of the calls, with a specific name."""
        for part in parts:  # pragma: no branch
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if part.tool_name == tool_name:
                    return part, self.tools[tool_name]

    def find_tool(
        self,
        parts: Iterable[_messages.ModelResponsePart],
    ) -> Iterator[tuple[_messages.ToolCallPart, OutputTool[OutputDataT]]]:
        """Find a tool that matches one of the calls."""
        for part in parts:
            if isinstance(part, _messages.ToolCallPart):  # pragma: no branch
                if result := self.tools.get(part.tool_name):
                    yield part, result


@dataclass(init=False)
class ToolOrTextOutputSchema(ToolOutputSchema[OutputDataT], PlainTextOutputSchema[OutputDataT]):
    def __init__(
        self,
        processor: PlainTextOutputProcessor[OutputDataT] | None,
        tools: dict[str, OutputTool[OutputDataT]],
    ):
        self.processor = processor
        self._tools = tools

    @property
    def mode(self) -> OutputMode:
        return 'tool_or_text'


@dataclass
class OutputObjectDefinition:
    json_schema: ObjectJsonSchema
    name: str | None = None
    description: str | None = None
    strict: bool | None = None


@dataclass(init=False)
class BaseOutputProcessor(ABC, Generic[OutputDataT]):
    @abstractmethod
    async def process(
        self,
        data: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function."""
        raise NotImplementedError()


@dataclass(init=False)
class ObjectOutputProcessor(BaseOutputProcessor[OutputDataT]):
    object_def: OutputObjectDefinition
    outer_typed_dict_key: str | None = None
    _validator: SchemaValidator
    _function_schema: _function_schema.FunctionSchema | None = None

    def __init__(
        self,
        output: OutputTypeOrFunction[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        if inspect.isfunction(output) or inspect.ismethod(output):
            self._function_schema = _function_schema.function_schema(output, GenerateToolJsonSchema)
            self._validator = self._function_schema.validator
            json_schema = self._function_schema.json_schema
            json_schema['description'] = self._function_schema.description
        else:
            type_adapter: TypeAdapter[Any]
            if _utils.is_model_like(output):
                type_adapter = TypeAdapter(output)
            else:
                self.outer_typed_dict_key = 'response'
                response_data_typed_dict = TypedDict(  # noqa: UP013
                    'response_data_typed_dict',
                    {'response': cast(type[OutputDataT], output)},  # pyright: ignore[reportInvalidTypeForm]
                )
                type_adapter = TypeAdapter(response_data_typed_dict)

            # Really a PluggableSchemaValidator, but it's API-compatible
            self._validator = cast(SchemaValidator, type_adapter.validator)
            json_schema = _utils.check_object_json_schema(
                type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
            )

            if self.outer_typed_dict_key:
                # including `response_data_typed_dict` as a title here doesn't add anything and could confuse the LLM
                json_schema.pop('title')

        if name is None and (json_schema_title := json_schema.get('title', None)):
            name = json_schema_title

        if json_schema_description := json_schema.pop('description', None):
            if description is None:
                description = json_schema_description
            else:
                description = f'{description}. {json_schema_description}'

        self.object_def = OutputObjectDefinition(
            name=name or getattr(output, '__name__', None),
            description=description,
            json_schema=json_schema,
            strict=strict,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message, performing validation and (if necessary) calling the output function.

        Args:
            data: The output data to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
            if isinstance(data, str):
                output = self._validator.validate_json(data or '{}', allow_partial=pyd_allow_partial)
            else:
                output = self._validator.validate_python(data or {}, allow_partial=pyd_allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=e.errors(include_url=False),
                )
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover

        if k := self.outer_typed_dict_key:
            output = output[k]

        if self._function_schema:
            try:
                output = await self._function_schema.call(output, run_context)
            except ModelRetry as r:
                if wrap_validation_errors:
                    m = _messages.RetryPromptPart(
                        content=r.message,
                    )
                    raise ToolRetryError(m) from r
                else:
                    raise  # pragma: lax no cover

        return output


@dataclass
class UnionOutputResult:
    kind: str
    data: ObjectJsonSchema


@dataclass
class UnionOutputModel:
    result: UnionOutputResult


@dataclass(init=False)
class UnionOutputProcessor(BaseOutputProcessor[OutputDataT]):
    object_def: OutputObjectDefinition
    _union_processor: ObjectOutputProcessor[UnionOutputModel]
    _processors: dict[str, ObjectOutputProcessor[OutputDataT]]

    def __init__(
        self,
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ):
        self._union_processor = ObjectOutputProcessor(output=UnionOutputModel)

        json_schemas: list[ObjectJsonSchema] = []
        self._processors = {}
        for output in outputs:
            processor = ObjectOutputProcessor(output=output, strict=strict)
            object_def = processor.object_def

            object_key = object_def.name or output.__name__
            i = 1
            original_key = object_key
            while object_key in self._processors:
                i += 1
                object_key = f'{original_key}_{i}'

            self._processors[object_key] = processor

            json_schema = object_def.json_schema
            if object_def.name:  # pragma: no branch
                json_schema['title'] = object_def.name
            if object_def.description:
                json_schema['description'] = object_def.description

            json_schemas.append(json_schema)

        json_schemas, all_defs = _utils.merge_json_schema_defs(json_schemas)

        discriminated_json_schemas: list[ObjectJsonSchema] = []
        for object_key, json_schema in zip(self._processors.keys(), json_schemas):
            title = json_schema.pop('title', None)
            description = json_schema.pop('description', None)

            discriminated_json_schema = {
                'type': 'object',
                'properties': {
                    'kind': {
                        'type': 'string',
                        'const': object_key,
                    },
                    'data': json_schema,
                },
                'required': ['kind', 'data'],
                'additionalProperties': False,
            }
            if title:  # pragma: no branch
                discriminated_json_schema['title'] = title
            if description:
                discriminated_json_schema['description'] = description

            discriminated_json_schemas.append(discriminated_json_schema)

        json_schema = {
            'type': 'object',
            'properties': {
                'result': {
                    'anyOf': discriminated_json_schemas,
                }
            },
            'required': ['result'],
            'additionalProperties': False,
        }
        if all_defs:
            json_schema['$defs'] = all_defs

        self.object_def = OutputObjectDefinition(
            json_schema=json_schema,
            strict=strict,
            name=name,
            description=description,
        )

    async def process(
        self,
        data: str | dict[str, Any] | None,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        union_object = await self._union_processor.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )

        result = union_object.result
        kind = result.kind
        data = result.data
        try:
            processor = self._processors[kind]
        except KeyError as e:  # pragma: no cover
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(content=f'Invalid kind: {kind}')
                raise ToolRetryError(m) from e
            else:
                raise

        return await processor.process(
            data, run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
        )


@dataclass(init=False)
class PlainTextOutputProcessor(BaseOutputProcessor[OutputDataT]):
    _function_schema: _function_schema.FunctionSchema
    _str_argument_name: str

    def __init__(
        self,
        output_function: TextOutputFunc[OutputDataT],
    ):
        self._function_schema = _function_schema.function_schema(output_function, GenerateToolJsonSchema)

        arguments_schema = self._function_schema.json_schema.get('properties', {})
        argument_name = next(iter(arguments_schema.keys()), None)
        if argument_name and arguments_schema.get(argument_name, {}).get('type') == 'string':
            self._str_argument_name = argument_name
            return

        raise UserError('TextOutput must take a function taking a `str`')

    @property
    def object_def(self) -> None:
        return None  # pragma: no cover

    async def process(
        self,
        data: str,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        args = {self._str_argument_name: data}

        try:
            output = await self._function_schema.call(args, run_context)
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=r.message,
                )
                raise ToolRetryError(m) from r
            else:
                raise  # pragma: lax no cover

        return cast(OutputDataT, output)


@dataclass(init=False)
class OutputTool(Generic[OutputDataT]):
    processor: ObjectOutputProcessor[OutputDataT]
    tool_def: ToolDefinition

    def __init__(self, *, name: str, processor: ObjectOutputProcessor[OutputDataT], multiple: bool):
        self.processor = processor
        object_def = processor.object_def

        description = object_def.description
        if not description:
            description = DEFAULT_OUTPUT_TOOL_DESCRIPTION
            if multiple:
                description = f'{object_def.name}: {description}'

        self.tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters_json_schema=object_def.json_schema,
            strict=object_def.strict,
            outer_typed_dict_key=processor.outer_typed_dict_key,
        )

    async def process(
        self,
        tool_call: _messages.ToolCallPart,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> OutputDataT:
        """Process an output message.

        Args:
            tool_call: The tool call from the LLM to validate.
            run_context: The current run context.
            allow_partial: If true, allow partial validation.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Either the validated output data (left) or a retry message (right).
        """
        try:
            output = await self.processor.process(
                tool_call.args, run_context, allow_partial=allow_partial, wrap_validation_errors=False
            )
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=e.errors(include_url=False, include_context=False),
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from e
            else:
                raise  # pragma: lax no cover
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    tool_name=tool_call.tool_name,
                    content=r.message,
                    tool_call_id=tool_call.tool_call_id,
                )
                raise ToolRetryError(m) from r
            else:
                raise  # pragma: lax no cover
        else:
            return output


def _flatten_output_spec(output_spec: T | Sequence[T]) -> list[T]:
    outputs: Sequence[T]
    if isinstance(output_spec, Sequence):
        outputs = output_spec
    else:
        outputs = (output_spec,)

    outputs_flat: list[T] = []
    for output in outputs:
        if union_types := _utils.get_union_args(output):
            outputs_flat.extend(union_types)
        else:
            outputs_flat.append(output)
    return outputs_flat
