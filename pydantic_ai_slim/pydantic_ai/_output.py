from __future__ import annotations as _annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Union, cast, overload

from pydantic import TypeAdapter, ValidationError
from pydantic_core import SchemaValidator, to_json
from typing_extensions import Self, TypedDict, TypeVar, assert_never

from . import _function_schema, _utils, messages as _messages
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UserError
from .output import (
    DeferredToolCalls,
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
    _OutputSpecItem,  # type: ignore[reportPrivateUsage]
)
from .tools import GenerateToolJsonSchema, ObjectJsonSchema, ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool

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


async def execute_traced_output_function(
    function_schema: _function_schema.FunctionSchema,
    run_context: RunContext[AgentDepsT],
    args: dict[str, Any] | Any,
    wrap_validation_errors: bool = True,
) -> Any:
    """Execute an output function within a traced span with error handling.

    This function executes the output function within an OpenTelemetry span for observability,
    automatically records the function response, and handles ModelRetry exceptions by converting
    them to ToolRetryError when wrap_validation_errors is True.

    Args:
        function_schema: The function schema containing the function to execute
        run_context: The current run context containing tracing and tool information
        args: Arguments to pass to the function
        wrap_validation_errors: If True, wrap ModelRetry exceptions in ToolRetryError

    Returns:
        The result of the function execution

    Raises:
        ToolRetryError: When wrap_validation_errors is True and a ModelRetry is caught
        ModelRetry: When wrap_validation_errors is False and a ModelRetry occurs
    """
    # Set up span attributes
    tool_name = run_context.tool_name or getattr(function_schema.function, '__name__', 'output_function')
    attributes = {
        'gen_ai.tool.name': tool_name,
        'logfire.msg': f'running output function: {tool_name}',
    }
    if run_context.tool_call_id:
        attributes['gen_ai.tool.call.id'] = run_context.tool_call_id
    if run_context.trace_include_content:
        attributes['tool_arguments'] = to_json(args).decode()
        attributes['logfire.json_schema'] = json.dumps(
            {
                'type': 'object',
                'properties': {
                    'tool_arguments': {'type': 'object'},
                    'tool_response': {'type': 'object'},
                },
            }
        )

    with run_context.tracer.start_as_current_span('running output function', attributes=attributes) as span:
        try:
            output = await function_schema.call(args, run_context)
        except ModelRetry as r:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=r.message,
                    tool_name=run_context.tool_name,
                )
                if run_context.tool_call_id:
                    m.tool_call_id = run_context.tool_call_id  # pragma: no cover
                raise ToolRetryError(m) from r
            else:
                raise

        # Record response if content inclusion is enabled
        if run_context.trace_include_content and span.is_recording():
            from .models.instrumented import InstrumentedModel

            span.set_attribute(
                'tool_response',
                output if isinstance(output, str) else json.dumps(InstrumentedModel.serialize_any(output)),
            )

        return output


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
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> T:
        """Validate a result but calling the function.

        Args:
            result: The result data after Pydantic validation the message content.
            run_context: The current run context.
            wrap_validation_errors: If true, wrap the validation errors in a retry message.

        Returns:
            Result of either the validated result data (ok) or a retry message (Err).
        """
        if self._takes_ctx:
            args = run_context, result
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
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=r.message,
                    tool_name=run_context.tool_name,
                )
                if run_context.tool_call_id:  # pragma: no cover
                    m.tool_call_id = run_context.tool_call_id
                raise ToolRetryError(m) from r
            else:
                raise r
        else:
            return result_data


@dataclass
class BaseOutputSchema(ABC, Generic[OutputDataT]):
    allows_deferred_tool_calls: bool

    @abstractmethod
    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        raise NotImplementedError()

    @property
    def toolset(self) -> OutputToolset[Any] | None:
        """Get the toolset for this output schema."""
        return None


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
    def build(  # noqa: C901
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        default_mode: StructuredOutputMode | None = None,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> BaseOutputSchema[OutputDataT]:
        """Build an OutputSchema dataclass from an output type."""
        raw_outputs = _flatten_output_spec(output_spec)

        outputs = [output for output in raw_outputs if output is not DeferredToolCalls]
        allows_deferred_tool_calls = len(outputs) < len(raw_outputs)
        if len(outputs) == 0 and allows_deferred_tool_calls:
            raise UserError('At least one output type must be provided other than `DeferredToolCalls`.')

        if output := next((output for output in outputs if isinstance(output, NativeOutput)), None):
            if len(outputs) > 1:
                raise UserError('`NativeOutput` must be the only output type.')  # pragma: no cover

            return NativeOutputSchema(
                processor=cls._build_processor(
                    _flatten_output_spec(output.outputs),
                    name=output.name,
                    description=output.description,
                    strict=output.strict,
                ),
                allows_deferred_tool_calls=allows_deferred_tool_calls,
            )
        elif output := next((output for output in outputs if isinstance(output, PromptedOutput)), None):
            if len(outputs) > 1:
                raise UserError('`PromptedOutput` must be the only output type.')  # pragma: no cover

            return PromptedOutputSchema(
                processor=cls._build_processor(
                    _flatten_output_spec(output.outputs),
                    name=output.name,
                    description=output.description,
                ),
                template=output.template,
                allows_deferred_tool_calls=allows_deferred_tool_calls,
            )

        text_outputs: Sequence[type[str] | TextOutput[OutputDataT]] = []
        tool_outputs: Sequence[ToolOutput[OutputDataT]] = []
        other_outputs: Sequence[OutputTypeOrFunction[OutputDataT]] = []
        for output in outputs:
            if output is str:
                text_outputs.append(cast(type[str], output))
            elif isinstance(output, TextOutput):
                text_outputs.append(output)
            elif isinstance(output, ToolOutput):
                tool_outputs.append(output)
            elif isinstance(output, NativeOutput):
                # We can never get here because this is checked for above.
                raise UserError('`NativeOutput` must be the only output type.')  # pragma: no cover
            elif isinstance(output, PromptedOutput):
                # We can never get here because this is checked for above.
                raise UserError('`PromptedOutput` must be the only output type.')  # pragma: no cover
            else:
                other_outputs.append(output)

        toolset = OutputToolset.build(tool_outputs + other_outputs, name=name, description=description, strict=strict)

        if len(text_outputs) > 0:
            if len(text_outputs) > 1:
                raise UserError('Only one `str` or `TextOutput` is allowed.')
            text_output = text_outputs[0]

            text_output_schema = None
            if isinstance(text_output, TextOutput):
                text_output_schema = PlainTextOutputProcessor(text_output.output_function)

            if toolset:
                return ToolOrTextOutputSchema(
                    processor=text_output_schema, toolset=toolset, allows_deferred_tool_calls=allows_deferred_tool_calls
                )
            else:
                return PlainTextOutputSchema(
                    processor=text_output_schema, allows_deferred_tool_calls=allows_deferred_tool_calls
                )

        if len(tool_outputs) > 0:
            return ToolOutputSchema(toolset=toolset, allows_deferred_tool_calls=allows_deferred_tool_calls)

        if len(other_outputs) > 0:
            schema = OutputSchemaWithoutMode(
                processor=cls._build_processor(other_outputs, name=name, description=description, strict=strict),
                toolset=toolset,
                allows_deferred_tool_calls=allows_deferred_tool_calls,
            )
            if default_mode:
                schema = schema.with_default_mode(default_mode)
            return schema

        raise UserError('At least one output type must be provided.')

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
    _toolset: OutputToolset[Any] | None

    def __init__(
        self,
        processor: ObjectOutputProcessor[OutputDataT] | UnionOutputProcessor[OutputDataT],
        toolset: OutputToolset[Any] | None,
        allows_deferred_tool_calls: bool,
    ):
        super().__init__(allows_deferred_tool_calls)
        self.processor = processor
        self._toolset = toolset

    def with_default_mode(self, mode: StructuredOutputMode) -> OutputSchema[OutputDataT]:
        if mode == 'native':
            return NativeOutputSchema(
                processor=self.processor, allows_deferred_tool_calls=self.allows_deferred_tool_calls
            )
        elif mode == 'prompted':
            return PromptedOutputSchema(
                processor=self.processor, allows_deferred_tool_calls=self.allows_deferred_tool_calls
            )
        elif mode == 'tool':
            return ToolOutputSchema(toolset=self.toolset, allows_deferred_tool_calls=self.allows_deferred_tool_calls)
        else:
            assert_never(mode)

    @property
    def toolset(self) -> OutputToolset[Any] | None:
        """Get the toolset for this output schema."""
        # We return a toolset here as they're checked for name conflicts with other toolsets in the Agent constructor.
        # At that point we may not know yet what output mode we're going to use if no model was provided or it was deferred until agent.run time,
        # but we cover ourselves just in case we end up using the tool output mode.
        return self._toolset


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
            raise UserError('Native structured output is not supported by the model.')

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
    _toolset: OutputToolset[Any] | None

    def __init__(self, toolset: OutputToolset[Any] | None, allows_deferred_tool_calls: bool):
        super().__init__(allows_deferred_tool_calls)
        self._toolset = toolset

    @property
    def mode(self) -> OutputMode:
        return 'tool'

    def raise_if_unsupported(self, profile: ModelProfile) -> None:
        """Raise an error if the mode is not supported by the model."""
        if not profile.supports_tools:
            raise UserError('Output tools are not supported by the model.')

    @property
    def toolset(self) -> OutputToolset[Any] | None:
        """Get the toolset for this output schema."""
        return self._toolset


@dataclass(init=False)
class ToolOrTextOutputSchema(ToolOutputSchema[OutputDataT], PlainTextOutputSchema[OutputDataT]):
    def __init__(
        self,
        processor: PlainTextOutputProcessor[OutputDataT] | None,
        toolset: OutputToolset[Any] | None,
        allows_deferred_tool_calls: bool,
    ):
        super().__init__(toolset=toolset, allows_deferred_tool_calls=allows_deferred_tool_calls)
        self.processor = processor

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
    validator: SchemaValidator
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
            self.validator = self._function_schema.validator
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
            self.validator = cast(SchemaValidator, type_adapter.validator)
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
            output = self.validate(data, allow_partial)
        except ValidationError as e:
            if wrap_validation_errors:
                m = _messages.RetryPromptPart(
                    content=e.errors(include_url=False),
                )
                raise ToolRetryError(m) from e
            else:
                raise

        output = await self.call(output, run_context, wrap_validation_errors)

        return output

    def validate(
        self,
        data: str | dict[str, Any] | None,
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        if isinstance(data, str):
            return self.validator.validate_json(data or '{}', allow_partial=pyd_allow_partial)
        else:
            return self.validator.validate_python(data or {}, allow_partial=pyd_allow_partial)

    async def call(
        self,
        output: Any,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ):
        if k := self.outer_typed_dict_key:
            output = output[k]

        if self._function_schema:
            output = await execute_traced_output_function(
                self._function_schema, run_context, output, wrap_validation_errors
            )

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
        output = await execute_traced_output_function(self._function_schema, run_context, args, wrap_validation_errors)

        return cast(OutputDataT, output)


@dataclass(init=False)
class OutputToolset(AbstractToolset[AgentDepsT]):
    """A toolset that contains contains output tools for agent output types."""

    _tool_defs: list[ToolDefinition]
    """The tool definitions for the output tools in this toolset."""
    processors: dict[str, ObjectOutputProcessor[Any]]
    """The processors for the output tools in this toolset."""
    max_retries: int
    output_validators: list[OutputValidator[AgentDepsT, Any]]

    @classmethod
    def build(
        cls,
        outputs: list[OutputTypeOrFunction[OutputDataT] | ToolOutput[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> Self | None:
        if len(outputs) == 0:
            return None

        processors: dict[str, ObjectOutputProcessor[Any]] = {}
        tool_defs: list[ToolDefinition] = []

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
            object_def = processor.object_def

            if name is None:
                name = default_name
                if multiple:
                    name += f'_{object_def.name}'

            i = 1
            original_name = name
            while name in processors:
                i += 1
                name = f'{original_name}_{i}'

            description = object_def.description
            if not description:
                description = DEFAULT_OUTPUT_TOOL_DESCRIPTION
                if multiple:
                    description = f'{object_def.name}: {description}'

            tool_def = ToolDefinition(
                name=name,
                description=description,
                parameters_json_schema=object_def.json_schema,
                strict=object_def.strict,
                outer_typed_dict_key=processor.outer_typed_dict_key,
                kind='output',
            )
            processors[name] = processor
            tool_defs.append(tool_def)

        return cls(processors=processors, tool_defs=tool_defs)

    def __init__(
        self,
        tool_defs: list[ToolDefinition],
        processors: dict[str, ObjectOutputProcessor[Any]],
        max_retries: int = 1,
        output_validators: list[OutputValidator[AgentDepsT, Any]] | None = None,
    ):
        self.processors = processors
        self._tool_defs = tool_defs
        self.max_retries = max_retries
        self.output_validators = output_validators or []

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self.max_retries,
                args_validator=self.processors[tool_def.name].validator,
            )
            for tool_def in self._tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        output = await self.processors[name].call(tool_args, ctx, wrap_validation_errors=False)
        for validator in self.output_validators:
            output = await validator.validate(output, ctx, wrap_validation_errors=False)
        return output


@overload
def _flatten_output_spec(
    output_spec: OutputTypeOrFunction[T] | Sequence[OutputTypeOrFunction[T]],
) -> Sequence[OutputTypeOrFunction[T]]: ...


@overload
def _flatten_output_spec(output_spec: OutputSpec[T]) -> Sequence[_OutputSpecItem[T]]: ...


def _flatten_output_spec(output_spec: OutputSpec[T]) -> Sequence[_OutputSpecItem[T]]:
    outputs: Sequence[OutputSpec[T]]
    if isinstance(output_spec, Sequence):
        outputs = output_spec
    else:
        outputs = (output_spec,)

    outputs_flat: list[_OutputSpecItem[T]] = []
    for output in outputs:
        if isinstance(output, Sequence):
            outputs_flat.extend(_flatten_output_spec(cast(OutputSpec[T], output)))
        elif union_types := _utils.get_union_args(output):
            outputs_flat.extend(union_types)
        else:
            outputs_flat.append(cast(_OutputSpecItem[T], output))
    return outputs_flat
