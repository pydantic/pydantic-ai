from __future__ import annotations as _annotations

import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from types import NoneType
from typing import TYPE_CHECKING, Any, Generic, Literal, cast, get_origin, overload

from pydantic import Json, TypeAdapter, ValidationError
from pydantic_core import SchemaValidator, to_json
from typing_extensions import Self, TypedDict, TypeVar

from pydantic_ai._instrumentation import InstrumentationNames, get_agent_run_baggage_attributes
from pydantic_ai._utils import get_function_type_hints

from . import _function_schema, _utils, messages as _messages
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UserError
from .output import (
    DeferredToolRequests,
    NativeOutput,
    OutputContext,
    OutputDataT,
    OutputMode,
    OutputObjectDefinition,
    OutputSpec,
    OutputTypeOrFunction,
    PromptedOutput,
    TextOutput,
    TextOutputFunc,
    ToolOutput,
    _OutputSpecItem,  # type: ignore[reportPrivateUsage]
)
from .tools import GenerateToolJsonSchema, ObjectJsonSchema, ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool

if TYPE_CHECKING:
    from .capabilities.abstract import AbstractCapability, RawOutput

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

OutputValidatorFunc = (
    Callable[[RunContext[AgentDepsT], OutputDataT_inv], OutputDataT_inv]
    | Callable[[RunContext[AgentDepsT], OutputDataT_inv], Awaitable[OutputDataT_inv]]
    | Callable[[OutputDataT_inv], OutputDataT_inv]
    | Callable[[OutputDataT_inv], Awaitable[OutputDataT_inv]]
)
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `OutputValidatorFunc[AgentDepsT, T]`.
"""


DEFAULT_OUTPUT_TOOL_NAME = 'final_result'
DEFAULT_OUTPUT_TOOL_DESCRIPTION = 'The final response which ends this conversation'


def _build_output_handlers(
    processor: BaseOutputProcessor[OutputDataT],
    *,
    run_context: RunContext[AgentDepsT],
    allow_partial: bool,
    wrap_validation_errors: bool,
) -> tuple[
    Callable[[RawOutput], Awaitable[Any]],
    Callable[[Any], Awaitable[Any]],
]:
    """Build validate and process handlers that delegate to `processor.hook_validate`/`hook_execute`.

    Output hooks see the **semantic value** (what the model was asked to produce), not the
    internal dict-wrapped shape used by Pydantic validation. Each processor decides what its
    semantic value looks like via its `hook_validate`/`hook_execute` methods, and opaque
    per-invocation state (e.g. the resolved union member) flows through the closure.
    """
    state: Any = None

    async def do_validate(data: RawOutput) -> Any:
        nonlocal state
        semantic, state = processor.hook_validate(data, run_context=run_context, allow_partial=allow_partial)
        return semantic

    async def do_process(output: Any) -> Any:
        return await processor.hook_execute(
            output, state, run_context=run_context, wrap_validation_errors=wrap_validation_errors
        )

    return do_validate, do_process


def _isinstance_maybe_generic(value: Any, type_: type[Any]) -> bool:
    """`isinstance(value, type_)` that also works for generics like `list[Bar]`.

    `isinstance(x, list[Bar])` raises `TypeError`; we fall back to the generic origin
    (here `list`), so union output resolution still matches the collection type when the
    element type can't be checked at runtime.
    """
    try:
        return isinstance(value, type_)
    except TypeError:
        origin = get_origin(type_)
        return origin is not None and isinstance(value, origin)


def _make_retry_prompt(e: ValidationError | ModelRetry, run_context: RunContext[Any]) -> ToolRetryError:
    if isinstance(e, ValidationError):
        content: list[Any] | str = e.errors(include_url=False, include_context=False)
    else:
        content = e.message
    m = _messages.RetryPromptPart(content=content, tool_name=run_context.tool_name)
    if run_context.tool_call_id:
        m.tool_call_id = run_context.tool_call_id
    return ToolRetryError(m)


async def run_output_validate_hooks(
    capability: AbstractCapability[AgentDepsT],
    *,
    run_context: RunContext[AgentDepsT],
    output_context: OutputContext,
    output: RawOutput,
    do_validate: Callable[[RawOutput], Awaitable[Any]],
    allow_partial: bool = False,
    wrap_validation_errors: bool = True,
) -> Any:
    """Run the output validate hooks around `do_validate`.

    Validate hooks only fire for structured output that needs parsing.

    `ValidationError` and `ModelRetry` from any hook (before, after, wrap, on_error) are
    caught by the outer handler and converted to `ToolRetryError` when
    `wrap_validation_errors` is True. When False (streaming), errors propagate as-is.
    """
    try:
        output = await capability.before_output_validate(run_context, output_context=output_context, output=output)

        try:
            validated = await capability.wrap_output_validate(
                run_context, output_context=output_context, output=output, handler=do_validate
            )
        except (ValidationError, ModelRetry) as e:
            if allow_partial:
                if wrap_validation_errors and isinstance(e, ValidationError):  # pragma: no cover
                    raise _make_retry_prompt(e, run_context) from e
                raise
            try:
                validated = await capability.on_output_validate_error(
                    run_context, output_context=output_context, output=output, error=e
                )
            except (ValidationError, ModelRetry) as hook_error:
                if wrap_validation_errors:
                    raise _make_retry_prompt(hook_error, run_context) from hook_error
                raise

        return await capability.after_output_validate(run_context, output_context=output_context, output=validated)
    except ToolRetryError:
        raise  # Already wrapped, propagate
    except (ValidationError, ModelRetry) as e:
        # ValidationError or ModelRetry from before_output_validate or after_output_validate
        # (e.g. a user hook that does additional Pydantic validation on the validated output)
        if wrap_validation_errors:
            raise _make_retry_prompt(e, run_context) from e
        raise


async def run_output_process_hooks(
    capability: AbstractCapability[AgentDepsT],
    *,
    run_context: RunContext[AgentDepsT],
    output_context: OutputContext,
    output: Any,
    do_process: Callable[[Any], Awaitable[Any]],
    wrap_validation_errors: bool = True,
) -> Any:
    """Run the output process hooks around `do_process`.

    Process hooks fire for all output types (text, structured, image) — in every mode,
    including tool output.

    `ValidationError` and `ModelRetry` from any hook (before, after, wrap, on_error) are caught
    by the outer handler and converted to `ToolRetryError` when `wrap_validation_errors` is True.
    When False (streaming), errors propagate as-is.
    """
    try:
        output = await capability.before_output_process(run_context, output_context=output_context, output=output)

        try:
            result = await capability.wrap_output_process(
                run_context, output_context=output_context, output=output, handler=do_process
            )
        except ToolRetryError:
            raise  # Control flow, not error
        except ModelRetry:
            raise  # Propagate to outer handler, skip on_output_process_error
        except Exception as e:
            # If the error hook itself raises ValidationError/ModelRetry, it propagates out
            # to the outer handler below, where it's wrapped as ToolRetryError if needed.
            result = await capability.on_output_process_error(
                run_context, output_context=output_context, output=output, error=e
            )

        return await capability.after_output_process(run_context, output_context=output_context, output=result)
    except ToolRetryError:
        raise  # Already wrapped, propagate
    except (ValidationError, ModelRetry) as e:
        # ValidationError or ModelRetry from before_output_process, after_output_process, or
        # on_output_process_error (e.g. a user hook doing additional Pydantic validation).
        if wrap_validation_errors:
            raise _make_retry_prompt(e, run_context) from e
        raise


async def run_none_process_hooks(
    *,
    capability: AbstractCapability[AgentDepsT],
    run_context: RunContext[AgentDepsT],
    schema: OutputSchema[Any],
    wrap_validation_errors: bool = True,
    output_validators: Sequence[OutputValidator[AgentDepsT, Any]] = (),
) -> Any:
    """Run output process hooks for a `None` result (empty model response with `allows_none`).

    Output validators run inside process hooks, matching the text/structured/image paths.
    """
    output_context = OutputContext(
        mode='text',
        output_type=type(None),
        object_def=None,
        has_function=False,
        allows_text=schema.allows_text,
        allows_image=schema.allows_image,
        allows_deferred_tools=schema.allows_deferred_tools,
    )

    async def do_process(output: Any) -> Any:
        result = output
        for validator in output_validators:
            result = await validator.validate(result, run_context)
        return result

    return await run_output_process_hooks(
        capability,
        run_context=run_context,
        output_context=output_context,
        output=None,
        do_process=do_process,
        wrap_validation_errors=wrap_validation_errors,
    )


async def run_image_process_hooks(
    image: _messages.BinaryImage,
    *,
    capability: AbstractCapability[AgentDepsT],
    run_context: RunContext[AgentDepsT],
    schema: OutputSchema[Any],
    wrap_validation_errors: bool = True,
    output_validators: Sequence[OutputValidator[AgentDepsT, Any]] = (),
) -> Any:
    """Run output process hooks for image output (no validate hooks — nothing to parse).

    Output validators run inside process hooks, consistent with text/structured output.
    """
    output_context = OutputContext(
        mode='image',
        output_type=_messages.BinaryImage,
        object_def=None,
        has_function=False,
        allows_text=schema.allows_text,
        allows_image=True,
        allows_deferred_tools=schema.allows_deferred_tools,
    )

    async def do_process(output: Any) -> Any:
        result = output
        for validator in output_validators:
            result = await validator.validate(result, run_context)
        return result

    return await run_output_process_hooks(
        capability,
        run_context=run_context,
        output_context=output_context,
        output=image,
        do_process=do_process,
        wrap_validation_errors=wrap_validation_errors,
    )


async def run_output_with_hooks(
    processor: BaseOutputProcessor[OutputDataT],
    *,
    text: str,
    run_context: RunContext[AgentDepsT],
    capability: AbstractCapability[AgentDepsT],
    schema: OutputSchema[Any],
    allow_partial: bool = False,
    wrap_validation_errors: bool = True,
    output_validators: Sequence[OutputValidator[AgentDepsT, Any]] = (),
) -> OutputDataT:
    """Process output text through the processor with capability output hooks.

    Validate hooks only fire for structured output (BaseObjectOutputProcessor) where
    real parsing occurs. Process hooks fire for all output types.

    Output validators (`@agent.output_validator`) run inside process hooks, ensuring
    `wrap_output_process` wraps the complete output pipeline.
    """
    output_context = processor.get_output_context(schema)
    do_validate, base_do_process = _build_output_handlers(
        processor, run_context=run_context, allow_partial=allow_partial, wrap_validation_errors=wrap_validation_errors
    )

    # Wrap output validators into do_process so they run inside wrap_output_process.
    # Validators use wrap_validation_errors=False — the outer run_output_process_hooks
    # handles wrapping ModelRetry as ToolRetryError when appropriate.
    async def do_process(output: Any) -> Any:
        result = await base_do_process(output)
        for validator in output_validators:
            result = await validator.validate(result, run_context)
        return result

    if isinstance(processor, BaseObjectOutputProcessor):
        # Structured output: fire validate hooks (real parsing) then process hooks
        validated = await run_output_validate_hooks(
            capability,
            run_context=run_context,
            output_context=output_context,
            output=text,
            do_validate=do_validate,
            allow_partial=allow_partial,
            wrap_validation_errors=wrap_validation_errors,
        )
    else:
        # Text output: no real validation, just pass through the text
        validated = await do_validate(text)

    result = await run_output_process_hooks(
        capability,
        run_context=run_context,
        output_context=output_context,
        output=validated,
        do_process=do_process,
        wrap_validation_errors=wrap_validation_errors,
    )

    return cast(OutputDataT, result)


async def execute_traced_output_function(
    function_schema: _function_schema.FunctionSchema,
    *,
    run_context: RunContext[AgentDepsT],
    args: dict[str, Any],
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
    instrumentation_names = InstrumentationNames.for_version(run_context.instrumentation_version)
    # Set up span attributes
    tool_name = run_context.tool_name or getattr(function_schema.function, '__name__', 'output_function')
    attributes = {
        'gen_ai.operation.name': 'execute_tool',
        'gen_ai.tool.name': tool_name,
        **get_agent_run_baggage_attributes(),
        'logfire.msg': f'running output function: {tool_name}',
    }
    if run_context.tool_call_id:
        attributes['gen_ai.tool.call.id'] = run_context.tool_call_id
    if run_context.trace_include_content:
        attributes[instrumentation_names.tool_arguments_attr] = to_json(args).decode()

    attributes['logfire.json_schema'] = json.dumps(
        {
            'type': 'object',
            'properties': {
                **(
                    {
                        instrumentation_names.tool_arguments_attr: {'type': 'object'},
                        instrumentation_names.tool_result_attr: {'type': 'object'},
                    }
                    if run_context.trace_include_content
                    else {}
                ),
                'gen_ai.tool.name': {},
                **({'gen_ai.tool.call.id': {}} if run_context.tool_call_id else {}),
            },
        }
    )

    with run_context.tracer.start_as_current_span(
        instrumentation_names.get_output_tool_span_name(tool_name), attributes=attributes
    ) as span:
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
                instrumentation_names.tool_result_attr,
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
    ) -> T:
        """Run the validator function on `result`.

        Propagates `ModelRetry` raised by the user's validator unwrapped; the caller
        (`run_output_process_hooks`, `stream_text`, etc.) decides whether to wrap in
        `ToolRetryError` for retry handling or re-raise.
        """
        if self._takes_ctx:
            args = run_context, result
        else:
            args = (result,)

        if self._is_async:
            function = cast(Callable[[Any], Awaitable[T]], self.function)
            return await function(*args)
        function = cast(Callable[[Any], T], self.function)
        return await _utils.run_in_executor(function, *args)


@dataclass(kw_only=True)
class OutputSchema(ABC, Generic[OutputDataT]):
    allows_none: bool
    text_processor: BaseOutputProcessor[OutputDataT] | None = None
    toolset: OutputToolset[Any] | None = None
    object_def: OutputObjectDefinition | None = None
    allows_deferred_tools: bool = False
    allows_image: bool = False

    @property
    def mode(self) -> OutputMode:
        raise NotImplementedError()

    @property
    def allows_text(self) -> bool:
        return self.text_processor is not None

    @classmethod
    def build(  # noqa: C901
        cls,
        output_spec: OutputSpec[OutputDataT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> OutputSchema[OutputDataT]:
        """Build an OutputSchema dataclass from an output type."""
        outputs = _flatten_output_spec(output_spec)

        # `str | None` produces NoneType (the class) via get_union_args; bare `None` value produces None itself
        allows_none = NoneType in outputs or None in outputs
        if allows_none:
            outputs = [output for output in outputs if output is not NoneType and output is not None]
            if len(outputs) == 0:
                raise UserError('At least one output type must be provided other than `None`.')

        allows_deferred_tools = DeferredToolRequests in outputs
        if allows_deferred_tools:
            outputs = [output for output in outputs if output is not DeferredToolRequests]
            if len(outputs) == 0:
                raise UserError('At least one output type must be provided other than `DeferredToolRequests`.')

        allows_image = _messages.BinaryImage in outputs
        if allows_image:
            outputs = [output for output in outputs if output is not _messages.BinaryImage]

        if output := next((output for output in outputs if isinstance(output, NativeOutput)), None):  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
            if len(outputs) > 1:
                raise UserError('`NativeOutput` must be the only output type.')  # pragma: no cover

            flattened_outputs = _flatten_output_spec(output.outputs)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            if DeferredToolRequests in flattened_outputs:
                raise UserError(  # pragma: no cover
                    '`NativeOutput` cannot contain `DeferredToolRequests`. Include it alongside the native output marker instead: `output_type=[NativeOutput(...), DeferredToolRequests]`'
                )
            if _messages.BinaryImage in flattened_outputs:
                raise UserError(  # pragma: no cover
                    '`NativeOutput` cannot contain `BinaryImage`. Include it alongside the native output marker instead: `output_type=[NativeOutput(...), BinaryImage]`'
                )

            return NativeOutputSchema(
                template=output.template,
                processor=cls._build_processor(
                    flattened_outputs,
                    name=output.name,
                    description=output.description,
                    strict=output.strict,
                ),
                allows_deferred_tools=allows_deferred_tools,
                allows_image=allows_image,
                allows_none=allows_none,
            )
        elif output := next((output for output in outputs if isinstance(output, PromptedOutput)), None):  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
            if len(outputs) > 1:
                raise UserError('`PromptedOutput` must be the only output type.')  # pragma: no cover

            flattened_outputs = _flatten_output_spec(output.outputs)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

            if DeferredToolRequests in flattened_outputs:
                raise UserError(  # pragma: no cover
                    '`PromptedOutput` cannot contain `DeferredToolRequests`. Include it alongside the prompted output marker instead: `output_type=[PromptedOutput(...), DeferredToolRequests]`'
                )
            if _messages.BinaryImage in flattened_outputs:
                raise UserError(  # pragma: no cover
                    '`PromptedOutput` cannot contain `BinaryImage`. Include it alongside the prompted output marker instead: `output_type=[PromptedOutput(...), BinaryImage]`'
                )

            return PromptedOutputSchema(
                template=output.template,
                processor=cls._build_processor(
                    flattened_outputs,
                    name=output.name,
                    description=output.description,
                ),
                allows_deferred_tools=allows_deferred_tools,
                allows_image=allows_image,
                allows_none=allows_none,
            )

        text_outputs: Sequence[type[str] | TextOutput[OutputDataT]] = []
        tool_outputs: Sequence[ToolOutput[OutputDataT]] = []
        other_outputs: Sequence[OutputTypeOrFunction[OutputDataT]] = []
        for output in outputs:
            if output is str:
                text_outputs.append(cast(type[str], output))
            elif isinstance(output, TextOutput):
                text_outputs.append(output)  # pyright: ignore[reportUnknownArgumentType]
            elif isinstance(output, ToolOutput):
                tool_outputs.append(output)  # pyright: ignore[reportUnknownArgumentType]
            elif isinstance(output, NativeOutput):
                # We can never get here because this is checked for above.
                raise UserError('`NativeOutput` must be the only output type.')  # pragma: no cover
            elif isinstance(output, PromptedOutput):
                # We can never get here because this is checked for above.
                raise UserError('`PromptedOutput` must be the only output type.')  # pragma: no cover
            else:
                other_outputs.append(output)

        # If `None` is allowed and we're building output tools, expose `NoneType` as its own
        # output tool so the model can commit to `None` through the structured schema alongside
        # any other output types, matching how the model would pick between them.
        if allows_none and (tool_outputs or other_outputs):
            other_outputs.append(cast(OutputTypeOrFunction[OutputDataT], NoneType))

        toolset = OutputToolset.build(tool_outputs + other_outputs, name=name, description=description, strict=strict)

        text_processor: BaseOutputProcessor[OutputDataT] | None = None

        if len(text_outputs) > 0:
            if len(text_outputs) > 1:
                raise UserError('Only one `str` or `TextOutput` is allowed.')
            text_output = text_outputs[0]

            if isinstance(text_output, TextOutput):
                text_processor = TextFunctionOutputProcessor(text_output.output_function)
            else:
                text_processor = TextOutputProcessor()

            if toolset:
                return ToolOutputSchema(
                    toolset=toolset,
                    text_processor=text_processor,
                    allows_deferred_tools=allows_deferred_tools,
                    allows_image=allows_image,
                    allows_none=allows_none,
                )
            else:
                return TextOutputSchema(
                    text_processor=text_processor,
                    allows_deferred_tools=allows_deferred_tools,
                    allows_image=allows_image,
                    allows_none=allows_none,
                )

        if len(tool_outputs) > 0:
            return ToolOutputSchema(
                toolset=toolset,
                allows_deferred_tools=allows_deferred_tools,
                allows_image=allows_image,
                allows_none=allows_none,
            )

        if len(other_outputs) > 0:
            return AutoOutputSchema(
                processor=cls._build_processor(other_outputs, name=name, description=description, strict=strict),
                toolset=toolset,
                allows_deferred_tools=allows_deferred_tools,
                allows_image=allows_image,
                allows_none=allows_none,
            )

        if allows_image:
            return ImageOutputSchema(allows_deferred_tools=allows_deferred_tools, allows_none=allows_none)

        raise UserError('At least one output type must be provided.')

    @staticmethod
    def _build_processor(
        outputs: Sequence[OutputTypeOrFunction[OutputDataT]],
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> BaseObjectOutputProcessor[OutputDataT]:
        outputs = _flatten_output_spec(outputs)
        if len(outputs) == 1:
            return ObjectOutputProcessor(output=outputs[0], name=name, description=description, strict=strict)

        return UnionOutputProcessor(outputs=outputs, strict=strict, name=name, description=description)


@dataclass(init=False)
class AutoOutputSchema(OutputSchema[OutputDataT]):
    processor: BaseObjectOutputProcessor[OutputDataT]

    def __init__(
        self,
        processor: BaseObjectOutputProcessor[OutputDataT],
        toolset: OutputToolset[Any] | None,
        allows_deferred_tools: bool,
        allows_image: bool,
        allows_none: bool,
    ):
        # We set a toolset here as they're checked for name conflicts with other toolsets in the Agent constructor.
        # At that point we may not know yet what output mode we're going to use if no model was provided or it was deferred until agent.run time,
        # but we cover ourselves just in case we end up using the tool output mode.
        super().__init__(
            toolset=toolset,
            object_def=processor.object_def,
            text_processor=processor,
            allows_deferred_tools=allows_deferred_tools,
            allows_image=allows_image,
            allows_none=allows_none,
        )
        self.processor = processor

    @property
    def mode(self) -> OutputMode:
        return 'auto'


@dataclass(init=False)
class TextOutputSchema(OutputSchema[OutputDataT]):
    def __init__(
        self,
        *,
        text_processor: TextOutputProcessor[OutputDataT],
        allows_deferred_tools: bool,
        allows_image: bool,
        allows_none: bool,
    ):
        super().__init__(
            text_processor=text_processor,
            allows_deferred_tools=allows_deferred_tools,
            allows_image=allows_image,
            allows_none=allows_none,
        )

    @property
    def mode(self) -> OutputMode:
        return 'text'


class ImageOutputSchema(OutputSchema[OutputDataT]):
    def __init__(self, *, allows_deferred_tools: bool, allows_none: bool):
        super().__init__(allows_deferred_tools=allows_deferred_tools, allows_image=True, allows_none=allows_none)

    @property
    def mode(self) -> OutputMode:
        return 'image'


@dataclass(init=False)
class StructuredTextOutputSchema(OutputSchema[OutputDataT], ABC):
    processor: BaseObjectOutputProcessor[OutputDataT]
    template: str | Literal[False] | None

    def __init__(
        self,
        *,
        template: str | Literal[False] | None = None,
        processor: BaseObjectOutputProcessor[OutputDataT],
        allows_deferred_tools: bool,
        allows_image: bool,
        allows_none: bool,
    ):
        super().__init__(
            text_processor=processor,
            object_def=processor.object_def,
            allows_deferred_tools=allows_deferred_tools,
            allows_image=allows_image,
            allows_none=allows_none,
        )
        self.processor = processor
        self.template = template

    @classmethod
    def build_instructions(cls, template: str, object_def: OutputObjectDefinition) -> str:
        """Build instructions from a template and an object definition."""
        schema = object_def.json_schema.copy()
        if object_def.name:
            schema['title'] = object_def.name
        if object_def.description:
            schema['description'] = object_def.description

        if '{schema}' not in template:
            template = '\n\n'.join([template, '{schema}'])

        return template.format(schema=json.dumps(schema))


class NativeOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    @property
    def mode(self) -> OutputMode:
        return 'native'


@dataclass(init=False)
class PromptedOutputSchema(StructuredTextOutputSchema[OutputDataT]):
    @property
    def mode(self) -> OutputMode:
        return 'prompted'


@dataclass(init=False)
class ToolOutputSchema(OutputSchema[OutputDataT]):
    def __init__(
        self,
        *,
        toolset: OutputToolset[Any] | None,
        text_processor: BaseOutputProcessor[OutputDataT] | None = None,
        allows_deferred_tools: bool,
        allows_image: bool,
        allows_none: bool,
    ):
        super().__init__(
            toolset=toolset,
            allows_deferred_tools=allows_deferred_tools,
            text_processor=text_processor,
            allows_image=allows_image,
            allows_none=allows_none,
        )

    @property
    def mode(self) -> OutputMode:
        return 'tool'


class BaseOutputProcessor(ABC, Generic[OutputDataT]):
    def validate(
        self,
        data: str | dict[str, Any] | None,
        *,
        allow_partial: bool = False,
        validation_context: Any | None = None,
    ) -> Any:
        """Validate/parse raw output. Default: identity (returns data as-is)."""
        return data

    async def call(
        self,
        output: Any,
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute output function on validated output. Default: identity (returns output as-is)."""
        return output

    def hook_validate(
        self,
        data: str | dict[str, Any] | None,
        *,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
    ) -> tuple[Any, Any]:
        """Validate raw data and return `(semantic_value, state)`.

        `semantic_value` is what output hooks see — the logical value the model was asked to
        produce (e.g., `MyModel(...)`, `42`, the input to a single-arg output function).
        `state` is opaque per-invocation data that `hook_execute` needs to complete processing
        (e.g., the resolved union member), or `None` if unused.

        Default: runs `self.validate()` and returns the result as the semantic value.
        """
        validated = self.validate(data, allow_partial=allow_partial, validation_context=run_context.validation_context)
        return validated, None

    async def hook_execute(
        self,
        semantic: Any,
        state: Any,
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute the post-validation stage on a (possibly hook-modified) semantic value.

        `state` is the second element of the tuple returned by `hook_validate`.
        Default: calls `self.call()` with the semantic value.
        """
        return await self.call(semantic, run_context=run_context, wrap_validation_errors=wrap_validation_errors)

    @abstractmethod
    def get_output_context(
        self,
        schema: OutputSchema[Any],
        *,
        mode: OutputMode | None = None,
        tool_call: _messages.ToolCallPart | None = None,
        tool_def: ToolDefinition | None = None,
    ) -> OutputContext:
        """Return context information about this processor for output hooks.

        `schema` provides the run-wide output configuration (what the schema accepts).
        `mode` overrides the reported mode for the current dispatch path (e.g. `'tool'`
        when validating an output tool call within an `'auto'` schema); defaults to
        `schema.mode`.
        """
        raise NotImplementedError()


@dataclass(kw_only=True)
class BaseObjectOutputProcessor(BaseOutputProcessor[OutputDataT]):
    object_def: OutputObjectDefinition


@dataclass(init=False)
class ObjectOutputProcessor(BaseObjectOutputProcessor[OutputDataT]):
    output_type: type[Any] | None = None
    """The resolved semantic output type (e.g. `MyModel`, `int`). For output functions,
    this is the function's *input* type — i.e. what the model produces — not its return
    type. `None` only for processors without a resolvable input type."""
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
        self.output_type = None

        if inspect.isfunction(output) or inspect.ismethod(output):
            self._function_schema = _function_schema.function_schema(output, GenerateToolJsonSchema)
            self.validator = self._function_schema.validator
            json_schema = self._function_schema.json_schema
            json_schema['description'] = self._function_schema.description

            # Extract the function's input type (what the model produces) for output_type
            type_hints = _utils.get_function_type_hints(output)
            for hint_name, hint_type in type_hints.items():  # pragma: no branch
                if hint_name != 'return' and not _function_schema._is_call_ctx(hint_type):  # pyright: ignore[reportPrivateUsage]
                    self.output_type = hint_type
                    break
        else:
            self.output_type = cast(type[Any], output)
            json_schema_type_adapter: TypeAdapter[Any]
            validation_type_adapter: TypeAdapter[Any]
            if _utils.is_model_like(output):
                json_schema_type_adapter = validation_type_adapter = TypeAdapter(output)
            else:
                self.outer_typed_dict_key = 'response'
                output_type: type[OutputDataT] = cast(type[OutputDataT], output)

                response_data_typed_dict = TypedDict(  # noqa: UP013
                    'response_data_typed_dict',
                    {'response': output_type},  # pyright: ignore[reportInvalidTypeForm]
                )
                json_schema_type_adapter = TypeAdapter(response_data_typed_dict)

                # More lenient validator: allow either the native type or a JSON string containing it
                # i.e. `response: OutputDataT | Json[OutputDataT]`, as some models don't follow the schema correctly,
                # e.g. `BedrockConverseModel('us.meta.llama3-2-11b-instruct-v1:0')`
                response_validation_typed_dict = TypedDict(  # noqa: UP013
                    'response_validation_typed_dict',
                    {'response': output_type | Json[output_type]},  # pyright: ignore[reportInvalidTypeForm]
                )
                validation_type_adapter = TypeAdapter(response_validation_typed_dict)

            # Really a PluggableSchemaValidator, but it's API-compatible
            self.validator = cast(SchemaValidator, validation_type_adapter.validator)
            json_schema = _utils.check_object_json_schema(
                json_schema_type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
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

        super().__init__(
            object_def=OutputObjectDefinition(
                name=name or getattr(output, '__name__', None),
                description=description,
                json_schema=json_schema,
                strict=strict,
            )
        )

    def validate(
        self,
        data: str | dict[str, Any] | None,
        *,
        allow_partial: bool = False,
        validation_context: Any | None = None,
    ) -> dict[str, Any]:
        if isinstance(data, str):
            data = _utils.strip_markdown_fences(data)
        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        if isinstance(data, str):
            return self.validator.validate_json(
                data or '{}', allow_partial=pyd_allow_partial, context=validation_context
            )
        else:
            return self.validator.validate_python(
                data or {}, allow_partial=pyd_allow_partial, context=validation_context
            )

    async def call(
        self,
        output: dict[str, Any],
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        if k := self.outer_typed_dict_key:
            output = output[k]

        if self._function_schema:
            output = await execute_traced_output_function(
                self._function_schema,
                run_context=run_context,
                args=output,
                wrap_validation_errors=wrap_validation_errors,
            )

        return output

    @property
    def hook_unwrap_key(self) -> str | None:
        """The internal envelope key around a single semantic value, or `None`.

        Output hooks see the **semantic value** the model was asked to produce (e.g. a
        `MyModel` instance, `42`, or the dict the user is producing as `dict[str, str]`).
        Pydantic validation works in dict-shape (`{'response': 42}`, `{'data': my_dict}`)
        for primitive and function-arg cases, so this property returns the key to peel off
        at the hook boundary and re-add when calling the inner processor.

        - `outer_typed_dict_key` — set when wrapping a bare primitive (`output_type=int`,
          `output_type=dict[str, str]`, etc.) so the model produces `{'response': value}`.
          The wrapper key is purely a transport detail; the value's own structure is
          unchanged.
        - `FunctionSchema.single_field_name` — set when the output function takes exactly
          one value-carrying arg (`def f(x: SomeType)`). The wrapper key is the arg name.

        Returns `None` for bare `BaseModel` outputs and multi-arg output functions, where
        the validated value is the semantic value as-is (no envelope to peel).
        """
        if self.outer_typed_dict_key is not None:
            return self.outer_typed_dict_key
        if self._function_schema is not None:
            return self._function_schema.single_field_name
        return None

    def hook_validate(
        self,
        data: str | dict[str, Any] | None,
        *,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
    ) -> tuple[Any, Any]:
        validated = self.validate(data, allow_partial=allow_partial, validation_context=run_context.validation_context)
        if (k := self.hook_unwrap_key) is None:
            return validated, None
        return validated[k], None

    async def hook_execute(
        self,
        semantic: Any,
        state: Any,
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        # Re-wrap the (possibly hook-modified) semantic value into the internal dict shape `call()` expects.
        # No idempotency check: for `output_type=dict[...]`, the semantic value can itself be a dict
        # that happens to contain the unwrap key.
        if (k := self.hook_unwrap_key) is not None:
            semantic = {k: semantic}
        return await self.call(semantic, run_context=run_context, wrap_validation_errors=wrap_validation_errors)

    def get_output_context(
        self,
        schema: OutputSchema[Any],
        *,
        mode: OutputMode | None = None,
        tool_call: _messages.ToolCallPart | None = None,
        tool_def: ToolDefinition | None = None,
    ) -> OutputContext:
        return OutputContext(
            mode=mode if mode is not None else schema.mode,
            output_type=self.output_type,
            object_def=self.object_def,
            has_function=self._function_schema is not None,
            tool_call=tool_call,
            tool_def=tool_def,
            allows_text=schema.allows_text,
            allows_image=schema.allows_image,
            allows_deferred_tools=schema.allows_deferred_tools,
        )


@dataclass
class _UnionValidatedOutput:
    """Internal wrapper returned by `UnionOutputProcessor.validate()`.

    Holds the resolved union member's kind key and the **semantic value** already
    unwrapped by the inner processor (e.g. a `MyModel` instance or an `int`, not a
    `{'response': ...}` dict). `UnionOutputProcessor.call()` uses `kind` to find the
    right inner processor and rewraps `data` back into that processor's internal dict
    shape before delegating.

    This keeps `self._processors` lookup internal to the union processor — callers
    (validate hooks, `hook_validate`) get clean semantic data without private access.
    """

    kind: str
    data: Any


@dataclass
class UnionOutputResult:
    kind: str
    data: ObjectJsonSchema


@dataclass
class UnionOutputModel:
    result: UnionOutputResult


@dataclass(init=False)
class UnionOutputProcessor(BaseObjectOutputProcessor[OutputDataT]):
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

        super().__init__(
            object_def=OutputObjectDefinition(
                json_schema=json_schema,
                strict=strict,
                name=name,
                description=description,
            )
        )

    def validate(
        self,
        data: str | dict[str, Any] | None,
        *,
        allow_partial: bool = False,
        validation_context: Any | None = None,
    ) -> _UnionValidatedOutput:
        """Validate the union envelope, resolve the kind, and return the inner semantic value.

        The returned wrapper holds the kind key and the unwrapped semantic value (not the
        inner processor's dict form). `call()` rewraps before delegating to the inner
        processor.
        """
        union_validated: UnionOutputModel = self._union_processor.validate(  # pyright: ignore[reportAssignmentType]
            data, allow_partial=allow_partial, validation_context=validation_context
        )

        result = union_validated.result
        kind: str = result.kind
        inner_data: dict[str, Any] = result.data

        # Pydantic validation ensures kind is always valid, so KeyError can't happen.
        inner = self._processors[kind]
        inner_validated = inner.validate(inner_data, allow_partial=allow_partial, validation_context=validation_context)
        # Unwrap to semantic here so the wrapper's `data` is always what hooks / callers
        # expect — e.g. a `MyModel` instance or an `int`, not `{'response': 42}`.
        if (k := inner.hook_unwrap_key) is not None:
            inner_validated = inner_validated[k]
        return _UnionValidatedOutput(kind=kind, data=inner_validated)

    async def call(
        self,
        output: _UnionValidatedOutput,
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Delegate to the inner processor resolved by the kind key in the wrapper.

        Rewraps the semantic `data` into the inner processor's dict shape before calling,
        inverting the unwrap in `validate()`.
        """
        inner = self._processors[output.kind]
        inner_args: Any = output.data
        if (k := inner.hook_unwrap_key) is not None:
            inner_args = {k: inner_args}
        return await inner.call(inner_args, run_context=run_context, wrap_validation_errors=wrap_validation_errors)

    def hook_validate(
        self,
        data: str | dict[str, Any] | None,
        *,
        run_context: RunContext[AgentDepsT],
        allow_partial: bool = False,
    ) -> tuple[Any, Any]:
        # `validate()` already unwraps to semantic, so no private `_processors` access here.
        union_validated = self.validate(
            data, allow_partial=allow_partial, validation_context=run_context.validation_context
        )
        return union_validated.data, union_validated.kind

    async def hook_execute(
        self,
        semantic: Any,
        state: Any,
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Dispatch the (possibly hook-modified) semantic value to the right inner processor.

        If `state` (the resolved kind from `hook_validate`) is set and the semantic value's
        type still matches that inner's `output_type`, use that kind. Otherwise — hook swapped
        the value for a different union member, or we're on the error-recovery path with no
        kind — resolve by `isinstance` so the output function still runs for the new type.
        """
        kind: str | None = state
        if kind is not None:
            inner = self._processors[kind]
            if self._semantic_matches_inner(inner, semantic):
                return await self.call(
                    _UnionValidatedOutput(kind=kind, data=semantic),
                    run_context=run_context,
                    wrap_validation_errors=wrap_validation_errors,
                )
            # Type mismatch — hook returned a different union member than validation resolved.
            # Fall through to resolve-by-type so we reach the right inner processor.
        match = self._resolve_inner_for_value(semantic)
        if match is not None:
            return await self.call(match, run_context=run_context, wrap_validation_errors=wrap_validation_errors)
        # Value doesn't match any union member — pass through unmodified. The output function
        # (if any) doesn't run. Users hitting this should inspect their hook return type.
        return semantic

    @staticmethod
    def _semantic_matches_inner(inner: ObjectOutputProcessor[Any], semantic: Any) -> bool:
        """Check whether `semantic` matches the shape `inner` expects to receive.

        - **Multi-arg output function** (`def f(x: int, y: str) -> Foo`): no unwrap key, so
          `semantic` is the validated dict of args. `inner.output_type` is the first arg's
          type, which can't be compared against the dict, so just check `isinstance(dict)`.
        - **Single-value cases** (BaseModel, single-arg function, primitives): `semantic`
          is the unwrapped value, so isinstance against `inner.output_type` is correct.
        """
        # Both inners are `ObjectOutputProcessor` (same module), so `_function_schema` is internal,
        # not external; pyright's private-usage warning is a false positive here.
        fn_schema = inner._function_schema  # pyright: ignore[reportPrivateUsage]
        if fn_schema is not None and fn_schema.single_field_name is None:
            return isinstance(semantic, dict)
        if inner.output_type is None:
            return False  # pragma: no cover
        return _isinstance_maybe_generic(semantic, inner.output_type)

    def _resolve_inner_for_value(self, value: Any) -> _UnionValidatedOutput | None:
        """Find the inner processor whose `output_type` matches `value`.

        Used on the error-recovery and type-mismatch paths in `hook_execute`. Returns a
        `_UnionValidatedOutput` ready for `call()`, or `None` if no inner type matches.

        Multi-arg output function inners are skipped: their `output_type` is the first
        arg's type, not the dict shape `value` would have, so isinstance can't pick
        them out unambiguously. The kind-trust path in `hook_execute` already handles
        the normal multi-arg case (no swap).
        """
        for kind, inner in self._processors.items():
            if inner.output_type is None:  # pragma: no cover
                continue
            fn_schema = inner._function_schema  # pyright: ignore[reportPrivateUsage]
            if fn_schema is not None and fn_schema.single_field_name is None:
                continue  # multi-arg — see docstring
            if _isinstance_maybe_generic(value, inner.output_type):
                return _UnionValidatedOutput(kind=kind, data=value)
        return None

    def get_output_context(
        self,
        schema: OutputSchema[Any],
        *,
        mode: OutputMode | None = None,
        tool_call: _messages.ToolCallPart | None = None,
        tool_def: ToolDefinition | None = None,
    ) -> OutputContext:
        return OutputContext(
            mode=mode if mode is not None else schema.mode,
            output_type=None,
            object_def=self.object_def,
            has_function=any(p._function_schema is not None for p in self._processors.values()),  # pyright: ignore[reportPrivateUsage]
            tool_call=tool_call,
            tool_def=tool_def,
            allows_text=schema.allows_text,
            allows_image=schema.allows_image,
            allows_deferred_tools=schema.allows_deferred_tools,
        )


class TextOutputProcessor(BaseOutputProcessor[OutputDataT]):
    def get_output_context(
        self,
        schema: OutputSchema[Any],
        *,
        mode: OutputMode | None = None,
        tool_call: _messages.ToolCallPart | None = None,
        tool_def: ToolDefinition | None = None,
    ) -> OutputContext:
        return OutputContext(
            mode=mode if mode is not None else schema.mode,
            output_type=str,
            object_def=None,
            has_function=False,
            tool_call=tool_call,
            tool_def=tool_def,
            allows_text=schema.allows_text,
            allows_image=schema.allows_image,
            allows_deferred_tools=schema.allows_deferred_tools,
        )


@dataclass(init=False)
class TextFunctionOutputProcessor(TextOutputProcessor[OutputDataT]):
    _function_schema: _function_schema.FunctionSchema
    _str_argument_name: str

    def __init__(
        self,
        output_function: TextOutputFunc[OutputDataT],
    ):
        self._function_schema = _function_schema.function_schema(output_function, GenerateToolJsonSchema)

        if (
            not (arguments_schema := self._function_schema.json_schema.get('properties', {}))
            or len(arguments_schema) != 1
            or not (argument_name := next(iter(arguments_schema.keys()), None))
            or arguments_schema.get(argument_name, {}).get('type') != 'string'
        ):
            raise UserError('TextOutput must take a function taking a single `str` argument')

        self._str_argument_name = argument_name

    async def call(
        self,
        output: Any,
        *,
        run_context: RunContext[AgentDepsT],
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute the text output function."""
        args = {self._str_argument_name: output}
        return await execute_traced_output_function(
            self._function_schema, run_context=run_context, args=args, wrap_validation_errors=wrap_validation_errors
        )

    def get_output_context(
        self,
        schema: OutputSchema[Any],
        *,
        mode: OutputMode | None = None,
        tool_call: _messages.ToolCallPart | None = None,
        tool_def: ToolDefinition | None = None,
    ) -> OutputContext:
        return OutputContext(
            mode=mode if mode is not None else schema.mode,
            output_type=str,
            object_def=None,
            has_function=True,
            tool_call=tool_call,
            tool_def=tool_def,
            allows_text=schema.allows_text,
            allows_image=schema.allows_image,
            allows_deferred_tools=schema.allows_deferred_tools,
        )


@dataclass(init=False)
class OutputToolset(AbstractToolset[AgentDepsT]):
    """A toolset that contains output tools for agent output types."""

    _tool_defs: list[ToolDefinition]
    """The tool definitions for the output tools in this toolset."""
    processors: dict[str, ObjectOutputProcessor[Any]]
    """The processors for the output tools in this toolset."""
    max_retries: int | None
    """Default max retries for output tools, set by the Agent. Per-tool overrides from `ToolOutput.max_retries` take priority."""
    _max_retries_overrides: dict[str, int]
    """Per-tool max_retries overrides from `ToolOutput(max_retries=N)`."""
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

        max_retries_overrides: dict[str, int] = {}
        tool_output_max_retries: int | None = None

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
                tool_output_max_retries = output.max_retries

                output = output.output  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

            description = description or default_description
            if strict is None:
                strict = default_strict

            processor = ObjectOutputProcessor(output=output, description=description, strict=strict)  # pyright: ignore[reportUnknownArgumentType]
            object_def = processor.object_def

            if name is None:
                name = default_name
                if multiple:
                    # strip unsupported characters like "[" and "]" from generic class names
                    safe_name = _utils.TOOL_NAME_SANITIZER.sub('', object_def.name or '')
                    name += f'_{safe_name}'

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
            if tool_output_max_retries is not None:
                max_retries_overrides[name] = tool_output_max_retries
            tool_output_max_retries = None

        return cls(processors=processors, tool_defs=tool_defs, max_retries_overrides=max_retries_overrides)

    def __init__(
        self,
        tool_defs: list[ToolDefinition],
        processors: dict[str, ObjectOutputProcessor[Any]],
        max_retries: int | None = None,
        max_retries_overrides: dict[str, int] | None = None,
        output_validators: list[OutputValidator[AgentDepsT, Any]] | None = None,
    ):
        self.processors = processors
        self._tool_defs = tool_defs
        self.max_retries = max_retries
        self._max_retries_overrides = max_retries_overrides or {}
        self.output_validators = output_validators or []

    @property
    def id(self) -> str | None:
        return '<output>'  # pragma: no cover

    @property
    def label(self) -> str:
        return "the agent's output tools"

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        assert self.max_retries is not None, 'Agent must set OutputToolset.max_retries before the run'
        max_retries = self.max_retries
        return {
            tool_def.name: ToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=self._max_retries_overrides.get(tool_def.name, max_retries),
                args_validator=self.processors[tool_def.name].validator,
            )
            for tool_def in self._tool_defs
        }

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        # Output tools are handled by ToolManager.validate_output_tool_call/execute_output_tool_call,
        # not through the normal toolset.call_tool path.
        raise NotImplementedError('Output tools use validate_output_tool_call/execute_output_tool_call')


@overload
def _flatten_output_spec(
    output_spec: OutputTypeOrFunction[T] | Sequence[OutputTypeOrFunction[T]],
) -> Sequence[OutputTypeOrFunction[T]]: ...


@overload
def _flatten_output_spec(output_spec: OutputSpec[T]) -> Sequence[_OutputSpecItem[T]]: ...


def _flatten_output_spec(output_spec: OutputSpec[T]) -> Sequence[_OutputSpecItem[T]]:
    outputs: Sequence[OutputSpec[T]]
    if isinstance(output_spec, Sequence):
        outputs = output_spec  # pyright: ignore[reportUnknownVariableType]
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


def types_from_output_spec(output_spec: OutputSpec[T]) -> Sequence[T | type[str]]:
    outputs: Sequence[OutputSpec[T]]
    if isinstance(output_spec, Sequence):
        outputs = output_spec  # pyright: ignore[reportUnknownVariableType]
    else:
        outputs = (output_spec,)

    outputs_flat: list[T | type[str]] = []
    for output in outputs:
        if isinstance(output, NativeOutput):
            outputs_flat.extend(types_from_output_spec(output.outputs))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        elif isinstance(output, PromptedOutput):
            outputs_flat.extend(types_from_output_spec(output.outputs))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        elif isinstance(output, TextOutput):
            outputs_flat.append(str)
        elif isinstance(output, ToolOutput):
            outputs_flat.extend(types_from_output_spec(output.output))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        elif union_types := _utils.get_union_args(output):
            outputs_flat.extend(union_types)
        elif inspect.isfunction(output) or inspect.ismethod(output):
            type_hints = get_function_type_hints(output)
            if return_annotation := type_hints.get('return', None):
                outputs_flat.extend(types_from_output_spec(return_annotation))
            else:
                outputs_flat.append(str)
        else:
            outputs_flat.append(cast(T, output))

    return outputs_flat
