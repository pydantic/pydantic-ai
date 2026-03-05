from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal

from opentelemetry.trace import Tracer
from pydantic import ValidationError
from typing_extensions import deprecated

from . import messages as _messages
from ._instrumentation import InstrumentationNames
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior
from .messages import ToolCallPart
from .tools import ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool
from .usage import RunUsage

ParallelExecutionMode = Literal['parallel', 'sequential', 'parallel_ordered_events']

_parallel_execution_mode_ctx_var: ContextVar[ParallelExecutionMode] = ContextVar(
    'parallel_execution_mode', default='parallel'
)


@dataclass
class ValidatedToolCall(Generic[AgentDepsT]):
    """Result of validating a tool call's arguments (may represent success or failure).

    This separates validation from execution, allowing callers to:
    1. Know if validation passed before executing
    2. Emit accurate `args_valid` status in events
    3. Handle validation failures differently from execution failures
    """

    call: ToolCallPart
    """The original tool call part."""
    tool: ToolsetTool[AgentDepsT] | None
    """The tool definition, or None if the tool is unknown."""
    ctx: RunContext[AgentDepsT]
    """The run context for this tool call."""
    args_valid: bool
    """Whether argument validation (schema + custom validator) passed."""
    validated_args: dict[str, Any] | None = None
    """The validated arguments if validation passed, None otherwise."""
    validation_error: ToolRetryError | None = None
    """The validation error if validation failed, None otherwise."""


@dataclass
class ToolManager(Generic[AgentDepsT]):
    """Manages tools for an agent run step. It caches the agent run's toolset's tool definitions and handles calling tools and retries."""

    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provides the tools for this run step."""
    ctx: RunContext[AgentDepsT] | None = None
    """The agent run context for a specific run step."""
    tools: dict[str, ToolsetTool[AgentDepsT]] | None = None
    """The cached tools for this run step."""
    failed_tools: set[str] = field(default_factory=set[str])
    """Names of tools that failed in this run step."""
    default_max_retries: int = 1
    """Default number of times to retry a tool"""

    @classmethod
    @contextmanager
    def parallel_execution_mode(cls, mode: ParallelExecutionMode = 'parallel') -> Iterator[None]:
        """Set the parallel execution mode during the context.

        Args:
            mode: The execution mode for tool calls:
                - 'parallel': Run tool calls in parallel, yielding events as they complete (default).
                - 'sequential': Run tool calls one at a time in order.
                - 'parallel_ordered_events': Run tool calls in parallel, but events are emitted in order, after all calls complete.
        """
        token = _parallel_execution_mode_ctx_var.set(mode)
        try:
            yield
        finally:
            _parallel_execution_mode_ctx_var.reset(token)

    @classmethod
    @contextmanager
    @deprecated('Use `parallel_execution_mode("sequential")` instead.')
    def sequential_tool_calls(cls) -> Iterator[None]:
        """Run tool calls sequentially during the context."""
        with cls.parallel_execution_mode('sequential'):
            yield

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> ToolManager[AgentDepsT]:
        """Build a new tool manager for the next run step, carrying over the retries from the current run step."""
        if self.ctx is not None:
            if ctx.run_step == self.ctx.run_step:
                return self

            retries = {
                failed_tool_name: self.ctx.retries.get(failed_tool_name, 0) + 1
                for failed_tool_name in self.failed_tools
            }
            ctx = replace(ctx, retries=retries)

        return self.__class__(
            toolset=self.toolset,
            ctx=ctx,
            tools=await self.toolset.get_tools(ctx),
            default_max_retries=self.default_max_retries,
        )

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        """The tool definitions for the tools in this tool manager."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        return [tool.tool_def for tool in self.tools.values()]

    def get_parallel_execution_mode(self, calls: list[ToolCallPart]) -> ParallelExecutionMode:
        """Get the effective parallel execution mode for a list of tool calls.

        This takes into account both the context variable and whether any tool
        has `sequential=True` set. If any tool requires sequential execution,
        returns `'sequential'` regardless of the context variable.
        """
        # Check if any tool requires sequential execution
        if any(tool_def.sequential for call in calls if (tool_def := self.get_tool_def(call.tool_name))):
            return 'sequential'

        mode = _parallel_execution_mode_ctx_var.get()

        return mode

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        """Get the tool definition for a given tool name, or `None` if the tool is unknown."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        try:
            return self.tools[name].tool_def
        except KeyError:
            return None

    def _check_max_retries(self, name: str, max_retries: int, error: Exception) -> None:
        """Raise UnexpectedModelBehavior if the tool has exceeded its max retries."""
        assert self.ctx is not None
        if self.ctx.retries.get(name, 0) == max_retries:
            raise UnexpectedModelBehavior(f'Tool {name!r} exceeded max retries count of {max_retries}') from error

    @staticmethod
    def _wrap_error_as_retry(name: str, call: ToolCallPart, error: ValidationError | ModelRetry) -> ToolRetryError:
        """Convert a ValidationError or ModelRetry to a ToolRetryError with a RetryPromptPart."""
        if isinstance(error, ValidationError):
            content: list[Any] | str = error.errors(include_url=False, include_context=False)
        else:
            content = error.message
        m = _messages.RetryPromptPart(tool_name=name, content=content, tool_call_id=call.tool_call_id)
        return ToolRetryError(m)

    def _build_tool_context(
        self,
        call: ToolCallPart,
        tool: ToolsetTool[AgentDepsT],
        *,
        allow_partial: bool,
        approved: bool = False,
        metadata: Any = None,
    ) -> RunContext[AgentDepsT]:
        """Build the execution context for a tool call."""
        assert self.ctx is not None
        return replace(
            self.ctx,
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            retry=self.ctx.retries.get(call.tool_name, 0),
            max_retries=tool.max_retries,
            tool_call_approved=approved,
            tool_call_metadata=metadata,
            partial_output=allow_partial,
        )

    async def _validate_tool_args(
        self,
        call: ToolCallPart,
        tool: ToolsetTool[AgentDepsT],
        ctx: RunContext[AgentDepsT],
        *,
        allow_partial: bool,
    ) -> dict[str, Any]:
        """Validate tool arguments using Pydantic schema and custom args_validator_func.

        Returns:
            The validated arguments as a dictionary.

        Raises:
            ValidationError: If argument validation fails.
            ModelRetry: If argument validation fails with a retry request.
        """
        pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
        validator = tool.args_validator
        if isinstance(call.args, str):
            args_dict = validator.validate_json(
                call.args or '{}', allow_partial=pyd_allow_partial, context=ctx.validation_context
            )
        else:
            args_dict = validator.validate_python(
                call.args or {}, allow_partial=pyd_allow_partial, context=ctx.validation_context
            )

        if tool.args_validator_func is not None:
            result = tool.args_validator_func(ctx, **args_dict)
            if inspect.isawaitable(result):
                await result

        return args_dict

    async def validate_tool_call(
        self,
        call: ToolCallPart,
        *,
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
        approved: bool = False,
        metadata: Any = None,
    ) -> ValidatedToolCall[AgentDepsT]:
        """Validate tool arguments without executing the tool.

        This method validates arguments BEFORE the tool is executed, allowing the caller to:
        1. Emit FunctionToolCallEvent with accurate `args_valid` status
        2. Handle validation failures differently from execution failures
        3. Decide whether to execute or defer based on validation result

        Args:
            call: The tool call part to validate.
            allow_partial: Whether to allow partial validation of the tool arguments.
            wrap_validation_errors: Whether to wrap validation errors in ToolRetryError.
            approved: Whether the tool call has been approved.
            metadata: Additional metadata from DeferredToolResults.metadata.

        Returns:
            ValidatedToolCall with validation results, ready for execution via execute_tool_call().
        """
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        name = call.tool_name
        tool = self.tools.get(name)
        ctx = self.ctx

        try:
            if tool is None:
                if self.tools:
                    msg = f'Available tools: {", ".join(f"{n!r}" for n in self.tools)}'
                else:
                    msg = 'No tools available.'
                raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')

            ctx = self._build_tool_context(
                call, tool, allow_partial=allow_partial, approved=approved, metadata=metadata
            )
            validated_args = await self._validate_tool_args(call, tool, ctx, allow_partial=allow_partial)
            return ValidatedToolCall(
                call=call,
                tool=tool,
                ctx=ctx,
                args_valid=True,
                validated_args=validated_args,
                validation_error=None,
            )
        except (ValidationError, ModelRetry) as e:
            max_retries = tool.max_retries if tool is not None else self.default_max_retries
            self._check_max_retries(name, max_retries, e)

            if not allow_partial:
                # If we're validating partial arguments, we don't want to count this as a failed tool as it may still succeed once the full arguments are received.
                self.failed_tools.add(name)

            if not wrap_validation_errors:
                raise

            validation_error = self._wrap_error_as_retry(name, call, e)

            return ValidatedToolCall(
                call=call,
                tool=tool,
                ctx=ctx,
                args_valid=False,
                validated_args=None,
                validation_error=validation_error,
            )

    async def execute_tool_call(
        self,
        validated: ValidatedToolCall[AgentDepsT],
    ) -> Any:
        """Execute a validated tool call, within a trace span for function tools.

        For output tools, no tracing is performed. For function tools, a trace span is
        created using the tracer from the run context.

        Args:
            validated: The validation result from validate_tool_call().

        Returns:
            The tool result if validation passed and execution succeeded.

        Raises:
            ToolRetryError: If validation failed (contains the retry prompt).
            RuntimeError: If trying to execute an external tool.
        """
        if self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        if validated.tool is not None and validated.tool.tool_def.kind == 'output':
            return await self._execute_tool_call_impl(validated)

        return await self._execute_function_tool_call(
            validated,
            tracer=self.ctx.tracer,
            include_content=self.ctx.trace_include_content,
            instrumentation_version=self.ctx.instrumentation_version,
            usage=self.ctx.usage,
        )

    async def _execute_tool_call_impl(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        usage: RunUsage | None = None,
    ) -> Any:
        """Execute a validated tool call without tracing.

        Raises ToolRetryError if validation previously failed or the tool raises ModelRetry.
        Raises UnexpectedModelBehavior if max retries exceeded.
        """
        # Asserts narrow types for pyright; invariants guaranteed by ValidatedToolCall construction
        if not validated.args_valid:
            assert validated.validation_error is not None
            raise validated.validation_error

        assert validated.tool is not None
        assert validated.validated_args is not None

        if validated.tool.tool_def.kind == 'external':
            raise RuntimeError('External tools cannot be called')

        name = validated.call.tool_name
        try:
            tool_result = await self.toolset.call_tool(
                name,
                validated.validated_args,
                validated.ctx,
                validated.tool,
            )
        except ModelRetry as e:
            self._check_max_retries(name, validated.tool.max_retries, e)
            self.failed_tools.add(name)
            raise self._wrap_error_as_retry(name, validated.call, e) from e

        if usage is not None:
            usage.tool_calls += 1

        return tool_result

    async def _execute_function_tool_call(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        tracer: Tracer,
        include_content: bool,
        instrumentation_version: int,
        usage: RunUsage | None = None,
    ) -> Any:
        """Execute a validated function tool call within a trace span.

        See <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span>.
        """
        instrumentation_names = InstrumentationNames.for_version(instrumentation_version)
        call = validated.call

        span_attributes = {
            'gen_ai.tool.name': call.tool_name,
            # NOTE: this means `gen_ai.tool.call.id` will be included even if it was generated by pydantic-ai
            'gen_ai.tool.call.id': call.tool_call_id,
            **({instrumentation_names.tool_arguments_attr: call.args_as_json_str()} if include_content else {}),
            'logfire.msg': f'running tool: {call.tool_name}',
            # add the JSON schema so these attributes are formatted nicely in Logfire
            'logfire.json_schema': json.dumps(
                {
                    'type': 'object',
                    'properties': {
                        **(
                            {
                                instrumentation_names.tool_arguments_attr: {'type': 'object'},
                                instrumentation_names.tool_result_attr: {'type': 'object'},
                            }
                            if include_content
                            else {}
                        ),
                        'gen_ai.tool.name': {},
                        'gen_ai.tool.call.id': {},
                    },
                }
            ),
        }

        with tracer.start_as_current_span(
            instrumentation_names.get_tool_span_name(call.tool_name),
            attributes=span_attributes,
        ) as span:
            try:
                tool_result = await self._execute_tool_call_impl(validated, usage=usage)
            except ToolRetryError as e:
                part = e.tool_retry
                if include_content and span.is_recording():
                    span.set_attribute(instrumentation_names.tool_result_attr, part.model_response())
                raise

            if include_content and span.is_recording():
                span.set_attribute(
                    instrumentation_names.tool_result_attr,
                    tool_result
                    if isinstance(tool_result, str)
                    else _messages.tool_return_ta.dump_json(tool_result).decode(),
                )

        return tool_result

    async def handle_call(
        self,
        call: ToolCallPart,
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
        *,
        approved: bool = False,
        metadata: Any = None,
    ) -> Any:
        """Handle a tool call by validating the arguments, calling the tool, and handling retries.

        This is a convenience method that combines validate_tool_call() and execute_tool_call().

        Args:
            call: The tool call part to handle.
            allow_partial: Whether to allow partial validation of the tool arguments.
            wrap_validation_errors: Whether to wrap validation errors in a retry prompt part.
            approved: Whether the tool call has been approved.
            metadata: Additional metadata from DeferredToolResults.metadata.
        """
        validated = await self.validate_tool_call(
            call,
            allow_partial=allow_partial,
            wrap_validation_errors=wrap_validation_errors,
            approved=approved,
            metadata=metadata,
        )
        return await self.execute_tool_call(validated)
