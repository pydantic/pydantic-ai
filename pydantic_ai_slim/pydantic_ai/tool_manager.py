from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal

from opentelemetry.trace import StatusCode, Tracer
from pydantic import ValidationError
from typing_extensions import deprecated

from . import messages as _messages
from ._instrumentation import InstrumentationNames, get_agent_run_baggage_attributes
from ._output import (
    OutputSchema,
    OutputToolset,
    run_output_process_hooks,
    run_output_validate_hooks,
)
from ._run_context import AgentDepsT, RunContext
from .exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    SkipToolValidation,
    ToolRetryError,
    UnexpectedModelBehavior,
)
from .messages import ToolCallPart, ToolReturn
from .tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition, ToolDenied
from .toolsets.abstract import AbstractToolset, ToolsetTool
from .usage import RunUsage

if TYPE_CHECKING:
    from .capabilities.abstract import AbstractCapability

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
    """The validated arguments if validation passed, `None` otherwise.

    For regular tool calls, always a `dict[str, Any]` matching the tool schema. For
    output tool calls, this holds what the tool's `args_validator` produced — a dict
    for primitive / multi-arg outputs (e.g. `{'response': 42}`), or the model instance
    for bare `BaseModel` outputs (the dict typing is a mild lie in that case, preserved
    for consistency with regular tool calls). Output-tool semantic unwrapping happens
    inside `execute_output_tool_call` at the output hook boundary, not here.
    """
    validation_error: ToolRetryError | None = None
    """The validation error if validation failed, None otherwise."""


@dataclass
class ToolManager(Generic[AgentDepsT]):
    """Manages tools for an agent run step. It caches the agent run's toolset's tool definitions and handles calling tools and retries."""

    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provides the tools for this run step."""
    root_capability: AbstractCapability[AgentDepsT] | None = None
    """The root capability for hook invocation."""
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

        toolset = await self.toolset.for_run_step(ctx)

        new_tm = self.__class__(
            toolset=toolset,
            root_capability=self.root_capability,
            ctx=ctx,
            tools=await toolset.get_tools(ctx),
            default_max_retries=self.default_max_retries,
        )
        # Make the prepared ToolManager accessible from RunContext so that
        # wrapper toolsets (e.g. CodeModeToolset) can dispatch tool calls
        # through the standard validation/execution path.
        ctx.tool_manager = new_tm
        return new_tm

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
        args_override: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Validate tool arguments using Pydantic schema and custom args_validator_func.

        Returns:
            The validated arguments as a dictionary.

        Raises:
            ValidationError: If argument validation fails.
            ModelRetry: If argument validation fails with a retry request.
        """
        raw_args = args_override if args_override is not None else call.args
        pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
        validator = tool.args_validator
        if isinstance(raw_args, str):
            args_dict = validator.validate_json(
                raw_args or '{}', allow_partial=pyd_allow_partial, context=ctx.validation_context
            )
        else:
            args_dict = validator.validate_python(
                raw_args or {}, allow_partial=pyd_allow_partial, context=ctx.validation_context
            )

        if tool.args_validator_func is not None:
            result = tool.args_validator_func(ctx, **args_dict)
            if inspect.isawaitable(result):
                await result

        return args_dict

    async def _run_validate_hooks(
        self,
        call: ToolCallPart,
        tool: ToolsetTool[AgentDepsT],
        ctx: RunContext[AgentDepsT],
        *,
        allow_partial: bool,
    ) -> dict[str, Any]:
        """Run validation with before/wrap/after tool_validate hooks."""
        cap = self.root_capability

        async def do_validate(args: str | dict[str, Any]) -> dict[str, Any]:
            # Update call.args with the (possibly modified) args before validation
            validated = await self._validate_tool_args(call, tool, ctx, allow_partial=allow_partial, args_override=args)
            return validated

        # Output tools are internal — they don't fire user-facing tool hooks, matching how
        # `WrapperToolset` and `prepare_tools` exclude them.
        if cap is not None and tool.tool_def.kind != 'output':
            tool_def = tool.tool_def

            # before_tool_validate
            raw_args: str | dict[str, Any] = call.args if call.args is not None else {}
            raw_args = await cap.before_tool_validate(ctx, call=call, tool_def=tool_def, args=raw_args)

            # wrap_tool_validate wraps the validation; on_tool_validate_error on failure
            try:
                validated_args = await cap.wrap_tool_validate(
                    ctx, call=call, tool_def=tool_def, args=raw_args, handler=do_validate
                )
            except (ValidationError, ModelRetry) as e:
                validated_args = await cap.on_tool_validate_error(
                    ctx, call=call, tool_def=tool_def, args=raw_args, error=e
                )

            # after_tool_validate
            validated_args = await cap.after_tool_validate(ctx, call=call, tool_def=tool_def, args=validated_args)
        else:
            validated_args = await do_validate(call.args if call.args is not None else {})

        return validated_args

    async def _run_execute_hooks(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        usage: RunUsage,
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Run execution with before/wrap/after tool_execute hooks."""
        assert validated.tool is not None
        assert validated.validated_args is not None

        cap = self.root_capability
        call = validated.call
        ctx = validated.ctx

        async def do_execute(args: dict[str, Any]) -> Any:
            # Execute with potentially modified args
            modified_validated = replace(validated, validated_args=args)
            return await self._raw_execute(
                modified_validated, usage=usage, wrap_validation_errors=wrap_validation_errors
            )

        # Output tools are internal — they don't fire user-facing tool hooks, matching how
        # `WrapperToolset` and `prepare_tools` exclude them.
        if cap is not None and validated.tool.tool_def.kind != 'output':
            tool_def = validated.tool.tool_def

            try:
                # before_tool_execute
                args = await cap.before_tool_execute(ctx, call=call, tool_def=tool_def, args=validated.validated_args)

                # wrap_tool_execute wraps the execution; on_tool_execute_error on failure
                try:
                    tool_result = await cap.wrap_tool_execute(
                        ctx, call=call, tool_def=tool_def, args=args, handler=do_execute
                    )
                except (SkipToolExecution, CallDeferred, ApprovalRequired, ToolRetryError):
                    raise  # Control flow, not errors
                except ModelRetry:
                    raise  # Propagate to outer handler
                except Exception as e:
                    tool_result = await cap.on_tool_execute_error(ctx, call=call, tool_def=tool_def, args=args, error=e)

                # after_tool_execute
                tool_result = await cap.after_tool_execute(
                    ctx, call=call, tool_def=tool_def, args=args, result=tool_result
                )
            except (ValidationError, ModelRetry) as e:
                # Hook raised ValidationError or ModelRetry (e.g. before/after_tool_execute
                # doing additional Pydantic validation on args/result) — convert to
                # ToolRetryError for retry handling, unless the caller asked for raw errors.
                if not wrap_validation_errors:
                    raise
                name = call.tool_name
                self._check_max_retries(name, validated.tool.max_retries, e)
                self.failed_tools.add(name)
                raise self._wrap_error_as_retry(name, call, e) from e
        else:
            tool_result = await do_execute(validated.validated_args)

        return tool_result

    def _resolve_tool(self, call: ToolCallPart) -> tuple[str, ToolsetTool[AgentDepsT]]:
        """Resolve tool name to ResolvedTool, raising ModelRetry for unknown tools."""
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        name = call.tool_name
        tool = self.tools.get(name)
        if tool is None:
            if self.tools:
                msg = f'Available tools: {", ".join(f"{n!r}" for n in self.tools)}'
            else:
                msg = 'No tools available.'
            raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')
        return name, tool

    def _make_validation_success(
        self,
        call: ToolCallPart,
        tool: ToolsetTool[AgentDepsT] | None,
        ctx: RunContext[AgentDepsT],
        validated_args: dict[str, Any] | None,
    ) -> ValidatedToolCall[AgentDepsT]:
        """Build a successful `ValidatedToolCall`. Counterpart to `_make_validation_failure`."""
        return ValidatedToolCall(
            call=call,
            tool=tool,
            ctx=ctx,
            args_valid=True,
            validated_args=validated_args,
            validation_error=None,
        )

    def _make_validation_failure(
        self,
        name: str,
        call: ToolCallPart,
        tool: ToolsetTool[AgentDepsT] | None,
        ctx: RunContext[AgentDepsT],
        error: ToolRetryError | ValidationError | ModelRetry,
    ) -> ValidatedToolCall[AgentDepsT]:
        """Handle validation failure: check retries, mark failed, wrap error.

        Only called when wrapping is requested (`wrap_validation_errors=True`); when
        False (streaming, or sandboxed callers that want raw errors), the caller lets
        the exception propagate without going through this helper.
        """
        max_retries = tool.max_retries if tool is not None else self.default_max_retries
        cause = (
            error.__cause__ if isinstance(error, ToolRetryError) and isinstance(error.__cause__, Exception) else error
        )
        self._check_max_retries(name, max_retries, cause)
        self.failed_tools.add(name)
        validation_error = error if isinstance(error, ToolRetryError) else self._wrap_error_as_retry(name, call, error)
        return ValidatedToolCall(
            call=call,
            tool=tool,
            ctx=ctx,
            args_valid=False,
            validated_args=None,
            validation_error=validation_error,
        )

    async def validate_tool_call(
        self,
        call: ToolCallPart,
        *,
        approved: bool = False,
        metadata: Any = None,
        wrap_validation_errors: bool = True,
    ) -> ValidatedToolCall[AgentDepsT]:
        """Validate tool arguments without executing the tool.

        This method validates arguments BEFORE the tool is executed, allowing the caller to:
        1. Emit FunctionToolCallEvent with accurate `args_valid` status
        2. Handle validation failures differently from execution failures
        3. Decide whether to execute or defer based on validation result

        Args:
            call: The tool call part to validate.
            approved: Whether the tool call has been approved.
            metadata: Additional metadata from DeferredToolResults.metadata.
            wrap_validation_errors: If True (default), wrap `ValidationError` / `ModelRetry`
                as `ToolRetryError` on the returned `ValidatedToolCall.validation_error`,
                count the call against the retry budget, and add it to `failed_tools`.
                If False, propagate the raw `ValidationError` / `ModelRetry` and leave
                retry-budget state untouched — useful for nested callers (e.g. sandboxed
                tool dispatch) where validation failures shouldn't consume the agent's
                retry budget and the raw exception is what the caller wants to surface.

        Returns:
            ValidatedToolCall with validation results, ready for execution via execute_tool_call().
        """
        assert self.ctx is not None
        ctx = self.ctx
        tool: ToolsetTool[AgentDepsT] | None = None

        try:
            _name, tool = self._resolve_tool(call)
            ctx = self._build_tool_context(call, tool, allow_partial=False, approved=approved, metadata=metadata)
            validated_args = await self._run_validate_hooks(call, tool, ctx, allow_partial=False)
            return self._make_validation_success(call, tool, ctx, validated_args)
        except SkipToolValidation as e:
            assert tool is not None
            # Hook asked us to skip validation entirely; accept the args it provided.
            return self._make_validation_success(call, tool, ctx, e.validated_args)
        except (ValidationError, ModelRetry) as e:
            if not wrap_validation_errors:
                raise
            return self._make_validation_failure(call.tool_name, call, tool, ctx, e)

    async def execute_tool_call(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute a validated tool call, within a trace span for function tools.

        Args:
            validated: The validation result from validate_tool_call().
            wrap_validation_errors: If True (default), `ModelRetry` raised by the tool
                body or by execute-stage capability hooks (`before_tool_execute`,
                `after_tool_execute`, `wrap_tool_execute`) is wrapped as `ToolRetryError`
                after counting against the retry budget. If False, the raw
                `ModelRetry` / `ValidationError` propagates and retry-budget state is
                left untouched.

        Returns:
            The tool result if validation passed and execution succeeded.

        Raises:
            ToolRetryError: If validation failed (contains the retry prompt) or the tool
                raised `ModelRetry`. Only when `wrap_validation_errors=True`.
            ModelRetry / ValidationError: When `wrap_validation_errors=False`.
            RuntimeError: If trying to execute an external tool.
        """
        if self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        return await self._execute_function_tool_call(
            validated,
            tracer=self.ctx.tracer,
            include_content=self.ctx.trace_include_content,
            instrumentation_version=self.ctx.instrumentation_version,
            usage=self.ctx.usage,
            wrap_validation_errors=wrap_validation_errors,
        )

    # --- Output tool methods (output hooks, no tool hooks) ---

    async def validate_output_tool_call(
        self,
        call: ToolCallPart,
        *,
        schema: OutputSchema[Any],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> ValidatedToolCall[AgentDepsT]:
        """Validate output tool args through output validate hooks (skipping tool hooks).

        Output tools use output hooks for validation instead of tool hooks. The Pydantic
        schema validation is used as the inner handler wrapped by output validate hooks.

        `schema` is the run's output schema; it's forwarded to
        [`OutputContext`][pydantic_ai.output.OutputContext] so hooks can see the full shape
        of what the schema accepts.

        Raises:
            UnexpectedModelBehavior: If max retries exceeded.
        """
        assert self.ctx is not None
        # Output tool names are pre-classified by _classify_tool_calls, so _resolve_tool
        # should never fail here. The assert documents this invariant.
        name, tool = self._resolve_tool(call)
        assert isinstance(tool.toolset, OutputToolset), f'Expected output tool, got {type(tool.toolset).__name__}'
        ctx = self._build_tool_context(call, tool, allow_partial=allow_partial)

        toolset = tool.toolset
        processor = toolset.processors[name]
        output_context = processor.get_output_context(schema, mode='tool', tool_call=call, tool_def=tool.tool_def)

        # Output hooks see the semantic value (what the model was asked to produce), not the
        # internal dict-wrapped form. This differs from tool call validation hooks, which see
        # `dict[str, Any]` tool args — the schema contract the model satisfies.
        # `processor.hook_validate` runs Pydantic validation and unwraps; output tools are
        # always `ObjectOutputProcessor` (never union), so the opaque state is always `None`.
        async def do_validate(args: str | dict[str, Any]) -> Any:
            semantic, _state = processor.hook_validate(args, run_context=ctx, allow_partial=allow_partial)
            return semantic

        cap = self.root_capability
        assert cap is not None, 'validate_output_tool_call requires root_capability'

        try:
            raw_args: str | dict[str, Any] = call.args if call.args is not None else {}
            semantic_value = await run_output_validate_hooks(
                cap,
                run_context=ctx,
                output_context=output_context,
                output=raw_args,
                do_validate=do_validate,
                allow_partial=allow_partial,
                wrap_validation_errors=wrap_validation_errors,
            )
            # Rewrap the (possibly hook-modified) semantic value into the dict shape that
            # matches the tool's schema — `ValidatedToolCall.validated_args` is the
            # schema-contract form, consistent with regular tool calls. The semantic
            # unwrap happens again in `execute_output_tool_call` at the output hook boundary.
            # No unwrap key → `validated_args` holds the validated object itself (e.g. a
            # `BaseModel` instance); typed as `dict[str, Any] | None` for consistency with
            # tool calls, matching pre-refactor behavior.
            if (k := processor.hook_unwrap_key) is not None:
                validated_args: dict[str, Any] | None = {k: semantic_value}
            else:
                validated_args = semantic_value
            return self._make_validation_success(call, tool, ctx, validated_args)
        except (ToolRetryError, ValidationError, ModelRetry) as e:
            if not wrap_validation_errors:
                raise
            return self._make_validation_failure(name, call, tool, ctx, e)

    async def execute_output_tool_call(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        schema: OutputSchema[Any],
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute output tool through output process hooks (skipping tool hooks).

        Output validators run inside process hooks (inside wrap_output_process), ensuring
        the complete output pipeline is wrapped. Validators see the global output retry
        context (from self.ctx), not the per-tool context, matching the text output path.

        `schema` is the run's output schema; it's forwarded to
        [`OutputContext`][pydantic_ai.output.OutputContext] so hooks can see the full shape
        of what the schema accepts.

        Raises:
            ToolRetryError: If execution or output validation fails.
            UnexpectedModelBehavior: If max retries exceeded.
        """
        assert validated.args_valid
        assert validated.tool is not None
        # validated_args may be None for `output_type=int | None` (legitimate semantic value),
        # so we rely on args_valid above rather than asserting validated_args is not None
        assert self.ctx is not None

        name = validated.call.tool_name
        toolset = validated.tool.toolset
        assert isinstance(toolset, OutputToolset)

        tool = validated.tool
        processor = toolset.processors[name]
        output_context = processor.get_output_context(
            schema, mode='tool', tool_call=validated.call, tool_def=tool.tool_def
        )

        # Unwrap the dict-shaped `validated_args` back to the semantic value that output hooks
        # see. Inverse of the rewrap in `validate_output_tool_call`. For `BaseModel` outputs,
        # `validated_args` already holds the instance (no unwrap key), so this is a passthrough.
        if (k := processor.hook_unwrap_key) is not None:
            assert isinstance(validated.validated_args, dict)
            semantic_value: Any = validated.validated_args[k]
        else:
            semantic_value = validated.validated_args

        # Output validators see the *global* output-retry budget (`max_result_retries`), so the same
        # validator stays consistent across the text path and across multiple `ToolOutput`s. Output
        # functions, by contrast, see the *per-tool* `tool.max_retries` (the post-#4687 override) on
        # `validated.ctx`. Termination on the tool path checks `retries[name] == tool.max_retries`
        # (see `_check_max_retries` below), so when `ToolOutput(max_retries=N)` exceeds
        # `max_result_retries`, the validator's `ctx.last_attempt` can fire before the run actually
        # terminates. Tracked in #5238 — revisiting cleanly needs broader thought about
        # `ctx.retry`/`ctx.retries[name]` semantics and is intentionally out of scope here.
        assert toolset.max_retries is not None
        validator_ctx = replace(validated.ctx, retry=self.ctx.retry, max_retries=toolset.max_retries)

        async def do_process(output: Any) -> Any:
            # `processor.hook_execute` re-wraps the semantic value into the dict shape
            # `processor.call()` expects, then runs the output function (if any).
            # Output tools are always `ObjectOutputProcessor` (never union), so `state` is `None`.
            try:
                result = await processor.hook_execute(
                    output, None, run_context=validated.ctx, wrap_validation_errors=False
                )
            except ModelRetry:
                # When wrap_validation_errors=True, run_output_process_hooks below wraps
                # ModelRetry as ToolRetryError (caught by the outer handler for retry tracking).
                # When False (streaming, see result.py:validate_response_output), ModelRetry
                # must propagate unwrapped so the streaming handler can catch it.
                raise
            # Output validators run inside do_process so wrap_output_process wraps the
            # complete pipeline. Validators use wrap_validation_errors=False — the outer
            # run_output_process_hooks handles wrapping ModelRetry as ToolRetryError.
            for validator in toolset.output_validators:
                result = await validator.validate(result, validator_ctx)
            return result

        cap = self.root_capability
        assert cap is not None, 'execute_output_tool_call requires root_capability'
        try:
            result = await run_output_process_hooks(
                cap,
                run_context=validated.ctx,
                output_context=output_context,
                output=semantic_value,
                do_process=do_process,
                wrap_validation_errors=wrap_validation_errors,
            )
        except ToolRetryError as e:
            cause = e.__cause__ if isinstance(e.__cause__, Exception) else e
            self._check_max_retries(name, tool.max_retries, cause)
            self.failed_tools.add(name)
            raise

        return result

    async def handle_output_tool_call(
        self,
        call: ToolCallPart,
        *,
        schema: OutputSchema[Any],
        allow_partial: bool = False,
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Handle an output tool call using output hooks (not tool hooks).

        Convenience method combining validate_output_tool_call and execute_output_tool_call.
        Used by the streaming path in result.py.
        """
        validated = await self.validate_output_tool_call(
            call,
            schema=schema,
            allow_partial=allow_partial,
            wrap_validation_errors=wrap_validation_errors,
        )
        if not validated.args_valid:  # pragma: no cover — caller (result.py) uses wrap_validation_errors=False
            assert validated.validation_error is not None
            raise validated.validation_error
        return await self.execute_output_tool_call(
            validated,
            schema=schema,
            wrap_validation_errors=wrap_validation_errors,
        )

    async def _execute_tool_call_impl(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        usage: RunUsage,
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute a validated tool call without tracing, with capability hooks.

        `wrap_validation_errors` here only governs errors raised *during* execution
        (tool body or execute-stage hooks). A `ValidatedToolCall` that already failed
        validation carries a pre-wrapped `ToolRetryError`; raw-mode callers get raw
        errors at the `validate_tool_call(wrap_validation_errors=False)` boundary.

        Raises ToolRetryError if validation previously failed or the tool raises ModelRetry
        (when `wrap_validation_errors=True`); when False, ModelRetry from the tool body
        or hooks propagates raw. Raises UnexpectedModelBehavior if max retries exceeded.
        """
        # Asserts narrow types for pyright; invariants guaranteed by ValidatedToolCall construction
        if not validated.args_valid:
            assert validated.validation_error is not None
            raise validated.validation_error

        assert validated.tool is not None
        assert validated.validated_args is not None

        if validated.tool.tool_def.kind == 'external':
            raise RuntimeError('External tools cannot be called')

        try:
            tool_result = await self._run_execute_hooks(
                validated, usage=usage, wrap_validation_errors=wrap_validation_errors
            )
        except SkipToolExecution as e:
            usage.tool_calls += 1
            return e.result

        return tool_result

    async def _raw_execute(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        usage: RunUsage,
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute a validated tool call without hooks or tracing."""
        assert validated.tool is not None
        assert validated.validated_args is not None

        name = validated.call.tool_name

        try:
            tool_result = await self.toolset.call_tool(
                name,
                validated.validated_args,
                validated.ctx,
                validated.tool,
            )
        except ModelRetry as e:
            if not wrap_validation_errors:
                raise
            self._check_max_retries(name, validated.tool.max_retries, e)
            self.failed_tools.add(name)
            raise self._wrap_error_as_retry(name, validated.call, e) from e

        usage.tool_calls += 1

        return tool_result

    async def _execute_function_tool_call(
        self,
        validated: ValidatedToolCall[AgentDepsT],
        *,
        tracer: Tracer,
        include_content: bool,
        instrumentation_version: int,
        usage: RunUsage,
        wrap_validation_errors: bool = True,
    ) -> Any:
        """Execute a validated function tool call within a trace span.

        See <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span>.
        """
        instrumentation_names = InstrumentationNames.for_version(instrumentation_version)
        call = validated.call

        span_attributes = {
            'gen_ai.operation.name': 'execute_tool',
            'gen_ai.tool.name': call.tool_name,
            # NOTE: this means `gen_ai.tool.call.id` will be included even if it was generated by pydantic-ai
            'gen_ai.tool.call.id': call.tool_call_id,
            **({instrumentation_names.tool_arguments_attr: call.args_as_json_str()} if include_content else {}),
            **get_agent_run_baggage_attributes(),
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
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                tool_result = await self._execute_tool_call_impl(
                    validated, usage=usage, wrap_validation_errors=wrap_validation_errors
                )
                if include_content and span.is_recording():
                    span.set_attribute(
                        instrumentation_names.tool_result_attr,
                        tool_result
                        if isinstance(tool_result, str)
                        else _messages.tool_return_ta.dump_json(tool_result).decode(),
                    )
            except (CallDeferred, ApprovalRequired) as exc:
                span.set_attribute(instrumentation_names.tool_deferral_name_attr, type(exc).__name__)
                if include_content and span.is_recording() and exc.metadata is not None:
                    try:
                        metadata_str = json.dumps(exc.metadata)
                    except (TypeError, ValueError):
                        metadata_str = repr(exc.metadata)
                    span.set_attribute(instrumentation_names.tool_deferral_metadata_attr, metadata_str)
                if instrumentation_version < 5:
                    span.record_exception(exc, escaped=True)
                    span.set_status(StatusCode.ERROR)
                raise
            except ToolRetryError as e:
                part = e.tool_retry
                if include_content and span.is_recording():
                    span.set_attribute(instrumentation_names.tool_result_attr, part.model_response())
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise
            except BaseException as e:
                span.record_exception(e, escaped=True)
                span.set_status(StatusCode.ERROR)
                raise

        return tool_result

    async def handle_call(
        self,
        call: ToolCallPart,
        *,
        approved: bool = False,
        metadata: Any = None,
        wrap_validation_errors: bool = True,
    ) -> ToolDenied | ToolReturn[Any] | Any:
        """Handle a tool call by validating the arguments, calling the tool, and handling retries.

        This is a convenience method that combines validate_tool_call() and execute_tool_call().

        If the tool raises [`ApprovalRequired`][pydantic_ai.exceptions.ApprovalRequired] or
        [`CallDeferred`][pydantic_ai.exceptions.CallDeferred], the capability handler
        (if any) is invoked to resolve it inline; otherwise the exception propagates.

        Args:
            call: The tool call part to handle.
            approved: Whether the tool call has been approved.
            metadata: Additional metadata from DeferredToolResults.metadata.
            wrap_validation_errors: If True (default), validation failures surface as
                `ToolRetryError` (after counting against the retry budget). If False,
                the raw `ValidationError` / `ModelRetry` propagates and retry-budget
                state is left untouched — useful for nested callers (e.g. sandboxed
                tool dispatch) where the call shouldn't consume the agent's retry
                budget and the raw exception is what the caller wants to surface.

        Returns:
            The tool's return value on success — possibly a [`ToolReturn`][pydantic_ai.messages.ToolReturn]
            wrapper if the tool or handler supplied one.

            A [`ToolDenied`][pydantic_ai.tools.ToolDenied] instance if a
            [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]
            handler denied the call. **Callers must `isinstance`-check the result**
            before treating it as a successful tool return — `ToolDenied` is *not* a
            valid tool result and the message string alone is indistinguishable from
            a real return value. Surfacing the denial (e.g. recording
            `ToolReturnPart(outcome='denied')` in message history, or raising inside
            a sandbox) is the caller's responsibility.

        Raises:
            ToolRetryError: The handler requested a retry, or the (re-)executed tool
                raised `ModelRetry`. Only when `wrap_validation_errors=True`.
            ValidationError / ModelRetry: When `wrap_validation_errors=False` and the
                arguments fail validation or a hook raises `ModelRetry`.
            CallDeferred / ApprovalRequired: No handler resolved the call, or the
                approved tool re-raised a deferral.
        """
        validated = await self.validate_tool_call(
            call,
            approved=approved,
            metadata=metadata,
            wrap_validation_errors=wrap_validation_errors,
        )
        try:
            return await self.execute_tool_call(validated, wrap_validation_errors=wrap_validation_errors)
        except (CallDeferred, ApprovalRequired) as exc:
            return await self._resolve_single_deferred(call, exc, wrap_validation_errors=wrap_validation_errors)

    async def resolve_deferred_tool_calls(
        self,
        requests: DeferredToolRequests,
    ) -> DeferredToolResults | None:
        """Invoke the capability handler to resolve deferred tool calls.

        Args:
            requests: The deferred tool requests to resolve.

        Returns:
            `DeferredToolResults` with results for some or all calls, or `None` if
            no handler is available or the handler declined to handle the requests.
        """
        if self.root_capability is None or self.ctx is None:
            return None  # pragma: no cover
        return await self.root_capability.handle_deferred_tool_calls(self.ctx, requests=requests)

    async def _resolve_single_deferred(
        self,
        call: ToolCallPart,
        exc: CallDeferred | ApprovalRequired,
        *,
        wrap_validation_errors: bool = True,
    ) -> ToolDenied | ToolReturn[Any] | Any:
        """Resolve a single deferred tool call inline using the capability handler.

        Dispatches the handler's result for `call` on the same set of
        [`DeferredToolResult`][pydantic_ai.tools.DeferredToolResult] variants as the
        batch path in `_agent_graph._call_tool`, but returns a raw tool-like value
        (what the tool "would have returned") rather than a message-history part.

        NOTE: keep the dispatch branches here in sync with
        [`_call_tool`][pydantic_ai._agent_graph._call_tool] — both paths must accept the
        full [`DeferredToolResult`][pydantic_ai.tools.DeferredToolResult] surface.

        `wrap_validation_errors` is forwarded to the post-approval re-validation and
        re-execution so callers passing `False` (e.g. sandboxed dispatch) keep the
        same raw-error contract through deferred-tool resolution. Handler-constructed
        retry signals (`ModelRetry` / `RetryPromptPart` returned by the handler) still
        surface as `ToolRetryError` regardless — those are handler outputs, not
        exceptions raised by validation or the tool body.

        Returns:
            For approved calls, the raw tool return (possibly a `ToolReturn` wrapper).
            For external-call results, the value the handler supplied verbatim (plain
            value or `ToolReturn`).
            For denied calls, the [`ToolDenied`][pydantic_ai.tools.ToolDenied] instance
            from the handler — callers must `isinstance`-check before treating the
            return value as a successful tool result.

        Raises:
            ToolRetryError: Handler requested a retry via `ModelRetry` or `RetryPromptPart`,
                or the approved tool re-raised `ModelRetry` (only when
                `wrap_validation_errors=True`).
            ValidationError / ModelRetry: When `wrap_validation_errors=False` and the
                approved tool's re-validation fails or its body raises `ModelRetry`.
            CallDeferred / ApprovalRequired: Handler couldn't resolve the call, or the
                approved tool re-raised a deferral.
        """
        requests = DeferredToolRequests(
            approvals=[call] if isinstance(exc, ApprovalRequired) else [],
            calls=[call] if isinstance(exc, CallDeferred) else [],
            metadata={call.tool_call_id: exc.metadata} if exc.metadata else {},
        )
        deferred_results = await self.resolve_deferred_tool_calls(requests)
        if deferred_results is None:
            raise exc

        # Normalize via to_tool_call_results(): bool → ToolApproved/ToolDenied,
        # plain external values → ToolReturn(value).
        tool_call_result = deferred_results.to_tool_call_results().get(call.tool_call_id)
        if tool_call_result is None:
            raise exc

        if isinstance(tool_call_result, ToolDenied):
            # Surface the denial as a return value, not an exception. Callers must
            # `isinstance`-check the result of `handle_call` to distinguish a denial
            # from a successful tool return.
            return tool_call_result
        if isinstance(tool_call_result, ToolApproved):
            validate_call = call
            if tool_call_result.override_args is not None:
                validate_call = replace(call, args=tool_call_result.override_args)
            call_metadata = deferred_results.metadata.get(call.tool_call_id)
            validated = await self.validate_tool_call(
                validate_call,
                approved=True,
                metadata=call_metadata,
                wrap_validation_errors=wrap_validation_errors,
            )
            return await self.execute_tool_call(validated, wrap_validation_errors=wrap_validation_errors)
        if isinstance(tool_call_result, ModelRetry):
            raise ToolRetryError(
                _messages.RetryPromptPart(
                    content=tool_call_result.message,
                    tool_name=call.tool_name,
                    tool_call_id=call.tool_call_id,
                )
            )
        if isinstance(tool_call_result, _messages.RetryPromptPart):
            tool_call_result.tool_name = call.tool_name
            tool_call_result.tool_call_id = call.tool_call_id
            raise ToolRetryError(tool_call_result)
        # Must be a ToolReturn (the only remaining DeferredToolResult variant). Return
        # the handler's original value verbatim so handle_call's contract — "what the
        # tool would have returned" — is preserved for plain-vs-wrapped inputs.
        return deferred_results.calls[call.tool_call_id]
