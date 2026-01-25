from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from typing import Any, Generic

from opentelemetry.trace import Tracer
from pydantic import ValidationError
from typing_extensions import assert_never

from pydantic_ai._tool_usage_policy import ToolPolicy, ToolPolicyMode

from . import messages as _messages
from ._instrumentation import InstrumentationNames
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior, UsageLimitExceeded
from .messages import ToolCallPart
from .tools import ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool
from .usage import RunUsage

_sequential_tool_calls_ctx_var: ContextVar[bool] = ContextVar('sequential_tool_calls', default=False)


@dataclass
class ToolManager(Generic[AgentDepsT]):
    """Manages tools for an agent run step. It caches the agent run's toolset's tool definitions and handles calling tools and retries."""

    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provides the tools for this run step."""
    ctx: RunContext[AgentDepsT] | None = None
    """The agent run context for a specific run step."""
    tools: dict[str, ToolsetTool[AgentDepsT]] | None = None
    """The cached tools for this run step."""
    failed_tools: set[str] = field(default_factory=set)
    """Names of tools that failed in this run step."""
    default_max_retries: int = 1
    """Default number of times to retry a tool"""
    tools_use_policy: ToolPolicy | None = None
    """Policy for using tools, configured on the Agent or run methods. Applies to all tools collectively."""

    @classmethod
    @contextmanager
    def sequential_tool_calls(cls) -> Iterator[None]:
        """Run tool calls sequentially during the context."""
        token = _sequential_tool_calls_ctx_var.set(True)
        try:
            yield
        finally:
            _sequential_tool_calls_ctx_var.reset(token)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> ToolManager[AgentDepsT]:
        """Build a new tool manager for the next run step, carrying over the retries from the current run step."""
        if self.ctx is not None:
            if ctx.run_step == self.ctx.run_step:
                return self

            retries = {
                failed_tool_name: self.ctx.retries.get(failed_tool_name, 0) + 1
                for failed_tool_name in self.failed_tools
            }
            tools_use_counts = self.ctx.tools_use_counts.copy()
            ctx = replace(ctx, retries=retries, tools_use_counts=tools_use_counts)

        return self.__class__(
            toolset=self.toolset,
            ctx=ctx,
            tools=await self.toolset.get_tools(ctx),
            default_max_retries=self.default_max_retries,
            tools_use_policy=self.tools_use_policy,
        )

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        """The tool definitions for the tools in this tool manager.

        Tools that have reached their `max_uses` limit (based on successful calls) are filtered out.
        """
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        return [
            tool.tool_def
            for tool in self.tools.values()
            if (limits := tool.usage_policy) is None
            or (max_uses := limits.max_uses) is None
            or (self._get_current_uses_of_tool(tool.tool_def.name) < max_uses)
        ]

    def should_call_sequentially(self, calls: list[ToolCallPart]) -> bool:
        """Whether to require sequential tool calls for a list of tool calls."""
        return _sequential_tool_calls_ctx_var.get() or any(
            tool_def.sequential for call in calls if (tool_def := self.get_tool_def(call.tool_name))
        )

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        """Get the tool definition for a given tool name, or `None` if the tool is unknown."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        try:
            return self.tools[name].tool_def
        except KeyError:
            return None

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

        Args:
            call: The tool call part to handle.
            allow_partial: Whether to allow partial validation of the tool arguments.
            wrap_validation_errors: Whether to wrap validation errors in a retry prompt part.
            approved: Whether the tool call has been approved.
            metadata: Additional metadata from DeferredToolResults.metadata.
        """
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        if (tool := self.tools.get(call.tool_name)) and tool.tool_def.kind == 'output':
            # Output tool calls are not traced and not counted
            return await self._call_tool(
                call,
                allow_partial=allow_partial,
                wrap_validation_errors=wrap_validation_errors,
                approved=approved,
                metadata=metadata,
            )
        else:
            return await self._call_function_tool(
                call,
                allow_partial=allow_partial,
                wrap_validation_errors=wrap_validation_errors,
                approved=approved,
                metadata=metadata,
                tracer=self.ctx.tracer,
                include_content=self.ctx.trace_include_content,
                instrumentation_version=self.ctx.instrumentation_version,
                usage=self.ctx.usage,
            )

    async def _call_tool(
        self,
        call: ToolCallPart,
        *,
        allow_partial: bool,
        wrap_validation_errors: bool,
        approved: bool,
        metadata: Any = None,
    ) -> Any:
        if self.tools is None or self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        name = call.tool_name
        tool = self.tools.get(name)
        try:
            if tool is None:
                if self.tools:
                    msg = f'Available tools: {", ".join(f"{name!r}" for name in self.tools.keys())}'
                else:
                    msg = 'No tools available.'
                raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')

            if tool.tool_def.kind == 'external':
                raise RuntimeError('External tools cannot be called')

            ctx = replace(
                self.ctx,
                tool_name=name,
                tool_call_id=call.tool_call_id,
                retry=self.ctx.retries.get(name, 0),
                max_retries=tool.max_retries,
                tool_call_approved=approved,
                tool_call_metadata=metadata,
                partial_output=allow_partial,
            )

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

            result = await self.toolset.call_tool(name, args_dict, ctx, tool)
            self.ctx.tools_use_counts[name] = self.ctx.tools_use_counts.get(name, 0) + 1
            return result
        except (ValidationError, ModelRetry) as e:
            max_retries = tool.max_retries if tool is not None else self.default_max_retries
            current_retry = self.ctx.retries.get(name, 0)

            if current_retry == max_retries:
                raise UnexpectedModelBehavior(f'Tool {name!r} exceeded max retries count of {max_retries}') from e
            else:
                if wrap_validation_errors:
                    if isinstance(e, ValidationError):
                        m = _messages.RetryPromptPart(
                            tool_name=name,
                            content=e.errors(include_url=False, include_context=False),
                            tool_call_id=call.tool_call_id,
                        )
                        e = ToolRetryError(m)
                    elif isinstance(e, ModelRetry):
                        m = _messages.RetryPromptPart(
                            tool_name=name,
                            content=e.message,
                            tool_call_id=call.tool_call_id,
                        )
                        e = ToolRetryError(m)
                    else:
                        assert_never(e)

                if not allow_partial:
                    # If we're validating partial arguments, we don't want to count this as a failed tool as it may still succeed once the full arguments are received.
                    self.failed_tools.add(name)

                raise e

    async def _call_function_tool(
        self,
        call: ToolCallPart,
        *,
        allow_partial: bool,
        wrap_validation_errors: bool,
        approved: bool,
        metadata: Any = None,
        tracer: Tracer,
        include_content: bool,
        instrumentation_version: int,
        usage: RunUsage,
    ) -> Any:
        """See <https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span>."""
        instrumentation_names = InstrumentationNames.for_version(instrumentation_version)

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
                tool_result = await self._call_tool(
                    call,
                    allow_partial=allow_partial,
                    wrap_validation_errors=wrap_validation_errors,
                    approved=approved,
                    metadata=metadata,
                )
                usage.tool_calls += 1

            except ToolRetryError as e:
                part = e.tool_retry
                if include_content and span.is_recording():
                    span.set_attribute(instrumentation_names.tool_result_attr, part.model_response())
                raise e

            if include_content and span.is_recording():
                span.set_attribute(
                    instrumentation_names.tool_result_attr,
                    tool_result
                    if isinstance(tool_result, str)
                    else _messages.tool_return_ta.dump_json(tool_result).decode(),
                )

        return tool_result

    def _get_current_uses_of_tool(self, tool_name: str) -> int:
        """Get the current number of uses of a given tool."""
        ctx = self._assert_ctx()
        return ctx.tools_use_counts.get(tool_name, 0)

    def _reject_call(
        self, message: str, *, tool_policy: ToolPolicy | None = None, tools_policy: ToolPolicy | None = None
    ) -> str:
        """Reject a tool call. Raises UsageLimitExceeded if mode='error', otherwise returns message for model retry.

        Args:
            message: The rejection message.
            tool_policy: The per-tool policy (takes precedence for mode).
            tools_policy: The agent-level policy (fallback for mode).
        """
        # Per-tool mode takes precedence over agent-level mode
        mode: ToolPolicyMode = 'model_retry'
        if tool_policy:
            mode = tool_policy.mode
        elif tools_policy:
            mode = tools_policy.mode

        if mode == 'error':
            raise UsageLimitExceeded(message=message)
        return message

    def get_batch_rejection_reason(
        self,
        tool_calls_in_batch: int,
        current_total_tool_uses: int,
    ) -> str | None:
        """Get rejection reason for a batch of tool calls, or None if allowed.

        This method should be called once before processing a batch of tool calls.
        It rejects the entire batch early if tools_use_policy limits would be exceeded
        and partial_execution is disabled.

        Args:
            tool_calls_in_batch: Total number of tool calls in the current batch.
            current_total_tool_uses: Total tool calls executed in the run before this batch.

        Returns:
            Rejection message if batch should be rejected, None if allowed.
        """
        if not self.tools_use_policy or self.tools_use_policy.partial_execution is not False:
            return None  # No tools_use_policy or partial execution allowed

        batch_exceeds = (
            self.tools_use_policy.max_uses_per_step is not None
            and tool_calls_in_batch > self.tools_use_policy.max_uses_per_step
        ) or (
            self.tools_use_policy.max_uses is not None
            and current_total_tool_uses + tool_calls_in_batch > self.tools_use_policy.max_uses
        )

        if batch_exceeds:
            return self._reject_call('Tool usage limit reached for this run.', tools_policy=self.tools_use_policy)
        return None

    def get_tool_call_rejection_reason(
        self,
        tool_name: str,
        *,
        tool_accepted_in_step: int,
        projected_tool_uses: int,
        current_total_tool_uses: int,
        tool_calls_executed_in_step: int,
    ) -> str | None:
        """Get rejection reason for a tool call, or None if allowed.

        This method enforces both agent-level and per-tool limits, supporting partial execution
        where some calls in a batch may be accepted while others are rejected.

        Note: Agent-level batch checks with partial_execution=False should be handled by
        get_batch_rejection_reason() before calling this method.

        Agent-level and per-tool partial_execution settings are independent:
        - Agent partial_execution only affects agent-level limits (via get_batch_rejection_reason)
        - Tool partial_execution only affects that tool's limits

        Args:
            tool_name: The name of the tool to check.
            tool_accepted_in_step: Number of times this specific tool was already accepted in this batch.
            projected_tool_uses: The projected number of uses of this tool if all calls in the batch succeed.
            current_total_tool_uses: Total tool calls executed in the run before this batch.
            tool_calls_executed_in_step: Number of tool calls already accepted in this batch (all tools).

        Returns:
            None if the call is allowed.
            A string rejection reason if the call should be rejected.
        """
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        # Phase 0: Unknown tools pass through - let execution provide a better error message
        if tool_name not in self.tools:
            return None

        # Phase 1: Gather policies and current usage state
        tool = self.tools.get(tool_name)
        tool_policy = tool.usage_policy if tool else None
        tools_policy = self.tools_use_policy

        current_uses = self._get_current_uses_of_tool(tool_name)
        max_uses = tool_policy.max_uses if tool_policy else None
        max_uses_per_step = tool_policy.max_uses_per_step if tool_policy else None

        # Phase 2: Per-tool batch pre-check - reject if this tool's projected uses exceed limits and partial_execution=False
        tool_batch_exceeds = (max_uses_per_step is not None and projected_tool_uses > max_uses_per_step) or (
            max_uses is not None and projected_tool_uses + current_uses > max_uses
        )
        # If batch exceeds tool limits and no partial execution allowed, reject all calls for this tool upfront
        if tool_batch_exceeds and tool_policy and tool_policy.partial_execution is False:
            return self._reject_call(
                f'Tool "{tool_name}" has reached its usage limit.', tool_policy=tool_policy, tools_policy=tools_policy
            )

        # Phase 3: Incremental per-tool checks - enforce limits one call at a time (for partial execution)
        # Check if this tool has already been accepted max times in this step
        if max_uses_per_step is not None and tool_accepted_in_step >= max_uses_per_step:
            return self._reject_call(
                f'Tool "{tool_name}" has reached its usage limit.', tool_policy=tool_policy, tools_policy=tools_policy
            )

        # Check if this tool has already been used max times across the entire run
        if max_uses is not None and current_uses + tool_accepted_in_step >= max_uses:
            return self._reject_call(
                f'Tool "{tool_name}" has reached its usage limit.', tool_policy=tool_policy, tools_policy=tools_policy
            )

        # Phase 4: Incremental agent-level checks - enforce agent-wide limits one call at a time
        if tools_policy:
            # Check if we've already accepted max tool calls in this step (across all tools)
            if (
                tools_policy.max_uses_per_step is not None
                and tool_calls_executed_in_step >= tools_policy.max_uses_per_step
            ):
                return self._reject_call(
                    'Tool usage limit reached for this step.', tool_policy=tool_policy, tools_policy=tools_policy
                )

            # Check if we've already used max tool calls across the entire run (across all tools)
            if (
                tools_policy.max_uses is not None
                and current_total_tool_uses + tool_calls_executed_in_step >= tools_policy.max_uses
            ):
                return self._reject_call(
                    'Tool usage limit reached for this run.', tool_policy=tool_policy, tools_policy=tools_policy
                )

        # All checks passed - this call is allowed
        return None

    def _assert_ctx(self) -> RunContext[AgentDepsT]:
        if self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover
        return self.ctx
