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

from pydantic_ai._tool_usage_policy import ToolPolicy

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
    tool_use_policy: ToolPolicy | None = None
    """Tool use policy configured on the Agent or the run methods"""

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

    def _get_max_uses_of_tool(self, tool_name: str) -> int | None:
        """Get the maximum number of uses allowed for a given tool, or `None` if unlimited."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        tool = self.tools.get(tool_name, None)
        if tool is None or tool.usage_policy is None:
            return None

        return tool.usage_policy.max_uses

    def _get_current_uses_of_tool(self, tool_name: str) -> int:
        """Get the current number of uses of a given tool."""
        ctx = self._assert_ctx()
        return ctx.tools_use_counts.get(tool_name, 0)

    def _get_max_uses_per_step_of_tool(self, tool_name: str) -> int | None:
        """Get the maximum number of uses allowed for a given tool within a step, or `None` if unlimited."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover
        if (
            (tool := self.tools.get(tool_name)) is not None
            and (usage_policy := tool.usage_policy) is not None
            and (max_uses_per_step := usage_policy.max_uses_per_step) is not None
        ):
            return max_uses_per_step
        return None


    def check_tool_call_allowed(
        self,
        tool_name: str,
        *,
        tool_accepted_in_step: int,
        projected_tool_uses: int,
        tool_calls_in_step: int,
        projected_total_tools_uses: int
    ) -> str | None:
        """Check if a tool call is allowed based on per-tool limits.

        This is the check for partial execution. It checks per-tool limits (max_uses for this specific tool).

        Args:
            tool_name: The name of the tool to check.
            tool_accepted_in_step: Number of times this specific tool was accepted in the batch.
            projected_tool_uses: The projected number of uses of the tool in this step.
            tool_calls_in_step:
            projected_total_tools_uses:


        Returns:
            None if the call is allowed.
            A string error message if the call should be rejected.
        """
        # TODO: I have no idea how this logic went through but damn it is ugly need to read through and fix this signature to be more understanable
        # I have completely ignored the mode as of now, need to integrate that logic here to allow for flexibility
        # Also need to consider other Douwe suggestions on retries deprecating into tool and model retries I think? I'll need to read those parts as well
        #
        # Will think about it after I am done with mode part
        # Refactor and clean this mess up lol?
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        # For unknown tools, allow the call - error will be caught during execution
        # This provides a better error message than "tool limit reached" which would technically be incorrect.
        if tool_name not in self.tools:
            return None

        # Per-run limits
        agent_tool_use_policy = self.tool_use_policy
        will_batch_exceed = False

        if agent_tool_use_policy:
            if agent_tool_use_policy.max_uses_per_step:
                will_batch_exceed = will_batch_exceed or tool_calls_in_step > agent_tool_use_policy.max_uses_per_step
            if agent_tool_use_policy.max_uses:
                will_batch_exceed = will_batch_exceed or projected_total_tools_uses > agent_tool_use_policy.max_uses

        # Per-tool limits
        current_tool_uses = self._get_current_uses_of_tool(tool_name)
        max_uses = self._get_max_uses_of_tool(tool_name)
        max_uses_per_step = self._get_max_uses_per_step_of_tool(tool_name)

        # Check entire step for tool first - if the batch will exceed per-tool limits

        if (max_uses_per_step is not None and projected_tool_uses > max_uses_per_step) or (
            max_uses is not None and projected_tool_uses + current_tool_uses > max_uses
        ) or (will_batch_exceed):
            # If limits would be exceeded and partial execution is not allowed, reject all calls.
            # The tool's ToolPolicy must have partial_execution != False (None means inherit default True)
            tool = self.tools.get(tool_name)
            # Tool allows partial if: no tool, no usage_policy (inherits default True),
            # or usage_policy.partial_execution is not explicitly False
            tool_allows_partial = (agent_tool_use_policy is None or agent_tool_use_policy.partial_execution) and (
                tool is None  # Unknown tool - allow through, will fail later with proper error
                or tool.usage_policy is None  # No policy on tool - inherits default True behavior
                or tool.usage_policy.partial_execution is not False  # None means inherit default True
            )
            if not tool_allows_partial:
                if agent_tool_use_policy and agent_tool_use_policy.mode == 'error':
                    raise UsageLimitExceeded(message=f'Tool use limit reached for tool "{tool_name}".')
                # TODO: Should be configurable via PromptConfig #3656
                return f'Tool use limit reached for tool "{tool_name}".'

        # Check incremental call for tool

        if (max_uses_per_step is not None) and (tool_accepted_in_step == max_uses_per_step):
            if agent_tool_use_policy and agent_tool_use_policy.mode == 'error':
                raise UsageLimitExceeded(message=f'Tool use limit reached for tool "{tool_name}".')
            # If already equal, going through with this call will put us over the limit
            # TODO: Should be configurable via PromptConfig #3656
            return f'Tool use limit reached for tool "{tool_name}".'

        if (max_uses is not None) and (current_tool_uses + tool_accepted_in_step == max_uses):
            if agent_tool_use_policy and agent_tool_use_policy.mode == 'error':
                raise UsageLimitExceeded(message=f'Tool use limit reached for tool "{tool_name}".')
            # If already equal, going through with this call will put us over the limit
            # TODO: Should be configurable via PromptConfig #3656
            return f'Tool use limit reached for tool "{tool_name}".'

        return None

    def _assert_ctx(self) -> RunContext[AgentDepsT]:
        if self.ctx is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover
        return self.ctx
