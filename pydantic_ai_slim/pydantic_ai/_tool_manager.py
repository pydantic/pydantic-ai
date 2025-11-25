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

from . import messages as _messages
from ._instrumentation import InstrumentationNames
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior
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
            ctx = replace(ctx, retries=retries)

        return self.__class__(
            toolset=self.toolset,
            ctx=ctx,
            tools=await self.toolset.get_tools(ctx),
        )

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        """The tool definitions for the tools in this tool manager."""
        if self.tools is None:
            raise ValueError('ToolManager has not been prepared for a run step yet')  # pragma: no cover

        return [tool.tool_def for tool in self.tools.values()]

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
    ) -> Any:
        """Handle a tool call by validating the arguments, calling the tool, and handling retries.

        Args:
            call: The tool call part to handle.
            allow_partial: Whether to allow partial validation of the tool arguments.
            wrap_validation_errors: Whether to wrap validation errors in a retry prompt part.
            approved: Whether the tool call has been approved.
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
            )
        else:
            return await self._call_function_tool(
                call,
                allow_partial=allow_partial,
                wrap_validation_errors=wrap_validation_errors,
                approved=approved,
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
                partial_output=allow_partial,
            )

            pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
            validator = tool.args_validator
            if isinstance(call.args, str):
                args_dict = validator.validate_json(call.args or '{}', allow_partial=pyd_allow_partial)
            else:
                args_dict = validator.validate_python(call.args or {}, allow_partial=pyd_allow_partial)

            result = await self.toolset.call_tool(name, args_dict, ctx, tool)

            return result
        except (ValidationError, ModelRetry) as e:
            max_retries = tool.max_retries if tool is not None else 1
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
