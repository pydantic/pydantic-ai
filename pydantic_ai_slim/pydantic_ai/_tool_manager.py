from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Generic

from pydantic import ValidationError
from typing_extensions import assert_never

from pydantic_ai.output import DeferredToolCalls

from . import messages as _messages
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior
from .messages import ToolCallPart
from .tools import ToolDefinition
from .toolsets.abstract import AbstractToolset, ToolsetTool


@dataclass
class ToolManager(Generic[AgentDepsT]):
    """Manages tools for an agent run step. It caches the agent run's toolset's tool definitions and handles calling tools and retries."""

    ctx: RunContext[AgentDepsT]
    """The agent run context for a specific run step."""
    toolset: AbstractToolset[AgentDepsT]
    """The toolset that provides the tools for this run step."""
    tools: dict[str, ToolsetTool[AgentDepsT]]
    """The cached tools for this run step."""

    @classmethod
    async def build(cls, toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]) -> ToolManager[AgentDepsT]:
        """Build a new tool manager for a specific run step."""
        return cls(
            ctx=ctx,
            toolset=toolset,
            tools=await toolset.get_tools(ctx),
        )

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> ToolManager[AgentDepsT]:
        """Build a new tool manager for the next run step, carrying over the retries from the current run step."""
        return await self.__class__.build(self.toolset, replace(ctx, retries=self.ctx.retries))

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        """The tool definitions for the tools in this tool manager."""
        return [tool.tool_def for tool in self.tools.values()]

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        """Get the tool definition for a given tool name, or `None` if the tool is unknown."""
        try:
            return self.tools[name].tool_def
        except KeyError:
            return None

    async def handle_call(self, call: ToolCallPart, allow_partial: bool = False) -> Any:
        """Handle a tool call by validating the arguments, calling the tool, and handling retries.

        Args:
            call: The tool call part to handle.
            allow_partial: Whether to allow partial validation of the tool arguments.
        """
        name = call.tool_name
        tool = None
        try:
            try:
                tool = self.tools[name]
            except KeyError:
                if self.tools:
                    msg = f'Available tools: {", ".join(f"{name!r}" for name in self.tools.keys())}'
                else:
                    msg = 'No tools available.'
                raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')

            ctx = replace(
                self.ctx,
                tool_name=name,
                tool_call_id=call.tool_call_id,
                retry=self.ctx.retries.get(name, 0),
            )

            pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
            validator = tool.args_validator
            if isinstance(call.args, str):
                args_dict = validator.validate_json(call.args or '{}', allow_partial=pyd_allow_partial)
            else:
                args_dict = validator.validate_python(call.args or {}, allow_partial=pyd_allow_partial)

            output = await self.toolset.call_tool(name, args_dict, ctx, tool)
        except (ValidationError, ModelRetry) as e:
            max_retries = tool.max_retries if tool is not None else 1
            current_retry = self.ctx.retries.get(name, 0)

            if current_retry == max_retries:
                raise UnexpectedModelBehavior(f'Tool {name!r} exceeded max retries count of {max_retries}') from e
            else:
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

                self.ctx.retries[name] = current_retry + 1
                raise e
        else:
            self.ctx.retries.pop(name, None)
            return output

    def get_deferred_tool_calls(self, parts: Iterable[_messages.ModelResponsePart]) -> DeferredToolCalls | None:
        """Get the deferred tool calls from the model response parts."""
        deferred_calls_and_defs = [
            (part, tool_def)
            for part in parts
            if isinstance(part, _messages.ToolCallPart)
            and (tool_def := self.get_tool_def(part.tool_name))
            and tool_def.kind == 'deferred'
        ]
        if not deferred_calls_and_defs:
            return None

        deferred_calls: list[_messages.ToolCallPart] = []
        deferred_tool_defs: dict[str, ToolDefinition] = {}
        for part, tool_def in deferred_calls_and_defs:
            deferred_calls.append(part)
            deferred_tool_defs[part.tool_name] = tool_def

        return DeferredToolCalls(deferred_calls, deferred_tool_defs)
