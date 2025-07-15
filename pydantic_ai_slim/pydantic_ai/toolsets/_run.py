from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Generic

from pydantic import ValidationError

from pydantic_ai.output import DeferredToolCalls

from .. import messages as _messages
from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior, UserError
from ..messages import ToolCallPart
from ..tools import ToolDefinition
from .abstract import AbstractToolset


@dataclass
class RunToolset(Generic[AgentDepsT]):
    """TODO: Update docstring -- A toolset that caches the wrapped toolset's tool definitions for a specific run step and handles retries."""

    ctx: RunContext[AgentDepsT]
    toolset: AbstractToolset[AgentDepsT]
    toolset_for_run: AbstractToolset[AgentDepsT]
    tool_defs: list[ToolDefinition]
    tool_names: list[str]
    retries: dict[str, int]

    @classmethod
    async def build(
        cls, ctx: RunContext[AgentDepsT], toolset: AbstractToolset[AgentDepsT], retries: dict[str, int] | None = None
    ) -> RunToolset[AgentDepsT]:
        toolset_for_run = await toolset.for_run(ctx)
        tool_defs = await toolset_for_run.list_tool_defs(ctx)

        tool_names: list[str] = []
        for tool_def in tool_defs:
            name = tool_def.name
            if name in tool_names:
                raise UserError(
                    # TODO: f'{toolset.name} defines a tool whose name conflicts with existing tool from {existing_toolset.name}: {name!r}. {toolset.tool_name_conflict_hint}'
                    f'Tool name conflict: {name!r}. {toolset_for_run.tool_name_conflict_hint}'
                )
            tool_names.append(name)

        return cls(
            ctx=ctx,
            toolset=toolset,
            toolset_for_run=toolset_for_run,
            tool_defs=tool_defs,
            tool_names=tool_names,
            retries=retries or {},
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        if ctx == self.ctx:
            return self
        else:
            return await self.__class__.build(ctx, self.toolset, self.retries)

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        """Get the tool definition for a given tool name, or `None` if the tool is unknown."""
        return next((tool_def for tool_def in self.tool_defs if tool_def.name == name), None)

    async def handle_call(self, call: ToolCallPart, allow_partial: bool = False) -> Any:
        name = call.tool_name
        try:
            if name not in self.tool_names:
                if self.tool_names:
                    msg = f'Available tools: {", ".join(f"{name!r}" for name in self.tool_names)}'
                else:
                    msg = 'No tools available.'
                raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')

            ctx = replace(
                self.ctx,
                tool_name=name,
                tool_call_id=call.tool_call_id,
                retry=self.retries.get(name, 0),
            )

            pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
            validator = self.toolset_for_run.get_tool_args_validator(ctx, name)
            if isinstance(call.args, str):
                args_dict = validator.validate_json(call.args or '{}', allow_partial=pyd_allow_partial)
            else:
                args_dict = validator.validate_python(call.args or {}, allow_partial=pyd_allow_partial)
            output = await self.toolset_for_run.call_tool(ctx, name, args_dict)
        except (ValidationError, ModelRetry) as e:
            try:
                max_retries = self.toolset_for_run.max_retries_for_tool(name)
            except Exception:
                max_retries = 1
            current_retry = self.retries.get(name, 0)

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

                self.retries[name] = current_retry + 1
                raise e
        else:
            self.retries.pop(name, None)
            return output

    def get_deferred_tool_calls(self, parts: Iterable[_messages.ModelResponsePart]) -> DeferredToolCalls | None:
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
