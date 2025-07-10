from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError

from pydantic_ai.output import DeferredToolCalls

from .. import messages as _messages
from .._run_context import AgentDepsT, RunContext
from ..exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior
from ..messages import ToolCallPart
from ..tools import ToolDefinition
from . import AbstractToolset
from .wrapper import WrapperToolset


@dataclass(init=False)
class RunToolset(WrapperToolset[AgentDepsT]):
    """A toolset that caches the wrapped toolset's tool definitions for a specific run step and handles retries."""

    ctx: RunContext[AgentDepsT]
    _tool_defs: list[ToolDefinition]
    _tool_names: list[str]
    _retries: dict[str, int]
    _original: AbstractToolset[AgentDepsT]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        ctx: RunContext[AgentDepsT],
        tool_defs: list[ToolDefinition] | None = None,
        original: AbstractToolset[AgentDepsT] | None = None,
    ):
        self.wrapped = wrapped
        self.ctx = ctx
        self._tool_defs = wrapped.tool_defs if tool_defs is None else tool_defs
        self._tool_names = [tool_def.name for tool_def in self._tool_defs]
        self._retries = ctx.retries.copy()
        self._original = original or wrapped

    @property
    def name(self) -> str:
        return self.wrapped.name

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        if ctx == self.ctx:
            return self
        else:
            if self._retries and not ctx.retries:
                ctx = replace(ctx, retries=self._retries)
            return await self._original.prepare_for_run(ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self._tool_defs

    @property
    def tool_names(self) -> list[str]:
        return self._tool_names

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
                retry=self._retries.get(name, 0),
            )

            pyd_allow_partial = 'trailing-strings' if allow_partial else 'off'
            validator = self._get_tool_args_validator(ctx, name)
            if isinstance(call.args, str):
                args_dict = validator.validate_json(call.args or '{}', allow_partial=pyd_allow_partial)
            else:
                args_dict = validator.validate_python(call.args or {}, allow_partial=pyd_allow_partial)
            output = await self._call_tool(ctx, name, args_dict)
        except (ValidationError, ModelRetry, UnexpectedModelBehavior, ToolRetryError) as e:
            try:
                max_retries = self._max_retries_for_tool(name)
            except Exception:
                max_retries = 1
            current_retry = self._retries.get(name, 0)

            if isinstance(e, UnexpectedModelBehavior) and e.__cause__ is not None:
                e = e.__cause__

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

                self._retries[name] = current_retry + 1
                raise e
        else:
            self._retries.pop(name, None)
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
