from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, overload

from pydantic.json_schema import GenerateJsonSchema

from pydantic_ai._instructions import AgentInstructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import (
    ArgsValidatorFunc,
    DocstringFormat,
    GenerateToolJsonSchema,
    Tool,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)
from pydantic_ai.toolsets import AgentToolset, FunctionToolset


@dataclass
class Capability(AbstractCapability[AgentDepsT]):
    """Convenience capability for bundling instructions and a toolset without subclassing.

    Use this when you just need to attach static instructions, a toolset, or a description
    to an agent. For dynamic behavior or lifecycle hooks, subclass
    [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability] directly.
    """

    _: KW_ONLY

    instructions: AgentInstructions[AgentDepsT] | None = None
    """Instructions contributed by this capability."""

    toolset: AgentToolset[AgentDepsT] | None = None
    """Toolset to register with the agent."""

    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = ()
    """Function tools to register with the agent."""

    description: str | None = None
    """Human-readable description, surfaced in the `load_capability` catalog when `defer_loading=True`."""

    _function_toolset: FunctionToolset[AgentDepsT] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.toolset and self.tools:
            raise UserError(
                'Cannot use both `toolset` and `tools` on the same capability. '
                'Use `toolset` to register a toolset, or `tools` to register individual tools.'
            )
        self._function_toolset = FunctionToolset[AgentDepsT](self.tools)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return self.instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        return self.toolset or self._function_toolset

    def get_description(self, ctx: RunContext[AgentDepsT] | None) -> str | None:
        return self.description

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Any:
        if self.toolset is not None:
            raise UserError('`Capability.tool_plain()` cannot be used when `toolset=` is set.')

        decorator = self._function_toolset.tool_plain(
            name=name,
            description=description,
            retries=retries,
            prepare=prepare,
            args_validator=args_validator,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
            sequential=sequential,
            requires_approval=requires_approval,
            metadata=metadata,
            timeout=timeout,
            defer_loading=defer_loading,
            include_return_schema=include_return_schema,
        )
        return decorator if func is None else decorator(func)

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        args_validator: ArgsValidatorFunc[AgentDepsT, ToolParams] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
        defer_loading: bool = False,
        include_return_schema: bool | None = None,
    ) -> Any:
        if self.toolset is not None:
            raise UserError('`Capability.tool()` cannot be used when `toolset=` is set.')

        decorator = self._function_toolset.tool(
            name=name,
            description=description,
            retries=retries,
            prepare=prepare,
            args_validator=args_validator,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
            sequential=sequential,
            requires_approval=requires_approval,
            metadata=metadata,
            timeout=timeout,
            defer_loading=defer_loading,
            include_return_schema=include_return_schema,
        )
        return decorator if func is None else decorator(func)
