from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, overload

from pydantic.json_schema import GenerateJsonSchema

from pydantic_ai._instructions import AgentInstructions, normalize_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityDescription
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import (
    ArgsValidatorFunc,
    DocstringFormat,
    GenerateToolJsonSchema,
    SystemPromptFunc,
    Tool,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)
from pydantic_ai.toolsets import AbstractToolset, AgentToolset, FunctionToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.combined import CombinedToolset


@dataclass
class Capability(AbstractCapability[AgentDepsT]):
    """Convenience capability for bundling instructions, tools, and toolsets without subclassing.

    This groups related instructions, descriptions, function tools, and toolsets under
    a capability identity. Instructions passed via `instructions=` are available through
    `get_instructions()`;
    [`instructions`][pydantic_ai.capabilities.Capability.instructions] is the decorator
    for registering instruction functions. The constructor accepts static or callable
    `description=` values. For model settings, lifecycle hooks, native tools, wrapper
    toolsets, or custom per-run logic, subclass
    [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability].
    """

    _: KW_ONLY

    toolsets: Sequence[AgentToolset[AgentDepsT]] = ()
    """Toolsets to register with the agent. Combined via [`CombinedToolset`][pydantic_ai.toolsets.CombinedToolset] when more than one is provided."""

    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = ()
    """Function tools to register with the agent."""

    description: str | None = None
    """Static description mirrored on the instance.

    The constructor also accepts callable descriptions, stored internally and returned
    from `get_description()`.
    """

    _function_toolset: FunctionToolset[AgentDepsT] = field(init=False, repr=False)
    _instructions: list[str | SystemPromptFunc[AgentDepsT]] = field(init=False, repr=False, default_factory=lambda: [])
    _description: CapabilityDescription[AgentDepsT] | None = field(init=False, repr=False, default=None)

    def __init__(
        self,
        *,
        instructions: AgentInstructions[AgentDepsT] | None = None,
        toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        id: str | None = None,
        description: CapabilityDescription[AgentDepsT] | None = None,
        defer_loading: bool = False,
    ) -> None:
        resolved_toolsets: tuple[AgentToolset[AgentDepsT], ...]
        if toolsets is not None:
            resolved_toolsets = tuple(toolsets)
        else:
            resolved_toolsets = ()
        if resolved_toolsets and tools:
            raise UserError(
                'Cannot use both `toolsets` and `tools` on the same capability. '
                'Use `toolsets` to register toolsets, or `tools` to register individual tools.'
            )
        self.id = id
        self.description = description if isinstance(description, str) else None
        self._description = description
        self.defer_loading = defer_loading
        self.toolsets = resolved_toolsets
        self.tools = tools
        self._function_toolset = FunctionToolset[AgentDepsT](tools)
        self._instructions = list(normalize_instructions(instructions))

    def get_description(self) -> CapabilityDescription[AgentDepsT] | None:
        return self._description

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return list(self._instructions) if self._instructions else None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        if not self.toolsets:
            return self._function_toolset
        if len(self.toolsets) == 1:
            return self.toolsets[0]
        materialized: list[AbstractToolset[AgentDepsT]] = [
            ts if isinstance(ts, AbstractToolset) else DynamicToolset[AgentDepsT](toolset_func=ts)
            for ts in self.toolsets
        ]
        return CombinedToolset[AgentDepsT](materialized)

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
        if self.toolsets:
            raise UserError('`Capability.tool_plain()` cannot be used when `toolsets=` is set.')

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
        if self.toolsets:
            raise UserError('`Capability.tool()` cannot be used when `toolsets=` is set.')

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

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], str | None], /
    ) -> Callable[[RunContext[AgentDepsT]], str | None]: ...

    @overload
    def instructions(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str | None]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]: ...

    @overload
    def instructions(self, func: Callable[[], str | None], /) -> Callable[[], str | None]: ...

    @overload
    def instructions(self, func: Callable[[], Awaitable[str | None]], /) -> Callable[[], Awaitable[str | None]]: ...

    @overload
    def instructions(self, /) -> Callable[[SystemPromptFunc[AgentDepsT]], SystemPromptFunc[AgentDepsT]]: ...

    def instructions(
        self,
        func: SystemPromptFunc[AgentDepsT] | None = None,
        /,
    ) -> Callable[[SystemPromptFunc[AgentDepsT]], SystemPromptFunc[AgentDepsT]] | SystemPromptFunc[AgentDepsT]:
        """Decorator to register an instructions function on this capability.

        Mirrors `Agent.instructions`: the function may take
        [`RunContext`][pydantic_ai.tools.RunContext] (or no arguments), may be sync or async, and is
        appended to any instructions provided via the `instructions=` field.

        Example:
        ```python
        from pydantic_ai import Capability, RunContext

        cap = Capability[str](instructions='base instructions')

        @cap.instructions
        async def dynamic(ctx: RunContext[str]) -> str:
            return f'extra: {ctx.deps}'
        ```
        """
        if func is None:

            def decorator(
                func_: SystemPromptFunc[AgentDepsT],
            ) -> SystemPromptFunc[AgentDepsT]:
                self._instructions.append(func_)
                return func_

            return decorator
        else:
            self._instructions.append(func)
            return func
