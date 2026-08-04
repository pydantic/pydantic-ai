from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, overload

from pydantic.json_schema import GenerateJsonSchema

from pydantic_ai._instructions import AgentInstructions, normalize_instructions
from pydantic_ai._run_context import AgentDepsT, RunContext
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityDescription
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


@dataclass(init=False)
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
        """Build a capability from instructions, tools, toolsets, and an optional description.

        Args:
            instructions: Static instructions and/or instruction function(s), available via
                `get_instructions()`. Register more with the
                [`instructions`][pydantic_ai.capabilities.Capability.instructions] decorator.
            toolsets: Toolsets to register with the agent.
            tools: Function tools to register with the agent.
            id: Stable identifier for the capability. Required when `defer_loading=True`, so the
                model's `load_capability` call can reference it.
            description: Static string or callable description, returned from `get_description()`.
                For a deferred capability it is shown to the model so it can decide whether to load it.
            defer_loading: When `True`, the capability's tools and instructions stay hidden until the
                model loads it on demand via the `load_capability` tool; requires `id`.
        """
        resolved_toolsets: tuple[AgentToolset[AgentDepsT], ...]
        if toolsets is not None:
            resolved_toolsets = tuple(toolsets)
        else:
            resolved_toolsets = ()
        self.id = id
        self.description = description if isinstance(description, str) else None
        self._description = description
        self.defer_loading = defer_loading
        self.toolsets = resolved_toolsets
        self.tools = tools
        # Stamp the capability's `id` onto its contributed function toolset so it can be used with
        # durable execution, which wraps leaf toolsets by `id` at construction time (see
        # `docs/capabilities/`). User-provided `toolsets=` keep their own ids and are never overwritten.
        self._function_toolset = FunctionToolset[AgentDepsT](tools, id=id)
        self._instructions = list(normalize_instructions(instructions))

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-constructible: holds function tools, instructions, and callable
        # descriptions that don't round-trip through YAML/JSON. Matches the other
        # non-serializable capabilities (`Hooks`, `PrefixTools`, `WrapperCapability`, ...).
        return None

    def get_description(self) -> CapabilityDescription[AgentDepsT] | None:
        return self._description

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return list(self._instructions) if self._instructions else None

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        toolsets: list[AgentToolset[AgentDepsT]] = []
        if self._function_toolset.tools:
            toolsets.append(self._function_toolset)
        toolsets.extend(self.toolsets)

        if not toolsets:
            # Return the live (currently-empty) function toolset rather than `None` so tools
            # registered after construction via `@tool`/`@tool_plain` still surface: the agent
            # wires in this reference once, and `None` would drop it and hide late additions.
            return self._function_toolset
        if len(toolsets) == 1:
            return toolsets[0]
        materialized: list[AbstractToolset[AgentDepsT]] = [
            ts if isinstance(ts, AbstractToolset) else DynamicToolset[AgentDepsT](toolset_func=ts) for ts in toolsets
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
        """Decorator to register a plain (no-[`RunContext`][pydantic_ai.tools.RunContext]) function tool on this capability.

        Mirrors [`Agent.tool_plain`][pydantic_ai.agent.Agent.tool_plain]: the tool is added to this
        capability's function toolset and registered with the agent whenever the capability is active.
        """
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
        """Decorator to register a function tool (taking [`RunContext`][pydantic_ai.tools.RunContext]) on this capability.

        Mirrors [`Agent.tool`][pydantic_ai.agent.Agent.tool]: the tool is added to this capability's
        function toolset and registered with the agent whenever the capability is active.
        """
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
        from pydantic_ai import RunContext
        from pydantic_ai.capabilities import Capability

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
