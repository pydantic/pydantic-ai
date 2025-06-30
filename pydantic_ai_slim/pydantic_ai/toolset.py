from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Iterable, Iterator, Sequence
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Protocol, overload

from pydantic import ValidationError
from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import SchemaValidator
from typing_extensions import Self

from pydantic_ai.output import DeferredToolCalls

from . import messages as _messages
from ._output import BaseOutputSchema, OutputValidator, ToolRetryError
from ._run_context import AgentDepsT, RunContext
from .exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from .tools import (
    DocstringFormat,
    GenerateToolJsonSchema,
    Tool,
    ToolDefinition,
    ToolFuncEither,
    ToolParams,
    ToolPrepareFunc,
    ToolsPrepareFunc,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model


class AbstractToolset(ABC, Generic[AgentDepsT]):
    """A toolset is a collection of tools that can be used by an agent.

    It is responsible for:
    - Listing the tools it contains
    - Validating the arguments of the tools
    - Calling the tools
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def name_conflict_hint(self) -> str:
        return 'Consider renaming the tool or wrapping the toolset in a `PrefixedToolset` to avoid name conflicts.'

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return None

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        return RunToolset(self, ctx)

    @property
    @abstractmethod
    def tool_defs(self) -> list[ToolDefinition]:
        raise NotImplementedError()

    @property
    def tool_names(self) -> list[str]:
        return [tool_def.name for tool_def in self.tool_defs]

    def get_tool_def(self, name: str) -> ToolDefinition | None:
        return next((tool_def for tool_def in self.tool_defs if tool_def.name == name), None)

    @abstractmethod
    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        raise NotImplementedError()

    def validate_tool_args(
        self, ctx: RunContext[AgentDepsT], name: str, args: str | dict[str, Any] | None, allow_partial: bool = False
    ) -> dict[str, Any]:
        pyd_allow_partial: Literal['off', 'trailing-strings'] = 'trailing-strings' if allow_partial else 'off'
        validator = self._get_tool_args_validator(ctx, name)
        if isinstance(args, str):
            return validator.validate_json(args or '{}', allow_partial=pyd_allow_partial)
        else:
            return validator.validate_python(args or {}, allow_partial=pyd_allow_partial)

    @abstractmethod
    def _max_retries_for_tool(self, name: str) -> int:
        raise NotImplementedError()

    @abstractmethod
    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError()

    def set_mcp_sampling_model(self, model: Model) -> None:
        pass


@dataclass(init=False)
class FunctionToolset(AbstractToolset[AgentDepsT]):
    """A toolset that functions can be registered to as tools."""

    max_retries: int = field(default=1)
    tools: dict[str, Tool[Any]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return 'FunctionToolset'

    def __init__(self, tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [], max_retries: int = 1):
        self.max_retries = max_retries
        self.tools = {}
        for tool in tools:
            if isinstance(tool, Tool):
                self.register_tool(tool)
            else:
                self.register_function(tool)

    @overload
    def tool(self, func: ToolFuncEither[AgentDepsT, ToolParams], /) -> ToolFuncEither[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
    ) -> Callable[[ToolFuncEither[AgentDepsT, ToolParams]], ToolFuncEither[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=int)

        @agent.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
        """
        if func is None:

            def tool_decorator(
                func_: ToolFuncEither[AgentDepsT, ToolParams],
            ) -> ToolFuncEither[AgentDepsT, ToolParams]:
                # noinspection PyTypeChecker
                self.register_function(
                    func_,
                    None,
                    name,
                    retries,
                    prepare,
                    docstring_format,
                    require_parameter_descriptions,
                    schema_generator,
                    strict,
                )
                return func_

            return tool_decorator
        else:
            # noinspection PyTypeChecker
            self.register_function(
                func,
                None,
                name,
                retries,
                prepare,
                docstring_format,
                require_parameter_descriptions,
                schema_generator,
                strict,
            )
            return func

    def register_function(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams],
        takes_ctx: bool | None = None,
        name: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
    ) -> None:
        """Register a function as a tool."""
        tool = Tool[AgentDepsT](
            func,
            takes_ctx=takes_ctx,
            name=name,
            max_retries=retries,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
        )
        self.register_tool(tool)

    def register_tool(self, tool: Tool[AgentDepsT]) -> None:
        if tool.name in self.tools:
            raise UserError(f'Tool name conflicts with existing tool: {tool.name!r}')
        if tool.max_retries is None:
            tool.max_retries = self.max_retries
        self.tools[tool.name] = tool

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        self_for_run = RunToolset(self, ctx)
        prepared_for_run = await IndividuallyPreparedToolset(self_for_run, self._prepare_tool_def).prepare_for_run(ctx)
        return RunToolset(prepared_for_run, ctx, original=self)

    async def _prepare_tool_def(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
        tool_name = tool_def.name
        ctx = replace(ctx, tool_name=tool_name, retry=ctx.retries.get(tool_name, 0))
        return await self.tools[tool_name].prepare_tool_def(ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [tool.tool_def for tool in self.tools.values()]

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.tools[name].function_schema.validator

    def _max_retries_for_tool(self, name: str) -> int:
        tool = self.tools[name]
        return tool.max_retries if tool.max_retries is not None else self.max_retries

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.tools[name].function_schema.call(tool_args, ctx)


@dataclass
class OutputToolset(AbstractToolset[AgentDepsT]):
    """A toolset that contains output tools."""

    output_schema: BaseOutputSchema[Any]
    max_retries: int = field(default=1)  # TODO: Test this works
    output_validators: list[OutputValidator[AgentDepsT, Any]] = field(default_factory=list)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [tool.tool_def for tool in self.output_schema.tools.values()]

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.output_schema.tools[name].processor.validator

    def _max_retries_for_tool(self, name: str) -> int:
        return self.max_retries

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        output = await self.output_schema.tools[name].processor.call(tool_args, ctx)
        for validator in self.output_validators:
            output = await validator.validate(output, None, ctx, wrap_validation_errors=False)
        return output


@dataclass(init=False)
class CombinedToolset(AbstractToolset[AgentDepsT]):
    """A toolset that combines multiple toolsets."""

    toolsets: list[AbstractToolset[AgentDepsT]]
    _exit_stack: AsyncExitStack | None
    _toolset_per_tool_name: dict[str, AbstractToolset[AgentDepsT]]

    def __init__(self, toolsets: Sequence[AbstractToolset[AgentDepsT]]):
        self._exit_stack = None
        self.toolsets = list(toolsets)

        self._toolset_per_tool_name = {}
        for toolset in self.toolsets:
            for name in toolset.tool_names:
                try:
                    existing_toolset = self._toolset_per_tool_name[name]
                    raise UserError(
                        f'{toolset.name} defines a tool whose name conflicts with existing tool from {existing_toolset.name}: {name!r}. {toolset.name_conflict_hint}'
                    )
                except KeyError:
                    pass
                self._toolset_per_tool_name[name] = toolset

    async def __aenter__(self) -> Self:
        # TODO: running_count thing like in MCPServer?
        self._exit_stack = AsyncExitStack()
        for toolset in self.toolsets:
            await self._exit_stack.enter_async_context(toolset)
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        return None

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        toolsets_for_run = await asyncio.gather(*[toolset.prepare_for_run(ctx) for toolset in self.toolsets])
        combined_for_run = CombinedToolset(toolsets_for_run)
        return RunToolset(combined_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [tool_def for toolset in self.toolsets for tool_def in toolset.tool_defs]

    @property
    def tool_names(self) -> list[str]:
        return list(self._toolset_per_tool_name.keys())

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self._toolset_for_tool_name(name)._get_tool_args_validator(ctx, name)

    def validate_tool_args(
        self, ctx: RunContext[AgentDepsT], name: str, args: str | dict[str, Any] | None, allow_partial: bool = False
    ) -> dict[str, Any]:
        return self._toolset_for_tool_name(name).validate_tool_args(ctx, name, args, allow_partial)

    def _max_retries_for_tool(self, name: str) -> int:
        return self._toolset_for_tool_name(name)._max_retries_for_tool(name)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self._toolset_for_tool_name(name).call_tool(ctx, name, tool_args, *args, **kwargs)

    def set_mcp_sampling_model(self, model: Model) -> None:
        for toolset in self.toolsets:
            toolset.set_mcp_sampling_model(model)

    def _toolset_for_tool_name(self, name: str) -> AbstractToolset[AgentDepsT]:
        try:
            return self._toolset_per_tool_name[name]
        except KeyError as e:
            raise ValueError(f'Tool {name!r} not found in any toolset') from e


@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT], ABC):
    """A toolset that wraps another toolset and delegates to it."""

    wrapped: AbstractToolset[AgentDepsT]

    @property
    def name(self) -> str:
        return self.wrapped.name

    @property
    def name_conflict_hint(self) -> str:
        return self.wrapped.name_conflict_hint

    async def __aenter__(self) -> Self:
        await self.wrapped.__aenter__()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> bool | None:
        return await self.wrapped.__aexit__(exc_type, exc_value, traceback)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self.wrapped.tool_defs

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.wrapped._get_tool_args_validator(ctx, name)

    def _max_retries_for_tool(self, name: str) -> int:
        return self.wrapped._max_retries_for_tool(name)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.wrapped.call_tool(ctx, name, tool_args, *args, **kwargs)

    def set_mcp_sampling_model(self, model: Model) -> None:
        self.wrapped.set_mcp_sampling_model(model)

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover


@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prefixes the names of the tools it contains."""

    prefix: str

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        prefixed_for_run = PrefixedToolset(wrapped_for_run, self.prefix)
        return RunToolset(prefixed_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [replace(tool_def, name=self._prefixed_tool_name(tool_def.name)) for tool_def in super().tool_defs]

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return super()._get_tool_args_validator(ctx, self._unprefixed_tool_name(name))

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._unprefixed_tool_name(name))

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await super().call_tool(ctx, self._unprefixed_tool_name(name), tool_args, *args, **kwargs)

    def _prefixed_tool_name(self, tool_name: str) -> str:
        return f'{self.prefix}_{tool_name}'

    def _unprefixed_tool_name(self, tool_name: str) -> str:
        full_prefix = f'{self.prefix}_'
        if not tool_name.startswith(full_prefix):
            raise ValueError(f"Tool name '{tool_name}' does not start with prefix '{full_prefix}'")
        return tool_name[len(full_prefix) :]


@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a prepare function."""

    prepare_func: ToolsPrepareFunc[AgentDepsT]

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        original_tool_defs = wrapped_for_run.tool_defs
        prepared_tool_defs = await self.prepare_func(ctx, original_tool_defs) or []

        original_tool_names = {tool_def.name for tool_def in original_tool_defs}
        prepared_tool_names = {tool_def.name for tool_def in prepared_tool_defs}
        if len(prepared_tool_names - original_tool_names) > 0:
            raise UserError('Prepare function is not allowed to change tool names or add new tools.')

        prepared_for_run = PreparedToolset(wrapped_for_run, self.prepare_func)
        return RunToolset(prepared_for_run, ctx, prepared_tool_defs)


@dataclass(init=False)
class MappedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that maps the names of the tools it contains."""

    name_map: dict[str, str]
    _tool_defs: list[ToolDefinition]

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        tool_defs: list[ToolDefinition],
        name_map: dict[str, str],
    ):
        super().__init__(wrapped)
        self._tool_defs = tool_defs
        self.name_map = name_map

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        mapped_for_run = MappedToolset(wrapped_for_run, self._tool_defs, self.name_map)
        return RunToolset(mapped_for_run, ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self._tool_defs

    def _get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return super()._get_tool_args_validator(ctx, self._map_name(name))

    def _max_retries_for_tool(self, name: str) -> int:
        return super()._max_retries_for_tool(self._map_name(name))

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await super().call_tool(ctx, self._map_name(name), tool_args, *args, **kwargs)

    def _map_name(self, name: str) -> str:
        return self.name_map.get(name, name)


@dataclass
class IndividuallyPreparedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that prepares the tools it contains using a per-tool prepare function."""

    prepare_func: ToolPrepareFunc[AgentDepsT]

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)

        tool_defs: dict[str, ToolDefinition] = {}
        name_map: dict[str, str] = {}
        for original_tool_def in wrapped_for_run.tool_defs:
            original_name = original_tool_def.name
            tool_def = await self.prepare_func(ctx, original_tool_def)
            if not tool_def:
                continue

            new_name = tool_def.name
            if new_name in tool_defs:
                if new_name != original_name:
                    raise UserError(f"Renaming tool '{original_name}' to '{new_name}' conflicts with existing tool.")
                else:
                    raise UserError(f'Tool name conflicts with previously renamed tool: {new_name!r}.')
            name_map[new_name] = original_name

            tool_defs[new_name] = tool_def

        mapped_for_run = await MappedToolset(wrapped_for_run, list(tool_defs.values()), name_map).prepare_for_run(ctx)
        return RunToolset(mapped_for_run, ctx, original=self)


@dataclass(init=False)
class FilteredToolset(IndividuallyPreparedToolset[AgentDepsT]):
    """A toolset that filters the tools it contains using a filter function."""

    def __init__(
        self,
        toolset: AbstractToolset[AgentDepsT],
        filter_func: Callable[[RunContext[AgentDepsT], ToolDefinition], bool],
    ):
        async def filter_tool_def(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
            return tool_def if filter_func(ctx, tool_def) else None

        super().__init__(toolset, filter_tool_def)


class CallToolFunc(Protocol):
    """A function protocol that represents a tool call."""

    def __call__(self, name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any) -> Awaitable[Any]: ...


ToolProcessFunc = Callable[
    [
        RunContext[AgentDepsT],
        CallToolFunc,
        str,
        dict[str, Any],
    ],
    Awaitable[Any],
]


@dataclass
class ProcessedToolset(WrapperToolset[AgentDepsT]):
    """A toolset that lets the tool call arguments and return value be customized using a process function."""

    process: ToolProcessFunc[AgentDepsT]

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        wrapped_for_run = await self.wrapped.prepare_for_run(ctx)
        processed = ProcessedToolset(wrapped_for_run, self.process)
        return RunToolset(processed, ctx)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        return await self.process(ctx, partial(self.wrapped.call_tool, ctx), name, tool_args, *args, **kwargs)


@dataclass(init=False)
class RunToolset(WrapperToolset[AgentDepsT]):
    """A toolset that is frozen for a specific run."""

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
            ctx = replace(ctx, retries=self._retries)
            return await self._original.prepare_for_run(ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return self._tool_defs

    @property
    def tool_names(self) -> list[str]:
        return self._tool_names

    def validate_tool_args(
        self, ctx: RunContext[AgentDepsT], name: str, args: str | dict[str, Any] | None, allow_partial: bool = False
    ) -> dict[str, Any]:
        with self._with_retry(name, ctx) as ctx:
            return super().validate_tool_args(ctx, name, args, allow_partial)

    async def call_tool(
        self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        with self._with_retry(name, ctx) as ctx:
            try:
                output = await super().call_tool(ctx, name, tool_args, *args, **kwargs)
            except Exception as e:
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

    @contextmanager
    def _with_retry(self, name: str, ctx: RunContext[AgentDepsT]) -> Iterator[RunContext[AgentDepsT]]:
        try:
            if name not in self.tool_names:
                if self.tool_names:
                    msg = f'Available tools: {", ".join(self.tool_names)}'
                else:
                    msg = 'No tools available.'
                raise ModelRetry(f'Unknown tool name: {name!r}. {msg}')

            ctx = replace(ctx, tool_name=name, retry=self._retries.get(name, 0), retries={})
            yield ctx
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
                if ctx.tool_call_id:
                    if isinstance(e, ValidationError):
                        m = _messages.RetryPromptPart(
                            tool_name=name,
                            content=e.errors(include_url=False, include_context=False),
                            tool_call_id=ctx.tool_call_id,
                        )
                        e = ToolRetryError(m)
                    elif isinstance(e, ModelRetry):
                        m = _messages.RetryPromptPart(
                            tool_name=name,
                            content=e.message,
                            tool_call_id=ctx.tool_call_id,
                        )
                        e = ToolRetryError(m)

                self._retries[name] = current_retry + 1
                raise e
