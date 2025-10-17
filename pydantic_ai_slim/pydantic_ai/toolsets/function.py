from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, overload

from pydantic.json_schema import GenerateJsonSchema

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import (
    DocstringFormat,
    GenerateToolJsonSchema,
    Tool,
    ToolFuncEither,
    ToolParams,
    ToolPrepareFunc,
)
from .abstract import AbstractToolset, ToolsetTool


@dataclass(kw_only=True)
class FunctionToolsetTool(ToolsetTool[AgentDepsT]):
    """A tool definition for a function toolset tool that keeps track of the function to call."""

    call_func: Callable[[dict[str, Any], RunContext[AgentDepsT]], Awaitable[Any]]
    is_async: bool


class FunctionToolset(AbstractToolset[AgentDepsT]):
    """A toolset that lets Python functions be used as tools.

    See [toolset docs](../toolsets.md#function-toolset) for more information.
    """

    tools: dict[str, Tool[Any]]
    max_retries: int
    _id: str | None
    docstring_format: DocstringFormat
    require_parameter_descriptions: bool
    schema_generator: type[GenerateJsonSchema]

    def __init__(
        self,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [],
        *,
        max_retries: int = 1,
        docstring_format: DocstringFormat = 'auto',
        require_parameter_descriptions: bool = False,
        schema_generator: type[GenerateJsonSchema] = GenerateToolJsonSchema,
        strict: bool | None = None,
        sequential: bool = False,
        requires_approval: bool = False,
        metadata: dict[str, Any] | None = None,
        id: str | None = None,
    ):
        """Build a new function toolset.

        Args:
            tools: The tools to add to the toolset.
            max_retries: The maximum number of retries for each tool during a run.
                Applies to all tools, unless overridden when adding a tool.
            docstring_format: Format of tool docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                Defaults to `'auto'`, such that the format is inferred from the structure of the docstring.
                Applies to all tools, unless overridden when adding a tool.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing. Defaults to False.
                Applies to all tools, unless overridden when adding a tool.
            schema_generator: The JSON schema generator class to use for this tool. Defaults to `GenerateToolJsonSchema`.
                Applies to all tools, unless overridden when adding a tool.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
                Applies to all tools, unless overridden when adding a tool.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
                Applies to all tools, unless overridden when adding a tool.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
                Applies to all tools, unless overridden when adding a tool, which will be merged with the toolset's metadata.
            id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal,
                in which case the ID will be used to identify the toolset's activities within the workflow.
        """
        self.max_retries = max_retries
        self._id = id
        self.docstring_format = docstring_format
        self.require_parameter_descriptions = require_parameter_descriptions
        self.schema_generator = schema_generator
        self.strict = strict
        self.sequential = sequential
        self.requires_approval = requires_approval
        self.metadata = metadata

        self.tools = {}
        for tool in tools:
            if isinstance(tool, Tool):
                self.add_tool(tool)
            else:
                self.add_function(tool)

    @property
    def id(self) -> str | None:
        return self._id

    @overload
    def tool(self, func: ToolFuncEither[AgentDepsT, ToolParams], /) -> ToolFuncEither[AgentDepsT, ToolParams]: ...

    @overload
    def tool(
        self,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat | None = None,
        require_parameter_descriptions: bool | None = None,
        schema_generator: type[GenerateJsonSchema] | None = None,
        strict: bool | None = None,
        sequential: bool | None = None,
        requires_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Callable[[ToolFuncEither[AgentDepsT, ToolParams]], ToolFuncEither[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams] | None = None,
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat | None = None,
        require_parameter_descriptions: bool | None = None,
        schema_generator: type[GenerateJsonSchema] | None = None,
        strict: bool | None = None,
        sequential: bool | None = None,
        requires_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Decorator to register a tool function which takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@toolset.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, FunctionToolset, RunContext

        toolset = FunctionToolset()

        @toolset.tool
        def foobar(ctx: RunContext[int], x: int) -> int:
            return ctx.deps + x

        @toolset.tool(retries=2)
        async def spam(ctx: RunContext[str], y: float) -> float:
            return ctx.deps + y

        agent = Agent('test', toolsets=[toolset], deps_type=int)
        result = agent.run_sync('foobar', deps=1)
        print(result.output)
        #> {"foobar":1,"spam":1.0}
        ```

        Args:
            func: The tool function to register.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool,defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                If `None`, the default value is determined by the toolset.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing.
                If `None`, the default value is determined by the toolset.
            schema_generator: The JSON schema generator class to use for this tool.
                If `None`, the default value is determined by the toolset.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
                If `None`, the default value is determined by the toolset.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
                If `None`, the default value is determined by the toolset.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
                If `None`, the default value is determined by the toolset.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
                If `None`, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata.
        """

        def tool_decorator(
            func_: ToolFuncEither[AgentDepsT, ToolParams],
        ) -> ToolFuncEither[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            self.add_function(
                func=func_,
                takes_ctx=None,
                name=name,
                description=description,
                retries=retries,
                prepare=prepare,
                docstring_format=docstring_format,
                require_parameter_descriptions=require_parameter_descriptions,
                schema_generator=schema_generator,
                strict=strict,
                sequential=sequential,
                requires_approval=requires_approval,
                metadata=metadata,
            )
            return func_

        return tool_decorator if func is None else tool_decorator(func)

    def add_function(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams],
        takes_ctx: bool | None = None,
        name: str | None = None,
        description: str | None = None,
        retries: int | None = None,
        prepare: ToolPrepareFunc[AgentDepsT] | None = None,
        docstring_format: DocstringFormat | None = None,
        require_parameter_descriptions: bool | None = None,
        schema_generator: type[GenerateJsonSchema] | None = None,
        strict: bool | None = None,
        sequential: bool | None = None,
        requires_approval: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a function as a tool to the toolset.

        Can take a sync or async function.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        Args:
            func: The tool function to register.
            takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] as its first argument. If `None`, this is inferred from the function signature.
            name: The name of the tool, defaults to the function name.
            description: The description of the tool, defaults to the function docstring.
            retries: The number of retries to allow for this tool, defaults to the agent's default retries,
                which defaults to 1.
            prepare: custom method to prepare the tool definition for each step, return `None` to omit this
                tool from a given step. This is useful if you want to customise a tool at call time,
                or omit it completely from a step. See [`ToolPrepareFunc`][pydantic_ai.tools.ToolPrepareFunc].
            docstring_format: The format of the docstring, see [`DocstringFormat`][pydantic_ai.tools.DocstringFormat].
                If `None`, the default value is determined by the toolset.
            require_parameter_descriptions: If True, raise an error if a parameter description is missing.
                If `None`, the default value is determined by the toolset.
            schema_generator: The JSON schema generator class to use for this tool.
                If `None`, the default value is determined by the toolset.
            strict: Whether to enforce JSON schema compliance (only affects OpenAI).
                See [`ToolDefinition`][pydantic_ai.tools.ToolDefinition] for more info.
                If `None`, the default value is determined by the toolset.
            sequential: Whether the function requires a sequential/serial execution environment. Defaults to False.
                If `None`, the default value is determined by the toolset.
            requires_approval: Whether this tool requires human-in-the-loop approval. Defaults to False.
                See the [tools documentation](../deferred-tools.md#human-in-the-loop-tool-approval) for more info.
                If `None`, the default value is determined by the toolset.
            metadata: Optional metadata for the tool. This is not sent to the model but can be used for filtering and tool behavior customization.
                If `None`, the default value is determined by the toolset. If provided, it will be merged with the toolset's metadata.
        """
        if docstring_format is None:
            docstring_format = self.docstring_format
        if require_parameter_descriptions is None:
            require_parameter_descriptions = self.require_parameter_descriptions
        if schema_generator is None:
            schema_generator = self.schema_generator
        if strict is None:
            strict = self.strict
        if sequential is None:
            sequential = self.sequential
        if requires_approval is None:
            requires_approval = self.requires_approval

        tool = Tool[AgentDepsT](
            func,
            takes_ctx=takes_ctx,
            name=name,
            description=description,
            max_retries=retries,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
            sequential=sequential,
            requires_approval=requires_approval,
            metadata=metadata,
        )
        self.add_tool(tool)

    def add_tool(self, tool: Tool[AgentDepsT]) -> None:
        """Add a tool to the toolset.

        Args:
            tool: The tool to add.
        """
        if tool.name in self.tools:
            raise UserError(f'Tool name conflicts with existing tool: {tool.name!r}')
        if tool.max_retries is None:
            tool.max_retries = self.max_retries
        if self.metadata is not None:
            tool.metadata = self.metadata | (tool.metadata or {})
        self.tools[tool.name] = tool

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools: dict[str, ToolsetTool[AgentDepsT]] = {}
        for original_name, tool in self.tools.items():
            max_retries = tool.max_retries if tool.max_retries is not None else self.max_retries
            run_context = replace(
                ctx,
                tool_name=original_name,
                retry=ctx.retries.get(original_name, 0),
                max_retries=max_retries,
            )
            tool_def = await tool.prepare_tool_def(run_context)
            if not tool_def:
                continue

            new_name = tool_def.name
            if new_name in tools:
                if new_name != original_name:
                    raise UserError(f'Renaming tool {original_name!r} to {new_name!r} conflicts with existing tool.')
                else:
                    raise UserError(f'Tool name conflicts with previously renamed tool: {new_name!r}.')

            tools[new_name] = FunctionToolsetTool(
                toolset=self,
                tool_def=tool_def,
                max_retries=max_retries,
                args_validator=tool.function_schema.validator,
                call_func=tool.function_schema.call,
                is_async=tool.function_schema.is_async,
            )
        return tools

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        assert isinstance(tool, FunctionToolsetTool)
        return await tool.call_func(tool_args, ctx)
