from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, overload

from pydantic.json_schema import GenerateJsonSchema
from pydantic_core import SchemaValidator

from .._run_context import AgentDepsT, RunContext
from ..exceptions import UserError
from ..tools import (
    DocstringFormat,
    GenerateToolJsonSchema,
    Tool,
    ToolDefinition,
    ToolFuncEither,
    ToolParams,
    ToolPrepareFunc,
)
from ._individually_prepared import IndividuallyPreparedToolset
from ._run import RunToolset
from .base import BaseToolset


@dataclass(init=False)
class FunctionToolset(BaseToolset[AgentDepsT]):
    """A toolset that lets Python functions be used as tools."""

    max_retries: int = field(default=1)
    tools: dict[str, Tool[Any]] = field(default_factory=dict)

    def __init__(self, tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = [], max_retries: int = 1):
        """Build a new function toolset.

        Args:
            tools: The tools to add to the toolset.
            max_retries: The maximum number of retries for each tool during a run.
        """
        self.max_retries = max_retries
        self.tools = {}
        for tool in tools:
            if isinstance(tool, Tool):
                self.add_tool(tool)
            else:
                self.add_function(tool)

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
        so the signature of functions decorated with `@toolset.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext
        from pydantic_ai.toolsets.function import FunctionToolset

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

        def tool_decorator(
            func_: ToolFuncEither[AgentDepsT, ToolParams],
        ) -> ToolFuncEither[AgentDepsT, ToolParams]:
            # noinspection PyTypeChecker
            self.add_function(
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

        return tool_decorator if func is None else tool_decorator(func)

    def add_function(
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
        """Add a function as a tool to the toolset.

        Can take a sync or async function.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        Args:
            func: The tool function to register.
            takes_ctx: Whether the function takes a [`RunContext`][pydantic_ai.tools.RunContext] as its first argument. If `None`, this is inferred from the function signature.
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
        self.tools[tool.name] = tool

    async def prepare_for_run(self, ctx: RunContext[AgentDepsT]) -> RunToolset[AgentDepsT]:
        self_for_run = await super().prepare_for_run(ctx)
        prepared_for_run = await IndividuallyPreparedToolset(self_for_run, self._prepare_tool_def).prepare_for_run(ctx)
        return RunToolset(prepared_for_run, ctx, original=self)

    async def _prepare_tool_def(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
        return await self.tools[tool_def.name].prepare_tool_def(ctx)

    @property
    def tool_defs(self) -> list[ToolDefinition]:
        return [tool.tool_def for tool in self.tools.values()]

    def get_tool_args_validator(self, ctx: RunContext[AgentDepsT], name: str) -> SchemaValidator:
        return self.tools[name].function_schema.validator

    def max_retries_for_tool(self, name: str) -> int:
        tool = self.tools[name]
        return tool.max_retries if tool.max_retries is not None else self.max_retries

    async def call_tool(self, ctx: RunContext[AgentDepsT], name: str, tool_args: dict[str, Any]) -> Any:
        return await self.tools[name].function_schema.call(tool_args, ctx)
