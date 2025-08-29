from __future__ import annotations

from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, cast, overload

from pydantic import TypeAdapter
from pydantic_core import SchemaValidator, core_schema

from pydantic_ai import (
    messages as _messages,
)
from pydantic_ai.format_prompt import format_as_xml
from pydantic_ai.messages import UserContent
from pydantic_ai.run import AgentRunResult

from .._run_context import AgentDepsT, RunContext
from ..agent import AbstractAgent
from ..exceptions import UserError
from ..tools import (
    GenerateToolJsonSchema,
    ToolDefinition,
)
from .abstract import AbstractToolset, ToolsetTool

CallerDepsT = TypeVar('CallerDepsT', default=None)

HandoffDepsT = TypeVar('HandoffDepsT', default=None)
HandoffInputModelT = TypeVar('HandoffInputModelT', default=None)
HandoffOutputDataT = TypeVar('HandoffOutputDataT', default=None)

HandoffDepsFunc = Callable[[CallerDepsT, HandoffInputModelT], Awaitable[HandoffDepsT]]
"""A function that takes an input model and dependencies and returns the agent dependencies."""

HandoffUserPromptFunc = Callable[
    [CallerDepsT, HandoffInputModelT], Awaitable[str | Sequence[_messages.UserContent] | None]
]
"""A function that takes an input model and dependencies and returns the user prompt for the agent."""

HandoffRunFunc = Callable[
    [
        RunContext[CallerDepsT],
        AbstractAgent[HandoffDepsT, HandoffOutputDataT],
        HandoffDepsT,
        str | Sequence[UserContent] | None,
    ],
    Awaitable[AgentRunResult[HandoffOutputDataT] | HandoffOutputDataT],
]
"""A function that takes a run context, an agent, dependencies, and an input model and returns the result of the agent run."""


@dataclass
class AgentToolsetTool(
    ToolsetTool[CallerDepsT], Generic[CallerDepsT, HandoffInputModelT, HandoffDepsT, HandoffOutputDataT]
):
    """A tool definition for an Agent Tool."""

    agent: AbstractAgent[HandoffDepsT, HandoffOutputDataT]

    input_type: type[HandoffInputModelT] | None = None

    deps_func: HandoffDepsFunc[CallerDepsT, HandoffInputModelT, HandoffDepsT] | None = None
    user_prompt_func: HandoffUserPromptFunc[CallerDepsT, HandoffInputModelT] | None = None

    run_func: HandoffRunFunc[CallerDepsT, HandoffDepsT, HandoffOutputDataT] | None = None

    async def _build_deps(self, ctx: RunContext[CallerDepsT], input: HandoffInputModelT) -> HandoffDepsT:
        if self.deps_func is None:
            return cast(HandoffDepsT, ctx.deps)

        return await self.deps_func(ctx.deps, input)

    async def _build_user_prompt(
        self, ctx: RunContext[CallerDepsT], input: HandoffInputModelT
    ) -> str | Sequence[_messages.UserContent] | None:
        if self.user_prompt_func is None:
            return format_as_xml(obj=input)

        return await self.user_prompt_func(ctx.deps, input)

    async def convert_args_to_input_type(self, args: dict[str, Any]) -> HandoffInputModelT:
        if self.input_type is None:
            raise ValueError('Input type is not set')

        type_adapter = TypeAdapter(self.input_type)
        return type_adapter.validate_python(args)

    async def call(self, ctx: RunContext[CallerDepsT], input: HandoffInputModelT) -> HandoffOutputDataT:
        handoff_deps: HandoffDepsT = await self._build_deps(ctx, input)

        handoff_user_prompt: str | Sequence[UserContent] | None = await self._build_user_prompt(ctx, input)

        if self.run_func is not None:
            func_result: AgentRunResult[HandoffOutputDataT] | HandoffOutputDataT = await self.run_func(
                ctx, self.agent, handoff_deps, handoff_user_prompt
            )
            if isinstance(func_result, AgentRunResult):
                return cast(HandoffOutputDataT, func_result.output)
            return func_result

        agent_run_result: AgentRunResult[HandoffOutputDataT] = await self.agent.run(
            user_prompt=handoff_user_prompt,
            deps=handoff_deps,
        )

        return agent_run_result.output

    def as_toolset_tool(self) -> ToolsetTool[AgentDepsT]:
        return ToolsetTool[AgentDepsT](
            toolset=self.toolset,
            tool_def=self.tool_def,
            max_retries=self.max_retries,
            args_validator=self.args_validator,
        )


class AgentToolset(AbstractToolset[AgentDepsT]):
    """A toolset that lets Agents be used as tools.

    See [toolset docs](../toolsets.md#agent-toolset) for more information.
    """

    max_retries: int

    agent_tools: dict[str, AgentToolsetTool[AgentDepsT, Any, Any, Any]]

    _id: str | None

    def __init__(
        self,
        agent_tools: Sequence[AgentToolsetTool[AgentDepsT, Any, Any, Any]] | None = None,
        max_retries: int = 1,
        *,
        id: str | None = None,
    ):
        """Build a new agent toolset.

        Args:
            agent_tools: The tools to add to the toolset.
            max_retries: The maximum number of retries for each tool during a run.
            id: An optional unique ID for the toolset. A toolset needs to have an ID in order to be used in a durable execution environment like Temporal, in which case the ID will be used to identify the toolset's activities within the workflow.
        """
        self.max_retries = max_retries
        self._id = id

        self.agent_tools = {}
        for agent_tool in agent_tools or []:
            self.add_agent_tool(agent_tool)

    @property
    def id(self) -> str | None:
        return self._id

    def add_agent_tool(
        self,
        agent_tool: AgentToolsetTool[AgentDepsT, Any, Any, Any],
    ) -> None:
        """Add an Agent as a tool to the toolset."""
        self.agent_tools[agent_tool.tool_def.name] = agent_tool

    @overload
    def add_agent(
        self,
        agent: AbstractAgent[AgentDepsT, HandoffOutputDataT],
        input_type: type[HandoffInputModelT] = str,
        *,
        user_prompt_func: HandoffUserPromptFunc[AgentDepsT, HandoffInputModelT] | None = None,
        run_func: HandoffRunFunc[AgentDepsT, AgentDepsT, HandoffOutputDataT] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    """Add an Agent with the same dependencies as the Toolset."""

    @overload
    def add_agent(
        self,
        agent: AbstractAgent[AgentDepsT, HandoffOutputDataT],
        input_type: type[HandoffInputModelT],
        *,
        user_prompt_func: HandoffUserPromptFunc[AgentDepsT, HandoffInputModelT] | None = None,
        run_func: HandoffRunFunc[AgentDepsT, AgentDepsT, HandoffOutputDataT] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    """Add an Agent with the same dependencies as the Toolset and an input type."""

    @overload
    def add_agent(
        self,
        agent: AbstractAgent[HandoffDepsT, HandoffOutputDataT],
        input_type: type[HandoffInputModelT] = str,
        *,
        deps_func: HandoffDepsFunc[AgentDepsT, HandoffInputModelT, HandoffDepsT],
        user_prompt_func: HandoffUserPromptFunc[AgentDepsT, HandoffInputModelT] | None = None,
        run_func: HandoffRunFunc[AgentDepsT, HandoffDepsT, HandoffOutputDataT] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    """Add an Agent with different dependencies from the Toolset."""

    @overload
    def add_agent(
        self,
        agent: AbstractAgent[HandoffDepsT, HandoffOutputDataT],
        input_type: type[HandoffInputModelT],
        *,
        deps_func: HandoffDepsFunc[AgentDepsT, HandoffInputModelT, HandoffDepsT],
        user_prompt_func: HandoffUserPromptFunc[AgentDepsT, HandoffInputModelT] | None = None,
        run_func: HandoffRunFunc[AgentDepsT, AgentDepsT, HandoffOutputDataT] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    """Add an Agent with different dependencies from the Toolset and a custom input type."""

    def add_agent(
        self,
        agent: AbstractAgent[HandoffDepsT, HandoffOutputDataT],
        input_type: type[HandoffInputModelT] = str,
        *,
        deps_func: HandoffDepsFunc[AgentDepsT, HandoffInputModelT, HandoffDepsT] | None = None,
        user_prompt_func: HandoffUserPromptFunc[AgentDepsT, HandoffInputModelT] | None = None,
        run_func: HandoffRunFunc[AgentDepsT, HandoffDepsT, HandoffOutputDataT] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Add an Agent as a tool to the toolset."""
        if not (tool_name := name or agent.name):
            raise ValueError('Provide either `name` or an Agent with a name')

        agent_tool_def: ToolDefinition

        type_adapter = TypeAdapter(input_type)

        input_type_schema = type_adapter.json_schema(schema_generator=GenerateToolJsonSchema)
        input_type_schema['properties']['input'] = input_type_schema

        agent_tool_def = ToolDefinition(
            name=tool_name,
            description=description,
            parameters_json_schema=input_type_schema,
        )

        agent_toolset_tool: AgentToolsetTool[AgentDepsT, Any, Any, Any] = AgentToolsetTool(
            toolset=self,
            max_retries=self.max_retries,
            tool_def=agent_tool_def,
            args_validator=SchemaValidator(schema=core_schema.any_schema()),
            agent=agent,
            input_type=input_type,
            deps_func=deps_func,
            user_prompt_func=user_prompt_func,
            run_func=run_func,
        )

        self.agent_tools[tool_name] = agent_toolset_tool

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        return {name: tool.as_toolset_tool() for name, tool in self.agent_tools.items()}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not (agent_tool := self.agent_tools.get(name)):
            raise UserError(f'Unknown tool: {name!r}')

        if agent_tool.input_type is None:
            return await agent_tool.call(ctx, input={})

        agent_input = await agent_tool.convert_args_to_input_type(tool_args)

        return await agent_tool.call(ctx, input=agent_input)
