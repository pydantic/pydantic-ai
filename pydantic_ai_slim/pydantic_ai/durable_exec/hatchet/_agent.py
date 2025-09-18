from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Generic, overload

from hatchet_sdk import DurableContext, Hatchet
from hatchet_sdk.runnables.workflow import BaseWorkflow
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import Never

from pydantic_ai import (
    messages as _messages,
    models,
    usage as _usage,
)
from pydantic_ai.agent import AbstractAgent, AgentRunResult, EventStreamHandler, RunOutputDataT, WrapperAgent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import (
    AgentDepsT,
    DeferredToolResults,
)
from pydantic_ai.toolsets import AbstractToolset

from ._model import HatchetModel
from ._run_context import HatchetRunContext
from ._utils import TaskConfig


class RunAgentInput(BaseModel, Generic[RunOutputDataT, AgentDepsT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_prompt: str | Sequence[_messages.UserContent] | None = None
    output_type: OutputSpec[RunOutputDataT] | None = None
    message_history: list[_messages.ModelMessage] | None = None
    deferred_tool_results: DeferredToolResults | None = None
    model: models.Model | models.KnownModelName | str | None = None
    deps: AgentDepsT
    model_settings: ModelSettings | None = None
    usage_limits: _usage.UsageLimits | None = None
    usage: _usage.RunUsage | None = None
    infer_name: bool = True
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None
    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    deprecated_kwargs: dict[str, Any] = Field(default_factory=dict)


class HatchetAgent(WrapperAgent[AgentDepsT, OutputDataT]):
    def __init__(
        self,
        wrapped: AbstractAgent[AgentDepsT, OutputDataT],
        hatchet: Hatchet,
        *,
        name: str | None = None,
        mcp_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        run_context_type: type[HatchetRunContext[AgentDepsT]] = HatchetRunContext[AgentDepsT],
    ):
        """Wrap an agent to enable it with Hatchet durable tasks, by automatically offloading model requests, tool calls, and MCP server communication to Hatchet tasks.

        After wrapping, the original agent can still be used as normal outside of the Hatchet workflow.

        Args:
            wrapped: The agent to wrap.
            hatchet: The Hatchet instance to use for creating tasks.
            name: Optional unique agent name to use in the Hatchet tasks' names. If not provided, the agent's `name` will be used.
            mcp_task_config: The base Hatchet task config to use for MCP server tasks. If no config is provided, use the default settings.
            model_task_config: The Hatchet task config to use for model request tasks. If no config is provided, use the default settings.
            run_context_type: The `HatchetRunContext` (sub)class that's used to serialize and deserialize the run context.
        """
        super().__init__(wrapped)

        self._name = name or wrapped.name
        self._hatchet = hatchet

        if not self._name:
            raise UserError(
                "An agent needs to have a unique `name` in order to be used with Hatchet. The name will be used to identify the agent's workflows and tasks."
            )

        if not isinstance(wrapped.model, Model):
            raise UserError(
                'An agent needs to have a `model` in order to be used with Hatchet, it cannot be set at agent run time.'
            )

        self._model = HatchetModel(
            wrapped.model,
            task_name_prefix=self._name,
            task_config=model_task_config or TaskConfig(),
            hatchet=self._hatchet,
        )
        hatchet_agent_name = self._name
        self.run_context_type: type[HatchetRunContext[AgentDepsT]] = run_context_type

        def hatchetify_toolset(toolset: AbstractToolset[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
            from ._toolset import hatchetize_toolset

            return hatchetize_toolset(
                toolset,
                hatchet=hatchet,
                task_name_prefix=hatchet_agent_name,
                task_config=mcp_task_config or TaskConfig(),
                deps_type=self.deps_type,
                run_context_type=run_context_type,
            )

        self._toolsets = [toolset.visit_and_replace(hatchetify_toolset) for toolset in wrapped.toolsets]

        @hatchet.durable_task(name=f'{self._name}.run', input_validator=RunAgentInput[Any, Any])
        async def wrapped_run_workflow(
            input: RunAgentInput[RunOutputDataT, AgentDepsT],
            _ctx: DurableContext,
        ) -> AgentRunResult[Any]:
            with self._hatchet_overrides():
                return await super(WrapperAgent, self).run(
                    input.user_prompt,
                    output_type=input.output_type,
                    message_history=input.message_history,
                    deferred_tool_results=input.deferred_tool_results,
                    model=input.model,
                    deps=input.deps,
                    model_settings=input.model_settings,
                    usage_limits=input.usage_limits,
                    usage=input.usage,
                    infer_name=input.infer_name,
                    toolsets=input.toolsets,
                    event_stream_handler=input.event_stream_handler,
                    **input.deprecated_kwargs,
                )

        self.hatchet_wrapped_run_workflow = wrapped_run_workflow

    @property
    def name(self) -> str | None:
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:  # pragma: no cover
        raise UserError(
            'The agent name cannot be changed after creation. If you need to change the name, create a new agent.'
        )

    @property
    def model(self) -> Model:
        return self._model

    @property
    def toolsets(self) -> Sequence[AbstractToolset[AgentDepsT]]:
        with self._hatchet_overrides():
            return super().toolsets

    @contextmanager
    def _hatchet_overrides(self) -> Iterator[None]:
        with super().override(model=self._model, toolsets=self._toolsets, tools=[]):
            yield

    @property
    def workflows(self) -> Sequence[BaseWorkflow[Any]]:
        workflows: list[BaseWorkflow[Any]] = [
            self.hatchet_wrapped_run_workflow,
            self._model.hatchet_wrapped_request_task,
        ]

        for toolset in self._toolsets:
            from ._toolset import HatchetWrapperToolset

            if isinstance(toolset, HatchetWrapperToolset):
                workflows.extend(toolset.hatchet_tasks)

        return workflows

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[OutputDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
    ) -> AgentRunResult[RunOutputDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        output_type: OutputSpec[RunOutputDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        **_deprecated_kwargs: Never,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode."""
        result = await self.hatchet_wrapped_run_workflow.aio_run(
            RunAgentInput[RunOutputDataT, AgentDepsT](
                user_prompt=user_prompt,
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                event_stream_handler=event_stream_handler,
                deprecated_kwargs=_deprecated_kwargs,
            )
        )

        if isinstance(result, dict):
            return TypeAdapter(AgentRunResult[Any]).validate_python(result)

        return result
