from __future__ import annotations as _annotations

import dataclasses
import inspect
from collections.abc import AsyncIterator, Awaitable, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager, contextmanager
from copy import deepcopy
from types import FrameType
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, cast, final, overload

from opentelemetry.trace import NoOpTracer, use_span
from pydantic.json_schema import GenerateJsonSchema
from typing_extensions import TypeGuard, TypeVar, deprecated

from pydantic_graph import End, Graph, GraphRun, GraphRunContext
from pydantic_graph._utils import get_event_loop

from . import (
    _agent_graph,
    _result,
    _system_prompt,
    _utils,
    exceptions,
    messages as _messages,
    models,
    result,
    usage as _usage,
)
from .models.instrumented import InstrumentationSettings, InstrumentedModel
from .result import FinalResult, ResultDataT, StreamedRunResult
from .settings import ModelSettings, merge_model_settings
from .tools import (
    AgentDepsT,
    DocstringFormat,
    GenerateToolJsonSchema,
    RunContext,
    Tool,
    ToolFuncContext,
    ToolFuncEither,
    ToolFuncPlain,
    ToolParams,
    ToolPrepareFunc,
)

# Re-exporting like this improves auto-import behavior in PyCharm
capture_run_messages = _agent_graph.capture_run_messages
EndStrategy = _agent_graph.EndStrategy
CallToolsNode = _agent_graph.CallToolsNode
ModelRequestNode = _agent_graph.ModelRequestNode
UserPromptNode = _agent_graph.UserPromptNode

if TYPE_CHECKING:
    from pydantic_ai.mcp import MCPServer

__all__ = (
    'Agent',
    'AgentRun',
    'AgentRunResult',
    'capture_run_messages',
    'EndStrategy',
    'CallToolsNode',
    'ModelRequestNode',
    'UserPromptNode',
    'InstrumentationSettings',
)


T = TypeVar('T')
S = TypeVar('S')
NoneType = type(None)
RunResultDataT = TypeVar('RunResultDataT')
"""Type variable for the result data of a run where `result_type` was customized on the run call."""


@final
@dataclasses.dataclass(init=False)
class Agent(Generic[AgentDepsT, ResultDataT]):
    """Class for defining "agents" - a way to have a specific type of "conversation" with an LLM.

    Agents are generic in the dependency type they take [`AgentDepsT`][pydantic_ai.tools.AgentDepsT]
    and the result data type they return, [`ResultDataT`][pydantic_ai.result.ResultDataT].

    By default, if neither generic parameter is customised, agents have type `Agent[None, str]`.

    Minimal usage example:

    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')
    result = agent.run_sync('What is the capital of France?')
    print(result.data)
    #> Paris
    ```
    """

    model: models.Model | models.KnownModelName | str | None
    """The default model configured for this agent.

    We allow str here since the actual list of allowed models changes frequently.
    """

    name: str | None
    """The name of the agent, used for logging.

    If `None`, we try to infer the agent name from the call frame when the agent is first run.
    """
    end_strategy: EndStrategy
    """Strategy for handling tool calls when a final result is found."""

    model_settings: ModelSettings | None
    """Optional model request settings to use for this agents's runs, by default.

    Note, if `model_settings` is provided by `run`, `run_sync`, or `run_stream`, those settings will
    be merged with this value, with the runtime argument taking priority.
    """

    result_type: type[ResultDataT] = dataclasses.field(repr=False)
    """
    The type of the result data, used to validate the result data, defaults to `str`.
    """

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry."""

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

    _deps_type: type[AgentDepsT] = dataclasses.field(repr=False)
    _result_tool_name: str = dataclasses.field(repr=False)
    _result_tool_description: str | None = dataclasses.field(repr=False)
    _result_schema: _result.ResultSchema[ResultDataT] | None = dataclasses.field(repr=False)
    _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]] = dataclasses.field(repr=False)
    _system_prompts: tuple[str, ...] = dataclasses.field(repr=False)
    _system_prompt_functions: list[_system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(repr=False)
    _system_prompt_dynamic_functions: dict[str, _system_prompt.SystemPromptRunner[AgentDepsT]] = dataclasses.field(
        repr=False
    )
    _function_tools: dict[str, Tool[AgentDepsT]] = dataclasses.field(repr=False)
    _mcp_servers: Sequence[MCPServer] = dataclasses.field(repr=False)
    _default_retries: int = dataclasses.field(repr=False)
    _max_result_retries: int = dataclasses.field(repr=False)
    _override_deps: _utils.Option[AgentDepsT] = dataclasses.field(default=None, repr=False)
    _override_model: _utils.Option[models.Model] = dataclasses.field(default=None, repr=False)

    def __init__(
        self,
        model: models.Model | models.KnownModelName | str | None = None,
        *,
        result_type: type[ResultDataT] = str,
        system_prompt: str | Sequence[str] = (),
        deps_type: type[AgentDepsT] = NoneType,
        name: str | None = None,
        model_settings: ModelSettings | None = None,
        retries: int = 1,
        result_tool_name: str = 'final_result',
        result_tool_description: str | None = None,
        result_retries: int | None = None,
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        mcp_servers: Sequence[MCPServer] = (),
        defer_model_check: bool = False,
        end_strategy: EndStrategy = 'early',
        instrument: InstrumentationSettings | bool | None = None,
    ):
        """Create an agent.

        Args:
            model: The default model to use for this agent, if not provide,
                you must provide the model when calling it. We allow str here since the actual list of allowed models changes frequently.
            result_type: The type of the result data, used to validate the result data, defaults to `str`.
            system_prompt: Static system prompts to use for this agent, you can also register system
                prompts via a function with [`system_prompt`][pydantic_ai.Agent.system_prompt].
            deps_type: The type used for dependency injection, this parameter exists solely to allow you to fully
                parameterize the agent, and therefore get the best out of static type checking.
                If you're not using deps, but want type checking to pass, you can set `deps=None` to satisfy Pyright
                or add a type hint `: Agent[None, <return type>]`.
            name: The name of the agent, used for logging. If `None`, we try to infer the agent name from the call frame
                when the agent is first run.
            model_settings: Optional model request settings to use for this agent's runs, by default.
            retries: The default number of retries to allow before raising an error.
            result_tool_name: The name of the tool to use for the final result.
            result_tool_description: The description of the final result tool.
            result_retries: The maximum number of retries to allow for result validation, defaults to `retries`.
            tools: Tools to register with the agent, you can also register tools via the decorators
                [`@agent.tool`][pydantic_ai.Agent.tool] and [`@agent.tool_plain`][pydantic_ai.Agent.tool_plain].
            mcp_servers: MCP servers to register with the agent. You should register a [`MCPServer`][pydantic_ai.mcp.MCPServer]
                for each server you want the agent to connect to.
            defer_model_check: by default, if you provide a [named][pydantic_ai.models.KnownModelName] model,
                it's evaluated to create a [`Model`][pydantic_ai.models.Model] instance immediately,
                which checks for the necessary environment variables. Set this to `false`
                to defer the evaluation until the first run. Useful if you want to
                [override the model][pydantic_ai.Agent.override] for testing.
            end_strategy: Strategy for handling tool calls that are requested alongside a final result.
                See [`EndStrategy`][pydantic_ai.agent.EndStrategy] for more information.
            instrument: Set to True to automatically instrument with OpenTelemetry,
                which will use Logfire if it's configured.
                Set to an instance of [`InstrumentationSettings`][pydantic_ai.agent.InstrumentationSettings] to customize.
                If this isn't set, then the last value set by
                [`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all]
                will be used, which defaults to False.
                See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
        """
        if model is None or defer_model_check:
            self.model = model
        else:
            self.model = models.infer_model(model)

        self.end_strategy = end_strategy
        self.name = name
        self.model_settings = model_settings
        self.result_type = result_type
        self.instrument = instrument

        self._deps_type = deps_type

        self._result_tool_name = result_tool_name
        self._result_tool_description = result_tool_description
        self._result_schema = _result.ResultSchema[result_type].build(
            result_type, result_tool_name, result_tool_description
        )
        self._result_validators = []

        self._system_prompts = (system_prompt,) if isinstance(system_prompt, str) else tuple(system_prompt)
        self._system_prompt_functions = []
        self._system_prompt_dynamic_functions = {}

        self._function_tools = {}

        self._default_retries = retries
        self._max_result_retries = result_retries if result_retries is not None else retries
        self._mcp_servers = mcp_servers
        for tool in tools:
            if isinstance(tool, Tool):
                self._register_tool(tool)
            else:
                self._register_tool(Tool(tool))

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the instrumentation options for all agents where `instrument` is not set."""
        Agent._instrument_default = instrument

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[ResultDataT]: ...

    @overload
    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[RunResultDataT]: ...

    async def run(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[Any]:
        """Run the agent with a user prompt in async mode.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then
        runs the graph to completion. The result of the run is returned.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            agent_run = await agent.run('What is the capital of France?')
            print(agent_run.data)
            #> Paris
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        async with self.iter(
            user_prompt=user_prompt,
            result_type=result_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
        ) as agent_run:
            async for _ in agent_run:
                pass

        assert agent_run.result is not None, 'The graph run did not finish properly'
        return agent_run.result

    @asynccontextmanager
    async def iter(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[AgentRun[AgentDepsT, Any]]:
        """A contextmanager which can be used to iterate over the agent graph's nodes as they are executed.

        This method builds an internal agent graph (using system prompts, tools and result schemas) and then returns an
        `AgentRun` object. The `AgentRun` can be used to async-iterate over the nodes of the graph as they are
        executed. This is the API to use if you want to consume the outputs coming from each LLM model response, or the
        stream of events coming from the execution of tools.

        The `AgentRun` also provides methods to access the full message history, new messages, and usage statistics,
        and the final result of the run once it has completed.

        For more details, see the documentation of `AgentRun`.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            nodes = []
            async with agent.iter('What is the capital of France?') as agent_run:
                async for node in agent_run:
                    nodes.append(node)
            print(nodes)
            '''
            [
                UserPromptNode(
                    user_prompt='What is the capital of France?',
                    system_prompts=(),
                    system_prompt_functions=[],
                    system_prompt_dynamic_functions={},
                ),
                ModelRequestNode(
                    request=ModelRequest(
                        parts=[
                            UserPromptPart(
                                content='What is the capital of France?',
                                timestamp=datetime.datetime(...),
                                part_kind='user-prompt',
                            )
                        ],
                        kind='request',
                    )
                ),
                CallToolsNode(
                    model_response=ModelResponse(
                        parts=[TextPart(content='Paris', part_kind='text')],
                        model_name='gpt-4o',
                        timestamp=datetime.datetime(...),
                        kind='response',
                    )
                ),
                End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),
            ]
            '''
            print(agent_run.result.data)
            #> Paris
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        model_used = self._get_model(model)
        del model

        deps = self._get_deps(deps)
        new_message_index = len(message_history) if message_history else 0
        result_schema: _result.ResultSchema[RunResultDataT] | None = self._prepare_result_schema(result_type)

        # Build the graph
        graph = self._build_graph(result_type)

        # Build the initial state
        state = _agent_graph.GraphAgentState(
            message_history=message_history[:] if message_history else [],
            usage=usage or _usage.Usage(),
            retries=0,
            run_step=0,
        )

        # We consider it a user error if a user tries to restrict the result type while having a result validator that
        # may change the result type from the restricted type to something else. Therefore, we consider the following
        # typecast reasonable, even though it is possible to violate it with otherwise-type-checked code.
        result_validators = cast(list[_result.ResultValidator[AgentDepsT, RunResultDataT]], self._result_validators)

        # TODO: Instead of this, copy the function tools to ensure they don't share current_retry state between agent
        #  runs. Requires some changes to `Tool` to make them copyable though.
        for v in self._function_tools.values():
            v.current_retry = 0

        model_settings = merge_model_settings(self.model_settings, model_settings)
        usage_limits = usage_limits or _usage.UsageLimits()

        if isinstance(model_used, InstrumentedModel):
            tracer = model_used.settings.tracer
        else:
            tracer = NoOpTracer()
        agent_name = self.name or 'agent'
        run_span = tracer.start_span(
            'agent run',
            attributes={
                'model_name': model_used.model_name if model_used else 'no-model',
                'agent_name': agent_name,
                'logfire.msg': f'{agent_name} run',
            },
        )

        graph_deps = _agent_graph.GraphAgentDeps[AgentDepsT, RunResultDataT](
            user_deps=deps,
            prompt=user_prompt,
            new_message_index=new_message_index,
            model=model_used,
            model_settings=model_settings,
            usage_limits=usage_limits,
            max_result_retries=self._max_result_retries,
            end_strategy=self.end_strategy,
            result_schema=result_schema,
            result_tools=self._result_schema.tool_defs() if self._result_schema else [],
            result_validators=result_validators,
            function_tools=self._function_tools,
            mcp_servers=self._mcp_servers,
            run_span=run_span,
            tracer=tracer,
        )
        start_node = _agent_graph.UserPromptNode[AgentDepsT](
            user_prompt=user_prompt,
            system_prompts=self._system_prompts,
            system_prompt_functions=self._system_prompt_functions,
            system_prompt_dynamic_functions=self._system_prompt_dynamic_functions,
        )

        async with graph.iter(
            start_node,
            state=state,
            deps=graph_deps,
            span=use_span(run_span, end_on_exit=True),
            infer_name=False,
        ) as graph_run:
            yield AgentRun(graph_run)

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[ResultDataT]: ...

    @overload
    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT] | None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[RunResultDataT]: ...

    def run_sync(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AgentRunResult[Any]:
        """Synchronously run the agent with a user prompt.

        This is a convenience method that wraps [`self.run`][pydantic_ai.Agent.run] with `loop.run_until_complete(...)`.
        You therefore can't use this method inside async code or if there's an active event loop.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        result_sync = agent.run_sync('What is the capital of Italy?')
        print(result_sync.data)
        #> Rome
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        if infer_name and self.name is None:
            self._infer_name(inspect.currentframe())
        return get_event_loop().run_until_complete(
            self.run(
                user_prompt,
                result_type=result_type,
                message_history=message_history,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=False,
            )
        )

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, ResultDataT]]: ...

    @overload
    def run_stream(
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT],
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AbstractAsyncContextManager[result.StreamedRunResult[AgentDepsT, RunResultDataT]]: ...

    @asynccontextmanager
    async def run_stream(  # noqa C901
        self,
        user_prompt: str | Sequence[_messages.UserContent] | None = None,
        *,
        result_type: type[RunResultDataT] | None = None,
        message_history: list[_messages.ModelMessage] | None = None,
        model: models.Model | models.KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: _usage.UsageLimits | None = None,
        usage: _usage.Usage | None = None,
        infer_name: bool = True,
    ) -> AsyncIterator[result.StreamedRunResult[AgentDepsT, Any]]:
        """Run the agent with a user prompt in async mode, returning a streamed response.

        Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.run_stream('What is the capital of the UK?') as response:
                print(await response.get_data())
                #> London
        ```

        Args:
            user_prompt: User input to start/continue the conversation.
            result_type: Custom result type to use for this run, `result_type` may only be used if the agent has no
                result validators since result validators would expect an argument that matches the agent's result type.
            message_history: History of the conversation so far.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.

        Returns:
            The result of the run.
        """
        # TODO: We need to deprecate this now that we have the `iter` method.
        #   Before that, though, we should add an event for when we reach the final result of the stream.
        if infer_name and self.name is None:
            # f_back because `asynccontextmanager` adds one frame
            if frame := inspect.currentframe():  # pragma: no branch
                self._infer_name(frame.f_back)

        yielded = False
        async with self.iter(
            user_prompt,
            result_type=result_type,
            message_history=message_history,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=False,
        ) as agent_run:
            first_node = agent_run.next_node  # start with the first node
            assert isinstance(first_node, _agent_graph.UserPromptNode)  # the first node should be a user prompt node
            node = first_node
            while True:
                if self.is_model_request_node(node):
                    graph_ctx = agent_run.ctx
                    async with node._stream(graph_ctx) as streamed_response:  # pyright: ignore[reportPrivateUsage]

                        async def stream_to_final(
                            s: models.StreamedResponse,
                        ) -> FinalResult[models.StreamedResponse] | None:
                            result_schema = graph_ctx.deps.result_schema
                            async for maybe_part_event in streamed_response:
                                if isinstance(maybe_part_event, _messages.PartStartEvent):
                                    new_part = maybe_part_event.part
                                    if isinstance(new_part, _messages.TextPart):
                                        if _agent_graph.allow_text_result(result_schema):
                                            return FinalResult(s, None, None)
                                    elif isinstance(new_part, _messages.ToolCallPart) and result_schema:
                                        for call, _ in result_schema.find_tool([new_part]):
                                            return FinalResult(s, call.tool_name, call.tool_call_id)
                            return None

                        final_result_details = await stream_to_final(streamed_response)
                        if final_result_details is not None:
                            if yielded:
                                raise exceptions.AgentRunError('Agent run produced final results')
                            yielded = True

                            messages = graph_ctx.state.message_history.copy()

                            async def on_complete() -> None:
                                """Called when the stream has completed.

                                The model response will have been added to messages by now
                                by `StreamedRunResult._marked_completed`.
                                """
                                last_message = messages[-1]
                                assert isinstance(last_message, _messages.ModelResponse)
                                tool_calls = [
                                    part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
                                ]

                                parts: list[_messages.ModelRequestPart] = []
                                async for _event in _agent_graph.process_function_tools(
                                    tool_calls,
                                    final_result_details.tool_name,
                                    final_result_details.tool_call_id,
                                    graph_ctx,
                                    parts,
                                ):
                                    pass
                                # TODO: Should we do something here related to the retry count?
                                #   Maybe we should move the incrementing of the retry count to where we actually make a request?
                                # if any(isinstance(part, _messages.RetryPromptPart) for part in parts):
                                #     ctx.state.increment_retries(ctx.deps.max_result_retries)
                                if parts:
                                    messages.append(_messages.ModelRequest(parts))

                            yield StreamedRunResult(
                                messages,
                                graph_ctx.deps.new_message_index,
                                graph_ctx.deps.usage_limits,
                                streamed_response,
                                graph_ctx.deps.result_schema,
                                _agent_graph.build_run_context(graph_ctx),
                                graph_ctx.deps.result_validators,
                                final_result_details.tool_name,
                                on_complete,
                            )
                            break
                next_node = await agent_run.next(node)
                if not isinstance(next_node, _agent_graph.AgentNode):
                    raise exceptions.AgentRunError('Should have produced a StreamedRunResult before getting here')
                node = cast(_agent_graph.AgentNode[Any, Any], next_node)

        if not yielded:
            raise exceptions.AgentRunError('Agent run finished without producing a final result')

    @contextmanager
    def override(
        self,
        *,
        deps: AgentDepsT | _utils.Unset = _utils.UNSET,
        model: models.Model | models.KnownModelName | str | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override agent dependencies and model.

        This is particularly useful when testing.
        You can find an example of this [here](../testing.md#overriding-model-via-pytest-fixtures).

        Args:
            deps: The dependencies to use instead of the dependencies passed to the agent run.
            model: The model to use instead of the model passed to the agent run.
        """
        if _utils.is_set(deps):
            override_deps_before = self._override_deps
            self._override_deps = _utils.Some(deps)
        else:
            override_deps_before = _utils.UNSET

        if _utils.is_set(model):
            override_model_before = self._override_model
            self._override_model = _utils.Some(models.infer_model(model))
        else:
            override_model_before = _utils.UNSET

        try:
            yield
        finally:
            if _utils.is_set(override_deps_before):
                self._override_deps = override_deps_before
            if _utils.is_set(override_model_before):
                self._override_model = override_model_before

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], str], /
    ) -> Callable[[RunContext[AgentDepsT]], str]: ...

    @overload
    def system_prompt(
        self, func: Callable[[RunContext[AgentDepsT]], Awaitable[str]], /
    ) -> Callable[[RunContext[AgentDepsT]], Awaitable[str]]: ...

    @overload
    def system_prompt(self, func: Callable[[], str], /) -> Callable[[], str]: ...

    @overload
    def system_prompt(self, func: Callable[[], Awaitable[str]], /) -> Callable[[], Awaitable[str]]: ...

    @overload
    def system_prompt(
        self, /, *, dynamic: bool = False
    ) -> Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]: ...

    def system_prompt(
        self,
        func: _system_prompt.SystemPromptFunc[AgentDepsT] | None = None,
        /,
        *,
        dynamic: bool = False,
    ) -> (
        Callable[[_system_prompt.SystemPromptFunc[AgentDepsT]], _system_prompt.SystemPromptFunc[AgentDepsT]]
        | _system_prompt.SystemPromptFunc[AgentDepsT]
    ):
        """Decorator to register a system prompt function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its only argument.
        Can decorate a sync or async functions.

        The decorator can be used either bare (`agent.system_prompt`) or as a function call
        (`agent.system_prompt(...)`), see the examples below.

        Overloads for every possible signature of `system_prompt` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Args:
            func: The function to decorate
            dynamic: If True, the system prompt will be reevaluated even when `messages_history` is provided,
                see [`SystemPromptPart.dynamic_ref`][pydantic_ai.messages.SystemPromptPart.dynamic_ref]

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test', deps_type=str)

        @agent.system_prompt
        def simple_system_prompt() -> str:
            return 'foobar'

        @agent.system_prompt(dynamic=True)
        async def async_system_prompt(ctx: RunContext[str]) -> str:
            return f'{ctx.deps} is the best'
        ```
        """
        if func is None:

            def decorator(
                func_: _system_prompt.SystemPromptFunc[AgentDepsT],
            ) -> _system_prompt.SystemPromptFunc[AgentDepsT]:
                runner = _system_prompt.SystemPromptRunner[AgentDepsT](func_, dynamic=dynamic)
                self._system_prompt_functions.append(runner)
                if dynamic:
                    self._system_prompt_dynamic_functions[func_.__qualname__] = runner
                return func_

            return decorator
        else:
            assert not dynamic, "dynamic can't be True in this case"
            self._system_prompt_functions.append(_system_prompt.SystemPromptRunner[AgentDepsT](func, dynamic=dynamic))
            return func

    @overload
    def result_validator(
        self, func: Callable[[RunContext[AgentDepsT], ResultDataT], ResultDataT], /
    ) -> Callable[[RunContext[AgentDepsT], ResultDataT], ResultDataT]: ...

    @overload
    def result_validator(
        self, func: Callable[[RunContext[AgentDepsT], ResultDataT], Awaitable[ResultDataT]], /
    ) -> Callable[[RunContext[AgentDepsT], ResultDataT], Awaitable[ResultDataT]]: ...

    @overload
    def result_validator(
        self, func: Callable[[ResultDataT], ResultDataT], /
    ) -> Callable[[ResultDataT], ResultDataT]: ...

    @overload
    def result_validator(
        self, func: Callable[[ResultDataT], Awaitable[ResultDataT]], /
    ) -> Callable[[ResultDataT], Awaitable[ResultDataT]]: ...

    def result_validator(
        self, func: _result.ResultValidatorFunc[AgentDepsT, ResultDataT], /
    ) -> _result.ResultValidatorFunc[AgentDepsT, ResultDataT]:
        """Decorator to register a result validator function.

        Optionally takes [`RunContext`][pydantic_ai.tools.RunContext] as its first argument.
        Can decorate a sync or async functions.

        Overloads for every possible signature of `result_validator` are included so the decorator doesn't obscure
        the type of the function, see `tests/typed_agent.py` for tests.

        Example:
        ```python
        from pydantic_ai import Agent, ModelRetry, RunContext

        agent = Agent('test', deps_type=str)

        @agent.result_validator
        def result_validator_simple(data: str) -> str:
            if 'wrong' in data:
                raise ModelRetry('wrong response')
            return data

        @agent.result_validator
        async def result_validator_deps(ctx: RunContext[str], data: str) -> str:
            if ctx.deps in data:
                raise ModelRetry('wrong response')
            return data

        result = agent.run_sync('foobar', deps='spam')
        print(result.data)
        #> success (no tool calls)
        ```
        """
        self._result_validators.append(_result.ResultValidator[AgentDepsT, Any](func))
        return func

    @overload
    def tool(self, func: ToolFuncContext[AgentDepsT, ToolParams], /) -> ToolFuncContext[AgentDepsT, ToolParams]: ...

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
    ) -> Callable[[ToolFuncContext[AgentDepsT, ToolParams]], ToolFuncContext[AgentDepsT, ToolParams]]: ...

    def tool(
        self,
        func: ToolFuncContext[AgentDepsT, ToolParams] | None = None,
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
        print(result.data)
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
                func_: ToolFuncContext[AgentDepsT, ToolParams],
            ) -> ToolFuncContext[AgentDepsT, ToolParams]:
                # noinspection PyTypeChecker
                self._register_function(
                    func_,
                    True,
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
            self._register_function(
                func,
                True,
                name,
                retries,
                prepare,
                docstring_format,
                require_parameter_descriptions,
                schema_generator,
                strict,
            )
            return func

    @overload
    def tool_plain(self, func: ToolFuncPlain[ToolParams], /) -> ToolFuncPlain[ToolParams]: ...

    @overload
    def tool_plain(
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
    ) -> Callable[[ToolFuncPlain[ToolParams]], ToolFuncPlain[ToolParams]]: ...

    def tool_plain(
        self,
        func: ToolFuncPlain[ToolParams] | None = None,
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
        """Decorator to register a tool function which DOES NOT take `RunContext` as an argument.

        Can decorate a sync or async functions.

        The docstring is inspected to extract both the tool description and description of each parameter,
        [learn more](../tools.md#function-tools-and-schema).

        We can't add overloads for every possible signature of tool, since the return type is a recursive union
        so the signature of functions decorated with `@agent.tool` is obscured.

        Example:
        ```python
        from pydantic_ai import Agent, RunContext

        agent = Agent('test')

        @agent.tool
        def foobar(ctx: RunContext[int]) -> int:
            return 123

        @agent.tool(retries=2)
        async def spam(ctx: RunContext[str]) -> float:
            return 3.14

        result = agent.run_sync('foobar', deps=1)
        print(result.data)
        #> {"foobar":123,"spam":3.14}
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

            def tool_decorator(func_: ToolFuncPlain[ToolParams]) -> ToolFuncPlain[ToolParams]:
                # noinspection PyTypeChecker
                self._register_function(
                    func_,
                    False,
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
            self._register_function(
                func,
                False,
                name,
                retries,
                prepare,
                docstring_format,
                require_parameter_descriptions,
                schema_generator,
                strict,
            )
            return func

    def _register_function(
        self,
        func: ToolFuncEither[AgentDepsT, ToolParams],
        takes_ctx: bool,
        name: str | None,
        retries: int | None,
        prepare: ToolPrepareFunc[AgentDepsT] | None,
        docstring_format: DocstringFormat,
        require_parameter_descriptions: bool,
        schema_generator: type[GenerateJsonSchema],
        strict: bool | None,
    ) -> None:
        """Private utility to register a function as a tool."""
        retries_ = retries if retries is not None else self._default_retries
        tool = Tool[AgentDepsT](
            func,
            takes_ctx=takes_ctx,
            name=name,
            max_retries=retries_,
            prepare=prepare,
            docstring_format=docstring_format,
            require_parameter_descriptions=require_parameter_descriptions,
            schema_generator=schema_generator,
            strict=strict,
        )
        self._register_tool(tool)

    def _register_tool(self, tool: Tool[AgentDepsT]) -> None:
        """Private utility to register a tool instance."""
        if tool.max_retries is None:
            # noinspection PyTypeChecker
            tool = dataclasses.replace(tool, max_retries=self._default_retries)

        if tool.name in self._function_tools:
            raise exceptions.UserError(f'Tool name conflicts with existing tool: {tool.name!r}')

        if self._result_schema and tool.name in self._result_schema.tools:
            raise exceptions.UserError(f'Tool name conflicts with result schema name: {tool.name!r}')

        self._function_tools[tool.name] = tool

    def _get_model(self, model: models.Model | models.KnownModelName | str | None) -> models.Model:
        """Create a model configured for this agent.

        Args:
            model: model to use for this run, required if `model` was not set when creating the agent.

        Returns:
            The model used
        """
        model_: models.Model
        if some_model := self._override_model:
            # we don't want `override()` to cover up errors from the model not being defined, hence this check
            if model is None and self.model is None:
                raise exceptions.UserError(
                    '`model` must either be set on the agent or included when calling it. '
                    '(Even when `override(model=...)` is customizing the model that will actually be called)'
                )
            model_ = some_model.value
        elif model is not None:
            model_ = models.infer_model(model)
        elif self.model is not None:
            # noinspection PyTypeChecker
            model_ = self.model = models.infer_model(self.model)
        else:
            raise exceptions.UserError('`model` must either be set on the agent or included when calling it.')

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        if instrument and not isinstance(model_, InstrumentedModel):
            if instrument is True:
                instrument = InstrumentationSettings()

            model_ = InstrumentedModel(model_, instrument)

        return model_

    def _get_deps(self: Agent[T, ResultDataT], deps: T) -> T:
        """Get deps for a run.

        If we've overridden deps via `_override_deps`, use that, otherwise use the deps passed to the call.

        We could do runtime type checking of deps against `self._deps_type`, but that's a slippery slope.
        """
        if some_deps := self._override_deps:
            return some_deps.value
        else:
            return deps

    def _infer_name(self, function_frame: FrameType | None) -> None:
        """Infer the agent name from the call frame.

        Usage should be `self._infer_name(inspect.currentframe())`.
        """
        assert self.name is None, 'Name already set'
        if function_frame is not None:  # pragma: no branch
            if parent_frame := function_frame.f_back:  # pragma: no branch
                for name, item in parent_frame.f_locals.items():
                    if item is self:
                        self.name = name
                        return
                if parent_frame.f_locals != parent_frame.f_globals:
                    # if we couldn't find the agent in locals and globals are a different dict, try globals
                    for name, item in parent_frame.f_globals.items():
                        if item is self:
                            self.name = name
                            return

    @property
    @deprecated(
        'The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.', category=None
    )
    def last_run_messages(self) -> list[_messages.ModelMessage]:
        raise AttributeError('The `last_run_messages` attribute has been removed, use `capture_run_messages` instead.')

    def _build_graph(
        self, result_type: type[RunResultDataT] | None
    ) -> Graph[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[Any]]:
        return _agent_graph.build_agent_graph(self.name, self._deps_type, result_type or self.result_type)

    def _prepare_result_schema(
        self, result_type: type[RunResultDataT] | None
    ) -> _result.ResultSchema[RunResultDataT] | None:
        if result_type is not None:
            if self._result_validators:
                raise exceptions.UserError('Cannot set a custom run `result_type` when the agent has result validators')
            return _result.ResultSchema[result_type].build(
                result_type, self._result_tool_name, self._result_tool_description
            )
        else:
            return self._result_schema  # pyright: ignore[reportReturnType]

    @staticmethod
    def is_model_request_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeGuard[_agent_graph.ModelRequestNode[T, S]]:
        """Check if the node is a `ModelRequestNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.ModelRequestNode)

    @staticmethod
    def is_call_tools_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeGuard[_agent_graph.CallToolsNode[T, S]]:
        """Check if the node is a `CallToolsNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.CallToolsNode)

    @staticmethod
    def is_user_prompt_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeGuard[_agent_graph.UserPromptNode[T, S]]:
        """Check if the node is a `UserPromptNode`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, _agent_graph.UserPromptNode)

    @staticmethod
    def is_end_node(
        node: _agent_graph.AgentNode[T, S] | End[result.FinalResult[S]],
    ) -> TypeGuard[End[result.FinalResult[S]]]:
        """Check if the node is a `End`, narrowing the type if it is.

        This method preserves the generic parameters while narrowing the type, unlike a direct call to `isinstance`.
        """
        return isinstance(node, End)

    @asynccontextmanager
    async def run_mcp_servers(self) -> AsyncIterator[None]:
        """Run [`MCPServerStdio`s][pydantic_ai.mcp.MCPServerStdio] so they can be used by the agent.

        Returns: a context manager to start and shutdown the servers.
        """
        exit_stack = AsyncExitStack()
        try:
            for mcp_server in self._mcp_servers:
                await exit_stack.enter_async_context(mcp_server)
            yield
        finally:
            await exit_stack.aclose()


@dataclasses.dataclass(repr=False)
class AgentRun(Generic[AgentDepsT, ResultDataT]):
    """A stateful, async-iterable run of an [`Agent`][pydantic_ai.agent.Agent].

    You generally obtain an `AgentRun` instance by calling `async with my_agent.iter(...) as agent_run:`.

    Once you have an instance, you can use it to iterate through the run's nodes as they execute. When an
    [`End`][pydantic_graph.nodes.End] is reached, the run finishes and [`result`][pydantic_ai.agent.AgentRun.result]
    becomes available.

    Example:
    ```python
    from pydantic_ai import Agent

    agent = Agent('openai:gpt-4o')

    async def main():
        nodes = []
        # Iterate through the run, recording each node along the way:
        async with agent.iter('What is the capital of France?') as agent_run:
            async for node in agent_run:
                nodes.append(node)
        print(nodes)
        '''
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                            part_kind='user-prompt',
                        )
                    ],
                    kind='request',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='Paris', part_kind='text')],
                    model_name='gpt-4o',
                    timestamp=datetime.datetime(...),
                    kind='response',
                )
            ),
            End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),
        ]
        '''
        print(agent_run.result.data)
        #> Paris
    ```

    You can also manually drive the iteration using the [`next`][pydantic_ai.agent.AgentRun.next] method for
    more granular control.
    """

    _graph_run: GraphRun[
        _agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any], FinalResult[ResultDataT]
    ]

    @property
    def ctx(self) -> GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]]:
        """The current context of the agent run."""
        return GraphRunContext[_agent_graph.GraphAgentState, _agent_graph.GraphAgentDeps[AgentDepsT, Any]](
            self._graph_run.state, self._graph_run.deps
        )

    @property
    def next_node(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
        """The next node that will be run in the agent graph.

        This is the next node that will be used during async iteration, or if a node is not passed to `self.next(...)`.
        """
        next_node = self._graph_run.next_node
        if isinstance(next_node, End):
            return next_node
        if _agent_graph.is_agent_node(next_node):
            return next_node
        raise exceptions.AgentRunError(f'Unexpected node type: {type(next_node)}')  # pragma: no cover

    @property
    def result(self) -> AgentRunResult[ResultDataT] | None:
        """The final result of the run if it has ended, otherwise `None`.

        Once the run returns an [`End`][pydantic_graph.nodes.End] node, `result` is populated
        with an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult].
        """
        graph_run_result = self._graph_run.result
        if graph_run_result is None:
            return None
        return AgentRunResult(
            graph_run_result.output.data,
            graph_run_result.output.tool_name,
            graph_run_result.state,
            self._graph_run.deps.new_message_index,
        )

    def __aiter__(
        self,
    ) -> AsyncIterator[_agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]]:
        """Provide async-iteration over the nodes in the agent run."""
        return self

    async def __anext__(
        self,
    ) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
        """Advance to the next node automatically based on the last returned node."""
        next_node = await self._graph_run.__anext__()
        if _agent_graph.is_agent_node(next_node):
            return next_node
        assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
        return next_node

    async def next(
        self,
        node: _agent_graph.AgentNode[AgentDepsT, ResultDataT],
    ) -> _agent_graph.AgentNode[AgentDepsT, ResultDataT] | End[FinalResult[ResultDataT]]:
        """Manually drive the agent run by passing in the node you want to run next.

        This lets you inspect or mutate the node before continuing execution, or skip certain nodes
        under dynamic conditions. The agent run should be stopped when you return an [`End`][pydantic_graph.nodes.End]
        node.

        Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_graph import End

        agent = Agent('openai:gpt-4o')

        async def main():
            async with agent.iter('What is the capital of France?') as agent_run:
                next_node = agent_run.next_node  # start with the first node
                nodes = [next_node]
                while not isinstance(next_node, End):
                    next_node = await agent_run.next(next_node)
                    nodes.append(next_node)
                # Once `next_node` is an End, we've finished:
                print(nodes)
                '''
                [
                    UserPromptNode(
                        user_prompt='What is the capital of France?',
                        system_prompts=(),
                        system_prompt_functions=[],
                        system_prompt_dynamic_functions={},
                    ),
                    ModelRequestNode(
                        request=ModelRequest(
                            parts=[
                                UserPromptPart(
                                    content='What is the capital of France?',
                                    timestamp=datetime.datetime(...),
                                    part_kind='user-prompt',
                                )
                            ],
                            kind='request',
                        )
                    ),
                    CallToolsNode(
                        model_response=ModelResponse(
                            parts=[TextPart(content='Paris', part_kind='text')],
                            model_name='gpt-4o',
                            timestamp=datetime.datetime(...),
                            kind='response',
                        )
                    ),
                    End(data=FinalResult(data='Paris', tool_name=None, tool_call_id=None)),
                ]
                '''
                print('Final result:', agent_run.result.data)
                #> Final result: Paris
        ```

        Args:
            node: The node to run next in the graph.

        Returns:
            The next node returned by the graph logic, or an [`End`][pydantic_graph.nodes.End] node if
            the run has completed.
        """
        # Note: It might be nice to expose a synchronous interface for iteration, but we shouldn't do it
        # on this class, or else IDEs won't warn you if you accidentally use `for` instead of `async for` to iterate.
        next_node = await self._graph_run.next(node)
        if _agent_graph.is_agent_node(next_node):
            return next_node
        assert isinstance(next_node, End), f'Unexpected node type: {type(next_node)}'
        return next_node

    def usage(self) -> _usage.Usage:
        """Get usage statistics for the run so far, including token usage, model requests, and so on."""
        return self._graph_run.state.usage

    def __repr__(self) -> str:
        result = self._graph_run.result
        result_repr = '<run not finished>' if result is None else repr(result.output)
        return f'<{type(self).__name__} result={result_repr} usage={self.usage()}>'


@dataclasses.dataclass
class AgentRunResult(Generic[ResultDataT]):
    """The final result of an agent run."""

    data: ResultDataT  # TODO: rename this to output. I'm putting this off for now mostly to reduce the size of the diff

    _result_tool_name: str | None = dataclasses.field(repr=False)
    _state: _agent_graph.GraphAgentState = dataclasses.field(repr=False)
    _new_message_index: int = dataclasses.field(repr=False)

    def _set_result_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the result tool.

        Useful if you want to continue the conversation and want to set the response to the result tool call.
        """
        if not self._result_tool_name:
            raise ValueError('Cannot set result tool return content when the return type is `str`.')
        messages = deepcopy(self._state.message_history)
        last_message = messages[-1]
        for part in last_message.parts:
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._result_tool_name:
                part.content = return_content
                return messages
        raise LookupError(f'No tool call found with tool name {self._result_tool_name!r}.')

    def all_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        if result_tool_return_content is not None:
            return self._set_result_tool_return(result_tool_return_content)
        else:
            return self._state.message_history

    def all_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.agent.AgentRunResult.all_messages] as JSON bytes.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(result_tool_return_content=result_tool_return_content)
        )

    def new_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.agent.AgentRunResult.new_messages] as JSON bytes.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(result_tool_return_content=result_tool_return_content)
        )

    def usage(self) -> _usage.Usage:
        """Return the usage of the whole run."""
        return self._state.usage
