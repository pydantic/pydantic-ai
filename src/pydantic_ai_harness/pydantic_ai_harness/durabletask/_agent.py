from __future__ import annotations

import inspect
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from math import isfinite
from time import monotonic, sleep
from types import GenericAlias, NoneType
from typing import TYPE_CHECKING, Generic, TypeVar, Union, cast

from durabletask import task
from durabletask.client import OrchestrationFailedError, OrchestrationState, OrchestrationStatus, TaskHubGrpcClient
from durabletask.worker import TaskHubGrpcWorker
from pydantic import JsonValue, TypeAdapter
from typing_extensions import TypeForm

from pydantic_ai import Agent, AgentRunResult, DeferredToolRequests, DeferredToolResults
from pydantic_ai._output import _flatten_output_spec, types_from_output_spec  # pyright: ignore[reportPrivateUsage]
from pydantic_ai._utils import get_function_type_hints
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import UserContent
from pydantic_ai.output import (
    NativeOutput,
    OutputSpec,
    PromptedOutput,
    TextOutput,
    ToolOutput,
    _ChoicesActions,  # pyright: ignore[reportPrivateUsage]
)

from ._bridge import current_loop, run_durable
from ._capability import DurableTaskDurability
from ._run_context import _DepsProxy  # pyright: ignore[reportPrivateUsage]

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


@dataclass
class _AgentInput:
    prompt: str | list[UserContent]
    deps_input: JsonValue = None


@dataclass
class _PendingTools:
    requests: DeferredToolRequests
    turn: int


_INPUT = TypeAdapter(_AgentInput)
_PENDING = TypeAdapter(_PendingTools)
_JSON = TypeAdapter[JsonValue](JsonValue)
_RESULT = TypeAdapter(AgentRunResult[object])


def _callable_result_types(function: object) -> Sequence[object]:
    if isinstance(function, partial):
        function = cast(partial[object], function).func
    annotation = get_function_type_hints(function).get('return')
    return [JsonValue] if annotation is None else types_from_output_spec(annotation)


def _result_types(output_spec: OutputSpec[OutputT]) -> list[object]:
    """Describe returned values, rather than the model-side schema of output functions and choices."""
    outputs: list[object] = []
    for output in _flatten_output_spec(output_spec):
        if isinstance(output, (NativeOutput, PromptedOutput)):
            marker = cast(NativeOutput[object] | PromptedOutput[object], output)
            outputs.extend(_result_types(marker.outputs))
        elif isinstance(output, ToolOutput):
            outputs.extend(_result_types(cast(ToolOutput[object], output).output))
        elif isinstance(output, TextOutput):
            outputs.extend(_callable_result_types(output.output_function))
        elif (choices := _ChoicesActions.of(output)) is not None:
            for value in choices.values.values():
                if isinstance(value, type):
                    outputs.append(value)
                elif callable(value):
                    outputs.extend(_callable_result_types(value))
                else:
                    outputs.append(type(cast(object, value)))
        elif inspect.isfunction(output) or inspect.ismethod(output) or isinstance(output, partial):
            outputs.extend(_callable_result_types(cast(object, output)))
        else:
            outputs.extend(types_from_output_spec(output))
    return outputs


def _event_name(turn: int, tool_call_id: str) -> str:
    return f'pydantic_ai.tool:{turn}:{tool_call_id}'


class DurableTaskRun(Generic[OutputT]):
    """A scheduled agent orchestration, with typed results and deferred-tool helpers.

    Obtain a handle with `DurableTaskAgent.start`. Client methods are synchronous,
    matching the Durable Task SDK.
    """

    def __init__(self, client: TaskHubGrpcClient, instance_id: str, *, output_adapter: TypeAdapter[OutputT]) -> None:
        self.client = client
        self.instance_id = instance_id
        self._output_adapter = output_adapter

    def result(self, *, timeout: float = 60) -> AgentRunResult[OutputT]:
        """Wait for completion, raising on a missing instance, orchestration failure or timeout."""
        state = self._completed_state(self.client.wait_for_orchestration_completion(self.instance_id, timeout=timeout))
        result = _RESULT.validate_python(state.get_output())
        # JSON-mode validation supports strict fields; Python-mode revalidation can also consume `Json` twice.
        output = self._output_adapter.validate_json(_JSON.dump_json(_JSON.validate_python(result.output)))
        return cast(AgentRunResult[OutputT], replace(result, output=output))

    def _completed_state(self, state: OrchestrationState | None) -> OrchestrationState:
        if state is None:
            raise UserError(f'Durable Task orchestration {self.instance_id!r} was not found.')
        if state.failure_details is not None:
            raise OrchestrationFailedError(
                f'Orchestration {self.instance_id!r} failed: {state.failure_details.message}', state.failure_details
            )
        if state.runtime_status != OrchestrationStatus.COMPLETED:
            raise UserError(f'Orchestration {self.instance_id!r} ended with status {state.runtime_status.name}.')
        return state

    def wait_for_pause(self, *, timeout: float = 60) -> DeferredToolRequests | None:
        """Wait for deferred tools, returning `None` if the run completes first.

        Raise on a missing instance, orchestration failure or timeout. The timeout
        bounds polling; individual state reads use the SDK's synchronous RPC.
        """
        if not isfinite(timeout) or timeout < 0:
            raise UserError('`timeout` must be finite and non-negative.')
        deadline = monotonic() + timeout
        while True:
            state = self.client.get_orchestration_state(self.instance_id)
            if state is None or state.runtime_status in (
                OrchestrationStatus.COMPLETED,
                OrchestrationStatus.FAILED,
                OrchestrationStatus.TERMINATED,
            ):
                self._completed_state(state)
                return None
            pending = state.get_custom_status()
            if pending is not None:
                return _PENDING.validate_python(pending).requests
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f'Durable Task orchestration {self.instance_id!r} did not pause or finish within {timeout}s.'
                )
            sleep(min(0.1, remaining))

    def _pending(self) -> _PendingTools:
        state = self.client.get_orchestration_state(self.instance_id)
        if state is None or state.get_custom_status() is None:
            raise UserError(f'Orchestration {self.instance_id!r} has no pending tool requests.')
        return _PENDING.validate_python(state.get_custom_status())

    def pending_tools(self) -> DeferredToolRequests:
        """Read the approval requests and deferred calls currently awaiting external events."""
        return self._pending().requests

    def approve(self, tool_call_id: str, *, approved: bool = True) -> None:
        """Approve or deny a pending tool call."""
        pending = self._pending()
        if tool_call_id not in {call.tool_call_id for call in pending.requests.approvals}:
            raise UserError(f'Tool call {tool_call_id!r} is not awaiting approval.')
        self.client.raise_orchestration_event(self.instance_id, _event_name(pending.turn, tool_call_id), data=approved)

    def resolve(self, tool_call_id: str, result: JsonValue) -> None:
        """Supply the JSON-serializable result of a deferred tool call."""
        pending = self._pending()
        if tool_call_id not in {call.tool_call_id for call in pending.requests.calls}:
            raise UserError(f'Tool call {tool_call_id!r} is not awaiting a result.')
        self.client.raise_orchestration_event(
            self.instance_id, _event_name(pending.turn, tool_call_id), data=_JSON.validate_python(result)
        )


class DurableTaskAgent(Generic[DepsT, OutputT]):
    """Register an agent with a Durable Task worker and start typed agent runs.

    The agent must have `DurableTaskDurability` attached and a stable name. Register
    identical agents and tools on every worker that can receive the orchestration.
    """

    def __init__(self, agent: Agent[DepsT, OutputT]) -> None:
        durability = DurableTaskDurability[DepsT].from_agent(agent)
        if durability is None:
            raise UserError('Attach `DurableTaskDurability` before constructing `DurableTaskAgent`.')
        self.agent = agent
        self._durability = durability
        types = _result_types(agent.output_type)
        # Runtime output specs may contain markers or functions, not just a Python type.
        result_type: TypeForm[AgentRunResult[OutputT]]
        output_type: object
        if TYPE_CHECKING:
            result_type = AgentRunResult[OutputT]
            output_type = agent.output_type
        else:
            output_type = Union[tuple(types)]  # noqa: UP007
            result_type = GenericAlias(AgentRunResult, output_type)
        self._output_adapter = TypeAdapter[OutputT](output_type)
        self._result_adapter = TypeAdapter[AgentRunResult[OutputT]](result_type)

        def orchestrator(
            context: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, object]:
            wire = _INPUT.validate_python(payload)

            async def body() -> JsonValue:
                result = await self._run(wire)
                return _JSON.validate_python(
                    self._result_adapter.dump_python(result, mode='json', round_trip=True, warnings='error')
                )

            return (yield from run_durable(context, body))

        orchestrator.__name__ = f'{durability.name}__agent'
        self.orchestrator: task.Orchestrator[object, object] = orchestrator

    @contextmanager
    def register(self, worker: TaskHubGrpcWorker) -> Generator[None]:
        """Register the orchestrator and activities while owning their async runtime.

        Enter this context before starting the worker. Stop and drain the worker before
        exiting it, so activity dependency resources close on the loop that created them.
        """
        backend = self._durability.get_durable_operation_backend()
        with backend.worker_scope(worker):
            worker.add_orchestrator(self.orchestrator)
            yield

    def start(
        self,
        client: TaskHubGrpcClient,
        prompt: str | Sequence[UserContent],
        *,
        deps_input: object = None,
        instance_id: str | None = None,
    ) -> DurableTaskRun[OutputT]:
        """Schedule a run, serializing typed dependencies or JSON input for a dependency factory."""
        if self._durability._deps_factory is None and self.agent.deps_type is not NoneType:  # pyright: ignore[reportPrivateUsage]
            adapter = TypeAdapter[DepsT](self.agent.deps_type)
            deps_input = adapter.dump_python(adapter.validate_python(deps_input), mode='json', warnings='error')
        payload = _AgentInput(
            prompt=prompt if isinstance(prompt, str) else list(prompt), deps_input=_JSON.validate_python(deps_input)
        )
        instance_id = client.schedule_new_orchestration(
            self.orchestrator.__name__,
            input=_INPUT.dump_python(payload, mode='json'),
            instance_id=instance_id,
        )
        return self.get_run(client, instance_id)

    def get_run(self, client: TaskHubGrpcClient, instance_id: str) -> DurableTaskRun[OutputT]:
        """Reconnect to an existing run, including from a separate client process."""
        return DurableTaskRun(client, instance_id, output_adapter=self._output_adapter)

    async def _run(self, wire: _AgentInput) -> AgentRunResult[OutputT]:
        if self._durability._deps_factory is None:  # pyright: ignore[reportPrivateUsage]
            deps_adapter = TypeAdapter[DepsT](self.agent.deps_type)
            deps = deps_adapter.validate_python(wire.deps_input)
        else:
            # The proxy is deliberately unreadable as live dependencies in orchestration code.
            deps = cast(DepsT, _DepsProxy(wire.deps_input))
        result = await self.agent.run(wire.prompt, deps=deps)
        turn = 0
        while isinstance(result.output, DeferredToolRequests):
            deferred = await self._wait_for_tools(result.output, turn)
            turn += 1
            result = await self.agent.run(
                deps=deps,
                message_history=result.all_messages(),
                deferred_tool_results=deferred,
                usage=result.usage,
            )
        # This orchestration starts without previous history, so every accumulated message is new.
        return replace(result, _new_message_index=0)

    async def _wait_for_tools(self, requests: DeferredToolRequests, turn: int) -> DeferredToolResults:
        loop = current_loop()
        assert loop is not None
        context = loop.context
        context.set_custom_status(_PENDING.dump_python(_PendingTools(requests, turn), mode='json'))
        calls = [*requests.approvals, *requests.calls]
        events: list[task.Task[object]] = [
            context.wait_for_external_event(_event_name(turn, call.tool_call_id), data_type=object) for call in calls
        ]
        await loop.wait(task.when_all(events))
        results = DeferredToolResults()
        for call, event in zip(calls, events, strict=True):
            if call in requests.approvals:
                results.approvals[call.tool_call_id] = TypeAdapter(bool).validate_python(
                    event.get_result(), strict=True
                )
            else:
                results.calls[call.tool_call_id] = _JSON.validate_python(event.get_result())
        context.set_custom_status(None)
        return results
