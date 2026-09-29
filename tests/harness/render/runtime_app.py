"""Process-isolated nested-agent fixture used by the local-runtime test."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from dataclasses import dataclass

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel, TypeAdapter
from render.workflows import TaskContext, Workflows
from typing_extensions import TypedDict

from pydantic_ai import Agent, CustomEvent, ModelRetry, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import RenderWorkflows
from pydantic_ai_harness.memory import Memory, SqliteMemoryStore
from pydantic_ai_harness.subagents import SubAgent, SubAgents


class RuntimeDeps(TypedDict):
    prefix: str
    controller_pid: int


class MemoryRuntimeDeps(TypedDict):
    database: str
    tenant: str


class MemoryTaskResult(BaseModel):
    root_pid: int
    tool_pid: int
    content: str
    span_exported: bool


class ToolEvidence(BaseModel):
    pid: int
    retry_count: int
    value: str


class GrandchildOutput(BaseModel):
    model_pid: int
    tool: ToolEvidence


class ChildOutput(BaseModel):
    grandchild: GrandchildOutput
    model_pid: int
    sibling_tools: list[ToolEvidence]


class ParentOutput(BaseModel):
    child: ChildOutput
    model_pid: int


class RuntimeEventEvidence(BaseModel):
    label: str
    sequence: int


class RootTaskResult(BaseModel):
    controller_pid: int
    deps_prefix: str
    events: list[RuntimeEventEvidence]
    output: ParentOutput
    root_pid: int
    usage_markers: dict[str, int]


@dataclass(kw_only=True)
class RuntimeEffectEvent(CustomEvent, name='render_local_runtime.effect'):
    label: str
    sequence: int


def _tool_returns(messages: list[ModelMessage]) -> dict[str, ToolReturnPart]:
    """Index tool results from serialized model history by public tool name."""
    return {part.tool_name: part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)}


def grandchild_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Run the grandchild tool once, using only serialized history for state."""
    del info
    returned = _tool_returns(messages)
    if tool := returned.get('grandchild_lookup'):
        output = GrandchildOutput(model_pid=os.getpid(), tool=ToolEvidence.model_validate(tool.content))
        return ModelResponse(parts=[TextPart(output.model_dump_json())])
    return ModelResponse(parts=[ToolCallPart('grandchild_lookup', {}, tool_call_id='grandchild-lookup')])


def child_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Run sibling tools, delegate, then finish from serialized history."""
    del info
    returned = _tool_returns(messages)
    if delegated := returned.get('delegate_task'):
        output = ChildOutput(
            grandchild=GrandchildOutput.model_validate_json(str(delegated.content)),
            model_pid=os.getpid(),
            sibling_tools=[
                ToolEvidence.model_validate(returned[name].content) for name in ('child_alpha', 'child_beta')
            ],
        )
        return ModelResponse(parts=[TextPart(output.model_dump_json())])
    if {'child_alpha', 'child_beta'} <= returned.keys():
        return ModelResponse(
            parts=[
                ToolCallPart(
                    'delegate_task',
                    {'agent_name': 'runtime-grandchild', 'task': 'collect grandchild evidence'},
                    tool_call_id='child-to-grandchild',
                )
            ]
        )
    return ModelResponse(
        parts=[
            ToolCallPart('child_alpha', {}, tool_call_id='child-alpha'),
            ToolCallPart('child_beta', {}, tool_call_id='child-beta'),
        ]
    )


def parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Delegate once and finish from the child result in serialized history."""
    del info
    returned = _tool_returns(messages)
    if delegated := returned.get('delegate_task'):
        output = ParentOutput(
            child=ChildOutput.model_validate_json(str(delegated.content)),
            model_pid=os.getpid(),
        )
        return ModelResponse(parts=[TextPart(output.model_dump_json())])
    return ModelResponse(
        parts=[
            ToolCallPart(
                'delegate_task',
                {'agent_name': 'runtime-child', 'task': 'collect nested process evidence'},
                tool_call_id='parent-to-child',
            )
        ]
    )


async def _stream_response(response: ModelResponse) -> AsyncIterator[DeltaToolCalls | str]:
    """Convert one deterministic response to FunctionModel's streamed form."""
    for index, part in enumerate(response.parts):
        if isinstance(part, TextPart):
            yield part.content
        elif isinstance(part, ToolCallPart):
            yield {
                index: DeltaToolCall(
                    name=part.tool_name,
                    json_args=part.args_as_json_str(),
                    tool_call_id=part.tool_call_id,
                )
            }


async def stream_child_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    """Stream the same history-driven child response."""
    async for delta in _stream_response(child_model(messages, info)):
        yield delta


workflows = Workflows()
grandchild_runtime = RenderWorkflows[RuntimeDeps](workflows, deps_type=RuntimeDeps)
grandchild = Agent[RuntimeDeps, str](
    FunctionModel(grandchild_model, model_name='runtime-grandchild-model'),
    name='runtime-grandchild',
    deps_type=RuntimeDeps,
    retries=1,
    capabilities=[grandchild_runtime],
)


@grandchild.tool
async def grandchild_lookup(ctx: RunContext[RuntimeDeps]) -> ToolEvidence:
    """Retry once, then return grandchild tool-process evidence."""
    if ctx.retry == 0:
        raise ModelRetry(f'retry remote grandchild attempt from pid {os.getpid()}')
    return ToolEvidence(
        pid=os.getpid(),
        retry_count=ctx.retry,
        value=f'{ctx.deps["prefix"]}:grandchild',
    )


_seen_events: list[RuntimeEventEvidence] = []
child_hooks = Hooks[RuntimeDeps]()


@child_hooks.on.event(RuntimeEffectEvent)
async def record_runtime_effect(ctx: RunContext[RuntimeDeps], event: RuntimeEffectEvent) -> None:
    """Record effects only after they reach the child agent's active caller."""
    del ctx
    _seen_events.append(RuntimeEventEvidence(label=event.label, sequence=event.sequence))


child_runtime = RenderWorkflows[RuntimeDeps](workflows, deps_type=RuntimeDeps)
child = Agent[RuntimeDeps, str](
    FunctionModel(child_model, stream_function=stream_child_model, model_name='runtime-child-model'),
    name='runtime-child',
    deps_type=RuntimeDeps,
    capabilities=[
        SubAgents(agents=[SubAgent(grandchild)], agent_folders=None),
        child_hooks,
        child_runtime,
    ],
)


async def _sibling_effect(ctx: RunContext[RuntimeDeps], label: str, amount: int) -> ToolEvidence:
    ctx.usage.incr(RunUsage(details={'runtime_remote_marker': amount, f'{label}_marker': amount}))
    await ctx.emit(RuntimeEffectEvent(label=label, sequence=1))
    await ctx.emit(RuntimeEffectEvent(label=label, sequence=2))
    return ToolEvidence(pid=os.getpid(), retry_count=ctx.retry, value=f'{ctx.deps["prefix"]}:{label}')


@child.tool
async def child_alpha(ctx: RunContext[RuntimeDeps]) -> ToolEvidence:
    """Emit alpha effects from one concurrent remote operation."""
    return await _sibling_effect(ctx, 'alpha', 2)


@child.tool
async def child_beta(ctx: RunContext[RuntimeDeps]) -> ToolEvidence:
    """Emit beta effects from one concurrent remote operation."""
    return await _sibling_effect(ctx, 'beta', 5)


parent_runtime = RenderWorkflows[RuntimeDeps](workflows, deps_type=RuntimeDeps)
parent = Agent[RuntimeDeps, str](
    FunctionModel(parent_model, model_name='runtime-parent-model'),
    name='runtime-parent',
    deps_type=RuntimeDeps,
    capabilities=[
        SubAgents(agents=[SubAgent(child)], agent_folders=None),
        parent_runtime,
    ],
)


@parent_runtime.task(name='run-local-runtime-agent')
async def run_local_runtime_agent(ctx: TaskContext, prompt: str, deps: RuntimeDeps) -> dict[str, object]:
    """Run the nested public agent entry point inside a Render root task."""
    del ctx
    _seen_events.clear()
    usage = RunUsage(details={'root_marker': 3})
    validated_deps = TypeAdapter(RuntimeDeps).validate_python(deps)
    result = await parent.run(prompt, deps=validated_deps, usage=usage)
    payload = RootTaskResult(
        controller_pid=validated_deps['controller_pid'],
        deps_prefix=validated_deps['prefix'],
        events=list(_seen_events),
        output=ParentOutput.model_validate_json(result.output),
        root_pid=os.getpid(),
        usage_markers={
            name: usage.details.get(name, 0) for name in ('runtime_remote_marker', 'alpha_marker', 'beta_marker')
        },
    )
    dumped = TypeAdapter(RootTaskResult).dump_python(payload, mode='json')
    return TypeAdapter(dict[str, object]).validate_python(dumped)


def memory_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Write and read memory in separate tasks, then exercise the public tracer."""
    del info
    returned = _tool_returns(messages)
    if trace := returned.get('trace_memory'):
        return ModelResponse(parts=[TextPart(str(trace.content))])
    if read := returned.get('read_memory'):
        return ModelResponse(parts=[ToolCallPart('trace_memory', {'content': str(read.content)})])
    if 'write_memory' in returned:
        return ModelResponse(parts=[ToolCallPart('read_memory', {'file': 'MEMORY.md'})])
    return ModelResponse(parts=[ToolCallPart('write_memory', {'content': 'process-shared memory'})])


memory_exporter = InMemorySpanExporter()
memory_provider = TracerProvider()
memory_provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
memory_runtime = RenderWorkflows[MemoryRuntimeDeps](workflows, deps_type=MemoryRuntimeDeps)
memory_agent = Agent(
    FunctionModel(memory_model, model_name='runtime-memory-model'),
    name='runtime-memory',
    deps_type=MemoryRuntimeDeps,
    capabilities=[
        Memory(
            store_resolver=lambda ctx: SqliteMemoryStore(database=ctx.deps['database']),
            namespace=lambda ctx: ctx.deps['tenant'],
        ),
        memory_runtime,
    ],
)
memory_agent.instrument = InstrumentationSettings(tracer_provider=memory_provider, include_content=False)


@memory_agent.tool
async def trace_memory(ctx: RunContext[MemoryRuntimeDeps], content: str) -> str:
    """Return evidence that a worker-local span reached its configured exporter."""
    with ctx.tracer.start_as_current_span('memory.worker') as span:
        span.set_attribute('worker.pid', os.getpid())
    return MemoryTaskResult(
        root_pid=0,
        tool_pid=os.getpid(),
        content=content,
        span_exported=any(span.name == 'memory.worker' for span in memory_exporter.get_finished_spans()),
    ).model_dump_json()


@memory_runtime.task(name='run-local-memory-agent')
async def run_local_memory_agent(ctx: TaskContext, deps: MemoryRuntimeDeps) -> dict[str, object]:
    """Exercise memory and tracing with a shared SQLite file in the local runtime."""
    del ctx
    validated_deps = TypeAdapter(MemoryRuntimeDeps).validate_python(deps)
    result = await memory_agent.run('remember and read', deps=validated_deps)
    evidence = MemoryTaskResult.model_validate_json(result.output)
    evidence.root_pid = os.getpid()
    return evidence.model_dump(mode='json')


if __name__ == '__main__':
    workflows.start()
