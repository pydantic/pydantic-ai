"""Exercise repository policy through the agent and CLAI's headless entry point."""

import asyncio
import json
import sys
import time
import warnings
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterator
from contextvars import Context
from functools import partial
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent, PydanticAIDeprecationWarning, RunContext
from pydantic_ai.messages import (
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_clai2.cli.headless import run_headless
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore

sys.path.insert(0, str(Path(__file__).parent))

import agent_provider_health as health
from pydantic_ai_gh_aw_shim import agent as workflow, cli


def streaming_model(respond: Callable[[list[ModelMessage], AgentInfo], ModelResponse]) -> FunctionModel:
    """Supply streamed responses for Coder's argument-repair capability."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for part in respond(messages, info).parts:
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ToolCallPart):
                yield {0: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args_as_dict()))}

    return FunctionModel(respond, stream_function=stream)


@pytest.fixture
def anyio_backend() -> str:
    """The existing workflow runtime uses asyncio deadlines and delegate tasks."""
    return 'asyncio'


@pytest.fixture(autouse=True)
def isolated_workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workflow identity and repository context local to each test."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_WORKSPACE', str(tmp_path))
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_ID', 'CI Review:workflow_dispatch:branch:head:run')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_dispatch')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '2')
    sink = tmp_path / 'published-review.jsonl'
    sink.write_text('{"type":"submit_pull_request_review"}\n')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))


async def test_clai2_headless_keeps_coder_tools_and_read_only_task(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real headless path retains Coder tools, Task and health metadata."""
    agent = workflow.build_agent()
    tools: set[str] = set()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages
        tools.update(tool.name for tool in info.function_tools)
        yield 'Review completed'

    with agent.override(model=FunctionModel(stream_function=respond)):
        result = await run_headless(
            text='Review the checkout',
            settings=Settings(request_limit=400, code_mode=False),
            store=SettingsStore(tmp_path / 'clai2.db'),
            project=ProjectSettings(),
            agent=agent,
        )
    assert result == 0
    assert tools == {'read_file', 'write_file', 'edit_file', 'list_files', 'grep', 'shell', 'Task'}
    output = capsys.readouterr()
    assert 'Review completed' in output.out
    artifact = tmp_path / 'agent-stdio.log'
    artifact.write_text(output.err)
    assert health.parse_run_result(artifact) == health.RunResult(
        'CI Review', 'CI Review:workflow_dispatch:branch:head:run', 'workflow_dispatch', 2, None
    )


async def test_safe_output_must_be_emitted_before_clai2_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A premature text answer retries until the safe-output sink is populated."""
    sink = tmp_path / 'safeoutputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[TextPart('About to publish')])
        if requests == 2:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        'shell',
                        {'command': 'printf \'{"type":"submit_pull_request_review"}\\n\' > "$GH_AW_SAFE_OUTPUTS"'},
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('Published')])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        result = await agent.run('Publish a safe output', usage_limits=UsageLimits(request_limit=400))
    assert result.output == 'Published'
    assert requests == 3
    assert json.loads(sink.read_text()) == {'type': 'submit_pull_request_review'}


async def test_only_safe_output_tools_remain_inside_the_time_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside the time reserve, the engine offers only gh-aw's `safeoutputs` MCP tools."""
    sink = tmp_path / 'safeoutputs.jsonl'
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setattr(cli, 'run_timeout_secs', lambda: cli.RUN_TIME_RESERVE_SECS + 60)
    offered: list[set[str]] = []

    def submit_pull_request_review(message: str) -> str:
        sink.write_text(json.dumps({'type': 'submit_pull_request_review', 'message': message}) + '\n')
        return 'posted'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages
        offered.append({tool_def.name for tool_def in info.function_tools})
        if len(offered) == 1:
            # The reserve starts while this request is in flight. Not `monkeypatch`: it would restore
            # the run's deadline after `WorkflowPolicy` clears it, leaking it into later tests.
            cli.run_deadline = time.monotonic() + cli.RUN_TIME_RESERVE_SECS - 1
            return ModelResponse(parts=[ToolCallPart('safeoutputs_submit_pull_request_review', {'message': 'posted'})])
        return ModelResponse(parts=[TextPart('done')])

    # `load_mcp_toolsets` prefixes each gh-aw MCP server's tools with its name.
    safe_outputs = FunctionToolset([submit_pull_request_review]).prefixed('safeoutputs')
    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        result = await agent.run('Review', toolsets=[safe_outputs], usage_limits=UsageLimits(request_limit=400))
    assert result.output == 'done'
    assert {'Task', 'safeoutputs_submit_pull_request_review'} < offered[0]
    assert offered[1:] == [{'safeoutputs_submit_pull_request_review'}]


async def test_attention_request_limit_is_enforced_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Attention keeps read-only tools and reports its smaller request budget."""
    monkeypatch.setenv('GITHUB_WORKFLOW', cli.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(tmp_path / 'safeoutputs.jsonl'))
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        del messages
        requests += 1
        assert {tool.name for tool in info.function_tools} == {'read_file', 'list_files', 'grep'}
        return ModelResponse(parts=[ToolCallPart('list_files', {'path': '.'})])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        exit_code = await run_headless(
            text='Keep investigating',
            settings=Settings(request_limit=cli.run_request_limit(), code_mode=False),
            store=SettingsStore(tmp_path / 'clai2.db'),
            project=ProjectSettings(),
            agent=agent,
        )
    assert exit_code == 1
    assert requests == 50
    artifact = tmp_path / 'agent-stdio.log'
    artifact.write_text(capsys.readouterr().err)
    result = health.parse_run_result(artifact)
    assert result is not None
    assert result.failure == health.Failure('request_limit', None, None)
    canonical = tmp_path / 'agent-session.jsonl'
    canonical.write_text(
        json.dumps({'type': 'session.result', 'data': {'status': 'failure', 'metadata': workflow.result_metadata}})
        + '\n'
    )
    assert health.parse_run_result(canonical) == result


async def test_request_limit_after_safe_output_preserves_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last allowed request can publish the output without a further model call."""
    sink = tmp_path / 'safeoutputs.jsonl'
    sink.write_text('{"type":"submit_pull_request_review"}\n')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[ToolCallPart('list_files', {'path': '.'})])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        result = await agent.run('Finish after publication', usage_limits=UsageLimits(request_limit=1))
    assert result.output == 'Request limit reached after the safe output was emitted'
    assert result.usage.requests == 1


async def test_checkout_guidance_stays_out_of_system_instructions(tmp_path: Path) -> None:
    """File traversal supplies checkout guidance without promoting its trust."""
    guidance = 'Checkout-specific convention with a unique marker'
    (tmp_path / 'AGENTS.md').write_text(guidance)
    (tmp_path / 'source.py').write_text('value = 1\n')
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert guidance not in (info.instructions or '')
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'source.py'})])
        assert any(
            isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) and guidance in str(part.content) for part in message.parts)
            for message in messages
        )
        return ModelResponse(parts=[TextPart('Reviewed using the checkout convention')])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        result = await agent.run('Review the checkout')
    assert result.output == 'Reviewed using the checkout convention'
    assert requests == 2


async def test_attention_snapshot_read_does_not_load_checkout_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attention reads its candidate snapshot without importing repository guidance."""
    monkeypatch.setenv('GITHUB_WORKFLOW', cli.ATTENTION_WORKFLOW)
    guidance = 'Repository guidance outside the Attention candidate snapshot'
    (tmp_path / 'AGENTS.md').write_text(guidance)
    (tmp_path / 'attention-candidates.json').write_text('{"candidates": []}')
    sink = tmp_path / 'safeoutputs.jsonl'
    sink.write_text('{"type":"noop"}\n')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    requests = 0

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal requests
        requests += 1
        assert guidance not in (info.instructions or '')
        assert guidance not in str(messages)
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'attention-candidates.json'})])
        return ModelResponse(parts=[TextPart('No candidates to triage')])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        result = await agent.run('Read the Attention candidate snapshot')
    assert result.output == 'No candidates to triage'
    assert requests == 2


async def test_deadline_reports_time_limit_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A positive deadline cancels an active headless stream and reports its cause."""
    budget = 1234
    monkeypatch.setattr(cli, 'run_timeout_secs', lambda: budget)
    loop = asyncio.get_running_loop()
    original_call_later = loop.call_later
    callbacks: list[Callable[[], object]] = []
    stream_stopped = False

    def schedule(
        delay: float, callback: Callable[..., object], *args: object, context: Context | None = None
    ) -> asyncio.TimerHandle:
        if delay == budget:
            callbacks.append(partial(callback, *args))
            return loop.call_at(loop.time() + budget, callback, *args, context=context)
        return original_call_later(delay, callback, *args, context=context)

    monkeypatch.setattr(loop, 'call_later', schedule)

    async def pending_response(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        nonlocal stream_stopped
        assert len(callbacks) == 1
        callbacks[0]()
        try:
            await asyncio.Future()
        finally:
            stream_stopped = True
        yield 'Unreachable'

    agent = workflow.build_agent()
    with agent.override(model=FunctionModel(stream_function=pending_response)):
        result = await run_headless(
            text='Investigate',
            settings=Settings(request_limit=400, code_mode=False),
            store=SettingsStore(tmp_path / 'clai2.db'),
            project=ProjectSettings(),
            agent=agent,
        )
    assert result == 1
    assert stream_stopped
    artifact = tmp_path / 'agent-stdio.log'
    artifact.write_text(capsys.readouterr().err)
    result = health.parse_run_result(artifact)
    assert result is not None
    assert result.failure == health.Failure('time_limit', None, None)
    canonical = tmp_path / 'agent-session.jsonl'
    canonical.write_text(
        json.dumps({'type': 'session.result', 'data': {'status': 'failure', 'metadata': workflow.result_metadata}})
        + '\n'
    )
    assert health.parse_run_result(canonical) == result


@pytest.fixture
def agent_spans() -> Iterator[InMemorySpanExporter]:
    """Every agent's spans, instrumented as `cli.configure_observability` instruments a workflow run."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    yield exporter
    Agent.instrument_all(False)


@pytest.mark.parametrize('chunks_before_cutoff', [0, 1])
async def test_run_budget_marks_the_spans_it_cuts_off(
    chunks_before_cutoff: int, agent_spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run budget's cancellation marks the run and its cut-off request as timed out."""
    budget = 1234
    monkeypatch.setattr(cli, 'run_timeout_secs', lambda: budget)
    loop = asyncio.get_running_loop()
    original_call_later = loop.call_later
    expire: list[Callable[[], object]] = []

    def schedule(
        delay: float, callback: Callable[..., object], *args: object, context: Context | None = None
    ) -> asyncio.TimerHandle:
        if delay == budget:
            expire.append(partial(callback, *args))
        return original_call_later(delay, callback, *args, context=context)

    monkeypatch.setattr(loop, 'call_later', schedule)

    async def cut_off(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        del messages, info
        for _ in range(chunks_before_cutoff):
            yield 'partial'
        expire[0]()
        await asyncio.Future[None]()
        yield 'Unreachable'

    async def drain(ctx: RunContext[None], events: AsyncIterable[object]) -> None:
        async for _ in events:
            pass

    agent = workflow.build_agent()
    with agent.override(model=FunctionModel(stream_function=cut_off)), pytest.raises(cli.RunTimeLimitExceeded):
        await agent.run('Investigate', event_stream_handler=drain)
    outcomes = [
        (
            (span.attributes or {}).get('gen_ai.operation.name'),
            span.status.status_code.name,
            (span.attributes or {}).get('error.type'),
        )
        for span in agent_spans.get_finished_spans()
    ]
    assert sorted(outcomes, key=repr) == [('chat', 'ERROR', 'timeout'), ('invoke_agent', 'ERROR', 'timeout')]


async def test_tool_retry_is_reported_as_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    """The pinned runtime reports a tool retry as a `RetryPromptPart`, which gh-aw must not read as success."""
    with warnings.catch_warnings():
        # `main` deprecates the part the pinned runtime still emits.
        warnings.simplefilter('ignore', PydanticAIDeprecationWarning)
        retry = RetryPromptPart(content='Validation failed', tool_name='Bash', tool_call_id='c1')  # pyright: ignore[reportDeprecated]

    async def events() -> AsyncIterator[FunctionToolResultEvent]:
        # The pinned runtime types `part` as `ToolReturnPart | RetryPromptPart`; `main` dropped the latter.
        yield FunctionToolResultEvent(part=retry)  # pyright: ignore[reportArgumentType]

    ctx: RunContext[object] = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    await cli._stream_events(ctx, events())  # pyright: ignore[reportPrivateUsage]
    [line] = capsys.readouterr().out.splitlines()
    assert json.loads(line)['message']['content'][0]['is_error'] is True


async def test_cancelled_delegate_preserves_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Usage already reported by a cancelled delegate reaches its parent."""

    async def cancelled_run(*args: object, usage: RunUsage, **kwargs: object) -> None:
        del args, kwargs
        usage.incr(RunUsage(requests=3, input_tokens=23, output_tokens=9, cache_read_tokens=4, cache_write_tokens=2))
        raise asyncio.CancelledError

    monkeypatch.setattr(cli.Agent, 'run', cancelled_run)
    usage = RunUsage()
    ctx: RunContext[object] = RunContext(deps=None, model=TestModel(), usage=usage)
    with pytest.raises(asyncio.CancelledError):
        await cli.task(ctx, 'Read source', 'Investigate a focused question')
    assert usage == RunUsage(requests=3, input_tokens=23, output_tokens=9, cache_read_tokens=4, cache_write_tokens=2)


@pytest.mark.parametrize('summary_fails', [False, True])
async def test_compaction_retains_the_exact_workflow_task(monkeypatch: pytest.MonkeyPatch, summary_fails: bool) -> None:
    """Neither a summary nor its failure can erase CLAI's original task."""
    monkeypatch.setattr(cli, 'COMPACTION_TRIGGER_CHARS', 1)
    task = ModelRequest(parts=[UserPromptPart('Required verdict, target and safe-output constraints')])
    messages: list[ModelMessage] = [task, *[ModelResponse(parts=[TextPart('Evidence ' * 100)]) for _ in range(20)]]

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        if summary_fails:
            raise RuntimeError('Summary unavailable')
        return ModelResponse(parts=[TextPart('Condensed evidence')])

    usage = RunUsage()
    ctx: RunContext[object] = RunContext(deps=None, model=FunctionModel(respond), usage=usage)
    compacted = await cli.compact_history(ctx, messages, preserve_prompt=True)
    assert compacted[0] is task
    assert len(compacted) < len(messages)
    assert compacted[-cli.COMPACTION_KEEP_RECENT :] == messages[-cli.COMPACTION_KEEP_RECENT :]
