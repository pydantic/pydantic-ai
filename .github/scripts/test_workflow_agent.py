"""Exercise repository policy through the agent and CLAI's headless entry point."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextvars import Context
from functools import partial
from pathlib import Path

import pytest

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
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
    monkeypatch.delenv('GH_AW_SAFE_OUTPUTS', raising=False)


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
                parts=[ToolCallPart('shell', {'command': 'printf \'{"type":"noop"}\\n\' > "$GH_AW_SAFE_OUTPUTS"'})]
            )
        return ModelResponse(parts=[TextPart('Published')])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)):
        result = await agent.run('Publish a safe output', usage_limits=UsageLimits(request_limit=400))
    assert result.output == 'Published'
    assert requests == 3
    assert json.loads(sink.read_text()) == {'type': 'noop'}


async def test_attention_request_limit_is_enforced_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Attention's smaller request budget produces typed health failure metadata."""
    monkeypatch.setenv('GITHUB_WORKFLOW', cli.ATTENTION_WORKFLOW)
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(tmp_path / 'safeoutputs.jsonl'))

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        return ModelResponse(parts=[ToolCallPart('list_files', {'path': '.'})])

    agent = workflow.build_agent()
    with agent.override(model=streaming_model(respond)), pytest.raises(UsageLimitExceeded):
        await agent.run('Keep investigating', usage_limits=UsageLimits(request_limit=50))
    artifact = tmp_path / 'agent-stdio.log'
    artifact.write_text(capsys.readouterr().err)
    result = health.parse_run_result(artifact)
    assert result is not None
    assert result.failure == health.Failure('request_limit', None, None)


async def test_request_limit_after_safe_output_preserves_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last allowed request can publish the output without a further model call."""
    sink = tmp_path / 'safeoutputs.jsonl'
    sink.write_text('{"type":"noop"}\n')
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
