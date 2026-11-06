"""Tests for the workflow agent composition and its policies."""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

import pytest

from pydantic_ai import AgentRunResult, ModelRequestContext, RunContext, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from gh_aw_agent import _budgets, _compaction, _policies, _result, _safe_outputs


def _tool(name: str) -> ToolDefinition:
    return ToolDefinition(name=name)


def _context() -> RunContext[object]:
    return RunContext(deps=None, model=TestModel(call_tools=[]), usage=RunUsage())


def _request_context(*names: str) -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(call_tools=[]),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(function_tools=[_tool(name) for name in names]),
    )


def test_safe_output_type_matches_engine_and_shim_prefixes() -> None:
    assert _safe_outputs.safe_output_type('safeoutputs_noop') == 'noop'
    assert _safe_outputs.safe_output_type('mcp__safeoutputs__create_issue') == 'create_issue'
    assert _safe_outputs.safe_output_type('github_create_issue') is None
    assert _safe_outputs.safe_output_type('shell') is None


def test_review_run_needs_a_submitted_review(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = tmp_path / 'safe-outputs.jsonl'
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    assert _safe_outputs.safe_output_status() == (True, None)  # a missing sink is not a review
    sink.write_text('{"type":"create_pull_request_review_comment"}\n')
    assert _safe_outputs.safe_output_status() == (True, None)
    sink.write_text('{"type":"submit_pull_request_review"}\n')
    assert _safe_outputs.safe_output_status() == (False, None)


def test_attention_run_reports_missing_decisions(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'attention-candidates.json').write_text(json.dumps({'candidates': [{'number': 1}, {'number': 2}]}))
    sink = tmp_path / 'safe-outputs.jsonl'
    sink.write_text('{"type":"record_attention_decision","item_number":"1"}\n')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Attention Triage')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))
    monkeypatch.setenv('GITHUB_WORKSPACE', str(workspace))
    assert _safe_outputs.safe_output_status() == (True, [2])


def test_shell_disabled_withholds_and_answers_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('PYDANTIC_AI_SHELL_DISABLED', '1')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Docs Drift')
    request = _request_context('read_file', 'shell', 'safeoutputs_noop')
    updated = asyncio.run(_policies.ToolPolicy().before_model_request(_context(), request))
    assert [tool.name for tool in updated.model_request_parameters.function_tools] == [
        'read_file',
        'safeoutputs_noop',
    ]

    async def handler(args: dict[str, object]) -> object:
        raise AssertionError('a disabled shell must not run')

    answer = asyncio.run(
        _policies.ToolPolicy().wrap_tool_execute(
            _context(),
            call=ToolCallPart('shell', {'command': 'ls'}),
            tool_def=_tool('shell'),
            args={'command': 'ls'},
            handler=handler,
        )
    )
    assert answer == _policies.SHELL_DISABLED_MESSAGE


def test_shell_is_offered_without_the_disable_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PYDANTIC_AI_SHELL_DISABLED', raising=False)
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Docs Drift')
    request = _request_context('read_file', 'shell')
    updated = asyncio.run(_policies.ToolPolicy().before_model_request(_context(), request))
    assert [tool.name for tool in updated.model_request_parameters.function_tools] == ['read_file', 'shell']


def test_ci_review_withholds_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    request = _request_context('submit_pull_request_review', 'safeoutputs_noop', 'safeoutputs_missing_tool')
    updated = asyncio.run(_policies.ToolPolicy().before_model_request(_context(), request))
    assert [tool.name for tool in updated.model_request_parameters.function_tools] == [
        'submit_pull_request_review',
        'safeoutputs_missing_tool',
    ]


def test_ci_review_inside_the_reserve_offers_only_publishing_outputs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setattr(_policies, 'time_budget_notice', lambda: 'time is nearly up')
    request = _request_context(
        'safeoutputs_submit_pull_request_review', 'safeoutputs_create_issue', 'safeoutputs_report_incomplete', 'shell'
    )
    updated = asyncio.run(_policies.ToolPolicy().before_model_request(_context(), request))
    assert [tool.name for tool in updated.model_request_parameters.function_tools] == [
        'safeoutputs_submit_pull_request_review'
    ]


def test_other_workflows_offer_only_safe_outputs_inside_the_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Docs Drift')
    monkeypatch.setattr(_policies, 'time_budget_notice', lambda: 'time is nearly up')
    request = _request_context('read_file', 'safeoutputs_create_issue')
    updated = asyncio.run(_policies.ToolPolicy().before_model_request(_context(), request))
    assert [tool.name for tool in updated.model_request_parameters.function_tools] == ['safeoutputs_create_issue']


def test_bare_mcp_error_is_returned_to_the_model() -> None:
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    capability = _policies.RecoverMCPToolErrors()
    answer = asyncio.run(
        capability.on_tool_execute_error(
            _context(),
            call=ToolCallPart('github_issue_read', {}),
            tool_def=_tool('github_issue_read'),
            args={},
            error=McpError(ErrorData(code=-32602, message='validation failed')),
        )
    )
    assert answer == 'error: validation failed'


def test_limit_after_output_keeps_the_run_successful(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = tmp_path / 'safe-outputs.jsonl'
    sink.write_text('{"type":"create_issue"}\n')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Docs Drift')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    async def handler() -> AgentRunResult[object]:
        raise UsageLimitExceeded('request limit')

    result = asyncio.run(_budgets.LimitAfterOutputIsSuccess().wrap_run(_context(), handler=handler))
    assert result.output == 'request limit reached after the safe output was emitted'


def test_limit_without_output_still_fails(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sink = tmp_path / 'safe-outputs.jsonl'
    sink.write_text('')
    monkeypatch.setenv('GITHUB_WORKFLOW', 'Pydantic AI Docs Drift')
    monkeypatch.setenv('GH_AW_SAFE_OUTPUTS', str(sink))

    async def handler() -> AgentRunResult[object]:
        raise UsageLimitExceeded('request limit')

    with pytest.raises(UsageLimitExceeded):
        asyncio.run(_budgets.LimitAfterOutputIsSuccess().wrap_run(_context(), handler=handler))


def test_provider_health_record_success(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'provider-health-result.json'
    monkeypatch.setenv(_result.RESULT_PATH_ENV, str(path))
    monkeypatch.setenv('GITHUB_WORKFLOW', 'CI Review')
    monkeypatch.setenv('PYDANTIC_AI_TASK_KEY', 'task')
    monkeypatch.setenv('PYDANTIC_AI_TRIGGER_EVENT', 'workflow_run')
    monkeypatch.setenv('PYDANTIC_AI_RUN_ATTEMPT', '2')

    async def handler() -> AgentRunResult[object]:
        return AgentRunResult('done')

    asyncio.run(_result.RecordResult().wrap_run(_context(), handler=handler))
    assert json.loads(path.read_text()) == {
        'type': 'result',
        'provider_health': {
            'workflow': 'CI Review',
            'task_key': 'task',
            'trigger_event': 'workflow_run',
            'run_attempt': 2,
        },
    }


@pytest.mark.parametrize(
    ('status', 'body', 'kind'),
    [
        (401, {}, 'authentication'),
        (429, {'error': {'code': 1302}}, 'rate_limit'),
        (429, {'error': {'code': 1308}}, 'plan_limit'),
    ],
)
def test_provider_health_record_classifies_http_failures(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, status: int, body: dict[str, object], kind: str
) -> None:
    from pydantic_ai.exceptions import ModelHTTPError

    path = tmp_path / 'provider-health-result.json'
    monkeypatch.setenv(_result.RESULT_PATH_ENV, str(path))

    async def handler() -> AgentRunResult[object]:
        raise ModelHTTPError(status, 'zai', body)

    with pytest.raises(ModelHTTPError):
        asyncio.run(_result.RecordResult().wrap_run(_context(), handler=handler))
    record = json.loads(path.read_text())
    assert record['provider_health']['failure'] == {'kind': kind, 'http_status': status}


def test_compaction_keeps_the_prompt_and_trims_old_results(monkeypatch: pytest.MonkeyPatch) -> None:
    prompt = ModelRequest(parts=[UserPromptPart('the task')])
    messages: list[ModelMessage] = [prompt]
    for index in range(40):
        messages.append(ModelRequest(parts=[ToolReturnPart('read_file', 'x' * 12_000, f'call-{index}')]))

    class _UnavailableAgent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def run(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError('summariser unavailable')

    monkeypatch.setattr(_compaction, 'Agent', _UnavailableAgent)
    result = asyncio.run(_compaction.compact_history(_context(), messages, preserve_prompt=True))
    assert result[0] is prompt
    trimmed_result = next(part for part in result[1].parts if isinstance(part, ToolReturnPart))
    assert '[trimmed' in str(trimmed_result.content)


def test_agent_builds_for_the_engine() -> None:
    from gh_aw_agent import build_agent

    agent = build_agent()
    assert agent.name == 'gh-aw-agent'
