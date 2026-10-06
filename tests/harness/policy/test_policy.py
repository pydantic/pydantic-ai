"""`PolicyRules` against a real agent run with a scripted model calling a `shell` tool (hackathon)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.policy import Policy, PolicyDecision, PolicyRule, PolicyRules, command_matches

COMMAND = 'cd repo && git push --force origin main'


def _model(command: str) -> FunctionModel:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('shell', {'command': command})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond)


async def _run(
    rule: PolicyRule, *, command: str = COMMAND, approver: Callable[[PolicyDecision], Awaitable[bool]] | None = None
) -> tuple[list[str], str, list[PolicyDecision]]:
    ran: list[str] = []
    recorded: list[PolicyDecision] = []
    agent = Agent(
        _model(command),
        capabilities=[PolicyRules(policy=lambda: Policy(rules=[rule]), approver=approver, record=recorded.append)],
    )

    @agent.tool_plain
    def shell(command: str) -> str:
        ran.append(command)
        return 'ok'

    result = await agent.run('push it')
    [request] = [m for m in result.all_messages() if isinstance(m, ModelRequest) and m is not result.all_messages()[0]]
    [returned] = [p for p in request.parts if isinstance(p, ToolReturnPart)]
    return ran, str(returned.content), recorded


def _rule(**overrides: object) -> PolicyRule:
    return PolicyRule.model_validate(
        {
            'name': 'no-force-push',
            'description': 'Never force-push.',
            'match': {'tool': 'shell', 'command': 'git push*--force*'},
        }
        | overrides
    )


async def test_observe_records_and_runs() -> None:
    ran, content, recorded = await _run(_rule(mode='observe'))
    assert ran == [COMMAND] and content == 'ok'
    assert [(d.rule, d.outcome, d.subject) for d in recorded] == [('no-force-push', 'would_deny', COMMAND)]


async def test_enforce_deny_tells_the_model() -> None:
    ran, content, recorded = await _run(_rule(mode='enforce'))
    assert ran == []
    assert content == "Blocked by your organization policy 'no-force-push': Never force-push."
    assert recorded[0].outcome == 'denied'


@pytest.mark.parametrize(('approve', 'outcome', 'runs'), [(True, 'asked_approved', 1), (False, 'asked_rejected', 0)])
async def test_enforce_ask_goes_to_the_approver(approve: bool, outcome: str, runs: int) -> None:
    async def approver(decision: PolicyDecision) -> bool:
        return approve

    ran, _, recorded = await _run(_rule(mode='enforce', action='ask'), approver=approver)
    assert len(ran) == runs and recorded[0].outcome == outcome


async def test_non_matching_command_is_untouched() -> None:
    ran, _, recorded = await _run(_rule(mode='enforce'), command='git push origin main')
    assert ran == ['git push origin main'] and recorded == []


MONTY_RULE = """
cmd = args.get("command", "")
decision = "deny" if "rm -rf /" in cmd else "allow"
"""


async def test_monty_rule_decides() -> None:
    rule = _rule(mode='enforce', match={'tool': 'shell'}, monty=MONTY_RULE)
    ran, _, recorded = await _run(rule, command='rm -rf /etc')
    assert ran == [] and recorded[0].outcome == 'denied' and recorded[0].monty_ms is not None
    ran, _, recorded = await _run(rule, command='rm -rf build')
    assert ran == ['rm -rf build'] and recorded == []


@pytest.mark.parametrize(('mode', 'runs', 'outcome'), [('enforce', 0, 'denied'), ('observe', 1, 'would_deny')])
async def test_failing_monty_rule_fails_closed_or_open(mode: str, runs: int, outcome: str) -> None:
    ran, _, recorded = await _run(_rule(mode=mode, match={'tool': 'shell'}, monty='1 / 0'), command='ls')
    assert len(ran) == runs and recorded[0].outcome == outcome and recorded[0].error is not None


@pytest.mark.parametrize(
    ('pattern', 'command', 'expected'),
    [
        ('git push*--force*', 'cd x && git push --force', True),
        ('git push*--force*', 'echo --force; git push', False),
        ('git reset --hard*', 'git status\ngit reset --hard', True),
        ('curl*', 'echo a | curl x', True),
        ('*| sh', 'curl x | sh', True),
    ],
)
def test_command_globs_match_per_segment(pattern: str, command: str, expected: bool) -> None:
    assert command_matches(pattern, command) is expected
