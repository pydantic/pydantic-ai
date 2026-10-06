"""`PolicyRules`: apply managed tool-call rules, in observe or enforce mode, with an optional Monty decider."""

from __future__ import annotations

import fnmatch
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import anyio
from opentelemetry.trace import Tracer

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.exceptions import SkipToolExecution
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.policy._models import Policy, PolicyRule

Decision = Literal['allow', 'ask', 'deny']
Outcome = Literal['would_deny', 'would_ask', 'denied', 'asked_approved', 'asked_rejected']
MAX_SUBJECT_CHARS = 500


@dataclass(frozen=True, kw_only=True)
class PolicyDecision:
    """One rule that matched a tool call (or an MCP server), and what came of it."""

    rule: str
    mode: str
    action: str
    outcome: Outcome
    tool_name: str
    subject: str
    description: str = ''
    error: str | None = None
    """Why a Monty rule could not decide, when it failed open or closed."""
    server_name: str | None = None
    """For an MCP allowlist decision, the server's configured name (the subject is its URL or command)."""
    monty_ms: float | None = None


Approver = Callable[[PolicyDecision], Awaitable[bool]]
"""Puts an `ask` to the user; `True` lets the call run."""


def _glob(pattern: str, value: str) -> bool:
    return fnmatch.fnmatchcase(value, pattern)


def _subject(tool_name: str, args: Mapping[str, Any]) -> str:
    command = args.get('command')
    text = command if isinstance(command, str) else f'{tool_name}({", ".join(f"{k}={v!r}" for k, v in args.items())})'
    return text[:MAX_SUBJECT_CHARS]


_SEGMENT_SPLIT = re.compile(r'&&|\|\||;|\||\n')
_PIPE_OUTSIDE_CLASS = re.compile(r'\|(?![^\[]*\])')


def command_matches(pattern: str, command: str) -> bool:
    """Whether `pattern` matches one segment of `command`, split on `&&`, `||`, `;`, `|` and newlines.

    So `git push*--force*` catches `cd repo && git push --force`. A pattern with a literal `|` outside a
    `[...]` class is about a pipeline, so it is matched against the whole command instead.
    """
    if _PIPE_OUTSIDE_CLASS.search(pattern):
        return _glob(pattern, command.strip())
    return any(_glob(pattern, segment.strip()) for segment in _SEGMENT_SPLIT.split(command))


def matches(rule: PolicyRule, tool_name: str, args: Mapping[str, Any]) -> bool:
    """Whether every part of the rule's `match` holds for this call."""
    match = rule.match
    if not _glob(match.tool, tool_name):
        return False
    if match.command is not None:
        command = args.get('command')
        if not isinstance(command, str) or not command_matches(match.command, command):
            return False
    for name, pattern in (match.args or {}).items():
        value = args.get(name)
        if value is None or not _glob(pattern, str(value)):
            return False
    return True


async def run_monty(code: str, tool_name: str, args: Mapping[str, Any], *, timeout: float) -> Decision:
    """Evaluate a rule snippet in Monty with only `tool_name` and `args` in scope and no host functions.

    Raises on a timeout, an error in the snippet, or a result that is not `'allow'`, `'ask'` or `'deny'`.
    """
    from pydantic_ai_harness._monty_exec import MontyExecutor, MontyRunState

    async def dispatch(name: str, kwargs: dict[str, Any]) -> Any:
        raise RuntimeError(f'policy rules cannot call host functions ({name})')

    source = f'decision = None\n{code}\n'
    if 'decision' in code:
        source += 'decision\n'
    # The call is data, handed in as inputs: never spliced into the source, never live objects.
    inputs = {'tool_name': tool_name, 'args': json.loads(json.dumps(dict(args), default=str))}
    state = MontyRunState()
    with anyio.fail_after(timeout):
        session = await state.get_session(
            type_check=False,
            type_check_stubs=None,
            limits={'max_feed_duration_secs': timeout, 'max_memory': 32_000_000},
            in_temporal_workflow=False,
        )
        completed = await MontyExecutor(dispatch=dispatch, valid_names=set[str](), portal=state.portal).run(
            lambda: session.feed_start(source, inputs=inputs)
        )
    result = completed.output
    if result not in ('allow', 'ask', 'deny'):
        raise ValueError(f"policy snippet returned {result!r}, not 'allow', 'ask' or 'deny'")
    return result


@dataclass(kw_only=True)
class PolicyRules(AbstractCapability[Any]):
    """Apply a managed `policy` to tool calls before they run.

    Each rule matching a call either records what it would have done (`observe`) or does it (`enforce`):
    `deny` skips the call and tells the model why, `ask` puts it to `approver` and skips it unless approved
    (with no approver, an `ask` is denied). A rule with `monty` code decides per call; a snippet that times
    out or fails lets the call through in `observe` mode and blocks it in `enforce` mode. Every match is
    reported to `record` and as a `policy decision` span on the run's tracer.
    """

    policy: Callable[[], Policy | None]
    """Read for every call, so a new published policy applies from the next tool call."""
    approver: Approver | None = None
    record: Callable[[PolicyDecision], None] | None = None
    attribute_prefix: str = 'policy'
    monty_timeout: float = 2.0
    id: str | None = field(default='policy_rules')

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='innermost')

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        policy = self.policy()
        if policy is None:
            return args
        for rule in policy.rules:
            if not matches(rule, call.tool_name, args):
                continue
            decision, error, monty_ms = await self._decide(rule, call.tool_name, args)
            if decision == 'allow':
                continue
            subject = _subject(call.tool_name, args)
            if rule.mode == 'observe':
                outcome: Outcome = 'would_deny' if decision == 'deny' else 'would_ask'
                self._emit(ctx, rule, decision, outcome, call.tool_name, subject, error, monty_ms)
                continue
            if decision == 'ask':
                pending = self._decision(rule, decision, 'asked_rejected', call.tool_name, subject, error, monty_ms)
                approved = self.approver is not None and await self.approver(pending)
                outcome = 'asked_approved' if approved else 'asked_rejected'
                self._emit(ctx, rule, decision, outcome, call.tool_name, subject, error, monty_ms)
                if approved:
                    continue
                raise SkipToolExecution(f'The user did not approve this call (policy {rule.name!r}).')
            self._emit(ctx, rule, decision, 'denied', call.tool_name, subject, error, monty_ms)
            reason = f': {rule.description}' if rule.description else '.'
            raise SkipToolExecution(f'Blocked by your organization policy {rule.name!r}{reason}')
        return args

    async def _decide(
        self, rule: PolicyRule, tool_name: str, args: Mapping[str, Any]
    ) -> tuple[Decision, str | None, float | None]:
        if rule.monty is None:
            return rule.action, None, None
        start = time.perf_counter()
        try:
            decision = await run_monty(rule.monty, tool_name, args, timeout=self.monty_timeout)
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            # Enforce mode fails closed. Observe mode never blocks, so the same `deny` is only recorded:
            # as `would_deny`, with the error, since that is what enforcing it would have done.
            return 'deny', f'{type(exc).__name__}: {exc}'[:MAX_SUBJECT_CHARS], elapsed
        return decision, None, (time.perf_counter() - start) * 1000

    def _decision(
        self,
        rule: PolicyRule,
        decision: Decision,
        outcome: Outcome,
        tool_name: str,
        subject: str,
        error: str | None,
        monty_ms: float | None,
    ) -> PolicyDecision:
        return PolicyDecision(
            rule=rule.name,
            mode=rule.mode,
            action=decision,
            outcome=outcome,
            tool_name=tool_name,
            subject=subject,
            description=rule.description,
            error=error,
            monty_ms=monty_ms,
        )

    def _emit(
        self,
        ctx: RunContext[Any],
        rule: PolicyRule,
        decision: Decision,
        outcome: Outcome,
        tool_name: str,
        subject: str,
        error: str | None,
        monty_ms: float | None,
    ) -> None:
        recorded = self._decision(rule, decision, outcome, tool_name, subject, error, monty_ms)
        if self.record is not None:
            self.record(recorded)
        emit_decision(ctx.tracer, recorded, prefix=self.attribute_prefix)


def decision_attributes(decision: PolicyDecision, *, prefix: str) -> dict[str, str | float]:
    """The span attributes of one decision, under `prefix` (e.g. `clai2.policy`)."""
    attributes: dict[str, str | float] = {
        f'{prefix}.rule': decision.rule,
        f'{prefix}.mode': decision.mode,
        f'{prefix}.action': decision.action,
        f'{prefix}.outcome': decision.outcome,
        f'{prefix}.subject': decision.subject,
        'gen_ai.tool.name': decision.tool_name,
        'logfire.msg': f'policy {decision.rule}: {decision.outcome} {decision.tool_name}',
    }
    if decision.error is not None:
        attributes[f'{prefix}.error'] = decision.error
    if decision.server_name is not None:
        attributes[f'{prefix}.server_name'] = decision.server_name
    if decision.monty_ms is not None:
        attributes[f'{prefix}.monty_ms'] = round(decision.monty_ms, 2)
    return attributes


def emit_decision(tracer: Tracer, decision: PolicyDecision, *, prefix: str) -> None:
    """Record one decision as a zero-length `policy decision` span on `tracer`."""
    with tracer.start_as_current_span('policy decision', attributes=decision_attributes(decision, prefix=prefix)):
        pass
