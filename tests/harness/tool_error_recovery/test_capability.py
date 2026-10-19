"""Tests for the `ToolErrorRecovery` capability.

Behavior tests drive real agent runs with `FunctionModel`; no API keys are needed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.capabilities import AbstractCapability, Hooks, HookTimeoutError
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.tool_error_recovery import (
    DEFAULT_BUG_TYPES,
    RecoveryOutcome,
    RecoveryPolicy,
    ToolErrorRecovery,
)
from tests.harness._recording_durability import RecordingDurability
from tests.harness.tool_error_recovery._agents import build, call_then_echo

_LOGGER_NAME = 'pydantic_ai_harness.tool_error_recovery'


# Worst-case misconfiguration: retry every exception. Control flow is re-raised before the
# classifier runs, so the body is never reached when the tests below pass.
def _retry_everything(ctx: Any, call: Any, error: BaseException) -> RecoveryOutcome:  # pragma: no cover
    return RecoveryOutcome.retry(5)


def deadline_hooks(seconds: float) -> Hooks[None]:
    """pydantic-ai's own per-tool deadline: a pass-through hook carrying a timeout."""
    hooks: Hooks[None] = Hooks()

    async def bounded(ctx: Any, *, call: Any, tool_def: Any, args: Any, handler: Any) -> Any:
        return await handler(args)

    hooks.on.tool_execute(tools=['do_thing'], timeout=seconds)(bounded)
    return hooks


def build_with_deadline(
    seconds: float,
    tool_fn: Any,
    *,
    policy: RecoveryPolicy | None = None,
    max_calls: int = 1,
    recovery_first: bool = True,
) -> Agent[None, str]:
    recovery: ToolErrorRecovery[Any] = ToolErrorRecovery(policy) if policy is not None else ToolErrorRecovery()
    deadline = deadline_hooks(seconds)
    caps: list[AbstractCapability[Any]] = [recovery, deadline] if recovery_first else [deadline, recovery]
    agent: Agent[None, str] = Agent(call_then_echo(max_calls), capabilities=caps)
    agent.tool_plain(name='do_thing')(tool_fn)
    return agent


async def never_returns() -> str:
    """A tool that waits until the deadline cancels it; no real timer is waited out."""
    await anyio.Event().wait()
    return 'never'  # pragma: no cover - cancelled by the deadline


def _attempt_fields(record: logging.LogRecord) -> tuple[Any, Any, Any]:
    return (
        getattr(record, 'recovery_action', None),
        getattr(record, 'recovery_attempt', None),
        getattr(record, 'recovery_max_attempts', None),
    )


def one_parameter(call: Any) -> RecoveryOutcome:  # pragma: no cover - rejected at construction
    return RecoveryOutcome.inform()


def unreachable() -> str:
    raise ConnectionError('db unreachable')


def broken_format(call: Any, error: BaseException, label: str | None) -> str:
    raise AttributeError('formatter is buggy')


def propagate_bugs(error: BaseException) -> RecoveryOutcome:
    return RecoveryOutcome.propagate() if isinstance(error, DEFAULT_BUG_TYPES) else RecoveryOutcome.inform()


# The five accepted shapes of a classifier signature. Bodies are identical; only the
# parameters differ, which is what `ctx` detection reads.
def plain(call: Any, error: BaseException) -> RecoveryOutcome:
    return propagate_bugs(error)


def with_ctx(ctx: Any, call: Any, error: BaseException) -> RecoveryOutcome:
    return propagate_bugs(error)


def with_default(call: Any, error: BaseException, threshold: int = 5) -> RecoveryOutcome:
    return propagate_bugs(error)


def with_keyword_only(call: Any, error: BaseException, *, strict: bool = True) -> RecoveryOutcome:
    return propagate_bugs(error)


def with_ctx_and_default(ctx: Any, call: Any, error: BaseException, threshold: int = 5) -> RecoveryOutcome:
    return propagate_bugs(error)


class TestToolErrorRecovery:
    # tool recovery

    async def test_retry_transient_recovers_transparently(self) -> None:
        calls = [0]

        def flaky() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError('flaky network')
            return 'ok'

        r = await build(ToolErrorRecovery(), flaky).run('hi')
        assert 'RESULT:ok' in str(r.output)
        assert calls[0] == 2  # retried once, no model round-trip

    async def test_retry_httpx_errors_out_of_the_box(self) -> None:
        # httpx ships with pydantic-ai, so HTTP-based tools retry without any configuration.
        import httpx

        calls = [0]

        def flaky_http() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise httpx.ConnectTimeout('connect timed out')
            return 'ok'

        r = await build(ToolErrorRecovery(), flaky_http).run('hi')
        assert 'RESULT:ok' in str(r.output)
        assert calls[0] == 2

    async def test_inform_surfaces_error_type_not_message(self) -> None:
        # The model learns the tool and exception type; the exception message stays on the
        # operator surface, since free-form exception text is where secrets live.
        def bad() -> str:
            raise ValueError('secret detail')

        r = await build(ToolErrorRecovery(), bad).run('hi')
        assert 'failed' in str(r.output)
        assert 'ValueError' in str(r.output)
        assert 'secret detail' not in str(r.output)

    async def test_inform_label_replaces_raw_error(self) -> None:
        cap = ToolErrorRecovery(
            RecoveryPolicy(classify=lambda ctx, call, e: RecoveryOutcome.inform(label='service unavailable'))
        )

        def bad() -> str:
            raise ValueError('secret detail')

        r = await build(cap, bad).run('hi')
        assert 'service unavailable' in str(r.output)
        assert 'ValueError' not in str(r.output)
        assert 'secret detail' not in str(r.output)

    async def test_classify_without_ctx_supported(self) -> None:
        # The classifier may omit the leading RunContext (optional-ctx convention).
        def classify(call: ToolCallPart, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.inform(label=f'{call.tool_name} down')

        def bad() -> str:
            raise ValueError('x')

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), bad).run('hi')
        assert 'do_thing down' in str(r.output)

    async def test_fallback_returns_configured_value(self) -> None:
        cap = ToolErrorRecovery(RecoveryPolicy(classify=lambda ctx, call, e: RecoveryOutcome.fallback('FB')))

        def bad() -> str:
            raise ValueError('nope')

        r = await build(cap, bad).run('hi')
        assert 'FB' in str(r.output)

    async def test_bug_propagates(self) -> None:
        def buggy() -> str:
            raise KeyError('missing')

        with pytest.raises(Exception) as exc_info:
            await build(ToolErrorRecovery(), buggy).run('hi')
        assert 'KeyError' in repr(exc_info.value)

    async def test_budget_exhausted_propagates(self) -> None:
        def bad() -> str:
            raise ValueError('nope')

        cap: ToolErrorRecovery[None] = ToolErrorRecovery(max_recoveries=1)
        with pytest.raises(Exception) as exc_info:
            await build(cap, bad, max_calls=2).run('hi')
        assert 'ValueError' in repr(exc_info.value)

    async def test_per_tool_budget_exhausts(self) -> None:
        def bad() -> str:
            raise ValueError('x')

        cap: ToolErrorRecovery[None] = ToolErrorRecovery(per_tool_recoveries={'do_thing': 1})
        with pytest.raises(Exception) as exc_info:
            await build(cap, bad, max_calls=2).run('hi')
        assert 'ValueError' in repr(exc_info.value)

    async def test_no_budget_recovers_repeatedly(self) -> None:
        def bad() -> str:
            raise ValueError('nope')

        r = await build(ToolErrorRecovery(), bad, max_calls=2).run('hi')
        assert 'failed' in str(r.output)

    async def test_control_flow_modelretry_not_swallowed(self) -> None:
        cap = ToolErrorRecovery(RecoveryPolicy(classify=_retry_everything))
        calls = [0]

        def wants_retry() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise ModelRetry('give me better input')
            return 'ok'

        r = await build(cap, wants_retry).run('hi')
        assert 'RESULT:ok' in str(r.output)  # the model retried, was not fed an inform string
        assert calls[0] == 2

    async def test_control_flow_approval_not_swallowed(self) -> None:
        # ApprovalRequired must leave through the framework's deferred-tools mechanism: assert
        # how it exits (a DeferredToolRequests output naming the call), not merely that no
        # inform text appeared, which a broken capability could also satisfy.
        cap = ToolErrorRecovery(RecoveryPolicy(classify=_retry_everything))
        agent: Agent[None, str | DeferredToolRequests] = Agent(
            call_then_echo(), output_type=[str, DeferredToolRequests], capabilities=[cap]
        )

        def needs_approval() -> str:
            raise ApprovalRequired()

        agent.tool_plain(name='do_thing')(needs_approval)
        r = await agent.run('hi')
        assert isinstance(r.output, DeferredToolRequests)
        assert [c.tool_name for c in r.output.approvals] == ['do_thing']

    async def test_tool_failed_passes_through(self) -> None:
        # A tool raising ToolFailed reports a terminal failure to the model itself; recovery
        # must not intercept it, even under retry-everything.
        cap = ToolErrorRecovery(RecoveryPolicy(classify=_retry_everything))
        calls = [0]

        def report_failure() -> str:
            calls[0] += 1
            raise ToolFailed('upstream unavailable')

        r = await build(cap, report_failure).run('hi')
        assert calls[0] == 1  # passed through, not retried by recovery
        assert 'upstream unavailable' in str(r.output)

    async def test_inform_produces_failed_outcome(self) -> None:
        # inform surfaces the error via ToolFailed, so the model sees a failed tool result.
        outcomes: list[str] = []

        def capture(messages: list[Any], info: AgentInfo) -> ModelResponse:
            for m in messages:
                for p in m.parts:
                    if isinstance(p, ToolReturnPart):
                        outcomes.append(getattr(p, 'outcome', 'n/a'))
                        return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='do_thing', args={})])

        def bad() -> str:
            raise ValueError('nope')

        agent: Agent[None, str] = Agent(FunctionModel(capture), capabilities=[ToolErrorRecovery()])
        agent.tool_plain(name='do_thing')(bad)
        await agent.run('hi')
        assert outcomes == ['failed']

    async def test_deadline_informs_without_retrying_by_default(self) -> None:
        # A deadline is wall-clock and may have fired after the call reached the server, so the
        # default must not retry it, even though HookTimeoutError subclasses TimeoutError, whose
        # members are retried.
        calls = [0]

        async def too_slow() -> str:
            calls[0] += 1
            return await never_returns()

        r = await build_with_deadline(0.05, too_slow).run('hi')
        assert calls[0] == 1
        assert 'HookTimeoutError' in str(r.output)

    async def test_a_deadline_listed_before_recovery_is_out_of_reach(self) -> None:
        # Capabilities wrap in list order, so a deadline listed first wraps the recovery rather than
        # the other way round: its expiry ends the run before the classifier runs. Nothing warns
        # about the order, so the docs state it as a rule and this pins it.
        seen: list[str] = []

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:  # pragma: no cover
            seen.append(type(error).__name__)
            return RecoveryOutcome.inform(label='caught')

        agent = build_with_deadline(0.05, never_returns, policy=RecoveryPolicy(classify=classify), recovery_first=False)
        with pytest.raises(HookTimeoutError):
            await agent.run('hi')
        assert seen == []

    async def test_deadline_retry_is_opt_in(self) -> None:
        # An idempotent tool can opt back into the transparent retry.
        calls = [0]

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            assert isinstance(error, HookTimeoutError)
            return RecoveryOutcome.retry(3)

        async def slow_then_fast() -> str:
            calls[0] += 1
            if calls[0] == 1:
                return await never_returns()  # overruns the deadline on the first attempt
            return 'ok'

        r = await build_with_deadline(0.05, slow_then_fast, policy=RecoveryPolicy(classify=classify)).run('hi')
        assert 'RESULT:ok' in str(r.output)
        assert calls[0] == 2

    async def test_deadline_and_tool_raised_timeout_are_distinguishable(self) -> None:
        # The two mean opposite things: a tool-raised timeout means the request never got
        # through (safe to retry), a deadline may have executed.
        seen: list[type[BaseException]] = []

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            seen.append(type(error))
            return RecoveryOutcome.inform()

        def raises_own_timeout() -> str:
            raise TimeoutError('client read timeout')

        policy = RecoveryPolicy(classify=classify)
        await build_with_deadline(0.05, never_returns, policy=policy).run('hi')
        await build(ToolErrorRecovery(policy), raises_own_timeout).run('hi')

        assert seen == [HookTimeoutError, TimeoutError]

    async def test_sync_tool_deadline_informs_by_default(self) -> None:
        # A sync tool runs in a worker thread, and the deadline abandons that thread instead of
        # waiting for it, so a blocking tool overruns like an async one and the default must
        # inform, not retry: the thread keeps running and may still have side effects. The sleep
        # is real because a thread cannot be cancelled.
        def blocking() -> str:
            time.sleep(0.15)
            return 'sync-ok'

        r = await build_with_deadline(0.05, blocking).run('hi')
        assert 'HookTimeoutError' in str(r.output)
        assert 'sync-ok' not in str(r.output)

    async def test_async_classify_supported(self) -> None:
        async def classify(ctx: Any, call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.inform(label='async ok')

        def bad() -> str:
            raise ValueError('x')

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), bad).run('hi')
        assert 'async ok' in str(r.output)

    async def test_retry_with_backoff(self) -> None:
        calls = [0]

        def flaky() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError('x')
            return 'ok'

        def classify(ctx: Any, call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(3, backoff=0.001)

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), flaky).run('hi')
        assert 'RESULT:ok' in str(r.output)
        assert calls[0] == 2

    async def test_retry_exhaustion_falls_to_inform(self) -> None:
        # A tool that never recovers is retried max_attempts times, then informs the model.
        calls = [0]

        def always_timeout() -> str:
            calls[0] += 1
            raise TimeoutError('always')

        r = await build(ToolErrorRecovery(), always_timeout).run('hi')
        assert calls[0] == 3  # default retry(3): all attempts spent
        assert 'failed' in str(r.output)
        assert 'TimeoutError' in str(r.output)

    async def test_a_changed_error_type_ends_the_retrying(self) -> None:
        # `classify` decides again on every attempt, so the second failure's own branch
        # governs: retrying stops even though the first one asked for three attempts.
        calls = [0]

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            if isinstance(error, TimeoutError):
                return RecoveryOutcome.retry(3)
            return RecoveryOutcome.inform(label='stopped')

        def timeout_then_value_error() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError('x')
            raise ValueError('y')

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), timeout_then_value_error).run('hi')
        assert calls[0] == 2  # retried once for TimeoutError, then stopped on ValueError
        assert 'stopped' in str(r.output)

    async def test_retry_custom_on_exhausted(self) -> None:
        # An explicit on_exhausted outcome replaces the inform default after exhaustion.
        calls = [0]

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(2, on_exhausted=RecoveryOutcome.fallback('EXHAUSTED'))

        def always_timeout() -> str:
            calls[0] += 1
            raise TimeoutError('x')

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), always_timeout).run('hi')
        assert calls[0] == 2
        assert 'EXHAUSTED' in str(r.output)

    async def test_callable_backoff_invoked_per_attempt(self) -> None:
        seen: list[int] = []

        def backoff(attempt: int) -> float:
            seen.append(attempt)
            return 0.0

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(3, backoff=backoff)

        calls = [0]

        def flaky() -> str:
            calls[0] += 1
            if calls[0] < 3:
                raise TimeoutError('x')
            return 'ok'

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), flaky).run('hi')
        assert 'RESULT:ok' in str(r.output)
        assert seen == [0, 1]  # called once per retried attempt, 0-based

    def test_classifier_arity_is_rejected_at_construction(self) -> None:
        # Otherwise the mismatch surfaces as a TypeError inside the recovery path, at the first
        # tool failure, which is the worst moment to learn that the signature was wrong.
        with pytest.raises(UserError):
            RecoveryPolicy(classify=one_parameter)  # pyright: ignore[reportArgumentType]

    async def test_var_positional_classifier_is_left_alone(self) -> None:
        # `*args` says nothing about the shape, so no arity is enforced and the payload stays (call, error).
        seen: list[int] = []

        def classify(*args: Any) -> RecoveryOutcome:
            seen.append(len(args))
            return RecoveryOutcome.inform(label='varargs')

        def bad() -> str:
            raise ValueError('x')

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), bad).run('hi')
        assert 'varargs' in str(r.output)
        assert seen == [2]

    def test_max_message_len_must_be_positive(self) -> None:
        # max_message_len=0 would silently disable the cap (`text[:-1] + '…'` keeps the
        # original length); reject it like the other validated public fields.
        with pytest.raises(UserError):
            RecoveryPolicy(max_message_len=0)

    async def test_inform_expose_message_sends_error_text(self) -> None:
        # The explicit opt-in sends str(error) to the model, for tools whose exception text
        # is meant to be seen (validation detail the model can adapt to).
        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.inform(label='lookup failed', expose_message=True)

        def bad() -> str:
            raise ValueError('item 42 is out of stock')

        cap = ToolErrorRecovery(RecoveryPolicy(classify=classify))
        r = await build(cap, bad, max_calls=2).run('hi')
        assert 'lookup failed' in str(r.output)
        assert 'item 42 is out of stock' in str(r.output)

    async def test_expose_message_capped_at_max_message_len(self) -> None:
        # An exposed message is uncurated content, so the renderer cap applies to it, unlike a
        # pure label, which is curated operator config and exempt.
        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.inform(expose_message=True)

        def bad() -> str:
            raise ValueError('x' * 500)

        cap = ToolErrorRecovery(RecoveryPolicy(classify=classify, max_message_len=60))
        r = await build(cap, bad, max_calls=2).run('hi')
        assert len(str(r.output)) == len('RESULT:') + 60  # echo prefix + capped inform text

    def test_expose_message_only_valid_for_inform(self) -> None:
        with pytest.raises(UserError):
            RecoveryOutcome(action='fallback', expose_message=True)

    async def test_on_exhausted_propagate_logged_as_propagate(self, caplog: pytest.LogCaptureFixture) -> None:
        # A deliberate on_exhausted=propagate() must not be mislabeled as 'budget-exhausted'.
        calls = [0]

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(2, on_exhausted=RecoveryOutcome.propagate())

        def always_timeout() -> str:
            calls[0] += 1
            raise TimeoutError('x')

        with pytest.raises(Exception) as exc_info:
            await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), always_timeout).run('hi')
        assert calls[0] == 2
        assert 'TimeoutError' in repr(exc_info.value)
        actions = [getattr(rec, 'recovery_action', None) for rec in caplog.records]
        assert 'propagate' in actions
        assert 'budget-exhausted' not in actions

    async def test_retries_are_logged_at_info_with_attempt_counts(self, caplog: pytest.LogCaptureFixture) -> None:
        # DEBUG is below any level a deployment runs at, so a chain that needed three tries
        # would look like one that worked first time. The terminal record says how many tries
        # preceded it, so an incident after exhausted retries mentions them.
        calls = [0]

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(3)

        def always_timeout() -> str:
            calls[0] += 1
            raise TimeoutError('x')

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), always_timeout).run('hi')
        assert calls[0] == 3
        records = [(r.levelno, *_attempt_fields(r)) for r in caplog.records]
        assert records == [
            (logging.INFO, 'retry', 1, 3),
            (logging.INFO, 'retry', 2, 3),
            (logging.WARNING, 'inform', 3, 3),
        ]

    async def test_terminal_record_without_retries_counts_a_single_attempt(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def bad() -> str:
            raise ValueError('nope')

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await build(ToolErrorRecovery(), bad).run('hi')
        assert [_attempt_fields(r) for r in caplog.records] == [('inform', 1, 1)]

    async def test_attempt_count_survives_a_changed_verdict(self, caplog: pytest.LogCaptureFixture) -> None:
        # The inform verdict that ends the chain carries max_attempts=1, but two attempts
        # were made; the record reports the attempts made, not a cap smaller than the attempt.
        calls = [0]

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(3) if isinstance(error, TimeoutError) else RecoveryOutcome.inform()

        def timeout_then_value_error() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise TimeoutError('x')
            raise ValueError('y')

        with caplog.at_level(logging.INFO, logger=_LOGGER_NAME):
            await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), timeout_then_value_error).run('hi')
        assert [_attempt_fields(r) for r in caplog.records] == [('retry', 1, 3), ('inform', 2, 2)]

    async def test_for_run_resets_budget_across_runs(self) -> None:
        # The per-run counters are re-initialized by for_run's `replace` (fresh default_factory
        # dict, not shared), so a second run gets its full budget again.
        def bad() -> str:
            raise ValueError('nope')

        cap: ToolErrorRecovery[None] = ToolErrorRecovery(max_recoveries=1)
        agent = build(cap, bad)
        for _ in range(2):  # each run recovers its single failure; no cross-run exhaustion
            r = await agent.run('hi')
            assert 'failed' in str(r.output)

    async def test_budget_spans_the_run_under_a_durable_engine(self) -> None:
        # `wrap_tool_execute` is never a durable operation, so under a durable engine it runs on
        # the run's own copy of the capability for every tool call, and the budget that copy
        # holds is not re-created per call: the second failure of the run exhausts it.
        def bad() -> str:
            raise ValueError('nope')

        cap: ToolErrorRecovery[None] = ToolErrorRecovery(max_recoveries=1)
        caps: list[AbstractCapability[Any]] = [RecordingDurability(), cap]
        agent: Agent[None, str] = Agent(call_then_echo(2), name='tool_error_recovery', capabilities=caps)
        agent.tool_plain(name='do_thing')(bad)
        with pytest.raises(Exception) as exc_info:
            await agent.run('hi')
        assert 'ValueError' in repr(exc_info.value)

    # outcome & policy units

    def test_outcome_validation(self) -> None:
        with pytest.raises(UserError):
            RecoveryOutcome.retry(0)
        with pytest.raises(UserError):
            RecoveryOutcome.retry(3, on_exhausted=RecoveryOutcome.retry(2))
        with pytest.raises(UserError):
            RecoveryOutcome(action='inform', value='x')
        with pytest.raises(UserError):
            RecoveryOutcome(action='propagate', label='x')

    def test_render_variants(self) -> None:
        call = ToolCallPart(tool_name='t', args={})
        err = ValueError('x' * 500)
        assert RecoveryPolicy(format_error=lambda c, e, lbl: 'CUSTOM').render(call, err, None) == 'CUSTOM'
        out = RecoveryPolicy(include_traceback=True, max_message_len=40).render(call, err, None)
        assert out.endswith('…')
        assert len(out) == 40

    def test_serialization_opt_out(self) -> None:
        assert ToolErrorRecovery.get_serialization_name() is None

    # callable bugs

    async def test_classifier_bug_crashes_and_keeps_the_original(self) -> None:
        # A bug in `classify` stays fatal, like DEFAULT_BUG_TYPES for tools, but it must not
        # erase the failure it was judging. Implicit chaining does not survive the task-group
        # unwrapping around tool calls, so the original is attached explicitly.
        def broken_classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return dict[str, RecoveryOutcome]()['missing']

        cap = ToolErrorRecovery(RecoveryPolicy(classify=broken_classify))
        with pytest.raises(KeyError) as exc_info:
            await build(cap, unreachable).run('hi')
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    async def test_backoff_bug_crashes_and_keeps_the_original(self) -> None:
        def broken_backoff(attempt: int) -> float:
            raise ZeroDivisionError('backoff is buggy')

        def classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return RecoveryOutcome.retry(3, backoff=broken_backoff)

        cap = ToolErrorRecovery(RecoveryPolicy(classify=classify))
        with pytest.raises(ZeroDivisionError) as exc_info:
            await build(cap, unreachable).run('hi')
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    async def test_format_error_bug_crashes_and_keeps_the_original(self) -> None:
        cap = ToolErrorRecovery(
            RecoveryPolicy(classify=lambda call, e: RecoveryOutcome.inform(), format_error=broken_format)
        )
        with pytest.raises(AttributeError) as exc_info:
            await build(cap, unreachable).run('hi')
        assert isinstance(exc_info.value.__cause__, ConnectionError)

    async def test_callable_bug_logs_both_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        # Reading the log alone has to show both: that the classifier is broken, and the failure
        # it swallowed. The error dimension stays the incident, so a dashboard keeps counting
        # ConnectionErrors rather than the classifier's KeyError.
        def broken_classify(call: Any, error: BaseException) -> RecoveryOutcome:
            return dict[str, RecoveryOutcome]()['missing']

        cap = ToolErrorRecovery(RecoveryPolicy(classify=broken_classify))
        with pytest.raises(KeyError):
            await build(cap, unreachable).run('hi')
        records = [r for r in caplog.records if getattr(r, 'recovery_action', None) == 'classify-failed']
        assert len(records) == 1
        assert records[0].levelno == logging.ERROR
        assert getattr(records[0], 'recovery_error', None) == 'ConnectionError'
        assert getattr(records[0], 'recovery_bug', None) == 'KeyError'
        assert 'KeyError' in records[0].getMessage()
        assert 'db unreachable' in records[0].getMessage()

    async def test_no_recovery_warning_when_rendering_fails(self, caplog: pytest.LogCaptureFixture) -> None:
        # The WARNING states that a recovery happened, so it may only be written once the text
        # it announces exists. Logged first, it made a crashed run look recovered.
        cap = ToolErrorRecovery(
            RecoveryPolicy(classify=lambda call, e: RecoveryOutcome.inform(), format_error=broken_format)
        )
        with pytest.raises(AttributeError):
            await build(cap, unreachable).run('hi')
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    async def test_a_classifier_raising_control_flow_is_reported_as_a_bug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A classifier returns its verdict; raising `ModelRetry` is a back door around the four
        # outcomes. The signal still reaches the framework untouched, but it is logged as a bug.
        calls = [0]

        def classify_asks_the_model(call: Any, error: BaseException) -> RecoveryOutcome:
            raise ModelRetry('classifier wants the model to try again')

        def flaky() -> str:
            calls[0] += 1
            if calls[0] == 1:
                raise ConnectionError('transient')
            return 'ok'

        r = await build(ToolErrorRecovery(RecoveryPolicy(classify=classify_asks_the_model)), flaky).run('hi')
        assert 'RESULT:ok' in str(r.output)
        assert calls[0] == 2
        assert [getattr(r_, 'recovery_bug', None) for r_ in caplog.records] == ['ModelRetry']

    @pytest.mark.parametrize(
        'classify',
        [
            pytest.param(plain, id='(call, error)'),
            pytest.param(with_ctx, id='(ctx, call, error)'),
            pytest.param(with_default, id='(call, error, threshold=5)'),
            pytest.param(with_keyword_only, id='(call, error, *, strict=True)'),
            pytest.param(with_ctx_and_default, id='(ctx, call, error, threshold=5)'),
        ],
    )
    async def test_every_classifier_signature_receives_the_error(self, classify: Any) -> None:
        # `ctx` is optional, so its presence is inferred; a wrong inference shifts the arguments and
        # lands the `ToolCallPart` in `error`, leaving every bug check reading a part, not the exception.
        def buggy() -> str:
            raise KeyError('missing')

        with pytest.raises(KeyError):
            await build(ToolErrorRecovery(RecoveryPolicy(classify=classify)), buggy).run('hi')
