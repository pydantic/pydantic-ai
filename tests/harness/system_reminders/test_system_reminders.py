"""Tests for the SystemReminders capability."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.system_reminders import (
    DynamicReminder,
    GoalReanchor,
    LLMReminder,
    Reminder,
    SystemReminders,
)
from tests._recording_durability import (  # pyright: ignore[reportMissingTypeStubs]
    RecordingDurability,
    RestrictedRunContext,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _ctx(
    *,
    run_step: int = 1,
    messages: list[ModelMessage] | None = None,
    usage: RunUsage | None = None,
    usage_limits: UsageLimits | None = None,
) -> Any:
    ctx = MagicMock()
    ctx.run_step = run_step
    ctx.messages = messages if messages is not None else []
    ctx.usage = usage if usage is not None else RunUsage()
    ctx.usage_limits = usage_limits if usage_limits is not None else UsageLimits()
    return ctx


def _make_request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


async def _run_wrap(
    cap: SystemReminders[None],
    messages: list[ModelMessage],
    *,
    ctx: Any | None = None,
) -> list[ModelMessage]:
    """Invoke `wrap_model_request` with a recording handler; return what the model was given."""
    captured: dict[str, list[ModelMessage]] = {}

    async def handler(rc: ModelRequestContext) -> ModelResponse:
        captured['messages'] = list(rc.messages)
        return ModelResponse(parts=[TextPart('ok')])

    request_context = _make_request_context(messages)
    await cap.wrap_model_request(ctx or _ctx(), request_context=request_context, handler=handler)
    return captured['messages']


def _fired_text(messages: list[ModelMessage]) -> str | None:
    """The reminder text injected into the tail this turn, or None if nothing fired.

    A fired reminder is the only tail `UserPromptPart` with list content (a `CachePoint`
    followed by the joined text); an unfired turn leaves the plain string prompt as the tail.
    """
    part = messages[-1].parts[-1]
    if isinstance(part, UserPromptPart) and isinstance(part.content, list):
        return ''.join(c for c in part.content if isinstance(c, str))
    return None


def _fresh_request() -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart('hello')])]


def _all_text(messages: list[ModelMessage]) -> str:
    out: list[str] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                if isinstance(part.content, str):
                    out.append(part.content)
                else:
                    out.extend(c for c in part.content if isinstance(c, str))
            elif isinstance(part, TextPart):
                out.append(part.content)
    return '\n'.join(out)


def _has_cache_point(messages: list[ModelMessage]) -> bool:
    return any(
        isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and any(isinstance(c, CachePoint) for c in part.content)
        for msg in messages
        if isinstance(msg, ModelRequest)
        for part in msg.parts
    )


def _injected_leads_with_cache_point(messages: list[ModelMessage]) -> bool:
    """Whether the reminder appended this turn (the tail's last part) leads with a `CachePoint`."""
    part = messages[-1].parts[-1]
    return (
        isinstance(part, UserPromptPart)
        and isinstance(part.content, list)
        and len(part.content) > 0
        and isinstance(part.content[0], CachePoint)
    )


# --- Reminder validation ---


class TestReminderValidation:
    def test_defaults(self) -> None:
        r = Reminder('text')
        assert r.interval == 1
        assert r.first_after is None
        assert r.tag == 'system-reminder'

    def test_zero_interval_raises(self) -> None:
        with pytest.raises(ValueError, match='interval must be >= 1'):
            Reminder('t', interval=0)

    def test_zero_first_after_raises(self) -> None:
        with pytest.raises(ValueError, match='first_after must be >= 1'):
            Reminder('t', first_after=0)

    def test_zero_max_fires_raises(self) -> None:
        with pytest.raises(ValueError, match='max_fires must be >= 1'):
            Reminder('t', max_fires=0)


class TestRenderContent:
    async def test_default_tag_wraps(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('stay focused')])
        assert (
            _fired_text(await _run_wrap(cap, _fresh_request())) == '<system-reminder>\nstay focused\n</system-reminder>'
        )

    async def test_custom_tag(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('x', tag='note')])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == '<note>\nx\n</note>'

    async def test_no_tag_raw(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('x', tag=None)])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'x'


# --- SystemReminders validation ---


class TestSystemRemindersValidation:
    def test_requires_at_least_one(self) -> None:
        with pytest.raises(ValueError, match='At least one'):
            SystemReminders[None]()

    def test_static_only(self) -> None:
        assert len(SystemReminders[None](reminders=[Reminder('a')]).reminders) == 1

    def test_dynamic_only(self) -> None:
        assert len(SystemReminders[None](dynamic_reminders=[lambda ctx: 'a']).dynamic_reminders) == 1

    def test_not_serializable(self) -> None:
        assert SystemReminders.get_serialization_name() is None


# --- Static cadence ---


class TestStaticCadence:
    async def test_interval_1_fires_every_request(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('always', tag=None)])
        for _ in range(3):
            seen = await _run_wrap(cap, _fresh_request())
            assert _fired_text(seen) == 'always'

    async def test_interval_3_fires_on_third(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('third', interval=3, tag=None)])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'third'

    async def test_six_turn_matrix(self) -> None:
        cap = SystemReminders[None](
            reminders=[Reminder('every 2', interval=2, tag=None), Reminder('every 3', interval=3, tag=None)]
        )
        expected = [None, 'every 2', 'every 3', 'every 2', None, 'every 2\n\nevery 3']
        for want in expected:
            assert _fired_text(await _run_wrap(cap, _fresh_request())) == want

    async def test_first_after(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('anchor', interval=15, first_after=15, tag=None)])
        for turn in range(1, 31):
            fired = _fired_text(await _run_wrap(cap, _fresh_request()))
            assert fired == ('anchor' if turn in (15, 30) else None)


# --- Trigger ---


class TestTrigger:
    async def test_trigger_true_and_interval(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('t', trigger=lambda ctx: True, tag=None)])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 't'

    async def test_trigger_false_suppresses(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('t', trigger=lambda ctx: False, tag=None)])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_trigger_reads_context(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('late', trigger=lambda ctx: ctx.run_step > 10, tag=None)])
        assert _fired_text(await _run_wrap(cap, _fresh_request(), ctx=_ctx(run_step=5))) is None
        assert _fired_text(await _run_wrap(cap, _fresh_request(), ctx=_ctx(run_step=15))) == 'late'


# --- Max fires ---


class TestMaxFires:
    async def test_caps_fires(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('limited', max_fires=2, tag=None)])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'limited'
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'limited'
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_duplicate_instance_respects_max_fires(self) -> None:
        # The same Reminder instance listed twice shares one identity budget, so max_fires=1
        # fires it once per turn, not once per list entry.
        r = Reminder('r', max_fires=1, tag=None)
        cap = SystemReminders[None](reminders=[r, r])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'r'
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_per_reminder_independence(self) -> None:
        cap = SystemReminders[None](
            reminders=[Reminder('once', max_fires=1, tag=None), Reminder('twice', max_fires=2, tag=None)]
        )
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'once\n\ntwice'
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'twice'
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_callback_growing_reminders_does_not_raise(self) -> None:
        # A user callback that appends to the reminders list must not desync the fire counts:
        # `_fire_counts` is a dict, so a newly appended reminder just starts at 0 rather than
        # indexing past a once-sized list (which previously raised IndexError). The append
        # lands after this turn's eligibility snapshot, so the new reminder fires next turn.
        reminders: list[Reminder[None]] = [Reminder('first', tag=None)]

        def grow(_text: str) -> None:
            if len(reminders) == 1:
                reminders.append(Reminder('appended', tag=None))

        cap = SystemReminders[None](reminders=reminders, on_fire=grow)
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'first'
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'first\n\nappended'

    async def test_fire_budget_follows_identity_across_removal(self) -> None:
        # `_fire_counts` keys on reminder identity, not list index, so removing an earlier
        # reminder does not shift a survivor onto a stale budget slot.
        a = Reminder('a', interval=2, tag=None)  # does not fire on request 1
        b = Reminder('b', max_fires=1, tag=None)  # fires once on request 1
        reminders: list[Reminder[None]] = [a, b]
        cap = SystemReminders[None](reminders=reminders)
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'b'
        reminders.pop(0)  # remove `a`; `b` shifts from index 1 to index 0
        # `b`'s max_fires=1 is spent; the index shift must not revive it.
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_self_appending_dynamic_reminder_does_not_loop(self) -> None:
        # `_collect_dynamic` snapshots the sequence, so a reminder that appends to it does not
        # extend this turn's iteration into a hang.
        dynamics: list[DynamicReminder[None]] = []

        def self_append(ctx: Any) -> str | None:
            dynamics.append(lambda _c: 'never-reached')
            return 'once'

        dynamics.append(self_append)
        cap = SystemReminders[None](dynamic_reminders=dynamics)
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'once'

    async def test_dynamic_error_leaves_static_fire_state_untouched(self) -> None:
        # A raising dynamic reminder aborts the request; static fire state and on_fire are
        # committed only after injection, so the static reminder's budget is not consumed and
        # no false fire is reported.
        fired: list[str] = []

        def boom(ctx: Any) -> str | None:
            raise RuntimeError('dynamic down')

        cap = SystemReminders[None](
            reminders=[Reminder('s', max_fires=1, tag=None)],
            dynamic_reminders=[boom],
            on_fire=fired.append,
        )
        with pytest.raises(RuntimeError, match='dynamic down'):
            await _run_wrap(cap, _fresh_request())
        assert cap._fire_counts == {}
        assert fired == []


# --- Dynamic reminders ---


class TestDynamicReminders:
    async def test_sync_returns_text(self) -> None:
        cap = SystemReminders[None](dynamic_reminders=[lambda ctx: 'dyn'])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'dyn'

    async def test_sync_returns_none_skips(self) -> None:
        cap = SystemReminders[None](dynamic_reminders=[lambda ctx: None])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_async_returns_text(self) -> None:
        async def gen(ctx: Any) -> str | None:
            return 'async dyn'

        cap = SystemReminders[None](dynamic_reminders=[gen])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 'async dyn'

    async def test_async_returns_none(self) -> None:
        async def gen(ctx: Any) -> str | None:
            return None

        cap = SystemReminders[None](dynamic_reminders=[gen])
        assert _fired_text(await _run_wrap(cap, _fresh_request())) is None

    async def test_static_and_dynamic_join(self) -> None:
        cap = SystemReminders[None](
            reminders=[Reminder('s', tag=None)],
            dynamic_reminders=[lambda ctx: 'd'],
        )
        assert _fired_text(await _run_wrap(cap, _fresh_request())) == 's\n\nd'


# --- Injection / cache-safety mechanics ---


class TestInjectionMechanics:
    async def test_tail_leads_with_cache_point(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)], cache_ttl='1h')
        seen = await _run_wrap(cap, _fresh_request())
        last = seen[-1]
        assert isinstance(last, ModelRequest)
        reminder = last.parts[-1]
        assert isinstance(reminder, UserPromptPart)
        assert isinstance(reminder.content, list)
        assert isinstance(reminder.content[0], CachePoint)
        assert reminder.content[0].ttl == '1h'
        assert reminder.content[1] == 'r'

    async def test_default_tag_in_injection(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('stay focused')])
        assert (
            _fired_text(await _run_wrap(cap, _fresh_request())) == '<system-reminder>\nstay focused\n</system-reminder>'
        )

    async def test_original_request_not_mutated_in_place(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        original = ModelRequest(parts=[UserPromptPart('hello')])
        seen = await _run_wrap(cap, [original])
        assert len(original.parts) == 1  # append-only, fresh ModelRequest
        assert seen[-1] is not original

    async def test_no_fire_leaves_messages_untouched(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', interval=3, tag=None)])
        original = ModelRequest(parts=[UserPromptPart('hello')])
        seen = await _run_wrap(cap, [original])
        assert seen[-1] is original
        assert len(original.parts) == 1

    async def test_no_injection_when_tail_not_model_request(self) -> None:
        fired: list[str] = []
        cap = SystemReminders[None](reminders=[Reminder('r', max_fires=1, tag=None)], on_fire=fired.append)
        prior = ModelResponse(parts=[TextPart('prior')])
        seen = await _run_wrap(cap, [prior])
        assert seen[-1] is prior
        # No ModelRequest tail: no cadence slot spent, and the fire budget and on_fire untouched.
        assert cap._request_count == 0
        assert cap._fire_counts == {}
        assert fired == []

    async def test_empty_message_list_is_a_noop(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [])
        assert seen == []
        assert cap._request_count == 0


# --- CachePoint guard (leading CachePoint is illegal without preceding user content) ---


class TestCachePointGuard:
    async def test_cache_point_with_user_prompt(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, _fresh_request())
        assert _fired_text(seen) == 'r'
        assert _injected_leads_with_cache_point(seen)

    async def test_cache_point_with_tool_return(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        tail = ModelRequest(parts=[ToolReturnPart('tool', 'out', tool_call_id='c1')])
        seen = await _run_wrap(cap, [tail])
        assert _injected_leads_with_cache_point(seen)

    async def test_cache_point_with_retry_prompt(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        tail = ModelRequest(parts=[RetryPromptPart('try again', tool_name='tool', tool_call_id='c1')])
        seen = await _run_wrap(cap, [tail])
        assert _injected_leads_with_cache_point(seen)

    async def test_cache_point_with_binary_user_content(self) -> None:
        img = BinaryContent(data=b'\x00', media_type='image/png')
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [ModelRequest(parts=[UserPromptPart(content=[img])])])
        assert _injected_leads_with_cache_point(seen)

    async def test_cache_point_with_mixed_list_content(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        tail = ModelRequest(parts=[UserPromptPart(content=[CachePoint(), '', 'real'])])
        seen = await _run_wrap(cap, [tail])
        assert _injected_leads_with_cache_point(seen)

    async def test_cache_point_with_text_content(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [ModelRequest(parts=[UserPromptPart(content=[TextContent('goal')])])])
        assert _injected_leads_with_cache_point(seen)

    async def test_no_cache_point_with_empty_text_content(self) -> None:
        # An empty TextContent maps to nothing (like an empty str), so no CachePoint may lead.
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [ModelRequest(parts=[UserPromptPart(content=[TextContent('')])])])
        assert _fired_text(seen) == 'r'
        assert not _injected_leads_with_cache_point(seen)

    async def test_no_cache_point_with_system_prompt_only(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [ModelRequest(parts=[SystemPromptPart('sys')])])
        assert _fired_text(seen) == 'r'
        assert not _injected_leads_with_cache_point(seen)

    async def test_no_cache_point_with_empty_user_prompt(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [ModelRequest(parts=[UserPromptPart('')])])
        assert _fired_text(seen) == 'r'
        assert not _injected_leads_with_cache_point(seen)

    async def test_no_cache_point_with_cachepoint_only_content(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        seen = await _run_wrap(cap, [ModelRequest(parts=[UserPromptPart(content=[CachePoint()])])])
        assert _fired_text(seen) == 'r'
        assert not _injected_leads_with_cache_point(seen)


# --- on_fire callback ---


class TestOnFire:
    async def test_static_fires_callback(self) -> None:
        seen_texts: list[str] = []
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)], on_fire=seen_texts.append)
        await _run_wrap(cap, _fresh_request())
        assert seen_texts == ['r']

    async def test_dynamic_fires_callback(self) -> None:
        seen_texts: list[str] = []
        cap = SystemReminders[None](dynamic_reminders=[lambda ctx: 'd'], on_fire=seen_texts.append)
        await _run_wrap(cap, _fresh_request())
        assert seen_texts == ['d']


# --- for_run isolation ---


class TestForRun:
    async def test_resets_counters_preserves_config(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', max_fires=5, tag=None)], cache_ttl='1h')
        await _run_wrap(cap, _fresh_request())
        assert cap._request_count == 1
        assert cap._fire_counts == {id(cap.reminders[0]): 1}

        fresh = await cap.for_run(_ctx())
        assert fresh is not cap
        assert fresh._request_count == 0
        assert fresh._fire_counts == {}
        assert fresh.reminders is cap.reminders
        assert fresh.cache_ttl == '1h'

    async def test_two_runs_independent(self) -> None:
        cap = SystemReminders[None](reminders=[Reminder('r', tag=None)])
        run1 = await cap.for_run(_ctx())
        run2 = await cap.for_run(_ctx())
        await _run_wrap(run1, _fresh_request())
        assert run1._request_count == 1
        assert run2._request_count == 0


# --- GoalReanchor ---


class TestGoalReanchor:
    def test_extracts_first_user_message(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[TextPart('assistant first')]),
            ModelRequest(parts=[UserPromptPart('build the thing')]),
        ]
        text = GoalReanchor()(_ctx(messages=messages))
        assert text is not None
        assert 'build the thing' in text

    def test_skips_empty_user_text(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart('')]),
            ModelRequest(parts=[UserPromptPart('real goal')]),
        ]
        text = GoalReanchor()(_ctx(messages=messages))
        assert text is not None
        assert 'real goal' in text

    def test_list_content_joins_text_parts(self) -> None:
        img = BinaryContent(data=b'\x00', media_type='image/png')
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content=['find bugs', img])])]
        text = GoalReanchor()(_ctx(messages=messages))
        assert text is not None
        assert 'find bugs' in text

    def test_extracts_text_content_goal(self) -> None:
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=[TextContent('refactor the auth module')])])
        ]
        text = GoalReanchor()(_ctx(messages=messages))
        assert text is not None
        assert 'refactor the auth module' in text

    def test_fallback_when_no_user_message(self) -> None:
        assert GoalReanchor(fallback='hold').__call__(_ctx(messages=[])) == 'hold'

    async def test_composes_as_dynamic_reminder(self) -> None:
        cap = SystemReminders[None](dynamic_reminders=[GoalReanchor()])
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('do X')])]
        seen = await _run_wrap(cap, messages, ctx=_ctx(messages=messages))
        fired = _fired_text(seen)
        assert fired is not None
        assert 'do X' in fired


# --- LLMReminder ---


def _capture_model(store: dict[str, str], output: str = 'generated') -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        store['prompt'] = _all_text([messages[-1]])
        return ModelResponse(parts=[TextPart(output)])

    return FunctionModel(fn)


class TestLLMReminder:
    async def test_generation_dispatches_as_durable_operation(self) -> None:
        durability = RecordingDurability()
        capability = SystemReminders(
            dynamic_reminders=[
                LLMReminder(model=FunctionModel(lambda _messages, _info: ModelResponse(parts=[TextPart('refocus')])))
            ],
        )
        assert capability.id == 'system_reminders'
        agent = Agent(TestModel(call_tools=[]), name='system_reminders', capabilities=[capability, durability])

        await agent.run('stay focused')

        bound = RecordingDurability.from_agent(agent)
        assert bound is not None
        assert 'system_reminders__capability__system_reminders.generate_reminder' in {name for name, _ in bound.calls}

    async def test_durable_generation_does_not_read_worker_messages(self) -> None:
        captured: dict[str, str] = {}
        reminder = LLMReminder(model=_capture_model(captured, output='generated durably'))
        capability = SystemReminders(dynamic_reminders=[reminder])
        capability._dynamic_snapshot = (reminder,)  # pyright: ignore[reportPrivateUsage]
        ctx = RestrictedRunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            usage_limits=UsageLimits(),
        )
        ctx.unavailable_fields = frozenset({'messages'})

        result = await capability._generate_reminder(  # pyright: ignore[reportPrivateUsage]
            ctx, 0, 'user: fix durable reminders'
        )

        assert result == ('generated durably', None)
        assert captured['prompt'] == 'user: fix durable reminders'

    def test_restricted_run_context_rejects_an_excluded_field(self) -> None:
        # Negative control for the test above: without proof that the stand-in actually rejects
        # `messages`, that test would pass just as well against a stand-in that carried it.
        ctx = RestrictedRunContext(deps=None, model=TestModel(), usage=RunUsage(), usage_limits=UsageLimits())
        ctx.unavailable_fields = frozenset({'messages'})

        with pytest.raises(UserError, match="'messages' is not available in this durable operation"):
            _ = ctx.messages

    async def test_subclass_call_override_is_preserved(self) -> None:
        class GatedReminder(LLMReminder[None]):
            async def __call__(self, ctx: RunContext[None]) -> str | None:
                del ctx
                return 'subclass override'

        capability = SystemReminders(dynamic_reminders=[GatedReminder(model=_capture_model({}))])
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('keep going')])]

        seen = await _run_wrap(capability, messages, ctx=_ctx(messages=messages))

        assert _fired_text(seen) == 'subclass override'

    async def test_generation_uses_stable_snapshot_when_an_earlier_callback_mutates_configuration(self) -> None:
        def remove_self(ctx: RunContext[None]) -> None:
            del ctx
            capability.dynamic_reminders = ()
            return None

        capability = SystemReminders(
            dynamic_reminders=[remove_self, LLMReminder(model=_capture_model({}, output='generated from snapshot'))]
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('keep going')])]

        seen = await _run_wrap(capability, messages, ctx=_ctx(messages=messages))

        assert _fired_text(seen) == 'generated from snapshot'

    async def test_generation_operation_failure_falls_back_to_goal_reanchor(self) -> None:
        operation = 'system_reminders__capability__system_reminders.generate_reminder'
        seen: dict[str, list[ModelMessage]] = {}

        def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            seen['messages'] = messages
            return ModelResponse(parts=[TextPart('done')])

        capability = SystemReminders(
            id='system_reminders',
            dynamic_reminders=[LLMReminder(model=_capture_model({}, output='unreachable'))],
        )
        durability = RecordingDurability(fail_operations=frozenset({operation}))
        agent = Agent(FunctionModel(model), name='system_reminders', capabilities=[capability, durability])

        await agent.run('ship the durable fix')

        assert 'Check that your next action advances it.' in _all_text(seen['messages'])
        assert operation in {name for name, _ in durability.calls}

    async def test_generation_error_is_journaled_as_goal_reanchor(self) -> None:
        operation = 'system_reminders__capability__system_reminders.generate_reminder'
        seen: dict[str, list[ModelMessage]] = {}

        def fail_generation(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            raise RuntimeError('reminder unavailable')

        def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            seen['messages'] = messages
            return ModelResponse(parts=[TextPart('done')])

        durability = RecordingDurability()
        agent = Agent(
            FunctionModel(model),
            name='system_reminders',
            capabilities=[
                SystemReminders(
                    id='system_reminders',
                    dynamic_reminders=[LLMReminder(model=FunctionModel(fail_generation))],
                ),
                durability,
            ],
        )

        await agent.run('ship the durable fix')

        assert operation in {name for name, _ in durability.calls}
        assert 'Check that your next action advances it.' in _all_text(seen['messages'])

    async def test_generates_from_transcript(self) -> None:
        store: dict[str, str] = {}
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart('sys'), UserPromptPart('refactor auth')]),
            ModelRequest(parts=[UserPromptPart('')]),
            ModelResponse(parts=[TextPart('looking'), TextPart('')]),
        ]
        reminder = LLMReminder(model=_capture_model(store, output='  refocus now  '))
        result = await reminder(_ctx(messages=messages))
        assert result == 'refocus now'  # stripped
        assert 'Original goal: refactor auth' in store['prompt']
        assert 'user: refactor auth' in store['prompt']
        assert 'assistant: looking' in store['prompt']

    async def test_truncates_recent(self) -> None:
        store: dict[str, str] = {}
        parts = [ModelRequest(parts=[UserPromptPart('goal')])]
        parts += [ModelResponse(parts=[TextPart(f'step {i}')]) for i in range(5)]
        reminder = LLMReminder(model=_capture_model(store), max_context_messages=2)
        await reminder(_ctx(messages=parts))
        assert 'assistant: step 3' in store['prompt']
        assert 'assistant: step 4' in store['prompt']
        assert 'step 0' not in store['prompt']

    async def test_no_activity_transcript(self) -> None:
        store: dict[str, str] = {}
        reminder = LLMReminder(model=_capture_model(store))
        await reminder(_ctx(messages=[]))
        assert store['prompt'] == 'No activity yet.'

    async def test_caches_agent(self) -> None:
        store: dict[str, str] = {}
        reminder = LLMReminder(model=_capture_model(store))
        ctx = _ctx(messages=[ModelRequest(parts=[UserPromptPart('g')])])
        first = await reminder(ctx)
        agent_after_first = reminder._agent
        second = await reminder(ctx)
        assert first == second == 'generated'
        assert reminder._agent is agent_after_first

    async def test_blank_output_returns_none(self) -> None:
        store: dict[str, str] = {}
        reminder = LLMReminder(model=_capture_model(store, output='   '))
        assert await reminder(_ctx(messages=[ModelRequest(parts=[UserPromptPart('g')])])) is None

    async def test_falls_back_on_error(self) -> None:
        def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise RuntimeError('model down')

        reminder = LLMReminder(model=FunctionModel(boom))
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('ship the fix')])]
        result = await reminder(_ctx(messages=messages))
        assert result is not None
        assert 'ship the fix' in result  # GoalReanchor fallback text

    def test_zero_max_context_messages_raises(self) -> None:
        with pytest.raises(ValueError, match='max_context_messages must be >= 1'):
            LLMReminder(model=TestModel(), max_context_messages=0)

    async def test_threads_usage_onto_parent_run(self) -> None:
        store: dict[str, str] = {}
        ctx = _ctx(messages=[ModelRequest(parts=[UserPromptPart('g')])])
        await LLMReminder(model=_capture_model(store))(ctx)
        assert ctx.usage.requests == 1

    async def test_reserves_a_request_for_the_pending_parent_call(self) -> None:
        """The last slot belongs to the parent request that already cleared its preflight check."""
        store: dict[str, str] = {}
        ctx = _ctx(
            messages=[ModelRequest(parts=[UserPromptPart('ship the fix')])],
            usage=RunUsage(requests=1),
            usage_limits=UsageLimits(request_limit=2),
        )
        result = await LLMReminder(model=_capture_model(store))(ctx)
        assert result is not None
        assert 'ship the fix' in result  # GoalReanchor fallback, no nested request spent
        assert ctx.usage.requests == 1

    async def test_generates_while_budget_remains(self) -> None:
        store: dict[str, str] = {}
        ctx = _ctx(
            messages=[ModelRequest(parts=[UserPromptPart('g')])],
            usage=RunUsage(requests=1),
            usage_limits=UsageLimits(request_limit=3),
        )
        assert await LLMReminder(model=_capture_model(store))(ctx) == 'generated'
        assert ctx.usage.requests == 2

    async def test_unlimited_requests_are_passed_through(self) -> None:
        store: dict[str, str] = {}
        ctx = _ctx(messages=[ModelRequest(parts=[UserPromptPart('g')])], usage_limits=UsageLimits(request_limit=None))
        assert await LLMReminder(model=_capture_model(store))(ctx) == 'generated'

    async def test_absent_limits_are_passed_through(self) -> None:
        store: dict[str, str] = {}
        ctx = _ctx(messages=[ModelRequest(parts=[UserPromptPart('g')])])
        ctx.usage_limits = None
        assert await LLMReminder(model=_capture_model(store))(ctx) == 'generated'


# --- End-to-end through Agent ---


class TestEndToEnd:
    async def test_runs_with_test_model(self) -> None:
        agent = Agent(TestModel(), capabilities=[SystemReminders(reminders=[Reminder('stay')])])
        result = await agent.run('do the work')
        assert result.output is not None

    async def test_reminder_reaches_model_but_not_persisted(self) -> None:
        captured: dict[str, list[ModelMessage]] = {}

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            captured['messages'] = messages
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[SystemReminders(reminders=[Reminder('stay focused')])],
        )
        result = await agent.run('go')
        assert result.output == 'done'

        sent = _all_text(captured['messages'])
        assert '<system-reminder>' in sent
        assert 'stay focused' in sent
        assert _has_cache_point(captured['messages'])
        # The invariant: neither the reminder nor its CachePoint enters the durable history.
        assert '<system-reminder>' not in _all_text(result.all_messages())
        assert not _has_cache_point(result.all_messages())

    async def test_composes_with_planning(self) -> None:
        captured: dict[str, list[ModelMessage]] = {}
        calls = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            'write_plan',
                            {'items': [{'content': 'Step A', 'status': 'in_progress'}]},
                            tool_call_id='c1',
                        )
                    ]
                )
            captured['messages'] = messages
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[Planning(), SystemReminders(reminders=[Reminder('stay focused')])],
        )
        result = await agent.run('go')
        assert result.output == 'done'
        sent = _all_text(captured['messages'])
        assert '<plan-reminder>' in sent
        assert '<system-reminder>' in sent
        # Neither ephemeral reminder is persisted.
        durable = _all_text(result.all_messages())
        assert '<plan-reminder>' not in durable
        assert '<system-reminder>' not in durable
