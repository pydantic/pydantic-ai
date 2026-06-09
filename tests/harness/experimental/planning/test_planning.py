"""Tests for the Planning capability."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.experimental.planning import PlanItem, Planning, PlanningToolset, TaskStatus
from pydantic_ai_harness.experimental.planning._toolset import PlanState, render_plan

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests on the asyncio backend (matching upstream pydantic-ai)."""
    return 'asyncio'


def _make_request_context(messages: list[ModelMessage]) -> ModelRequestContext:
    return ModelRequestContext(
        model=TestModel(),
        messages=messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )


def _all_text(messages: list[ModelMessage]) -> str:
    """Flatten user-prompt and assistant text fragments across all messages."""
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


class TestTaskStatus:
    def test_values(self) -> None:
        assert TaskStatus.pending == 'pending'
        assert TaskStatus.in_progress == 'in_progress'
        assert TaskStatus.completed == 'completed'
        assert TaskStatus.cancelled == 'cancelled'


class TestPlanItem:
    def test_default_status(self) -> None:
        item = PlanItem(content='Do something')
        assert item.status is TaskStatus.pending


class TestRenderPlan:
    def test_empty(self) -> None:
        assert render_plan([]) == 'No plan yet.'

    def test_checkboxes_and_progress(self) -> None:
        result = render_plan(
            [
                PlanItem(content='First', status=TaskStatus.completed),
                PlanItem(content='Second', status=TaskStatus.in_progress),
                PlanItem(content='Third'),
                PlanItem(content='Fourth', status=TaskStatus.cancelled),
            ]
        )
        assert result == ('1. [x] First\n2. [~] Second\n3. [ ] Third\n4. [-] Fourth\n(1/4 completed)')


class TestPlanningToolset:
    async def test_write_plan_replaces_state(self) -> None:
        state = PlanState(items=[PlanItem(content='old', status=TaskStatus.completed)])
        toolset = PlanningToolset(state)
        result = await toolset.write_plan([PlanItem(content='A'), PlanItem(content='B', status=TaskStatus.in_progress)])
        assert [i.content for i in state.items] == ['A', 'B']
        assert 'Plan updated: 2 step(s).' in result
        assert '2. [~] B' in result

    async def test_write_plan_warns_on_multiple_in_progress(self) -> None:
        toolset = PlanningToolset(PlanState())
        result = await toolset.write_plan(
            [
                PlanItem(content='A', status=TaskStatus.in_progress),
                PlanItem(content='B', status=TaskStatus.in_progress),
            ]
        )
        # Exact tail so a reworded/wrapped note can't slip through.
        assert result.endswith('\n\nNote: keep only one step in_progress at a time.')

    async def test_write_plan_single_in_progress_no_warning(self) -> None:
        toolset = PlanningToolset(PlanState())
        result = await toolset.write_plan([PlanItem(content='A', status=TaskStatus.in_progress)])
        # No note is appended -- the reply ends exactly with the rendered plan.
        assert result == 'Plan updated: 1 step(s).\n\n1. [~] A\n(0/1 completed)'


class TestPlanningCapability:
    def test_serialization_name(self) -> None:
        assert Planning.get_serialization_name() == 'Planning'

    def test_default_instructions(self) -> None:
        assert Planning[None]().get_instructions() == Planning[None]().get_instructions()
        instructions = Planning[None]().get_instructions()
        assert isinstance(instructions, str)
        assert 'write_plan' in instructions

    def test_custom_guidance(self) -> None:
        assert Planning[None](guidance='Custom guidance.').get_instructions() == 'Custom guidance.'

    def test_empty_guidance_omitted(self) -> None:
        assert Planning[None](guidance='').get_instructions() is None

    def test_get_toolset_type(self) -> None:
        assert isinstance(Planning[None]().get_toolset(), PlanningToolset)

    async def test_for_run_isolates_state_and_preserves_config(self) -> None:
        cap = Planning[None](guidance='G', cache_ttl='1h')
        cap._state.items.append(PlanItem(content='leftover'))

        run = await cap.for_run(MagicMock())

        assert run is not cap
        assert run._state.items == []
        assert run.guidance == 'G'
        assert run.cache_ttl == '1h'
        assert len(cap._state.items) == 1  # original untouched

    async def test_two_runs_are_independent(self) -> None:
        cap = Planning[None]()
        run1 = await cap.for_run(MagicMock())
        run2 = await cap.for_run(MagicMock())
        run1._state.items.append(PlanItem(content='only run1'))
        assert run2._state.items == []


class TestEphemeralReminder:
    async def _run_hook(
        self, cap: Planning[None], messages: list[ModelMessage]
    ) -> tuple[list[ModelMessage], ModelResponse]:
        """Invoke `wrap_model_request` with a recording handler.

        Returns the messages the handler was actually given (i.e. what would be
        sent to the model) and the handler's response.
        """
        captured: dict[str, list[ModelMessage]] = {}

        async def handler(rc: ModelRequestContext) -> ModelResponse:
            captured['messages'] = list(rc.messages)
            return ModelResponse(parts=[TextPart('ok')])

        ctx = _make_request_context(messages)
        response = await cap.wrap_model_request(MagicMock(), request_context=ctx, handler=handler)
        return captured['messages'], response

    async def test_no_reminder_when_plan_empty(self) -> None:
        cap = Planning[None]()
        original = ModelRequest(parts=[UserPromptPart('hello')])
        seen, response = await self._run_hook(cap, [original])
        assert seen[-1] is original
        assert len(original.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == 'ok'  # handler was still called

    async def test_reminder_appended_behind_cachepoint(self) -> None:
        cap = Planning[None]()
        cap._state.items = [PlanItem(content='Do X', status=TaskStatus.in_progress)]
        original = ModelRequest(parts=[UserPromptPart('hello')])

        seen, _ = await self._run_hook(cap, [original])

        # Append-only: the original object is never mutated in place.
        assert len(original.parts) == 1
        last = seen[-1]
        assert isinstance(last, ModelRequest)
        assert last is not original
        assert len(last.parts) == 2
        reminder = last.parts[-1]
        assert isinstance(reminder, UserPromptPart)
        content = reminder.content
        assert isinstance(content, list)
        # The cache breakpoint precedes the reminder text, so the reminder
        # falls outside the cached prefix.
        assert isinstance(content[0], CachePoint)
        assert content[0].ttl == '5m'
        assert isinstance(content[1], str)
        assert '<plan-reminder>' in content[1]
        assert 'Do X' in content[1]

    async def test_cache_ttl_is_forwarded(self) -> None:
        cap = Planning[None](cache_ttl='1h')
        cap._state.items = [PlanItem(content='Do X')]
        seen, _ = await self._run_hook(cap, [ModelRequest(parts=[UserPromptPart('hi')])])
        reminder = seen[-1].parts[-1]
        assert isinstance(reminder, UserPromptPart)
        assert isinstance(reminder.content, list)
        cache_point = reminder.content[0]
        assert isinstance(cache_point, CachePoint)
        assert cache_point.ttl == '1h'

    async def test_no_injection_when_last_is_not_model_request(self) -> None:
        # Defensive: core guarantees a trailing ModelRequest, but if the last
        # message isn't one we leave it untouched and still call the handler.
        cap = Planning[None]()
        cap._state.items = [PlanItem(content='Do X')]
        prior = ModelResponse(parts=[TextPart('prior')])
        seen, response = await self._run_hook(cap, [prior])
        assert seen[-1] is prior
        assert len(prior.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == 'ok'


class TestEndToEnd:
    async def test_write_plan_runs_and_plan_is_visible(self) -> None:
        agent = Agent(TestModel(), capabilities=[Planning()])
        # TestModel calls every tool once, including write_plan.
        result = await agent.run('plan and do the work')
        assert result.output is not None

    async def test_reminder_reaches_model_then_is_ephemeral(self) -> None:
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

        agent: Agent[None, str] = Agent(FunctionModel(model_fn), capabilities=[Planning()])
        result = await agent.run('go')
        assert result.output == 'done'

        # The plan reminder reached the model on the second request...
        sent = _all_text(captured['messages'])
        assert '<plan-reminder>' in sent
        assert 'Step A' in sent
        # ...with a CachePoint marking the boundary before it.
        has_cache_point = any(
            isinstance(part, UserPromptPart)
            and not isinstance(part.content, str)
            and any(isinstance(c, CachePoint) for c in part.content)
            for msg in captured['messages']
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )
        assert has_cache_point
        # ...but neither the reminder nor its CachePoint is ever written to the
        # durable message history (the CachePoint only ever rides the reminder).
        assert '<plan-reminder>' not in _all_text(result.all_messages())
