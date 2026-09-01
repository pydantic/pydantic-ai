"""Tests for the Planning capability, toolset, renderers, types, and events."""

from __future__ import annotations

from typing import cast
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
from pydantic_ai.tools import RunContext

from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanEvent,
    PlanEventEmitter,
    PlanEventType,
    PlanItem,
    Planning,
    PlanningToolset,
    PlanStatusUpdate,
    SqlitePlanStore,
    TaskStatus,
    render_plan,
)
from pydantic_ai_harness.planning._toolset import (
    CORE_TOOL_NAMES,
    SUBTASK_TOOL_NAMES,
    has_cycle,
    is_blocked,
    render_flat,
    render_summary,
    render_tree,
    status_counts_line,
    status_icon,
    validate_hierarchy,
)
from tests._recording_durability import RecordingDurability  # pyright: ignore[reportMissingTypeStubs]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _ctx() -> RunContext[None]:
    return cast(RunContext[None], MagicMock())


def _toolset(*, subtasks: bool = False, store: InMemoryPlanStore | None = None) -> PlanningToolset[None]:
    cap = Planning[None](store=store or InMemoryPlanStore(), enable_subtasks=subtasks)
    return PlanningToolset[None](cap)


async def test_plan_read_dispatches_as_durable_operation() -> None:
    store = InMemoryPlanStore()
    await store.set_items([PlanItem(id='1', content='journal the plan')])
    durability = RecordingDurability()
    planning = Planning(store=store)
    assert planning.id == 'planning'
    agent = Agent(
        TestModel(),
        name='planning',
        capabilities=[planning, durability],
    )

    await agent.run('continue')

    bound = RecordingDurability.from_agent(agent)
    assert bound is not None
    assert 'planning__capability__planning.read_plan' in {name for name, _ in bound.calls}


# --- Types ------------------------------------------------------------------


class TestTypes:
    def test_task_status_values(self) -> None:
        assert [s.value for s in TaskStatus] == ['pending', 'in_progress', 'completed', 'cancelled', 'blocked']

    def test_plan_item_defaults(self) -> None:
        item = PlanItem(content='do')
        assert item.status is TaskStatus.pending
        assert item.active_form == ''
        assert item.parent_id is None
        assert item.depends_on == []
        assert len(item.id) == 8

    def test_status_update_fields(self) -> None:
        update = PlanStatusUpdate(task_id='abc', status=TaskStatus.completed)
        assert update.task_id == 'abc'
        assert update.status is TaskStatus.completed


# --- Events -----------------------------------------------------------------


class TestEventEmitter:
    async def test_sync_and_async_callbacks(self) -> None:
        seen: list[str] = []
        emitter = PlanEventEmitter()

        @emitter.on_created
        def sync_cb(event: PlanEvent) -> None:
            seen.append(f'sync:{event.item.content}')

        @emitter.on_created
        async def async_cb(event: PlanEvent) -> None:
            seen.append(f'async:{event.item.content}')

        await emitter.emit(PlanEvent(event_type=PlanEventType.created, item=PlanItem(content='x')))
        assert seen == ['sync:x', 'async:x']

    def test_all_decorators_register(self) -> None:
        emitter = PlanEventEmitter()
        cb: list[object] = []
        for register in (
            emitter.on_created,
            emitter.on_updated,
            emitter.on_status_changed,
            emitter.on_completed,
            emitter.on_deleted,
        ):
            register(cb.append)
        assert all(len(v) == 1 for v in emitter._listeners.values())

    def test_off(self) -> None:
        emitter = PlanEventEmitter()

        def cb(event: PlanEvent) -> None:  # pragma: no cover - registered then removed, never fired
            ...

        emitter.on(PlanEventType.created, cb)
        assert emitter.off(PlanEventType.created, cb) is True
        assert emitter.off(PlanEventType.created, cb) is False


# --- Renderers --------------------------------------------------------------


class TestRenderers:
    def test_render_plan_empty(self) -> None:
        assert render_plan([]) == 'No plan yet.'

    def test_render_plan_progress(self) -> None:
        result = render_plan(
            [
                PlanItem(content='First', status=TaskStatus.completed),
                PlanItem(content='Second', status=TaskStatus.in_progress),
                PlanItem(content='Third'),
                PlanItem(content='Fourth', status=TaskStatus.cancelled),
            ]
        )
        assert result == '1. [x] First\n2. [~] Second\n3. [ ] Third\n4. [-] Fourth\n(1/4 completed)'

    def test_render_flat_annotations(self) -> None:
        result = render_flat(
            [PlanItem(id='p', content='Parent'), PlanItem(id='c', content='Child', parent_id='p', depends_on=['p'])],
            subtasks=True,
        )
        assert '[p] Parent' in result
        assert '(subtask of: p)' in result
        assert '(depends on: p)' in result

    def test_render_tree_nests(self) -> None:
        result = render_tree(
            [
                PlanItem(id='p', content='Parent'),
                PlanItem(id='c', content='Child', parent_id='p', depends_on=['x']),
            ]
        )
        assert 'Current plan (hierarchical view):' in result
        assert '  2. [ ] [c] Child' in result
        assert '     depends on: x' in result

    def test_render_tree_survives_duplicate_ids(self) -> None:
        # Two items sharing an id would recurse forever without the visited guard.
        result = render_tree(
            [
                PlanItem(id='p', content='First'),
                PlanItem(id='p', content='Second', parent_id='p'),
            ]
        )
        assert '1. [ ] [p] First' in result
        assert 'Second' not in result

    def test_status_counts_line_blocked_and_cancelled(self) -> None:
        items = [
            PlanItem(content='a', status=TaskStatus.blocked),
            PlanItem(content='b', status=TaskStatus.cancelled),
        ]
        line = status_counts_line(items)
        assert '1 blocked' in line
        assert '1 cancelled' in line

    def test_render_summary_all_done_note(self) -> None:
        done = render_summary([PlanItem(content='a', status=TaskStatus.completed)])
        assert 'All steps are completed' in done
        pending = render_summary([PlanItem(content='a')])
        assert 'All steps are completed' not in pending
        cancelled_only = render_summary([PlanItem(content='a', status=TaskStatus.cancelled)])
        assert 'All steps are completed' not in cancelled_only

    def test_render_summary_blocked_suppresses_all_done_note(self) -> None:
        items = [
            PlanItem(content='a', status=TaskStatus.completed),
            PlanItem(content='b', status=TaskStatus.blocked),
        ]
        assert 'All steps are completed' not in render_summary(items)

    def test_status_icon_all(self) -> None:
        assert status_icon(TaskStatus.pending) == '[ ]'
        assert status_icon(TaskStatus.in_progress) == '[~]'
        assert status_icon(TaskStatus.completed) == '[x]'
        assert status_icon(TaskStatus.cancelled) == '[-]'
        assert status_icon(TaskStatus.blocked) == '[!]'


class TestGraphHelpers:
    def test_has_cycle_true(self) -> None:
        items = [PlanItem(id='b', content='B', depends_on=['a']), PlanItem(id='a', content='A')]
        assert has_cycle(items, 'a', 'b') is True

    def test_has_cycle_missing_node(self) -> None:
        items = [PlanItem(id='b', content='B', depends_on=['ghost'])]
        assert has_cycle(items, 'a', 'b') is False

    def test_has_cycle_revisits_shared_node(self) -> None:
        items = [
            PlanItem(id='b', content='B', depends_on=['c', 'd']),
            PlanItem(id='c', content='C', depends_on=['e']),
            PlanItem(id='d', content='D', depends_on=['e']),
            PlanItem(id='e', content='E'),
        ]
        # Diamond: 'e' is reached twice, exercising the visited short-circuit.
        assert has_cycle(items, 'a', 'b') is False

    def test_is_blocked_variants(self) -> None:
        done = PlanItem(id='d', content='D', status=TaskStatus.completed)
        pending = PlanItem(id='p', content='P')
        blocked_by_pending = PlanItem(content='T', depends_on=['p'])
        assert is_blocked([pending, blocked_by_pending], blocked_by_pending) is True
        # dependency completed -> not blocked (loop continues past it)
        after_done = PlanItem(content='T', depends_on=['d'])
        assert is_blocked([done, after_done], after_done) is False
        # dependency missing -> not blocked
        missing_dep = PlanItem(content='T', depends_on=['ghost'])
        assert is_blocked([missing_dep], missing_dep) is False

    def test_validate_hierarchy(self) -> None:
        assert validate_hierarchy([PlanItem(id='a', content='A'), PlanItem(id='b', content='B', parent_id='a')]) is None
        # Valid multi-dependency plan (no cycle) -- the DFS steps past a clean dependency.
        assert (
            validate_hierarchy(
                [
                    PlanItem(id='a', content='A', depends_on=['b', 'c']),
                    PlanItem(id='b', content='B'),
                    PlanItem(id='c', content='C'),
                ]
            )
            is None
        )
        dup = validate_hierarchy([PlanItem(id='x', content='A'), PlanItem(id='x', content='B')])
        assert dup is not None and 'Duplicate step ids' in dup
        dangling = validate_hierarchy([PlanItem(id='a', content='A', parent_id='ghost')])
        assert dangling is not None and 'not in the plan' in dangling
        cycle = validate_hierarchy(
            [PlanItem(id='a', content='A', parent_id='b'), PlanItem(id='b', content='B', parent_id='a')]
        )
        assert cycle is not None and 'parent cycle' in cycle
        bad_dep = validate_hierarchy([PlanItem(id='a', content='A', depends_on=['ghost'])])
        assert bad_dep is not None and 'depends on' in bad_dep
        dep_cycle = validate_hierarchy(
            [PlanItem(id='a', content='A', depends_on=['b']), PlanItem(id='b', content='B', depends_on=['a'])]
        )
        assert dep_cycle is not None and 'dependency cycle' in dep_cycle


# --- Toolset: base tools ----------------------------------------------------


class TestWritePlan:
    async def test_replaces_and_reports(self) -> None:
        ts = _toolset()
        result = await ts.write_plan(_ctx(), [PlanItem(content='A'), PlanItem(content='B')])
        assert result.startswith('Plan updated: 2 step(s).')
        assert '2. [ ] B' in result

    async def test_multi_in_progress_note(self) -> None:
        ts = _toolset()
        result = await ts.write_plan(
            _ctx(),
            [
                PlanItem(content='A', status=TaskStatus.in_progress),
                PlanItem(content='B', status=TaskStatus.in_progress),
            ],
        )
        assert result.endswith('\n\nNote: keep only one step in_progress at a time.')

    async def test_subtasks_off_rejects_hierarchy(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        result = await ts.write_plan(
            _ctx(), [PlanItem(id='y', content='B'), PlanItem(content='A', parent_id='y', depends_on=['y'])]
        )
        assert result == (
            "Plan not updated: step 'A' sets parent_id/depends_on, which are only valid with subtasks enabled."
        )
        assert await store.get_items() == []

    async def test_subtasks_off_rejects_depends_on_alone(self) -> None:
        ts = _toolset()
        result = await ts.write_plan(_ctx(), [PlanItem(id='y', content='B'), PlanItem(content='A', depends_on=['y'])])
        assert 'only valid with subtasks enabled' in result

    async def test_subtasks_on_keeps_hierarchy(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        await ts.write_plan(_ctx(), [PlanItem(id='x', content='P'), PlanItem(content='C', parent_id='x')])
        stored = (await store.get_items())[1]
        assert stored.parent_id == 'x'

    async def test_reconciles_blocked_status(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        # A pending step with an incomplete prerequisite is reconciled to `blocked`;
        # a `blocked` step whose prerequisite is already done is reconciled to `pending`;
        # a pending step whose prerequisite is already done is left pending.
        await ts.write_plan(
            _ctx(),
            [
                PlanItem(id='a', content='A'),
                PlanItem(id='b', content='B', depends_on=['a']),
                PlanItem(id='c', content='C', status=TaskStatus.completed),
                PlanItem(id='d', content='D', status=TaskStatus.blocked, depends_on=['c']),
                PlanItem(id='e', content='E', depends_on=['c']),
            ],
        )
        assert (await store.get_item('b')).status is TaskStatus.blocked  # type: ignore[union-attr]
        assert (await store.get_item('d')).status is TaskStatus.pending  # type: ignore[union-attr]
        assert (await store.get_item('e')).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_auto_blocking_stays_event_silent(self) -> None:
        # write_plan is a bulk replacement: it reconciles dependency blocks before
        # storing, so it must not emit per-step events even when a step is blocked.
        events: list[PlanEvent] = []
        emitter = PlanEventEmitter()
        for kind in PlanEventType:
            emitter.on(kind, events.append)
        store = InMemoryPlanStore(event_emitter=emitter)
        ts = _toolset(subtasks=True, store=store)
        await ts.write_plan(
            _ctx(),
            [PlanItem(id='a', content='A'), PlanItem(id='b', content='B', depends_on=['a'])],
        )
        assert (await store.get_item('b')).status is TaskStatus.blocked  # type: ignore[union-attr]
        assert events == []

    async def test_subtasks_rejects_bad_hierarchy(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        result = await ts.write_plan(_ctx(), [PlanItem(id='x', content='A'), PlanItem(id='x', content='B')])
        assert result.startswith('Plan not updated:') and 'Duplicate step ids' in result
        assert await store.get_items() == []

    async def test_subtasks_rejects_broken_hierarchy(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        result = await ts.write_plan(_ctx(), [PlanItem(id='child', content='Child', parent_id='missing')])
        assert result == "Plan not updated: Step 'child' has parent_id 'missing', which is not in the plan."
        assert await store.get_items() == []

    async def test_rejects_duplicate_ids_without_subtasks_before_replacing_plan(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        await ts.write_plan(_ctx(), [PlanItem(id='kept', content='Kept')])
        result = await ts.write_plan(_ctx(), [PlanItem(id='same', content='A'), PlanItem(id='same', content='B')])
        assert result == 'Plan not updated: Duplicate step ids: same. Every step needs a unique id.'
        assert [item.id for item in await store.get_items()] == ['kept']

    async def test_subtasks_plan_is_independent_of_input_items(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        parent = PlanItem(id='parent', content='Parent')
        child = PlanItem(id='child', content='Child', depends_on=['parent'])
        await ts.write_plan(_ctx(), [parent, child])
        parent.content = 'Changed parent'
        child.content = 'Changed child'
        child.depends_on.clear()
        child.status = TaskStatus.completed
        stored = await store.get_items()
        assert [(item.content, item.status, item.depends_on) for item in stored] == [
            ('Parent', TaskStatus.pending, []),
            ('Child', TaskStatus.blocked, ['parent']),
        ]

    async def test_rejects_blocked_status_without_subtasks(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        result = await ts.write_plan(_ctx(), [PlanItem(content='A', status=TaskStatus.blocked)])
        assert 'only valid with subtasks' in result
        assert await store.get_items() == []


class TestReadPlan:
    async def test_empty(self) -> None:
        assert 'No plan yet' in await _toolset().read_plan(_ctx())

    async def test_populated(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        await store.add_item(PlanItem(content='A'))
        result = await ts.read_plan(_ctx())
        assert 'Current plan:' in result
        assert 'Summary:' in result

    async def test_tree_empty_and_modes(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        assert 'No plan yet' in await ts.read_plan_tree(_ctx())
        await store.set_items([PlanItem(id='p', content='P'), PlanItem(content='C', parent_id='p')])
        assert 'Current plan:' in await ts.read_plan_tree(_ctx(), hierarchical=False)
        assert 'hierarchical view' in await ts.read_plan_tree(_ctx(), hierarchical=True)


class TestAddTask:
    async def test_adds(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        result = await ts.add_task(_ctx(), 'Do it', active_form='Doing it')
        assert result.startswith("Added step 'Do it' with id:")
        assert len(await store.get_items()) == 1


class TestUpdateTaskStatus:
    async def test_success(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        item = await store.add_item(PlanItem(content='A'))
        result = await ts.update_task_status(_ctx(), item.id, TaskStatus.completed)
        assert "status to 'completed'" in result

    async def test_not_found(self) -> None:
        assert 'not found' in await _toolset().update_task_status(_ctx(), 'nope', TaskStatus.completed)

    async def test_blocked_rejected_without_subtasks(self) -> None:
        result = await _toolset().update_task_status(_ctx(), 'x', TaskStatus.blocked)
        assert 'subtasks are not enabled' in result

    async def test_blocked_allowed_with_subtasks(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        item = await store.add_item(PlanItem(content='A'))
        assert "status to 'blocked'" in await ts.update_task_status(_ctx(), item.id, TaskStatus.blocked)
        # A manual block on a dependency-free step must persist, not auto-revert.
        assert (await store.get_item(item.id)).status is TaskStatus.blocked  # type: ignore[union-attr]

    async def test_an_executor_without_subtasks_still_unblocks_over_a_shared_store(self) -> None:
        """The README's handoff shares one store; the flag says which tools exist, not what the data means."""
        store = InMemoryPlanStore()
        planner = _toolset(subtasks=True, store=store)
        executor = _toolset(subtasks=False, store=store)
        dep = await store.add_item(PlanItem(content='dep'))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))
        await planner.update_task_status(_ctx(), task.id, TaskStatus.pending)
        assert (await store.get_item(task.id)).status is TaskStatus.blocked  # type: ignore[union-attr]

        await executor.update_task_status(_ctx(), dep.id, TaskStatus.completed)

        assert (await store.get_item(task.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_a_blocked_step_is_counted_whoever_renders_the_plan(self) -> None:
        """Omitting it from the tally hid a step the reader had no other way to see."""
        store = InMemoryPlanStore()
        planner = _toolset(subtasks=True, store=store)
        executor = _toolset(subtasks=False, store=store)
        dep = await store.add_item(PlanItem(content='dep'))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))
        await planner.update_task_status(_ctx(), task.id, TaskStatus.pending)

        assert '1 blocked' in await executor.read_plan(_ctx())

    async def test_a_plan_cannot_store_a_step_blocked_on_nothing(self) -> None:
        """Reconciliation only touches steps with dependencies, so nothing would ever free it."""
        ts = _toolset(subtasks=True)

        result = await ts.write_plan(_ctx(), [PlanItem(id='x', content='X', status=TaskStatus.blocked)])

        assert 'blocked but depends on nothing' in result

    async def test_a_dependency_added_to_a_blocked_step_reconciles_it(self) -> None:
        """The prerequisite is already terminal, so leaving it blocked strands it forever."""
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        dep = await store.add_item(PlanItem(content='dep', status=TaskStatus.completed))
        task = await store.add_item(PlanItem(content='task', status=TaskStatus.blocked))

        await ts.set_dependency(_ctx(), task.id, dep.id)

        assert (await store.get_item(task.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_blocking_a_step_whose_dependencies_are_resolved_is_refused(self) -> None:
        """Reconciliation would undo it in the same call while the reply said it worked."""
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        dep = await store.add_item(PlanItem(content='dep', status=TaskStatus.completed))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))

        result = await ts.update_task_status(_ctx(), task.id, TaskStatus.blocked)

        assert 'every step it depends on is already resolved' in result
        assert (await store.get_item(task.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_the_reply_reports_the_status_the_store_holds(self) -> None:
        """Asking for `pending` on a step with an unmet dependency stores `blocked`."""
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        dep = await store.add_item(PlanItem(content='dep'))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))

        result = await ts.update_task_status(_ctx(), task.id, TaskStatus.pending)

        assert "status to 'blocked'" in result
        assert (await store.get_item(task.id)).status is TaskStatus.blocked  # type: ignore[union-attr]

    async def test_a_batch_reports_the_statuses_the_store_holds(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        dep = await store.add_item(PlanItem(content='dep'))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))

        result = await ts.update_task_statuses(_ctx(), [PlanStatusUpdate(task_id=task.id, status=TaskStatus.pending)])

        assert f'- [{task.id}] task -> blocked' in result

    async def test_a_batch_refuses_an_inert_block_without_applying_anything(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        dep = await store.add_item(PlanItem(content='dep', status=TaskStatus.completed))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))
        other = await store.add_item(PlanItem(content='other'))

        result = await ts.update_task_statuses(
            _ctx(),
            [
                PlanStatusUpdate(task_id=other.id, status=TaskStatus.completed),
                PlanStatusUpdate(task_id=task.id, status=TaskStatus.blocked),
            ],
        )

        assert 'No changes applied' in result
        assert (await store.get_item(other.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_cannot_start_blocked_by_dependency(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        dep = await store.add_item(PlanItem(content='dep'))
        task = await store.add_item(PlanItem(content='task', depends_on=[dep.id]))
        result = await ts.update_task_status(_ctx(), task.id, TaskStatus.in_progress)
        assert 'incomplete dependencies' in result


class TestUpdateTaskStatuses:
    async def test_empty(self) -> None:
        assert await _toolset().update_task_statuses(_ctx(), []) == 'No updates provided.'

    async def test_success(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B'))
        result = await ts.update_task_statuses(
            _ctx(),
            [
                PlanStatusUpdate(task_id=a.id, status=TaskStatus.completed),
                PlanStatusUpdate(task_id=b.id, status=TaskStatus.in_progress),
            ],
        )
        assert 'Updated 2 step(s):' in result

    async def test_all_or_nothing_errors(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        dep = await store.add_item(PlanItem(content='dep'))
        blocked = await store.add_item(PlanItem(content='blocked', depends_on=[dep.id]))
        result = await ts.update_task_statuses(
            _ctx(),
            [
                PlanStatusUpdate(task_id=a.id, status=TaskStatus.completed),
                PlanStatusUpdate(task_id='missing', status=TaskStatus.completed),
                PlanStatusUpdate(task_id=blocked.id, status=TaskStatus.in_progress),
            ],
        )
        assert 'No changes applied' in result
        assert 'not found' in result
        assert 'incomplete dependencies' in result
        # nothing applied
        assert (await store.get_item(a.id)) is not None and (await store.get_item(a.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_invalid_blocked_without_subtasks(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        a = await store.add_item(PlanItem(content='A'))
        result = await ts.update_task_statuses(_ctx(), [PlanStatusUpdate(task_id=a.id, status=TaskStatus.blocked)])
        assert 'subtasks are not enabled' in result

    async def test_atomic_handoff_completes_prereq_and_starts_dependent(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B', depends_on=[a.id], status=TaskStatus.blocked))
        result = await ts.update_task_statuses(
            _ctx(),
            [
                PlanStatusUpdate(task_id=a.id, status=TaskStatus.completed),
                PlanStatusUpdate(task_id=b.id, status=TaskStatus.in_progress),
            ],
        )
        assert 'Updated 2 step(s):' in result
        assert (await store.get_item(b.id)).status is TaskStatus.in_progress  # type: ignore[union-attr]


class TestRemoveTask:
    async def test_success_and_missing(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(store=store)
        item = await store.add_item(PlanItem(content='A'))
        assert 'Removed step' in await ts.remove_task(_ctx(), item.id)
        assert 'not found' in await ts.remove_task(_ctx(), item.id)

    async def test_cascades_subtasks_and_cleans_dependencies(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        await store.add_item(PlanItem(id='p', content='Parent'))
        await store.add_item(PlanItem(id='c', content='Child', parent_id='p'))
        await store.add_item(PlanItem(id='d', content='Dependent', depends_on=['p'], status=TaskStatus.blocked))
        # An unrelated, manually-blocked step must be left untouched.
        await store.add_item(PlanItem(id='e', content='Unrelated', status=TaskStatus.blocked))
        result = await ts.remove_task(_ctx(), 'p')
        assert '1 subtask(s)' in result
        # Parent and its child are gone; the dangling dependency is dropped and the
        # dependent is unblocked since it no longer waits on anything.
        assert [i.id for i in await store.get_items()] == ['d', 'e']
        assert (await store.get_item('d')).depends_on == []  # type: ignore[union-attr]
        assert (await store.get_item('d')).status is TaskStatus.pending  # type: ignore[union-attr]
        assert (await store.get_item('e')).status is TaskStatus.blocked  # type: ignore[union-attr]


# --- Toolset: subtask tools -------------------------------------------------


class TestSubtaskTools:
    async def test_add_subtask(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        parent = await store.add_item(PlanItem(content='P'))
        result = await ts.add_subtask(_ctx(), parent.id, 'child', active_form='doing child')
        assert 'Added subtask' in result
        assert (await store.get_items())[1].parent_id == parent.id

    async def test_add_subtask_missing_parent(self) -> None:
        ts = _toolset(subtasks=True)
        assert 'not found' in await ts.add_subtask(_ctx(), 'nope', 'child')

    async def test_set_dependency_blocks(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B'))
        result = await ts.set_dependency(_ctx(), b.id, a.id)
        assert 'automatically blocked' in result
        assert (await store.get_item(b.id)).status is TaskStatus.blocked  # type: ignore[union-attr]

    async def test_set_dependency_leaves_cancelled_step_terminal(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B', status=TaskStatus.cancelled))
        result = await ts.set_dependency(_ctx(), b.id, a.id)
        assert 'automatically blocked' not in result
        assert (await store.get_item(b.id)).status is TaskStatus.cancelled  # type: ignore[union-attr]
        assert (await store.get_item(b.id)).depends_on == [a.id]  # type: ignore[union-attr]

    async def test_set_dependency_no_block_when_prereq_done(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A', status=TaskStatus.completed))
        b = await store.add_item(PlanItem(content='B'))
        result = await ts.set_dependency(_ctx(), b.id, a.id)
        assert 'automatically blocked' not in result
        assert (await store.get_item(b.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_set_dependency_no_block_when_prereq_cancelled(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A', status=TaskStatus.cancelled))
        b = await store.add_item(PlanItem(content='B'))
        result = await ts.set_dependency(_ctx(), b.id, a.id)
        assert 'automatically blocked' not in result
        assert (await store.get_item(b.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_set_dependency_validation(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B'))
        assert 'not found' in await ts.set_dependency(_ctx(), 'nope', a.id)
        assert 'Dependency step' in await ts.set_dependency(_ctx(), a.id, 'nope')
        assert 'cannot depend on itself' in await ts.set_dependency(_ctx(), a.id, a.id)
        await ts.set_dependency(_ctx(), b.id, a.id)
        assert 'already exists' in await ts.set_dependency(_ctx(), b.id, a.id)
        # cycle: a already depends on b (via b->a? no). Make a depend on b, then b->a would cycle
        assert 'cycle' in await ts.set_dependency(_ctx(), a.id, b.id)

    async def test_completing_prerequisite_unblocks_dependent(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B'))
        await ts.set_dependency(_ctx(), b.id, a.id)
        assert (await store.get_item(b.id)).status is TaskStatus.blocked  # type: ignore[union-attr]
        # `b` must not show up as available while `a` is unfinished.
        assert 'B' not in await ts.get_available_tasks(_ctx())
        # Completing the prerequisite unblocks `b` back to pending.
        await ts.update_task_status(_ctx(), a.id, TaskStatus.completed)
        assert (await store.get_item(b.id)).status is TaskStatus.pending  # type: ignore[union-attr]
        assert 'B' in await ts.get_available_tasks(_ctx())

    async def test_cancelling_prerequisite_unblocks_dependent(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B'))
        await ts.set_dependency(_ctx(), b.id, a.id)
        assert (await store.get_item(b.id)).status is TaskStatus.blocked  # type: ignore[union-attr]
        # A cancelled prerequisite will never complete, so it must free its dependent.
        await ts.update_task_status(_ctx(), a.id, TaskStatus.cancelled)
        assert (await store.get_item(b.id)).status is TaskStatus.pending  # type: ignore[union-attr]
        assert 'B' in await ts.get_available_tasks(_ctx())
        assert "status to 'in_progress'" in await ts.update_task_status(_ctx(), b.id, TaskStatus.in_progress)

    async def test_batch_completion_unblocks_dependent(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A'))
        b = await store.add_item(PlanItem(content='B'))
        await ts.set_dependency(_ctx(), b.id, a.id)
        await ts.update_task_statuses(_ctx(), [PlanStatusUpdate(task_id=a.id, status=TaskStatus.completed)])
        assert (await store.get_item(b.id)).status is TaskStatus.pending  # type: ignore[union-attr]

    async def test_regressing_prerequisite_reblocks_dependent(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        a = await store.add_item(PlanItem(content='A', status=TaskStatus.completed))
        b = await store.add_item(PlanItem(content='B', depends_on=[a.id]))
        await ts.update_task_status(_ctx(), a.id, TaskStatus.in_progress)
        assert (await store.get_item(b.id)).status is TaskStatus.blocked  # type: ignore[union-attr]

    async def test_get_available_tasks(self) -> None:
        store = InMemoryPlanStore()
        ts = _toolset(subtasks=True, store=store)
        assert 'No available steps' in await ts.get_available_tasks(_ctx())
        await store.add_item(PlanItem(content='ready'))
        await store.add_item(PlanItem(content='done', status=TaskStatus.completed))
        result = await ts.get_available_tasks(_ctx())
        assert 'ready' in result
        assert 'done' not in result


# --- Capability -------------------------------------------------------------


class TestStoreContract:
    """The shipped stores answer the same way, so a plan behaves the same on any backend."""

    async def test_a_duplicate_id_is_refused(self) -> None:
        """SQLite and Postgres have a primary key; the others used to accept a shadowing id."""
        store = InMemoryPlanStore()
        await store.add_item(PlanItem(id='a', content='A'))

        with pytest.raises(ValueError, match="id 'a' is already"):
            await store.add_item(PlanItem(id='a', content='A again'))

        assert len(await store.get_items()) == 1


class TestCapability:
    def test_serialization_name(self) -> None:
        assert Planning.get_serialization_name() == 'Planning'

    def test_get_instructions(self) -> None:
        assert 'write_plan' in cast(str, Planning[None]().get_instructions())
        assert Planning[None](guidance='Custom.').get_instructions() == 'Custom.'
        assert Planning[None](guidance='').get_instructions() is None

    def test_get_instructions_subtasks(self) -> None:
        base = cast(str, Planning[None]().get_instructions())
        assert 'add_subtask' not in base
        subtasks = cast(str, Planning[None](enable_subtasks=True).get_instructions())
        assert 'add_subtask' in subtasks and 'set_dependency' in subtasks and 'get_available_tasks' in subtasks
        assert Planning[None](guidance='Custom.', enable_subtasks=True).get_instructions() == 'Custom.'

    def test_get_toolset_type(self) -> None:
        assert isinstance(Planning[None]().get_toolset(), PlanningToolset)

    def test_resolve_store_explicit_and_resolver(self) -> None:
        store = InMemoryPlanStore()
        assert Planning[None](store=store).resolve_store(_ctx()) is store
        assert Planning[None](store_resolver=lambda ctx: store).resolve_store(_ctx()) is store

    async def test_direct_toolset_keeps_default_store(self) -> None:
        toolset = PlanningToolset[None](Planning[None]())
        await toolset.write_plan(_ctx(), [PlanItem(content='Kept')])
        assert 'Kept' in await toolset.read_plan(_ctx())

    async def test_for_run_isolates_default_store(self) -> None:
        cap = Planning[None](guidance='G', cache_ttl='1h', enable_subtasks=True)
        run1 = await cap.for_run(_ctx())
        run2 = await cap.for_run(_ctx())
        assert (run1.guidance, run1.cache_ttl, run1.enable_subtasks) == ('G', '1h', True)
        await run1.resolve_store(_ctx()).add_item(PlanItem(content='only-run1'))
        assert await run2.resolve_store(_ctx()).get_items() == []

    async def test_for_run_caches_store(self) -> None:
        run = await Planning[None]().for_run(_ctx())
        assert run.resolve_store(_ctx()) is run.resolve_store(_ctx())
        assert isinstance(run.resolve_store(_ctx()), InMemoryPlanStore)

    async def test_store_resolver_runs_once_per_run(self) -> None:
        stores: list[InMemoryPlanStore] = []

        def resolve_store(ctx: RunContext[None]) -> InMemoryPlanStore:
            store = InMemoryPlanStore()
            stores.append(store)
            return store

        cap = Planning[None](store_resolver=resolve_store)
        first = await cap.for_run(_ctx())
        second = await cap.for_run(_ctx())
        assert first.resolve_store(_ctx()) is first.resolve_store(_ctx())
        assert second.resolve_store(_ctx()) is second.resolve_store(_ctx())
        assert stores == [first.resolve_store(_ctx()), second.resolve_store(_ctx())]

    def test_from_spec(self, tmp_path: str) -> None:
        assert Planning.from_spec().store is None
        sqlite_cap = Planning.from_spec(backend='sqlite', database=str(tmp_path))
        assert isinstance(sqlite_cap.store, SqlitePlanStore)
        with pytest.raises(ValueError, match='database is only valid'):
            Planning.from_spec(backend='memory', database='custom.db')

    def test_from_spec_rejects_an_unknown_backend(self) -> None:
        """`backend` is a `Literal`, but a spec is deserialized text; nothing has checked it yet."""
        with pytest.raises(ValueError, match='Unknown planning backend'):
            Planning.from_spec(backend='postgres')  # type: ignore[arg-type]

    def test_from_spec_carries_the_tool_allowlist(self) -> None:
        assert Planning.from_spec(tools=['write_plan']).tools == ['write_plan']


class TestToolAllowlist:
    def _names(self, cap: Planning[None]) -> set[str]:
        return set(PlanningToolset[None](cap).tools)

    def test_default_registers_every_tool(self) -> None:
        assert self._names(Planning[None]()) == set(CORE_TOOL_NAMES)
        assert self._names(Planning[None](enable_subtasks=True)) == {*CORE_TOOL_NAMES, *SUBTASK_TOOL_NAMES}

    def test_allowlist_narrows_the_surface(self) -> None:
        assert self._names(Planning[None](tools=['write_plan'])) == {'write_plan'}
        assert self._names(Planning[None](tools=['read_plan'])) == {'read_plan'}
        assert self._names(Planning[None](tools=[])) == set()
        assert self._names(Planning[None](enable_subtasks=True, tools=['write_plan', 'set_dependency'])) == {
            'write_plan',
            'set_dependency',
        }

    def test_guidance_follows_the_allowlist(self) -> None:
        write_only = cast(str, Planning[None](tools=['write_plan']).get_instructions())
        assert 'write_plan' in write_only
        assert 'add_task' not in write_only
        granular_only = cast(str, Planning[None](tools=['read_plan']).get_instructions())
        assert 'write_plan' not in granular_only and 'read_plan' in granular_only
        assert Planning[None](tools=[]).get_instructions() is None

    def test_subtask_tools_need_subtasks_enabled(self) -> None:
        with pytest.raises(ValueError, match='Unknown planning tools: add_subtask'):
            PlanningToolset[None](Planning[None](tools=['write_plan', 'add_subtask']))

    def test_unknown_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match='Unknown planning tools: write_todos'):
            PlanningToolset[None](Planning[None](tools=['write_todos']))
        with pytest.raises(ValueError, match='Unknown planning descriptions: red_plan'):
            PlanningToolset[None](Planning[None](descriptions={'red_plan': 'typo'}))


class TestReminder:
    async def _run_hook(
        self, cap: Planning[None], messages: list[ModelMessage]
    ) -> tuple[list[ModelMessage], ModelResponse]:
        captured: dict[str, list[ModelMessage]] = {}

        async def handler(rc: ModelRequestContext) -> ModelResponse:
            captured['messages'] = list(rc.messages)
            return ModelResponse(parts=[TextPart('ok')])

        ctx = ModelRequestContext(
            model=TestModel(), messages=messages, model_settings=None, model_request_parameters=ModelRequestParameters()
        )
        response = await cap.wrap_model_request(_ctx(), request_context=ctx, handler=handler)
        return captured['messages'], response

    async def test_inject_disabled_passthrough(self) -> None:
        cap = Planning[None](inject=False, store=InMemoryPlanStore())
        await cap.store.add_item(PlanItem(content='X')) if cap.store else None
        seen, response = await self._run_hook(cap, [ModelRequest(parts=[UserPromptPart('hi')])])
        assert len(seen[-1].parts) == 1
        assert cast(TextPart, response.parts[0]).content == 'ok'

    async def test_empty_plan_passthrough(self) -> None:
        cap = Planning[None](store=InMemoryPlanStore())
        original = ModelRequest(parts=[UserPromptPart('hi')])
        seen, _ = await self._run_hook(cap, [original])
        assert seen[-1] is original

    async def test_reminder_behind_cachepoint(self) -> None:
        store = InMemoryPlanStore()
        cap = Planning[None](store=store, cache_ttl='1h')
        await store.add_item(PlanItem(content='Do X', status=TaskStatus.in_progress))
        original = ModelRequest(parts=[UserPromptPart('hi')])
        seen, _ = await self._run_hook(cap, [original])
        assert len(original.parts) == 1  # append-only
        reminder = cast(UserPromptPart, seen[-1].parts[-1])
        content = reminder.content
        assert isinstance(content, list)
        assert content[0] == '<plan-reminder>\n'
        assert isinstance(content[1], CachePoint)
        assert content[1].ttl == '1h'
        assert 'Do X' in cast(str, content[2])

    async def test_last_not_model_request_passthrough(self) -> None:
        store = InMemoryPlanStore()
        cap = Planning[None](store=store)
        await store.add_item(PlanItem(content='Do X'))
        prior = ModelResponse(parts=[TextPart('prior')])
        seen, _ = await self._run_hook(cap, [prior])
        assert seen[-1] is prior


# --- End to end -------------------------------------------------------------


class TestEndToEnd:
    async def test_runs_with_test_model(self) -> None:
        agent = Agent(TestModel(), capabilities=[Planning()])
        result = await agent.run('plan the work')
        assert result.output is not None

    async def test_reminder_reaches_model_but_is_ephemeral(self) -> None:
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
        sent = '\n'.join(
            c
            for msg in captured['messages']
            for part in msg.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for c in part.content
            if isinstance(c, str)
        )
        assert '<plan-reminder>' in sent
        assert 'Step A' in sent
        reminder = next(
            part
            for message in captured['messages']
            for part in message.parts
            if isinstance(part, UserPromptPart)
            and isinstance(part.content, list)
            and '<plan-reminder>\n' in part.content
        )
        assert reminder.content[0] == '<plan-reminder>\n'
        assert isinstance(reminder.content[1], CachePoint)
        assert 'Step A' in cast(str, reminder.content[2])
        # ephemeral: never written to durable history
        durable = '\n'.join(
            part.content
            for msg in result.all_messages()
            for part in msg.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        assert '<plan-reminder>' not in durable
