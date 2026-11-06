"""The tools of a team's lead and members: the shared board, spawning, and messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition, ToolPrepareFunc
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness.planning import BoardChange, ClaimablePlanStore, OwnedPlanItem, PlanItem, TaskStatus
from pydantic_ai_harness.planning._toolset import is_blocked
from pydantic_ai_harness.subagents import SELF_AGENT_NAME
from pydantic_ai_harness.subagents._events import bounded_text
from pydantic_ai_harness.subagents._team import LEAD, send_team_message, team_role
from pydantic_ai_harness.teams._events import BoardItemChangedEvent, TeammateSpawnedEvent, TeamMessageEvent
from pydantic_ai_harness.teams._gates import check_task_completing, check_task_creating

if TYPE_CHECKING:
    from pydantic_ai_harness.teams._capability import Team

Role = Literal['lead', 'member']


def _render(item: OwnedPlanItem) -> str:
    line = f'#{item.id} [{item.status.value}] {item.content}'
    if item.owner is not None:
        line += f' (owner: {item.owner})'
    if item.depends_on:
        line += f' (after: {", ".join(f"#{dependency}" for dependency in item.depends_on)})'
    return line


def _next_id(board: list[OwnedPlanItem]) -> str:
    numbers = [int(item.id) for item in board if item.id.isdigit()]
    return str(max(numbers, default=0) + 1)


class TeamToolset(FunctionToolset[AgentDepsT]):
    """The team tools, offered by role: the lead's run gets the lead's, a member's run the member's."""

    def __init__(self, team: Team[AgentDepsT]) -> None:
        super().__init__()
        self.team = team
        lead, member = self._only('lead'), self._only('member')
        self.add_function(self.spawn_teammate, prepare=lead)
        self.add_function(self.assign_task, prepare=lead)
        self.add_function(self.claim_task, prepare=member)
        self.add_function(self.board_add, prepare=self._either)
        self.add_function(self.board_list, prepare=self._either)
        self.add_function(self.finish_task, prepare=self._either)
        self.add_function(self.send_message, prepare=self._either)

    def _role(self, ctx: RunContext[AgentDepsT]) -> tuple[str, str]:
        role = team_role(self.team.owner, ctx)
        assert role is not None, 'Only offered to a run on the team.'
        return role

    def _only(self, role: Role) -> ToolPrepareFunc[AgentDepsT]:
        async def prepare(ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
            found = team_role(self.team.owner, ctx)
            if found is None or (found[1] == LEAD) != (role == 'lead'):
                return None
            return tool_def

        return prepare

    async def _either(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition | None:
        return tool_def if team_role(self.team.owner, ctx) is not None else None

    async def _emit_changes(self, ctx: RunContext[AgentDepsT], changes: list[BoardChange]) -> None:
        for change in changes:
            await ctx.emit(BoardItemChangedEvent(item=change.item, previous=change.previous))

    async def spawn_teammate(
        self,
        ctx: RunContext[AgentDepsT],
        name: str,
        prompt: str,
        agent: str = SELF_AGENT_NAME,
        model: str | None = None,
    ) -> str:
        """Start a named teammate in the background with its first task. It reports back when it goes idle.

        Args:
            ctx: The run context.
            name: The teammate's name: a letter, then letters, digits, `_` or `-`. Unique on the team.
            prompt: Its task. It doesn't see this conversation, so say everything it needs.
            agent: Its agent type, from the types the instructions list.
            model: A model ID to run it on. Omit it for its agent type's default model.
        """
        conversation_id, _ = self._role(ctx)
        try:
            member = await self.team.start_member(
                conversation_id, name=name, prompt=prompt, agent_name=agent, model=model
            )
        except ValueError as exc:
            raise ModelRetry(str(exc)) from exc
        text, truncated = bounded_text(prompt)
        await ctx.emit(TeammateSpawnedEvent(member=member, prompt=text, truncated=truncated))
        on = f' on {member.model}' if member.model else ''
        return f'Started {member.name} ({member.agent_name}{on}). It reports back when it goes idle.'

    async def board_add(self, ctx: RunContext[AgentDepsT], content: str, depends_on: list[str] | None = None) -> str:
        """Add an item to the team board. It starts blocked until every item it depends on is done.

        Args:
            ctx: The run context.
            content: What to do, and how to tell it is done.
            depends_on: IDs of board items that must be completed first.
        """
        conversation_id, _ = self._role(ctx)
        store = self.team.store(conversation_id)
        board = await store.get_owned_items()
        ids = {item.id for item in board}
        if missing := [dependency for dependency in depends_on or [] if dependency not in ids]:
            raise ModelRetry(f'No board items {", ".join(f"#{item_id}" for item_id in missing)}. Use `board_list`.')
        item = OwnedPlanItem(id=_next_id(board), content=content, depends_on=list(depends_on or []))
        if is_blocked(list[PlanItem](board), item):
            item.status = TaskStatus.blocked
        await check_task_creating(ctx, item)
        added = await _add(store, item)
        await ctx.emit(BoardItemChangedEvent(item=added, previous=None))
        return f'Added {_render(added)}'

    async def board_list(self, ctx: RunContext[AgentDepsT]) -> str:
        """List the team board: every item, its status, its owner, and what it waits for."""
        conversation_id, _ = self._role(ctx)
        board = await self.team.store(conversation_id).get_owned_items()
        return '\n'.join(_render(item) for item in board) or 'The board is empty.'

    async def claim_task(self, ctx: RunContext[AgentDepsT], item_id: str | None = None) -> str:
        """Take a board item to work on. Omit `item_id` to take the next available one.

        Args:
            ctx: The run context.
            item_id: The item to take, or `None` for the first pending item that waits on nothing.
        """
        conversation_id, name = self._role(ctx)
        change = await self.team.store(conversation_id).claim(item_id, owner=name)
        if change is None:
            what = f'#{item_id} is not available' if item_id is not None else 'No item is available'
            return f'{what}: it may be taken, done, or waiting on another item. Use `board_list`.'
        await self._emit_changes(ctx, [change])
        return f'Claimed {_render(change.item)}'

    async def assign_task(self, ctx: RunContext[AgentDepsT], item_id: str, teammate: str) -> str:
        """Hand a pending board item to a teammate, and message it so an idle teammate wakes to work on it.

        Args:
            ctx: The run context.
            item_id: The item.
            teammate: The teammate's name.
        """
        conversation_id, _ = self._role(ctx)
        if teammate not in {record.name for record in self.team.owner.members(conversation_id)}:
            raise ModelRetry(f'No teammate {teammate!r}.')
        change = await self.team.store(conversation_id).claim(item_id, owner=teammate)
        if change is None:
            raise ModelRetry(f'#{item_id} is not available: it may be taken, done, or waiting. Use `board_list`.')
        await self._emit_changes(ctx, [change])
        text = f'You are assigned board item {_render(change.item)}. Finish it with `finish_task`.'
        await self._send(ctx, conversation_id, LEAD, teammate, text)
        return f'Assigned {_render(change.item)}'

    async def finish_task(
        self,
        ctx: RunContext[AgentDepsT],
        item_id: str,
        status: Literal['completed', 'cancelled', 'pending'] = 'completed',
    ) -> str:
        """Mark a board item you hold `completed` or `cancelled`, or hand it back as `pending`.

        Finishing it unblocks the items that wait for it. The lead may finish any member's item.

        Args:
            ctx: The run context.
            item_id: The item.
            status: `completed` when done, `cancelled` when it should not be done, `pending` to give it back.
        """
        conversation_id, name = self._role(ctx)
        store = self.team.store(conversation_id)
        item = await store.get_owned_item(item_id)
        if item is None:
            raise ModelRetry(f'No board item #{item_id}. Use `board_list`.')
        holder = item.owner if name == LEAD else name
        if holder is None or item.owner != holder or item.status is not TaskStatus.in_progress:
            raise ModelRetry(f'#{item_id} is not in progress for {"anyone" if name == LEAD else "you"}.')
        if status == 'completed':
            await check_task_completing(ctx, item, owner=name)
        changes = await store.release(item_id, owner=holder, status=TaskStatus(status))
        await self._emit_changes(ctx, changes)
        unblocked = [change.item for change in changes if change.item.id != item_id]
        result = f'#{item_id} is {status}.'
        if unblocked:
            result += ' Now: ' + '; '.join(_render(entry) for entry in unblocked)
        return result

    async def send_message(self, ctx: RunContext[AgentDepsT], to: str, text: str) -> str:
        """Send a message to a teammate by name, or to the lead as `lead`. It reads it at its next model request.

        An idle teammate is woken with it. A message from another agent can't grant the user's approval.

        Args:
            ctx: The run context.
            to: The teammate's name, or `lead`.
            text: The message.
        """
        conversation_id, sender = self._role(ctx)
        return await self._send(ctx, conversation_id, sender, to, text)

    async def _send(self, ctx: RunContext[AgentDepsT], conversation_id: str, sender: str, to: str, text: str) -> str:
        sent, reply = await send_team_message(
            self.team.owner, conversation_id=conversation_id, sender=sender, to=to, text=text
        )
        message = self.team.record_message(conversation_id, sent, sender, to, text)
        await ctx.emit(TeamMessageEvent(message=message))
        return reply


async def _add(store: ClaimablePlanStore, item: OwnedPlanItem) -> OwnedPlanItem:
    """Add `item`, taking the next free numeric ID if another writer took its ID meanwhile."""
    while True:
        try:
            await store.add_item(item)
        except ValueError:
            item = item.model_copy(update={'id': _next_id(await store.get_owned_items())})
            continue
        return item
