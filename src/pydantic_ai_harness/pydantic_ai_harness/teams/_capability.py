"""The `Team` capability: a lead and named members working one board, checked by the same gates."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    CombinedCapability,
    NodeResult,
    WrapperCapability,
)
from pydantic_ai.output import PendingModelResponse
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness.planning import ClaimablePlanStore, OwnedPlanItem, TaskStatus
from pydantic_ai_harness.subagents import (
    SELF_AGENT_NAME,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    MemberRun,
    TeamMessages,
)
from pydantic_ai_harness.subagents._events import bounded_text
from pydantic_ai_harness.subagents._team import LEAD, MESSAGES_INSTRUCTIONS, SentMessage, team_role
from pydantic_ai_harness.teams._events import (
    BoardItemChangedEvent,
    TeammateIdleEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMessageEvent,
)
from pydantic_ai_harness.teams._gates import joined, trace_rejection
from pydantic_ai_harness.teams._toolset import TeamToolset
from pydantic_ai_harness.teams._types import MemberState, TeamMember, TeamMessage, TeamSnapshot, member_state
from pydantic_graph import End

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.agent import AbstractAgent

USER = 'user'
"""The sender name of a message the user sent a member through `Team.send`."""

TeamEvent = BoardItemChangedEvent | TeammateSpawnedEvent | TeammateStateEvent | TeamMessageEvent
"""An event `Team` sends its `observer`: a change the owner or the user made, outside any agent run."""

TeamObserver = Callable[[TeamEvent], Awaitable[None]]
BoardResolver = Callable[[str], ClaimablePlanStore]
"""Returns the board of a conversation, by its ID."""


@dataclass
class _IdleGate(AbstractCapability[AgentDepsT]):
    """Asks the team's gates whether a member may go idle, as its run is about to end."""

    member: str

    def get_ordering(self) -> CapabilityOrdering:
        # Outermost runs `after_node_run` last, so the gate sees what other capabilities enqueued
        # there (a background tool's result, say) and does not fire for a run that continues anyway.
        return CapabilityOrdering(position='outermost')

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        # Queued messages turn the end into another model request, so the member is not going idle.
        if not isinstance(result, End) or ctx.pending_messages:
            return result
        output = result.data.output
        # A run paused for deferred tools or a pending provider response is waiting, not idle.
        if isinstance(output, DeferredToolRequests | PendingModelResponse):
            return result
        text, truncated = bounded_text(str(output))
        event = await ctx.emit(TeammateIdleEvent(member=self.member, final_output=text, truncated=truncated))
        if event.feedback:
            trace_rejection(ctx, 'teammate_idle', event.feedback, member=self.member, item_id=None)
            # Core's end-of-run drain turns content enqueued here into the member's next model request.
            ctx.enqueue(joined(event.feedback), priority='asap')
        return result


@dataclass(init=False)
class Team(WrapperCapability[AgentDepsT]):
    """A team: a lead that spawns named members, a shared board they claim tasks from, messages, and gates.

    Built on a `DelegationTasks` owner, which runs the members, and a `ClaimablePlanStore` per conversation,
    which holds the board. Bind it to the lead's agent, or pass it to each lead run, inside `owner.bind()`. Run
    each member's turn with `member_run`, which adds the team to the member's run.

    The lead gets `spawn_teammate`, `assign_task`, `board_add`, `board_list`, `finish_task` and `send_message`;
    each member gets `claim_task`, `board_add`, `board_list`, `finish_task` and `send_message`. A member's own
    sub-agents, and any other delegation, get no team tools: teams don't nest.

    Args:
        owner: Runs the members. Give it `observer=` that calls `observe`, and `claims=claims`, so the roster
            stays current and a member that fails or stops hands its board items back.
        board: The board of every conversation, or a function returning a conversation's board by its ID.
        gates: Capabilities whose event listeners decide the gates (`TeammateIdleEvent`,
            `TeamTaskCreatingEvent`, `TeamTaskCompletingEvent`), on the lead's run and every member's run.
        models: The model ID a member of each agent type runs on, by agent type (`self` for the lead's own
            agent), unless `spawn_teammate` names one. A type with none runs on the lead's model, or a
            specialist on its agent's own model.
        member_usage_limits: Each member turn's budget. A gate that keeps rejecting keeps a member working,
            so set one; a turn over it fails, and the member is not woken again until a new message arrives.
        observer: Told about the changes the owner or the user makes, outside any agent run: member state,
            claims handed back, members the host spawns, and messages the user sends. Changes a tool call makes
            are emitted into that run's event stream instead.
        max_messages: How many of a conversation's latest messages `snapshot` keeps.
    """

    id: str | None = field(default='team', kw_only=True)
    """Fixed: a team's tool names are fixed, so an agent has one team. See `combine`."""

    owner: DelegationTasks
    board: ClaimablePlanStore | BoardResolver
    gates: tuple[AbstractCapability[AgentDepsT], ...]
    models: Mapping[str, str]
    member_usage_limits: UsageLimits | None
    observer: TeamObserver | None = field(compare=False)
    max_messages: int = field(compare=False)
    _messages: dict[str, deque[TeamMessage]] = field(compare=False, repr=False)
    _states: dict[str, MemberState] = field(compare=False, repr=False)

    def __init__(
        self,
        owner: DelegationTasks,
        *,
        board: ClaimablePlanStore | BoardResolver,
        gates: Sequence[AbstractCapability[AgentDepsT]] = (),
        models: Mapping[str, str] | None = None,
        member_usage_limits: UsageLimits | None = None,
        observer: TeamObserver | None = None,
        max_messages: int = 100,
    ) -> None:
        self.owner = owner
        self.board = board
        self.gates = tuple(gates)
        self.models = dict(models or {})
        self.member_usage_limits = member_usage_limits
        self.observer = observer
        self.max_messages = max_messages
        self._messages = {}
        self._states = {}
        super().__init__(CombinedCapability[AgentDepsT](list(self.gates)), id='team')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable -- the capability holds a live `DelegationTasks` owner and board."""
        return None

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """The same team stated twice is one; two different teams on one agent raise rather than mix."""
        return one_per_id(capabilities, remedy='Bind one `Team` per agent')

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        return self._instructions

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        return TeamToolset[AgentDepsT](self)

    def _instructions(self, ctx: RunContext[AgentDepsT]) -> str | None:
        role = team_role(self.owner, ctx)
        if role is None:
            return None
        _, name = role
        if name == LEAD:
            types = ', '.join(
                f'`{agent}` ({self.models.get(agent) or "default model"})' for agent in self.agent_types()
            )
            return (
                f'{MESSAGES_INSTRUCTIONS} You lead the team. Start teammates with `spawn_teammate` (agent types: '
                f'{types}), put the work on the shared board with `board_add`, and hand items out with '
                '`assign_task`; idle teammates also claim items themselves. Teammates report back to you when '
                'they go idle.'
            )
        return (
            f'{MESSAGES_INSTRUCTIONS} You are `{name}`. Take work from the shared board with `claim_task`, mark '
            'it done with `finish_task`, and add follow-up work with `board_add`. Message the lead when you are '
            'blocked or done.'
        )

    def agent_types(self) -> list[str]:
        """The agent types a member can be: `self`, then the owner's agents that can be woken."""
        agents = self.owner.agents
        return [
            SELF_AGENT_NAME,
            *(name for name, agent in agents.items() if agent.resumable and name not in self.owner.one_shot),
        ]

    def member_capability(self, name: str) -> AbstractCapability[AgentDepsT]:
        """The capability to add to the run of the member called `name`: this team, and its idle gate.

        When the member's run is about to end with a final output, the idle gate emits a `TeammateIdleEvent`,
        and feedback from `keep_working` becomes the member's next user message, so the run continues.
        `member_run` adds it for you.
        """
        return CombinedCapability[AgentDepsT]([self, _IdleGate[AgentDepsT](member=name)])

    def member_run(self, record: DelegationTask, *, self_agent: AbstractAgent[Any, Any]) -> MemberRun:
        """What a `member_runner` runs for this turn of the member `record`: `DelegationTasks.member_run`, with the team.

        The team's tools replace `TeamMessages`, and `member_usage_limits` is the turn's default budget.
        `MemberRun.model` is the model ID the member is pinned to, or `None` for the lead's model.
        """
        assert record.name is not None, 'Only team members run as members.'
        run = self.owner.member_run(record, self_agent=self_agent)
        capabilities = [capability for capability in run.capabilities if not isinstance(capability, TeamMessages)]
        return replace(
            run,
            capabilities=[*capabilities, self.member_capability(record.name)],
            usage_limits=run.usage_limits or self.member_usage_limits,
        )

    def store(self, conversation_id: str) -> ClaimablePlanStore:
        """The board of `conversation_id`."""
        return self.board if isinstance(self.board, ClaimablePlanStore) else self.board(conversation_id)

    def claims(self, conversation_id: str) -> _Claims:
        """The board the owner hands a failed or stopped member's claims back to; pass `owner.claims = team.claims`."""
        return _Claims(self, conversation_id)

    async def observe(self, update: DelegationTaskEvent) -> None:
        """Track a member's state from the owner's events; call it from the owner's `observer`."""
        record = update.task
        if record.name is None:
            return
        state = member_state(record)
        previous = self._states.get(record.id)
        if state == previous:
            return
        self._states[record.id] = state
        await self.publish(TeammateStateEvent(member=await self.member(record), previous=previous))

    async def publish(self, event: TeamEvent) -> None:
        """Tell the `observer` about a change made outside any agent run."""
        if self.observer is not None:
            await self.observer(event)

    async def member(self, record: DelegationTask) -> TeamMember:
        """The roster entry of the member `record`, with the board item it is working on."""
        assert record.name is not None, 'Only team members are on the roster.'
        items = await self.store(record.conversation_id).get_owned_items()
        current = next(
            (item.id for item in items if item.owner == record.name and item.status is TaskStatus.in_progress), None
        )
        return TeamMember(
            name=record.name,
            task_id=record.id,
            agent_name=record.agent_name,
            model=record.model,
            state=member_state(record),
            current_task=current,
        )

    def default_model(self, agent_name: str) -> str | None:
        """The model ID a new member of `agent_name` runs on when `spawn` names none."""
        resolved = self.owner.aliases.get(agent_name, agent_name)
        for agent, model in self.models.items():
            if self.owner.aliases.get(agent, agent) == resolved:
                return model
        return None

    async def spawn(
        self,
        conversation_id: str,
        *,
        name: str,
        prompt: str,
        agent_name: str = SELF_AGENT_NAME,
        model: str | None = None,
    ) -> TeamMember:
        """Start a member, pinned to `model`, or to `models[agent_name]` when `model` is `None`.

        The model is saved on the member's record, so it keeps it across idle, wake and restart. The host calls
        this for the user; the lead's `spawn_teammate` tool calls it too. Here, the `observer` hears of it.

        Raises:
            ValueError: The name is invalid or taken, or the agent type is unknown or one-shot.
        """
        member = await self.start_member(conversation_id, name=name, prompt=prompt, agent_name=agent_name, model=model)
        text, truncated = bounded_text(prompt)
        await self.publish(TeammateSpawnedEvent(member=member, prompt=text, truncated=truncated))
        return member

    async def start_member(
        self, conversation_id: str, *, name: str, prompt: str, agent_name: str, model: str | None
    ) -> TeamMember:
        """`spawn`, without telling the `observer`: for a tool, which emits the event into its own run instead."""
        record = await self.owner.start(
            name=name,
            agent_name=agent_name,
            prompt=prompt,
            conversation_id=conversation_id,
            model=model or self.default_model(agent_name),
        )
        return await self.member(record)

    async def send(self, conversation_id: str, *, to: str, text: str) -> SentMessage:
        """Send the user's `text` to the member named `to`: it arrives as their words, and wakes it if idle.

        It clears a stop the user made, as their message to a member does. The `observer` hears of it.

        Raises:
            ValueError: The text is empty, or no member of the conversation is named `to`.
        """
        record = next((member for member in self.owner.members(conversation_id) if member.name == to), None)
        if record is None:
            raise ValueError(f'No team member {to!r} in this conversation')
        sent = await self.owner.send(text, to=record.id, conversation_id=conversation_id)
        await self.publish(TeamMessageEvent(message=self.record_message(conversation_id, sent, USER, to, text)))
        return sent

    async def stop(self, conversation_id: str, name: str) -> None:
        """Stop the member named `name` as the user: only a message from the user wakes it again.

        Raises:
            ValueError: No member of the conversation is named `name`.
        """
        record = next((member for member in self.owner.members(conversation_id) if member.name == name), None)
        if record is None:
            raise ValueError(f'No team member {name!r} in this conversation')
        await self.owner.cancel(record.id)
        await self.owner.save(record)
        await self.observe(DelegationTaskEvent(task=record))

    def record_message(self, conversation_id: str, sent: SentMessage, sender: str, to: str, text: str) -> TeamMessage:
        """Keep a sent message for `snapshot`, and return it."""
        bounded, truncated = bounded_text(text)
        message = TeamMessage(id=sent.id, sender=sender, recipient=to, text=bounded, truncated=truncated)
        self._messages.setdefault(conversation_id, deque(maxlen=self.max_messages)).append(message)
        return message

    async def snapshot(self, conversation_id: str) -> TeamSnapshot:
        """The team of `conversation_id` as it is now: its roster, its board, and its latest messages."""
        return TeamSnapshot(
            conversation_id=conversation_id,
            members=[await self.member(record) for record in self.owner.members(conversation_id)],
            board=await self.store(conversation_id).get_owned_items(),
            messages=list(self._messages.get(conversation_id, ())),
        )


@dataclass
class _Claims:
    """A conversation's board, as the owner hands back a member's claims, telling the team's observer."""

    team: Team[Any]
    conversation_id: str

    async def get_owned_items(self) -> list[OwnedPlanItem]:
        return await self.team.store(self.conversation_id).get_owned_items()

    async def release(self, item_id: str, *, owner: str, status: TaskStatus) -> object:
        changes = await self.team.store(self.conversation_id).release(item_id, owner=owner, status=status)
        for change in changes:
            await self.team.publish(BoardItemChangedEvent(item=change.item, previous=change.previous))
        return changes
