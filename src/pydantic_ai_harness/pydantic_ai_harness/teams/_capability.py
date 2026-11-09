"""The `Team` capability: a lead and named members working one board, checked by the same gates."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    CombinedCapability,
    NodeResult,
    WrapperCapability,
    WrapRunHandler,
)
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.output import PendingModelResponse
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, DeferredToolRequests, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness._combine import one_per_id
from pydantic_ai_harness.planning import ClaimablePlanStore, OwnedPlanItem, TaskStatus
from pydantic_ai_harness.subagents import (
    SELF_AGENT_NAME,
    AgentMessage,
    DelegationTask,
    DelegationTaskEvent,
    DelegationTasks,
    MemberRun,
    PlanRequest,
    TeamMessages,
)
from pydantic_ai_harness.subagents._events import bounded_text
from pydantic_ai_harness.subagents._team import LEAD, MESSAGES_INSTRUCTIONS, SentMessage, team_role
from pydantic_ai_harness.teams._events import (
    BoardItemChangedEvent,
    PlanApprovalRequestedEvent,
    PlanReviewedEvent,
    TeammateIdleEvent,
    TeammateSpawnedEvent,
    TeammateStateEvent,
    TeamMessageEvent,
)
from pydantic_ai_harness.teams._gates import joined, trace_rejection
from pydantic_ai_harness.teams._toolset import TeamToolset
from pydantic_ai_harness.teams._types import (
    MemberState,
    ShutdownState,
    TeamMember,
    TeamMessage,
    TeamSnapshot,
    member_state,
    shutdown_state,
)
from pydantic_graph import End

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.agent import AbstractAgent

USER = 'user'
"""The sender name of a message the user sent a member through `Team.send`."""

TeamEvent = (
    BoardItemChangedEvent
    | TeammateSpawnedEvent
    | TeammateStateEvent
    | TeamMessageEvent
    | PlanApprovalRequestedEvent
    | PlanReviewedEvent
)
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


@dataclass
class _Accounting(AbstractCapability[AgentDepsT]):
    """Adds up the usage of the lead's and members' runs as they go, and holds them to the team's `usage_limits`."""

    team: Team[AgentDepsT] = field(repr=False, compare=False)
    _counted: dict[str | None, RunUsage] = field(default_factory=dict[str | None, RunUsage], repr=False, compare=False)
    """The usage of each run on the team, by run ID, as far as it was added up."""

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        if team_role(self.team.owner, ctx) is None:
            return await handler()
        self._counted[ctx.run_id] = copy(ctx.usage)
        try:
            return await handler()
        finally:
            await self._count(ctx)
            del self._counted[ctx.run_id]

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        if ctx.run_id in self._counted:
            conversation_id = await self._count(ctx)
            self.team.check_usage(conversation_id)
        return request_context

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        # Counted as each step ends, so a run waiting in a tool, say on a plan's review, shows what it used.
        if ctx.run_id in self._counted:
            await self._count(ctx)
        return result

    async def _count(self, ctx: RunContext[AgentDepsT]) -> str:
        """Add what the run used since it was last counted to its member's or its lead's usage; its conversation."""
        role = team_role(self.team.owner, ctx)
        assert role is not None, 'Only counted for runs on the team.'
        conversation_id, _ = role
        used = ctx.usage - self._counted[ctx.run_id]
        self._counted[ctx.run_id] = copy(ctx.usage)
        if not used.has_values():
            return conversation_id
        task_id = self.team.owner.child_id()
        if task_id is None:
            await self.team.owner.add_lead_usage(conversation_id, used)
        else:
            # Saved with the record at the end of the member's turn.
            record = self.team.owner.records[task_id]
            record.usage = record.usage + used
        return conversation_id


@dataclass(init=False)
class Team(WrapperCapability[AgentDepsT]):
    """A team: a lead that spawns named members, a shared board they claim tasks from, messages, and gates.

    Built on a `DelegationTasks` owner, which runs the members, and a `ClaimablePlanStore` per conversation,
    which holds the board. Bind it to the lead's agent, or pass it to each lead run, inside `owner.bind()`. Run
    each member's turn with `member_run`, which adds the team to the member's run.

    The lead gets `spawn_teammate`, `assign_task`, `board_add`, `board_list`, `finish_task`, `send_message`,
    `shutdown_teammate` and, when it reviews plans, `review_plan`; each member gets `claim_task`, `board_add`,
    `board_list`, `finish_task`, `send_message` and `request_plan_approval`. A member's own sub-agents, and any
    other delegation, get no team tools: teams don't nest.

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
        plan_reviewer: Who approves or rejects a member's `request_plan_approval`: `'lead'`, through its
            `review_plan` tool, or only `'user'`, through `approve` and `reject`. The user can always review.
        usage_limits: The whole team's budget, the lead's runs and every member's turns together, checked before
            each of their model requests. A request over it fails the run that would make it.
        shutdown_grace: How many seconds a member asked to shut down may take to wrap up before it is stopped
            anyway. `None` waits for its turn to end.
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
    plan_reviewer: Literal['lead', 'user']
    usage_limits: UsageLimits | None
    shutdown_grace: float | None
    _messages: dict[str, deque[TeamMessage]] = field(compare=False, repr=False)
    _states: dict[str, tuple[MemberState, ShutdownState | None]] = field(compare=False, repr=False)
    _plans: dict[str, asyncio.Future[str]] = field(compare=False, repr=False)

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
        plan_reviewer: Literal['lead', 'user'] = 'lead',
        usage_limits: UsageLimits | None = None,
        shutdown_grace: float | None = 120.0,
    ) -> None:
        self.owner = owner
        self.board = board
        self.gates = tuple(gates)
        self.models = dict(models or {})
        self.member_usage_limits = member_usage_limits
        self.observer = observer
        self.max_messages = max_messages
        self.plan_reviewer = plan_reviewer
        self.usage_limits = usage_limits
        self.shutdown_grace = shutdown_grace
        self._messages = {}
        self._states = {}
        self._plans = {}
        super().__init__(CombinedCapability[AgentDepsT]([*self.gates, _Accounting[AgentDepsT](self)]), id='team')

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
                "they go idle: don't wait or poll for them, end your turn, and their reports arrive as your next "
                'input. Shut down a teammate you no longer need with `shutdown_teammate`.'
                + (
                    ' Approve or reject the plans teammates submit with `review_plan`.'
                    if self.plan_reviewer == LEAD
                    else ''
                )
            )
        return (
            f'{MESSAGES_INSTRUCTIONS} You are `{name}`. Take work from the shared board with `claim_task`, mark '
            'it done with `finish_task`, and add follow-up work with `board_add`. Message the lead when you are '
            'blocked or done. When you are asked to plan first, submit the plan with `request_plan_approval` and '
            'follow the answer. When you are asked to shut down, finish or hand back (`pending`) the item you hold, '
            'then end your turn.'
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
        current = member_state(record), shutdown_state(record)
        previous = self._states.get(record.id)
        if current == previous:
            return
        self._states[record.id] = current
        await self.publish(
            TeammateStateEvent(member=await self.member(record), previous=None if previous is None else previous[0])
        )

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
            shutdown=shutdown_state(record),
            plan=record.plan_request,
            usage=copy(record.usage),
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
        record = self._record(conversation_id, to)
        sent = await self.owner.send(text, to=record.id, conversation_id=conversation_id)
        await self.publish(TeamMessageEvent(message=self.record_message(conversation_id, sent, USER, to, text)))
        return sent

    async def stop(self, conversation_id: str, name: str) -> None:
        """Stop the member named `name` as the user: only a message from the user wakes it again.

        Raises:
            ValueError: No member of the conversation is named `name`.
        """
        record = self._record(conversation_id, name)
        await self.owner.cancel(record.id)
        await self.owner.save(record)
        await self.observe(DelegationTaskEvent(task=record))

    def _record(self, conversation_id: str, name: str) -> DelegationTask:
        record = next((member for member in self.owner.members(conversation_id) if member.name == name), None)
        if record is None:
            raise ValueError(f'No team member {name!r} in this conversation')
        return record

    async def shutdown(self, conversation_id: str, name: str, reason: str = '') -> bool:
        """Shut the member named `name` down as the user: it wraps up, then stops; the `observer` hears of it.

        It is asked, in the user's words, to finish or hand back the board item it holds and end its turn. When
        that turn ends, the items it still holds go back to the board, its report reaches the lead, and it is
        `stopped`, with `TeamMember.shutdown` set to `'done'`, unlike a member the user stopped. Only a message
        from the user wakes it again. A member that is queued or stopped is shut down at once. A wrap-up longer
        than `shutdown_grace` is stopped.

        Returns:
            Whether the member was asked to wrap up; `False` when it was shut down at once.

        Raises:
            ValueError: No member of the conversation is named `name`, or it is shut down already.
        """
        message = await self.request_shutdown(conversation_id, name, sender=USER, reason=reason)
        if message is not None:
            await self.publish(TeamMessageEvent(message=message))
        return message is not None

    async def request_shutdown(
        self, conversation_id: str, name: str, *, sender: str, reason: str
    ) -> TeamMessage | None:
        """`shutdown`, for `sender`, the lead or the user, without telling the `observer` of the request.

        Returns the request as a message, or `None` when the member was shut down at once.
        """
        record = self._record(conversation_id, name)
        reason = reason.strip()
        if reason and not reason.endswith(('.', '!', '?')):
            reason += '.'
        text = (
            f'Shutdown requested by {sender}.{f" Reason: {reason}" if reason else ""} Wrap up now: finish the board '
            'item you hold with `finish_task`, or hand it back as `pending`, then end your turn. You are stopped '
            'when this turn ends.'
        )
        request = text if sender == USER else AgentMessage(sender=sender, text=text)
        sent = await self.owner.shutdown(
            request, to=record.id, conversation_id=conversation_id, grace=self.shutdown_grace
        )
        return None if sent is None else self.record_message(conversation_id, sent, sender, name, text)

    async def submit_plan(
        self, task_id: str, plan: str, *, announce: Callable[[PlanRequest], Awaitable[object]]
    ) -> str:
        """Save the plan of the member whose task is `task_id`, `announce` it, and wait for its review.

        For a tool: `request_plan_approval` announces it in its own run, and to the lead. Returns the review, as
        the member's tool result. If the member's run ends first, as on a restart, the plan stays saved, and its
        review arrives as a message.

        Raises:
            ValueError: The member already waits on a plan.
        """
        if task_id in self._plans:
            raise ValueError('Your plan already waits for approval.')
        # Registered before anything is awaited, so a second request racing this one is refused.
        waiting = self._plans[task_id] = asyncio.get_running_loop().create_future()
        try:
            record = self.owner.records[task_id]
            request = record.plan_request = PlanRequest(plan=plan)
            await self.owner.save(record)
            await announce(request)
            return await waiting
        finally:
            del self._plans[task_id]

    async def approve(self, conversation_id: str, name: str, feedback: str = '') -> None:
        """Approve, as the user, the plan the member named `name` waits on; the `observer` hears of it.

        Raises:
            ValueError: No member of the conversation is named `name`, or it has no plan waiting.
        """
        await self.publish(await self.review(conversation_id, name, approved=True, feedback=feedback, reviewer=USER))

    async def reject(self, conversation_id: str, name: str, feedback: str) -> None:
        """Reject, as the user, the plan the member named `name` waits on, telling it what to change.

        Raises:
            ValueError: No member of the conversation is named `name`, it has no plan waiting, or `feedback` is
                empty.
        """
        await self.publish(await self.review(conversation_id, name, approved=False, feedback=feedback, reviewer=USER))

    async def review(
        self, conversation_id: str, name: str, *, approved: bool, feedback: str, reviewer: str
    ) -> PlanReviewedEvent:
        """Approve or reject the plan of the member named `name`, for `reviewer`, without telling the `observer`.

        A member still waiting gets the review as its `request_plan_approval` result. One whose run ended
        meanwhile gets it as a message, which wakes it: the user's words from the user, else the lead's message.
        Returns the event to emit.

        Raises:
            ValueError: No member of the conversation is named `name`, it has no plan waiting, or a rejection
                has no `feedback`.
        """
        record = self._record(conversation_id, name)
        request = record.plan_request
        if request is None:
            raise ValueError(f'{name} has no plan waiting for approval')
        if not approved and not feedback.strip():
            raise ValueError('Say what to change when rejecting a plan')
        record.plan_request = None
        await self.owner.save(record)
        if approved:
            text = f'Your plan was approved by the {reviewer}. Go ahead.'
            text += f' Notes: {feedback}' if feedback.strip() else ''
        else:
            text = (
                f'Your plan was rejected by the {reviewer}: {feedback}\n\nRevise it, and submit it again with '
                '`request_plan_approval` before you act on it.'
            )
        waiting = self._plans.get(record.id)
        if waiting is not None and not waiting.done():
            waiting.set_result(text)
        else:
            content = text if reviewer == USER else AgentMessage(sender=reviewer, text=text)
            await self.owner.send(content, to=record.id, conversation_id=conversation_id)
        bounded, truncated = bounded_text(feedback)
        return PlanReviewedEvent(
            member=name,
            request_id=request.id,
            approved=approved,
            feedback=bounded,
            reviewer=reviewer,
            truncated=truncated,
        )

    def usage(self, conversation_id: str) -> RunUsage:
        """The whole team's usage so far: its lead's runs while leading it, and every member's turns."""
        total = self.owner.lead_usage(conversation_id)
        for record in self.owner.members(conversation_id):
            total.incr(record.usage)
        return total

    def check_usage(self, conversation_id: str) -> None:
        """Hold the team of `conversation_id` to `usage_limits` before one of its runs makes a model request.

        Raises:
            UsageLimitExceeded: The team's usage is at or over a limit.
        """
        if self.usage_limits is None:
            return
        usage = self.usage(conversation_id)
        try:
            self.usage_limits.check_before_request(usage)
            self.usage_limits.check_tokens(usage)
        except UsageLimitExceeded as exc:
            raise UsageLimitExceeded(f'Team usage limit: {exc}') from exc

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
            lead_usage=self.owner.lead_usage(conversation_id),
            usage=self.usage(conversation_id),
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
