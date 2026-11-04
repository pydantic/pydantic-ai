"""Events emitted by the sub-agents capability.

One delegation is one `delegate_task` call, so its start and end events share
the `tool_call_id` core stamps on every event a capability tool emits; that is
how a subscriber pairs them when several delegations run at once.

The text of `DelegationStartEvent` and `DelegationEndEvent` is bounded: `task`
and `output` are cut at `MAX_EVENT_TEXT_CHARS` with a `truncated` flag, so an
event stream that is persisted or forwarded to a UI cannot be flooded by one
verbose delegation. A `SubAgentMessage` is not cut: it is the message itself,
which a transport carries to the other side of the task, so its prompt, question,
answer, or output arrives whole. Nor is the model's text in `AgentCreatedEvent`.
A listener that persists or forwards these events bounds them itself.

As with any capability event, a listener that raises aborts the parent run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal, TypeAlias

from pydantic_ai import AgentStreamEvent, CapabilityEvent
from pydantic_ai.usage import RunUsage, UsageLimits

SUB_AGENTS_EVENTS = 'sub_agents'

MAX_EVENT_TEXT_CHARS = 4096
"""Characters kept of `DelegationStartEvent.task` and `DelegationEndEvent.output` before they are cut."""

DelegationOutcome = Literal['ok', 'timeout', 'budget', 'failed', 'contained', 'cancelled', 'error']
"""How a delegation ended, mirroring the failure-handling contract of `SubAgent`.

- `ok`: the child finished and its output went back to the parent.
- `timeout`: the child exceeded `SubAgent.timeout_seconds`; the parent got a steering message.
- `budget`: the child reached its own `SubAgent.usage_limits`; the parent got a steering message.
- `failed`: the child raised a soft model error (the parent got `SubAgent.on_failure` or a `ModelRetry`),
  or its run ended with `DeferredToolRequests` or `PendingModelResponse` (the parent got a steering message).
- `contained`: the child crashed with `contain_errors` on; the parent got a `ModelRetry`.
"""


def bounded_text(text: str) -> tuple[str, bool]:
    """`text` cut at `MAX_EVENT_TEXT_CHARS`, and whether it was cut."""
    if len(text) <= MAX_EVENT_TEXT_CHARS:
        return text, False
    return text[:MAX_EVENT_TEXT_CHARS], True


@dataclass(kw_only=True)
class DelegationStartEvent(CapabilityEvent, namespace=SUB_AGENTS_EVENTS, name='delegation_start'):
    """A sub-agent run is about to start for one delegation.

    Emitted once the delegation has passed every check that could refuse it (an
    unknown sub-agent, a model key off the menu, an exhausted `max_calls`
    budget), immediately before the child run starts.
    """

    agent_name: str
    task: str
    truncated: bool
    model: str | None
    """The menu key the delegation runs on, or `None` when no option was selected: there is
    no menu, or the delegate allows the whole menu and the parent named no key."""
    task_id: str | None = None
    parent_id: str | None = None
    inherits_tools: bool
    """Whether the parent's own tools were passed to the child run (`SubAgents.inherit_tools`)."""


@dataclass(kw_only=True)
class DelegationEndEvent(CapabilityEvent, namespace=SUB_AGENTS_EVENTS, name='delegation_end'):
    """A delegation settled into what the parent receives.

    `output` is what the delegate tool hands back to the parent: the child's
    output on `ok`, otherwise the steering message it returns or the
    `ModelRetry` it raises. An exception that propagates out of the delegate
    tool (a shared usage limit, an uncontained crash, a cancellation) ends
    without this event for ordinary `SubAgents` users. An opt-in `DelegationTasks`
    owner emits terminal `cancelled` or `error` outcomes with stable task identity.
    """

    agent_name: str
    outcome: DelegationOutcome
    output: str
    truncated: bool
    usage: RunUsage | None
    """The child's own usage when it has separate accounting (`SubAgent.usage_limits`
    set, or `forward_usage` off); `None` when it accrues into the parent's usage."""
    task_id: str | None = None
    parent_id: str | None = None
    duration_seconds: float
    """Wall-clock seconds from just after the start event was emitted until the delegation settled."""


@dataclass(kw_only=True)
class _Vetoable:
    """The veto of an immediately dispatched event: a listener can stop what the event announces.

    Named `veto`, not `cancel` as `FileChangeRequestEvent` has it, as these events sit beside task cancellation.
    """

    vetoed: bool = False
    """Whether a listener vetoed it."""

    veto_reason: str | None = None
    """Why it was vetoed, reported to the model that asked for it."""

    def veto(self, reason: str | None = None) -> None:
        """Stop it from taking effect."""
        self.vetoed = True
        self.veto_reason = reason


Recipient: TypeAlias = Literal['worker', 'task', 'parent']
"""Which side of a task a message is for: the worker serving the agent, the running task, or its parent."""


@dataclass(kw_only=True)
class SubAgentMessage(CapabilityEvent, namespace=SUB_AGENTS_EVENTS, dispatch='immediate', abstract=True):
    """A message between a delegating run and one of its tasks, with `SubAgents(task_control=...)`.

    Messages are dispatched immediately: listeners run before the message takes effect, so they can
    rewrite its fields or, for a `ParentMessage`, veto it. A `SubAgentTransport` carries them to and
    from a `SubAgentWorker` in another process.
    """

    recipient: ClassVar[Recipient]
    """Which side of the task receives this message."""

    task_id: str
    """The task this message belongs to."""

    agent_name: str
    """The name of the sub-agent running the task."""

    @property
    def mailbox(self) -> str:
        """The address a transport delivers this message to.

        `agent:<agent_name>` for a message that starts a run, `task:<task_id>:worker` for one the running
        task receives, and `task:<task_id>:parent` for one its parent receives.
        """
        if self.recipient == 'worker':
            return agent_mailbox(self.agent_name)
        return task_mailbox(self.task_id, 'worker' if self.recipient == 'task' else 'parent')


def agent_mailbox(agent_name: str) -> str:
    """The mailbox of the worker serving `agent_name`, which receives the messages that start a run."""
    return f'agent:{agent_name}'


def task_mailbox(task_id: str, side: Literal['worker', 'parent']) -> str:
    """The mailbox of one side of a task: the worker running it, or its delegating run."""
    return f'task:{task_id}:{side}'


@dataclass(kw_only=True)
class ParentMessage(_Vetoable, SubAgentMessage, abstract=True):
    """A message from a delegating run to its task. A listener can veto it, and then it has no effect."""

    recipient: ClassVar[Recipient] = 'task'


@dataclass(kw_only=True)
class ChildMessage(SubAgentMessage, abstract=True):
    """A message from a task to its delegating run."""

    recipient: ClassVar[Recipient] = 'parent'


@dataclass(kw_only=True)
class TaskRequestEvent(ParentMessage, abstract=True):
    """A request to a worker to start a run of the sub-agent: a new task or a follow-up.

    Not to be confused with `TaskStartedEvent`, the worker's reply once the run started.
    """

    recipient: ClassVar[Recipient] = 'worker'

    prompt: str
    """What the sub-agent is asked to do. A listener can rewrite it."""

    parent_task_id: str | None = None
    """The task whose run started this one, or `None` when the top-level run did."""

    depth: int = 2
    """The delegation level the task runs at, counting the top-level run as 1, like `SubAgents.max_depth`."""

    can_ask_questions: bool = True
    """Whether the sub-agent gets an `ask_parent` tool."""

    forward_events: bool = False
    """Whether the sub-agent's stream events are forwarded as `TaskProgressEvent`s."""

    usage_limits: UsageLimits | None = None
    """The delegating run's usage limits, which a worker in another process applies to the sub-agent's own usage."""

    trace_context: dict[str, str] = field(default_factory=dict[str, str])
    """W3C trace context of the delegation, so a worker in another process continues the delegating run's trace."""


@dataclass(kw_only=True)
class TaskAssignedEvent(TaskRequestEvent, name='task_assigned'):
    """The delegating run assigns a new task to a sub-agent."""


@dataclass(kw_only=True)
class TaskFollowUpEvent(TaskRequestEvent, name='task_follow_up'):
    """The delegating run resumes a finished task's conversation with a new prompt."""


@dataclass(kw_only=True)
class TaskMessageEvent(ParentMessage, name='task_message'):
    """The delegating run steers a running task, which sees the message at its next model request."""

    message: str
    """The steering message. A listener can rewrite it."""


@dataclass(kw_only=True)
class TaskAnswerEvent(ParentMessage, name='task_answer'):
    """The delegating run answers the question a task is waiting on."""

    question_id: str
    """The `TaskQuestionEvent.question_id` this answers."""

    answer: str
    """The answer. A listener can rewrite it."""


@dataclass(kw_only=True)
class TaskCancelRequestEvent(ParentMessage, name='task_cancel_request'):
    """The delegating run asks a running task to stop before its next model request."""


@dataclass(kw_only=True)
class TaskForceCancelEvent(ParentMessage, name='task_force_cancel'):
    """The delegating run stops a running task immediately, interrupting its current model request or tool calls."""


@dataclass(kw_only=True)
class TaskStartedEvent(ChildMessage, name='task_started'):
    """A worker picked the task up and its run is starting.

    A remote task that no worker picks up within `TaskControl.accept_timeout` fails.
    """


@dataclass(kw_only=True)
class TaskProgressEvent(ChildMessage, name='task_progress', dispatch='stream'):
    """An event from the sub-agent's own run, forwarded while the task runs, with `TaskControl.forward_events`.

    Dispatched at its stream position, like the event it carries.
    """

    event: AgentStreamEvent
    """The sub-agent's stream event."""


@dataclass(kw_only=True)
class TaskQuestionEvent(ChildMessage, name='task_question'):
    """The task asks its delegating run a question and waits for a `TaskAnswerEvent`.

    A listener can answer it by setting `answer`, and then the delegating model is not asked.
    """

    question_id: str
    """Identifies the question, so the answer can name it."""

    question: str
    """The question."""

    answer: str | None = None
    """An answer a listener supplies, sent to the task without involving the delegating model."""


@dataclass(kw_only=True)
class TaskRetryEvent(ChildMessage, name='task_retry'):
    """A remote task's run failed and its worker is running it again, as `SubAgentWorker.retries` allows."""

    attempt: int
    """The attempt about to start; the first retry is attempt 2."""

    error: str
    """The error the previous attempt failed with."""


@dataclass(kw_only=True)
class TaskSettledEvent(ChildMessage, abstract=True):
    """The task's run ended. This is the task's report to its delegating run."""

    usage: RunUsage | None = None
    """The sub-agent's usage when a worker in another process counted it separately.

    `None` for a sub-agent that ran in the delegating run's process: its usage was counted the way
    `SubAgents.forward_usage` and `SubAgent.usage_limits` decide.
    """


@dataclass(kw_only=True)
class TaskCompletedEvent(TaskSettledEvent, name='task_completed'):
    """The task finished with an output."""

    output: str
    """The sub-agent's output, as text. A listener can rewrite it before the delegating model reads it."""


@dataclass(kw_only=True)
class TaskFailedEvent(TaskSettledEvent, name='task_failed'):
    """The task's run failed, or ended in a way that delivered no output."""

    error: str
    """What went wrong, as reported to the delegating model."""


@dataclass(kw_only=True)
class TaskCancelledEvent(TaskSettledEvent, name='task_cancelled'):
    """The task stopped because it was cancelled. A resumable task can be followed up on."""

    forced: bool
    """Whether the task was stopped immediately rather than before its next model request."""


BackgroundReason: TypeAlias = Literal['requested', 'auto', 'question']
"""Why a task moved to the background.

- `requested`: the model delegated with `background=True`.
- `auto`: the task ran longer than `TaskControl.auto_background_after`.
- `question`: the task asked a question, which the delegating model can only answer once the delegation tool returns.
"""


@dataclass(kw_only=True)
class TaskBackgroundedEvent(CapabilityEvent, namespace=SUB_AGENTS_EVENTS, name='task_backgrounded'):
    """A task now runs in the background: the delegation tool returned a receipt, and its report arrives later."""

    task_id: str
    """The task now running in the background."""

    agent_name: str
    """The name of the sub-agent running the task."""

    reason: BackgroundReason
    """Why it moved to the background."""


@dataclass(kw_only=True)
class AgentCreatedEvent(
    _Vetoable, CapabilityEvent, namespace=SUB_AGENTS_EVENTS, name='agent_created', dispatch='immediate'
):
    """The model is creating a sub-agent with `create_agent` (`SubAgents.dynamic_agents`).

    Dispatched immediately, so a listener can rewrite `description` and `instructions`, or veto the
    creation.
    """

    name: str
    """The name the model gave the sub-agent, which it delegates to."""

    description: str
    """What the sub-agent is for. It reaches only a custom `DynamicAgents.factory`, as `DynamicAgentSpec.description`;
    the model is never shown it. A listener can rewrite it."""

    instructions: str
    """The sub-agent's instructions. A listener can rewrite them."""

    model: str | None
    """The `SubAgents.models` key the sub-agent runs on, or `None` for the delegating run's model."""
