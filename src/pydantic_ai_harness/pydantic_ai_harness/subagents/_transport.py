"""How sub-agent messages travel between a delegating run and the worker running a remote task."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import cache

import anyio
from pydantic import TypeAdapter

from pydantic_ai import CapabilityEvent

# Private import: core has no public way to tell when new event kinds were registered, which the
# cached adapter below has to be rebuilt for.
from pydantic_ai._event_registry import event_registry_version
from pydantic_ai_harness.subagents._events import SubAgentMessage


class SubAgentTransport(ABC):
    """Carries `SubAgentMessage`s between a delegating run and the `SubAgentWorker` serving its remote agents.

    A transport is a set of mailboxes. `send` delivers a message to the mailbox named by its `mailbox`
    property, and `receive` waits for the next message in a mailbox. Messages a mailbox receives before
    anyone waits on it are kept, in order.

    The delegating run relays a task's messages until the task settles, so the worker must end every task it
    picks up with one `TaskSettledEvent` (`TaskCompletedEvent`, `TaskFailedEvent` or `TaskCancelledEvent`) to
    its parent mailbox. `SubAgentWorker` does, including when it stops.

    Implement this class to reach agents in other processes or machines, over a WebSocket, gRPC, a Redis
    list, or anything else that can move bytes: use `serialize_message` and `deserialize_message` at the
    boundary. `InMemoryTransport` serves a worker in the same process.

    The transport is the trust boundary: anyone who can send to it can run the served agents, with their
    tools and dependencies. Authenticate and authorize senders in the transport.
    """

    @abstractmethod
    async def send(self, message: SubAgentMessage) -> None:
        """Deliver `message` to `message.mailbox`, without waiting for anyone to receive it."""
        raise NotImplementedError

    @abstractmethod
    async def receive(self, mailbox: str) -> SubAgentMessage:
        """Wait for the next message delivered to `mailbox` and return it, in the order they were sent.

        A stopping worker reports its running tasks as failed. Raise once a worker is gone without doing so,
        if the transport can tell: otherwise its delegating run waits for the task until it is cancelled.
        """
        raise NotImplementedError

    async def release(self, mailbox: str) -> None:
        """Discard `mailbox` and any messages left in it, once no one will receive from it again.

        Called when a task settles, for its two task mailboxes. The default does nothing.

        A message sent after the release, such as the late reply of a worker to a task that timed out,
        starts the mailbox again, and nobody may receive it. A transport that serves many tasks should
        expire such messages, for example with a time to live.
        """


class InMemoryTransport(SubAgentTransport):
    """A transport whose mailboxes are in-process queues, for a worker in the same process.

    Messages are passed by reference, not serialized.
    """

    def __init__(self) -> None:
        """Create a transport with no mailboxes; each is created when first sent to or received from."""
        self._mailboxes: dict[str, _Mailbox] = {}

    async def send(self, message: SubAgentMessage) -> None:
        """Append `message` to its mailbox's queue and wake whoever waits on that mailbox."""
        mailbox = self._mailboxes.setdefault(message.mailbox, _Mailbox())
        mailbox.messages.append(message)
        mailbox.arrived.set()

    async def receive(self, mailbox: str) -> SubAgentMessage:
        """Take the oldest message in `mailbox`, waiting until one arrives."""
        box = self._mailboxes.setdefault(mailbox, _Mailbox())
        while not box.messages:
            # Receivers waiting together share one event; the first to wake after it fired renews it.
            if box.arrived.is_set():
                box.arrived = anyio.Event()
            await box.arrived.wait()
        return box.messages.popleft()

    async def release(self, mailbox: str) -> None:
        """Drop `mailbox` and any messages still queued in it."""
        self._mailboxes.pop(mailbox, None)


@dataclass
class _Mailbox:
    messages: deque[SubAgentMessage] = field(default_factory=deque[SubAgentMessage])
    arrived: anyio.Event = field(default_factory=anyio.Event)


@cache
def _message_adapter(registry_version: int) -> TypeAdapter[CapabilityEvent]:
    # Keyed on the registry version: the adapter snapshots the event kinds registered when it is built.
    return TypeAdapter(CapabilityEvent)


def serialize_message(message: SubAgentMessage) -> bytes:
    """Serialize a sub-agent message to JSON, for a transport that crosses a process boundary."""
    return _message_adapter(event_registry_version()).dump_json(message)


def deserialize_message(data: str | bytes) -> SubAgentMessage:
    """Deserialize a sub-agent message serialized by `serialize_message`.

    Raises:
        ValueError: If `data` is not a sub-agent message.
    """
    event = _message_adapter(event_registry_version()).validate_json(data)
    if not isinstance(event, SubAgentMessage):
        raise ValueError(f'Expected a sub-agent message, got a {event.kind!r} event.')
    return event
