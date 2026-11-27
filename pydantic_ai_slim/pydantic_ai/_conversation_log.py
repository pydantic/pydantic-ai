"""A run's conversation: the messages it was given, as given, then every message it added, as it added them."""

from __future__ import annotations as _annotations

import dataclasses
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse

if TYPE_CHECKING:
    from ._run_context import RunContext


@dataclasses.dataclass
class ConversationLog:
    """The conversation a run records, which its result's `conversation` holds.

    The run's message history is what the model sees: the run tidies it before the first request, and a capability can
    trim or rewrite it. None of that changes this log. The run writes to it when it adds a message, and it takes three
    things from the history, each matched by object identity, never by content:

    - messages appended after the log's last message, like a capability's `ctx.messages.append(note)`;
    - a copy of an unsaved request that keeps every part of it and adds more, which replaces the request;
    - a new message holding a `CompactionPart`, a summary, which goes before the next logged message the history keeps.
    """

    messages: list[ModelMessage] | None = None
    """The history the run was given, then every message it added. `None` when the history is the conversation."""

    run_start: int = 0
    """Where the messages the run added start in `messages`."""

    stored: int | None = None
    """How many of the first messages the conversation's store has, or `None` if that isn't known."""

    checked: bool = True
    """Whether the stored messages are still the ones the store has, so it can append after them without comparing.

    `False` for a conversation that didn't say so, like one that was serialized, and once the run changed a stored
    message, so the store compares them and branches where they differ.
    """

    resent: ModelRequest | None = None
    """The request the run sent again because it had no new prompt: tidying merged it from the unsaved requests the
    run was given, and the conversation records it in their place."""

    continued: ModelResponse | None = None
    """The suspended response the run is continuing, which the response it completes it with replaces."""

    step: ModelRequest | None = None
    """The logged request of the step the run is preparing, which what the run records for the step goes on."""

    end: ModelMessage | None = None
    """The message in the history that stands for the end of the log, when tidying replaced the log's last message
    there, so messages appended after it are still the log's."""

    final: list[ModelMessage] | None = None
    """The history as the run ended, so the result can tell edits made to it afterwards."""

    def append(self, history: list[ModelMessage], *messages: ModelMessage) -> None:
        """Append `messages` to `history`, the run's message history, and log them."""
        logged = self._logged(history)
        history.extend(messages)
        for message in messages:
            if isinstance(message, ModelResponse) and self.continued is not None:
                continued, self.continued = self.continued, None
                # Tidying merges only responses no provider sends, so the suspended one is logged as given.
                if self.replace(continued, message):  # pragma: no branch
                    continue
            logged.append(message)
            self.step = message if isinstance(message, ModelRequest) else None

    def logged(self) -> int:
        """How many messages are logged."""
        return len(self.messages or ())

    def resend(self, history: list[ModelMessage], request: ModelRequest) -> None:
        """Append `request` to `history` as the request a run without a new prompt sends again, and log it as that."""
        self.append(history, request)
        self.resent = request

    def replace(self, message: ModelMessage, replacement: ModelMessage | None) -> bool:
        """Log that the run replaced `message` with `replacement`, or removed it when that's `None`.

        Returns whether `message` was logged. Changing a stored message makes the store compare the conversation.
        """
        logged = self.messages or []
        index = _index(logged, message)
        if index is None:
            return False
        if self.stored is not None and index < self.stored:
            self.checked = False
        if replacement is None:
            del logged[index]
            if index < self.run_start:
                self.run_start -= 1
            if self.stored is not None and index < self.stored:
                self.stored -= 1
        else:
            logged[index] = replacement
        for name in ('resent', 'step', 'continued', 'end'):
            if getattr(self, name) is message:
                setattr(self, name, replacement)
        return True

    def recording_targets(self, targets: list[ModelRequest], history: Sequence[ModelMessage]) -> list[ModelRequest]:
        """`targets`, the requests the run records what it sent on, with the step's logged request if `history` lost it.

        The run records instructions and deltas on the requests the model sees. A capability may have replaced the
        step's request in `history` with a copy, so the log's own request is recorded on too.
        """
        step = self.step
        if step is not None and not any(message is step for message in [*history, *targets]):
            targets = [*targets, step]
        if self.stored is not None and any(
            (index := _index(self.messages or [], target)) is not None and index < self.stored for target in targets
        ):
            self.checked = False
        return targets

    def shaped(self, history: Sequence[ModelMessage]) -> None:
        """Note `history`, as the run shaped it for the model, so what's appended after it is still logged."""
        logged = self.messages or []
        last = history[-1] if history else None
        self.end = last if not logged or last is not logged[-1] else None

    def ended(self, history: list[ModelMessage]) -> None:
        """Take what code other than the run added to `history`, as the run ended, and note `history` as it ended."""
        if self.final is not None:
            return
        self._start(history)
        self.adopt(history)
        self.final = list(history)

    def saved(self, messages: list[ModelMessage]) -> None:
        """Log that a store saved `messages` as the conversation, so all of them are stored."""
        self.messages = list(messages)
        self.run_start = min(self.run_start, len(messages))
        self.stored = len(messages)
        self.checked = True
        self.resent = self.continued = self.step = self.end = None

    def conversation(self, history: Sequence[ModelMessage]) -> tuple[list[ModelMessage], int | None, bool]:
        """The conversation for `history`, the run's message history, with how many of its messages are stored.

        Also returns whether the stored ones are checked: see `checked`. After the run ended, messages appended to its
        history are added. Any other edit makes the conversation the edited history, with none known to be stored, when
        it was the history the run ended with, and leaves the conversation as the run ended otherwise.
        """
        if self.messages is None:
            return list(history), None, False
        final = self.final
        if final is None:
            log = dataclasses.replace(self, messages=list(self.messages))
            log.adopt(list(history))
            return log._resolved(), log.stored, log.checked
        resolved = self._resolved()
        if len(history) >= len(final) and all(map(_same, history, final)):
            logged = {id(message) for message in resolved}
            added = [message for message in history[len(final) :] if id(message) not in logged]
            return [*resolved, *added], self.stored, self.checked
        if len(resolved) == len(final) and all(map(_same, resolved, final)):
            return list(history), None, False
        return resolved, self.stored, self.checked

    def settled(self, history: list[ModelMessage], start: int, logged_from: int) -> ConversationLog:
        """This log for the copy of the history a streamed result keeps, which holds the messages from `start` on itself.

        The stream records its own copy of the response it streamed, which the run logged as the first response after
        `logged_from`, and adds the request with the tool returns, which only it has. Content queued at a deferred
        pause is in both, in the copy's order.
        """
        assert self.messages is not None, 'A run logs the history it was given.'
        resolved = self._resolved()
        streamed = history[start] if start < len(history) else None
        committed = next((m for m in self.messages[logged_from:] if isinstance(m, ModelResponse)), None)
        if isinstance(streamed, ModelResponse) and (index := _index(resolved, committed)) is not None:
            resolved[index] = streamed
        added = [message for message in history[start:] if isinstance(message, ModelRequest)]
        copied = {id(message) for message in added}
        messages = [*(message for message in resolved if id(message) not in copied), *added]
        return ConversationLog(messages, self.run_start, self.stored, self.checked, final=list(history))

    def adopt(self, history: list[ModelMessage]) -> None:
        """Log what code other than the run added to `history` since the run last looked: see the class docstring.

        Anything else done to the history, like dropping, reordering or replacing messages with edited copies, only
        changes what the model sees.
        """
        logged = self.messages
        if logged is None or not history:
            return
        self._adopt_copies(history)
        logged_ids = {id(message) for message in logged}
        tail = self._tail(history)
        summaries = self._adopt_summaries(history[: len(history) if tail is None else tail], logged_ids)
        appended = [
            *summaries,
            *(m for m in history[len(history) if tail is None else tail :] if id(m) not in logged_ids),
        ]
        if appended:
            logged.extend(appended)
            self.end = None

    def _tail(self, history: list[ModelMessage]) -> int | None:
        """Where the messages appended after the log's last one, or what stands for it, start: see `end`."""
        logged = self.messages
        if not logged:
            return 0
        tail = _last_index(history, logged[-1])
        if tail is None and self.end is not None:
            tail = _last_index(history, self.end)
        return None if tail is None else tail + 1

    def _adopt_summaries(self, history: list[ModelMessage], logged_ids: set[int]) -> list[ModelMessage]:
        """Log the new summaries in `history` before the next logged message it keeps, and return the ones after all.

        Never among the stored messages or, when that isn't known, the given ones, as a store only adds after those.
        """
        logged = self.messages
        assert logged is not None
        logged_parts = {id(part) for message in logged for part in message.parts}
        floor = self.stored if self.stored is not None else self.run_start
        summaries: list[ModelMessage] = []
        for message in history:
            if id(message) not in logged_ids:
                if _is_new_summary(message, logged_parts):
                    summaries.append(message)
                    logged_ids.add(id(message))
                continue
            if not summaries:
                continue
            index = _index(logged, message)
            assert index is not None
            position = max(index, floor)
            logged[position:position] = summaries
            if position < self.run_start:
                self.run_start += len(summaries)
            summaries = []
        return summaries

    def _start(self, history: list[ModelMessage]) -> list[ModelMessage]:
        """The log, started from `history` if nothing logged it yet, like a run context made outside a run."""
        if self.messages is None:
            self.messages = list(history)
            self.run_start = len(history)
        return self.messages

    def _logged(self, history: list[ModelMessage]) -> list[ModelMessage]:
        logged = self._start(history)
        self.adopt(history)
        self.end = None
        return logged

    def _adopt_copies(self, history: list[ModelMessage]) -> None:
        """Put a copy of an unsaved request that keeps all of its parts and adds others in its place.

        Like `ReinjectSystemPrompt` putting the system prompt at the head of a new turn. Matched by part identity, so a
        copy that changed or dropped a part only changes what the model sees.
        """
        logged = self.messages
        assert logged is not None
        logged_ids = {id(message) for message in logged}
        # A later message owns a part it shares with an earlier one, as the request sent again does.
        owners = {id(part): index for index, message in enumerate(logged) for part in message.parts}
        for message in history:
            if id(message) in logged_ids or not isinstance(message, ModelRequest):
                continue
            indexes = {owners[id(part)] for part in message.parts if id(part) in owners}
            if len(indexes) != 1:
                continue
            (index,) = indexes
            original = logged[index]
            if index >= (self.stored or 0) and _extends(message, original):
                self.replace(original, message)
                logged_ids.add(id(message))

    def _resolved(self) -> list[ModelMessage]:
        """The logged messages, with the request the run sent again recorded on the unsaved requests it was made from.

        Tidying merges consecutive requests by moving their parts, so each part of the request sent again is one of a
        given request's or one the run or a capability added since: tidying adds a result for a call nothing answered
        only to the request after an earlier response, never to these. Each given request that isn't stored gets its
        own parts back and the fields the run filled in, and the last one the parts added since. A given request
        tidying dropped a part of keeps its parts as given. The request the run sent again isn't recorded itself, nor
        is anything on it a stored request already has.
        """
        assert self.messages is not None
        messages = list(self.messages)
        resent = self.resent
        if resent is None or (position := _index(messages, resent)) is None:
            return messages
        del messages[position]
        given = self.run_start
        first = given
        while first > 0 and isinstance(messages[first - 1], ModelRequest):
            first -= 1
        # A request the store has keeps its parts, so what the run sent with it goes on the unsaved ones.
        unsaved_from = max(first, self.stored or 0)
        owners = {id(part): index for index in range(first, given) for part in messages[index].parts}
        own: dict[int, list[Any]] = {index: [] for index in range(unsaved_from, given)}
        for part in resent.parts:
            index = owners.get(id(part), given - 1)
            if index >= unsaved_from:
                own[index].append(part)
        for index, parts in own.items():
            request = messages[index]
            assert isinstance(request, ModelRequest)
            messages[index] = _restored(request, parts, resent, owners)
        return messages


def _is_new_summary(message: ModelMessage, logged_parts: set[int]) -> bool:
    """Whether `message` holds a `CompactionPart` and none of the logged messages' parts."""
    return any(isinstance(part, CompactionPart) for part in message.parts) and not any(
        id(part) in logged_parts for part in message.parts
    )


def _extends(copy: ModelRequest, original: ModelMessage) -> bool:
    """Whether `copy` holds every part of `original`, a request, and more."""
    parts = {id(part) for part in copy.parts}
    return (
        isinstance(original, ModelRequest)
        and len(copy.parts) > len(original.parts)
        and all(id(part) in parts for part in original.parts)
    )


def _restored(request: ModelRequest, parts: list[Any], resent: ModelRequest, owners: dict[int, int]) -> ModelRequest:
    """`request`, a given request the run sent again as `resent`, with `parts` back and the fields the run filled in.

    A request tidying dropped a part of keeps its parts as given, with the ones added since.
    """
    kept = {id(part) for part in parts}
    if not all(id(part) in kept for part in request.parts):
        parts = [*request.parts, *(part for part in parts if id(part) not in owners)]
    return dataclasses.replace(
        request,
        parts=parts,
        timestamp=request.timestamp or resent.timestamp,
        instructions=request.instructions or resent.instructions,
        run_id=request.run_id or resent.run_id,
        conversation_id=request.conversation_id or resent.conversation_id,
    )


def _same(a: object, b: object) -> bool:
    return a is b


def _last_index(messages: Sequence[ModelMessage], message: object) -> int | None:
    return next((index for index in range(len(messages) - 1, -1, -1) if messages[index] is message), None)


def _index(messages: Sequence[ModelMessage], message: object) -> int | None:
    return next((index for index, logged in enumerate(messages) if logged is message), None)


def append_to_history(ctx: RunContext[Any], *messages: ModelMessage) -> None:
    """Append `messages` to the history of the run `ctx` belongs to, logging them as messages the run added."""
    ctx._conversation_log.append(ctx.messages, *messages)  # pyright: ignore[reportPrivateUsage]
