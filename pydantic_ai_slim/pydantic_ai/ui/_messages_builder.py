from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TypeVar, cast

from pydantic_ai._utils import get_union_args
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
)

_PartT = TypeVar('_PartT', bound=ModelRequestPart | ModelResponsePart)


@dataclass(frozen=True)
class BuilderCheckpoint:
    """Opaque snapshot of `MessagesBuilder` state, used to query what `add()` touched after the snapshot.

    Intended as an in-process correlation token: `last_message` holds a live `ModelMessage`
    reference whose identity matters for the matching `last_modified` lookup, so checkpoints
    are not meaningful across pickle/JSON roundtrips. `last_message_part_count` is compared
    against the length of `last_message.parts` at query time, which `MessagesBuilder.add` only
    ever grows; callers must avoid removing parts from it between snapshot and query.
    """

    message_count: int
    last_message: ModelMessage | None
    last_message_part_count: int


@dataclass
class MessagesBuilder:
    """Helper class to build Pydantic AI messages from request/response parts."""

    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])

    _sealed_tail: ModelMessage | None = field(default=None, init=False, repr=False, compare=False)
    """The last message `_append_whole` added, which `add` never extends."""

    _own_parts: Sequence[ModelRequestPart | ModelResponsePart] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    """The parts list `add` last extended, which it made itself and so appends to in place."""

    def add(self, part: ModelRequestPart | ModelResponsePart) -> None:
        """Add a new part, creating a new request or response message if necessary."""
        last_message = self.messages[-1] if self.messages else None
        if last_message is not None and last_message is self._sealed_tail:
            last_message = None
        if isinstance(part, CompactionPart):
            # A provider's compaction part belongs to its response; one no provider produced, to a request.
            to_request = part.provider_name is None and part.id is None and part.provider_details is None
        else:
            to_request = isinstance(part, get_union_args(ModelRequestPart))
        if to_request:
            part = cast(ModelRequestPart, part)
            if isinstance(last_message, ModelRequest):
                last_message.parts = self._extended(last_message.parts, part)
            else:
                self._own_parts = request_parts = [part]
                self.messages.append(ModelRequest(parts=request_parts))
        else:
            part = cast(ModelResponsePart, part)
            if isinstance(last_message, ModelResponse):
                last_message.parts = self._extended(last_message.parts, part)
            else:
                self._own_parts = response_parts = [part]
                self.messages.append(ModelResponse(parts=response_parts))

    def _extended(self, parts: Sequence[_PartT], part: _PartT) -> list[_PartT]:
        """`parts` with `part` appended: in place on the list made here last time, else on a copy, made once.

        A list that came from elsewhere is left as it is, and extending a message one part at a time stays linear.
        """
        if parts is not self._own_parts or not isinstance(parts, list):
            parts = list(parts)
            self._own_parts = parts
        parts.append(part)
        return parts

    def _append_whole(self, messages: Sequence[ModelMessage]) -> None:
        """Append complete messages, which later `add` calls start a new message after rather than extend."""
        if messages:
            self.messages.extend(messages)
            self._sealed_tail = messages[-1]

    def checkpoint(self) -> BuilderCheckpoint:
        """Snapshot the current builder state. Pair with [`last_modified`][pydantic_ai.ui.MessagesBuilder.last_modified]."""
        last = self.messages[-1] if self.messages else None
        return BuilderCheckpoint(
            message_count=len(self.messages),
            last_message=last,
            last_message_part_count=len(last.parts) if last else 0,
        )

    def last_modified(
        self,
        checkpoint: BuilderCheckpoint,
        *,
        of_type: type[ModelRequest] | type[ModelResponse],
    ) -> ModelMessage | None:
        """Find the most recently created or extended `ModelMessage` of `of_type` since `checkpoint`.

        A single round of `add()` calls can either grow the previous tail's parts list (if the new
        part matches the tail's type) or append fresh messages (which can be more than one when
        e.g. tool-return parts follow a response). Callers that need to attribute side metadata to
        a logical "the message I just built" use this rather than re-deriving it from `messages`.
        """
        return next((message for message in self._modified_since(checkpoint) if isinstance(message, of_type)), None)

    def _modified_since(self, checkpoint: BuilderCheckpoint) -> list[ModelMessage]:
        """The messages created or extended since `checkpoint`, in order."""
        candidates: list[ModelMessage] = []
        prev_last = checkpoint.last_message
        if prev_last is not None and len(prev_last.parts) > checkpoint.last_message_part_count:
            candidates.append(prev_last)
        candidates.extend(self.messages[checkpoint.message_count :])
        return candidates
