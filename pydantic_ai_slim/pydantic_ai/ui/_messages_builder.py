from dataclasses import dataclass, field
from typing import cast

from pydantic_ai._utils import get_union_args
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart


@dataclass(frozen=True)
class BuilderCheckpoint:
    """Opaque snapshot of `MessagesBuilder` state, used to query what `add()` touched after the snapshot.

    Intended as an in-process correlation token: `last_message` holds a live `ModelMessage`
    reference whose identity matters for the matching `last_modified` lookup, so checkpoints
    are not meaningful across pickle/JSON roundtrips. Callers must also avoid mutating
    `last_message.parts` in place between snapshot and query — `MessagesBuilder.add` reassigns
    the list rather than mutating it, but external in-place edits would silently invalidate
    `last_message_part_count`.
    """

    message_count: int
    last_message: ModelMessage | None
    last_message_part_count: int


@dataclass
class MessagesBuilder:
    """Helper class to build Pydantic AI messages from request/response parts."""

    messages: list[ModelMessage] = field(default_factory=list[ModelMessage])

    def add(self, part: ModelRequestPart | ModelResponsePart) -> None:
        """Add a new part, creating a new request or response message if necessary."""
        last_message = self.messages[-1] if self.messages else None
        if isinstance(part, get_union_args(ModelRequestPart)):
            part = cast(ModelRequestPart, part)
            if isinstance(last_message, ModelRequest):
                last_message.parts = [*last_message.parts, part]
            else:
                self.messages.append(ModelRequest(parts=[part]))
        else:
            part = cast(ModelResponsePart, part)
            if isinstance(last_message, ModelResponse):
                last_message.parts = [*last_message.parts, part]
            else:
                self.messages.append(ModelResponse(parts=[part]))

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
        candidates: list[ModelMessage] = []
        prev_last = checkpoint.last_message
        if prev_last is not None and len(prev_last.parts) > checkpoint.last_message_part_count:
            candidates.append(prev_last)
        candidates.extend(self.messages[checkpoint.message_count :])
        return next((message for message in candidates if isinstance(message, of_type)), None)
