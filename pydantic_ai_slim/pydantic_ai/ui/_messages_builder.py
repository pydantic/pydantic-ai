from dataclasses import dataclass, field
from typing import cast

from pydantic_ai._utils import get_union_args
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, ModelResponsePart


@dataclass
class MessagesBuilder:
    """Helper class to build Pydantic AI messages from request/response parts."""

    messages: list[ModelMessage] = field(default_factory=list)

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
