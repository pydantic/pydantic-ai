from __future__ import annotations

import hashlib
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._utils import format_inlined_text_file
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelRequest,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

_DEFAULT_INSTRUCTIONS = (
    'Describe this file in detail, so that someone who cannot see it can answer questions about it. '
    'Reply with the description only.'
)


@dataclass
class FileUnderstanding(AbstractCapability[AgentDepsT]):
    """Describe the files the agent's model cannot read, with a model that can.

    Before each request, every image, document or video in a user prompt that the model's
    [`ModelProfile`][pydantic_ai.profiles.ModelProfile] says it does not accept is replaced by a text
    description of it, written by `fallback_model`. The agent's model sees the description in the file's
    place; files it accepts are sent as they are.

    ```python
    from pydantic_ai import Agent, DocumentUrl
    from pydantic_ai.capabilities import FileUnderstanding

    agent = Agent(
        'typesafe:jev-latest',
        output_type=bool,
        instructions='Is this document about animals?',
        capabilities=[FileUnderstanding(fallback_model='openai:gpt-5.6')],
    )
    result = agent.run_sync([DocumentUrl('https://example.com/whatever.pdf')])
    ```

    Each file is described once per capability instance; the description is reused across steps and runs.
    """

    fallback_model: Model | KnownModelName | str
    """The model that describes the files, with its provider: `'openai:gpt-5.6'`, or a `Model`."""

    instructions: str | None = None
    """What to write about each file. The default asks for a description someone who cannot see the file could answer questions from."""

    _: KW_ONLY

    id: str | None = 'file_understanding'
    """One-off: an agent describes files one way, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

    _descriptions: dict[str, str] = field(default_factory=lambda: {}, init=False, repr=False)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        try:
            profile = request_context.model.profile
        except NotImplementedError:
            # A `FallbackModel` has no profile of its own; which model answers is not known yet.
            return request_context
        messages = request_context.messages
        for index, message in enumerate(messages):
            if not isinstance(message, ModelRequest):
                continue
            parts = list(message.parts)
            for part_index, part in enumerate(parts):
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    content = [await self._describe_if_unsupported(item, profile) for item in part.content]
                    if content != list(part.content):
                        parts[part_index] = replace(part, content=content)
            if parts != list(message.parts):
                messages[index] = replace(message, parts=parts)
        return request_context

    async def _describe_if_unsupported(self, item: UserContent, profile: ModelProfile) -> UserContent:
        if _accepted(item, profile):
            return item
        assert isinstance(item, ImageUrl | DocumentUrl | VideoUrl | BinaryContent)
        key = item.url if isinstance(item, ImageUrl | DocumentUrl | VideoUrl) else hashlib.sha256(item.data).hexdigest()
        if (description := self._descriptions.get(key)) is None:
            description = self._descriptions[key] = await self._describe(item)
        return format_inlined_text_file(description, media_type=item.media_type, identifier=item.identifier)

    async def _describe(self, item: ImageUrl | DocumentUrl | VideoUrl | BinaryContent) -> str:
        from pydantic_ai.agent import Agent

        agent: Agent[None, str] = Agent(
            self.fallback_model, output_type=str, instructions=self.instructions or _DEFAULT_INSTRUCTIONS
        )
        result = await agent.run([item])
        return result.output


def _accepted(item: UserContent, profile: ModelProfile) -> bool:
    """Whether the model takes this item as it is; text and anything but a file is left alone."""
    if isinstance(item, ImageUrl) or (isinstance(item, BinaryContent) and item.is_image):
        return profile.get('supports_image_input', True)
    if isinstance(item, DocumentUrl) or (isinstance(item, BinaryContent) and item.is_document):
        return profile.get('supports_document_input', True)
    if isinstance(item, VideoUrl) or (isinstance(item, BinaryContent) and item.is_video):
        return profile.get('supports_video_input', False)
    return True
