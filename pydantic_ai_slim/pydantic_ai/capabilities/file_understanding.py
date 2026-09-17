from __future__ import annotations

import hashlib
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING

from pydantic_ai._utils import format_inlined_text_file
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    UserContent,
    UserPromptPart,
    VideoUrl,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import AgentDepsT, RunContext

from .abstract import AbstractCapability
from .durable_operation import durable_operation

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
        capabilities=[FileUnderstanding(fallback_model='openai:gpt-5.6-sol')],
    )
    result = agent.run_sync([DocumentUrl('https://example.com/whatever.pdf')])
    ```

    Each file is described once per capability instance; the description is reused across steps and runs.

    Audio is left as it is: [`supports_audio_input`][pydantic_ai.profiles.ModelProfile.supports_audio_input]
    is about realtime speech history and is not set per model, so it cannot say which models read audio files.
    """

    fallback_model: Model | KnownModelName | str
    """The model that describes the files, with its provider: `'openai:gpt-5.6-sol'`, or a `Model`."""

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
        messages = [await self._replace_unsupported_files(message, profile) for message in request_context.messages]
        return replace(request_context, messages=messages)

    async def _replace_unsupported_files(self, message: ModelMessage, profile: ModelProfile) -> ModelMessage:
        if not isinstance(message, ModelRequest):
            return message

        original_parts = list(message.parts)
        parts: list[ModelRequestPart] = []
        for part in original_parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                part = replace(
                    part,
                    content=[await self._describe_if_unsupported(item, profile) for item in part.content],
                )
            parts.append(part)
        return replace(message, parts=parts) if parts != original_parts else message

    async def _describe_if_unsupported(self, item: UserContent, profile: ModelProfile) -> UserContent:
        if _accepted(item, profile):
            return item
        assert isinstance(item, ImageUrl | DocumentUrl | VideoUrl | BinaryContent)
        key = item.url if isinstance(item, ImageUrl | DocumentUrl | VideoUrl) else hashlib.sha256(item.data).hexdigest()
        if (description := self._descriptions.get(key)) is None:
            description = self._descriptions[key] = await self._describe(item)
        return format_inlined_text_file(description, media_type=item.media_type, identifier=item.identifier)

    @durable_operation(name='describe')
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
