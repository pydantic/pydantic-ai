from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, TypeAlias

from pydantic_ai._utils import format_inlined_text_file, gather, is_text_like_media_type
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
from pydantic_ai.models import KnownModelName, Model, download_item
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage

from .abstract import AbstractCapability
from .durable_operation import durable_operation

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

_File: TypeAlias = ImageUrl | DocumentUrl | VideoUrl | BinaryContent

_MAX_DESCRIPTIONS = 256
"""Descriptions kept per capability instance; the oldest goes first, so a long-lived agent stays bounded."""

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

    A text-like document (plain text, CSV, JSON, XML, YAML) is not described but inlined as it is, the way models
    that read text files do. A provider file reference ([`UploadedFile`][pydantic_ai.messages.UploadedFile]) is left alone.

    A file is described once and the description reused for the steps that follow, so a history that carries the
    same file into every request costs one description. Binary content is reused for as long as the capability
    lives, since the bytes are their own identity; a file given by URL is reused only within the run that fetched
    it, because the same URL can serve different bytes to different callers and stop resolving once a signature
    expires. Run ids come from the caller when one is passed to `run(..., run_id=...)`, so a caller that passes
    the same `run_id` to two runs gets the first run's description of a URL in the second, without the URL being
    fetched again.

    The distinct files a request needs described are described concurrently. Each describing call is a run of its
    own, but its requests, tokens and cost are added to the agent run's [usage][pydantic_ai.usage.RunUsage], and
    the run's [usage limits][pydantic_ai.usage.UsageLimits] are checked before any is sent, as if the describing
    requests were the run's next ones.

    Audio is left as it is: [`supports_audio_input`][pydantic_ai.profiles.ModelProfile.supports_audio_input]
    is about realtime speech history and is not set per model, so it cannot say which models read audio files.
    """

    fallback_model: Model | KnownModelName | str
    """The model that describes the files, with its provider: `'openai:gpt-5.6-sol'`, or a `Model`."""

    _: KW_ONLY

    instructions: str | None = None
    """What to write about each file. The default asks for a description someone who cannot see the file could answer questions from."""

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

        # Every file is found first, so the distinct ones can be described at once rather than one after another.
        descriptions: dict[str, str] = {}
        pending: dict[str, _File] = {}
        for item in _user_content(request_context.messages):
            if _accepted(item, profile):
                continue
            assert isinstance(item, ImageUrl | DocumentUrl | VideoUrl | BinaryContent)
            identity = _identity(item, ctx.run_id)
            if identity in descriptions or identity in pending:
                continue
            if (cached := self._descriptions.get(identity)) is not None:
                descriptions[identity] = cached
            else:
                pending[identity] = item
        if not pending:
            return request_context

        if (usage_limits := ctx.usage_limits) is not None:
            # The run checked its limits for this step's request before this hook, so the describing requests go
            # in front of it: a limit without room for all of them and the request itself stops the run before
            # any is sent.
            requests = sum(1 for item in pending.values() if not is_text_like_media_type(item.media_type))
            if requests:
                usage_limits.check_before_request(ctx.usage + RunUsage(requests=requests))

        results = await gather(*(self._describe(item) for item in pending.values()))
        for (identity, item), (description, usage) in zip(pending.items(), results):
            # usage-attribution: ok, the run's total only; the describing run's own span reports these requests
            ctx.usage.incr(usage)
            descriptions[identity] = description
            if _reuse_key(item, ctx.run_id) is not None:
                if len(self._descriptions) >= _MAX_DESCRIPTIONS:
                    del self._descriptions[next(iter(self._descriptions))]
                self._descriptions[identity] = description

        messages = [_substitute(message, profile, descriptions, ctx.run_id) for message in request_context.messages]
        return replace(request_context, messages=messages)

    @durable_operation(name='describe')
    async def _describe(self, item: _File) -> tuple[str, RunUsage]:
        """The file as text, with the usage it cost.

        A text-like document is its own content, at no cost; anything else is described by `fallback_model`. The
        usage is returned rather than added to the run's here, so it reaches the run whether this runs in place or
        under a durable engine, and the caller adds it.
        """
        from pydantic_ai.agent import Agent

        if isinstance(item, BinaryContent) and is_text_like_media_type(item.media_type):
            return item.data.decode('utf-8'), RunUsage()
        if isinstance(item, DocumentUrl) and is_text_like_media_type(item.media_type):
            return (await download_item(item, data_format='text'))['data'], RunUsage()

        agent: Agent[None, str] = Agent(
            self.fallback_model, output_type=str, instructions=self.instructions or _DEFAULT_INSTRUCTIONS
        )
        result = await agent.run([item])
        return result.output, result.usage


def _user_content(messages: list[ModelMessage]) -> Iterator[UserContent]:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    yield from part.content


def _substitute(
    message: ModelMessage, profile: ModelProfile, descriptions: dict[str, str], run_id: str | None
) -> ModelMessage:
    if not isinstance(message, ModelRequest):
        return message

    def text_for(item: UserContent) -> UserContent:
        if _accepted(item, profile):
            return item
        assert isinstance(item, ImageUrl | DocumentUrl | VideoUrl | BinaryContent)
        # What goes back is prose about the file, so it is text whatever the file was. A text-like document is
        # inlined as its own content instead of described, and that one keeps the media type it came with.
        media_type = item.media_type if is_text_like_media_type(item.media_type) else 'text/plain'
        description = descriptions[_identity(item, run_id)]
        return format_inlined_text_file(description, media_type=media_type, identifier=item.identifier)

    original_parts = list(message.parts)
    parts: list[ModelRequestPart] = [
        replace(part, content=[text_for(item) for item in part.content])
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        else part
        for part in original_parts
    ]
    return replace(message, parts=parts) if parts != original_parts else message


def _identity(item: _File, run_id: str | None) -> str:
    """What makes two items one file within this request: the reuse key, or for a URL with no run, the URL."""
    return _reuse_key(item, run_id) or f'url:{item.url}'


def _reuse_key(item: _File, run_id: str | None) -> str | None:
    """What identifies this file for reuse, or `None` when its description must not be reused.

    Bytes are their own identity, so a description of them holds for anyone who sends the same bytes. A URL is
    not its content: it can serve different bytes to different callers and stop resolving once a signature
    expires, so a description of one is kept only for the run that fetched it, and not at all for a run with no
    id to scope it to.
    """
    if isinstance(item, BinaryContent):
        # The media type is part of the identity, not a label on it: the same bytes are described when they
        # are an image and decoded when they are a CSV, so hashing the bytes alone returns the wrong one.
        return hashlib.sha256(item.media_type.encode() + b'\0' + item.data).hexdigest()
    return f'{run_id}:{item.url}' if run_id is not None else None


def _accepted(item: UserContent, profile: ModelProfile) -> bool:
    """Whether the model takes this item as it is; text and anything but a file is left alone."""
    if isinstance(item, ImageUrl) or (isinstance(item, BinaryContent) and item.is_image):
        return profile.get('supports_image_input', True)
    if isinstance(item, DocumentUrl) or (isinstance(item, BinaryContent) and item.is_document):
        return profile.get('supports_document_input', True)
    if isinstance(item, VideoUrl) or (isinstance(item, BinaryContent) and item.is_video):
        return profile.get('supports_video_input', False)
    return True
