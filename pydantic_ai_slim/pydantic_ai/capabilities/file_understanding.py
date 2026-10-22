from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, TypeGuard, TypeVar

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._file_input import SendableFile, inferred_media_type, unsupported_file_input
from pydantic_ai._utils import format_inlined_text_file, gather, is_str_dict, is_text_like_media_type
from pydantic_ai.messages import (
    BinaryContent,
    DocumentUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model, download_item
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage

from .abstract import AbstractCapability
from .durable_operation import durable_operation

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

_T = TypeVar('_T')

_DESCRIPTIONS_ADAPTER = TypeAdapter(dict[str, str])

_FRAMEWORK_METADATA_KEY = '__pydantic_ai__'
"""The namespace in `metadata` reserved for framework state, which the UI adapters never send to or read from a client."""

_NAMESPACE_KEY = 'file_understanding'

_DEFAULT_INSTRUCTIONS = (
    'Describe this file in detail, so that someone who cannot see it can answer questions about it. '
    'Reply with the description only.'
)


@dataclass
class FileUnderstanding(AbstractCapability[AgentDepsT]):
    """Describe the files the agent's model cannot read, with a model that can.

    Before each request, every image, document, audio file or video in a user prompt or a tool result that
    the model's [`ModelProfile`][pydantic_ai.profiles.ModelProfile] says it does not accept is replaced by a
    text description of it, written by `fallback_subagent_model`. The agent's model sees the description in
    the file's place; files it accepts are sent as they are. Without this capability, a file a tool returned
    that the model cannot read is replaced by a note saying so instead.

    ```python
    from pydantic_ai import Agent, DocumentUrl
    from pydantic_ai.capabilities import FileUnderstanding

    agent = Agent(
        'typesafe:jev-latest',
        output_type=bool,
        instructions='Is this document about animals?',
        capabilities=[FileUnderstanding(fallback_subagent_model='openai:gpt-5.6-sol')],
    )
    result = agent.run_sync([DocumentUrl('https://example.com/whatever.pdf')])
    ```

    A text-like document (plain text, CSV, JSON, XML, YAML) is not described but inlined as it is, the way models
    that read text files do. A provider file reference ([`UploadedFile`][pydantic_ai.messages.UploadedFile]) is left alone.

    A file is described once per conversation. A file given by URL is identified by its URL, and bytes by their
    content and media type. The description is recorded in the message history, under the framework's reserved
    `__pydantic_ai__` namespace in the [`metadata`][pydantic_ai.messages.ModelRequest.metadata] of the request
    that holds the file (which is not sent to the model), and later requests use that text instead of describing
    the file again. The downloaded text of a text document given by URL is recorded the same way, so it is not
    downloaded again. A turn that continues from `message_history`, another worker serving the next turn, and a
    durable run replaying its history all see the same description, so earlier requests keep the same content and
    the describer is not paid for twice. The UI adapters leave that namespace out of the history they send to a
    client, so a conversation whose history the client holds describes the file again each turn, and a new
    description can differ, which changes earlier requests and costs the cached prefix.

    The distinct files a request needs described are described concurrently. Each describing call is a run of its
    own, but its requests, tokens and cost are added to the agent run's [usage][pydantic_ai.usage.RunUsage], and
    the run's [usage limits][pydantic_ai.usage.UsageLimits] are checked before any is sent, as if the describing
    requests were the run's next ones.
    """

    fallback_subagent_model: Model | KnownModelName | str
    """The model of the subagent that describes the files, with its provider: `'openai:gpt-5.6-sol'`, or a `Model`.

    Named like `ImageGeneration` and `XSearch`'s `fallback_subagent_model`: a separate agent run, not a
    replacement for the agent's model.
    """

    _: KW_ONLY

    instructions: str | None = None
    """What to write about each file. The default asks for a description someone who cannot see the file could answer questions from."""

    id: str | None = 'file_understanding'
    """One-off: an agent describes files one way, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

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
        recorded = self._recorded_descriptions([*ctx.messages, *request_context.messages])
        descriptions: dict[str, str] = {}
        pending: dict[str, SendableFile] = {}
        unaccepted = False
        for content in _file_content(request_context.messages):
            if (item := _unaccepted_file(content, profile)) is None:
                continue
            unaccepted = True
            if _is_text_bytes(item):
                # Decoded where it is inlined: that costs nothing, so it isn't described or cached.
                continue
            identity = _identity(item)
            if identity in descriptions or identity in pending:
                continue
            if (found := recorded.get(identity)) is not None:
                descriptions[identity] = found
            else:
                pending[identity] = item
        if not unaccepted:
            return request_context

        if pending:
            await self._describe_pending(ctx, pending, descriptions)
            self._record(ctx.messages, {identity: descriptions[identity] for identity in pending})

        messages = [_substitute(message, profile, descriptions) for message in request_context.messages]
        return replace(request_context, messages=messages)

    @property
    def _metadata_key(self) -> str:
        """This capability's key in its `metadata` namespace: one per `id`, as instructions can differ."""
        return self.id or 'default'

    def _recorded_descriptions(self, messages: list[ModelMessage]) -> dict[str, str]:
        """The descriptions recorded in `messages`, by file identity."""
        recorded: dict[str, str] = {}
        for message in messages:
            if isinstance(message, ModelRequest) and message.metadata:
                recorded.update(_str_dict(_own_namespace(message.metadata).get(self._metadata_key)))
        return recorded

    def _record(self, history: list[ModelMessage], described: dict[str, str]) -> None:
        """Record `described` in the run's history, on the request that holds each file.

        `RunContext.messages` is the run's persistent history, so this is what the next request, the next turn
        and a replay read back. A file that is only in this request's messages, not in the history (a history
        processor can add one), is recorded on the latest request instead.
        """
        requests = [i for i, message in enumerate(history) if isinstance(message, ModelRequest)]
        if not requests:
            return
        by_index: dict[int, dict[str, str]] = {}
        for identity, description in described.items():
            holder = next((i for i in requests if identity in _identities(history[i])), requests[-1])
            by_index.setdefault(holder, {})[identity] = description
        for index, new in by_index.items():
            message = history[index]
            assert isinstance(message, ModelRequest)
            metadata = message.metadata or {}
            framework = metadata.get(_FRAMEWORK_METADATA_KEY)
            framework = framework if is_str_dict(framework) else {}
            own = _own_namespace(metadata)
            own = {**own, self._metadata_key: {**_str_dict(own.get(self._metadata_key)), **new}}
            # Other framework keys and the application's own keys are kept as they are.
            metadata = {**metadata, _FRAMEWORK_METADATA_KEY: {**framework, _NAMESPACE_KEY: own}}
            # A new message rather than a changed one, since the caller's `message_history` may hold this object.
            history[index] = replace(message, metadata=metadata)

    async def _describe_pending(
        self, ctx: RunContext[AgentDepsT], pending: dict[str, SendableFile], descriptions: dict[str, str]
    ) -> None:
        """Describe the `pending` files at once into `descriptions`, adding what that costs to the run."""
        if (usage_limits := ctx.usage_limits) is not None:
            # The run checked its limits for this step's request before this hook, so the describing requests go
            # in front of it: a limit without room for all of them and the request itself stops the run before
            # any is sent.
            requests = sum(1 for item in pending.values() if _text_media_type(item) is None)
            if requests:
                usage_limits.check_before_request(ctx.usage + RunUsage(requests=requests))

        results = await gather(*(self._describe(item) for item in pending.values()))
        for identity, (description, usage) in zip(pending, results):
            # usage-attribution: ok, the run's total only; the describing run's own span reports these requests
            ctx.usage.incr(usage)
            descriptions[identity] = description

    @durable_operation(name='describe')
    async def _describe(self, item: SendableFile) -> tuple[str, RunUsage]:
        """The file as text, with the usage it cost.

        A text-like document is its own content, at no cost; anything else is described by `fallback_subagent_model`. The
        usage is returned rather than added to the run's here, so it reaches the run whether this runs in place or
        under a durable engine, and the caller adds it.
        """
        from pydantic_ai.agent import Agent

        if isinstance(item, DocumentUrl) and _text_media_type(item) is not None:
            return (await download_item(item, data_format='text'))['data'], RunUsage()

        agent: Agent[None, str] = Agent(
            self.fallback_subagent_model, output_type=str, instructions=self.instructions or _DEFAULT_INSTRUCTIONS
        )
        result = await agent.run([item])
        return result.output, result.usage


def _file_content(messages: list[ModelMessage]) -> Iterator[object]:
    """The items of every user prompt and of every tool result that carries files."""
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                    yield from part.content
                elif isinstance(part, ToolReturnPart) and part.files:
                    yield from part.content_items()


def _own_namespace(metadata: dict[str, object]) -> dict[str, object]:
    """`FileUnderstanding`'s part of the framework namespace in a request's `metadata`."""
    framework = metadata.get(_FRAMEWORK_METADATA_KEY)
    own = framework.get(_NAMESPACE_KEY) if is_str_dict(framework) else None
    return own if is_str_dict(own) else {}


def _str_dict(value: object) -> dict[str, str]:
    """`value` as recorded descriptions, or none: `metadata` belongs to the application too and holds any JSON."""
    try:
        return _DESCRIPTIONS_ADAPTER.validate_python(value)
    except ValidationError:
        return {}


def _identities(message: ModelMessage) -> set[str]:
    """The identities of the files in `message`."""
    return {_identity(item) for item in _file_content([message]) if isinstance(item, SendableFile)}


def _substitute(message: ModelMessage, profile: ModelProfile, descriptions: dict[str, str]) -> ModelMessage:
    if not isinstance(message, ModelRequest):
        return message

    def text_for(content: _T) -> _T | str:
        if (item := _unaccepted_file(content, profile)) is None:
            return content
        # What goes back is prose about the file, so it is text whatever the file was. A text-like document is
        # inlined as its own content instead of described, and that one keeps the media type it came with.
        media_type = _text_media_type(item) or 'text/plain'
        if _is_text_bytes(item):
            description = item.data.decode('utf-8')
        else:
            description = descriptions[_identity(item)]
        return format_inlined_text_file(description, media_type=media_type, identifier=item.identifier)

    original_parts = list(message.parts)
    parts: list[ModelRequestPart] = []
    for part in original_parts:
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
            part = replace(part, content=[text_for(item) for item in part.content])
        elif isinstance(part, ToolReturnPart) and part.files:
            items = [text_for(item) for item in part.content_items()]
            part = replace(part, content=items if isinstance(part.content, list) else items[0])
        parts.append(part)
    return replace(message, parts=parts) if parts != original_parts else message


def _identity(item: SendableFile) -> str:
    """What makes two items one file: a URL is identified by itself, bytes by their content and media type.

    The media type is part of the identity, not a label on it: the same bytes are described when they are an
    image and decoded when they are a CSV, so hashing the bytes alone returns the wrong one.
    """
    if isinstance(item, BinaryContent):
        return hashlib.sha256(item.media_type.encode() + b'\0' + item.data).hexdigest()
    return f'url:{item.url}'


def _unaccepted_file(item: object, profile: ModelProfile) -> SendableFile | None:
    """The file in `item` the model doesn't take as it is, or `None`; text and anything but a file is left alone."""
    if isinstance(item, SendableFile) and unsupported_file_input(item, profile) is not None:
        return item
    return None


def _text_media_type(item: SendableFile) -> str | None:
    """The media type of a text-like file, or `None`; a URL whose media type can't be inferred counts as not text."""
    media_type = inferred_media_type(item)
    return media_type if media_type is not None and is_text_like_media_type(media_type) else None


def _is_text_bytes(item: SendableFile) -> TypeGuard[BinaryContent]:
    return isinstance(item, BinaryContent) and is_text_like_media_type(item.media_type)
