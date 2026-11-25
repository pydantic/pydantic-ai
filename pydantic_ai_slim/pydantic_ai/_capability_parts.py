"""Rendering [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]s into the parts every model already maps.

A capability part stays a typed part in the history and in the messages handed to
[`Model.prepare_messages`][pydantic_ai.models.Model.prepare_messages], which renders it in its lookalike pass. There
the part's own fields say who wrote its body and which text is the framework's frame, so nothing about it is written
into the rendered parts' metadata. Rendering is a pure function of the stored part: no id, timestamp or metadata of
the part reaches the model, so the same history always renders to the same bytes.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from typing_extensions import TypedDict

from . import _utils
from ._framing import (
    _NOTE_IDENTIFIER,  # pyright: ignore[reportPrivateUsage]
    CAPABILITY_DATA_TAG,
    NO_AUTHORITY,
    LookalikeSource,
    capability_frame_tags,
    frame_tag_lookalike,
    lookalike_tags,
    neutralize_tags,
    resembles_frame,
)
from .exceptions import UserError
from .messages import (
    _FRAME_PLACEHOLDER,  # pyright: ignore[reportPrivateUsage]
    Authority,
    BinaryContent,
    CapabilityInstructionSource,
    CapabilityLoadDeltaPart,
    CapabilityPart,
    CapabilityPartSource,
    CapabilitySource,
    CompactionPart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelSettingsDeltaPart,
    ModelSource,
    OperatorText,
    RetryFeedbackPart,
    RetryPromptPart,  # pyright: ignore[reportDeprecated]
    SpeechPart,
    StructuredContent,
    SystemPromptPart,
    TextContent,
    ToolAvailabilityDeltaPart,
    ToolResultSource,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
    UserSource,
    WorkspaceSource,
    _CapabilityContentPart,  # pyright: ignore[reportPrivateUsage]
    _CapabilityInstructionsPart,  # pyright: ignore[reportPrivateUsage]
    _lower_authority,  # pyright: ignore[reportPrivateUsage]
)


def has_capability_parts(messages: Sequence[ModelMessage]) -> bool:
    """Whether any request in `messages` holds a [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]."""
    return any(_has_capability_parts(message) for message in messages)


_PART_AUTHORITY: dict[type[object], Authority] = {
    SystemPromptPart: 'operator',
    # The framework's own records of the run's state, which the model reads in the application's voice.
    InstructionBaselinePart: 'operator',
    ToolAvailabilityDeltaPart: 'operator',
    CapabilityLoadDeltaPart: 'operator',
    ModelSettingsDeltaPart: 'operator',
    UserPromptPart: 'user',
    SpeechPart: 'user',
    # Tool output is data whatever the tool, and so are the typed returns `ToolSearchReturnPart` and
    # `LoadCapabilityReturnPart`, which subclass `ToolReturnPart`.
    ToolReturnPart: 'none',
    RetryPromptPart: 'none',  # pyright: ignore[reportDeprecated]  # TODO(v3): remove RetryPromptPart
    # Retry feedback quotes the model's own output.
    RetryFeedbackPart: 'none',
    # A summary another model wrote of what it replaces.
    CompactionPart: 'none',
}
"""The fixed authority of each request part kind whose authority doesn't depend on the part."""


def effective_authority(part: ModelRequestPart) -> Authority:
    """The [authority][pydantic_ai.messages.Authority] `part` reaches the model with.

    A pure function of the part: it reads nothing from the run, so every reader of a history, such as a check on what
    may reach the application's voice, gets the same answer.

    - A [`CapabilityPart`][pydantic_ai.messages.CapabilityPart]: the lower of its source's and the one its kind asks
      for. An instruction kind asks for `'operator'`, a data kind for `'none'`. It only ever downgrades.
    - An [`InstructionDeltaPart`][pydantic_ai.messages.InstructionDeltaPart]: the authority stored on it.
    - A system prompt and the framework's records of the run's state: `'operator'`.
    - A user prompt or the user's speech: `'user'`.
    - Tool output, retry feedback and a compaction summary: `'none'`.

    A part of any other kind is `'none'`, so a kind added later carries no authority until it's given one here.
    """
    if isinstance(part, CapabilityPart):
        asked: Authority = 'operator' if part.rendering.voice == 'instruction' else 'none'
        return _lower_authority(asked, part.source.authority)
    if isinstance(part, InstructionDeltaPart):
        return part.authority
    for kind, authority in _PART_AUTHORITY.items():
        if isinstance(part, kind):
            return authority
    return 'none'


def is_downgraded_request_scoped(part: ModelRequestPart) -> bool:
    """Whether `part` is a request-scoped capability part that reaches the model framed on the user channel.

    There it's user content, which no provider clears, so nothing but leaving it out ends its scope.
    """
    return isinstance(part, CapabilityPart) and part.scope == 'request' and effective_authority(part) != 'operator'


def check_operator_text(text: str, *, what: str) -> None:
    """Raise unless `text` can go out in the application's voice: it can't look like a framework frame.

    For fixed text a capability sends in the application's voice, checked when the capability is set up rather than
    mid-run, when [`ctx.enqueue`][pydantic_ai.tools.RunContext.enqueue] checks the part that holds it.

    Raises:
        UserError: If `text` contains something that looks like a framework frame, such as a `<system>` tag.
    """
    if resembles_frame(text):
        raise UserError(
            f"{what} is sent in the application's voice, but contains something that looks like a framework frame "
            '(such as a `<system` tag), which the model could take for a message from the application.'
        )


def operator_voice_lookalike(part: CapabilityPart) -> bool:
    """Whether `part` would say something that looks like a framework frame in the application's voice."""
    content = part.model_content()
    return effective_authority(part) == 'operator' and isinstance(content, str) and resembles_frame(content)


class LookalikeNaming(TypedDict, total=False):
    """The arguments of `lookalike_frame_note` that name a capability part's body and its author."""

    tool_name: str
    tool_call_id: str
    model_name: str
    frame_tag: str


@dataclass(frozen=True)
class RenderedCapabilityPart:
    """What a [`CapabilityPart`][pydantic_ai.messages.CapabilityPart] reaches the model as, within one `prepare_messages` pass.

    Never stored: what the lookalike pass needs to know about the rendered parts travels here, next to them, rather
    than in their metadata.
    """

    parts: list[SystemPromptPart | UserPromptPart]
    """The rendered parts, in order."""

    body: UserPromptPart | None
    """The framed content on the user channel, which `prepare_messages` counts as a frame to announce."""

    note: SystemPromptPart | None
    """The note about the body in the application's voice, which isn't a change to the model's instructions."""

    lookalikes: list[tuple[LookalikeSource, list[str]]]
    """Who wrote text in the body that resembles a framework frame or its own frame, with the tags it opens."""

    naming: LookalikeNaming
    """How a lookalike note about the body names it."""


def render_capability_part(part: CapabilityPart) -> RenderedCapabilityPart:
    """The parts `part` reaches the model as, by its [effective authority][pydantic_ai.messages.Authority].

    - `'operator'`: a system message, in the part's own scope.
    - `'user'`: the content framed on the user channel, behind a note in the application's voice saying the
      application attached it on the user's behalf.
    - `'none'`: the content framed on the user channel, behind its kind's note, which also names who wrote the content
      when that isn't the application.

    A data part's frame is `<pyai:data>` with the attributes `_data_frame_attributes` gives it, and so is a lowered
    capability's instructions', with its kind and capability; any other downgraded instruction part's is its kind's
    name. The body is the content as written: text in it that resembles a framework frame, or opens or closes its own
    frame, gets a lookalike note from `prepare_messages` naming the author its source gives.
    """
    rendering = part.rendering
    content = part.model_content()
    authority = effective_authority(part)
    if authority == 'operator':
        # An instruction part, whose content was checked to be a string when it was created.
        assert isinstance(content, str)
        system = SystemPromptPart(content, timestamp=part.timestamp, scope=part.scope)
        return RenderedCapabilityPart(parts=[system], body=None, note=None, lookalikes=[], naming=LookalikeNaming())
    if isinstance(part, _CapabilityInstructionsPart):
        # A lowered capability's instructions are framework data like any other: `kind` says what they are.
        tag = CAPABILITY_DATA_TAG
        attrs: dict[str, str | None] = {'kind': part.kind, 'capability': _capability_attribute(part)}
    elif rendering.voice == 'instruction':
        tag = part.kind.rpartition('.')[2]
        attrs = {}
    else:
        tag = CAPABILITY_DATA_TAG
        attrs = _data_frame_attributes(part)
    # TODO(#11029): deliver content from a source the operator doesn't control on the tool channel instead, as a
    # framework-inserted read and its result, rebuilding #11837.
    open_tag, close_tag = capability_frame_tags(tag, attrs=attrs)
    items: list[UserContent] = [content] if isinstance(content, str | StructuredContent) else list(content)
    # A multi-line body starts and ends on lines of its own, as a block in a document would.
    multi_line = any(
        '\n' in (item if isinstance(item, str) else item.content)
        for item in items
        if isinstance(item, str | TextContent)
    )
    line_break = '\n' if multi_line else ''
    if is_downgraded_request_scoped(part) and isinstance(content, str):
        # One text, which is how the adapters find a request-scoped tail on the wire to keep their cache breakpoints
        # ahead of it (`_request_scoped_tail_texts`).
        body = UserPromptPart(f'{open_tag}{line_break}{content}{line_break}{close_tag}', timestamp=part.timestamp)
    else:
        # Text goes out as `TextContent`, as in any user prompt with more than one item.
        body = UserPromptPart(
            [
                TextContent(f'{open_tag}{line_break}'),
                *(TextContent(item) if isinstance(item, str) else item for item in items),
                TextContent(f'{line_break}{close_tag}'),
            ],
            timestamp=part.timestamp,
        )
    lookalikes = _body_lookalikes(part, items, tag)
    naming = _lookalike_naming(part.source, tag)
    text = _note(part, authority, _note_frame(tag, attrs, open_tag))
    note = None if text is None else SystemPromptPart(text, timestamp=part.timestamp)
    parts: list[SystemPromptPart | UserPromptPart] = [body] if note is None else [note, body]
    return RenderedCapabilityPart(parts=parts, body=body, note=note, lookalikes=lookalikes, naming=naming)


def _data_frame_attributes(part: CapabilityPart) -> dict[str, str | None]:
    """The attributes of a data part's `<pyai:data>` frame, which say what's inside it.

    User content a capability without authority enqueued is named by the capability, `capability="<id>"`, when its id
    is a plain identifier; every other data part by its `kind="<namespace>.<name>"`, then the attributes its kind
    declares. The kind's own `tag` isn't rendered.
    """
    if isinstance(part, _CapabilityContentPart):
        return {'capability': _capability_attribute(part)}
    return {'kind': part.kind, **part._rendered_attributes()}  # pyright: ignore[reportPrivateUsage]


def _capability_attribute(part: CapabilityPart) -> str | None:
    """The `capability` attribute naming the capability a part is from, or `None` when its id isn't a plain identifier."""
    source = part.source
    capability_id = source.capability_id if isinstance(source, CapabilitySource) else part.capability_id
    return capability_id if capability_id and _NOTE_IDENTIFIER.fullmatch(capability_id) else None


def lowered_instruction_statement(
    change: InstructionDeltaPart, *, restated: bool, timestamp: datetime
) -> list[SystemPromptPart | CapabilityPart]:
    """What a statement of a capability's instructions below `'operator'` reaches the model as.

    `change` is an [`InstructionDeltaPart`][pydantic_ai.messages.InstructionDeltaPart] whose `authority` is below
    `'operator'`, and `restated` says whether an earlier statement of its block is still in effect. Its text goes out
    as a capability part at that authority, rendered by `render_capability_part` like any other: framed on the user
    channel, never in the application's voice. When it replaces or withdraws an earlier statement, a note in the
    application's voice says that one no longer applies first. The authority comes from the stored part alone.
    """
    assert change.authority != 'operator'
    source = CapabilitySource(_statement_capability_id(change), authority=change.authority)
    parts: list[SystemPromptPart | CapabilityPart] = []
    if restated:
        author = _author_phrase(source)
        assert author is not None
        # Saying what replaces the earlier block keeps models from voiding the new block too; a withdrawal has none.
        note = (
            f'What {author} said earlier is replaced by the block below.'
            if change.content is not None
            else f'What {author} said earlier no longer applies.'
        )
        parts.append(SystemPromptPart(note, timestamp=timestamp))
    if change.content is not None:
        # Safe for text the operator didn't write: the part is never stored or enqueued, only rendered, and its source
        # is below operator authority, so it renders as framed data rather than in the application's voice.
        content = OperatorText(change.content)
        parts.append(_CapabilityInstructionsPart(content=content, source=source, timestamp=timestamp))
    return parts


def _statement_capability_id(change: InstructionDeltaPart) -> str:
    """The capability a lowered block's statement is from, as its id `capability:<id>` names it.

    That id is the capability's `id` or its class's name, which the operator's configuration sets, never text a server
    or a tool reports. A statement under any other id names no capability: `_author_phrase` calls it "a capability".
    """
    source = change.id.source if change.id is not None else None
    return source.id if isinstance(source, CapabilityInstructionSource) else _UNNAMED_CAPABILITY


_UNNAMED_CAPABILITY = '<unnamed>'
"""A capability id that isn't a plain identifier, so a note names its capability by kind alone."""


def _note_frame(tag: str, attrs: dict[str, str | None], open_tag: str) -> str:
    """How a note in the application's voice names a part's frame: its opening tag, or the tag with its identity alone.

    An attribute's value can come from content, such as a path in a repository, so the note quotes the opening tag only
    when every value is a plain identifier or path; otherwise it keeps only what identifies the frame (`kind` or
    `capability`, which are plain identifiers), and the other values stay in the frame on the user channel.
    """
    if all(value is None or _NOTE_PATH.fullmatch(value) for value in attrs.values()):
        return open_tag
    identity = {
        name: value
        for name, value in attrs.items()
        if name in _IDENTITY_ATTRIBUTES and (value is None or _NOTE_IDENTIFIER.fullmatch(value))
    }
    return capability_frame_tags(tag, attrs=identity)[0]


_IDENTITY_ATTRIBUTES = frozenset({'kind', 'capability'})
"""The `<pyai:data>` attributes that say what's inside it, which no kind can declare (`check_capability_frame`).

They're still checked as plain identifiers before a note quotes them: an unknown part's `kind` comes from history."""


def _note(part: CapabilityPart, authority: Authority, frame: str) -> str | None:
    """The note in the application's voice that goes with a framed part, or `None` when it has none.

    It qualifies the content and never vouches for it: it says who wrote it, unless that's the application. Without a
    note of the kind's own, it then says the content has no authority (`NO_AUTHORITY`); a kind's own note says what
    the content is for instead.
    """
    if authority == 'user':
        return (
            f"The accompanying {frame} is from the user: the application attached it on the user's behalf, and the "
            "user didn't type it now."
        )
    author = _author_phrase(part.source)
    if part.rendering.voice == 'instruction':
        assert author is not None  # a part with 'operator' authority isn't framed
        return f'The accompanying {frame} {_attribution(part.source, author)}, so {NO_AUTHORITY}'
    if author is None:
        return None if part.rendering.note is None else part.rendering.note.replace(_FRAME_PLACEHOLDER, frame)
    if part.rendering.note is None:
        return f'The accompanying {frame} {_attribution(part.source, author)}, so {NO_AUTHORITY}'
    # A kind's own note says what its content is for, such as advice to follow, which "no authority" would contradict.
    note = part.rendering.note.replace(_FRAME_PLACEHOLDER, frame)
    return f'{note} The {frame} {_attribution(part.source, author)}.'


def _attribution(source: CapabilityPartSource, author: str) -> str:
    """Who the content of a framed part is from, as the predicate of a note's sentence about its frame.

    A tool's content is credited to what the tool returned rather than to the tool, the way the lookalike note for a
    tool result words it: a tool fetches pages and documents others wrote, and "the tool wrote it" led models to
    credit the tool with a document's text.
    """
    if isinstance(source, ToolResultSource):
        return (
            f'is part of what {author} returned (such as a page or document it fetched); neither the application nor '
            'the user wrote it'
        )
    from_user = isinstance(source, UserSource) or (isinstance(source, WorkspaceSource) and source.authority == 'user')
    others = 'the application' if from_user else 'the user or the application'
    return f'was written by {author}, not by {others}'


_NOTE_PATH: re.Pattern[str] = re.compile(r'[A-Za-z0-9_./:-]{1,256}')
"""A workspace path a note may quote."""

_NOTE_AUTHOR: re.Pattern[str] = re.compile(
    r'(?:[A-Za-z0-9_.:-]+|`[A-Za-z0-9_.:-]+`)(?: (?:[A-Za-z0-9_.:-]+|`[A-Za-z0-9_.:-]+`))*'
)
"""A `ModelSource.author` a note may name, such as `'trajectory judge'` or a sub-agent's name in backticks.

Words of identifier characters, each optionally in backticks.
"""


def _author_phrase(source: CapabilityPartSource) -> str | None:
    """Who wrote content from `source`, as a note names them, or `None` when that's the application or the operator.

    A note speaks in the application's voice, and a path, tool name, model or capability id comes from history or from
    code that may have taken it from content. Only a value that's a plain identifier (a path of identifier characters,
    a model's name in words of them) is named; anything else would carry that text into the application's voice, so the
    note names the kind of author alone.
    """
    if source.authority == 'operator':
        return None
    if isinstance(source, WorkspaceSource):
        return _workspace_author_phrase(source)
    if isinstance(source, ToolResultSource):
        return f'the `{source.tool_name}` tool' if _NOTE_IDENTIFIER.fullmatch(source.tool_name) else 'a tool'
    if isinstance(source, ModelSource):
        return f'the {source.author}' if _note_author(source.author) else 'another model'
    if isinstance(source, CapabilitySource):
        if _NOTE_IDENTIFIER.fullmatch(source.capability_id):
            return f'the `{source.capability_id}` capability'
        return 'a capability'
    return 'the user'


def _workspace_author_phrase(source: WorkspaceSource) -> str:
    """Who wrote a workspace file, as a note names them: the user for the user's own files, such as their uploads."""
    quotable = _NOTE_PATH.fullmatch(source.path) is not None
    if source.authority == 'user':
        return f'the user, in `{source.path}`' if quotable else 'the user, in a file'
    return f'the author of `{source.path}`' if quotable else 'the author of a workspace file'


def _note_author(author: str) -> bool:
    return len(author) <= 128 and _NOTE_AUTHOR.fullmatch(author) is not None


def _lookalike_naming(source: CapabilityPartSource, tag: str) -> LookalikeNaming:
    naming: LookalikeNaming = {'frame_tag': tag}
    if isinstance(source, ToolResultSource):
        naming['tool_name'] = source.tool_name
        if source.tool_call_id:
            naming['tool_call_id'] = source.tool_call_id
    elif isinstance(source, ModelSource):
        if _note_author(source.author):
            naming['model_name'] = source.author
    elif isinstance(source, CapabilitySource) and _NOTE_IDENTIFIER.fullmatch(source.capability_id):
        naming['model_name'] = f'`{source.capability_id}` capability'
    return naming


def render_capability_parts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """`messages` with every capability part rendered in place, with no lookalike notes, or `messages` itself when there's none.

    For the paths with no lookalike pass: a second model reading a transcript, and a realtime session's seeded history.
    """
    if not has_capability_parts(messages):
        return messages
    return [
        replace(message, parts=_render_parts(message.parts))
        if isinstance(message, ModelRequest) and _has_capability_parts(message)
        else message
        for message in messages
    ]


def render_capability_parts_for_live_channel(messages: list[ModelMessage]) -> list[ModelMessage]:
    """`messages` rendered as [`render_capability_parts`][pydantic_ai._capability_parts.render_capability_parts] does, for a realtime session.

    The live channel puts the rendered text in the conversation's own turn, where no lookalike note can say who wrote
    a framing tag in a body. So in the body of a part with less than operator authority, every framing tag (`<system>`
    included) is neutralized as `&lt;`, as the harness's frames did before they became capability parts. The stored
    parts are left as they are.

    TODO(#12507): give the live channel lookalike notes instead, and send these bodies as written.
    """
    if not has_capability_parts(messages):
        return messages
    return render_capability_parts(
        [
            replace(message, parts=[_neutralized(part) for part in message.parts])
            if isinstance(message, ModelRequest)
            else message
            for message in messages
        ]
    )


def _neutralized(part: ModelRequestPart) -> ModelRequestPart:
    """`part` with the framing tags in its body neutralized, if it's a capability part below operator authority."""
    if not isinstance(part, CapabilityPart) or effective_authority(part) == 'operator':
        return part
    content = part.model_content()
    if isinstance(content, str):
        return replace(part, content=neutralize_tags(content))
    if isinstance(content, StructuredContent):
        return part
    items = [
        neutralize_tags(item)
        if isinstance(item, str)
        else replace(item, content=neutralize_tags(item.content))
        if isinstance(item, TextContent)
        else item
        for item in content
    ]
    return replace(part, content=items)


def _has_capability_parts(message: ModelMessage) -> bool:
    return isinstance(message, ModelRequest) and any(isinstance(part, CapabilityPart) for part in message.parts)


def _render_parts(parts: Sequence[ModelRequestPart]) -> list[ModelRequestPart]:
    rendered: list[ModelRequestPart] = []
    for part in parts:
        if isinstance(part, CapabilityPart):
            rendered.extend(render_capability_part(part).parts)
        else:
            rendered.append(part)
    return rendered


def _body_lookalikes(
    part: CapabilityPart, items: Sequence[UserContent], tag: str
) -> list[tuple[LookalikeSource, list[str]]]:
    """Who wrote text in a framed part's body that resembles a framework frame or its own `tag`, with the tags it opens.

    The body's text is the source's author's; text from an `OperatorSource` is the application's own and gets no note.
    A text file in the body is its author's, as in a user prompt.
    """
    author = part.source._author  # pyright: ignore[reportPrivateUsage]
    found: dict[LookalikeSource, list[str]] = {}
    for item in items:
        if author != 'application' and isinstance(item, str | TextContent):
            _record_body_lookalikes(found, author, item if isinstance(item, str) else item.content, tag)
    for item in items:
        if isinstance(item, BinaryContent) and _utils.is_text_like_media_type(item.media_type):
            _record_body_lookalikes(found, 'file', item.data.decode(errors='replace'), tag)
    return list(found.items())


def _record_body_lookalikes(
    found: dict[LookalikeSource, list[str]], source: LookalikeSource, text: str, tag: str
) -> None:
    """Add the tags `text` opens that resemble a framework frame or the part's own `tag` to `source`'s entry in `found`, once each."""
    tags = lookalike_tags(text)
    if tag not in tags and frame_tag_lookalike(text, tag):
        tags.append(tag)
    if tags:
        found.setdefault(source, []).extend(t for t in tags if t not in found.get(source, ()))
