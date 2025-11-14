"""This module provides functionality to manage and update parts of a model's streamed response.

The manager tracks which parts (in particular, text and tool calls) correspond to which
vendor-specific identifiers (e.g., `index`, `tool_call_id`, etc., as appropriate for a given model),
and produces Pydantic AI-format events as appropriate for consumers of the streaming APIs.

The "vendor-specific identifiers" to use depend on the semantics of the responses of the responses from the vendor,
and are tightly coupled to the specific model being used, and the Pydantic AI Model subclass implementation.

This `ModelResponsePartsManager` is used in each of the subclasses of `StreamedResponse` as a way to consolidate
event-emitting logic.
"""

from __future__ import annotations as _annotations

from collections.abc import Generator, Hashable
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, model_validator

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

from ._utils import generate_tool_call_id as _generate_tool_call_id

VendorId = Hashable
"""
Type alias for a vendor part identifier, which can be any hashable type (e.g., a string, UUID, etc.)
"""

ThinkingTags = tuple[str, str]

ManagedPart = ModelResponsePart | ToolCallPartDelta
"""
A union of types that are managed by the ModelResponsePartsManager.
Because many vendors have streaming APIs that may produce not-fully-formed tool calls,
this includes ToolCallPartDelta's in addition to the more fully-formed ModelResponsePart's.
"""

PartT = TypeVar('PartT', bound=ModelResponsePart)


@dataclass
class _ExistingPart(Generic[PartT]):
    part: PartT
    index: int
    found_by: Literal['vendor_part_id', 'latest_part']


def suffix_prefix_overlap(s1: str, s2: str) -> int:
    """Return the length of the longest suffix of s1 that is a prefix of s2."""
    n = min(len(s1), len(s2))
    for k in range(n, 0, -1):
        if s1.endswith(s2[:k]):
            return k
    return 0


class PartialThinkingTag(BaseModel, validate_assignment=True):
    respective_tag: str
    buffer: str = ''
    previous_part_index: int
    vendor_part_id: VendorId | None = None

    @model_validator(mode='after')
    def validate_buffer(self) -> PartialThinkingTag:
        if not self.respective_tag.startswith(self.buffer):  # pragma: no cover
            raise ValueError(f"Buffer '{self.buffer}' does not match the start of tag '{self.respective_tag}'")
        return self

    @property
    def expected_next(self) -> str:
        return self.respective_tag[len(self.buffer) :]

    @property
    def is_complete(self) -> bool:
        return self.buffer == self.respective_tag

    @property
    def has_previous_part(self) -> bool:
        return self.previous_part_index >= 0


@dataclass
class StartTagValidation:
    flushed_buffer: str = ''
    """Any buffered content that was flushed because the tag was invalid."""

    thinking_content: str = ''
    """Any content following the valid opening tag."""


class PartialStartTag(PartialThinkingTag):
    def validate_new_content(self, new_content: str) -> StartTagValidation:
        combined = self.buffer + new_content
        if combined.startswith(self.respective_tag):
            # combined = '<think>content'
            self.buffer = combined[: len(self.respective_tag)]
            thinking_content = combined[len(self.respective_tag) :]
            return StartTagValidation(thinking_content=thinking_content)
        elif self.respective_tag.startswith(combined):
            # combined = '<thi' - buffer it
            self.buffer = combined
            return StartTagValidation()
        elif self.respective_tag.startswith(new_content):
            # new_content = '<thi' or '<think>' - buffer new_content, flush old buffer - handles stutter
            flushed_buffer = self.buffer
            self.buffer = new_content
            return StartTagValidation(flushed_buffer=flushed_buffer)
        elif new_content.startswith(self.respective_tag):
            # new_content = '<think>content'
            flushed_buffer = self.buffer
            self.buffer = new_content[: len(self.respective_tag)]
            thinking_content = new_content[len(self.respective_tag) :]
            return StartTagValidation(flushed_buffer=flushed_buffer, thinking_content=thinking_content)
        else:
            self.buffer = ''
            return StartTagValidation(flushed_buffer=combined)


@dataclass
class EndTagValidation:
    content_before_closed: str = ''
    """Any content before the tag was closed."""

    content_after_closed: str = ''
    """Any content remaining after the tag was closed."""


class PartialEndTag(PartialThinkingTag):
    """A partial end tag that tracks the closing of a thinking part.

    A PartialEndTag is created when an opening thinking tag completes (e.g., after seeing `<think>`).
    PartialEndTags are tracked in `_partial_tags_list` by their vendor_part_id and previous_part_index fields.

    The PartialEndTag.previous_part_index initially inherits from the preceding PartialStartTag,
    which may be -1 (if `<think>` was first content) or a TextPart index.

    If content follows the opening tag, a ThinkingPart is created and previous_part_index is updated to point to it.

    Lifecycle:
    - Empty thinking (`<think></think>`): PartialEndTag removed, no ThinkingPart created, no event emitted
    - Normal completion: PartialEndTag removed when closing tag completes
    - Stream ends with buffer: Buffered content (e.g., `</th`) emitted as delta to the ThinkingPart
    """

    respective_opening_tag: str = ''
    thinking_was_emitted: bool = False

    def flush(self) -> str:
        """Return buffered content for flushing.

        - if no ThinkingPart was emitted (delayed thinking), include opening tag.
        - if ThinkingPart was emitted, only return closing tag buffer.
        """
        if self.thinking_was_emitted:
            return self.buffer
        else:
            return self.respective_opening_tag + self.buffer

    def validate_new_content(self, new_content: str, trim_whitespace: bool = False) -> EndTagValidation:
        if trim_whitespace and not self.has_previous_part:  # pragma: no cover
            new_content = new_content.lstrip()

        if not new_content:
            return EndTagValidation()
        combined = self.buffer + new_content

        # check if the complete closing tag appears in combined
        if self.respective_tag in combined:
            self.buffer = self.respective_tag
            content_before_closed, content_after_closed = combined.split(self.respective_tag, 1)
            return EndTagValidation(
                content_before_closed=content_before_closed, content_after_closed=content_after_closed
            )

        if new_content.startswith(self.expected_next):  # pragma: no cover
            tag_content = combined[: len(self.respective_tag)]
            self.buffer = tag_content
            content_after_closed = combined[len(self.respective_tag) :]
            return EndTagValidation(content_after_closed=content_after_closed)
        elif (overlap := suffix_prefix_overlap(combined, self.respective_tag)) > 0:
            content_to_add = combined[:-overlap]
            content_to_buffer = combined[-overlap:]
            # buffer partial closing tags
            self.buffer = content_to_buffer
            return EndTagValidation(content_before_closed=content_to_add)
        else:
            content_before_closed = combined
            self.buffer = ''
            return EndTagValidation(content_before_closed=content_before_closed)


@dataclass
class ModelResponsePartsManager:
    """Manages a sequence of parts that make up a model's streamed response.

    Parts are generally added and/or updated by providing deltas, which are tracked by vendor-specific IDs.
    """

    _parts: list[ManagedPart] = field(default_factory=list, init=False)
    """A list of parts (text or tool calls) that make up the current state of the model's response."""

    _vendor_id_to_part_index: dict[VendorId, int] = field(default_factory=dict, init=False)
    """Tracks the vendor part IDs of parts to their indices in the `_parts` list.

    Not all parts arrive with vendor part IDs, so the length of the tracker doesn't mirror the length of the _parts.
    `ThinkingPart`s that are created via embedded thinking will stop being tracked once their closing tag is seen.
    """

    _partial_tags_list: list[PartialStartTag | PartialEndTag] = field(default_factory=list, init=False)
    """Tracks active partial thinking tags. Tags contain their own previous_part_index and vendor_part_id."""

    def _append_and_track_new_part(self, part: ManagedPart, vendor_part_id: VendorId | None) -> int:
        """Append a new part to the manager and track it by vendor part ID if provided.

        Will overwrite any existing mapping for the given vendor part ID.
        """
        new_part_index = len(self._parts)
        if vendor_part_id is not None:
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        self._parts.append(part)
        return new_part_index

    def _replace_part(self, part_index: int, part: ManagedPart, vendor_part_id: VendorId) -> int:
        """Replace an existing part at the given index."""
        self._parts[part_index] = part
        self._vendor_id_to_part_index[vendor_part_id] = part_index
        return part_index

    def _stop_tracking_vendor_id(self, vendor_part_id: VendorId) -> None:
        """Stop tracking the given vendor part ID.

        This is useful when a part is considered complete and should no longer be updated.

        Args:
            vendor_part_id: The vendor part ID to stop tracking.
        """
        self._vendor_id_to_part_index.pop(vendor_part_id, None)

    def _get_part_and_index_by_vendor_id(self, vendor_part_id: VendorId) -> tuple[ManagedPart | None, int | None]:
        """Get a part by its vendor part ID."""
        part_index = self._vendor_id_to_part_index.get(vendor_part_id)
        if part_index is not None:
            return self._parts[part_index], part_index
        return None, None

    def _get_partial_by_part_index(self, part_index: int) -> PartialStartTag | PartialEndTag | None:
        """Get a partial thinking tag by its associated part index."""
        for tag in self._partial_tags_list:
            if tag.previous_part_index == part_index:
                return tag
        return None

    def _stop_tracking_partial_tag(self, partial_tag: PartialStartTag | PartialEndTag) -> None:
        """Stop tracking a partial tag."""
        if partial_tag in self._partial_tags_list:  # pragma: no cover
            # this is a defensive check in case we try to remove a tag that wasn't tracked
            self._partial_tags_list.remove(partial_tag)

    def _get_active_partial_tag(
        self,
        existing_part: _ExistingPart[TextPart] | _ExistingPart[ThinkingPart] | None,
        vendor_part_id: VendorId | None = None,
    ) -> PartialStartTag | PartialEndTag | None:
        """Get the active partial tag.

        - if vendor_part_id provided: lookup by vendor_id first (most relevant)
        - if existing_part exists: lookup by that part's index
        - if no existing_part: lookup by latest part's index, or index -1 for unattached tags
        """
        if vendor_part_id is not None:
            for tag in self._partial_tags_list:
                if tag.vendor_part_id == vendor_part_id:
                    return tag

        if existing_part is not None:
            return self._get_partial_by_part_index(existing_part.index)
        elif self._parts:
            latest_index = len(self._parts) - 1
            return self._get_partial_by_part_index(latest_index)
        else:
            return self._get_partial_by_part_index(-1)

    def _emit_text_start(
        self,
        *,
        content: str,
        vendor_part_id: VendorId | None,
        id: str | None = None,
    ) -> PartStartEvent:
        new_text_part = TextPart(content=content, id=id)
        new_part_index = self._append_and_track_new_part(new_text_part, vendor_part_id=vendor_part_id)
        return PartStartEvent(index=new_part_index, part=new_text_part)

    def _emit_text_delta(
        self,
        *,
        text_part: TextPart,
        part_index: int,
        content: str,
    ) -> PartDeltaEvent:
        part_delta = TextPartDelta(content_delta=content)
        self._parts[part_index] = part_delta.apply(text_part)
        return PartDeltaEvent(index=part_index, delta=part_delta)

    def _emit_thinking_delta_from_text(
        self,
        *,
        thinking_part: ThinkingPart,
        part_index: int,
        content: str,
    ) -> PartDeltaEvent:
        """Emit a ThinkingPartDelta from text content. Used only for embedded thinking."""
        part_delta = ThinkingPartDelta(content_delta=content, signature_delta=None, provider_name=None)
        self._parts[part_index] = part_delta.apply(thinking_part)
        return PartDeltaEvent(index=part_index, delta=part_delta)

    def get_parts(self) -> list[ModelResponsePart]:
        """Return only model response parts that are complete (i.e., not ToolCallPartDelta's).

        Returns:
            A list of ModelResponsePart objects. ToolCallPartDelta objects are excluded.
        """
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def handle_text_delta(
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None = None,
        thinking_tags: ThinkingTags | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle incoming text content, creating or updating a TextPart in the manager as appropriate.

        This function also handles what we'll call "embedded thinking", which is the generation of
        `ThinkingPart`s via explicit thinking tags embedded in the text content.
        Activating embedded thinking requires `thinking_tags` to be provided as a tuple of `(opening_tag, closing_tag)`.

        ### Embedded thinking will be processed under the following constraints:
        - C1: Thinking tags are only processed when `thinking_tags` is provided.
        - C2: Opening thinking tags are only recognized at the start of a content chunk.
        - C3.0: Closing thinking tags are recognized anywhere within a content chunk.
            - C3.1: Any text following a closing thinking tag in the same content chunk is treated as a new TextPart.

        ### Supported edge cases of embedded thinking:
        - Thinking tags may arrive split across multiple content chunks. E.g., '<thi' in one chunk and 'nk>' in the next.
        - Partial Opening and Closing tags without adjacent content won't emit an event.
        - EC2: No event is emitted for opening tags until they are fully formed and there is content following them.
            - This is called 'delayed thinking'
        - No event is emitted for closing tags that complete a `ThinkingPart` without any adjacent content.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of text. If None, a new part will be created unless the latest part is already
                a TextPart.
            content: The text content to append to the appropriate TextPart.
            id: An optional id for the text part.
            thinking_tags: If provided, will handle content between the thinking tags as thinking parts.
            ignore_leading_whitespace: If True, will ignore leading whitespace in the content.

        Returns:
            - A `PartStartEvent` if a new part was created.
            - A `PartDeltaEvent` if an existing part was updated.
            - `None` if no new event is emitted (e.g., the first text part was all whitespace).

        Raises:
            UnexpectedModelBehavior: If attempting to apply text content to a part that is not a TextPart.
        """
        existing_part: _ExistingPart[TextPart] | _ExistingPart[ThinkingPart] | None = None

        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    existing_part = _ExistingPart(part=latest_part, index=part_index, found_by='latest_part')
                elif isinstance(latest_part, ThinkingPart):
                    # Only update ThinkingParts created by embedded thinking (have PartialEndTag)
                    # to avoid incorrectly updating ThinkingParts from handle_thinking_delta (native thinking)
                    partial = self._get_partial_by_part_index(part_index)
                    if isinstance(partial, PartialEndTag):
                        existing_part = _ExistingPart(part=latest_part, index=part_index, found_by='latest_part')
        else:
            maybe_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None:
                if isinstance(maybe_part, ThinkingPart):
                    existing_part = _ExistingPart(part=maybe_part, index=part_index, found_by='vendor_part_id')
                elif isinstance(maybe_part, TextPart):
                    existing_part = _ExistingPart(part=maybe_part, index=part_index, found_by='vendor_part_id')
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {maybe_part=}')

        if existing_part is None and ignore_leading_whitespace:
            content = content.lstrip()

        # NOTE this breaks `test_direct.py`, `test_streaming.py` and `test_ui.py` expectations.
        # `test.py` (`TestModel`) is set to generate an empty part at the beginning of the stream.
        # if not content:
        #     return

        # we quickly handle good ol' text
        if not thinking_tags:
            yield from self._handle_plain_text(existing_part, content, vendor_part_id, id)
            return

        # from here on we handle embedded thinking
        partial_tag = self._get_active_partial_tag(existing_part, vendor_part_id)

        # 6. Handle based on current state
        if existing_part is not None and isinstance(existing_part.part, ThinkingPart):
            # Must be closing a ThinkingPart
            thinking_part_existing = cast(_ExistingPart[ThinkingPart], existing_part)
            if partial_tag is None:  # pragma: no cover
                raise RuntimeError('Embedded ThinkingParts must have an associated PartialEndTag')
            if not isinstance(partial_tag, PartialEndTag):  # pragma: no cover
                raise RuntimeError('ThinkingPart cannot be associated with a PartialStartTag')

            yield from self._handle_thinking_closing(
                thinking_part_existing.part,
                thinking_part_existing.index,
                partial_tag,
                content,
                vendor_part_id,
                ignore_leading_whitespace,
            )
            return

        if isinstance(partial_tag, PartialEndTag):
            # Delayed thinking: have PartialEndTag but no ThinkingPart yet
            existing_part = cast(_ExistingPart[TextPart] | None, existing_part)
            yield from self._handle_delayed_thinking(
                existing_part, partial_tag, content, vendor_part_id, ignore_leading_whitespace
            )

        else:
            # Opening tag scenario (partial_tag is None or PartialStartTag)
            opening_tag, closing_tag = thinking_tags
            yield from self._handle_thinking_opening(
                existing_part,
                partial_tag,
                content,
                opening_tag,
                closing_tag,
                vendor_part_id,
                id,
                ignore_leading_whitespace,
            )

    def _handle_plain_text(
        self,
        existing_part: _ExistingPart[TextPart] | _ExistingPart[ThinkingPart] | None,
        content: str,
        vendor_part_id: VendorId | None,
        id: str | None,
    ) -> Generator[PartDeltaEvent | PartStartEvent, None, None]:
        """Handle plain text content (no thinking tags)."""
        if existing_part and isinstance(existing_part.part, TextPart):
            existing_part = cast(_ExistingPart[TextPart], existing_part)
            part_delta = TextPartDelta(content_delta=content)
            self._parts[existing_part.index] = part_delta.apply(existing_part.part)
            yield PartDeltaEvent(index=existing_part.index, delta=part_delta)
        else:
            new_text_part = TextPart(content=content, id=id)
            new_part_index = self._append_and_track_new_part(new_text_part, vendor_part_id)
            yield PartStartEvent(index=new_part_index, part=new_text_part)

    def _handle_thinking_closing(
        self,
        thinking_part: ThinkingPart,
        part_index: int,
        partial_end_tag: PartialEndTag,
        content: str,
        vendor_part_id: VendorId,
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle closing tag validation for an existing ThinkingPart."""
        end_tag_validation = partial_end_tag.validate_new_content(content, trim_whitespace=ignore_leading_whitespace)

        if end_tag_validation.content_before_closed:
            yield self._emit_thinking_delta_from_text(
                thinking_part=thinking_part,
                part_index=part_index,
                content=end_tag_validation.content_before_closed,
            )

        if partial_end_tag.is_complete:
            self._stop_tracking_vendor_id(vendor_part_id)
            self._stop_tracking_partial_tag(partial_end_tag)

            if end_tag_validation.content_after_closed:
                yield self._emit_text_start(
                    content=end_tag_validation.content_after_closed,
                    vendor_part_id=vendor_part_id,
                    id=None,
                )

    def _handle_delayed_thinking(
        self,
        text_part: _ExistingPart[TextPart] | None,
        partial_end_tag: PartialEndTag,
        content: str,
        vendor_part_id: VendorId | None,
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle delayed thinking: PartialEndTag exists but no ThinkingPart created yet."""
        end_tag_validation = partial_end_tag.validate_new_content(content, trim_whitespace=ignore_leading_whitespace)

        if end_tag_validation.content_before_closed:
            # Create ThinkingPart with this content
            new_thinking_part = ThinkingPart(content=end_tag_validation.content_before_closed)
            new_part_index = self._append_and_track_new_part(new_thinking_part, vendor_part_id)
            partial_end_tag.previous_part_index = new_part_index
            partial_end_tag.thinking_was_emitted = True

            yield PartStartEvent(index=new_part_index, part=new_thinking_part)

        if partial_end_tag.is_complete:
            self._stop_tracking_partial_tag(partial_end_tag)

            if end_tag_validation.content_after_closed:
                yield self._emit_text_start(
                    content=end_tag_validation.content_after_closed,
                    vendor_part_id=vendor_part_id,
                    id=None,
                )

    def _handle_thinking_opening(
        self,
        text_part: _ExistingPart[TextPart] | _ExistingPart[ThinkingPart] | None,
        partial_start_tag: PartialStartTag | None,
        content: str,
        opening_tag: str,
        closing_tag: str,
        vendor_part_id: VendorId | None,
        id: str | None,
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle opening tag validation and buffering."""
        text_part = cast(_ExistingPart[TextPart] | None, text_part)

        if partial_start_tag is None:
            partial_start_tag = PartialStartTag(
                respective_tag=opening_tag,
                # Use -1 as sentinel for "no existing part" to enable consistent lookups via _get_partial_by_part_index
                previous_part_index=text_part.index if text_part is not None else -1,
                vendor_part_id=vendor_part_id,
            )
            self._partial_tags_list.append(partial_start_tag)

        start_tag_validation = partial_start_tag.validate_new_content(content)

        # Emit flushed buffer as text
        if start_tag_validation.flushed_buffer:
            if text_part:
                yield self._emit_text_delta(
                    text_part=text_part.part,
                    part_index=text_part.index,
                    content=start_tag_validation.flushed_buffer,
                )
            else:
                text_start_event = self._emit_text_start(
                    content=start_tag_validation.flushed_buffer,
                    vendor_part_id=vendor_part_id,
                    id=id,
                )
                partial_start_tag.previous_part_index = text_start_event.index
                yield text_start_event

        # if tag completed, transition to PartialEndTag
        if partial_start_tag.is_complete:
            # Remove PartialStartTag before creating PartialEndTag to avoid tracking both simultaneously
            self._stop_tracking_partial_tag(partial_start_tag)

            # Create PartialEndTag to track closing tag and subsequent thinking content
            yield from self._create_partial_end_tag(
                closing_tag=closing_tag,
                preceeding_partial_start_tag=partial_start_tag,
                thinking_content=start_tag_validation.thinking_content,
                vendor_part_id=vendor_part_id,
                ignore_leading_whitespace=ignore_leading_whitespace,
            )

    def _create_partial_end_tag(
        self,
        *,
        closing_tag: str,
        preceeding_partial_start_tag: PartialStartTag,
        thinking_content: str,
        vendor_part_id: VendorId | None,
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Create a PartialEndTag and process any thinking content."""
        partial_end_tag = PartialEndTag(
            respective_tag=closing_tag,
            previous_part_index=preceeding_partial_start_tag.previous_part_index,
            respective_opening_tag=preceeding_partial_start_tag.buffer,
            thinking_was_emitted=False,
            vendor_part_id=vendor_part_id,
        )

        end_tag_validation = partial_end_tag.validate_new_content(
            thinking_content, trim_whitespace=ignore_leading_whitespace
        )

        if end_tag_validation.content_before_closed:
            new_thinking_part = ThinkingPart(content=end_tag_validation.content_before_closed)
            new_part_index = self._append_and_track_new_part(new_thinking_part, vendor_part_id)
            partial_end_tag.previous_part_index = new_part_index
            partial_end_tag.thinking_was_emitted = True

            # Track PartialEndTag
            self._partial_tags_list.append(partial_end_tag)

            yield PartStartEvent(index=new_part_index, part=new_thinking_part)

            if partial_end_tag.is_complete:
                self._stop_tracking_vendor_id(vendor_part_id)
                self._stop_tracking_partial_tag(partial_end_tag)
                if end_tag_validation.content_after_closed:
                    yield self._emit_text_start(
                        content=end_tag_validation.content_after_closed,
                        vendor_part_id=vendor_part_id,
                        id=None,
                    )
        elif partial_end_tag.is_complete:
            # Empty thinking: <think></think> - no part to track
            if end_tag_validation.content_after_closed:
                yield self._emit_text_start(
                    content=end_tag_validation.content_after_closed,
                    vendor_part_id=vendor_part_id,
                    id=None,
                )
        else:
            # Partial closing tag but no content yet - add to tracking list
            self._partial_tags_list.append(partial_end_tag)

    def final_flush(self) -> Generator[ModelResponseStreamEvent, None, None]:
        """Emit any buffered content from the last part in the manager.

        This function isn't used internally, it's used by the overarching StreamedResponse
        to ensure any buffered content is flushed when the stream ends.
        """
        last_part_index = len(self._parts) - 1

        if last_part_index >= 0:
            part = self._parts[last_part_index]
            partial_tag = self._get_partial_by_part_index(last_part_index)
        else:
            part = None
            partial_tag = None

        def remove_partial_and_emit_buffered(
            partial: PartialStartTag | PartialEndTag,
            part_index: int,
            part: TextPart | ThinkingPart,
        ) -> Generator[PartStartEvent | PartDeltaEvent, None, None]:
            buffered_content = partial.flush() if isinstance(partial, PartialEndTag) else partial.buffer

            self._stop_tracking_partial_tag(partial)

            if buffered_content:
                delta_type = TextPartDelta if isinstance(part, TextPart) else ThinkingPartDelta
                if part.content:
                    content_delta = delta_type(content_delta=buffered_content)
                    self._parts[part_index] = content_delta.apply(part)
                    yield PartDeltaEvent(index=part_index, delta=content_delta)
                else:
                    updated_part = replace(part, content=buffered_content)
                    self._parts[part_index] = updated_part
                    yield PartStartEvent(index=part_index, part=updated_part)

        if part is not None and isinstance(part, TextPart | ThinkingPart) and partial_tag is not None:
            yield from remove_partial_and_emit_buffered(partial_tag, last_part_index, part)

        # Flush remaining partial tags
        for partial_tag in list(self._partial_tags_list):
            buffered_content = partial_tag.flush() if isinstance(partial_tag, PartialEndTag) else partial_tag.buffer
            if not buffered_content:
                self._stop_tracking_partial_tag(partial_tag)  # partial tag has an associated part index of -1 here
                continue

            if not partial_tag.has_previous_part:
                # No associated part - create new TextPart
                self._stop_tracking_partial_tag(partial_tag)  # partial tag has an associated part index of -1 here

                new_text_part = TextPart(content='')
                new_part_index = self._append_and_track_new_part(new_text_part, vendor_part_id=None)
                yield from remove_partial_and_emit_buffered(partial_tag, new_part_index, new_text_part)
            else:
                # exclude the -1 sentinel (unattached tag) from part lookup
                part_index = partial_tag.previous_part_index
                part = self._parts[part_index]
                if isinstance(part, TextPart | ThinkingPart):
                    yield from remove_partial_and_emit_buffered(partial_tag, part_index, part)
                else:  # pragma: no cover
                    raise RuntimeError('Partial tag is associated with a non-text/non-thinking part')

    def handle_thinking_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        content: str | None = None,
        id: str | None = None,
        signature: str | None = None,
        provider_name: str | None = None,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle incoming thinking content, creating or updating a ThinkingPart in the manager as appropriate.

        When `vendor_part_id` is None, the latest part is updated if it exists and is a ThinkingPart;
        otherwise, a new ThinkingPart is created. When a non-None ID is specified, the ThinkingPart corresponding
        to that vendor part ID is either created or updated.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of thinking. If None, a new part will be created unless the latest part is already
                a ThinkingPart.
            content: The thinking content to append to the appropriate ThinkingPart.
            id: An optional id for the thinking part.
            signature: An optional signature for the thinking content.
            provider_name: An optional provider name for the thinking part.

        Returns:
            A `PartStartEvent` if a new part was created, or a `PartDeltaEvent` if an existing part was updated.

        Raises:
            UnexpectedModelBehavior: If attempting to apply a thinking delta to a part that is not a ThinkingPart.
        """
        existing_thinking_part_and_index: tuple[ThinkingPart, int] | None = None

        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):
                    existing_thinking_part_and_index = latest_part, part_index
        else:
            existing_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None:
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is not None or signature is not None:
                part = ThinkingPart(content=content or '', id=id, signature=signature, provider_name=provider_name)
                new_part_index = self._append_and_track_new_part(part, vendor_part_id)
                yield PartStartEvent(index=new_part_index, part=part)
            else:
                raise UnexpectedModelBehavior('Cannot create a ThinkingPart with no content or signature')
        else:
            if content is not None or signature is not None:
                existing_thinking_part, part_index = existing_thinking_part_and_index
                part_delta = ThinkingPartDelta(
                    content_delta=content, signature_delta=signature, provider_name=provider_name
                )
                self._parts[part_index] = part_delta.apply(existing_thinking_part)
                yield PartDeltaEvent(index=part_index, delta=part_delta)
            else:
                raise UnexpectedModelBehavior('Cannot update a ThinkingPart with no content or signature')

    def handle_tool_call_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str | None = None,
        args: str | dict[str, Any] | None = None,
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent | None:
        """Handle or update a tool call, creating or updating a `ToolCallPart`, `BuiltinToolCallPart`, or `ToolCallPartDelta`.

        Managed items remain as `ToolCallPartDelta`s until they have at least a tool_name, at which
        point they are upgraded to `ToolCallPart`s.

        If `vendor_part_id` is None, updates the latest matching ToolCallPart (or ToolCallPartDelta)
        if any. Otherwise, a new part (or delta) may be created.

        Args:
            vendor_part_id: The ID the vendor uses for this tool call.
                If None, the latest matching tool call may be updated.
            tool_name: The name of the tool. If None, the manager does not enforce
                a name match when `vendor_part_id` is None.
            args: Arguments for the tool call, either as a string, a dictionary of key-value pairs, or None.
            tool_call_id: An optional string representing an identifier for this tool call.

        Returns:
            - A `PartStartEvent` if a new ToolCallPart or BuiltinToolCallPart is created.
            - A `PartDeltaEvent` if an existing part is updated.
            - `None` if no new event is emitted (e.g., the part is still incomplete).

        Raises:
            UnexpectedModelBehavior: If attempting to apply a tool call delta to a part that is not
                a ToolCallPart, BuiltinToolCallPart, or ToolCallPartDelta.
        """
        existing_matching_part_and_index: tuple[ToolCallPartDelta | ToolCallPart | BuiltinToolCallPart, int] | None = (
            None
        )

        if vendor_part_id is None:
            if tool_name is None and self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta):
                    existing_matching_part_and_index = latest_part, part_index
        else:
            existing_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None:
                if not isinstance(existing_part, ToolCallPartDelta | ToolCallPart | BuiltinToolCallPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a tool call delta to {existing_part=}')
                existing_matching_part_and_index = existing_part, part_index

        if existing_matching_part_and_index is None:
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            part = delta.as_part() or delta
            new_part_index = self._append_and_track_new_part(part, vendor_part_id)
            if isinstance(part, ToolCallPart | BuiltinToolCallPart):
                return PartStartEvent(index=new_part_index, part=part)
        else:
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            updated_part = delta.apply(existing_part)
            self._parts[part_index] = updated_part
            if isinstance(updated_part, ToolCallPart | BuiltinToolCallPart):
                if isinstance(existing_part, ToolCallPartDelta):
                    return PartStartEvent(index=part_index, part=updated_part)
                else:
                    if updated_part.tool_call_id and not delta.tool_call_id:
                        delta = replace(delta, tool_call_id=updated_part.tool_call_id)
                    return PartDeltaEvent(index=part_index, delta=delta)

    def handle_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str,
        args: str | dict[str, Any] | None,
        tool_call_id: str | None = None,
        id: str | None = None,
    ) -> ModelResponseStreamEvent:
        """Immediately create or fully-overwrite a ToolCallPart with the given information.

        This does not apply a delta; it directly sets the tool call part contents.

        Args:
            vendor_part_id: The vendor's ID for this tool call part. If not
                None and an existing part is found, that part is overwritten.
            tool_name: The name of the tool being invoked.
            args: The arguments for the tool call, either as a string, a dictionary, or None.
            tool_call_id: An optional string identifier for this tool call.
            id: An optional identifier for this tool call part.

        Returns:
            ModelResponseStreamEvent: A `PartStartEvent` indicating that a new tool call part
            has been added to the manager, or replaced an existing part.
        """
        new_part = ToolCallPart(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id or _generate_tool_call_id(),
            id=id,
        )
        if vendor_part_id is None:
            # vendor_part_id is None, so we unconditionally append a new ToolCallPart to the end of the list
            new_part_index = self._append_and_track_new_part(new_part, vendor_part_id)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new ToolCallPart.
            maybe_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None and isinstance(maybe_part, ToolCallPart):
                new_part_index = self._replace_part(part_index, new_part, vendor_part_id)
            else:
                new_part_index = self._append_and_track_new_part(new_part, vendor_part_id)
        return PartStartEvent(index=new_part_index, part=new_part)

    def handle_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        part: ModelResponsePart,
    ) -> ModelResponseStreamEvent:
        """Create or overwrite a ModelResponsePart.

        Args:
            vendor_part_id: The vendor's ID for this tool call part. If not
                None and an existing part is found, that part is overwritten.
            part: The ModelResponsePart.

        Returns:
            ModelResponseStreamEvent: A `PartStartEvent` indicating that a new part
            has been added to the manager, or replaced an existing part.
        """
        if vendor_part_id is None:
            # vendor_part_id is None, so we unconditionally append a new part to the end of the list
            new_part_index = self._append_and_track_new_part(part, vendor_part_id)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new part.
            maybe_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None and isinstance(maybe_part, type(part)):
                new_part_index = self._replace_part(part_index, part, vendor_part_id)
            else:
                new_part_index = self._append_and_track_new_part(part, vendor_part_id)
        return PartStartEvent(index=new_part_index, part=part)
