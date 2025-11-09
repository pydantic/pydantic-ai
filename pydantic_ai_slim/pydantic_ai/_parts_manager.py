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

from collections.abc import Callable, Generator, Hashable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar, cast

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

TPart = TypeVar('TPart', bound=ModelResponsePart)


@dataclass
class _ExistingPart(Generic[TPart]):
    part: TPart
    index: int
    found_by: Literal['vendor_part_id', 'latest_part']


def suffix_prefix_overlap(s1: str, s2: str) -> int:
    """Return the length of the longest suffix of s1 that is a prefix of s2."""
    n = min(len(s1), len(s2))
    for k in range(n, 0, -1):
        if s1.endswith(s2[:k]):
            return k
    return 0


def is_empty_thinking(thinking_part: ThinkingPart, new_content: str, thinking_tags: ThinkingTags) -> bool:
    _, closing_tag = thinking_tags
    buffered_content = thinking_part.closing_tag_buffer + new_content
    return buffered_content == closing_tag and thinking_part.content == ''


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
    ThinkingParts that are created via the `handle_text_delta` will stop being tracked once their closing tag is seen.
    """

    def append_and_track_new_part(self, part: ManagedPart, vendor_part_id: VendorId | None) -> int:
        """Append a new part to the manager and track it by vendor part ID if provided.

        Will overwrite any existing mapping for the given vendor part ID.
        """
        new_part_index = len(self._parts)
        if vendor_part_id is not None:  # pragma: no branch
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        self._parts.append(part)
        return new_part_index

    def stop_tracking_vendor_id(self, vendor_part_id: VendorId) -> None:
        """Stop tracking the given vendor part ID.

        This is useful when a part is considered complete and should no longer be updated.

        Args:
            vendor_part_id: The vendor part ID to stop tracking.
        """
        self._vendor_id_to_part_index.pop(vendor_part_id, None)

    def get_parts(self) -> list[ModelResponsePart]:
        """Return only model response parts that are complete (i.e., not ToolCallPartDelta's).

        Returns:
            A list of ModelResponsePart objects. ToolCallPartDelta objects are excluded.
        """
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def handle_text_delta(  # noqa: C901
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None = None,
        thinking_tags: ThinkingTags | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> Sequence[ModelResponseStreamEvent]:
        """Handle incoming text content, creating or updating a TextPart in the manager as appropriate.

        This function also handles what we'll call "loose thinking", which is the generation of
        ThinkingParts via explicit thinking tags embedded in the text content.
        Activating loose thinking requires:
        - `thinking_tags` to be provided, which is a tuple of (opening_tag, closing_tag)
        - and a valid vendor_part_id to track ThinkingParts by.

        Loose thinking is handled by:
        - `_handle_text_with_thinking_closing`
        - `_handle_text_with_thinking_opening`

        Loose thinking will be processed under the following constraints:
        - C1: Thinking tags are only processed if `thinking_tags` is provided.
        - C2: Opening thinking tags are only recognized at the start of a content chunk.
        - C3.0: Closing thinking tags are recognized anywhere within a content chunk.
            - C3.1: Any text following a closing thinking tag in the same content chunk is treated as a new TextPart.
            - this could in theory be supported by calling the with_thinking_*` handlers in a while loop
                and having them return any content after a closing tag to be re-processed.
        - C4: Existing ThinkingParts are only updated if a `vendor_part_id` is provided.
            - the reason to require it is that ThinkingParts can also be produced via `handle_thinking_delta`,
            - so we may wrongly append to a latest_part = ThinkingPart that was created that way,
            - this shouldn't happen because in practice models generate thinking one way or the other, not both.
                - and the user would also explicitly ask for loose thinking by providing `thinking_tags`,
                - but it may cause bugginess, for instance when thinking about cases with mixed models.

        Supported edge cases of loose thinking:
        - Thinking tags may arrive split across multiple content chunks. E.g., '<thi' in one chunk and 'nk>' in the next.
        - EC1: Opening tags are buffered in the potential_opening_tag_buffer of a TextPart until fully formed.
        - Closing tags are buffered in the ThinkingPart until fully formed.
        - Partial Opening and Closing tags without adjacent content won't emit an event.
        - EC2: No event is emitted for opening tags until they are fully formed and there is content following them.
            - This is called 'delayed thinking'
        - No event is emitted for closing tags that complete a ThinkingPart without any adjacent content.

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
        potential_part: _ExistingPart[TextPart] | _ExistingPart[ThinkingPart] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a TextPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    potential_part = _ExistingPart(part=latest_part, index=part_index, found_by='latest_part')
                    # ✅ vendor_part_id and ✅ potential_part is a TextPart
                else:
                    # NOTE that the latest part could be a ThinkingPart but
                    #   -> C4: we require ThinkingParts come from/with vendor_part_id's
                    # ❌ vendor_part_id is None + ❌ potential_part is None -> new part!
                    pass
            else:
                # ❌ vendor_part_id is None + ❌ potential_part is None -> new part!
                pass
        else:
            # Otherwise, attempt to look up an existing TextPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if isinstance(existing_part, ThinkingPart):
                    potential_part = _ExistingPart(part=existing_part, index=part_index, found_by='vendor_part_id')
                elif isinstance(existing_part, TextPart):
                    potential_part = _ExistingPart(part=existing_part, index=part_index, found_by='vendor_part_id')
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')
                # ✅ vendor_part_id and ✅ potential_part ❔ can be either TextPart or ThinkingPart ❔
            else:
                # ✅ vendor_part_id but ❌ potential_part is None -> new part!
                pass

        if potential_part is None:
            # This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
            # which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.
            if ignore_leading_whitespace and (len(content) == 0 or content.isspace()):
                return []  # ReturnText 1 (RT1)

        def handle_as_text_part() -> list[PartDeltaEvent | PartStartEvent]:
            if potential_part and isinstance(potential_part.part, TextPart):
                has_buffer = bool(potential_part.part.potential_opening_tag_buffer)
                combined_buffer = potential_part.part.potential_opening_tag_buffer + content
                potential_part.part.potential_opening_tag_buffer = ''

                # Emit Delta if: part has content OR was created without buffering (already emitted Start)
                # Emit Start if: part has no content AND was created with buffering (delayed emission)
                if potential_part.part.content or not has_buffer:
                    part_delta = TextPartDelta(content_delta=combined_buffer)
                    self._parts[potential_part.index] = part_delta.apply(potential_part.part)
                    return [PartDeltaEvent(index=potential_part.index, delta=part_delta)]
                else:
                    # This is the delayed emission case - part was created with a buffer, no content
                    potential_part.part.content = combined_buffer
                    self._parts[potential_part.index] = potential_part.part
                    return [PartStartEvent(index=potential_part.index, part=potential_part.part)]
            else:
                new_text_part = TextPart(content=content, id=id)
                new_part_index = self.append_and_track_new_part(new_text_part, vendor_part_id)
                return [PartStartEvent(index=new_part_index, part=new_text_part)]

        if thinking_tags:
            # handle loose thinking
            if potential_part is not None and isinstance(potential_part.part, ThinkingPart):
                if is_empty_thinking(
                    potential_part.part, content, thinking_tags
                ):  # pragma: no cover - don't have a test case for this yet
                    # TODO discuss how to handle empty thinking
                    #  this applies to non-empty, whitespace-only thinking as well
                    #  -> for now we just untrack it
                    self.stop_tracking_vendor_id(vendor_part_id)
                    return []  # RT2

                potential_part = cast(_ExistingPart[ThinkingPart], potential_part)
                if potential_part.found_by == 'vendor_part_id':
                    # if there's an existing thinking part found by vendor_part_id, handle it directly
                    combined_buffer = potential_part.part.closing_tag_buffer + content
                    potential_part.part.closing_tag_buffer = ''

                    closing_events = list(
                        self._handle_text_with_thinking_closing(
                            thinking_part=potential_part.part,
                            part_index=potential_part.index,
                            thinking_tags=thinking_tags,
                            vendor_part_id=vendor_part_id,
                            combined_buffer=combined_buffer,
                        )
                    )
                    return closing_events  # RT3
                else:
                    # C4: Unhandled branch 1: if the latest part is a ThinkingPart without a vendor_part_id
                    # it will be ignored and a new TextPart will be created instead
                    pass
            else:
                if potential_part is not None and isinstance(potential_part.part, ThinkingPart):
                    # Unhandled branch 2: extension of the above
                    pass
                else:
                    text_part = cast(_ExistingPart[TextPart] | None, potential_part)
                    # we discarded this is a ThinkingPart above
                    events = list(
                        self._handle_text_with_thinking_opening(
                            existing_text_part=text_part,
                            thinking_tags=thinking_tags,
                            vendor_part_id=vendor_part_id,
                            new_content=content,
                            id=id,
                            handle_invalid_opening_tag=handle_as_text_part,
                        )
                    )

                    return events  # RT4

        return handle_as_text_part()  # RT5

    def _handle_text_with_thinking_closing(
        self,
        *,
        thinking_part: ThinkingPart,
        part_index: int,
        thinking_tags: ThinkingTags,
        vendor_part_id: VendorId,
        combined_buffer: str,
    ) -> Generator[PartStartEvent | PartDeltaEvent, None, None]:
        """Handle text content that may contain a closing thinking tag."""
        _, closing_tag = thinking_tags

        if closing_tag in combined_buffer:
            # covers '</think>', 'filling</think>' and 'filling</think>more filling' cases
            before_closing, after_closing = combined_buffer.split(closing_tag, 1)
            if before_closing:
                yield self._emit_thinking_delta_from_text(  # ReturnClosing 1 (RC1)
                    thinking_part=thinking_part,
                    part_index=part_index,
                    content=before_closing,
                )

            self.stop_tracking_vendor_id(vendor_part_id)

            if after_closing:
                new_text_part = TextPart(content=after_closing, id=None)
                new_text_part_index = self.append_and_track_new_part(new_text_part, vendor_part_id)
                yield PartStartEvent(index=new_text_part_index, part=new_text_part)

        elif (overlap := suffix_prefix_overlap(combined_buffer, closing_tag)) > 0:
            # handles split closing tag cases,
            #   e.g. 1 'more</th' becomes PartDelta('more'); buffer = '</th'
            #   e.g. 2 '</thfoo' becomes PartDelta('</thfoo'); buffer = ''
            content_to_add = combined_buffer[:-overlap]
            content_to_buffer = combined_buffer[-overlap:]

            thinking_part.closing_tag_buffer = content_to_buffer

            if content_to_add:
                yield self._emit_thinking_delta_from_text(  # RC2
                    thinking_part=thinking_part, part_index=part_index, content=content_to_add
                )
        else:
            thinking_part.closing_tag_buffer = ''
            yield self._emit_thinking_delta_from_text(
                thinking_part=thinking_part, part_index=part_index, content=combined_buffer
            )

    def _emit_thinking_delta_from_text(
        self,
        *,
        thinking_part: ThinkingPart,
        part_index: int,
        content: str,
    ) -> PartDeltaEvent:
        part_delta = ThinkingPartDelta(content_delta=content, signature_delta=None, provider_name=None)
        self._parts[part_index] = part_delta.apply(thinking_part)
        return PartDeltaEvent(index=part_index, delta=part_delta)

    def _handle_text_with_thinking_opening(  # noqa: C901
        self,
        *,
        existing_text_part: _ExistingPart[TextPart] | None,
        thinking_tags: ThinkingTags,
        vendor_part_id: VendorId | None,
        new_content: str,
        id: str | None = None,
        handle_invalid_opening_tag: Callable[[], Sequence[PartStartEvent | PartDeltaEvent]],
    ) -> Generator[PartStartEvent | PartDeltaEvent, None, None]:
        opening_tag, closing_tag = thinking_tags

        if opening_tag.startswith(new_content) or new_content.startswith(opening_tag):
            # handle stutter e.g. 1: buffer = '<th'; new_content = '<think>content</think>'
            # e.g. 2: buffer = '<th'; new_content = '<th'
            if (
                existing_text_part
                and existing_text_part.part.potential_opening_tag_buffer
                and existing_text_part.part.potential_opening_tag_buffer != opening_tag
            ):
                # if we have stuff in the buffer that is not a valid opening tag, we flush it as text
                # that way the new_content (that also looks like an opening tag) can be processed independently
                if existing_text_part.part.content:
                    text_delta = TextPartDelta(content_delta=existing_text_part.part.potential_opening_tag_buffer)
                    existing_text_part.part.potential_opening_tag_buffer = ''
                    self._parts[existing_text_part.index] = text_delta.apply(existing_text_part.part)
                    delta_event = PartDeltaEvent(index=existing_text_part.index, delta=text_delta)
                    yield delta_event
                else:
                    existing_text_part.part.content = existing_text_part.part.potential_opening_tag_buffer
                    existing_text_part.part.potential_opening_tag_buffer = ''
                    part_start_event = PartStartEvent(
                        index=existing_text_part.index,
                        part=existing_text_part.part,
                    )
                    yield part_start_event

        combined_buffer = (
            existing_text_part.part.potential_opening_tag_buffer + new_content
            if existing_text_part is not None
            else new_content
        )
        # after we handle stutter we can safely combine the new content with any existing buffer

        def _buffer_thinking() -> Sequence[PartStartEvent | PartDeltaEvent]:
            if vendor_part_id is None:
                # C4: can't buffer opening tags without a vendor_part_id
                return handle_invalid_opening_tag()
            if existing_text_part is not None:
                existing_text_part.part.potential_opening_tag_buffer = combined_buffer
                return []
            else:
                # EC1: create a new TextPart to hold the potential opening tag in the buffer
                # we don't emit an event until we determine exactly what this part is
                new_text_part = TextPart(content='', id=id, potential_opening_tag_buffer=combined_buffer)
                self.append_and_track_new_part(new_text_part, vendor_part_id)
                return []

        if opening_tag in combined_buffer:
            # covers cases like '<think>', 'content<think>' and 'pre<think>content'
            if combined_buffer == opening_tag:
                # this block covers the '<think>' case
                # EC2: delayed thinking - we don't emit an event until there's content after the tag
                yield from _buffer_thinking()  # RO1
            elif combined_buffer.startswith(opening_tag):
                # TODO this whole elif is very close to a duplicate of `_handle_text_with_thinking_closing`,
                #   but we can't delegate because we're generating different events (starting ThinkingPart vs updating it)
                #   and there's no easy abstraction that comes to mind, so I'll leave it as is for now.
                after_opening = combined_buffer[len(opening_tag) :]
                # this block handles the cases:
                #   1. where the content might close the thinking tag in the same chunk
                #   2. where the content ends with a partial closing tag: '</th' or '</thi'
                #   3. where the start opens with content without a hint of closing: '<think>content'
                if closing_tag in after_opening:
                    before_closing, after_closing = after_opening.split(closing_tag, 1)
                    if not before_closing:
                        # 1.a. '<think></think>more content'
                        yield from handle_invalid_opening_tag()  # RO2
                        return

                    yield from self._emit_thinking_start_from_text(
                        existing_part=existing_text_part,
                        content=before_closing,
                        vendor_part_id=vendor_part_id,
                    )
                    if after_closing:
                        # 1.b. '<think>content</think>more content'
                        # NOTE follows constraint C3.1: anything after the closing tag is treated as text
                        new_text_part = TextPart(content=after_closing, id=None)
                        new_text_part_index = self.append_and_track_new_part(new_text_part, vendor_part_id)
                        yield PartStartEvent(index=new_text_part_index, part=new_text_part)
                    else:
                        # 1.c. '<think>content</think>'
                        # if there was no content after closing, the thinking tag closed cleanly
                        self.stop_tracking_vendor_id(vendor_part_id)

                    return  # RO3
                elif (overlap := suffix_prefix_overlap(after_opening, closing_tag)) > 0:
                    # handles case 2.a. and 2.b.
                    before_closing = after_opening[:-overlap]
                    closing_buffer = after_opening[-overlap:]
                    if not before_closing:
                        # 2.a. content = '<think></th'
                        # NOTE: we're not covering the case where this is indeed a valid thinking part
                        # like: '<think></th' + 'eres a snake in my boot</think>'
                        yield from handle_invalid_opening_tag()  # RO4
                        return

                    # 2.b. content = '<think>content</th'
                    yield from self._emit_thinking_start_from_text(  # RO5
                        existing_part=existing_text_part,
                        content=before_closing,
                        vendor_part_id=vendor_part_id,
                        closing_buffer=closing_buffer,
                    )
                else:
                    # 3.: '<think>content'
                    yield from self._emit_thinking_start_from_text(  # RO6
                        existing_part=existing_text_part,
                        content=after_opening,
                        vendor_part_id=vendor_part_id,
                    )
            else:
                # constraint C2: we don't allow text before opening tags like 'pre<think>content'
                yield from handle_invalid_opening_tag()  # RO7
        elif combined_buffer in opening_tag:
            # here we handle cases like '<thi', 'hink', and 'nk>'
            if opening_tag.startswith(combined_buffer):
                yield from _buffer_thinking()  # RO8
            else:
                # not a valid opening tag, flush the buffer as text
                yield from handle_invalid_opening_tag()  # RO9
        else:
            # not a valid opening tag, flush the buffer as text
            yield from handle_invalid_opening_tag()  # RO10

    def _emit_thinking_start_from_text(
        self,
        *,
        existing_part: _ExistingPart[TextPart] | None,
        content: str,
        vendor_part_id: VendorId | None,
        closing_buffer: str = '',
    ) -> list[PartStartEvent | PartDeltaEvent]:
        """Emit a ThinkingPart start event from text content.

        If `previous_part` is provided and its content is empty, the ThinkingPart
        will replace that part in the parts list.

        Otherwise, a new ThinkingPart will be appended and the tracked vendor_part_id will be overwritten to point to the new part index.
        """
        # There is no existing thinking part that should be updated, so create a new one
        events: list[PartStartEvent | PartDeltaEvent] = []

        thinking_part = ThinkingPart(content=content, closing_tag_buffer=closing_buffer)

        if existing_part is not None and existing_part.part.content:
            new_part_index = self.append_and_track_new_part(thinking_part, vendor_part_id)
            if (
                existing_part.part.potential_opening_tag_buffer
            ):  # pragma: no cover - this can't happen by the current logic so it's more of a safeguard
                raise RuntimeError(
                    'The buffer of an existing TextPart should have been flushed before creating a ThinkingPart'
                )
        elif existing_part is not None and not existing_part.part.content:
            # C2: we probably used an empty TextPart (that emitted no event) for buffering
            # so instead of appending a new part, we replace that one
            new_part_index = existing_part.index
            self._parts[new_part_index] = thinking_part
        else:
            new_part_index = self.append_and_track_new_part(thinking_part, vendor_part_id)

        if vendor_part_id is not None:
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index

        events.append(PartStartEvent(index=new_part_index, part=thinking_part))
        return events

    def final_flush(self) -> Generator[ModelResponseStreamEvent, None, None]:
        """Emit any buffered content from the last part in the manager.

        This function isn't used internally, it's used by the overarching StreamedResponse
        to ensure any buffered content is flushed when the stream ends.
        """
        # finalize only flushes the buffered content of the last part
        if len(self._parts) == 0:
            return

        part = self._parts[-1]

        if isinstance(part, TextPart) and part.potential_opening_tag_buffer:
            # Flush any buffered potential opening tag as text
            buffered_content = part.potential_opening_tag_buffer
            part.potential_opening_tag_buffer = ''

            last_part_index = len(self._parts) - 1
            if part.content:
                text_delta = TextPartDelta(content_delta=buffered_content)
                self._parts[last_part_index] = text_delta.apply(part)
                yield PartDeltaEvent(index=last_part_index, delta=text_delta)
            else:
                updated_part = replace(part, content=buffered_content)
                self._parts[last_part_index] = updated_part
                yield PartStartEvent(index=last_part_index, part=updated_part)

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
            # If the vendor_part_id is None, check if the latest part is a ThinkingPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):  # pragma: no branch
                    existing_thinking_part_and_index = latest_part, part_index
        else:
            # Otherwise, attempt to look up an existing ThinkingPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is not None or signature is not None:
                # There is no existing thinking part that should be updated, so create a new one
                part = ThinkingPart(content=content or '', id=id, signature=signature, provider_name=provider_name)
                new_part_index = self.append_and_track_new_part(part, vendor_part_id)
                yield PartStartEvent(index=new_part_index, part=part)
            else:
                raise UnexpectedModelBehavior('Cannot create a ThinkingPart with no content or signature')
        else:
            if content is not None or signature is not None:
                # Update the existing ThinkingPart with the new content and/or signature delta
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
            # vendor_part_id is None, so check if the latest part is a matching tool call or delta to update
            # When the vendor_part_id is None, if the tool_name is _not_ None, assume this should be a new part rather
            # than a delta on an existing one. We can change this behavior in the future if necessary for some model.
            if tool_name is None and self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta):  # pragma: no branch
                    existing_matching_part_and_index = latest_part, part_index
        else:
            # vendor_part_id is provided, so look up the corresponding part or delta
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ToolCallPartDelta | ToolCallPart | BuiltinToolCallPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a tool call delta to {existing_part=}')
                existing_matching_part_and_index = existing_part, part_index

        if existing_matching_part_and_index is None:
            # No matching part/delta was found, so create a new ToolCallPartDelta (or ToolCallPart if fully formed)
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            part = delta.as_part() or delta
            new_part_index = self.append_and_track_new_part(part, vendor_part_id)
            # Only emit a PartStartEvent if we have enough information to produce a full ToolCallPart
            if isinstance(part, ToolCallPart | BuiltinToolCallPart):
                return PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing part or delta with the new information
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            updated_part = delta.apply(existing_part)
            self._parts[part_index] = updated_part
            if isinstance(updated_part, ToolCallPart | BuiltinToolCallPart):
                if isinstance(existing_part, ToolCallPartDelta):
                    # We just upgraded a delta to a full part, so emit a PartStartEvent
                    return PartStartEvent(index=part_index, part=updated_part)
                else:
                    # We updated an existing part, so emit a PartDeltaEvent
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
            new_part_index = len(self._parts)
            self._parts.append(new_part)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new ToolCallPart.
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], ToolCallPart):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = len(self._parts)
                self._parts.append(new_part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
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
            new_part_index = len(self._parts)
            self._parts.append(part)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new part.
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], type(part)):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = part
            else:
                new_part_index = len(self._parts)
                self._parts.append(part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=part)
