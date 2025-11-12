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
    previous_part_index: int | None = None

    @model_validator(mode='after')
    def validate_buffer(self) -> PartialThinkingTag:
        if not self.respective_tag.startswith(self.buffer):
            raise ValueError(f"Buffer '{self.buffer}' does not match the start of tag '{self.respective_tag}'")
        return self

    @property
    def was_emitted(self) -> bool:
        return self.previous_part_index is not None

    @property
    def expected_next(self) -> str:
        return self.respective_tag[len(self.buffer) :]

    @property
    def is_complete(self) -> bool:
        return self.buffer == self.respective_tag


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
            self.buffer = combined[: len(self.respective_tag)]  # -> complete the tag
            thinking_content = combined[len(self.respective_tag) :]
            return StartTagValidation(thinking_content=thinking_content)
        elif self.respective_tag.startswith(combined):
            # combined = '<thi'
            self.buffer = combined
            return StartTagValidation()
        elif self.respective_tag.startswith(new_content):
            # new_content = '<thi' or '<think>'
            flushed_buffer = self.buffer
            self.buffer = new_content  # -> may complete the tag
            return StartTagValidation(flushed_buffer=flushed_buffer)
        elif new_content.startswith(self.respective_tag):
            # new_content = '<think>content'
            flushed_buffer = self.buffer
            self.buffer = new_content[: len(self.respective_tag)]  # -> complete the tag
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
    def validate_new_content(self, new_content: str, trim_whitespace: bool = False) -> EndTagValidation:
        if trim_whitespace:
            # strings are passed by value, so the original string is not modified
            new_content = new_content.lstrip()

        if not new_content:
            return EndTagValidation()
        combined = self.buffer + new_content
        if new_content.startswith(self.expected_next):
            """check if the new_content completes the tag"""
            tag_content = combined[: len(self.respective_tag)]
            self.buffer = tag_content
            content_after_closed = combined[len(self.respective_tag) :]
            return EndTagValidation(content_after_closed=content_after_closed)
        elif (overlap := suffix_prefix_overlap(combined, self.respective_tag)) > 0:
            """check if the new content starts a partial closing tag"""
            content_to_add = combined[:-overlap]
            content_to_buffer = combined[-overlap:]
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
    `ThinkingPart`s that are created via the `handle_text_delta` will stop being tracked once their closing tag is seen.
    """

    _partial_tags_list: list[PartialStartTag | PartialEndTag] = field(default_factory=list, init=False)
    """A list of partial thinking tags being tracked."""

    def _append_and_track_new_part(self, part: ManagedPart, vendor_part_id: VendorId | None) -> int:
        """Append a new part to the manager and track it by vendor part ID if provided.

        Will overwrite any existing mapping for the given vendor part ID.
        """
        new_part_index = len(self._parts)
        if vendor_part_id is not None:  # pragma: no branch
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        self._parts.append(part)
        return new_part_index

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
        for partial in self._partial_tags_list:
            if partial.previous_part_index == part_index:
                return partial
        return None

    def _append_partial_tag(self, partial_tag: PartialStartTag | PartialEndTag) -> None:
        if partial_tag in self._partial_tags_list:
            # rigurosity check for us, that we're only appending new partial tags
            raise RuntimeError('Partial tag is already being tracked')
        self._partial_tags_list.append(partial_tag)

    def _emit_text_start(
        self,
        *,
        content: str,
        id: str | None = None,
    ) -> PartStartEvent:
        new_text_part = TextPart(content=content, id=id)
        new_part_index = self._append_and_track_new_part(new_text_part, vendor_part_id=None)
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

    def handle_text_delta(  # noqa: C901
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
        Activating embedded thinking requires:
        - `thinking_tags` to be provided,
        - and a valid `vendor_part_id` to track `ThinkingPart`s by.

        ### Embedded thinking will be processed under the following constraints:
        - C1: Thinking tags are only processed when `thinking_tags` is provided, which is a tuple of `(opening_tag, closing_tag)`.
        - C2: Opening thinking tags are only recognized at the start of a content chunk.
        - C3.0: Closing thinking tags are recognized anywhere within a content chunk.
            - C3.1: Any text following a closing thinking tag in the same content chunk is treated as a new TextPart.
            - this could in theory be supported by calling the with_thinking_*` handlers in a while loop
                and having them return any content after a closing tag to be re-processed.
        - C4: `ThinkingPart`s created via **embedded thinking** are only updated if a `vendor_part_id` is provided.
            - the reason to is that `ThinkingPart`s can also be produced via `handle_thinking_delta`,
            - so we may wrongly append to a latest_part = ThinkingPart that was created that way,
            - this shouldn't happen because in practice models generate `ThinkingPart`s one way or the other, not both.
                - and the user would also explicitly ask for embedded thinking by providing `thinking_tags`,
                - but it may cause bugginess, for instance in cases with mixed models.

        ### Supported edge cases of embedded thinking:
        - Thinking tags may arrive split across multiple content chunks. E.g., '<thi' in one chunk and 'nk>' in the next.
        - EC1: Opening tags are buffered in the potential_opening_tag_buffer of a TextPart until fully formed.
        - Closing tags are buffered in the `ThinkingPart` until fully formed.
        - Partial Opening and Closing tags without adjacent content won't emit an event.
        - EC2: No event is emitted for opening tags until they are fully formed and there is content following them.
            - This is called 'delayed thinking'
        - No event is emitted for closing tags that complete a `ThinkingPart` without any adjacent content.

        ### Embedded thinking is handled by:
        - `_handle_text_with_thinking_closing`
        - `_handle_text_with_thinking_opening`

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
            # If the vendor_part_id is None, check if the latest part is a TextPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    existing_part = _ExistingPart(part=latest_part, index=part_index, found_by='latest_part')
                else:
                    # NOTE that the latest part could be a ThinkingPart but
                    #   -> C4: we require `ThinkingPart`s come from/with vendor_part_id's
                    pass
            else:
                pass
        else:
            # Otherwise, attempt to look up an existing TextPart by vendor_part_id
            maybe_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None:
                if isinstance(maybe_part, ThinkingPart):
                    existing_part = _ExistingPart(part=maybe_part, index=part_index, found_by='vendor_part_id')
                elif isinstance(maybe_part, TextPart):
                    existing_part = _ExistingPart(part=maybe_part, index=part_index, found_by='vendor_part_id')
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {maybe_part=}')
            else:
                pass

        if existing_part is None:
            # Some models emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
            # which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.
            if ignore_leading_whitespace:
                content = content.lstrip()

        if not content:
            return

        if thinking_tags:
            opening_tag, closing_tag = thinking_tags

            # handle embedded thinking
            if existing_part is not None:
                partial_tag = self._get_partial_by_part_index(existing_part.index)
                if isinstance(existing_part.part, ThinkingPart):
                    existing_part = cast(_ExistingPart[ThinkingPart], existing_part)
                    if existing_part.found_by != 'vendor_part_id':
                        # C4: we currently disallow updating ThinkingParts created via embedded thinking without a vendor_part_id
                        raise RuntimeError('Updating of embedded ThinkingParts requires a vendor_part_id')
                    if partial_tag is None:
                        # we will always create a `PartialEndTag` ahead of a new `ThinkingPart`
                        raise RuntimeError('Embedded ThinkingParts must have an associated PartialEndTag')
                    if isinstance(partial_tag, PartialStartTag):
                        raise RuntimeError('ThinkingPart cannot be associated with a PartialStartTag')

                    end_tag_validation = partial_tag.validate_new_content(content)

                    if end_tag_validation.content_before_closed:
                        yield self._emit_thinking_delta_from_text(
                            thinking_part=existing_part.part,
                            part_index=existing_part.index,
                            content=end_tag_validation.content_before_closed,
                        )
                    if not partial_tag.is_complete:
                        return
                    else:
                        self._stop_tracking_vendor_id(vendor_part_id)
                        self._partial_tags_list.remove(partial_tag)

                        if end_tag_validation.content_after_closed:
                            yield self._emit_text_start(
                                content=end_tag_validation.content_after_closed,
                                id=None,  # TODO should we reuse the id here?
                            )
                        return
                    return  # this closes `if isinstance(existing_part.part, ThinkingPart):`
                else:
                    existing_part = cast(_ExistingPart[TextPart], existing_part)

                    if isinstance(partial_tag, PartialEndTag):
                        # a TextPart will only be associated with a PartialEndTag when a PartialStartTag was completed without content
                        end_tag_validation = partial_tag.validate_new_content(
                            content, trim_whitespace=ignore_leading_whitespace
                        )
                        if end_tag_validation.content_before_closed:
                            # there's content for a ThinkingPart, so we emit one
                            new_thinking_part = ThinkingPart(content=end_tag_validation.content_before_closed)
                            new_part_index = self._append_and_track_new_part(new_thinking_part, vendor_part_id)
                            partial_tag.previous_part_index = new_part_index
                            yield PartStartEvent(index=new_part_index, part=new_thinking_part)
                        else:
                            # there are two cases here:
                            # 1. new_content is a partial closing a it got buffered
                            # 2. new_content closes a thinking tag with no content -> empty thinking
                            if partial_tag.is_complete:
                                self._partial_tags_list.remove(partial_tag)
                    else:
                        if partial_tag is None:
                            # no partial tag exists yet - create one for the start tag
                            partial_tag = PartialStartTag(
                                respective_tag=opening_tag,
                                previous_part_index=existing_part.index,
                            )
                            self._append_partial_tag(partial_tag)

                        start_tag_validation = partial_tag.validate_new_content(content)

                        if start_tag_validation.flushed_buffer:
                            yield self._emit_text_delta(
                                text_part=existing_part.part,
                                part_index=existing_part.index,
                                content=start_tag_validation.flushed_buffer,
                            )

                        if not partial_tag.is_complete:
                            return
                        else:
                            # completed a start tag - we now expect a closing tag
                            self._partial_tags_list.remove(partial_tag)
                            yield from self._handle_new_partial_end_tag(
                                closing_tag=closing_tag,
                                preceeding_partial_start_tag=partial_tag,
                                start_tag_validation=start_tag_validation,
                                vendor_part_id=vendor_part_id,
                                ignore_leading_whitespace=ignore_leading_whitespace,
                            )
                    return
                return  # this closes `if existing_part is not None:`
            else:
                existing_partial_tag = self._partial_tags_list[-1] if self._partial_tags_list else None
                if existing_partial_tag is None:
                    partial_tag = PartialStartTag(respective_tag=opening_tag)
                    self._append_partial_tag(partial_tag)
                    start_tag_validation = partial_tag.validate_new_content(content)

                    if start_tag_validation.flushed_buffer:
                        text_start_event = self._emit_text_start(
                            content=start_tag_validation.flushed_buffer,
                            id=id,
                        )
                        partial_tag.previous_part_index = text_start_event.index
                        yield text_start_event
                    else:
                        if not partial_tag.is_complete:
                            return
                        else:
                            # completed a start tag
                            self._partial_tags_list.remove(partial_tag)
                            yield from self._handle_new_partial_end_tag(
                                closing_tag=closing_tag,
                                preceeding_partial_start_tag=partial_tag,
                                start_tag_validation=start_tag_validation,
                                vendor_part_id=vendor_part_id,
                                ignore_leading_whitespace=ignore_leading_whitespace,
                            )
                elif isinstance(existing_partial_tag, PartialStartTag):
                    start_tag_validation = existing_partial_tag.validate_new_content(content)

                    if start_tag_validation.flushed_buffer:
                        new_text_part = TextPart(content=start_tag_validation.flushed_buffer, id=id)
                        new_part_index = self._append_and_track_new_part(new_text_part, vendor_part_id)
                        existing_partial_tag.previous_part_index = new_part_index
                        yield self._emit_text_delta(
                            text_part=new_text_part,
                            part_index=new_part_index,
                            content=start_tag_validation.flushed_buffer,
                        )
                    if not existing_partial_tag.is_complete:
                        return
                    else:
                        # completed a start tag
                        self._partial_tags_list.remove(existing_partial_tag)
                        yield from self._handle_new_partial_end_tag(
                            closing_tag=closing_tag,
                            preceeding_partial_start_tag=existing_partial_tag,
                            start_tag_validation=start_tag_validation,
                            vendor_part_id=vendor_part_id,
                            ignore_leading_whitespace=ignore_leading_whitespace,
                        )
                else:
                    # existing_partial_tag is a PartialEndTag - this should only happen when a start tag was completed without content
                    end_tag_validation = existing_partial_tag.validate_new_content(
                        content, trim_whitespace=ignore_leading_whitespace
                    )
                    if end_tag_validation.content_before_closed:
                        # there's content for a ThinkingPart, so we emit one
                        new_thinking_part = ThinkingPart(content=end_tag_validation.content_before_closed)
                        new_part_index = self._append_and_track_new_part(new_thinking_part, vendor_part_id)
                        existing_partial_tag.previous_part_index = new_part_index
                        yield PartStartEvent(index=new_part_index, part=new_thinking_part)

                    if existing_partial_tag.is_complete:
                        self._partial_tags_list.remove(existing_partial_tag)
                        if end_tag_validation.content_after_closed:
                            yield self._emit_text_start(
                                content=end_tag_validation.content_after_closed,
                                id=None,  # TODO should we reuse the id here?
                            )
                    return
                return
            return  # this closes `if thinking_tags:`

        # no embedded thinking - handle as normal text part
        if existing_part and isinstance(existing_part.part, TextPart):
            existing_part = cast(_ExistingPart[TextPart], existing_part)
            part_delta = TextPartDelta(content_delta=content)
            self._parts[existing_part.index] = part_delta.apply(existing_part.part)
            yield PartDeltaEvent(index=existing_part.index, delta=part_delta)

        else:
            new_text_part = TextPart(content=content, id=id)
            new_part_index = self._append_and_track_new_part(new_text_part, vendor_part_id)
            yield PartStartEvent(index=new_part_index, part=new_text_part)

    def _handle_new_partial_end_tag(
        self,
        *,
        closing_tag: str,
        preceeding_partial_start_tag: PartialStartTag,
        start_tag_validation: StartTagValidation,
        vendor_part_id: VendorId,
        ignore_leading_whitespace: bool,
    ):
        """Handle a new PartialEndTag following a completed PartialStartTag.

        We call this function even if there's no content after the start tag.
        That was we ensure we have a related PartialEndTag to track the closing of the new ThinkingPart.
        """
        partial_end_tag = PartialEndTag(
            respective_tag=closing_tag,
            previous_part_index=preceeding_partial_start_tag.previous_part_index,
        )
        self._append_partial_tag(partial_end_tag)
        end_tag_validation = partial_end_tag.validate_new_content(
            start_tag_validation.thinking_content,
            trim_whitespace=ignore_leading_whitespace,
        )
        if not end_tag_validation.content_before_closed:
            # there's no content for a ThinkingPart, so it's either buffering a closing tag or empty thinking
            if partial_end_tag.is_complete:
                # is an empty thinking part
                self._partial_tags_list.remove(partial_end_tag)
            # in both cases we return without emitting an event
            return
        else:
            # there's content for a ThinkingPart, so we emit one
            new_thinking_part = ThinkingPart(content=end_tag_validation.content_before_closed)
            new_part_index = self._append_and_track_new_part(new_thinking_part, vendor_part_id)
            partial_end_tag.previous_part_index = new_part_index
            yield PartStartEvent(index=new_part_index, part=new_thinking_part)
            if partial_end_tag.is_complete:
                self._stop_tracking_vendor_id(vendor_part_id)
                self._partial_tags_list.remove(partial_end_tag)
                if end_tag_validation.content_after_closed:
                    yield self._emit_text_start(
                        content=end_tag_validation.content_after_closed,
                        id=None,  # TODO should we reuse the id here?
                    )
            return

    def final_flush(self) -> Generator[ModelResponseStreamEvent, None, None]:
        """Emit any buffered content from the last part in the manager.

        This function isn't used internally, it's used by the overarching StreamedResponse
        to ensure any buffered content is flushed when the stream ends.
        """
        # finalize only flushes the buffered content of the last part
        last_part_index = len(self._parts) - 1
        if last_part_index == -1:
            return

        part = self._parts[last_part_index]
        partial_tag = self._get_partial_by_part_index(last_part_index)

        if isinstance(part, TextPart) and partial_tag is not None:
            # Flush any buffered potential opening tag as text
            buffered_content = partial_tag.buffer
            partial_tag.buffer = ''

            if part.content:
                text_delta = TextPartDelta(content_delta=buffered_content)
                self._parts[last_part_index] = text_delta.apply(part)
                yield PartDeltaEvent(index=last_part_index, delta=text_delta)
            else:
                updated_part = replace(part, content=buffered_content)
                self._parts[last_part_index] = updated_part
                yield PartStartEvent(index=last_part_index, part=updated_part)
        elif isinstance(part, ThinkingPart) and partial_tag is not None:
            # Flush any buffered closing tag content as thinking
            buffered_content = partial_tag.buffer
            partial_tag.buffer = ''

            if part.content:
                thinking_delta = ThinkingPartDelta(content_delta=buffered_content, provider_name=part.provider_name)
                self._parts[last_part_index] = thinking_delta.apply(part)
                yield PartDeltaEvent(index=last_part_index, delta=thinking_delta)
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
            existing_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None:
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is not None or signature is not None:
                # There is no existing thinking part that should be updated, so create a new one
                part = ThinkingPart(content=content or '', id=id, signature=signature, provider_name=provider_name)
                new_part_index = self._append_and_track_new_part(part, vendor_part_id)
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
            existing_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None:
                if not isinstance(existing_part, ToolCallPartDelta | ToolCallPart | BuiltinToolCallPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a tool call delta to {existing_part=}')
                existing_matching_part_and_index = existing_part, part_index

        if existing_matching_part_and_index is None:
            # No matching part/delta was found, so create a new ToolCallPartDelta (or ToolCallPart if fully formed)
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            part = delta.as_part() or delta
            new_part_index = self._append_and_track_new_part(part, vendor_part_id)
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
            new_part_index = self._append_and_track_new_part(new_part, vendor_part_id)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new ToolCallPart.
            maybe_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None and isinstance(maybe_part, ToolCallPart):
                new_part_index = part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = self._append_and_track_new_part(new_part, vendor_part_id)
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
            new_part_index = self._append_and_track_new_part(part, vendor_part_id)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new part.
            maybe_part, part_index = self._get_part_and_index_by_vendor_id(vendor_part_id)
            if part_index is not None and isinstance(maybe_part, type(part)):
                new_part_index = part_index
                self._parts[new_part_index] = part
            else:
                new_part_index = self._append_and_track_new_part(part, vendor_part_id)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=part)
