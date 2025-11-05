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
from typing import Any

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    FilePart,
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
Type alias for a vendor identifier, which can be any hashable type (e.g., a string, UUID, etc.)
"""

ManagedPart = ModelResponsePart | ToolCallPartDelta
"""
A union of types that are managed by the ModelResponsePartsManager.
Because many vendors have streaming APIs that may produce not-fully-formed tool calls,
this includes ToolCallPartDelta's in addition to the more fully-formed ModelResponsePart's.
"""


def _parse_chunk_for_thinking_tags(
    content: str,
    buffered: str,
    start_tag: str,
    end_tag: str,
    in_thinking: bool,
) -> tuple[list[tuple[str, str]], str]:
    """Parse content for thinking tags, handling split tags across chunks.

    Args:
        content: New content chunk to parse
        buffered: Previously buffered content (for split tags)
        start_tag: Opening thinking tag (e.g., '<think>')
        end_tag: Closing thinking tag (e.g., '</think>')
        in_thinking: Whether currently inside a ThinkingPart

    Returns:
        (segments, new_buffer) where:
        - segments: List of (type, content) tuples
        - type: 'text'|'start_tag'|'thinking'|'end_tag'
        - new_buffer: Content to buffer for next chunk (empty if nothing to buffer)
    """
    combined = buffered + content
    segments: list[tuple[str, str]] = []
    current_thinking_state = in_thinking
    remaining = combined

    while remaining:
        if current_thinking_state:
            if end_tag in remaining:
                before_end, after_end = remaining.split(end_tag, 1)
                if before_end:
                    segments.append(('thinking', before_end))
                segments.append(('end_tag', ''))
                remaining = after_end
                current_thinking_state = False
            else:
                # Check for partial end tag at end of remaining content
                for i in range(len(remaining)):
                    suffix = remaining[i:]
                    if len(suffix) < len(end_tag) and end_tag.startswith(suffix):
                        if i > 0:
                            segments.append(('thinking', remaining[:i]))
                        return segments, suffix

                # No end tag or partial, emit all as thinking
                segments.append(('thinking', remaining))
                return segments, ''
        else:
            if start_tag in remaining:
                before_start, after_start = remaining.split(start_tag, 1)
                if before_start:
                    segments.append(('text', before_start))
                segments.append(('start_tag', ''))
                remaining = after_start
                current_thinking_state = True
            else:
                # Check for partial start tag (only if original content started with first char of tag)
                if content and remaining and content[0] == start_tag[0]:
                    if len(remaining) < len(start_tag) and start_tag.startswith(remaining):
                        return segments, remaining

                # No start tag, treat as text
                segments.append(('text', remaining))
                return segments, ''

    return segments, ''


@dataclass
class ModelResponsePartsManager:
    """Manages a sequence of parts that make up a model's streamed response.

    Parts are generally added and/or updated by providing deltas, which are tracked by vendor-specific IDs.
    """

    _parts: list[ManagedPart] = field(default_factory=list, init=False)
    """A list of parts (text or tool calls) that make up the current state of the model's response."""
    _vendor_id_to_part_index: dict[VendorId, int] = field(default_factory=dict, init=False)
    """Maps a vendor's "part" ID (if provided) to the index in `_parts` where that part resides."""
    _thinking_tag_buffer: dict[VendorId, str] = field(default_factory=dict, init=False)
    """Buffers partial content when thinking tags might be split across chunks."""
    _started_part_indices: set[int] = field(default_factory=set, init=False)
    """Tracks indices of parts for which a PartStartEvent has already been yielded."""
    _isolated_start_tags: dict[int, str] = field(default_factory=dict, init=False)
    """Tracks start tags for isolated ThinkingParts (created from standalone tags with no content)."""

    def get_parts(self) -> list[ModelResponsePart]:
        """Return only model response parts that are complete (i.e., not ToolCallPartDelta's).

        Returns:
            A list of ModelResponsePart objects. ToolCallPartDelta objects are excluded.
        """
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def has_incomplete_parts(self) -> bool:
        """Check if there are any incomplete ToolCallPartDeltas being managed.

        Returns:
            True if there are any ToolCallPartDelta objects in the internal parts list.
        """
        return any(isinstance(p, ToolCallPartDelta) for p in self._parts)

    def is_vendor_id_mapped(self, vendor_id: VendorId) -> bool:
        """Check if a vendor ID is currently mapped to a part index.

        Args:
            vendor_id: The vendor ID to check.

        Returns:
            True if the vendor ID is mapped to a part index, False otherwise.
        """
        return vendor_id in self._vendor_id_to_part_index

    def finalize(self) -> Generator[ModelResponseStreamEvent, None, None]:
        """Flush any buffered content, appending to ThinkingParts or creating TextParts.

        This should be called when streaming is complete to ensure no content is lost.
        Any content buffered in _thinking_tag_buffer will be appended to its corresponding
        ThinkingPart if one exists, otherwise it will be emitted as a TextPart.

        The only possible buffered content to append to ThinkingParts are incomplete closing tags like `</th`

        Yields:
            ModelResponseStreamEvent for any buffered content that gets flushed.
        """
        # convert isolated ThinkingParts to TextParts using their original start tags
        for part_index in range(len(self._parts)):
            if part_index not in self._started_part_indices:
                part = self._parts[part_index]
                # we only convert ThinkingParts from standalone tags (no metadata) to TextParts.
                # ThinkingParts from explicit model deltas have signatures/ids that the tests expect.
                if (
                    isinstance(part, ThinkingPart)
                    and not part.content
                    and not part.signature
                    and not part.id
                    and not part.provider_name
                ):
                    start_tag = self._isolated_start_tags.get(part_index, '<think>')
                    text_part = TextPart(content=start_tag)
                    self._parts[part_index] = text_part
                    yield PartStartEvent(index=part_index, part=text_part)
                    self._started_part_indices.add(part_index)

        # flush any remaining buffered content
        for vendor_part_id, buffered_content in list(self._thinking_tag_buffer.items()):
            if buffered_content:  # pragma: no branch - buffer should never contain empty string
                part_index = self._vendor_id_to_part_index.get(vendor_part_id)

                # If buffered content belongs to a ThinkingPart, append it to the ThinkingPart
                # (for orphaned buffers like '</th')
                if part_index is not None and isinstance(self._parts[part_index], ThinkingPart):
                    yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=buffered_content)
                    self._vendor_id_to_part_index.pop(vendor_part_id)
                else:
                    # Otherwise flush as TextPart
                    # (for orphaned buffers like '<thi')
                    self._vendor_id_to_part_index.pop(vendor_part_id, None)
                    yield from self._handle_text_delta_simple(
                        vendor_part_id=vendor_part_id,
                        content=buffered_content,
                        id=None,
                        thinking_tags=None,
                        ignore_leading_whitespace=False,
                    )

        self._thinking_tag_buffer.clear()

    def handle_text_delta(
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None = None,
        thinking_tags: tuple[str, str] | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle incoming text content, creating or updating a TextPart in the manager as appropriate.

        When `vendor_part_id` is None, the latest part is updated if it exists and is a TextPart;
        otherwise, a new TextPart is created. When a non-None ID is specified, the TextPart corresponding
        to that vendor ID is either created or updated.

        Thinking tags may be split across multiple chunks. When `thinking_tags` is provided and
        `vendor_part_id` is not None, this method buffers content that could be the start of a
        thinking tag appearing at the beginning of the current chunk.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of text. If None, a new part will be created unless the latest part is already
                a TextPart.
            content: The text content to append to the appropriate TextPart.
            id: An optional id for the text part.
            thinking_tags: If provided, will handle content between the thinking tags as thinking parts.
                Buffering for split tags requires a non-None vendor_part_id.
            ignore_leading_whitespace: If True, will ignore leading whitespace in the content.

        Yields:
            - `PartStartEvent` if a new part was created.
            - `PartDeltaEvent` if an existing part was updated.
            May yield multiple events from a single call if buffered content is flushed.

        Raises:
            UnexpectedModelBehavior: If attempting to apply text content to a part that is not a TextPart.
        """
        if thinking_tags and vendor_part_id is not None:
            yield from self._handle_text_delta_with_thinking_tags(
                vendor_part_id=vendor_part_id,
                content=content,
                id=id,
                thinking_tags=thinking_tags,
                ignore_leading_whitespace=ignore_leading_whitespace,
            )
        else:
            yield from self._handle_text_delta_simple(
                vendor_part_id=vendor_part_id,
                content=content,
                id=id,
                thinking_tags=thinking_tags,
                ignore_leading_whitespace=ignore_leading_whitespace,
            )

    def _handle_text_delta_simple(
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None,
        thinking_tags: tuple[str, str] | None,
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle text delta without split tag buffering."""
        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):
                    yield from self.handle_thinking_delta(vendor_part_id=None, content=content)
                    return

        # If a TextPart has already been created for this vendor_part_id, disable thinking tag detection
        else:
            existing_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if existing_part_index is not None and isinstance(self._parts[existing_part_index], TextPart):
                thinking_tags = None

        # Handle thinking tag detection for simple path (no buffering)
        if thinking_tags and thinking_tags[0] in content:
            start_tag = thinking_tags[0]
            before_start, after_start = content.split(start_tag, 1)

            if before_start:
                if ignore_leading_whitespace and before_start.isspace():
                    before_start = ''

                if before_start:
                    yield from self._emit_text_part(
                        vendor_part_id=vendor_part_id,
                        content=content,
                        id=id,
                        ignore_leading_whitespace=False,
                    )
                    return

            self._vendor_id_to_part_index.pop(vendor_part_id, None)
            part = ThinkingPart(content='')
            self._parts.append(part)

            if after_start:
                yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=after_start)
            return

        # emit as TextPart
        yield from self._emit_text_part(
            vendor_part_id=vendor_part_id,
            content=content,
            id=id,
            ignore_leading_whitespace=ignore_leading_whitespace,
        )

    def _handle_text_delta_with_thinking_tags(
        self,
        *,
        vendor_part_id: VendorId,
        content: str,
        id: str | None,
        thinking_tags: tuple[str, str],
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle text delta with thinking tag detection and buffering for split tags."""
        start_tag, end_tag = thinking_tags
        buffered = self._thinking_tag_buffer.get(vendor_part_id, '')

        part_index = self._vendor_id_to_part_index.get(vendor_part_id)
        existing_part = self._parts[part_index] if part_index is not None else None

        # Strip leading whitespace if enabled and no existing part
        if ignore_leading_whitespace and not buffered and not existing_part:
            content = content.lstrip()

        # If a TextPart has already been created for this vendor_part_id, disable thinking tag detection
        if existing_part is not None and isinstance(existing_part, TextPart):
            combined_content = buffered + content
            self._thinking_tag_buffer.pop(vendor_part_id, None)
            yield from self._emit_text_part(
                vendor_part_id=vendor_part_id,
                content=combined_content,
                id=id,
                ignore_leading_whitespace=False,
            )
            return

        in_thinking = existing_part is not None and isinstance(existing_part, ThinkingPart)

        segments, new_buffer = _parse_chunk_for_thinking_tags(
            content=content,
            buffered=buffered,
            start_tag=start_tag,
            end_tag=end_tag,
            in_thinking=in_thinking,
        )

        # Check for text before thinking tag - if so, treat entire combined content as text
        # this covers cases like `pre<think>` or `pre<thi`
        if segments and segments[0][0] == 'text':
            text_content = segments[0][1]

            if text_content:  # praga: no cover - line was always true
                combined_content = buffered + content
                self._thinking_tag_buffer.pop(vendor_part_id, None)
                yield from self._emit_text_part(
                    vendor_part_id=vendor_part_id,
                    content=combined_content,
                    id=id,
                    ignore_leading_whitespace=False,
                )
                return

        for i, (segment_type, segment_content) in enumerate(segments):
            if segment_type == 'text':
                # Skip whitespace-only text before a thinking tag when ignore_leading_whitespace=True
                skip_whitespace_before_tag = (
                    ignore_leading_whitespace
                    and segment_content.isspace()
                    and i + 1 < len(segments)
                    and segments[i + 1][0] == 'start_tag'
                )
                if not skip_whitespace_before_tag:  # praga: no cover - line was always true (this is probably dead code, will remove after double checking)
                    yield from self._emit_text_part(
                        vendor_part_id=vendor_part_id,
                        content=segment_content,
                        id=id,
                        ignore_leading_whitespace=ignore_leading_whitespace,
                    )
            elif segment_type == 'start_tag':
                self._vendor_id_to_part_index.pop(vendor_part_id, None)
                new_part_index = len(self._parts)
                part = ThinkingPart(content='')
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
                self._parts.append(part)
                self._isolated_start_tags[new_part_index] = start_tag
            elif segment_type == 'thinking':
                yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=segment_content)
            elif segment_type == 'end_tag':  # pragma: no cover
                self._vendor_id_to_part_index.pop(vendor_part_id)

        if new_buffer:
            self._thinking_tag_buffer[vendor_part_id] = new_buffer
        else:
            self._thinking_tag_buffer.pop(vendor_part_id, None)

    def _emit_text_part(
        self,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Create or update a TextPart, yielding appropriate events.

        Args:
            vendor_part_id: Vendor ID for tracking this part
            content: Text content to add
            id: Optional id for the text part
            ignore_leading_whitespace: Whether to ignore empty/whitespace content

        Yields:
            PartStartEvent if creating new part, PartDeltaEvent if updating existing part
        """
        if ignore_leading_whitespace and (len(content) == 0 or content.isspace()):
            return

        existing_text_part_and_index: tuple[TextPart, int] | None = None

        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    existing_text_part_and_index = latest_part, part_index
            # else: existing_text_part_and_index remains None
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if isinstance(existing_part, TextPart):
                    existing_text_part_and_index = existing_part, part_index
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')
            # else: existing_text_part_and_index remains None

        if existing_text_part_and_index is None:
            new_part_index = len(self._parts)
            part = TextPart(content=content, id=id)
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            yield PartStartEvent(index=new_part_index, part=part)
            self._started_part_indices.add(new_part_index)
        else:
            existing_text_part, part_index = existing_text_part_and_index
            part_delta = TextPartDelta(content_delta=content)
            updated_text_part = part_delta.apply(existing_text_part)
            self._parts[part_index] = updated_text_part
            if (
                part_index not in self._started_part_indices
            ):  # pragma: no cover - TextPart should have already emitted PartStartEvent when created
                self._started_part_indices.add(part_index)
                yield PartStartEvent(index=part_index, part=updated_text_part)
            else:
                yield PartDeltaEvent(index=part_index, delta=part_delta)

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
        to that vendor ID is either created or updated.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of thinking. If None, a new part will be created unless the latest part is already
                a ThinkingPart.
            content: The thinking content to append to the appropriate ThinkingPart.
            id: An optional id for the thinking part.
            signature: An optional signature for the thinking content.
            provider_name: An optional provider name for the thinking part.

        Returns:
            A Generator of a `PartStartEvent` if a new part was created, or a `PartDeltaEvent` if an existing part was updated.

        Raises:
            UnexpectedModelBehavior: If attempting to apply a thinking delta to a part that is not a ThinkingPart.
        """
        existing_thinking_part_and_index: tuple[ThinkingPart, int] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a ThinkingPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):
                    existing_thinking_part_and_index = latest_part, part_index
                elif isinstance(latest_part, TextPart):
                    raise UnexpectedModelBehavior(
                        'Cannot create ThinkingPart after TextPart: thinking must come before text in response'
                    )
                else:  # pragma: no cover - `handle_thinking_delta` should never be called when vendor_part_id is None the latest part is not a ThinkingPart or TextPart
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {latest_part=}')
        else:
            # Otherwise, attempt to look up an existing ThinkingPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is None and signature is None:
                raise UnexpectedModelBehavior('Cannot create a ThinkingPart with no content or signature')

            # There is no existing thinking part that should be updated, so create a new one
            new_part_index = len(self._parts)
            part = ThinkingPart(content=content or '', id=id, signature=signature, provider_name=provider_name)
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            yield PartStartEvent(index=new_part_index, part=part)
            self._started_part_indices.add(new_part_index)
        else:
            if content is None and signature is None:
                raise UnexpectedModelBehavior('Cannot update a ThinkingPart with no content or signature')

            # Update the existing ThinkingPart with the new content and/or signature delta
            existing_thinking_part, part_index = existing_thinking_part_and_index
            part_delta = ThinkingPartDelta(
                content_delta=content, signature_delta=signature, provider_name=provider_name
            )
            updated_thinking_part = part_delta.apply(existing_thinking_part)
            self._parts[part_index] = updated_thinking_part
            if part_index not in self._started_part_indices:
                self._started_part_indices.add(part_index)
                yield PartStartEvent(index=part_index, part=updated_thinking_part)
            else:
                yield PartDeltaEvent(index=part_index, delta=part_delta)

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
                if isinstance(latest_part, ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta):
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
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = len(self._parts)
            new_part_index = len(self._parts)
            self._parts.append(part)
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
        self, *, vendor_part_id: Hashable | None, part: BuiltinToolCallPart | BuiltinToolReturnPart | FilePart
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
