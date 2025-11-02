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
            if buffered_content:
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

    def _handle_text_delta_simple(  # noqa: C901
        self,
        *,
        vendor_part_id: VendorId | None,
        content: str,
        id: str | None,
        thinking_tags: tuple[str, str] | None,
        ignore_leading_whitespace: bool,
    ) -> Generator[ModelResponseStreamEvent, None, None]:
        """Handle text delta without split tag buffering (original logic)."""
        existing_text_part_and_index: tuple[TextPart, int] | None = None

        if vendor_part_id is None:
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, ThinkingPart):
                    yield from self.handle_thinking_delta(vendor_part_id=None, content=content)
                    return
                elif isinstance(latest_part, TextPart):
                    existing_text_part_and_index = latest_part, part_index
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]

                if thinking_tags and isinstance(existing_part, ThinkingPart):
                    end_tag = thinking_tags[1]
                    if end_tag in content:
                        before_end, after_end = content.split(end_tag, 1)

                        if before_end:
                            yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=before_end)

                        self._vendor_id_to_part_index.pop(vendor_part_id)

                        if after_end:
                            yield from self._handle_text_delta_simple(
                                vendor_part_id=vendor_part_id,
                                content=after_end,
                                id=id,
                                thinking_tags=thinking_tags,
                                ignore_leading_whitespace=ignore_leading_whitespace,
                            )
                        return

                    if content == end_tag:
                        self._vendor_id_to_part_index.pop(vendor_part_id)
                        return

                    yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=content)
                    return
                elif isinstance(existing_part, TextPart):
                    existing_text_part_and_index = existing_part, part_index
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')

        # If a TextPart has already been created for this vendor_part_id, disable thinking tag detection
        if vendor_part_id is not None:
            existing_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if existing_part_index is not None and isinstance(self._parts[existing_part_index], TextPart):
                thinking_tags = None

        if thinking_tags and thinking_tags[0] in content:
            start_tag = thinking_tags[0]
            before_start, after_start = content.split(start_tag, 1)

            if before_start:
                if ignore_leading_whitespace and before_start.isspace():
                    before_start = ''

                if before_start:
                    yield from self._handle_text_delta_simple(
                        vendor_part_id=vendor_part_id,
                        content=content,
                        id=id,
                        thinking_tags=None,
                        ignore_leading_whitespace=ignore_leading_whitespace,
                    )
                    return
            self._vendor_id_to_part_index.pop(vendor_part_id, None)

            # Create ThinkingPart but defer PartStartEvent until there is content
            new_part_index = len(self._parts)
            part = ThinkingPart(content='')
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)

            if after_start:
                yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=after_start)
            return

        if existing_text_part_and_index is None:
            # This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
            # which we don't want to end up treating as a final result when using `run_stream` with `str` as a valid `output_type`.
            if ignore_leading_whitespace and (len(content) == 0 or content.isspace()):
                return None

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
            if part_index not in self._started_part_indices:
                self._started_part_indices.add(part_index)
                yield PartStartEvent(index=part_index, part=updated_text_part)
            else:
                yield PartDeltaEvent(index=part_index, delta=part_delta)

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
        combined_content = buffered + content

        part_index = self._vendor_id_to_part_index.get(vendor_part_id)
        existing_part = self._parts[part_index] if part_index is not None else None

        # If a TextPart has already been created for this vendor_part_id, disable thinking tag detection
        if existing_part is not None and isinstance(existing_part, TextPart):
            self._thinking_tag_buffer.pop(vendor_part_id, None)
            yield from self._handle_text_delta_simple(
                vendor_part_id=vendor_part_id,
                content=combined_content,
                id=id,
                thinking_tags=None,
                ignore_leading_whitespace=ignore_leading_whitespace,
            )
            return

        if existing_part is not None and isinstance(existing_part, ThinkingPart):
            if end_tag in combined_content:
                before_end, after_end = combined_content.split(end_tag, 1)

                if before_end:
                    yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=before_end)

                self._vendor_id_to_part_index.pop(vendor_part_id)
                self._thinking_tag_buffer.pop(vendor_part_id, None)

                if after_end:
                    yield from self._handle_text_delta_with_thinking_tags(
                        vendor_part_id=vendor_part_id,
                        content=after_end,
                        id=id,
                        thinking_tags=thinking_tags,
                        ignore_leading_whitespace=ignore_leading_whitespace,
                    )
                return

            # Check if any suffix of combined_content could be the start of the end tag
            for i in range(len(combined_content)):
                suffix = combined_content[i:]
                if self._could_be_tag_start(suffix, end_tag):
                    prefix = combined_content[:i]
                    if prefix:
                        yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=prefix)
                    self._thinking_tag_buffer[vendor_part_id] = suffix
                    return

            # No suffix could be a tag start, so emit all content
            self._thinking_tag_buffer.pop(vendor_part_id, None)
            yield from self.handle_thinking_delta(vendor_part_id=vendor_part_id, content=combined_content)
            return

        if start_tag in combined_content:
            before_start, after_start = combined_content.split(start_tag, 1)

            if before_start:
                if ignore_leading_whitespace and before_start.isspace():
                    before_start = ''

                if before_start:
                    self._thinking_tag_buffer.pop(vendor_part_id, None)
                    yield from self._handle_text_delta_simple(
                        vendor_part_id=vendor_part_id,
                        content=combined_content,
                        id=id,
                        thinking_tags=None,
                        ignore_leading_whitespace=ignore_leading_whitespace,
                    )
                    return
            self._thinking_tag_buffer.pop(vendor_part_id, None)
            self._vendor_id_to_part_index.pop(vendor_part_id, None)

            # Create ThinkingPart but defer PartStartEvent until there is content
            new_part_index = len(self._parts)
            part = ThinkingPart(content='')
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            self._isolated_start_tags[new_part_index] = start_tag

            if after_start:
                yield from self._handle_text_delta_with_thinking_tags(
                    vendor_part_id=vendor_part_id,
                    content=after_start,
                    id=id,
                    thinking_tags=thinking_tags,
                    ignore_leading_whitespace=ignore_leading_whitespace,
                )
            return
        if content.startswith(start_tag[0]) and self._could_be_tag_start(combined_content, start_tag):
            self._thinking_tag_buffer[vendor_part_id] = combined_content
            return

        self._thinking_tag_buffer.pop(vendor_part_id, None)
        yield from self._handle_text_delta_simple(
            vendor_part_id=vendor_part_id,
            content=combined_content,
            id=id,
            thinking_tags=thinking_tags,
            ignore_leading_whitespace=ignore_leading_whitespace,
        )

    def _could_be_tag_start(self, content: str, tag: str) -> bool:
        """Check if content could be the start of a tag."""
        if len(content) >= len(tag):
            return False
        return tag.startswith(content)

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
