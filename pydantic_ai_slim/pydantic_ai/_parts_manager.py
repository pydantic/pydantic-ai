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

from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    BuiltinToolCallPart,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    ProviderDetailsDelta,
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

PartT = TypeVar('PartT', bound=ManagedPart)


@dataclass
class ModelResponsePartsManager:
    """Manages a sequence of parts that make up a model's streamed response.

    Parts are generally added and/or updated by providing deltas, which are tracked by vendor-specific IDs.
    """

    _parts: list[ManagedPart] = field(default_factory=list, init=False)
    """A list of parts (text or tool calls) that make up the current state of the model's response."""
    _vendor_id_to_part_index: dict[VendorId, int] = field(default_factory=dict, init=False)
    """Maps a vendor's "part" ID (if provided) to the index in `_parts` where that part resides."""

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
        provider_details: dict[str, Any] | None = None,
        thinking_tags: tuple[str, str] | None = None,
        ignore_leading_whitespace: bool = False,
    ) -> Iterator[ModelResponseStreamEvent]:
        """Handle incoming text content, creating or updating a TextPart in the manager as appropriate.

        When `vendor_part_id` is None, the latest part is updated if it exists and is a TextPart;
        otherwise, a new TextPart is created. When a non-None ID is specified, the TextPart corresponding
        to that vendor ID is either created or updated.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of text. If None, a new part will be created unless the latest part is already
                a TextPart.
            content: The text content to append to the appropriate TextPart.
            id: An optional id for the text part.
            provider_details: An optional dictionary of provider-specific details for the text part.
            thinking_tags: If provided, will handle content between the thinking tags as thinking parts.
            ignore_leading_whitespace: If True, will ignore leading whitespace in the content.

        Yields:
            A `PartStartEvent` if a new part was created, or a `PartDeltaEvent` if an existing part was updated.
            Yields nothing if no event should be emitted (e.g., the first text part was all whitespace).

        Raises:
            UnexpectedModelBehavior: If attempting to apply text content to a part that is not a TextPart.
        """
        existing_text_part_and_index: tuple[TextPart, int] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a TextPart to update
            existing_text_part_and_index = self._latest_part_if_of_type(TextPart)
        else:
            # Otherwise, attempt to look up an existing TextPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]

                if thinking_tags and isinstance(existing_part, ThinkingPart):
                    # We may be building a thinking part instead of a text part if we had previously seen a thinking tag
                    if content == thinking_tags[1]:
                        # When we see the thinking end tag, we're done with the thinking part and the next text delta will need a new part
                        self._handle_embedded_thinking_end(vendor_part_id)
                        return
                    yield from self._handle_embedded_thinking_content(
                        existing_part, part_index, content, provider_details
                    )
                    return
                elif isinstance(existing_part, TextPart):
                    existing_text_part_and_index = existing_part, part_index
                else:
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')

        if thinking_tags and content == thinking_tags[0]:
            # When we see a thinking start tag (which is a single token), we'll build a new thinking part instead
            yield from self._handle_embedded_thinking_start(vendor_part_id, provider_details)
            return

        if existing_text_part_and_index is None:
            # This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
            # which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.
            if ignore_leading_whitespace and (len(content) == 0 or content.isspace()):
                return

            # There is no existing text part that should be updated, so create a new one
            part = TextPart(content=content, id=id, provider_details=provider_details)
            new_part_index = self._append_part(part, vendor_part_id)
            yield PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing TextPart with the new content delta
            existing_text_part, part_index = existing_text_part_and_index
            part_delta = TextPartDelta(content_delta=content, provider_details=provider_details)
            self._parts[part_index] = part_delta.apply(existing_text_part)
            yield PartDeltaEvent(index=part_index, delta=part_delta)

    def handle_thinking_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        content: str | None = None,
        id: str | None = None,
        signature: str | None = None,
        provider_name: str | None = None,
        provider_details: ProviderDetailsDelta = None,
    ) -> Iterator[ModelResponseStreamEvent]:
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
            provider_details: Either a dict of provider-specific details, or a callable that takes
                the existing part's `provider_details` and returns the updated details. Callables
                allow provider-specific update logic without the parts manager knowing the details.

        Yields:
            A `PartStartEvent` if a new part was created, or a `PartDeltaEvent` if an existing part was updated.

        Raises:
            UnexpectedModelBehavior: If attempting to apply a thinking delta to a part that is not a ThinkingPart.
        """
        existing_thinking_part_and_index: tuple[ThinkingPart, int] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a ThinkingPart to update
            existing_thinking_part_and_index = self._latest_part_if_of_type(ThinkingPart)
        else:
            # Otherwise, attempt to look up an existing ThinkingPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, ThinkingPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a thinking delta to {existing_part=}')
                existing_thinking_part_and_index = existing_part, part_index

        if existing_thinking_part_and_index is None:
            if content is not None or signature is not None or provider_details is not None:
                # There is no existing thinking part that should be updated, so create a new one
                # Resolve provider_details if it's a callback (with None since there's no existing part)
                resolved_details: dict[str, Any] | None
                resolved_details = provider_details(None) if callable(provider_details) else provider_details
                part = ThinkingPart(
                    content=content or '',
                    id=id,
                    signature=signature,
                    provider_name=provider_name,
                    provider_details=resolved_details,
                )
                new_part_index = self._append_part(part, vendor_part_id)
                yield PartStartEvent(index=new_part_index, part=part)
            else:
                raise UnexpectedModelBehavior(
                    'Cannot create a ThinkingPart with no content, signature, or provider_details'
                )
        else:
            existing_thinking_part, part_index = existing_thinking_part_and_index

            # Skip if nothing to update
            if content is None and signature is None and provider_name is None and provider_details is None:
                return

            part_delta = ThinkingPartDelta(
                content_delta=content,
                signature_delta=signature,
                provider_name=provider_name,
                provider_details=provider_details,
            )
            self._parts[part_index] = part_delta.apply(existing_thinking_part)
            yield PartDeltaEvent(index=part_index, delta=part_delta)

    def handle_tool_call_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str | None = None,
        args: str | dict[str, Any] | None = None,
        tool_call_id: str | None = None,
        provider_details: dict[str, Any] | None = None,
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
            provider_details: An optional dictionary of provider-specific details for the tool call part.

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
            if tool_name is None:
                existing_matching_part_and_index = self._latest_part_if_of_type(
                    ToolCallPart, BuiltinToolCallPart, ToolCallPartDelta
                )
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
            delta = ToolCallPartDelta(
                tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id, provider_details=provider_details
            )
            part = delta.as_part() or delta
            new_part_index = self._append_part(part, vendor_part_id)
            # Only emit a PartStartEvent if we have enough information to produce a full ToolCallPart
            if isinstance(part, ToolCallPart | BuiltinToolCallPart):
                return PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing part or delta with the new information
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(
                tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id, provider_details=provider_details
            )
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
        provider_details: dict[str, Any] | None = None,
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
            provider_details: An optional dictionary of provider-specific details for the tool call part.

        Returns:
            ModelResponseStreamEvent: A `PartStartEvent` indicating that a new tool call part
            has been added to the manager, or replaced an existing part.
        """
        new_part = ToolCallPart(
            tool_name=tool_name,
            args=args,
            tool_call_id=tool_call_id or _generate_tool_call_id(),
            id=id,
            provider_details=provider_details,
        )
        if vendor_part_id is None:
            # vendor_part_id is None, so we unconditionally append a new ToolCallPart to the end of the list
            new_part_index = self._append_part(new_part)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new ToolCallPart.
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], ToolCallPart):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = self._append_part(new_part)
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
            new_part_index = self._append_part(part)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new part.
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None and isinstance(self._parts[maybe_part_index], type(part)):
                new_part_index = maybe_part_index
                self._parts[new_part_index] = part
            else:
                new_part_index = self._append_part(part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=part)

    def _stop_tracking_vendor_id(self, vendor_part_id: VendorId | None) -> None:
        """Stop tracking a vendor_part_id (no-op if None or not tracked)."""
        if vendor_part_id is not None:  # pragma: no branch
            self._vendor_id_to_part_index.pop(vendor_part_id, None)

    def _append_part(self, part: ManagedPart, vendor_part_id: VendorId | None = None) -> int:
        """Append a part, optionally track vendor_part_id, return new index."""
        new_index = len(self._parts)
        self._parts.append(part)
        if vendor_part_id is not None:
            self._vendor_id_to_part_index[vendor_part_id] = new_index
        return new_index

    def _latest_part_if_of_type(self, *part_types: type[PartT]) -> tuple[PartT, int] | None:
        """Get the latest part and its index if it's an instance of the given type(s)."""
        if self._parts:
            part_index = len(self._parts) - 1
            latest_part = self._parts[part_index]
            if isinstance(latest_part, part_types):
                return latest_part, part_index
        return None

    def _handle_embedded_thinking_start(
        self, vendor_part_id: VendorId, provider_details: dict[str, Any] | None
    ) -> Iterator[ModelResponseStreamEvent]:
        """Handle <think> tag - create new ThinkingPart."""
        self._stop_tracking_vendor_id(vendor_part_id)
        part = ThinkingPart(content='', provider_details=provider_details)
        new_index = self._append_part(part, vendor_part_id)
        yield PartStartEvent(index=new_index, part=part)

    def _handle_embedded_thinking_content(
        self, existing_part: ThinkingPart, part_index: int, content: str, provider_details: dict[str, Any] | None
    ) -> Iterator[ModelResponseStreamEvent]:
        """Handle content inside <think>...</think>."""
        part_delta = ThinkingPartDelta(content_delta=content, provider_details=provider_details)
        self._parts[part_index] = part_delta.apply(existing_part)
        yield PartDeltaEvent(index=part_index, delta=part_delta)

    def _handle_embedded_thinking_end(self, vendor_part_id: VendorId) -> None:
        """Handle </think> tag - stop tracking so next delta creates new part."""
        self._stop_tracking_vendor_id(vendor_part_id)
