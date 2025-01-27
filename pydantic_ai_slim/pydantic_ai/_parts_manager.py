"""This module provides functionality to manage and update parts of a model's streamed response.

The manager tracks which parts (in particular, text and tool calls) correspond to which
vendor-specific identifiers (e.g., `index`, `tool_call_id`, etc., as appropriate for a given model),
and produces PydanticAI-format events as appropriate for consumers of the streaming APIs.

The "vendor-specific identifiers" to use depend on the semantics of the responses of the responses from the vendor,
and are tightly coupled to the specific model being used, and the PydanticAI Model subclass implementation.

This `ModelResponsePartsManager` is used in each of the subclasses of `StreamedResponse` as a way to consolidate
event-emitting logic.
"""

from __future__ import annotations as _annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Union

from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

VendorId = Hashable
"""
Type alias for a vendor identifier, which can be any hashable type (e.g., a string, UUID, etc.)
"""

ManagedPart = Union[ModelResponsePart, ToolCallPartDelta]
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

    def get_parts(self) -> list[ModelResponsePart]:
        """Return only model response parts that are complete (i.e., not ToolCallPartDelta's).

        Returns:
            A list of ModelResponsePart objects. ToolCallPartDelta objects are excluded.
        """
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def handle_text_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        content: str,
    ) -> ModelResponseStreamEvent:
        """Handle incoming text content, creating or updating a TextPart in the manager as appropriate.

        When `vendor_part_id` is None, the latest part is updated if it exists and is a TextPart;
        otherwise, a new TextPart is created. When a non-None ID is specified, the TextPart corresponding
        to that vendor ID is either created or updated.

        Args:
            vendor_part_id: The ID the vendor uses to identify this piece
                of text. If None, a new part will be created unless the latest part is already
                a TextPart.
            content: The text content to append to the appropriate TextPart.

        Returns:
            A `PartStartEvent` if a new part was created, or a `PartDeltaEvent` if an existing part was updated.

        Raises:
            UnexpectedModelBehavior: If attempting to apply text content to a part that is
                not a TextPart.
        """
        existing_text_part_and_index: tuple[TextPart, int] | None = None

        if vendor_part_id is None:
            # If the vendor_part_id is None, check if the latest part is a TextPart to update
            if self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, TextPart):
                    existing_text_part_and_index = latest_part, part_index
        else:
            # Otherwise, attempt to look up an existing TextPart by vendor_part_id
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, TextPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')
                existing_text_part_and_index = existing_part, part_index

        if existing_text_part_and_index is None:
            # There is no existing text part that should be updated, so create a new one
            new_part_index = len(self._parts)
            part = TextPart(content=content)
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            return PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing TextPart with the new content delta
            existing_text_part, part_index = existing_text_part_and_index
            part_delta = TextPartDelta(content_delta=content)
            self._parts[part_index] = part_delta.apply(existing_text_part)
            return PartDeltaEvent(index=part_index, delta=part_delta)

    def handle_tool_call_delta(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str | None,
        args: str | dict[str, Any] | None,
        tool_call_id: str | None,
    ) -> ModelResponseStreamEvent | None:
        """Handle or update a tool call, creating or updating a `ToolCallPart` or `ToolCallPartDelta`.

        Managed items remain as `ToolCallPartDelta`s until they have both a tool_name and arguments, at which
        point they are upgraded to `ToolCallPart`s.

        If `vendor_part_id` is None, updates the latest matching ToolCallPart (or ToolCallPartDelta)
        if any. Otherwise, a new part (or delta) may be created.

        Args:
            vendor_part_id: The ID the vendor uses for this tool call.
                If None, the latest matching tool call may be updated.
            tool_name: The name of the tool. If None, the manager does not enforce
                a name match when `vendor_part_id` is None.
            args: Arguments for the tool call, either as a string or a dictionary of key-value pairs.
            tool_call_id: An optional string representing an identifier for this tool call.

        Returns:
            - A `PartStartEvent` if a new (fully realized) ToolCallPart is created.
            - A `PartDeltaEvent` if an existing part is updated.
            - `None` if no new event is emitted (e.g., the part is still incomplete).

        Raises:
            UnexpectedModelBehavior: If attempting to apply a tool call delta to a part that is not
                a ToolCallPart or ToolCallPartDelta.
        """
        existing_matching_part_and_index: tuple[ToolCallPartDelta | ToolCallPart, int] | None = None

        if vendor_part_id is None:
            # vendor_part_id is None, so check if the latest part is a matching tool call or delta to update
            # When the vendor_part_id is None, if the tool_name is _not_ None, assume this should be a new part rather
            # than a delta on an existing one. We can change this behavior in the future if necessary for some model.
            if tool_name is None and self._parts:
                part_index = len(self._parts) - 1
                latest_part = self._parts[part_index]
                if isinstance(latest_part, (ToolCallPart, ToolCallPartDelta)):
                    existing_matching_part_and_index = latest_part, part_index
        else:
            # vendor_part_id is provided, so look up the corresponding part or delta
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, (ToolCallPartDelta, ToolCallPart)):
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
            if isinstance(part, ToolCallPart):
                return PartStartEvent(index=new_part_index, part=part)
        else:
            # Update the existing part or delta with the new information
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            updated_part = delta.apply(existing_part)
            self._parts[part_index] = updated_part
            if isinstance(updated_part, ToolCallPart):
                if isinstance(existing_part, ToolCallPartDelta):
                    # We just upgraded a delta to a full part, so emit a PartStartEvent
                    return PartStartEvent(index=part_index, part=updated_part)
                else:
                    # We updated an existing part, so emit a PartDeltaEvent
                    return PartDeltaEvent(index=part_index, delta=delta)

    def handle_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str,
        args: str | dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent:
        """Immediately create or fully-overwrite a ToolCallPart with the given information.

        This does not apply a delta; it directly sets the tool call part contents.

        Args:
            vendor_part_id: The vendor's ID for this tool call part. If not
                None and an existing part is found, that part is overwritten.
            tool_name: The name of the tool being invoked.
            args: The arguments for the tool call, either as a string or a dictionary.
            tool_call_id: An optional string identifier for this tool call.

        Returns:
            ModelResponseStreamEvent: A `PartStartEvent` indicating that a new tool call part
            has been added to the manager, or replaced an existing part.
        """
        new_part = ToolCallPart(tool_name=tool_name, args=args, tool_call_id=tool_call_id)
        if vendor_part_id is None:
            # vendor_part_id is None, so we unconditionally append a new ToolCallPart to the end of the list
            new_part_index = len(self._parts)
            self._parts.append(new_part)
        else:
            # vendor_part_id is provided, so find and overwrite or create a new ToolCallPart.
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None:
                new_part_index = maybe_part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = len(self._parts)
                self._parts.append(new_part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=new_part)
