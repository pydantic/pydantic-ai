from __future__ import annotations as _annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Union

from pydantic_ai import UnexpectedModelBehavior
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


ManagedPart = Union[ModelResponsePart, ToolCallPartDelta]


@dataclass
class ModelResponsePartsManager:
    _vendor_id_to_part_index: dict[VendorId, int] = field(default_factory=dict, init=False)
    _parts: list[ManagedPart] = field(default_factory=list, init=False)

    def get_parts(self) -> list[ModelResponsePart]:
        return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]

    def handle_text_delta(self, *, vendor_part_id: Hashable | None, content: str) -> ModelResponseStreamEvent | None:
        # vendor_part_id=None means to use the latest part if it is a text part, otherwise make a new one
        if not content:
            return None

        existing_text_part_and_index: tuple[TextPart, int] | None = None
        if vendor_part_id is None:
            if self._parts:
                latest_part = self._parts[-1]
                part_index = len(self._parts) - 1
                if isinstance(latest_part, TextPart):
                    existing_text_part_and_index = latest_part, part_index
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, TextPart):
                    raise UnexpectedModelBehavior(f'Cannot apply a text delta to {existing_part=}')
                existing_text_part_and_index = existing_part, part_index

        if existing_text_part_and_index is None:
            new_part_index = len(self._parts)
            part = TextPart(content=content)
            if vendor_part_id is not None:
                self._vendor_id_to_part_index[vendor_part_id] = new_part_index
            self._parts.append(part)
            return PartStartEvent(index=new_part_index, part=part)
        else:
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
        # vendor_part_id=None means to use the latest part if it is a matching tool call part, otherwise make a new one
        existing_matching_part_and_index: tuple[ToolCallPartDelta | ToolCallPart, int] | None = None
        if vendor_part_id is None:
            # If vendor_part_id is not provided, the tool_name must match the latest part to perform updates
            if self._parts:
                latest_part = self._parts[-1]
                part_index = len(self._parts) - 1
                if (
                    isinstance(latest_part, ToolCallPart) and (tool_name is None or latest_part.tool_name == tool_name)
                ) or (
                    isinstance(latest_part, ToolCallPartDelta)
                    and (
                        tool_name is None
                        or latest_part.tool_name_delta is None
                        or latest_part.tool_name_delta == tool_name
                    )
                ):
                    existing_matching_part_and_index = latest_part, part_index
        else:
            part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if part_index is not None:
                existing_part = self._parts[part_index]
                if not isinstance(existing_part, (ToolCallPartDelta, ToolCallPart)):
                    raise UnexpectedModelBehavior(f'Cannot apply a tool call delta to {existing_part=}')
                existing_matching_part_and_index = existing_part, part_index

        if existing_matching_part_and_index is None:
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
            existing_part, part_index = existing_matching_part_and_index
            delta = ToolCallPartDelta(tool_name_delta=tool_name, args_delta=args, tool_call_id=tool_call_id)
            updated_part = delta.apply(existing_part)
            self._parts[part_index] = updated_part
            if isinstance(updated_part, ToolCallPart):
                if isinstance(existing_part, ToolCallPartDelta):
                    # In this case, we just upgraded a delta to a full part, so emit a PartStartEvent:
                    return PartStartEvent(index=part_index, part=updated_part)
                else:
                    # In this case, we just updated an existing part, so emit a PartDeltaEvent:
                    return PartDeltaEvent(index=part_index, delta=delta)

    def handle_tool_call_part(
        self,
        *,
        vendor_part_id: Hashable | None,
        tool_name: str,
        args: str | dict[str, Any],
        tool_call_id: str | None = None,
    ) -> ModelResponseStreamEvent:
        new_part = ToolCallPart.from_raw_args(tool_name=tool_name, args=args, tool_call_id=tool_call_id)
        if vendor_part_id is None:
            new_part_index = len(self._parts)
            self._parts.append(new_part)
        else:
            maybe_part_index = self._vendor_id_to_part_index.get(vendor_part_id)
            if maybe_part_index is not None:
                new_part_index = maybe_part_index
                self._parts[new_part_index] = new_part
            else:
                new_part_index = len(self._parts)
                self._parts.append(new_part)
            self._vendor_id_to_part_index[vendor_part_id] = new_part_index
        return PartStartEvent(index=new_part_index, part=new_part)
