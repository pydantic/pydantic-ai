from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import TypedDict

from pydantic_ai.messages import ModelRequest, ToolReturnPart

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage


LOAD_CAPABILITY_TOOL_NAME = 'load_capability'


class LoadCapabilityReturn(TypedDict):
    capability_id: str
    instructions: str | None


def extract_load_capability_return(content: Any) -> LoadCapabilityReturn | None:
    if not isinstance(content, dict):
        return None

    content = cast(dict[str, Any], content)
    capability_id = content.get('capability_id')
    if not isinstance(capability_id, str):
        return None

    instructions = content.get('instructions')
    if instructions is not None and not isinstance(instructions, str):
        return None

    return LoadCapabilityReturn(capability_id=capability_id, instructions=instructions)


def parse_loaded_capabilities(messages: Sequence[ModelMessage]) -> set[str]:
    """Parse message history to find capabilities loaded via ``load_capability``.

    Dispatches on the typed `LoadCapabilityReturnPart` subclass first (the normal path
    once a part has flowed through the narrower), and falls back to a `tool_name` check
    for pre-typed-parts histories so existing serialized runs keep loading correctly.
    Both paths re-validate the payload via `extract_load_capability_return` so a
    malformed dict can't sneak a `capability_id` into `loaded`.
    """
    # Lazy import to avoid a cycle with `messages.py` (which late-imports `_load_capability`,
    # which depends on `_deferred`).
    from ._load_capability import LoadCapabilityReturnPart

    loaded: set[str] = set()
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, LoadCapabilityReturnPart) or (
                    isinstance(part, ToolReturnPart) and part.tool_name == LOAD_CAPABILITY_TOOL_NAME
                ):
                    parsed = extract_load_capability_return(part.content)
                    if parsed is not None:
                        loaded.add(parsed['capability_id'])

    return loaded
