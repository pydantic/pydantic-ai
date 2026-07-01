"""Shared utilities for the AG-UI protocol integration."""

from __future__ import annotations

import importlib.metadata
import json
import re
from typing import Any, Final

from typing_extensions import Required, TypedDict

from ..._utils import is_str_dict
from ...messages import ThinkingPart, ToolPartKind, parse_tool_kind

REASONING_VERSION = (0, 1, 13)
"""AG-UI version that introduced REASONING_* events (replacing THINKING_*)."""

MULTIMODAL_VERSION = (0, 1, 15)
"""AG-UI version that introduced typed multimodal input content (Image/Audio/Video/Document).

Also changed `ReasoningMessageStartEvent.role` from `'assistant'` to `'reasoning'`.
"""

INTERRUPTS_VERSION = (0, 1, 19)
"""AG-UI version that introduced the interrupt-aware run lifecycle.

`RunFinishedEvent.outcome` (`RunFinishedSuccessOutcome` | `RunFinishedInterruptOutcome`),
`Interrupt`, `ResumeEntry`, and `RunAgentInput.resume` were added in
[ag-ui-protocol#1569](https://github.com/ag-ui-protocol/ag-ui/pull/1569).
"""

BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'

INTERRUPT_ID_PREFIX: Final[str] = 'int-'
"""Prefix used to derive an `Interrupt.id` from a `ToolCallPart.tool_call_id`.

The same prefix is stripped on resume to map `ResumeEntry.interrupt_id` back to a `tool_call_id`.
Keep this string stable — clients may persist `Interrupt.id` across page reloads.
"""

FILE_ACTIVITY_TYPE: Final[str] = 'pydantic_ai_file'
"""Activity type for agent-generated files stored as AG-UI ActivityMessages."""

UPLOADED_FILE_ACTIVITY_TYPE: Final[str] = 'pydantic_ai_uploaded_file'
"""Activity type for uploaded files stored as AG-UI ActivityMessages."""


class FileActivityContent(TypedDict, total=False):
    """Content schema for `ActivityMessage` with `activity_type=pydantic_ai_file`."""

    url: Required[str]
    media_type: str
    id: str
    provider_name: str
    provider_details: dict[str, Any]


class UploadedFileActivityContent(TypedDict, total=False):
    """Content schema for `ActivityMessage` with `activity_type=pydantic_ai_uploaded_file`."""

    file_id: Required[str]
    provider_name: Required[str]
    media_type: str
    identifier: str
    vendor_metadata: Any


_AG_UI_VERSION_RE = re.compile(r'(\d+(?:\.\d+)*)')


def parse_ag_ui_version(version: str) -> tuple[int, ...]:
    """Parse an AG-UI version string (e.g. `'0.1.13'`) into a comparable tuple.

    Pre-release suffixes like `a1`, `b2`, `rc1`, `.dev0` are stripped before parsing.
    """
    from ...exceptions import UserError

    match = _AG_UI_VERSION_RE.match(version)
    if not match:
        raise UserError(f"Invalid AG-UI version {version!r}: expected a dotted numeric version like '0.1.13'")
    return tuple(int(x) for x in match.group(1).split('.'))


def detect_ag_ui_version() -> str:
    """Detect the installed ag-ui-protocol version string.

    Returns the raw installed version (e.g. `'0.1.13'`), or `'0.1.10'` as fallback.
    """
    try:
        return importlib.metadata.version('ag-ui-protocol')
    except Exception:
        return '0.1.10'


DEFAULT_AG_UI_VERSION: str = detect_ag_ui_version()
"""The default AG-UI version, auto-detected from the installed `ag-ui-protocol` package."""

REASONING_MESSAGE_ROLE: str = (
    'reasoning' if parse_ag_ui_version(DEFAULT_AG_UI_VERSION) >= MULTIMODAL_VERSION else 'assistant'
)
"""The correct `role` value for `ReasoningMessageStartEvent`, based on the installed SDK version."""


def thinking_encrypted_metadata(part: ThinkingPart) -> dict[str, Any]:
    """Collect non-None metadata fields from a ThinkingPart for AG-UI encrypted_value."""
    encrypted: dict[str, Any] = {}
    if part.id is not None:
        encrypted['id'] = part.id
    if part.signature is not None:
        encrypted['signature'] = part.signature
    if part.provider_name is not None:
        encrypted['provider_name'] = part.provider_name
    if part.provider_details is not None:
        encrypted['provider_details'] = part.provider_details
    return encrypted


_ENCRYPTED_VALUE_NAMESPACE: Final = 'pydantic_ai'
"""Top-level key our payload is nested under inside an AG-UI `encrypted_value` blob, so a genuine
provider blob in the same slot is never mistaken for our data."""


def tool_kind_encrypted_value(tool_kind: ToolPartKind | None) -> str | None:
    """Pack a part's `tool_kind` into an AG-UI `encrypted_value` blob, namespaced under `pydantic_ai`.

    AG-UI has no generic per-tool metadata field, so we carry the `tool_kind` discriminator in
    `encrypted_value` — the protocol's opaque, client-echoed state-continuity slot (for Zero Data
    Retention), exposed on `ToolCall`/`ToolMessage` and streamed via
    `ReasoningEncryptedValueEvent(subtype='tool-call')`. Our payload is nested under a `pydantic_ai`
    key so a genuine provider blob landing in this slot (e.g. Google's encrypted thinking on a tool
    call) is never read as our data: `parse_encrypted_tool_kind` only returns a claim when that key
    is present. The nested dict also leaves room to carry more metadata later. The claim is untrusted
    coming back in and degrades to a plain part if it doesn't validate.

    Callers gate writes on `REASONING_VERSION` (0.1.13) for symmetry across the streaming and
    non-streaming sides: the `encrypted_value` field itself landed in 0.1.11, but the streaming
    carrier (`ReasoningEncryptedValueEvent`) is a `REASONING_*` event the codebase only emits from
    0.1.13, so a single cutover keeps `tool_kind` round-tripping identically whether the history
    was built by `dump_messages` or by streaming.
    """
    if tool_kind is None:
        return None
    return json.dumps({_ENCRYPTED_VALUE_NAMESPACE: {'tool_kind': tool_kind}})


def parse_encrypted_tool_kind(encrypted_value: str | None) -> ToolPartKind | None:
    """Read a `tool_kind` claim from the `pydantic_ai` namespace of an AG-UI `encrypted_value` blob.

    Client-supplied and untrusted: anything that isn't a JSON object carrying
    `{'pydantic_ai': {'tool_kind': <known ToolPartKind>}}` reads as `None`, so a genuine provider
    encrypted blob (no `pydantic_ai` key) or a forged claim degrades to a plain part.
    """
    if not encrypted_value:
        return None
    try:
        data = json.loads(encrypted_value)
    except json.JSONDecodeError:
        return None
    if not is_str_dict(data):
        return None
    namespaced = data.get(_ENCRYPTED_VALUE_NAMESPACE)
    if not is_str_dict(namespaced):
        return None
    return parse_tool_kind(namespaced.get('tool_kind'))


def parse_builtin_tool_call_id(tool_call_id: str) -> tuple[str, str] | None:
    """Split a builtin tool-call id into its `(provider_name, original_id)`.

    Inverse of the `'|'.join([prefix, provider_name, original_id])` encoding. Returns
    `None` when `tool_call_id` is not a well-formed builtin id, so a malformed
    client-supplied id degrades to the plain tool-call path instead of raising on unpack.
    """
    if not tool_call_id.startswith(BUILTIN_TOOL_CALL_ID_PREFIX):
        return None
    parts = tool_call_id.split('|', 2)
    if len(parts) != 3:
        return None
    return parts[1], parts[2]
