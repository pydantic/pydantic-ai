"""Shared utilities for the AG-UI protocol integration."""

from __future__ import annotations

import importlib.metadata
import re
from typing import Any, Final

from typing_extensions import Required, TypedDict

from ...messages import ThinkingPart

REASONING_VERSION = (0, 1, 13)
"""AG-UI version that introduced REASONING_* events (replacing THINKING_*)."""

MULTIMODAL_VERSION = (0, 1, 15)
"""AG-UI version that introduced typed multimodal input content (Image/Audio/Video/Document).

Also changed `ReasoningMessageStartEvent.role` from `'assistant'` to `'reasoning'`.
"""

BUILTIN_TOOL_CALL_ID_PREFIX: Final[str] = 'pyd_ai_builtin'

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
