"""Shared utilities for the AG-UI protocol integration."""

from __future__ import annotations

import importlib.metadata
import json
import re
import warnings
from typing import Any, Final

from typing_extensions import Required, TypedDict

from ..._utils import is_str_dict
from ...messages import ThinkingPart, ToolPartKind, parse_tool_kind, tool_return_content_ta

ENCRYPTED_VALUE_VERSION = (0, 1, 11)
"""AG-UI version that added the `encrypted_value` field to `ToolCall` and `ToolMessage`.

Gates the field-based `tool_kind` round-trip in `dump_messages`/`load_messages`. The streaming
carrier (`ReasoningEncryptedValueEvent`) is a separate `REASONING_*` event gated on
`REASONING_VERSION` (0.1.13) — see `tool_kind_encrypted_value`.
"""

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
    vendor_metadata: dict[str, Any]


class UploadedFileActivityContent(TypedDict, total=False):
    """Content schema for `ActivityMessage` with `activity_type=pydantic_ai_uploaded_file`."""

    file_id: Required[str]
    provider_name: Required[str]
    media_type: str
    identifier: str
    vendor_metadata: dict[str, Any]


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
    `encrypted_value` — the protocol's opaque, client-echoed state-continuity slot. Our payload is
    nested under a `pydantic_ai` key so a genuine provider blob in the same slot (e.g. Google's
    encrypted thinking on a tool call) is never read as our data. The claim is untrusted coming back
    in: `parse_encrypted_tool_kind` returns it only when the key is present, and it degrades to a
    plain part if it doesn't validate.
    """
    if tool_kind is None:
        return None
    return json.dumps({_ENCRYPTED_VALUE_NAMESPACE: {'tool_kind': tool_kind}})


class _EncryptedValueKwargs(TypedDict, total=False):
    """`encrypted_value` kwarg for a `ToolCall`/`ToolMessage`, absent when there's nothing to carry."""

    encrypted_value: str


def tool_kind_encrypted_value_kwargs(tool_kind: ToolPartKind | None, *, supported: bool) -> _EncryptedValueKwargs:
    """`ToolCall`/`ToolMessage` kwargs carrying `tool_kind` as an `encrypted_value`, or empty to omit it.

    Empty when the target version predates the `encrypted_value` field (`supported=False`) or the part
    has no `tool_kind`, so — like the streaming carrier — the field is only ever set when there's a
    claim to carry, never written as a bare `null` a pre-0.1.11 client wouldn't expect.
    """
    value = tool_kind_encrypted_value(tool_kind) if supported else None
    return {'encrypted_value': value} if value is not None else {}


def warn_tool_kind_not_persisted(ag_ui_version: str) -> None:
    """Warn that typed tool parts' `tool_kind` will be lost when dumping below `ENCRYPTED_VALUE_VERSION`.

    The `encrypted_value` carrier only exists from 0.1.11, so on older versions features like lazy
    capabilities and tool search silently forget their state across a round-trip; upgrading the client
    fixes it.
    """
    warnings.warn(
        f'ag-ui-protocol {ag_ui_version} predates the `encrypted_value` field (added in 0.1.11), so '
        'the `tool_kind` of typed tool parts (e.g. lazy capabilities, tool search) cannot be carried '
        'across a dump/load round-trip and those parts will reload as their base classes. Upgrade the '
        'client to ag-ui-protocol >= 0.1.11 to preserve it.',
        UserWarning,
        stacklevel=3,
    )


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
    tool_kind = namespaced.get('tool_kind')
    return parse_tool_kind(tool_kind) if isinstance(tool_kind, str) else None


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


def dump_tool_return_content(content: Any) -> str:
    """Serialize a tool-return `content` value into an AG-UI `ToolMessage.content` / `ToolCallResultEvent.content` string.

    Inverse of [`rehydrate_tool_return_content`][pydantic_ai.ui.ag_ui._utils.rehydrate_tool_return_content],
    kept symmetric with it so a `ToolReturnPart` round-trips faithfully. `.content` is the source of truth
    (`.files` is derived from it), so dumping the full content — multimodal files included — and validating
    it back reconstructs the part. Used for both history serialization (`dump_messages`) and the live event
    stream, so a file a tool returns survives the round-trip through a streaming frontend and can be sent
    back to the model on the next step.

    - A plain string is emitted verbatim, not JSON-wrapped, so the loader hands the original string back.
    - A mapping or sequence — structured returns and anything carrying files at any depth — is dumped
      through `tool_return_content_ta`, so nested `BinaryContent`/`ImageUrl`/... become base64/URL dicts
      that the loader restores to their subclasses.
    - A scalar is JSON-dumped too, but reloads as its string form because AG-UI content is text-only.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return tool_return_content_ta.dump_json(content).decode()


def rehydrate_tool_return_content(content: Any) -> Any:
    """Rehydrate an AG-UI tool-return `content` value into `ToolReturnContent`, restoring multimodal subclasses.

    Inverse of [`dump_tool_return_content`][pydantic_ai.ui.ag_ui._utils.dump_tool_return_content].
    Content is a string on the wire; for structured and file-bearing returns it's our own JSON dump, parsed
    back through `tool_return_content_ta` so multimodal items nested in a mapping or list (`BinaryContent`,
    `ImageUrl`, `UploadedFile`, ...) come back as their subclasses. Image `BinaryContent` is narrowed to
    `BinaryImage`.

    Only a parsed mapping or sequence is run through the discriminator, since nested multimodal items can
    only live inside those. A non-JSON string (plain-text return) and a parsed JSON scalar (`'123'`,
    `'true'`) are returned as the original string: content is text-only, so a scalar is indistinguishable
    from a string on the wire and rehydrating it would silently turn `'123'` into `123`. An
    already-structured (non-string) `content` is validated directly.
    """
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return content
        if not isinstance(parsed, (dict, list)):
            return content
        return tool_return_content_ta.validate_python(parsed)
    if isinstance(content, (dict, list)):
        return tool_return_content_ta.validate_python(content)
    return content
