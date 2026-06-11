"""JSON walkers that swap inline `BinaryContent` for externalized markers.

Used by step-persistence backends to keep large media out of snapshot
payloads. The walkers operate on JSON-shaped trees (`list`/`dict`/scalars)
so they never need to know about `pydantic_ai.messages` types directly --
they recognize the serialized shape (`{"kind": "binary", "data": "<b64>"}`)
emitted by `ModelMessagesTypeAdapter.dump_json`.

Round-trip: `externalize_media` (pre-write) and `restore_media` (post-read)
are inverses for bytes ≥ threshold. Bytes below the threshold pass through
unchanged.
"""

from __future__ import annotations

import base64
from typing import TypeGuard

from pydantic_ai_harness.experimental.media._store import MediaContext, MediaStore

_EXTERNAL_MARKER = '__harness_external_media__'


def _is_json_dict(node: object) -> TypeGuard[dict[str, object]]:
    """Treat any `dict` returned by `json.loads` as `dict[str, object]`.

    JSON guarantees string keys, so the runtime `isinstance(..., dict)` check
    is sufficient -- pyright just needs the explicit TypeGuard to propagate
    the value-type narrowing through the walker.
    """
    return isinstance(node, dict)


def _is_json_list(node: object) -> TypeGuard[list[object]]:
    """Treat any `list` returned by `json.loads` as `list[object]`.

    Same rationale as `_is_json_dict` -- JSON lists contain JSON-compatible
    scalars, and we walk every element regardless.
    """
    return isinstance(node, list)


def _is_binary_part(node: dict[str, object]) -> bool:
    return node.get('kind') == 'binary' and isinstance(node.get('data'), str)


def _is_external_marker(node: dict[str, object]) -> bool:
    return node.get(_EXTERNAL_MARKER) is True


async def externalize_media(node: object, *, media_store: MediaStore, threshold_bytes: int) -> object:
    """Walk `node` and replace inline `BinaryContent` parts ≥ threshold.

    Each qualifying part becomes a marker dict carrying the canonical
    `media+sha256://` URI; the raw bytes are written to `media_store`.
    Returns a new tree -- input is not mutated.
    """
    if _is_json_list(node):
        out_list: list[object] = []
        for item in node:
            out_list.append(await externalize_media(item, media_store=media_store, threshold_bytes=threshold_bytes))
        return out_list
    if _is_json_dict(node):
        if _is_binary_part(node):
            replaced = await _maybe_externalize_binary(node, media_store, threshold_bytes)
            if replaced is not None:
                return replaced
        out_dict: dict[str, object] = {}
        for key, value in node.items():
            out_dict[key] = await externalize_media(value, media_store=media_store, threshold_bytes=threshold_bytes)
        return out_dict
    return node


async def _maybe_externalize_binary(
    node: dict[str, object],
    media_store: MediaStore,
    threshold_bytes: int,
) -> dict[str, object] | None:
    # `_is_binary_part` already verified `data` is a string; the cast is safe.
    data_value = node['data']
    assert isinstance(data_value, str)
    raw = base64.b64decode(data_value)
    if len(raw) < threshold_bytes:
        return None
    media_type_value = node.get('media_type')
    media_type = media_type_value if isinstance(media_type_value, str) else None
    uri = await media_store.put(raw, context=MediaContext(media_type=media_type))
    # Preserve every field except the externalized `data`, so the round-trip
    # survives new `BinaryContent` fields added by pydantic_ai upstream.
    marker = {key: value for key, value in node.items() if key != 'data'}
    marker[_EXTERNAL_MARKER] = True
    marker['uri'] = uri
    return marker


async def restore_media(node: object, *, media_store: MediaStore) -> object:
    """Inverse of `externalize_media`. Walks `node` and re-inlines external refs.

    Each marker dict's `uri` is resolved via `media_store.get` and the bytes
    are re-encoded as a `kind=binary` part so the result round-trips
    through `ModelMessagesTypeAdapter.validate_python`.
    """
    if _is_json_list(node):
        out_list: list[object] = []
        for item in node:
            out_list.append(await restore_media(item, media_store=media_store))
        return out_list
    if _is_json_dict(node):
        if _is_external_marker(node):
            return await _restore_external(node, media_store)
        out_dict: dict[str, object] = {}
        for key, value in node.items():
            out_dict[key] = await restore_media(value, media_store=media_store)
        return out_dict
    return node


async def _restore_external(node: dict[str, object], media_store: MediaStore) -> dict[str, object]:
    uri_value = node.get('uri')
    if not isinstance(uri_value, str):
        raise ValueError(f'externalized media marker missing string uri: {node!r}')
    raw = await media_store.get(uri_value)
    # Inverse of `_maybe_externalize_binary`: drop the marker keys, restore `data`,
    # keep every other field the original part carried.
    restored = {key: value for key, value in node.items() if key not in (_EXTERNAL_MARKER, 'uri')}
    restored['data'] = base64.b64encode(raw).decode('ascii')
    return restored
