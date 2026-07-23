from __future__ import annotations


def image_media_type_from_bytes(data: bytes) -> str | None:
    """Return the image media type sniffed from magic bytes, or `None` if unrecognized.

    Image providers echo the requested `output_format` even when the bytes they return use a
    different one (gpt-image-2 has been observed returning PNG while echoing the requested webp:
    https://github.com/openai/openai-node/issues/1850). Sniffing the decoded bytes lets adapters
    report the media type the caller actually received instead of the provider's claim.
    """
    if data.startswith(b'\x89PNG'):
        return 'image/png'
    if data.startswith(b'\xff\xd8\xff'):
        return 'image/jpeg'
    if len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None
