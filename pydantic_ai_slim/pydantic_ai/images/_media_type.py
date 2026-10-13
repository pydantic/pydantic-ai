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
    if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        return 'image/webp'
    return None


def output_format_from_media_type(media_type: str) -> str | None:
    """Derive [`GeneratedImage.output_format`][pydantic_ai.images.GeneratedImage] from a media type.

    No provider reports a trustworthy per-image format — Google and xAI report none at all, and
    OpenAI echoes the requested one (openai-node#1850) — so every adapter derives it from the media
    type it settled on for the returned bytes.
    """
    if media_type.startswith('image/'):
        return media_type.removeprefix('image/')
    return None
