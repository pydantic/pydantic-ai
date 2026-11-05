"""Wire-format utilities shared by the Google API adapters."""


def character_index(text: str, byte_index: int) -> int | None:
    """Convert a Gemini UTF-8 byte offset to a Python string index."""
    encoded = text.encode()
    if not 0 <= byte_index <= len(encoded):
        return None
    try:
        return len(encoded[:byte_index].decode())
    except UnicodeDecodeError:
        return None
