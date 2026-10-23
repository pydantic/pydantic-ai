"""Tool results that a durable engine's JSON codec can't serialize."""


def circular_dict() -> dict[str, object]:
    """A dict that contains itself, which pydantic refuses to serialize with a `ValueError`."""
    value: dict[str, object] = {}
    value['self'] = value
    return value
