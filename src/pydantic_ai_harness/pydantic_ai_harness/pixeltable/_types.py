"""Parsing of the `type_` strings in `Table.get_metadata()`: `'T'` is non-nullable, `'T | None'` nullable."""


def column_base(type_: str) -> str:
    """Base type name, e.g. `'Array'` for `'Array[(8,), Float] | None'`."""
    return str(type_).split(' | ', 1)[0].split('[', 1)[0]


def is_nullable_type(type_: str) -> bool:
    return str(type_).endswith(' | None')
