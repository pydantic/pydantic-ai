from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, cast, final, override

from pydantic import BaseModel

__all__ = (
    'format_tag',
    'Dialect',
)


Dialect = Literal['xml']
BasicType = str | int | float | bool


def format_tag(tag: str, content: Any, dialect: Dialect = 'xml') -> str:
    """Format content into a tagged string representation using the specified dialect.

    Args:
        tag: The tag name to wrap the content with.
        content: The content to be formatted. Can be:
            - Basic types (str, int, float, bool)
            - Dictionaries
            - Lists, tuples, or other iterables
            - Pydantic models
            - Dataclasses
        dialect: The formatting dialect to use. Currently supports:
            - 'xml': XML format

    Returns:
        A string representation of the tagged content in the specified dialect.

    Raises:
        ValueError: If an unsupported dialect is specified.
        TypeError: If content is of an unsupported type.

    Examples:
        >>> format_tag('user', {'name': 'John', 'age': 30})
        '<user><name>John</name><age>30</age></user>'

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class User:
        ...     name: str
        ...     age: int
        >>> user = User('John', 30)
        >>> format_tag('user', user)
        '<user><name>John</name><age>30</age></user>'
    """
    try:
        builder = {
            'xml': XMLTagBuilder,
        }[dialect](tag, content)
    except KeyError:
        raise ValueError(f'Unsupported dialect: {dialect}')

    return builder.build()


def _normalize_type(content: Any) -> dict[str, Any] | list[Any] | BasicType:
    match content:
        case BaseModel():
            return content.model_dump()

        case _ if is_dataclass(content):
            return asdict(cast(Any, content))

        case _ if isinstance(content, Iterable) and not isinstance(content, (str, dict)):
            return list(cast(Iterable[Any], content))

        case str() | int() | float() | bool() | dict():
            return cast(BasicType | dict[str, Any], content)

        case _:
            raise TypeError(f'Unsupported content type: {type(content)}')


class TagBuilder(ABC):
    def __init__(self, tag: str, content: Any) -> None:
        self._content: dict[str, Any] = self._format_content(tag, content)

    @abstractmethod
    def build(self) -> str: ...

    def _format_content(self, tag: str, content: Any) -> dict[str, Any]:
        root: dict[str, Any] = {}
        normalized = _normalize_type(content)

        match normalized:
            case dict():
                root[tag] = {}

                for key, value in normalized.items():
                    root[tag][key] = self._format_content(key, value)[key]

            case list():
                assert isinstance(normalized, list)
                root[tag] = [self._format_content(tag, item)[tag] for item in normalized]

            case _:
                root[tag] = normalized

        return root


@final
class XMLTagBuilder(TagBuilder):
    @override
    def build(self) -> str:
        raise NotImplementedError
