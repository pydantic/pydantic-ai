from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, cast, final, override

from pydantic import BaseModel

__all__ = (
    'format_tag',
    'Dialect',
    'Content',
)


Dialect = Literal['xml']
BasicType = str | int | float | bool
Content = BaseModel | dict[str, Any] | Iterable[Any] | BasicType | object


def format_tag(tag: str, content: Iterable[Content] | Content, dialect: Dialect = 'xml') -> str:
    try:
        builder = {
            'xml': XMLTagBuilder,
        }[dialect](tag, content)
    except KeyError:
        raise ValueError(f'Unsupported dialect: {dialect}')

    return builder.build()


def _normalize_type(content: Content) -> dict[str, Any] | list[Any] | BasicType:
    match content:
        case BaseModel():
            return content.model_dump()

        case _ if is_dataclass(content):
            return asdict(cast(Any, content))

        case _ if isinstance(content, Iterable) and not isinstance(content, (str, dict)):
            return list(content)

        case str() | int() | float() | bool() | dict():
            return content

        case _:
            raise TypeError(f'Unsupported content type: {type(content)}')


class TagBuilder(ABC):
    def __init__(self, tag: str, content: Iterable[Content] | Content) -> None:
        self._content: dict[str, Any] = self._format_content(tag, content)

    @abstractmethod
    def build(self) -> str: ...

    def _format_content(self, tag: str, content: Iterable[Content] | Content) -> dict[str, Any]:
        root: dict[str, Any] = {}
        normalized = _normalize_type(content)

        match normalized:
            case str() | int() | float() | bool():
                root[tag] = normalized

            case dict():
                root[tag] = {}

                for key, value in normalized.items():
                    root[tag][key] = self._format_content(key, cast(Content, value))[key]

            case _ if isinstance(content, Iterable) and not isinstance(content, (str, dict)):
                root[tag] = [self._format_content(tag, item)[tag] for item in normalized]

            case _:
                raise TypeError(f'Unsupported content type: {type(normalized)}')

        return root


@final
class XMLTagBuilder(TagBuilder):
    @override
    def build(self) -> str:
        raise NotImplementedError
