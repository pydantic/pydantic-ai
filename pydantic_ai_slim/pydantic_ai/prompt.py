from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, final, override

from pydantic import BaseModel

if TYPE_CHECKING:
    from _typeshed import DataclassInstance


__all__ = (
    'format_tag',
    'Dialect',
    'Content',
)


Dialect = Literal['xml']
Content = str | dict[str, Any] | BaseModel | object


def format_tag(tag: str, content: Iterable[Content] | Content, dialect: Dialect = 'xml') -> str:
    try:
        builder = {
            'xml': XMLTagBuilder,
        }[dialect](tag, content)
    except KeyError:
        raise ValueError(f'Unsupported dialect: {dialect}')

    return builder.build()


class TagBuilder(ABC):
    def __init__(self, tag: str, content: Iterable[Content] | Content) -> None:
        self._content: dict[str, Any] = self._format_content(tag, content)

    @abstractmethod
    def build(self) -> str: ...

    def _format_content(self, tag: str, content: Iterable[Content] | Content) -> dict[str, Any]:
        root: dict[str, Any] = {}

        match content:
            case str():
                root[tag] = content

            case dict():
                root = cast(dict[str, Any], content)

            case BaseModel():
                root = content.model_dump()

            case _ if is_dataclass(content):
                root[tag] = asdict(cast(DataclassInstance, content))

            case _:
                raise TypeError(f'Unsupported content type: {type(content)}')

        return root


@final
class XMLTagBuilder(TagBuilder):
    @override
    def build(self) -> str:
        raise NotImplementedError
