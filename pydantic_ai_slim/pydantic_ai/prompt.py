from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from functools import partial
from typing import Any, Literal, Union, cast, final

from pydantic import BaseModel

__all__ = (
    'format_tag',
    'format_context',
    'format_examples',
    'format_rules',
    'Dialect',
)

Dialect = Literal['xml']


def format_tag(content: Any, tag: str, dialect: Dialect = 'xml') -> str:
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
        >>> format_tag({'name': 'John', 'age': 30}, tag='user')
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
    if dialect != 'xml':
        raise ValueError(f'Unsupported dialect: {dialect}')

    return XMLTagBuilder(tag, content).build()


format_examples = partial(format_tag, tag='examples')
format_rules = partial(format_tag, tag='rules')
format_context = partial(format_tag, tag='context')


def _normalize_type(content: Any) -> dict[str, Any] | list[Any] | str | int | float | bool:
    """Normalize various Python types into a consistent format for tag building.

    This internal function converts complex Python objects into basic types that can be
    easily formatted into tagged strings. It handles special cases like Pydantic models,
    dataclasses, and iterables.

    Args:
        content: The content to normalize. Can be:
            - Pydantic BaseModel (converted to dict)
            - Dataclass (converted to dict)
            - Iterable (converted to list, except str and dict)
            - Basic types (str, int, float, bool, dict)

    Returns:
        The normalized content as one of:
            - dict[str, Any]: For mapping types (dict, BaseModel, dataclass)
            - list[Any]: For iterable types
            - str: For string values
            - int: For integer values
            - float: For float values
            - bool: For boolean values

    Raises:
        TypeError: If the content type is not supported for normalization.

    Examples:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        >>> _normalize_type(User(name='John'))
        {'name': 'John'}

        >>> from dataclasses import dataclass
        >>> @dataclass
        >>> class Point:
        ...     x: int
        ...     y: int
        >>> _normalize_type(Point(1, 2))
        {'x': 1, 'y': 2}

        >>> _normalize_type(['a', 'b', 'c'])
        ['a', 'b', 'c']

        >>> _normalize_type(42)
        42
    """
    if isinstance(content, BaseModel):
        return content.model_dump()
    elif is_dataclass(content):
        return asdict(cast(Any, content))
    elif isinstance(content, (str, int, float, bool, dict)):
        return cast(Union[str, int, float, bool, dict[str, Any]], content)
    elif isinstance(content, Iterable):
        return list(cast(Iterable[Any], content))
    else:
        raise TypeError(f'Unsupported content type: {type(content)}')


class TagBuilder(ABC):
    """Abstract base class for building tagged string representations of content.

    This class provides the base functionality for converting various Python data types
    into a tagged string format. It handles normalization and formatting of content
    while leaving the specific string building implementation to subclasses.

    Args:
        tag: The root tag name to wrap the content with.
        content: The content to be formatted. Supported types include:
            - Basic types (str, int, float, bool)
            - Dictionaries
            - Lists, tuples, or other iterables
            - Pydantic models
            - Dataclasses

    Attributes:
        _content: Normalized and formatted content as a dictionary structure.

    Raises:
        TypeError: If content is of an unsupported type.
    """

    def __init__(self, tag: str, content: Any) -> None:
        self._content: dict[str, Any] = self._format_content(tag, content)

    @abstractmethod
    def build(self) -> str: ...

    def _format_content(self, tag: str, content: Any) -> dict[str, Any]:
        """Format content into a structured dictionary representation.

        This internal method processes the input content and creates a hierarchical dictionary
        structure where each level represents a tagged element. It handles nested structures
        and different types of content recursively.

        Args:
            tag: The tag name to use as the key in the resulting dictionary.
            content: The content to format. Can be:
                - Basic types (str, int, float, bool)
                - Dictionaries (processed recursively)
                - Lists (each item processed separately)
                - Complex objects (dataclasses or pydantic models)

        Returns:
            A dictionary with a single key (the tag) and a structured value that represents
            the formatted content. The structure depends on the input type:
                - For basic types: {tag: value}
                - For dicts: {tag: {key1: formatted1, key2: formatted2, ...}}
                - For lists: {tag: [formatted1, formatted2, ...]}

        Examples:
            >>> builder = TagBuilder(...)
            >>> builder._format_content('user', 'John')
            {'user': 'John'}

            >>> builder._format_content('user', {'name': 'John', 'age': 30})
            {'user': {'name': 'John', 'age': 30}}

            >>> builder._format_content('items', [1, 2, 3])
            {'items': [{'items': 1}, {'items': 2}, {'items': 3}]}
        """
        root: dict[str, Any] = {}
        normalized = _normalize_type(content)

        if isinstance(normalized, dict):
            root[tag] = {}
            for key, value in normalized.items():
                root[tag][key] = self._format_content(key, value)[key]
        elif isinstance(normalized, list):
            assert isinstance(normalized, list)
            root[tag] = [self._format_content(tag, item)[tag] for item in normalized]
        else:
            root[tag] = normalized

        return root


@final
class XMLTagBuilder(TagBuilder):
    """Concrete implementation of TagBuilder that produces XML-like formatted output.

    This class converts the normalized content into valid XML string representation.
    It handles nested structures, lists, and basic types while ensuring proper XML formatting.

    Examples:
        >>> builder = XMLTagBuilder('user', {'name': 'John', 'age': 30})
        >>> builder.build()
        '<user><name>John</name><age>30</age></user>'

        >>> builder = XMLTagBuilder('flags', [True, False])
        >>> builder.build()
        '<flags>true</flags><flags>false</flags>'

        >>> builder = XMLTagBuilder('nested', {'user': {'details': {'active': True}}})
        >>> builder.build()
        '<nested><user><details><active>true</active></details></user></nested>'
    """

    def build(self) -> str:
        return self._build_element(self._content)

    def _build_element(self, content: dict[str, Any]) -> str:
        result: list[str] = []

        for tag, value in content.items():
            if isinstance(value, dict):
                nested_content = ''

                for k, v in cast(dict[str, Any], value).items():
                    nested_content += self._build_element({k: v})

                result.append(f'<{tag}>{nested_content}</{tag}>')
            elif isinstance(value, list):
                nested_content = ''.join(
                    f'<{tag}>{self._format_value(item)}</{tag}>' for item in cast(list[str], value)
                )
                result.append(nested_content)
            else:
                result.append(f'<{tag}>{self._format_value(value)}</{tag}>')

        return ''.join(result)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, dict):
            return self._build_element(cast(dict[str, Any], value))
        elif isinstance(value, list):
            return ''.join(f'{self._format_value(item)}' for item in cast(list[Any], value))
        else:
            return str(value)
