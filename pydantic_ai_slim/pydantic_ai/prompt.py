from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from functools import partial
from typing import Any, Literal, Union, cast, final
from xml.etree import ElementTree as ET

from pydantic import BaseModel

__all__ = (
    'format_tag',
    'format_context',
    'format_examples',
    'format_rules',
)

Content = Union[str, int, float, bool, dict[str, 'Content'], list['Content']]
XMLContent = Union[ET.Element, list['XMLContent']]
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


def prepare_content(content: Any) -> Content:
    """Format content into a structured dictionary representation.

    This internal function processes the input content and creates a hierarchical dictionary
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
        >>> _format_content('user', 'John')
        {'user': 'John'}

        >>> _format_content('user', {'name': 'John', 'age': 30})
        {'user': {'name': 'John', 'age': 30}}

        >>> _format_content('items', [1, 2, 3])
        {'items': [{'items': 1}, {'items': 2}, {'items': 3}]}
    """
    normalized = _normalize_type(content)

    if isinstance(normalized, dict):
        return {key: prepare_content(value) for key, value in normalized.items()}
    elif isinstance(normalized, list):
        assert isinstance(normalized, list)
        return [prepare_content(item) for item in normalized]
    else:
        return normalized


def _normalize_type(content: Any) -> Content:
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


@final
class XMLTagBuilder:
    """This class converts the content into valid XML string representation.

    Examples:
        >>> builder = XMLTagBuilder('user', {'name': 'John & Jane', 'age': 30})
        >>> builder.build()
        '<user><name>John &amp; Jane</name><age>30</age></user>'

        >>> builder = XMLTagBuilder('flags', [True, False])
        >>> builder.build()
        '<flags>true</flags><flags>false</flags>'

        >>> builder = XMLTagBuilder('nested', {'user': {'details': {'active': True}}})
        >>> builder.build()
        '<nested><user><details><active>true</active></details></user></nested>'
    """

    def __init__(self, tag: str, content: Any):
        self._tag = tag
        self._content = prepare_content(content)

    def build(self, indent: bool = True) -> str:
        """Build the XML string representation of the content.

        Args:
            indent: If True, format the output with 2-space indentation.
                   Default is True.

        Returns:
            A string containing the XML representation of the content.
            Special characters are properly escaped, and boolean values are
            converted to lowercase 'true'/'false'.

        Examples:
            >>> print(XMLTagBuilder('user', {'name': 'John'}).build(indent=False))
            <user><name>John</name></user>

            >>> print(XMLTagBuilder('user', {'name': 'John'}).build())
            <user>
              <name>John</name>
            </user>

            >>> print(XMLTagBuilder('items', [1, 2]).build())
            <items>1</items>
            <items>2</items>
        """
        elements = self._build_element(self._tag, self._content)

        if isinstance(elements, list):
            if indent:
                return '\n'.join(
                    ET.indent(element, space='  ') or ET.tostring(element, encoding='unicode', method='xml').strip()
                    for element in cast(list[ET.Element], elements)
                )
            return ''.join(
                ET.tostring(element, encoding='unicode', method='xml').strip()
                for element in cast(list[ET.Element], elements)
            )

        if indent:
            ET.indent(elements, space='  ')
        return ET.tostring(elements, encoding='unicode', method='xml').strip()

    def _build_element(self, tag: str, content: Content) -> XMLContent:
        if isinstance(content, list):
            return [self._build_element(tag, item) for item in content]

        element = ET.Element(tag)
        if isinstance(content, dict):
            for key, value in content.items():
                sub_elements = self._build_element(key, value)

                if isinstance(sub_elements, list):
                    for sub_element in cast(list[ET.Element], sub_elements):
                        element.append(sub_element)
                else:
                    element.append(sub_elements)
        else:
            element.text = self._format_value(content)

        return element

    def _format_value(self, value: Content) -> str:
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)
