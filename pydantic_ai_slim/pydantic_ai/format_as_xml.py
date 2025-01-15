from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)


def format_as_xml(
    obj: Any,
    root_tag: str = 'examples',
    item_tag: str = 'example',
    include_root_tag: bool = True,
    none_str: str = 'null',
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        include_root_tag: Whether to include the root tag in the output
            (The root tag is always included if it includes a body - e.g. when the input is a simple value).
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai.format_as_xml import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    el = _ToXml(item_tag=item_tag, none_str=none_str).to_xml(obj, root_tag)
    if not include_root_tag and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        element = ElementTree.Element(self.item_tag if tag is None else tag)
        if value is None:
            element.text = self.none_str
        elif isinstance(value, str):
            element.text = value
        elif isinstance(value, (bytes, bytearray)):
            element.text = value.decode(errors='ignore')
        elif isinstance(value, (bool, int, float)):
            element.text = str(value)
        elif isinstance(value, date):
            element.text = value.isoformat()
        elif isinstance(value, Mapping):
            self._mapping_to_xml(element, value)  # pyright: ignore[reportUnknownArgumentType]
        elif is_dataclass(value) and not isinstance(value, type):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
        elif isinstance(value, BaseModel):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            self._mapping_to_xml(element, value.model_dump(mode='python'))
        elif isinstance(value, Iterable):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                item_el = self.to_xml(item, None)
                element.append(item_el)
        else:
            raise TypeError(f'Unsupported type for XML formatting: {type(value)}')
        return element

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
