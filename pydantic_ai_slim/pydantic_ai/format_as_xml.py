from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, is_dataclass
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
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Example:
    ```python
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

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable.
        include_root_tag: Whether to include the root tag in the output
            (The root tag is always included if it includes a body - e.g. when the input is a simple value).
        indent: Indentation string to use for pretty printing.

    Returns: XML representation of the object.
    """
    el = _to_xml(obj, root_tag, item_tag)
    if not include_root_tag and el.text is None:
        return '\n'.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
def _to_xml(value: Any, tag: str, item_tag: str) -> ElementTree.Element:
    element = ElementTree.Element(tag)
    if value is None:
        element.text = 'null'
    elif isinstance(value, str):
        element.text = value
    elif isinstance(value, (bytes, bytearray)):
        element.text = value.decode(errors='ignore')
    elif isinstance(value, (bool, int, float)):
        element.text = str(value)
    elif isinstance(value, date):
        element.text = value.isoformat()
    elif isinstance(value, Mapping):
        _mapping_to_xml(element, value, item_tag)
    elif isinstance(value, Iterable):
        for item in value:
            item_el = _to_xml(item, item_tag, item_tag)
            element.append(item_el)
    elif is_dataclass(value) and not isinstance(value, type):
        dc_dict = asdict(value)
        _mapping_to_xml(element, dc_dict, item_tag)
    elif isinstance(value, BaseModel):
        _mapping_to_xml(element, value.model_dump(mode='python'), item_tag)
    else:
        raise TypeError(f'Unsupported type for XML formatting: {type(value)}')
    return element


def _mapping_to_xml(element: ElementTree.Element, mapping: Mapping[Any, Any], item_tag: str) -> None:
    for key, value in mapping.items():
        if isinstance(key, int):
            key = str(key)
        elif not isinstance(key, str):
            raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
        element.append(_to_xml(value, key, item_tag))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
