from __future__ import annotations as _annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any, Literal
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_examples',)


def format_examples(examples: Iterable[dict[str, str] | BaseModel | Any], dialect: Literal['xml'] = 'xml') -> str:
    """Format a sequence of examples into the specified dialect (e.g., XML).

    Args:
        examples: A sequence of data items to format.
        dialect: The format to use. Currently supports "xml".

    Returns:
        The formatted examples.
    """
    if dialect == 'xml':
        return _to_xml(examples)
    else:
        raise ValueError(f'Unsupported dialect: {dialect}')


def _serialize_to_xml(key: str, value: Any) -> ElementTree.Element:
    """Serialize a value to XML based on its type."""
    element = ElementTree.Element(key)
    if isinstance(value, (str, bool, int, float)) or value is None:
        element.text = '' if value is None else str(value)
    elif isinstance(value, dict):
        for sub_key, sub_value in value.items():  # type: ignore
            element.append(_serialize_to_xml(sub_key, sub_value))  # type: ignore
    elif isinstance(value, (list, tuple)):
        for item in value:  # type: ignore
            element.append(_serialize_to_xml('item', item))
    elif is_dataclass(value) or isinstance(value, BaseModel):
        dict_data = _convert_to_dict(value)
        return _serialize_to_xml(key, dict_data)
    else:
        raise TypeError(f'Unsupported type for serialization: {type(value)}')
    return element


def _to_xml_element(example: dict[str, Any]) -> ElementTree.Element:
    """Convert example to an xml element."""
    element = ElementTree.Element('example')
    for sub_key, sub_value in example.items():
        sub_element = _serialize_to_xml(sub_key, sub_value)
        element.append(sub_element)
    return element


def _convert_to_dict(data: BaseModel | dict[str, str] | Any) -> dict[str, Any]:
    """Convert Pydantic model or dataclass data into a dictionary."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode='python')
    elif is_dataclass(data):
        return asdict(data)  # type: ignore
    elif isinstance(data, dict):
        return data  # type: ignore
    else:
        raise TypeError(f'example:{data} of {type(data)} type not allowed for xml conversion')


def _to_xml(data: Iterable[dict[str, str] | BaseModel | Any]) -> str:
    """Converts a list of dictionaries, Pydantic models, or a dataclasses to XML.

    Args:
        data: The input data to be converted to XML.

    Returns:
        The indented XML string representation of the input data.
    """
    examples_string: list[str] = []
    for item in data:
        dict_item = _convert_to_dict(item)
        example_element = _to_xml_element(dict_item)
        ElementTree.indent(example_element, space='  ')
        indented_string = ElementTree.tostring(example_element, encoding='unicode')
        examples_string.append(indented_string)
    return '\n'.join(examples_string)
