from __future__ import annotations as _annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any, Literal

from pydantic import BaseModel

__all__ = [
    'format_examples',
]


def _to_xml_element(example: dict[str, Any]) -> ET.Element:
    """Convert example to an xml element."""
    element = ET.Element('example')
    for sub_key, sub_value in example.items():
        sub_element = ET.Element(sub_key)
        sub_element.text = str(sub_value)
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
        data (Iterable[Union[dict[str, str], BaseModel, Any]]):
            The input data to be converted to XML.

    Returns:
        str: The indented XML string representation of the input data.
    """
    examples_string: list[str] = []
    for item in data:
        dict_item = _convert_to_dict(item)
        example_element = _to_xml_element(dict_item)
        ET.indent(example_element)
        indented_string = ET.tostring(example_element, encoding='unicode')
        examples_string.append(indented_string)
    return '\n'.join(examples_string)


def format_examples(examples: Iterable[dict[str, str] | BaseModel | Any], dialect: Literal['xml'] = 'xml') -> str:
    """Format a sequence of examples into the specified dialect (e.g., XML).

    Args:
        examples (Iterable[Union[dict[str, str], BaseModel, Any]]):
          A sequence of data items to format.
        dialect (str): The format to use. Currently supports "xml".

    Returns:
        str: The formatted examples.
    """
    if dialect == 'xml':
        return _to_xml(examples)
    else:
        raise ValueError(f'Unsupported dialect: {dialect}')
