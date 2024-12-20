from __future__ import annotations as _annotations

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from pydantic import BaseModel


def format_examples(examples: Sequence[Any], dialect: str = 'xml') -> str:
    """Format a sequence of examples into the specified dialect (e.g., XML).

    Args:
        examples (Sequence[Any]): A sequence of data items to format.
        dialect (str): The format to use. Currently supports "xml".

    Returns:
        str: The formatted examples.
    """
    if dialect == 'xml':
        return XMLConverter.to_xml(examples)
    else:
        raise ValueError(f'Unsupported dialect: {dialect}')


@dataclass
class XMLConverter:
    """Convert the examples to LLM friendly XML format."""

    @classmethod
    def to_xml(cls, data: Sequence[dict[str, str] | BaseModel | Any]) -> str:
        """Converts a list of dictionaries, Pydantic models, or a dataclasses to XML.

        Args:
            data (Sequence[Union[dict[str, str], BaseModel, Any]]):
                The input data to be converted to XML.

        Returns:
            str: The indented XML string representation of the input data.
        """

        def convert_to_element(key: str, value: dict[str, str]) -> ET.Element:
            element = ET.Element(key)
            for sub_key, sub_value in value.items():
                sub_element = ET.Element(sub_key)
                sub_element.text = str(sub_value)
                element.append(sub_element)
            return element

        ## Convert to python dictionaries
        def handle_data(data: BaseModel | dict[str, str] | Any) -> dict[str, str]:
            if isinstance(data, BaseModel):
                return data.model_dump(mode='python')
            elif is_dataclass(data):
                return asdict(data)  # type: ignore
            elif isinstance(data, dict):
                return data  # type: ignore
            else:
                raise ValueError(f'{type(data)} type not allowed as an example')

        root = ET.Element('examples')
        if isinstance(data, list):
            for idx, item in enumerate(data):
                processed_item = handle_data(item)
                root.append(convert_to_element(f'example{idx}', processed_item))
        else:
            raise ValueError(f'Unsupported examples input data type {type(data)} for XML conversion.')

        ET.indent(root)
        return ET.tostring(root, encoding='unicode')
