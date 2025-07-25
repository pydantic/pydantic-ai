from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)

from pydantic.fields import ComputedFieldInfo, FieldInfo


def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
    add_attributes: bool = False,
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
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.
        add_attributes: Whether to include attributes like Pydantic Field attributes (title, description, alias)
            as XML attributes.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

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
    el = _ToXml(data=obj, item_tag=item_tag, none_str=none_str, add_attributes=add_attributes).to_xml(root_tag)
    if root_tag is None and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    data: Any
    item_tag: str
    none_str: str
    add_attributes: bool
    _attributes: dict[str, dict[str, str]] | None = None
    # keep track of class names for dataclasses and Pydantic models in lists
    _element_names: dict[str, str] | None = None
    _FIELD_ATTRIBUTES = ('title', 'description', 'alias')

    def to_xml(self, tag: str | None) -> ElementTree.Element:
        return self._to_xml(self.data, tag)

    def _to_xml(self, value: Any, tag: str | None, path: str = '') -> ElementTree.Element:
        element = self._create_element(self.item_tag if tag is None else tag, path)
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
            if tag is None and self._element_names and path in self._element_names:
                element = self._create_element(self._element_names[path], path)
            self._mapping_to_xml(element, value, path)  # pyright: ignore[reportUnknownArgumentType]
        elif is_dataclass(value) and not isinstance(value, type):
            self._init_element_names()
            if tag is None:
                element = self._create_element(value.__class__.__name__, path)
            self._mapping_to_xml(element, asdict(value), path)
        elif isinstance(value, BaseModel):
            # before serializing the model and losing all the metadata of other data structures contained in it,
            # we extract all the field attributes and class names
            self._init_attributes()
            self._init_element_names()
            if tag is None:
                element = self._create_element(value.__class__.__name__, path)
            self._mapping_to_xml(element, value.model_dump(mode='python'), path)
        elif isinstance(value, Iterable):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                element.append(self._to_xml(item, None, f'{path}.[]' if path else '[]'))
        else:
            raise TypeError(f'Unsupported type for XML formatting: {type(value)}')
        return element

    def _create_element(self, tag: str, path: str) -> ElementTree.Element:
        element = ElementTree.Element(tag)
        if self._attributes:
            for k, v in self._attributes.get(path, {}).items():
                element.set(k, v)
        return element

    def _init_attributes(self):
        if self.add_attributes and self._attributes is None:
            self._attributes = {}
            self._parse_data_structures(self.data, attributes=self._attributes)

    def _init_element_names(self):
        if self._element_names is None:
            self._element_names = {}
            self._parse_data_structures(self.data, element_names=self._element_names)

    def _mapping_to_xml(
        self,
        element: ElementTree.Element,
        mapping: Mapping[Any, Any],
        path: str = '',
    ) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self._to_xml(value, key, f'{path}.{key}' if path else key))

    @classmethod
    def _parse_data_structures(
        cls,
        value: Any,
        element_names: dict[str, str] | None = None,
        attributes: dict[str, dict[str, str]] | None = None,
        path: str = '',
    ):
        """Parse data structures as dataclasses or Pydantic models to extract element names and attributes."""
        if value is None or isinstance(value, (str, int, float, date, bytearray, bytes, bool)):
            return
        elif isinstance(value, Mapping):
            for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
                cls._parse_data_structures(v, element_names, attributes, f'{path}.{k}' if path else f'{k}')
        elif is_dataclass(value) and not isinstance(value, type):
            if element_names is not None:
                element_names[path] = value.__class__.__name__
            for k, v in asdict(value).items():
                cls._parse_data_structures(v, element_names, attributes, f'{path}.{k}' if path else f'{k}')
        elif isinstance(value, BaseModel):
            if element_names is not None:
                element_names[path] = value.__class__.__name__
            for model_fields in (value.__class__.model_fields, value.__class__.model_computed_fields):
                for field, info in model_fields.items():
                    new_path = f'{path}.{field}' if path else field
                    if (attributes is not None) and (isinstance(info, ComputedFieldInfo) or not info.exclude):
                        attributes.update(cls._extract_attributes(info, new_path))
                    cls._parse_data_structures(getattr(value, field), element_names, attributes, new_path)
        elif isinstance(value, Iterable):
            new_path = f'{path}.[]' if path else '[]'
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                cls._parse_data_structures(item, element_names, attributes, new_path)

    @classmethod
    def _extract_attributes(cls, info: FieldInfo | ComputedFieldInfo, path: str) -> dict[str, dict[str, str]]:
        ret: dict[str, dict[str, str]] = {}
        attributes = {}
        for attr in cls._FIELD_ATTRIBUTES:
            attr_value = getattr(info, attr, None)
            if attr_value is not None:
                attributes[attr] = str(attr_value)
        if attributes:
            ret[path] = attributes
        return ret


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
