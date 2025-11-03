from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import date, time, timedelta
from enum import Enum
from typing import Any, Literal
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
    include_field_info: Literal['once'] | bool = False,
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `time`, `timedelta`, `Enum`,
    `Mapping`, `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.
        include_field_info: Whether to include attributes like Pydantic `Field` attributes and dataclasses `field()`
            `metadata` as XML attributes. In both cases the allowed `Field` attributes and `field()` metadata keys are
            `title` and `description`. If a field is repeated in the data (e.g. in a list) by setting `once`
            the attributes are included only in the first occurrence of an XML element relative to the same field.

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
    el = _ToXml(
        data=obj,
        item_tag=item_tag,
        none_str=none_str,
        include_field_info=include_field_info,
    ).to_xml(root_tag)
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
    include_field_info: Literal['once'] | bool
    # a map of Pydantic and dataclasses Field paths to their metadata:
    # a field unique string representation and its class
    _fields_info: dict[str, tuple[str, FieldInfo | ComputedFieldInfo]] = field(default_factory=dict)
    # keep track of fields we have extracted attributes from
    _included_fields: set[str] = field(default_factory=set)
    # keep track of class names for dataclasses and Pydantic models, that occur in lists
    _element_names: dict[str, str] = field(default_factory=dict)
    # flag for parsing dataclasses and Pydantic models once
    _is_info_extracted: bool = False
    _FIELD_ATTRIBUTES = ('title', 'description')

    def to_xml(self, tag: str | None = None) -> ElementTree.Element:
        return self._to_xml(value=self.data, path='', tag=tag)

    def _to_xml(self, value: Any, path: str, tag: str | None = None) -> ElementTree.Element:
        element = self._create_element(self.item_tag if tag is None else tag, path)
        if value is None:
            element.text = self.none_str
        elif isinstance(value, str):
            element.text = value
        elif isinstance(value, bytes | bytearray):
            element.text = value.decode(errors='ignore')
        elif isinstance(value, bool | int | float | Enum):
            element.text = str(value)
        elif isinstance(value, date | time):
            element.text = value.isoformat()
        elif isinstance(value, timedelta):
            element.text = str(value)
        elif isinstance(value, Mapping):
            if tag is None and path in self._element_names:
                element.tag = self._element_names[path]
            self._mapping_to_xml(element, value, path)  # pyright: ignore[reportUnknownArgumentType]
        elif is_dataclass(value) and not isinstance(value, type):
            self._init_structure_info()
            if tag is None:
                element.tag = value.__class__.__name__
            self._mapping_to_xml(element, asdict(value), path)
        elif isinstance(value, BaseModel):
            self._init_structure_info()
            if tag is None:
                element.tag = value.__class__.__name__
            # by dumping the model we loose all metadata in nested data structures,
            # but we have collected it when called _init_structure_info
            self._mapping_to_xml(element, value.model_dump(), path)
        elif isinstance(value, Iterable):
            for n, item in enumerate(value):  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
                element.append(self._to_xml(value=item, path=f'{path}.[{n}]' if path else f'[{n}]'))
        else:
            raise TypeError(f'Unsupported type for XML formatting: {type(value)}')
        return element

    def _create_element(self, tag: str, path: str) -> ElementTree.Element:
        element = ElementTree.Element(tag)
        if path in self._fields_info:
            field_repr, field_info = self._fields_info[path]
            if self.include_field_info and self.include_field_info != 'once' or field_repr not in self._included_fields:
                field_attributes = self._extract_attributes(field_info)
                for k, v in field_attributes.items():
                    element.set(k, v)
                self._included_fields.add(field_repr)
        return element

    def _init_structure_info(self):
        """Create maps with all data information (fields info and class names), if not already created."""
        if not self._is_info_extracted:
            self._parse_data_structures(self.data)
            self._is_info_extracted = True

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
            element.append(self._to_xml(value=value, path=f'{path}.{key}' if path else key, tag=key))

    def _parse_data_structures(
        self,
        value: Any,
        path: str = '',
    ):
        """Parse data structures as dataclasses or Pydantic models to extract element names and attributes."""
        if value is None or isinstance(value, (str | int | float | date | time | timedelta | bytearray | bytes | bool)):
            return
        elif isinstance(value, Mapping):
            for k, v in value.items():  # pyright: ignore[reportUnknownVariableType]
                self._parse_data_structures(v, f'{path}.{k}' if path else f'{k}')
        elif is_dataclass(value) and not isinstance(value, type):
            self._element_names[path] = value.__class__.__name__
            for field in fields(value):
                new_path = f'{path}.{field.name}' if path else field.name
                if self.include_field_info and field.metadata:
                    attributes = {k: v for k, v in field.metadata.items() if k in self._FIELD_ATTRIBUTES}
                    if attributes:
                        field_repr = f'{value.__class__.__name__}.{field.name}'
                        self._fields_info[new_path] = (field_repr, FieldInfo(**attributes))
                self._parse_data_structures(getattr(value, field.name), new_path)
        elif isinstance(value, BaseModel):
            self._element_names[path] = value.__class__.__name__
            for model_fields in (value.__class__.model_fields, value.__class__.model_computed_fields):
                for field, info in model_fields.items():
                    new_path = f'{path}.{field}' if path else field
                    if self.include_field_info and (isinstance(info, ComputedFieldInfo) or not info.exclude):
                        field_repr = f'{value.__class__.__name__}.{field}'
                        self._fields_info[new_path] = (field_repr, info)
                    self._parse_data_structures(getattr(value, field), new_path)
        elif isinstance(value, Iterable):
            for n, item in enumerate(value):  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
                new_path = f'{path}.[{n}]' if path else f'[{n}]'
                self._parse_data_structures(item, new_path)

    @classmethod
    def _extract_attributes(cls, info: FieldInfo | ComputedFieldInfo) -> dict[str, str]:
        return {attr: str(value) for attr in cls._FIELD_ATTRIBUTES if (value := getattr(info, attr, None)) is not None}


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
