from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from itertools import chain
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)

from pydantic.fields import ComputedFieldInfo


def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
    fields_attributes: bool = True,
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
        fields_attributes: Whether to include field attributes (title, description, alias) as attributes
            on the XML elements for Pydantic models.

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
    el = _ToXml(item_tag=item_tag, none_str=none_str, add_fields_attributes=fields_attributes).to_xml(obj, root_tag)
    if root_tag is None and el.text is None:
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
    add_fields_attributes: bool

    def to_xml(self, value: Any, tag: str | None, attributes: dict[str, str] | None = None) -> ElementTree.Element:
        element = ElementTree.Element(self.item_tag if tag is None else tag)
        if attributes is not None:
            for k, v in attributes.items():
                element.set(k, v)
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
            self._mapping_to_xml(
                element=element,
                mapping=self._partial_model_dump(value),
                fields_attributes=self._fields_attributes(value) if self.add_fields_attributes else None,
            )
        elif isinstance(value, Iterable):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                item_el = self.to_xml(item, None)
                element.append(item_el)
        else:
            raise TypeError(f'Unsupported type for XML formatting: {type(value)}')
        return element

    def _mapping_to_xml(
        self,
        element: ElementTree.Element,
        mapping: Mapping[Any, Any],
        fields_attributes: dict[str, dict[str, str]] | None = None,
    ) -> None:
        fields_attributes = fields_attributes or {}
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key, fields_attributes.get(key)))

    @staticmethod
    def _partial_model_dump(model: BaseModel) -> dict[str, Any]:
        """Dump only primitive types in order to keep fields information on models in sub-fields."""
        exclude: set[str] = set()
        for field in chain(
            (k for k, v in model.model_fields.items() if not v.exclude), model.model_computed_fields.keys()
        ):
            value = getattr(model, field)
            if value is not None and (
                not isinstance(value, (str, int, float, date, bytearray, bytes, bool))
                or not (is_dataclass(value) and not isinstance(value, type))
            ):
                exclude.add(field)
        # FIXME we iteratively dump again sub-fields, we could set exclude parameter, but we would lost fields order
        dump: dict[str, Any] = model.model_dump(mode='python')
        # FIXME what is excluded won't follow serialization rules defined in the model
        for field in exclude:
            dump[field] = getattr(model, field)
        return dump

    @staticmethod
    def _fields_attributes(value: BaseModel) -> dict[str, dict[str, str]]:
        """Obtain a map of xml attributes for each Pydantic field."""
        ret: dict[str, dict[str, str]] = {}
        for model_fields in (value.model_fields, value.model_computed_fields):
            for field, info in model_fields.items():
                if isinstance(info, ComputedFieldInfo) or not info.exclude:
                    attributes = {}
                    for attr in ('title', 'description', 'alias'):
                        attr_value = getattr(info, attr, None)
                        if attr_value is not None:
                            attributes[attr] = str(attr_value)
                    if attributes:
                        ret[field] = attributes
        return ret


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
