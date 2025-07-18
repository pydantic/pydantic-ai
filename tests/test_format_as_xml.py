from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Self

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, computed_field

from pydantic_ai import format_as_xml


@dataclass
class ExampleDataclass:
    name: str
    age: int


class ExamplePydanticModel(BaseModel):
    name: str
    age: int


class ExamplePydanticModelFields(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description='Years', title='Age', default=18)
    height: float = Field(description="The person's height", exclude=True)
    children: list[Self] | None = Field(alias='child', default=None)

    @computed_field(title='Location')
    def location(self) -> str | None:
        if self.name == 'John':
            return 'Australia'
        return None


@pytest.mark.parametrize(
    'input_obj,output',
    [
        pytest.param('a string', snapshot('<examples>a string</examples>'), id='string'),
        pytest.param(42, snapshot('<examples>42</examples>'), id='int'),
        pytest.param(None, snapshot('<examples>null</examples>'), id='null'),
        pytest.param(
            ExampleDataclass(name='John', age=42),
            snapshot("""\
<examples>
  <name>John</name>
  <age>42</age>
</examples>\
"""),
            id='dataclass',
        ),
        pytest.param(
            ExamplePydanticModel(name='John', age=42),
            snapshot("""\
<examples>
  <name>John</name>
  <age>42</age>
</examples>\
"""),
            id='pydantic model',
        ),
        pytest.param(
            [ExampleDataclass(name='John', age=42)],
            snapshot("""\
<examples>
  <ExampleDataclass>
    <name>John</name>
    <age>42</age>
  </ExampleDataclass>
</examples>\
"""),
            id='list[dataclass]',
        ),
        pytest.param(
            [ExamplePydanticModel(name='John', age=42)],
            snapshot("""\
<examples>
  <ExamplePydanticModel>
    <name>John</name>
    <age>42</age>
  </ExamplePydanticModel>
</examples>\
"""),
            id='list[pydantic model]',
        ),
        pytest.param(
            [1, 2, 3],
            snapshot("""\
<examples>
  <example>1</example>
  <example>2</example>
  <example>3</example>
</examples>\
"""),
            id='list[int]',
        ),
        pytest.param(
            (1, 'x'),
            snapshot("""\
<examples>
  <example>1</example>
  <example>x</example>
</examples>\
"""),
            id='tuple[int,str]',
        ),
        pytest.param(
            [[1, 2], [3]],
            snapshot("""\
<examples>
  <example>
    <example>1</example>
    <example>2</example>
  </example>
  <example>
    <example>3</example>
  </example>
</examples>\
"""),
            id='list[list[int]]',
        ),
        pytest.param(
            {'x': 1, 'y': 3, 3: 'z', 4: {'a': -1, 'b': -2}},
            snapshot("""\
<examples>
  <x>1</x>
  <y>3</y>
  <3>z</3>
  <4>
    <a>-1</a>
    <b>-2</b>
  </4>
</examples>\
"""),
            id='dict',
        ),
    ],
)
def test(input_obj: Any, output: str):
    assert format_as_xml(input_obj) == output


@pytest.mark.parametrize(
    'input_obj,use_fields,output',
    [
        pytest.param(
            ExamplePydanticModelFields(
                name='John',
                age=42,
                height=160.0,
                child=[
                    ExamplePydanticModelFields(name='Liam', height=150),
                    ExamplePydanticModelFields(name='Alice', height=160),
                ],
            ),
            True,
            snapshot("""\
<examples>
  <name description="The person's name">John</name>
  <age title="Age" description="Years">42</age>
  <children alias="child">
    <ExamplePydanticModelFields>
      <name description="The person's name">Liam</name>
      <age title="Age" description="Years">18</age>
      <children alias="child">null</children>
      <location title="Location">null</location>
    </ExamplePydanticModelFields>
    <ExamplePydanticModelFields>
      <name description="The person's name">Alice</name>
      <age title="Age" description="Years">18</age>
      <children alias="child">null</children>
      <location title="Location">null</location>
    </ExamplePydanticModelFields>
  </children>
  <location title="Location">Australia</location>
</examples>\
"""),
            id='pydantic model with fields',
        ),
        pytest.param(
            [
                ExamplePydanticModelFields(
                    name='John',
                    age=42,
                    height=160.0,
                    child=[
                        ExamplePydanticModelFields(name='Liam', height=150),
                        ExamplePydanticModelFields(name='Alice', height=160),
                    ],
                )
            ],
            True,
            snapshot("""\
<examples>
  <ExamplePydanticModelFields>
    <name description="The person's name">John</name>
    <age title="Age" description="Years">42</age>
    <children alias="child">
      <ExamplePydanticModelFields>
        <name description="The person's name">Liam</name>
        <age title="Age" description="Years">18</age>
        <children alias="child">null</children>
        <location title="Location">null</location>
      </ExamplePydanticModelFields>
      <ExamplePydanticModelFields>
        <name description="The person's name">Alice</name>
        <age title="Age" description="Years">18</age>
        <children alias="child">null</children>
        <location title="Location">null</location>
      </ExamplePydanticModelFields>
    </children>
    <location title="Location">Australia</location>
  </ExamplePydanticModelFields>
</examples>\
"""),
            id='list[pydantic model with fields]',
        ),
        pytest.param(
            ExamplePydanticModelFields(
                name='John',
                age=42,
                height=160.0,
                child=[
                    ExamplePydanticModelFields(name='Liam', height=150),
                    ExamplePydanticModelFields(name='Alice', height=160),
                ],
            ),
            False,
            snapshot("""\
<examples>
  <name>John</name>
  <age>42</age>
  <children>
    <ExamplePydanticModelFields>
      <name>Liam</name>
      <age>18</age>
      <children>null</children>
      <location>null</location>
    </ExamplePydanticModelFields>
    <ExamplePydanticModelFields>
      <name>Alice</name>
      <age>18</age>
      <children>null</children>
      <location>null</location>
    </ExamplePydanticModelFields>
  </children>
  <location>Australia</location>
</examples>\
"""),
            id='pydantic model without fields',
        ),
    ],
)
def test_fields(input_obj: Any, use_fields: bool, output: str):
    assert format_as_xml(input_obj, fields_attributes=use_fields) == output


@pytest.mark.parametrize(
    'input_obj,output',
    [
        pytest.param('a string', snapshot('<examples>a string</examples>'), id='string'),
        pytest.param('a <ex>foo</ex>', snapshot('<examples>a &lt;ex&gt;foo&lt;/ex&gt;</examples>'), id='string'),
        pytest.param(42, snapshot('<examples>42</examples>'), id='int'),
        pytest.param(
            [1, 2, 3],
            snapshot("""\
<example>1</example>
<example>2</example>
<example>3</example>\
"""),
            id='list[int]',
        ),
        pytest.param(
            [[1, 2], [3]],
            snapshot("""\
<example>
  <example>1</example>
  <example>2</example>
</example>
<example>
  <example>3</example>
</example>\
"""),
            id='list[list[int]]',
        ),
        pytest.param(
            {'binary': b'my bytes', 'barray': bytearray(b'foo')},
            snapshot("""\
<binary>my bytes</binary>
<barray>foo</barray>\
"""),
            id='dict[str, bytes]',
        ),
        pytest.param(
            [datetime(2025, 1, 1, 12, 13), date(2025, 1, 2)],
            snapshot("""\
<example>2025-01-01T12:13:00</example>
<example>2025-01-02</example>\
"""),
            id='list[date]',
        ),
    ],
)
def test_no_root(input_obj: Any, output: str):
    assert format_as_xml(input_obj, include_root_tag=False) == output


def test_no_indent():
    assert format_as_xml([1, 2, 3], indent=None) == snapshot(
        '<examples><example>1</example><example>2</example><example>3</example></examples>'
    )
    assert format_as_xml([1, 2, 3], indent=None, include_root_tag=False) == snapshot(
        '<example>1</example><example>2</example><example>3</example>'
    )


def test_invalid_value():
    with pytest.raises(TypeError, match='Unsupported type'):
        format_as_xml(object())


def test_invalid_key():
    with pytest.raises(TypeError, match='Unsupported key type for XML formatting'):
        format_as_xml({(1, 2): 42})


def test_set():
    assert '<example>1</example>' in format_as_xml({1, 2, 3})


def test_custom_null():
    assert format_as_xml(None, none_str='nil') == snapshot('<examples>nil</examples>')
