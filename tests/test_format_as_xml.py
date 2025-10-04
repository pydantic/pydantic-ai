from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field, computed_field
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import Self

from pydantic_ai import format_as_xml


@dataclass
class ExampleDataclass:
    name: str
    age: int


class ExamplePydanticModel(BaseModel):
    name: str
    age: int


class ExampleEnum(Enum):
    FOO = 1
    BAR = 2


class ExampleStrEnum(str, Enum):
    FOO = 'foo'
    BAR = 'bar'


class ExamplePydanticFields(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(description='Years', title='Age', default=18)
    height: float = Field(description="The person's height", exclude=True)
    children: list[Self] | None = Field(title='child', alias='child_list', default=None)

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
        pytest.param(ExampleEnum.FOO, snapshot('<examples>ExampleEnum.FOO</examples>'), id='enum'),
        pytest.param(ExampleStrEnum.FOO, snapshot('<examples>foo</examples>'), id='str enum'),
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
def test_root_tag(input_obj: Any, output: str):
    assert format_as_xml(input_obj, root_tag='examples', item_tag='example', include_field_info=False) == output
    assert format_as_xml(input_obj, root_tag='examples', item_tag='example', include_field_info='once') == output


@pytest.mark.parametrize(
    'input_obj,use_fields,output',
    [
        pytest.param(
            ExamplePydanticFields(
                name='John',
                age=42,
                height=160.0,
                child_list=[
                    ExamplePydanticFields(name='Liam', height=150),
                    ExamplePydanticFields(name='Alice', height=160),
                ],
            ),
            'once',
            snapshot("""\
<name description="The person's name">John</name>
<age title="Age" description="Years">42</age>
<children title="child">
  <ExamplePydanticFields>
    <name>Liam</name>
    <age>18</age>
    <children>null</children>
    <location title="Location">null</location>
  </ExamplePydanticFields>
  <ExamplePydanticFields>
    <name>Alice</name>
    <age>18</age>
    <children>null</children>
    <location>null</location>
  </ExamplePydanticFields>
</children>
<location>Australia</location>\
"""),
            id='pydantic model with fields',
        ),
        pytest.param(
            [
                ExamplePydanticFields(
                    name='John',
                    age=42,
                    height=160.0,
                    child_list=[
                        ExamplePydanticFields(name='Liam', height=150),
                        ExamplePydanticFields(name='Alice', height=160),
                    ],
                )
            ],
            'once',
            snapshot("""\
<ExamplePydanticFields>
  <name description="The person's name">John</name>
  <age title="Age" description="Years">42</age>
  <children title="child">
    <ExamplePydanticFields>
      <name>Liam</name>
      <age>18</age>
      <children>null</children>
      <location title="Location">null</location>
    </ExamplePydanticFields>
    <ExamplePydanticFields>
      <name>Alice</name>
      <age>18</age>
      <children>null</children>
      <location>null</location>
    </ExamplePydanticFields>
  </children>
  <location>Australia</location>
</ExamplePydanticFields>\
"""),
            id='list[pydantic model with fields]',
        ),
        pytest.param(
            ExamplePydanticFields(
                name='John',
                age=42,
                height=160.0,
                child_list=[
                    ExamplePydanticFields(name='Liam', height=150),
                    ExamplePydanticFields(name='Alice', height=160),
                ],
            ),
            False,
            snapshot("""\
<name>John</name>
<age>42</age>
<children>
  <ExamplePydanticFields>
    <name>Liam</name>
    <age>18</age>
    <children>null</children>
    <location>null</location>
  </ExamplePydanticFields>
  <ExamplePydanticFields>
    <name>Alice</name>
    <age>18</age>
    <children>null</children>
    <location>null</location>
  </ExamplePydanticFields>
</children>
<location>Australia</location>\
"""),
            id='pydantic model without fields',
        ),
    ],
)
def test_fields(input_obj: Any, use_fields: bool, output: str):
    assert format_as_xml(input_obj, include_field_info=use_fields) == output


def test_repeated_field_attributes():
    class DataItem(BaseModel):
        user1: ExamplePydanticFields
        user2: ExamplePydanticFields

    data = ExamplePydanticFields(
        name='John',
        age=42,
        height=160.0,
        child_list=[
            ExamplePydanticFields(name='Liam', height=150),
            ExamplePydanticFields(name='Alice', height=160),
        ],
    )
    assert (
        format_as_xml(data, include_field_info=True)
        == """\
<name description="The person's name">John</name>
<age title="Age" description="Years">42</age>
<children title="child">
  <ExamplePydanticFields>
    <name description="The person's name">Liam</name>
    <age title="Age" description="Years">18</age>
    <children title="child">null</children>
    <location title="Location">null</location>
  </ExamplePydanticFields>
  <ExamplePydanticFields>
    <name description="The person's name">Alice</name>
    <age title="Age" description="Years">18</age>
    <children title="child">null</children>
    <location title="Location">null</location>
  </ExamplePydanticFields>
</children>
<location title="Location">Australia</location>\
"""
    )

    assert (
        format_as_xml(DataItem(user1=data, user2=data.model_copy()), include_field_info=True)
        == """\
<user1>
  <name description="The person's name">John</name>
  <age title="Age" description="Years">42</age>
  <children title="child">
    <ExamplePydanticFields>
      <name description="The person's name">Liam</name>
      <age title="Age" description="Years">18</age>
      <children title="child">null</children>
      <location title="Location">null</location>
    </ExamplePydanticFields>
    <ExamplePydanticFields>
      <name description="The person's name">Alice</name>
      <age title="Age" description="Years">18</age>
      <children title="child">null</children>
      <location title="Location">null</location>
    </ExamplePydanticFields>
  </children>
  <location title="Location">Australia</location>
</user1>
<user2>
  <name description="The person's name">John</name>
  <age title="Age" description="Years">42</age>
  <children title="child">
    <ExamplePydanticFields>
      <name description="The person's name">Liam</name>
      <age title="Age" description="Years">18</age>
      <children title="child">null</children>
      <location title="Location">null</location>
    </ExamplePydanticFields>
    <ExamplePydanticFields>
      <name description="The person's name">Alice</name>
      <age title="Age" description="Years">18</age>
      <children title="child">null</children>
      <location title="Location">null</location>
    </ExamplePydanticFields>
  </children>
  <location title="Location">Australia</location>
</user2>\
"""
    )

    assert (
        format_as_xml(DataItem(user1=data, user2=data.model_copy()), include_field_info='once')
        == """\
<user1>
  <name description="The person's name">John</name>
  <age title="Age" description="Years">42</age>
  <children title="child">
    <ExamplePydanticFields>
      <name>Liam</name>
      <age>18</age>
      <children>null</children>
      <location title="Location">null</location>
    </ExamplePydanticFields>
    <ExamplePydanticFields>
      <name>Alice</name>
      <age>18</age>
      <children>null</children>
      <location>null</location>
    </ExamplePydanticFields>
  </children>
  <location>Australia</location>
</user1>
<user2>
  <name>John</name>
  <age>42</age>
  <children>
    <ExamplePydanticFields>
      <name>Liam</name>
      <age>18</age>
      <children>null</children>
      <location>null</location>
    </ExamplePydanticFields>
    <ExamplePydanticFields>
      <name>Alice</name>
      <age>18</age>
      <children>null</children>
      <location>null</location>
    </ExamplePydanticFields>
  </children>
  <location>Australia</location>
</user2>\
"""
    )


def test_nested_data():
    @dataclass
    class DataItem1:
        id: str | None = None
        source: str = field(default='none', metadata={'description': 'the source', 'date': '19990805'})

    class ModelItem1(BaseModel):
        name: str = Field(description='Name')
        value: int
        items: list[DataItem1] = Field(description='Items')

    @pydantic_dataclass
    class DataItem2:
        model: ModelItem1 = field(metadata={'title': 'the model', 'description': 'info'})
        others: tuple[ModelItem1] | None = None
        count: int = field(default=10, metadata={'info': 'a count'})

    data = {
        'values': [
            DataItem2(
                ModelItem1(name='Alice', value=42, items=[DataItem1('xyz')]),
                (ModelItem1(name='Liam', value=3, items=[]),),
            ),
            DataItem2(
                ModelItem1(
                    name='Bob',
                    value=7,
                    items=[
                        DataItem1('a'),
                        DataItem1(source='xx'),
                    ],
                ),
                count=42,
            ),
        ]
    }

    assert (
        format_as_xml(data, include_field_info='once')
        == """
<values>
  <DataItem2>
    <model title="the model" description="info">
      <name description="Name">Alice</name>
      <value>42</value>
      <items description="Items">
        <DataItem1>
          <id>xyz</id>
          <source description="the source">none</source>
        </DataItem1>
      </items>
    </model>
    <others>
      <ModelItem1>
        <name>Liam</name>
        <value>3</value>
        <items />
      </ModelItem1>
    </others>
    <count>10</count>
  </DataItem2>
  <DataItem2>
    <model>
      <name>Bob</name>
      <value>7</value>
      <items>
        <DataItem1>
          <id>a</id>
          <source>none</source>
        </DataItem1>
        <DataItem1>
          <id>null</id>
          <source>xx</source>
        </DataItem1>
      </items>
    </model>
    <others>null</others>
    <count>42</count>
  </DataItem2>
</values>
""".strip()
    )

    assert (
        format_as_xml(data, include_field_info=False)
        == """
<values>
  <DataItem2>
    <model>
      <name>Alice</name>
      <value>42</value>
      <items>
        <DataItem1>
          <id>xyz</id>
          <source>none</source>
        </DataItem1>
      </items>
    </model>
    <others>
      <ModelItem1>
        <name>Liam</name>
        <value>3</value>
        <items />
      </ModelItem1>
    </others>
    <count>10</count>
  </DataItem2>
  <DataItem2>
    <model>
      <name>Bob</name>
      <value>7</value>
      <items>
        <DataItem1>
          <id>a</id>
          <source>none</source>
        </DataItem1>
        <DataItem1>
          <id>null</id>
          <source>xx</source>
        </DataItem1>
      </items>
    </model>
    <others>null</others>
    <count>42</count>
  </DataItem2>
</values>
""".strip()
    )


@pytest.mark.parametrize(
    'input_obj,output',
    [
        pytest.param('a string', snapshot('<item>a string</item>'), id='string'),
        pytest.param('a <ex>foo</ex>', snapshot('<item>a &lt;ex&gt;foo&lt;/ex&gt;</item>'), id='string'),
        pytest.param(42, snapshot('<item>42</item>'), id='int'),
        pytest.param(
            [1, 2, 3],
            snapshot("""\
<item>1</item>
<item>2</item>
<item>3</item>\
"""),
            id='list[int]',
        ),
        pytest.param(
            [[1, 2], [3]],
            snapshot("""\
<item>
  <item>1</item>
  <item>2</item>
</item>
<item>
  <item>3</item>
</item>\
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
<item>2025-01-01T12:13:00</item>
<item>2025-01-02</item>\
"""),
            id='list[date]',
        ),
        pytest.param(
            [time(12, 30, 45), time(8, 15)],
            snapshot("""\
<item>12:30:45</item>
<item>08:15:00</item>\
"""),
            id='list[time]',
        ),
        pytest.param(
            [timedelta(days=1, hours=2, minutes=30), timedelta(seconds=90)],
            snapshot("""\
<item>1 day, 2:30:00</item>
<item>0:01:30</item>\
"""),
            id='list[timedelta]',
        ),
    ],
)
def test_no_root(input_obj: Any, output: str):
    assert format_as_xml(input_obj) == output


def test_no_indent():
    assert format_as_xml([1, 2, 3], indent=None, root_tag='example') == snapshot(
        '<example><item>1</item><item>2</item><item>3</item></example>'
    )
    assert format_as_xml([1, 2, 3], indent=None) == snapshot('<item>1</item><item>2</item><item>3</item>')


def test_invalid_value():
    with pytest.raises(TypeError, match='Unsupported type'):
        format_as_xml(object())


def test_invalid_key():
    with pytest.raises(TypeError, match='Unsupported key type for XML formatting'):
        format_as_xml({(1, 2): 42})


def test_parse_invalid_value():
    class Invalid(BaseModel):
        name: str = Field(default='Alice', title='Name')
        bad: Any = object()

    with pytest.raises(TypeError, match='Unsupported type'):
        format_as_xml(Invalid(), include_field_info='once')


def test_set():
    assert '<example>1</example>' in format_as_xml({1, 2, 3}, item_tag='example')


def test_custom_null():
    assert format_as_xml(None, none_str='nil') == snapshot('<item>nil</item>')
