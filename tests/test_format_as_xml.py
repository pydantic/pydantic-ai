from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai.format_as_xml import format_as_xml


class ExamplePydanticModel(BaseModel):
    key01: str
    key02: int


@dataclass
class ExampleDataclass:
    key01: str
    key02: int


@pytest.mark.parametrize(
    'input_obj,output',
    [
        pytest.param('a string', snapshot('<examples>a string</examples>'), id='string'),
        pytest.param(42, snapshot('<examples>42</examples>'), id='int'),
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
def test_format_xml(input_obj: Any, output: str):
    assert format_as_xml(input_obj) == output


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
def test_format_xml_no_root(input_obj: Any, output: str):
    assert format_as_xml(input_obj, include_root_tag=False) == output
