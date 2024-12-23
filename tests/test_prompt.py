from collections.abc import Iterable
import json
from dataclasses import dataclass
from typing import final, override

import pytest
from pydantic import BaseModel

from pydantic_ai.prompt import TagBuilder


@final
class DummyTagBuilder(TagBuilder):
    @override
    def build(self) -> str:
        return json.dumps(self._content)


class DummyModel(BaseModel):
    key01: str
    key02: int


@dataclass
class DummyDataclass:
    key01: str
    key02: int


class DummyClass:
    key = 'Hello, world!'


class TestSimpleContentFormatting:
    def test_str(self) -> None:
        builder = DummyTagBuilder('test', 'Hello, world!')
        content = json.loads(builder.build())

        assert content['test'] == 'Hello, world!'

    def test_dict(self) -> None:
        builder = DummyTagBuilder(
            'test',
            {
                'key01': 'value',
                'key02': 42,
            },
        )
        content = json.loads(builder.build())

        assert content['test']['key01'] == 'value'
        assert content['test']['key02'] == 42

    def test_base_model(self) -> None:
        builder = DummyTagBuilder('test', DummyModel(key01='value', key02=42))
        content = json.loads(builder.build())

        assert content['test']['key01'] == 'value'
        assert content['test']['key02'] == 42

    def test_dataclass(self) -> None:
        builder = DummyTagBuilder('test', DummyDataclass(key01='value', key02=42))
        content = json.loads(builder.build())

        assert content['test']['key01'] == 'value'
        assert content['test']['key02'] == 42

    @pytest.mark.parametrize(
        'iterable_content',
        [
            ['Element 1', 'Element 2', 'Element 3'],
            ('Element 1', 'Element 2', 'Element 3'),
            {'Element 1', 'Element 2', 'Element 3'},
        ],
    )
    def test_iterable(self, iterable_content: Iterable[str]) -> None:
        builder = DummyTagBuilder('test', iterable_content)
        content = json.loads(builder.build())

        assert len(content['test']) == 3

    def test_unsupported_content(self) -> None:
        with pytest.raises(TypeError, match='Unsupported content type'):
            DummyTagBuilder('test', DummyClass())


# class TestXMLEncoder:
#     pass
