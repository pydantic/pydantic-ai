import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, final, override

import pytest
from pydantic import BaseModel

from pydantic_ai.prompt import TagBuilder, XMLTagBuilder


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


class Location(BaseModel):
    country: str
    city: str


class UserInfo(BaseModel):
    age: int
    location: Location


class UserContext(BaseModel):
    user_name: str
    user_info: UserInfo


@dataclass
class LocationDataclass:
    country: str
    city: str


@dataclass
class UserInfoDataclass:
    age: int
    location: LocationDataclass


@dataclass
class UserContextDataclass:
    user_name: str
    user_info: UserInfoDataclass


class TestComplexContentFormatting:
    @pytest.mark.parametrize(
        'content',
        (
            {
                'user_name': 'John',
                'user_info': {
                    'age': 42,
                    'location': {
                        'country': 'UK',
                        'city': 'London',
                    },
                },
            },
            UserContext(
                user_name='John',
                user_info=UserInfo(age=42, location=Location(country='UK', city='London')),
            ),
            UserContextDataclass(
                user_name='John',
                user_info=UserInfoDataclass(age=42, location=LocationDataclass(country='UK', city='London')),
            ),
        ),
    )
    def test_nested_dict(self, content: Any) -> None:
        builder = DummyTagBuilder('context', content)
        got = json.loads(builder.build())

        assert got['context']['user_name'] == 'John'
        assert got['context']['user_info']['age'] == 42
        assert got['context']['user_info']['location']['country'] == 'UK'

    def test_iterable_with_dict(self) -> None:
        builder = DummyTagBuilder(
            'rules',
            {
                'general': ['rule #1', 'rule #2'],
                'specific': [
                    {'name': 'rule #3', 'description': 'Some description'},
                    {'name': 'rule #4', 'description': 'Another description'},
                ],
            },
        )
        got = json.loads(builder.build())

        assert got['rules']['general'][0] == 'rule #1'
        assert got['rules']['specific'][0]['name'] == 'rule #3'

    @pytest.mark.parametrize(
        'content',
        (
            (
                UserContext(
                    user_name='John',
                    user_info=UserInfo(age=42, location=Location(country='UK', city='London')),
                ),
                UserContext(
                    user_name='Jane',
                    user_info=UserInfo(age=24, location=Location(country='USA', city='New York')),
                ),
            ),
            [
                UserContextDataclass(
                    user_name='John',
                    user_info=UserInfoDataclass(age=42, location=LocationDataclass(country='UK', city='London')),
                ),
                UserContextDataclass(
                    user_name='Jane',
                    user_info=UserInfoDataclass(age=24, location=LocationDataclass(country='USA', city='New York')),
                ),
            ],
        ),
    )
    def test_iterable_with_models(self, content: Any) -> None:
        builder = DummyTagBuilder('context', content)
        got = json.loads(builder.build())

        assert got['context'][0]['user_name'] == 'John'
        assert got['context'][1]['user_info']['location']['country'] == 'USA'


class Example(BaseModel):
    text: str
    decision: bool


@dataclass
class ExampleDataclass:
    text: str
    decision: bool


class TestXMLContentFormatting:
    @pytest.mark.parametrize(
        'content',
        (
            {
                'age': 42,
                'location': {
                    'country': 'UK',
                    'city': 'London',
                },
            },
            UserInfo(
                age=42,
                location=Location(country='UK', city='London'),
            ),
            UserInfoDataclass(
                age=42,
                location=LocationDataclass(country='UK', city='London'),
            ),
        ),
    )
    def test_simple_schema(self, content: Any) -> None:
        builder = XMLTagBuilder('context', content)
        got = builder.build()

        assert got == '<context><age>42</age><location><country>UK</country><city>London</city></location></context>'

    @pytest.mark.parametrize(
        'content',
        (
            [
                {'text': 'Example #1', 'decision': True},
                {'text': 'Example #2', 'decision': False},
            ],
            [
                Example(text='Example #1', decision=True),
                Example(text='Example #2', decision=False),
            ],
            [
                ExampleDataclass(text='Example #1', decision=True),
                ExampleDataclass(text='Example #2', decision=False),
            ],
        ),
    )
    def test_list(self, content: Any) -> None:
        builder = XMLTagBuilder('examples', content)
        got = builder.build()

        assert (
            got
            == '<examples><text>Example #1</text><decision>true</decision></examples><examples><text>Example #2</text><decision>false</decision></examples>'
        )

    def test_str(self) -> None:
        rules = ['Rule #1', 'Rule #2']
        builder = XMLTagBuilder('rules', '\n'.join(f'- {rule}' for rule in rules))
        got = builder.build()

        assert got == '<rules>- Rule #1\n- Rule #2</rules>'
