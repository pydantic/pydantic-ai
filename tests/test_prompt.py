from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai.prompt import XMLTagBuilder, format_context, format_examples, format_rules, prepare_content


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
        content = prepare_content('Hello, world!')

        assert content == 'Hello, world!'

    def test_dict(self) -> None:
        content = prepare_content(
            {
                'key01': 'value',
                'key02': 42,
            },
        )

        assert isinstance(content, dict)
        assert content['key01'] == 'value'
        assert content['key02'] == 42

    def test_base_model(self) -> None:
        content = prepare_content(DummyModel(key01='value', key02=42))

        assert isinstance(content, dict)
        assert content['key01'] == 'value'
        assert content['key02'] == 42

    def test_dataclass(self) -> None:
        content = prepare_content(DummyDataclass(key01='value', key02=42))

        assert isinstance(content, dict)
        assert content['key01'] == 'value'
        assert content['key02'] == 42

    @pytest.mark.parametrize(
        'iterable_content',
        [
            ['Element 1', 'Element 2', 'Element 3'],
            ('Element 1', 'Element 2', 'Element 3'),
            {'Element 1', 'Element 2', 'Element 3'},
        ],
    )
    def test_iterable(self, iterable_content: Iterable[str]) -> None:
        content = prepare_content(iterable_content)

        assert isinstance(content, list)
        assert len(content) == 3

    def test_unsupported_content(self) -> None:
        with pytest.raises(TypeError, match='Unsupported content type'):
            prepare_content(DummyClass())


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
        got = prepare_content(content)

        assert isinstance(got, dict)
        assert isinstance(got['user_info'], dict)
        assert isinstance(got['user_info']['location'], dict)

        assert got['user_name'] == 'John'
        assert got['user_info']['age'] == 42
        assert got['user_info']['location']['country'] == 'UK'

    def test_iterable_with_dict(self) -> None:
        got = prepare_content(
            {
                'general': ['rule #1', 'rule #2'],
                'specific': [
                    {'name': 'rule #3', 'description': 'Some description'},
                    {'name': 'rule #4', 'description': 'Another description'},
                ],
            },
        )

        assert isinstance(got, dict)
        assert isinstance(got['general'], list)
        assert isinstance(got['specific'], list)
        assert isinstance(got['specific'][0], dict)

        assert got['general'][0] == 'rule #1'
        assert got['specific'][0]['name'] == 'rule #3'

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
        got = prepare_content(content)

        assert isinstance(got, list)
        assert isinstance(got[0], dict)
        assert isinstance(got[1], dict)
        assert isinstance(got[1]['user_info'], dict)
        assert isinstance(got[1]['user_info']['location'], dict)

        assert got[0]['user_name'] == 'John'
        assert got[1]['user_info']['location']['country'] == 'USA'


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
        got = builder.build(indent=False)

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
        got = builder.build(indent=False)

        assert (
            got
            == '<examples><text>Example #1</text><decision>true</decision></examples><examples><text>Example #2</text><decision>false</decision></examples>'
        )

    def test_dict_with_list(self) -> None:
        builder = XMLTagBuilder(
            'context',
            {'users': ['John', 'Jane']},
        )
        got = builder.build(indent=False)

        assert got == '<context><users>John</users><users>Jane</users></context>'

    def test_str(self) -> None:
        rules = ['Rule #1', 'Rule #2']
        builder = XMLTagBuilder('rules', '\n'.join(f'- {rule}' for rule in rules))
        got = builder.build(indent=False)

        assert got == '<rules>- Rule #1\n- Rule #2</rules>'

    def test_escaping(self) -> None:
        builder = XMLTagBuilder('user', {'name': '</name>John & Jane', 'age': 30})
        got = builder.build(indent=False)

        assert got == '<user><name>&lt;/name&gt;John &amp; Jane</name><age>30</age></user>'

    def test_indent(self) -> None:
        builder = XMLTagBuilder('user', {'name': 'John', 'age': 30})
        got = builder.build(indent=True)

        assert got == '\n'.join(
            line
            for line in (
                '<user>',
                '  <name>John</name>',
                '  <age>30</age>',
                '</user>',
            )
        )

    def test_indent_list(self) -> None:
        builder = XMLTagBuilder(
            'users',
            ['John', 'Jane'],
        )
        got = builder.build(indent=True)

        assert got == '\n'.join(
            line
            for line in (
                '<users>John</users>',
                '<users>Jane</users>',
            )
        )


@pytest.mark.parametrize(
    ['tag', 'function'],
    (
        ('context', format_context),
        ('examples', format_examples),
        ('rules', format_rules),
    ),
)
def test_aliases(tag: str, function: Callable[[Any], str]) -> None:
    result = function('Hello, world!')

    assert result == f'<{tag}>Hello, world!</{tag}>'
