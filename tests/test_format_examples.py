from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from pydantic_ai.format_examples import format_examples


def test_format_examples_with_dict():
    dict_example = {
        'text': "I absolutely love this car! It's smooth and efficient.",
        'sentiment': 'Positive',
    }
    expected_output = (
        '<example>\n'
        "  <text>I absolutely love this car! It's smooth and efficient.</text>\n"
        '  <sentiment>Positive</sentiment>\n'
        '</example>'
    )
    assert format_examples([dict_example], dialect='xml') == expected_output


def test_format_examples_with_pydantic_model():
    class PydanticExample(BaseModel):
        text: str
        sentiment: str

    pydantic_example = PydanticExample(
        text='The design of this model is terrible, and the fuel efficiency is poor.',
        sentiment='Negative',
    )
    expected_output = (
        '<example>\n'
        '  <text>The design of this model is terrible, and the fuel efficiency is poor.</text>\n'
        '  <sentiment>Negative</sentiment>\n'
        '</example>'
    )
    assert format_examples([pydantic_example], dialect='xml') == expected_output


def test_format_examples_with_dataclass():
    @dataclass
    class DataclassExample:
        text: str
        sentiment: str

    dataclass_example = DataclassExample(
        text='The car is okay, nothing special. Just an average experience.',
        sentiment='Neutral',
    )
    expected_output = (
        '<example>\n'
        '  <text>The car is okay, nothing special. Just an average experience.</text>\n'
        '  <sentiment>Neutral</sentiment>\n'
        '</example>'
    )
    assert format_examples([dataclass_example], dialect='xml') == expected_output


def test_format_examples_with_multiple_examples():
    dict_example = {
        'text': "I absolutely love this car! It's smooth and efficient.",
        'sentiment': 'Positive',
    }

    class PydanticExample(BaseModel):
        text: str
        sentiment: str

    pydantic_example = PydanticExample(
        text='The design of this model is terrible, and the fuel efficiency is poor.',
        sentiment='Negative',
    )

    @dataclass
    class DataclassExample:
        text: str
        sentiment: str

    dataclass_example = DataclassExample(
        text='The car is okay, nothing special. Just an average experience.',
        sentiment='Neutral',
    )

    examples_combined = [dict_example, pydantic_example, dataclass_example]

    expected_output = (
        '<example>\n'
        "  <text>I absolutely love this car! It's smooth and efficient.</text>\n"
        '  <sentiment>Positive</sentiment>\n'
        '</example>\n'
        '<example>\n'
        '  <text>The design of this model is terrible, and the fuel efficiency is poor.</text>\n'
        '  <sentiment>Negative</sentiment>\n'
        '</example>\n'
        '<example>\n'
        '  <text>The car is okay, nothing special. Just an average experience.</text>\n'
        '  <sentiment>Neutral</sentiment>\n'
        '</example>'
    )
    assert format_examples(examples_combined, dialect='xml') == expected_output


def test_format_examples_with_empty_list():
    assert format_examples([], dialect='xml') == ''


def test_format_examples_with_invalid_data_type():
    with pytest.raises(TypeError, match="example:1 of <class 'int'> type not allowed for xml conversion"):
        format_examples([1], dialect='xml')


def test_format_examples_with_values_as_list():
    dict_example = {
        'texts': [
            "I absolutely love this car! It's smooth and efficient.",
            'The design of this model is terrible, and the fuel efficiency is poor.',
            'The car is okay, nothing special. Just an average experience.',
        ],
        'sentiments': ['Positive', 'Negative', 'Neutral'],
    }
    expected_output = (
        '<example>\n'
        '  <texts>\n'
        "    <item>I absolutely love this car! It's smooth and efficient.</item>\n"
        '    <item>The design of this model is terrible, and the fuel efficiency is poor.</item>\n'
        '    <item>The car is okay, nothing special. Just an average experience.</item>\n'
        '  </texts>\n'
        '  <sentiments>\n'
        '    <item>Positive</item>\n'
        '    <item>Negative</item>\n'
        '    <item>Neutral</item>\n'
        '  </sentiments>\n'
        '</example>'
    )
    assert format_examples([dict_example], dialect='xml') == expected_output


def test_format_examples_with_values_as_other_types():
    class math_example(BaseModel):
        math_question: str
        output: int

    other_examples = [
        {
            'fact': 'Is the capital of Italy Rome?',
            'output': True,
        },
        {
            'fact': 'Is the capital of France London?',
            'output': False,
        },
        math_example(
            math_question='What is 2+2?',
            output=4,
        ),
    ]

    expected_output = (
        '<example>\n'
        '  <fact>Is the capital of Italy Rome?</fact>\n'
        '  <output>True</output>\n'
        '</example>\n'
        '<example>\n'
        '  <fact>Is the capital of France London?</fact>\n'
        '  <output>False</output>\n'
        '</example>\n'
        '<example>\n'
        '  <math_question>What is 2+2?</math_question>\n'
        '  <output>4</output>\n'
        '</example>'
    )

    assert format_examples(other_examples, dialect='xml') == expected_output
