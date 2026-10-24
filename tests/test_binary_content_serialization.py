from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, SerializationInfo, TypeAdapter, field_serializer
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic.version import VERSION as pydantic_version
from pydantic_core import PydanticSerializationError

from pydantic_ai.messages import (
    BinaryAudio,
    BinaryContent,
    BinaryImage,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    UserPromptPart,
)

if TYPE_CHECKING:
    from pydantic.main import IncEx


def binary_prompt(content: BinaryContent) -> UserPromptPart:
    return UserPromptPart(['look', content], timestamp=datetime(2026, 10, 23, tzinfo=UTC))


def binary_messages(content: BinaryContent) -> list[ModelMessage]:
    return [ModelRequest(parts=[binary_prompt(content)])]


@pytest.mark.parametrize('identifier', [None, 'file-id'])
@pytest.mark.parametrize(('content_type', 'media_type'), [(BinaryImage, 'image/png'), (BinaryAudio, 'audio/wav')])
def test_binary_subclass_serialization_does_not_repr(
    content_type: type[BinaryContent], media_type: str, identifier: str | None, monkeypatch: pytest.MonkeyPatch
):
    content = content_type(
        b'\x00\xffabc', media_type=media_type, identifier=identifier, vendor_metadata={'detail': 'high'}
    )
    base = BinaryContent(content.data, media_type=media_type, identifier=identifier, vendor_metadata={'detail': 'high'})
    expected_json = ModelMessagesTypeAdapter.dump_json(binary_messages(base))
    expected_python = ModelMessagesTypeAdapter.dump_python(binary_messages(base))
    repr_mock = Mock(return_value='binary repr')
    monkeypatch.setattr(content_type, '__repr__', repr_mock)

    encoded = ModelMessagesTypeAdapter.dump_json(binary_messages(content), warnings='error')
    assert encoded == expected_json
    assert ModelMessagesTypeAdapter.dump_python(binary_messages(content), warnings='error') == expected_python
    repr_mock.assert_not_called()

    restored = ModelMessagesTypeAdapter.validate_json(encoded)
    assert isinstance(restored[0], ModelRequest)
    part = restored[0].parts[0]
    assert isinstance(part, UserPromptPart)
    assert not isinstance(part.content, str)
    binary = part.content[1]
    assert type(binary) is (BinaryImage if media_type.startswith('image/') else BinaryContent)
    assert isinstance(binary, BinaryContent)
    assert binary.data == content.data
    assert binary.identifier == content.identifier
    assert binary.vendor_metadata == {'detail': 'high'}


@pydantic_dataclass
class CustomImage(BinaryImage):
    secret: str = 'private'

    @property
    def identifier(self) -> str:
        return 'custom-id'


FILTERS: list[tuple[IncEx | None, IncEx | None]] = [
    (None, None),
    ({'content': {1: {'data', 'identifier', 'secret'}}}, None),
    (None, {'content': {1: {'data', 'identifier'}}}),
    (None, {'content': {1: {'vendor_metadata': {'hidden'}}}}),
]


@pytest.mark.parametrize(('include', 'exclude'), FILTERS)
@pytest.mark.parametrize('round_trip', [False, True])
@pytest.mark.parametrize('exclude_defaults', [False, True])
def test_binary_subclass_dump_options(
    include: IncEx | None, exclude: IncEx | None, round_trip: bool, exclude_defaults: bool
):
    content = CustomImage(b'\x00\xffabc', media_type='image/png', vendor_metadata={'hidden': 1, 'keep': 2})
    base = BinaryContent(
        content.data, media_type=content.media_type, identifier='custom-id', vendor_metadata=content.vendor_metadata
    )
    for mode in ('python', 'json'):
        assert TypeAdapter(UserPromptPart).dump_python(
            binary_prompt(content),
            mode=mode,
            include=include,
            exclude=exclude,
            round_trip=round_trip,
            exclude_defaults=exclude_defaults,
            by_alias=True,
        ) == TypeAdapter(UserPromptPart).dump_python(
            binary_prompt(base),
            mode=mode,
            include=include,
            exclude=exclude,
            round_trip=round_trip,
            exclude_defaults=exclude_defaults,
            by_alias=True,
            # Pydantic 2.12 warns on partial includes of the exact base dataclass in a union.
            warnings=False,
        )
    assert TypeAdapter(UserPromptPart).dump_json(
        binary_prompt(content),
        include=include,
        exclude=exclude,
        round_trip=round_trip,
        exclude_defaults=exclude_defaults,
        by_alias=True,
    ) == TypeAdapter(UserPromptPart).dump_json(
        binary_prompt(base),
        include=include,
        exclude=exclude,
        round_trip=round_trip,
        exclude_defaults=exclude_defaults,
        by_alias=True,
        warnings=False,
    )


class LazyIdentifierImage(BinaryImage):
    @property
    def identifier(self) -> str:
        raise RuntimeError('identifier accessed')


@pytest.mark.parametrize('exclusion', ['exclude', 'include', 'exclude_computed_fields', 'round_trip'])
def test_binary_subclass_excluded_identifier_is_lazy(exclusion: str):
    content = LazyIdentifierImage(b'abc', media_type='image/png')
    include: IncEx | None = {'content': {1: {'data'}}} if exclusion == 'include' else None
    exclude: IncEx | None = {'content': {1: {'identifier'}}} if exclusion == 'exclude' else None
    for mode in ('python', 'json'):
        TypeAdapter(UserPromptPart).dump_python(
            binary_prompt(content),
            mode=mode,
            include=include,
            exclude=exclude,
            exclude_computed_fields=exclusion == 'exclude_computed_fields',
            round_trip=exclusion == 'round_trip',
        )
    TypeAdapter(UserPromptPart).dump_json(
        binary_prompt(content),
        include=include,
        exclude=exclude,
        exclude_computed_fields=exclusion == 'exclude_computed_fields',
        round_trip=exclusion == 'round_trip',
    )


class ContextMetadata(BaseModel):
    value: str

    @field_serializer('value')
    def serialize_value(self, value: str, info: SerializationInfo) -> str:
        return 'redacted' if info.context else value


def test_binary_subclass_metadata_context():
    content = BinaryImage(b'abc', media_type='image/png', vendor_metadata={'item': ContextMetadata(value='secret')})
    assert TypeAdapter(UserPromptPart).dump_python(
        UserPromptPart([content]),
        mode='json',
        context={'redact': True},
        exclude_none=True,
    )['content'][0]['vendor_metadata'] == {'item': {'value': 'redacted'}}


@pytest.mark.parametrize(
    ('attribute', 'value', 'warns'),
    [
        # Pydantic 2.12 silently accepts an unknown discriminator during serialization.
        ('kind', 'changed-kind', not pydantic_version.startswith('2.12.')),
        ('data', 'invalid-bytes', True),
        ('vendor_metadata', 123, True),
    ],
)
def test_binary_subclass_invalid_field_warnings(
    attribute: str, value: object, warns: bool, monkeypatch: pytest.MonkeyPatch
):
    content = BinaryImage(b'abc', media_type='image/png')
    monkeypatch.setattr(content, attribute, value)
    messages = binary_messages(content)
    if not warns:
        assert ModelMessagesTypeAdapter.dump_json(messages, warnings='error') == ModelMessagesTypeAdapter.dump_json(
            messages
        )
        return
    with pytest.warns(UserWarning, match='Pydantic serializer warnings'):
        encoded = ModelMessagesTypeAdapter.dump_json(messages)
    assert ModelMessagesTypeAdapter.dump_json(messages, warnings=False) == encoded
    with pytest.raises(PydanticSerializationError, match='Pydantic serializer warnings'):
        ModelMessagesTypeAdapter.dump_json(messages, warnings='error')


def test_binary_subclass_metadata_fallback():
    content = BinaryImage(b'abc', media_type='image/png', vendor_metadata={'item': object()})
    assert TypeAdapter(UserPromptPart).dump_python(
        UserPromptPart([content]),
        mode='json',
        fallback=lambda _: 'fallback',
    )['content'][0]['vendor_metadata'] == {'item': 'fallback'}


@pytest.mark.parametrize('serialize_as_any', [False, True])
def test_binary_subclass_explicit_duck_typing(serialize_as_any: bool):
    content = CustomImage(b'abc', media_type='image/png')
    dumped = TypeAdapter(UserPromptPart).dump_python(
        UserPromptPart([content]), mode='json', serialize_as_any=serialize_as_any
    )
    assert ('secret' in dumped['content'][0]) is serialize_as_any
