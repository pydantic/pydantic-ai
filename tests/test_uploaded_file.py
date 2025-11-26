from __future__ import annotations

import hashlib

from pydantic_ai.messages import (
    UploadedFile,
    UserPromptPart,
    _uploaded_file_identifier_source,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models.instrumented import InstrumentationSettings


def test_uploaded_file_identifier_source_prefers_known_fields():
    class WithId:
        id = 'file-id'

    class WithUri:
        uri = 'gs://bucket/file'

    class WithName:
        name = 'named-file'

    class WithRepr:
        def __repr__(self) -> str:
            return 'repr-value'

    assert _uploaded_file_identifier_source('direct-id') == 'direct-id'
    assert _uploaded_file_identifier_source(WithId()) == 'file-id'
    assert _uploaded_file_identifier_source(WithUri()) == 'gs://bucket/file'
    assert _uploaded_file_identifier_source(WithName()) == 'named-file'
    assert _uploaded_file_identifier_source(WithRepr()) == 'repr-value'


def test_uploaded_file_identifier_defaults_to_hash_and_respects_override():
    file_id = 'file-abc'
    uploaded_file = UploadedFile(file=file_id)

    expected = hashlib.sha1(file_id.encode('utf-8')).hexdigest()[:6]
    assert uploaded_file.identifier == expected

    overridden = UploadedFile(file=file_id, identifier='explicit-id')
    assert overridden.identifier == 'explicit-id'


def test_uploaded_file_instrumentation_parts_include_identifier_and_optional_file():
    uploaded_file = UploadedFile(file='file-123')
    part = UserPromptPart(content=[uploaded_file])

    without_content = part.otel_message_parts(InstrumentationSettings(include_content=False))
    assert without_content == [{'type': 'uploaded-file', 'identifier': uploaded_file.identifier}]

    with_content = part.otel_message_parts(InstrumentationSettings(include_content=True))
    assert with_content == [
        {'type': 'uploaded-file', 'identifier': uploaded_file.identifier, 'file': 'file-123'},
    ]
