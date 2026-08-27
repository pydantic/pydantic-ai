"""Recorded responses must replay without an optional decompressor.

A cassette body stored under `Content-Encoding: br` or `zstd` only decodes when
`brotli`/`brotlicffi` or `zstandard` happens to be importable. Nothing in this
repo asks for either: they arrive through an extra of a transitive dependency,
so an upstream that drops that extra silently takes the decoder away. That is
what #709 was -- `ddgs` 9.16.0 dropped `httpx[brotli, ...]`, and the two agent
integration cassettes stopped decoding on every leg that resolves without the
lock, while the locked matrix stayed green.

`gzip`, `deflate` and `identity` come from the standard library, so a cassette
may keep those.
"""

from __future__ import annotations as _annotations

from collections.abc import Iterable
from pathlib import Path

import pytest
from _pytest.mark import ParameterSet
from pydantic import BaseModel
from vcr.serializers import yamlserializer  # pyright: ignore[reportMissingTypeStubs]

_ROOT = Path(__file__).parent.parent

# `httpx._decoders.SUPPORTED_DECODERS` minus the two entries it pops when their
# optional package is absent: `br` needs `brotli`/`brotlicffi`, `zstd` needs
# `zstandard`. An encoding it cannot look up falls through to `IdentityDecoder`,
# which is what leaves a compressed body sitting in `Response.content`.
_STDLIB_ENCODINGS = frozenset({'gzip', 'deflate', 'identity'})


def _optional_encodings(header_value: str) -> list[str]:
    """The encodings in one `Content-Encoding` value that httpx cannot decode from the stdlib.

    Normalizes the way httpx does -- it reads the header with `split_commas=True`
    and then `.strip().lower()` -- so `gzip, deflate` is two stdlib decoders rather
    than one unrecognized encoding.
    """
    optional: list[str] = []
    for part in header_value.split(','):
        encoding = part.strip().lower()
        if encoding and encoding not in _STDLIB_ENCODINGS:
            optional.append(encoding)
    return optional


class _Request(BaseModel):
    uri: str


class _Response(BaseModel):
    headers: dict[str, list[str]]


class _Interaction(BaseModel):
    request: _Request
    response: _Response


class _Cassette(BaseModel):
    interactions: list[_Interaction]


def _cassettes() -> Iterable[ParameterSet]:
    for directory in ('tests', 'integration_tests'):
        for path in sorted((_ROOT / directory).glob('**/cassettes/**/*.yaml')):
            yield pytest.param(path, id=str(path.relative_to(_ROOT)))


def test_cassettes_discovered() -> None:
    # Guard against a discovery break silently making the check vacuous.
    assert sum(1 for _ in _cassettes()) >= 7


@pytest.mark.parametrize('path', _cassettes())
def test_cassette_replays_without_an_optional_decompressor(path: Path) -> None:
    # `vcr`'s own loader, because a cassette can carry tags `yaml.safe_load` rejects.
    document = yamlserializer.deserialize(path.read_text())  # pyright: ignore[reportUnknownMemberType]
    cassette = _Cassette.model_validate(document)
    needs = [
        f'{interaction.request.uri} responds `Content-Encoding: {encoding}`'
        for interaction in cassette.interactions
        for value in interaction.response.headers.get('content-encoding', [])
        for encoding in _optional_encodings(value)
    ]
    assert not needs, (
        '\n'.join(needs) + f'\nDecode the body in {path.name} and drop the header, or re-record it '
        'without that encoding. Relying on an optional decompressor makes the cassette depend on '
        'whichever extra a transitive dependency happens to pull in.'
    )


def test_optional_encodings_normalizes_like_httpx() -> None:
    assert _optional_encodings('gzip') == []
    assert _optional_encodings(' GZIP , deflate ') == []  # a list value, and httpx lowercases
    assert _optional_encodings('') == []  # no encoding declared is not an unknown one
    assert _optional_encodings('br') == ['br']
    assert _optional_encodings('gzip, zstd') == ['zstd']
