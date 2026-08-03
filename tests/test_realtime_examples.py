"""Unit tests for the realtime examples' own helpers.

`test_examples.py` runs the documented snippets; these cover the parts of the runnable examples that
no snippet reaches — the playback buffer's eviction bounds and the camera server's origin check —
because both are load-bearing and neither is exercised by simply importing the module.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from .conftest import try_import

with try_import() as imports_successful:
    from examples.pydantic_ai_examples import realtime_voice
    from examples.pydantic_ai_examples.realtime_camera import app as realtime_camera

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='extras not installed'),
]


def test_playback_buffer_evicts_carry_before_adding_audio() -> None:
    playback = realtime_voice.PlaybackBuffer(max_bytes=6)
    playback.add(b'abcdef')
    playback.fill(bytearray(2))

    playback.add(b'ghij')
    output = bytearray(6)
    playback.fill(output)

    assert output == b'efghij'


def test_playback_buffer_truncates_oversized_chunk() -> None:
    playback = realtime_voice.PlaybackBuffer(max_bytes=4)
    playback.add(b'abcdef')
    output = bytearray(4)
    playback.fill(output)

    assert output == b'cdef'


def test_playback_buffer_new_turn_discards_old_audio() -> None:
    playback = realtime_voice.PlaybackBuffer(max_bytes=8)
    playback.start_turn()
    playback.add(b'old')

    playback.start_turn()
    playback.add(b'new')
    output = bytearray(3)
    playback.fill(output)

    assert output == b'new'


def test_playback_buffer_interrupt_tracks_active_turn_during_underrun() -> None:
    playback = realtime_voice.PlaybackBuffer(max_bytes=8)
    playback.start_turn()
    playback.add(b'ab')
    playback.fill(bytearray(2))

    assert playback.interrupt() == 0
    assert playback.interrupt() is None


def test_camera_websocket_origin_requires_loopback_host() -> None:
    assert realtime_camera._same_origin(  # pyright: ignore[reportPrivateUsage]
        Mock(headers={'origin': 'http://localhost:8000', 'host': 'localhost:8000'})
    )
    assert not realtime_camera._same_origin(  # pyright: ignore[reportPrivateUsage]
        Mock(headers={'origin': 'http://attacker.example:8000', 'host': 'attacker.example:8000'})
    )


async def test_camera_defaults_are_safe_to_embed_in_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime_camera, 'VOICE', '</script><script>alert(1)</script>')

    response = await realtime_camera.index()
    html = bytes(response.body).decode()

    assert '</script><script>alert(1)</script>' not in html
    assert r'\u003c/script\u003e\u003cscript\u003ealert(1)\u003c/script\u003e' in html
