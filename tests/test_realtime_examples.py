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


def test_playback_buffer_interrupt_only_reports_dropped_audio() -> None:
    """`interrupt()` reports played milliseconds only when it actually drops unheard audio.

    A turn the user heard in full needs no truncation — reporting one anyway would make the provider
    discard part of a completed reply — so after everything was played, `interrupt()` is a no-op.
    """
    bytes_per_ms = realtime_voice.SAMPLE_RATE * realtime_voice.CHANNELS * 2 // 1000
    playback = realtime_voice.PlaybackBuffer(max_bytes=8 * bytes_per_ms)

    playback.start_turn()
    playback.add(b'\x00' * (2 * bytes_per_ms))
    playback.fill(bytearray(2 * bytes_per_ms))  # the user heard the whole turn
    assert playback.interrupt() is None

    playback.start_turn()
    playback.add(b'\x00' * (4 * bytes_per_ms))
    playback.fill(bytearray(3 * bytes_per_ms))  # barge-in with 1 ms still unheard
    assert playback.interrupt() == 3
    assert playback.interrupt() is None  # already flushed; nothing further to report


def test_camera_websocket_origin_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct loopback and proxied (forwarded-host) origins connect; cross-site and DNS-rebinding don't."""
    same_origin = realtime_camera._same_origin  # pyright: ignore[reportPrivateUsage]

    assert same_origin(Mock(headers={'origin': 'http://localhost:8000', 'host': 'localhost:8000'}))
    # DNS rebinding: an attacker domain resolving to 127.0.0.1 matches `Host` with its own origin,
    # so a bare same-origin comparison is not enough — non-loopback origins need a proxy or allowlist.
    assert not same_origin(Mock(headers={'origin': 'http://attacker.example:8000', 'host': 'attacker.example:8000'}))
    # A reverse proxy that rewrites `Host` forwards the browser-facing host; browsers cannot send
    # `X-Forwarded-Host`, so it proves a proxy hop.
    assert same_origin(
        Mock(
            headers={
                'origin': 'https://app.proxy.example',
                'host': '127.0.0.1:8000',
                'x-forwarded-host': 'app.proxy.example',
            }
        )
    )
    assert not same_origin(
        Mock(
            headers={
                'origin': 'https://evil.example',
                'host': '127.0.0.1:8000',
                'x-forwarded-host': 'app.proxy.example',
            }
        )
    )
    assert not same_origin(Mock(headers={'host': '127.0.0.1:8000'}))
    # Proxies that forward neither `Host` nor `X-Forwarded-Host` are covered by the explicit allowlist.
    monkeypatch.setenv('CAMERA_ALLOWED_ORIGINS', 'https://tunnel.example, https://other.example')
    assert same_origin(Mock(headers={'origin': 'https://tunnel.example', 'host': '127.0.0.1:8000'}))


async def test_camera_defaults_are_safe_to_embed_in_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime_camera, 'VOICE', '</script><script>alert(1)</script>')

    response = await realtime_camera.index()
    html = bytes(response.body).decode()

    assert '</script><script>alert(1)</script>' not in html
    assert r'\u003c/script\u003e\u003cscript\u003ealert(1)\u003c/script\u003e' in html
