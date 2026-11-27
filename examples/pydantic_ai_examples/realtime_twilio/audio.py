"""Convert between a phone line's audio and a realtime session's.

Twilio Media Streams carry 8 kHz G.711 mu-law, while a realtime session takes mono PCM16
at its model's `audio_input_sample_rate` and speaks at its `audio_output_sample_rate`
(16 or 24 kHz). This module converts between them with the standard library alone:
mu-law is a 256-entry table, and both session rates are whole multiples of 8 kHz, so
resampling repeats or averages samples rather than needing a filter library.
"""

from __future__ import annotations

import sys
from array import array

LINE_RATE = 8000
"""The sample rate of the phone line's audio."""

_BIAS = 0x84
_CLIP = 32635


def _decode(code: int) -> int:
    inverted = ~code & 0xFF
    exponent = (inverted >> 4) & 0x07
    magnitude = (((inverted & 0x0F) << 3) + _BIAS) << exponent
    return _BIAS - magnitude if inverted & 0x80 else magnitude - _BIAS


def _encode(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    biased = min(abs(sample), _CLIP) + _BIAS
    exponent = biased.bit_length() - 8
    mantissa = (biased >> (exponent + 3)) & 0x0F
    return ~(sign | exponent << 4 | mantissa) & 0xFF


_DECODED = array('h', (_decode(code) for code in range(256)))
_ENCODED = bytes(_encode(sample) for sample in range(-32768, 32768))


def _factor(rate: int) -> int:
    if rate % LINE_RATE:
        raise ValueError(f'{rate} Hz is not a whole multiple of {LINE_RATE} Hz')
    return rate // LINE_RATE


def mulaw_to_pcm16(mulaw: bytes, *, rate: int) -> bytes:
    """Decode 8 kHz mu-law, as Twilio sends it, to PCM16 at `rate`.

    Each line sample is followed by straight-line steps to the next one. The last sample of
    a chunk is held, as the next chunk's first isn't known yet.
    """
    factor = _factor(rate)
    decoded = [_DECODED[code] for code in mulaw]
    out = array('h')
    for current, following in zip(decoded, decoded[1:] + decoded[-1:]):
        out.extend(
            current + (following - current) * step // factor for step in range(factor)
        )
    # Realtime sessions exchange little-endian PCM16.
    if sys.byteorder == 'big':
        out.byteswap()
    return out.tobytes()


def pcm16_to_mulaw(pcm: bytes, *, rate: int) -> bytes:
    """Encode PCM16 at `rate`, as a session speaks it, to 8 kHz mu-law for Twilio to play.

    Each line sample is the average of the samples it replaces, which also filters out the
    frequencies 8 kHz can't carry. A partial group at the end of a chunk is dropped.
    """
    factor = _factor(rate)
    samples = array('h')
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])
    if sys.byteorder == 'big':
        samples.byteswap()
    return bytes(
        _ENCODED[sum(samples[start : start + factor]) // factor + 32768]
        for start in range(0, len(samples) - factor + 1, factor)
    )
