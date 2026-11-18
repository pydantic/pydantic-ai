"""What the contract's one dimensioned setting means, and how to convert it without surprises.

`timeout` is the only canonical setting carrying a unit, and it is the only one an adapter has to
convert before it can be used: `AbortSignal.timeout`, `httpx`, and most CLI flags want integer
milliseconds. Left to each adapter, that conversion has already drifted three ways -- one truncating,
one rounding, one passing a fraction of a millisecond straight into an API that throws on it -- so
the range and the rounding are defined here, once, and the cores share these vectors.
"""

from __future__ import annotations

import math

MAX_TIMEOUT_MILLISECONDS = 2**31 - 1
"""The largest delay a timer can be given: `2**31 - 1` ms, about 24.9 days.

Not an opinion about sensible timeouts, but the smallest ceiling the targets share. Browsers, Node,
and `AbortSignal.timeout` all take a signed 32-bit millisecond delay, and a larger one silently wraps
or fires immediately -- which is the failure mode a managed setting must never introduce, since it
turns "no real limit" into "cancel at once".
"""

MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MILLISECONDS / 1000
"""`MAX_TIMEOUT_MILLISECONDS` as the seconds the contract states timeouts in."""


def is_representable_timeout(seconds: float) -> bool:
    """Whether a published `timeout` is a request budget an adapter can actually install.

    A timeout is representable when it is a finite, non-negative number of seconds no larger than
    `MAX_TIMEOUT_SECONDS`. Everything else -- a negative
    budget, a `nan` that compares false against every deadline, an `inf` or an oversized value that
    wraps a 32-bit timer, an integer too large to be a float at all -- is refused rather than clamped,
    because each of them would make a request behave in a way nobody published: silently unlimited, or
    cancelled before it was sent.

    `0` is representable and means exactly what it says: a budget of no time at all.
    """
    # Range first, and it is the whole test: `nan` compares false against both bounds and `inf`
    # exceeds the upper one, so nothing infinite survives it. Calling `math.isfinite` first instead
    # would raise `OverflowError` on an int too large to turn into a float -- which is exactly the
    # oversized value this is here to refuse, so refusing it must not be the thing that crashes.
    return 0 <= seconds <= MAX_TIMEOUT_SECONDS


def to_milliseconds(seconds: float) -> int:
    """A representable `timeout` in the integer milliseconds most SDKs and timers want.

    Rounded half up -- `floor(seconds * 1000 + 0.5)` -- which is the one rounding JavaScript's
    `Math.round` and Python agree on for non-negative values, so both cores return the same integer
    for the same published value. (Python's own `round` does not: it rounds halves to even, so
    `round(0.0025 * 1000)` is `2`, not `3`.)

    A *positive* budget never rounds down to `0`: anything under half a millisecond comes back as
    `1`. Zero milliseconds means "already expired" to a timer, so rounding a small positive budget
    into it would turn a very short timeout into a request that cannot be made at all. An exact `0`
    is passed through, since that is what it was published as.

    Raises:
        ValueError: when `seconds` is not representable; see `is_representable_timeout`, which is
            what an adapter should call first so the value is reported under its `OnUnmatched` policy
            rather than raised at conversion time.
    """
    if not is_representable_timeout(seconds):
        raise ValueError(
            f'Timeout {seconds!r} is not a representable request budget: it has to be a finite, '
            f'non-negative number of seconds no larger than {MAX_TIMEOUT_SECONDS}.'
        )
    milliseconds = math.floor(seconds * 1000 + 0.5)
    return max(milliseconds, 1) if seconds > 0 else milliseconds
