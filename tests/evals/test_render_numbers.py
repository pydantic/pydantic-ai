from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.reporting.render_numbers import (
        default_render_duration,
        default_render_duration_diff,
        default_render_number,
        default_render_number_diff,
    )

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


@pytest.mark.parametrize(
    'value,expected',
    [
        (0, snapshot('0')),
        (0.0, snapshot('0.000')),
        (17348, snapshot('17,348')),
        (-17348, snapshot('-17,348')),
        (17347.0, snapshot('17,347.0')),
        (-17347.0, snapshot('-17,347.0')),
        (0.1234, snapshot('0.123')),
        (-0.1234, snapshot('-0.123')),
        (0.1, snapshot('0.100')),
        (-0.1, snapshot('-0.100')),
        (2.0, snapshot('2.00')),
        (12.0, snapshot('12.0')),
        (2398723.123, snapshot('2,398,723.1')),
        (0.00000000000001, snapshot('0.0000000000000100')),
    ],
)
def test_default_render_number(value: float | int, expected: str):
    assert default_render_number(value) == expected


@pytest.mark.parametrize(
    'old,new,expected',
    [
        (3, 3, snapshot(None)),
        (127.3, 127.3, snapshot(None)),
        (3, 4, snapshot('+1')),
        (4, 3, snapshot('-1')),
        (1.0, 1.7, snapshot('+0.7 / +70.0%')),
        (2.5, 1.0, snapshot('-1.5 / -60.0%')),
        (10.023, 10.024, snapshot('+0.001')),
        (1.00024, 1.00023, snapshot('-1e-05')),
        (2.0, 25.0, snapshot('+23.0 / 12.5x')),
        (2.0, -25.0, snapshot('-27.0 / -12.5x')),
        (0.02, 25.0, snapshot('+25.0 / 1,250x')),
        (0.02, -25.0, snapshot('-25.0 / -1,250x')),
        (0.001, 25.0, snapshot('+25.0')),
        (0.0, 25.0, snapshot('+25.0')),
    ],
)
def test_default_render_number_diff(old: int | float, new: int | float, expected: str | None):
    assert default_render_number_diff(old, new) == expected


@pytest.mark.parametrize(
    'value,expected',
    [
        (-123.4567, snapshot('-123.5s')),
        (-0.1234567, snapshot('-123.5ms')),
        (-0.00001234567, snapshot('-12µs')),
        (-0.0000001234567, snapshot('-0.1µs')),
        (-0.00000001234567, snapshot('-0.0µs')),
        (0, snapshot('0s')),
        (0.00000001234567, snapshot('0.0µs')),
        (0.0000001234567, snapshot('0.1µs')),
        (0.000001234567, snapshot('1µs')),
        (0.0001234567, snapshot('123µs')),
        (0.001234567, snapshot('1.2ms')),
        (0.1234567, snapshot('123.5ms')),
        (1.234567, snapshot('1.2s')),
        (12.34567, snapshot('12.3s')),
        (123.4567, snapshot('123.5s')),
        (1234.567, snapshot('1,234.6s')),
        (12345.67, snapshot('12,345.7s')),
        (123456.7, snapshot('123,456.7s')),
    ],
)
def test_default_render_duration(value: float, expected: str):
    assert default_render_duration(value) == expected


@pytest.mark.parametrize(
    'old,new,expected',
    [
        (3, 3, snapshot(None)),
        (127.3, 127.3, snapshot(None)),
        (3, 4, snapshot('+1.0s / +33.3%')),
        (4, 3, snapshot('-1.0s / -25.0%')),
        (1.0, 1.7, snapshot('+700.0ms / +70.0%')),
        (2.5, 1.0, snapshot('-1.5s / -60.0%')),
        (10.023, 10.024, snapshot('+1,000µs')),
        (1.00024, 1.00023, snapshot('-10µs')),
        (2.0, 25.0, snapshot('+23.0s / 12.5x')),
        (2.0, -25.0, snapshot('-27.0s / -12.5x')),
        (0.02, 25.0, snapshot('+25.0s / 1,250x')),
        (0.02, -25.0, snapshot('-25.0s / -1,250x')),
        (0.001, 25.0, snapshot('+25.0s')),
    ],
)
def test_default_render_duration_diff(old: float, new: float, expected: str | None):
    assert default_render_duration_diff(old, new) == expected
