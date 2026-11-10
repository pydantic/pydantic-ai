import pytest

from pydantic_evals.reporting.render_numbers import (
    default_render_duration_diff,
    default_render_number_diff,
)


@pytest.mark.parametrize(
    ('old', 'new'),
    [
        (1.0, float('inf')),
        (1.0, float('-inf')),
        (float('inf'), 1.0),
        (float('nan'), 1.0),
        (1.0, float('nan')),
    ],
)
def test_non_finite_duration_diff_matches_number_diff(old, new):
    assert default_render_duration_diff(old, new) is None
    assert default_render_number_diff(old, new) is None


def test_finite_duration_diff_unchanged():
    assert default_render_duration_diff(5.0, 5.0) is None
    assert default_render_duration_diff(1.0, 2.0) is not None
