from inline_snapshot._code_repr import mocked_code_repr  # pyright: ignore[reportUnknownVariableType]

from tests.conftest import SNAPSHOT_BYTES_COLLAPSE_THRESHOLD


def test_long_bytes_are_collapsed_in_snapshots():
    long_value = b'x' * (SNAPSHOT_BYTES_COLLAPSE_THRESHOLD + 1)

    assert mocked_code_repr(long_value) == 'IsBytes()'


def test_short_bytes_keep_full_repr():
    short_value = b'hello world'

    assert mocked_code_repr(short_value) == repr(short_value)
