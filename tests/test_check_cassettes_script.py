"""Tests for cassette-marker collection in `scripts/check_cassettes.py`."""

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.check_cassettes import _collect_vcr_tests_from_file  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        (
            """
            import pytest

            pytestmark = [pytest.mark.anyio, pytest.mark.ws_cassette]

            async def test_module_marked():
                pass
            """,
            {'test_module_marked'},
        ),
        (
            """
            import pytest

            @pytest.mark.ws_cassette
            class TestMarked:
                async def test_method(self):
                    pass
            """,
            {'TestMarked.test_method'},
        ),
    ],
)
def test_collect_ws_cassette_scoped_markers(tmp_path: Path, source: str, expected: set[str]) -> None:
    test_file = tmp_path / 'test_marked.py'
    test_file.write_text(dedent(source), encoding='utf-8')

    assert _collect_vcr_tests_from_file(test_file) == expected
