"""Tests for `scripts/check_cassettes.py`."""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


@pytest.mark.parametrize(
    ('source', 'cassette_name'),
    [
        (
            """
            import pytest

            pytestmark = [pytest.mark.anyio, pytest.mark.ws_cassette]

            async def test_module_marked():
                pass
            """,
            'test_module_marked',
        ),
        (
            """
            import pytest

            @pytest.mark.ws_cassette
            class TestMarked:
                async def test_method(self):
                    pass
            """,
            'TestMarked.test_method',
        ),
    ],
)
def test_check_cassettes_finds_ws_cassette_scoped_markers(tmp_path: Path, source: str, cassette_name: str) -> None:
    """Exercise local CLI parsing; there is no provider request for VCR to record."""
    tests_dir = tmp_path / 'tests'
    test_file = tests_dir / 'test_marked.py'
    test_file.parent.mkdir()
    test_file.write_text(dedent(source), encoding='utf-8')

    cassette_dir = tests_dir / 'cassettes' / 'test_marked'
    cassette_dir.mkdir(parents=True)
    (cassette_dir / f'{cassette_name}.yaml').write_text('version: 1\ninteractions: []\n', encoding='utf-8')

    script = Path(__file__).parents[1] / 'scripts' / 'check_cassettes.py'
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Orphaned cassettes check: 1 matched, 0 orphaned' in result.stdout
