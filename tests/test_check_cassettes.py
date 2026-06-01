from pathlib import Path

from scripts.check_cassettes import _collect_vcr_tests_from_file


def test_collect_vcr_tests_reads_utf8_source(tmp_path: Path) -> None:
    test_file = tmp_path / 'test_unicode_source.py'
    test_file.write_text(
        """\
import pytest


@pytest.mark.vcr
def test_unicode_source_comment():
    message = 'Olá, 世界'
    assert message
""",
        encoding='utf-8',
    )

    assert _collect_vcr_tests_from_file(test_file) == {'test_unicode_source_comment'}
