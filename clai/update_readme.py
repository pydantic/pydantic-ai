import os
import re
import sys
from pathlib import Path

import pytest

from pydantic_ai._cli import cli


@pytest.mark.skipif(sys.version_info >= (3, 13), reason='slightly different output with 3.13')
def test_cli_help(capfd: pytest.CaptureFixture[str]):
    """Check README.md help output matches `clai --help`."""
    os.environ['COLUMNS'] = '150'
    with pytest.raises(SystemExit):
        cli(['--help'], prog_name='clai')

    help_output = capfd.readouterr().out.strip()
    # TODO change when we reach v1
    help_output = re.sub(r'(Pydantic AI CLI v).+', r'\1...', help_output)

    this_dir = Path(__file__).parent
    readme = this_dir / 'README.md'
    content = readme.read_text()

    new_content, count = re.subn('^(## Help\n+```).+?```', rf'\1\n{help_output}\n```', content, flags=re.M | re.S)
    assert count, 'help section not found'
    if new_content != content:
        readme.write_text(new_content)
        pytest.fail('`clai --help` output changed.')
