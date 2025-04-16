import re
import sys
from pathlib import Path

import pytest

from pydantic_ai._cli import cli


@pytest.mark.skipif(sys.version_info >= (3, 13), reason='slightly different output with 3.13')
def test_cli_help(capfd: pytest.CaptureFixture[str]):
    """Check README.md help output matches `clai --help`."""
    with pytest.raises(SystemExit):
        cli(['--help'])

    help_output = capfd.readouterr().out

    this_dir = Path(__file__).parent
    readme = this_dir / 'README.md'
    content = readme.read_text()

    new_content = re.sub('^(## Help\n+```).+?```', content, rf'\1{help_output}```', flags=re.M | re.S)
    if new_content != content:
        readme.write_text(new_content)
        pytest.fail('help output updated')
