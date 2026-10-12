import os
import subprocess
import sys

import pytest


@pytest.mark.subprocess(reason='asserts what a fresh interpreter imports')
def test_media_import_does_not_load_pydantic_ai() -> None:
    """Importing a light submodule must not pull in `pydantic_ai` via the root package (#9916).

    Runs in a fresh interpreter so imports made elsewhere in the test session can't hide a regression.
    """
    code = 'import sys, pydantic_ai_harness.media; print("pydantic_ai" in sys.modules)'
    env: dict[str, str] = {key: value for key, value in os.environ.items() if not key.startswith('COVERAGE_')}
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip() == 'False'
