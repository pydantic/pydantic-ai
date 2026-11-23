import subprocess
import sys

import pytest

from tests.loaded_distributions import SNAPSHOT_PRELUDE, Allowlist, Loads, assert_loads_only, child_env, snapshots

IMPORT_MEDIA: Allowlist = {
    'anyio': Loads('the Postgres media store locks its schema setup with AnyIO'),
    'backports-tarfile': Loads(
        "on Python 3.11, HTTPX2 runs this distribution's `backports/__init__.py` as it looks for `backports.zstd`"
    ),
    'brotli': Loads('HTTPX2 imports it when it is installed, to decode Brotli responses'),
    'httpx2': Loads("the S3 media store's HTTP client"),
    'idna': Loads('HTTPX2 imports it'),
    'sniffio': Loads('AnyIO imports it'),
    'typing-extensions': Loads('`pydantic_ai_harness._warn` imports it'),
}


@pytest.mark.subprocess(reason='asserts what a fresh interpreter imports')
def test_media_import_does_not_load_pydantic_ai() -> None:
    """Importing a light submodule must not pull in `pydantic_ai` via the root package (#9916).

    Its third-party distributions are checked against `IMPORT_MEDIA`. Runs in a fresh interpreter so imports made
    elsewhere in the test session can't hide a regression.
    """
    code = """
import sys
import pydantic_ai_harness.media
snapshot('import pydantic_ai_harness.media')
assert 'pydantic_ai' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, '-c', SNAPSHOT_PRELUDE + code], capture_output=True, text=True, check=False, env=child_env()
    )
    assert result.returncode == 0, result.stderr
    loaded = snapshots(result.stdout)['import pydantic_ai_harness.media']
    assert_loads_only('`import pydantic_ai_harness.media`', loaded, IMPORT_MEDIA, table_name='IMPORT_MEDIA')
