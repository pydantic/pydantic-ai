"""Exercise the chat example's real frontend on fragmented response streams."""

import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.ui_roundtrip
def test_chat_app_stream():
    # Node executes the actual TypeScript client; Python cannot reproduce browser stream parsing.
    result = subprocess.run(
        ['node', '--experimental-vm-modules', str(Path(__file__).with_name('chat_app_stream.test.mjs'))],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=False,
        timeout=10,
    )
    if result.returncode == 77:
        reason = 'Node >=22.13 is required to strip TypeScript; the UI round-trip CI job uses Node 22'
        if os.environ.get('UI_ROUNDTRIP_REQUIRED'):
            pytest.fail(reason)
        pytest.skip(reason)
    assert result.returncode == 0, result.stdout + result.stderr
