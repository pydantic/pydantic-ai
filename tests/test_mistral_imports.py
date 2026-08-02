from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    'target_module,error_expression,expected_message',
    [
        (
            'pydantic_ai.models.mistral',
            "ModuleNotFoundError('blocked dependency: mistralai', name='mistralai')",
            'Please install `mistralai` to use the Mistral model',
        ),
        (
            'pydantic_ai.providers.mistral',
            "ModuleNotFoundError('blocked dependency: mistralai', name='mistralai')",
            'Please install the `mistralai` package to use the Mistral provider',
        ),
        (
            'pydantic_ai.models.mistral',
            "ModuleNotFoundError('blocked dependency: httpx', name='httpx')",
            'blocked dependency: httpx',
        ),
        (
            'pydantic_ai.providers.mistral',
            'ImportError("cannot import name \'Mistral\'")',
            "cannot import name 'Mistral'",
        ),
    ],
)
def test_mistral_import_error(target_module: str, error_expression: str, expected_message: str) -> None:
    """Mistral imports add guidance only when the SDK itself is missing.

    Unit test rather than VCR: this exercises module import behavior before any request.
    """
    code = f"""
import importlib.abc
import importlib.util
import sys

class BlockMistral(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mistralai':
            return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise {error_expression}

sys.meta_path.insert(0, BlockMistral())
import {target_module}
"""

    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)

    assert result.returncode == 1
    assert expected_message in result.stderr
    if not expected_message.startswith('Please install'):
        assert 'pip install "pydantic-ai-slim[mistral]"' not in result.stderr
