"""Names renamed by the harness naming pass (#11321) still import, with a `HarnessDeprecationWarning`."""

from __future__ import annotations

import importlib
import re
import sys
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning


@pytest.mark.parametrize(
    ('module', 'old', 'new'),
    [
        ('pydantic_ai_harness.experimental.acp', 'AcpFileSystemToolset', 'ACPFileSystemToolset'),
        ('pydantic_ai_harness.experimental.acp', 'AcpSession', 'ACPSession'),
        ('pydantic_ai_harness.experimental.acp', 'AcpSessionConfig', 'ACPSessionConfig'),
        ('pydantic_ai_harness.experimental.acp', 'AcpTerminalToolset', 'ACPTerminalToolset'),
        ('pydantic_ai_harness.memory', 'FileStore', 'FileMemoryStore'),
        ('pydantic_ai_harness.memory', 'InMemoryStore', 'InMemoryMemoryStore'),
        ('pydantic_ai_harness.media', 'DiskMediaStore', 'FileMediaStore'),
        ('pydantic_ai_harness.tool_output_limits', 'LocalFileStore', 'LocalFileOverflowStore'),
        ('pydantic_ai_harness.tool_output_limits', 'WorkspaceStore', 'WorkspaceOverflowStore'),
    ],
)
def test_renamed_class_alias_warns_and_resolves(module: str, old: str, new: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        package = importlib.import_module(module)

    message = f'`{module}.{old}` has been renamed to `{module}.{new}`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        alias = getattr(package, old)

    assert alias is getattr(package, new)
    assert new in package.__all__
    assert old not in package.__all__
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(package, 'DoesNotExist')


@pytest.mark.parametrize(
    ('old', 'new'),
    [('filesystem', 'file_system'), ('subagents', 'sub_agents'), ('spend', 'spend_limits')],
)
def test_renamed_module_shim_warns_and_reexports(old: str, new: str) -> None:
    package = importlib.import_module(f'pydantic_ai_harness.{new}')
    sys.modules.pop(f'pydantic_ai_harness.{old}', None)

    message = f'`pydantic_ai_harness.{old}` has been renamed to `pydantic_ai_harness.{new}`.'
    with pytest.warns(HarnessDeprecationWarning, match=re.escape(message)):
        shim = importlib.import_module(f'pydantic_ai_harness.{old}')

    assert shim.__all__ == package.__all__
    with warnings.catch_warnings():
        # `sub_agents` and `spend_limits` serve deprecated exports lazily, which warn on access.
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        assert all(getattr(shim, name) is getattr(package, name) for name in package.__all__)
    with pytest.raises(AttributeError, match='has no attribute'):
        getattr(shim, 'DoesNotExist')


@pytest.mark.parametrize(
    ('module', 'old', 'new'),
    [
        ('pydantic_ai_harness.experimental.media', 'DiskMediaStore', 'pydantic_ai_harness.media.FileMediaStore'),
        (
            'pydantic_ai_harness.experimental.overflow',
            'LocalFileStore',
            'pydantic_ai_harness.tool_output_limits.LocalFileOverflowStore',
        ),
        (
            'pydantic_ai_harness.overflowing_tool_output',
            'LocalFileStore',
            'pydantic_ai_harness.tool_output_limits.LocalFileOverflowStore',
        ),
    ],
)
def test_deprecated_modules_keep_old_class_names(module: str, old: str, new: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HarnessDeprecationWarning)
        shim = importlib.import_module(module)
    new_module, _, new_name = new.rpartition('.')

    assert old in shim.__all__
    assert getattr(shim, old) is getattr(importlib.import_module(new_module), new_name)
