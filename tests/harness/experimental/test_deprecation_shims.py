"""The old `experimental.<name>` paths still import, with a `DeprecationWarning` to the new path.

Each capability that graduated out of `experimental` leaves a compatibility shim at its former
path. Importing the shim must keep working (re-exporting the moved capability) and must warn.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

# (old experimental subpackage, new top-level module, a public symbol to check re-export)
_SHIMS = [
    ('authoring', 'runtime_authoring', 'RuntimeAuthoring'),
    ('overflow', 'overflowing_tool_output', 'OverflowingToolOutput'),
    ('compaction', 'compaction', 'TieredCompaction'),
    ('context', 'context', 'RepoContext'),
    ('docs', 'docs', 'PyaiDocs'),
    ('dynamic_workflow', 'dynamic_workflow', 'DynamicWorkflow'),
    ('media', 'media', 'S3MediaStore'),
    ('planning', 'planning', 'Planning'),
    ('step_persistence', 'step_persistence', 'StepPersistence'),
    ('subagents', 'subagents', 'SubAgents'),
]


@pytest.mark.parametrize('old, new, symbol', _SHIMS)
def test_shim_warns_and_reexports(old: str, new: str, symbol: str) -> None:
    # Import once quietly so the deprecation fires on the reload we assert on, even if
    # another test already imported the shim.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shim = importlib.import_module(f'pydantic_ai_harness.experimental.{old}')

    with pytest.warns(DeprecationWarning, match=rf'experimental\.{old}` has moved'):
        importlib.reload(shim)

    new_module = importlib.import_module(f'pydantic_ai_harness.{new}')
    assert getattr(shim, symbol) is getattr(new_module, symbol)
