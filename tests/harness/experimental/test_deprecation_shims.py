"""The old `experimental.<name>` paths still import, with a `HarnessDeprecationWarning` to the new path.

Each capability that graduated out of `experimental` leaves a compatibility shim at its former
path. Importing the shim must keep working (re-exporting the moved capability) and must warn.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

from pydantic_ai_harness import HarnessDeprecationWarning

# (old experimental subpackage, new top-level module, shim symbol, symbol on the new module)
_SHIMS = [
    ('authoring', 'capability_creation', 'RuntimeAuthoring', 'CapabilityCreation'),
    ('overflow', 'tool_output_limits', 'OverflowingToolOutput', 'ToolOutputLimits'),
    ('compaction', 'compaction', 'TieredCompaction', 'TieredCompaction'),
    ('context', 'repo_context', 'RepoContext', 'RepoContext'),
    # `PyaiDocs` is a subclass (it keeps the old spec name), not an alias, so the
    # identity check uses the toolset.
    ('docs', 'pydantic_ai_docs', 'PyaiDocsToolset', 'PydanticAIDocsToolset'),
    ('dynamic_workflow', 'dynamic_workflow', 'DynamicWorkflow', 'DynamicWorkflow'),
    ('media', 'media', 'S3MediaStore', 'S3MediaStore'),
    ('planning', 'planning', 'Planning', 'Planning'),
    ('step_persistence', 'step_persistence', 'StepPersistence', 'StepPersistence'),
    ('subagents', 'subagents', 'SubAgents', 'SubAgents'),
]


@pytest.mark.parametrize('old, new, shim_symbol, new_symbol', _SHIMS)
def test_shim_warns_and_reexports(old: str, new: str, shim_symbol: str, new_symbol: str) -> None:
    # Import once quietly so the deprecation fires on the reload we assert on, even if
    # another test already imported the shim.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        shim = importlib.import_module(f'pydantic_ai_harness.experimental.{old}')

    with pytest.warns(HarnessDeprecationWarning, match=rf'experimental\.{old}` has moved'):
        importlib.reload(shim)

    new_module = importlib.import_module(f'pydantic_ai_harness.{new}')
    assert getattr(shim, shim_symbol) is getattr(new_module, new_symbol)
