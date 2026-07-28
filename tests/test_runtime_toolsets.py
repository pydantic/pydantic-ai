from __future__ import annotations

import pytest

from pydantic_ai.durable_exec._runtime_toolsets import reject_unsupported_runtime_toolsets
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets.function import FunctionToolset


def test_reject_unsupported_runtime_toolsets_lists_ids() -> None:
    """When several unsupported toolsets share a kind, the error names each one by id."""
    toolsets = [FunctionToolset(id='search-tools'), FunctionToolset(id='billing-tools')]
    with pytest.raises(UserError) as exc_info:
        reject_unsupported_runtime_toolsets(toolsets, unsupported_kinds=frozenset({'function'}), engine='Prefect')
    message = str(exc_info.value)
    assert "'search-tools'" in message
    assert "'billing-tools'" in message
    assert 'FunctionToolset' in message


def test_reject_unsupported_runtime_toolset_falls_back_to_type_name_without_id() -> None:
    """A toolset with no id keeps the bare kind label in the error (no empty quotes)."""
    with pytest.raises(UserError) as exc_info:
        reject_unsupported_runtime_toolsets(
            [FunctionToolset()], unsupported_kinds=frozenset({'function'}), engine='Prefect'
        )
    message = str(exc_info.value)
    assert "FunctionToolset 'FunctionToolset'" not in message
    assert message.startswith('FunctionToolset cannot be passed')


def test_reject_unsupported_runtime_toolsets_mixed_id_and_no_id() -> None:
    """When id'd and anonymous toolsets of the same kind are mixed, only id'd ones are quoted."""
    with pytest.raises(UserError) as exc_info:
        reject_unsupported_runtime_toolsets(
            [FunctionToolset(id='named'), FunctionToolset()],
            unsupported_kinds=frozenset({'function'}),
            engine='Prefect',
        )
    message = str(exc_info.value)
    assert "'named'" in message
    # The anonymous one must not introduce an empty/quoted placeholder.
    assert "''" not in message


def test_reject_unsupported_runtime_toolsets_no_offenders() -> None:
    """No unsupported kinds means no error is raised."""
    reject_unsupported_runtime_toolsets([FunctionToolset(id='ok')], unsupported_kinds=frozenset(), engine='Prefect')
