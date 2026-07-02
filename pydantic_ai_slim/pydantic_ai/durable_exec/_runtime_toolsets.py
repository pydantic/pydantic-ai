"""Validation of per-run toolsets for durable execution engines.

Durable execution engines (DBOS, Prefect, Temporal) durably wrap the *executing* toolsets an agent is
constructed with — function tools become steps/tasks/activities and MCP servers get their I/O
checkpointed — so their side effects are recorded and replayed deterministically. Toolsets passed
per-run via `run(toolsets=...)` arrive after that wrapping has happened (and, for Temporal, after
activities have been registered with the worker), so an *executing* runtime toolset would run
un-checkpointed inside the workflow.

We therefore reject executing runtime toolsets, while still allowing non-executing ones like
`ExternalToolset` whose tools are resolved outside the agent run and so need no durable wrapping.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..exceptions import UserError
from ..toolsets import AbstractToolset

_KIND_LABELS = {
    'function': 'FunctionToolset',
    'mcp': 'MCPToolset',
    'dynamic': 'DynamicToolset',
}


def _runtime_toolset_kind(toolset: AbstractToolset[Any]) -> str | None:
    """Classify a leaf toolset for durable-execution runtime support, or `None` if it needs no wrapping."""
    from ..toolsets._dynamic import DynamicToolset
    from ..toolsets.function import FunctionToolset

    # A dynamic toolset resolves its inner toolset lazily per run/run step, so we can't tell ahead of
    # time whether it produces executing leaves that would need durable wrapping.
    if isinstance(toolset, DynamicToolset):
        return 'dynamic'
    if isinstance(toolset, FunctionToolset):
        return 'function'
    try:
        from ..mcp import MCPToolset
    except ImportError:  # pragma: no cover
        pass
    else:
        if isinstance(toolset, MCPToolset):
            return 'mcp'
    return None


def reject_unsupported_runtime_toolsets(
    toolsets: Sequence[AbstractToolset[Any]] | None,
    *,
    unsupported_kinds: frozenset[str],
    engine: str,
) -> None:
    """Raise a `UserError` if any per-run toolset contains a leaf `engine` can't durably wrap at runtime.

    Args:
        toolsets: The toolsets passed to `run`/`run_sync`/`iter` for this run.
        unsupported_kinds: The leaf kinds (`'function'`, `'mcp'`, `'dynamic'`) this engine cannot handle
            when added per-run. Engines that run function tools inline (DBOS) omit `'function'`.
        engine: Human-readable engine name for the error message (e.g. `'DBOS'`).
    """
    if not toolsets:
        return

    found: set[str] = set()

    def collect(leaf: AbstractToolset[Any]) -> None:
        kind = _runtime_toolset_kind(leaf)
        if kind in unsupported_kinds:
            found.add(kind)

    for toolset in toolsets:
        toolset.apply(collect)

    if found:
        labels = ', '.join(_KIND_LABELS[kind] for kind in sorted(found))
        raise UserError(
            f'{labels} cannot be passed to `run(toolsets=...)` at runtime with {engine}, because toolsets '
            'that execute their own tools or resolve dynamically must be registered for durable execution '
            'when the agent is constructed. Pass them to the agent constructor instead. Non-executing '
            'toolsets like `ExternalToolset` can be passed at runtime.'
        )
