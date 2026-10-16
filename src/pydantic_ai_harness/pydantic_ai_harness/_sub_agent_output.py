"""Shared handling of sub-agent run outputs that only the run's own caller can act on."""

from __future__ import annotations

from collections.abc import Sequence
from types import UnionType
from typing import Union, cast, get_args, get_origin

from pydantic_ai import DeferredToolRequests
from pydantic_ai.output import OutputSpec

# Output types that end a run early so its caller can resume it. A top-level caller can do that;
# a delegating agent or a workflow script cannot, so for a sub-agent they are an interruption, not
# an output. Each maps to the reason shown when a sub-agent run ends with one.
_RESUMABLE_OUTPUT_REASONS: dict[type, str] = {
    DeferredToolRequests: 'it needs tool approval or deferred tool results',
}


def _is_resumable_output_type(output_type: object) -> bool:
    return isinstance(output_type, type) and issubclass(output_type, tuple(_RESUMABLE_OUTPUT_REASONS))


def without_resumable_outputs(output_type: OutputSpec[object]) -> OutputSpec[object]:
    """`output_type` minus the resumable output types listed directly in it as a sequence or union.

    Output mode wrappers like `ToolOutput` are kept as they are. Core requires an agent to have
    at least one other output type, so something is always left.
    """
    members: Sequence[object]
    if isinstance(output_type, Sequence):
        members = output_type  # pyright: ignore[reportUnknownVariableType]
    elif get_origin(output_type) in (Union, UnionType):
        members = get_args(output_type)
    else:
        return output_type
    kept = [member for member in members if not _is_resumable_output_type(member)]
    # Each member of a valid output spec is a valid output spec item, so the kept members form one too.
    return cast('OutputSpec[object]', kept or members)


def resumable_output_reason(output: object) -> str | None:
    """Why a sub-agent run that ended with `output` has no result, or `None` if `output` is its result.

    Phrased to follow the sub-agent's name, e.g. "ended with `DeferredToolRequests` because ...".
    """
    for output_type, reason in _RESUMABLE_OUTPUT_REASONS.items():
        if isinstance(output, output_type):
            return f'ended with `{output_type.__name__}` because {reason}'
    return None
