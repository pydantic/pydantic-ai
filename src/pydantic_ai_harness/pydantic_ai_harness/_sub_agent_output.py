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

    Nested sequences and unions, and output mode wrappers like `ToolOutput`, are kept as they are:
    the rendered signature may then still mention a marker, but a run that ends with one fails
    the call all the same.
    """
    members: Sequence[object]
    if isinstance(output_type, Sequence):
        members = output_type  # pyright: ignore[reportUnknownVariableType]
    elif get_origin(output_type) in (Union, UnionType):
        members = get_args(output_type)
    else:
        return output_type
    # The members of a valid output spec are valid output spec items, so any subset of them is an output
    # spec too; pyright cannot see that through `object`. Core rejects an output type that is only markers.
    return cast('OutputSpec[object]', [member for member in members if not _is_resumable_output_type(member)])


def resumable_output_reason(output: object) -> str | None:
    """Why a sub-agent run that ended with `output` has no result, or `None` if `output` is its result.

    Phrased to follow the sub-agent's name, e.g. "ended with `DeferredToolRequests` because ...".
    """
    for output_type, reason in _RESUMABLE_OUTPUT_REASONS.items():
        if isinstance(output, output_type):
            return f'ended with `{output_type.__name__}` because {reason}'
    return None
