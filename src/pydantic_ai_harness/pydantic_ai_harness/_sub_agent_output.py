"""Shared handling of sub-agent run outputs that only the run's own caller can act on."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from types import UnionType
from typing import Any, ClassVar, Union, cast, get_args, get_origin

from pydantic_ai import DeferredToolRequests, ModelResponse, PendingModelResponse, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.output import OutputSpec

# Output types that end a run early so its caller can resume it. A top-level caller can do that;
# a delegating agent or a workflow script cannot, so for a sub-agent they are an interruption, not
# an output. Each maps to the reason shown when a sub-agent run ends with one.
_RESUMABLE_OUTPUT_REASONS: dict[type, str] = {
    DeferredToolRequests: 'it needs tool approval or deferred tool results',
    PendingModelResponse: 'it needs the result of a model request the provider is still processing',
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


class ServingModelRecorder(AbstractCapability[Any]):
    """Records the model that served a sub-agent run's last request, to cancel a request it left pending.

    The model is the one capabilities like `BatchModelRequests` wrapped, so cancelling through it also releases the
    request from a batch shared with other runs.
    """

    # It only observes the request, so it can be added to a sub-agent run under durable execution too.
    _safe_at_runtime: ClassVar[bool] = True

    def __init__(self) -> None:
        self.model: Model | None = None

    async def after_model_request(
        self, ctx: RunContext[Any], *, request_context: ModelRequestContext, response: ModelResponse
    ) -> ModelResponse:
        self.model = request_context.model
        return response


async def cancel_pending_output(output: object, recorder: ServingModelRecorder) -> None:
    """Cancel the provider request a sub-agent run left pending, as its caller can't check on it later.

    Best effort, like the cancellations in core: a failing cancel must not replace the outcome the caller reports.
    """
    if isinstance(output, PendingModelResponse) and recorder.model is not None:
        with suppress(Exception):
            await recorder.model.cancel_suspended_response(output.response)


def unresumable_output_message(agent_name: str, reason: str) -> str:
    """What the delegating model is told when sub-agent `agent_name` ended for `reason` from `resumable_output_reason`."""
    return (
        f'Sub-agent {agent_name!r} {reason}, which a delegating agent cannot provide. '
        'Do the task another way, or without this sub-agent.'
    )
