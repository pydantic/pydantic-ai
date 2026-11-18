"""Guards on the declarations of the projection passes `Model.prepare_messages` runs (#11030).

Each pass declares the part kinds it reads, the ones it consumes (renders as other parts) and the ones it drops (leaves
out with nothing in their place). The order is checked from the declarations, so no pass drops a kind a later pass
reads, and the declarations are checked against what each pass does on the histories in `tests/history_cases.py`, for
every target `tests/test_history_properties.py` renders them for. That file checks the rendered requests themselves.
"""

from __future__ import annotations as _annotations

from dataclasses import replace
from typing import Any, get_args

import pytest
from pydantic import JsonValue

from pydantic_ai.messages import ModelMessage, ModelRequestPart, SystemPromptPart, repair_messages
from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    _PartKind,  # pyright: ignore[reportPrivateUsage]
    _Projection,  # pyright: ignore[reportPrivateUsage]
    _ProjectionPass,  # pyright: ignore[reportPrivateUsage]
    _UnpreparedPart,  # pyright: ignore[reportPrivateUsage]
)

from .history_cases import HistoryCase, message_json
from .test_history_properties import MapperCase, history_case_params

PROJECTION_PASSES = Model._PROJECTION_PASSES  # pyright: ignore[reportPrivateUsage]

REMOVED_KINDS: frozenset[_PartKind] = frozenset(
    kind for projection_pass in PROJECTION_PASSES for kind in projection_pass.consumes | projection_pass.drops
)
"""The part kinds some pass consumes or drops: the ones whose readers the declarations have to name."""


def _reads(projection_pass: _ProjectionPass) -> frozenset[_PartKind]:
    return projection_pass.reads | projection_pass.consumes


def test_no_pass_drops_a_kind_a_later_pass_reads():
    violations = [
        (earlier.name, later.name, kind.__name__)
        for index, earlier in enumerate(PROJECTION_PASSES)
        for later in PROJECTION_PASSES[index + 1 :]
        for kind in sorted(earlier.drops & _reads(later), key=lambda kind: kind.__name__)
    ]
    assert violations == []


def test_pass_names_are_unique():
    names = [projection_pass.name for projection_pass in PROJECTION_PASSES]
    assert len(names) == len(set(names))


def test_unprepared_part_covers_every_request_kind_a_pass_removes():
    """Every request part kind a pass consumes or drops is in `_UnpreparedPart`, the type the adapters' one guard
    branch checks, except `SystemPromptPart`: every adapter maps one, and the `system_voice` pass consumes it only
    for a model without mid-conversation system messages.

    This checks the declaration, not each adapter's behavior. Being in `_UnpreparedPart` doesn't mean every path
    raises: adapters still send a deprecated `RetryPromptPart` through `_sendable_request_parts`, and `FunctionModel`
    estimates a `SpeechPart`'s tokens from its transcript."""
    # `ModelRequestPart` is `Annotated[Union, Discriminator]`, and each union member is `Annotated[Part, Tag]`.
    (request_part_union, *_) = get_args(ModelRequestPart)
    request_kinds = {get_args(member)[0] for member in get_args(request_part_union)}
    assert {kind for kind in REMOVED_KINDS if kind in request_kinds} - {SystemPromptPart} <= set(
        get_args(_UnpreparedPart)
    )


def _part_kinds(messages: list[ModelMessage]) -> set[type]:
    return {type(part) for message in messages for part in message.parts}


def _without(messages: list[ModelMessage], kind: _PartKind) -> list[ModelMessage]:
    """`messages` without their parts of `kind`, keeping a message left with none."""
    return [
        replace(message, parts=[part for part in message.parts if not isinstance(part, kind)]) for message in messages
    ]


def _dump(messages: list[ModelMessage]) -> list[JsonValue]:
    return [message_json(message) for message in messages]


@pytest.mark.parametrize(('history_case', 'case'), history_case_params())
async def test_passes_drop_only_what_they_declare(history_case: HistoryCase, case: MapperCase[Any]):
    """No pass leaves a part kind out of the history that it doesn't declare it consumes or drops."""
    model = case.model()
    projection = _Projection(repair_messages(await history_case.history()), ModelRequestParameters())
    for projection_pass in PROJECTION_PASSES:
        projected = projection_pass.run(model, projection)
        removed = _part_kinds(projection.messages) - _part_kinds(projected.messages)
        declared = tuple(projection_pass.consumes | projection_pass.drops)
        assert [kind.__name__ for kind in removed if not issubclass(kind, declared)] == [], projection_pass.name
        projection = projected


@pytest.mark.parametrize(('history_case', 'case'), history_case_params())
async def test_passes_read_only_what_they_declare(history_case: HistoryCase, case: MapperCase[Any]):
    """A pass's output doesn't depend on the parts of a kind some pass consumes or drops, unless it declares reading it.

    Each pass runs on the stored history, which still holds every kind, so a pass that reads a kind an earlier pass
    drops shows up here even though, run in order, it never sees one.
    """
    model = case.model()
    history = repair_messages(await history_case.history())
    present = _part_kinds(history)
    for projection_pass in PROJECTION_PASSES:
        for kind in sorted(REMOVED_KINDS - _reads(projection_pass), key=lambda kind: kind.__name__):
            if not any(issubclass(present_kind, kind) for present_kind in present):
                continue
            stripped = _without(history, kind)
            full = projection_pass.run(model, _Projection(history, ModelRequestParameters(), authored=history))
            without = projection_pass.run(model, _Projection(stripped, ModelRequestParameters(), authored=stripped))
            assert _dump(_without(full.messages, kind)) == _dump(without.messages), (
                projection_pass.name,
                kind.__name__,
            )
