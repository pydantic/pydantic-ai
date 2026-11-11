"""The coordinate grid of a model whose profile sets `computer_use_coordinate_grid`, applied to the `computer` tool.

The agent applies both per attempt, from the model serving it, so the tool's description and the grid recorded on the
calls always agree, also when a capability like `ModelFallback` moves the request to another model.
"""

from __future__ import annotations as _annotations

from dataclasses import replace

from . import messages as _messages
from .computer_use import _COORDINATE_GRID, COMPUTER_KIND  # pyright: ignore[reportPrivateUsage]
from .models import Model, ModelRequestParameters
from .models.fallback import FallbackModel
from .models.wrapper import WrapperModel


def _grid_description(grid: int) -> str:
    """The sentence the `computer` tool's description gets for a model that points on a grid of `grid` points."""
    last = grid - 1
    return (
        f'Positions are on a 0-{last} grid over the latest screenshot: '
        f'(0, 0) is the top left, ({last}, {last}) the bottom right.'
    )


def _answering_models(model: Model) -> list[Model]:
    """The models that may answer a request to `model`: itself, a `FallbackModel`'s models, or those a wrapper wraps."""
    if isinstance(model, FallbackModel):
        return [answering for inner in model.models for answering in _answering_models(inner)]
    try:
        model.profile
    except NotImplementedError:
        return _answering_models(model.wrapped) if isinstance(model, WrapperModel) else []
    return [model]


def coordinate_grid(model: Model) -> int | None:
    """The grid `model` points on, when every model that may answer it uses the same one."""
    grids = {answering.profile.get('computer_use_coordinate_grid') for answering in _answering_models(model)}
    return grids.pop() if len(grids) == 1 else None


def describe_grid(model: Model, params: ModelRequestParameters) -> ModelRequestParameters:
    """Tell a model that points on a grid so in the `computer` tool's description."""
    name = params.tool_name_for_kind(COMPUTER_KIND)
    if name is None or (grid := coordinate_grid(model)) is None:
        return params
    sentence = _grid_description(grid)
    tools = [
        replace(tool, description=f'{tool.description} {sentence}' if tool.description else sentence)
        if tool.name == name
        else tool
        for tool in params.function_tools
    ]
    return replace(params, function_tools=tools)


def record_grid(model: Model, params: ModelRequestParameters, response: _messages.ModelResponse) -> None:
    """Record on the response's `computer` calls the grid `model` was told to point on."""
    name = params.tool_name_for_kind(COMPUTER_KIND)
    if name is None or (grid := coordinate_grid(model)) is None:
        return
    for part in response.parts:
        if isinstance(part, _messages.ToolCallPart) and part.tool_name == name:
            part.provider_name = part.provider_name or response.provider_name
            part.provider_details = {**(part.provider_details or {}), _COORDINATE_GRID: grid}
