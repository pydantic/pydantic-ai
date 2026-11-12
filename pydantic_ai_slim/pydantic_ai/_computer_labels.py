"""Tells a model that points on a grid so, and labels its `computer` calls with its grid and screenshot limits.

The agent applies this per attempt, so it follows the model serving the request, also after `ModelFallback` switches.
"""

from __future__ import annotations as _annotations

from dataclasses import replace
from typing import Literal

from . import messages as _messages
from .computer_use import (
    _COORDINATE_GRID,  # pyright: ignore[reportPrivateUsage]
    _SCREENSHOT_MAX_PIXELS,  # pyright: ignore[reportPrivateUsage]
    _SCREENSHOT_MAX_SIDE,  # pyright: ignore[reportPrivateUsage]
    COMPUTER_KIND,
)
from .models import Model, ModelRequestParameters
from .models.fallback import FallbackModel
from .models.wrapper import WrapperModel


def _grid_hint(grid: int) -> str:
    """The hint added to the `computer` tool's description for a model that points on a grid of `grid` points."""
    last = grid - 1
    return (
        f'Positions are on a 0-{last} grid over the latest screenshot: '
        f'(0, 0) is the top left, ({last}, {last}) the bottom right.'
    )


def _candidate_models(model: Model) -> list[Model]:
    """The models that may answer a request to `model`: itself, a `FallbackModel`'s models, or those a wrapper wraps."""
    if isinstance(model, FallbackModel):
        return [answering for inner in model.models for answering in _candidate_models(inner)]
    try:
        model.profile
    except NotImplementedError:
        return _candidate_models(model.wrapped) if isinstance(model, WrapperModel) else []
    return [model]


def shared_grid(model: Model) -> int | None:
    """The grid `model` points on, when every model that may answer it uses the same one."""
    grids = {answering.profile.get('computer_use_coordinate_grid') for answering in _candidate_models(model)}
    return grids.pop() if len(grids) == 1 else None


def _smallest_limit(
    models: list[Model], key: Literal['computer_use_screenshot_max_side', 'computer_use_screenshot_max_pixels']
) -> int | None:
    """The smallest of the models' screenshot limits under `key`, so every one of them can see the screenshot."""
    return min((limit for answering in models if (limit := answering.profile.get(key))), default=None)


def add_grid_hint(model: Model, params: ModelRequestParameters) -> ModelRequestParameters:
    """Tell a model that points on a grid so in the `computer` tool's description."""
    name = params.tool_name_for_kind(COMPUTER_KIND)
    if name is None or (grid := shared_grid(model)) is None:
        return params
    hint = _grid_hint(grid)
    tools = [
        replace(tool, description=f'{tool.description} {hint}' if tool.description else hint)
        if tool.name == name
        else tool
        for tool in params.function_tools
    ]
    return replace(params, function_tools=tools)


def label_calls(model: Model, params: ModelRequestParameters, response: _messages.ModelResponse) -> None:
    """Record `model`'s grid and screenshot limits on the response's `computer` calls."""
    name = params.tool_name_for_kind(COMPUTER_KIND)
    if name is None:
        return
    models = _candidate_models(model)
    labels = {
        _COORDINATE_GRID: shared_grid(model),
        _SCREENSHOT_MAX_SIDE: _smallest_limit(models, 'computer_use_screenshot_max_side'),
        _SCREENSHOT_MAX_PIXELS: _smallest_limit(models, 'computer_use_screenshot_max_pixels'),
    }
    labels = {key: value for key, value in labels.items() if value is not None}
    if not labels:
        return
    for part in response.parts:
        if isinstance(part, _messages.ToolCallPart) and part.tool_name == name:
            part.provider_name = part.provider_name or response.provider_name
            part.provider_details = {**(part.provider_details or {}), **labels}
