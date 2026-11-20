"""How a model applies its profile's computer use settings to its own requests and responses.

A model that points on a grid is told so in the `computer` tool's description. Its `computer` calls are labelled with
its grid and screenshot limits, so the tool can scale positions to pixels. A request it can't serve, because it can't
see images or the latest screenshot is larger than it can see, is refused. Each model does this for itself, so under a
`FallbackModel` the model that answers decides.
"""

from __future__ import annotations as _annotations

import math
from collections.abc import Container, Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from . import messages as _messages
from .computer_use import (
    _COORDINATE_GRID,  # pyright: ignore[reportPrivateUsage]
    _SCREENSHOT_MAX_PIXELS,  # pyright: ignore[reportPrivateUsage]
    _SCREENSHOT_MAX_SIDE,  # pyright: ignore[reportPrivateUsage]
    COMPUTER_KIND,
    ComputerReturnPart,
)
from .exceptions import UserError
from .profiles import ModelProfile

if TYPE_CHECKING:
    from .models import ModelRequestParameters

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def _grid_hint(grid: int) -> str:
    """The sentence added to the `computer` tool's description for a model that points on a grid of `grid` points."""
    last = grid - 1
    return (
        f'Positions are on a 0-{last} grid over the latest screenshot: '
        f'(0, 0) is the top left, ({last}, {last}) the bottom right.'
    )


def with_computer_grid_hint(params: ModelRequestParameters, profile: ModelProfile) -> ModelRequestParameters:
    """Tell a model that points on a grid so, in each `computer` tool's description.

    A request can be prepared more than once, so a description that already has the sentence is left as it is.
    """
    if (grid := profile.get('computer_use_coordinate_grid')) is None or grid < 2:
        return params
    hint = _grid_hint(grid)
    tools = [
        replace(tool, description=f'{tool.description} {hint}' if tool.description else hint)
        if tool.tool_kind == COMPUTER_KIND.id and hint not in (tool.description or '')
        else tool
        for tool in params.function_tools
    ]
    return replace(params, function_tools=tools) if tools != params.function_tools else params


def computer_tool_names(params: ModelRequestParameters) -> set[str]:
    """The names of the request's `computer` tools."""
    return {tool.name for tool in params.function_tools if tool.tool_kind == COMPUTER_KIND.id}


def computer_call_labels(profile: ModelProfile) -> dict[str, int]:
    """The grid and screenshot limits a model's `computer` calls are labelled with, from its profile."""
    labels = {
        _COORDINATE_GRID: profile.get('computer_use_coordinate_grid'),
        _SCREENSHOT_MAX_SIDE: profile.get('computer_use_screenshot_max_side'),
        _SCREENSHOT_MAX_PIXELS: profile.get('computer_use_screenshot_max_pixels'),
    }
    return {key: value for key, value in labels.items() if value is not None}


def label_calls(parts: Iterable[object], tool_names: Container[str], labels: Mapping[str, int]) -> None:
    """Record a model's `labels` in the `provider_details` of each call in `parts` to one of the `computer` tools."""
    if not labels:
        return
    for part in parts:
        if isinstance(part, _messages.ToolCallPart) and part.tool_name in tool_names:
            # `provider_name` is left as sent, because the response digest includes it. The digest leaves these keys
            # out of a `computer` call, so labelling after the digest is taken changes nothing.
            part.provider_details = {**(part.provider_details or {}), **labels}


def _png_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's header chunk, which the signature puts at bytes 16 to 24."""
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b'IHDR':
        return None
    return int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')


def _latest_screenshot_size(messages: list[_messages.ModelMessage]) -> tuple[int, int] | None:
    for message in reversed(messages):
        if not isinstance(message, _messages.ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ComputerReturnPart) and (screenshot := part.screenshot) is not None:
                return _png_size(screenshot.data)
    return None


def refuse_unservable_computer_request(
    messages: list[_messages.ModelMessage],
    params: ModelRequestParameters | None,
    profile: ModelProfile,
    model_name: str,
) -> None:
    """Refuse a request with a `computer` tool that this model can't serve.

    Every `computer` call returns a screenshot, so a model that can't see images can't use the tool. A model shown a
    screenshot larger than it can see would see it scaled down and give positions on the smaller image.
    """
    if params is None or not computer_tool_names(params):
        return
    if not profile.get('supports_image_input', True):
        raise UserError(
            f'The `computer` tool needs a model that accepts images, and {model_name!r} does not: every `computer` '
            'call returns a screenshot.'
        )
    if (size := _latest_screenshot_size(messages)) is None:
        return
    width, height = size
    max_side = profile.get('computer_use_screenshot_max_side')
    max_pixels = profile.get('computer_use_screenshot_max_pixels')
    scale = min(
        max_side / max(width, height) if max_side else 1.0,
        math.sqrt(max_pixels / (width * height)) if max_pixels else 1.0,
    )
    if scale < 1:
        fitted = f'{max(1, int(width * scale))}x{max(1, int(height * scale))}'
        raise UserError(
            f'The latest screenshot is {width}x{height}, larger than {model_name!r} can see. Start the display at '
            f'{fitted} or smaller to use this model on it.'
        )
