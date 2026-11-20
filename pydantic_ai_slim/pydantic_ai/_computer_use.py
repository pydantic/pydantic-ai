"""How a model applies its profile's computer use settings to its own requests and responses.

A model that points on a grid is told so in the `computer` tool's description. Its `computer` calls are labelled with
its grid and screenshot limits, so the tool can scale positions to pixels. A request it can't serve, because it can't
see images or the latest screenshot is larger than it can see, is refused before it's sent. Each model does this for
itself, so under a fallback the model that answers decides, and a model that refuses is passed over.
"""

from __future__ import annotations as _annotations

import math
from collections.abc import Container, Iterable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

from . import messages as _messages
from .exceptions import UserError
from .profiles import ModelProfile

if TYPE_CHECKING:
    from .models import ModelRequestParameters

COORDINATE_GRID = 'coordinate_grid'
"""The `provider_details` key where a model records the size of the grid its call's positions are on."""

SCREENSHOT_MAX_SIDE = 'screenshot_max_side'
"""The `provider_details` key where a model records the longest screenshot side it can see."""

SCREENSHOT_MAX_PIXELS = 'screenshot_max_pixels'
"""The `provider_details` key where a model records the most screenshot pixels it can see."""

LABEL_KEYS = frozenset({COORDINATE_GRID, SCREENSHOT_MAX_SIDE, SCREENSHOT_MAX_PIXELS})
"""The `provider_details` keys a model adds to its `computer` calls. The provider never sent or saw them."""

_PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class UnservableRequestError(UserError):
    """A request with a `computer` tool that the model about to answer it can't serve.

    `FallbackModel` and the `ModelFallback` capability move on to their next model when one raises this, whatever
    their `fallback_on` says, since another model may be able to serve it.
    """


def _computer_kind_id() -> str:
    # `computer_use` imports the label keys from this module, so it's imported here when first needed.
    from .computer_use import COMPUTER_KIND

    return COMPUTER_KIND.id


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
    if (grid := computer_call_labels(profile).get(COORDINATE_GRID)) is None:
        return params
    hint = _grid_hint(grid)
    tools = [
        replace(tool, description=f'{tool.description} {hint}' if tool.description else hint)
        if tool.tool_kind == _computer_kind_id() and hint not in (tool.description or '')
        else tool
        for tool in params.function_tools
    ]
    return replace(params, function_tools=tools) if tools != params.function_tools else params


def computer_tool_names(params: ModelRequestParameters) -> set[str]:
    """The names of the request's `computer` tools."""
    kind = _computer_kind_id()
    return {tool.name for tool in params.function_tools if tool.tool_kind == kind}


def computer_call_labels(profile: ModelProfile) -> dict[str, int]:
    """The grid and screenshot limits a model's `computer` calls are labelled with, from its profile."""
    grid = profile.get('computer_use_coordinate_grid')
    labels = {
        # A grid of one point can't say where anything is, so it isn't one.
        COORDINATE_GRID: grid if grid is not None and grid >= 2 else None,
        SCREENSHOT_MAX_SIDE: profile.get('computer_use_screenshot_max_side'),
        SCREENSHOT_MAX_PIXELS: profile.get('computer_use_screenshot_max_pixels'),
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


def png_size(data: bytes) -> tuple[int, int] | None:
    """Width and height from a PNG's header chunk, which the signature puts at bytes 16 to 24."""
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b'IHDR':
        return None
    return int.from_bytes(data[16:20], 'big'), int.from_bytes(data[20:24], 'big')


def latest_screenshot_size(messages: Iterable[_messages.ModelMessage]) -> tuple[int, int] | None:
    """The size of the latest screenshot a `computer` call returned in `messages`."""
    from .computer_use import ComputerReturnPart

    for message in reversed(list(messages)):
        if not isinstance(message, _messages.ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ComputerReturnPart) and (screenshot := part.screenshot) is not None:
                return png_size(screenshot.data)
    return None


def fit_screenshot(size: tuple[int, int], *, max_side: int | None, max_pixels: int | None) -> tuple[int, int]:
    """The largest size with the same shape as `size` within the limits, `size` itself when it fits.

    A limit left `None` is no limit.
    """
    width, height = size
    scale = min(
        max_side / max(width, height) if max_side else 1.0,
        math.sqrt(max_pixels / (width * height)) if max_pixels else 1.0,
    )
    return size if scale >= 1 else (max(1, int(width * scale)), max(1, int(height * scale)))


def refuse_unservable_computer_request(
    messages: Sequence[_messages.ModelMessage],
    params: ModelRequestParameters | None,
    profile: ModelProfile,
    model_name: str,
) -> None:
    """Refuse a request with a `computer` tool that this model can't serve, with an `UnservableRequestError`.

    Every `computer` call returns a screenshot, so a model that can't see images can't use the tool. A model shown a
    screenshot larger than it can see would see it scaled down and give positions on the smaller image.

    `messages` are the ones about to be sent. A model that takes images but not PNGs gets a note in place of each
    screenshot, so there is no size to check and it isn't refused for one.
    """
    if params is None or not computer_tool_names(params):
        return
    if not profile.get('supports_image_input', True):
        raise UnservableRequestError(
            f'The `computer` tool needs a model that accepts images, and {model_name!r} does not: every `computer` '
            'call returns a screenshot.'
        )
    if (size := latest_screenshot_size(messages)) is None:
        return
    max_side = profile.get('computer_use_screenshot_max_side')
    max_pixels = profile.get('computer_use_screenshot_max_pixels')
    if (fitted := fit_screenshot(size, max_side=max_side, max_pixels=max_pixels)) != size:
        raise UnservableRequestError(
            f'The latest screenshot is {size[0]}x{size[1]}, larger than {model_name!r} can see. Start the display at '
            f'{fitted[0]}x{fitted[1]} or smaller to use this model on it.'
        )
