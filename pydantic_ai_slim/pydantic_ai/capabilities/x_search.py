from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai.builtin_tools import XSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .builtin_or_local import BuiltinOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.common_tools.x_search import XSearchFallbackModel


@dataclass(init=False)
class XSearch(BuiltinOrLocalTool[AgentDepsT]):
    """X (Twitter) search capability.

    On xAI models, uses the native builtin X search directly with no extra configuration.

    On non-xAI models, you must explicitly set `fallback_model` to an xAI model
    (e.g. `'xai:grok-4-1-fast-non-reasoning'`) to enable a subagent-based fallback.
    There is no default fallback model — attempting to use `XSearch` on a non-xAI
    model without `fallback_model` will error.
    """

    fallback_model: XSearchFallbackModel
    """Model to use for X search when the agent's model doesn't support it natively.

    Required for non-xAI models; leave as `None` (the default) when running on an xAI
    model. Must be a model that supports X search via the
    [`XSearchTool`][pydantic_ai.builtin_tools.XSearchTool] builtin (i.e. an xAI model),
    for example `'xai:grok-4-1-fast-non-reasoning'`.

    Can be a model name string, `Model` instance, or a callable taking `RunContext`
    that returns a `Model` instance.
    """

    allowed_x_handles: list[str] | None
    """If provided, only posts from these X handles will be included (max 10). Requires builtin support."""

    excluded_x_handles: list[str] | None
    """If provided, posts from these X handles will be excluded (max 10). Requires builtin support."""

    from_date: datetime | None
    """If provided, only posts created on or after this datetime will be included."""

    to_date: datetime | None
    """If provided, only posts created on or before this datetime will be included."""

    enable_image_understanding: bool
    """Enable image analysis from X posts. Defaults to `False`."""

    enable_video_understanding: bool
    """Enable video analysis from X content. Defaults to `False`."""

    include_output: bool
    """Include raw X search results in the response as
    [`BuiltinToolReturnPart`][pydantic_ai.messages.BuiltinToolReturnPart]. Defaults to `False`.
    """

    def __init__(
        self,
        *,
        builtin: XSearchTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[XSearchTool | None] | XSearchTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | Literal[False] | None = None,
        fallback_model: Model
        | KnownModelName
        | str
        | Callable[[RunContext[AgentDepsT]], Awaitable[Model] | Model]
        | None = None,
        allowed_x_handles: list[str] | None = None,
        excluded_x_handles: list[str] | None = None,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        enable_image_understanding: bool = False,
        enable_video_understanding: bool = False,
        include_output: bool = False,
    ) -> None:
        if fallback_model is not None and local is not None:
            raise UserError(
                'XSearch: cannot specify both `fallback_model` and `local` — '
                'use `fallback_model` for the default subagent fallback, or `local` for a custom tool'
            )
        self.builtin = builtin
        self.local = local
        self.fallback_model = fallback_model
        self.allowed_x_handles = allowed_x_handles
        self.excluded_x_handles = excluded_x_handles
        self.from_date = from_date
        self.to_date = to_date
        self.enable_image_understanding = enable_image_understanding
        self.enable_video_understanding = enable_video_understanding
        self.include_output = include_output
        self.__post_init__()

    def _default_builtin(self) -> XSearchTool:
        return XSearchTool(
            allowed_x_handles=self.allowed_x_handles,
            excluded_x_handles=self.excluded_x_handles,
            from_date=self.from_date,
            to_date=self.to_date,
            enable_image_understanding=self.enable_image_understanding,
            enable_video_understanding=self.enable_video_understanding,
            include_output=self.include_output,
        )

    def _builtin_unique_id(self) -> str:
        return XSearchTool.kind

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        if self.fallback_model is None:
            return None
        from pydantic_ai.common_tools.x_search import x_search_tool

        return x_search_tool(model=self.fallback_model, builtin_tool=self._resolved_builtin())

    def _resolved_builtin(self) -> XSearchTool:
        """Get the XSearchTool for the fallback, with capability-level settings applied."""
        if isinstance(self.builtin, XSearchTool):
            return self.builtin
        return self._default_builtin()

    def _requires_builtin(self) -> bool:
        if self.fallback_model is not None:
            # Subagent's xAI model enforces handle constraints via its native builtin
            return False
        return self.allowed_x_handles is not None or self.excluded_x_handles is not None
