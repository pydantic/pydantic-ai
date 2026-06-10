from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from pydantic_ai._utils import install_deprecated_kwarg_alias
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.native_tools import XSearchTool
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset

from .native_or_local import NativeOrLocalTool

if TYPE_CHECKING:
    from pydantic_ai.common_tools.x_search import XSearchFallbackModel


@dataclass(init=False)
class XSearch(NativeOrLocalTool[AgentDepsT]):
    """X (Twitter) search capability.

    On xAI models, uses the native X search directly with no extra configuration.

    On non-xAI models, you must explicitly set `fallback_model` to an xAI model
    (e.g. `'xai:grok-4.3'`) to enable a subagent-based fallback.
    There is no default fallback model — attempting to use `XSearch` on a non-xAI
    model without `fallback_model` will error.
    """

    fallback_model: XSearchFallbackModel
    """Model to use for X search when the agent's model doesn't support it natively.

    Required for non-xAI models; leave as `None` (the default) when running on an xAI
    model. Must be a model that supports X search via the
    [`XSearchTool`][pydantic_ai.native_tools.XSearchTool] native tool (i.e. an xAI model),
    for example `'xai:grok-4.3'`.

    Can be a model name string, `Model` instance, or a callable taking `RunContext`
    that returns a `Model` instance.
    """

    allowed_x_handles: list[str] | None
    """If provided, only posts from these X handles will be included (max 10).

    Honored by the native X search tool, whether used directly on an xAI model or via the `fallback_model` subagent.
    """

    excluded_x_handles: list[str] | None
    """If provided, posts from these X handles will be excluded (max 10).

    Honored by the native X search tool, whether used directly on an xAI model or via the `fallback_model` subagent.
    """

    from_date: datetime | None
    """If provided, only posts created on or after this datetime will be included."""

    to_date: datetime | None
    """If provided, only posts created on or before this datetime will be included."""

    enable_image_understanding: bool | None
    """Enable image analysis from X posts. When unset, inherits the native tool's default (`False`)."""

    enable_video_understanding: bool | None
    """Enable video analysis from X content. When unset, inherits the native tool's default (`False`)."""

    include_output: bool | None
    """Include raw X search results in the response as
    [`NativeToolReturnPart`][pydantic_ai.messages.NativeToolReturnPart].

    When unset, inherits the native tool's default (`False`).
    """

    def __init__(
        self,
        *,
        native: XSearchTool
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
        enable_image_understanding: bool | None = None,
        enable_video_understanding: bool | None = None,
        include_output: bool | None = None,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        if fallback_model is not None and local is not None:
            raise UserError(
                'XSearch: cannot specify both `fallback_model` and `local` — '
                'use `fallback_model` for the default subagent fallback, or `local` for a custom tool'
            )
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.native = native
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

    def _xsearch_kwargs(self) -> dict[str, Any]:
        """Collect non-None XSearchTool config fields."""
        kwargs: dict[str, Any] = {}
        if self.allowed_x_handles is not None:
            kwargs['allowed_x_handles'] = self.allowed_x_handles
        if self.excluded_x_handles is not None:
            kwargs['excluded_x_handles'] = self.excluded_x_handles
        if self.from_date is not None:
            kwargs['from_date'] = self.from_date
        if self.to_date is not None:
            kwargs['to_date'] = self.to_date
        if self.enable_image_understanding is not None:
            kwargs['enable_image_understanding'] = self.enable_image_understanding
        if self.enable_video_understanding is not None:
            kwargs['enable_video_understanding'] = self.enable_video_understanding
        if self.include_output is not None:
            kwargs['include_output'] = self.include_output
        return kwargs

    def _default_native(self) -> XSearchTool:
        return XSearchTool(**self._xsearch_kwargs())

    def _native_unique_id(self) -> str:
        return XSearchTool.kind

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        if self.fallback_model is None:
            return None
        from pydantic_ai.common_tools.x_search import x_search_tool

        return x_search_tool(model=self.fallback_model, native_tool=self._resolved_native())

    def _requires_native(self) -> bool:
        # Handle constraints can only be enforced by the native XSearchTool.
        # When a `fallback_model` is set, the subagent runs the native tool too,
        # so the local fallback can satisfy the constraints — don't require native.
        if self.fallback_model is not None:
            return False
        return self.allowed_x_handles is not None or self.excluded_x_handles is not None

    def _resolved_native(self) -> XSearchTool:
        """Get the XSearchTool for the fallback, with capability-level overrides applied."""
        base = self.native if isinstance(self.native, XSearchTool) else XSearchTool()
        overrides = self._xsearch_kwargs()
        if not overrides:
            return base
        return replace(base, **overrides)


install_deprecated_kwarg_alias(XSearch, old='builtin', new='native')
