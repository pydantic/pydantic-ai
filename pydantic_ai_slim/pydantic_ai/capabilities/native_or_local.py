from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar, cast

from pydantic_ai._utils import replace_no_init
from pydantic_ai.exceptions import UserError
from pydantic_ai.native_tools import AbstractNativeTool
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext, Tool, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.prepared import PreparedToolset

from ._merge import merge_capability_fields
from ._native_resolution import resolve_native_tool
from .abstract import (
    AbstractCapability,
)

_NativeToolT = TypeVar('_NativeToolT', bound=AbstractNativeTool)


@dataclass(init=False)
class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    """Capability that pairs a provider-native tool with a local fallback.

    When the model supports the native tool, the local fallback is removed.
    When the model doesn't support the native tool, it is removed and the local tool stays.

    Can be used directly:

    ```python {test="skip" lint="skip"}
    from pydantic_ai.capabilities import NativeOrLocalTool

    cap = NativeOrLocalTool(native=WebSearchTool(), local=my_search_func)
    ```

    Or subclassed to set defaults by overriding `_default_native`, `_default_local`,
    `_resolve_local`, `_has_local_fallback`, and `_requires_native`.
    The built-in [`WebSearch`][pydantic_ai.capabilities.WebSearch],
    [`WebFetch`][pydantic_ai.capabilities.WebFetch], and
    [`ImageGeneration`][pydantic_ai.capabilities.ImageGeneration] capabilities
    are all subclasses.
    """

    native: AgentNativeTool[AgentDepsT] | bool = True
    """Configure the provider-native tool.

    - `True` (default): use the default native tool configuration (subclasses only).
    - `False`: disable the native tool; always use the local tool.
    - An `AbstractNativeTool` instance: use this specific configuration.
    - A callable (`NativeToolFunc`): dynamically create the native tool per-run via `RunContext`.
      Returning `None` omits the native tool.

    The field keeps what was passed, so `dataclasses.replace` and merging resolve it again from the
    new configuration. [`get_native_tools()`][pydantic_ai.capabilities.AbstractCapability.get_native_tools]
    returns the tool it resolves to.
    """

    local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None
    """Configure the local fallback tool.

    - `None` (default): auto-detect a local fallback via `_default_local`.
    - `True`: opt in to the default local fallback (resolved via `_resolve_local_strategy`).
    - `False`: disable the local fallback; only use the native tool.
    - A named strategy (e.g. `'duckduckgo'`): resolved via `_resolve_local_strategy` in subclasses.
    - A `Tool` or `AbstractToolset` instance: use this specific local tool.
    - A bare callable: automatically wrapped in a `Tool`.

    The field keeps what was passed, like `native`.
    [`get_toolset()`][pydantic_ai.capabilities.AbstractCapability.get_toolset] returns the toolset it
    resolves to.
    """

    def __init__(
        self,
        *,
        native: AgentNativeTool[AgentDepsT] | bool = True,
        local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None,
        id: str | None = None,
        defer_loading: bool = False,
        description: str | None = None,
    ) -> None:
        self.id = id
        self.description = description
        self.defer_loading = defer_loading
        self.native = native
        self.local = local
        self.__post_init__()

    _native_tool: AgentNativeTool[AgentDepsT] | None = field(init=False, repr=False, compare=False, default=None)
    """The native tool `native` resolves to, or `None` for `native=False`.

    Resolved once per instance by `__post_init__`, so every `get_native_tools()` call returns the
    same tool. Excluded from `compare` and `repr` because `native` already states it, and `init=False`
    so `dataclasses.replace` never feeds it back.
    """

    _local_tool: Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None = field(
        init=False, repr=False, compare=False, default=None
    )
    """The tool or toolset `local` resolves to, or `None` when there is none. See `_native_tool`.

    Kept for the instance's lifetime rather than rebuilt per `get_toolset()` call: that runs at
    agent construction and again per run, and a toolset such as `MCPToolset` holds the connection
    an entered agent reuses.
    """

    def __post_init__(self) -> None:
        if self.native is False and self.local is False:
            raise UserError(f'{type(self).__name__}: both `native` and `local` cannot be False')

        # Resolve native=True → default instance (subclass hook)
        native = self.native
        if native is True:
            native = self._default_native()
            if native is None:
                raise UserError(
                    f'{type(self).__name__}: native=True requires a subclass that overrides '
                    f'`_default_native()`, or pass an `AbstractNativeTool` instance directly'
                )
        # Assigned directly rather than through the field default: the subclasses declare their own
        # `__init__`, which never runs the dataclass field initializers.
        self._native_tool = None if native is False else native
        self._local_tool = self._resolve_local()

        # Catch contradictory config: native disabled but constraint fields require it.
        # Checked first because adding `local=` can't fix it — the user needs to either drop
        # the constraint or re-enable native.
        if self.native is False and self._requires_native():
            raise UserError(f'{type(self).__name__}: constraint fields require the native tool, but native=False')

        # Disallow `native=False` without an explicit local — would produce a silent no-op capability.
        if self.native is False and not self._has_local_fallback():
            raise UserError(
                f'{type(self).__name__}(native=False) requires an explicit local tool — '
                'pass `local=...` (e.g. a strategy string, `True`, a callable, or a `Tool`/`AbstractToolset`).'
            )

    # --- Subclass hooks (not abstract — direct use is supported) ---

    def _default_native(self) -> AbstractNativeTool | None:
        """Create the default native tool instance.

        Override in subclasses. Returns None by default (direct use requires
        passing an explicit `AbstractNativeTool` instance as `native`).
        """
        return None

    def _native_unique_id(self) -> str:
        """The unique_id used for `unless_native` on local tool definitions.

        By default, derived from the native tool's `unique_id` property.
        Override in subclasses for custom behavior.
        """
        native = self._native_tool
        if isinstance(native, AbstractNativeTool):
            return native.unique_id
        raise UserError(
            f'{type(self).__name__}: cannot derive native unique_id — override `_native_unique_id()` in your subclass'
        )

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        """Auto-detect a local fallback. Override in subclasses that have one."""
        return None

    def _resolve_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        """Resolve `local` to the tool or toolset it declares.

        `None` → `_default_local()`, `True` or a string → `_resolve_local_strategy()`, a bare callable
        → a `Tool`. Override in a subclass whose `local` takes a shape of its own.
        """
        local = self.local
        if isinstance(local, (Tool, AbstractToolset)):
            # Narrowing the `Callable` arm of the union leaves the type parameter unknown, though only
            # the `Tool[AgentDepsT]` and `AbstractToolset[AgentDepsT]` arms can reach here.
            return cast('Tool[AgentDepsT] | AbstractToolset[AgentDepsT]', local)
        if local is None:
            return self._default_local()
        if local is True or isinstance(local, str):
            return self._resolve_local_strategy(local)
        if local is False:
            return None
        return Tool(local)

    def _has_local_fallback(self) -> bool:
        """Whether a local fallback is configured, once `local` has been resolved."""
        return self._local_tool is not None

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        """Resolve a named local strategy (e.g. `'duckduckgo'`) or `local=True` to a concrete tool.

        Override in subclasses that expose named strategies. The default implementation raises
        `UserError`.
        """
        raise UserError(
            f'{type(self).__name__}: `local={name!r}` is not supported. '
            'Pass a `Tool`, `AbstractToolset`, or callable directly.'
        )

    def _requires_native(self) -> bool:
        """Return True if capability-level constraint fields require the native tool.

        When True, the local fallback is suppressed. If the model doesn't support
        the native tool, `UserError` is raised — preventing silent constraint violation.

        Override in subclasses that expose native-only constraint fields
        (e.g. `allowed_domains`, `blocked_domains`).
        """
        return False

    # --- Shared logic ---

    def get_native_tools(self) -> Sequence[AgentNativeTool[AgentDepsT]]:
        return [] if self._native_tool is None else [self._native_tool]

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        local = self._local_tool
        if local is None or self._requires_native():
            return None

        # When wrapping a bare local callable, stamp the capability's `id` onto the toolset so it can
        # be used with durable execution (which wraps leaf toolsets by `id`). An `AbstractToolset`
        # passed as `local=` keeps its own id and is never overwritten.
        toolset: AbstractToolset[AgentDepsT] = (
            local if isinstance(local, AbstractToolset) else FunctionToolset([local], id=self.id)
        )

        if self.native is not False:
            uid = self._native_unique_id()

            async def _add_unless_native(
                ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]
            ) -> list[ToolDefinition]:
                return [replace(d, unless_native=uid) for d in tool_defs]

            return PreparedToolset(wrapped=toolset, prepare_func=_add_unless_native)
        return toolset

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge the declared configuration, then resolve the tools again from the result.

        `__post_init__` copies this capability's configuration into the tools it resolves, and those
        tools -- not the capability -- are what reach the provider and the run. Merging the fields
        alone would leave a merged `allowed_domains` beside a native tool still carrying one
        instance's, so a composed restriction would read as applied while the request went out
        without it.

        A native tool the user passed in is left alone: it states its own configuration, and
        rebuilding would discard it. Two of those take the later, like any other value the merge
        cannot reconcile.

        The merged instance is validated the way a constructed one is. `replace_no_init` skips
        `__post_init__`, and a merge can reach a combination no constructor would accept -- a
        `native=False` instance beside one carrying native-only constraints leaves a capability that
        contributes neither the native tool nor a local fallback. Re-running the check turns that
        into the same `UserError` writing it by hand would raise.
        """
        # Copied even when the merge changed nothing, which hands back the last instance itself:
        # resolving again in place would replace the tools of a capability the caller still holds.
        merged = replace_no_init(cls._merge_fields(capabilities))
        assert isinstance(merged, cls)
        merged.__post_init__()
        return merged

    @classmethod
    def _merge_fields(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge the declared fields; `combine` then resolves the tools from the result.

        Override in a subclass with a field the default merge would get wrong.
        """
        return merge_capability_fields(capabilities)

    def _resolve_native_with_overrides(
        self, tool_cls: type[_NativeToolT], overrides: dict[str, Any]
    ) -> _NativeToolT | Callable[[RunContext[AgentDepsT]], Awaitable[_NativeToolT] | _NativeToolT]:
        """Resolve the native tool for the fallback subagent, with capability-level overrides applied.

        Handles every `native` shape: an instance (overridden via `dataclasses.replace`), `True` or
        `False` (a default instance with overrides), or a factory (wrapped so its resolved result is
        overridden the same way). A factory that returns `None` raises `UserError` rather than
        substituting a default instance, and anything else raises too.

        Only the `fallback_subagent_model` path reaches here: a subclass builds its subagent tool only
        when `fallback_subagent_model` is set, so a capability configured without one never runs this
        check. Validating unconditionally instead would reject configurations that construct fine
        today for users who never opted into a fallback subagent.
        """
        if isinstance(self.native, tool_cls):
            return replace(self.native, **overrides) if overrides else self.native

        if isinstance(self.native, bool):
            return tool_cls(**overrides)

        native_factory = self.native
        if not callable(native_factory):
            raise UserError(
                f'{type(self).__name__}: `native` must be `True`, `False`, a callable, or an instance of '
                f'`{tool_cls.__name__}`, not {native_factory!r}'
            )

        async def resolve_native(ctx: RunContext[AgentDepsT]) -> _NativeToolT:
            native_tool = await resolve_native_tool(tool_cls, native_factory, ctx)
            return replace(native_tool, **overrides) if overrides else native_tool

        return resolve_native
