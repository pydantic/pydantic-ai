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

from ._merge import merge_capability_fields, merge_field_values
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
    and `_requires_native`.
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
    """

    local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = None
    """Configure the local fallback tool.

    - `None` (default): auto-detect a local fallback via `_default_local`.
    - `True`: opt in to the default local fallback (resolved via `_resolve_local_strategy`).
    - `False`: disable the local fallback; only use the native tool.
    - A named strategy (e.g. `'duckduckgo'`): resolved via `_resolve_local_strategy` in subclasses.
    - A `Tool` or `AbstractToolset` instance: use this specific local tool.
    - A bare callable: automatically wrapped in a `Tool`.
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

    _declared_native: AgentNativeTool[AgentDepsT] | bool | None = field(
        init=False, repr=False, compare=False, default=None
    )
    """What the caller passed as `native`, before `__post_init__` resolved it.

    Resolution is destructive: `native=True` becomes a tool instance, so afterwards there is no way
    to tell configuration the caller stated from configuration this class derived. `combine` needs
    that distinction to rebuild the derived half from merged configuration. Excluded from `compare`
    because it restates a field that is already compared.
    """

    _declared_local: str | Tool[AgentDepsT] | Callable[..., Any] | AbstractToolset[AgentDepsT] | bool | None = field(
        init=False, repr=False, compare=False, default=None
    )
    """What the caller passed as `local`, before `__post_init__` resolved it. See `_declared_native`."""

    def __post_init__(self) -> None:
        # Assigned through `object.__setattr__` rather than relying on the field defaults: the
        # subclasses declare their own `__init__`, which never runs the dataclass field
        # initializers. Every caller reaches here with `native`/`local` as stated rather than
        # resolved -- a fresh `__init__`, or `combine` having just put the merged declarations
        # back -- so capturing unconditionally records declarations, never resolved values.
        object.__setattr__(self, '_declared_native', self.native)
        object.__setattr__(self, '_declared_local', self.local)
        if self.native is False and self.local is False:
            raise UserError(f'{type(self).__name__}: both `native` and `local` cannot be False')

        # Resolve native=True → default instance (subclass hook)
        if self.native is True:
            default = self._default_native()
            if default is None:
                raise UserError(
                    f'{type(self).__name__}: native=True requires a subclass that overrides '
                    f'`_default_native()`, or pass an `AbstractNativeTool` instance directly'
                )
            self.native = default

        # Resolve local: None → default, True/str → named strategy, callable → Tool
        if self.local is None:
            self.local = self._default_local()
        elif self.local is True or isinstance(self.local, str):
            self.local = self._resolve_local_strategy(self.local)
        elif self.local is False:
            pass
        elif callable(self.local) and not isinstance(self.local, (Tool, AbstractToolset)):
            self.local = Tool(self.local)

        # Catch contradictory config: native disabled but constraint fields require it.
        # Checked first because adding `local=` can't fix it — the user needs to either drop
        # the constraint or re-enable native.
        if self.native is False and self._requires_native():
            raise UserError(f'{type(self).__name__}: constraint fields require the native tool, but native=False')

        # Disallow `native=False` without an explicit local — would produce a silent no-op capability.
        if self.native is False and self.local is None:  # pyright: ignore[reportUnknownMemberType]
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
        native = self.native
        if isinstance(native, AbstractNativeTool):
            return native.unique_id
        raise UserError(
            f'{type(self).__name__}: cannot derive native unique_id — override `_native_unique_id()` in your subclass'
        )

    def _default_local(self) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT] | None:
        """Auto-detect a local fallback. Override in subclasses that have one."""
        return None

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
        if self.native is False:
            return []
        # After __post_init__, native=True is resolved to an AbstractNativeTool instance
        assert not isinstance(self.native, bool)
        return [self.native]

    def get_toolset(self) -> AbstractToolset[AgentDepsT] | None:
        local = self.local
        if local is None or local is False or self._requires_native():
            return None

        # local is Tool | AbstractToolset after __post_init__ resolution.
        # When wrapping a bare local callable, stamp the capability's `id` onto the toolset so it can
        # be used with durable execution (which wraps leaf toolsets by `id`). An `AbstractToolset`
        # passed as `local=` keeps its own id and is never overwritten.
        toolset: AbstractToolset[AgentDepsT] = (
            cast(AbstractToolset[AgentDepsT], local)
            if isinstance(local, AbstractToolset)
            else FunctionToolset([cast(Tool[AgentDepsT], local)], id=self.id)
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
        """Merge the declared configuration, then rebuild the native tool from the result.

        `__post_init__` copies this capability's configuration into the native tool it builds, and
        that tool -- not the capability -- is what reaches the provider. Merging the capability's
        fields alone would leave a merged `allowed_domains` beside a native tool still carrying one
        instance's, so a composed restriction would read as applied while the request went out
        without it. Anything `_default_native` produced is therefore produced again from the merged
        configuration.

        A native tool the user passed in is left alone: it states its own configuration, and
        rebuilding would discard it. Two of those take the later, like any other value the merge
        cannot reconcile.

        The merged instance is validated the way a constructed one is. `replace_no_init` skips
        `__post_init__`, and a merge can reach a combination no constructor would accept -- a
        `native=False` instance beside one carrying native-only constraints leaves a capability that
        contributes neither the native tool nor a local fallback. Re-running the check turns that
        into the same `UserError` writing it by hand would raise.
        """
        nol_capabilities = [capability for capability in capabilities if isinstance(capability, NativeOrLocalTool)]
        assert len(nol_capabilities) == len(capabilities)
        merged = merge_capability_fields(capabilities)
        assert isinstance(merged, cls)
        # `native`/`local` are set to the *declarations* rather than the resolved values, so the
        # `__post_init__` below re-records them as such and then resolves them once, exactly as a
        # constructor would.
        merged = replace_no_init(
            merged,
            native=merge_field_values([capability._declared_native for capability in nol_capabilities]),
            local=merge_field_values([capability._declared_local for capability in nol_capabilities]),
        )
        merged.__post_init__()
        return merged

    def _resolve_native_with_overrides(
        self, tool_cls: type[_NativeToolT], overrides: dict[str, Any]
    ) -> _NativeToolT | Callable[[RunContext[AgentDepsT]], Awaitable[_NativeToolT] | _NativeToolT]:
        """Resolve the native tool for the fallback subagent, with capability-level overrides applied.

        Handles every `native` shape reaching here: an instance (overridden via
        `dataclasses.replace`), `False` (a default instance with overrides), or a factory (wrapped
        so its resolved result is overridden the same way). `True` never arrives — `__post_init__`
        has already resolved it to an instance. A factory that returns `None` raises `UserError`
        rather than substituting a default instance, and anything else raises too.

        Only the `fallback_model` path reaches here: `__post_init__` calls `_default_local()`, which
        returns early when `fallback_model` is unset, so a capability configured without one never
        runs this check. Validating in `__post_init__` instead would reject configurations that
        construct fine today for users who never opted into a fallback subagent.
        """
        if isinstance(self.native, tool_cls):
            return replace(self.native, **overrides) if overrides else self.native

        if self.native is False:
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
