from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic_ai.exceptions import UserError

from ._codec import IDENTITY_CODEC, DurabilityCodec
from ._operation import ToolsetKind
from ._runtime_toolsets import RuntimeToolsetKind
from ._toolset import Lifecycle


@dataclass(frozen=True, kw_only=True)
class DurabilityEngineSpec:
    """Declarative configuration for a durable execution engine."""

    engine_name: str
    """Human-readable engine name used in error messages (e.g. `'Temporal'`)."""

    durable_unit_noun: str
    """Name for one durable unit of work, such as `'activity'`, `'step'`, or `'task'`."""

    durable_container_noun: str
    """Name for the durable container, such as `'workflow'` or `'flow'`."""

    durable_unit_plural: str | None = None
    """Plural name for durable units of work; defaults to `durable_unit_noun + 's'`."""

    codec: DurabilityCodec = IDENTITY_CODEC
    """How the base serializes at every durable boundary. Identity for object-passing engines
    (Temporal/DBOS/Prefect), JSON for journal engines (Restate/Lambda/Absurd)."""

    serialization_failure: Callable[[Exception], BaseException] | None = None
    """Convert deterministic codec failures into an engine's terminal error type.

    JSON-journal engines set this to map `PydanticSerializationError` and `TypeError` failures so
    the engine does not retry values that can never be encoded.
    """

    wrapped_toolset_kinds: frozenset[ToolsetKind] = frozenset({'function', 'mcp', 'dynamic'})
    """Which leaf-toolset kinds this engine wraps in a durable unit. DBOS omits `'function'`
    (function tools run inline via `@DBOS.step`)."""

    toolset_lifecycles: Mapping[ToolsetKind, Lifecycle] = field(
        default_factory=lambda: {
            'function': 'enter-always',
            'mcp': 'enter-always',
            'dynamic': 'enter-never',
        }
    )
    """Per-kind lifecycle profile (`enter-always` / `enter-outside-durable` / `enter-never`).
    Forced explicit because two real bugs came from defaulted gates (#5477 requirement 3).
    Restate opts function tools out of entry (`enter-never`)."""

    tool_call_result_upgrade_lenient: bool = False
    """Decode tool results recorded before control-flow exceptions were wrapped as values.

    Enable this only for stores containing recordings from released DBOS or Prefect integrations,
    or from a sibling integration package that used the earlier payload format. New engines must
    leave this `False`: lenient decoding cannot distinguish a raw earlier recording from a corrupt
    or mis-decoded payload, so strict decoding keeps corruption visible instead of passing garbage
    to the model.
    """

    journal_discovery: bool = True
    """Whether toolset discovery (`get_tools`/`get_instructions`) runs in its own durable unit.
    Durable engines normally journal discovery; integrations may disable it when discovery must
    run in the durable container itself."""

    sequential_tools_in_durable_context: bool = False
    """Whether tool calls must run sequentially inside the durable container."""

    unsupported_runtime_toolset_kinds: frozenset[RuntimeToolsetKind] = frozenset()
    """Runtime toolset kinds rejected inside the durable container because they bypass registration."""

    tool_config_key: str | None = None
    """Tool metadata key containing engine-specific durable configuration, if supported."""

    def __post_init__(self) -> None:
        errors: list[str] = []
        if not self.durable_unit_noun:
            errors.append('`durable_unit_noun` must not be empty')
        if self.durable_unit_plural == '':
            errors.append('`durable_unit_plural` must not be empty')
        if not self.durable_container_noun:
            errors.append('`durable_container_noun` must not be empty')
        missing_lifecycles = self.wrapped_toolset_kinds - self.toolset_lifecycles.keys()
        if missing_lifecycles:
            errors.append(f'missing toolset lifecycles for: {sorted(missing_lifecycles)!r}')
        if errors:
            raise UserError(f'Invalid {self.engine_name} durability engine spec: {"; ".join(errors)}.')
