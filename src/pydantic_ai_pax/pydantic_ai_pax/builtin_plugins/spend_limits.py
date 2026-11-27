"""Stop a run or a conversation once it has spent its budget, with the totals in the status row.

The built-in `spend_limits` plugin, off by default: harness `SpendLimits`, built with `SpendLimits.from_spec` from
the `budgets` the user saved, and a status-row fragment showing each budget's spend against its ceiling.

Pax keeps the counters in the plugin's `InMemorySpendStore`, so only the `run` and `conversation` windows are
offered: a `day` or `month` window would start again from zero whenever Pax restarted, which a ceiling must not do.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SerializerFunctionWrapHandler,
    TypeAdapter,
    field_validator,
    model_serializer,
    model_validator,
)

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, on_event
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.spend_limits import BudgetSpec, SpendLimitExceeded, SpendLimits, SpendRecordedEvent, Window
from pydantic_ai_pax.plugins import Plugin, PluginHost, TurnEnd, TurnStart
from pydantic_ai_pax.plugins.menus import FieldRow, SettingsMenu
from pydantic_ai_pax.ui.menus.field_menu import first_error, shown
from pydantic_ai_pax.ui.rendering import theme
from pydantic_ai_pax.ui.rendering.status import StatusSegment
from pydantic_ai_pax.ui.rendering.usage_report import format_cost

_RECONFIGURE = '/plugins configure spend_limits'


class BudgetSetting(BaseModel):
    """One budget as saved: the part of harness `BudgetSpec` Pax offers.

    Only the windows Pax can keep, and no `warn_at`: nothing in Pax would show its warning. `retain` is left to
    harness, since the counters do not outlive the process anyway.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)
    window: Window = Field(description='`run` (one prompt and its tool calls) or `conversation`.')
    usd: Decimal | None = Field(
        default=None, gt=0, allow_inf_nan=False, description='Ceiling in US dollars; unset counts without a ceiling.'
    )
    tokens: int | None = Field(default=None, gt=0, description='Ceiling in total tokens; unset sets none.')
    name: str | None = Field(default=None, description='Tells apart two budgets on one window.')

    @field_validator('window')
    @classmethod
    def _kept_in_memory(cls, window: Window) -> Window:
        if window not in ('run', 'conversation'):
            raise ValueError(
                f'window {window!r} is not offered: Pax keeps spend in memory, so a {window} total would start '
                "again from zero whenever Pax restarts. Use 'run' or 'conversation'."
            )
        return window

    @model_serializer(mode='wrap')
    def _only_given(self, handler: SerializerFunctionWrapHandler) -> dict[str, JsonValue]:
        """Save only the fields someone gave, as harness `BudgetSpec` would read them."""
        dumped: dict[str, JsonValue] = handler(self)
        return {key: value for key, value in dumped.items() if key in self.model_fields_set}

    def spec(self) -> BudgetSpec:
        """The harness spec entry, leaving out what was not set so harness defaults apply."""
        spec: BudgetSpec = {'window': self.window}
        if self.usd is not None:
            # A string, so the amount does not round through a float on its way to `Decimal`.
            spec['usd'] = str(self.usd)
        if self.tokens is not None:
            spec['tokens'] = self.tokens
        if self.name is not None:
            spec['name'] = self.name
        return spec


class SpendLimitsSettings(BaseModel):
    """What `/plugins add spend_limits pydantic_ai_pax.builtin_plugins.spend_limits '{...}'` may set."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    budgets: tuple[BudgetSetting, ...] = Field(
        default=(),
        description='JSON list of budgets, such as [{"usd": "5", "window": "run"}]. Each needs a `window`.',
    )
    on_unpriced: Literal['zero', 'raise'] = Field(
        default='zero',
        description='For a model with no known price: `zero` counts it as $0 and marks the total `>=`; '
        '`raise` stops the run.',
    )

    @model_validator(mode='after')
    def _builds(self) -> Self:
        """Refuse what harness would refuse, such as two USD ceilings on one counter, when the settings are saved."""
        try:
            self.limits()
        except UserError as exc:
            raise ValueError(str(exc)) from exc
        return self

    @model_serializer(mode='wrap')
    def _only_chosen(self, handler: SerializerFunctionWrapHandler) -> dict[str, JsonValue]:
        """Save only the settings someone chose, so menu edits never pin the other defaults."""
        dumped: dict[str, JsonValue] = handler(self)
        return {key: value for key, value in dumped.items() if key in self.model_fields_set}

    def limits(self) -> SpendLimits[None]:
        """The harness capability these settings describe, with a counter store of its own."""
        return SpendLimits[None].from_spec(
            budgets=[budget.spec() for budget in self.budgets], on_unpriced=self.on_unpriced
        )


@dataclass(frozen=True, kw_only=True)
class _Reading:
    """What one budget's counter held after the latest response a conversation recorded."""

    usd: Decimal
    tokens: int
    unpriced: int


@dataclass
class _SpendGauge(AbstractCapability[None]):
    """Keep each budget's latest reading per conversation, for the status row."""

    readings: dict[tuple[str | None, str, str], _Reading]

    @on_event(SpendRecordedEvent)
    async def _record(self, ctx: RunContext[None], event: SpendRecordedEvent) -> None:
        for status in event.budgets:
            self.readings[ctx.conversation_id, status.name, status.window] = _Reading(
                usd=status.spent_usd, tokens=status.spent_tokens, unpriced=status.unpriced_requests
            )


class SpendLimitsSource:
    """The budgets as one JSON row and the unpriced-model policy, saved through the validated model."""

    title = 'Spend limits settings'

    def __init__(self, host: PluginHost[None]) -> None:
        self.host = host

    def rows(self) -> tuple[FieldRow, ...]:
        fields = SpendLimitsSettings.model_fields
        return (
            FieldRow(
                key='budgets', label='Budgets', description=fields['budgets'].description or '', default=shown([])
            ),
            FieldRow(
                key='on_unpriced',
                label='Unpriced models',
                description=fields['on_unpriced'].description or '',
                default='zero',
                choices=('zero', 'raise'),
                allow_custom=False,
            ),
        )

    def current(self, row: FieldRow) -> str:
        values: dict[str, JsonValue] = self.host.settings(SpendLimitsSettings).model_dump(mode='json')
        return shown(values.get(row.key, [] if row.key == 'budgets' else 'zero'))

    def _updated(self, row: FieldRow, raw: str) -> SpendLimitsSettings:
        data: dict[str, JsonValue] = self.host.settings(SpendLimitsSettings).model_dump(mode='json')
        data[row.key] = TypeAdapter(JsonValue).validate_json(raw) if row.key == 'budgets' else raw
        return SpendLimitsSettings.model_validate(data)

    def problem(self, row: FieldRow, text: str) -> str | None:
        try:
            self._updated(row, text)
        except ValueError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self.host.save_settings(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        data: dict[str, JsonValue] = self.host.settings(SpendLimitsSettings).model_dump(mode='json')
        data.pop(row.key, None)
        self.host.save_settings(SpendLimitsSettings.model_validate(data))
        return f'Reset {row.label}.'


class SpendLimitsPlugin(Plugin[SpendLimitsSettings]):
    """Harness `SpendLimits` over the saved budgets, and their totals in the status row.

    Counters live as long as the plugin: changing its settings or reloading it starts them again.
    """

    def __init__(self, host: PluginHost[None], settings: SpendLimitsSettings) -> None:
        super().__init__(host, settings)
        self.limits = settings.limits()
        self._readings: dict[tuple[str | None, str, str], _Reading] = {}

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self.limits, _SpendGauge(readings=self._readings))

    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (self._segment,)

    def settings_menu(self) -> SettingsMenu:
        return SettingsMenu(source=SpendLimitsSource(self.host))

    async def on_turn_start(self, event: TurnStart) -> None:
        # Each prompt is a new run, whose counter starts at zero.
        for key in [key for key in self._readings if key[2] == 'run']:
            del self._readings[key]

    async def on_turn_end(self, event: TurnEnd) -> None:
        if isinstance(event.error, SpendLimitExceeded):
            self.host.console.print(
                f'A spend limit stopped this turn. Change the budgets with {_RECONFIGURE}.',
                style=theme.color(theme.MUTED),
            )

    def _segment(self) -> str:
        """Each budget's spend in the shown conversation against its ceilings; empty without budgets."""
        conversation = self.host.conversation.conversation_id
        figures = [
            _figure(budget, self._readings.get((conversation, budget.name or _DEFAULT_NAME, budget.window), _NOTHING))
            for budget in self.settings.budgets
        ]
        return f'spend: {", ".join(figures)}' if figures else ''


_DEFAULT_NAME = 'default'
"""The name harness `Budget` takes when none is given."""
_NOTHING = _Reading(usd=Decimal(0), tokens=0, unpriced=0)
"""A counter nothing has been recorded against yet, as a new run's or conversation's is."""


def _figure(budget: BudgetSetting, reading: _Reading) -> str:
    """`run $0.0125/$1.0000`, with `>=` and a count when a response could not be priced."""
    label = budget.window if budget.name is None else f'{budget.window} {budget.name}'
    parts: list[str] = []
    if budget.usd is not None or budget.tokens is None:
        ceiling = '' if budget.usd is None else f'/{format_cost(budget.usd)}'
        parts.append(f'{">=" if reading.unpriced else ""}{format_cost(reading.usd)}{ceiling}')
    if budget.tokens is not None:
        parts.append(f'{reading.tokens:,}/{budget.tokens:,} tokens')
    unpriced = f' ({reading.unpriced} unpriced)' if reading.unpriced else ''
    return f'{label} {" ".join(parts)}{unpriced}'
