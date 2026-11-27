"""Let the model consult a second, stronger model; the built-in `advisor` plugin, off until enabled.

Harness `Advisor` gives the model an `advisor` tool. Pax builds the advisor model each run through the
session's own resolver, the one `/model` uses, so saved `/keys` entries, `/login` sign-ins, accounts,
plugin model prefixes, and `chain:` names work for it. A name Pax leaves to core stays a name, so an
`anthropic:` advisor beside an Anthropic conversation model still gets Anthropic's own advisor tool.

It binds its capability eagerly rather than deferred: it is one small tool the user turned on to be
used, and a deferred one would cost the model a `load_capability` call before every first consultation.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.advisor import Advisor
from pydantic_ai_pax.models.profiles import parse_model
from pydantic_ai_pax.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_ai_pax.plugins.menus import FieldRow, SettingsMenu
from pydantic_ai_pax.runtime.session import WorkspacelessCapability
from pydantic_ai_pax.ui.menus.field_menu import first_error, shown
from pydantic_ai_pax.ui.rendering import theme

CONFIGURE = '/plugins configure advisor'
NO_MODEL = f'Advisor: no model is set, so the model has no advisor to consult. Choose one with {CONFIGURE}.'


class AdvisorSettings(BaseModel):
    """The JSON an `advisor` declaration may carry."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    model: str | None = Field(
        default=None,
        description=(
            'The model to consult, named as in /model: PROVIDER[@ACCOUNT]:MODEL or chain:NAME. '
            'Until one is set, the advisor adds nothing.'
        ),
    )

    @field_validator('model')
    @classmethod
    def _valid_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        ref = parse_model(value)
        if not ref.provider or not ref.name or value != value.strip() or not value.isprintable():
            raise ValueError('Use a nonempty PROVIDER[@ACCOUNT]:MODEL or chain:NAME.')
        return value


class AdvisorPlugin(Plugin[AdvisorSettings, DepsT]):
    """Harness `Advisor` on a model Pax resolves, and a settings menu to choose it."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        if self.settings.model is None:
            return ()
        return (WorkspacelessCapability[DepsT](self._for_run),)

    async def _for_run(self, _ctx: RunContext[DepsT]) -> Advisor[DepsT]:
        # Resolved per run, so a key or sign-in changed in `/keys` or `/login` reaches the next prompt.
        name = self.settings.model
        assert name is not None, '`get_capabilities` binds nothing without a model'
        try:
            model = await self.host.resolve_model(name)
        except UserError as exc:
            raise UserError(f'Advisor model {name}: {exc} Choose another with {CONFIGURE}.') from exc
        return Advisor[DepsT](model)

    def settings_menu(self) -> SettingsMenu:
        return SettingsMenu(source=_Settings(self.host))

    async def on_session_start(self, event: SessionStart) -> None:
        if self.settings.model is None:
            self.host.console.print(NO_MODEL, style=theme.color(theme.WARNING), markup=False)


class _Settings:
    """The advisor's one row, saved through the validated model; empty input resets it."""

    title = 'Advisor settings'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        self._host = host

    def rows(self) -> tuple[FieldRow, ...]:
        return (
            FieldRow(
                key='model',
                label='Advisor model',
                description=AdvisorSettings.model_fields['model'].description or '',
                default=shown(None),
            ),
        )

    def current(self, row: FieldRow) -> str:
        return shown(self._host.settings(AdvisorSettings).model)

    def problem(self, row: FieldRow, text: str) -> str | None:
        try:
            AdvisorSettings(model=text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self._host.save_settings(AdvisorSettings(model=raw))
        return f'The advisor is {raw} from the next prompt.'

    def reset(self, row: FieldRow) -> str:
        self._host.save_settings(AdvisorSettings())
        return 'The advisor has no model, so it adds nothing.'
