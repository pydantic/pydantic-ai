"""Conversation-local settings and actions behind `/set`."""

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter, ValidationError

from pydantic_ai.settings import ModelSettings
from pydantic_clai2.commands import Command, set_completions
from pydantic_clai2.config import SETTING_FIELDS, STRING_SETTINGS, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.ui import telemetry


@runtime_checkable
class CommandProvider(Protocol):
    """Compatibility interface for capability-provided terminal commands."""

    def get_commands(self, context: 'CommandContext') -> Sequence[Command]:
        """Declare commands at startup."""
        ...


@dataclass(kw_only=True)
class CommandContext:
    """Conversation-local settings and actions, without global mutable state."""

    settings: Settings
    store: SettingsStore
    clear_history: Callable[[], None]
    apply_setting: Callable[[str, Settings], None]
    project: ProjectSettings = field(default_factory=ProjectSettings)
    """Read-only here: `/set` writes the user store, and the project file wins again at next start."""
    plugin_models: Callable[[], Sequence[str]] = lambda: ()
    """Models loaded plugins offer with `PluginHost.model_provider`, as `PREFIX:NAME`."""
    settings_model: Callable[[str], str] = lambda model: model
    """The model whose `/model settings` controls and defaults a model takes.

    Differs for a plugin's `settings_from`, and for a chain, which takes its first model's.
    """

    def __post_init__(self) -> None:
        """Keep the configured model selectable, including preferences saved before the model list existed."""
        if self.settings.model:
            self.store.add_model(name=self.settings.model)

    def from_project(self, key: str) -> bool:
        """Whether the project file sets `key`, so a saved value only lasts for this session."""
        return key in self.project.overrides

    def set_setting(self, args: list[str]) -> str:
        """Validate, persist, and apply a preference to the current conversation."""
        if len(args) == 1 and args[0] in SETTING_FIELDS:
            return str(self.settings.model_dump()[SETTING_FIELDS[args[0]]])
        if len(args) != 2 or args[0] not in SETTING_FIELDS:
            raise ValueError('Usage: /set SETTING VALUE. Press Tab for suggestions.')
        key, raw = args
        value, settings = self.validate(key, raw)
        self.store.set(key, value)
        self._apply(key, settings)
        return f'Saved {key}. ' + self._when(key)

    def validate(self, key: str, raw: str) -> tuple[JsonValue, Settings]:
        """Parse typed text for `key` and check it against the whole settings model; nothing is saved.

        A model whose provider SDK is not installed raises `ValueError` here, not on the next turn.
        """
        adapter: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
        value: JsonValue = (
            raw
            if key in STRING_SETTINGS or (key == 'sessions.naming_model' and raw != 'null')
            else adapter.validate_json(raw)
        )
        updated = self.settings.model_dump()
        updated[SETTING_FIELDS[key]] = value
        settings = Settings.model_validate(updated)
        if key == 'model' and settings.model:
            from pydantic_clai2.models.model_catalog import check_installed

            check_installed(settings.model)
        return value, settings

    def model_settings(self, model: str) -> ModelSettings | None:
        """Family defaults plus saved overrides, ready for `agent.run`."""
        from pydantic_clai2.models.model_settings import model_settings_from_json

        form = model_settings_from_json(self.store.model_settings(model), model=self.settings_model(model))
        return form.to_model_settings()

    def live_model_settings(self, model: str) -> Callable[[object], ModelSettings]:
        """`model_settings` for a run, read again before each model request.

        Core resolves a callable per request, so edits saved while a turn runs reach its next
        request. Invalid saved settings raise `ValidationError` now, before the run starts; an
        invalid read later keeps the last valid settings instead of failing the running turn.
        """
        current = self.model_settings(model) or ModelSettings()

        def resolve(_: object) -> ModelSettings:
            nonlocal current
            with suppress(ValidationError):
                current = self.model_settings(model) or ModelSettings()
            return current

        return resolve

    def reset_setting(self, key: str) -> str:
        """Forget the saved override and apply the default now."""
        self.store.reset(key)
        updated = self.settings.model_dump()
        field = SETTING_FIELDS[key]
        updated[field] = Settings().model_dump()[field]
        self._apply(key, Settings.model_validate(updated))
        return f'Reset {key}. ' + self._when(key)

    def _apply(self, key: str, settings: Settings) -> None:
        self.settings = settings
        self.apply_setting(key, settings)
        telemetry.record('setting {setting} changed', setting=key, value=_shown(key, settings))

    def _when(self, key: str) -> str:
        when = 'Applies at next startup.' if key == 'display.splash' else 'Applied.'
        if self.from_project(key):
            when += ' The project file sets it again at next start.'
        return when


def _shown(key: str, settings: Settings) -> bool | int | float | str:
    """A setting's new value for UI telemetry: flags, numbers, and listed choices, but never typed text."""
    value: object = getattr(settings, SETTING_FIELDS[key])
    if value is None:
        return 'null'
    if isinstance(value, bool | int | float):
        return value
    return value if isinstance(value, str) and value in set(set_completions([key, ''])) else 'custom'
