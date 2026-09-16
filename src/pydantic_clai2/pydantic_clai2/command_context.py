"""Conversation-local settings and actions behind `/set`."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pydantic import JsonValue, TypeAdapter
from pydantic_ai.settings import ModelSettings

from .commands import Command
from .config import SETTING_FIELDS, Settings
from .model_settings import model_settings_from_json
from .settings_store import SettingsStore


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
        """Parse typed text for `key` and check it against the whole settings model; nothing is saved."""
        adapter: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
        value: JsonValue = raw if key == 'model' else adapter.validate_json(raw)
        updated = self.settings.model_dump()
        updated[SETTING_FIELDS[key]] = value
        return value, Settings.model_validate(updated)

    def model_settings(self, model: str) -> ModelSettings | None:
        """Saved overrides for `model`, ready for `agent.run`; `None` when there are none."""
        return model_settings_from_json(self.store.model_settings(model)).to_model_settings()

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

    @staticmethod
    def _when(key: str) -> str:
        return 'Applies at next startup.' if key == 'display.splash' else 'Applied.'
