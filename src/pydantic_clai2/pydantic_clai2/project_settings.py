"""Project-local settings: `.clai/settings.json`, found by walking up from the workspace to the git root."""

from dataclasses import dataclass, field
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from .config import SETTING_FIELDS, PluginSettings, resolve_settings

PROJECT_FILE = Path('.clai') / 'settings.json'
"""The file name, relative to any directory between the workspace and the git root."""

_FILE: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])
_PLUGINS: TypeAdapter[list[PluginSettings]] = TypeAdapter(list[PluginSettings])
_SET_KEYS = {name: key for key, name in SETTING_FIELDS.items()}
"""`Settings` field name to `/set` key, so the file uses the model's own names."""


@dataclass(frozen=True, kw_only=True)
class ProjectSettings:
    """What the project file contributes. The default instance is what "no file" looks like."""

    path: Path | None = None
    overrides: dict[str, JsonValue] = field(default_factory=dict[str, JsonValue])
    """Validated setting values by `/set` key, ready to layer over the user store."""
    plugins: tuple[PluginSettings, ...] = ()
    """Declarations from the file, every one disabled: the user approves each with `/plugins enable`."""
    unknown: tuple[str, ...] = ()
    """Keys the `Settings` model does not know; reported once at startup and ignored."""


def _present(path: Path) -> bool:
    """Like `exists()`, except a permission failure counts as present: it is a boundary, not an absence."""
    try:
        path.stat()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


def find_project_file(workspace: Path) -> Path | None:
    """The nearest project file at or above `workspace`. The search stops at the first `.git`.

    An unreadable project file is still returned, so loading it fails loudly rather than silently
    dropping the repository's settings; an unreadable `.git` still ends the walk.
    """
    start = workspace.resolve()
    for directory in (start, *start.parents):
        candidate = directory / PROJECT_FILE
        if _present(candidate):
            return candidate
        if _present(directory / '.git'):
            return None
    return None


def load_project_settings(workspace: Path) -> ProjectSettings:
    """Read and validate the project file for `workspace`.

    An unreadable file or a bad value fails startup with the path in the message; an unknown key does not.
    """
    path = find_project_file(workspace)
    if path is None:
        return ProjectSettings()
    try:
        raw = _FILE.validate_json(path.read_bytes())
        plugins = _PLUGINS.validate_python(raw.pop('plugins', []))
        overrides = {_SET_KEYS[name]: value for name, value in raw.items() if name in _SET_KEYS}
        resolve_settings(overrides)
    except (OSError, ValidationError) as exc:
        raise ValueError(f'{path}: {exc}') from exc
    return ProjectSettings(
        path=path,
        overrides=overrides,
        plugins=tuple(plugin.model_copy(update={'enabled': False}) for plugin in plugins),
        unknown=tuple(sorted(raw.keys() - _SET_KEYS.keys())),
    )
