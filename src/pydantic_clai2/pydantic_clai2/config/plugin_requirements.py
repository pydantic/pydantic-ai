"""Which saved plugin settings need which features, and what a build lacking them uses instead.

The requirements live in their own `plugin_requirements` table, one JSON object per plugin id
mapping a setting key to the feature names it needs: `{"sub_agents": ["stock-bound-delegation"]}`.
Builds that predate the table never read it, so they keep loading every declaration unchanged.
Pure functions only; `SettingsStore` and the loader do the reading and writing.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from pydantic_clai2.config import features
from pydantic_clai2.config.features import check_feature_name

Requirements = dict[str, frozenset[str]]
"""Setting key to the feature names its saved value needs."""

UNREADABLE = '(unreadable)'
"""Stands in for a requirement this build cannot parse. It is not a valid feature name, so no build supports it."""


def declared_requirements(requires: Mapping[str, Iterable[str]]) -> Requirements:
    """Validate a plugin's declaration; raise `ValueError` naming the first bad entry."""
    result: Requirements = {}
    for key, required in requires.items():
        names = frozenset(check_feature_name(name) for name in required)
        if not names:
            raise ValueError(f'Setting {key!r} lists no required features; leave it out instead.')
        result[key] = names
    return result


def stored_requirements(row: JsonValue | None, settings: Mapping[str, JsonValue]) -> Requirements:
    """What each of `settings` needs according to a stored row.

    A row that is not a JSON object, or an entry that is not a list of strings, cannot be read, so
    the settings it covers need `UNREADABLE`: ignoring a setting is safe, applying it may not be.
    """
    if row is None:
        return {}
    if not isinstance(row, dict):
        return {key: frozenset({UNREADABLE}) for key in settings}
    result: Requirements = {}
    for key, value in row.items():
        if key not in settings:
            continue
        if isinstance(value, list) and all(isinstance(name, str) for name in value):
            result[key] = frozenset(name for name in value if isinstance(name, str))
        else:
            result[key] = frozenset({UNREADABLE})
    return result


@dataclass(frozen=True, kw_only=True)
class Applied:
    """Saved settings with unsupported ones replaced by defaults, and what was ignored."""

    settings: dict[str, JsonValue]
    ignored: Requirements
    """Each ignored key and the features it needs that this build lacks."""


def apply_requirements(
    settings: Mapping[str, JsonValue],
    requirements: Requirements,
    *,
    defaults: Mapping[str, JsonValue],
    supported: frozenset[str] | None = None,
) -> Applied:
    """Drop every setting needing a feature outside `supported`, by default this build's `SUPPORTED_FEATURES`.

    A dropped key takes its value from `defaults` (the shipped declaration), or is left out so the
    plugin's own default applies. Nothing else is ever substituted.
    """
    known = features.SUPPORTED_FEATURES if supported is None else supported
    kept = dict(settings)
    ignored: Requirements = {}
    for key, needs in requirements.items():
        missing = needs - known
        if key not in kept or not missing:
            continue
        ignored[key] = missing
        if key in defaults:
            kept[key] = defaults[key]
        else:
            del kept[key]
    return Applied(settings=kept, ignored=ignored)


def withheld(settings: Mapping[str, JsonValue], requirements: Requirements) -> dict[str, JsonValue]:
    """The saved values this build ignores, to write back unchanged when it saves the plugin's settings."""
    ignored = apply_requirements(settings, requirements, defaults={}).ignored
    return {key: settings[key] for key in ignored}


def ignored_notice(plugin: str, ignored: Requirements) -> str:
    """One line per plugin, such as `coder: ignored saved sub_agents (needs stock-bound-delegation); using defaults.`."""
    keys = ', '.join(sorted(ignored))
    features = ', '.join(sorted({name for names in ignored.values() for name in names}))
    return f'{plugin}: ignored saved {keys} (needs {features}); using defaults.'


def merged_requirements(
    *,
    old_settings: Mapping[str, JsonValue],
    old_row: JsonValue | None,
    new_settings: Mapping[str, JsonValue],
    declared: Requirements,
) -> dict[str, JsonValue]:
    """The row to store with `new_settings`: the writer's tags, plus stored tags on values it did not change.

    A tag disappears only when its value changes, so a build that does not know a feature cannot
    strip the tag from a value another build saved. An unreadable stored entry on an unchanged value
    is kept as it is, so it stays ignored everywhere.
    """
    row: dict[str, JsonValue] = {}
    for key, value in new_settings.items():
        names = set(declared.get(key, ()))
        previous = _previous(old_row, key) if key in old_settings and old_settings[key] == value else None
        if isinstance(previous, list) and all(isinstance(name, str) for name in previous):
            names.update(name for name in previous if isinstance(name, str))
        elif previous is not None:
            row[key] = previous
            continue
        if names:
            row[key] = list[JsonValue](sorted(names))
    return row


def _previous(old_row: JsonValue | None, key: str) -> JsonValue | None:
    """The stored entry for `key`; a row that is not an object makes every entry unreadable."""
    if isinstance(old_row, dict):
        if key not in old_row:
            return None
        # A stored `null` entry is unreadable, not absent, so it must survive the save.
        entry = old_row[key]
        return [UNREADABLE] if entry is None else entry
    return None if old_row is None else [UNREADABLE]
