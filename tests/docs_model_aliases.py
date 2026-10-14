"""Resolve the model aliases that doc pages use, as `docs/model_aliases.py` defines them.

`{{ openai:medium }}` stands for the full model string, like `openai:<model>`, and
`{{ openai:medium.name }}` for the bare model name, like `<model>`. The docs site expands them the same way.
"""

from __future__ import annotations as _annotations

import difflib
import re
import runpy
from pathlib import Path

_MODEL_ALIASES_FILE = Path(__file__).parent.parent / 'docs' / 'model_aliases.py'
MODEL_ALIASES: dict[str, str] = runpy.run_path(str(_MODEL_ALIASES_FILE))['MODEL_ALIASES']

_ALIAS_RE = re.compile(r'\{\{\s*([a-z0-9-]+:[a-z0-9-]+)(\.name)?\s*\}\}')


def resolve_model_aliases(text: str) -> str:
    """Replace each model alias in `text` with the model it stands for."""

    def resolve(match: re.Match[str]) -> str:
        alias, bare = match.groups()
        if (model := MODEL_ALIASES.get(alias)) is None:
            raise ValueError(f'Unknown model alias {match.group()!r}, see `docs/model_aliases.py`')
        return model.partition(':')[2] if bare else model

    return _ALIAS_RE.sub(resolve, text)


def _model_name_aliases() -> dict[str, str]:
    """Map each aliased model, as a full and a bare name, back to its alias; the first alias wins for a shared model."""
    aliases: dict[str, str] = {}
    for alias, model in MODEL_ALIASES.items():
        aliases.setdefault(model, f'{{{{ {alias} }}}}')
    for alias, model in MODEL_ALIASES.items():
        aliases.setdefault(model.partition(':')[2], f'{{{{ {alias}.name }}}}')
    return aliases


_MODEL_NAME_ALIASES = _model_name_aliases()
_MODEL_NAME_RE = re.compile(
    r'(?<![\w.:-])(' + '|'.join(map(re.escape, sorted(_MODEL_NAME_ALIASES, key=len, reverse=True))) + r')(?![\w.-])'
)


def alias_model_names(text: str) -> str:
    """Replace each aliased model in `text` with its alias, reversing `resolve_model_aliases`."""
    return _MODEL_NAME_RE.sub(lambda match: _MODEL_NAME_ALIASES[match.group()], text)


def restore_model_aliases(source: str, resolved: str, updated: str) -> str:
    """Return `source` with the changes that turned `resolved` into `updated`, the models in them aliased.

    `resolved` is `source` with its aliases resolved, and `updated` is what `--update-examples` made of it,
    so unchanged lines keep their aliases and only the lines the update wrote are aliased.
    """
    source_lines = source.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, resolved.splitlines(keepends=True), updated_lines, autojunk=False)
    restored: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        restored += source_lines[i1:i2] if tag == 'equal' else map(alias_model_names, updated_lines[j1:j2])
    return ''.join(restored)
