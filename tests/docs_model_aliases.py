"""Resolve the model aliases that doc pages use, as `docs/model_aliases/aliases.json` defines them.

`{{ openai:medium }}` stands for the full model string, like `openai:<model>`, and
`{{ openai:medium.name }}` for the bare model name, like `<model>`. The docs site expands them the same way.
"""

from __future__ import annotations as _annotations

import difflib
import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import TypeAdapter

_MODEL_ALIASES_FILE = Path(__file__).parent.parent / 'docs' / 'model_aliases' / 'aliases.json'
MODEL_ALIASES = TypeAdapter(dict[str, str]).validate_json(_MODEL_ALIASES_FILE.read_bytes())

ALIAS_RE = re.compile(r'\{\{ ([a-z0-9-]+:[a-z]+)(\.name)? \}\}')
"""A model alias, written exactly as the docs site expects it."""


def resolve_model_aliases(text: str) -> str:
    """Replace each model alias in `text` with the model it stands for."""

    def resolve(match: re.Match[str]) -> str:
        alias, bare = match.groups()
        if (model := MODEL_ALIASES.get(alias)) is None:
            raise ValueError(f'Unknown model alias {match.group()!r}, see `docs/model_aliases/aliases.json`')
        return model.partition(':')[2] if bare else model

    return ALIAS_RE.sub(resolve, text)


def alias_model_names(text: str, prefer: Iterable[str] = ()) -> str:
    """Replace each aliased model in `text`, as a full or a bare name, with its alias.

    A model that several aliases stand for gets the first of them in `prefer`, else the first in `MODEL_ALIASES`.
    """
    full: dict[str, str] = {}
    bare: dict[str, str] = {}
    for alias in [*(alias for alias in prefer if alias in MODEL_ALIASES), *MODEL_ALIASES]:
        model = MODEL_ALIASES[alias]
        full.setdefault(model, f'{{{{ {alias} }}}}')
        bare.setdefault(model.partition(':')[2], f'{{{{ {alias}.name }}}}')
    # Full names first, so bare names only match where no full name did, as in `openai-chat:<model>`.
    return _replace_names(_replace_names(text, full, r'(?<![\w.-])'), bare, r'(?<![\w./-])')


def _replace_names(text: str, replacements: dict[str, str], lookbehind: str) -> str:
    names = '|'.join(map(re.escape, sorted(replacements, key=len, reverse=True)))
    return re.sub(rf'{lookbehind}({names})(?![\w.-])', lambda match: replacements[match.group()], text)


def restore_model_aliases(source: str, resolved: str, updated: str) -> str:
    """Return `source` with the changes that turned `resolved` into `updated`, the models in them aliased.

    `resolved` is `source` with its aliases resolved, and `updated` is what `--update-examples` made of it,
    so unchanged lines keep their aliases and only the lines the update wrote are aliased.
    """
    prefer = [match.group(1) for match in ALIAS_RE.finditer(source)]
    source_lines = source.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(None, resolved.splitlines(keepends=True), updated_lines, autojunk=False)
    restored: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            restored += source_lines[i1:i2]
        else:
            restored += (alias_model_names(line, prefer) for line in updated_lines[j1:j2])
    return ''.join(restored)
