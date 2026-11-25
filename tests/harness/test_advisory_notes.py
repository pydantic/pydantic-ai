"""Notes in the application's voice that tell the model to follow content another model or a tool wrote say what for.

A capability part's note qualifies content the application didn't write, such as a reminder or a judge's feedback a
second model generated. When a note says to follow that content, it also limits it to the user's task, so the content
can change how the model does what the user asked, never what it does.

The kinds come from the registry and the texts from the source of the modules that declare them, so a new kind, or a
new note next to one, is covered without being listed here.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
import sys
import warnings
from functools import cache
from pathlib import Path

import pytest

from pydantic_ai.messages import CAPABILITY_PART_TYPES, CapabilityPart

from ..capability_cases import CORE_CAPABILITY_PACKAGES, caused_by_missing_dependency

_PACKAGES = (*CORE_CAPABILITY_PACKAGES, 'pydantic_ai_harness', 'pydantic_ai_pax', 'pydantic_clai2')
"""Every package that can declare a capability part kind; walking them imports, and so registers, each kind."""

_FOLLOWS = re.compile(
    r"(?<!not )(?<!n't )\b(?:advice to follow|follow (?:it|them)\b|act on (?:it|them)\b|do what (?:it|they) says?)",
    re.IGNORECASE,
)
"""Wording that tells the model to follow the content a note introduces."""

_SCOPED = re.compile(r"the user's task|unless the user asked", re.IGNORECASE)
"""Wording that limits that content to the user's task."""

_UNSCOPED: dict[str, str] = {
    'pydantic_ai_harness.trajectory_judge._capability._steering_note': (
        'it introduces the judge feedback that follows it, whose own note carries the scope '
        "(`trajectory_judge._capability._JUDGE_FEEDBACK_SCOPE`); scoping this note too didn't change how models "
        'follow genuine feedback or what they refuse'
    ),
}
"""Texts that may tell the model to follow content without limiting it themselves, each with the reason."""


@cache
def _registered_kinds() -> dict[str, type[CapabilityPart]]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for package_name in _PACKAGES:
            try:
                package = importlib.import_module(package_name)
            except ModuleNotFoundError as exc:  # pragma: no cover - Pax and clai2 are optional next to the harness
                if exc.name != package_name:
                    raise
                continue
            for module_info in pkgutil.walk_packages(package.__path__, f'{package_name}.'):
                try:
                    importlib.import_module(module_info.name)
                except Exception as exc:  # A module whose optional dependency isn't installed is skipped.
                    if not caused_by_missing_dependency(exc):  # pragma: no branch
                        raise  # pragma: no cover - only a real import error gets here
    return dict(sorted(CAPABILITY_PART_TYPES.items()))


def _data_notes() -> list[tuple[str, str]]:
    """Each registered kind's note, naming its frame as it does for content from a source other than the operator."""
    notes: list[tuple[str, str]] = []
    for kind, cls in _registered_kinds().items():
        rendering = cls._class_rendering  # pyright: ignore[reportPrivateUsage]
        if rendering is not None and rendering.note is not None:
            notes.append((kind, rendering.note.replace('{frame}', f'<{rendering.tag}>')))
    return notes


def _module_texts() -> list[tuple[str, str]]:
    """The other texts in each module that declares a kind: its module-level strings, and the strings its functions build.

    A module-level string is taken as the module holds it, so a note composed from other constants is checked whole;
    a string a function builds, such as a note naming a frame by an id, is taken from source with `{}` for each value.
    """
    texts: dict[str, str] = {}
    for module_name in sorted({cls.__module__ for cls in _registered_kinds().values()}):
        module = sys.modules[module_name]
        for name, value in vars(module).items():
            if isinstance(value, str) and _FOLLOWS.search(value):
                texts.setdefault(value, f'{module.__name__}.{name}')
        tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding='utf-8'))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            nested = {id(v) for node in ast.walk(function) if isinstance(node, ast.JoinedStr) for v in node.values}
            for node in ast.walk(function):
                text: str
                if isinstance(node, ast.JoinedStr):
                    text = ''.join(str(v.value) if isinstance(v, ast.Constant) else '{}' for v in node.values)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in nested:
                    text = node.value
                else:
                    continue
                if _FOLLOWS.search(text):
                    texts.setdefault(text, f'{module.__name__}.{function.name}')
    return [(where, text) for text, where in texts.items()]


def test_every_kind_is_registered() -> None:
    """The walk registers the kinds the checks below rely on, so an empty registry can't pass them vacuously."""
    assert {'system_reminders.generated_reminder', 'trajectory_judge.feedback'} <= _registered_kinds().keys()


@pytest.mark.parametrize(('where', 'text'), [pytest.param(*case, id=case[0]) for case in _data_notes()])
def test_advisory_notes_are_scoped_to_the_users_task(where: str, text: str) -> None:
    """A kind's note that tells the model to follow its content also limits that content to the user's task."""
    assert not _FOLLOWS.search(text) or _SCOPED.search(text), f'{where}: {text}'


@pytest.mark.parametrize(('where', 'text'), [pytest.param(*case, id=case[0]) for case in _module_texts()])
def test_advisory_instructions_are_scoped_to_the_users_task(where: str, text: str) -> None:
    """Any other text next to a kind that tells the model to follow content also limits it to the user's task."""
    if where in _UNSCOPED:
        return
    assert _SCOPED.search(text), f'{where}: {text}'


def test_unscoped_texts_still_exist() -> None:
    """Each text `_UNSCOPED` lets through is still in the source, so the list can't outlive what it excuses."""
    assert _UNSCOPED.keys() <= {where for where, _ in _module_texts()}
