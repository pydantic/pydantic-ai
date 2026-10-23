"""Every page under `docs/` is linked from `docs/navigation.yml`, or listed in `UNPUBLISHED_PAGES` with the reason.

unified-docs publishes only the pages `docs/navigation.yml` links, so a page without an entry is unreachable,
and an entry whose page does not exist fails the unified-docs build, which runs on deploy.
"""

from __future__ import annotations as _annotations

from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict

DOCS = Path(__file__).parent.parent / 'docs'

# Instructions for coding agents working in `docs/`, not pages.
AGENT_INSTRUCTION_FILES = {'AGENTS.md', 'CLAUDE.md'}

# Pages left out of `docs/navigation.yml` on purpose. Publishing one is a decision of its own: add its
# navigation entry and remove it from here.
UNPUBLISHED_PAGES = {
    'harness/clai2.md': (
        'Disagrees with the CLAI2 README and code in several places: https://github.com/pydantic/pydantic-ai/issues/8938'
    ),
    'harness/mutation-testing.md': (
        "Contributor guidance for mutation-testing the harness's own filesystem and shell toolsets, "
        'not something a reader of these docs can use.'
    ),
}


class NavigationEntry(TypedDict, total=False):
    path: str
    contents: list[NavigationEntry]


class Navigation(TypedDict):
    navigation: list[NavigationEntry]


def _linked_paths(entries: list[NavigationEntry]) -> Iterator[str]:
    for entry in entries:
        if 'path' in entry:
            yield entry['path']
        yield from _linked_paths(entry.get('contents', []))


_navigation: object = yaml.safe_load((DOCS / 'navigation.yml').read_text(encoding='utf-8'))
LINKED = set(_linked_paths(TypeAdapter(Navigation).validate_python(_navigation)['navigation']))
PAGES = {path.relative_to(DOCS).as_posix() for path in DOCS.rglob('*.md') if path.name not in AGENT_INSTRUCTION_FILES}


def test_every_page_is_linked_unless_unpublished():
    assert sorted(PAGES - LINKED) == sorted(UNPUBLISHED_PAGES), (
        'Link every docs page from `docs/navigation.yml`, or add it to `UNPUBLISHED_PAGES` with the reason it '
        'stays unpublished. Remove an `UNPUBLISHED_PAGES` entry once its page is linked or deleted.'
    )


def test_every_linked_page_exists():
    assert sorted(LINKED - PAGES) == []
