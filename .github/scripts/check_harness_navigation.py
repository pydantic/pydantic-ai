#!/usr/bin/env python
"""Check `docs/navigation.yml` against the Pydantic AI Harness release it publishes.

The harness is documented in its own repository, but which of its pages are
published is decided here: unified-docs enumerates them from the `source:
"harness"` entries in `docs/navigation.yml`. Nothing else connects the two, so
each side can drift without noticing.

Both directions of drift have bitten us:

* An entry whose page does not exist fails the unified-docs build at config
  load, which is the *deploy*, not a check. Twelve such entries were approved
  and waiting to merge before this check existed (unified-docs#279).
* A released page with no entry is simply unreachable. `mutation-testing`
  shipped in harness v0.31.0 and stayed unpublished, with nothing linking to it
  so no link check ever fired.

Pages are compared against the harness's newest release rather than its `main`,
because that is what unified-docs builds (`refPattern: 'latest:v0.*'`). Linking
a page before its release would publish docs for something nobody can install.

Usage:
    python .github/scripts/check_harness_navigation.py [--verbose]
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

HARNESS_REPO = 'pydantic/pydantic-ai-harness'
"""Kept in sync with the `extraRepos` entry for the `ai` library in unified-docs."""

RELEASE_TAG_PATTERN = re.compile(r'^v0\.\d+(\.\d+)*$')
"""Mirrors unified-docs' `refPattern: 'latest:v0.*'`, which skips prereleases."""

REPO_ROOT = Path(__file__).parent.parent.parent
NAVIGATION = REPO_ROOT / 'docs' / 'navigation.yml'
ALLOWLIST = REPO_ROOT / '.github' / 'harness-navigation-allowlist.json'


def _parse_version(tag: str) -> tuple[int, ...]:
    return tuple(int(part) for part in tag.lstrip('v').split('.'))


def latest_harness_release(client: httpx.Client) -> str:
    """The newest non-prerelease `v0.*` tag, chosen the way unified-docs chooses it."""
    tags: list[str] = []
    url = f'https://api.github.com/repos/{HARNESS_REPO}/tags?per_page=100'
    while url:
        response = client.get(url)
        response.raise_for_status()
        tags.extend(tag['name'] for tag in response.json())
        url = response.links.get('next', {}).get('url', '')

    matching = [tag for tag in tags if RELEASE_TAG_PATTERN.match(tag)]
    if not matching:  # pragma: no cover - only reachable if the harness retags
        raise RuntimeError(f'No v0.* release tags found in {HARNESS_REPO}')
    return max(matching, key=_parse_version)


def harness_doc_pages(client: httpx.Client, ref: str) -> set[str]:
    """Every `docs/*.md` page present in the harness at `ref`, as navigation spells it."""
    response = client.get(f'https://api.github.com/repos/{HARNESS_REPO}/git/trees/{ref}?recursive=1')
    response.raise_for_status()
    tree = response.json()
    if tree.get('truncated'):  # pragma: no cover - the harness tree is far below the cap
        raise RuntimeError(f'The git tree for {HARNESS_REPO}@{ref} was truncated')
    return {
        entry['path'].removeprefix('docs/')
        for entry in tree['tree']
        if entry['type'] == 'blob'
        and entry['path'].startswith('docs/')
        and entry['path'].endswith('.md')
        and '/' not in entry['path'].removeprefix('docs/')
    }


def navigation_harness_paths(navigation: Any) -> dict[str, str]:
    """Maps each `source: "harness"` page path to the slug it is published under."""
    paths: dict[str, str] = {}

    def walk(entries: Any) -> None:
        for entry in entries or ():
            if not isinstance(entry, dict):  # pragma: no cover - manifest is schema-checked
                continue
            if entry.get('source') == 'harness' and 'path' in entry:
                paths[entry['path']] = entry.get('slug', '')
            walk(entry.get('contents'))

    walk(navigation.get('navigation'))
    return paths


def load_allowlist() -> dict[str, str]:
    """Harness pages deliberately left unpublished, each with the reason why."""
    if not ALLOWLIST.exists():
        return {}
    return json.loads(ALLOWLIST.read_text())['unpublished']


def navigation_problems(
    linked_pages: dict[str, str],
    released_pages: set[str],
    allowlist: dict[str, str],
    release: str,
) -> list[str]:
    """Both directions of drift between the manifest and the released harness."""
    problems: list[str] = []

    for path, slug in sorted(linked_pages.items()):
        if path not in released_pages:
            problems.append(
                f'docs/navigation.yml links `{path}` (as /ai/{slug}), but {HARNESS_REPO}@{release} '
                f'has no docs/{path}. Merging this would fail the unified-docs build, which runs '
                f'on deploy. Wait for the harness release that ships the page.'
            )

    for path in sorted(released_pages - linked_pages.keys()):
        if path in allowlist:
            continue
        problems.append(
            f'{HARNESS_REPO}@{release} ships docs/{path}, but docs/navigation.yml does not link '
            f'it, so the page is unreachable. Add an entry with `source: "harness"`, or record why '
            f'it stays unpublished in .github/harness-navigation-allowlist.json.'
        )

    return problems


def main() -> int:
    """Compare the committed manifest with the released harness; non-zero on drift."""
    verbose = '--verbose' in sys.argv

    headers = {'accept': 'application/vnd.github+json'}
    if token := os.environ.get('GITHUB_TOKEN'):
        headers['authorization'] = f'Bearer {token}'

    with httpx.Client(headers=headers, timeout=30, follow_redirects=True) as client:
        release = latest_harness_release(client)
        released_pages = harness_doc_pages(client, release)

    navigation = yaml.safe_load(NAVIGATION.read_text())
    linked_pages = navigation_harness_paths(navigation)
    allowlist = load_allowlist()

    if verbose:
        print(f'{HARNESS_REPO} latest release: {release}')
        print(f'{len(released_pages)} released pages, {len(linked_pages)} linked from navigation')
        for path in sorted(released_pages & allowlist.keys() - linked_pages.keys()):
            print(f'allowed: {path} ({allowlist[path]})')

    problems = navigation_problems(linked_pages, released_pages, allowlist, release)
    for problem in problems:
        print(f'ERROR: {problem}', file=sys.stderr)

    if problems:
        print(f'\n{len(problems)} harness navigation problem(s) found.', file=sys.stderr)
        return 1

    print(f'docs/navigation.yml matches {HARNESS_REPO}@{release}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
