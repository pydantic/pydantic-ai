from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

import check_harness_navigation as check

REPO_ROOT = Path(__file__).parent.parent.parent


def test_walks_nested_sections_and_ignores_pydantic_ai_pages():
    navigation = yaml.safe_load("""
navigation:
  - section: Capabilities
    contents:
      - page: Pydantic AI Harness
        path: index.md
        source: harness
        slug: harness
      - section: Control & Safety
        contents:
          - page: Handle Deferred Tool Calls
            path: capabilities/handle-deferred-tool-calls.md
            slug: capabilities/handle-deferred-tool-calls
          - page: Guardrails
            path: guardrails.md
            source: harness
            slug: harness/guardrails
""")

    assert check.navigation_harness_paths(navigation) == {
        'index.md': 'harness',
        'guardrails.md': 'harness/guardrails',
    }


def test_flags_an_entry_whose_page_is_not_in_the_release():
    # The failure mode behind the twelve approved-but-unmergeable navigation PRs:
    # the page does not exist, so unified-docs aborts at config load on deploy.
    problems = check.navigation_problems({'notion.md': 'harness/notion'}, set(), {}, 'v0.31.0')

    assert len(problems) == 1
    assert 'has no docs/notion.md' in problems[0]
    assert 'would fail the unified-docs build' in problems[0]


def test_flags_a_released_page_nothing_links():
    problems = check.navigation_problems({}, {'mutation-testing.md'}, {}, 'v0.31.0')

    assert len(problems) == 1
    assert 'does not link it, so the page is unreachable' in problems[0]


def test_allowlisted_pages_are_not_flagged():
    assert (
        check.navigation_problems({}, {'clai2.md'}, {'clai2.md': 'experimental client, not a capability'}, 'v0.31.0')
        == []
    )


def test_an_allowlist_entry_does_not_excuse_a_missing_page():
    """Allowlisting says 'do not publish', never 'publish something that is absent'."""
    problems = check.navigation_problems({'clai2.md': 'harness/clai2'}, set(), {'clai2.md': 'experimental'}, 'v0.31.0')

    assert len(problems) == 1
    assert 'has no docs/clai2.md' in problems[0]


def test_a_matching_manifest_reports_nothing():
    assert check.navigation_problems({'guardrails.md': 'harness/guardrails'}, {'guardrails.md'}, {}, 'v0.31.0') == []


@pytest.mark.parametrize(
    'tags,expected',
    [
        (['v0.9.0', 'v0.10.0', 'v0.31.0'], 'v0.31.0'),
        # Ordering is by version, not lexicographic, and prereleases are skipped
        # the way unified-docs' `latest:v0.*` skips them.
        (['v0.31.0', 'v0.32.0a1', 'v1.0.0', 'nightly'], 'v0.31.0'),
    ],
)
def test_picks_the_newest_release_tag(tags: list[str], expected: str):
    matching = [tag for tag in tags if check.RELEASE_TAG_PATTERN.match(tag)]
    assert max(matching, key=check._parse_version) == expected  # pyright: ignore[reportPrivateUsage]


def test_the_committed_allowlist_is_shaped_as_the_script_expects():
    allowlist = json.loads((REPO_ROOT / '.github' / 'harness-navigation-allowlist.json').read_text())

    assert isinstance(allowlist['unpublished'], dict)
    # Every omission has to be a decision somebody recorded, not a blank entry.
    assert all(reason.strip() for reason in allowlist['unpublished'].values())


def test_every_harness_entry_in_the_committed_manifest_is_a_docs_page():
    navigation = yaml.safe_load((REPO_ROOT / 'docs' / 'navigation.yml').read_text())
    paths = check.navigation_harness_paths(navigation)

    assert paths, 'the manifest should carry harness entries'
    assert all(path.endswith('.md') and '/' not in path for path in paths)
