"""Keep the Pax front pages in sync: `docs/harness/clai2.md` and the Pax README.

The page used to be generated from the README. It is a page in its own right now, so these tests
take over from the generator: the two surfaces tell one story, and only the markup each renderer
needs may differ. See the "Pax front pages" section of `docs/AGENTS.md` for the contract.
"""

from __future__ import annotations as _annotations

import re
from pathlib import Path

from .docs_model_aliases import resolve_model_aliases

ROOT = Path(__file__).parent.parent
PAGE = ROOT / 'docs' / 'harness' / 'clai2.md'
README = ROOT / 'src' / 'pydantic_ai_pax' / 'README.md'


def _code_blocks(text: str) -> list[str]:
    """Every fenced block's body, in order, whatever the language."""
    return [m.group(2).strip() for m in re.finditer(r'^```([a-z]*)\n(.*?)\n```', text, re.S | re.M)]


def test_code_blocks_match() -> None:
    """Commands a reader copies must be identical on both surfaces.

    The page names models by alias, since the docs site resolves them; the README names the model
    itself, because GitHub and PyPI do not.
    """
    page = _code_blocks(resolve_model_aliases(PAGE.read_text(encoding='utf-8')))
    readme = _code_blocks(README.read_text(encoding='utf-8'))
    assert page == readme, (
        'the Pax page and the Pax README disagree on a code block. They are two surfaces of one '
        'story and no longer generated from each other, so edit both.'
    )


def test_page_links_within_the_docs_are_relative() -> None:
    """Relative links keep previews working and let the offline link check verify them."""
    absolute = re.findall(r'\]\((https://pydantic\.dev/docs/ai/[^)]*)\)', PAGE.read_text(encoding='utf-8'))
    assert not absolute, (
        f'{PAGE.relative_to(ROOT)} links to the docs site by URL: {absolute}. Use a relative path '
        'so the link check and doc previews resolve it.'
    )


def test_readme_links_to_the_docs_site_by_url() -> None:
    """The README renders on PyPI and GitHub, where a relative docs path goes nowhere."""
    relative = re.findall(r'\]\((?!https?://|#)([^)]*\.md(?:#[^)]*)?)\)', README.read_text(encoding='utf-8'))
    assert not relative, (
        f'{README.relative_to(ROOT)} has relative links to Markdown: {relative}. PyPI and GitHub '
        'cannot resolve those, so link to the docs site by URL.'
    )


def test_no_em_dashes() -> None:
    for path in (PAGE, README):
        assert '—' not in path.read_text(encoding='utf-8'), f'em dash found in {path.relative_to(ROOT)}'


def _prose(text: str, *, is_readme: bool) -> list[str]:
    """The document's paragraphs with the surface-specific markup normalized away.

    Link targets, the README's badges and the page's front matter are the parts that are meant to
    differ. Everything else is shared wording, and the generator used to guarantee it byte for
    byte; this keeps that guarantee now that nothing regenerates either file.

    Paragraphs rather than lines, because an absolute documentation URL is far longer than the
    relative path it replaces, so the two surfaces cannot wrap at the same places.
    """
    if not is_readme:
        text = re.sub(r'\A---\n.*?\n---\n', '', text, flags=re.S)
    text = re.sub(r'^```.*?^```', '', text, flags=re.S | re.M)  # code has its own test
    kept: list[str] = []
    for line in text.splitlines():
        if is_readme and line.startswith('[!['):
            continue  # badge row, PyPI only
        kept.append(re.sub(r'\]\([^)]*\)', '](LINK)', line))
    paragraphs = re.split(r'\n\s*\n', '\n'.join(kept))
    return [' '.join(p.split()) for p in paragraphs if p.split()]


def test_prose_matches() -> None:
    """Shared wording stays shared. Only links, badges and front matter may differ."""
    page = _prose(PAGE.read_text(encoding='utf-8'), is_readme=False)
    readme = _prose(README.read_text(encoding='utf-8'), is_readme=True)
    assert page == readme, (
        'the Pax page and the Pax README disagree on wording. Edit both: they are two surfaces of '
        'one story and nothing regenerates them. Only link targets, the README badges and the '
        "page's front matter are allowed to differ."
    )
