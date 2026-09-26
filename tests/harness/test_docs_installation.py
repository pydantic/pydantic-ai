"""Keep user-facing Python installation examples available for pip and uv."""

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_PAGES = sorted((_ROOT / 'docs').rglob('*.md'))
_READMES = [_ROOT / 'README.md', *sorted((_ROOT / 'pydantic_ai_harness').rglob('README.md'))]
_INSTALL = re.compile(r'^[ \t]*(?:pip[ \t]+install|uv[ \t]+add)[ \t]+(.+)$', re.MULTILINE)


@pytest.mark.parametrize('path', _PAGES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_docs_installation_commands_use_shorthand(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    assert not _INSTALL.search(text), f'{path}: use pip/uv-add in a bash fence instead of hand-written install tabs'
    shorthand = re.compile(r'^```bash\n(?:pip/uv-add|py-cli) [^\n]+\n```', re.MULTILINE)
    assert not re.search(r'\b(?:pip/uv-add|py-cli)\b', shorthand.sub('', text)), (
        f'{path}: each shorthand command needs its own single-line bash fence'
    )


@pytest.mark.parametrize('path', _READMES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_installation_commands_are_paired(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    assert not re.search(r'\b(?:pip/uv-add|py-cli)\b', text), f'{path}: READMEs need executable commands'
    pair = re.compile(
        r'^uv:\n\n```bash\n'
        r'uv add (?P<args>[^\n]+)\n'
        r'(?P<uv_followup>(?:(?!```)[^\n]*\n)*)```\n\n'
        r'pip:\n\n```bash\n'
        r'pip install (?P=args)\n'
        r'(?P<pip_followup>(?:(?!```)[^\n]*\n)*)```',
        re.MULTILINE,
    )
    for match in pair.finditer(text):
        assert not _INSTALL.search(match['pip_followup'] + match['uv_followup']), (
            f'{path}: additional installation commands need their own paired blocks'
        )
    assert not _INSTALL.search(pair.sub('', text)), (
        f'{path}: each installation needs adjacent labeled uv / pip blocks, uv first, with identical package arguments'
    )


@pytest.mark.parametrize('path', [*_PAGES, *_READMES], ids=lambda path: str(path.relative_to(_ROOT)))
def test_installation_commands_are_not_inline(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    assert not re.search(r'`(?:pip[ \t]+install|uv[ \t]+add)[ \t]+[^`]+`', text), (
        f'{path}: move inline installation commands into fenced installation examples'
    )
