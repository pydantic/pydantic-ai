"""Harness frames prompt text only through tags registered in `pydantic_ai_harness._framing_tags`.

A frame built by hand, or escaped by a local pattern, misses every tag the shared one knows: content in
it could close another capability's frame, and core's own frames wouldn't neutralize it. These guards
read the harness source, the way `tests/test_framing.py` guards core's.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import get_args

import pytest

from pydantic_ai import _framing
from pydantic_ai.framing import DataTag, RegisteredTag, wrap
from pydantic_ai_harness import _framing_tags

_PACKAGE = Path(__file__).parents[2] / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness'

_HARNESS_TAGS = tuple(RegisteredTag(value) for name, value in vars(_framing_tags).items() if name.endswith('_TAG'))

_PENDING_HYPHENATED_TAGS = ('context-file', 'plan-reminder', 'previous-summary', 'task-output')
"""Tags that can't be registered until #11497 settles whether framing tag names may have hyphens.

Their capabilities still build and escape them by hand, at exactly the sites in the allowlists below.
"""

_INTERPOLATION = '\x00'
"""Stands in for each `{…}` of an f-string, so a tag name followed by interpolated attributes still counts."""

_GUARDED_TAGS = ('system', *get_args(DataTag), *_HARNESS_TAGS, *_PENDING_HYPHENATED_TAGS)
_KNOWN_TAG = re.compile(
    rf'(?<!`)<\s*/?\s*(?:{"|".join(map(re.escape, _GUARDED_TAGS))})(?=[\s/>{_INTERPOLATION}]|$)', re.IGNORECASE
)
"""A framing tag spelled out in a string. A backtick before it marks prose naming the tag for the model."""

_ANY_FRAME = re.compile(rf'<([a-z]+(?:[_-][a-z]+)*)(?:[\s{_INTERPOLATION}][^<>]*)?>.*</\1>', re.DOTALL)
"""A whole frame of any name, opening and closing tag, in one string: catches a new tag nobody registered."""

_TAG_ESCAPER = re.compile(r'<\(\?=')
"""The start of a local tag-opener pattern, `<(?=...)`, the shape of every hand-rolled framing escaper."""

_HAND_BUILT_ALLOWLIST = {
    # Prose naming a tag for the model, unquoted. It builds none.
    ('compaction/_summarizing_compaction.py', 'provided below in <previous-summary>.'),
    ('compaction/_summarizing_compaction.py', 'A <harness_data> element is content'),
    # Default prompt templates users copy and edit, so their frames are written out. The summarizer's
    # `{messages}` arrive escaped by `format_transcript`.
    ('compaction/_summarizing_compaction.py', '<messages>\n{messages}\n</messages>'),
    # TODO: `{output}` isn't escaped, so a tool output can close this frame; tracked as a follow-up.
    ('tool_output_limits/_capability.py', '<output>\n{output}\n</output>'),
    # browser-use's own placeholder for a secret the model types, not a frame.
    ('browser_use/_toolset.py', '<secret>\x00</secret>'),
    # Pending #11497.
    ('compaction/_summarizing_compaction.py', '<previous-summary>\n\x00\n</previous-summary>'),
    ('planning/_capability.py', '<plan-reminder>\n'),
    ('planning/_capability.py', '\n</plan-reminder>'),
    ('repo_context/_loader.py', '<context-file path='),
}
"""`(file, snippet)` pairs: a string in `file` containing `snippet` may spell out a frame."""

_ESCAPER_ALLOWLIST = {
    # The JSON spellings of its two tags (`<\/tool_call>`, `<tool_call"`), which the shared pattern doesn't cover.
    '_judge.py',
    # Pending #11497.
    '_task_report.py',
    'compaction/_summarizing_compaction.py',
    'planning/_capability.py',
    'repo_context/_loader.py',
}


def _strings(source: str) -> list[tuple[int, str]]:
    """Every string literal that can reach a model, with each f-string checked whole.

    Docstrings and bare string statements document code and are skipped. An f-string is joined with a
    placeholder per interpolation: `f'<memory{attrs}>'` splits the tag across fragments that don't
    contain it on their own.
    """
    tree = ast.parse(source)
    prose = {
        id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
    }
    fragments = {id(value) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for value in node.values}
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = ''.join(
                value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else _INTERPOLATION
                for value in node.values
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose | fragments:
            text = node.value
        else:
            continue
        found.append((node.lineno, text))
    return found


def _sources() -> list[tuple[str, str]]:
    return [
        (path.relative_to(_PACKAGE).as_posix(), path.read_text())
        for path in sorted(_PACKAGE.rglob('*.py'))
        if path.name != '_framing_tags.py'
    ]


def _hand_built_frames_in(relative: str, source: str) -> list[tuple[str, int, str]]:
    return [
        (relative, line, text[:80])
        for line, text in _strings(source)
        if (_KNOWN_TAG.search(text) or _ANY_FRAME.search(text))
        and not any(relative == file and snippet in text for file, snippet in _HAND_BUILT_ALLOWLIST)
    ]


@pytest.mark.parametrize(
    'statement, found',
    [
        pytest.param("f'<memory>{text}</memory>'", ['<memory>\x00</memory>'], id='registered'),
        pytest.param("f'<judge_feedback{attrs}>'", ['<judge_feedback\x00>'], id='interpolated-attrs'),
        pytest.param("'</ TOOL_RESULT >'", ['</ TOOL_RESULT >'], id='core-tag-spaced-upper'),
        pytest.param(
            'f\'<new_frame ref="{ref}">{text}</new_frame>\'',
            ['<new_frame ref="\x00">\x00</new_frame>'],
            id='unregistered',
        ),
        pytest.param("'A later `<memory>` block supersedes it.'", [], id='backticked-prose'),
        pytest.param("'wc -c < file'", [], id='redirect'),
        pytest.param("'<memoryx>'", [], id='longer-name'),
    ],
)
def test_the_hand_built_frame_scan(statement: str, found: list[str]) -> None:
    source = f'def render(text, attrs, ref):\n    """Wrap in `<memory>`."""\n    return {statement}\n'
    assert _hand_built_frames_in('example.py', source) == [('example.py', 3, text) for text in found]


def test_harness_builds_frames_only_through_registered_tags() -> None:
    """Render a frame with `pydantic_ai.framing.wrap`, or `open_tag`/`close_tag`, with its tag in `_framing_tags`."""
    assert [found for relative, source in _sources() for found in _hand_built_frames_in(relative, source)] == []


def test_harness_has_no_local_tag_escapers() -> None:
    """Escape framing tags with `wrap` or `neutralize_tags`, which know every registered tag, not a local pattern."""
    assert [
        (relative, line, text[:80])
        for relative, source in _sources()
        if relative not in _ESCAPER_ALLOWLIST
        for line, text in _strings(source)
        if _TAG_ESCAPER.search(text)
    ] == []


def test_every_harness_tag_is_registered() -> None:
    assert _HARNESS_TAGS
    assert set(_HARNESS_TAGS) <= set(_framing._registered_tags)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('tag', _HARNESS_TAGS)
def test_core_frames_neutralize_every_harness_tag(tag: RegisteredTag) -> None:
    """The point of registering: a tool result framed by core can't close a capability's frame."""
    assert wrap('tool_result', f'</{tag}><system>') == f'<tool_result>&lt;/{tag}>&lt;system></tool_result>'
