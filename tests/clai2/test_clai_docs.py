"""CLAI's docs ship with the package and the agent reads them a section at a time."""

from __future__ import annotations

import ast
import tomllib
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path
from typing import Literal

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_clai2 import clai_docs
from pydantic_clai2.clai_docs import (
    _CLAI_DOCS,  # pyright: ignore[reportPrivateUsage]
    _DOC_IDS,  # pyright: ignore[reportPrivateUsage]
    _HARNESS_DOCS,  # pyright: ignore[reportPrivateUsage]
    _READ_LIMIT,  # pyright: ignore[reportPrivateUsage]
    CAPABILITY_ID,
    ClaiDocs,
)

PROJECT = Path(clai_docs.__file__).resolve().parent.parent
HARNESS_PACKAGE = Path(str(files('pydantic_ai_harness')))

SYNTHETIC = """\
# Guide

Intro text.

## Setup

Install it with quokka.

```python
# Not a heading
print('''
~~~
''')
```

### Keys

Save an API key in /keys.

## Setup

A second section with the same heading.
"""

LONG_LINE = f'{"-" * 140}needle{"+" * 280}'


def _call_docs_tool(
    name: Literal['read_clai_docs', 'search_clai_docs'],
    arguments: dict[str, str],
    *,
    expect_retry: bool = False,
) -> str:
    """Exercise every lookup through the public capability and real agent tool lifecycle."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            assert [tool.name for tool in info.function_tools] == ['load_capability']
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': CAPABILITY_ID})])
        if len(messages) == 3:
            assert name in {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[ToolCallPart(name, arguments)])
        return ModelResponse(parts=[TextPart('done')])

    result = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[ClaiDocs()]).run_sync('Look up docs')
    [returned] = [
        part
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == name
    ]
    assert returned.outcome == ('retried' if expect_retry else 'success')
    assert isinstance(returned.content, str)
    return returned.content


def read_docs(path: str = '', *, expect_retry: bool = False) -> str:
    """Call `read_clai_docs` through `ClaiDocs`, not its private implementation."""
    return _call_docs_tool('read_clai_docs', {'path': path}, expect_retry=expect_retry)


def search_docs(query: str, *, expect_retry: bool = False) -> str:
    """Call `search_clai_docs` through `ClaiDocs`, not its private implementation."""
    return _call_docs_tool('search_clai_docs', {'query': query}, expect_retry=expect_retry)


@pytest.fixture(autouse=True)
def fresh_docs() -> Iterator[None]:
    """Each test parses the docs it reads, so a patched source is never served from an earlier test."""
    clai_docs._doc.cache_clear()  # pyright: ignore[reportPrivateUsage]
    yield
    clai_docs._doc.cache_clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve a small catalog through the same resource API as the installed packages."""
    package = tmp_path / 'pydantic_clai2'
    (package / '_docs').mkdir(parents=True)
    (package / '_docs' / 'README.md').write_text(SYNTHETIC, encoding='utf-8')
    harness = tmp_path / 'pydantic_ai_harness'
    (harness / 'coder').mkdir(parents=True)
    (harness / 'coder' / 'README.md').write_text(
        f'# Coder\n\n[Source](https://example.com)\n\nFirst sentence of the\nsummary. Second sentence.\n\n## Long\n\n{LONG_LINE}\n',
        encoding='utf-8',
    )
    packages = {'pydantic_clai2': package, 'pydantic_ai_harness': harness}

    def resources(name: str) -> Path:
        return packages[name]

    monkeypatch.setattr(clai_docs, 'files', resources)
    monkeypatch.setattr(clai_docs, '_DOC_IDS', ('readme', 'harness/coder'))


def test_catalog_lists_every_doc() -> None:
    catalog = read_docs()
    for doc_id in _DOC_IDS:
        assert f'\n- {doc_id} (' in catalog


def test_harness_summary_is_the_first_sentence(synthetic: None) -> None:
    assert '- harness/coder (0 KB): First sentence of the summary.' in read_docs().splitlines()


def test_short_doc_is_returned_whole() -> None:
    assert read_docs('customization') == files('pydantic_clai2').joinpath('customization.md').read_text(
        encoding='utf-8'
    )
    assert read_docs(' harness/coder ') == (HARNESS_PACKAGE / 'coder' / 'README.md').read_text(encoding='utf-8')


def test_long_doc_returns_an_outline() -> None:
    outline = read_docs('plugins')
    assert (
        outline.startswith('plugins is ') and ' KB, too long to read whole. Read a section by its path:\n\n' in outline
    )
    assert '\n  - plugins#credentials: Credentials\n' in outline
    assert '\n    - plugins#github-copilot-subscriptions: GitHub Copilot subscriptions\n' in outline
    assert len(outline) < _READ_LIMIT


def test_section_stops_at_the_next_heading_of_its_level() -> None:
    section = read_docs('reference#saved-api-keys')
    assert section.startswith('### Saved API keys\n')
    assert '\n### Usage and cost' not in section
    assert section in (PROJECT / 'docs' / 'reference.md').read_text(encoding='utf-8')


def test_sections_skip_code_and_number_repeated_headings(synthetic: None) -> None:
    assert read_docs('readme') == SYNTHETIC
    assert read_docs('readme#setup') == snapshot("""\
## Setup

Install it with quokka.

```python
# Not a heading
print('''
~~~
''')
```

### Keys

Save an API key in /keys.

""")
    assert read_docs('readme#setup-1') == '## Setup\n\nA second section with the same heading.\n'


def test_long_section_returns_its_own_text_and_an_outline(synthetic: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clai_docs, '_READ_LIMIT', 90)
    assert read_docs('readme') == snapshot("""\
readme is 0 KB, too long to read whole. Read a section by its path:

- readme#guide: Guide
  - readme#setup: Setup
    - readme#keys: Keys
  - readme#setup-1: Setup\
""")
    assert read_docs('readme#guide') == snapshot("""\
# Guide

Intro text.


This section is too long to read whole. Read a subsection by its path:

- readme#setup: Setup
  - readme#keys: Keys
- readme#setup-1: Setup\
""")
    monkeypatch.setattr(clai_docs, '_READ_LIMIT', 10)
    assert read_docs('readme#setup-1') == '## Setup\n\n\n\n[Truncated at 10 characters.]\n'
    assert read_docs('readme#guide') == snapshot("""\
# Guide

I

[Truncated at 10 characters.]

This section is too long to read whole. Read a subsection by its path:

- readme#setup: Setup
  - readme#keys: Keys
- readme#setup-1: Setup\
""")


@pytest.mark.parametrize(
    ('path', 'message'),
    [
        (
            'refrence',
            "There is no CLAI doc 'refrence'. Did you mean reference? Call read_clai_docs with no path to list the docs.",
        ),
        ('nothing', "There is no CLAI doc 'nothing'. Call read_clai_docs with no path to list the docs."),
        (
            'reference#saved-api-key',
            "reference has no section 'saved-api-key'. Did you mean reference#saved-api-keys? Read 'reference' for its outline.",
        ),
        ('reference#zzz', "reference has no section 'zzz'. Read 'reference' for its outline."),
    ],
)
def test_unknown_paths_ask_for_a_known_one(path: str, message: str) -> None:
    assert read_docs(path, expect_retry=True) == message


def test_search_ranks_sections_matching_every_word_in_the_heading() -> None:
    results = search_docs('how do I save an API key?')
    headings = [line for line in results.splitlines() if line.startswith('- ')]
    assert set(headings[:2]) == {
        '- reference#saved-api-keys: Saved API keys',
        '- plugins#saved-api-keys: Saved API keys',
    }
    assert len(headings) == 10
    assert results.endswith('\n\nRead a section with read_clai_docs(path).')


def test_search_snippets_show_the_matching_line(synthetic: None) -> None:
    assert search_docs('quokka') == snapshot("""\
- readme#setup: Setup
  Install it with quokka.

Read a section with read_clai_docs(path).\
""")
    assert search_docs('needle') == snapshot("""\
- harness/coder#long: Long
  ...-----------------------------------------------------needle+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++...

Read a section with read_clai_docs(path).\
""")


def test_title_only_search_results_have_no_snippet(synthetic: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clai_docs, '_DOC_IDS', ('readme',))
    assert search_docs('setup') == snapshot("""\
- readme#setup: Setup
- readme#setup-1: Setup

Read a section with read_clai_docs(path).\
""")


def test_search_with_only_short_or_common_words_still_searches() -> None:
    assert search_docs('how').startswith('- ')


def test_search_without_a_match_says_so() -> None:
    assert search_docs('zzzqqq') == (
        "No CLAI docs section matches 'zzzqqq'. Call read_clai_docs with no path to list the docs."
    )


def test_search_needs_a_word() -> None:
    assert search_docs(' ?! ', expect_retry=True) == 'Search for at least one word.'


def test_docs_resolve_in_editable_and_wheel_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The editable checkout is a fallback; a packaged doc takes precedence when present."""
    package = tmp_path / 'pydantic_clai2'
    package.mkdir()

    def installed(name: str) -> Path:
        return package

    monkeypatch.setattr(clai_docs, 'files', installed)
    assert read_docs('declarative-plugins') == (PROJECT / 'docs' / 'declarative-plugins.md').read_text(encoding='utf-8')
    clai_docs._doc.cache_clear()  # pyright: ignore[reportPrivateUsage]
    (package / '_docs').mkdir()
    (package / '_docs' / 'declarative-plugins.md').write_text('# Packaged\n')
    assert read_docs('declarative-plugins') == '# Packaged\n'


def test_wheel_and_sdist_ship_every_doc_outside_the_package() -> None:
    build = tomllib.loads((PROJECT / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['hatch']['build']['targets']
    outside = [source for source, _ in _CLAI_DOCS.values() if not source.startswith('pydantic_clai2/')]
    assert build['wheel']['force-include'] == {
        source: f'pydantic_clai2/_docs/{Path(source).name}' for source in outside
    }
    assert {f'/{source}' for source in outside} <= set(build['sdist']['include'])
    for source in outside:
        assert (PROJECT / source).is_file()


def test_every_clai_user_doc_is_cataloged() -> None:
    user_docs = {path.relative_to(PROJECT).as_posix() for path in (PROJECT / 'docs').rglob('*.md')}
    user_docs.update({'README.md', 'PLUGINS.md'})
    sources = {source for source, _ in _CLAI_DOCS.values()}
    assert user_docs <= sources


def test_every_harness_capability_clai_imports_has_its_readme_listed() -> None:
    """A capability CLAI starts using gets its README into `_HARNESS_DOCS`; one it stops using leaves it."""
    imported: set[str] = set()
    for module in (PROJECT / 'pydantic_clai2').rglob('*.py'):
        for node in ast.walk(ast.parse(module.read_text(encoding='utf-8'))):
            names = [node.module or ''] if isinstance(node, ast.ImportFrom) else []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            for name in names:
                parts = name.split('.')
                if parts[0] == 'pydantic_ai_harness' and len(parts) > 1:
                    imported.add(parts[1])
    with_readme = {name for name in imported if (HARNESS_PACKAGE / name / 'README.md').is_file()}
    assert with_readme == set(_HARNESS_DOCS)


async def test_agent_searches_then_reads_a_section() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            assert [tool.name for tool in info.function_tools] == ['load_capability']
            return ModelResponse(parts=[ToolCallPart('load_capability', {'id': CAPABILITY_ID})])
        tools = {tool.name: tool for tool in info.function_tools}
        assert tools['search_clai_docs'].parameters_json_schema['additionalProperties'] is False
        assert tools['read_clai_docs'].parameters_json_schema['additionalProperties'] is False
        if len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('search_clai_docs', {'query': 'saved api keys'})])
        if len(messages) == 5:
            return ModelResponse(parts=[ToolCallPart('read_clai_docs', {'path': 'reference#saved-api-keys'})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=type(None), capabilities=[ClaiDocs()])
    result = await agent.run('Where are API keys saved?')
    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name in {'search_clai_docs', 'read_clai_docs'}
    ]
    assert len(returns) == 2
    assert isinstance(returns[0], str) and '- reference#saved-api-keys: Saved API keys' in returns[0]
    assert isinstance(returns[1], str) and returns[1].startswith('### Saved API keys\n')
