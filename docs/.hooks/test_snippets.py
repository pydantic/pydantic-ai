from __future__ import annotations as _annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from snippets import (
    REPO_ROOT,
    LineRange,
    ParsedFile,
    RenderedSnippet,
    SnippetDirective,
    format_highlight_lines,
    inject_snippets,
    parse_file_sections,
    parse_snippet_directive,
)


@contextmanager
def temp_text_file(content: str):
    """Context manager for temporary text file with common params."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', encoding='utf-8', delete=False) as f:
        f.write(content)
        temp_name = f.name

    try:
        yield Path(temp_name)
    finally:
        os.unlink(temp_name)


def test_parse_snippet_directive_basic():
    """Test basic parsing of snippet directives."""
    line = '```snippet {path="test.py"}```'
    result = parse_snippet_directive(line)
    assert result == snapshot(
        SnippetDirective(path='test.py', title=None, fragment=None, highlight=None, extra_attrs=None)
    )


def test_parse_snippet_directive_all_attrs():
    """Test parsing with all standard attributes."""
    line = '```snippet {path="src/main.py" title="Main Module" fragment="init setup" highlight="error-handling"}'
    result = parse_snippet_directive(line)
    assert result == snapshot(
        SnippetDirective(
            path='src/main.py', title='Main Module', fragment='init setup', highlight='error-handling', extra_attrs=None
        )
    )


def test_parse_snippet_directive_extra_attrs():
    """Test parsing with extra attributes."""
    line = '```snippet {path="test.py" custom="value" another="attr"}'
    result = parse_snippet_directive(line)
    assert result == snapshot(
        SnippetDirective(
            path='test.py',
            title=None,
            fragment=None,
            highlight=None,
            extra_attrs={'another': 'attr', 'custom': 'value'},
        )
    )


def test_parse_snippet_directive_missing_path():
    """Test that missing path raises ValueError."""
    line = '```snippet {title="Test"}'
    with pytest.raises(ValueError, match='Missing required key "path" in snippet directive'):
        parse_snippet_directive(line)


def test_parse_snippet_directive_invalid_format():
    """Test that invalid format returns None."""
    assert parse_snippet_directive('```python') is None
    assert parse_snippet_directive("snippet {path='test.py'}") is None
    assert parse_snippet_directive('```snippet') is None
    assert parse_snippet_directive('```snippet```') is None


def test_parse_snippet_directive_whitespace():
    """Test parsing with various whitespace."""
    line = '   ```snippet   {   path="test.py"   }   '
    result = parse_snippet_directive(line)
    assert result == snapshot(
        SnippetDirective(path='test.py', title=None, fragment=None, highlight=None, extra_attrs=None)
    )


def test_parse_file_sections_basic():
    """Test basic section parsing."""
    content = """line 1
### [section1]
content 1
content 2
### [/section1]
line 6"""

    with temp_text_file(content) as temp_path:
        result = parse_file_sections(temp_path)

    assert result == snapshot(
        ParsedFile(
            lines=['line 1', 'content 1', 'content 2', 'line 6'],
            sections={'section1': [LineRange(start_line=1, end_line=3)]},
            lines_mapping={0: 0, 1: 2, 2: 3, 3: 5},
        )
    )


def test_parse_file_sections_multiple_ranges():
    """Test section with multiple disjoint ranges."""
    content = """line 1
### [section1]
content 1
### [/section1]
middle line
### [section1]
content 2
### [/section1]
end line"""

    with temp_text_file(content) as temp_path:
        result = parse_file_sections(temp_path)

    assert result == snapshot(
        ParsedFile(
            lines=[
                'line 1',
                'content 1',
                'middle line',
                'content 2',
                'end line',
            ],
            sections={'section1': [LineRange(start_line=1, end_line=2), LineRange(start_line=3, end_line=4)]},
            lines_mapping={0: 0, 1: 2, 2: 4, 3: 6, 4: 8},
        )
    )


def test_parse_file_sections_comment_style():
    """Test parsing with /// comment style."""
    content = """line 1
/// [section1]
content 1
/// [/section1]
line 5"""

    with temp_text_file(content) as temp_path:
        result = parse_file_sections(temp_path)

    assert result == snapshot(
        ParsedFile(
            lines=['line 1', 'content 1', 'line 5'],
            sections={'section1': [LineRange(start_line=1, end_line=2)]},
            lines_mapping={0: 0, 1: 2, 2: 4},
        )
    )


def test_parse_file_sections_nested():
    """Test nested sections with different names."""
    content = """line 1
### [outer]
outer content
### [inner]
inner content
### [/inner]
more outer
### [/outer]
end"""

    with temp_text_file(content) as temp_path:
        result = parse_file_sections(temp_path)

    assert result == snapshot(
        ParsedFile(
            lines=[
                'line 1',
                'outer content',
                'inner content',
                'more outer',
                'end',
            ],
            sections={
                'inner': [LineRange(start_line=2, end_line=3)],
                'outer': [LineRange(start_line=1, end_line=4)],
            },
            lines_mapping={0: 0, 1: 2, 2: 4, 3: 6, 4: 8},
        )
    )


def test_extract_fragment_content_entire_file():
    """Test extracting entire file when no fragments specified."""
    content = """line 1
### [section1]
content 1
### [/section1]
line 5"""

    with temp_text_file(content) as temp_path:
        parsed = parse_file_sections(temp_path)

    assert parsed.render([], []) == snapshot(
        RenderedSnippet(
            content="""\
line 1
content 1
line 5\
""",
            highlights=[],
            original_range=LineRange(start_line=0, end_line=5),
        )
    )

    assert parsed.render(['section1'], []) == snapshot(
        RenderedSnippet(
            content="""\
content 1

...\
""",
            highlights=[],
            original_range=LineRange(start_line=2, end_line=3),
        )
    )

    assert parsed.render([], ['section1']) == snapshot(
        RenderedSnippet(
            content="""\
line 1
content 1
line 5\
""",
            highlights=[LineRange(start_line=1, end_line=2)],
            original_range=LineRange(start_line=0, end_line=5),
        )
    )


def test_extract_fragment_content_specific_section():
    """Test extracting specific section."""
    content = """line 1
### [section1]
content 1
content 2
### [/section1]
line 6"""

    with temp_text_file(content) as temp_path:
        parsed = parse_file_sections(temp_path)

    assert parsed.render([], []) == snapshot(
        RenderedSnippet(
            content="""\
line 1
content 1
content 2
line 6\
""",
            highlights=[],
            original_range=LineRange(start_line=0, end_line=6),
        )
    )

    assert parsed.render(['section1'], []) == snapshot(
        RenderedSnippet(
            content="""\
content 1
content 2

...\
""",
            highlights=[],
            original_range=LineRange(start_line=2, end_line=4),
        )
    )

    assert parsed.render([], ['section1']) == snapshot(
        RenderedSnippet(
            content="""\
line 1
content 1
content 2
line 6\
""",
            highlights=[LineRange(start_line=1, end_line=3)],
            original_range=LineRange(start_line=0, end_line=6),
        )
    )


def test_extract_fragment_content_multiple_sections():
    """Test extracting multiple disjoint sections."""
    content = """line 1
### [section1]
content 1
### [/section1]
middle
### [section2]
content 2
### [/section2]
end"""

    with temp_text_file(content) as temp_path:
        parsed = parse_file_sections(temp_path)

    assert parsed.render([], []) == snapshot(
        RenderedSnippet(
            content="""\
line 1
content 1
middle
content 2
end\
""",
            highlights=[],
            original_range=LineRange(start_line=0, end_line=9),
        )
    )

    assert parsed.render(['section1', 'section2'], []) == snapshot(
        RenderedSnippet(
            content="""\
content 1

...

content 2

...\
""",
            highlights=[],
            original_range=LineRange(start_line=2, end_line=7),
        )
    )

    assert parsed.render(['section1', 'section2'], ['section1']) == snapshot(
        RenderedSnippet(
            content="""\
content 1

...

content 2

...\
""",
            highlights=[LineRange(start_line=0, end_line=1)],
            original_range=LineRange(start_line=2, end_line=7),
        )
    )

    assert parsed.render(['section1', 'section2'], ['section1', 'section2']) == snapshot(
        RenderedSnippet(
            content="""\
content 1

...

content 2

...\
""",
            highlights=[LineRange(start_line=0, end_line=1), LineRange(start_line=2, end_line=3)],
            original_range=LineRange(start_line=2, end_line=7),
        )
    )

    assert parsed.render(['section1'], ['section2']) == snapshot(
        RenderedSnippet(
            content="""\
content 1

...\
""",
            highlights=[],
            original_range=LineRange(start_line=2, end_line=3),
        )
    )

    assert parsed.render([], ['section1', 'section2']) == snapshot(
        RenderedSnippet(
            content="""\
line 1
content 1
middle
content 2
end\
""",
            highlights=[LineRange(start_line=1, end_line=2), LineRange(start_line=3, end_line=4)],
            original_range=LineRange(start_line=0, end_line=9),
        )
    )


def test_complicated_example():
    """Test extracting multiple overlapping sections."""
    content = """line 1
### [fragment1]
line 2
### [fragment2]
line 3
### [highlight1,highlight2]
line 4
### [/fragment1,/highlight1]
line 5
### [/fragment2]
line 6
### [/highlight2]
"""

    with temp_text_file(content) as temp_path:
        parsed = parse_file_sections(temp_path)

    assert parsed.render([], []) == snapshot(
        RenderedSnippet(
            content="""\
line 1
line 2
line 3
line 4
line 5
line 6\
""",
            highlights=[],
            original_range=LineRange(start_line=0, end_line=11),
        )
    )

    assert parsed.render(['fragment1'], ['highlight1']) == snapshot(
        RenderedSnippet(
            content="""\
line 2
line 3
line 4

...\
""",
            highlights=[LineRange(start_line=2, end_line=3)],
            original_range=LineRange(start_line=2, end_line=7),
        )
    )

    assert parsed.render(['fragment1'], ['highlight2']) == snapshot(
        RenderedSnippet(
            content="""\
line 2
line 3
line 4

...\
""",
            highlights=[LineRange(start_line=2, end_line=5)],
            original_range=LineRange(start_line=2, end_line=7),
        )
    )

    assert parsed.render(['fragment2'], ['highlight2']) == snapshot(
        RenderedSnippet(
            content="""\
...

line 3
line 4
line 5

...\
""",
            highlights=[LineRange(start_line=2, end_line=5)],
            original_range=LineRange(start_line=4, end_line=9),
        )
    )

    assert parsed.render(['fragment1', 'fragment2'], []) == snapshot(
        RenderedSnippet(
            content="""\
line 2
line 3
line 4
line 5

...\
""",
            highlights=[],
            original_range=LineRange(start_line=2, end_line=9),
        )
    )


def test_format_highlight_lines_empty():
    """Test formatting empty highlight ranges."""
    assert format_highlight_lines([]) == ''


def test_format_highlight_lines_single():
    """Test formatting single line highlight."""
    assert format_highlight_lines([LineRange(0, 1)]) == '1'
    assert format_highlight_lines([LineRange(5, 6)]) == '6'


def test_format_highlight_lines_range():
    """Test formatting line range highlight."""
    assert format_highlight_lines([LineRange(0, 3)]) == '1-3'
    assert format_highlight_lines([LineRange(5, 9)]) == '6-9'


def test_format_highlight_lines_multiple():
    """Test formatting multiple highlights."""
    assert format_highlight_lines([LineRange(0, 1), LineRange(2, 5), LineRange(6, 7)]) == '1 3-5 7'


def test_inject_snippets_basic():
    """Test basic snippet injection."""
    content = """def hello():
    return "world" """

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)

        # Mock the docs directory resolution by copying file
        target_file = docs_dir / 'test.py'
        target_file.write_text(content, encoding='utf-8')

        markdown = '```snippet {path="test.py"}'
        result = inject_snippets(markdown, docs_dir)

    assert result == snapshot('```snippet {path="test.py"}')


def test_inject_snippets_with_title():
    """Test snippet injection with custom title."""
    content = "print('hello')"

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        target_file = docs_dir / 'test.py'
        target_file.write_text(content, encoding='utf-8')

        markdown = '```snippet {path="test.py" title="Custom Title"}'

        result = inject_snippets(markdown, docs_dir)

    assert result == snapshot('```snippet {path="test.py" title="Custom Title"}')


def test_inject_snippets_with_fragments():
    """Test snippet injection with fragments."""
    content = """line 1
### [important]
key_function()
### [/important]
line 5"""

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        target_file = docs_dir / 'test.py'
        target_file.write_text(content, encoding='utf-8')

        markdown = '```snippet {path="test.py" fragment="important"}'

        result = inject_snippets(markdown, docs_dir)

    assert result == snapshot('```snippet {path="test.py" fragment="important"}')


def test_inject_snippets_with_highlights():
    """Test snippet injection with highlights."""
    content = """def normal():
    pass

### [important]
def important():
    return True
### [/important]

def other():
    pass"""

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        target_file = docs_dir / 'test.py'
        target_file.write_text(content, encoding='utf-8')

        markdown = '```snippet {path="test.py" highlight="important"}'

        result = inject_snippets(markdown, docs_dir)

    assert result == snapshot('```snippet {path="test.py" highlight="important"}')


def test_inject_snippets_nonexistent_file():
    """Test that nonexistent files raise an error.."""
    markdown = '```snippet {path="nonexistent.py"}```'
    with pytest.raises(FileNotFoundError):
        inject_snippets(markdown, REPO_ROOT)


def test_inject_snippets_multiple():
    """Test injecting multiple snippets in one markdown."""
    content1 = "print('file1')"
    content2 = "print('file2')"

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        file1 = docs_dir / 'test1.py'
        file2 = docs_dir / 'test2.py'
        file1.write_text(content1, encoding='utf-8')
        file2.write_text(content2, encoding='utf-8')

        markdown = """Some text
```snippet {path="test1.py"}
More text
```snippet {path="test2.py"}
Final text"""

        result = inject_snippets(markdown, docs_dir)

    assert result == snapshot(
        """\
Some text
```snippet {path="test1.py"}
More text
```snippet {path="test2.py"}
Final text\
"""
    )


def test_inject_snippets_extra_attrs():
    """Test snippet injection with extra attributes."""
    content = "print('test')"

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        target_file = docs_dir / 'test.py'
        target_file.write_text(content, encoding='utf-8')

        markdown = '```snippet {path="test.py" custom="value" another="attr"}'

        result = inject_snippets(markdown, docs_dir)

    assert result == snapshot('```snippet {path="test.py" custom="value" another="attr"}')
