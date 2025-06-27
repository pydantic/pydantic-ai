from __future__ import annotations as _annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from inline_snapshot import snapshot

REPO_ROOT = Path(__file__).parent.parent.parent
PYDANTIC_AI_EXAMPLES_ROOT = REPO_ROOT / 'examples' / 'pydantic_ai_examples'


@dataclass
class SnippetDirective:
    path: str
    title: str | None = None
    fragment: str | None = None
    highlight: str | None = None
    extra_attrs: dict[str, str] | None = None


@dataclass
class LineRange:
    start_line: int  # first line in file is 0
    end_line: int  # unlike start_line, this line is interpreted as excluded from the range; this should always be larger than the start_line

    def intersection(self, ranges: list[LineRange]) -> list[LineRange]:
        new_ranges: list[LineRange] = []
        for r in ranges:
            new_start_line = max(r.start_line, self.start_line)
            new_end_line = min(r.end_line, self.end_line)
            if new_start_line < new_end_line:
                new_ranges.append(r)
        return new_ranges

    @staticmethod
    def merge(ranges: list[LineRange]) -> list[LineRange]:
        if not ranges:
            return []

        # Sort ranges by start_line
        sorted_ranges = sorted(ranges, key=lambda r: r.start_line)
        merged: list[LineRange] = []

        for current in sorted_ranges:
            if not merged or merged[-1].end_line < current.start_line:
                # No overlap with previous range, add as new range
                merged.append(current)
            else:
                # Overlap or adjacent, merge with previous range
                merged[-1] = LineRange(merged[-1].start_line, max(merged[-1].end_line, current.end_line))

        return merged


@dataclass
class RenderedSnippet:
    content: str
    highlights: list[LineRange]
    original_range: LineRange


@dataclass
class ParsedFile:
    lines: list[str]
    sections: dict[str, list[LineRange]]
    lines_mapping: dict[int, int]

    def render(self, fragment_sections: list[str], highlight_sections: list[str]) -> RenderedSnippet:
        fragment_ranges: list[LineRange] = []
        if fragment_sections:
            for k in fragment_sections:
                if k not in self.sections:
                    raise ValueError(f'Unrecognized fragment section: {k!r} (expected {list(self.sections)})')
                fragment_ranges.extend(self.sections[k])
            fragment_ranges = LineRange.merge(fragment_ranges)
        else:
            fragment_ranges = [LineRange(0, len(self.lines))]

        highlight_ranges: list[LineRange] = []
        for k in highlight_sections:
            if k not in self.sections:
                raise ValueError(f'Unrecognized highlight section: {k!r} (expected {list(self.sections)})')
            highlight_ranges.extend(self.sections[k])
        highlight_ranges = LineRange.merge(highlight_ranges)

        rendered_highlight_ranges = list[LineRange]()
        rendered_lines: list[str] = []
        last_end_line = 1
        current_line = 0
        for fragment_range in fragment_ranges:
            if fragment_range.start_line > last_end_line:
                if current_line == 0:
                    rendered_lines.append('...\n')
                else:
                    rendered_lines.append('\n...\n')

                current_line += 1
            fragment_highlight_ranges = fragment_range.intersection(highlight_ranges)
            for fragment_highlight_range in fragment_highlight_ranges:
                rendered_highlight_ranges.append(
                    LineRange(
                        fragment_highlight_range.start_line - fragment_range.start_line + current_line,
                        fragment_highlight_range.end_line - fragment_range.start_line + current_line,
                    )
                )

            for i in range(fragment_range.start_line, fragment_range.end_line):
                rendered_lines.append(self.lines[i])
                current_line += 1
            last_end_line = fragment_range.end_line

        if last_end_line < len(self.lines):
            rendered_lines.append('\n...')

        original_range = LineRange(
            self.lines_mapping[fragment_ranges[0].start_line],
            self.lines_mapping[fragment_ranges[-1].end_line - 1] + 1,
        )
        return RenderedSnippet('\n'.join(rendered_lines), LineRange.merge(rendered_highlight_ranges), original_range)


def parse_snippet_directive(line: str) -> SnippetDirective | None:
    """Parse a line like: ```snippet {path="..." title="..." fragment="..." highlight="..."}```"""
    pattern = r'```snippet\s+\{([^}]+)\}'
    match = re.match(pattern, line.strip())
    if not match:
        return None

    attrs_str = match.group(1)
    attrs: dict[str, str] = {}

    # Parse key="value" pairs
    for attr_match in re.finditer(r'(\w+)="([^"]*)"', attrs_str):
        key, value = attr_match.groups()
        attrs[key] = value

    if 'path' not in attrs:
        raise ValueError('Missing required key "path" in snippet directive')

    extra_attrs = {k: v for k, v in attrs.items() if k not in ['path', 'title', 'fragment', 'highlight']}

    return SnippetDirective(
        path=attrs['path'],
        title=attrs.get('title'),
        fragment=attrs.get('fragment'),
        highlight=attrs.get('highlight'),
        extra_attrs=extra_attrs if extra_attrs else None,
    )


def parse_file_sections(file_path: Path) -> ParsedFile:
    """Parse a file and extract sections marked with ### [section] or /// [section]"""
    input_lines = file_path.read_text().splitlines()
    output_lines: list[str] = []
    lines_mapping: dict[int, int] = {}

    sections: dict[str, list[LineRange]] = {}
    section_starts: dict[str, int] = {}

    output_line_no = 0
    for line_no, line in enumerate(input_lines, 1):
        match: re.Match[str] | None = None
        for match in re.finditer(r'\s*(?:###|///)\s*\[([^]]+)]\s*$', line):
            break
        else:
            output_lines.append(line)
            output_line_no += 1
            lines_mapping[output_line_no - 1] = line_no - 1
            continue

        pre_matches_line = line[: match.start()]
        sections_to_start: set[str] = set()
        sections_to_end: set[str] = set()
        for item in match.group(1).split(','):
            if item in sections_to_end or item in sections_to_start:
                raise ValueError(f'Duplicate section reference: {item!r} at {file_path}:{line_no}')
            if item.startswith('/'):
                sections_to_end.add(item[1:])
            else:
                sections_to_start.add(item)

        for section_name in sections_to_start:
            if section_name in section_starts:
                raise ValueError(f'Cannot nest section with the same name {section_name!r} at {file_path}:{line_no}')
            section_starts[section_name] = output_line_no

        for section_name in sections_to_end:
            start_line = section_starts.pop(section_name, None)
            if start_line is None:
                raise ValueError(f'Cannot end unstarted section {section_name!r} at {file_path}:{line_no}')
            if section_name not in sections:
                sections[section_name] = []
            end_line = output_line_no + 1 if pre_matches_line else output_line_no
            sections[section_name].append(LineRange(start_line, end_line))

        if pre_matches_line:
            output_lines.append(pre_matches_line)
            output_line_no += 1
            lines_mapping[output_line_no - 1] = line_no - 1

    if section_starts:
        raise ValueError(f'Some sections were not finished in {file_path}: {list(section_starts)}')

    return ParsedFile(lines=output_lines, sections=sections, lines_mapping=lines_mapping)


def format_highlight_lines(highlight_ranges: list[LineRange]) -> str:
    """Convert highlight ranges to mkdocs hl_lines format"""
    if not highlight_ranges:
        return ''

    parts: list[str] = []
    for range in highlight_ranges:
        start = range.start_line + 1  # convert to 1-based indexing
        end = range.end_line  # SectionRanges exclude the end, so just don't add 1 here
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f'{start}-{end}')

    return ' '.join(parts)


def inject_snippets(markdown: str, relative_path_root: Path) -> str:  # noqa C901
    def replace_snippet(match: re.Match[str]) -> str:
        line = match.group(0)
        directive = parse_snippet_directive(line)
        if not directive:
            return line

        if directive.path.startswith('/'):
            # If directive path is absolute, treat it as relative to the repo root:
            file_path = (REPO_ROOT / directive.path[1:]).resolve()
        else:
            # Else, resolve as a relative path
            file_path = (relative_path_root / directive.path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f'File {file_path} not found')

        # Parse the file sections
        parsed_file = parse_file_sections(file_path)

        # Determine fragments to extract
        fragment_names = directive.fragment.split() if directive.fragment else []
        highlight_names = directive.highlight.split() if directive.highlight else []

        # Extract content
        rendered = parsed_file.render(fragment_names, highlight_names)

        # Get file extension for syntax highlighting
        file_extension = file_path.suffix.lstrip('.')

        # Determine title
        if directive.title:
            title = directive.title
        else:
            if file_path.is_relative_to(PYDANTIC_AI_EXAMPLES_ROOT):
                title_path = str(file_path.relative_to(PYDANTIC_AI_EXAMPLES_ROOT))
            else:
                title_path = file_path.name
            title = title_path
            range_spec: str | None = None
            if directive.fragment:
                range_spec = f'L{rendered.original_range.start_line + 1}-L{rendered.original_range.end_line}'
                title = f'{title_path} ({range_spec})'
            if file_path.is_relative_to(REPO_ROOT):
                relative_path = file_path.relative_to(REPO_ROOT)
                url = f'https://github.com/pydantic/pydantic-ai/blob/main/{relative_path}'
                if range_spec is not None:
                    url += f'#{range_spec}'
                title = f"<a href='{url}' target='_blank' rel='noopener noreferrer'>{title}</a>"
        # Build attributes for the code block
        attrs: list[str] = []
        if title:
            attrs.append(f'title="{title}"')

        # Add highlight lines
        if rendered.highlights:
            hl_lines = format_highlight_lines(rendered.highlights)
            if hl_lines:
                attrs.append(f'hl_lines="{hl_lines}"')

        # Add extra attributes
        if directive.extra_attrs:
            for key, value in directive.extra_attrs.items():
                attrs.append(f'{key}="{value}"')

        # Build the replacement
        attrs_str = ' '.join(attrs)
        if attrs_str:
            attrs_str = ' {' + attrs_str + '}'

        result = f'```{file_extension}{attrs_str}\n{rendered.content}\n```'

        return result

    # Find and replace all snippet directives
    pattern = r'^```snippet\s+\{[^}]+\}```$'
    return re.sub(pattern, replace_snippet, markdown, flags=re.MULTILINE)


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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            result = parse_file_sections(Path(f.name))
            assert result == snapshot(
                ParsedFile(
                    lines=['line 1', 'content 1', 'content 2', 'line 6'],
                    sections={'section1': [LineRange(start_line=1, end_line=3)]},
                    lines_mapping={0: 0, 1: 2, 2: 3, 3: 5},
                )
            )
        finally:
            os.unlink(f.name)


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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            result = parse_file_sections(Path(f.name))
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
        finally:
            os.unlink(f.name)


def test_parse_file_sections_comment_style():
    """Test parsing with /// comment style."""
    content = """line 1
/// [section1]
content 1
/// [/section1]
line 5"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            result = parse_file_sections(Path(f.name))
            assert result == snapshot(
                ParsedFile(
                    lines=['line 1', 'content 1', 'line 5'],
                    sections={'section1': [LineRange(start_line=1, end_line=2)]},
                    lines_mapping={0: 0, 1: 2, 2: 4},
                )
            )
        finally:
            os.unlink(f.name)


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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            result = parse_file_sections(Path(f.name))
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
        finally:
            os.unlink(f.name)


def test_extract_fragment_content_entire_file():
    """Test extracting entire file when no fragments specified."""
    content = """line 1
### [section1]
content 1
### [/section1]
line 5"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            parsed = parse_file_sections(Path(f.name))
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
        finally:
            os.unlink(f.name)


def test_extract_fragment_content_specific_section():
    """Test extracting specific section."""
    content = """line 1
### [section1]
content 1
content 2
### [/section1]
line 6"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            parsed = parse_file_sections(Path(f.name))
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
        finally:
            os.unlink(f.name)


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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            parsed = parse_file_sections(Path(f.name))
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
        finally:
            os.unlink(f.name)


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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

        try:
            parsed = parse_file_sections(Path(f.name))
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
        finally:
            os.unlink(f.name)


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

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(content)
        f.flush()

    try:
        # Create a temporary docs directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)

            # Mock the docs directory resolution by copying file
            target_file = docs_dir / 'test.py'
            target_file.write_text(content)

            markdown = '```snippet {path="test.py"}'
            result = inject_snippets(markdown, docs_dir)
        assert result == snapshot('```snippet {path="test.py"}')

    finally:
        os.unlink(f.name)


def test_inject_snippets_with_title():
    """Test snippet injection with custom title."""
    content = "print('hello')"

    with tempfile.TemporaryDirectory() as temp_dir:
        docs_dir = Path(temp_dir)
        target_file = docs_dir / 'test.py'
        target_file.write_text(content)

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
        target_file.write_text(content)

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
        target_file.write_text(content)

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
        file1.write_text(content1)
        file2.write_text(content2)

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
        target_file.write_text(content)

        markdown = '```snippet {path="test.py" custom="value" another="attr"}'

        result = inject_snippets(markdown, docs_dir)
    assert result == snapshot('```snippet {path="test.py" custom="value" another="attr"}')
