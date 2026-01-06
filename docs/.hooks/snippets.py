from __future__ import annotations as _annotations

import re
from dataclasses import dataclass
from pathlib import Path

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
    input_lines = file_path.read_text(encoding='utf-8').splitlines()
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


def inject_snippets(markdown: str, relative_path_root: Path) -> str:  # noqa: C901
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
