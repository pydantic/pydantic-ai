"""Bounded rendering of the memory snapshot that `Memory` injects beside the user prompt."""

from __future__ import annotations

from pydantic_ai_harness.memory._store import MemoryStore
from pydantic_ai_harness.memory._toolset import MAIN_FILENAME

_CHARS_PER_TOKEN = 4
_MIN_FILENAME_LIST_CHARS = 6


async def list_subfiles(store: MemoryStore, scope: str, *, limit: int) -> tuple[list[str], bool]:
    """Return a bounded filename list and whether additional paths may exist."""
    prefix = f'{scope}/'
    names: list[str] = []
    request_limit = limit + 2  # one main notebook plus one truncation sentinel
    returned_paths = await store.list_paths(prefix, limit=request_limit)
    paths = returned_paths[:request_limit]
    for path in paths:
        if not path.startswith(prefix):
            raise RuntimeError('memory backend returned a path outside the requested scope')
        name = path.removeprefix(prefix)
        if '/' not in name and name != MAIN_FILENAME and name.endswith('.md'):
            names.append(name)
    names.sort()
    return names[:limit], len(names) > limit or len(returned_paths) >= request_limit


def injection_listing_limit(max_tokens: int) -> int:
    """Derive a finite backend listing bound from the complete prompt budget."""
    return max(1, max_tokens * _CHARS_PER_TOKEN // _MIN_FILENAME_LIST_CHARS)


def render_memory_prompt(
    main_content: str,
    subfiles: list[str],
    *,
    heading: str,
    guidance: str,
    max_lines: int,
    max_tokens: int,
    main_truncated: bool = False,
    files_truncated: bool = False,
) -> str:
    """Render a complete memory section within the strict approximate token budget."""
    budget = max_tokens * _CHARS_PER_TOKEN
    rendered_guidance = guidance
    source_lines = main_content.rstrip().splitlines()
    kept_lines = source_lines[-max_lines:] if max_lines else []
    dropped_lines = len(source_lines) - len(kept_lines)

    max_filename_candidates = min(len(subfiles), budget // 4)
    shown_files = subfiles[:max_filename_candidates]
    omitted_files = len(subfiles) - len(shown_files)

    rendered = _render_sections(
        heading,
        rendered_guidance,
        kept_lines,
        dropped_lines,
        shown_files,
        omitted_files,
        main_truncated,
        files_truncated,
    )
    while len(rendered) > budget and shown_files:
        shown_files.pop()
        omitted_files += 1
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    while len(rendered) > budget and kept_lines:
        kept_lines.pop(0)
        dropped_lines += 1
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) > budget and dropped_lines:
        dropped_lines = 0
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) > budget and rendered_guidance:
        rendered_guidance = rendered_guidance[: max(0, len(rendered_guidance) - (len(rendered) - budget))]
        rendered = _render_sections(
            heading,
            rendered_guidance,
            kept_lines,
            dropped_lines,
            shown_files,
            omitted_files,
            main_truncated,
            files_truncated,
        )
    if len(rendered) <= budget:
        return rendered
    if omitted_files or files_truncated:
        marker = '[Memory files omitted; use search_memory]'
        return marker[:budget]
    return rendered[:budget]


def _render_sections(
    heading: str,
    guidance: str,
    main_lines: list[str],
    dropped_lines: int,
    subfiles: list[str],
    omitted_files: int,
    main_truncated: bool,
    files_truncated: bool,
) -> str:
    sections = [f'## {heading}'] if heading else []
    if guidance:
        sections.append(guidance)
    if main_lines or dropped_lines or main_truncated:
        lines = list(main_lines)
        if dropped_lines:
            lines.insert(
                0,
                f'... [{dropped_lines} earlier lines; use read_memory("{MAIN_FILENAME}") for the full notebook] ...',
            )
        if main_truncated:
            lines.append('... [notebook exceeds `max_memory_size`; bounded prefix shown] ...')
        sections.append(f'### {MAIN_FILENAME}\n\n' + '\n'.join(lines))
    if subfiles or omitted_files or files_truncated:
        listing = [f'- {name}' for name in subfiles]
        if files_truncated:
            listing.append('- ... [more files omitted; use search_memory to find relevant memory]')
        elif omitted_files:
            listing.append(f'- ... [{omitted_files} more files; use search_memory to find relevant memory]')
        sections.append('### Other memory files\n\n' + '\n'.join(listing))
    return '\n\n'.join(sections)
