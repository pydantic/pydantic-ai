"""Keep the README honest about what ships.

Every capability package must document itself with a `README.md` and be linked
from the top-level `README.md`. A capability cannot land without showing up in
the docs, so the "what's available today" tables cannot silently fall behind the
code. This is the mechanical half of docs parity; the semantic half (does the
prose match the code as written) is a review-time concern, not a unit test.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
_PACKAGE = _ROOT / 'pydantic_ai_harness'

# The `experimental` package is a namespace/warning shim, not a capability, so it
# has no standalone README and is not listed in the top-level tables.
_NAMESPACE_PACKAGES = {_PACKAGE / 'experimental'}


def _is_deprecation_shim(package: Path) -> bool:
    """A package left at a moved or renamed capability's old path re-exports it and warns.

    Such shims carry no docs of their own, so they are excluded from the capability tables.
    """
    source = (package / '__init__.py').read_text(encoding='utf-8')
    return 'warn_moved(' in source or 'warn_module_renamed(' in source


# Packages that are supporting infrastructure rather than capabilities, kept out of the README
# capability tables per review. `media` exports the content-addressed stores Step Persistence
# uses; its docs placement is being reworked in
# https://github.com/pydantic/pydantic-ai-harness/issues/625.
_NOT_A_CAPABILITY = frozenset({'media'})


def _capability_packages() -> list[Path]:
    """Directories that are importable packages and represent a capability's public surface."""
    candidates: list[Path] = []
    for parent in (_PACKAGE, _PACKAGE / 'experimental'):
        for child in sorted(parent.iterdir()):
            if not child.is_dir() or child.name.startswith(('_', '.')):
                continue
            # A non-package dir under the capability roots does not occur in a clean tree, so this guard stays uncovered.
            if not (child / '__init__.py').exists():  # pragma: no cover
                continue
            if child.name in _NOT_A_CAPABILITY or child in _NAMESPACE_PACKAGES or _is_deprecation_shim(child):
                continue
            candidates.append(child)
    return candidates


_CAPABILITY_PACKAGES = _capability_packages()


def test_capability_packages_discovered() -> None:
    # Guard against the discovery silently finding nothing (e.g. a moved package root),
    # which would make the parametrized checks below vacuously pass.
    assert len(_CAPABILITY_PACKAGES) >= 10


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_has_readme(package: Path) -> None:
    readme = package / 'README.md'
    assert readme.exists(), (
        f'{package.relative_to(_ROOT)} is an importable capability package but has no README.md. '
        'Add one (start from an existing capability README) so its public surface is documented.'
    )


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_linked_from_top_readme(package: Path) -> None:
    top_readme = (_ROOT / 'README.md').read_text(encoding='utf-8')
    link_target = f'{package.relative_to(_ROOT).as_posix()}/'
    # Require an actual Markdown link to the package, not just the path appearing
    # anywhere (prose or an unrelated URL would otherwise satisfy the check).
    linked = any(t.startswith(link_target) for t in _markdown_link_targets(top_readme))
    assert linked, (
        f'{package.relative_to(_ROOT)} is not linked from the top-level README.md. '
        f'Add a row for it (linking `{link_target}`) to the "What\'s available today" or "Roadmap" tables '
        'so the README stays in step with the code.'
    )


# --- Unified-docs page checks (docs/*.md) -----------------------------------
#
# The flat pages under `docs/` render on the unified site. These mechanical
# checks encode the capability-authoring rules agreed in the 2026-07-10 team
# sync: purpose-first leads, a source link on every page, names that match the
# capability, and no leftover "experimental" framing on graduated capabilities.
# ACP is the one page that stays experimental.

_DOCS_DIR = _ROOT / 'docs'
_NON_CAPABILITY_PAGES = {'examples.md', 'index.md', 'mutation-testing.md'}
_ACP_PAGE = 'acp.md'

_SOURCE_LINK = 'github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/'
# Framing that must not appear on a graduated (non-ACP) capability page.
_EXPERIMENTAL_MARKERS = ('HarnessExperimentalWarning', 'removed in any release', '!!! warning "Experimental')
# Lifecycle hook names must not lead a page -- mechanism goes below the purpose.
_LEAD_HOOK_NAMES = ('before_model_request', 'after_model_request', 'before_tool_execute', 'after_tool_execute')
# ClassName-style headings are a smell, except where the class name IS the name.
_ALLOWED_CLASSNAME_HEADINGS = {'FileSystem', 'LocalStack', 'StackOne'}
_FORBIDDEN_HEADINGS = {'overflow', 'authoring', 'overflow capability', 'compaction capabilities'}


def _capability_doc_pages() -> list[Path]:
    return [p for p in sorted(_DOCS_DIR.glob('*.md')) if p.name not in _NON_CAPABILITY_PAGES]


_CAPABILITY_DOC_PAGES = _capability_doc_pages()

# Each capability doc page maps to its source module (for the source-link check)
# and its expected H1 (for the heading check). Keeping this explicit makes the
# checks page-specific: a page that links the wrong module, or carries a generic
# or empty heading, fails instead of passing on a substring match.
_CAPABILITY_PAGE_META = {
    'advisor.md': ('advisor', 'Advisor'),
    'code-mode.md': ('code_mode', 'Code Mode'),
    'coder.md': ('coder', 'Coder'),
    'skills.md': ('skills', 'Skills'),
    'filesystem.md': ('filesystem', 'FileSystem'),
    'shell.md': ('shell', 'Shell'),
    'managed-prompt.md': ('logfire', 'Managed Prompt'),
    'memory.md': ('memory', 'Memory'),
    'modal-sandbox.md': ('modal_sandbox', 'Modal Sandbox'),
    'repo-context.md': ('repo_context', 'Repo Context'),
    'researcher.md': ('researcher', 'Researcher'),
    'pydantic-ai-docs.md': ('pydantic_ai_docs', 'Pydantic AI Docs'),
    'exa-search.md': ('exa', 'Exa Search'),
    'youdotcom.md': ('youdotcom', 'You.com'),
    'macroscope.md': ('macroscope', 'Macroscope'),
    'browser-use.md': ('browser_use', 'Browser Use'),
    'compaction.md': ('compaction', 'Compaction'),
    'tool-output-limits.md': ('tool_output_limits', 'Tool Output Limits'),
    'warn-on-cache-busts.md': ('warn_on_cache_busts', 'Warn On Cache Busts'),
    'step-persistence.md': ('step_persistence', 'Step Persistence'),
    'conversation-search.md': ('conversation_search', 'Conversation Search'),
    'media.md': ('media', 'Media Externalization'),
    'subagents.md': ('subagents', 'Subagents'),
    'dynamic-workflow.md': ('dynamic_workflow', 'Dynamic Workflow'),
    'planning.md': ('planning', 'Planning'),
    'system-reminders.md': ('system_reminders', 'System Reminders'),
    'capability-creation.md': ('capability_creation', 'Runtime Capability Creation'),
    'guardrails.md': ('guardrails', 'Input, Output & Tool Guardrails'),
    'prompt-injection-defender.md': ('prompt_injection_defender', 'Prompt Injection Defender'),
    'spend.md': ('spend', 'Spend'),
    'localstack.md': ('localstack', 'LocalStack'),
    'stackone.md': ('stackone', 'StackOne'),
    'acp.md': ('experimental/acp', 'ACP (Agent Client Protocol)'),
}


def _markdown_link_targets(text: str) -> list[str]:
    """Every `](target)` destination in the text -- so a bare path mention is not a link."""
    return re.findall(r'\]\(([^)\s]+)\)', text)


def _strip_frontmatter(text: str) -> str:
    if text.startswith('---\n'):
        end = text.find('\n---', 4)
        if end != -1:
            return text[end + 4 :]
    return text


def _heading_problem(h1: str) -> str | None:
    """Return why an H1 fails the name rule, or None if it is fine."""
    name = h1[2:].strip() if h1.startswith('# ') else h1.strip()
    if not name:
        return 'missing H1 heading'
    if name.lower() in _FORBIDDEN_HEADINGS:
        return f'"{name}" is a short/legacy form -- use the full capability name'
    for word in name.split():
        if word in _ALLOWED_CLASSNAME_HEADINGS:
            continue
        if re.search(r'[a-z][A-Z]', word):
            return f'"{name}" is ClassName-style ("{word}") -- use spaced words'
    return None


def _h1(text: str) -> str:
    for line in _strip_frontmatter(text).splitlines():
        if line.startswith('# '):
            return line
    return ''


def _lead_paragraph(text: str) -> str:
    """The first prose paragraph after the H1, skipping links, notes, and admonitions."""
    lines = _strip_frontmatter(text).splitlines()
    start = next((i + 1 for i, ln in enumerate(lines) if ln.startswith('# ')), 0)
    collected: list[str] = []
    in_fence = False
    for line in lines[start:]:
        stripped = line.strip()
        if in_fence:
            if stripped.startswith('```'):
                in_fence = False
            continue
        if not collected:
            if not stripped:
                continue
            if stripped.startswith('```'):
                in_fence = True
                continue
            # Skip non-prose preamble: headings, blockquotes, admonitions,
            # links/images, tables, and indented admonition bodies.
            if stripped.startswith(('#', '>', '!!!', '[', '![', '|')) or line.startswith('    '):
                continue
            collected.append(stripped)
        elif not stripped:
            break
        else:
            collected.append(stripped)
    return ' '.join(collected)


def test_strip_frontmatter_handles_missing_close() -> None:
    assert _strip_frontmatter('---\ntitle: x\n---\nbody') == '\nbody'
    # Opened but never closed -- returned unchanged rather than swallowing the file.
    assert _strip_frontmatter('---\nnot closed') == '---\nnot closed'
    assert _strip_frontmatter('no frontmatter') == 'no frontmatter'


def test_h1_missing_returns_empty() -> None:
    assert _h1('# Title\nbody') == '# Title'
    assert _h1('just prose, no heading') == ''


def test_heading_problem_flags_each_failure_mode() -> None:
    assert _heading_problem('') == 'missing H1 heading'
    assert _heading_problem('# Overflow') is not None  # forbidden short form
    assert _heading_problem('# SubAgents') is not None  # ClassName-style
    assert _heading_problem('# FileSystem') is None  # allowlisted ClassName
    assert _heading_problem('# Code Mode') is None


def test_lead_paragraph_skips_preamble_and_fences() -> None:
    doc = (
        '---\ntitle: x\n---\n'
        '# Title\n\n'
        '```python\ncode\n```\n\n'
        '> a note\n'
        '[Source](x)\n'
        '    indented body\n\n'
        'The real lead.\n\n'
        'second paragraph\n'
    )
    # Fenced block, blockquote, link, and indented lines are skipped; collection
    # stops at the blank line after the first prose paragraph.
    assert _lead_paragraph(doc) == 'The real lead.'
    # A lead that runs to EOF exercises multi-line collection and loop exhaustion.
    assert _lead_paragraph('# T\n\nline one\nline two') == 'line one line two'


def test_capability_doc_pages_discovered() -> None:
    # Guard against a moved docs root making every check below vacuously pass.
    assert len(_CAPABILITY_DOC_PAGES) >= 12
    # A new capability page must declare its module + expected heading here, so
    # the source-link and heading checks below stay page-specific.
    unmapped = sorted(p.name for p in _CAPABILITY_DOC_PAGES if p.name not in _CAPABILITY_PAGE_META)
    assert not unmapped, f'add {unmapped} to _CAPABILITY_PAGE_META (source module + expected heading)'


@pytest.mark.parametrize('page', _CAPABILITY_DOC_PAGES, ids=lambda p: p.name)
def test_doc_page_links_its_source(page: Path) -> None:
    module, _ = _CAPABILITY_PAGE_META[page.name]
    expected = f'{_SOURCE_LINK}{module}/'
    targets = _markdown_link_targets(page.read_text(encoding='utf-8'))
    assert any(expected in t for t in targets), (
        f'{page.relative_to(_ROOT)} must link its own source module as a Markdown link '
        f'(target containing `{expected}`), not just mention the prefix or link a different module.'
    )


@pytest.mark.parametrize('page', _CAPABILITY_DOC_PAGES, ids=lambda p: p.name)
def test_doc_page_heading_matches_capability(page: Path) -> None:
    _, expected_name = _CAPABILITY_PAGE_META[page.name]
    h1 = _h1(page.read_text(encoding='utf-8'))
    assert h1, f'{page.relative_to(_ROOT)} has no H1 heading.'
    actual = h1[2:].strip()
    assert actual == expected_name, (
        f'{page.relative_to(_ROOT)} H1 is "{actual}"; expected "{expected_name}" '
        '(doc filename, H1, and capability name must agree).'
    )


@pytest.mark.parametrize('page', _CAPABILITY_DOC_PAGES, ids=lambda p: p.name)
def test_doc_page_lead_is_purpose_first(page: Path) -> None:
    lead = _lead_paragraph(page.read_text(encoding='utf-8'))
    hit = next((h for h in _LEAD_HOOK_NAMES if h in lead), None)
    assert hit is None, (
        f'{page.relative_to(_ROOT)}: opening paragraph names the `{hit}` hook. '
        'Lead with the purpose (what it is for, when to use it); move mechanism lower.'
    )


@pytest.mark.parametrize(
    'page',
    [p for p in _CAPABILITY_DOC_PAGES if p.name != _ACP_PAGE],
    ids=lambda p: p.name,
)
def test_graduated_doc_page_has_no_experimental_framing(page: Path) -> None:
    text = page.read_text(encoding='utf-8')
    hit = next((m for m in _EXPERIMENTAL_MARKERS if m in text), None)
    assert hit is None, (
        f'{page.relative_to(_ROOT)}: graduated capability still carries experimental framing ({hit!r}). '
        'Only ACP keeps an experimental note; soften the rest to the README stability note.'
    )


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_readme_heading_matches_capability(package: Path) -> None:
    problem = _heading_problem(_h1((package / 'README.md').read_text(encoding='utf-8')))
    assert problem is None, f'{package.relative_to(_ROOT) / "README.md"}: {problem}'


@pytest.mark.parametrize('package', _CAPABILITY_PACKAGES, ids=lambda p: str(p.relative_to(_ROOT)))
def test_capability_readme_links_source(package: Path) -> None:
    module = package.relative_to(_ROOT / 'pydantic_ai_harness').as_posix()
    expected = f'{_SOURCE_LINK}{module}/'
    targets = _markdown_link_targets((package / 'README.md').read_text(encoding='utf-8'))
    assert any(expected in t for t in targets), (
        f'{package.relative_to(_ROOT) / "README.md"} must link its own source module '
        f'(a Markdown link containing `{expected}`), so parity tooling can find the implementation.'
    )


# The Coder blown-out example is repeated on four markdown surfaces (plus, parameterized,
# in examples/coding_agent.py). They must stay byte-identical so no page drifts from what
# `coder_agent` actually is; see agent_docs/docs-conventions.md.
_BLOWN_OUT_MARKER = '<!-- Keep this blown-out example in sync across'
_BLOWN_OUT_SURFACES = (
    'docs/coder.md',
    'docs/index.md',
    'README.md',
    'pydantic_ai_harness/coder/README.md',
)


def _blown_out_block(path: Path) -> str:
    text = path.read_text(encoding='utf-8')
    assert _BLOWN_OUT_MARKER in text, f'{path.relative_to(_ROOT)} lost its blown-out keep-in-sync marker'
    after = text.split(_BLOWN_OUT_MARKER, 1)[1]
    match = re.search(r'```python\n(.*?)```', after, flags=re.DOTALL)
    assert match, f'{path.relative_to(_ROOT)} has no python block after the blown-out marker'
    return match.group(1)


@pytest.mark.parametrize('surface', _BLOWN_OUT_SURFACES[1:])
def test_blown_out_example_is_identical_across_surfaces(surface: str) -> None:
    canonical = _blown_out_block(_ROOT / _BLOWN_OUT_SURFACES[0])
    assert _blown_out_block(_ROOT / surface) == canonical, (
        f'{surface} blown-out example differs from {_BLOWN_OUT_SURFACES[0]}; '
        'update all surfaces named in the keep-in-sync comment together.'
    )


def test_blown_out_example_matches_coder_defaults() -> None:
    from pydantic_ai_harness.coder import DEFAULT_ALLOWED_COMMANDS

    block = _blown_out_block(_ROOT / _BLOWN_OUT_SURFACES[0])
    listed = re.findall(r"'([a-z]+)'", block.split('allowed_commands = [', 1)[1].split(']', 1)[0])
    assert tuple(listed) == tuple(DEFAULT_ALLOWED_COMMANDS), (
        'the written-out allowlist no longer matches pydantic_ai_harness.coder.DEFAULT_ALLOWED_COMMANDS'
    )
    agent_source = (_PACKAGE / 'coder' / '_agent.py').read_text(encoding='utf-8')
    identity_match = re.search(r"instructions='([^']+)'", agent_source)
    assert identity_match, 'coder/_agent.py no longer defines an identity instruction'
    identity = f"instructions='{identity_match.group(1)}'"
    assert identity in block, "the blown-out example must carry coder_agent's identity instruction verbatim"
    example = (_ROOT / 'examples/coding_agent.py').read_text(encoding='utf-8')
    assert identity in example and 'denied_env_patterns=LLM_API_KEY_ENV_PATTERNS' in example, (
        'examples/coding_agent.py drifted from the coder_agent composition'
    )
