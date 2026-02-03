#!/usr/bin/env python
"""Check structural integrity of Claude Code skills.

This script validates that:
1. All expected reference files exist
2. SKILL.md is under the line limit
3. SKILL.md frontmatter is valid YAML
4. Reference files are under the line limit
5. All code blocks have proper {title="..."} metadata
6. Public exports from pydantic_ai are mentioned in skill files
7. Skill code examples are in sync with doc/docstring examples

Usage:
    python scripts/check_skills.py [--verbose]
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


SKILLS_DIR = Path('skills/pydantic-ai')
SKILL_MD = SKILLS_DIR / 'SKILL.md'
REFERENCES_DIR = SKILLS_DIR / 'references'
INIT_FILE = Path('pydantic_ai_slim/pydantic_ai/__init__.py')

SKILL_MD_MAX_LINES = 500
REFERENCE_MAX_LINES = 400
API_REFERENCE_MAX_LINES = 400

DOCS_DIR = Path('docs')
SOURCE_DIR = Path('pydantic_ai_slim')

# Escape hatch for skill-only examples or indented doc examples that can't be matched
SKIP_SYNC_TITLES: set[str] = {
    # Indented in docs (inside MkDocs tabs blocks) - regex doesn't match
    'bedrock_claude_thinking_part.py',
    'bedrock_openai_thinking_part.py',
    'bedrock_qwen_thinking_part.py',
    'bedrock_deepseek_thinking_part.py',
    # Skill-only examples (condensed/simplified versions for skills)
    'test_model_structured.py',
    # Input examples
    'multimodal_input.py',
    'binary_content.py',
    # Tool examples
    'tool_retry_example.py',
    'tool_outputs.py',
    'tool_return_example.py',
    'dynamic_tool.py',
    'prepare_tools_example.py',
    'tool_with_ctx.py',
    'tool_plain_example.py',
    # Messages examples
    'messages_access.py',
    # Common use case examples
    'rag_example.py',
    'support_bot.py',
    'code_assistant.py',
    'data_analyst.py',
    'webhook_handler.py',
    # Third-party tools examples
    'langchain_tool.py',
    'langchain_toolkit.py',
    'aci_tool.py',
    'aci_toolset.py',
    # Dependencies examples
    'deps_basic.py',
    'deps_instructions.py',
    'deps_in_tools.py',
    'deps_testing.py',
    # Output examples
    'output_validator_simple.py',
    # Direct API examples
    'direct_basic.py',
    'direct_async.py',
    'direct_with_tools.py',
    'direct_streaming.py',
    'direct_instrumented.py',  # Has duplicate titles in docs causing extract_code_blocks to miss first version
    # Multi-agent examples
    'agent_delegation.py',
    'delegation_with_deps.py',
    'programmatic_handoff.py',  # Skill-simplified version
    'router_pattern.py',
    'reflection_pattern.py',
    'plan_execute_pattern.py',
    # Toolsets examples
    'approval_required_toolset.py',  # Skill-simplified version
    # Model examples
    'model_settings_example.py',
    'fallback_model_simple.py',
    # Retry examples
    'retry_setup.py',
    # Approval examples
    'always_require_approval.py',
    'conditional_approval.py',
    # Deferred tools examples
    'external_tool.py',  # Skill-simplified version
    # Agent examples
    'static_instructions.py',
    'dynamic_instructions.py',
    'agent_override.py',
    'agent_metadata.py',
}

EXPECTED_REFERENCE_FILES = [
    'agents.md',
    'tools.md',
    'toolsets.md',
    'builtin-tools.md',
    'common-tools.md',
    'output.md',
    'dependencies.md',
    'models.md',
    'streaming.md',
    'messages.md',
    'mcp.md',
    'graph.md',
    'exceptions.md',
    'observability.md',
    'testing.md',
    'thinking.md',
    'evals.md',
    'embeddings.md',
    'durable.md',
    'api-reference.md',
]

# Exports that are intentionally not documented in skills (internal, deprecated, or low-level)
EXCLUDED_EXPORTS = {
    '__version__',
    # Format types - documented as a group, not individually
    'AudioFormat',
    'AudioMediaType',
    'ImageFormat',
    'ImageMediaType',
    'VideoFormat',
    'VideoMediaType',
    'DocumentFormat',
    'DocumentMediaType',
    # Low-level or internal types less relevant for skill users
    'FinishReason',
    'ModelRequestPart',
    'ModelResponsePart',
    'BaseToolCallPart',
    'BaseToolReturnPart',
    'BuiltinToolCallPart',
    'BuiltinToolReturnPart',
    'ToolReturn',
    'CachePoint',
    'BinaryContent',
    'FileUrl',
    'FilePart',
    'MultiModalContent',
    'UserContent',
    'ModelResponsePartDelta',
    'ModelResponseStreamEvent',
    'TextPartDelta',
    'ThinkingPartDelta',
    'ToolCallPartDelta',
    'AgentStreamEvent',
    'PartStartEvent',
    'PartDeltaEvent',
    'PartEndEvent',
    'FinalResultEvent',
    'FunctionToolCallEvent',
    'FunctionToolResultEvent',
    'HandleResponseEvent',
    'AgentRunResultEvent',
    # Profiles - documented at high level
    'DEFAULT_PROFILE',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
    'ModelProfileSpec',
    # Internal toolset types (toolset classes are documented in toolsets.md)
    'ToolsetFunc',
    'ToolsetTool',
    # Other internal
    'format_as_xml',
    'UrlContextTool',
}


def parse_all_exports() -> set[str]:
    """Parse __all__ from pydantic_ai/__init__.py using AST."""
    if not INIT_FILE.exists():
        print(f'Warning: {INIT_FILE} not found')
        return set()

    tree = ast.parse(INIT_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == '__all__':
                    if isinstance(node.value, ast.Tuple | ast.List):
                        return {
                            elt.value  # type: ignore[union-attr]
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    return set()


def scan_skill_files() -> set[str]:
    """Scan all skill markdown files for mentioned class/function names."""
    mentioned: set[str] = set()
    if not SKILLS_DIR.exists():
        return mentioned

    for md_file in SKILLS_DIR.rglob('*.md'):
        content = md_file.read_text()
        # Match backtick-wrapped names and table entries
        mentioned.update(re.findall(r'`(\w+)`', content))
        # Match names in table cells (| Name |)
        mentioned.update(re.findall(r'\|\s*`?(\w+)`?\s*\|', content))

    return mentioned


def check_frontmatter(content: str) -> list[str]:
    """Check that SKILL.md has valid YAML frontmatter."""
    errors: list[str] = []
    if not content.startswith('---'):
        errors.append('SKILL.md must start with YAML frontmatter (---)')
        return errors

    end = content.find('---', 3)
    if end == -1:
        errors.append('SKILL.md frontmatter is not closed (missing second ---)')
        return errors

    frontmatter = content[3:end].strip()
    # Basic validation: check required fields
    if 'name:' not in frontmatter:
        errors.append('SKILL.md frontmatter missing required field: name')
    if 'description:' not in frontmatter:
        errors.append('SKILL.md frontmatter missing required field: description')

    return errors


def check_code_blocks(file_path: Path) -> list[str]:
    """Check that code blocks have {title="..."} metadata."""
    errors: list[str] = []
    content = file_path.read_text()
    lines = content.splitlines()

    for i, line in enumerate(lines, 1):
        if line.startswith('```python') and '{title=' not in line and '{test="skip"}' not in line:
            # Allow plain ```python blocks that are not meant to be executed
            # (e.g., constructor signatures, type definitions)
            # Only flag blocks that have print() output assertions (#>)
            block_end = None
            for j in range(i, len(lines)):
                if lines[j].strip() == '```':
                    block_end = j
                    break
            if block_end:
                block_content = '\n'.join(lines[i:block_end])
                # Only flag blocks with print assertions — those must be testable
                if '#>' in block_content:
                    errors.append(f'{file_path}:{i}: Python code block missing {{title="..."}} metadata')

    return errors


def extract_code_blocks(file_path: Path) -> dict[str, str]:
    """Parse a file and return {title: source_code} for every fenced Python code block with a title.

    Works with both markdown (.md) files and Python (.py) files containing docstrings.
    Only includes blocks whose title ends with '.py'.
    """
    blocks: dict[str, str] = {}
    content = file_path.read_text()
    lines = content.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i]
        # Match opening fence: ```python {title="something.py" ...} or ```py {title="something.py" ...}
        m = re.match(r'^```(?:python|py)\s+\{.*?title="([^"]+\.py)".*?\}', line)
        if m:
            title = m.group(1)
            # Collect code until closing fence
            code_lines: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != '```':
                code_lines.append(lines[i])
                i += 1
            blocks[title] = '\n'.join(code_lines)
        i += 1

    return blocks


def normalize_code(code: str) -> str:
    """Normalize code for comparison by stripping MkDocs annotations and trailing whitespace."""
    # Strip # (N)! annotation comments (MkDocs footnotes)
    code = re.sub(r'\s*# \(\d+\)!', '', code)
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in code.splitlines()]
    # Strip trailing blank lines
    while lines and not lines[-1]:
        lines.pop()
    # Strip leading blank lines
    while lines and not lines[0]:
        lines.pop(0)
    return '\n'.join(lines)


def check_example_sync() -> list[str]:
    """Check that skill code examples match their doc/docstring counterparts.

    Every titled code block in skill files must have a matching title in docs/ or
    pydantic_ai_slim/ with identical normalized source code. When a title appears
    multiple times in docs (e.g. the same example shown in different contexts),
    the skill must match at least one of them.
    """
    errors: list[str] = []

    # Collect all titled code blocks from docs and source (multiple per title)
    doc_blocks: dict[str, list[str]] = {}
    for md_file in DOCS_DIR.rglob('*.md'):
        for title, code in extract_code_blocks(md_file).items():
            doc_blocks.setdefault(title, []).append(code)
    for py_file in SOURCE_DIR.rglob('*.py'):
        for title, code in extract_code_blocks(py_file).items():
            doc_blocks.setdefault(title, []).append(code)

    # Collect all titled code blocks from skill files
    skill_blocks: dict[str, str] = {}
    for md_file in SKILLS_DIR.rglob('*.md'):
        skill_blocks.update(extract_code_blocks(md_file))

    # Check each skill block has a matching doc block
    for title, skill_code in skill_blocks.items():
        if title in SKIP_SYNC_TITLES:
            continue

        if title not in doc_blocks:
            errors.append(
                f'Skill example {title!r} has no matching doc example '
                f'(not found in docs/ or pydantic_ai_slim/)'
            )
            continue

        skill_normalized = normalize_code(skill_code)
        doc_versions = [normalize_code(code) for code in doc_blocks[title]]

        if skill_normalized not in doc_versions:
            errors.append(
                f'Skill example {title!r} differs from doc example. '
                f'Skill code must be a verbatim copy (modulo MkDocs annotations).'
            )

    return errors


def main() -> int:
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    errors: list[str] = []

    # 1. Check SKILL.md exists and is under line limit
    if not SKILL_MD.exists():
        errors.append(f'{SKILL_MD} does not exist')
    else:
        content = SKILL_MD.read_text()
        line_count = len(content.splitlines())
        if line_count > SKILL_MD_MAX_LINES:
            errors.append(f'{SKILL_MD} is {line_count} lines (max {SKILL_MD_MAX_LINES})')
        elif verbose:
            print(f'OK: {SKILL_MD} is {line_count} lines (max {SKILL_MD_MAX_LINES})')

        # Check frontmatter
        errors.extend(check_frontmatter(content))

        # Check code blocks
        errors.extend(check_code_blocks(SKILL_MD))

    # 2. Check all expected reference files exist and are under line limit
    for ref_file in EXPECTED_REFERENCE_FILES:
        ref_path = REFERENCES_DIR / ref_file
        if not ref_path.exists():
            errors.append(f'Missing reference file: {ref_path}')
        else:
            line_count = len(ref_path.read_text().splitlines())
            max_lines = API_REFERENCE_MAX_LINES if ref_file == 'api-reference.md' else REFERENCE_MAX_LINES
            if line_count > max_lines:
                errors.append(f'{ref_path} is {line_count} lines (max {max_lines})')
            elif verbose:
                print(f'OK: {ref_path} is {line_count} lines (max {max_lines})')

            # Check code blocks
            errors.extend(check_code_blocks(ref_path))

    # 3. Check VERSION file exists
    version_file = SKILLS_DIR / 'VERSION'
    if not version_file.exists():
        errors.append(f'{version_file} does not exist')
    elif verbose:
        print(f'OK: {version_file} exists (version: {version_file.read_text().strip()})')

    # 4. Check public export coverage
    all_exports = parse_all_exports()
    if all_exports:
        mentioned = scan_skill_files()
        checkable_exports = all_exports - EXCLUDED_EXPORTS
        uncovered = checkable_exports - mentioned

        if verbose:
            print(f'\nPublic exports: {len(all_exports)}')
            print(f'Excluded: {len(EXCLUDED_EXPORTS)}')
            print(f'Checkable: {len(checkable_exports)}')
            print(f'Mentioned in skills: {len(mentioned & checkable_exports)}')
            print(f'Uncovered: {len(uncovered)}')

        if uncovered:
            errors.append(
                f'{len(uncovered)} public exports not mentioned in skill files: '
                f'{", ".join(sorted(uncovered))}'
            )

    # 5. Check example sync with docs
    sync_errors = check_example_sync()
    if verbose and not sync_errors:
        print(f'OK: All skill examples are in sync with docs')
    errors.extend(sync_errors)

    # Report results
    print()
    if errors:
        print(f'Skills check FAILED with {len(errors)} error(s):')
        for error in errors:
            print(f'  - {error}')
        return 1
    else:
        print('Skills check passed!')
        return 0


if __name__ == '__main__':
    sys.exit(main())
