"""Tests for scripts/check_skills.py — structural validation of Claude Code skills."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

from scripts.check_skills import (
    EXPECTED_REFERENCE_FILES,
    check_code_blocks,
    check_example_sync,
    check_frontmatter,
    extract_code_blocks,
    main,
    normalize_code,
    parse_all_exports,
    scan_skill_files,
)


class TestCheckFrontmatter:
    def test_valid_frontmatter(self) -> None:
        content = textwrap.dedent("""\
            ---
            name: pydantic-ai
            description: A skill
            ---
            # Content
        """)
        assert check_frontmatter(content) == []

    def test_missing_frontmatter(self) -> None:
        content = '# No frontmatter'
        errors = check_frontmatter(content)
        assert len(errors) == 1
        assert 'must start with YAML frontmatter' in errors[0]

    def test_unclosed_frontmatter(self) -> None:
        content = '---\nname: test\n'
        errors = check_frontmatter(content)
        assert len(errors) == 1
        assert 'not closed' in errors[0]

    def test_missing_name_field(self) -> None:
        content = '---\ndescription: A skill\n---\n'
        errors = check_frontmatter(content)
        assert len(errors) == 1
        assert 'name' in errors[0]

    def test_missing_description_field(self) -> None:
        content = '---\nname: test\n---\n'
        errors = check_frontmatter(content)
        assert len(errors) == 1
        assert 'description' in errors[0]


class TestCheckCodeBlocks:
    def test_code_block_with_title(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python {title="test.py"}\nimport os\nresult = os.getcwd()\nprint(result)\n```\n')
        assert check_code_blocks(md_file) == []

    def test_non_executable_block_allowed(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python\nctx.deps  # access deps\n```\n')
        assert check_code_blocks(md_file) == []

    def test_block_with_assertions_without_title(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python\nimport os\nresult = os.getcwd()\nprint(result)\n#> /some/path\n```\n')
        errors = check_code_blocks(md_file)
        assert len(errors) == 1
        assert 'missing {title="..."}' in errors[0]

    def test_block_without_assertions_allowed(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python\nimport os\nresult = os.getcwd()\nprint(result)\n```\n')
        assert check_code_blocks(md_file) == []


class TestParseAllExports:
    def test_parse_real_init(self) -> None:
        exports = parse_all_exports()
        # Should find a non-trivial number of exports
        assert len(exports) > 50
        # Should include key types
        assert 'Agent' in exports
        assert 'RunContext' in exports
        assert 'ModelRetry' in exports

    def test_missing_init_file(self) -> None:
        with patch('scripts.check_skills.INIT_FILE', Path('nonexistent/__init__.py')):
            exports = parse_all_exports()
            assert exports == set()


class TestScanSkillFiles:
    def test_scan_real_skills(self) -> None:
        mentioned = scan_skill_files()
        # Should find key types mentioned in skill files
        assert 'Agent' in mentioned
        assert 'RunContext' in mentioned

    def test_empty_directory(self) -> None:
        with patch('scripts.check_skills.SKILLS_DIR', Path('nonexistent')):
            mentioned = scan_skill_files()
            assert mentioned == set()


class TestExpectedReferenceFiles:
    def test_all_reference_files_exist(self) -> None:
        """Verify that all expected reference files actually exist on disk."""
        references_dir = Path('skills/pydantic-ai/references')
        for ref_file in EXPECTED_REFERENCE_FILES:
            ref_path = references_dir / ref_file
            assert ref_path.exists(), f'Missing reference file: {ref_path}'


class TestMain:
    def test_main_passes(self) -> None:
        """The main check should pass on the current codebase."""
        result = main()
        assert result == 0, 'check_skills.py main() should return 0 (pass)'

    def test_main_verbose(self) -> None:
        """Verbose mode should also pass."""
        with patch('sys.argv', ['check_skills.py', '--verbose']):
            result = main()
            assert result == 0

    def test_main_fails_missing_skill_md(self, tmp_path: Path) -> None:
        """Should fail if SKILL.md doesn't exist."""
        with patch('scripts.check_skills.SKILL_MD', tmp_path / 'nonexistent' / 'SKILL.md'):
            with patch('scripts.check_skills.REFERENCES_DIR', tmp_path / 'nonexistent' / 'references'):
                with patch('scripts.check_skills.SKILLS_DIR', tmp_path / 'nonexistent'):
                    result = main()
                    assert result == 1

    def test_main_fails_skill_md_too_long(self, tmp_path: Path) -> None:
        """Should fail if SKILL.md exceeds line limit."""
        skill_md = tmp_path / 'SKILL.md'
        skill_md.write_text('---\nname: test\ndescription: test\n---\n' + ('line\n' * 600))
        with (
            patch('scripts.check_skills.SKILL_MD', skill_md),
            patch('scripts.check_skills.REFERENCES_DIR', tmp_path / 'references'),
            patch('scripts.check_skills.SKILLS_DIR', tmp_path),
        ):
            result = main()
            assert result == 1


class TestExtractCodeBlocks:
    def test_extracts_titled_block(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('# Example\n\n```python {title="example.py"}\nprint("hello")\n```\n')
        blocks = extract_code_blocks(md_file)
        assert blocks == {'example.py': 'print("hello")'}

    def test_ignores_untitled_blocks(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python\nprint("hello")\n```\n')
        blocks = extract_code_blocks(md_file)
        assert blocks == {}

    def test_multiple_blocks(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python {title="a.py"}\ncode_a\n```\n\n```python {title="b.py"}\ncode_b\n```\n')
        blocks = extract_code_blocks(md_file)
        assert blocks == {'a.py': 'code_a', 'b.py': 'code_b'}

    def test_py_fence_variant(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```py {title="example.py"}\ncode\n```\n')
        blocks = extract_code_blocks(md_file)
        assert blocks == {'example.py': 'code'}

    def test_ignores_non_py_titles(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python {title="Terminal"}\nnot python\n```\n')
        blocks = extract_code_blocks(md_file)
        assert blocks == {}

    def test_preserves_multiline_content(self, tmp_path: Path) -> None:
        md_file = tmp_path / 'test.md'
        md_file.write_text('```python {title="multi.py"}\nimport os\n\nprint(os.getcwd())\n```\n')
        blocks = extract_code_blocks(md_file)
        assert blocks == {'multi.py': 'import os\n\nprint(os.getcwd())'}


class TestNormalizeCode:
    def test_strips_mkdocs_annotations(self) -> None:
        code = 'x = 1  # (1)!\ny = 2  # (2)!'
        assert normalize_code(code) == 'x = 1\ny = 2'

    def test_strips_trailing_whitespace(self) -> None:
        code = 'x = 1   \ny = 2  '
        assert normalize_code(code) == 'x = 1\ny = 2'

    def test_strips_trailing_blank_lines(self) -> None:
        code = 'x = 1\n\n\n'
        assert normalize_code(code) == 'x = 1'

    def test_strips_leading_blank_lines(self) -> None:
        code = '\n\nx = 1'
        assert normalize_code(code) == 'x = 1'

    def test_identical_code_matches(self) -> None:
        code = 'from pydantic_ai import Agent\n\nagent = Agent("openai:gpt-5")'
        assert normalize_code(code) == normalize_code(code)

    def test_annotation_difference_ignored(self) -> None:
        doc_code = 'agent = Agent("openai:gpt-5")  # (1)!\nresult = agent.run_sync("hi")  # (2)!'
        skill_code = 'agent = Agent("openai:gpt-5")\nresult = agent.run_sync("hi")'
        assert normalize_code(doc_code) == normalize_code(skill_code)


class TestCheckExampleSync:
    def test_matching_blocks_pass(self, tmp_path: Path) -> None:
        """Matching code blocks should produce no errors."""
        docs_dir = tmp_path / 'docs'
        docs_dir.mkdir()
        doc_file = docs_dir / 'agents.md'
        doc_file.write_text(
            '```python {title="example.py"}\nfrom pydantic_ai import Agent\nagent = Agent("openai:gpt-5")\n```\n'
        )
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        skill_file = skills_dir / 'SKILL.md'
        skill_file.write_text(
            '```python {title="example.py"}\nfrom pydantic_ai import Agent\nagent = Agent("openai:gpt-5")\n```\n'
        )
        with (
            patch('scripts.check_skills.DOCS_DIR', docs_dir),
            patch('scripts.check_skills.SOURCE_DIR', tmp_path / 'empty'),
            patch('scripts.check_skills.SKILLS_DIR', skills_dir),
        ):
            (tmp_path / 'empty').mkdir()
            errors = check_example_sync()
            assert errors == []

    def test_mismatched_blocks_produce_errors(self, tmp_path: Path) -> None:
        """Different code should produce an error."""
        docs_dir = tmp_path / 'docs'
        docs_dir.mkdir()
        doc_file = docs_dir / 'agents.md'
        doc_file.write_text(
            '```python {title="example.py"}\nfrom pydantic_ai import Agent\nagent = Agent("openai:gpt-5")\n```\n'
        )
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        skill_file = skills_dir / 'SKILL.md'
        skill_file.write_text(
            '```python {title="example.py"}\nfrom pydantic_ai import Agent\nagent = Agent("openai:gpt-4o")\n```\n'
        )
        with (
            patch('scripts.check_skills.DOCS_DIR', docs_dir),
            patch('scripts.check_skills.SOURCE_DIR', tmp_path / 'empty'),
            patch('scripts.check_skills.SKILLS_DIR', skills_dir),
        ):
            (tmp_path / 'empty').mkdir()
            errors = check_example_sync()
            assert len(errors) == 1
            assert 'differs from doc example' in errors[0]

    def test_annotations_stripped_for_comparison(self, tmp_path: Path) -> None:
        """MkDocs annotations in docs should be stripped before comparison."""
        docs_dir = tmp_path / 'docs'
        docs_dir.mkdir()
        doc_file = docs_dir / 'example.md'
        doc_file.write_text('```python {title="annotated.py"}\nx = 1  # (1)!\ny = 2  # (2)!\n```\n')
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        skill_file = skills_dir / 'SKILL.md'
        skill_file.write_text('```python {title="annotated.py"}\nx = 1\ny = 2\n```\n')
        with (
            patch('scripts.check_skills.DOCS_DIR', docs_dir),
            patch('scripts.check_skills.SOURCE_DIR', tmp_path / 'empty'),
            patch('scripts.check_skills.SKILLS_DIR', skills_dir),
        ):
            (tmp_path / 'empty').mkdir()
            errors = check_example_sync()
            assert errors == []

    def test_missing_doc_counterpart_produces_error(self, tmp_path: Path) -> None:
        """Skill example with no doc counterpart should produce an error."""
        docs_dir = tmp_path / 'docs'
        docs_dir.mkdir()
        (docs_dir / 'empty.md').write_text('# Empty\n')
        skills_dir = tmp_path / 'skills'
        skills_dir.mkdir()
        skill_file = skills_dir / 'SKILL.md'
        skill_file.write_text('```python {title="orphan.py"}\nprint("no doc match")\n```\n')
        with (
            patch('scripts.check_skills.DOCS_DIR', docs_dir),
            patch('scripts.check_skills.SOURCE_DIR', tmp_path / 'empty'),
            patch('scripts.check_skills.SKILLS_DIR', skills_dir),
        ):
            (tmp_path / 'empty').mkdir()
            errors = check_example_sync()
            assert len(errors) == 1
            assert 'no matching doc example' in errors[0]

    def test_real_codebase_sync(self) -> None:
        """All skill examples in the real codebase should be in sync with docs."""
        errors = check_example_sync()
        assert errors == [], 'Skill examples out of sync:\n' + '\n'.join(f'  - {e}' for e in errors)
