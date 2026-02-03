"""Tests for scripts/check_skills.py — structural validation of Claude Code skills."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

from scripts.check_skills import main, validate_frontmatter


class TestValidateFrontmatter:
    def test_valid_frontmatter(self) -> None:
        content = textwrap.dedent("""\
            ---
            name: pydantic-ai
            description: A skill
            ---
            # Content
        """)
        assert validate_frontmatter(content) == []

    def test_missing_frontmatter(self) -> None:
        content = '# No frontmatter'
        errors = validate_frontmatter(content)
        assert len(errors) == 1
        assert 'must start with YAML frontmatter' in errors[0]

    def test_unclosed_frontmatter(self) -> None:
        content = '---\nname: test\n'
        errors = validate_frontmatter(content)
        assert len(errors) == 1
        assert 'not closed' in errors[0]

    def test_missing_name_field(self) -> None:
        content = '---\ndescription: A skill\n---\n'
        errors = validate_frontmatter(content)
        assert any('name' in e for e in errors)

    def test_missing_description_field(self) -> None:
        content = '---\nname: test\n---\n'
        errors = validate_frontmatter(content)
        assert any('description' in e for e in errors)

    def test_invalid_yaml(self) -> None:
        content = '---\nname: [unclosed\n---\n'
        errors = validate_frontmatter(content)
        assert any('Invalid YAML' in e for e in errors)

    def test_unexpected_keys(self) -> None:
        content = '---\nname: test\ndescription: A skill\nunknown: value\n---\n'
        errors = validate_frontmatter(content)
        assert any('Unexpected' in e for e in errors)


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
            with patch('scripts.check_skills.SKILLS_DIR', tmp_path / 'nonexistent'):
                result = main()
                assert result == 1

    def test_main_fails_skill_md_too_long(self, tmp_path: Path) -> None:
        """Should fail if SKILL.md exceeds line limit."""
        skill_md = tmp_path / 'SKILL.md'
        skill_md.write_text('---\nname: test\ndescription: test\n---\n' + ('line\n' * 600))
        with (
            patch('scripts.check_skills.SKILL_MD', skill_md),
            patch('scripts.check_skills.SKILLS_DIR', tmp_path),
        ):
            result = main()
            assert result == 1
