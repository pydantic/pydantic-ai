"""Tests for skills toolset."""

import asyncio
import os
from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai._run_context import RunContext as InternalRunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.skills import (
    Skill,
    SkillNotFoundError,
    SkillResource,
    SkillScript,
    SkillsDirectory,
    SkillValidationError,
)
from pydantic_ai.skills._directory import (
    _discover_skills,  # pyright: ignore[reportPrivateUsage]
    _parse_skill_md,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.skills._local import (
    create_file_based_resource,
)
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import SkillsToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


# ==================== Fixtures ====================


@pytest.fixture
def sample_skills_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample skills."""
    # Create skill 1
    skill1_dir = tmp_path / 'skill-one'
    skill1_dir.mkdir()
    (skill1_dir / 'SKILL.md').write_text("""---
name: skill-one
description: First test skill for basic operations
---

# Skill One

Use this skill for basic operations.

## Instructions

1. Do something simple
2. Return results
""")

    # Create skill 2 with resources
    skill2_dir = tmp_path / 'skill-two'
    skill2_dir.mkdir()
    (skill2_dir / 'SKILL.md').write_text("""---
name: skill-two
description: Second test skill with resources
---

# Skill Two

Advanced skill with resources.

See FORMS.md for details.
""")
    (skill2_dir / 'FORMS.md').write_text('# Forms\n\nForm filling guide.')
    (skill2_dir / 'REFERENCE.md').write_text('# API Reference\n\nDetailed reference.')

    # Create skill 3 with scripts
    skill3_dir = tmp_path / 'skill-three'
    skill3_dir.mkdir()
    (skill3_dir / 'SKILL.md').write_text("""---
name: skill-three
description: Third test skill with executable scripts
---

# Skill Three

Skill with executable scripts.
""")

    scripts_dir = skill3_dir / 'scripts'
    scripts_dir.mkdir()
    (scripts_dir / 'hello.py').write_text("""#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='World')
args = parser.parse_args()
print(f"Hello, {args.name}!")
""")
    (scripts_dir / 'echo.py').write_text("""#!/usr/bin/env python3
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--message', type=str, required=True)
args = parser.parse_args()
print(args.message)
""")

    return tmp_path


# ==================== Type Tests ====================


def test_skill_resource_creation() -> None:
    """Test creating SkillResource."""
    resource = SkillResource(name='FORMS.md', uri='/tmp/skill/FORMS.md')

    assert resource.name == 'FORMS.md'
    assert resource.uri == '/tmp/skill/FORMS.md'
    assert resource.content is None


def test_skill_script_creation() -> None:
    """Test creating SkillScript."""
    script = SkillScript(name='test_script.py', uri='/tmp/skill/scripts/test_script.py', skill_name='test-skill')

    assert script.name == 'test_script.py'
    assert script.uri == '/tmp/skill/scripts/test_script.py'
    assert script.skill_name == 'test-skill'


def test_skill_creation() -> None:
    """Test creating a complete Skill."""
    resource = SkillResource(name='FORMS.md', uri='/tmp/skill/FORMS.md')
    script = SkillScript(name='test_script.py', uri='/tmp/skill/scripts/test_script.py', skill_name='test-skill')

    skill = Skill(
        name='test-skill',
        description='A test skill',
        content='# Instructions\n\nTest instructions.',
        uri='/tmp/skill',
        resources=[resource],
        scripts=[script],
    )

    assert skill.name == 'test-skill'
    assert skill.uri == '/tmp/skill'
    assert skill.description == 'A test skill'
    assert skill.content == '# Instructions\n\nTest instructions.'
    assert len(skill.resources) == 1
    assert len(skill.scripts) == 1


def test_skill_metadata() -> None:
    """Test Skill metadata field."""
    skill = Skill(
        name='test-skill',
        description='A test skill',
        content='# Instructions\n\nTest instructions.',
        metadata={'version': '1.0.0', 'author': 'Test Author'},
    )

    assert skill.metadata == {'version': '1.0.0', 'author': 'Test Author'}


# ==================== Parsing Tests ====================


def test_parse_skill_md_with_frontmatter() -> None:
    """Test parsing SKILL.md with valid frontmatter."""
    content = """---
name: test-skill
description: A test skill for testing
version: 1.0.0
---

# Test Skill

This is the main content.
"""

    frontmatter, instructions = _parse_skill_md(content)

    assert frontmatter['name'] == 'test-skill'
    assert frontmatter['description'] == 'A test skill for testing'
    assert frontmatter['version'] == '1.0.0'
    assert instructions.startswith('# Test Skill')


def test_parse_skill_md_without_frontmatter() -> None:
    """Test parsing SKILL.md without frontmatter."""
    content = """# Test Skill

This skill has no frontmatter.
"""

    frontmatter, instructions = _parse_skill_md(content)

    assert frontmatter == {}
    assert instructions.startswith('# Test Skill')


def test_parse_skill_md_empty_frontmatter() -> None:
    """Test parsing SKILL.md with empty frontmatter."""
    content = """---
---

# Test Skill

Content here.
"""

    frontmatter, instructions = _parse_skill_md(content)

    assert frontmatter == {}
    assert instructions.startswith('# Test Skill')


def test_parse_skill_md_invalid_yaml() -> None:
    """Test parsing SKILL.md with invalid YAML."""
    content = """---
name: test-skill
description: [unclosed array
---

Content.
"""

    with pytest.raises(SkillValidationError, match='Failed to parse YAML frontmatter'):
        _parse_skill_md(content)


def test_parse_skill_md_multiline_description() -> None:
    """Test parsing SKILL.md with multiline description."""
    content = """---
name: test-skill
description: |
  This is a multiline
  description for testing
---

# Content
"""

    frontmatter, _ = _parse_skill_md(content)

    assert 'multiline' in frontmatter['description']
    assert 'description for testing' in frontmatter['description']


def test_parse_skill_md_complex_frontmatter() -> None:
    """Test parsing SKILL.md with complex frontmatter."""
    content = """---
name: complex-skill
description: Complex skill with metadata
version: 2.0.0
author: Test Author
tags:
  - testing
  - example
metadata:
  category: test
  priority: high
---

# Complex Skill
"""

    frontmatter, _ = _parse_skill_md(content)

    assert frontmatter['name'] == 'complex-skill'
    assert frontmatter['tags'] == ['testing', 'example']
    assert frontmatter['metadata']['category'] == 'test'


# ==================== Discovery Tests ====================


def test_discover_skills_single_skill(tmp_path: Path) -> None:
    """Test discovering a single skill."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    skill_md = skill_dir / 'SKILL.md'
    skill_md.write_text("""---
name: test-skill
description: A test skill
---

# Test Skill

Instructions here.
""")

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].name == 'test-skill'
    assert skills[0].description == 'A test skill'
    assert 'Instructions here' in skills[0].content


def test_discover_skills_multiple_skills(tmp_path: Path) -> None:
    """Test discovering multiple skills."""
    # Create first skill
    skill1_dir = tmp_path / 'skill-one'
    skill1_dir.mkdir()
    (skill1_dir / 'SKILL.md').write_text("""---
name: skill-one
description: First skill
---

Content 1.
""")

    # Create second skill
    skill2_dir = tmp_path / 'skill-two'
    skill2_dir.mkdir()
    (skill2_dir / 'SKILL.md').write_text("""---
name: skill-two
description: Second skill
---

Content 2.
""")

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 2
    skill_names = {s.name for s in skills}
    assert skill_names == {'skill-one', 'skill-two'}


def test_discover_skills_with_resources(tmp_path: Path) -> None:
    """Test discovering skills with resource files."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with resources
---

See FORMS.md for details.
""")

    (skill_dir / 'FORMS.md').write_text('# Forms\n\nForm documentation.')
    (skill_dir / 'REFERENCE.md').write_text('# Reference\n\nAPI reference.')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 2
    resource_names = {r.name for r in skills[0].resources}
    # Resource names should be relative paths
    assert resource_names == {'FORMS.md', 'REFERENCE.md'}


def test_discover_skills_with_scripts(tmp_path: Path) -> None:
    """Test discovering skills with scripts."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with scripts
---

Use the search script.
""")

    scripts_dir = skill_dir / 'scripts'
    scripts_dir.mkdir()
    (scripts_dir / 'search.py').write_text('#!/usr/bin/env python3\nprint("searching")')
    (scripts_dir / 'process.py').write_text('#!/usr/bin/env python3\nprint("processing")')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].scripts is not None and len(skills[0].scripts) == 2
    script_names = {s.name for s in skills[0].scripts}
    # Script names should be relative paths from skill folder
    assert script_names == {'scripts/search.py', 'scripts/process.py'}


def test_discover_skills_nested_directories(tmp_path: Path) -> None:
    """Test discovering skills in nested directories."""
    nested_dir = tmp_path / 'category' / 'subcategory' / 'test-skill'
    nested_dir.mkdir(parents=True)

    (nested_dir / 'SKILL.md').write_text("""---
name: nested-skill
description: Nested skill
---

Content.
""")

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].name == 'nested-skill'


def test_discover_skills_missing_name_with_validation(tmp_path: Path) -> None:
    """Test discovering skill missing name field with validation enabled."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
description: Missing name field
---

Content.
""")

    # With validation, should skip this skill (log warning)
    with pytest.warns(UserWarning, match='missing required "name" field'):
        skills = _discover_skills(tmp_path, validate=True)
    assert len(skills) == 0


def test_discover_skills_missing_name_without_validation(tmp_path: Path) -> None:
    """Test discovering skill missing name field without validation."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
description: Missing name field
---

Content.
""")

    # Without validation, uses folder name
    skills = _discover_skills(tmp_path, validate=False)
    assert len(skills) == 1
    assert skills[0].name == 'test-skill'  # Uses folder name


def test_discover_skills_nonexistent_directory(tmp_path: Path) -> None:
    """Test discovering skills from non-existent directory."""
    nonexistent = tmp_path / 'does-not-exist'

    # Should not raise, just log warning
    skills = _discover_skills(nonexistent, validate=True)
    assert len(skills) == 0


def test_discover_skills_resources_subdirectory(tmp_path: Path) -> None:
    """Test discovering resources in references/ subdirectory."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with references subdirectory
---

Content.
""")

    references_dir = skill_dir / 'references'
    references_dir.mkdir()
    (references_dir / 'schema.md').write_text('# Schema\n\nSchema documentation.')
    (references_dir / 'template.md').write_text('# Template\n\nTemplate documentation.')

    nested_dir = references_dir / 'nested'
    nested_dir.mkdir()
    (nested_dir / 'data.md').write_text('# Data\n\nData documentation.')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 3

    resource_names = {r.name for r in skills[0].resources}
    # Resource names should be relative paths from skill folder
    assert 'references/schema.md' in resource_names
    assert 'references/template.md' in resource_names
    assert 'references/nested/data.md' in resource_names


def test_discover_skills_with_json_resources(tmp_path: Path) -> None:
    """Test discovering skills with JSON resource files."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with JSON resources
---

See data.json for configuration.
""")

    (skill_dir / 'data.json').write_text('{"key": "value", "count": 42}')
    (skill_dir / 'config.json').write_text('{"setting": true}')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 2
    resource_names = {r.name for r in skills[0].resources}
    assert resource_names == {'data.json', 'config.json'}

    # Verify file extension can be extracted from name
    for resource in skills[0].resources:
        from pathlib import Path

        assert Path(resource.name).suffix == '.json'


def test_discover_skills_with_yaml_resources(tmp_path: Path) -> None:
    """Test discovering skills with YAML resource files."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with YAML resources
---

See config.yaml for settings.
""")

    (skill_dir / 'config.yaml').write_text('key: value\ncount: 42')
    (skill_dir / 'data.yml').write_text('setting: true')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 2
    resource_names = {r.name for r in skills[0].resources}
    assert resource_names == {'config.yaml', 'data.yml'}

    # Verify file extension can be extracted from name
    from pathlib import Path

    yaml_resource = next(r for r in skills[0].resources if r.name == 'config.yaml')
    yml_resource = next(r for r in skills[0].resources if r.name == 'data.yml')
    assert Path(yaml_resource.name).suffix == '.yaml'
    assert Path(yml_resource.name).suffix == '.yml'


def test_discover_skills_with_csv_resources(tmp_path: Path) -> None:
    """Test discovering skills with CSV resource files."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with CSV resources
---

See data.csv for samples.
""")

    (skill_dir / 'data.csv').write_text('name,age\nAlice,30\nBob,25')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 1
    assert skills[0].resources[0].name == 'data.csv'
    from pathlib import Path

    assert Path(skills[0].resources[0].name).suffix == '.csv'


def test_discover_skills_with_xml_resources(tmp_path: Path) -> None:
    """Test discovering skills with XML resource files."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with XML resources
---

See schema.xml for structure.
""")

    (skill_dir / 'schema.xml').write_text('<?xml version="1.0"?><root><item>test</item></root>')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 1
    assert skills[0].resources[0].name == 'schema.xml'
    from pathlib import Path

    assert Path(skills[0].resources[0].name).suffix == '.xml'


def test_discover_skills_with_txt_resources(tmp_path: Path) -> None:
    """Test discovering skills with TXT resource files."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with text resources
---

See notes.txt for additional info.
""")

    (skill_dir / 'notes.txt').write_text('These are plain text notes.\nLine 2.')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 1
    assert skills[0].resources[0].name == 'notes.txt'
    from pathlib import Path

    assert Path(skills[0].resources[0].name).suffix == '.txt'


def test_discover_skills_with_mixed_resources(tmp_path: Path) -> None:
    """Test discovering skills with multiple file types."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with mixed resource types
---

Multiple resource types available.
""")

    # Create resources of different types
    (skill_dir / 'README.md').write_text('# README')
    (skill_dir / 'data.json').write_text('{"key": "value"}')
    (skill_dir / 'config.yaml').write_text('key: value')
    (skill_dir / 'samples.csv').write_text('a,b,c')
    (skill_dir / 'schema.xml').write_text('<root/>')
    (skill_dir / 'notes.txt').write_text('notes')

    # Create nested resources
    assets_dir = skill_dir / 'assets'
    assets_dir.mkdir()
    (assets_dir / 'nested.json').write_text('{"nested": true}')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 7
    resource_names = {r.name for r in skills[0].resources}
    assert resource_names == {
        'README.md',
        'data.json',
        'config.yaml',
        'samples.csv',
        'schema.xml',
        'notes.txt',
        'assets/nested.json',
    }


async def test_load_json_resource_parsed(tmp_path: Path) -> None:
    """Test loading JSON resource returns parsed dict."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    json_file = skill_dir / 'data.json'
    json_content = '{"name": "test", "count": 42, "active": true}'
    json_file.write_text(json_content)

    resource = create_file_based_resource(
        name='data.json',
        uri=str(json_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return parsed dict, not string
    assert isinstance(result, dict)
    assert result == {'name': 'test', 'count': 42, 'active': True}


async def test_load_yaml_resource_parsed(tmp_path: Path) -> None:
    """Test loading YAML resource returns parsed dict."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    yaml_file = skill_dir / 'config.yaml'
    yaml_content = 'name: test\ncount: 42\nactive: true'
    yaml_file.write_text(yaml_content)

    resource = create_file_based_resource(
        name='config.yaml',
        uri=str(yaml_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return parsed dict, not string
    assert isinstance(result, dict)
    assert result == {'name': 'test', 'count': 42, 'active': True}


async def test_load_yml_resource_parsed(tmp_path: Path) -> None:
    """Test loading YML resource returns parsed dict."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    yml_file = skill_dir / 'config.yml'
    yml_content = 'setting: enabled'
    yml_file.write_text(yml_content)

    resource = create_file_based_resource(
        name='config.yml',
        uri=str(yml_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return parsed dict, not string
    assert isinstance(result, dict)
    assert result == {'setting': 'enabled'}


async def test_load_json_resource_invalid_fallback(tmp_path: Path) -> None:
    """Test loading invalid JSON falls back to text."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    json_file = skill_dir / 'invalid.json'
    invalid_content = '{not valid json}'
    json_file.write_text(invalid_content)

    resource = create_file_based_resource(
        name='invalid.json',
        uri=str(json_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should fall back to returning raw text
    assert isinstance(result, str)
    assert result == invalid_content


async def test_load_yaml_resource_invalid_fallback(tmp_path: Path) -> None:
    """Test loading invalid YAML falls back to text."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    yaml_file = skill_dir / 'invalid.yaml'
    invalid_content = 'key: [invalid\nnot: closed'
    yaml_file.write_text(invalid_content)

    resource = create_file_based_resource(
        name='invalid.yaml',
        uri=str(yaml_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should fall back to returning raw text
    assert isinstance(result, str)
    assert result == invalid_content


async def test_load_csv_resource_as_text(tmp_path: Path) -> None:
    """Test loading CSV resource returns raw text."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    csv_file = skill_dir / 'data.csv'
    csv_content = 'name,age\nAlice,30\nBob,25'
    csv_file.write_text(csv_content)

    resource = create_file_based_resource(
        name='data.csv',
        uri=str(csv_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return raw text
    assert isinstance(result, str)
    assert result == csv_content


async def test_load_xml_resource_as_text(tmp_path: Path) -> None:
    """Test loading XML resource returns raw text."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    xml_file = skill_dir / 'schema.xml'
    xml_content = '<?xml version="1.0"?><root><item>test</item></root>'
    xml_file.write_text(xml_content)

    resource = create_file_based_resource(
        name='schema.xml',
        uri=str(xml_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return raw text
    assert isinstance(result, str)
    assert result == xml_content


async def test_load_txt_resource_as_text(tmp_path: Path) -> None:
    """Test loading TXT resource returns raw text."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    txt_file = skill_dir / 'notes.txt'
    txt_content = 'These are notes.\nMultiple lines.'
    txt_file.write_text(txt_content)

    resource = create_file_based_resource(
        name='notes.txt',
        uri=str(txt_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return raw text
    assert isinstance(result, str)
    assert result == txt_content


async def test_load_markdown_resource_as_text(tmp_path: Path) -> None:
    """Test loading markdown resource returns raw text."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    md_file = skill_dir / 'README.md'
    md_content = '# README\n\nSome markdown content.'
    md_file.write_text(md_content)

    resource = create_file_based_resource(
        name='README.md',
        uri=str(md_file.resolve()),
    )

    result = await resource.load(ctx=None, args=None)

    # Should return raw text (markdown is not parsed)
    assert isinstance(result, str)
    assert result == md_content


# ==================== SkillsToolset Tests ====================


def test_toolset_initialization(sample_skills_dir: Path) -> None:
    """Test SkillsToolset initialization with directory path."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    assert len(toolset.skills) == 3
    assert 'skill-one' in toolset.skills
    assert 'skill-two' in toolset.skills
    assert 'skill-three' in toolset.skills


def test_toolset_default_initialization(tmp_path: Path) -> None:
    """Test SkillsToolset initialization with default ./skills directory."""
    # Create a ./skills directory in tmp_path
    skills_dir = tmp_path / 'skills'
    skills_dir.mkdir()

    # Create a test skill
    skill_dir = skills_dir / 'test-skill'
    skill_dir.mkdir()
    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Test skill
---

# Test Skill
""")

    # Change to tmp_path and create toolset
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        toolset = SkillsToolset()  # Should default to ./skills
        assert len(toolset.skills) == 1
        assert 'test-skill' in toolset.skills
    finally:
        os.chdir(original_cwd)


def test_toolset_tool_definitions(sample_skills_dir: Path) -> None:
    """Test SkillsToolset tool definitions with snapshot."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Build a run context to get tool definitions via ToolManager
    context = InternalRunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )

    # Get tool manager and prepare for run step
    async def get_tool_defs():
        tool_manager = await ToolManager(toolset).for_run_step(context)
        return tool_manager.tool_defs

    tool_defs = asyncio.run(get_tool_defs())

    # Verify tool definitions match expected structure
    assert tool_defs == snapshot(
        [
            ToolDefinition(
                name='list_skills',
                description="""\
<summary>List all available skills with their descriptions.

Only use this tool if the available skills are not in your system prompt.</summary>
<returns>
<description>Dictionary mapping skill names to brief descriptions.
Empty dictionary if no skills are available.</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {},
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='load_skill',
                description="""\
<summary>Load complete instructions and metadata for a specific skill.

Do NOT infer or guess resource names or script names - they must come from
the output of this tool.</summary>
<returns>
<type>Complete skill documentation including</type>
<description>
- Skill description and purpose
- List of available resource files (e.g., FORMS.md, REFERENCE.md)
- List of available scripts with their names
- Detailed usage instructions and examples</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'skill_name': {'description': 'Exact name of the skill.', 'type': 'string'}},
                    'required': ['skill_name'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='read_skill_resource',
                description="""\
<summary>Read a resource file from a skill or invoke a callable resource.

Do NOT guess or infer resource names, use load_skill first to get the resource names.

Resources contain supplementary documentation like form templates,
reference guides, or data schemas. They can be static content or dynamic callables.</summary>
<returns>
<description>Complete content of the requested resource.</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {
                        'resource_name': {
                            'description': """\
Exact resource filename as listed in load_skill output
(e.g., "FORMS.md", "REFERENCE.md").\
""",
                            'type': 'string',
                        },
                        'skill_name': {
                            'description': 'Exact name of the skill (from list_skills or load_skill).',
                            'type': 'string',
                        },
                        'args': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'description': """\
Named arguments as a dictionary matching the resource's parameter schema.
Keys should match parameter names from the resource's schema.\
""",
                        },
                    },
                    'required': ['skill_name', 'resource_name'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='run_skill_script',
                description="""\
<summary>Execute a script provided by a skill.

Do NOT guess or infer script names or arguments.
Use load_skill first to get the script names and usage instructions.</summary>
<returns>
<description>Script output including both stdout and stderr.</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {
                        'args': {
                            'anyOf': [{'additionalProperties': True, 'type': 'object'}, {'type': 'null'}],
                            'default': None,
                            'description': """\
Named arguments as a dictionary matching the script's parameter schema.
Keys should match parameter names from the script's schema.\
""",
                        },
                        'script_name': {
                            'description': 'Exact script name as listed in load_skill output (includes .py extension).',
                            'type': 'string',
                        },
                        'skill_name': {
                            'description': 'Exact name of the skill (from list_skills or load_skill).',
                            'type': 'string',
                        },
                    },
                    'required': ['skill_name', 'script_name'],
                    'type': 'object',
                },
            ),
        ]
    )


def test_toolset_get_skill(sample_skills_dir: Path) -> None:
    """Test getting a specific skill."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    skill = toolset.get_skill('skill-one')
    assert skill.name == 'skill-one'
    assert skill.description == 'First test skill for basic operations'


def test_toolset_get_skill_not_found(sample_skills_dir: Path) -> None:
    """Test getting a non-existent skill."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    with pytest.raises(SkillNotFoundError, match="Skill 'nonexistent' not found"):
        toolset.get_skill('nonexistent')


async def test_list_skills_tool(sample_skills_dir: Path) -> None:
    """Test the list_skills tool by checking skills were loaded."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Verify all three skills were discovered
    assert len(toolset.skills) == 3
    assert 'skill-one' in toolset.skills
    assert 'skill-two' in toolset.skills
    assert 'skill-three' in toolset.skills

    # Verify descriptions
    assert toolset.skills['skill-one'].description == 'First test skill for basic operations'
    assert toolset.skills['skill-two'].description == 'Second test skill with resources'
    assert toolset.skills['skill-three'].description == 'Third test skill with executable scripts'


async def test_load_skill_tool(sample_skills_dir: Path) -> None:
    """Test the load_skill tool."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # The tools are internal, so we test via the public methods
    # We can check that the skills were loaded correctly
    skill = toolset.get_skill('skill-one')
    assert skill is not None
    assert skill.name == 'skill-one'
    assert 'First test skill for basic operations' in skill.description
    assert 'Use this skill for basic operations' in skill.content


async def test_load_skill_not_found(sample_skills_dir: Path) -> None:
    """Test loading a non-existent skill."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test that nonexistent skill raises an error
    with pytest.raises(SkillNotFoundError):
        toolset.get_skill('nonexistent-skill')


async def test_read_skill_resource_tool(sample_skills_dir: Path) -> None:
    """Test the read_skill_resource tool."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test that skill-two has the expected resources
    skill = toolset.get_skill('skill-two')
    assert skill.resources is not None and len(skill.resources) == 2

    resource_names = [r.name for r in skill.resources]
    assert 'FORMS.md' in resource_names
    assert 'REFERENCE.md' in resource_names

    # Check that resources can be read
    for resource in skill.resources:
        assert resource.uri is not None
        resource_path = Path(resource.uri)
        assert resource_path.exists()
        assert resource_path.is_file()


async def test_read_skill_resource_not_found(sample_skills_dir: Path) -> None:
    """Test reading a non-existent resource."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test skill with no resources
    skill_one = toolset.get_skill('skill-one')
    assert skill_one.resources is None or len(skill_one.resources) == 0

    # Test skill with resources
    skill_two = toolset.get_skill('skill-two')
    assert skill_two.resources is not None
    resource_names = [r.name for r in skill_two.resources]
    assert 'NONEXISTENT.md' not in resource_names


async def test_toolset_with_json_resources(tmp_path: Path) -> None:
    """Test SkillsToolset with JSON resources."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with JSON
---

Use data.json for configuration.
""")

    json_data = {'api_key': 'test123', 'timeout': 30}
    (skill_dir / 'data.json').write_text('{"api_key": "test123", "timeout": 30}')

    toolset = SkillsToolset(directories=[tmp_path])
    skill = toolset.get_skill('test-skill')

    assert skill.resources is not None and len(skill.resources) == 1
    resource = skill.resources[0]
    assert resource.name == 'data.json'
    from pathlib import Path

    assert Path(resource.name).suffix == '.json'

    # Load and verify it's parsed as dict
    result = await resource.load(ctx=None, args=None)
    assert isinstance(result, dict)
    assert result == json_data


async def test_toolset_with_yaml_resources(tmp_path: Path) -> None:
    """Test SkillsToolset with YAML resources."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with YAML
---

Use config.yaml for settings.
""")

    (skill_dir / 'config.yaml').write_text('database:\n  host: localhost\n  port: 5432')

    toolset = SkillsToolset(directories=[tmp_path])
    skill = toolset.get_skill('test-skill')

    assert skill.resources is not None and len(skill.resources) == 1
    resource = skill.resources[0]
    assert resource.name == 'config.yaml'
    from pathlib import Path

    assert Path(resource.name).suffix == '.yaml'

    # Load and verify it's parsed as dict
    result = await resource.load(ctx=None, args=None)
    assert isinstance(result, dict)
    assert result == {'database': {'host': 'localhost', 'port': 5432}}


async def test_toolset_with_csv_resources(tmp_path: Path) -> None:
    """Test SkillsToolset with CSV resources."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with CSV
---

Use data.csv for samples.
""")

    csv_content = 'name,age,city\nAlice,30,NYC\nBob,25,SF'
    (skill_dir / 'data.csv').write_text(csv_content)

    toolset = SkillsToolset(directories=[tmp_path])
    skill = toolset.get_skill('test-skill')

    assert skill.resources is not None and len(skill.resources) == 1
    resource = skill.resources[0]
    assert resource.name == 'data.csv'
    from pathlib import Path

    assert Path(resource.name).suffix == '.csv'

    # Load and verify it's returned as text
    result = await resource.load(ctx=None, args=None)
    assert isinstance(result, str)
    assert result == csv_content


async def test_toolset_with_multiple_file_types(tmp_path: Path) -> None:
    """Test SkillsToolset with multiple resource file types."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Multi-format skill
---

Multiple resource formats available.
""")

    # Create resources of different types
    (skill_dir / 'README.md').write_text('# Documentation')
    (skill_dir / 'config.json').write_text('{"key": "value"}')
    (skill_dir / 'settings.yaml').write_text('option: enabled')
    (skill_dir / 'data.csv').write_text('a,b\n1,2')
    (skill_dir / 'schema.xml').write_text('<schema/>')
    (skill_dir / 'notes.txt').write_text('Notes here')

    toolset = SkillsToolset(directories=[tmp_path])
    skill = toolset.get_skill('test-skill')

    assert skill.resources is not None and len(skill.resources) == 6

    # Check all resources are discovered with correct extensions
    from pathlib import Path

    resource_map = {r.name: r for r in skill.resources}
    assert Path(resource_map['README.md'].name).suffix == '.md'
    assert Path(resource_map['config.json'].name).suffix == '.json'
    assert Path(resource_map['settings.yaml'].name).suffix == '.yaml'
    assert Path(resource_map['data.csv'].name).suffix == '.csv'
    assert Path(resource_map['schema.xml'].name).suffix == '.xml'
    assert Path(resource_map['notes.txt'].name).suffix == '.txt'

    # Test loading and parsing behavior
    json_result = await resource_map['config.json'].load(ctx=None, args=None)
    assert isinstance(json_result, dict)

    yaml_result = await resource_map['settings.yaml'].load(ctx=None, args=None)
    assert isinstance(yaml_result, dict)

    md_result = await resource_map['README.md'].load(ctx=None, args=None)
    assert isinstance(md_result, str)

    csv_result = await resource_map['data.csv'].load(ctx=None, args=None)
    assert isinstance(csv_result, str)


async def test_run_skill_script_tool(sample_skills_dir: Path) -> None:
    """Test the run_skill_script tool."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test that skill-three has scripts
    skill = toolset.get_skill('skill-three')
    assert skill.scripts is not None and len(skill.scripts) == 2

    script_names = [s.name for s in skill.scripts]
    # Script names should be relative paths from skill folder
    assert 'scripts/hello.py' in script_names
    assert 'scripts/echo.py' in script_names

    # Check that scripts can be found
    for script in skill.scripts:
        assert script.uri is not None
        script_path = Path(script.uri)
        assert script_path.exists()
        assert script_path.is_file()
        assert script_path.suffix == '.py'


async def test_run_skill_script_not_found(sample_skills_dir: Path) -> None:
    """Test running a non-existent script."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test skill with no scripts
    skill_one = toolset.get_skill('skill-one')
    assert skill_one.scripts is None or len(skill_one.scripts) == 0

    # Test skill with scripts
    skill_three = toolset.get_skill('skill-three')
    assert skill_three.scripts is not None
    script_names = [s.name for s in skill_three.scripts]
    # Script names should be relative paths
    assert 'nonexistent.py' not in script_names
    assert 'scripts/nonexistent.py' not in script_names


async def test_get_instructions_returns_system_prompt(sample_skills_dir: Path) -> None:
    """Test that get_instructions() returns the skills system prompt."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Create a minimal run context
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    instructions = await toolset.get_instructions(ctx)

    assert instructions is not None
    # Should include all skill names and descriptions
    assert 'skill-one' in instructions
    assert 'skill-two' in instructions
    assert 'skill-three' in instructions
    assert 'First test skill for basic operations' in instructions
    assert 'Second test skill with resources' in instructions
    assert 'Third test skill with executable scripts' in instructions
    # Should include usage instructions
    assert 'load_skill' in instructions
    assert 'read_skill_resource' in instructions
    assert 'run_skill_script' in instructions
    # Should include progressive disclosure guidance
    assert 'Progressive disclosure' in instructions or 'progressive disclosure' in instructions


async def test_get_instructions_empty_toolset() -> None:
    """Test that get_instructions() returns None for empty toolset."""
    toolset = SkillsToolset(directories=[])

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    instructions = await toolset.get_instructions(ctx)
    assert instructions is None


async def test_get_instructions_with_custom_template(sample_skills_dir: Path) -> None:
    """Test get_instructions uses custom template when provided."""
    custom_template = """# My Custom Skills

Available:
{skills_list}

Use load_skill(name) for details.
"""

    toolset = SkillsToolset(directories=[sample_skills_dir], instruction_template=custom_template)

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    instructions = await toolset.get_instructions(ctx)

    assert instructions is not None
    # Should use custom template
    assert '# My Custom Skills' in instructions
    assert 'Available:' in instructions
    assert 'Use load_skill(name) for details.' in instructions
    # Should still have skill list
    assert 'skill-one' in instructions
    assert 'skill-two' in instructions
    assert 'skill-three' in instructions
    # Should NOT have default template text
    assert 'Progressive disclosure' not in instructions


@pytest.mark.skip(reason='TestModel behavior is non-deterministic and calls random skills')
async def test_skills_instructions_injected_into_agent(sample_skills_dir: Path) -> None:
    """Test that SkillsToolset instructions are automatically injected into agent runs."""
    toolset = SkillsToolset(directories=[sample_skills_dir])
    agent: Agent[None, str] = Agent(TestModel(), toolsets=[toolset])

    result = await agent.run('Hello')

    # Check that the instructions were included in the model request
    # The instructions should be in the ModelRequest.instructions field
    model_requests = [m for m in result.all_messages() if isinstance(m, ModelRequest)]
    assert any(m.instructions is not None and 'skill-one' in m.instructions for m in model_requests), (
        'Skills instructions should be injected into model request'
    )


# ==================== New Architecture Tests ====================


def test_skills_toolset_with_directories(sample_skills_dir: Path) -> None:
    """Test SkillsToolset with directories parameter."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Verify skills were loaded
    assert len(toolset.skills) == 3
    assert 'skill-one' in toolset.skills
    assert 'skill-two' in toolset.skills
    assert 'skill-three' in toolset.skills


def test_skills_toolset_with_skills_list() -> None:
    """Test SkillsToolset with pre-loaded skills list."""
    # Create custom skills
    skill1 = Skill(
        name='custom-skill-1',
        description='First custom skill',
        content='Custom skill content 1',
        uri='',
    )
    skill2 = Skill(
        name='custom-skill-2',
        description='Second custom skill',
        content='Custom skill content 2',
        uri='',
    )

    toolset = SkillsToolset(skills=[skill1, skill2])

    # Verify skills were loaded
    assert len(toolset.skills) == 2
    assert 'custom-skill-1' in toolset.skills
    assert 'custom-skill-2' in toolset.skills


@pytest.mark.filterwarnings("ignore:Duplicate skill 'skill-three' found.*:UserWarning")
def test_skills_toolset_duplicate_detection(sample_skills_dir: Path, tmp_path: Path) -> None:
    """Test SkillsToolset duplicate detection across directories."""
    # Create isolated directories to avoid recursive discovery conflicts
    # sample_skills_dir is tmp_path, so we create siblings
    first_dir = tmp_path.parent / f'{tmp_path.name}_first'
    first_dir.mkdir()

    # Copy skill-three from sample_skills_dir to first_dir
    skill3_source = sample_skills_dir / 'skill-three'
    skill3_first = first_dir / 'skill-three'
    skill3_first.mkdir()
    (skill3_first / 'SKILL.md').write_text((skill3_source / 'SKILL.md').read_text())

    second_dir = tmp_path.parent / f'{tmp_path.name}_second'
    second_dir.mkdir()

    # Create skill-three (duplicate) in second directory
    skill3_dir = second_dir / 'skill-three'
    skill3_dir.mkdir()
    (skill3_dir / 'SKILL.md').write_text("""---
name: skill-three
description: Duplicate skill-three from second source
---

# Skill Three (Duplicate)

Duplicate of skill-three.
""")

    # Create skill-four in second directory
    skill4_dir = second_dir / 'skill-four'
    skill4_dir.mkdir()
    (skill4_dir / 'SKILL.md').write_text("""---
name: skill-four
description: Fourth test skill from second source
---

# Skill Four

Another skill from a different source.
""")

    toolset = SkillsToolset(directories=[first_dir, second_dir])

    # Verify skills from both directories were loaded
    # skill-three appears in both - last one wins (overrides first)
    assert len(toolset.skills) == 2  # skill-three (overridden), skill-four
    assert 'skill-three' in toolset.skills  # Overridden by second_dir (last one wins)
    assert 'skill-four' in toolset.skills  # From second_dir

    # Verify skill-three is from second directory (last occurrence overrides)
    skill_three_uri = toolset.skills['skill-three'].uri
    assert skill_three_uri is not None
    assert 'second' in skill_three_uri


# ==================== New Combined Mode Tests ====================


def test_toolset_with_both_skills_and_directories(sample_skills_dir: Path) -> None:
    """Test toolset with both programmatic skills and directories."""
    # Create programmatic skills
    prog_skill1 = Skill(
        name='programmatic-skill-one',
        description='Programmatic skill 1',
        content='Programmatic instructions 1',
        uri='memory://prog1',
    )
    prog_skill2 = Skill(
        name='programmatic-skill-two',
        description='Programmatic skill 2',
        content='Programmatic instructions 2',
        uri='memory://prog2',
    )

    toolset = SkillsToolset(skills=[prog_skill1, prog_skill2], directories=[sample_skills_dir])

    # Should have both programmatic and directory skills
    assert 'programmatic-skill-one' in toolset.skills
    assert 'programmatic-skill-two' in toolset.skills
    assert 'skill-one' in toolset.skills  # From sample_skills_dir
    assert 'skill-two' in toolset.skills  # From sample_skills_dir


@pytest.mark.filterwarnings('ignore:Duplicate skill.*:UserWarning')
def test_toolset_with_skills_directory_instances(sample_skills_dir: Path, tmp_path: Path) -> None:
    """Test toolset with SkillsDirectory instances."""
    # Create second directory
    second_dir = tmp_path / 'second'
    second_dir.mkdir()
    (second_dir / 'skill-extra').mkdir()
    (second_dir / 'skill-extra' / 'SKILL.md').write_text("""---
name: skill-extra
description: Extra skill
---

Extra content.
""")

    # Create SkillsDirectory instances
    dir1 = SkillsDirectory(path=sample_skills_dir)
    dir2 = SkillsDirectory(path=second_dir)

    toolset = SkillsToolset(directories=[dir1, dir2])

    # Should have skills from both directories
    assert 'skill-one' in toolset.skills  # From dir1
    assert 'skill-extra' in toolset.skills  # From dir2


@pytest.mark.filterwarnings('ignore:Duplicate skill.*:UserWarning')
def test_toolset_mixed_directory_types(sample_skills_dir: Path, tmp_path: Path) -> None:
    """Test toolset with mixed directory types (str, Path, SkillsDirectory)."""
    # Create second directory
    second_dir = tmp_path / 'second'
    second_dir.mkdir()
    (second_dir / 'skill-mixed').mkdir()
    (second_dir / 'skill-mixed' / 'SKILL.md').write_text("""---
name: skill-mixed
description: Mixed test skill
---

Mixed content.
""")

    # Create SkillsDirectory instance
    skills_dir_instance = SkillsDirectory(path=second_dir)

    # Mix str, Path, and SkillsDirectory
    toolset = SkillsToolset(
        directories=[
            str(sample_skills_dir),  # str
            skills_dir_instance,  # SkillsDirectory
        ]
    )

    # Should have skills from all sources
    assert 'skill-one' in toolset.skills
    assert 'skill-mixed' in toolset.skills


@pytest.mark.filterwarnings('ignore:Duplicate skill.*:UserWarning')
def test_toolset_combined_with_duplicate_override(tmp_path: Path) -> None:
    """Test that directory skills override programmatic skills with same name."""
    # Create directory with skill
    skill_dir = tmp_path / 'skills'
    skill_dir.mkdir()
    (skill_dir / 'override-skill').mkdir()
    (skill_dir / 'override-skill' / 'SKILL.md').write_text("""---
name: override-skill
description: Directory version
---

Directory content.
""")

    # Create programmatic skill with same name
    prog_skill = Skill(
        name='override-skill',
        description='Programmatic version',
        content='Programmatic content',
        uri='memory://override',
    )

    # Directory skills should override programmatic skills
    toolset = SkillsToolset(skills=[prog_skill], directories=[skill_dir])

    assert 'override-skill' in toolset.skills
    # Directory version should win
    assert 'Directory content' in toolset.skills['override-skill'].content
    assert 'Programmatic content' not in toolset.skills['override-skill'].content


async def test_toolset_combined_mode_tools(sample_skills_dir: Path) -> None:
    """Test that tools work correctly with combined mode."""
    prog_skill = Skill(
        name='prog-test',
        description='Test skill',
        content='Test instructions',
        uri='memory://test',
    )

    toolset = SkillsToolset(skills=[prog_skill], directories=[sample_skills_dir])

    # Create context for tool calls
    ctx = InternalRunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)

    # Test list_skills - should include both
    result = await toolset.tools['list_skills'].function(ctx)
    assert 'prog-test' in result
    assert 'skill-one' in result

    # Test load_skill for programmatic skill
    result = await toolset.tools['load_skill'].function(ctx, skill_name='prog-test')
    assert 'Test instructions' in result

    # Test load_skill for directory skill
    result = await toolset.tools['load_skill'].function(ctx, skill_name='skill-one')
    assert 'Skill One' in result
