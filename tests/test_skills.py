"""Tests for skills toolset."""

from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.skills import (
    CallableSkillScriptExecutor,
    LocalSkillScriptExecutor,
    Skill,
    SkillMetadata,
    SkillNotFoundError,
    SkillResource,
    SkillScript,
    SkillsDirectory,
    SkillsToolset,
    SkillValidationError,
)
from pydantic_ai.toolsets.skills._directory import (
    _discover_skills,  # pyright: ignore[reportPrivateUsage]
    _parse_skill_md,  # pyright: ignore[reportPrivateUsage]
)

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
import sys
print(f"Hello, {sys.argv[1] if len(sys.argv) > 1 else 'World'}!")
""")
    (scripts_dir / 'echo.py').write_text("""#!/usr/bin/env python3
import sys
print(' '.join(sys.argv[1:]))
""")

    return tmp_path


# ==================== Type Tests ====================


def test_skill_metadata_creation() -> None:
    """Test creating SkillMetadata with required fields."""
    metadata = SkillMetadata(name='test-skill', description='A test skill')

    assert metadata.name == 'test-skill'
    assert metadata.description == 'A test skill'
    assert metadata.extra is None


def test_skill_metadata_with_extra_fields() -> None:
    """Test SkillMetadata with additional fields."""
    metadata = SkillMetadata(
        name='test-skill', description='A test skill', extra={'version': '1.0.0', 'author': 'Test Author'}
    )

    assert metadata.extra is not None
    assert metadata.extra['version'] == '1.0.0'
    assert metadata.extra['author'] == 'Test Author'


def test_skill_resource_creation() -> None:
    """Test creating SkillResource."""
    resource = SkillResource(name='FORMS.md', uri='/tmp/skill/FORMS.md')

    assert resource.name == 'FORMS.md'
    assert resource.uri == '/tmp/skill/FORMS.md'
    assert resource.content is None


def test_skill_script_creation() -> None:
    """Test creating SkillScript."""
    script = SkillScript(name='test_script', uri='/tmp/skill/scripts/test_script.py', skill_name='test-skill')

    assert script.name == 'test_script'
    assert script.uri == '/tmp/skill/scripts/test_script.py'
    assert script.skill_name == 'test-skill'


def test_skill_creation() -> None:
    """Test creating a complete Skill."""
    from pydantic_ai.toolsets.skills import LocalSkill

    metadata = SkillMetadata(name='test-skill', description='A test skill')
    resource = SkillResource(name='FORMS.md', uri='/tmp/skill/FORMS.md')
    script = SkillScript(name='test_script', uri='/tmp/skill/scripts/test_script.py', skill_name='test-skill')

    skill = LocalSkill(
        name='test-skill',
        uri='/tmp/skill',
        metadata=metadata,
        content='# Instructions\n\nTest instructions.',
        resources=[resource],
        scripts=[script],
    )

    assert skill.name == 'test-skill'
    assert skill.uri == '/tmp/skill'
    assert skill.metadata.name == 'test-skill'
    assert skill.content == '# Instructions\n\nTest instructions.'
    assert skill.resources is not None
    assert len(skill.resources) == 1
    assert skill.scripts is not None
    assert len(skill.scripts) == 1


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
    assert script_names == {'search', 'process'}


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
    """Test discovering resources in resources/ subdirectory."""
    skill_dir = tmp_path / 'test-skill'
    skill_dir.mkdir()

    (skill_dir / 'SKILL.md').write_text("""---
name: test-skill
description: Skill with resources subdirectory
---

Content.
""")

    resources_dir = skill_dir / 'resources'
    resources_dir.mkdir()
    (resources_dir / 'schema.json').write_text('{}')
    (resources_dir / 'template.txt').write_text('template')

    nested_dir = resources_dir / 'nested'
    nested_dir.mkdir()
    (nested_dir / 'data.csv').write_text('col1,col2')

    skills = _discover_skills(tmp_path, validate=True)

    assert len(skills) == 1
    assert skills[0].resources is not None and len(skills[0].resources) == 3

    resource_names = {r.name for r in skills[0].resources}
    assert 'resources/schema.json' in resource_names
    assert 'resources/template.txt' in resource_names
    assert 'resources/nested/data.csv' in resource_names


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
    import os

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
    from pydantic_ai._run_context import RunContext
    from pydantic_ai._tool_manager import ToolManager
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.usage import RunUsage

    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Build a run context to get tool definitions via ToolManager
    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )

    # Get tool manager and prepare for run step
    import asyncio

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
<description>Dictionary mapping skill names to their descriptions.
    An empty dictionary if no skills are available.</description>
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
<summary>Load full instructions for a skill.

Always load the skill before using read_skill_resource
or run_skill_script to understand the skill's capabilities, available
resources, scripts, and their usage patterns.</summary>
<returns>
<description>Full skill instructions including available resources and scripts.</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'skill_name': {'description': 'Name of the skill to load.', 'type': 'string'}},
                    'required': ['skill_name'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='read_skill_resource',
                description="""\
<summary>Read a resource file from a skill (e.g., FORMS.md, REFERENCE.md).

Call load_skill first to see which resources are available.</summary>
<returns>
<description>The resource file content.</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {
                        'resource_name': {
                            'description': 'The resource filename (e.g., "FORMS.md").',
                            'type': 'string',
                        },
                        'skill_name': {'description': 'Name of the skill.', 'type': 'string'},
                    },
                    'required': ['skill_name', 'resource_name'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='run_skill_script',
                description="""\
<summary>Execute a skill script with command-line arguments.

Call load_skill first to understand the script's expected arguments,
usage patterns, and example invocations. Running scripts without
loading instructions first will likely fail.</summary>
<returns>
<description>The script's output (stdout and stderr combined).</description>
</returns>\
""",
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {
                        'args': {
                            'anyOf': [{'items': {'type': 'string'}, 'type': 'array'}, {'type': 'null'}],
                            'default': None,
                            'description': 'Optional list of command-line arguments (positional args, flags, values).',
                        },
                        'script_name': {
                            'description': 'The script name (without .py extension).',
                            'type': 'string',
                        },
                        'skill_name': {'description': 'Name of the skill.', 'type': 'string'},
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
    assert skill.metadata.description == 'First test skill for basic operations'


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
    assert toolset.skills['skill-one'].metadata.description == 'First test skill for basic operations'
    assert toolset.skills['skill-two'].metadata.description == 'Second test skill with resources'
    assert toolset.skills['skill-three'].metadata.description == 'Third test skill with executable scripts'


async def test_load_skill_tool(sample_skills_dir: Path) -> None:
    """Test the load_skill tool."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # The tools are internal, so we test via the public methods
    # We can check that the skills were loaded correctly
    skill = toolset.get_skill('skill-one')
    assert skill is not None
    assert skill.name == 'skill-one'
    assert 'First test skill for basic operations' in skill.metadata.description
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


async def test_run_skill_script_tool(sample_skills_dir: Path) -> None:
    """Test the run_skill_script tool."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test that skill-three has scripts
    skill = toolset.get_skill('skill-three')
    assert skill.scripts is not None and len(skill.scripts) == 2

    script_names = [s.name for s in skill.scripts]
    assert 'hello' in script_names
    assert 'echo' in script_names

    # Check that scripts can be found
    for script in skill.scripts:
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
    assert 'nonexistent' not in script_names


async def test_get_instructions_returns_system_prompt(sample_skills_dir: Path) -> None:
    """Test that get_instructions() returns the skills system prompt."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage

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
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage

    toolset = SkillsToolset(directories=[])

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())

    instructions = await toolset.get_instructions(ctx)
    assert instructions is None


async def test_get_instructions_with_custom_template(sample_skills_dir: Path) -> None:
    """Test get_instructions uses custom template when provided."""
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage

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
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelRequest
    from pydantic_ai.models.test import TestModel

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


async def test_skill_read_resource(sample_skills_dir: Path) -> None:
    """Test reading a resource from a Skill."""
    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    skills = _discover_skills(sample_skills_dir)
    skill_two = next(s for s in skills if s.name == 'skill-two')

    # Read a resource
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    resource_content = await skill_two.read_resource(ctx, 'FORMS.md')

    assert 'Forms' in resource_content
    assert 'Form filling guide' in resource_content


async def test_skill_run_script(sample_skills_dir: Path) -> None:
    """Test running a script via a Skill."""
    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    skills = _discover_skills(sample_skills_dir)
    skill_three = next(s for s in skills if s.name == 'skill-three')

    # Run a script
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    output = await skill_three.run_script(ctx, 'hello', args=['World'])

    assert 'Hello, World!' in output


async def test_skill_run_script_with_custom_executor(sample_skills_dir: Path) -> None:
    """Test custom executor with Skill."""
    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.toolsets.skills import SkillScript
    from pydantic_ai.usage import RunUsage

    # Create a custom executor as a callable
    def custom_executor(skill: Skill, script: SkillScript, args: list[str] | None = None) -> str:
        return f'Custom execution of {script.name} from skill {skill.name}'

    executor = CallableSkillScriptExecutor(custom_executor)
    skills = _discover_skills(sample_skills_dir)
    skill_three = next(s for s in skills if s.name == 'skill-three')

    # Attach custom executor
    skill_three.script_executor = executor

    # Run a script - should use custom executor
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    output = await skill_three.run_script(ctx, 'hello', args=['Test'])

    assert 'Custom execution of hello from skill skill-three' in output


async def test_skill_run_script_with_custom_python(sample_skills_dir: Path) -> None:
    """Test Skill with custom Python executable."""
    import sys

    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    # Use current Python executable explicitly
    executor = LocalSkillScriptExecutor(python_executable=sys.executable)
    skills = _discover_skills(sample_skills_dir)
    skill_three = next(s for s in skills if s.name == 'skill-three')

    # Attach executor
    skill_three.script_executor = executor

    # Run a script
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    output = await skill_three.run_script(ctx, 'echo', args=['test', 'message'])

    assert 'test message' in output


async def test_skills_directory_with_callable_executor(sample_skills_dir: Path) -> None:
    """Test SkillsDirectory with a callable script executor."""
    from pydantic_ai.usage import RunUsage

    # Track execution
    executed_scripts: list[dict[str, Any]] = []

    async def custom_executor(skill: Skill, script: SkillScript, args: list[str] | None = None) -> str:
        """Custom async executor function."""
        executed_scripts.append({'skill': skill.name, 'script': script.name, 'args': args})
        return f'Custom execution of {script.name} with args {args}'

    # Create SkillsDirectory with callable executor
    skills_dir = SkillsDirectory(
        path=sample_skills_dir,
        script_executor=custom_executor,
    )

    # Load a skill with scripts
    skill = skills_dir.load_skill(next(uri for uri in skills_dir.skills if 'skill-three' in uri))

    # Run a script
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    output = await skill.run_script(ctx, 'echo', args=['hello'])

    # Verify custom executor was called
    assert len(executed_scripts) == 1
    assert executed_scripts[0]['skill'] == 'skill-three'
    assert executed_scripts[0]['script'] == 'echo'
    assert executed_scripts[0]['args'] == ['hello']
    assert 'Custom execution' in output


async def test_skills_directory_with_sync_callable_executor(sample_skills_dir: Path) -> None:
    """Test SkillsDirectory with a synchronous callable script executor."""
    from pydantic_ai.usage import RunUsage

    # Track execution
    executed_scripts: list[dict[str, str]] = []

    def sync_executor(skill: Skill, script: SkillScript, args: list[str] | None = None) -> str:
        """Custom sync executor function."""
        executed_scripts.append({'skill': skill.name, 'script': script.name})
        return f'Sync execution of {script.name}'

    # Create SkillsDirectory with sync callable executor
    skills_dir = SkillsDirectory(
        path=sample_skills_dir,
        script_executor=sync_executor,
    )

    # Load a skill with scripts
    skill = skills_dir.load_skill(next(uri for uri in skills_dir.skills if 'skill-three' in uri))

    # Run a script
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)
    output = await skill.run_script(ctx, 'echo')

    # Verify sync executor was called
    assert len(executed_scripts) == 1
    assert 'Sync execution' in output


async def test_skills_directory_with_custom_timeout(sample_skills_dir: Path) -> None:
    """Test SkillsDirectory with custom script executor timeout."""
    # Create SkillsDirectory with custom executor that has timeout
    executor = LocalSkillScriptExecutor(timeout=60)
    skills_dir = SkillsDirectory(
        path=sample_skills_dir,
        script_executor=executor,
    )

    # Load a skill and verify it can execute scripts (timeout is configured in executor)
    skill = skills_dir.load_skill(next(uri for uri in skills_dir.skills if 'skill-three' in uri))
    assert skill is not None


def test_skills_toolset_with_directories(sample_skills_dir: Path) -> None:
    """Test SkillsToolset with directories parameter."""
    from pydantic_ai.toolsets.skills import SkillsToolset

    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Verify skills were loaded
    assert len(toolset.skills) == 3
    assert 'skill-one' in toolset.skills
    assert 'skill-two' in toolset.skills
    assert 'skill-three' in toolset.skills


def test_skills_toolset_with_skills_list() -> None:
    """Test SkillsToolset with pre-loaded skills list."""
    from pydantic_ai.toolsets.skills import LocalSkill, SkillsToolset

    # Create custom skills
    skill1 = LocalSkill(
        name='custom-skill-1',
        uri='',
        metadata=SkillMetadata(name='custom-skill-1', description='First custom skill'),
        content='Custom skill content 1',
    )
    skill2 = LocalSkill(
        name='custom-skill-2',
        uri='',
        metadata=SkillMetadata(name='custom-skill-2', description='Second custom skill'),
        content='Custom skill content 2',
    )

    toolset = SkillsToolset(skills=[skill1, skill2])

    # Verify skills were loaded
    assert len(toolset.skills) == 2
    assert 'custom-skill-1' in toolset.skills
    assert 'custom-skill-2' in toolset.skills


@pytest.mark.filterwarnings("ignore:Duplicate skill 'skill-three' found.*:UserWarning")
def test_skills_toolset_duplicate_detection(sample_skills_dir: Path, tmp_path: Path) -> None:
    """Test SkillsToolset duplicate detection across directories."""
    from pydantic_ai.toolsets.skills import SkillsToolset

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
    assert 'second' in toolset.skills['skill-three'].uri


# ==================== New Combined Mode Tests ====================


def test_toolset_with_both_skills_and_directories(sample_skills_dir: Path) -> None:
    """Test toolset with both programmatic skills and directories."""
    from pydantic_ai.toolsets.skills import LocalSkill, SkillMetadata, SkillsToolset

    # Create programmatic skills
    prog_skill1 = LocalSkill(
        name='programmatic-skill-one',
        uri='memory://prog1',
        metadata=SkillMetadata(name='programmatic-skill-one', description='Programmatic skill 1'),
        content='Programmatic instructions 1',
    )
    prog_skill2 = LocalSkill(
        name='programmatic-skill-two',
        uri='memory://prog2',
        metadata=SkillMetadata(name='programmatic-skill-two', description='Programmatic skill 2'),
        content='Programmatic instructions 2',
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
    from pydantic_ai.toolsets.skills import SkillsDirectory, SkillsToolset

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
    from pydantic_ai.toolsets.skills import SkillsDirectory, SkillsToolset

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
    from pydantic_ai.toolsets.skills import LocalSkill, SkillMetadata, SkillsToolset

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
    prog_skill = LocalSkill(
        name='override-skill',
        uri='memory://override',
        metadata=SkillMetadata(name='override-skill', description='Programmatic version'),
        content='Programmatic content',
    )

    # Directory skills should override programmatic skills
    toolset = SkillsToolset(skills=[prog_skill], directories=[skill_dir])

    assert 'override-skill' in toolset.skills
    # Directory version should win
    assert 'Directory content' in toolset.skills['override-skill'].content
    assert 'Programmatic content' not in toolset.skills['override-skill'].content


async def test_toolset_combined_mode_tools(sample_skills_dir: Path) -> None:
    """Test that tools work correctly with combined mode."""
    from pydantic_ai.toolsets.skills import LocalSkill, SkillMetadata, SkillsToolset

    prog_skill = LocalSkill(
        name='prog-test',
        uri='memory://test',
        metadata=SkillMetadata(name='prog-test', description='Test skill'),
        content='Test instructions',
    )

    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    toolset = SkillsToolset(skills=[prog_skill], directories=[sample_skills_dir])

    # Create context for tool calls
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), prompt=None, messages=[], run_step=0)

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
