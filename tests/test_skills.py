"""Tests for skills toolset."""

from pathlib import Path

import pytest
from inline_snapshot import snapshot

from pydantic_ai.toolsets.skills import (
    Skill,
    SkillMetadata,
    SkillNotFoundError,
    SkillResource,
    SkillScript,
    SkillsToolset,
    SkillValidationError,
    discover_skills,
    parse_skill_md,
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
    assert metadata.extra == {}


def test_skill_metadata_with_extra_fields() -> None:
    """Test SkillMetadata with additional fields."""
    metadata = SkillMetadata(
        name='test-skill', description='A test skill', extra={'version': '1.0.0', 'author': 'Test Author'}
    )

    assert metadata.extra['version'] == '1.0.0'
    assert metadata.extra['author'] == 'Test Author'


def test_skill_resource_creation() -> None:
    """Test creating SkillResource."""
    resource = SkillResource(name='FORMS.md', path=Path('/tmp/skill/FORMS.md'))

    assert resource.name == 'FORMS.md'
    assert resource.path == Path('/tmp/skill/FORMS.md')
    assert resource.content is None


def test_skill_script_creation() -> None:
    """Test creating SkillScript."""
    script = SkillScript(name='test_script', path=Path('/tmp/skill/scripts/test_script.py'), skill_name='test-skill')

    assert script.name == 'test_script'
    assert script.path == Path('/tmp/skill/scripts/test_script.py')
    assert script.skill_name == 'test-skill'


def test_skill_creation() -> None:
    """Test creating a complete Skill."""
    metadata = SkillMetadata(name='test-skill', description='A test skill')
    resource = SkillResource(name='FORMS.md', path=Path('/tmp/skill/FORMS.md'))
    script = SkillScript(name='test_script', path=Path('/tmp/skill/scripts/test_script.py'), skill_name='test-skill')

    skill = Skill(
        name='test-skill',
        path=Path('/tmp/skill'),
        metadata=metadata,
        content='# Instructions\n\nTest instructions.',
        resources=[resource],
        scripts=[script],
    )

    assert skill.name == 'test-skill'
    assert skill.path == Path('/tmp/skill')
    assert skill.metadata.name == 'test-skill'
    assert skill.content == '# Instructions\n\nTest instructions.'
    assert len(skill.resources) == 1
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

    frontmatter, instructions = parse_skill_md(content)

    assert frontmatter['name'] == 'test-skill'
    assert frontmatter['description'] == 'A test skill for testing'
    assert frontmatter['version'] == '1.0.0'
    assert instructions.startswith('# Test Skill')


def test_parse_skill_md_without_frontmatter() -> None:
    """Test parsing SKILL.md without frontmatter."""
    content = """# Test Skill

This skill has no frontmatter.
"""

    frontmatter, instructions = parse_skill_md(content)

    assert frontmatter == {}
    assert instructions.startswith('# Test Skill')


def test_parse_skill_md_empty_frontmatter() -> None:
    """Test parsing SKILL.md with empty frontmatter."""
    content = """---
---

# Test Skill

Content here.
"""

    frontmatter, instructions = parse_skill_md(content)

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
        parse_skill_md(content)


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

    frontmatter, _ = parse_skill_md(content)

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

    frontmatter, _ = parse_skill_md(content)

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

    skills = discover_skills([tmp_path], validate=True)

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

    skills = discover_skills([tmp_path], validate=True)

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

    skills = discover_skills([tmp_path], validate=True)

    assert len(skills) == 1
    assert len(skills[0].resources) == 2
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

    skills = discover_skills([tmp_path], validate=True)

    assert len(skills) == 1
    assert len(skills[0].scripts) == 2
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

    skills = discover_skills([tmp_path], validate=True)

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
    skills = discover_skills([tmp_path], validate=True)
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
    skills = discover_skills([tmp_path], validate=False)
    assert len(skills) == 1
    assert skills[0].name == 'test-skill'  # Uses folder name


def test_discover_skills_nonexistent_directory(tmp_path: Path) -> None:
    """Test discovering skills from non-existent directory."""
    nonexistent = tmp_path / 'does-not-exist'

    # Should not raise, just log warning
    skills = discover_skills([nonexistent], validate=True)
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

    skills = discover_skills([tmp_path], validate=True)

    assert len(skills) == 1
    assert len(skills[0].resources) == 3

    resource_names = {r.name for r in skills[0].resources}
    assert 'resources/schema.json' in resource_names
    assert 'resources/template.txt' in resource_names
    assert 'resources/nested/data.csv' in resource_names


# ==================== SkillsToolset Tests ====================


def test_toolset_initialization(sample_skills_dir: Path) -> None:
    """Test SkillsToolset initialization."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    assert len(toolset.skills) == 3
    assert 'skill-one' in toolset.skills
    assert 'skill-two' in toolset.skills
    assert 'skill-three' in toolset.skills


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
<description>Formatted list of available skills with names and descriptions.</description>
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
    assert len(skill.resources) == 2

    resource_names = [r.name for r in skill.resources]
    assert 'FORMS.md' in resource_names
    assert 'REFERENCE.md' in resource_names

    # Check that resources can be read
    for resource in skill.resources:
        assert resource.path.exists()
        assert resource.path.is_file()


async def test_read_skill_resource_not_found(sample_skills_dir: Path) -> None:
    """Test reading a non-existent resource."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test skill with no resources
    skill_one = toolset.get_skill('skill-one')
    assert len(skill_one.resources) == 0

    # Test skill with resources
    skill_two = toolset.get_skill('skill-two')
    resource_names = [r.name for r in skill_two.resources]
    assert 'NONEXISTENT.md' not in resource_names


async def test_run_skill_script_tool(sample_skills_dir: Path) -> None:
    """Test the run_skill_script tool."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test that skill-three has scripts
    skill = toolset.get_skill('skill-three')
    assert len(skill.scripts) == 2

    script_names = [s.name for s in skill.scripts]
    assert 'hello' in script_names
    assert 'echo' in script_names

    # Check that scripts can be found
    for script in skill.scripts:
        assert script.path.exists()
        assert script.path.is_file()
        assert script.path.suffix == '.py'


async def test_run_skill_script_not_found(sample_skills_dir: Path) -> None:
    """Test running a non-existent script."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    # Test skill with no scripts
    skill_one = toolset.get_skill('skill-one')
    assert len(skill_one.scripts) == 0

    # Test skill with scripts
    skill_three = toolset.get_skill('skill-three')
    script_names = [s.name for s in skill_three.scripts]
    assert 'nonexistent' not in script_names


def test_toolset_refresh(sample_skills_dir: Path) -> None:
    """Test refreshing skills."""
    toolset = SkillsToolset(directories=[sample_skills_dir])

    initial_count = len(toolset.skills)

    # Add a new skill
    new_skill_dir = sample_skills_dir / 'skill-four'
    new_skill_dir.mkdir()
    (new_skill_dir / 'SKILL.md').write_text("""---
name: skill-four
description: Fourth skill added after initialization
---

New skill content.
""")

    # Refresh
    toolset.refresh()

    assert len(toolset.skills) == initial_count + 1
    assert 'skill-four' in toolset.skills


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

    toolset = SkillsToolset(directories=[], auto_discover=False)

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
