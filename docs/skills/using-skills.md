# Using Skills

This guide covers how to integrate and use the Skills framework in your Pydantic AI agents.

## SkillsToolset API

The [`SkillsToolset`][pydantic_ai.toolsets.skills.SkillsToolset] is the main entry point for working with skills.

### Initialization

```python
from pydantic_ai.toolsets import SkillsToolset

toolset = SkillsToolset(
    directories=["./skills", "./shared-skills"],
    auto_discover=True,           # Auto-discover skills on init (default: True)
    validate=True,                # Validate skill structure (default: True)
    id=None,                      # Unique identifier (default: None)
    script_timeout=30,            # Script execution timeout in seconds (default: 30)
    python_executable=None,       # Python executable path (default: sys.executable)
    instruction_template=None,    # Custom instruction template (default: None)
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `get_skill(name)` | Get a specific skill object by name. Raises `SkillNotFoundError` if not found |
| `refresh()` | Re-scan directories for skills |

### Properties

| Property | Description |
|----------|-------------|
| `skills` | Dictionary of loaded skills (`dict[str, Skill]`) |

### Customizing Instructions

You can customize the instruction template that gets injected into the agent's system prompt:

```python
custom_template = """# My Custom Skills Section

Available tools:
{skills_list}

Use load_skill(name) to get details.
"""

toolset = SkillsToolset(
    directories=["./skills"],
    instruction_template=custom_template
)
```

The template must include the `{skills_list}` placeholder, which will be replaced with the formatted list of available skills.

## The Four Tools

The `SkillsToolset` provides four tools to agents:

### 1. `list_skills()`

Lists all available skills with their descriptions.

**Returns**: Formatted markdown with skill names and descriptions

**When to use**: Optional - skills are already listed in the system prompt automatically. Use only if the agent needs to re-check available skills dynamically.

**Example**:

```python
# Agent can call this tool
list_skills()

# Output:
# Available Skills:
# - arxiv-search: Search arXiv for research papers
# - web-research: Research topics on the web
# - data-analyzer: Analyze CSV and JSON files
```

### 2. `load_skill(skill_name)`

Loads the complete instructions for a specific skill.

**Parameters**:

- `skill_name` (str) - Name of the skill to load

**Returns**: Full SKILL.md content (as a string) including detailed instructions, available resources, and scripts

**When to use**: When the agent needs detailed instructions for using a skill

**Example**:

```python
# Agent loads skill details
load_skill("arxiv-search")

# Returns full SKILL.md content with:
# - When to use
# - Step-by-step instructions
# - Example invocations
# - Available resources and scripts
```

### 3. `read_skill_resource(skill_name, resource_name)`

Reads additional resource files from a skill.

**Parameters**:

- `skill_name` (str) - Name of the skill
- `resource_name` (str) - Resource filename (e.g., "FORMS.md")

**Returns**: Content of the resource file

**When to use**: When a skill references additional documentation or data files

**Example**:

```python
# Agent reads a skill resource
read_skill_resource("web-research", "FORMS.md")

# Returns content of the FORMS.md file
```

### 4. `run_skill_script(skill_name, script_name, args)`

Executes a Python script from a skill.

**Parameters**:

- `skill_name` (str) - Name of the skill
- `script_name` (str) - Script name without .py extension
- `args` (list[str] | None, optional) - Command-line arguments passed to the script

**Returns**: Script output (stdout and stderr combined)

**When to use**: When a skill needs to execute custom code

**Example**:

```python
# Agent executes a script
run_skill_script(
    skill_name="arxiv-search",
    script_name="arxiv_search",
    args=["machine learning", "--max-papers", "3"]
)

# Returns script output with search results
```

## Skill Discovery

Skills can be discovered programmatically using the [`discover_skills`][pydantic_ai.toolsets.skills.discover_skills] function:

```python
from pydantic_ai.toolsets import discover_skills

skills = discover_skills(
    directories=["./skills"],
    validate=True
)

for skill in skills:
    print(f"{skill.name}: {skill.metadata.description}")
    print(f"  Resources: {[r.name for r in skill.resources]}")
    print(f"  Scripts: {[s.name for s in skill.scripts]}")
```

This is useful for:

- Listing available skills before creating an agent
- Validating skill structure in tests
- Building custom skill management tools
- Generating documentation about available skills

## Usage Patterns

### Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset

# Create toolset with skills
skills_toolset = SkillsToolset(directories=["./skills"])

# Create agent with skills
agent = Agent(
    model='openai:gpt-4o',
    instructions="You are a helpful assistant.",
    toolsets=[skills_toolset]
)

# Agent automatically has access to all skill tools
result = await agent.run("Search for papers about transformers")
```

### Multiple Skill Directories

```python
# Load skills from multiple directories
skills_toolset = SkillsToolset(
    directories=[
        "./my-skills",           # Project-specific skills
        "./shared-skills",       # Shared across projects
        "~/.pydantic-ai/skills"  # Global skills
    ]
)
```

### Custom Script Timeout

```python
# Increase timeout for long-running scripts
skills_toolset = SkillsToolset(
    directories=["./skills"],
    script_timeout=120  # 2 minutes
)
```

### Programmatic Access

```python
# Access skills programmatically
toolset = SkillsToolset(directories=["./skills"])

# Get a specific skill
skill = toolset.get_skill("arxiv-search")
print(f"Skill: {skill.name}")
print(f"Description: {skill.metadata.description}")
print(f"Scripts: {[s.name for s in skill.scripts]}")

# Refresh skills (rescans directories)
toolset.refresh()
```

### Custom Instructions Template

```python
# Customize how skills appear in system prompt
template = """
## Available Research Tools

The following specialized tools are available:
{skills_list}

To use a tool, first load its instructions with load_skill(name).
"""

toolset = SkillsToolset(
    directories=["./skills"],
    instruction_template=template
)
```

## Error Handling

The toolset raises specific exceptions for different error conditions:

```python
from pydantic_ai.toolsets.skills import (
    SkillNotFoundError,
    SkillValidationError,
    SkillResourceLoadError,
    SkillScriptExecutionError,
)

try:
    toolset = SkillsToolset(directories=["./skills"])
    skill = toolset.get_skill("non-existent")
except SkillNotFoundError as e:
    print(f"Skill not found: {e}")
except SkillValidationError as e:
    print(f"Invalid skill structure: {e}")
```

## Best Practices

### Organization

- **Organize by domain**: Group related skills in subdirectories
- **Use descriptive directories**: `./skills/research/`, `./skills/data-analysis/`

### Testing

- **Test skills independently**: Run scripts directly before adding to skills
- **Validate structure**: Use `validate=True` during development
- **Use programmatic discovery**: Test skill loading in your test suite

### Security

We strongly recommend using Skills only from trusted sources: those you created yourself or obtained from trusted sources. Skills provide AI Agents with new capabilities through instructions and code, and while this makes them powerful, it also means a malicious Skill can direct agents to invoke tools or execute code in ways that don't match the Skill's stated purpose.

!!! warning
    If you must use a Skill from an untrusted or unknown source, exercise extreme caution and thoroughly audit it before use. Depending on the access agents have when executing the Skill, malicious Skills could lead to data exfiltration, unauthorized system access, or other security risks.
