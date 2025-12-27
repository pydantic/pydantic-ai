# Using Skills

This guide covers how to integrate and use the Skills framework in your Pydantic AI agents.

## SkillsToolset API

The [`SkillsToolset`][pydantic_ai.toolsets.skills.SkillsToolset] is the main entry point for working with skills.

### Initialization

```python
from pydantic_ai.toolsets import SkillsToolset

# Default: uses ./skills directory
toolset = SkillsToolset()

# Multiple directories (can be str, Path, or SkillsDirectory instances)
toolset = SkillsToolset(
    directories=["./skills", "./shared-skills"]
)

# Programmatic skills (must use LocalSkill or custom Skill subclass)
from pydantic_ai.toolsets.skills import LocalSkill, SkillMetadata

custom_skill = LocalSkill(
    name="my-skill",
    uri="/path/to/custom",
    metadata=SkillMetadata(name="my-skill", description="Custom skill"),
    content="Instructions here",
)
toolset = SkillsToolset(skills=[custom_skill])

# Combined mode: both programmatic and directory-based
toolset = SkillsToolset(
    skills=[custom_skill],
    directories=["./skills"]
)

# Using SkillsDirectory instances directly
from pydantic_ai.toolsets.skills import SkillsDirectory

skills_dir = SkillsDirectory(path="./skills", validate=True)
toolset = SkillsToolset(
    directories=[skills_dir, "./more-skills"]  # Mix SkillsDirectory and paths
)

# Configuration options
toolset = SkillsToolset(
    directories=["./skills"],
    validate=True,                # Validate skill structure (default: True)
    max_depth=3,                  # Max directory depth for discovery (default: 3)
    id=None,                      # Unique identifier (default: None)
    instruction_template=None,    # Custom instruction template (default: None)
)
```

### Handling Duplicate Skills

When the same skill name appears multiple times, **the last occurrence wins**:

```python
# Programmatic skills loaded first
prog_skill = LocalSkill(name="data-tool", ...)

# Then directory skills (which override programmatic ones)
toolset = SkillsToolset(
    skills=[prog_skill],
    directories=["./skills"]  # If contains "data-tool", it overrides prog_skill
)

# Multiple directories: later directories override earlier ones
toolset = SkillsToolset(
    directories=["./skills", "./override-skills"]
    # Skills in ./override-skills override those in ./skills
)
```

A warning is emitted when duplicates are detected:

```text
UserWarning: Duplicate skill 'data-tool' found. Overriding previous occurrence.
```

**Best Practice:** Use unique skill names, or intentionally use this behavior for environment-specific overrides (e.g., dev/prod skill variations).

### Key Methods

| Method | Description |
|--------|-------------|
| `get_skill(name)` | Get a specific skill object by name. Raises `SkillNotFoundError` if not found |
| `get_instructions(ctx)` | Return instructions to inject into agent's system prompt. Called automatically by the agent |

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

**Returns**: `dict[str, str]` - Dictionary mapping skill names to descriptions

**When to use**: Optional - skills are already listed in the system prompt automatically. Use only if the agent needs to re-check available skills dynamically.

**Example**:

```python
# Agent can call this tool
result = list_skills()

# Returns:
{
    "arxiv-search": "Search arXiv for research papers",
    "web-research": "Research topics on the web",
    "data-analyzer": "Analyze CSV and JSON files"
}
```

### 2. `load_skill(skill_name)`

Loads the complete instructions for a specific skill.

**Parameters**:

- `skill_name` (str) - Name of the skill to load

**Returns**: `str` - Formatted string with complete skill details including:

- Skill name and description
- File path/URI
- List of available resources
- List of available scripts
- Full SKILL.md content

**Example return format**:

```text
# Skill: arxiv-search
**Description:** Search arXiv for research papers
**Path:** /path/to/skills/arxiv-search
**Available Resources:**
- FORMS.md
- REFERENCE.md
**Available Scripts:**
- arxiv_search

---

# arXiv Search

[Full SKILL.md content here...]
```

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

## Skill Types

The skills framework uses an abstract base class pattern:

- [`Skill`][pydantic_ai.toolsets.skills.Skill]: Abstract base class defining the interface for all skills. Subclasses must implement:
  - `read_resource(ctx, resource_uri)`: Read a resource file from the skill
  - `run_script(ctx, script_uri, args)`: Execute a script from the skill
- [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill]: Filesystem-based implementation that reads resources and runs scripts from local directories

This design allows you to create custom skill implementations (e.g., for remote skills, database-backed skills, or skills with custom execution environments).

### Skill URI Semantics

Every skill has a `uri` field that identifies its location:

- **For [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill]**: Absolute filesystem path to skill directory

  ```python
  skill.uri  # "/Users/you/project/skills/my-skill"
  ```

- **For custom skills**: Any identifier (URLs, database IDs, etc.)

  ```python
  custom_skill.uri  # "https://api.example.com/skills/analyzer"
  database_skill.uri  # "db://skills/12345"
  ```

**The URI is used for:**

- Script execution working directory (for [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill])
- Error messages and logging
- [`SkillsDirectory.skills`][pydantic_ai.toolsets.skills.SkillsDirectory.skills] dictionary keys
- Identifying skills programmatically

!!! note
    For [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill], the URI is the resolved absolute path, not the original path provided to discovery.

### Creating Custom Skill Implementations

```python
from pydantic_ai.toolsets.skills import Skill, SkillMetadata
from pydantic_ai import RunContext

class RemoteSkill(Skill):
    """Skill implementation that fetches resources and runs scripts remotely."""

    async def read_resource(self, ctx: RunContext, resource_uri: str) -> str:
        # Implement remote resource fetching
        response = await http_client.get(f"{self.uri}/resources/{resource_uri}")
        return response.text

    async def run_script(self, ctx: RunContext, script_uri: str, args: list[str] | None = None) -> str:
        # Implement remote script execution
        response = await http_client.post(
            f"{self.uri}/scripts/{script_uri}",
            json={"args": args}
        )
        return response.text

# Use custom skill with toolset
remote_skill = RemoteSkill(
    name="remote-analyzer",
    uri="https://api.example.com/skills/analyzer",
    metadata=SkillMetadata(name="remote-analyzer", description="Remote data analysis"),
    content="Instructions for remote analyzer...",
)

toolset = SkillsToolset(skills=[remote_skill])
```

## Skill Discovery

The [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory] class provides low-level skill discovery from filesystem directories:

```python
from pydantic_ai.toolsets.skills import SkillsDirectory

# Create a skills directory for discovery
skills_dir = SkillsDirectory(
    path="./skills",
    validate=True,
    max_depth=3,
    script_timeout=30,  # Script execution timeout in seconds
)

# Load a specific skill by URI (filesystem path)
skill = skills_dir.load_skill("/path/to/skills/my-skill")

# Access skills dictionary (keyed by URI)
for skill_uri, skill in skills_dir.skills.items():
    print(f"Skill: {skill.name} at {skill_uri}")

# Pass to SkillsToolset
toolset = SkillsToolset(directories=[skills_dir])
```

This is useful for:

- Listing available skills before creating an agent
- Validating skill structure in tests
- Building custom skill management tools
- Generating documentation about available skills
- Fine-grained control over skill loading and script execution

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

## Configuring Script Execution

When creating a [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory], you can control how scripts are executed using the `script_executor` parameter. The parameter accepts three types of values:

### Option 1: Use Default (LocalSkillScriptExecutor)

When `script_executor` is `None` (default), skills use [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor] with default settings:

```python
# Default: local subprocess execution with 30s timeout
skills_dir = SkillsDirectory(path="./skills")
```

### Option 2: Custom LocalSkillScriptExecutor

Create a [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor] instance with custom configuration:

```python
from pydantic_ai.toolsets.skills import LocalSkillScriptExecutor, SkillsDirectory

executor = LocalSkillScriptExecutor(
    python_executable="/usr/bin/python3.11",  # Custom Python
    timeout=120  # 2 minutes
)

skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=executor
)
```

### Option 3: Callable Function (Sync or Async)

Pass any callable function - it will be automatically wrapped in [`CallableSkillScriptExecutor`][pydantic_ai.toolsets.skills.CallableSkillScriptExecutor]:

```python
from pydantic_ai.toolsets.skills import SkillsDirectory

# Async function - automatically wrapped
async def my_executor(skill, script, args=None):
    print(f"Executing {script.name} from skill {skill.name}")
    # Custom logic here
    result = await execute_with_monitoring(script.uri, args)
    return result

skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=my_executor  # Wrapped automatically
)

# Sync function - also supported
def sync_executor(skill, script, args=None):
    # Synchronous execution logic
    return execute_in_sandbox(script.uri, args)

skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=sync_executor
)
```

### Option 4: Custom SkillScriptExecutor

Implement the [`SkillScriptExecutor`][pydantic_ai.toolsets.skills.SkillScriptExecutor] protocol for full control:

```python
class DockerExecutor:
    async def run(self, skill, script, args=None):
        # Execute in Docker container
        pass

skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=DockerExecutor()
)
```

**Summary:**

- `None`: Uses [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor] with defaults
- [`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor] instance: Custom timeout and Python executable
- Callable (sync or async): Automatically wrapped in [`CallableSkillScriptExecutor`][pydantic_ai.toolsets.skills.CallableSkillScriptExecutor]
- Custom protocol implementation: Full control over execution

## Script Executors

Skills can execute Python scripts through the [`SkillScriptExecutor`][pydantic_ai.toolsets.skills.SkillScriptExecutor] protocol. The framework provides two built-in executors:

### LocalSkillScriptExecutor

[`LocalSkillScriptExecutor`][pydantic_ai.toolsets.skills.LocalSkillScriptExecutor] runs scripts using the local Python interpreter in a subprocess:

```python
from pydantic_ai.toolsets.skills import LocalSkillScriptExecutor, SkillsDirectory

# Create executor with custom timeout and Python executable
executor = LocalSkillScriptExecutor(
    python_executable="/usr/bin/python3.11",  # Optional, defaults to sys.executable
    timeout=120  # 2 minutes
)

# Use with SkillsDirectory
skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=executor
)
```

### CallableSkillScriptExecutor

[`CallableSkillScriptExecutor`][pydantic_ai.toolsets.skills.CallableSkillScriptExecutor] wraps a callable function as a script executor, supporting both sync and async functions:

```python
from pydantic_ai.toolsets.skills import SkillsDirectory

# Async executor with custom logic
async def custom_async_executor(skill, script, args=None):
    """Custom script execution with logging and monitoring."""
    print(f"Executing {script.name} from skill {skill.name}")

    # Add custom logic: logging, monitoring, sandboxing, etc.
    start_time = time.time()
    result = await execute_in_sandbox(script.uri, args)
    duration = time.time() - start_time

    logger.info(f"Script {script.name} completed in {duration:.2f}s")
    return result

# Use callable executor (automatically wrapped in CallableSkillScriptExecutor)
skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=custom_async_executor
)

# Synchronous executors are also supported
def sync_executor(skill, script, args=None):
    """Sync executor for simple scripts."""
    print(f"Running {script.name}")
    # Custom synchronous execution logic
    result = subprocess.run(
        ["python", script.uri] + (args or []),
        capture_output=True,
        text=True
    )
    return result.stdout

skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=sync_executor
)
```

### Implementing Custom SkillScriptExecutor

For advanced use cases, you can implement the [`SkillScriptExecutor`][pydantic_ai.toolsets.skills.SkillScriptExecutor] protocol directly:

```python
from pydantic_ai.toolsets.skills import SkillScriptExecutor, Skill, SkillScript

class DockerScriptExecutor:
    """Execute scripts in Docker containers for isolation."""

    def __init__(self, image: str = "python:3.11-slim", timeout: int = 30):
        self.image = image
        self.timeout = timeout

    async def run(self, skill: Skill, script: SkillScript, args: list[str] | None = None) -> str:
        """Run script in Docker container."""
        import anyio

        # Build docker command
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{skill.uri}:/skill:ro",  # Mount skill directory read-only
            self.image,
            "python", f"/skill/{Path(script.uri).name}"
        ]
        if args:
            cmd.extend(args)

        # Execute with timeout
        with anyio.move_on_after(self.timeout) as scope:
            result = await anyio.run_process(cmd, check=False)

        if scope.cancelled_caught:
            raise SkillScriptExecutionError(
                f"Script timed out after {self.timeout}s"
            )

        return result.stdout.decode("utf-8")

# Use custom executor
executor = DockerScriptExecutor(image="python:3.11-slim", timeout=60)
skills_dir = SkillsDirectory(
    path="./skills",
    script_executor=executor
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
