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

# Programmatic skills with decorators
from pydantic_ai.toolsets.skills import Skill, SkillResource
from pydantic_ai import RunContext

my_skill = Skill(
    name="my-skill",
    description="Custom skill",
    content="Instructions here",
    resources=[
        SkillResource(name="readme", content="Static readme")
    ]
)

@my_skill.resource
def get_config() -> str:
    """Get configuration."""
    return "Config data"

@my_skill.script
async def process(ctx: RunContext[MyDeps], data: str) -> str:
    """Process data."""
    return f"Processed: {data}"

toolset = SkillsToolset(skills=[my_skill])

# Combined mode: both programmatic and directory-based
toolset = SkillsToolset(
    skills=[my_skill],
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

### 3. `read_skill_resource(skill_name, resource_name, args)`

Reads additional resources from a skill (files or callable resources).

**Parameters**:

- `skill_name` (str) - Name of the skill
- `resource_name` (str) - Resource name (e.g., "FORMS.md" or "get_schema")
- `args` (dict[str, Any] | None, optional) - Named arguments for callable resources

**Returns**: Content of the resource (can be str, dict, or any type returned by the resource)

**When to use**: When a skill references additional documentation, data files, or dynamic resources

**Example**:

```python
# File-based resource
read_skill_resource("web-research", "FORMS.md")

# Callable resource (no args)
read_skill_resource("data-skill", "get_config")

# Callable resource with args
read_skill_resource("data-skill", "get_samples", args={"count": 10})
```

### 4. `run_skill_script(skill_name, script_name, args)`

Executes a Python script from a skill (file-based or programmatic).

**Parameters**:

- `skill_name` (str) - Name of the skill
- `script_name` (str) - Script name without .py extension
- `args` (dict[str, Any] | None, optional) - Named arguments as dictionary matching the script's parameter schema

**Returns**: Script output (can be str, dict, or any JSON-serializable type)

**When to use**: When a skill needs to execute custom code

**Example**:

```python
# File-based script (subprocess execution with named arguments)
run_skill_script(
    skill_name="arxiv-search",
    script_name="arxiv_search",
    args={"query": "machine learning", "max-papers": 3}
)

# Programmatic script (direct function call)
run_skill_script(
    skill_name="data-processor",
    script_name="load_dataset",
    args={"path": "data.csv"}
)
```

!!! note "Argument Format for File-Based Scripts"
    For file-based scripts, arguments are passed as **named command-line arguments**.
    Dictionary keys are used exactly as provided. For example:
    ```python
    args={"query": "test", "max-papers": 5}
    ```
    is converted to:
    ```bash
    python script.py --query "test" --max-papers 5
    ```
    
    **Important:** All file-based scripts must use named arguments (argparse or similar).
    Positional arguments are not supported.

## Skill Types

Skills can be created in two ways:

### File-Based Skills

Discovered from filesystem directories using [`SkillsDirectory`][pydantic_ai.toolsets.skills.SkillsDirectory]:

- Resources are loaded from markdown files
- Scripts are Python files executed via subprocess (with JSON stdin)
- Automatically discovered from directory structure

### Programmatic Skills

Created using the [`Skill`][pydantic_ai.toolsets.skills.Skill] class:

- Resources can be static strings or callable functions
- Scripts are Python functions (sync or async)
- Full access to `RunContext` and dependencies

**Key differences:**

| Feature | File-Based | Programmatic |
|---------|------------|--------------|
| Resource type | Files on disk | Static strings or callables |
| Script type | Subprocess execution | Direct function calls |
| Script arguments | Named CLI arguments (e.g., `--query value`) | Function parameters |
| Dependencies | Not available | Full `RunContext` access |
| Discovery | Automatic | Manual creation |
| Execution | Slower (subprocess) | Faster (in-process) |

**Example comparison:**

```python
# File-based skill
# Script executed as: python arxiv_search.py --query "transformers" --max-papers 5
toolset = SkillsToolset(directories=["./skills"])

# Programmatic skill
@my_skill.script
async def arxiv_search(ctx: RunContext[MyDeps], query: str, max_papers: int = 10) -> str:
    # Direct function call with dependency access
    return await ctx.deps.api.search_arxiv(query, max_papers)

toolset = SkillsToolset(skills=[my_skill])
```

### Skill URI

Every skill has an optional `uri` field that identifies its location:

- **For file-based skills**: Absolute filesystem path to skill directory
- **For programmatic skills**: Optional identifier (URLs, database IDs, or None)

```python
# File-based skill
file_skill.uri  # "/Users/you/project/skills/my-skill"

# Programmatic skill (optional)
prog_skill = Skill(name="processor", description="...", content="...", uri=None)
```

**The URI is used for:**

- Error messages and logging
- Skill identification in toolset
- Working directory for file-based script execution

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
