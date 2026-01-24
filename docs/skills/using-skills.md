# Using Skills

This guide covers integrating skills into your agents, both file-based skills from directories and programmatic skills created in code.

## Quick Start

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset

# Default: uses ./skills directory
toolset = SkillsToolset()

# Create agent with skills
agent = Agent(
    model='openai:gpt-5.2',
    toolsets=[toolset]
)

# Agent automatically has access to skill tools
result = await agent.run("Search for papers about transformers")
```

## SkillsToolset Initialization

The [`SkillsToolset`][pydantic_ai.toolsets.SkillsToolset] supports multiple initialization modes:

### Directory-based Skills

Load skills from filesystem directories:

```python
from pydantic_ai.toolsets import SkillsToolset

# Single directory
toolset = SkillsToolset(directories=["./skills"])

# Multiple directories
toolset = SkillsToolset(
    directories=["./skills", "./shared-skills"]
)

# Using SkillsDirectory instances for fine-grained control
from pydantic_ai.skills import SkillsDirectory

skills_dir = SkillsDirectory(path="./skills", validate=True)
toolset = SkillsToolset(directories=[skills_dir, "./more-skills"])
```

### Programmatic Skills

Create skills in code with full access to dependencies:

```python
from pydantic_ai.skills import Skill, SkillResource
from pydantic_ai import RunContext

my_skill = Skill(
    name="data-processor",
    description="Process data using various algorithms",
    content="Use this skill for data processing tasks..."
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
```

See [Programmatic Skills](#programmatic-skills) for details.

### Combined Mode

Mix both directory-based and programmatic skills:

```python
toolset = SkillsToolset(
    skills=[my_skill],
    directories=["./skills"]
)
```

### Configuration Options

```python
toolset = SkillsToolset(
    directories=["./skills"],
    skills=[],                    # Programmatic skills (default: [])
    validate=True,                # Validate skill structure (default: True)
    max_depth=3,                  # Max directory depth for discovery (default: 3)
    id=None,                      # Unique identifier (default: None)
    instruction_template=None,    # Custom instruction template (default: None)
    exclude_tools=None,           # Tools to exclude (default: None)
)
```

### Excluding Tools

For security or capability restrictions, you can exclude specific skill tools from being available to agents:

```python
# Disable script execution only
toolset = SkillsToolset(
    directories=["./skills"],
    exclude_tools={'run_skill_script'}
)

# Disable multiple tools
toolset = SkillsToolset(
    directories=["./skills"],
    exclude_tools={'run_skill_script', 'read_skill_resource'}
)
```

Valid tool names are:

- `list_skills`: List available skills
- `load_skill`: Load skill instructions
- `read_skill_resource`: Access skill resources (files or callables)
- `run_skill_script`: Execute skill scripts

!!! warning "Excluding load_skill"
    Excluding `load_skill` severely limits skill functionality and will emit a warning. Agents need this tool to work effectively with skills.

**Best Practice:** Only exclude tools you intentionally want to restrict. For example:

- Exclude `run_skill_script` if you want to prevent arbitrary code execution
- Exclude `read_skill_resource` if you want to limit resource access

### Default Directory Behavior

When initializing without arguments, it defaults to `./skills`:

```python
# These are equivalent
toolset = SkillsToolset()
toolset = SkillsToolset(directories=["./skills"])
```

**Important:** The default is NOT used when providing programmatic skills:

```python
# No default directory - only programmatic skills
toolset = SkillsToolset(skills=[custom_skill])

# To use both, explicitly specify directories
toolset = SkillsToolset(skills=[custom_skill], directories=["./skills"])
```

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

**Best Practice:** Use unique skill names, or intentionally use this behavior for environment-specific overrides (e.g., dev/prod variations).

## SkillsToolset API

### Key Methods

| Method | Description |
|--------|-------------|
| `get_skill(name)` | Get a specific skill by name. Raises [`SkillNotFoundError`][pydantic_ai.toolsets.skills.SkillNotFoundError] if not found |

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

## Skill Management Tools

!!! note "Model-Facing Tools"
    These tools are available to the AI model during execution. You don't call them directly - they're documented here to help you understand how skills work.

The toolset provides four tools to agents. Skills are automatically listed in the system prompt, so agents know what's available before calling any tools.

### list_skills()

Lists all available skills with descriptions. **Optional** - skills are already in the system prompt.

**Returns**: `dict[str, str]` - skill names to descriptions

### load_skill(skill_name)

Loads complete instructions for a specific skill.

**Returns**: Formatted string with:

- Skill name and description
- Available resources and scripts
- Full SKILL.md content

### read_skill_resource(skill_name, resource_name, args)

Reads resources from a skill (files or callables). Supports multiple file types including Markdown, JSON, YAML, CSV, XML, and TXT files.

**Parameters**:

- `skill_name`: Name of the skill
- `resource_name`: Resource name (e.g., "FORMS.md", "config.json", "data.csv")
- `args`: Optional arguments for callable resources

**Examples**:

```python
# Markdown resource (returned as text)
read_skill_resource(skill_name="web-research", resource_name="FORMS.md")
read_skill_resource(skill_name="web-research", resource_name="references/report-template.md")

# JSON resource (automatically parsed to dict)
read_skill_resource(skill_name="api-skill", resource_name="config.json")

# YAML resource (automatically parsed to dict)
read_skill_resource(skill_name="api-skill", resource_name="settings.yaml")

# CSV resource (returned as text)
read_skill_resource(skill_name="data-skill", resource_name="samples.csv")

# Callable resource with args
read_skill_resource(skill_name="data-skill", resource_name="get_samples", args={"count": 10})
```

### run_skill_script(skill_name, script_name, args)

Executes a script from a skill (file-based or programmatic).

**Parameters**:

- `skill_name`: Name of the skill
- `script_name`: Script name (includes .py extension)
- `args`: Named arguments as dictionary

**Examples**:

```python
# File-based script (subprocess with named args)
run_skill_script(
    skill_name="arxiv-search",
    script_name="scripts/arxiv_search.py",
    args={"query": "machine learning", "max-papers": 3}
)

# Programmatic script (direct function call)
run_skill_script(
    skill_name="data-processor",
    script_name="scripts/load_dataset.py",
    args={"path": "data.csv"}
)
```

!!! note "File-Based Script Arguments"
    For file-based scripts, arguments are passed as **named command-line arguments**:
    ```python
    args={"query": "test", "max-papers": 5}
    # Becomes: python script.py --query "test" --max-papers 5
    ```
    All file-based scripts must use named arguments. Positional arguments are not supported.

## Creating Programmatic Skills

Create skills in code when you need dynamic resources or scripts that interact with your application's dependencies.

### Basic Skill

```python
from pydantic_ai.skills import Skill, SkillResource

my_skill = Skill(
    name='data-processor',
    description='Process data using various algorithms',
    content='Use this skill for data processing tasks...',
    resources=[
        SkillResource(name='readme', content='# README\nSupports CSV and JSON.')
    ]
)
```

### Adding Dynamic Resources

Use `@skill.resource` for callable resources that can access dependencies:

```python
from pydantic_ai import RunContext

@my_skill.resource
def get_config() -> str:
    """Get configuration (static)."""
    return "Config: mode=production"

@my_skill.resource
async def get_data_schema(ctx: RunContext[MyDeps]) -> str:
    """Get data schema from database (dynamic)."""
    schema = await ctx.deps.db.get_schema()
    return f"Schema: {schema}"

@my_skill.resource
async def get_samples(ctx: RunContext[MyDeps], count: int = 5) -> str:
    """Get sample data.

    Args:
        count: Number of samples to return.
    """
    samples = await ctx.deps.db.fetch_samples(count)
    return f"Samples: {samples}"
```

**Key points:**

- Resources can be sync or async
- `RunContext` parameter is auto-detected for dependency access
- Additional parameters become tool arguments
- Description inferred from docstring

### Adding Executable Scripts

Use `@skill.script` for callable scripts:

```python
@my_skill.script
async def load_dataset(ctx: RunContext[MyDeps], path: str) -> str:
    """Load a dataset from path.

    Args:
        path: Path to the dataset file.
    """
    await ctx.deps.data_loader.load(path)
    return f'Dataset loaded from {path}'

@my_skill.script
async def run_query(ctx: RunContext[MyDeps], query: str, limit: int = 10) -> str:
    """Execute a query on the dataset.

    Args:
        query: SQL-like query string.
        limit: Maximum results to return.
    """
    result = await ctx.deps.db.execute(query, limit)
    return str(result)
```

**Key points:**

- Scripts can be sync or async
- `RunContext` parameter is auto-detected
- Arguments match function signature
- Description inferred from docstring

### Complete Example

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import SkillsToolset
from pydantic_ai.skills import Skill, SkillResource
from dataclasses import dataclass

@dataclass
class MyDeps:
    db: DatabaseConnection
    api_key: str

# Create programmatic skill
data_skill = Skill(
    name='data-analyzer',
    description='Analyze datasets with various algorithms',
    content='''# Data Analyzer

Use this skill to analyze datasets.

## Available Operations

1. Load dataset using `load_dataset` script
2. Get schema using `get_schema` resource
3. Run analysis using `analyze` script
'''
)

@data_skill.resource
async def get_schema(ctx: RunContext[MyDeps]) -> str:
    """Get current dataset schema."""
    schema = await ctx.deps.db.get_schema()
    return f"Schema: {schema}"

@data_skill.script
async def load_dataset(ctx: RunContext[MyDeps], path: str) -> str:
    """Load dataset from path."""
    await ctx.deps.db.load(path)
    return f'Loaded dataset from {path}'

@data_skill.script
async def analyze(ctx: RunContext[MyDeps], metric: str) -> str:
    """Analyze dataset with specified metric."""
    result = await ctx.deps.db.analyze(metric)
    return f'{metric}: {result}'

# Use with agent
agent = Agent(
    model='openai:gpt-5.2',
    deps_type=MyDeps,
    toolsets=[SkillsToolset(skills=[data_skill])]
)

deps = MyDeps(db=my_database, api_key='...')
result = await agent.run('Load data.csv and compute mean', deps=deps)
```

## File-Based vs Programmatic Skills

| Feature | File-Based | Programmatic |
|---------|------------|--------------|
| Resource type | Files on disk | Static strings or callables |
| Script type | Subprocess execution | Direct function calls |
| Script arguments | Named CLI arguments | Function parameters |
| Dependencies | Not available | Full `RunContext` access |
| Discovery | Automatic from directories | Manual creation |
| Execution speed | Slower (subprocess) | Faster (in-process) |

**Example comparison:**

```python
# File-based: python arxiv_search.py --query "transformers" --max-papers 5
toolset = SkillsToolset(directories=["./skills"])

# Programmatic: Direct function call with dependency access
my_skill = Skill(
    name='arxiv-search',
    description='Search arXiv for research papers',
    content='Use this skill to search arXiv...'
)

@my_skill.script
async def arxiv_search(ctx: RunContext[MyDeps], query: str, max_papers: int = 10) -> str:
    return await ctx.deps.api.search_arxiv(query, max_papers)

toolset = SkillsToolset(skills=[my_skill])
```

## Advanced Usage

### Skill URIs

Every skill has an optional `uri` field:

- **File-based**: Absolute filesystem path to skill directory
- **Programmatic**: Optional identifier (URLs, database IDs, or None)

The URI is used for error messages, logging, and as the working directory for file-based script execution.

### Skill Discovery

The [`SkillsDirectory`][pydantic_ai.skills.SkillsDirectory] class provides low-level skill discovery:

```python
from pydantic_ai.skills import SkillsDirectory, LocalSkillScriptExecutor

# Basic usage with default settings (30s timeout)
skills_dir = SkillsDirectory(
    path="./skills",
    validate=True,
    max_depth=3,
)

# With custom timeout via script_executor
executor = LocalSkillScriptExecutor(timeout=60)
skills_dir = SkillsDirectory(
    path="./skills",
    validate=True,
    max_depth=3,
    script_executor=executor,
)

# Load specific skill
skill = skills_dir.load_skill("/path/to/skills/my-skill")

# Access all skills (keyed by URI)
for skill_uri, skill in skills_dir.skills.items():
    print(f"{skill.name} at {skill_uri}")

# Pass to toolset
toolset = SkillsToolset(directories=[skills_dir])
```

Useful for:

- Listing skills before creating agents
- Validating skill structure in tests
- Building custom skill management tools
- Generating documentation

### Skill Validation

Control skill structure validation with the `validate` parameter:

**Enabled (default)**:

```python
toolset = SkillsToolset(directories=["./skills"], validate=True)
```

- Missing `name`: Skill skipped with warning
- Invalid name format: Warning emitted, skill still loaded
- Description >1024 chars: Warning emitted, skill still loaded
- Instructions >500 lines: Warning emitted, skill still loaded
- YAML parse errors: [`SkillValidationError`][pydantic_ai.skills.SkillValidationError] raised

**Disabled**:

```python
toolset = SkillsToolset(directories=["./skills"], validate=False)
```

- Missing `name`: Uses folder name as fallback
- No format checks: All validation warnings suppressed
- YAML parse errors: Still raises [`SkillValidationError`][pydantic_ai.skills.SkillValidationError]

**Recommendation:** Keep enabled during development, disable in production if needed.

### Script Executors

Control how file-based scripts are executed using the `script_executor` parameter with [`SkillsDirectory`][pydantic_ai.skills.SkillsDirectory]:

**Option 1: Default ([`LocalSkillScriptExecutor`][pydantic_ai.skills.LocalSkillScriptExecutor])**:

```python
skills_dir = SkillsDirectory(path="./skills")  # 30s timeout, default Python
```

**Option 2: Custom LocalSkillScriptExecutor**:

```python
from pydantic_ai.skills import LocalSkillScriptExecutor

executor = LocalSkillScriptExecutor(
    python_executable="/usr/bin/python3.11",
    timeout=120
)
skills_dir = SkillsDirectory(path="./skills", script_executor=executor)
```

**Option 3: Callable Function** (auto-wrapped in [`CallableSkillScriptExecutor`][pydantic_ai.skills.CallableSkillScriptExecutor]):

```python
from pydantic_ai.skills import SkillScript

async def my_executor(script: SkillScript, args=None):
    print(f"Executing {script.name} from {skill.name}")
    result = await execute_with_monitoring(script.uri, args)
    return result

skills_dir = SkillsDirectory(path="./skills", script_executor=my_executor)

# Sync functions also supported
def sync_executor(script: SkillScript, args=None):
    return execute_in_sandbox(script.uri, args)
```

**Option 4: Custom Protocol**:

```python
from pydantic_ai.skills import SkillScriptExecutor, SkillScript

class DockerExecutor:
    async def run(self, script: SkillScript, args: dict[str, Any] | None = None) -> str:
        # Execute in Docker container
        pass

skills_dir = SkillsDirectory(path="./skills", script_executor=DockerExecutor())
```

## Error Handling

The toolset raises specific exceptions:

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

## Security

!!! warning "Use Skills from Trusted Sources Only"

    Skills provide agents with instructions and executable code. Use only skills from trusted sources you control or thoroughly audit. Malicious skills could misuse agent capabilities or execute harmful code depending on what access agents have.

Best practices:

- Only load skills from trusted sources
- Review skill content and scripts before use
- Use custom script executors for additional sandboxing
- Limit agent permissions and access
- Monitor skill execution in production
