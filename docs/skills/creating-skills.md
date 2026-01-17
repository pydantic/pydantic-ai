# Creating Skills

This guide covers everything you need to know about creating your own Agent Skills, from basic structure to advanced patterns.

## Skill Structure

Every skill must have at minimum a `SKILL.md` file:

```markdown
my-skill/
├── SKILL.md           # Required: Instructions and metadata
├── scripts/           # Optional: Executable scripts
│   └── my_script.py
└── resources/         # Optional: Additional files
    ├── reference.md
    └── data.json
```

## SKILL.md Format

The `SKILL.md` file uses **YAML frontmatter** for metadata and **Markdown** for instructions:

````markdown {title="SKILL.md"}
---
name: arxiv-search
description: Search arXiv for research papers
version: 1.0.0
author: Your Name
tags: [papers, arxiv, academic]
---

# arXiv Search Skill

## When to Use

Use this skill when you need to:
- Find recent preprints in physics, math, or computer science
- Search for papers not yet published in journals
- Access cutting-edge research

## Instructions

To search arXiv, use the `run_skill_script` tool with:

1. **skill_name**: "arxiv-search"
2. **script_name**: "arxiv_search"
3. **args**: Your search query and options

## Example

```python
run_skill_script(
    skill_name="arxiv-search",
    script_name="arxiv_search",
    args=["machine learning", "--max-papers", "5"]
)
```
````

## Required Fields

The YAML frontmatter must include:

- `name`: Unique identifier (lowercase letters, numbers, and hyphens only)
- `description`: Brief summary (appears in skill listings, max 1024 characters)

All other fields are optional and stored in the `metadata` dictionary of the [`Skill`](../api/skills.md#pydantic_ai.toolsets.skills.Skill) object.

!!! note "Validation Behavior"
    When `validate=True` (default), skills missing the `name` field are skipped with a warning. When `validate=False`, the folder name is used as a fallback for missing `name` fields. See [Validation](#skill-validation) for details.

## Naming Conventions

Following Anthropic's skill naming conventions:

| Requirement | Example |
|------------|---------|
| Lowercase only | `arxiv-search` ✅, `ArxivSearch` ❌ |
| Hyphens for spaces | `web-research` ✅, `web_research` ❌ |
| Max 64 characters | `data-analyzer` ✅ |
| No reserved words | Avoid "anthropic" or "claude" in names |

## Resource Discovery

Resources are discovered automatically in two locations:

1. **Root-level markdown files** (except SKILL.md):
   - `FORMS.md`, `REFERENCE.md`, `GUIDE.md`, etc.
   - Referenced as: `"FORMS.md"`

2. **Files in `resources/` subdirectory** (recursive with depth limits):
   - `resources/schema.json`
   - `resources/data/sample.csv`
   - `resources/nested/file.txt`
   - Referenced with full relative path: `"resources/data/sample.csv"`

**Example structure:**

```markdown
my-skill/
├── SKILL.md
├── FORMS.md              # Referenced as "FORMS.md"
├── REFERENCE.md          # Referenced as "REFERENCE.md"
└── resources/
    ├── schema.json       # Referenced as "resources/schema.json"
    └── data/
        └── sample.csv    # Referenced as "resources/data/sample.csv"
```

**Usage in agent:**

```python
# Root-level resource
read_skill_resource("my-skill", "FORMS.md")

# Nested resource
read_skill_resource("my-skill", "resources/data/sample.csv")
```

## Adding Scripts to Skills

Scripts enable skills to perform custom operations that aren't available as standard agent tools.

### Script Location

Place scripts in either:

- `scripts/` subdirectory (recommended)
- Directly in the skill folder

```markdown
my-skill/
├── SKILL.md
└── scripts/
    ├── process_data.py
    └── fetch_info.py
```

### Writing Scripts

Scripts must accept **named arguments only** via command-line flags. Positional arguments are not supported.

**Requirements:**

- Use command-line argument parser (e.g., `argparse`) for named arguments
- Print output to stdout
- Exit with code 0 on success, non-zero on error
- Handle errors gracefully

**Example using argparse:**

```python {title="process_data.py"}
#!/usr/bin/env python3
"""Example skill script with named arguments."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='Process data')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input data to process'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='json',
        help='Output format (default: json)'
    )
    
    args = parser.parse_args()

    try:
        # Process the input
        result = args.input.upper()
        print(f'Processed: {result}')
        print(f'Format: {args.format}')
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**How arguments are passed:**

When the agent calls `run_skill_script` with:

```python
run_skill_script(
    skill_name="my-skill",
    script_name="process_data",
    args={"input": "test data", "format": "xml"}
)
```

The framework converts the dictionary to command-line arguments:

```bash
python process_data.py --input "test data" --format "xml"
```

!!! note "Argument Naming Convention"
    Dictionary keys are used exactly as provided without any conversion.
    Ensure your script's argument names match the dictionary keys you use.
    For example, `{"max-papers": 5}` becomes `--max-papers 5`.

!!! warning "Known Limitation"
    **Positional arguments are not supported.** All scripts must use named arguments (flags) only.
    This ensures consistent and predictable argument passing across different script types.

## Complete Example

Here's a complete example combining file-based skills with programmatic enhancements:

### File-Based Skill with Scripts

```markdown
skills/
└── arxiv-search/
    ├── SKILL.md
    └── scripts/
        └── arxiv_search.py
```

````markdown {title="SKILL.md"}
---
name: arxiv-search
description: Search arXiv for research papers by query
---

# arXiv Search

Search the arXiv preprint server for academic papers.

## Usage

Use `run_skill_script` with:
- **script_name**: "arxiv_search"
- **args**: {"query": "your search query", "max-papers": 5}

## Example

To find papers about transformers:

```python
run_skill_script(
    skill_name="arxiv-search",
    script_name="arxiv_search",
    args={"query": "transformers attention mechanism", "max-papers": 3}
)
```
````

```python {title="arxiv_search.py"}
#!/usr/bin/env python3
"""Search arXiv for papers."""

import argparse
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv API."""
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url) as response:
        data = response.read()

    root = ET.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    results = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()[:200]
        link = entry.find("atom:id", ns).text
        results.append({"title": title, "summary": summary, "link": link})

    return results

def main():
    parser = argparse.ArgumentParser(description='Search arXiv for research papers')
    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Search query string'
    )
    parser.add_argument(
        '--max-papers',
        type=int,
        default=5,
        help='Maximum number of papers to retrieve (default: 5)'
    )
    args = parser.parse_args()

    results = search_arxiv(args.query, args.max_papers)

    for i, paper in enumerate(results, 1):
        print(f"{i}. {paper['title']}")
        print(f"   {paper['summary']}...")
        print(f"   Link: {paper['link']}")
        print()

if __name__ == "__main__":
    main()
```

### Programmatic Skill

```python {title="programmatic_skill.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import SkillsToolset
from pydantic_ai.toolsets.skills import Skill, SkillResource
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
''',
    resources=[
        SkillResource(name='readme', content='# README\n\nSupports CSV and JSON formats.')
    ]
)

@data_skill.resource
async def get_schema(ctx: RunContext[MyDeps]) -> str:
    """Get current dataset schema."""
    schema = await ctx.deps.db.get_schema()
    return f"Schema: {schema}"

@data_skill.script
async def load_dataset(ctx: RunContext[MyDeps], path: str) -> str:
    """Load dataset from path.
    
    Args:
        path: Path to dataset file.
    """
    await ctx.deps.db.load(path)
    return f'Loaded dataset from {path}'

@data_skill.script
async def analyze(ctx: RunContext[MyDeps], metric: str) -> str:
    """Analyze dataset with specified metric.
    
    Args:
        metric: Analysis metric (mean, median, mode).
    """
    result = await ctx.deps.db.analyze(metric)
    return f'{metric}: {result}'

# Use with agent
agent = Agent(
    model='openai:gpt-4o',
    deps_type=MyDeps,
    toolsets=[SkillsToolset(
        skills=[data_skill],
        directories=['./skills']  # Also load file-based skills
    )]
)

# Run agent
deps = MyDeps(db=my_database, api_key='...')
result = await agent.run(
    'Search for papers about LLMs and analyze the results',
    deps=deps
)
```

## Programmatic Skills

You can create skills programmatically using the [`Skill`][pydantic_ai.toolsets.skills.Skill] class. Programmatic skills are ideal when you need dynamic resources or scripts that interact with your application's dependencies.

### Basic Programmatic Skill

```python
from pydantic_ai.toolsets.skills import Skill, SkillResource

# Create a skill with static content
my_skill = Skill(
    name='data-processor',
    description='Process data using various algorithms',
    content='Use this skill for data processing tasks...',
    resources=[
        SkillResource(name='algorithms', content='Available algorithms: sort, filter, transform')
    ]
)
```

### Adding Dynamic Resources

Use the `@skill.resource` decorator to add callable resources that can access dependencies:

```python
from pydantic_ai import RunContext

@my_skill.resource
def get_config() -> str:
    """Get current configuration (static)."""
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

- Resources can be sync or async functions
- Resources can optionally take `RunContext` to access dependencies (auto-detected)
- Resources can accept additional parameters (will be available as tool arguments)
- Description is inferred from docstring if not provided

### Adding Executable Scripts

Use the `@skill.script` decorator to add callable scripts:

```python
@my_skill.script
async def load_dataset(ctx: RunContext[MyDeps], path: str) -> str:
    """Load a dataset from the given path.
    
    Args:
        path: Path to the dataset file.
    """
    await ctx.deps.data_loader.load(path)
    return f'Dataset loaded from {path}'

@my_skill.script
async def run_query(ctx: RunContext[MyDeps], query: str, limit: int = 10) -> str:
    """Execute a query on the loaded dataset.
    
    Args:
        query: SQL-like query string.
        limit: Maximum number of results to return.
    """
    result = await ctx.deps.db.execute(query, limit)
    return str(result)
```

**Key points:**

- Scripts can be sync or async functions
- Scripts can optionally take `RunContext` to access dependencies (auto-detected)
- Scripts accept named arguments matching their function signature
- Description is inferred from docstring if not provided

### Using with Agent

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset

agent = Agent(
    model='openai:gpt-4o',
    toolsets=[SkillsToolset(skills=[my_skill])]
)

result = await agent.run('Load dataset from data.csv and show 3 samples')
```

### Mixing Static and Callable Resources/Scripts

You can combine static and callable resources/scripts in the same skill:

```python
skill = Skill(
    name='mixed-skill',
    description='Skill with mixed resources',
    content='Instructions here...',
    resources=[
        SkillResource(name='static-doc', content='Static documentation')
    ]
)

@skill.resource
async def dynamic_data(ctx: RunContext[MyDeps]) -> str:
    return await ctx.deps.fetch_live_data()
```

## Skill Validation

The `validate` parameter controls skill structure validation during discovery:

### Validation Enabled (Default)

```python
toolset = SkillsToolset(directories=["./skills"], validate=True)
```

**Behavior:**

- **Missing `name` field**: Skill is skipped with warning
- **Invalid name format**: Warning emitted, skill still loaded
- **Description too long** (>1024 chars): Warning emitted, skill still loaded
- **Instructions too long** (>500 lines): Warning emitted, skill still loaded
- **YAML parse errors**: [`SkillValidationError`][pydantic_ai.toolsets.skills.SkillValidationError] raised

### Validation Disabled

```python
toolset = SkillsToolset(directories=["./skills"], validate=False)
```

**Behavior:**

- **Missing `name` field**: Uses folder name as fallback
- **No format checks**: All validation warnings suppressed
- **YAML parse errors**: Still raises [`SkillValidationError`][pydantic_ai.toolsets.skills.SkillValidationError]

**Recommendation:** Keep validation enabled during development to catch issues early. Disable in production if you need more permissive loading (e.g., accepting skills without strict naming conventions).

## Best Practices

### Documentation

- Write clear, concise descriptions (they appear in skill listings)
- Include "When to Use" sections to guide agents
- Provide multiple examples showing different usage patterns
- Document all script arguments and expected output formats

### Scripts

- Keep scripts focused on a single responsibility
- Use descriptive script names (e.g., `search_papers.py` not `script1.py`)
- Include helpful error messages
- Return structured output (JSON) when possible
- Test scripts independently before adding to skills

### Resources

- Use the `resources/` directory for reference documentation
- Keep resource files small and focused
- Use clear, descriptive filenames
- Reference resources in your `SKILL.md` instructions

### Organization

- Group related skills in subdirectories
- Use consistent naming across your skills
- Version your skills in metadata for tracking
- Document dependencies in `SKILL.md`
