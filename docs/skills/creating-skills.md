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

All other fields are optional and stored in the `extra` dictionary of [`SkillMetadata`](../api/skills.md#pydantic_ai.toolsets.skills.SkillMetadata).

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

Scripts should:

- Accept command-line arguments via `sys.argv`
- Print output to stdout
- Exit with code 0 on success, non-zero on error
- Handle errors gracefully

```python {title="process_data.py"}
#!/usr/bin/env python3
"""Example skill script."""

import sys
import json

def main():
    if len(sys.argv) < 2:
        print("Usage: process_data.py <input>")
        sys.exit(1)

    input_data = sys.argv[1]

    try:
        # Process the input
        result = {"processed": input_data.upper()}
        print(json.dumps(result, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## Complete Example

Here's a complete example with a skill that searches for research papers:

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
- **args**: ["your search query", "--max-papers", "5"]

## Example

To find papers about transformers:

```python
run_skill_script(
    skill_name="arxiv-search",
    script_name="arxiv_search",
    args=["transformers attention mechanism", "--max-papers", "3"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-papers", type=int, default=5)
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

```python {title="agent_example.py"}
import asyncio
from pydantic_ai import Agent
from pydantic_ai.toolsets import SkillsToolset

async def main():
    # Initialize Skills Toolset
    skills_toolset = SkillsToolset(directories=["./skills"])

    # Create agent with skills
    agent = Agent(
        model='openai:gpt-4o',
        instructions="You are a research assistant.",
        toolsets=[skills_toolset]
    )
    # Skills instructions are automatically injected via get_instructions()

    # Run agent - skills tools are automatically available
    result = await agent.run(
        "Find the 3 most recent papers about large language models"
    )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

## Understanding Skill Types

The skills framework uses an abstract base class pattern with two main classes:

- **[`Skill`][pydantic_ai.toolsets.skills.Skill]** (abstract base class): Defines the interface all skills must implement
  - Cannot be instantiated directly
  - Used for type hints and custom implementations
  - Subclasses must implement `read_resource()` and `run_script()` methods

- **[`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill]** (concrete implementation): Filesystem-based skills
  - Used by automatic directory-based discovery
  - Used for programmatic skills with filesystem access
  - Implements resource reading and script execution for local files

**When creating skills programmatically, use `LocalSkill`** (or create your own `Skill` subclass for custom backends):

```python
from pydantic_ai.toolsets.skills import LocalSkill  # ✅ Use this

# Correct:
skill = LocalSkill(name="my-skill", ...)

# Wrong (will fail):
from pydantic_ai.toolsets.skills import Skill
skill = Skill(name="my-skill", ...)  # ❌ Error: can't instantiate abstract class
```

## Programmatic Skills

You can create skills programmatically using [`LocalSkill`][pydantic_ai.toolsets.skills.LocalSkill]:

```python
from pydantic_ai.toolsets.skills import LocalSkill, SkillMetadata, SkillResource, SkillScript
from pydantic_ai.toolsets import SkillsToolset
from pathlib import Path

# Create skill with full control
skill = LocalSkill(
    name="custom-analyzer",
    uri=str(Path("./custom-skills/analyzer").resolve()),
    metadata=SkillMetadata(
        name="custom-analyzer",
        description="Custom data analysis skill",
        extra={"version": "2.0.0", "author": "Your Name"}
    ),
    content="""# Custom Analyzer

## When to Use
Use this skill for advanced data analysis tasks.

## Instructions
1. Load data using the load_data script
2. Process with analyze_data script
3. Export results
""",
    resources=[
        SkillResource(
            name="REFERENCE.md",
            uri=str(Path("./custom-skills/analyzer/REFERENCE.md").resolve())
        )
    ],
    scripts=[
        SkillScript(
            name="load_data",
            uri=str(Path("./custom-skills/analyzer/scripts/load_data.py").resolve()),
            skill_name="custom-analyzer"
        ),
        SkillScript(
            name="analyze_data",
            uri=str(Path("./custom-skills/analyzer/scripts/analyze_data.py").resolve()),
            skill_name="custom-analyzer"
        )
    ],
    # Optional: provide custom script executor
    script_executor=None  # Uses LocalSkillScriptExecutor by default
)

# Use with toolset
toolset = SkillsToolset(skills=[skill])
```

### Custom Skill Implementations

For advanced scenarios, you can create custom [`Skill`][pydantic_ai.toolsets.skills.Skill] subclasses:

```python
from pydantic_ai.toolsets.skills import Skill, SkillMetadata
from pydantic_ai import RunContext
import httpx

class APISkill(Skill):
    """Skill that fetches resources and executes scripts via API."""

    def __init__(self, api_base_url: str, **kwargs):
        super().__init__(**kwargs)
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient()

    async def read_resource(self, ctx: RunContext, resource_uri: str) -> str:
        """Fetch resource from API."""
        response = await self.client.get(
            f"{self.api_base_url}/resources/{resource_uri}"
        )
        response.raise_for_status()
        return response.text

    async def run_script(self, ctx: RunContext, script_uri: str, args: list[str] | None = None) -> str:
        """Execute script via API."""
        response = await self.client.post(
            f"{self.api_base_url}/scripts/{script_uri}",
            json={"args": args or []}
        )
        response.raise_for_status()
        return response.text

# Create API-based skill
api_skill = APISkill(
    name="remote-processor",
    uri="https://api.example.com/skills/processor",
    api_base_url="https://api.example.com/skills/processor",
    metadata=SkillMetadata(
        name="remote-processor",
        description="Remote data processing via API"
    ),
    content="Instructions for using the remote processor..."
)

toolset = SkillsToolset(skills=[api_skill])
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
