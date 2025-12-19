# Skills

A standardized, composable framework for building and managing Agent Skills. Skills are modular collections of instructions, scripts, tools, and resources that enable AI agents to progressively discover, load, and execute specialized capabilities for domain-specific tasks.

## What are Agent Skills?

Agent Skills are **modular packages** that extend your agent's capabilities without hardcoding every possible feature into your agent's instructions. Think of them as plugins that agents can discover and load on-demand.

Key benefits:

- **🔍 Progressive Discovery**: Agents list available skills and load only what they need
- **📦 Modular Design**: Each skill is a self-contained directory with instructions and resources
- **🛠️ Script Execution**: Skills can include executable Python scripts
- **📚 Resource Management**: Support for additional documentation and data files
- **🚀 Easy Integration**: Simple toolset interface that works with any Pydantic AI agent

## Quick Example

```python
from pydantic_ai import Agent, SkillsToolset

# Initialize Skills Toolset with skill directories
skills_toolset = SkillsToolset(directories=["./skills"])

# Create agent with skills
agent = Agent(
    model='openai:gpt-4o',
    instructions="You are a helpful research assistant.",
    toolsets=[skills_toolset]
)

# Add skills system prompt
@agent.system_prompt
async def add_skills_to_system_prompt() -> str:
    return skills_toolset.get_skills_system_prompt()

# Use agent - skills tools are automatically available
result = await agent.run(
    "What are the last 3 papers on arXiv about machine learning?"
)
print(result.output)
```

!!! note "Alternative Import"
    You can also import `SkillsToolset` from `pydantic_ai.toolsets`:
    ```python
    from pydantic_ai.toolsets import SkillsToolset
    ```

## How It Works

1. **Discovery**: The toolset scans specified directories for skills (folders with `SKILL.md` files)
2. **Registration**: Skills are registered as tools on your agent
3. **Progressive Loading**: Agents can:
   - List all available skills with `list_skills()` (optional, as skills are in system prompt)
   - Load detailed instructions with `load_skill(name)`
   - Read additional resources with `read_skill_resource(skill_name, resource_name)`
   - Execute scripts with `run_skill_script(skill_name, script_name, args)`

## Creating Skills

### Basic Skill Structure

Every skill must have at minimum a `SKILL.md` file:

```markdown
my-skill/
├── SKILL.md # Required: Instructions and metadata
├── scripts/ # Optional: Executable scripts
│ └── my_script.py
└── resources/ # Optional: Additional files
├── reference.md
└── data.json
```

### SKILL.md Format

The `SKILL.md` file uses **YAML frontmatter** for metadata and **Markdown** for instructions:

````markdown
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

### Required Fields

- `name`: Unique identifier (lowercase letters, numbers, and hyphens only)
- `description`: Brief summary (appears in skill listings, max 1024 characters)

### Naming Conventions

Following Anthropic's skill naming conventions:

| Requirement        | Example                                |
| ------------------ | -------------------------------------- |
| Lowercase only     | `arxiv-search` ✅, `ArxivSearch` ❌    |
| Hyphens for spaces | `web-research` ✅, `web_research` ❌   |
| Max 64 characters  | `data-analyzer` ✅                     |
| No reserved words  | Avoid "anthropic" or "claude" in names |

## Progressive Disclosure

The toolset implements **progressive disclosure** - exposing information only when needed:

```markdown
┌─────────────────────────────────────────────────────────────┐
│  System Prompt (via get_skills_system_prompt())             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Available Skills:                                     │  │
│  │ - arxiv-search: Search arXiv for research papers      │  │
│  │ - web-research: Research topics on the web            │  │
│  │ - data-analyzer: Analyze CSV and JSON files           │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
          Agent sees skill names & descriptions
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  load_skill("arxiv-search")                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Returns full SKILL.md instructions:                   │  │
│  │ - When to use                                         │  │
│  │ - Step-by-step guide                                  │  │
│  │ - Example invocations                                 │  │
│  │ - Available resources and scripts                     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
      Agent loads detailed instructions when needed
```

This approach:

- **Reduces initial context size** - Only metadata is in the system prompt
- **Lets agents discover capabilities dynamically** - Load what's needed
- **Improves token efficiency** - Don't pay for unused instructions
- **Scales to many skills** - Add hundreds of skills without bloating prompts

## The Four Tools

The `SkillsToolset` provides four tools to agents:

### 1. `list_skills()`

Lists all available skills with their descriptions.

**Returns**: Formatted markdown with skill names and descriptions

**When to use**: Optional - skills are already listed in the system prompt via `get_skills_system_prompt()`. Use only if the agent needs to re-check available skills dynamically.

### 2. `load_skill(skill_name)`

Loads the complete instructions for a specific skill.

**Parameters**:

- `skill_name` (str) - Name of the skill to load

**Returns**: Full SKILL.md content including detailed instructions, available resources, and scripts

**When to use**: When the agent needs detailed instructions for using a skill

### 3. `read_skill_resource(skill_name, resource_name)`

Reads additional resource files from a skill.

**Parameters**:

- `skill_name` (str) - Name of the skill
- `resource_name` (str) - Resource filename (e.g., "FORMS.md")

**Returns**: Content of the resource file

**When to use**: When a skill references additional documentation or data files

### 4. `run_skill_script(skill_name, script_name, args)`

Executes a Python script from a skill.

**Parameters**:

- `skill_name` (str) - Name of the skill
- `script_name` (str) - Script name without .py extension
- `args` (list[str], optional) - Command-line arguments

**Returns**: Script output (stdout and stderr combined)

**When to use**: When a skill needs to execute custom code

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

```python
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

## SkillsToolset API

### Initialization

```python
from pydantic_ai.toolsets import SkillsToolset

toolset = SkillsToolset(
    directories=["./skills", "./shared-skills"],
    auto_discover=True,      # Auto-discover skills on init (default: True)
    validate=True,           # Validate skill structure (default: True)
    id="skills",             # Unique identifier (default: "skills")
    script_timeout=30,       # Script execution timeout in seconds (default: 30)
    python_executable=None,  # Python executable path (default: sys.executable)
)
```

### Key Methods

| Method                       | Description                                    |
| ---------------------------- | ---------------------------------------------- |
| `get_skills_system_prompt()` | Get system prompt text with all skill metadata |
| `get_skill(name)`            | Get a specific skill object by name            |
| `refresh()`                  | Re-scan directories for skills                 |

### Properties

| Property | Description                                      |
| -------- | ------------------------------------------------ |
| `skills` | Dictionary of loaded skills (`dict[str, Skill]`) |

## Skill Discovery

Skills can be discovered programmatically:

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

## Type Reference

### Skill

```python
from pydantic_ai import SkillsToolset
from pydantic_ai.toolsets.skills import Skill

skill = Skill(
    name="my-skill",
    path=Path("./skills/my-skill"),
    metadata=SkillMetadata(...),
    content="# Instructions...",
    resources=[SkillResource(...)],
    scripts=[SkillScript(...)],
)
```

### SkillMetadata

```python
from pydantic_ai.toolsets.skills import SkillMetadata

metadata = SkillMetadata(
    name="my-skill",
    description="My skill description",
    extra={"version": "1.0.0", "author": "Me"}
)
```

### SkillResource

```python
from pydantic_ai.toolsets.skills import SkillResource

resource = SkillResource(
    name="FORMS.md",
    path=Path("./skills/my-skill/FORMS.md"),
    content=None,  # Lazy-loaded
)
```

### SkillScript

```python
from pydantic_ai.toolsets.skills import SkillScript

script = SkillScript(
    name="process_data",
    path=Path("./skills/my-skill/scripts/process_data.py"),
    skill_name="my-skill",
)
```

## Security Considerations

!!! warning "Use Skills from Trusted Sources Only"

    Skills provide AI agents with new capabilities through instructions and code. While this makes them powerful, it also means a malicious skill can direct agents to invoke tools or execute code in ways that don't match the skill's stated purpose.

    If you must use a skill from an untrusted or unknown source, exercise extreme caution and thoroughly audit it before use. Depending on what access agents have when executing the skill, malicious skills could lead to data exfiltration, unauthorized system access, or other security risks.

The toolset includes security measures:

- **Path traversal prevention**: Resources and scripts are validated to stay within the skill directory
- **Script timeout**: Scripts have a configurable timeout (default: 30 seconds)
- **Sandboxed execution**: Scripts run in a subprocess with limited access

## Complete Example

Here's a complete example with a skill that searches for research papers:

### Skill Structure

```markdown
skills/
└── arxiv-search/
    ├── SKILL.md
    └── scripts/
        └── arxiv_search.py
```

### SKILL.md

````markdown
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

### arxiv_search.py

```python
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

### Agent Code

```python
import asyncio
from pydantic_ai import Agent, SkillsToolset

async def main():
    skills_toolset = SkillsToolset(directories=["./skills"])

    agent = Agent(
        model='openai:gpt-4o',
        instructions="You are a research assistant.",
        toolsets=[skills_toolset]
    )

    @agent.system_prompt
    async def add_skills():
        return skills_toolset.get_skills_system_prompt()

    result = await agent.run(
        "Find the 3 most recent papers about large language models"
    )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

## References

This implementation is inspired by:

- [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents/tree/master)
- [vstorm-co/pydantic-deepagents](https://github.com/vstorm-co/pydantic-deepagents/tree/main)
- [Introducing Agent Skills | Anthropic](https://www.anthropic.com/news/agent-skills)
- [Using skills with Deep Agents | LangChain](https://blog.langchain.com/using-skills-with-deep-agents/)
