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

## Naming Conventions

Following Anthropic's skill naming conventions:

| Requirement | Example |
|------------|---------|
| Lowercase only | `arxiv-search` ✅, `ArxivSearch` ❌ |
| Hyphens for spaces | `web-research` ✅, `web_research` ❌ |
| Max 64 characters | `data-analyzer` ✅ |
| No reserved words | Avoid "anthropic" or "claude" in names |

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
from pydantic_ai import Agent, SkillsToolset

async def main():
    skills_toolset = SkillsToolset(directories=["./skills"])

    agent = Agent(
        model='openai:gpt-4o',
        instructions="You are a research assistant.",
        toolsets=[skills_toolset]
    )
    # Skills instructions are automatically injected

    result = await agent.run(
        "Find the 3 most recent papers about large language models"
    )
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

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
