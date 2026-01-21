# Creating Skills

This guide covers creating file-based Agent Skills - modular directories that agents can discover and load from the filesystem. For programmatic skills created in code, see [Using Skills - Programmatic Skills](using-skills.md#programmatic-skills).

## Skill Structure

Every skill is a directory with at minimum a `SKILL.md` file. For the complete specification, see [Agent Skills Specification](https://agentskills.io/specification).

```markdown
my-skill/
├── SKILL.md           # Required: Instructions and metadata
├── scripts/           # Optional: Executable scripts
│   └── my_script.py
├── references/        # Optional: Additional documentation
│   └── REFERENCE.md
└── assets/            # Optional: Static resources
    └── data.json
```

## SKILL.md Format

The `SKILL.md` file uses **YAML frontmatter** for metadata and **Markdown** for instructions:

````markdown {title="SKILL.md"}
---
name: arxiv-search
description: Search arXiv for research papers
license: MIT
compatibility: Requires network access to arxiv.org API and arxiv package
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

To search arXiv, use the `arxiv_search.py` script:

**Arguments:**
- `query` (required): Your search query
- `max-papers` (optional): Maximum number of papers to return (default: 5)

**Example:**

Search for papers about machine learning with 5 results.
````

## Required Fields

The YAML frontmatter must include:

- `name`: Unique identifier (lowercase letters, numbers, and hyphens only)
- `description`: Brief summary (appears in skill listings, max 1024 characters)

## Optional Standard Fields

The following optional fields are defined by the [Agent Skills specification](https://agentskills.io/specification):

- `license`: License information (e.g., "MIT", "Apache-2.0", or "Proprietary. See LICENSE.txt")
- `compatibility`: Environment requirements (max 500 characters, e.g., "Requires git, docker, and internet access")

All other fields are optional and stored in the `metadata` dictionary of the [`Skill`](../api/skills.md#pydantic_ai.skills.Skill) object.

!!! note "Validation Behavior"
    When `validate=True` (default), skills missing the `name` field are skipped with a warning. When `validate=False`, the folder name is used as a fallback for missing `name` fields. See [Validation](#validation) for details.

The [Agent Skills specification](https://agentskills.io/specification#name-field) defines the following requirements for skill names:

| Requirement             | Example                                    |
|-------------------------|--------------------------------------------|
| Lowercase only          | `arxiv-search` ✅, `ArxivSearch` ❌        |
| Hyphens for spaces      | `web-research` ✅, `web_research` ❌       |
| No consecutive hyphens  | `arxiv-search` ✅, `arxiv--search` ❌      |
| Max 64 characters       | `data-analyzer` ✅                         |
| No reserved words       | Avoid "anthropic" or "claude" in names     |

## Resource Discovery

Resources are discovered automatically. For the complete specification, see [Optional directories](https://agentskills.io/specification#optional-directories).

1. **Root-level markdown files** (except SKILL.md):
   - `REFERENCE.md`, `FORMS.md`, `GUIDE.md`, etc.
   - Referenced as: `"REFERENCE.md"`

2. **Files in optional subdirectories** (`references/`, `assets/`, recursive with depth limits):
   - `references/technical.md`
   - `assets/schema.json`
   - `assets/data/sample.csv`
   - Referenced with full relative path: `"references/technical.md"`

**Example structure:**

```markdown
my-skill/
├── SKILL.md
├── FORMS.md              # Referenced as "FORMS.md"
├── REFERENCE.md          # Referenced as "REFERENCE.md"
├── references/
│   └── technical.md      # Referenced as "references/technical.md"
└── assets/
    ├── schema.json       # Referenced as "assets/schema.json"
    └── data/
        └── sample.csv    # Referenced as "assets/data/sample.csv"
```

**Usage:**

Resources can be accessed by their path:

- Root-level: `FORMS.md`
- Nested: `assets/data/sample.csv`

## Adding Scripts to Skills

Scripts enable skills to perform custom operations that aren't available as standard agent tools.

### Script Location

All scripts must be placed in the `scripts/` subdirectory within your skill folder:

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

**Script execution:**

Scripts accept named arguments. For example, the `process_data` script accepts:

- `input`: Input data to process
- `format`: Output format

!!! note "Argument Naming Convention"
    Dictionary keys are used exactly as provided without any conversion.
    Ensure your script's argument names match the dictionary keys you use.
    For example, `{"max-papers": 5}` becomes `--max-papers 5`.

!!! warning "Known Limitation"
    **Positional arguments are not supported.** All scripts must use named arguments (flags) only.
    This ensures consistent and predictable argument passing across different script types.

## Complete Example

Here's a complete file-based skill with a script:

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

Use the `arxiv_search.py` script:

**Arguments:**
- `query` (required): Your search query
- `max-papers` (optional): Maximum number of results (default: 5)

**Example:**

Find papers about transformers (limited to 3 results):
- query: "transformers attention mechanism"
- max-papers: 3
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

## Best Practices

### Documentation

- Write clear, concise descriptions (appear in skill listings, max 1024 characters)
- Include "When to Use" sections to guide agents
- Provide multiple examples showing different usage patterns
- Document all script arguments and expected output formats

### Scripts

- Keep scripts focused on a single responsibility
- Use descriptive names (e.g., `search_papers.py` not `script1.py`)
- Include helpful error messages
- Return structured output (JSON) when possible
- Test scripts independently before adding to skills

### Resources

- Use `references/` for reference documentation
- Use `assets/` for data files and static content
- Keep resource files small and focused
- Use clear, descriptive filenames
- Reference resources in your `SKILL.md` instructions

### Organization

- Group related skills in subdirectories
- Use consistent naming across your skills
- Version your skills in metadata for tracking
- Document dependencies in `SKILL.md`

## Validation

By default, skills are validated during discovery (`validate=True`). See [Using Skills - Skill Validation](using-skills.md#skill-validation) for details on validation behavior and how to configure it.
