---
name: arxiv-search
description: Search arXiv preprint repository for papers in physics, mathematics, computer science, quantitative biology, and related fields.
compatibility: Requires arxiv installed in the Python environment and access to the internet
---

# arXiv Search Skill

This skill provides access to arXiv, a free distribution service and open-access archive for scholarly articles in physics, mathematics, computer science, quantitative biology, quantitative finance, statistics, electrical engineering, systems science, and economics.

## When to Use This Skill

Use this skill when you need to:

- Find preprints and recent research papers before journal publication
- Search for papers in computational biology, bioinformatics, or systems biology
- Access mathematical or statistical methods papers relevant to biology
- Find machine learning papers applied to biological problems
- Get the latest research that may not yet be in PubMed

## Skill Scripts

### arxiv_search

The `arxiv_search` script accepts the following arguments:

- `query` (required): Search query string (e.g., "neural networks protein structure", "single cell RNA-seq")
- `max-papers` (optional): Maximum number of papers to retrieve (default: 10)

### Usage Examples

**Search for computational biology papers (default 10 results):**

- query: "protein folding prediction"

**Search for machine learning papers with limited results:**

- query: "transformer attention mechanism"
- max-papers: 3

**General search with custom limit:**

- query: "your search query"
- max-papers: 5

## Output Format

The script returns formatted results with:

- Paper title
- Summary/abstract
- arXiv URL

## Dependencies

This script requires the `arxiv` package, which should be installed in your environment.
