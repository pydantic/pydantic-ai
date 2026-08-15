"""Keep the front pages in sync: docs/index.md and README.md tell one story on two surfaces.

See the "Front pages" section of docs/AGENTS.md for the full contract.
"""

from __future__ import annotations as _annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Each marker identifies one example that must stay code-identical between the
# docs index and the README (comments and annotation markers excluded).
MIRRORED_EXAMPLE_MARKERS = [
    "Advisor('openai:gpt-5.6-sol')",
    'ClearToolResults(), WarnNearLimits(), ToolOutputLimits()',
    'class Sentiment(BaseModel):',
    'class ResearchWorkflow(PydanticAIWorkflow):',
    'output_type=BinaryImage',
    "Embedder('openai:text-embedding-3-small')",
    "agent.realtime('openai:gpt-realtime-2.1')",
    'class SupportDependencies:',
]

FRONT_PAGES = [ROOT / 'docs' / 'index.md', ROOT / 'README.md', ROOT / 'docs' / 'interfaces.md']


def _python_blocks(text: str) -> list[str]:
    """Extract python fence bodies, dedenting tab-indented docs blocks."""
    blocks: list[str] = []
    for m in re.finditer(r'^( *)```python[^`\n]*\n(.*?)\n\1```', text, re.S | re.M):
        indent, body = m.group(1), m.group(2)
        if indent:
            body = '\n'.join(line[len(indent) :] if line.strip() else '' for line in body.splitlines())
        blocks.append(body)
    return blocks


def _normalize(block: str) -> str:
    """Strip comments and annotation markers so annotated and plain-comment variants compare on code alone."""
    lines: list[str] = []
    for line in block.splitlines():
        line = re.sub(r'\s*#.*$', '', line).rstrip()
        lines.append(line)
    return '\n'.join(line for line in lines if line).strip()


def test_mirrored_examples_are_code_identical():
    index_blocks = [_normalize(b) for b in _python_blocks((ROOT / 'docs' / 'index.md').read_text())]
    readme_blocks = [_normalize(b) for b in _python_blocks((ROOT / 'README.md').read_text())]
    for marker in MIRRORED_EXAMPLE_MARKERS:
        index_matches = [b for b in index_blocks if marker in b]
        readme_matches = [b for b in readme_blocks if marker in b]
        assert index_matches, f'mirrored example missing from docs/index.md: {marker!r}'
        assert readme_matches, f'mirrored example missing from README.md: {marker!r}'
        assert index_matches[0] == readme_matches[0], (
            f'front-page example diverged between docs/index.md and README.md: {marker!r}'
        )


def test_front_pages_have_no_em_dashes():
    for path in FRONT_PAGES:
        assert '—' not in path.read_text(), f'em dash found in {path.relative_to(ROOT)}'
