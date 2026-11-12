"""Classify a pull request with OpenCode when the Claude subscription refused `Category Classify`."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

from douwebot_opencode_result import final_text, rejection_reason
from douwebot_opencode_run import run_read_only
from pydantic import BaseModel, ConfigDict

# Appended to the Claude prompt, which tells the model to run `gh`. The `Prefetch PR context` step in `bots.yml`
# writes these files.
TOOL_GUIDANCE = """

You have read, grep, and glob tools only. Shell and the GitHub CLI are unavailable, so run no command.
The working directory holds the output of each `gh` command above:
- `pr-view.txt` holds `gh pr view`.
- `changed-files.txt` holds `gh pr diff --name-only`.
- `pr.diff` holds the full `gh pr diff`. It exists only when a `pyproject.toml` file changed.

Return only the JSON object, with no other text.
"""


class Classification(BaseModel):
    """The answer `category-apply` accepts: one allowed category and nothing else."""

    model_config = ConfigDict(extra='forbid', strict=True)

    category: Literal['bug', 'feature', 'docs', 'chore', 'dependency']


def main() -> None:
    """Classify from the prefetched context and write the validated category as a job output."""
    runner_temp = Path(os.environ['RUNNER_TEMP'])
    stream = runner_temp / 'category-classify-opencode.ndjson'
    run_read_only(
        os.environ['CLASSIFY_PROMPT'] + TOOL_GUIDANCE,
        cwd=runner_temp / 'category-context',
        agent='classifier',
        output=stream,
    )
    try:
        classification = Classification.model_validate_json(final_text(stream))
    except ValueError as exc:
        sys.exit(f'OpenCode returned an invalid category: {rejection_reason(exc)}')
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
        output.write(f'category={classification.category}\n')


if __name__ == '__main__':
    main()
