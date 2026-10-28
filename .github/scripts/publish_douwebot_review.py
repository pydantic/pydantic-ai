"""Publish structured douwebot findings from a trusted host, with a deterministic receipt."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Literal

from pydantic import BaseModel, Field


class Finding(BaseModel):
    """One inline finding; GitHub validates the position against the pinned diff."""

    path: str
    line: int = Field(gt=0)
    side: Literal['LEFT', 'RIGHT']
    body: str = Field(min_length=1)


class ReviewResult(BaseModel):
    """Structured review judgment; source identity is supplied by the host."""

    verdict: Literal['APPROVE', 'REQUEST_CHANGES', 'INCOMPLETE']
    comments: list[Finding]
    discussion: str


def publication(result: ReviewResult, head: str, run_id: int) -> str:
    """Bind the formal review and all inline findings to this run and reviewed head."""
    bodies = {
        'APPROVE': 'No blocking findings.',
        'REQUEST_CHANGES': 'Blocking findings are recorded inline.',
        'INCOMPLETE': 'Review incomplete.',
    }
    body = bodies[result.verdict]
    if result.discussion:
        body += '\n\n' + result.discussion
    body += f'\n\n<!-- douwebot:review run={run_id} head={head} -->'
    payload: dict[str, object] = {
        'commit_id': head,
        'event': 'COMMENT' if result.verdict == 'INCOMPLETE' else result.verdict,
        'body': body,
        'comments': [comment.model_dump() for comment in result.comments],
    }
    return json.dumps(payload)


def main() -> None:
    """Read only host-bound repository/PR/head values, then publish through `gh`."""
    env = os.environ
    result = ReviewResult.model_validate_json(env['REVIEW_RESULT'])
    head = env['HEAD_SHA']
    endpoint = f'repos/{env["GITHUB_REPOSITORY"]}/pulls/{int(env["PR_NUMBER"])}'
    live = subprocess.run(
        ['gh', 'api', endpoint, '--jq', r'"\(.state) at \(.head.sha)"'], check=True, capture_output=True, text=True
    )
    if live.stdout.strip() != f'open at {head}':
        raise RuntimeError('PR closed or moved while douwebot was reviewing; no review published.')
    subprocess.run(
        ['gh', 'api', '--method', 'POST', '--input', '-', f'{endpoint}/reviews'],
        input=publication(result, head, int(env['GITHUB_RUN_ID'])),
        check=True,
        text=True,
        stdout=subprocess.DEVNULL,
    )


if __name__ == '__main__':
    main()
