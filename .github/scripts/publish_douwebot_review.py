"""Publish structured douwebot findings from a trusted host, with a deterministic receipt."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel, Field

APPROVAL_BODY = 'No blocking findings.'
DOUWEBOT_MARKER = re.compile(r'\n\n<!-- douwebot:review run=([1-9][0-9]*) head=([a-f0-9]{40}) -->\Z')
ReviewVerdict = Literal['APPROVE', 'REQUEST_CHANGES', 'INCOMPLETE']


class Finding(BaseModel):
    """One inline finding; GitHub validates the position against the pinned diff."""

    path: str
    line: int = Field(gt=0)
    side: Literal['LEFT', 'RIGHT']
    body: str = Field(min_length=1)


class ReviewResult(BaseModel):
    """Structured review judgment; source identity is supplied by the host."""

    verdict: ReviewVerdict
    comments: list[Finding]
    discussion: str


def publication(result: ReviewResult, head: str, run_id: int, *, inline: bool = True) -> str:
    """Bind the formal review and all findings to this run and reviewed head."""
    bodies: dict[ReviewVerdict, str] = {
        'APPROVE': APPROVAL_BODY,
        'REQUEST_CHANGES': 'Blocking findings are recorded inline.'
        if inline
        else 'Blocking findings are recorded below.',
        'INCOMPLETE': 'Review incomplete.',
    }
    body = bodies[result.verdict]
    if result.discussion:
        body += '\n\n' + result.discussion
    if not inline:
        for comment in result.comments:
            body += f'\n\n### `{comment.path}:{comment.line}` (`{comment.side}`)\n\n{comment.body}'
    body += f'\n\n<!-- douwebot:review run={run_id} head={head} -->'
    payload: dict[str, object] = {
        'commit_id': head,
        'event': 'COMMENT' if result.verdict == 'INCOMPLETE' else result.verdict,
        'body': body,
        'comments': [comment.model_dump() for comment in result.comments] if inline else [],
    }
    return json.dumps(payload)


def main() -> None:
    """Read only host-bound repository/PR/head values, then publish through `gh`."""
    env = os.environ
    head = env['HEAD_SHA']
    endpoint = f'repos/{env["GITHUB_REPOSITORY"]}/pulls/{int(env["PR_NUMBER"])}'
    result: ReviewResult | None = None
    for inline in (True, False):
        live = subprocess.run(
            ['gh', 'api', endpoint, '--jq', r'"\(.state) at \(.head.sha)"'], check=True, capture_output=True, text=True
        )
        if live.stdout.strip() != f'open at {head}':
            print('PR closed or moved while douwebot was reviewing; no review needed.')
            return
        if result is None:
            result = ReviewResult.model_validate_json(env['REVIEW_RESULT'])
        posted = subprocess.run(
            ['gh', 'api', '--method', 'POST', '--input', '-', f'{endpoint}/reviews'],
            input=publication(result, head, int(env['GITHUB_RUN_ID']), inline=inline),
            check=False,
            capture_output=True,
            text=True,
        )
        if posted.returncode == 0:
            return
        if inline and result.comments and '(HTTP 422)' in posted.stderr:
            print('GitHub rejected the inline review; retrying with findings in the review body.')
            continue
        print(posted.stdout, file=sys.stderr)
        print(posted.stderr, file=sys.stderr)
        posted.check_returncode()


if __name__ == '__main__':
    main()
