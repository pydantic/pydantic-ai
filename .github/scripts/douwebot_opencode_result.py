"""Validate a completed OpenCode review before passing it to the publisher."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from publish_douwebot_review import ReviewResult
from pydantic import TypeAdapter

_EVENT = TypeAdapter(dict[str, object])
# Some models wrap the requested JSON object in one Markdown code fence despite the prompt.
_FENCE = re.compile(r'\A\s*```[\w+-]*[ \t]*\n(.*)\n[ \t]*```\s*\Z', re.DOTALL)


def final_text(path: Path) -> str:
    """Return the final assistant text only after a normal completed stream."""
    events = [_EVENT.validate_json(line) for line in path.read_text(encoding='utf-8').splitlines()]
    if not events or events[-1].get('type') != 'step_finish':
        raise ValueError('OpenCode review did not complete')
    finish = _EVENT.validate_python(events[-1].get('part'), strict=True)
    if finish.get('reason') != 'stop':
        raise ValueError('OpenCode review did not stop normally')
    message_id = finish.get('messageID')
    if not isinstance(message_id, str) or not message_id:
        raise ValueError('OpenCode final message is missing its identity')
    if any(event.get('type') == 'error' for event in events):
        raise ValueError('OpenCode review reported an error')
    for event in reversed(events[:-1]):
        if event.get('type') == 'text':
            part = _EVENT.validate_python(event.get('part'), strict=True)
            if part.get('messageID') != message_id:
                continue
            content = part.get('text')
            if not isinstance(content, str):
                raise ValueError('Invalid OpenCode text event')
            return content
    raise ValueError('OpenCode review did not return text')


def review_result(path: Path) -> ReviewResult:
    """Return the final assistant result only after a normal completed stream."""
    text = final_text(path)
    if fenced := _FENCE.match(text):
        text = fenced[1]
    return ReviewResult.model_validate_json(text)


def main() -> None:
    """Publish one compact validated result as a GitHub Actions output."""
    result = review_result(Path(sys.argv[1]))
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as output:
        output.write('result=' + json.dumps(result.model_dump(), separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
