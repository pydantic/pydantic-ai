"""Run gh-aw threat detection with OpenCode and record the verdict `threat-detect conclude` reads."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

from douwebot_opencode_result import final_text, rejection_reason
from douwebot_opencode_run import check_symlinks, run_read_only
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

DETECTION_DIR = Path('/tmp/gh-aw/threat-detection')
RESULT_PREFIX = 'THREAT_DETECTION_RESULT:'
# The detection step's timeout-minutes is 15. Stop retrying in time to record why no verdict exists.
DEADLINE_SECONDS = 13 * 60
# Appended to the detection prompt gh-aw renders, which assumes a shell and the repository checkout.
TOOL_GUIDANCE = (
    '\n\nYou have read, grep, and glob tools only, inside /tmp/gh-aw/threat-detection. '
    'Shell, network, and the repository workspace are unavailable, so judge a patch from its own content.\n'
)


class Verdict(BaseModel):
    """The result schema `threat-detect conclude` v0.5.3 accepts: no other fields, booleans, bounded reasons."""

    model_config = ConfigDict(extra='forbid', strict=True)

    prompt_injection: bool
    secret_leak: bool
    malicious_patch: bool
    reasons: Annotated[
        list[Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]],
        Field(max_length=20),
    ]


def parse_verdict(text: str) -> Verdict:
    """Return the verdict on the last `THREAT_DETECTION_RESULT:` line of the final message."""
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith(RESULT_PREFIX)]
    if not lines:
        raise ValueError(f'OpenCode did not return a {RESULT_PREFIX} line')
    return Verdict.model_validate_json(lines[-1].removeprefix(RESULT_PREFIX))


def report_status(reason: str, code: int) -> None:
    """Append threat-detect's terminal status line, which `conclude` maps to a failure reason."""
    line = f'THREAT_DETECTION_STATUS: reason={reason} exit={code}'
    with (DETECTION_DIR / 'detection.log').open('a', encoding='utf-8') as log:
        log.write(line + '\n')
    print(line)


def main() -> None:
    """Record a verdict, or the status line that explains its absence."""
    result_path = DETECTION_DIR / 'detection_result.json'
    full_path = DETECTION_DIR / 'detection_result_full.json'
    for path in (result_path, full_path):
        path.unlink(missing_ok=True)
    deadline = time.monotonic() + DEADLINE_SECONDS
    retries = int(os.environ['GH_AW_HARNESS_MAX_RETRIES'])
    try:
        check_symlinks(DETECTION_DIR)
        prompt = Path(os.environ['GH_AW_PROMPT']).read_text(encoding='utf-8') + TOOL_GUIDANCE
        attempt = 1
        while True:
            # One stream per attempt, so the usage report counts the retried attempts too.
            stream = Path(os.environ['RUNNER_TEMP']) / f'gh-aw-opencode-detector-{attempt}.ndjson'
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired('opencode', DEADLINE_SECONDS)
            try:
                run_read_only(prompt, cwd=DETECTION_DIR, agent='detector', output=stream, timeout=remaining)
                text = final_text(stream)
                break
            except (ValueError, subprocess.CalledProcessError) as exc:
                reason = rejection_reason(exc) if isinstance(exc, ValueError) else type(exc).__name__
                print(f'OpenCode threat detection attempt {attempt} failed: {reason}')
                if attempt > retries:
                    raise
                attempt += 1
    except subprocess.TimeoutExpired:
        report_status('engine_timeout', 2)
        sys.exit('OpenCode threat detection ran out of time')
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        report_status('engine_error', 2)
        sys.exit(f'OpenCode threat detection failed: {type(exc).__name__}')
    try:
        verdict = parse_verdict(text)
    except ValueError as exc:
        report_status('invalid_report_exhausted', 1)
        sys.exit(f'OpenCode returned an invalid threat detection result: {rejection_reason(exc)}')
    # Only the uploaded result drops the model-authored reasons; `conclude` reads them from the full result.
    full_path.write_text(verdict.model_dump_json(indent=2), encoding='utf-8')
    result_path.write_text(verdict.model_copy(update={'reasons': []}).model_dump_json(indent=2), encoding='utf-8')
    report_status('result_recorded', 0)


if __name__ == '__main__':
    main()
