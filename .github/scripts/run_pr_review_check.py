"""Run one PR review check with read-only OpenCode tools inside its step and time budget."""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
import urllib.request
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Literal, TextIO
from urllib.parse import quote

from douwebot_opencode_result import final_text, rejection_reason, unfenced
from douwebot_opencode_run import Attempt, check_symlinks, export_spans, otlp_span, run_with_fallback, step_tokens
from publish_pr_review_checks import CheckResult, ReviewFinding, Severity
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pr_review_checks import HUNK, INSTRUCTION_DIRECTORIES, ReviewCheck, changed_files, check_named, select

PROMPTS = Path('.github/pr-review-checks/prompts')
REVIEW_CONTEXT = Path('.github/.review-context')
MAX_FINDINGS = 20
# The instruction layer an agent could load from the head checkout. Their changes stay reviewable in the diffs.
INSTRUCTION_NAMES = frozenset({'AGENTS.md', 'CLAUDE.md', 'CLAUDE.local.md', 'THREAT_MODEL.md', '.mcp.json'})
_TSV_ESCAPES: dict[str, str] = {'\\': '\\', 't': '\t', 'n': '\n', 'r': '\r'}
# Slack reads `<url|label>`, `<!channel>` and entities in webhook text as formatting, so model text arrives escaped.
_SLACK_ESCAPES = str.maketrans({'&': '&amp;', '<': '&lt;', '>': '&gt;'})
PRIVATE_REPORT_LIMIT = 3000


class ModelResult(BaseModel):
    """What the model returns."""

    outcome: Literal['COMPLETE', 'INCOMPLETE']
    findings: list[ReviewFinding] = Field(max_length=MAX_FINDINGS)
    coverage: str
    private_report: str


@dataclass
class Usage:
    """Steps and tokens the stream reported, and whether the runner stopped it at its time budget."""

    steps: int = 0
    tokens: int = 0
    stopped: bool = False
    seconds: int = 0


def strip_instructions(head: Path) -> None:
    """Delete files an agent could read as instructions, so the head checkout stays data."""
    for directory, directories, files in os.walk(head):
        if '.git' in directories:
            directories.remove('.git')
        for name in [name for name in directories if name in INSTRUCTION_DIRECTORIES]:
            path = Path(directory) / name
            # `os.walk` lists a symlinked directory here, and `rmtree` refuses one.
            if path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
            directories.remove(name)
        for name in files:
            if name in INSTRUCTION_NAMES:
                (Path(directory) / name).unlink()


def write_context(check: str, files: list[str], workspace: Path, *, base: str, head: str) -> Path:
    """Copy only this check's diffs and file list next to the PR details."""
    context = workspace / '.github' / '.pr-review-checks' / check
    (context / 'diff' / 'git').mkdir(parents=True)
    shutil.copy(workspace / REVIEW_CONTEXT / 'pr-details.json', context / 'pr-details.json')
    unseen = set(files)
    rows: list[str] = []
    # Rows end only at LF, and jq's `@tsv` escapes a backslash, tab, newline or CR inside a path.
    shared = (workspace / REVIEW_CONTEXT / 'changed-files.txt').read_bytes().decode()
    listed = [row.split('\t') for row in shared.split('\n') if row]
    # Two rows naming one diff file would let one file's diff stand in for the other's.
    shared_names = Counter(fields[2] for fields in listed if len(fields) > 2)
    for listed_path, counts, *diff in listed:
        path = re.sub(r'\\([\\tnr])', lambda escape: _TSV_ESCAPES[escape[1]], listed_path)
        if path not in unseen:
            continue
        unseen.remove(path)
        if diff and shared_names[diff[0]] == 1:
            shutil.copy(workspace / REVIEW_CONTEXT / diff[0], context / diff[0])
        else:
            # The shared context also leaves out `uv.lock` and cassette diffs, which only some checks read. A
            # directory of their own and a reversible name keep two paths from writing one diff file.
            diff = ['diff/git/' + quote(path, safe='') + '.diff']
            (context / diff[0]).write_text(_annotated_diff(workspace / 'pr-head', base, head, path), encoding='utf-8')
        rows.append('\t'.join([listed_path, counts, diff[0]]))
    # The context lists files from the PR files API, which stops at 3,000; an unlisted file would go unreviewed.
    if unseen:
        raise ValueError(f'The review context has no diff for {sorted(unseen)}')
    (context / 'changed-files.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return context


def _annotated_diff(repository: Path, base: str, head: str, path: str) -> str:
    """Number the changed lines of one file's diff the way the shared context does, for inline comments."""
    patch = subprocess.run(
        ['git', '-C', str(repository), 'diff', '--no-ext-diff', '--no-textconv', '--no-color', '--text', '-U0']
        + [f'{base}...{head}', '--', path],
        check=True,
        capture_output=True,
    ).stdout.decode(errors='replace')
    lines: list[str] = []
    old = new = 0
    in_hunk = False
    # Git ends a line only at LF. Text mode and `splitlines` would also split at CR and Unicode separators inside a line.
    for line in patch.removesuffix('\n').split('\n'):
        if hunk := HUNK.match(line):
            old, new, in_hunk = int(hunk[1]), int(hunk[2]), True
            lines.append(line)
        elif in_hunk and line.startswith('+'):
            lines.append(f'NL:{new} {line}')
            new += 1
        elif in_hunk and line.startswith('-'):
            lines.append(f'OL:{old} {line}')
            old += 1
        else:
            lines.append(line)
    return '\n'.join(lines) + '\n'


def build_prompt(check: ReviewCheck, context: Path, workspace: Path, *, pr_number: int, repository: str) -> str:
    """Fill the shared prompt with this check's fragment and limits."""
    values = {
        '{check_title}': check.title,
        '{check_scope}': check.scope,
        '{pr_number}': str(pr_number),
        '{repository}': repository,
        '{context_dir}': context.relative_to(workspace).as_posix(),
        '{max_steps}': str(check.max_steps),
        '{max_findings}': str(MAX_FINDINGS),
        '{check_fragment}': '\n\n'.join(
            (workspace / PROMPTS / f'{name}.md').read_text(encoding='utf-8').strip()
            for name in (['security'] if check.security_rubric else []) + [check.name]
        ),
        '{schema}': json.dumps(ModelResult.model_json_schema(), separators=(',', ':')),
    }
    prompt = (workspace / PROMPTS / 'shared.md').read_text(encoding='utf-8')
    for placeholder, value in values.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def run_opencode(
    command: list[str], prompt: str, *, cwd: Path, env: dict[str, str], log: TextIO, seconds: float
) -> Usage:
    """Stream the event log to `log`, counting steps and tokens, and stop the run at its time budget."""
    usage = Usage()
    started = time.monotonic()
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=cwd, env=env, encoding='utf-8'
    ) as process:

        def stop() -> None:
            usage.stopped = True
            process.kill()

        timer = threading.Timer(seconds, stop)
        timer.start()
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(prompt)
            process.stdin.close()
            for line in process.stdout:
                log.write(line)
                tokens = step_tokens(line) if line.strip() else None
                if tokens is None:
                    continue
                usage.steps += 1
                usage.tokens += tokens
            process.wait()
        finally:
            timer.cancel()
    usage.seconds = round(time.monotonic() - started)
    if not usage.stopped and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return usage


def check_result(
    check: ReviewCheck,
    files: list[str],
    stream: Path,
    usage: Usage,
    *,
    changed: Mapping[str, frozenset[tuple[str, int]]],
) -> tuple[CheckResult, str]:
    """Validate the model's result; a spent budget is an incomplete review, never a crash."""
    outcome: Literal['COMPLETE', 'INCOMPLETE'] = 'INCOMPLETE'
    findings: list[ReviewFinding] = []
    dropped: Counter[Severity] = Counter()
    private_report = ''
    at_step_budget = usage.steps >= check.max_steps
    if usage.stopped:
        coverage = f'Review stopped at its time budget of {check.max_minutes} minutes.'
    else:
        try:
            result = ModelResult.model_validate_json(unfenced(final_text(stream)))
        except (ValueError, ValidationError) as error:
            if not at_step_budget:
                # The error would repeat the answer, private report included, into the public job log.
                raise ValueError(f'The review answer was rejected: {rejection_reason(error)}') from None
            coverage = f'Review stopped at its step budget of {check.max_steps}.'
        else:
            # The result artifact is public, and a finding on a line the pull request did not change can describe a
            # weakness it did not add. A finding on a file outside this check is misplaced. Only the severity of
            # either leaves this job, so the publisher can fail a blocking one.
            for finding in result.findings:
                if finding.path in files and (finding.side, finding.line) in changed.get(finding.path, frozenset()):
                    findings.append(finding)
                else:
                    dropped[finding.severity] += 1
            coverage, private_report = result.coverage, result.private_report
            # OpenCode withdraws the tools at the step budget, so a review that reached it may have stopped short.
            if at_step_budget:
                coverage = f'Review reached its step budget of {check.max_steps}. {coverage}'
            else:
                outcome = result.outcome
    return CheckResult(
        check=check.name,
        files=files,
        outcome=outcome,
        findings=findings,
        dropped=dict(dropped),
        coverage=coverage,
        steps=usage.steps,
        tokens=usage.tokens,
        seconds=usage.seconds,
    ), private_report


def report_privately(webhook: str, check: str, pr_number: int, repository: str, report: str) -> None:
    """Send a vulnerability that already exists on the default branch to the private triage channel.

    The report is model output that the pull request's author can steer, so it arrives escaped, capped and quoted.
    """
    quoted = report[:PRIVATE_REPORT_LIMIT].replace('```', "'''").translate(_SLACK_ESCAPES)
    text = f'PR Review Check ({check}) on {repository}#{pr_number} found a pre-existing issue:\n```\n{quoted}\n```'
    request = urllib.request.Request(
        webhook, data=json.dumps({'text': text}).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=30):
        pass


def _interrupt(signum: int, frame: FrameType | None) -> None:
    """Raise `KeyboardInterrupt` once, then ignore the cancellation's later signals while the spans export."""
    for each in (signal.SIGINT, signal.SIGTERM):
        signal.signal(each, signal.SIG_IGN)
    raise KeyboardInterrupt


def main() -> None:
    """Review one check and write its publishable result for the publisher job."""
    env = os.environ
    check = check_named(env['CHECK'])
    pr_number = int(env['PR_NUMBER'])
    repository = env['GITHUB_REPOSITORY']
    runner_temp = Path(env['RUNNER_TEMP'])
    # These attributes tie OpenCode's own telemetry, and the runner's span, to this review.
    attributes = {
        'pr_review_checks.check': check.name,
        'github.repository': repository,
        'github.pr_number': str(pr_number),
        'github.head_sha': env['HEAD_SHA'],
        'github.run_id': env['GITHUB_RUN_ID'],
        'github.run_attempt': env['GITHUB_RUN_ATTEMPT'],
    }
    # Actions cancels a step with SIGINT to its entry process, which the Review step `exec`s into this script, then
    # SIGTERM 7.5 s later, and kills it 2.5 s after that. The first raises `KeyboardInterrupt`, so the `finally` below
    # exports the review's span, and the rest cannot cut that export short.
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _interrupt)
    started = time.time_ns()
    # The review's span, which each OpenCode run's span names as its parent.
    trace_id, span_id = secrets.token_hex(16), secrets.token_hex(8)
    attempts: list[Attempt] = []
    values: dict[str, str | int | bool] = {}
    try:
        workspace = Path(env['GITHUB_WORKSPACE']).resolve(strict=True)
        head = workspace / 'pr-head'
        changes = changed_files(head, env['BASE_SHA'], env['HEAD_SHA'])
        selected = select(changes)
        files = selected[check.name]
        strip_instructions(head)
        check_symlinks(workspace, skip_pr_head=True)
        check_symlinks(head)
        context = write_context(check.name, files, workspace, base=env['BASE_SHA'], head=env['HEAD_SHA'])
        prompt = build_prompt(check, context, workspace, pr_number=pr_number, repository=repository)
        stream = runner_temp / f'pr-review-checks-{check.name}.ndjson'
        # A run on the fallback provider gets only what is left of the time budget.
        deadline = time.monotonic() + check.max_minutes * 60
        usage, _ = run_with_fallback(
            lambda command, opencode_env, log: run_opencode(
                command, prompt, cwd=workspace, env=opencode_env, log=log, seconds=deadline - time.monotonic()
            ),
            agent='reviewer',
            output=stream,
            steps=check.max_steps,
            attributes=attributes,
            # The `detect` job builds the matrix in `select` order.
            matrix_index=list(selected).index(check.name),
            attempts=attempts,
            trace_id=trace_id,
            parent_id=span_id,
        )
        values = {
            'pr_review_checks.steps': usage.steps,
            'pr_review_checks.tokens': usage.tokens,
            'pr_review_checks.seconds': usage.seconds,
        }
        changed = {change.path: change.positions for change in changes}
        result, private_report = check_result(check, files, stream, usage, changed=changed)
        values |= {'pr_review_checks.outcome': result.outcome, 'pr_review_checks.findings': len(result.findings)}
    # A failed, rejected or interrupted review is exported too, under the exception's class name: its message can
    # repeat model or provider text.
    except BaseException as error:
        values |= {'pr_review_checks.outcome': 'FAILED', 'pr_review_checks.exception': type(error).__name__}
        raise
    finally:
        if attempts:
            # The plan of the last OpenCode run, which served the review or failed it.
            last = attempts[-1]
            failure = {'error_type': last.error_type, 'error_status': last.status, 'error_code': last.code}
            values |= {
                'pr_review_checks.provider': last.provider.name,
                'pr_review_checks.plan': last.provider.plan,
                'pr_review_checks.attempts': len(attempts),
                **{f'pr_review_checks.{key}': value for key, value in failure.items() if value is not None},
                **({} if last.finished is None else {'pr_review_checks.step_finish': last.finished}),
                # Only a restart moves the chain past a run that finished a step.
                **({'pr_review_checks.restarted': True} if any(run.finished for run in attempts[:-1]) else {}),
            }
        if token := env.get('LOGFIRE_TOKEN', ''):
            # OpenCode's own spans carry no model, token or step counts.
            span = otlp_span(
                f'PR Review Check {check.name}',
                {**attributes, **values},
                started,
                time.time_ns(),
                trace_id=trace_id,
                span_id=span_id,
                failed='pr_review_checks.exception' in values,
            )
            export_spans(token, 'pr-review-checks', [span])
        else:
            print('::warning::LOGFIRE_TOKEN is empty, so this review exports no telemetry.')
    if private_report:
        if not (webhook := env.get('TRIAGE_WEBHOOK_URL', '')):
            raise ValueError('The review produced a private report, but TRIAGE_WEBHOOK_URL is empty')
        report_privately(webhook, check.name, pr_number, repository, private_report)
    output = runner_temp / 'pr-review-checks-result'
    output.mkdir(exist_ok=True)
    (output / f'{check.name}.json').write_text(result.model_dump_json(), encoding='utf-8')
    print(
        f'{check.name}: {result.outcome}, {len(result.findings)} findings, '
        f'{sum(result.dropped.values())} dropped outside the changed lines, {usage.steps} steps, '
        f'{usage.tokens:,} tokens, {usage.seconds} s'
    )


if __name__ == '__main__':
    main()
