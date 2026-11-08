"""Run one Security Review check with read-only OpenCode tools inside a step, token and time budget."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from douwebot_opencode_result import final_text
from douwebot_opencode_run import MODEL, check_symlinks, read_only_config
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from security_review_checks import CHECKS, changed_files, select

PROMPTS = Path('.github/security-review/prompts')
REVIEW_CONTEXT = Path('.github/.review-context')
MAX_FINDINGS = 20
# Budget values are pending maintainer sign-off.
MAX_STEPS = 60
MAX_TOKENS = 3_000_000
TIMEOUT_SECONDS = 25 * 60
# The instruction layer an agent could load from the head checkout. Their changes stay reviewable in the diffs.
INSTRUCTION_NAMES = frozenset({'AGENTS.md', 'CLAUDE.md', 'CLAUDE.local.md', 'THREAT_MODEL.md', '.mcp.json'})
INSTRUCTION_DIRECTORIES = frozenset({'.claude', '.agents', 'agent_docs', '.opencode'})

Severity = Literal['critical', 'high', 'medium']
BLOCKING: frozenset[Severity] = frozenset({'critical', 'high'})

_EVENT = TypeAdapter(dict[str, object])
_TOKENS = TypeAdapter(dict[str, float | dict[str, float]])


class SecurityFinding(BaseModel):
    """One inline finding on a file this check reviews."""

    path: str
    line: int = Field(gt=0)
    side: Literal['LEFT', 'RIGHT']
    severity: Severity
    body: str = Field(min_length=1)


class ModelResult(BaseModel):
    """What the model returns."""

    outcome: Literal['COMPLETE', 'INCOMPLETE']
    findings: list[SecurityFinding] = Field(max_length=MAX_FINDINGS)
    coverage: str
    private_report: str


class CheckResult(BaseModel):
    """What the trusted publisher receives; it never carries the private report."""

    check: str
    files: list[str]
    outcome: Literal['COMPLETE', 'INCOMPLETE']
    findings: list[SecurityFinding]
    coverage: str
    steps: int
    tokens: int


@dataclass
class Usage:
    """Steps and tokens the stream reported, and why the runner stopped it early."""

    steps: int = 0
    tokens: int = 0
    stopped: str = ''


def strip_instructions(head: Path) -> None:
    """Delete files an agent could read as instructions, so the head checkout stays data."""
    for directory, directories, files in os.walk(head):
        if '.git' in directories:
            directories.remove('.git')
        for name in [name for name in directories if name in INSTRUCTION_DIRECTORIES]:
            shutil.rmtree(Path(directory) / name)
            directories.remove(name)
        for name in files:
            if name in INSTRUCTION_NAMES:
                (Path(directory) / name).unlink()


def write_context(check: str, files: list[str], workspace: Path) -> Path:
    """Copy only this check's diffs and file list next to the PR details."""
    context = workspace / '.github' / '.security-review' / check
    (context / 'diff').mkdir(parents=True)
    shutil.copy(workspace / REVIEW_CONTEXT / 'pr-details.json', context / 'pr-details.json')
    selected = set(files)
    rows: list[str] = []
    for row in (workspace / REVIEW_CONTEXT / 'changed-files.txt').read_text(encoding='utf-8').splitlines():
        fields = row.split('\t')
        if fields[0] not in selected:
            continue
        rows.append(row)
        if len(fields) == 3:
            shutil.copy(workspace / REVIEW_CONTEXT / fields[2], context / fields[2])
    (context / 'changed-files.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
    return context


def build_prompt(check: str, context: Path, workspace: Path, *, pr_number: int, repository: str) -> str:
    """Fill the shared prompt with this check's fragment and limits."""
    entry = next(entry for entry in CHECKS if entry.name == check)
    values = {
        '{check_title}': entry.title,
        '{check_scope}': entry.scope,
        '{pr_number}': str(pr_number),
        '{repository}': repository,
        '{context_dir}': context.relative_to(workspace).as_posix(),
        '{max_steps}': str(MAX_STEPS),
        '{max_findings}': str(MAX_FINDINGS),
        '{check_fragment}': (workspace / PROMPTS / f'{check}.md').read_text(encoding='utf-8').strip(),
        '{schema}': json.dumps(ModelResult.model_json_schema(), separators=(',', ':')),
    }
    prompt = (workspace / PROMPTS / 'shared.md').read_text(encoding='utf-8')
    for placeholder, value in values.items():
        prompt = prompt.replace(placeholder, value)
    return prompt


def step_tokens(line: str) -> int | None:
    """Return the tokens a `step_finish` event reports, or None for any other event."""
    event = _EVENT.validate_json(line)
    if event.get('type') != 'step_finish':
        return None
    part = _EVENT.validate_python(event.get('part'))
    tokens = _TOKENS.validate_python(part.get('tokens', {}))
    total = 0.0
    for value in tokens.values():
        total += sum(value.values()) if isinstance(value, dict) else value
    return int(total)


def run_opencode(command: list[str], prompt: str, *, cwd: Path, env: dict[str, str], output: Path) -> Usage:
    """Stream the event log to `output`, stopping the run when it passes the token or time budget."""
    usage = Usage()
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=cwd, env=env, encoding='utf-8'
    )

    def stop(reason: str) -> None:
        usage.stopped = usage.stopped or reason
        process.kill()

    timer = threading.Timer(TIMEOUT_SECONDS, stop, args=(f'time budget of {TIMEOUT_SECONDS // 60} minutes',))
    timer.start()
    try:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(prompt)
        process.stdin.close()
        with output.open('w', encoding='utf-8') as log:
            for line in process.stdout:
                log.write(line)
                tokens = step_tokens(line) if line.strip() else None
                if tokens is None:
                    continue
                usage.steps += 1
                usage.tokens += tokens
                if usage.tokens > MAX_TOKENS:
                    stop(f'token budget of {MAX_TOKENS:,}')
        returncode = process.wait()
    finally:
        timer.cancel()
    if not usage.stopped and returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return usage


def check_result(check: str, files: list[str], stream: Path, usage: Usage) -> tuple[CheckResult, str]:
    """Validate the model's result; a spent budget is an incomplete review, never a crash."""
    outcome: Literal['COMPLETE', 'INCOMPLETE'] = 'INCOMPLETE'
    findings: list[SecurityFinding] = []
    private_report = ''
    if usage.stopped:
        coverage = f'Review stopped at its {usage.stopped}.'
    else:
        try:
            result = ModelResult.model_validate_json(final_text(stream))
        except (ValueError, ValidationError):
            if usage.steps < MAX_STEPS:
                raise
            coverage = f'Review stopped at its step budget of {MAX_STEPS}.'
        else:
            if outside := sorted({finding.path for finding in result.findings} - set(files)):
                raise ValueError(f'Findings outside this check: {outside}')
            outcome, findings, coverage, private_report = (
                result.outcome,
                result.findings,
                result.coverage,
                result.private_report,
            )
    return CheckResult(
        check=check,
        files=files,
        outcome=outcome,
        findings=findings,
        coverage=coverage,
        steps=usage.steps,
        tokens=usage.tokens,
    ), private_report


def report_privately(webhook: str, check: str, pr_number: int, repository: str, report: str) -> None:
    """Send a vulnerability that already exists on the base branch to the private triage channel."""
    text = f'Security Review ({check}) on {repository}#{pr_number} found a pre-existing issue:\n\n{report}'
    request = urllib.request.Request(
        webhook, data=json.dumps({'text': text}).encode(), headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(request, timeout=30):
        pass


def main() -> None:
    """Review one check and write its publishable result for the publisher job."""
    env = os.environ
    check = env['CHECK']
    pr_number = int(env['PR_NUMBER'])
    repository = env['GITHUB_REPOSITORY']
    workspace = Path(env['GITHUB_WORKSPACE']).resolve(strict=True)
    runner_temp = Path(env['RUNNER_TEMP'])
    head = workspace / 'pr-head'
    files = select(changed_files(head, env['BASE_SHA'], env['HEAD_SHA']))[check]
    strip_instructions(head)
    check_symlinks(workspace, skip_pr_head=True)
    check_symlinks(head)
    context = write_context(check, files, workspace)
    prompt = build_prompt(check, context, workspace, pr_number=pr_number, repository=repository)
    api_key = env['OPENCODE_API_KEY']
    if not api_key:
        raise ValueError('OPENCODE_API_KEY is empty')
    stream = runner_temp / f'security-review-{check}.ndjson'
    with tempfile.TemporaryDirectory(dir=runner_temp, prefix='security-review-opencode-') as temp:
        temp_root = Path(temp)
        config_dir = temp_root / 'config' / 'opencode'
        config_dir.mkdir(parents=True)
        (config_dir / 'opencode.json').write_text(json.dumps(read_only_config(steps=MAX_STEPS)), encoding='utf-8')
        opencode_env = {
            'HOME': str(temp_root),
            'PATH': env['PATH'],
            'XDG_CONFIG_HOME': str(temp_root / 'config'),
            'XDG_DATA_HOME': str(temp_root / 'data'),
            'XDG_CACHE_HOME': str(temp_root / 'cache'),
            'OPENCODE_API_KEY': api_key,
        }
        command = ['opencode', 'run', '--standalone', '--model', MODEL, '--agent', 'reviewer', '--format', 'json']
        usage = run_opencode(command, prompt, cwd=workspace, env=opencode_env, output=stream)
    result, private_report = check_result(check, files, stream, usage)
    if private_report:
        if not (webhook := env.get('TRIAGE_WEBHOOK_URL', '')):
            raise ValueError('The review produced a private report, but TRIAGE_WEBHOOK_URL is empty')
        report_privately(webhook, check, pr_number, repository, private_report)
    output = runner_temp / 'security-review-result'
    output.mkdir(exist_ok=True)
    (output / f'{check}.json').write_text(result.model_dump_json(), encoding='utf-8')
    print(f'{check}: {result.outcome}, {len(result.findings)} findings, {usage.steps} steps, {usage.tokens:,} tokens')


if __name__ == '__main__':
    main()
