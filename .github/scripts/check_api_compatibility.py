from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

PACKAGES = {
    'pydantic_ai': 'pydantic_ai_slim',
    'pydantic_graph': 'pydantic_graph',
    'pydantic_evals': 'pydantic_evals',
    'clai': 'clai',
}
_FINDING_RE = re.compile(r'^(?P<path>.*):(?P<line>\d+): (?P<message>.*)$')


class Waiver(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)

    against: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    fingerprint: Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    pull_request: Annotated[str, StringConstraints(pattern=r'^https://github\.com/pydantic/pydantic-ai/pull/\d+$')]


class Allowlist(BaseModel):
    model_config = ConfigDict(extra='forbid')

    allowed_breakages: list[Waiver]


@dataclass(frozen=True)
class Finding:
    package: str
    path: str
    line: int
    message: str

    @property
    def fingerprint(self) -> str:
        value = f'{self.package}\0{self.path}\0{self.message}'.encode()
        return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare the public Python API with a released Git tag.')
    parser.add_argument('--against', required=True, help='Released Git tag to compare against.')
    parser.add_argument(
        '--allowlist',
        type=Path,
        default=Path('.github/api-compatibility-allowlist.json'),
        help='Exact reviewed compatibility-impact waivers.',
    )
    args = parser.parse_args()

    waivers = load_waivers(args.allowlist)
    used_waivers: set[tuple[str, str]] = set()
    failed = False
    for package, search_path in PACKAGES.items():
        findings = run_griffe(package, search_path, args.against)
        for finding in findings:
            waiver_key = (args.against, finding.fingerprint)
            waiver = waivers.get(waiver_key)
            if waiver is None:
                failed = True
                emit_annotation(
                    'error',
                    finding,
                    f'{finding.message} [fingerprint: {finding.fingerprint}]. '
                    'Preserve compatibility or follow the allowed compatibility-impact process.',
                )
            else:
                used_waivers.add(waiver_key)
                emit_annotation(
                    'warning',
                    finding,
                    f'{finding.message} Allowed by {waiver.pull_request}: {waiver.reason}',
                )

    unused_waivers = {key for key in waivers if key[0] == args.against} - used_waivers
    for _, fingerprint in sorted(unused_waivers):
        failed = True
        print(f'::error::Unused API compatibility waiver for {args.against}: {fingerprint}')

    if failed:
        raise SystemExit(1)


def load_waivers(path: Path) -> dict[tuple[str, str], Waiver]:
    allowlist = Allowlist.model_validate_json(path.read_text(encoding='utf-8'))
    waivers: dict[tuple[str, str], Waiver] = {}
    for waiver in allowlist.allowed_breakages:
        key = (waiver.against, waiver.fingerprint)
        if key in waivers:
            raise ValueError(f'{path}: duplicate waiver for {waiver.against}: {waiver.fingerprint}')
        waivers[key] = waiver
    return waivers


def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'griffecli',
            'check',
            package,
            '--search',
            search_path,
            '--against',
            against,
            '--format',
            'oneline',
            '--no-color',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in {0, 1}:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return parse_findings(package, result.stdout)


def parse_findings(package: str, output: str) -> list[Finding]:
    findings: list[Finding] = []
    for output_line in output.splitlines():
        match = _FINDING_RE.fullmatch(output_line)
        if match is None:
            raise ValueError(f'Unexpected Griffe output: {output_line!r}')
        findings.append(
            Finding(
                package=package,
                path=match.group('path'),
                line=int(match.group('line')),
                message=match.group('message'),
            )
        )
    return findings


def emit_annotation(level: str, finding: Finding, message: str) -> None:
    escaped = message.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
    print(f'::{level} file={finding.path},line={finding.line}::{escaped}')


if __name__ == '__main__':
    main()
