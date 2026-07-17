#!/usr/bin/env python3
"""Verify declarative eval output against evaluator-owned behavioral oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROVED_GAPS = {
    'prompt_replacement_vs_composition',
    'retry_call_counts',
    'interrupt_and_approval',
    'run_vs_stream',
}
_UNPROVED_GAPS = {
    'application_event_contract',
    'approval_resume_durability',
    'durability_replay',
    'exactly_once',
    'fanout_reducer_cancellation_order',
    'limit_domains',
    'message_history_protocol',
    'stream_reconnect',
    'structured_output_transport',
    'return_direct_bridge',
    'middleware_order',
    'tool_schema_authorization',
}
_STATE_BOUNDARIES = {
    'dependencies',
    'messages',
    'workflow_state',
    'checkpoint_state',
    'long_term_memory',
}
_WORKAROUND_OUTCOMES = {
    'validated-native',
    'validated-adapter',
    'integration-required',
    'blocker',
}
_REQUIRED_WORKAROUND_OUTCOMES = {
    'prompt_replacement_vs_composition': 'validated-adapter',
    'retry_call_counts': 'validated-adapter',
    'interrupt_and_approval': 'validated-adapter',
    'run_vs_stream': 'validated-native',
    'application_event_contract': 'integration-required',
    'approval_resume_durability': 'integration-required',
    'durability_replay': 'integration-required',
    'exactly_once': 'integration-required',
    'fanout_reducer_cancellation_order': 'integration-required',
    'limit_domains': 'integration-required',
    'message_history_protocol': 'integration-required',
    'stream_reconnect': 'integration-required',
    'structured_output_transport': 'integration-required',
    'return_direct_bridge': 'integration-required',
    'middleware_order': 'integration-required',
    'tool_schema_authorization': 'integration-required',
}
_GAP_TERMS = {
    'prompt_replacement_vs_composition': ('replace', 'compose'),
    'retry_call_counts': ('local', 'model call'),
    'interrupt_and_approval': ('restart', 'tool-call id', 'principal'),
    'run_vs_stream': ('first', 'final'),
    'application_event_contract': ('correlation', 'sequence', 'terminal'),
    'approval_resume_durability': ('tool-call id', 'principal', 'consumed'),
    'durability_replay': ('checkpoint', 'crash', 'replay'),
    'exactly_once': ('idempot', 'unknown'),
    'fanout_reducer_cancellation_order': ('reducer', 'cancellation', 'order', 'race'),
    'limit_domains': ('recursion', 'thread', 'request', 'tool-call'),
    'message_history_protocol': ('history', 'tool-call', 'conversion'),
    'middleware_order': ('order', 'retry', 'resume'),
    'return_direct_bridge': ('return_direct', 'model-call', 'bridge'),
    'stream_reconnect': ('reconnect', 'duplicate', 'cursor'),
    'structured_output_transport': ('provider', 'tool', 'output'),
    'tool_schema_authorization': ('identity', 'schema', 'authorization'),
}
_ROW_TEXT_FIELDS = (
    'title',
    'source_guarantee',
    'target_behavior',
    'workaround',
    'decision',
    'probe',
    'residual',
    'owner',
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--pydantic-repo', required=True, type=Path)
    parser.add_argument(
        '--write-evidence',
        action='store_true',
        help='write evaluator-measured evidence.json, then exit without running the audit',
    )
    return parser


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f'Artifact must be a regular file: {path}')
    if path.stat().st_size > 1_000_000:
        raise SystemExit(f'Artifact is unexpectedly large: {path}')
    return path.read_text(encoding='utf-8')


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise SystemExit(f'Invalid JSON artifact {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise SystemExit(f'Expected a JSON object in {path}')
    return value


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, 'O_NOFOLLOW'):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise SystemExit(f'Refusing to replace existing evaluator output: {path}') from exc
    with os.fdopen(descriptor, 'w', encoding='utf-8') as file:
        json.dump(value, file, indent=2)
        file.write('\n')


def _git(repository: Path, *args: str) -> bytes:
    return subprocess.run(
        ['git', '-C', str(repository), *args],
        check=True,
        capture_output=True,
    ).stdout


def _repo_fingerprint(repository: Path) -> str:
    digest = hashlib.sha256()
    digest.update(_git(repository, 'rev-parse', 'HEAD'))
    digest.update(_git(repository, 'diff', '--binary', 'HEAD', '--'))
    untracked = _git(repository, 'ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
    for raw_path in sorted(item for item in untracked if item):
        path = repository / os.fsdecode(raw_path)
        digest.update(raw_path)
        if path.is_symlink():
            digest.update(os.readlink(path).encode())
        elif path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkout_sha(repository: Path) -> str:
    return _git(repository, 'rev-parse', 'HEAD').decode().strip()


def _measure_provenance(repository: Path, oracle_provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        'versions': oracle_provenance['versions'],
        'import_origins': oracle_provenance['import_origins'],
        'pydantic_checkout_sha': _checkout_sha(repository),
    }


def _sanitized_env() -> dict[str, str]:
    allowed = ('HOME', 'LANG', 'LC_ALL', 'PATH', 'SYSTEMROOT', 'TEMP', 'TMP', 'TMPDIR')
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(
        {
            'NO_PROXY': '*',
            'PYTHONNOUSERSITE': '1',
        }
    )
    return env


def _run_oracles(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = repository / 'pydantic_ai_slim'
    oracle = Path(__file__).with_name('semantic_oracles.py').resolve()
    before = _repo_fingerprint(repository)
    with tempfile.TemporaryDirectory(prefix='semantic-gap-oracles-') as temporary_directory:
        launcher = (
            'import runpy, sys; '
            f'sys.path.insert(0, {str(source_root)!r}); '
            f'runpy.run_path({str(oracle)!r}, run_name="__main__")'
        )
        command = [sys.executable, '-I', '-c', launcher]
        result = subprocess.run(
            command,
            cwd=temporary_directory,
            env=_sanitized_env(),
            capture_output=True,
            text=True,
        )
    after = _repo_fingerprint(repository)
    if before != after:
        raise SystemExit('Evaluator-owned oracles changed the Pydantic AI repository')
    if result.returncode:
        raise SystemExit(f'Evaluator-owned oracles failed:\n{result.stdout}\n{result.stderr}')
    try:
        measured = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f'Oracles returned invalid JSON: {exc}') from exc
    return measured, {
        'command': command,
        'exit_code': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
        'repository_fingerprint': before,
    }


def _verify_entrypoint(entrypoint: Any) -> None:
    if not isinstance(entrypoint, dict) or entrypoint.get('public_stream_invokes_agent') is not False:
        raise SystemExit('Register must report that the public stream does not invoke the agent factory')
    trace = entrypoint.get('trace')
    if not isinstance(trace, str) or len(trace) < 80:
        raise SystemExit('Register entrypoint needs a substantive source trace')
    if any(term not in trace for term in ('stream', 'graph.astream', 'build_agent')):
        raise SystemExit('Register entrypoint trace does not name the active and latent paths')


def _verify_gap_content(markdown: str, by_id: dict[Any, dict[str, Any]]) -> None:
    for gap_id in _PROVED_GAPS | _UNPROVED_GAPS:
        gap = by_id[gap_id]
        for field in _ROW_TEXT_FIELDS:
            value = gap.get(field)
            minimum = 5 if field in {'title', 'owner'} else 30
            if not isinstance(value, str) or len(value.strip()) < minimum:
                raise SystemExit(f'Gap {gap_id!r} needs a substantive {field!r} field')
        combined = ' '.join(str(gap[field]).lower() for field in _ROW_TEXT_FIELDS)
        missing_terms = [term for term in _GAP_TERMS[gap_id] if term not in combined]
        if missing_terms:
            raise SystemExit(f'Gap {gap_id!r} omits required concepts: {missing_terms}')
        if gap['title'].lower() not in markdown.lower():
            raise SystemExit(f'Markdown register omits the JSON row titled {gap["title"]!r}')


def _verify_gap_statuses(by_id: dict[Any, dict[str, Any]], measured: dict[str, Any]) -> None:
    for gap_id in _PROVED_GAPS:
        gap = by_id[gap_id]
        if gap.get('observed') != measured[gap_id]:
            raise SystemExit(f'Observed values for {gap_id!r} do not match evaluator oracles')
        if gap.get('status') not in {'adapted', 'intentional-change', 'non-equivalent', 'redesign'}:
            raise SystemExit(f'Proved gap {gap_id!r} has a misleading status')

    for gap_id in _UNPROVED_GAPS:
        if by_id[gap_id].get('status') not in {'cutover-blocker', 'external-owner', 'unproved'}:
            raise SystemExit(f'Unproved gap {gap_id!r} is presented as complete')

    for gap_id in _PROVED_GAPS | _UNPROVED_GAPS:
        outcome = by_id[gap_id].get('workaround_status')
        if outcome not in _WORKAROUND_OUTCOMES:
            raise SystemExit(f'Gap {gap_id!r} needs a recognized workaround_status')
        expected = _REQUIRED_WORKAROUND_OUTCOMES[gap_id]
        if outcome != expected:
            raise SystemExit(f'Gap {gap_id!r} workaround_status must be {expected!r}, got {outcome!r}')


def _verify_register(markdown: str, register: dict[str, Any], measured: dict[str, Any]) -> None:
    if len(markdown) < 500 or 'not 1:1' not in markdown.lower():
        raise SystemExit('semantic_gap_register.md needs a substantive upfront non-1:1 warning')
    required_terms = ('checkpointer', 'exactly-once', 'reconnect', 'middleware', 'owner')
    if any(term not in markdown.lower() for term in required_terms):
        raise SystemExit('semantic_gap_register.md omits required operational gaps or ownership')

    warning = register.get('warning')
    if not isinstance(warning, str) or 'not 1:1' not in warning.lower():
        raise SystemExit('semantic_gap_register.json needs an upfront non-1:1 warning')
    _verify_entrypoint(register.get('entrypoint'))
    boundaries = register.get('state_boundaries')
    if not isinstance(boundaries, list) or set(boundaries) != _STATE_BOUNDARIES:
        raise SystemExit('Register does not separate all five state boundaries')
    gaps = register.get('gaps')
    if not isinstance(gaps, list) or not all(isinstance(item, dict) for item in gaps):
        raise SystemExit('Register gaps must be a list of objects')
    by_id = {item.get('id'): item for item in gaps}
    if not (_PROVED_GAPS | _UNPROVED_GAPS) <= by_id.keys():
        raise SystemExit('Register is missing required proved or unproved gap rows')
    _verify_gap_content(markdown, by_id)
    _verify_gap_statuses(by_id, measured)


def main() -> None:
    """Run trusted oracles and verify declarative agent output against them."""
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    repository = args.pydantic_repo.resolve()
    if output_dir == repository or repository in output_dir.parents:
        raise SystemExit('Evaluation output must be outside the Pydantic AI repository')
    source_root = repository / 'pydantic_ai_slim'
    if not source_root.is_dir():
        raise SystemExit(f'Not a Pydantic AI checkout: {repository}')

    sys.path.insert(0, str(source_root))
    measured, oracle_evidence = _run_oracles(repository)
    oracle_provenance = measured.pop('provenance')
    verified = _measure_provenance(repository, oracle_provenance)
    if args.write_evidence:
        output_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = output_dir / 'evidence.json'
        evidence = {**verified, 'oracle_observations': measured}
        _write_new_json(evidence_path, evidence)
        print(f'Wrote evaluator-measured evidence: {evidence_path}')
        return

    markdown = _read_text(output_dir / 'semantic_gap_register.md')
    register = _load_json(output_dir / 'semantic_gap_register.json')
    claimed = _load_json(output_dir / 'evidence.json')

    expected_claims = {**verified, 'oracle_observations': measured}
    for key, expected in expected_claims.items():
        if claimed.get(key) != expected:
            raise SystemExit(f'evidence.json {key!r} does not match evaluator measurements')

    _verify_register(markdown, register, measured)
    verified['oracles'] = measured
    verified['oracle_execution'] = oracle_evidence
    verified_path = output_dir / 'verified_evidence.json'
    _write_new_json(verified_path, verified)
    print(f'Verified semantic eval: {verified_path}')


if __name__ == '__main__':
    main()
