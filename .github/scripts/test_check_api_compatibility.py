from __future__ import annotations

import json
from pathlib import Path

import check_api_compatibility
import pytest
from check_api_compatibility import Finding, emit_annotation, load_waivers, parse_findings


def test_fingerprint_identifies_the_exact_finding():
    first = Finding('pydantic_ai', 'old.py', 1, 'Thing.method(arg): Parameter was added as required')
    moved = Finding('pydantic_ai', 'new.py', first.line, first.message)
    shifted = Finding(first.package, first.path, 99, first.message)
    other_package = Finding('pydantic_graph', first.path, first.line, first.message)

    assert first.fingerprint != moved.fingerprint
    assert first.fingerprint == shifted.fingerprint
    assert first.fingerprint != other_package.fingerprint


def test_parse_findings():
    output = (
        'pydantic_ai/messages.py:12: RequestPart: Attribute value was changed: Old -> New\n'
        'pydantic_ai/tools.py:34: Tool.run(ctx): Parameter was added as required\n'
    )

    assert parse_findings('pydantic_ai', output) == [
        Finding('pydantic_ai', 'pydantic_ai/messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New'),
        Finding('pydantic_ai', 'pydantic_ai/tools.py', 34, 'Tool.run(ctx): Parameter was added as required'),
    ]


def test_parse_findings_rejects_unknown_output():
    with pytest.raises(ValueError, match='Unexpected Griffe output'):
        parse_findings('pydantic_ai', 'not a finding')


def test_load_waivers(tmp_path: Path):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New')
    path = tmp_path / 'allowlist.json'
    path.write_text(
        json.dumps(
            {
                'allowed_breakages': [
                    {
                        'against': 'v2.32.1',
                        'fingerprint': finding.fingerprint,
                        'reason': 'The version policy permits adding public union members.',
                        'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/123',
                    }
                ]
            }
        ),
        encoding='utf-8',
    )

    assert load_waivers(path)[('v2.32.1', finding.fingerprint)].reason == (
        'The version policy permits adding public union members.'
    )


def test_main_fails_for_unapproved_breakage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New')
    path = tmp_path / 'allowlist.json'
    path.write_text('{"allowed_breakages": []}', encoding='utf-8')

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        assert (package, search_path, against) == ('pydantic_ai', 'pydantic_ai_slim', 'v2.32.1')
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', '--allowlist', str(path)])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    output = capsys.readouterr().out
    assert '::error file=messages.py,line=12::' in output
    assert finding.fingerprint in output


def test_main_warns_for_approved_breakage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New')
    path = tmp_path / 'allowlist.json'
    path.write_text(
        json.dumps(
            {
                'allowed_breakages': [
                    {
                        'against': 'v2.32.1',
                        'fingerprint': finding.fingerprint,
                        'reason': 'The version policy permits this change.',
                        'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/123',
                    }
                ]
            }
        ),
        encoding='utf-8',
    )

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', '--allowlist', str(path)])

    check_api_compatibility.main()

    output = capsys.readouterr().out
    assert '::warning file=messages.py,line=12::' in output
    assert 'Allowed by https://github.com/pydantic/pydantic-ai/pull/123' in output


def test_main_rejects_unused_waiver_for_current_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    path = tmp_path / 'allowlist.json'
    path.write_text(
        json.dumps(
            {
                'allowed_breakages': [
                    {
                        'against': 'v2.32.1',
                        'fingerprint': '0' * 64,
                        'reason': 'The version policy permits this change.',
                        'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/123',
                    }
                ]
            }
        ),
        encoding='utf-8',
    )

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {})
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', '--allowlist', str(path)])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    assert '::error::Unused API compatibility waiver for v2.32.1' in capsys.readouterr().out


_INVALID_ALLOWLISTS: list[tuple[object, str]] = [
    ({}, 'allowed_breakages'),
    ({'allowed_breakages': {}}, 'list_type'),
    ({'allowed_breakages': [{}]}, 'fingerprint'),
    (
        {
            'allowed_breakages': [
                {
                    'against': 'v2.32.1',
                    'fingerprint': 'bad',
                    'reason': 'reason',
                    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/1',
                }
            ]
        },
        'string_pattern_mismatch',
    ),
    (
        {
            'allowed_breakages': [
                {
                    'against': 'v2.32.1',
                    'fingerprint': '0' * 64,
                    'reason': '',
                    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/1',
                }
            ]
        },
        'string_too_short',
    ),
    (
        {
            'allowed_breakages': [
                {'against': 'v2.32.1', 'fingerprint': '0' * 64, 'reason': 'reason', 'pull_request': 'not-a-pr'}
            ]
        },
        'string_pattern_mismatch',
    ),
    (
        {
            'allowed_breakages': [
                {
                    'against': 'v2.32.1',
                    'fingerprint': '0' * 64,
                    'reason': 'reason',
                    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/1',
                },
                {
                    'against': 'v2.32.1',
                    'fingerprint': '0' * 64,
                    'reason': 'reason',
                    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/2',
                },
            ]
        },
        'duplicate waiver',
    ),
]


@pytest.mark.parametrize('value, error', _INVALID_ALLOWLISTS)
def test_load_waivers_rejects_invalid_entries(tmp_path: Path, value: object, error: str):
    path = tmp_path / 'allowlist.json'
    path.write_text(json.dumps(value), encoding='utf-8')

    with pytest.raises(ValueError, match=error):
        load_waivers(path)


def test_emit_annotation_escapes_workflow_commands(capsys: pytest.CaptureFixture[str]):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'message')

    emit_annotation('error', finding, '100%\nfailed')

    assert capsys.readouterr().out == '::error file=messages.py,line=12::100%25%0Afailed\n'
