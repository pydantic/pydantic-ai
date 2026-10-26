from __future__ import annotations

import json
import subprocess
from pathlib import Path

import check_api_compatibility
import pytest
from check_api_compatibility import (
    Finding,
    emit_annotation,
    literal_member_changes,
    load_allowlist,
    parse_findings,
    run_griffe,
)


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


@pytest.mark.subprocess(reason='parses the output stream and exit code of a real `griffecli` process')
def test_run_griffe_reports_real_griffe_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = tmp_path / 'fixture_pkg' / '__init__.py'
    module.parent.mkdir()
    module.write_text('from ._names import ModelNames as ModelNames\n\n\ndef greet(name): ...\n', encoding='utf-8')
    # Re-exported from a private module, so only `griffe-public-redundant-aliases` makes Griffe check it.
    names = module.parent / '_names.py'
    names.write_text("from typing import Literal\n\nModelNames = Literal['a']\n", encoding='utf-8')
    git = [
        'git',
        '-C',
        str(tmp_path),
        '-c',
        'user.name=t',
        '-c',
        'user.email=t@example.com',
        '-c',
        'commit.gpgsign=false',
        '-c',
        'tag.gpgSign=false',
        '-c',
        'core.hooksPath=/dev/null',
    ]
    subprocess.run([*git, 'init', '-q'], check=True)
    subprocess.run([*git, 'add', '.'], check=True)
    subprocess.run([*git, 'commit', '-q', '-m', 'release'], check=True)
    subprocess.run([*git, 'tag', 'v1'], check=True)
    monkeypatch.chdir(tmp_path)

    assert run_griffe('fixture_pkg', '.', 'v1') == []

    module.write_text(
        'from ._names import ModelNames as ModelNames\n\n\ndef greet(name, greeting): ...\n', encoding='utf-8'
    )
    names.write_text("from typing import Literal\n\nModelNames = Literal['a', 'b']\n", encoding='utf-8')

    assert run_griffe('fixture_pkg', '.', 'v1') == [
        Finding(
            'fixture_pkg',
            'fixture_pkg/_names.py',
            3,
            "ModelNames: Attribute value was changed: Literal['a'] -> Literal['a', 'b']",
        ),
        Finding('fixture_pkg', 'fixture_pkg/__init__.py', 4, 'greet(greeting): Parameter was added as required'),
    ]

    with pytest.raises(SystemExit, match='griffecli check missing_pkg failed with exit code 1'):
        run_griffe('missing_pkg', '.', 'v1')


def test_load_allowlist(tmp_path: Path):
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
                ],
                'model_name_literals': [{'path': 'models/foo.py', 'symbol': 'LatestFooModelNames'}],
            }
        ),
        encoding='utf-8',
    )

    waivers, model_name_literals = load_allowlist(path)
    assert waivers[('v2.32.1', finding.fingerprint)].reason == (
        'The version policy permits adding public union members.'
    )
    assert model_name_literals == {('models/foo.py', 'LatestFooModelNames')}


_LITERAL_MEMBER_CHANGES: list[tuple[str, tuple[set[str], set[str]] | None]] = [
    ("Literal['a'] -> Literal['a', 'b']", ({'b'}, set())),
    ("str | Literal['a'] -> str | Literal['a', 'b']", ({'b'}, set())),
    ("TypeAliasType('Names', Literal['p:a']) -> TypeAliasType('Names', Literal['p:a', 'p:b'])", ({'p:b'}, set())),
    ("str | AllModels -> str | AllModels | Literal['x']", ({'x'}, set())),
    ("Literal['a', 'b'] -> Literal['a']", (set(), {'b'})),
    ("Literal['a', 'b'] -> Literal['b', 'a']", (set(), set())),
    ("Literal['a -> b'] -> Literal['a -> b', 'c']", ({'c'}, set())),
    ("str | Literal['a'] -> Literal['a']", None),
    ("str | Literal['a'] -> unset", None),
    ("str | Literal['a'] -> str | Literal['a'] | int", None),
    ("Literal['a'] -> Literal['a', 1]", None),
    ("TypeAliasType('Old', Literal['a']) -> TypeAliasType('New', Literal['a'])", None),
    ("TypeAliasType('Names', Literal['a'], type_params=()) -> TypeAliasType('Names', Literal['a', 'b'])", None),
    ("Literal['a' -> Literal['a', 'b']", None),
]


@pytest.mark.parametrize('values, changes', _LITERAL_MEMBER_CHANGES)
def test_literal_member_changes(values: str, changes: tuple[set[str], set[str]] | None):
    assert literal_member_changes(f'Names: Attribute value was changed: {values}') == changes


def test_literal_member_changes_ignores_other_findings():
    assert literal_member_changes('Names: Public object was removed') is None


@pytest.mark.parametrize(
    'message, listed, annotation',
    [
        (
            "Names: Attribute value was changed: Literal['a'] -> Literal['a', 'b']",
            True,
            "::notice file=models.py,line=12::Names: model names added: ['b'], removed: []. "
            'Model-name alias changes need no waiver.\n',
        ),
        (
            "Names: Attribute value was changed: Literal['a', 'b'] -> Literal['a']",
            True,
            "::warning file=models.py,line=12::Names: model names added: [], removed: ['b']. "
            'Model-name alias changes need no waiver. Removing a name from an alias without a `str |` arm is a '
            'permitted compatibility impact: add the `compatibility impact` label and a release note.\n',
        ),
        ("Names: Attribute value was changed: Literal['a'] -> Literal['a', 'b']", False, '::error'),
        ("Names: Attribute value was changed: str | Literal['a'] -> Literal['a']", True, '::error'),
        ('Names: Public object was removed', True, '::error'),
    ],
)
def test_main_allows_listed_model_name_literal_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    message: str,
    listed: bool,
    annotation: str,
):
    finding = Finding('pydantic_ai', 'models.py', 12, message)
    path = tmp_path / 'allowlist.json'
    path.write_text(
        json.dumps(
            {
                'allowed_breakages': [],
                'model_name_literals': [{'path': 'models.py', 'symbol': 'Names'}] if listed else [],
            }
        ),
        encoding='utf-8',
    )

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', '--allowlist', str(path)])

    if annotation == '::error':
        with pytest.raises(SystemExit, match='1'):
            check_api_compatibility.main()
    else:
        check_api_compatibility.main()

    assert annotation in capsys.readouterr().out


def test_main_rejects_waiver_for_listed_model_name_literal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    finding = Finding(
        'pydantic_ai', 'models.py', 12, "Names: Attribute value was changed: Literal['a'] -> Literal['a', 'b']"
    )
    path = tmp_path / 'allowlist.json'
    path.write_text(
        json.dumps(
            {
                'allowed_breakages': [
                    {
                        'against': 'v2.32.1',
                        'fingerprint': finding.fingerprint,
                        'reason': 'Adds a model name.',
                        'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/123',
                    }
                ],
                'model_name_literals': [{'path': 'models.py', 'symbol': 'Names'}],
            }
        ),
        encoding='utf-8',
    )

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', '--allowlist', str(path)])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    assert f'::error::Unused API compatibility waiver for v2.32.1: {finding.fingerprint}' in capsys.readouterr().out


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


def test_main_skips_the_clai2_web_ui_modules(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    plugins = 'src/pydantic_clai2/pydantic_clai2/builtin_plugins'
    skipped = [
        Finding('pydantic_clai2', f'{plugins}/web/__init__.py', 1, 'WebPlugin: Public object was removed'),
        Finding('pydantic_clai2', f'{plugins}/web/service.py', 2, 'WebService.host: Public object was removed'),
        Finding('pydantic_clai2', f'{plugins}/workspaces/sessions.py', 3, '<module>: Public object was removed'),
    ]
    reported = [
        Finding('pydantic_clai2', f'{plugins}/web_search.py', 4, 'WebSearch: Public object was removed'),
        Finding(
            'pydantic_clai2',
            'src/pydantic_clai2/pydantic_clai2/plugins/__init__.py',
            5,
            'Plugin: Public object was removed',
        ),
    ]
    path = tmp_path / 'allowlist.json'
    path.write_text('{"allowed_breakages": []}', encoding='utf-8')

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [*skipped, *reported]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_clai2': 'src/pydantic_clai2'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', '--allowlist', str(path)])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    output = capsys.readouterr().out
    assert [line.split('::')[1] for line in output.splitlines()] == [
        f'error file={finding.path},line={finding.line}' for finding in reported
    ]


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
def test_load_allowlist_rejects_invalid_entries(tmp_path: Path, value: object, error: str):
    path = tmp_path / 'allowlist.json'
    path.write_text(json.dumps(value), encoding='utf-8')

    with pytest.raises(ValueError, match=error):
        load_allowlist(path)


def test_emit_annotation_escapes_workflow_commands(capsys: pytest.CaptureFixture[str]):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'message')

    emit_annotation('error', finding, '100%\nfailed')

    assert capsys.readouterr().out == '::error file=messages.py,line=12::100%25%0Afailed\n'
