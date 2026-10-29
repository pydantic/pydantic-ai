from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import check_api_compatibility
import pytest
from check_api_compatibility import (
    Finding,
    collection_entry_changes,
    emit_annotation,
    literal_member_changes,
    load_allowlist,
    load_waivers,
    parse_findings,
    run_griffe,
)

WAIVER = {
    'against': 'v2.32.1',
    'reason': 'The version policy permits this change.',
    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/123',
}


def _write_allowlist(tmp_path: Path, allowlist: object = None) -> Path:
    path = tmp_path / 'allowlist.json'
    path.write_text(json.dumps(allowlist or {}), encoding='utf-8')
    return path


def _write_waivers(tmp_path: Path, waivers: Mapping[str, object] | None = None) -> Path:
    """Write a directory with a file per named waiver."""
    directory = tmp_path / 'waivers'
    directory.mkdir()
    for name, waiver in (waivers or {}).items():
        (directory / name).write_text(json.dumps(waiver), encoding='utf-8')
    return directory


def _config(tmp_path: Path, allowlist: object = None, waivers: Mapping[str, object] | None = None) -> list[str]:
    """Write an allowlist and waivers, and return the arguments naming them."""
    return [
        '--allowlist',
        str(_write_allowlist(tmp_path, allowlist)),
        '--waivers',
        str(_write_waivers(tmp_path, waivers)),
    ]


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


def test_load_waivers_treats_a_missing_directory_as_empty(tmp_path: Path):
    assert load_waivers(tmp_path / 'waivers') == {}


def test_load_allowlist_and_waivers(tmp_path: Path):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New')
    allowlist = _write_allowlist(
        tmp_path,
        {
            'model_name_literals': [{'path': 'models/foo.py', 'symbol': 'LatestFooModelNames'}],
            'growing_collections': [{'path': 'config.py', 'symbol': 'SETTINGS'}],
        },
    )
    waivers = _write_waivers(tmp_path, {f'{finding.fingerprint}.json': WAIVER})

    assert {fingerprint: waiver.against for fingerprint, waiver in load_waivers(waivers).items()} == {
        finding.fingerprint: 'v2.32.1'
    }
    model_name_literals, growing_collections = load_allowlist(allowlist)
    assert model_name_literals == {('models/foo.py', 'LatestFooModelNames')}
    assert growing_collections == {('config.py', 'SETTINGS')}


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
    config = _config(tmp_path, {'model_name_literals': [{'path': 'models.py', 'symbol': 'Names'}] if listed else []})

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

    if annotation == '::error':
        with pytest.raises(SystemExit, match='1'):
            check_api_compatibility.main()
    else:
        check_api_compatibility.main()

    assert annotation in capsys.readouterr().out


_COLLECTION_ENTRY_CHANGES: list[tuple[str, tuple[list[str], list[str]] | None]] = [
    ("{'a': 'x'} -> {'a': 'x', 'b': 'y'}", (["'b': 'y'"], [])),
    ("{'a': 'x', 'b': 'y'} -> {'b': 'y', 'a': 'x'}", ([], [])),
    ("{'a': 'x', 'b': 'y'} -> {'a': 'x'}", ([], ["'b': 'y'"])),
    ("{'a': 'x'} -> {'a': 'z'}", (["'a': 'z'"], ["'a': 'x'"])),
    ("{'a -> b': 'x'} -> {'a -> b': 'x', 'c': 'y'}", (["'c': 'y'"], [])),
    ("frozenset({'a'}) -> frozenset({'a', 'b'})", (["'b'"], [])),
    ("{'a'} -> {'b'}", (["'b'"], ["'a'"])),
    ("{'a': 'x'} -> frozenset({'a'})", None),
    ("{'a': Name} -> {'a': Name, 'b': Other}", None),
    ("['a'] -> ['a', 'b']", None),
    ("frozenset({'a'}, key=1) -> frozenset({'a', 'b'})", None),
    ("{'a': 'x' -> {'a': 'x', 'b': 'y'}", None),
    ('{c.name: c for c in (A, B)} -> {c.name: c for c in (A, C, B)}', (['C'], [])),
    ('{c.name: c for c in (A, B)} -> {c.name: c for c in (A,)}', ([], ['B'])),
    ('{c for c in [A]} -> {c for c in [A, B]}', (['B'], [])),
    ('{c.name: c for c in (A, B)} -> {c.id: c for c in (A, B, C)}', None),
    ('{c.name: c for c in (A, B) if c.ok} -> {c.name: c for c in (A, B, C)}', None),
    ('{c.name: c for c in (A, B)} -> {c for c in (A, B, C)}', None),
    ('{c.name: c for c in TYPES} -> {c.name: c for c in TYPES}', None),
]


@pytest.mark.parametrize('values, changes', _COLLECTION_ENTRY_CHANGES)
def test_collection_entry_changes(values: str, changes: tuple[list[str], list[str]] | None):
    assert collection_entry_changes(f'SETTINGS: Attribute value was changed: {values}') == changes


@pytest.mark.parametrize(
    'message, listed, annotation',
    [
        (
            "SETTINGS: Attribute value was changed: {'a': 'x'} -> {'a': 'x', 'b': 'y'}",
            True,
            "::notice file=config.py,line=12::SETTINGS: entries added: [\"'b': 'y'\"]. "
            'Additions to a growing collection need no waiver.\n',
        ),
        ("SETTINGS: Attribute value was changed: {'a': 'x'} -> {'a': 'x', 'b': 'y'}", False, '::error'),
        ("SETTINGS: Attribute value was changed: {'a': 'x', 'b': 'y'} -> {'a': 'x'}", True, '::error'),
        ("SETTINGS: Attribute value was changed: {'a': 'x'} -> {'a': 'z'}", True, '::error'),
        ('SETTINGS: Public object was removed', True, '::error'),
    ],
)
def test_main_allows_additions_to_listed_growing_collections(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    message: str,
    listed: bool,
    annotation: str,
):
    finding = Finding('pydantic_clai2', 'config.py', 12, message)
    config = _config(tmp_path, {'growing_collections': [{'path': 'config.py', 'symbol': 'SETTINGS'}] if listed else []})

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_clai2': 'src/pydantic_clai2'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

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
    config = _config(
        tmp_path,
        {'model_name_literals': [{'path': 'models.py', 'symbol': 'Names'}]},
        {f'{finding.fingerprint}.json': WAIVER},
    )

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    assert f'::error::Unused API compatibility waiver for v2.32.1: {finding.fingerprint}' in capsys.readouterr().out


# A waiver against an older release does not cover the same finding against the current one.
@pytest.mark.parametrize('waived_against', [None, 'v2.31.0'])
def test_main_fails_for_unapproved_breakage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str], waived_against: str | None
):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New')
    waivers = {} if waived_against is None else {f'{finding.fingerprint}.json': {**WAIVER, 'against': waived_against}}
    config = _config(tmp_path, waivers=waivers)

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        assert (package, search_path, against) == ('pydantic_ai', 'pydantic_ai_slim', 'v2.32.1')
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    output = capsys.readouterr().out
    assert '::error file=messages.py,line=12::' in output
    assert f'with a waiver in `{tmp_path}/waivers/{finding.fingerprint}.json`' in output


def test_main_warns_for_approved_breakage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'RequestPart: Attribute value was changed: Old -> New')
    config = _config(tmp_path, waivers={f'{finding.fingerprint}.json': WAIVER})

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [finding]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_ai': 'pydantic_ai_slim'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

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
    config = _config(tmp_path)

    def run_griffe(package: str, search_path: str, against: str) -> list[Finding]:
        return [*skipped, *reported]

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {'pydantic_clai2': 'src/pydantic_clai2'})
    monkeypatch.setattr(check_api_compatibility, 'run_griffe', run_griffe)
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    output = capsys.readouterr().out
    assert [line.split('::')[1] for line in output.splitlines()] == [
        f'error file={finding.path},line={finding.line}' for finding in reported
    ]


def test_main_rejects_unused_waiver_for_current_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    # A waiver against an older release is kept but not required.
    config = _config(
        tmp_path, waivers={f'{"0" * 64}.json': WAIVER, f'{"1" * 64}.json': {**WAIVER, 'against': 'v2.31.0'}}
    )

    monkeypatch.setattr(check_api_compatibility, 'PACKAGES', {})
    monkeypatch.setattr('sys.argv', ['check_api_compatibility.py', '--against', 'v2.32.1', *config])

    with pytest.raises(SystemExit, match='1'):
        check_api_compatibility.main()

    assert '::error::Unused API compatibility waiver for v2.32.1' in capsys.readouterr().out


_FINGERPRINT = 'a' * 64

_INVALID_ALLOWLISTS: list[tuple[object, str]] = [
    # An entry under the old layout, from a PR opened before waivers moved to their own files.
    ({'allowed_breakages': []}, 'extra_forbidden'),
    ({'model_name_literals': {}}, 'list_type'),
    ({'growing_collections': [{'path': 'config.py'}]}, 'symbol'),
]


@pytest.mark.parametrize('allowlist, error', _INVALID_ALLOWLISTS)
def test_load_allowlist_rejects_invalid_entries(tmp_path: Path, allowlist: object, error: str):
    path = _write_allowlist(tmp_path, allowlist)

    with pytest.raises(ValueError, match=error):
        load_allowlist(path)


_INVALID_WAIVERS: list[tuple[str, object, str]] = [
    (f'{_FINGERPRINT}.json', {}, 'against'),
    ('bad.json', WAIVER, 'must be named'),
    (_FINGERPRINT, WAIVER, 'must be named'),
    (f'{_FINGERPRINT.upper()}.json', WAIVER, 'must be named'),
    (f'{_FINGERPRINT}.json', {**WAIVER, 'fingerprint': _FINGERPRINT}, 'extra_forbidden'),
    (f'{_FINGERPRINT}.json', {**WAIVER, 'reason': ''}, 'string_too_short'),
    (f'{_FINGERPRINT}.json', {**WAIVER, 'pull_request': 'not-a-pr'}, 'string_pattern_mismatch'),
]


@pytest.mark.parametrize('name, waiver, error', _INVALID_WAIVERS)
def test_load_waivers_rejects_invalid_files(tmp_path: Path, name: str, waiver: object, error: str):
    directory = _write_waivers(tmp_path, {name: waiver})

    with pytest.raises(ValueError, match=error):
        load_waivers(directory)


def test_emit_annotation_escapes_workflow_commands(capsys: pytest.CaptureFixture[str]):
    finding = Finding('pydantic_ai', 'messages.py', 12, 'message')

    emit_annotation('error', finding, '100%\nfailed')

    assert capsys.readouterr().out == '::error file=messages.py,line=12::100%25%0Afailed\n'
