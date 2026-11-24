"""`scripts/durable_exec_contract.py --touched` names the manifest entries a pull request's changed lines touch.

CI labels a pull request `needs-durable-exec-review` and gives CI Review the entries when the set is not empty, and
`--signed-off` keeps the label off while a reviewer's sign-off marker carries the pull request's current fingerprint.
The classifier reads the pull request through `gh api`; these tests pass the files listing, the comments, the
permissions and the sources it would fetch, so they are not VCR tests.
"""

from __future__ import annotations

import difflib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from inline_snapshot import snapshot

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))

import durable_exec_contract as generator
from durable_exec_contract import ChangedFile, Manifest

CORE = 'pydantic_ai_slim/pydantic_ai'
MODELS = f'{CORE}/models/__init__.py'
SETTINGS = f'{CORE}/settings.py'

MODELS_SOURCE = '''\
from typing import Literal, TypeAlias

KnownModelName: TypeAlias = Literal['a', 'b']


class Model:
    """A model."""

    def __init__(self, label: str) -> None:
        self.label = label

    @property
    def profile(self) -> str:
        # The profile is the label.
        return self.label

    def request(self) -> str:
        return 'request'


def infer_model(name: str) -> Model:
    return Model(name)
'''

SETTINGS_SOURCE = """\
DEFAULT = 0


class ModelSettings:
    timeout = 1"""


def _entry(kind: str) -> dict[str, list[str]]:
    return {
        'engines': ['temporal'],
        'kinds': [kind],
        'sites': ['pydantic_ai_slim/pydantic_ai/durable_exec/x.py::<module>'],
    }


MANIFEST: Manifest = {
    MODELS: {
        'KnownModelName': _entry('annotation_only'),
        'Model': _entry('import'),
        'Model.label': _entry('attribute_read'),
        'Model.profile': _entry('attribute_read'),
        'infer_model': _entry('import'),
    },
    SETTINGS: {'DEFAULT': _entry('import'), 'ModelSettings': _entry('import')},
}


def _patch(base: str, head: str) -> str:
    """The patch GitHub lists for a change from `base` to `head`: the hunks, without the file header."""
    return '\n'.join(list(difflib.unified_diff(base.splitlines(), head.splitlines(), lineterm=''))[2:])


@dataclass
class _Sources:
    """The core file sources at one side of a pull request, recording which ones are read."""

    sources: dict[str, str]
    read: list[str] = field(default_factory=list[str])

    def __call__(self, path: str) -> str:
        self.read.append(path)
        return self.sources[path]


def _touched(*files: ChangedFile, base: _Sources, head: _Sources) -> dict[str, list[str]]:
    touched = generator.touched_entries(MANIFEST, files, base, head)
    return {core_file: list(entries) for core_file, entries in touched.items()}


@pytest.mark.parametrize(
    'head, touched',
    [
        pytest.param(MODELS_SOURCE.replace("'request'", "'response'"), ['Model'], id='method'),
        pytest.param(
            MODELS_SOURCE.replace('return self.label', 'return self.label.upper()'),
            ['Model', 'Model.profile'],
            id='property',
        ),
        pytest.param(
            MODELS_SOURCE.replace('= label', '= label.strip()'), ['Model', 'Model.label'], id='instance-attribute'
        ),
        pytest.param(MODELS_SOURCE.replace('Model(name)', 'Model(name.lower())'), ['infer_model'], id='function'),
        pytest.param(
            MODELS_SOURCE.replace("    def request(self) -> str:\n        return 'request'\n\n", ''),
            ['Model'],
            id='removed-method',
        ),
        pytest.param(MODELS_SOURCE.replace('A model.', 'A language model.'), [], id='docstring'),
        pytest.param(MODELS_SOURCE.replace('# The profile is the label.', '# The label.'), [], id='comment'),
        pytest.param(MODELS_SOURCE.replace("'b']", "'b', 'c']"), [], id='annotation-only-entry'),
        pytest.param(MODELS_SOURCE.replace('Literal, TypeAlias', 'Final, Literal, TypeAlias'), [], id='import'),
        pytest.param(MODELS_SOURCE + '\n\nclass Other:\n    pass\n', [], id='new-definition'),
    ],
)
def test_a_changed_line_touches_the_entries_of_the_definitions_around_it(head: str, touched: list[str]):
    """Removed lines are placed in the merge-base source and added ones in the head source; only gating entries count."""
    base_sources = _Sources({MODELS: MODELS_SOURCE})
    head_sources = _Sources({MODELS: head})
    files = [
        ChangedFile(MODELS, 'modified', patch=_patch(MODELS_SOURCE, head)),
        ChangedFile('tests/models/test_model.py', 'modified', patch='@@ -1 +1 @@\n-a\n+b'),
    ]

    result = _touched(*files, base=base_sources, head=head_sources)

    assert result == ({MODELS: touched} if touched else {})
    assert set(base_sources.read + head_sources.read) <= {MODELS}


@pytest.mark.parametrize(
    'changed, head',
    [
        pytest.param(ChangedFile(MODELS, 'removed'), MODELS_SOURCE, id='removed'),
        pytest.param(
            ChangedFile(f'{CORE}/models/base.py', 'renamed', previous_filename=MODELS), MODELS_SOURCE, id='renamed'
        ),
        pytest.param(ChangedFile(MODELS, 'modified'), MODELS_SOURCE, id='patch-left-out'),
        pytest.param(
            ChangedFile(MODELS, 'modified', patch='@@ -6 +6 @@\n-class Model:\n+class Mdl:'), MODELS_SOURCE, id='stale'
        ),
        pytest.param(
            ChangedFile(MODELS, 'modified', patch='@@ -40 +40 @@\n-pass\n+pass'), MODELS_SOURCE, id='past-the-end'
        ),
        pytest.param(
            ChangedFile(MODELS, 'modified', patch='@@ -22,0 +23 @@\n+def ('), MODELS_SOURCE + 'def (', id='unparsable'
        ),
    ],
)
def test_a_core_file_whose_lines_cannot_be_placed_touches_all_its_gating_entries(changed: ChangedFile, head: str):
    """A removed or renamed core file leaves the path the engines import it from; an unplaced change could be anywhere."""
    assert _touched(changed, base=_Sources({MODELS: MODELS_SOURCE}), head=_Sources({MODELS: head})) == snapshot(
        {MODELS: ['Model', 'Model.label', 'Model.profile', 'infer_model']}
    )


@pytest.mark.parametrize(
    'status, previous_filename',
    [('added', None), ('copied', f'{CORE}/old_settings.py'), ('renamed', f'{CORE}/old_settings.py')],
)
def test_a_file_new_at_a_core_path_is_read_on_the_head_side_only(
    status: generator.FileStatus, previous_filename: str | None
):
    """An added, copied or renamed file's patch compares it with another file, so its removed lines are not placed."""
    head = _Sources({SETTINGS: SETTINGS_SOURCE})
    base = _Sources({})
    changed = ChangedFile(
        SETTINGS, status, previous_filename=previous_filename, patch='@@ -5 +5 @@\n-    a = 1\n+    timeout = 1'
    )

    touched = generator.touched_entries(MANIFEST, [changed], base, head)

    assert list(touched[SETTINGS]) == ['ModelSettings']
    assert (base.read, head.read) == ([], [SETTINGS])


def test_a_no_newline_marker_belongs_to_neither_side():
    """`\\ No newline at end of file` does not shift the line numbers of the lines after it."""
    patch = '@@ -4,2 +4,2 @@\n class ModelSettings:\n-    timeout = 1\n\\ No newline at end of file\n+    timeout = 2\n\\ No newline at end of file'

    touched = _touched(
        ChangedFile(SETTINGS, 'modified', patch=patch),
        base=_Sources({SETTINGS: SETTINGS_SOURCE}),
        head=_Sources({SETTINGS: SETTINGS_SOURCE.replace('= 1', '= 2')}),
    )

    assert touched == {SETTINGS: ['ModelSettings']}


REPO = 'pydantic/pydantic-ai'
BASE_SHA = 'b' * 40
HEAD_SHA = 'c' * 40
MERGE_BASE = 'a' * 40


@dataclass
class _FakeGh:
    """`gh api`, answering from `responses` by endpoint and recording each call's arguments."""

    responses: dict[str, str]
    calls: list[tuple[str, ...]] = field(default_factory=list[tuple[str, ...]])

    def __call__(self, *arguments: str) -> str:
        self.calls.append(arguments)
        return self.responses[next(argument for argument in arguments if argument.startswith('repos/'))]


def _fake_gh(merge_base: str = MERGE_BASE, extra_files: int = 0) -> _FakeGh:
    head = MODELS_SOURCE.replace("'request'", "'response'")
    models = {'filename': MODELS, 'status': 'modified', 'patch': _patch(MODELS_SOURCE, head)}
    others = [{'filename': f'docs/{index}.md', 'status': 'added'} for index in range(extra_files)]
    return _FakeGh(
        {
            f'repos/{REPO}/compare/{BASE_SHA}...{HEAD_SHA}': f'{merge_base}\n',
            f'repos/{REPO}/pulls/42/files?per_page=100': json.dumps(
                [[models], [{'filename': 'README.md', 'status': 'modified'}, *others]]
            ),
            f'repos/{REPO}/contents/{MODELS}?ref={MERGE_BASE}': MODELS_SOURCE,
            f'repos/{REPO}/contents/{MODELS}?ref={HEAD_SHA}': head,
        }
    )


@pytest.fixture
def manifest_root(tmp_path: Path) -> Path:
    (tmp_path / 'scripts').mkdir()
    (tmp_path / generator.MANIFEST_PATH).write_text(json.dumps(MANIFEST), encoding='utf-8')
    return tmp_path


def test_touched_reads_the_pull_request_through_gh(manifest_root: Path, capsys: pytest.CaptureFixture[str]):
    """It reads the merge base, every page of the files listing, and each listed core file at the merge base and head."""
    gh = _fake_gh()

    assert generator.main(['--touched', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=gh) == 0

    assert json.loads(capsys.readouterr().out) == {MODELS: {'Model': MANIFEST[MODELS]['Model']}}
    assert gh.calls == snapshot(
        [
            (
                'repos/pydantic/pydantic-ai/compare/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb...cccccccccccccccccccccccccccccccccccccccc',
                '--jq',
                '.merge_base_commit.sha',
            ),
            ('--paginate', '--slurp', 'repos/pydantic/pydantic-ai/pulls/42/files?per_page=100'),
            (
                '-H',
                'Accept: application/vnd.github.raw+json',
                'repos/pydantic/pydantic-ai/contents/pydantic_ai_slim/pydantic_ai/models/__init__.py?ref=cccccccccccccccccccccccccccccccccccccccc',
            ),
            (
                '-H',
                'Accept: application/vnd.github.raw+json',
                'repos/pydantic/pydantic-ai/contents/pydantic_ai_slim/pydantic_ai/models/__init__.py?ref=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            ),
        ]
    )


@pytest.mark.parametrize(
    'arguments, manifest',
    [
        pytest.param([REPO, '42', 'main', HEAD_SHA], MANIFEST, id='base-is-not-a-commit'),
        pytest.param([REPO, '42', BASE_SHA, HEAD_SHA[:12]], MANIFEST, id='head-is-abbreviated'),
        pytest.param([REPO, '0', BASE_SHA, HEAD_SHA], MANIFEST, id='pull-request-zero'),
        pytest.param(['pydantic', '42', BASE_SHA, HEAD_SHA], MANIFEST, id='repository-without-owner'),
        pytest.param([REPO, '42', BASE_SHA, HEAD_SHA], {f'{CORE}/../x.py': {}}, id='core-file-outside-the-tree'),
    ],
)
def test_touched_requests_nothing_for_an_unexpected_input(
    manifest_root: Path, arguments: list[str], manifest: Manifest, capsys: pytest.CaptureFixture[str]
):
    """Every value that reaches a request URL is checked before the first request."""
    (manifest_root / generator.MANIFEST_PATH).write_text(json.dumps(manifest), encoding='utf-8')
    gh = _fake_gh()

    with pytest.raises(SystemExit) as exit_info:
        generator.main(['--touched', *arguments], root=manifest_root, gh=gh)

    assert exit_info.value.code == 2
    assert 'refusing to request anything for' in capsys.readouterr().err
    assert gh.calls == []


@pytest.mark.parametrize(
    'gh, error',
    [
        pytest.param(
            _fake_gh(merge_base='null'), "cannot classify: merge base 'null', 2 files listed\n", id='merge-base'
        ),
        pytest.param(
            _fake_gh(extra_files=2998),
            "cannot classify: merge base 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 3000 files listed\n",
            id='file-cap',
        ),
    ],
)
def test_touched_fails_when_the_pull_request_cannot_be_classified(
    manifest_root: Path, gh: _FakeGh, error: str, capsys: pytest.CaptureFixture[str]
):
    """A missing merge base, or a listing at GitHub's 3000-file cap that may leave core files out, is an error."""
    assert generator.main(['--touched', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=gh) == 1
    assert capsys.readouterr().err == error
    assert not any(call[0] == '-H' for call in gh.calls)


def _signoff_digest(manifest_root: Path, capsys: pytest.CaptureFixture[str]) -> str:
    assert generator.main(['--fingerprint', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=_fake_gh()) == 0
    return capsys.readouterr().out.strip()


REQUEST_PATCH = "@@ -45,2 +45,2 @@ class Model:\n     def request(self) -> str:\n-        return 'request'\n+        return 'response'"
# The same change after merging a base branch that added lines above it and next to it.
MERGED_REQUEST_PATCH = (
    '@@ -52,3 +52,3 @@ class BaseModel:\n     def request(self) -> str:\n'
    "-        return 'request'\n+        return 'response'\n \n"
)


def _fingerprint(*files: ChangedFile) -> str:
    return generator.fingerprint(files, {MODELS: MANIFEST[MODELS]})


@pytest.mark.parametrize(
    'other',
    [
        pytest.param(ChangedFile(MODELS, 'modified', patch=MERGED_REQUEST_PATCH), id='moved-hunk-and-context'),
        pytest.param(
            ChangedFile(MODELS, 'modified', patch='diff --git a/x b/x\n--- a/x\n+++ b/x\n' + REQUEST_PATCH),
            id='file-header',
        ),
    ],
)
def test_the_fingerprint_ignores_what_merging_the_base_branch_in_moves(other: ChangedFile):
    """Only the added and removed lines count, so new hunk line numbers and context leave the fingerprint as it was."""
    assert _fingerprint(other) == _fingerprint(ChangedFile(MODELS, 'modified', patch=REQUEST_PATCH))


@pytest.mark.parametrize(
    'other',
    [
        pytest.param(ChangedFile(MODELS, 'modified', patch=REQUEST_PATCH.replace("'response'", "'reply'")), id='added'),
        pytest.param(
            ChangedFile(MODELS, 'modified', patch=REQUEST_PATCH.replace("-        return 'request'\n", '')),
            id='removed',
        ),
        pytest.param(ChangedFile(MODELS, 'modified', patch=REQUEST_PATCH + '\n----'), id='removed-dashes'),
        pytest.param(
            ChangedFile(f'{CORE}/models/base.py', 'renamed', previous_filename=MODELS, patch=REQUEST_PATCH),
            id='renamed',
        ),
    ],
)
def test_the_fingerprint_changes_with_a_changed_line_of_a_touched_core_file(other: ChangedFile):
    """A different added or removed line, a removed line that reads like a file header, or a rename is a new change."""
    assert _fingerprint(other) != _fingerprint(ChangedFile(MODELS, 'modified', patch=REQUEST_PATCH))


def test_the_fingerprint_ignores_files_without_touched_gating_entries():
    """A file `touched_entries` does not name, a core file included, is left out, even with no patch."""
    models = ChangedFile(MODELS, 'modified', patch=REQUEST_PATCH)
    others = [
        ChangedFile('README.md', 'modified', patch='@@ -1 +1 @@\n-a\n+b'),
        ChangedFile(SETTINGS, 'modified', patch='@@ -1 +1 @@\n-DEFAULT = 0\n+DEFAULT = 1'),
        ChangedFile('docs/big.md', 'modified'),
    ]

    assert _fingerprint(models, *others) == _fingerprint(models)


def test_the_fingerprint_refuses_a_touched_core_file_without_a_patch():
    """GitHub leaves out the patch of a large diff, and then the changed lines are unknown."""
    with pytest.raises(ValueError, match='GitHub left out the patch of'):
        _fingerprint(ChangedFile(MODELS, 'modified'))


FINGERPRINT = 'f' * 64
MARKER = f'<!-- durable-exec-signoff fingerprint={FINGERPRINT} head={HEAD_SHA} -->'


@pytest.mark.parametrize(
    'body, signoff',
    [
        pytest.param(f'Durable review done.\n\n{MARKER}\n', (FINGERPRINT, HEAD_SHA), id='own-line'),
        pytest.param(f'Done.\r\n  {MARKER}  \r\n', (FINGERPRINT, HEAD_SHA), id='crlf-and-spaces'),
        pytest.param(f'{MARKER.replace("f" * 64, "e" * 64)}\n{MARKER}', (FINGERPRINT, HEAD_SHA), id='last-wins'),
        pytest.param(f'> {MARKER}', None, id='quoted'),
        pytest.param(f'LGTM {MARKER}', None, id='inline'),
        pytest.param(MARKER.replace('f' * 64, 'F' * 64), None, id='uppercase-fingerprint'),
        pytest.param(MARKER.replace('f' * 64, 'f' * 63), None, id='short-fingerprint'),
        pytest.param(MARKER.replace(HEAD_SHA, HEAD_SHA[:12]), None, id='short-head'),
        pytest.param(MARKER.replace(' -->', ' x=1 -->'), None, id='extra-field'),
    ],
)
def test_a_signoff_is_a_marker_line_of_its_own(body: str, signoff: tuple[str, str] | None):
    """Only an exact, well-formed marker line counts, so a quote of someone else's marker does not."""
    parsed = generator.parse_signoff(body, 'DouweM')

    assert (parsed and (parsed.fingerprint, parsed.head)) == signoff


def _comment(comment_id: int, login: str, body: str, user_type: str = 'User') -> dict[str, object]:
    return {'id': comment_id, 'user': {'login': login, 'type': user_type}, 'body': body}


def _signoff_gh(comments: list[dict[str, object]], permissions: dict[str, str]) -> _FakeGh:
    gh = _fake_gh()
    gh.responses[f'repos/{REPO}/issues/42/comments?per_page=100'] = json.dumps([comments[:1], comments[1:]])
    for login, permission in permissions.items():
        gh.responses[f'repos/{REPO}/collaborators/{login}/permission'] = f'{permission}\n'
    return gh


def _marker(digest: str) -> str:
    return f'Signed off.\n{generator.signoff_marker(digest, HEAD_SHA)}'


def test_signoff_marker_prints_the_marker_for_the_current_fingerprint(
    manifest_root: Path, capsys: pytest.CaptureFixture[str]
):
    digest = _signoff_digest(manifest_root, capsys)

    assert generator.main(['--signoff-marker', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=_fake_gh()) == 0

    out = capsys.readouterr().out
    assert out == f'<!-- durable-exec-signoff fingerprint={digest} head={HEAD_SHA} -->\n'
    assert generator.parse_signoff(out, 'x') == (digest, HEAD_SHA, 'x')


def test_signed_off_accepts_the_latest_signoff_by_a_writer(manifest_root: Path, capsys: pytest.CaptureFixture[str]):
    """The newest marker comment whose author has write access counts; newer ones by others are skipped.

    Each author's permission is read once.
    """
    digest = _signoff_digest(manifest_root, capsys)
    gh = _signoff_gh(
        [
            _comment(1, 'maintainer', _marker('0' * 64)),
            _comment(2, 'DouweM', _marker(digest)),
            _comment(3, 'author', _marker('1' * 64)),
            _comment(4, 'github-actions[bot]', _marker('2' * 64), user_type='Bot'),
            _comment(5, 'author', _marker('3' * 64)),
            _comment(6, 'DouweM', 'Thanks!'),
        ],
        {'author': 'read', 'DouweM': 'admin'},
    )

    assert generator.main(['--signed-off', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=gh) == 0

    assert capsys.readouterr().out == (
        f'The durable-exec sign-off by DouweM at {HEAD_SHA} still covers fingerprint {digest}.\n'
    )
    permission_calls = [call[0] for call in gh.calls if '/collaborators/' in call[0]]
    assert permission_calls == [
        f'repos/{REPO}/collaborators/author/permission',
        f'repos/{REPO}/collaborators/DouweM/permission',
    ]


@pytest.mark.parametrize(
    'comments, permissions',
    [
        pytest.param([], {}, id='no-comments'),
        pytest.param([_comment(1, 'author', 'current')], {'author': 'read'}, id='reader'),
        pytest.param([_comment(1, 'author', 'current')], {'author': 'triage'}, id='triage'),
        pytest.param([_comment(1, 'pydanty', 'current', user_type='Bot')], {}, id='bot'),
        pytest.param([_comment(1, 'evil/../x', 'current')], {}, id='login-outside-the-shape'),
        pytest.param(
            [_comment(1, 'DouweM', 'current'), _comment(2, 'DouweM', 'stale')], {'DouweM': 'write'}, id='newer-stale'
        ),
    ],
)
def test_signed_off_fails_without_a_writers_signoff_on_the_current_fingerprint(
    manifest_root: Path,
    capsys: pytest.CaptureFixture[str],
    comments: list[dict[str, object]],
    permissions: dict[str, str],
):
    """A sign-off by a user without write access, by a bot, or superseded by a newer one on other changes does not count."""
    digest = _signoff_digest(manifest_root, capsys)
    bodies = {'current': _marker(digest), 'stale': _marker('0' * 64)}
    gh = _signoff_gh([{**comment, 'body': bodies[str(comment['body'])]} for comment in comments], permissions)

    assert generator.main(['--signed-off', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=gh) == 1

    assert capsys.readouterr().err.startswith(f'no sign-off covers fingerprint {digest}; the latest sign-off is ')
    assert {call[0] for call in gh.calls if '/collaborators/' in call[0]} == {
        f'repos/{REPO}/collaborators/{login}/permission' for login in permissions
    }


def test_signed_off_fails_when_a_touched_core_file_cannot_be_fingerprinted(
    manifest_root: Path, capsys: pytest.CaptureFixture[str]
):
    """Without the patch of a touched core file no sign-off can cover it, so the label job adds the label."""
    gh = _fake_gh()
    pages = json.loads(gh.responses[f'repos/{REPO}/pulls/42/files?per_page=100'])
    del pages[0][0]['patch']
    gh.responses[f'repos/{REPO}/pulls/42/files?per_page=100'] = json.dumps(pages)

    assert generator.main(['--signed-off', REPO, '42', BASE_SHA, HEAD_SHA], root=manifest_root, gh=gh) == 1

    assert capsys.readouterr().err == f"cannot fingerprint: GitHub left out the patch of '{MODELS}'\n"
    assert not any('/comments' in call[0] for call in gh.calls)
