"""Execute the size-label workflow shell with a recording, stateful gh fake."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from typing_extensions import TypedDict

WORKFLOW = Path(__file__).parents[1] / 'workflows' / 'bots.yml'
PULL = 'repos/pydantic/pydantic-ai/pulls/1234'
LABELS = 'repos/pydantic/pydantic-ai/issues/1234/labels'
SIZES = ['size: S', 'size: M', 'size: L', 'size: XL']

FAKE_GH = r"""#!/usr/bin/env python3
import json
import os
import subprocess
import sys
from pathlib import Path

args = sys.argv[1:]
assert args[0] == 'api', args
endpoint = args[1]
method = args[args.index('--method') + 1] if '--method' in args else 'GET'
state = Path(os.environ['FAKE_STATE'])
labels = json.loads((state / 'labels.json').read_text())

def record():
    with (state / 'calls.txt').open('a') as handle:
        handle.write(f'{method} {endpoint}\n')

if endpoint.endswith('/files'):
    pages = json.loads(os.environ['FAKE_FILE_PAGES'])
    for page in pages if '--paginate' in args else pages[:1]:
        record()
        if os.environ.get('FAKE_FAILURE') == 'files':
            print('[]')
            sys.exit('file read failed')
        print(os.environ.get('FAKE_FILE_RESPONSE', json.dumps(page)))
    sys.exit(0)

record()
if method == 'GET' and endpoint.endswith('/pulls/1234'):
    if os.environ.get('FAKE_FAILURE') == 'labels':
        sys.exit('label read failed')
    payload = os.environ.get('FAKE_PULL_RESPONSE', json.dumps({'labels': [{'name': label} for label in labels]}))
    if '--jq' in args:
        # gh parses JSON before evaluating its jq filter, including on empty input.
        json.loads(payload)
        result = subprocess.run(['jq', '-r', args[args.index('--jq') + 1]], input=payload, text=True)
        sys.exit(result.returncode)
    print(payload)
elif method == 'DELETE' and '/labels/' in endpoint:
    if os.environ.get('FAKE_FAILURE') == 'delete':
        sys.exit('label delete failed')
    label = endpoint.split('/labels/', 1)[1]
    if label not in labels:
        sys.exit('label absent (HTTP 404)')
    labels.remove(label)
elif method == 'POST' and endpoint.endswith('/labels'):
    if os.environ.get('FAKE_FAILURE') == 'post':
        sys.exit('label add failed')
    assert args[args.index('-f') + 1].startswith('labels[]=')
    label = args[args.index('-f') + 1].removeprefix('labels[]=')
    if label not in labels:
        labels.append(label)
else:
    sys.exit('unexpected gh call: ' + repr(args))
(state / 'labels.json').write_text(json.dumps(labels))
"""


class ChangedFile(TypedDict):
    """File fields consumed by the size calculation."""

    filename: str
    additions: int
    deletions: int


@dataclass
class StepResult:
    """Shell outcome, REST request log and resulting label state."""

    process: subprocess.CompletedProcess[str]
    calls: list[str]
    labels: list[str]


def _run_step(
    tmp_path: Path,
    source: str,
    labels: list[str],
    pages: list[list[ChangedFile]],
    overrides: dict[str, str] | None = None,
    attributes: dict[str, str] | None = None,
) -> StepResult:
    tmp_path.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '--quiet', str(tmp_path)], check=True)
    (tmp_path / '.gitattributes').write_text(
        (WORKFLOW.parents[2] / '.gitattributes').read_text(encoding='utf-8'), encoding='utf-8'
    )
    for filename, content in (attributes or {}).items():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    gh = tmp_path / 'gh'
    gh.write_text(FAKE_GH, encoding='utf-8')
    gh.chmod(0o755)
    (tmp_path / 'labels.json').write_text(json.dumps(labels), encoding='utf-8')
    script = source.replace('${{ github.repository }}', 'pydantic/pydantic-ai').replace(
        '${{ github.event.pull_request.number }}', '1234'
    )
    process = subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', script],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
            'GH_TOKEN': 'fake-token',
            'FAKE_STATE': str(tmp_path),
            'FAKE_FILE_PAGES': json.dumps(pages),
            **(overrides or {}),
        },
    )
    calls = tmp_path / 'calls.txt'
    return StepResult(
        process,
        calls.read_text(encoding='utf-8').splitlines(),
        json.loads((tmp_path / 'labels.json').read_text(encoding='utf-8')),
    )


@pytest.fixture
def script() -> str:
    """Load the candidate run block without executing privileged workflow jobs."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    return next(step['run'] for step in workflow['jobs']['size-label']['steps'] if 'run' in step)


@pytest.mark.parametrize(
    ('initial_sizes', 'writes'),
    [
        (['size: S'], []),
        ([], [f'POST {LABELS}']),
        (['size: M'], [f'DELETE {LABELS}/size: M', f'POST {LABELS}']),
        (['size: S', 'size: L'], [f'DELETE {LABELS}/size: L']),
        (SIZES[1:], [*[f'DELETE {LABELS}/{label}' for label in SIZES[1:]], f'POST {LABELS}']),
        (SIZES, [f'DELETE {LABELS}/{label}' for label in SIZES[1:]]),
    ],
    ids=['correct', 'missing', 'wrong', 'correct-and-wrong', 'three-wrong', 'all-four'],
)
def test_label_updates(tmp_path: Path, script: str, initial_sizes: list[str], writes: list[str]):
    """Write only necessary size changes and preserve every unrelated label."""
    unrelated = ['chore', 'pkg:core', 'size: custom', 'contains "quotes" and spaces']
    result = _run_step(tmp_path, script, [*unrelated, *initial_sizes], [[]])

    assert result.process.returncode == 0, result.process.stderr
    assert result.calls == [f'GET {PULL}/files', f'GET {PULL}', *writes]
    assert sorted(result.labels) == sorted([*unrelated, 'size: S'])
    # One file page plus one live label read plus exactly the necessary writes.
    assert len(result.calls) == 2 + len(writes)


@pytest.mark.parametrize(
    ('files', 'score', 'target'),
    [
        ([], 0, 'size: S'),
        ([('src/code.py', 50, 50)], 100, 'size: S'),
        ([('src/code.py', 50, 51)], 101, 'size: M'),
        ([('src/code.py', 250, 250)], 500, 'size: M'),
        ([('src/code.py', 250, 251)], 501, 'size: L'),
        ([('src/code.py', 750, 750)], 1500, 'size: L'),
        ([('src/code.py', 750, 751)], 1501, 'size: XL'),
        (
            [
                ('src/code.py', 80, 0),
                ('docs/guide.py', 10, 5),
                ('README.md', 0, 6),
                ('tests/test_code.py', 10, 1),
                ('tests/README.md', 4, 0),
                ('uv.lock', 9999, 9999),
                ('tests/cassettes/provider/response.yaml', 9999, 9999),
                ('src/cassettes/response.json', 9999, 9999),
                ('docs/cassettes/README.md', 9999, 9999),
            ],
            97,  # 80 code + floor(25 docs / 2) + floor(11 tests / 2).
            'size: S',
        ),
    ],
)
def test_scoring(tmp_path: Path, script: str, files: list[tuple[str, int, int]], score: int, target: str):
    """Preserve cutoffs, category weights, integer rounding and exclusions."""
    changes: list[ChangedFile] = [
        {'filename': filename, 'additions': additions, 'deletions': deletions}
        for filename, additions, deletions in files
    ]
    result = _run_step(tmp_path, script, [], [changes])

    assert result.process.returncode == 0, result.process.stderr
    assert f'Weighted score: {score}\n' in result.process.stdout
    assert result.labels == [target]
    assert result.calls == [f'GET {PULL}/files', f'GET {PULL}', f'POST {LABELS}']


def test_file_pagination(tmp_path: Path, script: str):
    """Include changes on later file pages when choosing the target size."""
    first_page: list[ChangedFile] = [
        {'filename': f'src/file_{index}.py', 'additions': 1, 'deletions': 0} for index in range(100)
    ]
    second_page: list[ChangedFile] = [{'filename': 'src/last.py', 'additions': 1, 'deletions': 0}]
    result = _run_step(tmp_path, script, ['size: S'], [first_page, second_page])

    assert result.process.returncode == 0, result.process.stderr
    assert 'Weighted score: 101\n' in result.process.stdout
    assert result.labels == ['size: M']
    assert result.calls == [
        f'GET {PULL}/files',
        f'GET {PULL}/files',
        f'GET {PULL}',
        f'DELETE {LABELS}/size: S',
        f'POST {LABELS}',
    ]


def test_live_labels_beyond_first_label_page(tmp_path: Path, script: str):
    """Read the pull's complete label list without a paginated labels endpoint."""
    unrelated = [f'topic-{index}' for index in range(100)]
    result = _run_step(tmp_path, script, [*unrelated, 'size: S'], [[]])

    assert result.process.returncode == 0, result.process.stderr
    assert result.calls == [f'GET {PULL}/files', f'GET {PULL}']
    assert result.labels == [*unrelated, 'size: S']


def test_stale_event_labels(tmp_path: Path, script: str):
    """Use current API labels even when the queued event carries an old size."""
    event = tmp_path / 'event.json'
    event.write_text(json.dumps({'pull_request': {'labels': [{'name': 'size: XL'}]}}), encoding='utf-8')
    result = _run_step(tmp_path, script, ['size: S'], [[]], {'GITHUB_EVENT_PATH': str(event)})

    assert result.process.returncode == 0, result.process.stderr
    assert result.calls == [f'GET {PULL}/files', f'GET {PULL}']
    assert result.labels == ['size: S']


@pytest.mark.parametrize(
    ('overrides', 'expected_calls'),
    [
        ({'FAKE_FAILURE': 'files'}, [f'GET {PULL}/files']),
        ({'GIT_DIR': '/nonexistent-size-label-repository'}, [f'GET {PULL}/files']),
        ({'FAKE_FILE_RESPONSE': 'not json'}, [f'GET {PULL}/files']),
        ({'FAKE_FAILURE': 'labels'}, [f'GET {PULL}/files', f'GET {PULL}']),
        *[
            ({'FAKE_PULL_RESPONSE': response}, [f'GET {PULL}/files', f'GET {PULL}'])
            for response in ['', 'not json', '{}', '{"labels": null}', '{"labels": {}}', '{"labels": [{}]}']
        ],
        ({'FAKE_FAILURE': 'delete'}, [f'GET {PULL}/files', f'GET {PULL}', f'DELETE {LABELS}/size: M']),
        (
            {'FAKE_FAILURE': 'post'},
            [f'GET {PULL}/files', f'GET {PULL}', f'DELETE {LABELS}/size: M', f'POST {LABELS}'],
        ),
    ],
)
def test_api_errors(tmp_path: Path, script: str, overrides: dict[str, str], expected_calls: list[str]):
    """Stop on failed or unreadable reads and failed writes without reporting success."""
    result = _run_step(tmp_path, script, ['chore', 'size: M'], [[]], overrides)

    assert result.process.returncode != 0
    assert result.process.stderr
    assert result.calls == expected_calls
    assert 'Set label:' not in result.process.stdout
    assert 'chore' in result.labels


@pytest.mark.parametrize(
    ('filename', 'score'),
    [
        ('.github/workflows/pydantic-ai-pr-review.lock.yml', 0),
        ('src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/static/index.html', 0),
        ('src/pydantic_clai2/pydantic_clai2/builtin_plugins/web/static/assets/app.js', 0),
        ('src/pydantic_clai2/web-ui/package-lock.json', 0),
        ('.github/workflows/pydantic-ai-pr-review.md', 1000),
        ('src/pydantic_clai2/web-ui/src/app.ts', 2000),
        ('src/pydantic_clai2/web-ui/package.json', 2000),
    ],
)
def test_generated_files(tmp_path: Path, script: str, filename: str, score: int):
    """Exclude generated artifacts while preserving their editable source neighbors."""
    changes: list[ChangedFile] = [{'filename': filename, 'additions': 1000, 'deletions': 1000}]
    result = _run_step(tmp_path, script, [], [changes])

    assert result.process.returncode == 0, result.process.stderr
    assert f'Weighted score: {score}\n' in result.process.stdout


def test_attribute_overrides(tmp_path: Path, script: str):
    """Resolve nested declarations, explicit opt-outs and bare boolean attributes."""
    filenames = [
        'src/generated.js',
        'src/handwritten.js',
        'src/bare.js',
        'src/unset.js',
        'src/unspecified.js',
        'docs/generated.md',
        'tests/generated.py',
    ]
    changes: list[ChangedFile] = [{'filename': filename, 'additions': 10, 'deletions': 10} for filename in filenames]
    result = _run_step(
        tmp_path,
        script,
        [],
        [changes],
        attributes={
            '.gitattributes': '*.js linguist-generated=true\ndocs/** linguist-generated=true\n'
            'tests/** linguist-generated=true\n',
            'src/.gitattributes': 'handwritten.js linguist-generated=false\nbare.js linguist-generated\n'
            'unset.js -linguist-generated\nunspecified.js !linguist-generated\n',
        },
    )

    assert result.process.returncode == 0, result.process.stderr
    assert 'Code: 60, Docs: 0, Tests: 0\n' in result.process.stdout
    assert 'Weighted score: 60\n' in result.process.stdout
    assert result.labels == ['size: S']


def test_generated_paths_on_later_pages(tmp_path: Path, script: str):
    """Handle deleted or new paths, shell metacharacters and NUL-delimited Git output."""
    filenames = ['new file.js', 'quote".js', 'line\nbreak.js', '$(touch injected).js', '-option.js', 'café.js']
    changes: list[ChangedFile] = [{'filename': filename, 'additions': 1000, 'deletions': 1000} for filename in filenames]
    result = _run_step(
        tmp_path,
        script,
        [],
        [[{'filename': 'src/code.py', 'additions': 50, 'deletions': 51}], changes],
        attributes={'.gitattributes': '*.js linguist-generated=true\n'},
    )

    assert result.process.returncode == 0, result.process.stderr
    assert 'Weighted score: 101\n' in result.process.stdout
    assert result.labels == ['size: M']
    assert result.calls == [f'GET {PULL}/files', f'GET {PULL}/files', f'GET {PULL}', f'POST {LABELS}']
    assert not (tmp_path / 'injected').exists()


def test_trusted_checkout():
    """Pin declarations to base code without persisting the privileged token."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    steps = workflow['jobs']['size-label']['steps']

    assert steps[0]['uses'].startswith('actions/checkout@')
    assert steps[0]['with'] == {'ref': '${{ github.event.pull_request.base.sha }}', 'persist-credentials': False}
    assert steps[1]['name'] == 'Calculate PR size and set label'
