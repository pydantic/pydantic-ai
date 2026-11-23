from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import packaging_smoke
from packaging_smoke import ImportFailure, Shape, ShardReport, Wheel

EXTRAS: dict[str, tuple[str, ...]] = {
    'pydantic-ai-slim': ('duckduckgo', 'mcp', 'openai', 'voyageai', 'web-fetch'),
    'pydantic-ai-harness': ('coder', 'day-ai', 'repair-tool-arguments'),
}


def _import_error(message: str, *, missing: str | None = None) -> ImportFailure:
    return ImportFailure(module='pkg.mod', exception='ImportError', message=message, missing=missing, import_error=True)


# Copied from the packages, so a rephrased hint that stops matching fails here rather than in a clean install.
HINTS = [
    pytest.param(
        'Please install `openai` to use the OpenAI model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`',
        id='slim-optional-group',
    ),
    pytest.param(
        'Please install the fastmcp client to use `MCPToolset` — '
        '`pip install "pydantic-ai-slim[mcp]"` pulls `fastmcp-slim[client]`, '
        'or install the full `fastmcp` package directly.',
        id='slim-mcp-names-another-requirement-too',
    ),
    pytest.param(
        'MCP support is required for the DayAI capability. Install it with uv add "pydantic-ai-harness[day-ai]"',
        id='harness-uv-add',
    ),
    pytest.param(
        '`json-repair` is required for `RepairToolArguments`. Install `pydantic-ai-harness[coder]` to use `Coder`, '
        'or `pydantic-ai-harness[repair-tool-arguments]` to use `RepairToolArguments` on its own.',
        id='harness-two-extras',
    ),
    pytest.param(
        'Please install `ddgs` to use the DuckDuckGo search tool, '
        "you can use the `duckduckgo` optional group — `pip install 'pydantic-ai-slim[duckduckgo,web-fetch]'`",
        id='several-extras-single-quotes',
    ),
]


@pytest.mark.parametrize('message', HINTS)
def test_install_hint_passes(message: str):
    """An `ImportError` naming an extra of a published package is a hint the shape passes with."""
    assert packaging_smoke.failure_problem(_import_error(message), extras=EXTRAS, installed=None) is None


@pytest.mark.parametrize(
    ('failure', 'problem'),
    [
        pytest.param(
            ImportFailure(
                module='pydantic_ai_harness._workspace_provider',
                exception='ModuleNotFoundError',
                message="No module named 'sniffio'",
                missing='sniffio',
                import_error=True,
            ),
            'no install hint names an extra of a published package',
            id='bare-module-not-found',
        ),
        pytest.param(
            _import_error('Please install `fastmcp-slim[client]` to use the MCP toolset.'),
            'no install hint names an extra of a published package',
            id='hint-names-an-unpublished-distribution',
        ),
        pytest.param(
            _import_error('Please install `cohere`, you can use `pip install "pydantic-ai-slim[cohere]"`'),
            'the install hint names pydantic-ai-slim[cohere], which is not an extra',
            id='hint-names-a-missing-extra',
        ),
        pytest.param(
            ImportFailure(
                module='pkg.mod',
                exception='AttributeError',
                message='module has no attribute `X`; pip install "pydantic-ai-slim[openai]"',
                missing=None,
                import_error=False,
            ),
            'raised AttributeError, not an ImportError',
            id='not-an-import-error',
        ),
    ],
)
def test_import_failure_fails(failure: ImportFailure, problem: str):
    """Any other import failure fails the shape, with the reason it is not a hint."""
    assert packaging_smoke.failure_problem(failure, extras=EXTRAS, installed=None) == problem


def test_hint_naming_the_installed_extra_fails():
    """A hint to install the extra the shape installed fails it; a hint naming another extra does not."""
    message = 'MCP support is required for the DayAI capability. Install it with uv add "pydantic-ai-harness[day-ai]"'
    assert packaging_smoke.failure_problem(
        _import_error(message), extras=EXTRAS, installed=('pydantic-ai-harness', 'day-ai')
    ) == ('the install hint names the extra this shape installed')
    assert (
        packaging_smoke.failure_problem(_import_error(message), extras=EXTRAS, installed=('pydantic-ai-slim', 'mcp'))
        is None
    )


def _write_wheel(directory: Path, name: str, files: dict[str, str], metadata: str) -> Path:
    path = directory / f'{name.replace("-", "_")}-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        for file, content in files.items():
            archive.writestr(file, content)
        archive.writestr(f'{name.replace("-", "_")}-1.0.0.dist-info/METADATA', metadata)
    return path


SLIM_METADATA = """\
Metadata-Version: 2.4
Name: pydantic-ai-slim
Version: 1.0.0
Requires-Dist: anyio>=4.7.0
Requires-Dist: openai<4,>=3.50.0; extra == 'openai'
Requires-Dist: voyageai<1,>=0.3.7; python_version < '3.0' and extra == 'voyageai'
Requires-Dist: hf-xet<1.5.0; python_version >= '3.0' and extra == 'huggingface'
Provides-Extra: openai
Provides-Extra: voyageai
Provides-Extra: huggingface
"""


def test_read_wheel(tmp_path: Path):
    """The metadata and top-level packages come from the wheel, and a marker can leave out an extra's requirement."""
    path = _write_wheel(
        tmp_path,
        'pydantic-ai-slim',
        {'pydantic_ai/__init__.py': '', 'pydantic_ai/models/__init__.py': '', 'docs/guide.md': '', 'top.py': ''},
        SLIM_METADATA,
    )
    wheel = packaging_smoke.read_wheel(path)
    assert (wheel.name, wheel.extras, wheel.packages) == (
        'pydantic-ai-slim',
        ('openai', 'voyageai', 'huggingface'),
        ('pydantic_ai', 'top'),
    )
    assert packaging_smoke.installs_completely(wheel, 'openai')
    assert packaging_smoke.installs_completely(wheel, 'huggingface')
    assert not packaging_smoke.installs_completely(wheel, 'voyageai')


def test_published_distributions(tmp_path: Path):
    """The root project and each workspace member are published, by canonical name."""
    (tmp_path / 'pyproject.toml').write_text(
        '[project]\nname = "pydantic-ai"\n\n[tool.uv.workspace]\nmembers = ["pydantic_ai_slim", "src/pydantic_ai_harness"]\n'
    )
    for member, name in [('pydantic_ai_slim', 'pydantic-ai-slim'), ('src/pydantic_ai_harness', 'pydantic_ai_harness')]:
        (tmp_path / member).mkdir(parents=True)
        (tmp_path / member / 'pyproject.toml').write_text(f'[project]\nname = "{name}"\n')
    assert packaging_smoke.published_distributions(tmp_path) == {
        'pydantic-ai',
        'pydantic-ai-slim',
        'pydantic-ai-harness',
    }


def _wheel(name: str, extras: tuple[str, ...] = ()) -> Wheel:
    return Wheel(path=Path(f'{name}.whl'), name=name, extras=extras, requirements=(), packages=())


def test_enumerate_shapes_runs_each_distribution_bare_then_per_extra():
    """Each distribution runs bare and once per extra, and an excluded distribution not at all."""
    shapes = packaging_smoke.enumerate_shapes(
        [_wheel('pydantic-graph'), _wheel('pydantic-ai-slim', ('openai', 'mcp')), _wheel('pydantic-ai', ('openai',))]
    )
    assert [shape.label for shape in shapes] == [
        'pydantic-ai-slim',
        'pydantic-ai-slim[mcp]',
        'pydantic-ai-slim[openai]',
        'pydantic-graph',
    ]


SHAPES = [Shape('a'), Shape('a', 'x'), Shape('a', 'y'), Shape('b'), Shape('b', 'z')]


def test_select_shapes_shards_the_selection():
    """Shards split the selection round-robin, and a package or shape filter selects before sharding."""
    shards = [packaging_smoke.select_shapes(SHAPES, packages=(), labels=(), shard=(i, 2)) for i in (1, 2)]
    assert [[shape.label for shape in shard] for shard in shards] == [['a', 'a[y]', 'b[z]'], ['a[x]', 'b']]
    selected = packaging_smoke.select_shapes(SHAPES, packages={'b'}, labels={'a[x]'}, shard=(1, 1))
    assert [shape.label for shape in selected] == ['a[x]', 'b', 'b[z]']


def test_select_shapes_refuses_a_name_it_does_not_know():
    """A misspelled filter fails instead of running fewer shapes."""
    with pytest.raises(SystemExit, match=r'no such package or shape: a\[w\], c'):
        packaging_smoke.select_shapes(SHAPES, packages={'c'}, labels={'a[w]'}, shard=(1, 1))


def test_select_shapes_refuses_an_empty_shard():
    """A shard that selects nothing fails instead of passing empty."""
    with pytest.raises(SystemExit, match='shard 2/2 of 1 shapes selects none'):
        packaging_smoke.select_shapes(SHAPES, packages=(), labels={'a'}, shard=(2, 2))


def _module_not_found(module: str, missing: str) -> ImportFailure:
    return ImportFailure(
        module=module,
        exception='ModuleNotFoundError',
        message=f"No module named '{missing}'",
        missing=missing,
        import_error=True,
    )


def test_baselined_failure_passes_and_a_new_one_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A failure `BASELINE` lists passes its shape and is reported; any other failure fails it."""
    monkeypatch.setattr(packaging_smoke, 'BASELINE', {('pkg.known', 'gone'): packaging_smoke.GUARDED})
    report = {
        'imported': 3,
        'failures': [
            _module_not_found('pkg.known', 'gone').__dict__,
            _module_not_found('pkg.new', 'gone').__dict__,
        ],
    }

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        (tmp_path / 'import-report.json').write_text(json.dumps(report))
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(packaging_smoke.subprocess, 'run', fake_run)
    wheel = _wheel('pkg')
    result = packaging_smoke.import_every_module(Shape('pkg'), wheel, {'pkg': wheel}, tmp_path / 'python', tmp_path, 0)
    assert result.baselined == {('pkg.known', 'gone')}
    assert result.problems == (
        "pkg.new: ModuleNotFoundError: No module named 'gone' (missing `gone`): "
        'no install hint names an extra of a published package',
    )
    assert result.hints == 0


def _report(shard: tuple[int, int], *baselined: tuple[str, str], filtered: bool = False) -> ShardReport:
    return {'shard': list(shard), 'filtered': filtered, 'baselined': [list(key) for key in baselined]}


def test_stale_entries_needs_every_shard_of_one_full_run(monkeypatch: pytest.MonkeyPatch):
    """An entry no shard reported is stale, and reports that miss a shard or ran a filter are refused."""
    monkeypatch.setattr(packaging_smoke, 'BASELINE', {('a', 'x'): 'r', ('b', 'y'): 'r', ('c', 'z'): 'r'})
    assert packaging_smoke.stale_entries([_report((1, 2), ('a', 'x')), _report((2, 2), ('b', 'y'))]) == [('c', 'z')]
    with pytest.raises(SystemExit, match=r'not every shard of one run'):
        packaging_smoke.stale_entries([_report((1, 2), ('a', 'x'))])
    with pytest.raises(SystemExit, match=r'not every shard of one run'):
        packaging_smoke.stale_entries([_report((1, 2)), _report((2, 3))])
    with pytest.raises(SystemExit, match='filtered by --package or --shape'):
        packaging_smoke.stale_entries([_report((1, 1), filtered=True)])


def test_stale_check_fails_on_a_stale_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """`--stale-check` reads the shard reports and names each entry to remove."""
    monkeypatch.setattr(packaging_smoke, 'BASELINE', {('a', 'x'): 'r', ('b', 'y'): 'r'})
    path = tmp_path / 'report.json'
    path.write_text(json.dumps(_report((1, 1), ('a', 'x'))))
    assert packaging_smoke.main(['--stale-check', str(path)]) == 1
    assert capsys.readouterr().err == 'b now imports without `y` failing: remove its `BASELINE` entry\n'
    path.write_text(json.dumps(_report((1, 1), ('a', 'x'), ('b', 'y'))))
    assert packaging_smoke.main(['--stale-check', str(path)]) == 0


def test_baseline_entries_are_guarded_or_known_defects():
    """Each entry says why it may fail: a guarded private module, or a defect tracked for a fix."""
    for (module, _), reason in packaging_smoke.BASELINE.items():
        if reason == packaging_smoke.GUARDED:
            assert any(part.startswith('_') for part in module.split('.')), module
        else:
            assert reason.startswith('known defect: '), module
