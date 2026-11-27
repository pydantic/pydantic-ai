from __future__ import annotations

from pathlib import Path

import pytest

from inline_snapshot import snapshot

from pydantic_ai import FileCitationSource, MarkerCitationAnchor, WorkspaceLocation
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, file_links

from .workspace_fakes import FakeWorkspace, FilesystemOnlyWorkspaceBackend


def test_file_links_reads_sandbox_links_only():
    """Not a VCR test: `file_links` only parses text, so no model response is involved."""
    text = (
        'Here is the [report](sandbox:/work/out/report.csv), the chart: ![a chart](<sandbox:/work/my%20chart.png>), '
        'the [copy](sandbox:/work/report(1).csv), [notes](sandbox:/work/notes.md "my notes") and [win](sandbox:C:/work/w.txt). '
        'Not these: [relative](sandbox:out/x.csv), [nul](sandbox:/work/a%00b), [docs](https://example.com), `[code](sandbox:/work/a.txt)`,\n'
        '```\nprint("[fenced](sandbox:/work/b.txt)")\n```\n'
        '~~~\n~~~ not a closing fence\n[still fenced](sandbox:/work/f.txt)\n~~~\n'
        '1. Run ```pip install x``` and get [after inline](sandbox:/work/c.txt)\n'
        '   ```\n   [indented fence](sandbox:/work/d.txt)\n   ```\n'
        '2. Then [after list](sandbox:/work/e.txt)'
    )

    assert [
        (c.sources, text[c.anchor.start : c.anchor.end])
        for c in file_links(text)
        if isinstance(c.anchor, MarkerCitationAnchor)
    ] == snapshot(
        [
            (
                [FileCitationSource(location=WorkspaceLocation(path='/work/out/report.csv'), title='report')],
                '[report](sandbox:/work/out/report.csv)',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(path='/work/my chart.png'), title='a chart')],
                '![a chart](<sandbox:/work/my%20chart.png>)',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(path='/work/report(1).csv'), title='copy')],
                '[copy](sandbox:/work/report(1).csv)',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(path='/work/notes.md'), title='notes')],
                '[notes](sandbox:/work/notes.md "my notes")',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(path='C:/work/w.txt'), title='win')],
                '[win](sandbox:C:/work/w.txt)',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(path='/work/c.txt'), title='after inline')],
                '[after inline](sandbox:/work/c.txt)',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(path='/work/e.txt'), title='after list')],
                '[after list](sandbox:/work/e.txt)',
            ),
        ]
    )


async def test_in_working_dir(tmp_path: Path):
    """A model can link any path, so an app checks one before serving it.

    Not a VCR test: it checks paths in a local directory, and no model response is involved.
    """
    project = tmp_path / 'project'
    (project / 'out').mkdir(parents=True)
    (tmp_path / 'secret.txt').write_text('key')
    (project / 'out' / 'link.txt').symlink_to(tmp_path / 'secret.txt')
    workspace = Workspace(LocalWorkspaceBackend(project))
    root = await workspace.working_dir()

    assert await workspace.in_working_dir(f'{root}/out/report.csv')
    assert not await workspace.in_working_dir(f'{root}/../secret.txt')
    assert not await workspace.in_working_dir('/etc/passwd')
    assert not await workspace.in_working_dir(f'{root}/out/link.txt')


async def test_in_working_dir_refuses_without_symlink_resolution():
    """A backend that can't resolve symlinks can't show a path stays inside, so no path is.

    Not a VCR test: it uses an in-memory backend, and no model response is involved.
    """
    workspace = Workspace(FilesystemOnlyWorkspaceBackend(FakeWorkspace('files-only')))

    assert not await workspace.in_working_dir(f'{await workspace.working_dir()}/report.csv')


async def test_read_file_reads_a_cited_file_only_inside_the_working_dir(tmp_path: Path):
    """Not a VCR test: it reads files in a local directory, and no model response is involved."""
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'out.csv').write_text('a,b\n')
    (tmp_path / 'secret.txt').write_text('key')
    (project / 'link.csv').symlink_to(tmp_path / 'secret.txt')
    workspace = Workspace(LocalWorkspaceBackend(project))
    root = await workspace.working_dir()

    [out, escape, link] = [
        citation.sources[0]
        for citation in file_links(
            f'[out](sandbox:{root}/out.csv) [x](sandbox:{root}/../secret.txt) [y](sandbox:{root}/link.csv)'
        )
    ]
    assert isinstance(out, FileCitationSource)
    file = await workspace.read_file(out)
    assert (file.data, file.media_type) == (b'a,b\n', 'text/csv')
    for source in (escape, link):
        assert isinstance(source, FileCitationSource)
        with pytest.raises(PermissionError, match='is not inside the working directory'):
            await workspace.read_file(source)
