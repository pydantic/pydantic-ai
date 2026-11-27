from __future__ import annotations

from pathlib import Path

from inline_snapshot import snapshot

from pydantic_ai import FileCitationSource, MarkerCitationAnchor, WorkspaceLocation
from pydantic_ai.workspaces import LocalWorkspaceBackend, Workspace, WorkspaceRef, file_links

REF = WorkspaceRef(provider='e2b', id='sbx-1')


def test_file_links_reads_sandbox_links_only():
    text = (
        'Here is the [report](sandbox:/work/out/report.csv), the chart: ![a chart](<sandbox:/work/my%20chart.png>), '
        'the [copy](sandbox:/work/report(1).csv), and [notes](sandbox:/work/notes.md "my notes"). '
        'Not these: [relative](sandbox:out/x.csv), [docs](https://example.com), `[code](sandbox:/work/a.txt)`,\n'
        '```\nprint("[fenced](sandbox:/work/b.txt)")\n```'
    )

    assert [
        (c.sources, text[c.anchor.start : c.anchor.end])
        for c in file_links(text, REF)
        if isinstance(c.anchor, MarkerCitationAnchor)
    ] == snapshot(
        [
            (
                [
                    FileCitationSource(
                        location=WorkspaceLocation(workspace=REF, path='/work/out/report.csv'), title='report'
                    )
                ],
                '[report](sandbox:/work/out/report.csv)',
            ),
            (
                [
                    FileCitationSource(
                        location=WorkspaceLocation(workspace=REF, path='/work/my chart.png'), title='a chart'
                    )
                ],
                '![a chart](<sandbox:/work/my%20chart.png>)',
            ),
            (
                [
                    FileCitationSource(
                        location=WorkspaceLocation(workspace=REF, path='/work/report(1).csv'), title='copy'
                    )
                ],
                '[copy](sandbox:/work/report(1).csv)',
            ),
            (
                [FileCitationSource(location=WorkspaceLocation(workspace=REF, path='/work/notes.md'), title='notes')],
                '[notes](sandbox:/work/notes.md "my notes")',
            ),
        ]
    )
    # A reply from a run with no workspace has nothing to read the files from.
    assert file_links(text, None) == []


async def test_in_working_dir(tmp_path: Path):
    """A model can link any path, so an app checks one before serving it."""
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
