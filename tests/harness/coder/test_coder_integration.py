"""Deterministic end-to-end Coder exercise; no provider requests or stale cassettes."""

import shlex
import sys
from pathlib import Path

import pytest

from .test_tools import call

pytestmark = pytest.mark.anyio


async def test_coder_completes_task(tmp_path: Path) -> None:
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'AGENTS.md').write_text(
        'Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.\n'
    )
    (workspace / 'README.md').write_text(
        '# Incident desk\n\nIncidents enter through `cli.py`, public report composition lives in `service.py`, '
        'visibility rules live in `policy.py`, and each output format has its own renderer module. Internal incidents '
        'must not appear in exports unless a caller explicitly opts in.\n'
    )
    (workspace / 'incidents.py').write_text(
        'from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Incident:\n'
        '    title: str\n    severity: str\n    internal: bool = False\n'
    )
    (workspace / 'policy.py').write_text(
        'from incidents import Incident\n\n\ndef visible_incidents(\n'
        '    incidents: list[Incident], *, include_internal: bool\n) -> list[Incident]:\n'
        '    if include_internal:\n        return incidents\n'
        '    return [incident for incident in incidents if not incident.internal]\n'
    )
    (workspace / 'text_renderer.py').write_text(
        'from incidents import Incident\n\n\ndef render_text(incidents: list[Incident]) -> str:\n'
        "    return '\\n'.join(f'[{item.severity.upper()}] {item.title}' for item in incidents)\n"
    )
    (workspace / 'service.py').write_text(
        'from incidents import Incident\nfrom policy import visible_incidents\nfrom text_renderer import render_text\n\n\n'
        'def render_report(incidents: list[Incident], *, include_internal: bool = False) -> str:\n'
        '    visible = visible_incidents(incidents, include_internal=include_internal)\n'
        '    return render_text(visible)\n'
    )
    (workspace / 'cli.py').write_text(
        'from incidents import Incident\nfrom service import render_report\n\n\ndef run(\n'
        '    incidents: list[Incident], *, include_internal: bool = False\n) -> str:\n'
        '    return render_report(incidents, include_internal=include_internal)\n'
    )
    (workspace / 'dashboard.py').write_text(
        'from incidents import Incident\nfrom service import render_report\n\n\ndef incident_card(items: list[Incident]) -> str:\n'
        '    return render_report(items)\n'
    )
    (workspace / 'digest.py').write_text(
        'from incidents import Incident\nfrom service import render_report\n\n\ndef internal_digest(items: list[Incident]) -> str:\n'
        '    return render_report(items, include_internal=True)\n'
    )
    (workspace / 'test_reports.py').write_text(
        'from cli import run\nfrom incidents import Incident\n\n\ndef test_text_report_hides_internal_by_default() -> None:\n'
        "    incidents = [Incident('API down', 'critical'), Incident('Secret host', 'warning', internal=True)]\n"
        "    assert run(incidents) == '[CRITICAL] API down'\n\n\ndef test_text_report_can_include_internal() -> None:\n"
        "    incidents = [Incident('Secret host', 'warning', internal=True)]\n"
        "    assert run(incidents, include_internal=True) == '[WARNING] Secret host'\n"
    )

    files = await call(workspace, 'list_files', {'glob': '*.py'})
    assert 'service.py' in files and 'policy.py' in files
    matches = await call(workspace, 'grep', {'pattern': 'render_report'})
    assert 'dashboard.py' in matches and 'digest.py' in matches
    source = await call(workspace, 'read_file', {'path': 'service.py'})
    assert 'include_internal: bool = False' in source
    await call(
        workspace,
        'edit_file',
        {
            'path': 'service.py',
            'replacements': [
                {
                    'old_text': 'from incidents import Incident',
                    'new_text': 'import json\nfrom incidents import Incident',
                },
                {
                    'old_text': 'include_internal: bool = False)',
                    'new_text': 'include_internal: bool = False, format_name: str = "text")',
                },
                {
                    'old_text': '    return render_text(visible)',
                    'new_text': '    if format_name == "json":\n'
                    '        return json.dumps({"incidents": [{"title": item.title, "severity": item.severity} '
                    'for item in visible]})\n    return render_text(visible)',
                },
            ],
        },
    )
    await call(
        workspace,
        'write_file',
        {
            'path': 'test_json.py',
            'content': 'import json\nfrom service import render_report\nfrom incidents import Incident\n'
            'def test_json():\n'
            '    items = [Incident("public", "warning"), Incident("private", "critical", internal=True)]\n'
            '    assert json.loads(render_report(items, format_name="json")) == '
            '{"incidents": [{"title": "public", "severity": "warning"}]}\n',
        },
    )
    # Plugin autoload off: the repo's own `-p no:` settings don't reach this nested run, and two
    # installed plugins both register `--record-mode`.
    command = f'PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 {shlex.quote(sys.executable)} -m pytest -q'
    output = await call(workspace, 'shell', {'command': command})
    assert '3 passed' in output
