"""Keep the CodSpeed benchmark workflow on the configuration decided in #8780/#8781.

Keep in sync with .github/workflows/benchmark.yml (same pattern as the ci.yml
assertions in tests/test_embeddings.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(__file__).parent.parent / '.github' / 'workflows' / 'benchmark.yml'


def _load_workflow() -> dict[str, Any]:
    """Load and parse .github/workflows/benchmark.yml."""
    if not WORKFLOW.is_file():  # pragma: lax no cover
        pytest.skip('not running from a repo checkout')
    workflow: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    return workflow


def test_benchmark_workflow_uses_graviton_walltime() -> None:
    workflow = _load_workflow()
    job = workflow['jobs']['benchmarks']
    assert job['runs-on'] == 'codspeed-macro-arm64-graviton-ubuntu-22-04'  # benchmark.yml:18
    assert job['timeout-minutes'] == 10  # benchmark.yml:19

    # Workflow-level: PR runs cancel each other; each main push keys its own baseline.
    concurrency = workflow['concurrency']
    assert concurrency['cancel-in-progress'] is True  # benchmark.yml:13
    assert 'github.run_id' in concurrency['group']  # benchmark.yml:12

    setup_uv = next(s for s in job['steps'] if 'setup-uv' in str(s.get('uses', '')))
    assert setup_uv['with']['python-version'] == '3.14'  # benchmark.yml:31

    codspeed_action = next(s for s in job['steps'] if 'codspeed' in str(s.get('uses', '')).lower())
    assert codspeed_action['with']['mode'] == 'walltime'  # benchmark.yml:40
