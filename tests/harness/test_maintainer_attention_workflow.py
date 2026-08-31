from __future__ import annotations

import re
from pathlib import Path

UPSTREAM_SHA = '22b3d6a4dc9e409c5a581e4657311b0dfff16256'


def test_maintainer_attention_uses_merged_upstream_workflows():
    workflow = (Path(__file__).parents[1] / '.github/workflows/maintainer-attention.yml').read_text()

    refs = re.findall(r'uses: pydantic/pydantic-ai/\.github/workflows/[^@]+@([0-9a-f]{40})', workflow)
    assert refs == [UPSTREAM_SHA] * 3

    owner_routing = workflow.split('  owner-routing:', 1)[1].split('\n  operations:', 1)[0]
    assert 'pull-requests: write' in owner_routing
