"""Test that the emem geospatial MCP example exists and is valid."""

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def test_emem_example_module_exists_and_is_valid_python() -> None:
    """The emem geospatial example module must exist and be valid Python."""
    module_path = REPO_ROOT / 'examples' / 'pydantic_ai_examples' / 'emem_geospatial_agent.py'
    assert module_path.exists(), f'{module_path} does not exist'
    source = module_path.read_text()
    ast.parse(source)


def test_emem_example_docs_page_exists() -> None:
    """The emem geospatial example docs page must exist."""
    docs_path = REPO_ROOT / 'docs' / 'examples' / 'emem-geospatial-agent.md'
    assert docs_path.exists(), f'{docs_path} does not exist'


def test_emem_example_in_nav() -> None:
    """The emem geospatial example must be referenced in docs/nav.json."""
    nav_path = REPO_ROOT / 'docs' / 'nav.json'
    nav = json.loads(nav_path.read_text())
    nav_str = json.dumps(nav)
    assert 'emem-geospatial-agent' in nav_str, 'emem-geospatial-agent not found in docs/nav.json'
