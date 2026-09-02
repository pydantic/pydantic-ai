from __future__ import annotations

import email
import zipfile
from importlib.metadata import requires
from pathlib import Path

from packaging.requirements import Requirement


def wheel_requirements(pattern: str) -> list[Requirement]:
    wheel = next(Path('dist').glob(pattern))
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith('.dist-info/METADATA'))
        metadata = email.message_from_bytes(archive.read(metadata_path))
    return [Requirement(value) for value in metadata.get_all('Requires-Dist', [])]


slim = wheel_requirements('pydantic_ai_slim-*.whl')
graph = wheel_requirements('pydantic_graph-*.whl')
root = wheel_requirements('pydantic_ai-[0-9]*.whl')

slim_httpx = [requirement for requirement in slim if requirement.name == 'httpx']
assert len(slim_httpx) == 1
assert str(slim_httpx[0].marker) == 'extra == "retries"'
assert not any(requirement.name == 'httpx' for requirement in graph)
assert not any(requirement.name == 'httpx' for requirement in root)

slim_httpx2 = [requirement for requirement in slim if requirement.name == 'httpx2']
assert len(slim_httpx2) == 1 and slim_httpx2[0].marker is None

openai_extras = {
    'openai',
    'openrouter',
    'bedrock-mantle',
    'zai',
    'snowflake',
    'crusoe',
    'cerebras',
    'openai-realtime',
    'xai-realtime',
}
openai_requirements = [requirement for requirement in slim if requirement.name == 'openai']
assert len(openai_requirements) == len(openai_extras)
for requirement in openai_requirements:
    assert not requirement.extras
    assert requirement.specifier.contains('3.0.0')
    assert requirement.marker is not None
    assert str(requirement.marker).removeprefix('extra == ').strip('"') in openai_extras

openai_dependencies = [Requirement(value) for value in requires('openai') or []]
assert not any(requirement.name == 'httpx' for requirement in openai_dependencies)
assert any(
    requirement.name == 'httpx2' and requirement.specifier.contains('2.7') for requirement in openai_dependencies
)

anthropic_requirements = [requirement for requirement in slim if requirement.name == 'anthropic']
assert len(anthropic_requirements) == 1
assert anthropic_requirements[0].specifier.contains('1.0.0')
assert str(anthropic_requirements[0].marker) == 'extra == "anthropic"'

anthropic_dependencies = [Requirement(value) for value in requires('anthropic') or []]
assert not any(requirement.name == 'httpx' for requirement in anthropic_dependencies)
assert any(
    requirement.name == 'httpx2' and requirement.specifier.contains('2.7') for requirement in anthropic_dependencies
)

google_extras = {'google', 'google-realtime'}
google_requirements = [requirement for requirement in slim if requirement.name == 'google-genai']
assert len(google_requirements) == len(google_extras)
for requirement in google_requirements:
    assert requirement.specifier.contains('2.18.0')
    assert requirement.marker is not None
    assert str(requirement.marker).removeprefix('extra == ').strip('"') in google_extras

# Google Gen AI 2.18 accepts injected HTTPX2 clients, but still depends directly on legacy HTTPX.
google_dependencies = [Requirement(value) for value in requires('google-genai') or []]
assert any(requirement.name == 'httpx' for requirement in google_dependencies)

# `pydantic_ai.mcp` imports legacy HTTPX at module level, and the `mcp` extra never declares it: the
# module is importable only because `fastmcp-slim[client]` pulls HTTPX in transitively. Assert that
# transitive dependency still exists, so `mcp.py` losing its HTTP client fails here rather than at import.
fastmcp_requirements = [requirement for requirement in slim if requirement.name == 'fastmcp-slim']
assert fastmcp_requirements
assert all(requirement.extras == {'client'} for requirement in fastmcp_requirements)

fastmcp_dependencies = [Requirement(value) for value in requires('fastmcp-slim') or []]
assert any(
    requirement.name == 'httpx' and str(requirement.marker) == 'extra == "client"'
    for requirement in fastmcp_dependencies
)

root_slim = [requirement for requirement in root if requirement.name == 'pydantic-ai-slim']
assert root_slim
assert all('retries' not in requirement.extras for requirement in root_slim if requirement.marker is None)
assert any(requirement.extras == {'retries'} and requirement.marker is not None for requirement in root_slim)
