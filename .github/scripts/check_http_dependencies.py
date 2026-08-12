from __future__ import annotations

import email
import zipfile
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
    assert requirement.extras == {'httpx2'}
    assert requirement.specifier.contains('2.47.0')
    assert requirement.marker is not None
    assert str(requirement.marker).removeprefix('extra == ').strip('"') in openai_extras

root_slim = [requirement for requirement in root if requirement.name == 'pydantic-ai-slim']
assert root_slim
assert all('retries' not in requirement.extras for requirement in root_slim if requirement.marker is None)
assert any(requirement.extras == {'retries'} and requirement.marker is not None for requirement in root_slim)
