"""Guard `ModelSettings` "Supported by:" docstring lists against the code.

Every `ModelSettings` field documents which providers read it. A setting that
isn't read by a provider is silently dropped — not rejected — so a stale list
is indistinguishable from a broken provider. This test keeps the two in sync:

* the expected table below was derived from the provider modules (one pass,
  documented at https://github.com/pydantic/pydantic-ai/issues/6856);
* the assertions re-derive the *direct* reads from the source, so a future
  provider that grows or drops a read can't leave the docs behind.

`tool_choice` and `thinking` are deliberately excluded: every model receives
them through `resolve_tool_choice` / `ModelRequestParameters.thinking` whether
it honours them or not, so they can't be machine-checked and stay hand-maintained.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pydantic_ai.settings import ModelSettings

REPO_ROOT = Path(__file__).parents[1]
SETTINGS_FILE = REPO_ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / 'settings.py'
MODELS_DIR = REPO_ROOT / 'pydantic_ai_slim' / 'pydantic_ai' / 'models'

# Provider module -> the display name(s) used in the docstrings. A module may
# appear under several names ("Gemini"/"Google" both mean google.py).
DOC_NAME_TO_MODULE = {
    'Gemini': 'google.py',
    'Google': 'google.py',
    'OpenAI': 'openai.py',
    'Anthropic': 'anthropic.py',
    'Groq': 'groq.py',
    'Cohere': 'cohere.py',
    'Mistral': 'mistral.py',
    'Bedrock': 'bedrock.py',
    'HuggingFace': 'huggingface.py',
    'xAI': 'xai.py',
    'Cerebras': 'cerebras.py',
    'OpenRouter': 'openrouter.py',
    'Z.AI': 'zai.py',
    'MCP Sampling': 'mcp_sampling.py',
}

PROVIDER_FILES = sorted(DOC_NAME_TO_MODULE.values())

# xAI reads settings through `_XAI_MODEL_SETTINGS_MAPPING` rather than
# `model_settings.get(...)`, so its reads can't be detected by the text scan.
XAI_MAPPING = {
    'max_tokens',
    'temperature',
    'top_p',
    'stop_sequences',
    'parallel_tool_calls',
    'presence_penalty',
    'frequency_penalty',
    'seed',
}

# Reads that don't use the literal `model_settings.get('<field>'` shape
# (a local variable named `settings`, or an `in model_settings` membership test).
EXTRA_READ_PATTERNS = {
    ('extra_headers', 'bedrock.py'): "settings.get('extra_headers'",
    ('extra_body', 'cerebras.py'): "settings.get('extra_body'",
    ('parallel_tool_calls', 'anthropic.py'): "'parallel_tool_calls' in model_settings",
}

# Expected "Supported by:" lists, keyed by module name. Derived from the
# provider sources (see the issue above); update this table AND the docstrings
# when a provider starts or stops reading a general setting.
_SUPPORTED_BY: dict[str, set[str]] = {
    'max_tokens': {'google.py', 'anthropic.py', 'openai.py', 'groq.py', 'cohere.py', 'mistral.py', 'bedrock.py', 'mcp_sampling.py', 'xai.py', 'huggingface.py'},
    'temperature': {'google.py', 'anthropic.py', 'openai.py', 'groq.py', 'cohere.py', 'mistral.py', 'bedrock.py', 'xai.py', 'huggingface.py', 'mcp_sampling.py'},
    'top_p': {'google.py', 'anthropic.py', 'openai.py', 'groq.py', 'cohere.py', 'mistral.py', 'bedrock.py', 'xai.py', 'huggingface.py'},
    'top_k': {'google.py', 'anthropic.py', 'cohere.py', 'bedrock.py'},
    'timeout': {'google.py', 'anthropic.py', 'openai.py', 'groq.py', 'mistral.py'},
    'parallel_tool_calls': {'openai.py', 'groq.py', 'anthropic.py', 'xai.py', 'mistral.py'},
    'seed': {'openai.py', 'groq.py', 'cohere.py', 'mistral.py', 'google.py', 'xai.py', 'huggingface.py'},
    'presence_penalty': {'openai.py', 'groq.py', 'cohere.py', 'google.py', 'mistral.py', 'xai.py', 'huggingface.py'},
    'frequency_penalty': {'openai.py', 'groq.py', 'cohere.py', 'google.py', 'mistral.py', 'xai.py', 'huggingface.py'},
    'logit_bias': {'openai.py', 'groq.py', 'huggingface.py'},
    'stop_sequences': {'openai.py', 'anthropic.py', 'bedrock.py', 'mistral.py', 'groq.py', 'cohere.py', 'google.py', 'xai.py', 'huggingface.py', 'mcp_sampling.py'},
    'extra_headers': {'openai.py', 'anthropic.py', 'bedrock.py', 'google.py', 'groq.py'},
    'extra_body': {'openai.py', 'anthropic.py', 'groq.py', 'cerebras.py', 'huggingface.py', 'openrouter.py', 'zai.py'},
    'service_tier': {'openai.py', 'anthropic.py', 'bedrock.py', 'google.py'},
}

_FIELD_DOCSTRING_RE = re.compile(r'^\s{4}([a-z_]+): [^\n]*\n    """.*?"""', re.M | re.S)


def _docstring_providers(field: str) -> set[str]:
    """Parse the field's "Supported by:" docstring block into module names."""
    text = SETTINGS_FILE.read_text(encoding='utf-8')
    match = _FIELD_DOCSTRING_RE.search(text)
    # locate the field's own docstring: walk matches until the field line
    for m in _FIELD_DOCSTRING_RE.finditer(text):
        if m.group(1) == field:
            doc = m.group(0)
            break
    else:  # pragma: no cover - guarded by the test
        raise AssertionError(f'field {field!r} not found in settings.py')
    block = doc.split('Supported by:', 1)[1]
    providers: set[str] = set()
    for line in block.splitlines():
        line = line.strip()
        if line.startswith('* '):
            name = line[2:].split(' (', 1)[0].strip()
            module = DOC_NAME_TO_MODULE.get(name)
            if module is None:  # pragma: no cover - unknown display name
                raise AssertionError(f'unknown provider name {name!r} in {field} docstring')
            providers.add(module)
    return providers


def _source_reads(field: str) -> set[str]:
    """Find which provider modules directly read the field from model settings."""
    reads: set[str] = set()
    for filename in PROVIDER_FILES:
        if filename == 'xai.py':
            if field in XAI_MAPPING:
                reads.add(filename)
            continue
        source = (MODELS_DIR / filename).read_text(encoding='utf-8')
        if f"model_settings.get('{field}'" in source:
            reads.add(filename)
            continue
        pattern = EXTRA_READ_PATTERNS.get((field, filename))
        if pattern is not None and pattern in source:
            reads.add(filename)
    return reads


@pytest.mark.parametrize('field', sorted(_SUPPORTED_BY))
def test_supported_by_list_matches_source(field: str) -> None:
    assert field in ModelSettings.__annotations__, f'unknown field {field!r} in the expected table'
    assert _docstring_providers(field) == _SUPPORTED_BY[field], field
    assert _source_reads(field) <= _SUPPORTED_BY[field], field


def test_expected_table_covers_all_checkable_fields() -> None:
    """Every ModelSettings field must be in the table or be a hand-maintained field."""
    hand_maintained = {'tool_choice', 'thinking'}
    missing = set(ModelSettings.__annotations__) - set(_SUPPORTED_BY) - hand_maintained
    assert not missing, f'fields missing from the expected table: {missing}'
