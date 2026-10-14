"""Doc pages name models through the aliases in `docs/model_aliases.py`, so updating a model is a one-line change.

See the "Model names" section of `docs/AGENTS.md`.
"""

from __future__ import annotations as _annotations

import re
from pathlib import Path

import pytest

from pydantic_ai.models import known_model_names

from .docs_model_aliases import MODEL_ALIASES, resolve_model_aliases, restore_model_aliases

DOCS = Path(__file__).parent.parent / 'docs'

# Pages that are deliberately about specific models or versions, so they may name them directly.
EXEMPT_PAGES = {
    'changelog.md': 'release notes name the models of their time',
    'migration.md': 'migration notes name the models of their version',
    'capabilities/thinking.md': 'thinking support and settings differ per model',
    'evals/evaluators/built-in.md': "documents the judge's default model",
    'harness/clai2.md': 'lists the models Codex offers',
    'harness/compaction.md': 'context windows differ per model',
    'realtime/azure.md': 'Azure Voice Live serves specific models',
    'realtime/overview.md': 'Azure Voice Live serves specific models',
    'realtime/troubleshooting.md': 'Azure Voice Live serves specific models',
}
EXEMPT_DIRS = {
    'examples/': 'example pages describe apps whose code names its models',
    'models/': 'provider pages document model-specific behavior',
}
CHECKED_PAGES = {'models/overview.md'}

_PROVIDERS = {alias.partition(':')[0] for alias in MODEL_ALIASES}
_TIERS = ('small', 'medium', 'large')
# Models the tiers don't cover, such as image, audio and research models, may be named directly.
_UNTIERED = re.compile(r'image|audio|tts|whisper|guard|moderation|computer-use|research|codex|multi-agent|^o\d')


def _tiered_model_names() -> list[str]:
    names: set[str] = set()
    for model in known_model_names():
        provider, _, name = model.partition(':')
        if provider in _PROVIDERS and not _UNTIERED.search(name):
            # Bedrock's aliases stand for its Claude models only.
            if provider != 'bedrock' or 'anthropic.claude' in name:
                names.add(name)
    return sorted(names, key=len, reverse=True)


_HARDCODED_RE = re.compile(
    r'(?<![\w./-])(?:(?P<prefix>[a-z][a-z0-9-]*):)?(?P<name>'
    + '|'.join(map(re.escape, _tiered_model_names()))
    + r')(?![\w.-])'
)


def _pages() -> list[Path]:
    # All-caps files like `AGENTS.md` are contributor notes, not published pages.
    return sorted(path for path in DOCS.rglob('*.md') if not path.stem.isupper())


def _is_exempt(page: str) -> bool:
    return page not in CHECKED_PAGES and (page in EXEMPT_PAGES or page.startswith(tuple(EXEMPT_DIRS)))


def test_aliases_name_known_models():
    for alias, model in MODEL_ALIASES.items():
        provider, _, tier = alias.partition(':')
        assert tier in _TIERS, f'{alias!r} should use one of the tiers {_TIERS}'
        assert model.startswith(f'{provider}:'), f'{alias!r} should name a `{provider}:` model'
        assert model in known_model_names(), f'{alias!r} names {model!r}, which is not a `KnownModelName`'
    for provider in _PROVIDERS:
        assert {f'{provider}:{tier}' for tier in _TIERS} <= MODEL_ALIASES.keys(), f'{provider!r} lacks a tier'


def test_docs_use_known_aliases():
    for page in _pages():
        try:
            resolve_model_aliases(page.read_text(encoding='utf-8'))
        except ValueError as e:
            pytest.fail(f'{page.relative_to(DOCS)}: {e}')


def test_docs_name_models_by_alias():
    hardcoded: list[str] = []
    for page in _pages():
        relative = page.relative_to(DOCS).as_posix()
        if _is_exempt(relative):
            continue
        text = page.read_text(encoding='utf-8')
        for match in _HARDCODED_RE.finditer(text):
            prefix = match.group('prefix')
            # Other providers, like `openrouter:` or `heroku:`, serve these models under the same names.
            if prefix is None or prefix.split('-')[0] in _PROVIDERS:
                hardcoded.append(f'docs/{relative}:{text.count(chr(10), 0, match.start()) + 1}: {match.group()}')
    assert not hardcoded, (
        'Name models with an alias from docs/model_aliases.py, like `{{ openai:medium }}`, '
        'or add the page to `EXEMPT_PAGES` if it is about a specific model:\n' + '\n'.join(hardcoded)
    )


def test_resolve_model_aliases():
    model = MODEL_ALIASES['openai:medium']
    assert resolve_model_aliases("Agent('{{ openai:medium }}')  # {{openai:medium.name}}") == (
        f"Agent('{model}')  # {model.partition(':')[2]}"
    )
    with pytest.raises(ValueError, match=r"Unknown model alias '\{\{ openai:huge \}\}'"):
        resolve_model_aliases('{{ openai:huge }}')


def test_restore_model_aliases_aliases_only_updated_lines():
    model = MODEL_ALIASES['anthropic:small']
    source = "agent = Agent('{{ anthropic:small }}')\nprint(agent.model.model_name)\n#> old\n"
    resolved = resolve_model_aliases(source)
    updated = resolved.replace('#> old', f'#> {model.partition(":")[2]}\n#> {model}')

    assert restore_model_aliases(source, resolved, resolved) == source
    assert restore_model_aliases(source, resolved, updated) == (
        "agent = Agent('{{ anthropic:small }}')\n"
        'print(agent.model.model_name)\n'
        '#> {{ anthropic:small.name }}\n'
        '#> {{ anthropic:small }}\n'
    )
