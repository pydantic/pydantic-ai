"""Doc pages name models through the aliases in `docs/model_aliases.py`, so updating a model is a one-line change.

See the "Model names" section of `docs/AGENTS.md`.
"""

from __future__ import annotations as _annotations

import re
from pathlib import Path

import pytest

from pydantic_ai.models import known_model_names

from .docs_model_aliases import (
    ALIAS_RE,
    MODEL_ALIASES,
    alias_model_names,
    resolve_model_aliases,
    restore_model_aliases,
)

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
    'models/': 'provider pages document model-specific behavior and catalogs',
}
# The models overview is a general guide, not a provider page.
CHECKED_PAGES = {'models/overview.md'}

_KNOWN_MODELS = frozenset(known_model_names())
_KNOWN_PREFIXES = {model.partition(':')[0] for model in _KNOWN_MODELS}
_PROVIDERS = {alias.partition(':')[0] for alias in MODEL_ALIASES}
_TIERS = ('small', 'medium', 'large')
# Models the tiers don't cover, such as image, audio and research models, may be named directly.
_UNTIERED = re.compile(r'image|audio|tts|whisper|guard|moderation|computer-use|research|codex|multi-agent|^o\d')


def _tiered_model_names() -> list[str]:
    names: set[str] = set()
    for model in _KNOWN_MODELS:
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
# Anything meant as an alias, so a misspelled one can't reach the docs site unresolved.
_ALIAS_LIKE_RE = re.compile(r'\{\{[^{}\n]*:[^{}\n]*\}\}')
# An alias with a provider prefix, like `gateway/{{ openai:small }}` or `google-cloud:{{ google:small.name }}`.
_PREFIXED_ALIAS_RE = re.compile(r'(?<![\w/-])(?P<prefix>(?:gateway/)?(?:[a-z][a-z0-9-]*:)?)' + ALIAS_RE.pattern)


def _pages() -> list[Path]:
    # All-caps files like `AGENTS.md` are contributor notes, not published pages.
    return sorted(path for path in DOCS.rglob('*.md') if not path.stem.isupper())


def _is_exempt(page: str) -> bool:
    return page not in CHECKED_PAGES and (page in EXEMPT_PAGES or page.startswith(tuple(EXEMPT_DIRS)))


def _location(page: Path, text: str, index: int) -> str:
    return f'docs/{page.relative_to(DOCS).as_posix()}:{text.count(chr(10), 0, index) + 1}'


def test_aliases_name_known_models():
    for alias, model in MODEL_ALIASES.items():
        provider, _, tier = alias.partition(':')
        assert tier in _TIERS, f'{alias!r} should use one of the tiers {_TIERS}'
        assert model.startswith(f'{provider}:'), f'{alias!r} should name a `{provider}:` model'
        assert model in _KNOWN_MODELS, f'{alias!r} names {model!r}, which is not a `KnownModelName`'
    for provider in _PROVIDERS:
        assert {f'{provider}:{tier}' for tier in _TIERS} <= MODEL_ALIASES.keys(), f'{provider!r} lacks a tier'


def test_docs_aliases_resolve_to_known_models():
    problems: list[str] = []
    for page in _pages():
        text = page.read_text(encoding='utf-8')
        for match in _ALIAS_LIKE_RE.finditer(text):
            alias = ALIAS_RE.fullmatch(match.group())
            if alias is None or alias.group(1) not in MODEL_ALIASES:
                problems.append(f'{_location(page, text, match.start())}: {match.group()} is not a known alias')
        for match in _PREFIXED_ALIAS_RE.finditer(text):
            if match.group(2) not in MODEL_ALIASES:
                continue
            prefix = match.group('prefix')
            model = prefix + resolve_model_aliases(match.group()[len(prefix) :])
            if model.partition(':')[0] in _KNOWN_PREFIXES and model not in _KNOWN_MODELS:
                problems.append(f'{_location(page, text, match.start())}: {match.group()} is unknown {model!r}')
    assert not problems, 'Use the model aliases from docs/model_aliases.py, like `{{ openai:medium }}`:\n' + '\n'.join(
        problems
    )


def test_docs_name_models_by_alias():
    hardcoded: list[str] = []
    for page in _pages():
        if _is_exempt(page.relative_to(DOCS).as_posix()):
            continue
        text = page.read_text(encoding='utf-8')
        for match in _HARDCODED_RE.finditer(text):
            prefix = match.group('prefix')
            # Other providers, like `snowflake:` or `heroku:`, serve some of these models under the same names.
            if prefix is None or prefix.split('-')[0] in _PROVIDERS:
                hardcoded.append(f'{_location(page, text, match.start())}: {match.group()}')
    assert not hardcoded, (
        'Name models with an alias from docs/model_aliases.py, like `{{ openai:medium }}`, '
        'or add the page to `EXEMPT_PAGES` if it is about specific models:\n' + '\n'.join(hardcoded)
    )


def test_resolve_model_aliases():
    model = MODEL_ALIASES['openai:medium']
    assert resolve_model_aliases("Agent('{{ openai:medium }}')  # {{ openai:medium.name }}") == (
        f"Agent('{model}')  # {model.partition(':')[2]}"
    )
    with pytest.raises(ValueError, match=r"Unknown model alias '\{\{ openai:huge \}\}'"):
        resolve_model_aliases('{{ openai:huge }}')


def test_alias_model_names_prefers_the_given_alias():
    model = MODEL_ALIASES['xai:large']
    assert alias_model_names(f"'{model}'", prefer=['xai:large']) == "'{{ xai:large }}'"


def test_restore_model_aliases_aliases_only_updated_lines():
    model = MODEL_ALIASES['anthropic:small']
    name = model.partition(':')[2]
    source = "agent = Agent('{{ anthropic:small }}')\nprint(agent.model.model_name)\n#> old\n"
    resolved = resolve_model_aliases(source)
    updated = resolved.replace('#> old', f'#> {name}\n#> {model}\n#> gateway/{model}\n#> anthropic-direct:{name}')

    assert restore_model_aliases(source, resolved, resolved) == source
    assert restore_model_aliases(source, resolved, updated) == (
        "agent = Agent('{{ anthropic:small }}')\n"
        'print(agent.model.model_name)\n'
        '#> {{ anthropic:small.name }}\n'
        '#> {{ anthropic:small }}\n'
        '#> gateway/{{ anthropic:small }}\n'
        '#> anthropic-direct:{{ anthropic:small.name }}\n'
    )
