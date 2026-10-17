"""Doc pages name models through the aliases in `docs/model_aliases/aliases.json`.

Updating a model is then a one-line change. See the "Model names" section of `docs/AGENTS.md`.
"""

from __future__ import annotations as _annotations

import re
from collections.abc import Callable, Collection
from pathlib import Path

import pytest
from typing_inspection.introspection import get_literal_values

from pydantic_ai.embeddings import KnownEmbeddingModelName
from pydantic_ai.images import KnownImageGenerationModelName
from pydantic_ai.models import known_model_names
from pydantic_ai.realtime import KnownRealtimeModelName

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
    'harness/gh-aw.md': 'the engine runs `openai/` models on Chat Completions, where the aliased ones reject tools at their default reasoning effort',
    'realtime/azure.md': 'Azure Voice Live serves specific models',
    'realtime/overview.md': 'Azure Voice Live serves specific models',
    'realtime/troubleshooting.md': 'Azure Voice Live serves specific models',
    'realtime/openai.md': 'documents GPT-Live and gpt-realtime model behavior',
    'realtime/gemini.md': 'documents the behavior of each Gemini Live model',
    'realtime/xai.md': 'documents Grok Voice model names',
    'image-generation.md': 'the size tables differ per image model',
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
# Chat models are aliased by tier, and a provider that defines one tier defines all of them.
_TIERS = ('small', 'medium', 'large', 'xlarge')
# Other kinds of model have one alias per provider, checked against their own known names.
_KINDS = {
    kind: frozenset(get_literal_values(names.__value__, unpack_type_aliases='eager'))
    for kind, names in (
        ('realtime', KnownRealtimeModelName),
        ('image', KnownImageGenerationModelName),
        ('embedding', KnownEmbeddingModelName),
    )
}
# Models the providers serve that Pydantic AI doesn't list as known yet; drop each once it does.
_NOT_YET_KNOWN = {'openai:gpt-image-2.5-flare'}
# Chat models the tiers don't cover, such as audio and research models, may be named directly.
_UNTIERED = re.compile(r'image|audio|tts|whisper|guard|moderation|computer-use|research|codex|multi-agent|^o\d')


def _providers_with(kinds: Collection[str]) -> set[str]:
    return {provider for provider, _, kind in (alias.partition(':') for alias in MODEL_ALIASES) if kind in kinds}


def _chat_model_names() -> set[str]:
    names: set[str] = set()
    providers = _providers_with(_TIERS)
    for model in _KNOWN_MODELS:
        provider, _, name = model.partition(':')
        if provider in providers and not _UNTIERED.search(name):
            # Bedrock's aliases stand for its Claude models only.
            if provider != 'bedrock' or 'anthropic.claude' in name:
                names.add(name)
    return names


def _kind_model_names() -> set[str]:
    names: set[str] = set()
    for kind, models in _KINDS.items():
        providers = _providers_with([kind])
        names |= {
            name for provider, _, name in map(lambda m: m.partition(':'), models) if provider.split('-')[0] in providers
        }
    return names


def _names_pattern(names: set[str]) -> str:
    return '(?P<name>' + '|'.join(map(re.escape, sorted(names, key=len, reverse=True))) + r')(?![\w.-])'


# A chat model named anywhere. Realtime, image and embedding names also name model families in prose
# (`gpt-realtime`), so for those only code is checked: a quoted name, or one behind a provider prefix.
_HARDCODED_RES = (
    re.compile(r'(?<![\w./-])(?:(?P<prefix>[a-z][a-z0-9-]*):)?' + _names_pattern(_chat_model_names())),
    re.compile(r'(?:(?<![\w./-])(?P<prefix>[a-z][a-z0-9-]*):|(?<=[\'"]))' + _names_pattern(_kind_model_names())),
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


def _alias_problems(text: str) -> list[tuple[int, str]]:
    """Each alias in `text` that isn't known or resolves to an unknown model, with its position."""
    problems: list[tuple[int, str]] = []
    for match in _ALIAS_LIKE_RE.finditer(text):
        alias = ALIAS_RE.fullmatch(match.group())
        if alias is None or alias.group(1) not in MODEL_ALIASES:
            problems.append((match.start(), f'{match.group()} is not a known alias'))
    for match in _PREFIXED_ALIAS_RE.finditer(text):
        # Only chat models have provider routes like `gateway/` or `google-cloud:` to check.
        if match.group(2) not in MODEL_ALIASES or match.group(2).partition(':')[2] not in _TIERS:
            continue
        prefix = match.group('prefix')
        model = prefix + resolve_model_aliases(match.group()[len(prefix) :])
        if model.partition(':')[0] in _KNOWN_PREFIXES and model not in _KNOWN_MODELS | _NOT_YET_KNOWN:
            problems.append((match.start(), f'{match.group()} resolves to unknown {model!r}'))
    return problems


def _hardcoded_models(text: str) -> list[tuple[int, str]]:
    """Each aliased model in `text` that is named directly instead of by alias, with its position."""
    return [
        (match.start(), match.group())
        for pattern in _HARDCODED_RES
        for match in pattern.finditer(text)
        # Other providers, like `snowflake:` or `heroku:`, serve some of these models under the same names.
        if (prefix := match.group('prefix')) is None or prefix.split('-')[0] in _PROVIDERS
    ]


def _report(pages: list[Path], find: Callable[[str], list[tuple[int, str]]]) -> list[str]:
    report: list[str] = []
    for page in pages:
        text = page.read_text(encoding='utf-8')
        for index, problem in find(text):
            report.append(f'docs/{page.relative_to(DOCS).as_posix()}:{text.count(chr(10), 0, index) + 1}: {problem}')
    return report


def test_aliases_name_known_models():
    for alias, model in MODEL_ALIASES.items():
        provider, _, tier = alias.partition(':')
        assert model.startswith(f'{provider}:'), f'{alias!r} should name a `{provider}:` model'
        known = _KNOWN_MODELS if tier in _TIERS else _KINDS.get(tier)
        assert known is not None, f'{alias!r} should use a tier from {_TIERS} or a kind from {tuple(_KINDS)}'
        assert model in known | _NOT_YET_KNOWN, (
            f'{alias!r} names {model!r}, which Pydantic AI does not list as a known model'
        )
    all_known = _KNOWN_MODELS.union(*_KINDS.values())
    assert not _NOT_YET_KNOWN & all_known, 'Drop the models Pydantic AI now lists from `_NOT_YET_KNOWN`'


def test_chat_providers_define_every_tier():
    # A page can then switch an example to another provider without checking which tiers it has.
    missing = sorted(
        f'{provider}:{tier}'
        for provider in _providers_with(_TIERS)
        for tier in _TIERS
        if f'{provider}:{tier}' not in MODEL_ALIASES
    )
    assert not missing, f'Define every chat tier, see `docs/model_aliases/AGENTS.md`: {missing}'


def test_docs_aliases_resolve_to_known_models():
    problems = _report(_pages(), _alias_problems)
    assert not problems, 'Use the model aliases from docs/model_aliases/aliases.json:\n' + '\n'.join(problems)


def test_docs_name_models_by_alias():
    pages = [page for page in _pages() if not _is_exempt(page.relative_to(DOCS).as_posix())]
    hardcoded = _report(pages, _hardcoded_models)
    assert not hardcoded, (
        'Name models with an alias from docs/model_aliases/aliases.json, like `{{ openai:medium }}`, '
        'or add the page to `EXEMPT_PAGES` if it is about specific models:\n' + '\n'.join(hardcoded)
    )


def test_checks_flag_misspelled_aliases_and_model_ids():
    model = MODEL_ALIASES['openai:medium']
    image = min(model for model in _KINDS['image'] if model.startswith('openai:'))
    embedding = MODEL_ALIASES['openai:embedding'].partition(':')[2]
    text = (
        "Agent('{{openai:medium}}'), Agent('{{ openai:huge }}'), Agent('azure:{{ openai:medium.name }}'),\n"
        f"Agent('google-cloud:{{{{ openai:medium.name }}}}'), Agent('{model}'), Agent('heroku:{model.partition(':')[2]}')\n"
        f"ImageGenerator('{image}'), Embedder('azure:{embedding}'), the `{image.partition(':')[2]}` model"
    )
    assert [problem for _, problem in _alias_problems(text)] == [
        '{{openai:medium}} is not a known alias',
        '{{ openai:huge }} is not a known alias',
        f"google-cloud:{{{{ openai:medium.name }}}} resolves to unknown 'google-cloud:{model.partition(':')[2]}'",
    ]
    assert [name for _, name in _hardcoded_models(text)] == [model, image]
    assert _report([DOCS / 'index.md'], lambda text: [(0, 'problem')]) == ['docs/index.md:1: problem']


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
