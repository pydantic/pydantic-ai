"""Auth profiles: several accounts for one provider, named in the model as `PROVIDER@PROFILE:NAME`.

A model without `@PROFILE` uses the provider's default account, stored where it always was, unless the
`accounts.pool` setting runs it on every signed-in account; `@default` names that account alone. A profile's
credentials are stored under the account `PROVIDER@PROFILE`. Only the prefix before the first `:` is
read, so model IDs that contain `@` themselves, such as Vertex's `claude-...@20240620`, are left alone.
"""

import re
from dataclasses import dataclass

_NAME = re.compile(r'[a-z0-9][a-z0-9_-]{0,31}')

ALL = '*'
"""The profile meaning every signed-in account of the provider, tried in `/accounts` order: `openai-codex@*:gpt-6`."""

DEFAULT = 'default'
"""The profile meaning the account without a profile alone, even when `accounts.pool` pools plain names."""


def check_name(value: str, *, kind: str) -> str:
    """Profile and chain names share one short, file-name-safe format."""
    if not _NAME.fullmatch(value):
        raise ValueError(
            f'{kind} {value!r} must be 1 to 32 lowercase letters, digits, hyphens, or underscores, '
            'starting with a letter or digit.'
        )
    return value


@dataclass(frozen=True, kw_only=True)
class ModelRef:
    """A model name split into provider, profile, and the provider's own model name."""

    provider: str
    profile: str | None
    name: str

    @property
    def account(self) -> str:
        """The credential-store account: the provider for the default profile, else `PROVIDER@PROFILE`."""
        return account(self.provider, self.profile)


def parse_model(model: str) -> ModelRef:
    """Split `PROVIDER[@PROFILE]:NAME`, rejecting a malformed profile; a name without `:` has no provider."""
    prefix, separator, name = model.partition(':')
    if not separator:
        return ModelRef(provider='', profile=None, name=model)
    provider, profile = split_profile(prefix)
    return ModelRef(provider=provider, profile=profile, name=name)


def split_profile(name: str) -> tuple[str, str | None]:
    """`openai-codex@work` as `('openai-codex', 'work')`; no `@` means the default profile.

    `ALL` and `DEFAULT` come back as they are, for the caller to handle.
    """
    provider, at, profile = name.partition('@')
    if not at:
        return provider, None
    if profile in (ALL, DEFAULT):
        return provider, profile
    return provider, check_name(profile, kind='Profile')


def account(provider: str, profile: str | None) -> str:
    """Where a profile's credentials are stored; the default profile keeps the provider's own account."""
    return provider if profile is None else f'{provider}@{profile}'


def provider_of(model: str) -> str:
    """The provider prefix without any profile: `openai-codex@work:gpt-6` gives `openai-codex`."""
    return model.partition(':')[0].partition('@')[0]


def base_model(model: str) -> str:
    """The model without its profile, for provider-specific controls and defaults."""
    prefix, separator, name = model.partition(':')
    return f'{prefix.partition("@")[0]}{separator}{name}'


def with_profile(model: str, profile: str | None) -> str:
    """`PROVIDER:NAME` as `PROVIDER@PROFILE:NAME`; the default profile leaves it unchanged."""
    if profile is None:
        return model
    prefix, _, name = model.partition(':')
    return f'{prefix}@{profile}:{name}'
