"""Spec loading shared by capabilities whose `from_spec` restates `__init__` with narrower types."""

from __future__ import annotations

import inspect

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT


def use_default_from_spec(
    cls: type[AbstractCapability[AgentDepsT]], narrowed: type[AbstractCapability[AgentDepsT]]
) -> None:
    """Load `cls`, a subclass of `narrowed`, through the default `from_spec`, unless it inherits another override.

    `narrowed.from_spec` restates `narrowed.__init__`'s parameters and defaults, so a subclass loading through it
    would reject the fields it adds and replace the defaults it changes. The default passes a spec's arguments on.
    Core then reads the default's signature for the subclass's spec schema, which publishes only its name; a subclass
    that wants its fields in the schema overrides `from_spec`.
    """
    if inspect.getattr_static(cls, 'from_spec') is vars(narrowed)['from_spec']:
        cls.from_spec = vars(AbstractCapability)['from_spec']
