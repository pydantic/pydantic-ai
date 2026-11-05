"""Experimental pydantic-ai-harness capabilities.

Anything under `pydantic_ai_harness.experimental` may change or be removed in any release,
without a deprecation period.  Importing an experimental capability emits a
`HarnessExperimentalWarning` that tells you how to silence the whole category at once.

Importing this module on its own does **not** emit a warning, so you can pull in
`HarnessExperimentalWarning` to silence the warnings before importing a capability.
"""

from pydantic_ai_harness.experimental._warn import HarnessExperimentalWarning

__all__ = ['HarnessExperimentalWarning']
