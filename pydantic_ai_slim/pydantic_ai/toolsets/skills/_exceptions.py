"""Exceptions for skill-related errors.

This module defines the exception hierarchy for the skills toolset.
All skill exceptions inherit from [`SkillException`][pydantic_ai.toolsets.skills.SkillException],
making it easy to catch all skill-related errors.

Exception hierarchy:
- [`SkillException`][pydantic_ai.toolsets.skills.SkillException]: Base exception for all skill errors
    - [`SkillNotFoundError`][pydantic_ai.toolsets.skills.SkillNotFoundError]: Skill not found in any source
    - [`SkillValidationError`][pydantic_ai.toolsets.skills.SkillValidationError]: Skill validation failed
    - [`SkillResourceNotFoundError`][pydantic_ai.toolsets.skills.SkillResourceNotFoundError]: Resource file not found
    - [`SkillResourceLoadError`][pydantic_ai.toolsets.skills.SkillResourceLoadError]: Failed to load resource
    - [`SkillScriptExecutionError`][pydantic_ai.toolsets.skills.SkillScriptExecutionError]: Script execution failed
"""

from __future__ import annotations


class SkillException(Exception):
    """Base exception for skill-related errors."""


class SkillNotFoundError(SkillException):
    """Skill not found in any source."""


class SkillValidationError(SkillException):
    """Skill validation failed."""


class SkillResourceNotFoundError(SkillException):
    """Skill resource not found."""


class SkillResourceLoadError(SkillException):
    """Failed to load skill resources."""


class SkillScriptExecutionError(SkillException):
    """Skill script execution failed."""
