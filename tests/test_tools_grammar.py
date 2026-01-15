# pyright: reportPrivateUsage=false
"""Tests for grammar constraint tools: FreeformText, RegexGrammar, LarkGrammar."""

from __future__ import annotations

import importlib.util
from typing import Annotated
from unittest.mock import patch

import pytest
from pydantic.json_schema import GenerateJsonSchema

from pydantic_ai import UserError
from pydantic_ai._function_schema import extract_text_format, function_schema
from pydantic_ai.tools import FreeformText, LarkGrammar, RegexGrammar


class TestRegexGrammar:
    """Tests for RegexGrammar class."""

    def test_validate_success(self):
        """Test that validation passes for matching strings."""
        grammar = RegexGrammar(r'\d{3}-\d{4}')
        result = grammar._validate('555-1234')
        assert result == '555-1234'

    def test_validate_failure(self):
        """Test that validation fails for non-matching strings."""
        grammar = RegexGrammar(r'\d{3}-\d{4}')
        with pytest.raises(ValueError, match='String does not match regex pattern'):
            grammar._validate('invalid-phone')

    def test_validate_failure_empty_string(self):
        """Test that validation fails for empty strings when pattern requires content."""
        grammar = RegexGrammar(r'\d+')
        with pytest.raises(ValueError, match='String does not match regex pattern'):
            grammar._validate('')

    def test_invalid_pattern_raises(self):
        """Test that invalid regex patterns raise ValueError during construction."""
        with pytest.raises(ValueError, match='Regex pattern is invalid'):
            RegexGrammar(r'[invalid')

    def test_get_description(self):
        """Test that get_description returns the pattern description."""
        grammar = RegexGrammar(r'\d{3}-\d{4}')
        assert grammar.get_description() == r'Input must match regex pattern: \d{3}-\d{4}'


@pytest.mark.skipif(not importlib.util.find_spec('lark'), reason='lark not installed')
class TestLarkGrammar:
    """Tests for LarkGrammar class."""

    def test_validate_success(self):
        """Test that validation passes for strings matching the grammar."""
        grammar = LarkGrammar('start: "hello"')
        result = grammar._validate('hello')
        assert result == 'hello'

    def test_validate_failure(self):
        """Test that validation fails for strings not matching the grammar."""
        grammar = LarkGrammar('start: "hello"')
        with pytest.raises(ValueError, match='String does not match Lark grammar'):
            grammar._validate('goodbye')

    def test_validate_import_error_graceful_degradation(self):
        """Test that validation gracefully degrades when lark is not installed."""
        import builtins
        from collections.abc import Sequence
        from types import ModuleType
        from typing import Any

        grammar = LarkGrammar.__new__(LarkGrammar)
        grammar.definition = 'start: "hello"'

        # Mock lark import to fail
        original_import = builtins.__import__

        def mock_import(
            name: str,
            globals: dict[str, Any] | None = None,
            locals: dict[str, Any] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> ModuleType:
            if name == 'lark':
                raise ImportError('No module named lark')
            return original_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, '__import__', mock_import):
            result = grammar._validate('anything')
            assert result == 'anything'

    def test_invalid_grammar_raises(self):
        """Test that invalid Lark grammars raise ValueError during construction."""
        with pytest.raises(ValueError, match='Lark grammar is invalid'):
            LarkGrammar('invalid grammar syntax ][')

    def test_get_description_short(self):
        """Test that get_description returns full grammar when short."""
        grammar = LarkGrammar('start: "hello"')
        assert grammar.get_description() == 'Input must match Lark grammar:\nstart: "hello"'

    def test_get_description_long_truncated(self):
        """Test that get_description truncates long grammars."""
        # Create a grammar with more than 500 characters
        long_rules = '\n'.join(f'rule{i}: "value{i}"' for i in range(100))
        long_grammar = f'start: rule0\n{long_rules}'
        assert len(long_grammar) > 500

        grammar = LarkGrammar(long_grammar)
        description = grammar.get_description()

        assert description.startswith('Input must match Lark grammar (truncated):\n')
        assert description.endswith('...')
        # Should only include first 500 chars of grammar
        assert len(description) < len(long_grammar)


class TestFreeformText:
    """Tests for FreeformText class."""

    def test_get_description_returns_none(self):
        """Test that FreeformText.get_description returns None."""
        ft = FreeformText()
        assert ft.get_description() is None

    def test_hash(self):
        """Test that FreeformText instances are hashable and equal."""
        ft1 = FreeformText()
        ft2 = FreeformText()
        assert hash(ft1) == hash(ft2)


class TestMultipleTextFormatParameters:
    """Tests for error handling when multiple TextFormat parameters are used."""

    def test_multiple_text_format_error(self):
        """Test that multiple TextFormat parameters raise UserError."""

        def tool_with_multiple_grammars(
            param1: Annotated[str, RegexGrammar(r'\d+')],
            param2: Annotated[str, FreeformText()],
        ) -> str:
            return f'{param1} {param2}'  # pragma: no cover

        with pytest.raises(UserError, match='Only one parameter may have a TextFormat annotation'):
            function_schema(
                function=tool_with_multiple_grammars,
                takes_ctx=False,
                schema_generator=GenerateJsonSchema,
            )


class TestDescriptionCombination:
    """Tests for combining docstring descriptions with grammar descriptions."""

    def test_description_with_grammar_combined(self):
        """Test that parameter description is combined with grammar constraint description."""

        def tool_with_description(
            phone: Annotated[str, RegexGrammar(r'\d{3}-\d{4}')],
        ) -> str:
            """Look up a phone number.

            Args:
                phone: The phone number to look up

            Returns:
                Contact info
            """
            return f'Found: {phone}'  # pragma: no cover

        schema = function_schema(
            function=tool_with_description,
            takes_ctx=False,
            schema_generator=GenerateJsonSchema,
        )

        # Get the JSON schema to check the description
        json_schema = schema.json_schema
        phone_desc = json_schema['properties']['phone'].get('description', '')

        # Description should combine docstring description with grammar description
        assert 'The phone number to look up' in phone_desc
        assert r'Input must match regex pattern: \d{3}-\d{4}' in phone_desc

    def test_description_without_docstring(self):
        """Test that grammar description is used when no docstring description exists."""

        def tool_no_docstring(
            phone: Annotated[str, RegexGrammar(r'\d{3}-\d{4}')],
        ) -> str:
            return f'Found: {phone}'  # pragma: no cover

        schema = function_schema(
            function=tool_no_docstring,
            takes_ctx=False,
            schema_generator=GenerateJsonSchema,
        )

        json_schema = schema.json_schema
        phone_desc = json_schema['properties']['phone'].get('description', '')

        # Description should just be the grammar description
        assert phone_desc == r'Input must match regex pattern: \d{3}-\d{4}'


class TestExtractTextFormat:
    """Tests for extract_text_format function edge cases."""

    def test_non_annotated_returns_none(self):
        """Test that non-Annotated types return None."""

        assert extract_text_format(str) is None
        assert extract_text_format(int) is None

    def test_annotated_with_text_format_returns_format(self):
        """Test that Annotated with TextFormat returns the format."""

        regex = RegexGrammar(r'\d+')
        result = extract_text_format(Annotated[str, regex])
        assert result == regex

    def test_annotated_non_str_returns_none(self):
        """Test that Annotated[non-str, TextFormat] returns None."""

        # TextFormat on non-str type should return None
        assert extract_text_format(Annotated[int, RegexGrammar(r'\d+')]) is None

    def test_annotated_without_text_format_returns_none(self):
        """Test that Annotated without TextFormat returns None."""

        assert extract_text_format(Annotated[str, 'some metadata']) is None
