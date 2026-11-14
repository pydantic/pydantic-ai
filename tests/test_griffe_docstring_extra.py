"""Direct regression tests for `pydantic_ai._griffe` docstring parsing.

The existing coverage in `tests/test_tools.py` only exercises `doc_descriptions`
indirectly through full `Tool`/`Agent` snapshot tests, and only for the main
happy paths. These unit tests pin down branches that no existing test covers:

- functions without a docstring (`func.__doc__ is None`)
- `_infer_docstring_style` detection of sphinx / numpy / google markers and its
  fallback when no marker is present
- parameters documented only in Google `Keyword Args:` / `Other Parameters:`
  sections (parsed into subclasses of `DocstringSectionParameters`)
- first-description-wins when the same parameter appears in two sections
- star-stripping of `*args` / `**kwargs` parameter names
- a returns section without an annotation (no `<type>` tag emitted)
- an explicit `docstring_format` overriding what `auto` would infer

These functions build the tool descriptions sent to LLMs, so a regression here
silently changes every agent's tool schema; a wrong `<summary>`/`<returns>`
XML shape or a dropped parameter description would be user-visible.

Docstrings are written the way real `__doc__` strings look (indented inside a
function body) because griffe's section parsing is indentation-sensitive.
"""

from __future__ import annotations as _annotations

import inspect

import pytest

from pydantic_ai._griffe import _infer_docstring_style, doc_descriptions  # pyright: ignore[reportPrivateUsage]


def test_doc_descriptions_no_docstring():
    def no_doc() -> None:  # pragma: no cover
        pass

    assert no_doc.__doc__ is None
    desc, params = doc_descriptions(no_doc, docstring_format='auto')
    assert desc is None
    assert params == {}


def test_infer_docstring_style_sphinx_marker():
    def fn(x: int) -> str:  # pragma: no cover
        """
        Do the thing.

        :param x: The input value.
        """
        return str(x)

    assert _infer_docstring_style(fn.__doc__ or '') == 'sphinx'


def test_infer_docstring_style_numpy_marker():
    def fn(x: int) -> str:  # pragma: no cover
        """
        Do the thing.

        Parameters
        ----------
        x : int
            The input value.
        """
        return str(x)

    assert _infer_docstring_style(fn.__doc__ or '') == 'numpy'


def test_infer_docstring_style_google_marker():
    def fn(x: int) -> str:  # pragma: no cover
        """
        Do the thing.

        Args:
            x: The input value.
        """
        return str(x)

    assert _infer_docstring_style(fn.__doc__ or '') == 'google'


def test_infer_docstring_style_no_marker_falls_back_to_google():
    def fn() -> None:  # pragma: no cover
        """
        Just some prose, no sections.
        """
        pass

    assert _infer_docstring_style(fn.__doc__ or '') == 'google'


def test_doc_descriptions_auto_infers_sphinx():
    def fn(x: int) -> str:  # pragma: no cover
        """Do the thing.

        :param x: The input value.
        :returns: The output value.
        """
        return str(x)

    desc, params = doc_descriptions(fn, docstring_format='auto')
    assert params == {'x': 'The input value.'}
    # a sphinx returns field carries no type annotation, so no <type> tag is emitted
    assert (
        desc == '<summary>Do the thing.</summary>\n<returns>\n<description>The output value.</description>\n</returns>'
    )


def test_doc_descriptions_auto_infers_numpy():
    def fn(x: int) -> str:  # pragma: no cover
        """Do the thing.

        Parameters
        ----------
        x : int
            The input value.

        Returns
        -------
        str
            The output value.
        """
        return str(x)

    desc, params = doc_descriptions(fn, docstring_format='auto')
    assert params == {'x': 'The input value.'}
    assert desc == (
        '<summary>Do the thing.</summary>\n'
        '<returns>\n<type>str</type>\n<description>The output value.</description>\n</returns>'
    )


def test_doc_descriptions_auto_falls_back_to_google():
    def fn() -> None:  # pragma: no cover
        """Just some prose, no sections."""
        pass

    desc, params = doc_descriptions(fn, docstring_format='auto')
    assert desc == 'Just some prose, no sections.'
    assert params == {}


def test_doc_descriptions_keyword_args_section_params_included():
    def fn(x: int, *, verbose: bool) -> str:  # pragma: no cover
        """Do the thing.

        Args:
            x: The input value.

        Keyword Args:
            verbose: Whether to be verbose.
        """
        return str(x)

    desc, params = doc_descriptions(fn, docstring_format='auto')
    assert params == {'x': 'The input value.', 'verbose': 'Whether to be verbose.'}
    assert desc == 'Do the thing.'


def test_doc_descriptions_other_parameters_section_params_included():
    def fn(x: int, timeout: float) -> str:  # pragma: no cover
        """Do the thing.

        Args:
            x: The input value.

        Other Parameters:
            timeout: Maximum wait time.
        """
        return str(x)

    _, params = doc_descriptions(fn, docstring_format='auto')
    assert params == {'x': 'The input value.', 'timeout': 'Maximum wait time.'}


def test_doc_descriptions_first_description_wins():
    def fn(x: int) -> str:  # pragma: no cover
        """Do the thing.

        Args:
            x: First description.

        Keyword Args:
            x: Second description.
        """
        return str(x)

    _, params = doc_descriptions(fn, docstring_format='auto')
    assert params == {'x': 'First description.'}


def test_doc_descriptions_stars_stripped_from_var_positional_and_keyword():
    def fn(*args: int, **kwargs: str) -> None:  # pragma: no cover
        """Do the thing.

        Args:
            *args: Extra positional arguments.
            **kwargs: Extra keyword arguments.
        """
        pass

    _, params = doc_descriptions(fn, docstring_format='auto')
    assert params == {'args': 'Extra positional arguments.', 'kwargs': 'Extra keyword arguments.'}


def test_doc_descriptions_returns_without_annotation_omits_type_tag():
    def fn() -> str:  # pragma: no cover
        """Do the thing.

        Returns:
            Whatever the tool produced.
        """
        return ''

    desc, params = doc_descriptions(fn, docstring_format='auto')
    assert (
        desc
        == '<summary>Do the thing.</summary>\n<returns>\n<description>Whatever the tool produced.</description>\n</returns>'
    )
    assert params == {}


def test_doc_descriptions_empty_description_before_returns_omits_summary():
    def fn() -> str:  # pragma: no cover
        """
        Returns:
            The result as a string.
        """
        return ''

    desc, _ = doc_descriptions(fn, docstring_format='auto')
    assert desc == '<returns>\n<description>The result as a string.</description>\n</returns>'


def test_doc_descriptions_explicit_format_overrides_inference():
    def fn(x: int) -> str:  # pragma: no cover
        """Do the thing.

        Parameters
        ----------
        x : int
            The input value.
        """
        return str(x)

    # `auto` would infer numpy for this docstring (asserted in
    # test_infer_docstring_style_numpy_marker); an explicit format must take
    # precedence. Parsing it as google yields no recognizable sections, hence
    # no params and a plain-text description.
    assert _infer_docstring_style(fn.__doc__ or '') == 'numpy'
    desc, params = doc_descriptions(fn, docstring_format='google')
    assert params == {}
    # griffe dedents the docstring before parsing, so compare against the dedented form
    # (`inspect.cleandoc` normalizes the indentation that only CPython <3.13 keeps in `__doc__`)
    assert desc == inspect.cleandoc(fn.__doc__ or '')


@pytest.mark.parametrize(
    'doc, expected',
    [
        ('\nDo the thing.\n\n:param x: The input value.\n', 'sphinx'),
        ('\nDo the thing.\n\nParameters\n----------\nx : int\n    The input value.\n', 'numpy'),
        ('\nDo the thing.\n\nArgs:\n    x: The input value.\n', 'google'),
        ('\nJust some prose.\n', 'google'),
    ],
)
def test_infer_docstring_style_table(doc: str, expected: str):
    assert _infer_docstring_style(doc) == expected
