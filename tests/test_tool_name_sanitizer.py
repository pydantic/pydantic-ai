"""Regression tests for ``pydantic_ai._utils.sanitize_tool_name``.

``sanitize_tool_name`` maps any character outside ``[a-zA-Z0-9_-]`` to ``_`` so
tool names are safe to hand to model providers. It had no direct coverage
despite being on a security-sensitive path (tool names flow into the model's
tool schema), so these pin its behavior.
"""

import pytest

from pydantic_ai._utils import sanitize_tool_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("get-weather", "get-weather"),  # hyphen preserved
        ("normal_name", "normal_name"),  # underscore preserved
        ("my tool!", "my_tool_"),  # space and bang -> underscore
        ("foo.bar", "foo_bar"),  # dot -> underscore
        ("a@b#c", "a_b_c"),  # punctuation -> underscore
        ("get/User:Data(1)", "get_User_Data_1_"),  # mixed separators
        ("café_名", "caf___"),  # non-ASCII -> underscore
        ("工具", "__"),  # all non-ASCII -> underscores
        ("", ""),  # empty stays empty
    ],
)
def test_sanitize_tool_name(raw: str, expected: str) -> None:
    assert sanitize_tool_name(raw) == expected


def test_sanitize_tool_name_only_allows_safe_chars():
    out = sanitize_tool_name("a!b@c#d$e%f^g&h*i(j)k-l_m+n=")
    assert out == "a_b_c_d_e_f_g_h_i_j_k-l_m_n_"
    assert set(out) <= set("abcdefghijklmn_-")
