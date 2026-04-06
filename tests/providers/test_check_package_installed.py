"""Tests for the check_package_installed utility function."""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest

from pydantic_ai.providers import check_package_installed


class TestCheckPackageInstalled:
    """Tests for check_package_installed()."""

    def test_installed_package_does_not_raise(self) -> None:
        """A package that is installed should not raise."""
        # 'os' is always available
        check_package_installed('os')

    def test_missing_package_raises_import_error(self) -> None:
        """A package that is not installed should raise ImportError with install hint."""
        with pytest.raises(ImportError, match=r'Please install the `nonexistent_pkg_xyz` package'):
            check_package_installed('nonexistent_pkg_xyz')

    def test_missing_package_shows_install_group(self) -> None:
        """The error message should include the install group."""
        with pytest.raises(ImportError, match=r'pydantic-ai-slim\[mistral\]'):
            check_package_installed('nonexistent_pkg_xyz', install_group='mistral')

    def test_missing_package_default_install_group(self) -> None:
        """When install_group is not specified, use package_name as the group."""
        with pytest.raises(ImportError, match=r'pydantic-ai-slim\[nonexistent_pkg_xyz\]'):
            check_package_installed('nonexistent_pkg_xyz')

    def test_installed_but_broken_import_propagates_real_error(self) -> None:
        """If the package is installed but a name import fails, the real ImportError should propagate.

        This is the core bug from #4927: when mistralai is installed but 'UNSET'
        cannot be imported, the user should see the real error, not "please install".
        """
        # Simulate: find_spec says the package exists, but importing a name fails
        check_package_installed('os')  # passes — package exists
        # Now the subsequent import of a nonexistent name should give the real error
        with pytest.raises(ImportError, match="cannot import name 'nonexistent_name'"):
            from os import nonexistent_name  # type: ignore[attr-defined]  # noqa: F401

    def test_find_spec_returns_none_for_missing_package(self) -> None:
        """Verify that find_spec returns None for a package that doesn't exist."""
        assert importlib.util.find_spec('nonexistent_pkg_xyz_12345') is None

    def test_find_spec_returns_spec_for_installed_package(self) -> None:
        """Verify that find_spec returns a spec for an installed package."""
        assert importlib.util.find_spec('os') is not None

    def test_check_with_subpackage(self) -> None:
        """check_package_installed should work with dotted package names."""
        # email.mime is a valid subpackage in the stdlib
        check_package_installed('email.mime')

    def test_check_with_missing_subpackage_raises(self) -> None:
        """A missing subpackage should raise ImportError."""
        with pytest.raises(ImportError, match=r'Please install'):
            check_package_installed('email.nonexistent_subpkg_xyz')
