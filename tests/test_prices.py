from collections.abc import Callable
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

import pydantic_ai.prices as prices_mod
from pydantic_ai.prices import update_in_background


@pytest.fixture(autouse=True)
def _reset_updater():
    """Reset the module-level singleton before each test."""
    prices_mod._updater = None  # pyright: ignore[reportPrivateUsage]
    yield
    prices_mod._updater = None  # pyright: ignore[reportPrivateUsage]


def test_update_in_background_calls_start(mocker: MockerFixture):
    """Verify update_in_background() creates an UpdatePrices instance and starts it."""
    mock_cls = mocker.patch('pydantic_ai.prices.UpdatePrices')
    update_in_background()
    mock_cls.assert_called_once()
    mock_cls.return_value.start.assert_called_once()


def test_update_in_background_is_idempotent(mocker: MockerFixture):
    """Verify calling update_in_background() twice only creates one updater."""
    mock_cls = mocker.patch('pydantic_ai.prices.UpdatePrices')
    update_in_background()
    update_in_background()
    mock_cls.assert_called_once()


def _start_raises(m: MagicMock) -> None:
    m.return_value.start.side_effect = RuntimeError('already started')


def _constructor_raises(m: MagicMock) -> None:
    m.side_effect = Exception('import failed')


@pytest.mark.parametrize(
    'setup_mock',
    [
        pytest.param(_start_raises, id='start-raises'),
        pytest.param(_constructor_raises, id='constructor-raises'),
    ],
)
def test_update_in_background_suppresses_errors(mocker: MockerFixture, setup_mock: Callable[[MagicMock], None]):
    """Verify update_in_background() silently catches exceptions and allows retry."""
    mock_cls = mocker.patch('pydantic_ai.prices.UpdatePrices')
    setup_mock(mock_cls)
    update_in_background()

    # After failure, a subsequent call should retry (not be stuck thinking an updater exists).
    mock_cls.reset_mock()
    mock_cls.side_effect = None
    mock_cls.return_value.start.side_effect = None
    update_in_background()
    mock_cls.assert_called_once()
    mock_cls.return_value.start.assert_called_once()
