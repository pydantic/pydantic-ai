"""`LocalComputer` against stand-ins for the display (`mss`) and the input devices (`pynput`).

Real input would move the pointer of whoever runs the suite, and CI has no display, so the
`pynput.mouse` and `pynput.keyboard` modules are replaced in `sys.modules` and `mss.MSS` is
patched. Pillow is real, so the PNG the model would see is decoded and measured.
"""

from __future__ import annotations

import enum
import platform
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import mss
import pytest
from PIL import Image

from pydantic_ai import Agent
from pydantic_ai.capabilities.abstract import leaf_capabilities
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness import ComputerUse
from pydantic_ai_harness.computer_use import ComputerError, LocalComputer

pytestmark = pytest.mark.anyio


class Key(enum.Enum):
    """The `pynput.keyboard.Key` members these tests need; `insert` is absent, as on some platforms."""

    ctrl = 'ctrl'
    shift = 'shift'
    cmd = 'cmd'
    enter = 'enter'
    page_down = 'page_down'


@dataclass(frozen=True)
class KeyCode:
    char: str

    @classmethod
    def from_char(cls, char: str) -> KeyCode:
        return cls(char)


class Button(enum.Enum):
    left = 'left'
    right = 'right'
    middle = 'middle'


@dataclass
class Devices:
    """Everything the fake pointer and keyboard were asked to do, in order."""

    log: list[tuple[Any, ...]] = field(default_factory=list[tuple[Any, ...]])


class InvalidCharacterException(Exception):
    pass


def install_devices(monkeypatch: pytest.MonkeyPatch) -> Devices:
    devices = Devices()

    class MouseController:
        @property
        def position(self) -> tuple[int, int]:  # pragma: no cover
            raise NotImplementedError

        @position.setter
        def position(self, value: tuple[int, int]) -> None:
            devices.log.append(('position', value))

        def click(self, button: Button, count: int) -> None:
            devices.log.append(('click', button.name, count))

        def press(self, button: Button) -> None:
            devices.log.append(('mouse_down', button.name))

        def release(self, button: Button) -> None:
            devices.log.append(('mouse_up', button.name))

        def scroll(self, dx: int, dy: int) -> None:
            devices.log.append(('scroll', dx, dy))

    class KeyboardController:
        InvalidCharacterException = InvalidCharacterException

        def press(self, key: Key | KeyCode) -> None:
            devices.log.append(('key_down', key))

        def release(self, key: Key | KeyCode) -> None:
            devices.log.append(('key_up', key))

        def type(self, text: str) -> None:
            if '\x00' in text:
                raise InvalidCharacterException(text.index('\x00'), '\x00')
            devices.log.append(('type', text))

    mouse = types.ModuleType('pynput.mouse')
    mouse.Controller = MouseController  # type: ignore[attr-defined]
    mouse.Button = Button  # type: ignore[attr-defined]
    keyboard = types.ModuleType('pynput.keyboard')
    keyboard.Controller = KeyboardController  # type: ignore[attr-defined]
    keyboard.Key = Key  # type: ignore[attr-defined]
    keyboard.KeyCode = KeyCode  # type: ignore[attr-defined]
    package = types.ModuleType('pynput')
    package.mouse = mouse  # type: ignore[attr-defined]
    package.keyboard = keyboard  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'pynput', package)
    monkeypatch.setitem(sys.modules, 'pynput.mouse', mouse)
    monkeypatch.setitem(sys.modules, 'pynput.keyboard', keyboard)
    return devices


@dataclass
class Shot:
    size: tuple[int, int]
    rgb: bytes


def install_display(monkeypatch: pytest.MonkeyPatch, *, width: int, height: int, density: int = 1) -> None:
    """One 'monitor' of `width`x`height` OS units whose captures have `density` pixels per unit."""

    class FakeMSS:
        monitors = [
            {'left': 0, 'top': 0, 'width': width + 100, 'height': height},
            {'left': 100, 'top': 0, 'width': width, 'height': height},
        ]

        def grab(self, monitor: dict[str, int]) -> Shot:
            assert monitor == {'left': 100, 'top': 0, 'width': width, 'height': height}
            size = (width * density, height * density)
            return Shot(size=size, rgb=b'\x80' * (size[0] * size[1] * 3))

        def close(self) -> None:
            pass

    monkeypatch.setattr(mss, 'MSS', FakeMSS)


@pytest.fixture(autouse=True)
def macos_permissions(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, bool]]:
    """Answer the macOS permission probes, so a Mac running the suite never consults its real settings."""
    granted = {'screen': True, 'accessibility': True, 'requested': False}
    quartz = types.ModuleType('Quartz')
    quartz.CGPreflightScreenCaptureAccess = lambda: granted['screen']  # type: ignore[attr-defined]
    quartz.CGRequestScreenCaptureAccess = lambda: granted.update(requested=True)  # type: ignore[attr-defined]
    services = types.ModuleType('ApplicationServices')
    services.AXIsProcessTrusted = lambda: granted['accessibility']  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, 'Quartz', quartz)
    monkeypatch.setitem(sys.modules, 'ApplicationServices', services)
    yield granted


@pytest.fixture
def devices(monkeypatch: pytest.MonkeyPatch) -> Devices:
    install_display(monkeypatch, width=200, height=100)
    return install_devices(monkeypatch)


class TestLocalComputer:
    @pytest.fixture(autouse=True)
    def off_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'linux')

    async def test_screenshots_are_scaled_to_fit_and_encoded_as_png(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_display(monkeypatch, width=200, height=100, density=2)

        png = await LocalComputer(max_width=100, max_height=100).screenshot()

        assert Image.open(BytesIO(png)).size == (100, 50)
        assert png.startswith(b'\x89PNG')

    async def test_a_screen_within_bounds_is_not_resized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_display(monkeypatch, width=200, height=100)

        png = await LocalComputer().screenshot()

        assert Image.open(BytesIO(png)).size == (200, 100)

    async def test_clicks_map_screenshot_coordinates_to_the_monitor(self, devices: Devices) -> None:
        computer = LocalComputer(max_width=100, max_height=100)

        await computer.click(50, 25, button='right', count=2, modifiers=['cmd+shift'])

        assert devices.log == [
            ('position', (200, 50)),
            ('key_down', Key.cmd),
            ('key_down', Key.shift),
            ('click', 'right', 2),
            ('key_up', Key.shift),
            ('key_up', Key.cmd),
        ]

    async def test_coordinates_off_the_screenshot_are_refused(self, devices: Devices) -> None:
        with pytest.raises(ComputerError, match=r'\(100, 0\) is outside the 100x50 screenshot'):
            await LocalComputer(max_width=100, max_height=100).move(100, 0)
        assert devices.log == []

    async def test_move_scroll_and_drag(self, devices: Devices) -> None:
        computer = LocalComputer()

        await computer.move(10, 20)
        await computer.scroll(1, 1, direction='up', amount=2)
        await computer.scroll(1, 1, direction='down', amount=2)
        await computer.scroll(1, 1, direction='left', amount=1)
        await computer.scroll(1, 1, direction='right', amount=1)
        await computer.drag([(0, 0), (10, 0)])

        assert devices.log[:9] == [
            ('position', (110, 20)),
            ('position', (101, 1)),
            ('scroll', 0, 2),
            ('position', (101, 1)),
            ('scroll', 0, -2),
            ('position', (101, 1)),
            ('scroll', -1, 0),
            ('position', (101, 1)),
            ('scroll', 1, 0),
        ]
        drag = devices.log[9:]
        assert drag[:2] == [('position', (100, 0)), ('mouse_down', 'left')]
        assert drag[2:-1] == [('position', (100 + step, 0)) for step in range(1, 11)]
        assert drag[-1] == ('mouse_up', 'left')

    async def test_typing(self, devices: Devices) -> None:
        computer = LocalComputer()

        await computer.type_text('Hello\n')
        with pytest.raises(ComputerError, match=r"Typed the first 1 characters, then could not type '\\x00'"):
            await computer.type_text('a\x00b')

        assert devices.log == [('type', 'Hello\n')]

    @pytest.mark.parametrize(
        ('keys', 'pressed'),
        [
            (['Return'], [Key.enter]),
            (['Page_Down'], [Key.page_down]),
            (['A'], [KeyCode('A')]),
            (['+'], [KeyCode('+')]),
            (['ctrl+C'], [Key.ctrl, KeyCode('c')]),
            (['ctrl++'], [Key.ctrl, KeyCode('+')]),
            (['control', 'v'], [Key.ctrl, KeyCode('v')]),
        ],
    )
    async def test_key_chords(self, devices: Devices, keys: list[str], pressed: list[Key | KeyCode]) -> None:
        await LocalComputer().press_keys(keys)

        assert devices.log == [
            *(('key_down', key) for key in pressed),
            *(('key_up', key) for key in reversed(pressed)),
        ]

    @pytest.mark.parametrize('key', ['hyper', 'insert'])
    async def test_unknown_keys_are_reported(self, devices: Devices, key: str) -> None:
        with pytest.raises(ComputerError, match=f'Unknown key {key!r}'):
            await LocalComputer().press_keys(['ctrl', key])
        assert devices.log == []

    async def test_a_missing_monitor_is_a_setup_error(self, devices: Devices) -> None:
        with pytest.raises(UserError, match='Monitor 2 does not exist; this machine has 1'):
            await LocalComputer(monitor=2).screenshot()

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [({'monitor': 0}, 'monitor must be 1 or more'), ({'max_width': 0}, 'must be positive')],
    )
    def test_rejects_invalid_settings(self, kwargs: dict[str, int], message: str) -> None:
        with pytest.raises(UserError, match=message):
            LocalComputer(**kwargs)


class TestLocalComputerOnMacOS:
    @pytest.fixture(autouse=True)
    def on_macos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, 'platform', 'darwin')

    async def test_missing_permissions_explain_how_to_grant_them(
        self, devices: Devices, macos_permissions: dict[str, bool]
    ) -> None:
        macos_permissions.update(screen=False, accessibility=False)

        with pytest.raises(UserError, match='needs the macOS Screen Recording and Accessibility permission'):
            await LocalComputer().screenshot()

        assert macos_permissions['requested']
        assert devices.log == []

    async def test_granted_permissions_are_checked_once(
        self, devices: Devices, macos_permissions: dict[str, bool]
    ) -> None:
        computer = LocalComputer()
        await computer.move(1, 1)
        macos_permissions.update(accessibility=False)

        await computer.type_text('still allowed')

        assert devices.log[-1] == ('type', 'still allowed')


class TestComputerUseDefaults:
    @pytest.mark.parametrize('computer', [None, LocalComputer(monitor=2)])
    def test_a_local_computer_is_described_by_its_operating_system(self, computer: LocalComputer | None) -> None:
        instructions = ComputerUse(computer=computer).get_instructions()

        system = {'Darwin': 'macOS'}.get(platform.system(), platform.system())
        assert f'You can see and control this {system} computer with the `computer` tool' in instructions

    def test_a_spec_drives_this_machine(self) -> None:
        agent = Agent.from_spec(
            {'model': 'test', 'capabilities': [{'ComputerUse': {'require_approval': True, 'keep_screenshots': 5}}]},
            custom_capability_types=[ComputerUse],
        )

        [capability] = [c for c in leaf_capabilities(agent.root_capability) if isinstance(c, ComputerUse)]
        assert capability.require_approval
        assert capability.keep_screenshots == 5
        assert isinstance(capability.computer, type(None))
