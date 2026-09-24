"""The working-indicator animations, following Code Puppy's `puppy_spinner` catalogue.

Three layers make up the catalogue, later layers winning on a name collision:

1. The builtins below: CLAI2's own `working` braille (the default) plus every Code Puppy builtin.
2. Spinners plugins register with `host.spinner(...)`.
3. The user's `spinners.json` next to CLAI2's settings (`/spinner init` writes a starter file).
   An entry without `frames` that names an existing spinner only changes its speed or description.

The chosen name is the `display.spinner` setting. Painters call `Spinners.active()` on every
tick, so a new choice, a plugin load, or an edit to `spinners.json` shows on the next frame.
"""

import math
import os
import sys
import unicodedata
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from rich.cells import cell_len

from .settings_store import config_dir
from .spinner_frames import EXTRA_SPECS

if sys.platform != 'win32':  # pragma: no branch
    import fcntl

SpinnerSource = Literal['builtin', 'plugin', 'user']
DEFAULT_SPINNER = 'working'
MIN_INTERVAL = 0.02
MAX_INTERVAL = 1.0
MAX_FRAME_LENGTH = 40
_PACK_INTERVAL = 0.2
"""Code Puppy's one speed for every pack builtin; `puppy` and `working` keep their own tuned speeds."""
_BRAILLE = tuple('\u280b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f')


@dataclass(frozen=True, kw_only=True)
class Spinner:
    """One animation: frames of equal cell width, seconds per frame, and where it came from."""

    name: str
    frames: tuple[str, ...]
    interval: float
    description: str = ''
    source: SpinnerSource = 'builtin'

    def frame(self, now: float) -> str:
        """The frame to show at `now` seconds on a monotonic clock."""
        return self.frames[int(now / self.interval) % len(self.frames)]


def clamp_interval(seconds: float) -> float:
    """Keep a speed between `MIN_INTERVAL` and `MAX_INTERVAL`, rounded to centiseconds.

    NaN and infinity raise instead: comparisons with NaN are all false, so it would pass the clamp
    and fail later in `Spinner.frame`, on every repaint.
    """
    if not math.isfinite(seconds):
        raise ValueError(f'A spinner interval must be a finite number of seconds, not {seconds}.')
    return round(min(max(seconds, MIN_INTERVAL), MAX_INTERVAL), 2)


def make_spinner(
    name: str,
    frames: Iterable[str],
    *,
    interval: float = _PACK_INTERVAL,
    description: str = '',
    source: SpinnerSource = 'builtin',
) -> Spinner:
    """Validate and normalize a spinner: frames are capped, padded to one cell width, and the speed clamped.

    Padding by terminal cells rather than `len` keeps the text after an emoji frame from jumping.
    """
    name = name.strip()
    shown = tuple(frame[:MAX_FRAME_LENGTH] for frame in frames)
    if not name or any(char.isspace() for char in name):
        raise ValueError(f'Spinner names need at least one character and no spaces: {name!r}')
    if not shown or not all(shown):
        raise ValueError(f'Spinner {name!r} needs at least one non-empty frame.')
    if any(unicodedata.category(char).startswith('C') for frame in shown for char in frame):
        raise ValueError(f'Spinner {name!r} has a control character in a frame.')
    width = max(cell_len(frame) for frame in shown)
    return Spinner(
        name=name,
        frames=tuple(frame + ' ' * (width - cell_len(frame)) for frame in shown),
        interval=clamp_interval(interval),
        description=description,
        source=source,
    )


def _kennel_bounce(critter: str) -> tuple[str, ...]:
    return tuple(f'({" " * i}{critter}{" " * (4 - i)}) ' for i in (0, 1, 2, 3, 4, 3, 2, 1))


_DOG, _DASH, _PAW, _BONE, _PUPPY = '\U0001f415', '\U0001f4a8', '\U0001f43e', '\U0001f9b4', '\U0001f436'
BUILTIN_SPINNERS: dict[str, Spinner] = {
    spinner.name: spinner
    for spinner in (
        make_spinner('working', _BRAILLE, interval=0.1, description="CLAI2's braille Working indicator (default)"),
        make_spinner('puppy', _kennel_bounce(_PUPPY), interval=0.06, description='the classic kennel bounce'),
        make_spinner('bone', _kennel_bounce(_BONE), description='same kennel, chewier occupant'),
        make_spinner(
            'zoomies',
            (f'{" " * i}{_DOG}{_DASH}{" " * (5 - i)}' for i in (5, 4, 3, 2, 1, 0)),
            description='full-speed dog, dust trailing',
        ),
        make_spinner('paws', (_PAW * count for count in (1, 2, 3, 4)), description='a trail of paw prints'),
        make_spinner('dots', _BRAILLE, description='classic braille dots'),
        *(make_spinner(name, frames, description=blurb) for name, (frames, blurb) in EXTRA_SPECS.items()),
    )
}


class _Entry(BaseModel):
    """One `spinners.json` entry; `frames` may be omitted to retune an existing spinner."""

    model_config = ConfigDict(extra='forbid', strict=True)
    frames: list[str] | None = Field(default=None, min_length=1)
    interval: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    description: str | None = None


_FILE: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])
STARTER_FILE = """{
  "sniffer": {
    "frames": ["( .    ) ", "(  .   ) ", "(   .  ) ", "(    . ) "],
    "interval": 0.1,
    "description": "a very minimalist puppy"
  },
  "zoomies": {"interval": 0.2}
}
"""
"""What `/spinner init` writes: one new spinner, and one frameless entry that only retunes a builtin."""


def user_spinners_path() -> Path:
    """`spinners.json` in the CLAI2 config folder."""
    return config_dir() / 'spinners.json'


def _parse_user_file(text: str, base: dict[str, Spinner]) -> tuple[dict[str, Spinner], tuple[str, ...]]:
    """Apply each valid entry over `base`; a bad entry is reported and skipped, not fatal."""
    try:
        entries = _FILE.validate_json(text)
    except ValidationError:
        return {}, ('spinners.json must be a JSON object of spinner entries.',)
    spinners: dict[str, Spinner] = {}
    problems: list[str] = []
    for key, raw in entries.items():
        name = key.strip()
        try:
            entry = _Entry.model_validate(raw)
            if entry.frames is None:
                if name not in base:
                    raise ValueError('needs "frames" unless it names an existing spinner')
                found = base[name]
                interval = found.interval if entry.interval is None else clamp_interval(entry.interval)
                description = found.description if entry.description is None else entry.description
                spinners[name] = replace(found, interval=interval, description=description, source='user')
            else:
                spinner = make_spinner(
                    name,
                    entry.frames,
                    interval=_PACK_INTERVAL if entry.interval is None else entry.interval,
                    description=entry.description or '',
                    source='user',
                )
                spinners[spinner.name] = spinner
        except (ValidationError, ValueError) as exc:
            reason = exc.errors()[0]['msg'] if isinstance(exc, ValidationError) else str(exc)
            problems.append(f'spinners.json: skipped {key!r}: {reason}')
    return spinners, tuple(problems)


@contextmanager
def _locked(path: Path) -> Generator[None]:
    """Hold an exclusive lock on a sibling lock file; Windows has no `fcntl`, so there it is unlocked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(f'.{path.name}.lock').open('a') as handle:
        if sys.platform != 'win32':  # pragma: no branch
            fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _replace(path: Path, text: str) -> None:
    """Replace the file atomically through a staging file no other writer touches."""
    staging = path.with_name(f'.{path.name}.{uuid4().hex}.tmp')
    try:
        staging.write_text(text, encoding='utf-8')
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


class Spinners:
    """The live catalogue: builtins, then plugin spinners, then the user file, read fresh when it changes."""

    def __init__(
        self,
        *,
        selected: Callable[[], str],
        registered: Callable[[], Iterable[Spinner]] = tuple,
        path: Path | None = None,
    ) -> None:
        """`selected` returns the `display.spinner` setting; `registered` returns plugin spinners."""
        self.selected = selected
        self.registered = registered
        self.path = path or user_spinners_path()
        self._cached: tuple[tuple[Spinner, ...], str] | None = None
        self._loaded: tuple[dict[str, Spinner], tuple[str, ...]] = ({}, ())

    def _user_text(self) -> str:
        """The file's contents; empty when absent."""
        try:
            return self.path.read_text(encoding='utf-8')
        except FileNotFoundError:
            return ''

    def _load(self) -> tuple[dict[str, Spinner], tuple[str, ...]]:
        """Read the file on every call and parse it only when it or the plugin spinners change.

        Comparing contents rather than the modification time also catches a same-size edit within
        one timestamp tick; the file is small, so the read is cheap at the repaint rate.
        """
        registered = tuple(self.registered())
        catalogue = {**BUILTIN_SPINNERS, **{spinner.name: spinner for spinner in registered}}
        try:
            text = self._user_text()
        except (OSError, UnicodeDecodeError) as exc:
            return catalogue, (f'spinners.json could not be read: {exc}',)
        if self._cached != (registered, text):
            user, problems = _parse_user_file(text, catalogue) if text.strip() else ({}, ())
            self._cached, self._loaded = (registered, text), ({**catalogue, **user}, problems)
        return self._loaded

    def catalogue(self) -> dict[str, Spinner]:
        """Every selectable spinner, sorted by name without regard to case."""
        return dict(sorted(self._load()[0].items(), key=lambda item: item[0].lower()))

    def problems(self) -> tuple[str, ...]:
        """Why entries in `spinners.json` were skipped."""
        return self._load()[1]

    def find(self, name: str) -> Spinner | None:
        """Look a name up exactly, then without regard to case, since builtins use camelCase."""
        catalogue = self._load()[0]
        exact = catalogue.get(name)
        if exact is not None:
            return exact
        return next((spinner for key, spinner in catalogue.items() if key.lower() == name.lower()), None)

    def active(self) -> Spinner:
        """The selected spinner; an unknown name, such as an unloaded plugin's, falls back to `working`."""
        return self.find(self.selected()) or BUILTIN_SPINNERS[DEFAULT_SPINNER]

    def save_interval(self, name: str, seconds: float) -> None:
        """Record a speed in `spinners.json`, keeping the entry's other keys; the file is the only record.

        The read-modify-write holds a lock, so two CLAI2 processes saving at once both keep their change.
        """
        with _locked(self.path):
            text = self._user_text()
            entries = _FILE.validate_json(text) if text.strip() else {}
            # Keys are matched as the catalogue matches them, so a padded key is updated rather than shadowed.
            key = next((key for key in entries if key.strip() == name), name)
            current = entries.get(key)
            entry = _FILE.validate_python(current) if isinstance(current, dict) else {}
            entry['interval'] = clamp_interval(seconds)
            entries[key] = entry
            _replace(self.path, _FILE.dump_json(entries, indent=2).decode() + '\n')

    def init(self) -> bool:
        """Write the starter file; `False` when one already exists, which is left alone.

        Exclusive creation, so a file another process writes meanwhile is never replaced.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open('x', encoding='utf-8') as file:
                file.write(STARTER_FILE)
        except FileExistsError:
            return False
        return True
