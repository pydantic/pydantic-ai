#!/usr/bin/env bash
# Regenerates the README screenshots in `src/pydantic_clai2/docs/images/`.
#
#   src/pydantic_clai2/scripts/screenshots/capture.sh [TAPE...]
#
# Needs the checkout's `.venv` (`make install`), plus VHS (https://github.com/charmbracelet/vhs),
# pngquant, oxipng, and the JetBrains Mono font. On macOS:
#   brew install vhs pngquant oxipng && brew install --cask font-jetbrains-mono
#
# Every tape runs the checkout's `clai2` in a fixed-size VHS terminal against a throwaway HOME and
# XDG_CONFIG_HOME, with a clean environment: no API keys, sign-ins, saved sessions, or real paths
# reach the images. Turns use the scripted `demo:tour` model from `demo_model.py`, a drop-in plugin,
# so no model is called and every built-in plugin still loads. With TAPE names (such as `hero`),
# only those tapes run; the setup tape always runs first.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
images="$repo/src/pydantic_clai2/docs/images"
[ -x "$repo/.venv/bin/clai2" ] || { echo "no $repo/.venv/bin/clai2: run make install" >&2; exit 1; }
tools=""
for tool in vhs pngquant oxipng git; do
  found=$(command -v "$tool") || { echo "missing $tool; see the top of $0" >&2; exit 1; }
  tools="$tools:$(dirname "$found")"
done

root=$(mktemp -d "${TMPDIR:-/tmp}/clai2-shots.XXXXXX")
root=$(cd "$root" && pwd -P)
# CLAI_SHOTS_KEEP=1 keeps the throwaway directory for debugging a tape.
if [ -n "${CLAI_SHOTS_KEEP:-}" ]; then echo "keeping $root"; else trap 'rm -rf "$root"' EXIT; fi
home="$root/home"
project="$home/code/weather"
mkdir -p "$home/.config/pydantic-clai2/plugins" "$project" "$root/shots"
cp "$here/demo_model.py" "$home/.config/pydantic-clai2/plugins/demo_model.py"

# A clean environment: only what the terminal, VHS, Git, and the checkout's `clai2` need. Your own
# Git configuration, API keys, and keychain stay out.
run() {
  env -i \
    HOME="$home" XDG_CONFIG_HOME="$home/.config" CLAUDE_CONFIG_DIR="$home/.claude" CODEX_HOME="$home/.codex" \
    PATH="$home/bin:$repo/.venv/bin$tools:/usr/bin:/bin" \
    TERM=xterm-256color COLORTERM=truecolor LANG=en_US.UTF-8 \
    PYTHONPATH="$here" PYTHON_KEYRING_BACKEND=demo_keyring.Keyring \
    PROJECT="$project" \
    "$@"
}

# A small project for the model to work on, with a short Git history.
mkdir -p "$project/weather" "$project/tests"
cat > "$project/AGENTS.md" <<'EOF'
# weather

A tiny forecast CLI. Run the tests with `python3 -m unittest`.
EOF
cat > "$project/weather/__init__.py" <<'EOF'
"""Forecasts from bundled sample data."""
EOF
cat > "$project/weather/__main__.py" <<'EOF'
from weather.cli import main

main()
EOF
cat > "$project/weather/forecast.py" <<'EOF'
"""Forecast lookups. The sample data keeps the tests offline."""

from dataclasses import dataclass


@dataclass
class Day:
    date: str
    summary: str
    high: int
    low: int


SAMPLE = {
    'lisbon': [
        Day('Mon', 'Sunny', 24, 16),
        Day('Tue', 'Clouds', 21, 15),
        Day('Wed', 'Showers', 19, 14),
    ],
}


def fetch_forecast(city, days=3):
    """Return up to `days` forecast days for `city`."""
    return SAMPLE.get(city.lower(), SAMPLE['lisbon'])[:days]
EOF
cat > "$project/weather/cli.py" <<'EOF'
"""Command line entry point: `python3 -m weather forecast CITY`."""

import argparse

from weather.forecast import fetch_forecast


def main(argv=None):
    parser = argparse.ArgumentParser(prog='weather')
    commands = parser.add_subparsers(dest='command', required=True)
    forecast = commands.add_parser('forecast', help='Show the forecast for a city')
    forecast.add_argument('city')
    forecast.add_argument('--days', type=int, default=3)
    args = parser.parse_args(argv)

    for day in fetch_forecast(args.city, days=args.days):
        print(f'{day.date}  {day.summary:<12} {day.high}°/{day.low}°')
EOF
cat > "$project/tests/__init__.py" <<'EOF'
EOF
cat > "$project/tests/test_cli.py" <<'EOF'
import io
import unittest
from contextlib import redirect_stdout

from weather.cli import main


def run(*argv):
    out = io.StringIO()
    with redirect_stdout(out):
        main(list(argv))
    return out.getvalue()


class ForecastTests(unittest.TestCase):
    def test_default_days(self):
        self.assertEqual(len(run('forecast', 'lisbon').splitlines()), 3)

    def test_days(self):
        self.assertEqual(len(run('forecast', 'lisbon', '--days', '2').splitlines()), 2)

    def test_unknown_city_falls_back(self):
        self.assertTrue(run('forecast', 'atlantis').startswith('Mon  Sunny'))
EOF
run git -C "$project" init -q -b main
run git -C "$project" config user.name 'Demo User'
run git -C "$project" config user.email 'demo@example.com'
run git -C "$project" add weather/__init__.py weather/forecast.py
run git -C "$project" commit -q -m 'Add sample forecast data'
run git -C "$project" add .
run git -C "$project" commit -q -m 'Add the forecast command'

# A second project, so the session browser has more than one repository.
other="$home/code/shop-api"
mkdir -p "$other"
run git -C "$other" init -q -b main
echo '# shop-api' > "$other/README.md"

# VHS keeps its headless browser under $HOME/.cache/rod; share the real one so it downloads once.
mkdir -p "$HOME/.cache/rod" "$home/.cache"
ln -s "$HOME/.cache/rod" "$home/.cache/rod"
# Tapes name each other and their screenshots relative to the directory VHS runs in.
ln -s "$here/tapes" "$root/tapes"

# Every tape starts CLAI with `clai-demo`: the project as committed, and the settings a tape may change reset.
mkdir -p "$home/bin"
cat > "$home/bin/clai-demo" <<'EOF'
#!/usr/bin/env bash
set -e
cd "$PROJECT"
git checkout -q . && git clean -qfd
clai2 config set model demo:tour > /dev/null
clai2 config reset run.speculative_code_mode > /dev/null
clai2 config reset display.theme > /dev/null
exec clai2 "$@"
EOF
chmod +x "$home/bin/clai-demo"
run clai2 config set model demo:tour > /dev/null

# Saved sessions for the browser, made headless in each project.
(cd "$other" && run clai2 -p 'Add retry with backoff to the orders client' > /dev/null)
(cd "$other" && run clai2 -p 'Add an index to speed up order search' > /dev/null)
(cd "$project" && run clai2 -p 'Rewrite the README install section' > /dev/null)
(cd "$project" && run clai2 -p 'Explain how a forecast is fetched' > /dev/null)
(cd "$project" && run clai2 -p 'Write tests for the parser' > /dev/null)

# Every tape, in order: the browser goes first, so it shows only the sessions made above. Add new tapes here.
tapes=("$@")
if [ ${#tapes[@]} -eq 0 ]; then
  tapes=(resume model model-add accounts keys plugins set theme spinner system-prompt
    hero ask-user shell code-mode rewind usage fork)
fi
failed=""
cd "$root"
run vhs "$here/tapes/setup.tape" || run vhs "$here/tapes/setup.tape"
for name in "${tapes[@]}"; do
  echo "tape: $name"
  # VHS sometimes fails to start its headless browser; one retry gets past that.
  run vhs "$here/tapes/$name.tape" > "$root/shots/$name.log" 2>&1 ||
    run vhs "$here/tapes/$name.tape" > "$root/shots/$name.log" 2>&1 ||
    { echo "tape $name failed: see $root/shots/$name.log"; failed="$failed $name"; }
done

mkdir -p "$images"
for shot in "$root"/shots/*.png; do
  [ -e "$shot" ] || continue
  pngquant --force --skip-if-larger --quality 80-95 --output "$shot" "$shot" || true
  oxipng --quiet --opt 4 --strip safe "$shot"
  cp "$shot" "$images/"
  echo "wrote ${shot##*/}"
done

if [ -n "$failed" ]; then
  echo "failed tapes:$failed" >&2
  exit 1
fi
