"""Very simply CLI to aid in running the examples, and for copying examples code to a new directory."""
import argparse
import os
import re
import sys
from pathlib import Path


def cli():
    this_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(prog='pydantic_ai_examples', description=get_description(this_dir), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--version', action='store_true', help='show the version and exit')
    parser.add_argument('--copy-to', dest='DEST', help='Copy all examples to a new directory')

    args = parser.parse_args()
    if args.version:
        from pydantic_ai import __version__

        print(f'pydantic_ai v{__version__}')
    elif args.DEST:
        copy_to(this_dir, Path(args.DEST))
    else:
        parser.print_help()


def get_description(this_dir: Path) -> str:
    description = f"""\
{__doc__}

The following examples are available:
(you might need to prefix the command you run with `uv run` or similar depending on your environment)

"""
    for file in this_dir.glob('*.py'):
        if file.name == '__main__.py':
            continue
        file_descr = re.match(r'"""(.+)', file.read_text())
        if file_descr:
            description += f"""
## {file.name}

{file_descr.group(1).strip()}

    python -m {this_dir.name}.{file.stem}
"""

    return description


def copy_to(this_dir: Path, dst: Path):
    if dst.exists():
        print(f'Error: destination path "{dst}" already exists', file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True)

    count = 0
    for file in this_dir.glob('*.*'):
        with open(file, 'rb') as src:
            with open(dst / file.name, 'wb') as dst:
                dst.write(src.read())
        count += 1

    print(f'Copied {count} example files to "{dst}"')


if __name__ == '__main__':
    cli()
