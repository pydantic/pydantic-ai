"""Very simply CLI to aid in copying examples code to a new directory.

To run examples in place, run:

    uv run -m pydantic_ai_examples.<example_module_name>

For examples:

    uv run -m pydantic_ai_examples.pydantic_model

To copy all examples to a new directory, run:

    uv run -m pydantic_ai_examples --copy-to <destination_path>

See https://ai.pydantic.dev/examples/ for more information.
"""

import argparse
import sys
from pathlib import Path


def cli():
    this_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        prog='pydantic_ai_examples',
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-v', '--version', action='store_true', help='show the version and exit'
    )
    parser.add_argument(
        '--copy-to', dest='DEST', help='Copy all examples to a new directory'
    )

    args = parser.parse_args()
    if args.version:
        from pydantic_ai import __version__

        print(f'pydantic_ai v{__version__}')
    elif args.DEST:
        copy_to(this_dir, Path(args.DEST))
    else:
        parser.print_help()


def copy_to(this_dir: Path, dst: Path):
    if dst.exists():
        print(f'Error: destination path "{dst}" already exists', file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True)

    count = 0
    for file in this_dir.glob('*.*'):
        with open(file, 'rb') as src_file:
            with open(dst / file.name, 'wb') as dst_file:
                dst_file.write(src_file.read())
        count += 1

    print(f'Copied {count} example files to "{dst}"')


if __name__ == '__main__':
    cli()
