"""
Used to check that examples are at least valid syntax and can be imported without errors.

Called in CI.
"""

from pathlib import Path

examples_dir = Path(__file__).parent.parent / 'examples'
for example in examples_dir.glob('*.py'):
    __import__(f'examples.{example.stem}')
