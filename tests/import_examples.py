"""
Used to check that examples are at least valid syntax and can be imported without errors.

Called in CI.
"""

import os
from pathlib import Path

os.environ.update(OPENAI_API_KEY='fake-key', GEMINI_API_KEY='fake-key')

examples_dir = Path(__file__).parent.parent / 'examples'
for example in examples_dir.glob('*.py'):
    __import__(f'examples.{example.stem}')
