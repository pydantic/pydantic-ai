"""
Used to check that examples are at least valid syntax and can be imported without errors.

Called in CI.
"""

import os
from pathlib import Path

os.environ.update(OPENAI_API_KEY='fake-key', GEMINI_API_KEY='fake-key', GROQ_API_KEY='fake-key')

examples_dir = Path(__file__).parent.parent / 'examples' / 'pydantic_ai_examples'
assert examples_dir.is_dir(), f'No examples directory found at {examples_dir}'
count = 0
for example in examples_dir.glob('*.py'):
    print(f'Importing {example.stem}...')
    __import__(f'pydantic_ai_examples.{example.stem}')
    count += 1

assert count > 0, 'No examples found'
