import re
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print('Usage: python uprev.py <new_version>')
    sys.exit(1)

new_version = sys.argv[1]
this_dir = Path(__file__).parent
root_dir = (this_dir / '..').resolve()

path_regexes = [
    (this_dir / 'deno.jsonc', r'^\s+"version": "(.+?)"'),
    (this_dir / 'src/main.ts', "^const VERSION = '(.+?)'"),
    (root_dir / 'pydantic_ai_slim/pydantic_ai/mcp_run_python.py', "^MCP_RUN_PYTHON_VERSION = '(.+?)'"),
]


if __name__ == '__main__':
    for path, regex in path_regexes:
        path_pretty = path.relative_to(root_dir)

        def replace_version(m: re.Match[str]) -> str:
            version = m.group(1)
            print(f'Updated version from {version} to {new_version} in {path_pretty}')
            return m.group(0).replace(version, new_version)

        content = path.read_text()
        content, count = re.subn(regex, replace_version, content, count=1, flags=re.M)
        if count != 1:
            raise ValueError(f'Failed to update version in {path}')
        path.write_text(content)
