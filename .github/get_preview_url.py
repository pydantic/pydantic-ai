import os
import re

deploy_output = os.environ['DEPLOY_OUTPUT']

m = re.search(r'https://(\S+)\.workers\.dev', deploy_output)
assert m, f'Could not find worker URL in {deploy_output!r}'
worker_name = m.group(1)

m = re.search(r'Current Version ID: ([^-]+)', deploy_output)
assert m, f'Could not find version ID in {deploy_output!r}'
version_id = m.group(1)

preview_url = f'https://{version_id}-{worker_name}.workers.dev'

print(f'preview_url={preview_url}')
