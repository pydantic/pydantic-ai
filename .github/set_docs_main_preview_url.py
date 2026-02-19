import json
import os
import re
import typing

import httpx

DEPLOY_OUTPUT = os.environ['DEPLOY_OUTPUT']
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
REPOSITORY = os.environ['REPOSITORY']
REF = os.environ['REF']
ENVIRONMENT = 'deploy-docs-preview'

m = re.search(r'https://(\S+)\.workers\.dev', DEPLOY_OUTPUT)
assert m, f'Could not find worker URL in {DEPLOY_OUTPUT!r}'

worker_name = m.group(1)
m = re.search(r'Current Version ID: ([^-]+)', DEPLOY_OUTPUT)
assert m, f'Could not find version ID in {DEPLOY_OUTPUT!r}'

version_id = m.group(1)
preview_url = f'https://{version_id}-{worker_name}.workers.dev'
print('CloudFlare worker preview URL:', preview_url, flush=True)

gh_headers = {
    'Accept': 'application/vnd.github+json',
    'Authorization': f'Bearer {GITHUB_TOKEN}',
    'X-GitHub-Api-Version': '2022-11-28',
}

deployment_url = f'https://api.github.com/repos/{REPOSITORY}/deployments'
deployment_data: dict[str, typing.Any] = {
    'ref': REF,
    'task': 'docs preview',
    'environment': ENVIRONMENT,
    'auto_merge': False,
    'required_contexts': [],
    'payload': json.dumps({
        'preview_url': preview_url,
        'worker_name': worker_name,
        'version_id': version_id,
    })
}
r = httpx.post(deployment_url, headers=gh_headers, json=deployment_data)
print(f'POST {deployment_url} {r.status_code} {r.text}', flush=True)
r.raise_for_status()
deployment_id = r.json()['id']

status_url = f'https://api.github.com/repos/{REPOSITORY}/deployments/{deployment_id}/statuses'
status_data = {
    'environment': ENVIRONMENT,
    'environment_url': preview_url,
    'state': 'success',
}
r = httpx.post(status_url, headers=gh_headers, json=status_data)
print(f'POST {status_url} {r.status_code} {r.text}', flush=True)
r.raise_for_status()
