import os
import re

import httpx

DEPLOY_OUTPUT = os.environ['DEPLOY_OUTPUT']
GITHUB_TOKEN = os.environ['GITHUB_TOKEN']
REPOSITORY = os.environ['REPOSITORY']
PULL_REQUEST_NUMBER = os.environ['PULL_REQUEST_NUMBER']
REF = os.environ['REF']

m = re.search(r'https://(\S+)\.workers\.dev', DEPLOY_OUTPUT)
assert m, f'Could not find worker URL in {DEPLOY_OUTPUT!r}'

worker_name = m.group(1)
m = re.search(r'Current Version ID: ([^-]+)', DEPLOY_OUTPUT)
assert m, f'Could not find version ID in {DEPLOY_OUTPUT!r}'

version_id = m.group(1)
preview_url = f'https://{version_id}-{worker_name}.workers.dev'
print('Docs preview URL:', preview_url)

gh_headers = {
    'Accept': 'application/vnd.github+json',
    'Authorization': f'Bearer {GITHUB_TOKEN}',
    'X-GitHub-Api-Version': '2022-11-28',
}

# now create or update a comment on the PR with the preview URL
if not PULL_REQUEST_NUMBER:
    print('Pull request number not set')
    exit(1)

comments_url = f'https://api.github.com/repos/{REPOSITORY}/issues/{PULL_REQUEST_NUMBER}/comments'
r = httpx.get(comments_url, headers=gh_headers)
print(f'{r.request.method} {r.request.url} {r.status_code}')
if r.status_code != 200:
    print(f'Failed to get comments, status {r.status_code}, response:\n{r.text}')
    exit(1)

comment_update_url = None

for comment in r.json():
    if comment['user']['login'] == 'github-actions[bot]' and comment['body'].startswith('## Docs Preview'):
        comment_update_url = comment['url']
        break

body = f"""\
## Docs Preview

<table>
<tr>
<td><strong>commit:</strong></td>
<td><code>{REF:.7}</code></td>
</tr>
<tr>
<td><strong>Preview URL:</strong></td>
<td><a href="{preview_url}">{preview_url}</a></td>
</tr>
</table>
"""
comment_data = {'body': body}

if comment_update_url:
    print('Updating existing comment...')
    r = httpx.patch(comment_update_url, headers=gh_headers, json=comment_data)
else:
    print('Creating new comment...')
    r = httpx.post(comments_url, headers=gh_headers, json=comment_data)

print(f'{r.request.method} {r.request.url} {r.status_code}')
r.raise_for_status()
