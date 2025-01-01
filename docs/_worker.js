// cloudflare worker to building warning if the docs are ahead of the latest release
// see https://developers.cloudflare.com/pages/functions/advanced-mode/

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname === '/version-warning.html') {
      try {
        const html = await versionWarning(request, env)
        const headers = {
          'Content-Type': 'text/plain',
          'Cache-Control': 'max-age=2592000', // 30 days
        }
        return new Response(html, { headers })
      } catch (e) {
        console.error(e)
        return new Response(
          `Error getting ahead HTML: ${e}`,
          { status: 500, headers: {'Content-Type': 'text/plain'} }
        )
      }
    } else {
      return env.ASSETS.fetch(request)
    }
  },
}

// env looks like
// {"CF_PAGES":"1","CF_PAGES_BRANCH":"ahead-warning","CF_PAGES_COMMIT_SHA":"...","CF_PAGES_URL":"https://..."}
async function versionWarning(request, env) {
  const headers = {
    'User-Agent': request.headers.get('User-Agent') || 'pydantic-ai-docs',
    'Accept': 'application/vnd.github.v3+json',
  }
  const r1 = await fetch('https://api.github.com/repos/pydantic/pydantic-ai/releases/latest', {headers})
  if (!r1.ok) {
    const text = await r1.text()
    throw new Error(`Failed to fetch latest release, response status ${r1.status}:\n${text}`)
  }
  const {html_url, name, tag_name} = await r1.json()
  const r2 = await fetch(
    `https://api.github.com/repos/pydantic/pydantic-ai/compare/${tag_name}...${env.CF_PAGES_COMMIT_SHA}`,
    {headers}
  )
  if (!r2.ok) {
    const text = await r2.text()
    throw new Error(`Failed to fetch compare, response status ${r2.status}:\n${text}`)
  }
  const {ahead_by} = await r2.json()

  if (ahead_by === 0) {
    return `<div class="admonition success" style="margin: 0">
  <p class="admonition-title">Version</p>
  <p>Showing documentation for the latest release <a href="${html_url}">${name}</a>.</p>
</div>`
  }

  const branch = env.CF_PAGES_BRANCH
  const diff_html_url = `https://github.com/pydantic/pydantic-ai/compare/${tag_name}...${branch}`
  return `<div class="admonition info" style="margin: 0">
  <p class="admonition-title">Version Notice</p>
  <p>
    ${branch === 'main' ? '' : `(<b>${branch}</b> preview)`}
    This documentation is ahead of the last release by
    <a href="${diff_html_url}">${ahead_by} commit${ahead_by === 1 ? '' : 's'}</a>.
    You may see documentation for features not yet supported in the latest release <a href="${html_url}">${name}</a>.
  </p>
</div>`
}
