export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname === '/version-warning.html') {
      try {
        const html = await versionWarning(env)
        return new Response(html, { headers: {'Content-Type': 'text/html'} })
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
// {"ASSETS":{},"CF_PAGES":"1","CF_PAGES_BRANCH":"ahead-warning","CF_PAGES_COMMIT_SHA":"...","CF_PAGES_URL":"https://..."}
async function versionWarning(env) {
  const headers = new Headers({
    'User-Agent': 'pydantic-ai-docs',
    'Accept': 'application/vnd.github.v3+json',
  })
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
    return `<div class="admonition note">
  <p class="admonition-title">Version</p>
  <p>Showing documentation for the latest release <a href="${html_url}">${name}</a>.</p>
</div>`
  }

  const commits_plural = ahead_by === 1 ? 'commit' : 'commits'
  let msg
  if (env.CF_PAGES_BRANCH === 'main') {
    msg = `These docs are ahead of the latest release by <b>${ahead_by}</b> ${commits_plural}.`
  } else {
    msg = `These preview for <b>${env.CF_PAGES_BRANCH}</b> docs are ahead of the latest release by <b>${ahead_by}</b> ${commits_plural}.`
  }

  return `<div class="admonition note">
  <p class="admonition-title">Notice</p>
  <p>${msg}</p>
  <p>
    You may see documentation for features not yet supported in the latest release <a href="${html_url}">${name}</a>.
  </p>
</div>`
}
