export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    if (url.pathname === '/env.json') {
      const headers = new Headers({'Content-Type': 'application/json'})
      return new Response(JSON.stringify(env), { headers })
    }
    if (url.pathname === '/ahead-warning.html') {
      try {
        const html = await aheadOfRelease()
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

async function aheadOfRelease() {
  const r1 = await fetch('https://api.github.com/repos/pydantic/pydantic-ai/releases/latest')
  if (!r1.ok) {
    throw new Error(`Failed to fetch latest release, response status ${r.status}`)
  }
  const {tag_name} = await r1.json()
  const r2 = await fetch(`https://api.github.com/repos/pydantic/pydantic-ai/compare/${tag_name}...main`)
  if (!r2.ok) {
    throw new Error(`Failed to fetch compare, response status ${r.status}`)
  }
  const {ahead_by} = await r2.json()

  return `<div class="admonition note">
  <p class="admonition-title">Warning</p>
  <p>These docs are ahead of the latest release by <b>${ahead_by}</b> commits.</p>
  <p>
    You may see documentation for features not yet supported in the
    <a href="https://github.com/pydantic/pydantic-ai/releases/latest">latest release</a>.
  </p>
</div>`
}
