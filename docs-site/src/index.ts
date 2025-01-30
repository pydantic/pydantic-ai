export default {
  async fetch(request, env): Promise<Response> {
    const r = await env.ASSETS.fetch(request)
    if (r.status == 404) {
      const url = new URL(request.url)
      url.pathname = '/404.html'
      const r = await env.ASSETS.fetch(url)
      return new Response(r.body, { status: 404 })
    } else if (r.status == 200) {
      const contentType = r.headers.get('content-type')
      if (contentType && contentType.includes('text/html')) {
        const handler = new InsertVersionNoticeHandler(request, env)
        return new HTMLRewriter().on('div#version-notice', handler).transform(r);
      }
    }
    return r
  },
} satisfies ExportedHandler<Env>

class InsertVersionNoticeHandler {
  readonly request: Request
  readonly env: Env

  constructor(request: Request, env: Env) {
    this.request = request
    this.env = env
  }

  async element(element: Element): Promise<void> {
    try {
      const warning = await this.getVersionNotice()
      element.setInnerContent(warning, {html: true})
    } catch (e) {
      // catch the error and log it, but do not raise it so the site can still be served
      console.error(e)
    }
  }
  async getVersionNotice(): Promise<string> {
    const cacheKey = `version-notice-${this.env.GIT_BRANCH}-${this.env.GIT_COMMIT_SHA}`
    let versionNotice = await this.env.VERSION_NOTICE_CACHE.get(cacheKey, {cacheTtl: 60})
    if (versionNotice === null) {
      versionNotice = await this.fetchVersionNotice()
      await this.env.VERSION_NOTICE_CACHE.put(cacheKey, versionNotice, {expirationTtl: 300})
    }
    return versionNotice
  }

  async fetchVersionNotice(): Promise<string> {
    const headers = {
      'User-Agent': this.request.headers.get('User-Agent') || 'pydantic-ai-docs',
      'Accept': 'application/vnd.github.v3+json',
    }
    const r1 = await fetch('https://api.github.com/repos/pydantic/pydantic-ai/releases/latest', {headers})
    if (!r1.ok) {
      const text = await r1.text()
      throw new Error(`Failed to fetch latest release, response status ${r1.status}:\n${text}`)
    }
    const {html_url, name, tag_name}: ReleaseInfo = await r1.json()
    const r2 = await fetch(
      `https://api.github.com/repos/pydantic/pydantic-ai/compare/${tag_name}...${this.env.GIT_COMMIT_SHA}`,
      {headers}
    )
    if (!r2.ok) {
      const text = await r2.text()
      throw new Error(`Failed to fetch compare, response status ${r2.status}:\n${text}`)
    }
    const {ahead_by}: TagInfo = await r2.json()

    if (ahead_by === 0) {
      return `<div class="admonition success" style="margin: 0">
    <p class="admonition-title">Version</p>
    <p>Showing documentation for the latest release <a href="${html_url}">${name}</a>.</p>
  </div>`
    }

    const branch = this.env.GIT_BRANCH
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
}

interface ReleaseInfo {
  html_url: string
  name: string
  tag_name: string
}
interface TagInfo {
  ahead_by: number
}
