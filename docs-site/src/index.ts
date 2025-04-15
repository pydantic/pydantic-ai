 import { marked } from 'marked'

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname === '/changelog.html') {
      const changelog = await getChangelog(env.KV, env.GIT_COMMIT_SHA)
      return new Response(changelog, { headers: {'content-type': 'text/html'} })
    }
    const r = await env.ASSETS.fetch(request)
    if (r.status == 404) {
      const redirectPath = redirect(url.pathname)
      if (redirectPath) {
        url.pathname = redirectPath
        return Response.redirect(url.toString(), 301)
      }
      url.pathname = '/404.html'
      const r = await env.ASSETS.fetch(url)
      return new Response(r.body, { status: 404, headers: {'content-type': 'text/html'} })
    }
    return r
  },
} satisfies ExportedHandler<Env>

const redirect_lookup: Record<string, string> = {
  '/common_tools': '/common-tools/',
  '/testing-evals': '/testing/',
  '/result': '/output/',
}

function redirect(pathname: string): string | null {
  return redirect_lookup[pathname.replace(/\/+$/, '')] ?? null
}

async function getChangelog(kv: KVNamespace, commitSha: string): Promise<string> {
  const cache_key = `changelog:${commitSha}`
  const cached = await kv.get(cache_key, {cacheTtl: 60})
  if (cached) {
    return cached
  }
  const headers = {
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'pydantic-ai-docs'
  }
  let url: string | undefined = 'https://api.github.com/repos/pydantic/pydantic-ai/releases'
  const releases: Release[] = []
  while (typeof url == 'string') {
    const response = await fetch(url, { headers })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`Failed to fetch changelog: ${response.status} ${response.statusText} ${text}`)
    }
    const newReleases = await response.json() as Release[]
    releases.push(...newReleases)
    const linkHeader = response.headers.get('link')
    if (!linkHeader) {
      break
    }
    url = linkHeader.match(/<([^>]+)>; rel="next"/)?.[1]
  }
  marked.use({pedantic: false, gfm: true})
  const html = marked(releases.map(prepRelease).join('\n\n')) as string
  await kv.put(cache_key, html, {expirationTtl: 300})
  return html
}

interface Release {
  name: string
  body: string
  html_url: string
}

function prepRelease(release: Release): string {
  const body = release.body
    .replace(/(#+)/g, (m) => `##${m}`)
    .replace(/https:\/\/github.com\/pydantic\/pydantic-ai\/pull\/(\d+)/g, (url, id) => `[#${id}](${url})`)
    .replace(/\*\*Full Changelog\*\*: (\S+)/, (_, url) => `[Compare diff](${url})`)
  return `
### ${release.name}

${body}

[View on GitHub](${release.html_url})
`
}
