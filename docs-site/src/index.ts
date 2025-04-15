 import { marked } from 'marked'

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname === '/changelog.html') {
      const changelog = await getChangelog()
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

async function getChangelog(): Promise<string> {
  const headers = {
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'pydantic-ai-docs'
  }
  let url: string = 'https://api.github.com/repos/pydantic/pydantic-ai/releases'
  const releases: Release[] = []
  while (typeof url === 'string') {
    const response = await fetch(url, { headers })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`Failed to fetch changelog: ${response.status} ${response.statusText} ${text}`)
    }
    const newReleases = await response.json() as Release[]
    releases.push(...newReleases)
    const linkHeader = response.headers.get('link')
    if (!linkHeader) break
    const nextUrl = linkHeader.match(/<([^>]+)>; rel="next"/)?.[1]
    if (!nextUrl) break
    url = nextUrl
  }
  console.log(releases.length)
  return marked(releases.map(prepRelease).join('\n\n'))
}

interface Release {
  name: string
  body: string
  html_url: string
}

function prepRelease(release: Release): string {
  // console.log(release.body)
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
