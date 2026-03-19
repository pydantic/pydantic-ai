import { instrument } from '@pydantic/logfire-cf-workers'
import { marked } from 'marked'

const handler = {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url)
    if (url.pathname === '/changelog.html') {
      const changelog = await getChangelog(env.KV, env.GIT_COMMIT_SHA)
      return new Response(changelog, { headers: {'content-type': 'text/html'} })
    }
    const maybeTextResponse = await maybeGetTextResponse(request, env)
    if (maybeTextResponse) {
      return maybeTextResponse
    }
    const r = await env.ASSETS.fetch(request)
    if (r.status === 404) {
      const redirectPath = redirect(url.pathname)
      if (redirectPath) {
        if (redirectPath.startsWith('http')) {
          return Response.redirect(redirectPath, 301)
        } else {
          url.pathname = redirectPath
          return Response.redirect(url.toString(), 301)
        }
      }
      url.pathname = '/404.html'
      const r = await env.ASSETS.fetch(url)
      return new Response(r.body, { status: 404, headers: {'content-type': 'text/html'} })
    }
    return r
  },
} satisfies ExportedHandler<Env>

export default instrument(handler, {
	service: {
		name: 'pai-docs',
	},
	baseUrl: 'https://api.logfire.dev',
})

const redirect_lookup: Record<string, string> = {
  '/common_tools': '/common-tools/',
  '/testing-evals': '/testing/',
  '/result': '/output/',
  '/mcp/run-python': 'https://github.com/pydantic/mcp-run-python',
  '/temporal': '/durable_execution/temporal/',
  '/api': '/api/agent/',
  '/examples/question-graph': '/graph/',
  '/api/models/vertexai': '/models/google/',
  '/models/gemini': '/models/google/',
  '/api/models/gemini': '/api/models/google/',
  '/contributing': '/contributing/',
  '/api/format_as_xml': '/api/format_prompt/',
  '/api/models/ollama': '/models/openai/#ollama',
  '/examples': 'examples/setup/',
  '/mcp': '/mcp/overview/',
  '/models': '/models/overview/',
  '/ag-ui': '/ui/ag-ui/',
  '/agents': '/agent/'
}

function redirect(pathname: string): string | null {
  return redirect_lookup[pathname.replace(/[/:]+$/, '')] ?? null
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
  while (typeof url === 'string') {
    const response: Response = await fetch(url, { headers })
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
  tag_name: string
  body: string
  html_url: string
}

const githubIcon = `<span class="twemoji"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"></path></svg></span>`

function prepRelease(release: Release): string {
  const body = release.body
    .replace(/(#+)/g, (m) => `##${m}`)
    .replace(/https:\/\/github.com\/pydantic\/pydantic-ai\/pull\/(\d+)/g, (url, id) => `[#${id}](${url})`)
    .replace(/(\s)@([\w-]+)/g, (_, s, u) => `${s}[@${u}](https://github.com/${u})`)
    .replace(/\*\*Full Changelog\*\*: (\S+)/, (_, url) => `[${githubIcon} Compare diff](${url}).`)
  return `
### ${release.name}

${body}

[${githubIcon} View ${release.tag_name} release](${release.html_url}).
`
}

/** Logic to return text (the markdown document where available) when the Accept header prefers plain text over html
 * See https://x.com/threepointone/status/1971988718052651300
 */
async function maybeGetTextResponse(request: Request, env: Env): Promise<Response | undefined> {
  if (!preferText(request)) {
    return
  }
  const url = new URL(request.url)
  url.pathname = `${url.pathname.replace(/[/:]+$/, '')}/index.md`
  const r = await env.ASSETS.fetch(url)
  if (r.status === 200) {
    return new Response(r.body, {
      headers: {
        'content-type': 'text/plain',
      },
    })
  }
}

function preferText(request: Request): boolean {
  const accept = request.headers.get('accept')
  if (!accept || request.method !== 'GET') {
    return false
  }
  for (const option of accept.split(',')) {
    const lowerOption = option.toLowerCase()
    if (lowerOption.includes('html')) {
      return false
    } else if (lowerOption.includes('text/plain')) {
      return true
    }
  }
  return false
}
