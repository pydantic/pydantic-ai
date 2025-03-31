export default {
  async fetch(request, env): Promise<Response> {
    const r = await env.ASSETS.fetch(request)
    if (r.status == 404) {
      const url = new URL(request.url)
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
}

function redirect(pathname: string): string | null {
  return redirect_lookup[pathname.replace(/\/+$/, '')] ?? null
}
