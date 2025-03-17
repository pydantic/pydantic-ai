export default {
  async fetch(request, env): Promise<Response> {
    const r = await env.ASSETS.fetch(request)
    if (r.status == 404) {
      const url = new URL(request.url)
      url.pathname = '/404.html'
      const r = await env.ASSETS.fetch(url)
      return new Response(r.body, { status: 404 })
    }
    return r
  },
} satisfies ExportedHandler<Env>
