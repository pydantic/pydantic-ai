import DOMPurify from 'dompurify'
import hljs from 'highlight.js/lib/core'
import bash from 'highlight.js/lib/languages/bash'
import css from 'highlight.js/lib/languages/css'
import diff from 'highlight.js/lib/languages/diff'
import go from 'highlight.js/lib/languages/go'
import javascript from 'highlight.js/lib/languages/javascript'
import json from 'highlight.js/lib/languages/json'
import markdown from 'highlight.js/lib/languages/markdown'
import python from 'highlight.js/lib/languages/python'
import rust from 'highlight.js/lib/languages/rust'
import sql from 'highlight.js/lib/languages/sql'
import typescript from 'highlight.js/lib/languages/typescript'
import xml from 'highlight.js/lib/languages/xml'
import yaml from 'highlight.js/lib/languages/yaml'
import { Marked } from 'marked'

for (const [name, language] of Object.entries({
  bash, css, diff, go, javascript, json, markdown, python, rust, sql, typescript, xml, yaml,
})) {
  hljs.registerLanguage(name, language)
}
hljs.registerAliases(['sh', 'shell', 'zsh', 'console'], { languageName: 'bash' })
hljs.registerAliases(['js', 'jsx', 'mjs'], { languageName: 'javascript' })
hljs.registerAliases(['ts', 'tsx'], { languageName: 'typescript' })
hljs.registerAliases(['py'], { languageName: 'python' })
hljs.registerAliases(['html', 'svg'], { languageName: 'xml' })
hljs.registerAliases(['yml', 'toml'], { languageName: 'yaml' })
hljs.registerAliases(['md'], { languageName: 'markdown' })

const escape = (text: string) =>
  text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

export function highlight(code: string, language?: string): string {
  if (language && hljs.getLanguage(language)) {
    return hljs.highlight(code, { language, ignoreIllegals: true }).value
  }
  return escape(code)
}

/** The highlight.js language for a file, from its extension (`py`, `tsx`, `rs`, `yml`, …), if one is loaded. */
export function languageOf(path: string): string | undefined {
  const extension = /\.([^./]+)$/.exec(path)?.[1]?.toLowerCase()
  return extension && hljs.getLanguage(extension) ? extension : undefined
}

/** Lines of code highlighted as one text, so a string or comment spanning lines is coloured right,
 * then split back into lines, each closing the spans it leaves open and reopening them on the next. */
export function highlightLines(lines: string[], language: string | undefined): string[] {
  let open: string[] = []
  return highlight(lines.join('\n'), language)
    .split('\n')
    .map((line) => {
      const start = open.join('')
      for (const [tag] of line.matchAll(/<span[^>]*>|<\/span>/g)) {
        if (tag === '</span>') open.pop()
        else open.push(tag)
      }
      open = [...open]
      return start + line + '</span>'.repeat(open.length)
    })
}

// `breaks` keeps a single line break, as the terminal renderer does; code blocks are unaffected.
const marked = new Marked({
  gfm: true,
  breaks: true,
  renderer: {
    code({ text, lang }) {
      const language = (lang ?? '').split(/\s/)[0] ?? ''
      const label = language ? `<span class="code-lang">${escape(language)}</span>` : ''
      return `<div class="code-block"><div class="code-head">${label}<button class="copy" data-copy>Copy</button></div><pre><code class="hljs">${highlight(text, language)}</code></pre></div>`
    },
    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens)
      const titled = title ? ` title="${escape(title)}"` : ''
      return `<a href="${escape(href)}"${titled} target="_blank" rel="noreferrer noopener">${text}</a>`
    },
  },
})

DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank')
    node.setAttribute('rel', 'noreferrer noopener')
  }
})
// Set once: a config passed to each `sanitize` call is parsed and its allow-lists copied again every time.
DOMPurify.setConfig({ ADD_ATTR: ['target', 'data-copy'], FORBID_TAGS: ['style', 'form', 'input'] })

/** Model output as safe HTML: Markdown parsed, then sanitized, since the page holds a token that runs commands. */
export function renderMarkdown(text: string): string {
  return DOMPurify.sanitize(marked.parse(text, { async: false }))
}

/** How many characters of Markdown and its HTML `cachedMarkdown` keeps, before dropping the least recent. */
const CACHED = 8_000_000
const rendered = new Map<string, string>()
let cachedSize = 0

/**
 * `renderMarkdown`, kept for the page's lifetime by text: opening a session again, or remounting a transcript,
 * renders each finished message once. Text still streaming changes with every delta, so it is not cached.
 */
export function cachedMarkdown(text: string): string {
  const found = rendered.get(text)
  if (found !== undefined) {
    // Most recently used last, so the oldest is the first to go.
    rendered.delete(text)
    rendered.set(text, found)
    return found
  }
  const html = renderMarkdown(text)
  rendered.set(text, html)
  cachedSize += text.length + html.length
  for (const [oldest, oldHtml] of rendered) {
    if (cachedSize <= CACHED) break
    rendered.delete(oldest)
    cachedSize -= oldest.length + oldHtml.length
  }
  return html
}
