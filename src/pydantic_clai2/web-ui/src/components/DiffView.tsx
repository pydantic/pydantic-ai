import { memo, useState } from 'react'
import { highlightLines, languageOf } from '../markdown'
import type { Diff } from '../protocol'

/** A diff's lines, with its `---`/`+++` file headers left out: only before its first hunk, so a
 * removed line that itself starts with `--` (a SQL comment, a Markdown rule) is kept. */
function body(diff: string): string[] {
  const lines = diff.replace(/\n$/, '').split('\n')
  const first = lines.findIndex((line) => line.startsWith('@@ '))
  if (first < 0) return lines
  return lines.filter((line, index) => index >= first || !(line.startsWith('--- ') || line.startsWith('+++ ')))
}

export function diffStats(diff: string): { added: number; removed: number } {
  let added = 0
  let removed = 0
  for (const line of body(diff)) {
    if (line.startsWith('+')) added++
    else if (line.startsWith('-')) removed++
  }
  return { added, removed }
}

interface Row {
  kind: 'hunk' | 'add' | 'del' | 'ctx'
  old: string
  new: string
  text: string
  html: string
}

/** Each side of every hunk highlighted as a whole: the old one (context and removals) and the new one
 * (context and additions), so code that spans lines, such as a docstring, is coloured as in the file. */
function colour(rows: Row[], language: string | undefined): void {
  for (let start = 0; start < rows.length; ) {
    let end = start
    while (end < rows.length && rows[end]?.kind !== 'hunk') end++
    const hunk = rows.slice(start, end)
    for (const side of ['del', 'add'] as const) {
      const lines = hunk.filter((row) => row.kind === side || row.kind === 'ctx')
      highlightLines(
        lines.map((row) => row.text),
        language,
      ).forEach((html, index) => {
        const row = lines[index]
        // Context lines are on both sides; they take the new side's colours.
        if (row && (row.kind === side || side === 'add')) row.html = html
      })
    }
    start = end + 1
  }
}

/** A unified diff with old and new line numbers, additions and removals tinted, code highlighted by file type.
 * With a `preview`, only that many code lines (and the hunk headers among them) are highlighted and mounted until
 * the user asks for the rest: a long diff's every line costs page nodes and layout that a glance does not need. */
export const DiffView = memo(function DiffView({ diff, preview }: { diff: Diff; preview?: number }) {
  const [whole, setWhole] = useState(false)
  const limit = whole ? undefined : preview
  const rows: Row[] = []
  let oldLine = 0
  let newLine = 0
  let numbered = false
  for (const line of body(diff.diff)) {
    const hunk = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)/.exec(line)
    if (hunk) {
      oldLine = Number(hunk[1])
      newLine = Number(hunk[2])
      numbered = hunk[1] !== '0' || hunk[2] !== '1' || rows.length > 0
      // An unnumbered hunk still separates the sides, so it is kept and only its row is hidden.
      rows.push({ kind: 'hunk', old: '', new: '', text: line, html: '' })
    } else if (line.startsWith('+')) {
      rows.push({ kind: 'add', old: '', new: String(newLine++), text: line.slice(1), html: '' })
    } else if (line.startsWith('-')) {
      rows.push({ kind: 'del', old: String(oldLine++), new: '', text: line.slice(1), html: '' })
    } else if (line.startsWith('\\')) {
      continue
    } else if (line || rows.length) {
      rows.push({ kind: 'ctx', old: String(oldLine++), new: String(newLine++), text: line.slice(1), html: '' })
    }
  }
  let end = rows.length
  if (limit !== undefined) {
    // Stop right after the last code line shown, so the preview never ends on the next hunk's header.
    let shown = 0
    for (end = 0; end < rows.length && shown < limit; end++) if (rows[end]?.kind !== 'hunk') shown++
  }
  const visible = rows.slice(0, end)
  const more = rows.slice(end).filter((row) => row.kind !== 'hunk').length
  colour(visible, languageOf(diff.path))
  return (
    <div className="diff">
      <table>
        <tbody>
          {visible.map((row, index) =>
            row.kind === 'hunk' ? (
              numbered && (
                <tr key={index} className="diff-hunk">
                  <td className="ln" />
                  <td className="ln" />
                  <td className="sign" />
                  <td className="code">{row.text}</td>
                </tr>
              )
            ) : (
              <tr key={index} className={`diff-${row.kind}`}>
                {numbered && <td className="ln">{row.old}</td>}
                {numbered && <td className="ln">{row.new}</td>}
                <td className="sign">{row.kind === 'add' ? '+' : row.kind === 'del' ? '−' : ''}</td>
                {/* `highlight` escapes the text, and highlight.js adds only its own spans. */}
                <td className="code hljs" dangerouslySetInnerHTML={{ __html: row.html || ' ' }} />
              </tr>
            ),
          )}
        </tbody>
      </table>
      {more > 0 && (
        <button className="diff-note diff-more" onClick={() => setWhole(true)}>
          {more} more {more === 1 ? 'line' : 'lines'}
        </button>
      )}
      {diff.truncated && <div className="diff-note">Diff truncated</div>}
    </div>
  )
})
