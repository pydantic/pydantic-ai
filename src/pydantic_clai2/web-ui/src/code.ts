/** The `code` and `restart` of a `run_code` cell's arguments, readable while they stream as partial JSON. */
export interface Cell {
  code: string
  restart: boolean
}

const ESCAPES: Record<string, string> = { n: '\n', t: '\t', r: '\r', b: '\b', f: '\f', '"': '"', '\\': '\\', '/': '/' }

/** Decode a JSON string body up to its closing quote, or as far as it has arrived. */
function decode(text: string, start: number): string {
  let out = ''
  for (let index = start; index < text.length; index++) {
    const char = text[index]
    if (char === '"') return out
    if (char !== '\\') {
      out += char
      continue
    }
    const next = text[index + 1]
    if (next === undefined) return out
    if (next === 'u') {
      const hex = text.slice(index + 2, index + 6)
      if (hex.length < 4) return out
      out += String.fromCharCode(parseInt(hex, 16))
      index += 5
    } else {
      out += ESCAPES[next] ?? next
      index += 1
    }
  }
  return out
}

export function cell(args: string): Cell {
  try {
    const value = JSON.parse(args) as { code?: unknown; restart?: unknown }
    return { code: typeof value.code === 'string' ? value.code : '', restart: value.restart === true }
  } catch {
    const match = /"code"\s*:\s*"/.exec(args)
    return { code: match ? decode(args, match.index + match[0].length) : '', restart: /"restart"\s*:\s*true/.test(args) }
  }
}
