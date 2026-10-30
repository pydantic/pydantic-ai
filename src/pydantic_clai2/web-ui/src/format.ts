/** `1234` as `1.2k`, `56789` as `57k`; `?` when unknown. */
export function compact(tokens: number | null | undefined): string {
  if (tokens === null || tokens === undefined) return '?'
  if (tokens < 1000) return String(tokens)
  return `${(tokens / 1000).toFixed(tokens >= 10000 ? 0 : 1)}k`
}

/** `75` seconds as `1m 15s`, `4.2` as `4.2s`. */
export function duration(seconds: number): string {
  if (seconds < 10) return `${Math.min(seconds, 9.9).toFixed(1)}s`
  if (seconds < 60) return `${Math.floor(seconds)}s`
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
}
