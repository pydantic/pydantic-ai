import type { Palette } from './protocol'

// The slots CLAI paints its roles with in the terminal (`ui/rendering/theme.py`), so `/theme` looks the same here.
const ACCENT = 12
const ERROR = 1
const THINKING = 5
const SUCCESS = 14
const WARNING = 3
const INFO = 6
const DIFF_TINT = 0.28

function channels(hex: string): [number, number, number] {
  const value = hex.replace('#', '')
  return [0, 2, 4].map((index) => parseInt(value.slice(index, index + 2), 16)) as [number, number, number]
}

/** `from` moved `amount` of the way to `to`. */
export function blend(from: string, to: string, amount: number): string {
  const a = channels(from)
  const b = channels(to)
  return `#${a.map((value, index) => Math.round(value + ((b[index] ?? 0) - value) * amount).toString(16).padStart(2, '0')).join('')}`
}

function alpha(hex: string, amount: number): string {
  const [red, green, blue] = channels(hex)
  return `rgba(${red}, ${green}, ${blue}, ${amount})`
}

/** As the terminal decides a palette is light: by its background's weighted brightness. */
export function isLight(palette: Palette): boolean {
  const [red, green, blue] = channels(palette.bg)
  return 0.299 * red + 0.587 * green + 0.114 * blue > 128
}

/** The page's colour variables for a palette, in place of Pydantic's own. */
export function paletteVariables(palette: Palette): Record<string, string> {
  const { bg, fg } = palette
  const slot = (index: number) => palette.ansi[index] ?? fg
  const light = isLight(palette)
  const accent = slot(ACCENT)
  const [red, green, blue] = channels(accent)
  const accentText = 0.299 * red + 0.587 * green + 0.114 * blue > 150 ? blend(bg, '#000000', 0.6) : '#ffffff'
  return {
    '--bg': bg,
    '--bg-sidebar': blend(bg, light ? '#000000' : fg, light ? 0.035 : 0.03),
    '--surface': blend(bg, fg, 0.07),
    '--surface-raised': light ? blend(bg, '#ffffff', 0.6) : blend(bg, fg, 0.12),
    '--composer': light ? blend(bg, '#ffffff', 0.6) : blend(bg, fg, 0.06),
    '--hover': alpha(fg, 0.06),
    '--active': alpha(accent, 0.14),
    '--border': alpha(fg, 0.1),
    '--border-strong': alpha(fg, 0.18),
    '--text': fg,
    '--text-soft': blend(fg, bg, 0.12),
    '--muted': blend(fg, bg, 0.35),
    '--faint': blend(fg, bg, 0.5),
    '--lithium': accent,
    '--lithium-bright': blend(accent, light ? '#000000' : '#ffffff', 0.15),
    '--lithium-deep': blend(accent, '#000000', 0.2),
    '--accent': accent,
    '--accent-hover': blend(accent, light ? '#000000' : '#ffffff', 0.15),
    '--accent-strong-hover': blend(accent, '#000000', 0.34),
    '--accent-text': accentText,
    '--link': accent,
    '--focus': alpha(accent, 0.55),
    '--green': slot(SUCCESS),
    '--red': slot(ERROR),
    '--amber': slot(WARNING),
    '--add-bg': blend(bg, slot(2), DIFF_TINT),
    '--add-text': fg,
    '--del-bg': blend(bg, slot(1), DIFF_TINT),
    '--del-text': fg,
    '--code-bg': blend(bg, '#000000', light ? 0.03 : 0.25),
    '--shadow': `0 16px 48px ${alpha('#000000', light ? 0.12 : 0.5)}, 0 0 0 1px ${alpha(fg, 0.1)}`,
    '--glow': `0 10px 32px ${alpha('#000000', light ? 0.08 : 0.4)}`,
    '--hl-keyword': slot(THINKING),
    '--hl-string': slot(SUCCESS),
    '--hl-title': slot(ACCENT),
    '--hl-number': slot(WARNING),
    '--hl-comment': blend(fg, bg, 0.45),
    '--hl-attr': slot(INFO),
  }
}
