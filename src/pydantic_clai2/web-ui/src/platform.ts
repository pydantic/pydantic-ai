/** Cmd on a Mac, Ctrl elsewhere: on a Mac, Ctrl+K and Ctrl+B are text editing keys. */
export const MAC = /Mac|iPhone|iPad/.test(navigator.platform)
export const MOD = MAC ? '⌘' : 'Ctrl+'
/** The key that steers a running turn, as in the terminal: Option+Enter on a Mac. */
export const STEER_KEY = MAC ? '⌥Enter' : 'Alt+Enter'
