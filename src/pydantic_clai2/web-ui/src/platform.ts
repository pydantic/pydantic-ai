/** Cmd on a Mac, Ctrl elsewhere: on a Mac, Ctrl+K and Ctrl+B are text editing keys. */
export const MAC = /Mac|iPhone|iPad/.test(navigator.platform)
export const MOD = MAC ? '⌘' : 'Ctrl+'
