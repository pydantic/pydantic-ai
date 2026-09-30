"""The host `PATH`, minus the directories a sandboxed command could have written to."""

TRUSTED_PATH_FUNCTIONS = r"""__pai_inside() {
  if [ "$2" = / ]; then
    return 0
  fi
  case $1 in
    "$2"|"$2"/*) return 0 ;;
  esac
  return 1
}
__pai_trusted_path() {
  __pai_kept=
  __pai_saved=$IFS
  IFS=:
  for __pai_dir in $PATH; do
    case $__pai_dir in
      /*) ;;
      *) continue ;;
    esac
    __pai_canon=$(cd "$__pai_dir" 2>/dev/null && pwd -P) || continue
    case $__pai_canon in
      *:*) continue ;;
    esac
    if [ -n "$1" ] && __pai_inside "$__pai_canon" "$1"; then
      continue
    fi
    __pai_kept=${__pai_kept:+$__pai_kept:}$__pai_canon
  done
  IFS=$__pai_saved
  printf '%s\n' "$__pai_kept"
}
"""
"""POSIX shell functions for scripts that run on the host, outside the sandbox.

`__pai_trusted_path DIR` prints `$PATH` with every entry canonical, leaving out relative entries
and any entry inside the canonical directory `DIR` (none when `DIR` is empty): a command that can
write there could otherwise plant a program the host runs next. It uses only shell builtins, so it
can't be tricked by the directories it filters. `__pai_inside PATH DIR` tells whether `PATH` is
`DIR` or under it.
"""
