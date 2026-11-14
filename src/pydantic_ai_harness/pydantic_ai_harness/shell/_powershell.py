"""The PowerShell side of `Job`, for a `'windows'` workspace, which has no `sh`, `setsid` or `kill`.

The launcher writes the command to `command.ps1` in the job directory and starts the wrapper as a
hidden PowerShell of its own. The wrapper publishes `status.json` as the `sh` one does, runs the
command through `cmd.exe`, whose redirection writes its output to the log files byte for byte, and
publishes the exit code when it ends. Stopping a job ends the wrapper's process tree with `taskkill`.
"""

from __future__ import annotations

from pydantic_ai.workspaces.windows import (
    _EPILOGUE,  # pyright: ignore[reportPrivateUsage]
    _PREAMBLE,  # pyright: ignore[reportPrivateUsage]
)

MISSING_PROCESS = 128
"""`taskkill`'s exit status when no process has the PID."""

KILLED = 1
"""The exit code `Job.kill` publishes for a job `taskkill` ended, which `taskkill /F` gives it."""


def literal(value: str) -> str:
    """`value` as a PowerShell string literal."""
    return "'" + value.replace("'", "''") + "'"


_WRAPPER = """function Publish($code) {
  $tmp = Join-Path $dir 'status.tmp'
  [IO.File]::WriteAllText($tmp, ('{"pid": ' + $PID + ', "exit_code": ' + $code + '}'))
  # A reader holding `status.json` open makes the replace fail for a moment, so it is retried.
  for ($try = 0; $try -lt 100; $try++) {
    try { Move-Item -LiteralPath $tmp -Destination (Join-Path $dir 'status.json') -Force -ErrorAction Stop; return }
    catch { Start-Sleep -Milliseconds 20 }
  }
}
Publish 'null'
$ready = Join-Path $dir 'launch.ready'
$launcher = [int][IO.File]::ReadAllText((Join-Path $dir 'launcher'))
while (-not (Test-Path -LiteralPath $ready)) {
  if (-not (Get-Process -Id $launcher -ErrorAction Ignore) -and -not (Test-Path -LiteralPath $ready)) { Publish 143; exit }
  Start-Sleep -Milliseconds 100
}
if (Test-Path -LiteralPath (Join-Path $dir 'stop')) { Publish 143; exit }
$run = "& ([ScriptBlock]::Create([IO.File]::ReadAllText('{0}'))); exit `$LASTEXITCODE" -f `
  (Join-Path $dir 'command.ps1').Replace("'", "''")
$command = '"{0}" -NoLogo -NoProfile -NonInteractive -Command "{1}" > "{2}" {3}' -f `
  (Get-Process -Id $PID).Path, $run, (Join-Path $dir $out), $redirect
$process = Start-Process -FilePath $env:ComSpec -ArgumentList ('/d /s /c "' + $command + '"') `
  -WorkingDirectory $cwd -WindowStyle Hidden -PassThru
$null = $process.Handle
$process.WaitForExit()
Publish $process.ExitCode
"""
"""The job's supervisor, given `$dir`, `$cwd`, the stdout log `$out` and the stderr redirection `$redirect`.

It waits for `launch.ready`, like the `sh` wrapper, and never starts the command when its launcher died first
or a `stop` file appeared. The command reads `command.ps1` as text rather than running it as a script file,
so no execution policy applies to it.
"""

_LAUNCHER = """try {
  New-Item -ItemType Directory -Force -Path (Split-Path -LiteralPath $dir) -ErrorAction Stop | Out-Null
  # Without `-Force`, so of two launches that claim the same directory, only one creates it.
  New-Item -ItemType Directory -Path $dir -ErrorAction Stop | Out-Null
} catch { if ($exclusive -and (Test-Path -LiteralPath $dir)) { exit 126 }; exit 125 }
foreach ($log in $logs) { [IO.File]::WriteAllBytes((Join-Path $dir $log), [byte[]]@()) }
[IO.File]::WriteAllText((Join-Path $dir 'command.ps1'), $script, [Text.UTF8Encoding]::new($true))
[IO.File]::WriteAllText((Join-Path $dir 'launcher'), "$PID")
$wrapper = Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -PassThru `
  -ArgumentList '-NoLogo', '-NoProfile', '-NonInteractive', '-EncodedCommand', `
  ([Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($supervisor)))
while (-not (Test-Path -LiteralPath (Join-Path $dir 'status.json'))) { Start-Sleep -Milliseconds 100 }
[IO.File]::WriteAllText((Join-Path $dir 'launch.ready'), '')
[IO.File]::WriteAllText((Join-Path $dir 'handle'), "$($wrapper.Id) -")
[Console]::Out.Write("$($wrapper.Id) -")
"""
"""Create the job directory, write the command to `command.ps1`, start the wrapper, and print `<pid> -`.

Like the `sh` launcher, an exclusive launch exits 126 when the directory exists already and 125 when it cannot
be created, and it returns once the wrapper has published its status. The `-` says there is no process group
to signal: `Job.kill` ends the wrapper's process tree instead.
"""


def launch_script(directory: str, cwd: str, command: str, *, exclusive: bool, combined: bool) -> str:
    """The PowerShell that launches `command` as a job in `directory`, starting in `cwd`."""
    out, err = ('output.log', 'output.log') if combined else ('stdout.log', 'stderr.log')
    # `cmd.exe` cannot open the same file for both streams, so a combined job's stderr follows its stdout.
    redirect = '2>&1' if combined else f'2> "{directory}\\{err}"'
    wrapper = (
        f'$dir = {literal(directory)}\n$cwd = {literal(cwd)}\n$out = {literal(out)}\n$redirect = {literal(redirect)}\n'
        + _WRAPPER
    )
    logs = ', '.join(literal(log) for log in dict.fromkeys((out, err)))
    return (
        f'$dir = {literal(directory)}\n$exclusive = ${str(exclusive).lower()}\n$logs = @({logs})\n'
        f'$script = {literal(_PREAMBLE + command + _EPILOGUE)}\n$supervisor = {literal(wrapper)}\n' + _LAUNCHER
    )


def read_script(path: str, offset: int, length: int) -> str:
    """PowerShell printing, in base64, up to `length` bytes of `path` from `offset`; it exits 66 when `path` is missing.

    Like the `sh` side, it reads only those bytes, whatever the log's size, and exits 67 when the log is unreadable.
    """
    return f"""$path = {literal(path)}
if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {{ exit 66 }}
try {{ $file = [IO.File]::Open($path, 'Open', 'Read', 'ReadWrite, Delete') }}
catch {{ if (Test-Path -LiteralPath $path) {{ exit 67 }}; exit 66 }}
try {{
  $null = $file.Seek({offset}, 'Begin')
  $buffer = [byte[]]::new([Math]::Min({length}, [Math]::Max(0, $file.Length - {offset})))
  $read = 0
  while ($read -lt $buffer.Length) {{
    $count = $file.Read($buffer, $read, $buffer.Length - $read)
    if ($count -eq 0) {{ break }}
    $read += $count
  }}
}} finally {{ $file.Dispose() }}
[Console]::Out.Write([Convert]::ToBase64String($buffer, 0, $read))"""


def kill_command(pid: int) -> list[str]:
    """The argv that ends process `pid` and every process it started that is still its descendant."""
    return ['taskkill', '/T', '/F', '/PID', str(pid)]
