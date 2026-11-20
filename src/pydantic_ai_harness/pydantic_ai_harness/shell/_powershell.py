"""The PowerShell side of `Job`, for a `'windows'` workspace, which has no `sh`, `setsid` or `kill`.

The launcher writes the command to `command.ps1` and the wrapper to `supervisor.ps1` in the job directory,
and starts the wrapper as a hidden PowerShell of its own. Each script reaches its PowerShell as a file, not
on the command line, where antivirus software flags an `-EncodedCommand` launch. The wrapper publishes
`status.json` as the `sh` one does, runs the command, copies its output into the log files byte for byte,
and publishes the exit code when it ends. Stopping a job runs `stop.ps1`, which ends the wrapper's process
tree with `taskkill`.
"""

from __future__ import annotations

from pydantic_ai.workspaces.windows import powershell_script

MISSING_PROCESS = 128
"""`taskkill`'s exit status when no process has the PID."""

KILLED = 1
"""The exit code a stop publishes for a job `taskkill` ended, which `taskkill /F` gives it."""


def literal(value: str) -> str:
    """`value` as a PowerShell string literal."""
    return "'" + value.replace("'", "''") + "'"


_PUBLISH = """function Publish($code, $owner = $PID) {
  $tmp = Join-Path $dir 'status.tmp'
  [IO.File]::WriteAllText($tmp, ('{"pid": ' + $owner + ', "exit_code": ' + $code + '}'))
  # A reader holding `status.json` open makes the replace fail for a moment, so it is retried.
  for ($try = 0; $try -lt 100; $try++) {
    try { Move-Item -LiteralPath $tmp -Destination (Join-Path $dir 'status.json') -Force -ErrorAction Stop; return }
    catch { Start-Sleep -Milliseconds 20 }
  }
}
"""
"""Publish `status.json` for the job in `$dir`: the exit code `$code` of process `$owner`, the wrapper by default."""

_WRAPPER = (
    _PUBLISH
    + """Publish 'null'
$ready = Join-Path $dir 'launch.ready'
$launcher = [int][IO.File]::ReadAllText((Join-Path $dir 'launcher'))
while (-not (Test-Path -LiteralPath $ready)) {
  if (-not (Get-Process -Id $launcher -ErrorAction Ignore) -and -not (Test-Path -LiteralPath $ready)) { Publish 143; exit }
  Start-Sleep -Milliseconds 100
}
if (Test-Path -LiteralPath (Join-Path $dir 'stop')) { Publish 143; exit }
function Open-Log($name) {
  $stream = [IO.FileStream]::new((Join-Path $dir $name), 'Open', 'Write', 'ReadWrite, Delete')
  $null = $stream.Seek(0, 'End')
  $stream
}
function Add-Clear {
  $clears = [IO.File]::Open((Join-Path $dir 'log_clears'), 'Append', 'Write', 'ReadWrite, Delete')
  $clears.WriteByte(120)
  $clears.Dispose()
}
function Clear-Logs {
  $notice = [Text.Encoding]::UTF8.GetBytes("[Log exceeded max_log_bytes=$limit; earlier output was cleared.]`n")
  foreach ($stream in $logs.Values) {
    if ($stream.Length -gt $limit -and $stream.Length -gt $notice.Length) {
      Add-Clear
      $stream.SetLength(0)
      $stream.Position = 0
      $stream.Write($notice, 0, $notice.Length)
      $stream.Flush()
      Add-Clear
    }
  }
}
$start = [Diagnostics.ProcessStartInfo]::new((Get-Process -Id $PID).Path)
$start.Arguments = '-NoLogo -NoProfile -NonInteractive -Command "& ([ScriptBlock]::Create([IO.File]::ReadAllText(''{0}'')))"' -f `
  (Join-Path $dir 'command.ps1').Replace("'", "''")
$start.WorkingDirectory = $cwd
$start.UseShellExecute = $false
$start.RedirectStandardInput = $true
$start.RedirectStandardOutput = $true
$start.RedirectStandardError = $true
$process = [Diagnostics.Process]::Start($start)
$process.StandardInput.Close()
$logs = @{}
$pipes = @(@{ source = $process.StandardOutput.BaseStream; log = $out }, @{ source = $process.StandardError.BaseStream; log = $err })
foreach ($pipe in $pipes) {
  if (-not $logs.ContainsKey($pipe.log)) { $logs[$pipe.log] = Open-Log $pipe.log }
  $pipe.buffer = [byte[]]::new(65536)
  $pipe.read = $pipe.source.ReadAsync($pipe.buffer, 0, $pipe.buffer.Length)
}
$check = [DateTime]::UtcNow.AddSeconds($interval)
while ($open = @($pipes | Where-Object { $_.read })) {
  $null = [Threading.Tasks.Task]::WaitAny([Threading.Tasks.Task[]]@($open | ForEach-Object { $_.read }), 250)
  foreach ($pipe in $open) {
    if (-not $pipe.read.IsCompleted) { continue }
    try { $count = $pipe.read.Result } catch { $count = 0 }
    if ($count -le 0) { $pipe.read = $null; continue }
    $stream = $logs[$pipe.log]
    $stream.Write($pipe.buffer, 0, $count)
    $stream.Flush()
    $pipe.read = $pipe.source.ReadAsync($pipe.buffer, 0, $pipe.buffer.Length)
  }
  if ($limit -and [DateTime]::UtcNow -ge $check) { Clear-Logs; $check = [DateTime]::UtcNow.AddSeconds($interval) }
}
# Once more for the output since the last check, so a finished job's logs end within the limit.
if ($limit) { Clear-Logs }
foreach ($stream in $logs.Values) { $stream.Dispose() }
$process.WaitForExit()
Publish $process.ExitCode
"""
)
"""The job's supervisor, given `$dir`, `$cwd`, the logs `$out` and `$err`, the log limit `$limit` (`0` for none),
and the seconds between log checks `$interval`.

It waits for `launch.ready`, like the `sh` wrapper, and never starts the command when its launcher died first
or a `stop` file appeared. It copies the command's output into the logs itself, so it can empty a log past the
limit the way the `sh` watcher does, notice and `log_clears` bytes included: a log another process writes keeps
that process's offset, so emptying it would leave the earlier bytes as zeros. The command reads `command.ps1` as
text rather than running it as a script file, so no execution policy applies to it.
"""

_LAUNCHER = """try {
  New-Item -ItemType Directory -Force -Path (Split-Path -LiteralPath $dir) -ErrorAction Stop | Out-Null
  # Without `-Force`, so of two launches that claim the same directory, only one creates it.
  New-Item -ItemType Directory -Path $dir -ErrorAction Stop | Out-Null
} catch { if ($exclusive -and (Test-Path -LiteralPath $dir)) { exit 126 }; exit 125 }
# Owner-only, before any job file is written, as `umask 077` makes the `sh` side's: the command and its output
# may hold secrets, and a checkout's own ACL can let other local users read it.
try {
  $acl = [Security.AccessControl.DirectorySecurity]::new()
  $acl.SetAccessRuleProtection($true, $false)
  $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new(
    [Security.Principal.WindowsIdentity]::GetCurrent().User, 'FullControl', 'ContainerInherit, ObjectInherit', 'None', 'Allow'))
  Set-Acl -LiteralPath $dir -AclObject $acl -ErrorAction Stop
} catch { Remove-Item -LiteralPath $dir -Recurse -Force -ErrorAction Ignore; exit 125 }
foreach ($log in $logs) { [IO.File]::WriteAllBytes((Join-Path $dir $log), [byte[]]@()) }
[IO.File]::WriteAllText((Join-Path $dir 'command.ps1'), $script, [Text.UTF8Encoding]::new($true))
[IO.File]::WriteAllText((Join-Path $dir 'supervisor.ps1'), $supervisor, [Text.UTF8Encoding]::new($true))
[IO.File]::WriteAllText((Join-Path $dir 'launcher'), "$PID")
$wrapper = Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -PassThru -ArgumentList (
  '-NoLogo -NoProfile -NonInteractive -Command "& ([ScriptBlock]::Create([IO.File]::ReadAllText(''{0}'')))"' -f `
  (Join-Path $dir 'supervisor.ps1').Replace("'", "''"))
while (-not (Test-Path -LiteralPath (Join-Path $dir 'status.json'))) { Start-Sleep -Milliseconds 100 }
[IO.File]::WriteAllText((Join-Path $dir 'launch.ready'), '')
# Its start time tells the wrapper from a later process that reuses its PID, which Windows does soon.
[IO.File]::WriteAllText((Join-Path $dir 'wrapper.started'), "$($wrapper.StartTime.ToFileTimeUtc())")
[IO.File]::WriteAllText((Join-Path $dir 'stop.ps1'), $stop, [Text.UTF8Encoding]::new($true))
[IO.File]::WriteAllText((Join-Path $dir 'handle'), "$($wrapper.Id) -")
[Console]::Out.Write("$($wrapper.Id) -")
"""
"""Create the job directory, write the command and the wrapper to it, start the wrapper, and print `<pid> -`.

Like the `sh` launcher, an exclusive launch exits 126 when the directory exists already and 125 when it cannot
be created, and it returns once the wrapper has published its status. The `-` says there is no process group
to signal: `Job.kill` ends the wrapper's process tree instead.
"""


def launch_script(
    directory: str, cwd: str, command: str, *, exclusive: bool, combined: bool, log_limit: int | None, interval: float
) -> str:
    """The PowerShell that launches `command` as a job in `directory`, starting in `cwd`."""
    stdout_log, stderr_log = ('output.log', 'output.log') if combined else ('stdout.log', 'stderr.log')
    wrapper = (
        f'$dir = {literal(directory)}\n$cwd = {literal(cwd)}\n'
        f'$out = {literal(stdout_log)}\n$err = {literal(stderr_log)}\n'
        f'$limit = {log_limit or 0}\n$interval = {interval:g}\n' + _WRAPPER
    )
    logs = ', '.join(literal(log) for log in dict.fromkeys((stdout_log, stderr_log)))
    return (
        f'$dir = {literal(directory)}\n$exclusive = ${str(exclusive).lower()}\n$logs = @({logs})\n'
        f'$script = {literal(powershell_script(command))}\n$supervisor = {literal(wrapper)}\n'
        f'$stop = {literal(stop_script(directory))}\n' + _LAUNCHER
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


def stop_script(directory: str) -> str:
    """PowerShell that stops the job in `directory`, as `stop.ps1` there does for a model's `shell` command.

    It ends the wrapper's process tree with `taskkill`, but only while the wrapper's PID has the start time the
    launcher recorded: a wrapper that died without its status leaves its PID free, and Windows soon gives it to
    another process. `taskkill /F` gives the wrapper no chance to publish, so this publishes `KILLED` for it, and
    the job's status file never reads as running after a stop.
    """
    return f"""$dir = {literal(directory)}
{_PUBLISH}[IO.File]::WriteAllText((Join-Path $dir 'stop'), '')
$wrapper = [IO.File]::ReadAllText((Join-Path $dir 'handle')).Split(' ')[0]
$started = [IO.File]::ReadAllText((Join-Path $dir 'wrapper.started'))
try {{ $current = "$((Get-Process -Id $wrapper -ErrorAction Stop).StartTime.ToFileTimeUtc())" }} catch {{ $current = $null }}
if ($current -eq $started) {{
  taskkill /T /F /PID $wrapper | Out-Null
  if ($LASTEXITCODE -notin 0, {MISSING_PROCESS}) {{ exit $LASTEXITCODE }}
}}
if ([IO.File]::ReadAllText((Join-Path $dir 'status.json')) -match '"exit_code": null') {{ Publish {KILLED} $wrapper }}
exit 0"""
