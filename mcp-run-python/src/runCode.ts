/* eslint @typescript-eslint/no-explicit-any: off */
import { loadPyodide } from 'pyodide'
import { preparePythonCode } from './prepareEnvCode.ts'
import { type LoggingLevel, SetLevelRequestSchema } from '@modelcontextprotocol/sdk/types.js'
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import * as path from '@std/path'
import z from 'zod'

export interface CodeFile {
  name: string
  content: string
  active: boolean
}

export async function runCode(
  files: CodeFile[],
  log: (level: LoggingLevel, data: string) => void,
  rootDir: string | null,
  mountDir: string | null,
): Promise<RunSuccess | RunError> {
  const output: string[] = []
  const pyodide = await loadPyodide({
    stdout: (msg) => {
      log('info', msg)
      output.push(msg)
    },
    stderr: (msg) => {
      log('warning', msg)
      output.push(msg)
    },
  })

  // Mount file system
  if (mountDir != null && rootDir != null) {
    // Ensure emscriptem directory is created
    pyodide.FS.mkdirTree(mountDir)
    // Mount local directory
    pyodide.FS.mount(
      pyodide.FS.filesystems.NODEFS,
      { root: rootDir },
      mountDir,
    )
  }

  // see https://github.com/pyodide/pyodide/discussions/5512
  const origLoadPackage = pyodide.loadPackage
  pyodide.loadPackage = (pkgs, options) =>
    origLoadPackage(pkgs, {
      // stop pyodide printing to stdout/stderr
      messageCallback: (msg: string) => log('debug', `loadPackage: ${msg}`),
      errorCallback: (msg: string) => {
        log('error', `loadPackage: ${msg}`)
        output.push(`install error: ${msg}`)
      },
      ...options,
    })

  await pyodide.loadPackage(['micropip', 'pydantic'])
  const sys = pyodide.pyimport('sys')

  // This is in the virtual in-memory emscriptem file system
  const dirPath = '/tmp/mcp_run_python'
  sys.path.append(dirPath)
  const pathlib = pyodide.pyimport('pathlib')
  pathlib.Path(dirPath).mkdir()
  const moduleName = '_prepare_env'

  pathlib.Path(`${dirPath}/${moduleName}.py`).write_text(preparePythonCode)

  const preparePyEnv: PreparePyEnv = pyodide.pyimport(moduleName)

  const prepareStatus = await preparePyEnv.prepare_env(pyodide.toPy(files))

  let runResult: RunSuccess | RunError
  if (prepareStatus.kind == 'error') {
    runResult = {
      status: 'install-error',
      output,
      error: prepareStatus.message,
    }
  } else {
    const { dependencies } = prepareStatus
    const activeFile = files.find((f) => f.active)! || files[0]
    try {
      const rawValue = await pyodide.runPythonAsync(activeFile.content, {
        globals: pyodide.toPy({ __name__: '__main__' }),
        filename: activeFile.name,
      })
      runResult = {
        status: 'success',
        dependencies,
        output,
        returnValueJson: preparePyEnv.dump_json(rawValue),
      }
    } catch (err) {
      runResult = {
        status: 'run-error',
        dependencies,
        output,
        error: formatError(err),
      }
    }
  }
  sys.stdout.flush()
  sys.stderr.flush()
  return runResult
}

interface RunSuccess {
  status: 'success'
  // we could record stdout and stderr separately, but I suspect simplicity is more important
  output: string[]
  dependencies: string[]
  returnValueJson: string | null
}

interface RunError {
  status: 'install-error' | 'run-error'
  output: string[]
  dependencies?: string[]
  error: string
}

export function asXml(runResult: RunSuccess | RunError): string {
  const xml = [`<status>${runResult.status}</status>`]
  if (runResult.dependencies?.length) {
    xml.push(
      `<dependencies>${JSON.stringify(runResult.dependencies)}</dependencies>`,
    )
  }
  if (runResult.output.length) {
    xml.push('<output>')
    const escapeXml = escapeClosing('output')
    xml.push(...runResult.output.map(escapeXml))
    xml.push('</output>')
  }
  if (runResult.status == 'success') {
    if (runResult.returnValueJson) {
      xml.push('<return_value>')
      xml.push(escapeClosing('return_value')(runResult.returnValueJson))
      xml.push('</return_value>')
    }
  } else {
    xml.push('<error>')
    xml.push(escapeClosing('error')(runResult.error))
    xml.push('</error>')
  }
  return xml.join('\n')
}

function escapeClosing(closingTag: string): (str: string) => string {
  const regex = new RegExp(`</?\\s*${closingTag}(?:.*?>)?`, 'gi')
  const onMatch = (match: string) => {
    return match.replace(/</g, '&lt;').replace(/>/g, '&gt;')
  }
  return (str) => str.replace(regex, onMatch)
}

// deno-lint-ignore no-explicit-any
function formatError(err: any): string {
  let errStr = err.toString()
  errStr = errStr.replace(/^PythonError: +/, '')
  // remove frames from inside pyodide
  errStr = errStr.replace(
    / {2}File "\/lib\/python\d+\.zip\/_pyodide\/.*\n {4}.*\n(?: {4,}\^+\n)?/g,
    '',
  )
  return errStr
}

interface PrepareSuccess {
  kind: 'success'
  dependencies: string[]
}
interface PrepareError {
  kind: 'error'
  message: string
}
interface PreparePyEnv {
  prepare_env: (files: CodeFile[]) => Promise<PrepareSuccess | PrepareError>
  // deno-lint-ignore no-explicit-any
  dump_json: (value: any) => string | null
}

// list of log levels to use for level comparison
const LogLevels: LoggingLevel[] = [
  'debug',
  'info',
  'notice',
  'warning',
  'error',
  'critical',
  'alert',
  'emergency',
]

/*
 * Resolve a mountDir cli option to a specific directory
 */
export function resolveMountDir(mountDir: string): string {
  // Base dir created by emscriptem
  // See https://emscripten.org/docs/api_reference/Filesystem-API.html#file-system-api
  const baseDir = '/home/pyodide'

  if (mountDir.trim() === '') {
    return path.join(baseDir, 'persistent')
  }

  if (path.isAbsolute(mountDir)) {
    return mountDir
  }

  // relative path
  return path.join(baseDir, mountDir)
}

export function registerCodeFunctions(server: McpServer, rootDir: string | null, mount: string | boolean) {
  // Resolve CLI mount option
  let mountDirDescription: string
  let mountDir: string | null
  if (mount !== false) {
    // Resolve mounted directory
    mountDir = resolveMountDir(typeof mount === 'string' ? mount : '')
    mountDirDescription = `To store files permanently use the directory at: ${mountDir}\n`
  } else {
    mountDir = null
    mountDirDescription = ''
  }

  const toolDescription = `Tool to execute Python code and return stdout, stderr, and return value.

The code may be async, and the value on the last line will be returned as the return value.

The code will be executed with Python 3.12.
${mountDirDescription}
Dependencies may be defined via PEP 723 script metadata, e.g. to install "pydantic", the script should start
with a comment of the form:

# /// script
# dependencies = ['pydantic']
# ///
print('python code here')
`
  let setLogLevel: LoggingLevel = 'emergency'

  server.server.setRequestHandler(SetLevelRequestSchema, (request) => {
    setLogLevel = request.params.level
    return {}
  })

  // Main tool to run code
  server.registerTool(
    'run_python_code',
    {
      title: 'Run Python Code',
      description: toolDescription,
      inputSchema: { python_code: z.string().describe('Python code to run') },
    },
    async ({ python_code }: { python_code: string }) => {
      const logPromises: Promise<void>[] = []
      const result = await runCode(
        [
          {
            name: 'main.py',
            content: python_code,
            active: true,
          },
        ],
        (level, data) => {
          if (LogLevels.indexOf(level) >= LogLevels.indexOf(setLogLevel)) {
            logPromises.push(server.server.sendLoggingMessage({ level, data }))
          }
        },
        rootDir,
        mountDir,
      )
      await Promise.all(logPromises)
      return {
        content: [{ type: 'text', text: asXml(result) }],
      }
    },
  )
}
