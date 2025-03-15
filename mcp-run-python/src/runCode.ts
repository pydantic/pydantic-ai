/* eslint @typescript-eslint/no-explicit-any: off */
import { loadPyodide } from 'pyodide'
import { preparePythonCode } from './prepareEnvCode.js'

export interface CodeFile {
  name: string
  content: string
  active: boolean
}

export async function runCode(files: CodeFile[]): Promise<RunSuccess | RunError> {
  const output: string[] = []
  const pyodide = await loadPyodide({
    stdout: (msg) => {
      output.push(msg)
    },
    stderr: (msg) => {
      output.push(msg)
    },
  })
  await pyodide.loadPackage(['micropip', 'pydantic'], {
    // stop pyodide printing to stdout/stderr
    messageCallback: () => {},
    errorCallback: (msg: string) => {
      output.push(`install error: ${msg}`)
    },
  })
  const sys = pyodide.pyimport('sys')

  const dirPath = '/tmp/mcp_run_python'
  sys.path.append(dirPath)
  const pathlib = pyodide.pyimport('pathlib')
  pathlib.Path(dirPath).mkdir()
  const moduleName = '_prepare_env'

  pathlib.Path(`${dirPath}/${moduleName}.py`).write_text(preparePythonCode)

  const preparePyEnv: PreparePyEnv = pyodide.pyimport(moduleName)
  const prepareStatus = await preparePyEnv.prepare_env(pyodide.toPy(files))
  sys.stdout.flush()
  sys.stderr.flush()
  if (prepareStatus.kind == 'error') {
    return {
      status: 'prepare-error',
      output,
      error: prepareStatus.message,
    }
  }
  const activeFile = files.find((f) => f.active)! || files[0]
  try {
    const rawValue = await pyodide.runPythonAsync(activeFile.content, {
      globals: pyodide.toPy({ __name__: '__main__' }),
      filename: activeFile.name,
    })
    const returnValueJson = preparePyEnv.dump_json(rawValue)
    sys.stdout.flush()
    sys.stderr.flush()
    return {
      status: 'success',
      output,
      returnValueJson,
    }
  } catch (err) {
    sys.stdout.flush()
    sys.stderr.flush()
    return {
      status: 'run-error',
      output,
      error: formatError(err),
    }
  }
}

interface RunSuccess {
  status: 'success'
  // we could record stdout and stderr separately, but I suspect simplicity is more important
  output: string[]
  returnValueJson: string | null
}

interface RunError {
  status: 'prepare-error' | 'install-error' | 'run-error'
  output: string[]
  error: string
}

export function asXml(runResult: RunSuccess | RunError): string {
  const xml = [`<status>${runResult.status}</status>`]
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

function formatError(err: any): string {
  let errStr = err.toString()
  errStr = errStr.replace(/^PythonError: +/, '')
  // remove frames from inside pyodide
  errStr = errStr.replace(/ {2}File "\/lib\/python\d+\.zip\/_pyodide\/.*\n {4}.*\n(?: {4,}\^+\n)?/g, '')
  return errStr
}

interface PrepareSuccess {
  kind: 'success'
  message: string
}
interface PrepareError {
  kind: 'error'
  message: string
}
interface PreparePyEnv {
  prepare_env: (files: CodeFile[]) => Promise<PrepareSuccess | PrepareError>
  dump_json: (value: any) => string | null
}
