import { loadPyodide } from 'pyodide'
import { preparePythonCode } from './prepareEnvCode.js'

export interface CodeFile {
  name: string
  content: string
  active: boolean
}

interface RunSuccess {
  status: 'success'
  // we could record stdout and stderr separately, but I suspect simplicity is more important
  output: string
  returnValue: any
  pythonVersion: string
}

interface RunError {
  status: 'prepare-error' | 'run-error'
  output: string
  error: string
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
  await pyodide.loadPackage(['micropip', 'pydantic'])
  const sys = pyodide.pyimport('sys')
  const pythonVersion = `${sys.version_info.major}.${sys.version_info.minor}.${sys.version_info.micro}`

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
      output: output.join(''),
      error: prepareStatus.message,
    }
  }
  const activeFile = files.find((f) => f.active)! || files[0]
  try {
    const rawValue = await pyodide.runPythonAsync(activeFile.content, {
      globals: pyodide.toPy({ __name__: '__main__' }),
      filename: activeFile.name,
    })
    const returnValue = JSON.parse(preparePyEnv.dump_json(rawValue))
    sys.stdout.flush()
    sys.stderr.flush()
    return {
      status: 'success',
      output: output.join(''),
      returnValue,
      pythonVersion,
    }
  } catch (err) {
    sys.stdout.flush()
    sys.stderr.flush()
    return {
      status: 'run-error',
      output: output.join(''),
      error: formatError(err),
    }
  }
}

function formatError(err: any): string {
  let errStr = (err as any).toString()
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
  prepare_env: (files: any) => Promise<PrepareSuccess | PrepareError>
  dump_json: (value: any) => string
}
