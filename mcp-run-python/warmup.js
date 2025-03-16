import { loadPyodide } from 'pyodide'

// this runs pyodide to download packages which can otherwise interrupt tests running
const pyodide = await loadPyodide({
  stdout: (msg) => {
    console.log('stdout:', msg)
  },
  stderr: (msg) => {
    console.error('stderr:', msg)
  },
})
await pyodide.loadPackage(['micropip', 'pygments', 'pydantic', 'numpy'])
await pyodide.runPythonAsync(`print('hello world')`)
console.log('\nwarmup successful 🎉')
