import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js'
import { z } from 'zod'

import { runCode } from './runCode.js'

export const server = new McpServer({
  name: 'Run Python',
  version: '0.0.1',
})

interface ToolParams {
  python_code: string
}

const toolDescription = `Tool to execute Python code and return stdout, stderr, and return value.

The code may be async, and the value on the last line will be returned as the return value.

Dependencies may be defined via PEP 723 script metadata, e.g. to install "pydantic", the script should start
with a comment of the form:

# /// script
# dependencies = ['pydantic']
# ///
`

server.tool(
  'run_python_code',
  toolDescription,
  { python_code: z.string().describe('Python code to run') },
  async ({ python_code }: ToolParams) => {
    const result = await runCode([{ name: 'main.py', content: python_code, active: true }])
    return {
      content: [{ type: 'text', text: JSON.stringify(result, null, 2) }],
    }
  },
)
