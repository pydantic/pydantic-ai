import express, { Request, Response } from 'express'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'

import { server } from './mcpServer.js'

const app = express()
let transport: SSEServerTransport | null = null

app.get('/sse', async (_: Request, res: Response) => {
  transport = new SSEServerTransport('/messages', res)
  await server.connect(transport)
})

app.post('/messages', async (req: Request, res: Response) => {
  // Note: to support multiple simultaneous connections, these messages will
  // need to be routed to a specific matching transport. (This logic isn't
  // implemented here, for simplicity.)
  if (!transport) {
    res.status(400).send('No transport connected')
    return
  }
  await transport.handlePostMessage(req, res)
})

app.listen(3001)
