import express, { Request, Response } from 'express'
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js'

import { server } from './mcpServer.js'

const app = express()
const transports: { [sessionId: string]: SSEServerTransport } = {}

app.get('/sse', async (_: Request, res: Response) => {
  const transport = new SSEServerTransport('/messages', res)
  transports[transport.sessionId] = transport
  res.on('close', () => {
    delete transports[transport.sessionId]
  })
  await server.connect(transport)
})

app.post('/messages', async (req: Request, res: Response) => {
  const sessionId = req.query.sessionId as string
  const transport = transports[sessionId]
  if (transport) {
    await transport.handlePostMessage(req, res)
  } else {
    res.status(400).send('No transport found for sessionId')
  }
})

app.listen(3001)
