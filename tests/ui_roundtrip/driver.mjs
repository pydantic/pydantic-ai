// Drives the real JavaScript UI clients for the round-trip tests in this directory.
//
// Requests and responses are JSON lines on stdin and stdout: `{"id", "op", ...}` in, `{"id", "result"}` or
// `{"id", "error"}` out. Each conversation is a session that holds the client's state, as a browser tab would.
// A turn takes two requests, because the server's reply depends on what the client sent:
//
// - `<protocol>.submit` adds a user message and returns the request body the client sends to the server.
//   Passing `messages` first replaces the client's state, the way an app hydrates it from storage.
// - `<protocol>.receive` feeds the server's stream to the pending request and returns the client's messages.
//
// Vercel AI goes through the AI SDK's `AbstractChat` (what `useChat` wraps), with a transport that hands the
// request to the test and streams back what the test receives. AG-UI goes through `AbstractAgent.runAgent()` of
// `@ag-ui/client` (what `HttpAgent` extends), with `run()` replaced the same way.
import { createInterface } from 'node:readline';

import { AbstractAgent } from '@ag-ui/client';
import { AbstractChat } from 'ai';
import { Subject } from 'rxjs';

// stdout carries the protocol, so library logging goes to stderr.
console.log = console.error;
console.info = console.error;
console.debug = console.error;

class ChatState {
  constructor(messages) {
    this.messages = messages;
    this.status = 'ready';
    this.error = undefined;
  }

  pushMessage = (message) => {
    this.messages = this.messages.concat(message);
  };

  popMessage = () => {
    this.messages = this.messages.slice(0, -1);
  };

  replaceMessage = (index, message) => {
    this.messages = [...this.messages.slice(0, index), message, ...this.messages.slice(index + 1)];
  };

  snapshot = (value) => structuredClone(value);
}

class Chat extends AbstractChat {
  constructor({ messages, ...init }) {
    super({ ...init, state: new ChatState(messages) });
  }
}

/** A promise with its `resolve`, for a value that arrives later. */
function deferred() {
  let resolve;
  const promise = new Promise((r) => (resolve = r));
  return { promise, resolve };
}

/** The request the client sent, or the error it finished with before sending one. */
async function sentOrFailed(sent, done) {
  const finished = done.then((error) => {
    throw new Error(`the client finished without sending a request: ${error}`);
  });
  return await Promise.race([sent, finished]);
}

class VercelSession {
  constructor(chatId) {
    this.chatId = chatId;
    // One counter per session, not per chat, so a client hydrated again keeps minting fresh ids.
    this.counter = 0;
    this.chat = this.newChat([]);
  }

  newChat(messages) {
    return new Chat({
      id: this.chatId,
      messages,
      // Deterministic ids, so failures reproduce.
      generateId: () => `${this.chatId}-client-${++this.counter}`,
      transport: {
        sendMessages: async ({ trigger, chatId, messageId, messages }) => {
          let controller;
          const stream = new ReadableStream({ start: (c) => (controller = c) });
          // The body `DefaultChatTransport` posts.
          this.turn.sent.resolve({ trigger, id: chatId, messageId, messages: structuredClone(messages) });
          this.turn.controller = controller;
          return stream;
        },
        reconnectToStream: async () => null,
      },
    });
  }

  async submit({ messages, message }) {
    if (messages !== undefined) {
      this.chat = this.newChat(messages);
    }
    this.turn = { sent: deferred() };
    // `AbstractChat` keeps a failed request's error rather than rejecting.
    this.turn.done = this.chat.sendMessage(message).then(() => this.chat.error);
    return await sentOrFailed(this.turn.sent.promise, this.turn.done);
  }

  async receive({ chunks }) {
    for (const chunk of chunks) {
      this.turn.controller.enqueue(chunk);
    }
    this.turn.controller.close();
    const error = await this.turn.done;
    return { error: error && String(error), messages: structuredClone(this.chat.messages) };
  }
}

class ReplayAgent extends AbstractAgent {
  run(input) {
    this.turn.sent.resolve(structuredClone(input));
    return this.turn.events.asObservable();
  }
}

class AGUISession {
  constructor(threadId) {
    this.threadId = threadId;
    this.agent = new ReplayAgent({ threadId });
  }

  async submit({ messages, message }) {
    if (messages !== undefined) {
      this.agent = new ReplayAgent({ threadId: this.threadId, initialMessages: messages });
    }
    this.agent.addMessage(message);
    this.agent.turn = { sent: deferred(), events: new Subject() };
    const { turn } = this.agent;
    turn.done = this.agent.runAgent().then(
      () => undefined,
      (error) => error,
    );
    return await sentOrFailed(turn.sent.promise, turn.done);
  }

  async receive({ events }) {
    const { turn } = this.agent;
    for (const event of events) {
      turn.events.next(event);
    }
    turn.events.complete();
    const error = await turn.done;
    return { error: error && String(error), messages: structuredClone(this.agent.messages) };
  }
}

const sessions = new Map();

function session(protocol, id) {
  const key = `${protocol}:${id}`;
  if (!sessions.has(key)) {
    sessions.set(key, protocol === 'vercel' ? new VercelSession(id) : new AGUISession(id));
  }
  return sessions.get(key);
}

async function handle({ op, session: id, ...request }) {
  const [protocol, action] = op.split('.');
  if (!['vercel', 'agui'].includes(protocol) || !['submit', 'receive'].includes(action)) {
    throw new Error(`unknown op ${op}`);
  }
  return await session(protocol, id)[action](request);
}

// Requests are handled one at a time, in order: the test waits for each response before sending the next.
const lines = createInterface({ input: process.stdin });
for await (const line of lines) {
  const { id, ...request } = JSON.parse(line);
  let response;
  try {
    response = { id, result: await handle(request) };
  } catch (error) {
    response = { id, error: error instanceof Error ? (error.stack ?? error.message) : String(error) };
  }
  process.stdout.write(`${JSON.stringify(response)}\n`);
}
