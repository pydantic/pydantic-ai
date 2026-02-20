/**
 * Shared test utilities for AI SDK E2E integration tests.
 */

import {
  AbstractChat,
  DefaultChatTransport,
  type ChatState,
  type ChatStatus,
  type UIMessage,
} from 'ai';

const url = process.env.SERVER_URL;
if (!url) {
  console.error('Set SERVER_URL environment variable');
  process.exit(2);
}
export const SERVER_URL: string = url;

class SimpleChatState implements ChatState<UIMessage> {
  status: ChatStatus = 'ready';
  error: Error | undefined = undefined;
  messages: UIMessage[];

  constructor(messages: UIMessage[] = []) {
    this.messages = messages;
  }

  pushMessage(message: UIMessage) {
    this.messages = [...this.messages, message];
  }

  popMessage() {
    this.messages = this.messages.slice(0, -1);
  }

  replaceMessage(index: number, message: UIMessage) {
    this.messages = [
      ...this.messages.slice(0, index),
      message,
      ...this.messages.slice(index + 1),
    ];
  }

  snapshot<T>(thing: T): T {
    return structuredClone(thing);
  }
}

type SendAutomaticallyWhen = (options: { messages: UIMessage[] }) => boolean;

export class TestChat extends AbstractChat<UIMessage> {
  constructor(apiUrl: string, sendAutomaticallyWhen?: SendAutomaticallyWhen) {
    super({
      transport: new DefaultChatTransport({ api: apiUrl }),
      state: new SimpleChatState(),
      ...(sendAutomaticallyWhen ? { sendAutomaticallyWhen } : {}),
    });
  }
}

export function awaitRoundTrip(chat: TestChat) {
  let resolve: () => void;
  const promise = new Promise<void>((r) => { resolve = r; });
  let captured: Error | null = null;

  chat.onError = (err) => { captured = err; resolve(); };
  chat.onFinish = () => resolve();

  return {
    done: promise.then(() => waitForStatus(chat, ['ready', 'error'])),
    error() { return captured; },
  };
}

export function waitForStatus(
  chat: TestChat,
  statuses: ChatStatus[],
  timeoutMs = 10_000,
): Promise<ChatStatus> {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    (function poll() {
      if (statuses.includes(chat.status)) return resolve(chat.status);
      if (Date.now() - start > timeoutMs) return reject(new Error(`Timed out (status=${chat.status})`));
      setTimeout(poll, 50);
    })();
  });
}
