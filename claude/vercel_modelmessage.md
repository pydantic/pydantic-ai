# FILENAME: git_clones/ai/packages/provider-utils/src/types/model-message.ts

import { AssistantModelMessage } from './assistant-model-message';
import { SystemModelMessage } from './system-model-message';
import { ToolModelMessage } from './tool-model-message';
import { UserModelMessage } from './user-model-message';

/**
A message that can be used in the `messages` field of a prompt.
It can be a user message, an assistant message, or a tool message.
 */
export type ModelMessage =
  | SystemModelMessage
  | UserModelMessage
  | AssistantModelMessage
  | ToolModelMessage;
[ERROR: can't do nonzero end-relative seeks]

# FILENAME: git_clones/ai/packages/provider-utils/src/types/system-model-message.ts

import { ProviderOptions } from './provider-options';

/**
 A system message. It can contain system information.

 Note: using the "system" part of the prompt is strongly preferred
 to increase the resilience against prompt injection attacks,
 and because not all providers support several system messages.
 */
export type SystemModelMessage = {
  role: 'system';
  content: string;

  /**
    Additional provider-specific metadata. They are passed through
    to the provider from the AI SDK and enable provider-specific
    functionality that can be fully encapsulated in the provider.
     */
  providerOptions?: ProviderOptions;
};
[ERROR: can't do nonzero end-relative seeks]

# FILENAME: git_clones/ai/packages/provider-utils/src/types/user-model-message.ts

import { FilePart, ImagePart, TextPart } from './content-part';
import { ProviderOptions } from './provider-options';

/**
A user message. It can contain text or a combination of text and images.
 */
export type UserModelMessage = {
  role: 'user';
  content: UserContent;

  /**
    Additional provider-specific metadata. They are passed through
    to the provider from the AI SDK and enable provider-specific
    functionality that can be fully encapsulated in the provider.
     */
  providerOptions?: ProviderOptions;
};

/**
  Content of a user message. It can be a string or an array of text and image parts.
   */
export type UserContent = string | Array<TextPart | ImagePart | FilePart>;
[ERROR: can't do nonzero end-relative seeks]

# FILENAME: git_clones/ai/packages/provider-utils/src/types/assistant-model-message.ts

import {
  FilePart,
  ReasoningPart,
  TextPart,
  ToolCallPart,
  ToolResultPart,
} from './content-part';
import { ProviderOptions } from './provider-options';

/**
An assistant message. It can contain text, tool calls, or a combination of text and tool calls.
 */
export type AssistantModelMessage = {
  role: 'assistant';
  content: AssistantContent;

  /**
  Additional provider-specific metadata. They are passed through
  to the provider from the AI SDK and enable provider-specific
  functionality that can be fully encapsulated in the provider.
   */
  providerOptions?: ProviderOptions;
};

/**
Content of an assistant message.
It can be a string or an array of text, image, reasoning, redacted reasoning, and tool call parts.
 */
export type AssistantContent =
  | string
  | Array<TextPart | FilePart | ReasoningPart | ToolCallPart | ToolResultPart>;
[ERROR: can't do nonzero end-relative seeks]

# FILENAME: git_clones/ai/packages/provider-utils/src/types/tool-model-message.ts

import { ToolResultPart } from './content-part';
import { ProviderOptions } from './provider-options';

/**
A tool message. It contains the result of one or more tool calls.
 */
export type ToolModelMessage = {
  role: 'tool';
  content: ToolContent;

  /**
  Additional provider-specific metadata. They are passed through
  to the provider from the AI SDK and enable provider-specific
  functionality that can be fully encapsulated in the provider.
   */
  providerOptions?: ProviderOptions;
};

/**
Content of a tool message. It is an array of tool result parts.
 */
export type ToolContent = Array<ToolResultPart>;
[ERROR: can't do nonzero end-relative seeks]

# FILENAME: git_clones/ai/packages/provider-utils/src/types/content-part.ts

import { LanguageModelV2ToolResultOutput } from '@ai-sdk/provider';
import { ProviderOptions } from './provider-options';
import { DataContent } from './data-content';

/**
Text content part of a prompt. It contains a string of text.
 */
export interface TextPart {
  type: 'text';

  /**
The text content.
   */
  text: string;

  /**
Additional provider-specific metadata. They are passed through
to the provider from the AI SDK and enable provider-specific
functionality that can be fully encapsulated in the provider.
 */
  providerOptions?: ProviderOptions;
}

/**
Image content part of a prompt. It contains an image.
 */
export interface ImagePart {
  type: 'image';

  /**
Image data. Can either be:

- data: a base64-encoded string, a Uint8Array, an ArrayBuffer, or a Buffer
- URL: a URL that points to the image
   */
  image: DataContent | URL;

  /**
Optional IANA media type of the image.

@see https://www.iana.org/assignments/media-types/media-types.xhtml
   */
  mediaType?: string;

  /**
Additional provider-specific metadata. They are passed through
to the provider from the AI SDK and enable provider-specific
functionality that can be fully encapsulated in the provider.
 */
  providerOptions?: ProviderOptions;
}

/**
File content part of a prompt. It contains a file.
 */
export interface FilePart {
  type: 'file';

  /**
File data. Can either be:

- data: a base64-encoded string, a Uint8Array, an ArrayBuffer, or a Buffer
- URL: a URL that points to the image
   */
  data: DataContent | URL;

  /**
Optional filename of the file.
   */
  filename?: string;

  /**
IANA media type of the file.

@see https://www.iana.org/assignments/media-types/media-types.xhtml
   */
  mediaType: string;

  /**
Additional provider-specific metadata. They are passed through
to the provider from the AI SDK and enable provider-specific
functionality that can be fully encapsulated in the provider.
 */
  providerOptions?: ProviderOptions;
}

/**
 * Reasoning content part of a prompt. It contains a reasoning.
 */
export interface ReasoningPart {
  type: 'reasoning';

  /**
The reasoning text.
   */
  text: string;

  /**
Additional provider-specific metadata. They are passed through
to the provider from the AI SDK and enable provider-specific
functionality that can be fully encapsulated in the provider.
 */
  providerOptions?: ProviderOptions;
}

/**
Tool call content part of a prompt. It contains a tool call (usually generated by the AI model).
 */
export interface ToolCallPart {
  type: 'tool-call';

  /**
ID of the tool call. This ID is used to match the tool call with the tool result.
 */
  toolCallId: string;

  /**
Name of the tool that is being called.
 */
  toolName: string;

  /**
Arguments of the tool call. This is a JSON-serializable object that matches the tool's input schema.
   */
  input: unknown;

  /**
Additional provider-specific metadata. They are passed through
to the provider from the AI SDK and enable provider-specific
functionality that can be fully encapsulated in the provider.
 */
  providerOptions?: ProviderOptions;

  /**
Whether the tool call was executed by the provider.
 */
  providerExecuted?: boolean;
}

/**
Tool result content part of a prompt. It contains the result of the tool call with the matching ID.
 */
export interface ToolResultPart {
  type: 'tool-result';

  /**
ID of the tool call that this result is associated with.
 */
  toolCallId: string;

  /**
Name of the tool that generated this result.
  */
  toolName: string;

  /**
Result of the tool call. This is a JSON-serializable object.
   */
  output: LanguageModelV2ToolResultOutput;

  /**
Additional provider-specific metadata. They are passed through
to the provider from the AI SDK and enable provider-specific
functionality that can be fully encapsulated in the provider.
 */
  providerOptions?: ProviderOptions;
}
[ERROR: can't do nonzero end-relative seeks]
