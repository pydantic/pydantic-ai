# Agent2Agent (A2A) Protocol

!!! warning
    This page is WIP.

The Agent2Agent (A2A) Protocol is an open standard introduced by Google
that enables communication and interoperability between AI agents, regardless
of the framework or vendor they are built on. This protocol addresses one of
the biggest challenges in enterprise AI adoption: getting agents built on
different platforms to work together seamlessly.

## What is A2A?

A2A provides a common language for AI agents to communicate with each other,
allowing them to discover capabilities, negotiate interaction modes, and
securely work together. The protocol facilitates:

- **Agent Discovery**: Agents can advertise their capabilities through "Agent Cards" (JSON metadata files) that describe what they can do
- **Task Management**: A standardized way to initiate, track, and complete tasks between agents
- **Secure Collaboration**: Enterprise-grade authentication and authorization mechanisms
- **Rich Communication**: Support for multimodal content exchange (text, structured data, files)
- **Real-time Updates**: Streaming capabilities for long-running tasks

## How A2A Works

At its core, A2A is built on existing web standards (HTTP, JSON-RPC,
Server-Sent Events) and follows a client-server model where:

1. A client agent formulates a task request
2. A remote agent processes the request and sends back results
3. Both agents exchange messages and artifacts through a well-defined protocol

The central unit of work in A2A is a **Task**, which progresses through various
states (submitted, working, input-required, completed, failed, canceled).
Tasks contain messages and can produce artifacts as results.
