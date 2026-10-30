# Vstorm subagents-pydantic-ai

Task control in `pydantic_ai_harness/subagents/` (`SubAgents(task_control=TaskControl())`, remote agents served by
`SubAgentWorker`, and agents the model creates with `dynamic_agents`) is a port of
https://github.com/vstorm-co/subagents-pydantic-ai: its background tasks, steering, parent questions,
task-control tools, dynamic agents, and typed task messages over a transport. The implementation is
rewritten for Pydantic AI Harness, on `DelegationTasks`.

MIT License

Copyright (c) 2026 Vstorm

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
