# Agent Queue

This folder is for overnight or queued Copilot work.

Files:
- `TASKS.md`: queued work items.
- `EVENTS.md`: append-only completion log.

How to use:
1. Add tasks to `TASKS.md` with status `pending`.
2. In chat, send `continue` or `next`.
3. Copilot should pick the first `pending` task, do it, update the task status, and append an event.

Limitation:
Copilot cannot start a new turn without a new prompt from you. The queue can be consumed automatically on the next prompt, but not fully unattended without further prompts.
