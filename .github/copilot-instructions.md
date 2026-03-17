# Queue Workflow

When the user indicates they want queued autonomous work, follow this workflow.

Files:
- `agent_queue/TASKS.md`: source of queued tasks.
- `agent_queue/EVENTS.md`: append-only event log for completed or blocked work.

Rules:
- If the user's prompt is `continue`, `next`, `继续`, `做下一项`, `work on queue`, or similarly generic, read `agent_queue/TASKS.md` first.
- Select the first task whose status is `pending`.
- Complete exactly one queued task per turn unless the user explicitly asks for more than one.
- After finishing a queued task, update its status in `agent_queue/TASKS.md`.
- Append a short event entry to `agent_queue/EVENTS.md` with timestamp, task id, outcome, and key notes.
- If a queued task is blocked, mark it `blocked` and log the blocker in `agent_queue/EVENTS.md`.
- If the user's prompt specifies a different explicit task, follow the user's prompt instead of the queue.
- Never silently skip queue bookkeeping after finishing queued work.

Status values:
- `pending`
- `in_progress`
- `done`
- `blocked`

Expected task line format in `agent_queue/TASKS.md`:

| id | status | title | prompt |
| --- | --- | --- | --- |
| 1 | pending | Example task | Do something concrete |
