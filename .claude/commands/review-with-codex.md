---
description: Create a plan and have Codex CLI review it before implementing
---

You are helping the user implement a feature with Codex CLI's guidance.

Follow these steps:

1. **Understand the request**: Analyze what the user wants to implement
2. **Create an implementation plan**: Draft a detailed plan with specific steps
3. **Send to Codex CLI for review**: Use the `mcp__codex-cli__start_review` tool to start background review
4. **Continue working**: Codex runs in the background, so you can continue other tasks
5. **Check status**: Use `mcp__codex-cli__get_task_status` to check progress
6. **Get results**: Use `mcp__codex-cli__get_task_result` when the review is complete
7. **Show both versions**: Present the original plan and Codex's feedback
8. **Create revised plan**: Incorporate Codex's suggestions into an improved plan
9. **Wait for approval**: Ask the user if they want to proceed with implementation
10. **Implement**: Only after user approval, implement the revised plan

**Important**:
- Codex runs asynchronously - you can continue working while it thinks
- **DO NOT poll in a loop** - start the task and return immediately
- Do other work (analyze codebase, answer questions, etc.) for ~30 seconds
- Then check results with `mcp__codex-cli__get_task_result`
- If still running, continue with other work and check again later
- **When to use blocking mode**: If the review is expected to be quick (<60s) or user wants immediate results, you can use `mcp__codex-cli__wait_for_task` to block until complete. This waits directly on the async task (no polling delay) and returns results immediately when done. If timeout occurs, the background task continues running and can be checked later with `get_task_result`.
- **When to use async mode**: For large codebase reviews, complex analysis, or when you want to do other work while waiting, use the async approach with manual checking. This is recommended for tasks that may take longer than 60 seconds.
- Always wait for user approval before implementing
- Show both the original plan and Codex's feedback clearly
- The revised plan should address all of Codex's concerns

**Example flow**:
```
User: Add user authentication with OAuth2

1. I'll create a plan...
2. Starting Codex CLI review in background... (task_id: abc-123)
3. While Codex reviews, let me analyze your current codebase...
   [Do actual work here - read files, understand structure, etc.]
4. [After ~30 seconds] Checking if Codex is done...
5. Codex review complete! Here's the original plan: [...]
6. Codex CLI feedback: [...]
7. Here's the revised plan incorporating feedback: [...]
8. Would you like me to proceed with implementation?
```
