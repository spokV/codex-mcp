---
description: Create a plan and have Codex CLI review it before implementing
---

You are helping the user implement a feature with Codex CLI's guidance.

Follow these steps:

1. **Understand the request**: Analyze what the user wants to implement
2. **Create an implementation plan**: Draft a detailed plan with specific steps
3. **Send to Codex CLI for review**: Use the `mcp__codex-cli__review_plan` tool to get feedback
4. **Show both versions**: Present the original plan and Codex's feedback
5. **Create revised plan**: Incorporate Codex's suggestions into an improved plan
6. **Wait for approval**: Ask the user if they want to proceed with implementation
7. **Implement**: Only after user approval, implement the revised plan

**Important**:
- Always wait for user approval before implementing
- Show both the original plan and Codex's feedback clearly
- The revised plan should address all of Codex's concerns

**Example flow**:
```
User: Add user authentication with OAuth2

1. I'll create a plan...
2. Sending to Codex CLI for review...
3. Here's the original plan: [...]
4. Codex CLI feedback: [...]
5. Here's the revised plan incorporating feedback: [...]
6. Would you like me to proceed with implementation?
```
