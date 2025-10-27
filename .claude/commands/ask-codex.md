---
description: Ask Codex CLI a question or get quick analysis
---

You are helping the user get quick feedback from Codex CLI.

Take the user's question or request and send it directly to Codex CLI using the `mcp__codex-cli__review_plan` tool.

**Usage examples**:
- `/ask-codex What are the security risks in this code?`
- `/ask-codex Review this API design for best practices`
- `/ask-codex How can I improve the performance of this function?`

**Instructions**:
1. Take the user's question/request
2. Send it to Codex CLI using the review_plan tool
3. Return Codex's response to the user

**For code reviews**:
If the user wants to review specific files, use the `working_directory` and `files` parameters to let Codex read the code directly:
- `working_directory`: Set to current directory
- `files`: Array of file paths to review

This is more efficient than sending code inline and gives Codex full codebase context.
