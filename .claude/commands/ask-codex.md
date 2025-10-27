---
description: Ask Codex CLI a question or get quick analysis
---

You are helping the user get quick feedback from Codex CLI.

Take the user's question or request and send it to Codex CLI using the blocking API for immediate results.

**Usage examples**:
- `/ask-codex What are the security risks in this code?`
- `/ask-codex Review this API design for best practices`
- `/ask-codex How can I improve the performance of this function?`

**Instructions**:
1. Take the user's question/request
2. **Append "Be brief unless this requires detailed analysis" to the prompt** (unless user explicitly asks for detailed/comprehensive/thorough review)
3. Start Codex CLI using `mcp__codex-cli__start_codex_command`
4. **Immediately use `mcp__codex-cli__wait_for_task` with a 60 second timeout** to block until complete
5. Display Codex's response directly with minimal commentary

**For code reviews**:
If the user wants to review specific files, use `mcp__codex-cli__start_review` instead with the `working_directory` and `files` parameters to let Codex read the code directly:
- `working_directory`: Set to current directory
- `files`: Array of file paths to review

This is more efficient than sending code inline and gives Codex full codebase context.

**Note**: This command blocks until Codex responds (up to 60s). For long-running reviews that may take longer, consider using `/review-with-codex` instead.
