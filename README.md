# Owlex

MCP server for running Codex CLI sessions from Claude Code while maintaining context.

## Features

- **Fresh or resumed sessions** - Start a new Codex session with fresh context, or resume from the last session with full conversation history preserved
- **Async task execution** - Tasks run in the background; wait for completion with `wait_for_task` or continue working and check results later with `get_task_result`
- **Working directory support** - Point Codex to any project directory
- **Web search** - Enable Codex web search for up-to-date information

## Installation

```bash
uv tool install /path/to/owlex
```

## Configuration

Add to your `.mcp.json`:

```json
{
  "mcpServers": {
    "owlex": {
      "command": "owlex-server",
      "env": {
        "CODEX_CLEAN_OUTPUT": "true"
      }
    }
  }
}
```

## Tools

### `start_codex_session`

Start a new Codex session (no prior context).

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Question or request to send |
| `working_directory` | No | Working directory for Codex |
| `enable_search` | No | Enable web search |

### `resume_codex_session`

Resume an existing Codex session to ask for advice.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Question or request to send |
| `session_id` | No | Session ID to resume (uses `--last` if omitted) |
| `working_directory` | No | Working directory for Codex |
| `enable_search` | No | Enable web search |

### `wait_for_task`

Wait for a task to complete and return its result.

### `get_task_result`

Get the result of a completed task.

## Environment Variables

- `CODEX_CLEAN_OUTPUT`: Remove echoed prompts from output (default: `true`)
- `CODEX_BYPASS_APPROVALS`: Bypass sandbox mode (default: `false`)
