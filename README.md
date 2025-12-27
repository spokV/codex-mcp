# Owlex

[![Version](https://img.shields.io/badge/version-0.1.1-blue)](https://github.com/agentic-mcp-tools/owlex/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-purple)](https://modelcontextprotocol.io)

MCP server for running Codex CLI and Gemini CLI sessions from Claude Code while maintaining context.

## Features

- **Multi-AI delegation** - Invoke Codex CLI or Gemini CLI from Claude Code
- **Fresh or resumed sessions** - Start a new session with fresh context, or resume from the last session with full conversation history preserved
- **Async task execution** - Tasks run in the background; wait for completion with `wait_for_task` or continue working and check results later with `get_task_result`
- **Working directory support** - Point Codex/Gemini to any project directory
- **Web search** - Enable Codex web search for up-to-date information

## When to Use Each Provider

| Task Type | Recommended | Reason |
|-----------|-------------|--------|
| Large codebase analysis | Gemini | 1M token context window |
| Code review & bug finding | Codex | Purpose-built for finding critical flaws |
| Complex multi-step implementation | Claude (caller) | Best agentic coding (SWE-bench) |
| PRD & requirements | Codex | Excellent Socratic questioning |
| Multimodal tasks (images, video) | Gemini | State-of-the-art multimodal reasoning |

**Tip:** Configure your preferred models in your Claude Code config (`CLAUDE.md`) for project-specific recommendations.

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

### Codex CLI Tools

#### `start_codex_session`

Start a new Codex session (no prior context).

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Question or request to send |
| `working_directory` | No | Working directory for Codex |
| `enable_search` | No | Enable web search |

#### `resume_codex_session`

Resume an existing Codex session to ask for advice.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Question or request to send |
| `session_id` | No | Session ID to resume (uses `--last` if omitted) |
| `working_directory` | No | Working directory for Codex |
| `enable_search` | No | Enable web search |

### Gemini CLI Tools

#### `start_gemini_session`

Start a new Gemini CLI session (no prior context).

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Question or request to send |
| `working_directory` | No | Working directory for Gemini context |

#### `resume_gemini_session`

Resume an existing Gemini CLI session with full conversation history.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `prompt` | Yes | Question or request to send |
| `session_ref` | No | Session to resume: `latest` (default) or index number |
| `working_directory` | No | Working directory for Gemini context |

### Task Management

#### `wait_for_task`

Wait for a task (Codex or Gemini) to complete and return its result.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `task_id` | Yes | Task ID returned by session tools |
| `timeout` | No | Maximum seconds to wait (default: 300) |

#### `get_task_result`

Get the result of a task (Codex or Gemini) without blocking.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `task_id` | Yes | Task ID returned by session tools |

## Environment Variables

### Codex Settings
- `CODEX_CLEAN_OUTPUT`: Remove echoed prompts from output (default: `true`)
- `CODEX_BYPASS_APPROVALS`: Bypass sandbox mode (default: `false`)

### Gemini Settings
- `GEMINI_YOLO_MODE`: Auto-approve actions via `--approval-mode yolo` (default: `true`)
- `GEMINI_CLEAN_OUTPUT`: Remove noise from output (default: `true`)
