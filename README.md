# Codex CLI MCP Server

An MCP server that integrates Codex CLI with Claude Code, enabling AI-assisted code reviews and analysis directly from your Claude Code workflow.

## Features

- **AI Code Reviews**: Send code and plans to Codex CLI for expert review
- **Slash Commands**: Quick access via `/review-with-codex` and `/ask-codex`
- **Working Directory Support**: Review entire codebases or specific files
- **Secure by Default**: Sandboxed execution with `--full-auto` mode

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Codex CLI is installed:**
   ```bash
   which codex
   ```

   If not found, install Codex CLI from its documentation.

4. **Configure Claude Code:**

   Add to `~/.claude.json` for your project directory:
   ```json
   {
     "mcpServers": {
       "codex-cli": {
         "type": "stdio",
         "command": "/path/to/codex-mcp/venv/bin/python3",
         "args": ["/path/to/codex-mcp/codex_mcp_server.py"],
         "env": {
           "CODEX_CLEAN_OUTPUT": "true"
         }
       }
     }
   }
   ```

5. **Restart Claude Code**

## Slash Commands

### `/review-with-codex` - Plan and Review Workflow

Creates an implementation plan, sends it to Codex CLI for review, and waits for your approval before implementing.

**Example:**
```
/review-with-codex Add user authentication with OAuth2
```

### `/ask-codex` - Quick Questions

Send direct questions or requests to Codex CLI for immediate feedback.

**Examples:**
```
/ask-codex What are security risks in this code?
/ask-codex Review the API design for best practices
```

## Environment Variables

- **`CODEX_BYPASS_APPROVALS`** (default: `false`): Set to `true` to disable sandboxing (not recommended)
- **`CODEX_CLEAN_OUTPUT`** (default: `true`): Set to `false` to show full Codex output including prompts

## MCP Tools

The server provides two MCP tools that can be used directly:

- **`review_plan`**: Send plans or code for Codex CLI review
  - Supports `working_directory` and `files` parameters for reviewing code in place

- **`execute_codex_command`**: Run custom Codex CLI commands

You can ask Claude Code to use these tools without the slash commands:
- "Can you have Codex CLI review this code?"
- "Use Codex to analyze security risks"

## Troubleshooting

**MCP server not responding:**
```bash
# Check if server runs manually
./venv/bin/python3 codex_mcp_server.py

# Check Claude Code logs
claude config logs
```

**Codex CLI not found:**
Ensure `codex` is in your PATH or update the configuration with the absolute path.
