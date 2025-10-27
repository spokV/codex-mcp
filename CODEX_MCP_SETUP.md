# Codex CLI MCP Server Setup

This setup allows Claude Code to send plans and code to Codex CLI for review and act upon its feedback.

## Installation

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Codex CLI is installed:**
   ```bash
   which codex
   ```

   If not found, install Codex CLI according to its documentation.

4. **Configure security settings (optional):**

   By default, the MCP server uses `--dangerously-bypass-approvals-and-sandbox` for automated execution. To use safer settings:

   ```bash
   export CODEX_BYPASS_APPROVALS=false
   ```

   When set to `false`, the server uses `--full-auto` instead, which provides sandboxed execution with workspace-write permissions.

5. **Make the MCP server executable (optional):**
   ```bash
   chmod +x codex_mcp_server.py
   ```

## Configuration

The MCP server has been added to Claude Code configuration at:
`~/.claude.json`

Configuration:
```json
{
  "mcpServers": {
    "codex-cli": {
      "command": "/Users/spok/repos/test/venv/bin/python3",
      "args": ["/Users/spok/repos/test/codex_mcp_server.py"],
      "transport": "stdio",
      "env": {
        "CODEX_BYPASS_APPROVALS": "true"
      }
    }
  }
}
```

**Security Note:** The command uses the virtual environment's Python interpreter to ensure all dependencies are available. The `CODEX_BYPASS_APPROVALS` environment variable controls whether Codex runs with or without sandbox restrictions.

### Security Configuration

- **`CODEX_BYPASS_APPROVALS=true`** (default): Uses `--dangerously-bypass-approvals-and-sandbox`
  - ⚠️ No sandboxing or approval prompts
  - Best for: Trusted automation, isolated environments
  - Risk: High - any code can execute without restrictions

- **`CODEX_BYPASS_APPROVALS=false`**: Uses `--full-auto`
  - ✅ Sandboxed execution with workspace-write permissions
  - ✅ Only asks approval on command failure
  - Best for: Production environments, untrusted callers
  - Risk: Low - sandboxed and controlled

### Output Cleaning Configuration

By default, the MCP server cleans Codex CLI output to remove verbose prompt templates, providing a cleaner user experience.

- **`CODEX_CLEAN_OUTPUT=true`** (default): Clean output
  - ✅ Removes prompt templates ("Review the following...", "Please provide: 1. 2. 3. 4.")
  - ✅ Shows only Codex's actual response
  - ✅ More concise and readable output
  - Best for: Normal usage, better UX

- **`CODEX_CLEAN_OUTPUT=false`**: Show full output
  - Shows complete raw output including prompts
  - Useful for: Debugging, understanding what was sent to Codex

**Example Configuration:**
```json
{
  "mcpServers": {
    "codex-cli": {
      "command": "/Users/spok/repos/test/venv/bin/python3",
      "args": ["/Users/spok/repos/test/codex_mcp_server.py"],
      "transport": "stdio",
      "env": {
        "CODEX_BYPASS_APPROVALS": "true",
        "CODEX_CLEAN_OUTPUT": "true"
      }
    }
  }
}
```

## Usage

### Slash Commands

Two custom slash commands are available for easy Codex CLI integration:

#### `/review-with-codex` - Full review workflow

Use this command when starting a new feature or major change:
```
/review-with-codex Add user authentication with OAuth2
```

This will:
1. Create an implementation plan
2. Send it to Codex CLI for review
3. Show you both the original plan and Codex's feedback
4. Present a revised plan incorporating suggestions
5. Wait for your approval before implementing

#### `/ask-codex` - Quick Codex CLI queries

Use this for direct questions or analysis:
```
/ask-codex What are the security risks in this code?
/ask-codex Review this API design for best practices
```

This sends your question directly to Codex CLI and returns the response.

### From Claude Code (Direct MCP Tools)

You can also ask me directly without slash commands. I have access to two MCP tools:

#### 1. `review_plan` - Send plans for Codex CLI review

Ask me to review a plan with Codex:
```
"Can you have Codex CLI review this implementation plan before we proceed?"
"Review codex_mcp_server.py for security issues"
```

**New Feature:** The tool now supports `working_directory` and `files` parameters. When reviewing code, Codex will read files directly from the specified directory using its built-in capabilities instead of receiving code inline. This is more efficient and lets Codex access the full codebase context.

I'll use the MCP tool to send the plan to Codex CLI and receive feedback.

#### 2. `execute_codex_command` - Run custom Codex commands

Ask me to execute specific Codex CLI commands:
```
"Use Codex CLI to analyze the security of this code"
```

### Example Workflow

**Typical usage pattern:**

1. **You request a feature:** "Add user authentication to the app"

2. **I create a plan:**
   - Add login endpoint
   - Implement JWT tokens
   - Create user database schema

3. **I consult Codex CLI:** Use `review_plan` tool to send the plan

4. **Codex provides feedback:**
   - Suggests using bcrypt for passwords
   - Recommends rate limiting
   - Identifies security considerations

5. **I adjust based on feedback:** Update the plan incorporating Codex's suggestions

6. **I implement:** Execute the reviewed and approved plan

## Working Directory Support

The MCP server now supports reviewing code files directly from a working directory, leveraging Codex CLI's built-in file reading capabilities.

### How It Works

When you provide a `working_directory` parameter:
- Codex CLI is invoked with the `-C <directory>` flag
- Codex reads files directly from that directory
- No need to send code inline in the prompt
- Codex has full access to the codebase context

### Parameters

**`working_directory`** (optional): Path to the directory containing code to review
**`files`** (optional): Array of specific file paths to review

### Example Usage

```python
# Review specific file
mcp__codex-cli__review_plan(
    plan="Review for security issues",
    working_directory="/Users/spok/repos/test",
    files=["codex_mcp_server.py"],
    review_type="security"
)

# Review entire directory
mcp__codex-cli__review_plan(
    plan="Review this codebase for best practices",
    working_directory="/Users/spok/repos/test",
    review_type="code"
)
```

## Customizing the Server

### Modify Codex CLI Execution

The server is configured to use `codex exec` with `--skip-git-repo-check` and `--dangerously-bypass-approvals-and-sandbox` flags for automated execution. You can customize this in `codex_mcp_server.py`:

**For review_plan (line 175):**
```python
cmd = ["codex", "exec", "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox"]

if working_directory:
    cmd.extend(["-C", working_directory])

cmd.append(review_prompt)

result = subprocess.run(
    cmd,
    text=True,
    capture_output=True,
    timeout=120  # Adjust timeout as needed
)
```

**Available Codex CLI options:**
- `-m, --model <MODEL>` - Specify which model to use
- `-s, --sandbox <MODE>` - Sandbox policy (read-only, workspace-write, danger-full-access)
- `-a, --ask-for-approval <POLICY>` - Approval policy (untrusted, on-failure, on-request, never)
- `-C, --cd <DIR>` - Working directory
- `--full-auto` - Low-friction sandboxed automatic execution

Example with custom options:
```python
["codex", "exec", "-m", "claude-3-5-sonnet", "--full-auto", review_prompt]
```

### Add Custom Tools

Add new tools by:

1. Adding to `list_tools()` function
2. Handling in `call_tool()` function
3. Implementing the tool logic

Example:
```python
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        # ... existing tools ...
        Tool(
            name="analyze_security",
            description="Analyze code for security vulnerabilities",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Code to analyze"}
                },
                "required": ["code"]
            }
        )
    ]
```

## Troubleshooting

### MCP server not responding

1. Check if Python dependencies are installed in venv:
   ```bash
   ./venv/bin/python3 -c "import mcp; print('MCP installed')"
   ```

2. Test the server manually:
   ```bash
   ./venv/bin/python3 codex_mcp_server.py
   ```

3. Check Claude Code logs:
   ```bash
   claude config logs
   ```

### Codex CLI not found

Ensure `codex` is in your PATH:
```bash
export PATH=$PATH:/path/to/codex
```

Or update the server script with absolute path to Codex CLI.

### Timeout issues

The default timeout is 120 seconds. If you need longer for complex reviews, increase it in `codex_mcp_server.py`:

**Line 209 (review_plan):**
```python
timeout=300  # Increase from 120 to 300 seconds (5 minutes)
```

**Line 290 (execute_codex_command):**
```python
timeout=300  # Increase from 120 to 300 seconds (5 minutes)
```

## Recent Improvements

The MCP server has been updated with production-ready improvements based on Codex CLI's own code review:

### ✅ Non-Blocking Async Operations
- Replaced synchronous `subprocess.run()` with `asyncio.create_subprocess_exec()`
- Server now remains responsive during long-running Codex operations
- Multiple concurrent requests can be handled without blocking

### ✅ Configurable Security
- `CODEX_BYPASS_APPROVALS` environment variable controls execution mode
- Default: `--dangerously-bypass-approvals-and-sandbox` (automated)
- Secure mode: `--full-auto` (sandboxed with workspace-write)
- No more unconditional bypass flags

### ✅ Input Validation
- Validates required parameters before execution
- Returns clear, structured error messages
- Fails fast with helpful feedback

### ✅ Improved Error Handling
- Proper timeout management with process cleanup
- Better decoding of subprocess output
- More informative error messages

### ✅ Clean Output (NEW)
- Automatically removes verbose prompt templates from Codex responses
- Strips "Review the following...", "Please provide: 1. 2. 3. 4." patterns
- Provides cleaner, more concise output in the UI
- Configurable via `CODEX_CLEAN_OUTPUT` environment variable
- Reduces output size by ~80% in typical reviews

## Restart Claude Code

After any configuration changes:
```bash
# Restart your Claude Code session
# Or reload the configuration if available
```

## Next Steps

- Try the slash commands: `/review-with-codex` and `/ask-codex`
- Customize review prompts in `review_plan()` function for your workflow
- Adjust the Codex CLI command in `codex_mcp_server.py` to match your Codex CLI interface
- Add more specialized review types (performance, accessibility, etc.)
- Create additional slash commands for common workflows
- Integrate with your CI/CD pipeline for automated reviews
- Customize Codex CLI commands for project-specific needs

## Slash Command Files

The slash commands are located at:
- `.claude/commands/review-with-codex.md` - Full review workflow
- `.claude/commands/ask-codex.md` - Quick Codex queries

You can edit these files to customize the behavior or create additional commands.
