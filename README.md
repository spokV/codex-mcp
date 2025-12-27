# Owlex

MCP server for multi-AI provider integration with usage analytics. Access Codex CLI, Gemini CLI, and 500+ API models (via OpenRouter, Kimi, MiniMax, etc.) from Claude Code.

## Features

- **Multi-AI delegation** - Invoke any AI provider from Claude Code
- **Plugin system** - Add new API providers via simple config
- **Usage tracking** - SQLite database tracks tokens, cost, duration
- **Quality ratings** - Claude auto-rates responses for comparison
- **Monthly statistics** - View provider performance over time
- **Statusline integration** - Show available providers in Claude Code

## When to Use Each Provider

| Task Type | Recommended | Reason |
|-----------|-------------|--------|
| Large codebase analysis | Gemini | 1M token context window |
| Code review & bug finding | Codex | Purpose-built for finding critical flaws |
| Complex multi-step implementation | Claude (caller) | Best agentic coding (SWE-bench) |
| PRD & requirements | Codex | Excellent Socratic questioning |
| Multimodal tasks (images, video) | Gemini | State-of-the-art multimodal reasoning |
| Cost-sensitive tasks | OpenRouter | Route to cheapest provider with `:floor` |

## Installation

```bash
uv tool install /path/to/owlex
```

## Configuration

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "owlex": {
      "command": "owlex-server",
      "env": {
        "CODEX_CLEAN_OUTPUT": "true",
        "GEMINI_YOLO_MODE": "true"
      }
    }
  }
}
```

### Adding API Providers

Create `~/.owlex/providers.json`:

```json
{
  "kimi": {
    "type": "openai_api",
    "base_url": "https://api.moonshot.ai/v1",
    "api_key_env": "KIMI_API_KEY",
    "default_model": "kimi-k2-turbo-preview",
    "cost_per_1k_input": 0.0001,
    "cost_per_1k_output": 0.0003
  },
  "minimax": {
    "type": "openai_api",
    "base_url": "https://api.minimax.io/v1",
    "api_key_env": "MINIMAX_API_KEY",
    "default_model": "minimax-m2"
  },
  "openrouter": {
    "type": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "default_model": "anthropic/claude-sonnet-4",
    "site_url": "https://github.com/agentic-mcp-tools/owlex",
    "app_name": "owlex"
  }
}
```

Or use the `add_provider` tool to add providers programmatically.

## Tools

### Unified Provider Interface

#### `list_providers`

List all available AI providers with status.

#### `call_provider`

Call any configured AI provider.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `provider` | Yes | Provider name (codex, gemini, kimi, openrouter, etc.) |
| `prompt` | Yes | The prompt to send |
| `model` | No | Model override (uses default if not specified) |
| `working_directory` | No | Working directory context |

#### `rate_response`

Rate a provider response for quality tracking.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `task_id` | Yes | Task ID to rate |
| `helpfulness` | Yes | Rating 1-5 |
| `accuracy` | Yes | Rating 1-5 |
| `completeness` | Yes | Rating 1-5 |
| `notes` | No | Optional notes |

#### `get_provider_stats`

Get usage statistics for providers.

| Parameter | Required | Description |
|-----------|----------|-------------|
| `period` | No | `this_month`, `last_month`, `all_time`, or `YYYY-MM` |
| `provider` | No | Filter by specific provider |

#### `add_provider`

Add a new API provider via MCP tool.

### Legacy CLI Tools (Backward Compatible)

#### `start_codex_session` / `resume_codex_session`

Direct Codex CLI integration (unchanged from v0.2).

#### `start_gemini_session` / `resume_gemini_session`

Direct Gemini CLI integration (unchanged from v0.2).

### Task Management

#### `wait_for_task`

Wait for a task to complete and return its result.

#### `get_task_result`

Get the result of a task without blocking.

## Usage Workflow

```
1. Claude: call_provider(provider="kimi", prompt="Explain async in Python")
   → Returns task_id

2. Claude: wait_for_task(task_id)
   → Returns Kimi's response, usage auto-tracked

3. Claude: rate_response(task_id, helpfulness=4, accuracy=5, completeness=4)
   → Rating saved

4. Claude: get_provider_stats(period="this_month")
   → Shows: kimi (10 calls, 50k tokens, 4.3 avg rating, $0.05)
```

## OpenRouter Special Features

OpenRouter provides access to 500+ models through one API:

- **Model routing**: Use `:nitro` suffix for fastest, `:floor` for cheapest
- **Example models**: `anthropic/claude-sonnet-4`, `openai/gpt-4o`, `google/gemini-2.0-flash`
- **Exact costs**: Queries generation stats API for precise billing

## Statusline Integration

Add available providers to your Claude Code statusline:

```bash
# Install the helper
chmod +x ~/.local/bin/owlex-status

# Add to statusline (in ~/.claude/settings.json)
# Append to your statusline command:
providers=$($HOME/.local/bin/owlex-status)
echo "... ${providers}"
```

Output: `[codex,gemini,kimi,openrouter]`

## Environment Variables

### Codex Settings
- `CODEX_CLEAN_OUTPUT`: Remove echoed prompts (default: `true`)
- `CODEX_SANDBOX_MODE`: Sandbox mode - `read-only` (suggest only), `workspace-write`, or `danger-full-access` (default: `read-only`)

### Gemini Settings
- `GEMINI_CLEAN_OUTPUT`: Remove noise from output (default: `true`)
- `GEMINI_SANDBOX_MODE`: Enable sandbox for suggest-only behavior (default: `true`)
- `GEMINI_APPROVAL_MODE`: Approval mode - `default` (prompt), `auto_edit`, or `yolo` (default: `default`)

### API Keys

Create a `.env` file in `~/.owlex/` with your API keys:

```bash
# Copy the example and edit
cp .env.example ~/.owlex/.env

# Or create manually
cat > ~/.owlex/.env << 'EOF'
OPENROUTER_API_KEY=sk-or-v1-...
KIMI_API_KEY=sk-...
MINIMAX_API_KEY=...
DEEPSEEK_API_KEY=sk-...
EOF
```

Available keys:
- `OPENROUTER_API_KEY`: [OpenRouter](https://openrouter.ai/keys) - 500+ models
- `KIMI_API_KEY`: [Kimi/Moonshot](https://platform.moonshot.cn/)
- `MINIMAX_API_KEY`: [MiniMax](https://platform.minimax.chat/)
- `DEEPSEEK_API_KEY`: [DeepSeek](https://platform.deepseek.com/)

## Data Storage

- **API keys**: `~/.owlex/.env`
- **Provider config**: `~/.owlex/providers.json`
- **Usage database**: `~/.owlex/usage.db` (SQLite)

## Architecture

```
owlex/
├── server.py           # MCP server with all tools
├── providers/
│   ├── protocol.py     # Provider interface
│   ├── codex.py        # Codex CLI provider
│   ├── gemini.py       # Gemini CLI provider
│   ├── openai_api.py   # OpenAI-compatible API provider
│   ├── openrouter.py   # OpenRouter provider
│   └── registry.py     # Provider discovery
└── storage/
    ├── schema.py       # SQLite schema
    └── usage.py        # Usage tracking
```
