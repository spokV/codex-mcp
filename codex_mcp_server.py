#!/usr/bin/env python3
"""
MCP Server for Codex CLI Integration
Allows Claude Code to send plans to Codex CLI for review
"""

import asyncio
import os
import re
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Configuration
# Set to False to require approval for Codex operations (more secure)
# Set to True for automated execution without approval (less secure)
BYPASS_APPROVALS = os.environ.get("CODEX_BYPASS_APPROVALS", "false").lower() == "true"

# Set to False to show full Codex output including prompt templates
# Set to True to clean output and show only Codex's actual response (default)
CLEAN_OUTPUT = os.environ.get("CODEX_CLEAN_OUTPUT", "true").lower() == "true"

# Initialize MCP server
app = Server("codex-cli-server")


def clean_codex_output(raw_output: str, original_prompt: str = "") -> str:
    """
    Clean Codex CLI output by removing echoed prompt templates.

    Args:
        raw_output: Raw output from Codex CLI
        original_prompt: The prompt we sent to Codex (optional)

    Returns:
        Cleaned output with only the actual response
    """
    if not CLEAN_OUTPUT:
        return raw_output

    cleaned = raw_output

    # Remove the exact prompt template if it appears at the start of output
    # This is more conservative and only removes the template we injected
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()

    # Remove common prompt template patterns only at the very start of output
    # Use \A to anchor to absolute start, not ^ which matches line starts with MULTILINE
    patterns_to_remove = [
        # Remove "Review the following files: X" followed by newlines at start only
        r"\AReview the following files:.*?\n+",
        # Remove "Review the code in this directory" at absolute start only
        r"\AReview the code in this directory\.?\n+",
        # Remove "Review type:" line only at absolute start
        r"\AReview type:.*?\n+",
        # Remove "Context:" line only at absolute start
        r"\AContext:.*?\n+",
    ]

    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, "", cleaned, count=1)

    # Remove extra blank lines (3+ consecutive newlines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for Codex CLI interaction"""
    return [
        Tool(
            name="review_plan",
            description="Send a plan or code changes to Codex CLI for review and get feedback",
            inputSchema={
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "The plan, code, or changes to review"
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context about the task or goal (optional)"
                    },
                    "review_type": {
                        "type": "string",
                        "enum": ["plan", "code", "architecture", "security"],
                        "description": "Type of review to perform",
                        "default": "plan"
                    },
                    "working_directory": {
                        "type": "string",
                        "description": "Working directory for Codex to use when reviewing code (optional). If provided, Codex will review files in this directory instead of receiving code inline."
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific files to review (optional). If not provided, Codex will review the entire working directory."
                    }
                },
                "required": ["plan"]
            }
        ),
        Tool(
            name="execute_codex_command",
            description="Execute a custom Codex CLI command and get the response",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The Codex CLI command to execute"
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Arguments to pass to the command"
                    }
                },
                "required": ["command"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from Claude Code"""

    if name == "review_plan":
        return await review_plan(
            arguments.get("plan"),
            arguments.get("context", ""),
            arguments.get("review_type", "plan"),
            arguments.get("working_directory"),
            arguments.get("files", [])
        )

    elif name == "execute_codex_command":
        return await execute_codex_command(
            arguments.get("command"),
            arguments.get("args", [])
        )

    else:
        raise ValueError(f"Unknown tool: {name}")


async def review_plan(plan: str, context: str, review_type: str,
                      working_directory: str = None, files: list[str] = None) -> list[TextContent]:
    """
    Send a plan to Codex CLI for review

    This creates a prompt for Codex CLI asking it to review the provided plan.
    If working_directory is provided, Codex will review files in that directory
    using its built-in file reading capabilities instead of receiving code inline.
    """

    # Construct the review prompt
    if working_directory and files:
        # Ask Codex to review specific files in the working directory
        files_list = ", ".join(files)
        review_prompt = f"""Review the following files: {files_list}

Review type: {review_type}

{plan}

{f"Context: {context}" if context else ""}

Please provide:
1. Overall assessment (approve/needs changes/reject)
2. Specific issues or concerns
3. Suggestions for improvement
4. Any risks or considerations

Format your response as structured feedback."""
    elif working_directory:
        # Ask Codex to review the working directory
        review_prompt = f"""Review the code in this directory.

Review type: {review_type}

{plan}

{f"Context: {context}" if context else ""}

Please provide:
1. Overall assessment (approve/needs changes/reject)
2. Specific issues or concerns
3. Suggestions for improvement
4. Any risks or considerations

Format your response as structured feedback."""
    else:
        # Traditional inline review
        review_prompt = f"""Review the following {review_type}:

{plan}

{f"Context: {context}" if context else ""}

Please provide:
1. Overall assessment (approve/needs changes/reject)
2. Specific issues or concerns
3. Suggestions for improvement
4. Any risks or considerations

Format your response as structured feedback."""

    # Validate inputs
    if not plan or not plan.strip():
        return [TextContent(
            type="text",
            text="Error: 'plan' parameter is required and cannot be empty."
        )]

    if working_directory and not working_directory.strip():
        return [TextContent(
            type="text",
            text="Error: 'working_directory' cannot be empty if provided."
        )]

    try:
        # Build command with optional working directory
        cmd = ["codex", "exec", "--skip-git-repo-check"]

        # Only add bypass flag if configured
        if BYPASS_APPROVALS:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            # Use full-auto for safer automated execution
            cmd.append("--full-auto")

        if working_directory:
            cmd.extend(["-C", working_directory])

        cmd.append(review_prompt)

        # Execute Codex CLI using async subprocess for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120  # Increased timeout for complex reviews
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return [TextContent(
                type="text",
                text="Codex CLI review timed out after 120 seconds"
            )]

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            # Clean the output to remove prompt templates
            cleaned_output = clean_codex_output(stdout_text, review_prompt)
            return [TextContent(
                type="text",
                text=f"Codex CLI Review Results:\n\n{cleaned_output}"
            )]
        else:
            return [TextContent(
                type="text",
                text=f"Codex CLI returned an error:\n{stderr_text}"
            )]

    except FileNotFoundError:
        return [TextContent(
            type="text",
            text="Error: 'codex' command not found. Please ensure Codex CLI is installed and in your PATH."
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing Codex CLI: {str(e)}"
        )]


async def execute_codex_command(command: str, args: list[str]) -> list[TextContent]:
    """
    Execute a custom Codex CLI command

    This allows Claude Code to invoke any Codex CLI command directly.
    Supports both subcommands (exec, login, etc.) and direct prompts.
    """

    # Validate inputs
    if not command or not command.strip():
        return [TextContent(
            type="text",
            text="Error: 'command' parameter is required and cannot be empty."
        )]

    try:
        # Build the full command
        # Track whether this is a prompt (for output cleaning)
        is_prompt = False
        prompt_text = ""

        # If command looks like a subcommand, use it as such; otherwise treat as a prompt
        if command in ["exec", "login", "logout", "mcp", "mcp-server", "app-server",
                       "completion", "sandbox", "apply", "resume", "cloud", "features"]:
            full_command = ["codex", command] + args
        else:
            # Treat command as a prompt for exec subcommand
            is_prompt = True
            prompt_text = command

            full_command = ["codex", "exec", "--skip-git-repo-check"]

            # Only add bypass flag if configured
            if BYPASS_APPROVALS:
                full_command.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                # Use full-auto for safer automated execution
                full_command.append("--full-auto")

            # Add all args before the prompt to preserve flag/value pairs
            # CLI flags and their values must come before the prompt
            full_command.extend(args)
            # Add the prompt last
            full_command.append(command)

        # Execute Codex CLI using async subprocess for non-blocking execution
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=120  # Increased timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return [TextContent(
                type="text",
                text="Codex CLI command timed out after 120 seconds"
            )]

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            # Clean the output if this was a prompt (not a subcommand)
            if is_prompt:
                cleaned_output = clean_codex_output(stdout_text, prompt_text)
                return [TextContent(
                    type="text",
                    text=f"Codex CLI Output:\n\n{cleaned_output}"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"Codex CLI Output:\n\n{stdout_text}"
                )]
        else:
            return [TextContent(
                type="text",
                text=f"Codex CLI Error:\n{stderr_text}"
            )]

    except FileNotFoundError:
        return [TextContent(
            type="text",
            text="Error: 'codex' command not found. Please ensure Codex CLI is installed and in your PATH."
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error executing Codex CLI: {str(e)}"
        )]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
