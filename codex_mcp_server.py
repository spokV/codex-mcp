#!/usr/bin/env python3
"""
MCP Server for Codex CLI Integration
Allows Claude Code to send plans to Codex CLI for review
"""

import asyncio
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession


# Configuration
# Set to False to require approval for Codex operations (more secure)
# Set to True for automated execution without approval (less secure)
BYPASS_APPROVALS = os.environ.get("CODEX_BYPASS_APPROVALS", "false").lower() == "true"

# Set to False to show full Codex output including prompt templates
# Set to True to clean output and show only Codex's actual response (default)
CLEAN_OUTPUT = os.environ.get("CODEX_CLEAN_OUTPUT", "true").lower() == "true"


# Task Management
@dataclass
class Task:
    """Represents a background Codex task"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    command: str
    args: dict
    start_time: datetime
    context: Optional[Context[ServerSession, None]] = field(default=None, repr=False)
    completion_time: Optional[datetime] = None
    result: Optional[str] = None
    error: Optional[str] = None
    async_task: Optional[asyncio.Task] = field(default=None, repr=False)
    process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)


# Global task storage
tasks: dict[str, Task] = {}


async def cleanup_old_tasks():
    """Background task to clean up completed tasks after 5 minutes"""
    while True:
        try:
            await asyncio.sleep(60)  # Check every minute
            now = datetime.now()
            tasks_to_remove = []

            for task_id, task in tasks.items():
                if task.completion_time and (now - task.completion_time) > timedelta(minutes=5):
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del tasks[task_id]

        except Exception as e:
            # Log but don't crash the cleanup task
            print(f"Error in cleanup_old_tasks: {e}", flush=True)


# Initialize FastMCP server
mcp = FastMCP("codex-cli-server")


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
    prompt_was_removed = False
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()
        prompt_was_removed = True

    # Only apply pattern-based cleaning if we didn't already remove the exact prompt
    # This prevents accidentally removing legitimate Codex responses that happen to
    # start with these phrases
    if not prompt_was_removed:
        # Remove common prompt template patterns only at the very start of output
        # Use \A to anchor to absolute start, not ^ which matches line starts with MULTILINE
        # These patterns match the structure of prompts we generate, not arbitrary text
        patterns_to_remove = [
            # Remove "Review the following files: X\n\nReview type: Y\n\n" at start only
            r"\AReview the following files:.*?\n+Review type:.*?\n+",
            # Remove "Review the code in this directory.\n\nReview type: Y\n\n" at start only
            r"\AReview the code in this directory\.?\n+Review type:.*?\n+",
        ]

        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, count=1)

    # Remove extra blank lines (3+ consecutive newlines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


async def _run_codex_review(task_id: str, plan: str, context: str, review_type: str,
                            working_directory: str = None, files: list[str] = None):
    """
    Background coroutine that runs Codex review and updates task status
    """
    task = tasks[task_id]

    try:
        task.status = "running"

        # Construct the review prompt
        if working_directory and files:
            files_list = ", ".join(files)
            review_prompt = f"""{review_type.capitalize()} review of {files_list}:

{plan}

{f"Context: {context}" if context else ""}"""
        elif working_directory:
            review_prompt = f"""{review_type.capitalize()} review:

{plan}

{f"Context: {context}" if context else ""}"""
        else:
            review_prompt = f"""{review_type.capitalize()} review:

{plan}

{f"Context: {context}" if context else ""}"""

        # Build command
        cmd = ["codex", "exec", "--skip-git-repo-check"]

        if BYPASS_APPROVALS:
            cmd.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            cmd.append("--full-auto")

        cmd.append(review_prompt)

        # Execute Codex CLI
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory if working_directory else None
        )

        task.process = process

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            task.status = "failed"
            task.error = "Codex CLI review timed out after 300 seconds"
            task.completion_time = datetime.now()
            return

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            cleaned_output = clean_codex_output(stdout_text, review_prompt)
            task.status = "completed"
            task.result = f"Codex CLI Review Results:\n\n{cleaned_output}"
            task.completion_time = datetime.now()

            # Send MCP notification via context
            if task.context:
                try:
                    await task.context.info(f"Codex review task {task_id} completed successfully")
                except Exception as e:
                    print(f"[ERROR] Failed to send notification: {e}", file=sys.stderr, flush=True)
        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")

            combined_error = "\n\n".join(error_output) if error_output else "No output"
            task.status = "failed"
            task.error = f"Codex CLI returned an error (exit code {process.returncode}):\n\n{combined_error}"
            task.completion_time = datetime.now()

    except asyncio.CancelledError:
        # Task was cancelled
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        raise

    except FileNotFoundError as e:
        if e.filename == "codex":
            task.error = "Error: 'codex' command not found. Please ensure Codex CLI is installed and in your PATH."
        else:
            task.error = f"Error: Path not found or inaccessible: {e.filename or working_directory or 'unknown'}"
        task.status = "failed"
        task.completion_time = datetime.now()

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Codex CLI: {str(e)}"
        task.completion_time = datetime.now()


@mcp.tool()
async def start_review(
    plan: str,
    ctx: Context[ServerSession, None],
    context: str = "",
    review_type: str = "plan",
    working_directory: str = None,
    files: list[str] = None
) -> str:
    """
    Start a background Codex review and return task ID immediately.

    Args:
        plan: The plan, code, or changes to review
        context: Additional context about the task or goal (optional)
        review_type: Type of review to perform (plan, code, architecture, security)
        working_directory: Working directory for Codex to use (optional)
        files: Specific files to review (optional)
    """
    # Validate inputs
    if not plan or not plan.strip():
        return "Error: 'plan' parameter is required and cannot be empty."

    if working_directory and not working_directory.strip():
        return "Error: 'working_directory' cannot be empty if provided."

    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="review",
        args={
            "plan": plan,
            "context": context,
            "review_type": review_type,
            "working_directory": working_directory,
            "files": files
        },
        start_time=datetime.now(),
        context=ctx  # Store context for notifications
    )

    tasks[task_id] = task

    # Launch background task
    task.async_task = asyncio.create_task(_run_codex_review(
        task_id, plan, context, review_type, working_directory, files
    ))

    return f"Codex review started. Task ID: {task_id}\n\nUse get_task_status or get_task_result to check progress."


async def _run_codex_command(task_id: str, command: str, args: list[str]):
    """
    Background coroutine that runs Codex command and updates task status
    """
    task = tasks[task_id]

    try:
        task.status = "running"

        # Track whether this is a prompt
        is_prompt = False
        prompt_text = ""

        # Normalize args
        args = list(args or [])

        # Build command
        if command in ["exec", "login", "logout", "mcp", "mcp-server", "app-server",
                       "completion", "sandbox", "apply", "resume", "cloud", "features"]:
            full_command = ["codex", command] + args
        else:
            # Treat command as a prompt for exec subcommand
            is_prompt = True
            prompt_text = command

            full_command = ["codex", "exec", "--skip-git-repo-check"]

            if BYPASS_APPROVALS:
                full_command.append("--dangerously-bypass-approvals-and-sandbox")
            else:
                full_command.append("--full-auto")

            full_command.extend(args)
            full_command.append(command)

        # Execute Codex CLI
        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        task.process = process

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=300
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            task.status = "failed"
            task.error = "Codex CLI command timed out after 300 seconds"
            task.completion_time = datetime.now()
            return

        stdout_text = stdout.decode('utf-8') if stdout else ""
        stderr_text = stderr.decode('utf-8') if stderr else ""

        if process.returncode == 0:
            # Clean the output if this was a prompt
            if is_prompt:
                cleaned_output = clean_codex_output(stdout_text, prompt_text)
                task.result = f"Codex CLI Output:\n\n{cleaned_output}"
            else:
                task.result = f"Codex CLI Output:\n\n{stdout_text}"

            task.status = "completed"
            task.completion_time = datetime.now()

            # Send MCP notification via context
            if task.context:
                try:
                    await task.context.info(f"Codex command task {task_id} completed successfully")
                except Exception as e:
                    print(f"[ERROR] Failed to send notification: {e}", file=sys.stderr, flush=True)
        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")

            combined_error = "\n\n".join(error_output) if error_output else "No output"
            task.status = "failed"
            task.error = f"Codex CLI Error (exit code {process.returncode}):\n\n{combined_error}"
            task.completion_time = datetime.now()

    except asyncio.CancelledError:
        # Task was cancelled
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        raise

    except FileNotFoundError as e:
        if e.filename == "codex":
            task.error = "Error: 'codex' command not found. Please ensure Codex CLI is installed and in your PATH."
        else:
            task.error = f"Error: Path not found or inaccessible: {e.filename or 'unknown'}"
        task.status = "failed"
        task.completion_time = datetime.now()

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Codex CLI: {str(e)}"
        task.completion_time = datetime.now()


@mcp.tool()
async def start_codex_command(
    command: str,
    ctx: Context[ServerSession, None],
    args: list[str] = None
) -> str:
    """
    Start a background Codex command and return task ID immediately.

    Args:
        command: The Codex CLI command to execute
        args: Arguments to pass to the command
    """
    # Validate inputs
    if not command or not command.strip():
        return "Error: 'command' parameter is required and cannot be empty."

    # Create task
    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="execute",
        args={
            "command": command,
            "args": args or []
        },
        start_time=datetime.now(),
        context=ctx  # Store context for notifications
    )

    tasks[task_id] = task

    # Launch background task
    task.async_task = asyncio.create_task(_run_codex_command(
        task_id, command, args or []
    ))

    return f"Codex command started. Task ID: {task_id}\n\nUse get_task_status or get_task_result to check progress."


@mcp.tool()
async def get_task_status(task_id: str) -> str:
    """
    Get the status of a background task.

    Args:
        task_id: The task ID returned by start_review or start_codex_command
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    status_info = f"""Task ID: {task_id}
Status: {task.status}
Command: {task.command}
Started: {task.start_time.strftime('%Y-%m-%d %H:%M:%S')}"""

    if task.completion_time:
        duration = (task.completion_time - task.start_time).total_seconds()
        status_info += f"\nCompleted: {task.completion_time.strftime('%Y-%m-%d %H:%M:%S')}"
        status_info += f"\nDuration: {duration:.1f} seconds"

    if task.status == "pending":
        status_info += "\n\nTask is queued and will start soon."
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        status_info += f"\n\nTask is running (elapsed: {elapsed:.1f} seconds)."
    elif task.status == "completed":
        status_info += "\n\nTask completed successfully. Use get_task_result to retrieve the output."
    elif task.status == "failed":
        status_info += f"\n\nTask failed: {task.error}"
    elif task.status == "cancelled":
        status_info += "\n\nTask was cancelled."

    return status_info


@mcp.tool()
async def get_task_result(task_id: str) -> str:
    """
    Get the result of a completed background task.

    Args:
        task_id: The task ID returned by start_review or start_codex_command
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    if task.status == "pending":
        return f"Task {task_id} is still pending. Use get_task_status to check its status."
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        return f"Task {task_id} is still running (elapsed: {elapsed:.1f} seconds). Use get_task_status to check its status."
    elif task.status == "completed":
        return task.result or "Task completed but no output was captured."
    elif task.status == "failed":
        return f"Task {task_id} failed:\n\n{task.error}"
    elif task.status == "cancelled":
        return f"Task {task_id} was cancelled."
    else:
        return f"Task {task_id} has unknown status: {task.status}"


@mcp.tool()
async def wait_for_task(task_id: str, timeout: int = 300) -> str:
    """
    Wait for a background task to complete and return its result.
    Blocks until the task finishes or timeout is reached.

    Args:
        task_id: The task ID returned by start_review or start_codex_command
        timeout: Maximum seconds to wait (default: 300)
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    # If task is already done, return immediately
    if task.status in ["completed", "failed", "cancelled"]:
        if task.status == "completed":
            return f"Task completed!\n\n{task.result or 'No output was captured.'}"
        elif task.status == "failed":
            return f"Task failed:\n\n{task.error}"
        else:
            return f"Task was cancelled."

    # Wait for the asyncio task to complete
    # Use shield to prevent timeout from cancelling the background task
    if task.async_task:
        try:
            await asyncio.wait_for(asyncio.shield(task.async_task), timeout=timeout)
        except asyncio.TimeoutError:
            # Task keeps running in background despite timeout
            elapsed = (datetime.now() - task.start_time).total_seconds()
            return f"Task is still running after {timeout}s timeout (total elapsed: {elapsed:.1f}s). Use get_task_result to check later."
        except asyncio.CancelledError:
            return f"Task was cancelled."
        except Exception as e:
            return f"Task failed with exception: {str(e)}"

    # Return the result based on final status
    if task.status == "completed":
        return f"Task completed!\n\n{task.result or 'No output was captured.'}"
    elif task.status == "failed":
        return f"Task failed:\n\n{task.error}"
    elif task.status == "cancelled":
        return f"Task was cancelled."
    else:
        return f"Task ended with unexpected status: {task.status}"


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """
    Cancel a running background task.

    Args:
        task_id: The task ID to cancel
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found. It may have expired (tasks are kept for 5 minutes after completion)."

    task = tasks[task_id]

    if task.status in ["completed", "failed", "cancelled"]:
        return f"Task {task_id} is already {task.status} and cannot be cancelled."

    try:
        # Kill the subprocess if it exists
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except Exception as e:
                # Process might already be dead
                pass

        # Cancel the asyncio task
        if task.async_task:
            task.async_task.cancel()
            try:
                await task.async_task
            except asyncio.CancelledError:
                pass

        # Update task status
        task.status = "cancelled"
        task.completion_time = datetime.now()

        return f"Task {task_id} has been cancelled."

    except Exception as e:
        return f"Error cancelling task {task_id}: {str(e)}"


# FastMCP auto-discovers tools via @mcp.tool() decorators
# No need for manual list_tools() or call_tool() handlers


if __name__ == "__main__":
    # Start cleanup task in background
    async def run_with_cleanup():
        cleanup_task = asyncio.create_task(cleanup_old_tasks())
        try:
            # Use run_stdio_async to avoid double event loop
            await mcp.run_stdio_async()
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_with_cleanup())
