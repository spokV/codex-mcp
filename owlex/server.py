#!/usr/bin/env python3
"""
MCP Server for Multi-AI Provider Integration with Analytics
Supports CLI providers (Codex, Gemini) and API providers (Kimi, MiniMax, OpenRouter)
"""

import asyncio
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

# Load .env file from ~/.owlex/.env or current directory
_env_paths = [
    Path.home() / ".owlex" / ".env",
    Path.cwd() / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

# Import providers and storage
from .providers.registry import registry
from .providers.protocol import ProviderResult
from .storage import usage as usage_db
from .storage.schema import init_database


# Configuration - Codex
CODEX_CLEAN_OUTPUT = os.environ.get("CODEX_CLEAN_OUTPUT", "true").lower() == "true"
# Sandbox mode: read-only (suggest only), workspace-write, or danger-full-access
CODEX_SANDBOX_MODE = os.environ.get("CODEX_SANDBOX_MODE", "read-only")

# Configuration - Gemini
GEMINI_CLEAN_OUTPUT = os.environ.get("GEMINI_CLEAN_OUTPUT", "true").lower() == "true"
# Sandbox mode: default is enabled (suggest only)
GEMINI_SANDBOX_MODE = os.environ.get("GEMINI_SANDBOX_MODE", "true").lower() == "true"
# Approval mode: default, auto_edit, or yolo
GEMINI_APPROVAL_MODE = os.environ.get("GEMINI_APPROVAL_MODE", "default")


@dataclass
class Task:
    """Represents a background Codex task"""
    task_id: str
    status: str  # pending, running, completed, failed, cancelled
    command: str
    args: dict
    start_time: datetime
    context: Context[ServerSession, None] | None = field(default=None, repr=False)
    completion_time: datetime | None = None
    result: str | None = None
    error: str | None = None
    async_task: asyncio.Task | None = field(default=None, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)


# Global task storage
tasks: dict[str, Task] = {}


async def _send_context_message(task: Task, level: str, message: str):
    if not task.context:
        return
    handler = getattr(task.context, level, None)
    if not callable(handler):
        handler = task.context.info
    try:
        await asyncio.shield(handler(message))
    except Exception as e:
        print(f"[ERROR] Failed to send {level} notification: {e}", file=sys.stderr, flush=True)


async def _emit_task_notification(task: Task):
    if not task.context:
        return
    prefix = "[owlex]"
    if task.status == "completed":
        # Include short preview of result
        preview = ""
        if task.result:
            lines = task.result.strip().split('\n')
            preview = f": {lines[-1][:100]}" if lines else ""
        await _send_context_message(task, "info", f"{prefix} Task {task.task_id[:8]} completed{preview}")
    elif task.status == "failed":
        error_preview = (task.error or "")[:100]
        await _send_context_message(task, "error", f"{prefix} Task {task.task_id[:8]} failed: {error_preview}")


async def cleanup_old_tasks():
    """Background task to clean up completed tasks after 5 minutes"""
    while True:
        try:
            await asyncio.sleep(60)
            now = datetime.now()
            tasks_to_remove = [
                task_id for task_id, task in tasks.items()
                if task.completion_time and (now - task.completion_time) > timedelta(minutes=5)
            ]
            for task_id in tasks_to_remove:
                del tasks[task_id]
        except Exception as e:
            print(f"Error in cleanup_old_tasks: {e}", flush=True)


# Initialize FastMCP server
mcp = FastMCP("owlex-server")


def clean_codex_output(raw_output: str, original_prompt: str = "") -> str:
    """Clean Codex CLI output by removing echoed prompt templates."""
    if not CODEX_CLEAN_OUTPUT:
        return raw_output
    cleaned = raw_output
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


async def _run_codex_resume(
    task_id: str,
    session_ref: str,
    prompt: str,
    working_directory: str | None = None,
    enable_search: bool = False
):
    """Background coroutine that runs Codex resume and updates task status"""
    task = tasks[task_id]

    try:
        task.status = "running"

        # Build command: codex exec [options] resume [--last | session-id] -
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        # Add --cd and --search before subcommand
        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.append("--search")

        # Use sandbox mode (defaults to read-only for suggest-only behavior)
        full_command.extend(["--sandbox", CODEX_SANDBOX_MODE])

        # Add resume subcommand
        full_command.append("resume")
        if session_ref == "--last":
            full_command.append("--last")
        else:
            full_command.append(session_ref)
        full_command.append("-")  # Read prompt from stdin

        stdin_input = prompt.encode('utf-8')

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        task.process = process

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_input),
                timeout=300
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            task.status = "failed"
            task.error = "Codex resume timed out after 300 seconds"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
            return

        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        if process.returncode == 0:
            cleaned_output = clean_codex_output(stdout_text, prompt)
            task.result = f"Codex Resume Output:\n\n{cleaned_output}"
            task.status = "completed"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            task.status = "failed"
            task.error = f"Codex resume failed (exit code {process.returncode}):\n\n" + ("\n\n".join(error_output) or "No output")
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        task.error = "Error: 'codex' command not found. Please ensure Codex CLI is installed and in your PATH." if e.filename == "codex" else f"Error: {e}"
        task.status = "failed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Codex resume: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


async def _run_codex_exec(
    task_id: str,
    prompt: str,
    working_directory: str | None = None,
    enable_search: bool = False
):
    """Background coroutine that runs Codex exec (new session) and updates task status"""
    task = tasks[task_id]

    try:
        task.status = "running"

        # Build command: codex exec [options] -
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.append("--search")

        # Use sandbox mode (defaults to read-only for suggest-only behavior)
        full_command.extend(["--sandbox", CODEX_SANDBOX_MODE])

        full_command.append("-")  # Read prompt from stdin

        stdin_input = prompt.encode('utf-8')

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        task.process = process

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_input),
                timeout=300
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            task.status = "failed"
            task.error = "Codex exec timed out after 300 seconds"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
            return

        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        if process.returncode == 0:
            cleaned_output = clean_codex_output(stdout_text, prompt)
            task.result = f"Codex Output:\n\n{cleaned_output}"
            task.status = "completed"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            task.status = "failed"
            task.error = f"Codex exec failed (exit code {process.returncode}):\n\n" + ("\n\n".join(error_output) or "No output")
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        task.error = "Error: 'codex' command not found. Please ensure Codex CLI is installed and in your PATH." if e.filename == "codex" else f"Error: {e}"
        task.status = "failed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Codex: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


def clean_gemini_output(raw_output: str, original_prompt: str = "") -> str:
    """Clean Gemini CLI output by removing noise."""
    if not GEMINI_CLEAN_OUTPUT:
        return raw_output
    cleaned = raw_output
    # Remove YOLO mode warning if present
    if cleaned.startswith("YOLO mode is enabled."):
        lines = cleaned.split('\n', 2)
        if len(lines) > 2:
            cleaned = lines[2]
        elif len(lines) > 1:
            cleaned = lines[1]
    # Remove "Loaded cached credentials." line
    cleaned = re.sub(r'^Loaded cached credentials\.\n?', '', cleaned, flags=re.MULTILINE)
    # Collapse triple+ newlines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


async def _run_gemini_exec(
    task_id: str,
    prompt: str,
    working_directory: str | None = None,
):
    """Background coroutine that runs Gemini CLI (new session) and updates task status"""
    task = tasks[task_id]

    try:
        task.status = "running"

        # Build command: gemini [options] "prompt"
        full_command = ["gemini"]

        # Sandbox mode for suggest-only behavior (default: enabled)
        if GEMINI_SANDBOX_MODE:
            full_command.append("--sandbox")

        # Approval mode: default (prompt), auto_edit, or yolo
        if GEMINI_APPROVAL_MODE != "default":
            full_command.extend(["--approval-mode", GEMINI_APPROVAL_MODE])

        if working_directory:
            full_command.extend(["--include-directories", working_directory])

        full_command.append(prompt)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory
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
            task.error = "Gemini exec timed out after 300 seconds"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
            return

        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        if process.returncode == 0:
            cleaned_output = clean_gemini_output(stdout_text, prompt)
            task.result = f"Gemini Output:\n\n{cleaned_output}"
            task.status = "completed"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            task.status = "failed"
            task.error = f"Gemini exec failed (exit code {process.returncode}):\n\n" + ("\n\n".join(error_output) or "No output")
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        task.error = "Error: 'gemini' command not found. Please ensure Gemini CLI is installed (brew install gemini-cli)." if e.filename == "gemini" else f"Error: {e}"
        task.status = "failed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Gemini: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


async def _run_gemini_resume(
    task_id: str,
    session_ref: str,
    prompt: str,
    working_directory: str | None = None,
):
    """Background coroutine that runs Gemini resume and updates task status"""
    task = tasks[task_id]

    try:
        task.status = "running"

        # Build command: gemini [options] -r <session> "prompt"
        full_command = ["gemini"]

        # Sandbox mode for suggest-only behavior (default: enabled)
        if GEMINI_SANDBOX_MODE:
            full_command.append("--sandbox")

        # Approval mode: default (prompt), auto_edit, or yolo
        if GEMINI_APPROVAL_MODE != "default":
            full_command.extend(["--approval-mode", GEMINI_APPROVAL_MODE])

        if working_directory:
            full_command.extend(["--include-directories", working_directory])

        # Add resume flag
        full_command.extend(["-r", session_ref])
        full_command.append(prompt)

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_directory
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
            task.error = "Gemini resume timed out after 300 seconds"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
            return

        stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""

        if process.returncode == 0:
            cleaned_output = clean_gemini_output(stdout_text, prompt)
            task.result = f"Gemini Resume Output:\n\n{cleaned_output}"
            task.status = "completed"
            task.completion_time = datetime.now()
            await _emit_task_notification(task)
        else:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            task.status = "failed"
            task.error = f"Gemini resume failed (exit code {process.returncode}):\n\n" + ("\n\n".join(error_output) or "No output")
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        task.status = "cancelled"
        task.completion_time = datetime.now()
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        task.error = "Error: 'gemini' command not found. Please ensure Gemini CLI is installed (brew install gemini-cli)." if e.filename == "gemini" else f"Error: {e}"
        task.status = "failed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing Gemini resume: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


@mcp.tool()
async def start_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=False, description="Enable web search (--search flag)")
) -> str:
    """Start a new Codex session (no prior context)."""
    if not prompt or not prompt.strip():
        return "Error: 'prompt' parameter is required."

    if working_directory:
        working_directory = os.path.expanduser(working_directory)
        if not os.path.isdir(working_directory):
            return f"Error: working_directory '{working_directory}' does not exist or is not a directory."

    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="exec",
        args={
            "prompt": prompt.strip(),
            "working_directory": working_directory,
            "enable_search": enable_search
        },
        start_time=datetime.now(),
        context=ctx
    )

    tasks[task_id] = task

    task.async_task = asyncio.create_task(_run_codex_exec(
        task_id, prompt.strip(),
        working_directory, enable_search
    ))

    return f"Codex session started. Task ID: {task_id}\n\nUse wait_for_task to get result."


@mcp.tool()
async def resume_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_id: str | None = Field(default=None, description="Session ID to resume (uses --last if not provided)"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=False, description="Enable web search (--search flag)")
) -> str:
    """Resume an existing Codex session and ask for advice."""
    if not prompt or not prompt.strip():
        return "Error: 'prompt' parameter is required."

    # Validate working_directory if provided
    if working_directory:
        working_directory = os.path.expanduser(working_directory)
        if not os.path.isdir(working_directory):
            return f"Error: working_directory '{working_directory}' does not exist or is not a directory."

    # Use --last if no session_id provided
    use_last = not session_id or not session_id.strip()
    session_ref = "--last" if use_last else session_id.strip()

    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="resume",
        args={
            "session_id": session_ref,
            "prompt": prompt.strip(),
            "working_directory": working_directory,
            "enable_search": enable_search
        },
        start_time=datetime.now(),
        context=ctx
    )

    tasks[task_id] = task

    task.async_task = asyncio.create_task(_run_codex_resume(
        task_id, session_ref, prompt.strip(),
        working_directory, enable_search
    ))

    return f"Codex resume started{' (last session)' if use_last else f' for session {session_id}'}. Task ID: {task_id}\n\nUse wait_for_task to get result."


@mcp.tool()
async def start_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> str:
    """Start a new Gemini CLI session (no prior context)."""
    if not prompt or not prompt.strip():
        return "Error: 'prompt' parameter is required."

    if working_directory:
        working_directory = os.path.expanduser(working_directory)
        if not os.path.isdir(working_directory):
            return f"Error: working_directory '{working_directory}' does not exist or is not a directory."

    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="gemini_exec",
        args={
            "prompt": prompt.strip(),
            "working_directory": working_directory,
        },
        start_time=datetime.now(),
        context=ctx
    )

    tasks[task_id] = task

    task.async_task = asyncio.create_task(_run_gemini_exec(
        task_id, prompt.strip(),
        working_directory
    ))

    return f"Gemini session started. Task ID: {task_id}\n\nUse wait_for_task to get result."


@mcp.tool()
async def resume_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_ref: str = Field(default="latest", description="Session to resume: 'latest' for most recent, or index number"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> str:
    """Resume an existing Gemini CLI session with full conversation history."""
    if not prompt or not prompt.strip():
        return "Error: 'prompt' parameter is required."

    if working_directory:
        working_directory = os.path.expanduser(working_directory)
        if not os.path.isdir(working_directory):
            return f"Error: working_directory '{working_directory}' does not exist or is not a directory."

    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="gemini_resume",
        args={
            "session_ref": session_ref,
            "prompt": prompt.strip(),
            "working_directory": working_directory,
        },
        start_time=datetime.now(),
        context=ctx
    )

    tasks[task_id] = task

    task.async_task = asyncio.create_task(_run_gemini_resume(
        task_id, session_ref, prompt.strip(),
        working_directory
    ))

    return f"Gemini resume started (session: {session_ref}). Task ID: {task_id}\n\nUse wait_for_task to get result."


@mcp.tool()
async def get_task_result(task_id: str) -> str:
    """
    Get the result of a task (Codex or Gemini).

    Args:
        task_id: The task ID returned by resume_codex_session
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found."

    task = tasks[task_id]

    if task.status == "pending":
        return f"Task {task_id} is still pending."
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        return f"Task {task_id} is still running ({elapsed:.1f}s elapsed)."
    elif task.status == "completed":
        return task.result or "Task completed but no output was captured."
    elif task.status == "failed":
        return f"Task {task_id} failed:\n\n{task.error}"
    else:
        return f"Task {task_id} status: {task.status}"


@mcp.tool()
async def wait_for_task(task_id: str, timeout: int = 300) -> str:
    """
    Wait for a task to complete and return its result.

    Args:
        task_id: The task ID to wait for
        timeout: Maximum seconds to wait (default: 300)
    """
    if not task_id or task_id not in tasks:
        return f"Error: Task '{task_id}' not found."

    task = tasks[task_id]

    if task.status in ["completed", "failed", "cancelled"]:
        if task.status == "completed":
            return task.result or "No output."
        return f"Task {task.status}: {task.error or ''}"

    if task.async_task:
        try:
            await asyncio.wait_for(asyncio.shield(task.async_task), timeout=timeout)
        except asyncio.TimeoutError:
            return f"Task still running after {timeout}s. Use get_task_result to check later."
        except Exception as e:
            return f"Task failed: {str(e)}"

    if task.status == "completed":
        return task.result or "No output."
    return f"Task {task.status}: {task.error or ''}"


# =============================================================================
# New Marketplace Tools
# =============================================================================


async def _run_provider_call(
    task_id: str,
    provider_name: str,
    prompt: str,
    model: str | None = None,
    working_directory: str | None = None,
):
    """Background task to run a provider call with usage tracking."""
    task = tasks[task_id]
    provider = registry.get_provider(provider_name)

    if not provider:
        task.status = "failed"
        task.error = f"Provider '{provider_name}' not found"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)
        return

    try:
        task.status = "running"

        # Call the provider
        result: ProviderResult = await provider.call(
            prompt=prompt,
            model=model,
            working_directory=working_directory,
        )

        # Record usage
        await usage_db.record_usage(
            task_id=task_id,
            provider_name=result.provider_name,
            model_name=result.model_name,
            provider_type=provider.provider_type,
            status="completed",
            prompt=prompt,
            working_directory=working_directory,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            estimated_cost=result.cost,
            duration_seconds=result.duration_seconds,
            generation_id=result.generation_id,
        )

        task.result = f"{result.provider_name} ({result.model_name}) Output:\n\n{result.content}"
        task.status = "completed"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except TimeoutError as e:
        await usage_db.record_usage(
            task_id=task_id,
            provider_name=provider_name,
            model_name=model or "unknown",
            provider_type=provider.provider_type,
            status="timeout",
            prompt=prompt,
            error_message=str(e),
        )
        task.status = "failed"
        task.error = str(e)
        task.completion_time = datetime.now()
        await _emit_task_notification(task)

    except Exception as e:
        await usage_db.record_usage(
            task_id=task_id,
            provider_name=provider_name,
            model_name=model or "unknown",
            provider_type=provider.provider_type,
            status="failed",
            prompt=prompt,
            error_message=str(e),
        )
        task.status = "failed"
        task.error = f"Error: {str(e)}"
        task.completion_time = datetime.now()
        await _emit_task_notification(task)


@mcp.tool()
async def list_providers(ctx: Context[ServerSession, None]) -> str:
    """
    List all available AI providers.

    Returns providers configured in ~/.owlex/providers.json plus built-in
    providers (codex, gemini).
    """
    registry.initialize()
    providers = registry.list_all()

    lines = ["Available providers:\n"]
    for p in providers:
        status = "ready" if p["available"] else "not configured"
        model_info = f" [{p['model']}]" if p.get("model") else ""
        lines.append(f"  - {p['name']} ({p['type']}){model_info}: {status}")

    lines.append("\nTo add a new provider, edit ~/.owlex/providers.json")
    return "\n".join(lines)


@mcp.tool()
async def call_provider(
    ctx: Context[ServerSession, None],
    provider: str = Field(description="Provider name (codex, gemini, kimi, minimax, openrouter, etc.)"),
    prompt: str = Field(description="The prompt to send"),
    model: str | None = Field(default=None, description="Model override (uses default if not specified)"),
    working_directory: str | None = Field(default=None, description="Working directory context"),
) -> str:
    """
    Call any configured AI provider.

    Supports both CLI-based (codex, gemini) and API-based (kimi, minimax, openrouter) providers.
    Usage is automatically tracked in ~/.owlex/usage.db.

    After receiving the response, you should rate it using rate_response.
    """
    registry.initialize()

    if not prompt or not prompt.strip():
        return "Error: 'prompt' parameter is required."

    provider_instance = registry.get_provider(provider)
    if not provider_instance:
        available = ", ".join(registry.list_available())
        return f"Error: Provider '{provider}' not found. Available: {available}"

    if not provider_instance.is_available:
        return f"Error: Provider '{provider}' is not configured. Check API key or CLI installation."

    if working_directory:
        working_directory = os.path.expanduser(working_directory)
        if not os.path.isdir(working_directory):
            return f"Error: working_directory '{working_directory}' does not exist."

    task_id = str(uuid.uuid4())
    task = Task(
        task_id=task_id,
        status="pending",
        command="call_provider",
        args={"provider": provider, "prompt": prompt[:200], "model": model},
        start_time=datetime.now(),
        context=ctx,
    )
    tasks[task_id] = task

    task.async_task = asyncio.create_task(
        _run_provider_call(task_id, provider, prompt.strip(), model, working_directory)
    )

    return f"Provider call started ({provider}). Task ID: {task_id}\n\nUse wait_for_task to get result, then rate_response to rate quality."


@mcp.tool()
async def rate_response(
    ctx: Context[ServerSession, None],
    task_id: str = Field(description="Task ID to rate"),
    helpfulness: int = Field(description="Helpfulness rating (1-5)"),
    accuracy: int = Field(description="Accuracy rating (1-5)"),
    completeness: int = Field(description="Completeness rating (1-5)"),
    notes: str | None = Field(default=None, description="Optional notes about the rating"),
) -> str:
    """
    Rate a provider response for quality tracking.

    Claude should call this after evaluating a response from any provider.
    Ratings are stored and used for provider comparison in get_provider_stats.

    Scale: 1=Poor, 2=Below Average, 3=Average, 4=Good, 5=Excellent
    """
    if any(r < 1 or r > 5 for r in [helpfulness, accuracy, completeness]):
        return "Error: All ratings must be between 1 and 5."

    success = await usage_db.record_rating(
        task_id=task_id,
        helpfulness=helpfulness,
        accuracy=accuracy,
        completeness=completeness,
        notes=notes,
    )

    if not success:
        return f"Error: No usage record found for task_id '{task_id}'. Make sure the task completed."

    overall = (helpfulness + accuracy + completeness) / 3.0
    return f"Rating saved for task {task_id[:8]}. Overall: {overall:.1f}/5.0"


@mcp.tool()
async def get_provider_stats(
    ctx: Context[ServerSession, None],
    period: str = Field(default="this_month", description="Time period: 'this_month', 'last_month', 'all_time', or 'YYYY-MM'"),
    provider: str | None = Field(default=None, description="Filter by specific provider"),
) -> str:
    """
    Get usage statistics for AI providers.

    Returns:
    - Calls per provider/model
    - Total tokens used
    - Average quality rating
    - Estimated costs
    - Success rate
    """
    stats = await usage_db.get_provider_stats(period=period, provider=provider)

    if not stats:
        return f"No usage data found for period: {period}"

    lines = [f"Provider Statistics ({period}):", "=" * 50]

    for s in stats:
        lines.append(f"\n{s['provider_name']} ({s['model_name']}):")
        lines.append(f"  Calls: {s['call_count']} ({s['success_count']} successful)")
        if s['total_tokens']:
            lines.append(f"  Tokens: {s['total_tokens']:,}")
        if s['avg_rating']:
            lines.append(f"  Avg Rating: {s['avg_rating']:.1f}/5.0")
        else:
            lines.append("  Avg Rating: Not rated yet")
        if s['total_cost']:
            lines.append(f"  Est. Cost: ${s['total_cost']:.4f}")
        if s['avg_duration']:
            lines.append(f"  Avg Duration: {s['avg_duration']:.1f}s")

    return "\n".join(lines)


@mcp.tool()
async def add_provider(
    ctx: Context[ServerSession, None],
    name: str = Field(description="Provider name (e.g., 'kimi', 'deepseek')"),
    provider_type: str = Field(description="Provider type: 'openai_api' or 'openrouter'"),
    base_url: str = Field(description="API base URL (e.g., 'https://api.moonshot.ai/v1')"),
    api_key_env: str = Field(description="Environment variable name for API key (e.g., 'KIMI_API_KEY')"),
    default_model: str = Field(description="Default model name (e.g., 'kimi-k2-turbo-preview')"),
    cost_per_1k_input: float | None = Field(default=None, description="Cost per 1k input tokens (USD)"),
    cost_per_1k_output: float | None = Field(default=None, description="Cost per 1k output tokens (USD)"),
    site_url: str | None = Field(default=None, description="For OpenRouter: Site URL (HTTP-Referer header)"),
    app_name: str | None = Field(default=None, description="For OpenRouter: App name (X-Title header)"),
) -> str:
    """
    Add a new API provider to ~/.owlex/providers.json.

    Example:
    - name: "kimi"
    - provider_type: "openai_api"
    - base_url: "https://api.moonshot.ai/v1"
    - api_key_env: "KIMI_API_KEY"
    - default_model: "kimi-k2-turbo-preview"
    """
    success = registry.add_provider(
        name=name,
        provider_type=provider_type,
        base_url=base_url,
        api_key_env=api_key_env,
        default_model=default_model,
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
        site_url=site_url,
        app_name=app_name,
    )

    if success:
        # Check if API key is set
        provider = registry.get_provider(name)
        if provider and provider.is_available:
            return f"Provider '{name}' added and ready. Use call_provider(provider='{name}', ...) to test."
        else:
            return f"Provider '{name}' added but API key not found. Set {api_key_env} in your environment."
    else:
        return f"Failed to add provider '{name}'."


def main():
    """Entry point for owlex-server command."""
    async def run_with_cleanup():
        # Initialize database and registry
        await init_database()
        registry.initialize()

        cleanup_task = asyncio.create_task(cleanup_old_tasks())
        try:
            await mcp.run_stdio_async()
        finally:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    asyncio.run(run_with_cleanup())


if __name__ == "__main__":
    main()
