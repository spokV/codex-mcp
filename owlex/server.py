#!/usr/bin/env python3
"""
MCP Server for Codex CLI and Gemini CLI Integration
Allows Claude Code to start/resume sessions with Codex or Gemini for advice
"""

import asyncio
import json
import os
import re
import sys
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession


# Configuration - Codex
BYPASS_APPROVALS = os.environ.get("CODEX_BYPASS_APPROVALS", "false").lower() == "true"
CLEAN_OUTPUT = os.environ.get("CODEX_CLEAN_OUTPUT", "true").lower() == "true"

# Configuration - Gemini
# YOLO mode: false = read-only (safe), true = full access (can write/execute)
GEMINI_YOLO_MODE = os.environ.get("GEMINI_YOLO_MODE", "false").lower() == "true"
GEMINI_CLEAN_OUTPUT = os.environ.get("GEMINI_CLEAN_OUTPUT", "true").lower() == "true"

DEFAULT_TIMEOUT = 300


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
    # Streaming support
    output_lines: list[str] = field(default_factory=list)
    stream_complete: bool = False


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
    if not CLEAN_OUTPUT:
        return raw_output
    cleaned = raw_output
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


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


async def _read_stream_lines(
    stream: asyncio.StreamReader,
    task: Task,
    stream_name: str,
) -> str:
    """Read stream line-by-line, storing lines and emitting notifications."""
    lines = []
    while True:
        try:
            line = await stream.readline()
            if not line:
                break
            decoded = line.decode('utf-8', errors='replace').rstrip('\n\r')
            lines.append(decoded)
            task.output_lines.append(f"[{stream_name}] {decoded}")
            # Emit real-time notification if context available
            if task.context:
                try:
                    await task.context.info(f"[{task.task_id[:8]}] {decoded}")
                except Exception:
                    pass  # Don't fail on notification errors
        except Exception:
            break
    return '\n'.join(lines)


async def _run_command_task(
    task_id: str,
    command: list[str],
    prompt: str,
    output_cleaner: Callable[[str, str], str],
    output_prefix: str,
    cwd: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    not_found_hint: str | None = None,
    stream: bool = False,
):
    """
    Generic helper to run a subprocess command for a task.
    Handles process creation, timeouts, output cleaning, and status updates.

    If stream=True, reads stdout/stderr line-by-line with real-time notifications.
    """
    task = tasks[task_id]

    try:
        task.status = "running"
        task.output_lines = []
        task.stream_complete = False

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        task.process = process

        # Write stdin if provided
        if prompt:
            process.stdin.write(prompt.encode('utf-8'))
            await process.stdin.drain()
        process.stdin.close()

        if stream:
            # Streaming mode: read line-by-line with notifications
            try:
                async def read_with_timeout():
                    stdout_task = asyncio.create_task(
                        _read_stream_lines(process.stdout, task, "stdout")
                    )
                    stderr_task = asyncio.create_task(
                        _read_stream_lines(process.stderr, task, "stderr")
                    )
                    stdout_text, stderr_text = await asyncio.gather(stdout_task, stderr_task)
                    await process.wait()
                    return stdout_text, stderr_text

                stdout_text, stderr_text = await asyncio.wait_for(
                    read_with_timeout(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                task.status = "failed"
                task.error = f"{command[0]} command timed out after {timeout} seconds"
                task.completion_time = datetime.now()
                task.stream_complete = True
                await _emit_task_notification(task)
                return
        else:
            # Buffered mode: use communicate()
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input=prompt.encode('utf-8') if prompt else None),
                    timeout=timeout
                )
                stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                task.status = "failed"
                task.error = f"{command[0]} command timed out after {timeout} seconds"
                task.completion_time = datetime.now()
                await _emit_task_notification(task)
                return

        task.stream_complete = True

        if process.returncode == 0:
            cleaned_output = output_cleaner(stdout_text, prompt)
            task.result = f"{output_prefix}:\n\n{cleaned_output}"
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
            task.error = f"{command[0]} failed (exit code {process.returncode}):\n\n" + ("\n\n".join(error_output) or "No output")
            task.completion_time = datetime.now()
            await _emit_task_notification(task)

    except asyncio.CancelledError:
        task.status = "cancelled"
        task.completion_time = datetime.now()
        task.stream_complete = True
        if task.process:
            try:
                task.process.kill()
                await task.process.wait()
            except:
                pass
        await _emit_task_notification(task)
        raise

    except FileNotFoundError as e:
        cmd_name = command[0]
        if e.filename == cmd_name:
            hint = not_found_hint or "Please ensure it is installed and in your PATH."
            task.error = f"Error: '{cmd_name}' command not found. {hint}"
        else:
            task.error = f"Error: {e}"
        task.status = "failed"
        task.completion_time = datetime.now()
        task.stream_complete = True
        await _emit_task_notification(task)

    except Exception as e:
        task.status = "failed"
        task.error = f"Error executing {command[0]}: {str(e)}"
        task.completion_time = datetime.now()
        task.stream_complete = True
        await _emit_task_notification(task)


async def _run_codex_resume(
    task_id: str,
    session_ref: str,
    prompt: str,
    working_directory: str | None = None,
    enable_search: bool = False
):
    """Background coroutine that runs Codex resume and updates task status"""
    # Build command: codex exec [options] resume [--last | session-id] -
    full_command = ["codex", "exec", "--skip-git-repo-check"]

    # Add --cd and --search before subcommand
    if working_directory:
        full_command.extend(["--cd", working_directory])
    if enable_search:
        full_command.append("--search")

    if BYPASS_APPROVALS:
        full_command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        full_command.append("--full-auto")

    # Add resume subcommand
    full_command.append("resume")
    if session_ref == "--last":
        full_command.append("--last")
    else:
        full_command.append(session_ref)
    full_command.append("-")  # Read prompt from stdin

    await _run_command_task(
        task_id=task_id,
        command=full_command,
        prompt=prompt,
        output_cleaner=clean_codex_output,
        output_prefix="Codex Resume Output",
        not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
    )


async def _run_codex_exec(
    task_id: str,
    prompt: str,
    working_directory: str | None = None,
    enable_search: bool = False,
    stream: bool = True,
):
    """Background coroutine that runs Codex exec (new session) and updates task status"""
    # Build command: codex exec [options] -
    full_command = ["codex", "exec", "--skip-git-repo-check"]

    if working_directory:
        full_command.extend(["--cd", working_directory])
    if enable_search:
        full_command.append("--search")

    if BYPASS_APPROVALS:
        full_command.append("--dangerously-bypass-approvals-and-sandbox")
    else:
        full_command.append("--full-auto")

    full_command.append("-")  # Read prompt from stdin

    await _run_command_task(
        task_id=task_id,
        command=full_command,
        prompt=prompt,
        output_cleaner=clean_codex_output,
        output_prefix="Codex Output",
        not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
        stream=stream,
    )


async def _run_gemini_exec(
    task_id: str,
    prompt: str,
    working_directory: str | None = None,
    stream: bool = True,
):
    """Background coroutine that runs Gemini CLI (new session) and updates task status"""
    # Build command: gemini [options] "prompt"
    full_command = ["gemini"]

    if GEMINI_YOLO_MODE:
        full_command.extend(["--approval-mode", "yolo"])

    if working_directory:
        full_command.extend(["--include-directories", working_directory])

    full_command.append(prompt)  # Gemini takes prompt as argument, not stdin

    await _run_command_task(
        task_id=task_id,
        command=full_command,
        prompt="",
        output_cleaner=clean_gemini_output,
        output_prefix="Gemini Output",
        cwd=working_directory,
        not_found_hint="Please ensure Gemini CLI is installed (npm install -g @google/gemini-cli).",
        stream=stream,
    )


async def _run_gemini_resume(
    task_id: str,
    session_ref: str,
    prompt: str,
    working_directory: str | None = None,
):
    """Background coroutine that runs Gemini resume and updates task status"""
    # Build command: gemini [options] -r <session> "prompt"
    full_command = ["gemini"]

    if GEMINI_YOLO_MODE:
        full_command.extend(["--approval-mode", "yolo"])

    if working_directory:
        full_command.extend(["--include-directories", working_directory])

    # Add resume flag
    full_command.extend(["-r", session_ref])
    full_command.append(prompt)

    await _run_command_task(
        task_id=task_id,
        command=full_command,
        prompt="",
        output_cleaner=clean_gemini_output,
        output_prefix="Gemini Resume Output",
        cwd=working_directory,
        not_found_hint="Please ensure Gemini CLI is installed (npm install -g @google/gemini-cli).",
    )


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
async def wait_for_task(task_id: str, timeout: int = DEFAULT_TIMEOUT) -> str:
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


def _extract_content(result: str | None, prefix: str) -> str:
    """Extract raw content from task result, stripping the prefix."""
    if not result:
        return ""
    # Strip prefix like "Codex Output:\n\n" or "Gemini Output:\n\n"
    if result.startswith(prefix):
        return result[len(prefix):].strip()
    return result.strip()


def _build_agent_response(task: Task, agent: str) -> dict[str, Any]:
    """Build structured response for an agent."""
    prefix_map = {
        "codex": "Codex Output:\n\n",
        "gemini": "Gemini Output:\n\n",
    }
    prefix = prefix_map.get(agent, "")

    return {
        "agent": agent,
        "status": task.status,
        "content": _extract_content(task.result, prefix) if task.status == "completed" else None,
        "error": task.error if task.status == "failed" else None,
        "duration_seconds": (
            (task.completion_time - task.start_time).total_seconds()
            if task.completion_time else None
        ),
        "task_id": task.task_id,
    }


@mcp.tool()
async def council_ask(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or task to send to the council"),
    working_directory: str | None = Field(default=None, description="Working directory for context"),
    deliberate: bool = Field(default=True, description="If true, share answers between agents for a second round of deliberation"),
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Timeout per agent in seconds"),
) -> str:
    """
    Ask the council (Codex + Gemini) a question and collect their answers.

    Sends the prompt to both Codex and Gemini in parallel, waits for responses,
    and returns all answers for the MCP client (Claude Code) to synthesize.

    If deliberate=True, shares all initial answers with each agent for a second
    round, allowing them to revise their opinions after seeing others' responses.
    """
    if not prompt or not prompt.strip():
        return json.dumps({"error": "'prompt' parameter is required."})

    if working_directory:
        working_directory = os.path.expanduser(working_directory)
        if not os.path.isdir(working_directory):
            return json.dumps({"error": f"working_directory '{working_directory}' does not exist."})

    prompt = prompt.strip()
    council_start = datetime.now()

    # === Round 1: Parallel initial queries ===
    await ctx.info("[council] Starting round 1: querying Codex and Gemini in parallel...")

    # Create tasks for both agents
    codex_task_id = str(uuid.uuid4())
    gemini_task_id = str(uuid.uuid4())

    codex_task = Task(
        task_id=codex_task_id,
        status="pending",
        command="council_codex",
        args={"prompt": prompt, "working_directory": working_directory},
        start_time=datetime.now(),
        context=ctx
    )
    gemini_task = Task(
        task_id=gemini_task_id,
        status="pending",
        command="council_gemini",
        args={"prompt": prompt, "working_directory": working_directory},
        start_time=datetime.now(),
        context=ctx
    )

    tasks[codex_task_id] = codex_task
    tasks[gemini_task_id] = gemini_task

    # Start both in parallel with streaming enabled
    codex_task.async_task = asyncio.create_task(_run_codex_exec(
        codex_task_id, prompt, working_directory, enable_search=False
    ))
    gemini_task.async_task = asyncio.create_task(_run_gemini_exec(
        gemini_task_id, prompt, working_directory
    ))

    # Wait for both to complete
    await asyncio.gather(
        asyncio.wait_for(asyncio.shield(codex_task.async_task), timeout=timeout),
        asyncio.wait_for(asyncio.shield(gemini_task.async_task), timeout=timeout),
        return_exceptions=True
    )

    await ctx.info("[council] Round 1 complete.")

    # Build structured response
    response: dict[str, Any] = {
        "prompt": prompt,
        "working_directory": working_directory,
        "deliberation": deliberate,
        "round_1": {
            "codex": _build_agent_response(codex_task, "codex"),
            "gemini": _build_agent_response(gemini_task, "gemini"),
        },
    }

    # === Round 2: Deliberation (optional) ===
    if deliberate:
        await ctx.info("[council] Starting round 2: deliberation phase...")

        # Get content for deliberation prompt
        codex_content = response["round_1"]["codex"]["content"] or response["round_1"]["codex"]["error"] or "(no response)"
        gemini_content = response["round_1"]["gemini"]["content"] or response["round_1"]["gemini"]["error"] or "(no response)"

        deliberation_prompt = f"""You previously answered a question. Now review all council members' answers and provide your revised opinion.

ORIGINAL QUESTION:
{prompt}

CODEX'S ANSWER:
{codex_content}

GEMINI'S ANSWER:
{gemini_content}

Please provide your revised answer after considering the other perspectives. Note any points of agreement or disagreement."""

        # Create new tasks for deliberation
        codex_delib_id = str(uuid.uuid4())
        gemini_delib_id = str(uuid.uuid4())

        codex_delib_task = Task(
            task_id=codex_delib_id,
            status="pending",
            command="council_codex_delib",
            args={"prompt": deliberation_prompt, "working_directory": working_directory},
            start_time=datetime.now(),
            context=ctx
        )
        gemini_delib_task = Task(
            task_id=gemini_delib_id,
            status="pending",
            command="council_gemini_delib",
            args={"prompt": deliberation_prompt, "working_directory": working_directory},
            start_time=datetime.now(),
            context=ctx
        )

        tasks[codex_delib_id] = codex_delib_task
        tasks[gemini_delib_id] = gemini_delib_task

        # Start deliberation round in parallel
        codex_delib_task.async_task = asyncio.create_task(_run_codex_exec(
            codex_delib_id, deliberation_prompt, working_directory, enable_search=False
        ))
        gemini_delib_task.async_task = asyncio.create_task(_run_gemini_exec(
            gemini_delib_id, deliberation_prompt, working_directory
        ))

        await asyncio.gather(
            asyncio.wait_for(asyncio.shield(codex_delib_task.async_task), timeout=timeout),
            asyncio.wait_for(asyncio.shield(gemini_delib_task.async_task), timeout=timeout),
            return_exceptions=True
        )

        await ctx.info("[council] Round 2 complete.")

        response["round_2"] = {
            "codex": _build_agent_response(codex_delib_task, "codex"),
            "gemini": _build_agent_response(gemini_delib_task, "gemini"),
        }

    # Add metadata
    response["metadata"] = {
        "total_duration_seconds": (datetime.now() - council_start).total_seconds(),
        "rounds": 2 if deliberate else 1,
    }

    return json.dumps(response, indent=2, default=str)


def main():
    """Entry point for owlex-server command."""
    async def run_with_cleanup():
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
