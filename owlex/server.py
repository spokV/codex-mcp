#!/usr/bin/env python3
"""
MCP Server for Codex CLI and Gemini CLI Integration
Allows Claude Code to start/resume sessions with Codex or Gemini for advice
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from .models import (
    TaskResponse,
    ClaudeOpinion,
    CouncilResponse,
    CouncilRound,
    CouncilMetadata,
)
from .engine import (
    engine,
    build_agent_response,
    DEFAULT_TIMEOUT,
)


# Initialize FastMCP server
mcp = FastMCP("owlex-server")


def _log(msg: str):
    """Log progress to stderr for CLI visibility."""
    print(msg, file=sys.stderr, flush=True)


def _validate_working_directory(working_directory: str | None) -> tuple[str | None, str | None]:
    """Validate and expand working directory. Returns (expanded_path, error_message)."""
    if not working_directory:
        return None, None
    expanded = os.path.expanduser(working_directory)
    if not os.path.isdir(expanded):
        return None, f"working_directory '{working_directory}' does not exist or is not a directory."
    return expanded, None


async def _kill_task_subprocess(task):
    """Kill subprocess for a task if it's still running."""
    if task.process and task.process.returncode is None:
        try:
            task.process.kill()
            await task.process.wait()
        except Exception:
            pass
    if task.async_task and not task.async_task.done():
        task.async_task.cancel()
        try:
            await task.async_task
        except asyncio.CancelledError:
            pass


async def _run_council_tasks(tasks: list, timeout: int, log_func) -> list[Exception | None]:
    """
    Run council tasks with proper timeout handling and cleanup.
    Returns list of exceptions (or None for successful tasks).
    """
    results = await asyncio.gather(
        *[t.async_task for t in tasks],
        return_exceptions=True
    )

    # Check for timeouts and cleanup any still-running processes
    for i, (task, result) in enumerate(zip(tasks, results)):
        if isinstance(result, asyncio.TimeoutError):
            log_func(f"{task.command} timed out after {timeout}s")
            await _kill_task_subprocess(task)
            task.status = "failed"
            task.error = f"Timed out after {timeout} seconds"
            task.completion_time = datetime.now()
        elif isinstance(result, Exception):
            log_func(f"{task.command} failed: {result}")
            await _kill_task_subprocess(task)

    return results


# === Codex Tools ===

@mcp.tool()
async def start_codex_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Codex (--cd flag)"),
    enable_search: bool = Field(default=False, description="Enable web search (--search flag)")
) -> str:
    """Start a new Codex session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.").model_dump_json()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error).model_dump_json()

    task = engine.create_task(
        command="codex_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory, "enable_search": enable_search},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_codex_exec(
        task, prompt.strip(), working_directory, enable_search
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Codex session started. Use wait_for_task to get result.",
    ).model_dump_json()


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
        return TaskResponse(success=False, error="'prompt' parameter is required.").model_dump_json()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error).model_dump_json()

    use_last = not session_id or not session_id.strip()
    session_ref = "--last" if use_last else session_id.strip()

    task = engine.create_task(
        command="codex_resume",
        args={"session_id": session_ref, "prompt": prompt.strip(), "working_directory": working_directory, "enable_search": enable_search},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_codex_resume(
        task, session_ref, prompt.strip(), working_directory, enable_search
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Codex resume started{' (last session)' if use_last else f' for session {session_id}'}. Use wait_for_task to get result.",
    ).model_dump_json()


# === Gemini Tools ===

@mcp.tool()
async def start_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> str:
    """Start a new Gemini CLI session (no prior context)."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.").model_dump_json()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error).model_dump_json()

    task = engine.create_task(
        command="gemini_exec",
        args={"prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_gemini_exec(
        task, prompt.strip(), working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message="Gemini session started. Use wait_for_task to get result.",
    ).model_dump_json()


@mcp.tool()
async def resume_gemini_session(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or request to send to the resumed session"),
    session_ref: str = Field(default="latest", description="Session to resume: 'latest' for most recent, or index number"),
    working_directory: str | None = Field(default=None, description="Working directory for Gemini context"),
) -> str:
    """Resume an existing Gemini CLI session with full conversation history."""
    if not prompt or not prompt.strip():
        return TaskResponse(success=False, error="'prompt' parameter is required.").model_dump_json()

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return TaskResponse(success=False, error=error).model_dump_json()

    task = engine.create_task(
        command="gemini_resume",
        args={"session_ref": session_ref, "prompt": prompt.strip(), "working_directory": working_directory},
        context=ctx,
    )

    task.async_task = asyncio.create_task(engine.run_gemini_resume(
        task, session_ref, prompt.strip(), working_directory
    ))

    return TaskResponse(
        success=True,
        task_id=task.task_id,
        status=task.status,
        message=f"Gemini resume started (session: {session_ref}). Use wait_for_task to get result.",
    ).model_dump_json()


# === Task Management Tools ===

@mcp.tool()
async def get_task_result(task_id: str) -> str:
    """
    Get the result of a task (Codex or Gemini).

    Args:
        task_id: The task ID returned by start/resume session
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.").model_dump_json()

    if task.status == "pending":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            message="Task is still pending.",
        ).model_dump_json()
    elif task.status == "running":
        elapsed = (datetime.now() - task.start_time).total_seconds()
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            message=f"Task is still running ({elapsed:.1f}s elapsed).",
        ).model_dump_json()
    elif task.status == "completed":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            content=task.result,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump_json()
    elif task.status == "failed":
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump_json()
    else:
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
        ).model_dump_json()


@mcp.tool()
async def wait_for_task(task_id: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """
    Wait for a task to complete and return its result.

    Args:
        task_id: The task ID to wait for
        timeout: Maximum seconds to wait (default: 300)
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.").model_dump_json()

    if task.status in ["completed", "failed", "cancelled"]:
        if task.status == "completed":
            return TaskResponse(
                success=True,
                task_id=task_id,
                status=task.status,
                content=task.result,
                duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
            ).model_dump_json()
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=task.error,
        ).model_dump_json()

    if task.async_task:
        try:
            await asyncio.wait_for(asyncio.shield(task.async_task), timeout=timeout)
        except asyncio.TimeoutError:
            return TaskResponse(
                success=False,
                task_id=task_id,
                status="timeout",
                error=f"Task still running after {timeout}s. Use get_task_result to check later.",
            ).model_dump_json()
        except Exception as e:
            return TaskResponse(
                success=False,
                task_id=task_id,
                error=f"Task failed: {str(e)}",
            ).model_dump_json()

    if task.status == "completed":
        return TaskResponse(
            success=True,
            task_id=task_id,
            status=task.status,
            content=task.result,
            duration_seconds=(task.completion_time - task.start_time).total_seconds() if task.completion_time else None,
        ).model_dump_json()

    return TaskResponse(
        success=False,
        task_id=task_id,
        status=task.status,
        error=task.error,
    ).model_dump_json()


@mcp.tool()
async def list_tasks(
    status_filter: str | None = Field(default=None, description="Filter by status: pending, running, completed, failed, cancelled"),
    limit: int = Field(default=20, description="Maximum number of tasks to return"),
) -> str:
    """
    List all tracked tasks with their current status.

    Args:
        status_filter: Optional filter by task status
        limit: Maximum number of tasks to return (default: 20)
    """
    tasks_list = []
    for task_id, task in list(engine.tasks.items())[-limit:]:
        if status_filter and task.status != status_filter:
            continue
        elapsed = (datetime.now() - task.start_time).total_seconds()
        tasks_list.append({
            "task_id": task_id,
            "command": task.command,
            "status": task.status,
            "elapsed_seconds": round(elapsed, 1),
            "has_result": task.result is not None,
            "has_error": task.error is not None,
        })

    return json.dumps({
        "success": True,
        "count": len(tasks_list),
        "tasks": tasks_list,
    }, indent=2)


@mcp.tool()
async def cancel_task(task_id: str) -> str:
    """
    Cancel a running task and kill its subprocess.

    Args:
        task_id: The task ID to cancel
    """
    task = engine.get_task(task_id)
    if not task:
        return TaskResponse(success=False, error=f"Task '{task_id}' not found.").model_dump_json()

    if task.status in ["completed", "failed", "cancelled"]:
        return TaskResponse(
            success=False,
            task_id=task_id,
            status=task.status,
            error=f"Task already {task.status}, cannot cancel.",
        ).model_dump_json()

    # Kill the subprocess and cancel the async task
    await _kill_task_subprocess(task)
    task.status = "cancelled"
    task.error = "Cancelled by user"
    task.completion_time = datetime.now()

    return TaskResponse(
        success=True,
        task_id=task_id,
        status=task.status,
        message="Task cancelled successfully.",
    ).model_dump_json()


# === Council Tool ===

@mcp.tool()
async def council_ask(
    ctx: Context[ServerSession, None],
    prompt: str = Field(description="The question or task to send to the council"),
    claude_opinion: str | None = Field(default=None, description="Claude's initial opinion to share with the council"),
    working_directory: str | None = Field(default=None, description="Working directory for context"),
    deliberate: bool = Field(default=True, description="If true, share answers between agents for a second round of deliberation"),
    timeout: int = Field(default=DEFAULT_TIMEOUT, description="Timeout per agent in seconds"),
) -> str:
    """
    Ask the council (Codex + Gemini) a question and collect their answers.

    Sends the prompt to both Codex and Gemini in parallel, waits for responses,
    and returns all answers for the MCP client (Claude Code) to synthesize.

    If claude_opinion is provided, it will be shared with other council members
    during deliberation so they can consider Claude's perspective.

    If deliberate=True, shares all answers (including Claude's) with each agent
    for a second round, allowing them to revise after seeing others' responses.
    """
    if not prompt or not prompt.strip():
        return json.dumps({"error": "'prompt' parameter is required."})

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return json.dumps({"error": error})

    prompt = prompt.strip()
    council_start = datetime.now()
    log_entries: list[str] = []

    def log(msg: str):
        """Add to log and print to stderr."""
        log_entries.append(msg)
        _log(msg)

    # === Round 1: Parallel initial queries ===
    if claude_opinion and claude_opinion.strip():
        log(f"Claude's opinion received ({len(claude_opinion)} chars)")
    log("Round 1: querying Codex and Gemini...")

    codex_task = engine.create_task(
        command="council_codex",
        args={"prompt": prompt, "working_directory": working_directory},
        context=ctx,
    )
    gemini_task = engine.create_task(
        command="council_gemini",
        args={"prompt": prompt, "working_directory": working_directory},
        context=ctx,
    )

    round1_start = datetime.now()

    async def run_and_notify_codex():
        await engine.run_codex_exec(codex_task, prompt, working_directory, enable_search=False)
        elapsed = (datetime.now() - round1_start).total_seconds()
        status = "completed" if codex_task.status == "completed" else "failed"
        log(f"Codex {status} ({elapsed:.1f}s)")

    async def run_and_notify_gemini():
        await engine.run_gemini_exec(gemini_task, prompt, working_directory)
        elapsed = (datetime.now() - round1_start).total_seconds()
        status = "completed" if gemini_task.status == "completed" else "failed"
        log(f"Gemini {status} ({elapsed:.1f}s)")

    codex_task.async_task = asyncio.create_task(run_and_notify_codex())
    gemini_task.async_task = asyncio.create_task(run_and_notify_gemini())

    # Wait for both tasks with timeout, then cleanup any that didn't complete
    done, pending = await asyncio.wait(
        [codex_task.async_task, gemini_task.async_task],
        timeout=timeout,
        return_when=asyncio.ALL_COMPLETED
    )

    # Kill subprocesses for any tasks that timed out
    for task in [codex_task, gemini_task]:
        if task.async_task in pending:
            log(f"{task.command} timed out")
            await _kill_task_subprocess(task)
            task.status = "failed"
            task.error = f"Timed out after {timeout} seconds"
            task.completion_time = datetime.now()

    round1_elapsed = (datetime.now() - round1_start).total_seconds()
    log(f"Round 1 complete ({round1_elapsed:.1f}s)")

    round_1 = CouncilRound(
        codex=build_agent_response(codex_task, "codex"),
        gemini=build_agent_response(gemini_task, "gemini"),
    )

    round_2 = None

    # === Round 2: Deliberation (optional) ===
    if deliberate:
        log("Round 2: deliberation phase...")

        codex_content = round_1.codex.content or round_1.codex.error or "(no response)"
        gemini_content = round_1.gemini.content or round_1.gemini.error or "(no response)"
        claude_content = claude_opinion.strip() if claude_opinion else None

        # Build deliberation prompt with all available opinions
        deliberation_parts = [
            "You previously answered a question. Now review all council members' answers and provide your revised opinion.",
            "",
            "ORIGINAL QUESTION:",
            prompt,
        ]

        if claude_content:
            deliberation_parts.extend(["", "CLAUDE'S ANSWER:", claude_content])

        deliberation_parts.extend([
            "",
            "CODEX'S ANSWER:",
            codex_content,
            "",
            "GEMINI'S ANSWER:",
            gemini_content,
            "",
            "Please provide your revised answer after considering the other perspectives. Note any points of agreement or disagreement.",
        ])

        deliberation_prompt = "\n".join(deliberation_parts)

        codex_delib_task = engine.create_task(
            command="council_codex_delib",
            args={"prompt": deliberation_prompt, "working_directory": working_directory},
            context=ctx,
        )
        gemini_delib_task = engine.create_task(
            command="council_gemini_delib",
            args={"prompt": deliberation_prompt, "working_directory": working_directory},
            context=ctx,
        )

        round2_start = datetime.now()

        async def run_and_notify_codex_delib():
            await engine.run_codex_exec(codex_delib_task, deliberation_prompt, working_directory, enable_search=False)
            elapsed = (datetime.now() - round2_start).total_seconds()
            log(f"Codex revised ({elapsed:.1f}s)")

        async def run_and_notify_gemini_delib():
            await engine.run_gemini_exec(gemini_delib_task, deliberation_prompt, working_directory)
            elapsed = (datetime.now() - round2_start).total_seconds()
            log(f"Gemini revised ({elapsed:.1f}s)")

        codex_delib_task.async_task = asyncio.create_task(run_and_notify_codex_delib())
        gemini_delib_task.async_task = asyncio.create_task(run_and_notify_gemini_delib())

        # Wait for both tasks with timeout, then cleanup any that didn't complete
        done, pending = await asyncio.wait(
            [codex_delib_task.async_task, gemini_delib_task.async_task],
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )

        # Kill subprocesses for any tasks that timed out
        for task in [codex_delib_task, gemini_delib_task]:
            if task.async_task in pending:
                log(f"{task.command} timed out")
                await _kill_task_subprocess(task)
                task.status = "failed"
                task.error = f"Timed out after {timeout} seconds"
                task.completion_time = datetime.now()

        round2_elapsed = (datetime.now() - round2_start).total_seconds()
        log(f"Round 2 complete ({round2_elapsed:.1f}s)")

        round_2 = CouncilRound(
            codex=build_agent_response(codex_delib_task, "codex"),
            gemini=build_agent_response(gemini_delib_task, "gemini"),
        )

    # Build Claude opinion object if provided
    claude_opinion_obj = None
    if claude_opinion and claude_opinion.strip():
        claude_opinion_obj = ClaudeOpinion(
            content=claude_opinion.strip(),
            provided_at=council_start.isoformat(),
        )

    response = CouncilResponse(
        prompt=prompt,
        working_directory=working_directory,
        deliberation=deliberate,
        claude_opinion=claude_opinion_obj,
        round_1=round_1,
        round_2=round_2,
        metadata=CouncilMetadata(
            total_duration_seconds=(datetime.now() - council_start).total_seconds(),
            rounds=2 if deliberate else 1,
            log=log_entries,
        ),
    )

    return response.model_dump_json(indent=2)


def main():
    """Entry point for owlex-server command."""
    async def run_with_cleanup():
        engine.start_cleanup_loop()
        try:
            await mcp.run_stdio_async()
        finally:
            engine.stop_cleanup_loop()

    asyncio.run(run_with_cleanup())


if __name__ == "__main__":
    main()
