#!/usr/bin/env python3
"""
MCP Server for Codex CLI and Gemini CLI Integration
Allows Claude Code to start/resume sessions with Codex or Gemini for advice
"""

import asyncio
import json
import os
from datetime import datetime

from pydantic import Field
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

from .models import (
    TaskResponse,
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


def _validate_working_directory(working_directory: str | None) -> tuple[str | None, str | None]:
    """Validate and expand working directory. Returns (expanded_path, error_message)."""
    if not working_directory:
        return None, None
    expanded = os.path.expanduser(working_directory)
    if not os.path.isdir(expanded):
        return None, f"working_directory '{working_directory}' does not exist or is not a directory."
    return expanded, None


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


# === Council Tool ===

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

    working_directory, error = _validate_working_directory(working_directory)
    if error:
        return json.dumps({"error": error})

    prompt = prompt.strip()
    council_start = datetime.now()

    # === Round 1: Parallel initial queries ===
    await ctx.info("[council] Starting round 1: querying Codex and Gemini in parallel...")

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

    codex_task.async_task = asyncio.create_task(engine.run_codex_exec(
        codex_task, prompt, working_directory, enable_search=False
    ))
    gemini_task.async_task = asyncio.create_task(engine.run_gemini_exec(
        gemini_task, prompt, working_directory
    ))

    await asyncio.gather(
        asyncio.wait_for(asyncio.shield(codex_task.async_task), timeout=timeout),
        asyncio.wait_for(asyncio.shield(gemini_task.async_task), timeout=timeout),
        return_exceptions=True
    )

    await ctx.info("[council] Round 1 complete.")

    round_1 = CouncilRound(
        codex=build_agent_response(codex_task, "codex"),
        gemini=build_agent_response(gemini_task, "gemini"),
    )

    round_2 = None

    # === Round 2: Deliberation (optional) ===
    if deliberate:
        await ctx.info("[council] Starting round 2: deliberation phase...")

        codex_content = round_1.codex.content or round_1.codex.error or "(no response)"
        gemini_content = round_1.gemini.content or round_1.gemini.error or "(no response)"

        deliberation_prompt = f"""You previously answered a question. Now review all council members' answers and provide your revised opinion.

ORIGINAL QUESTION:
{prompt}

CODEX'S ANSWER:
{codex_content}

GEMINI'S ANSWER:
{gemini_content}

Please provide your revised answer after considering the other perspectives. Note any points of agreement or disagreement."""

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

        codex_delib_task.async_task = asyncio.create_task(engine.run_codex_exec(
            codex_delib_task, deliberation_prompt, working_directory, enable_search=False
        ))
        gemini_delib_task.async_task = asyncio.create_task(engine.run_gemini_exec(
            gemini_delib_task, deliberation_prompt, working_directory
        ))

        await asyncio.gather(
            asyncio.wait_for(asyncio.shield(codex_delib_task.async_task), timeout=timeout),
            asyncio.wait_for(asyncio.shield(gemini_delib_task.async_task), timeout=timeout),
            return_exceptions=True
        )

        await ctx.info("[council] Round 2 complete.")

        round_2 = CouncilRound(
            codex=build_agent_response(codex_delib_task, "codex"),
            gemini=build_agent_response(gemini_delib_task, "gemini"),
        )

    response = CouncilResponse(
        prompt=prompt,
        working_directory=working_directory,
        deliberation=deliberate,
        round_1=round_1,
        round_2=round_2,
        metadata=CouncilMetadata(
            total_duration_seconds=(datetime.now() - council_start).total_seconds(),
            rounds=2 if deliberate else 1,
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
