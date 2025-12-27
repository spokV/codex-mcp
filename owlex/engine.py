"""
Task execution engine for owlex.
Handles subprocess management, streaming, and task lifecycle.
"""

import asyncio
import os
import re
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from .models import Task, TaskStatus, AgentResponse


# Configuration - Codex
BYPASS_APPROVALS = os.environ.get("CODEX_BYPASS_APPROVALS", "false").lower() == "true"
CLEAN_OUTPUT = os.environ.get("CODEX_CLEAN_OUTPUT", "true").lower() == "true"

# Configuration - Gemini
GEMINI_YOLO_MODE = os.environ.get("GEMINI_YOLO_MODE", "false").lower() == "true"
GEMINI_CLEAN_OUTPUT = os.environ.get("GEMINI_CLEAN_OUTPUT", "true").lower() == "true"

DEFAULT_TIMEOUT = 300


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
    if cleaned.startswith("YOLO mode is enabled."):
        lines = cleaned.split('\n', 2)
        if len(lines) > 2:
            cleaned = lines[2]
        elif len(lines) > 1:
            cleaned = lines[1]
    cleaned = re.sub(r'^Loaded cached credentials\.\n?', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def extract_content(result: str | None, prefix: str) -> str:
    """Extract raw content from task result, stripping the prefix."""
    if not result:
        return ""
    if result.startswith(prefix):
        return result[len(prefix):].strip()
    return result.strip()


def build_agent_response(task: Task, agent: str) -> AgentResponse:
    """Build structured response for an agent."""
    prefix_map = {
        "codex": "Codex Output:\n\n",
        "gemini": "Gemini Output:\n\n",
    }
    prefix = prefix_map.get(agent, "")

    return AgentResponse(
        agent=agent,
        status=task.status,
        content=extract_content(task.result, prefix) if task.status == "completed" else None,
        error=task.error if task.status == "failed" else None,
        duration_seconds=(
            (task.completion_time - task.start_time).total_seconds()
            if task.completion_time else None
        ),
        task_id=task.task_id,
    )


# Type alias for notification callback
NotifyCallback = Callable[[str, str], Any] | None


class TaskEngine:
    """
    Core task execution engine.
    Manages task lifecycle, subprocess execution, and streaming.
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self._cleanup_task: asyncio.Task | None = None

    def start_cleanup_loop(self):
        """Start background task cleanup loop."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_tasks())

    def stop_cleanup_loop(self):
        """Stop background task cleanup loop."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def _cleanup_old_tasks(self):
        """Background task to clean up completed tasks after 5 minutes."""
        while True:
            try:
                await asyncio.sleep(60)
                now = datetime.now()
                tasks_to_remove = [
                    task_id for task_id, task in self.tasks.items()
                    if task.completion_time and (now - task.completion_time) > timedelta(minutes=5)
                ]
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in cleanup_old_tasks: {e}", flush=True)

    def create_task(
        self,
        command: str,
        args: dict,
        context: Any = None,
    ) -> Task:
        """Create and register a new task."""
        task_id = str(uuid.uuid4())
        task = Task(
            task_id=task_id,
            status=TaskStatus.PENDING.value,
            command=command,
            args=args,
            start_time=datetime.now(),
            context=context,
        )
        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    async def _send_notification(self, task: Task, level: str, message: str):
        """Send notification via task context if available."""
        if not task.context:
            return
        handler = getattr(task.context, level, None)
        if not callable(handler):
            handler = getattr(task.context, 'info', None)
        if handler:
            try:
                await asyncio.shield(handler(message))
            except Exception as e:
                print(f"[ERROR] Failed to send {level} notification: {e}", file=sys.stderr, flush=True)

    async def _emit_task_notification(self, task: Task):
        """Emit task completion/failure notification."""
        if not task.context:
            return
        prefix = "[owlex]"
        if task.status == "completed":
            preview = ""
            if task.result:
                lines = task.result.strip().split('\n')
                preview = f": {lines[-1][:100]}" if lines else ""
            await self._send_notification(task, "info", f"{prefix} Task {task.task_id[:8]} completed{preview}")
        elif task.status == "failed":
            error_preview = (task.error or "")[:100]
            await self._send_notification(task, "error", f"{prefix} Task {task.task_id[:8]} failed: {error_preview}")

    async def _read_stream_lines(
        self,
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
                if task.context:
                    try:
                        await task.context.info(f"[{task.task_id[:8]}] {decoded}")
                    except Exception:
                        pass
            except Exception:
                break
        return '\n'.join(lines)

    async def run_command(
        self,
        task: Task,
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
        Run a subprocess command for a task.
        Handles process creation, timeouts, output cleaning, and status updates.
        """
        try:
            task.status = TaskStatus.RUNNING.value
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

            if prompt:
                process.stdin.write(prompt.encode('utf-8'))
                await process.stdin.drain()
            process.stdin.close()

            if stream:
                try:
                    async def read_with_timeout():
                        stdout_task = asyncio.create_task(
                            self._read_stream_lines(process.stdout, task, "stdout")
                        )
                        stderr_task = asyncio.create_task(
                            self._read_stream_lines(process.stderr, task, "stderr")
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
                    task.status = TaskStatus.FAILED.value
                    task.error = f"{command[0]} command timed out after {timeout} seconds"
                    task.completion_time = datetime.now()
                    task.stream_complete = True
                    await self._emit_task_notification(task)
                    return
            else:
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
                    task.status = TaskStatus.FAILED.value
                    task.error = f"{command[0]} command timed out after {timeout} seconds"
                    task.completion_time = datetime.now()
                    await self._emit_task_notification(task)
                    return

            task.stream_complete = True

            if process.returncode == 0:
                cleaned_output = output_cleaner(stdout_text, prompt)
                task.result = f"{output_prefix}:\n\n{cleaned_output}"
                task.status = TaskStatus.COMPLETED.value
                task.completion_time = datetime.now()
                await self._emit_task_notification(task)
            else:
                error_output = []
                if stdout_text.strip():
                    error_output.append(f"stdout:\n{stdout_text}")
                if stderr_text.strip():
                    error_output.append(f"stderr:\n{stderr_text}")
                task.status = TaskStatus.FAILED.value
                task.error = f"{command[0]} failed (exit code {process.returncode}):\n\n" + ("\n\n".join(error_output) or "No output")
                task.completion_time = datetime.now()
                await self._emit_task_notification(task)

        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED.value
            task.completion_time = datetime.now()
            task.stream_complete = True
            if task.process:
                try:
                    task.process.kill()
                    await task.process.wait()
                except:
                    pass
            await self._emit_task_notification(task)
            raise

        except FileNotFoundError as e:
            cmd_name = command[0]
            if e.filename == cmd_name:
                hint = not_found_hint or "Please ensure it is installed and in your PATH."
                task.error = f"Error: '{cmd_name}' command not found. {hint}"
            else:
                task.error = f"Error: {e}"
            task.status = TaskStatus.FAILED.value
            task.completion_time = datetime.now()
            task.stream_complete = True
            await self._emit_task_notification(task)

        except Exception as e:
            task.status = TaskStatus.FAILED.value
            task.error = f"Error executing {command[0]}: {str(e)}"
            task.completion_time = datetime.now()
            task.stream_complete = True
            await self._emit_task_notification(task)

    # === Codex Runners ===

    async def run_codex_exec(
        self,
        task: Task,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        stream: bool = True,
    ):
        """Run Codex exec (new session)."""
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.append("--search")

        if BYPASS_APPROVALS:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            full_command.append("--full-auto")

        full_command.append("-")

        await self.run_command(
            task=task,
            command=full_command,
            prompt=prompt,
            output_cleaner=clean_codex_output,
            output_prefix="Codex Output",
            not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
            stream=stream,
        )

    async def run_codex_resume(
        self,
        task: Task,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
    ):
        """Run Codex resume (existing session)."""
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.append("--search")

        if BYPASS_APPROVALS:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            full_command.append("--full-auto")

        full_command.append("resume")
        if session_ref == "--last":
            full_command.append("--last")
        else:
            full_command.append(session_ref)
        full_command.append("-")

        await self.run_command(
            task=task,
            command=full_command,
            prompt=prompt,
            output_cleaner=clean_codex_output,
            output_prefix="Codex Resume Output",
            not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
        )

    # === Gemini Runners ===

    async def run_gemini_exec(
        self,
        task: Task,
        prompt: str,
        working_directory: str | None = None,
        stream: bool = True,
    ):
        """Run Gemini CLI (new session)."""
        full_command = ["gemini"]

        if GEMINI_YOLO_MODE:
            full_command.extend(["--approval-mode", "yolo"])

        if working_directory:
            full_command.extend(["--include-directories", working_directory])

        full_command.append(prompt)

        await self.run_command(
            task=task,
            command=full_command,
            prompt="",
            output_cleaner=clean_gemini_output,
            output_prefix="Gemini Output",
            cwd=working_directory,
            not_found_hint="Please ensure Gemini CLI is installed (npm install -g @google/gemini-cli).",
            stream=stream,
        )

    async def run_gemini_resume(
        self,
        task: Task,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
    ):
        """Run Gemini resume (existing session)."""
        full_command = ["gemini"]

        if GEMINI_YOLO_MODE:
            full_command.extend(["--approval-mode", "yolo"])

        if working_directory:
            full_command.extend(["--include-directories", working_directory])

        full_command.extend(["-r", session_ref])
        full_command.append(prompt)

        await self.run_command(
            task=task,
            command=full_command,
            prompt="",
            output_cleaner=clean_gemini_output,
            output_prefix="Gemini Resume Output",
            cwd=working_directory,
            not_found_hint="Please ensure Gemini CLI is installed (npm install -g @google/gemini-cli).",
        )


# Global engine instance
engine = TaskEngine()
