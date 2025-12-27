"""
Task execution engine for owlex.
Handles subprocess management, streaming, and task lifecycle.
"""

import asyncio
import sys
import uuid
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from .config import config
from .models import Task, TaskStatus, AgentResponse, Agent
from .agents import CodexRunner, GeminiRunner
from .agents.base import AgentRunner, AgentCommand


def extract_content(result: str | None, prefix: str) -> str:
    """Extract raw content from task result, stripping the prefix."""
    if not result:
        return ""
    if result.startswith(prefix):
        return result[len(prefix):].strip()
    return result.strip()


def build_agent_response(task: Task, agent: Agent | str) -> AgentResponse:
    """Build structured response for an agent."""
    # Normalize to string for backward compatibility
    agent_name = agent.value if isinstance(agent, Agent) else agent

    prefix_map = {
        Agent.CODEX.value: "Codex Output:\n\n",
        Agent.GEMINI.value: "Gemini Output:\n\n",
    }
    prefix = prefix_map.get(agent_name, "")

    return AgentResponse(
        agent=agent_name,
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

# Agent runner instances - available for import by other modules
codex_runner = CodexRunner()
gemini_runner = GeminiRunner()

# Map Agent enum to runner instances
AGENT_RUNNERS: dict[Agent, AgentRunner] = {
    Agent.CODEX: codex_runner,
    Agent.GEMINI: gemini_runner,
}


class TaskEngine:
    """
    Core task execution engine.
    Manages task lifecycle, subprocess execution, and streaming.
    """

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self._cleanup_task: asyncio.Task | None = None
        # Print security warnings on initialization
        config.print_warnings()

    def start_cleanup_loop(self):
        """Start background task cleanup loop."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_tasks())

    def stop_cleanup_loop(self):
        """Stop background task cleanup loop."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    async def _terminate_process(self, process: asyncio.subprocess.Process, grace_period: float = 2.0):
        """Gracefully terminate a process: SIGTERM first, then SIGKILL after grace period."""
        if process.returncode is not None:
            return  # Already terminated
        try:
            process.terminate()  # SIGTERM - allow graceful cleanup
            try:
                await asyncio.wait_for(process.wait(), timeout=grace_period)
            except asyncio.TimeoutError:
                # Process didn't exit gracefully, force kill
                process.kill()  # SIGKILL
                await process.wait()
        except Exception:
            pass

    async def kill_task_subprocess(self, task: Task):
        """Kill subprocess for a task if it's still running."""
        if task.process and task.process.returncode is None:
            await self._terminate_process(task.process)
        if task.async_task and not task.async_task.done():
            task.async_task.cancel()
            try:
                await task.async_task
            except asyncio.CancelledError:
                pass

    async def kill_all_tasks(self):
        """Kill all running tasks and their subprocesses. Used for graceful shutdown."""
        for task_id, task in list(self.tasks.items()):
            if task.status in [TaskStatus.PENDING.value, TaskStatus.RUNNING.value]:
                await self.kill_task_subprocess(task)
                task.status = TaskStatus.CANCELLED.value
                task.error = "Server shutdown"
                task.completion_time = datetime.now()

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
        timeout: int | None = None,
        not_found_hint: str | None = None,
        stream: bool = False,
    ):
        """
        Run a subprocess command for a task (legacy interface).
        Wraps run_agent_command for backward compatibility.
        """
        agent_cmd = AgentCommand(
            command=command,
            prompt=prompt,
            cwd=cwd,
            output_prefix=output_prefix,
            not_found_hint=not_found_hint,
            stream=stream,
        )
        await self.run_agent_command(task, agent_cmd, timeout)
        # Apply output cleaner to result
        if task.result and task.status == "completed":
            prefix = f"{output_prefix}:\n\n"
            if task.result.startswith(prefix):
                content = task.result[len(prefix):]
                cleaned = output_cleaner(content, prompt)
                task.result = f"{prefix}{cleaned}"

    async def run_agent_command(
        self,
        task: Task,
        agent_cmd: AgentCommand,
        timeout: int | None = None,
    ):
        """
        Run a subprocess command for a task using an AgentCommand specification.
        This is the unified method for running any agent.
        """
        if timeout is None:
            timeout = config.default_timeout

        command = agent_cmd.command
        prompt = agent_cmd.prompt
        output_cleaner = lambda stdout, p: stdout  # Default no-op cleaner
        output_prefix = agent_cmd.output_prefix
        cwd = agent_cmd.cwd
        not_found_hint = agent_cmd.not_found_hint
        stream = agent_cmd.stream

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

            if stream:
                # Streaming mode: manually write stdin, then read line-by-line
                if prompt:
                    process.stdin.write(prompt.encode('utf-8'))
                    await process.stdin.drain()
                process.stdin.close()

                # Create reader tasks - track them for explicit cleanup on timeout
                stdout_task = asyncio.create_task(
                    self._read_stream_lines(process.stdout, task, "stdout")
                )
                stderr_task = asyncio.create_task(
                    self._read_stream_lines(process.stderr, task, "stderr")
                )

                try:
                    async def read_with_timeout():
                        stdout_text, stderr_text = await asyncio.gather(stdout_task, stderr_task)
                        await process.wait()
                        return stdout_text, stderr_text

                    stdout_text, stderr_text = await asyncio.wait_for(
                        read_with_timeout(),
                        timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Cancel reader tasks explicitly to prevent zombie tasks
                    stdout_task.cancel()
                    stderr_task.cancel()
                    try:
                        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                    except Exception:
                        pass
                    # Graceful termination: SIGTERM first, then SIGKILL
                    await self._terminate_process(process)
                    task.status = TaskStatus.FAILED.value
                    task.error = f"{command[0]} command timed out after {timeout} seconds"
                    task.completion_time = datetime.now()
                    task.stream_complete = True
                    await self._emit_task_notification(task)
                    return
            else:
                # Non-streaming mode: use communicate() which handles stdin internally
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=prompt.encode('utf-8') if prompt else None),
                        timeout=timeout
                    )
                    stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
                    stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
                except asyncio.TimeoutError:
                    # Graceful termination: SIGTERM first, then SIGKILL
                    await self._terminate_process(process)
                    task.status = TaskStatus.FAILED.value
                    task.error = f"{command[0]} command timed out after {timeout} seconds"
                    task.completion_time = datetime.now()
                    await self._emit_task_notification(task)
                    return

            task.stream_complete = True

            if process.returncode == 0:
                cleaned_output = output_cleaner(stdout_text, prompt)
                task.result = f"{output_prefix}:\n\n{cleaned_output}"
                # Capture stderr as warnings even on success (CLI tools often emit warnings)
                if stderr_text.strip():
                    task.warnings = stderr_text.strip()
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

    # === Unified agent execution method ===

    async def run_agent(
        self,
        task: Task,
        runner: AgentRunner,
        mode: str = "exec",
        prompt: str = "",
        working_directory: str | None = None,
        session_ref: str | None = None,
        enable_search: bool = False,
        timeout: int | None = None,
    ):
        """
        Run an agent using the unified polymorphic pattern.

        Args:
            task: The Task object to execute
            runner: AgentRunner instance (codex_runner or gemini_runner)
            mode: "exec" for new session, "resume" for existing session
            prompt: The prompt to send
            working_directory: Working directory for the agent
            session_ref: Session reference (required for resume mode)
            enable_search: Enable web search (Codex only)
            timeout: Timeout in seconds
        """
        if mode == "exec":
            agent_cmd = runner.build_exec_command(
                prompt=prompt,
                working_directory=working_directory,
                enable_search=enable_search,
            )
        elif mode == "resume":
            if session_ref is None:
                raise ValueError("session_ref is required for resume mode")
            agent_cmd = runner.build_resume_command(
                session_ref=session_ref,
                prompt=prompt,
                working_directory=working_directory,
                enable_search=enable_search,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'exec' or 'resume'")

        await self.run_agent_command(task, agent_cmd, timeout=timeout)

        # Apply output cleaner to result
        if task.result and task.status == "completed":
            prefix = f"{agent_cmd.output_prefix}:\n\n"
            if task.result.startswith(prefix):
                content = task.result[len(prefix):]
                cleaner = runner.get_output_cleaner()
                cleaned = cleaner(content, prompt)
                task.result = f"{prefix}{cleaned}"


# Global engine instance
engine = TaskEngine()

# Re-export DEFAULT_TIMEOUT for backward compatibility
DEFAULT_TIMEOUT = config.default_timeout
