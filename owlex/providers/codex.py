"""Codex CLI provider for owlex."""

import asyncio
import os
import re
import shutil
from datetime import datetime

from .protocol import Provider, ProviderConfig, ProviderResult


class CodexProvider:
    """Provider for Codex CLI integration."""

    def __init__(self, config: ProviderConfig | None = None):
        self._config = config or ProviderConfig(
            name="codex",
            type="cli",
            default_model="gpt-5-codex",
        )
        self._bypass_approvals = os.environ.get("CODEX_BYPASS_APPROVALS", "false").lower() == "true"
        self._clean_output = os.environ.get("CODEX_CLEAN_OUTPUT", "true").lower() == "true"

    @property
    def name(self) -> str:
        return "codex"

    @property
    def provider_type(self) -> str:
        return "cli"

    @property
    def is_available(self) -> bool:
        """Check if codex CLI is available."""
        return shutil.which("codex") is not None

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _clean_output(self, raw_output: str, original_prompt: str = "") -> str:
        """Clean Codex CLI output by removing echoed prompt templates."""
        if not self._clean_output:
            return raw_output
        cleaned = raw_output
        if original_prompt and cleaned.startswith(original_prompt):
            cleaned = cleaned[len(original_prompt) :].lstrip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    async def call(
        self,
        prompt: str,
        model: str | None = None,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> ProviderResult:
        """Execute a new Codex session."""
        start_time = datetime.now()

        # Build command: codex exec [options] -
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            working_directory = os.path.expanduser(working_directory)
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.append("--search")

        if self._bypass_approvals:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            full_command.append("--full-auto")

        full_command.append("-")  # Read prompt from stdin

        stdin_input = prompt.encode("utf-8")

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_input), timeout=self._config.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Codex exec timed out after {self._config.timeout} seconds")

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

        duration = (datetime.now() - start_time).total_seconds()

        if process.returncode != 0:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            raise RuntimeError(
                f"Codex exec failed (exit code {process.returncode}):\n\n"
                + ("\n\n".join(error_output) or "No output")
            )

        cleaned_output = self._clean_output(stdout_text, prompt)

        return ProviderResult(
            content=cleaned_output,
            provider_name=self.name,
            model_name=model or self._config.default_model or "codex",
            duration_seconds=duration,
        )

    async def resume(
        self,
        prompt: str,
        session_ref: str = "--last",
        model: str | None = None,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> ProviderResult:
        """Resume an existing Codex session."""
        start_time = datetime.now()

        # Build command: codex exec [options] resume [--last | session-id] -
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            working_directory = os.path.expanduser(working_directory)
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.append("--search")

        if self._bypass_approvals:
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

        stdin_input = prompt.encode("utf-8")

        process = await asyncio.create_subprocess_exec(
            *full_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=stdin_input), timeout=self._config.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(f"Codex resume timed out after {self._config.timeout} seconds")

        stdout_text = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_text = stderr.decode("utf-8", errors="replace") if stderr else ""

        duration = (datetime.now() - start_time).total_seconds()

        if process.returncode != 0:
            error_output = []
            if stdout_text.strip():
                error_output.append(f"stdout:\n{stdout_text}")
            if stderr_text.strip():
                error_output.append(f"stderr:\n{stderr_text}")
            raise RuntimeError(
                f"Codex resume failed (exit code {process.returncode}):\n\n"
                + ("\n\n".join(error_output) or "No output")
            )

        cleaned_output = self._clean_output(stdout_text, prompt)

        return ProviderResult(
            content=cleaned_output,
            provider_name=self.name,
            model_name=model or self._config.default_model or "codex",
            duration_seconds=duration,
        )
