"""
Codex CLI agent runner.
"""

import re
from typing import Callable

from ..config import config
from .base import AgentRunner, AgentCommand


def clean_codex_output(raw_output: str, original_prompt: str = "") -> str:
    """Clean Codex CLI output by removing echoed prompt templates."""
    if not config.codex.clean_output:
        return raw_output
    cleaned = raw_output
    if original_prompt and cleaned.startswith(original_prompt):
        cleaned = cleaned[len(original_prompt):].lstrip()
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


class CodexRunner(AgentRunner):
    """Runner for OpenAI Codex CLI."""

    @property
    def name(self) -> str:
        return "codex"

    def build_exec_command(
        self,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for starting a new Codex session."""
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.extend(["--enable", "web_search_request"])

        if config.codex.bypass_approvals:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            full_command.append("--full-auto")

        # Use stdin for prompt input
        full_command.append("-")

        return AgentCommand(
            command=full_command,
            prompt=prompt,
            output_prefix="Codex Output",
            not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
            stream=True,
        )

    def build_resume_command(
        self,
        session_ref: str,
        prompt: str,
        working_directory: str | None = None,
        enable_search: bool = False,
        **kwargs,
    ) -> AgentCommand:
        """Build command for resuming an existing Codex session."""
        full_command = ["codex", "exec", "--skip-git-repo-check"]

        if working_directory:
            full_command.extend(["--cd", working_directory])
        if enable_search:
            full_command.extend(["--enable", "web_search_request"])

        if config.codex.bypass_approvals:
            full_command.append("--dangerously-bypass-approvals-and-sandbox")
        else:
            full_command.append("--full-auto")

        full_command.append("resume")
        if session_ref == "--last":
            full_command.append("--last")
        else:
            # Validate session_ref to prevent flag injection
            # Session IDs should be alphanumeric/UUID-like
            if session_ref.startswith("-"):
                raise ValueError(f"Invalid session_ref: '{session_ref}' - cannot start with '-'")
            full_command.append(session_ref)

        # Use stdin for prompt input
        full_command.append("-")

        return AgentCommand(
            command=full_command,
            prompt=prompt,
            output_prefix="Codex Resume Output",
            not_found_hint="Please ensure Codex CLI is installed and in your PATH.",
            stream=False,  # Resume uses non-streaming mode
        )

    def get_output_cleaner(self) -> Callable[[str, str], str]:
        return clean_codex_output
