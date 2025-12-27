"""
Council orchestration logic for multi-agent deliberation.
Handles parallel execution and deliberation rounds.
"""

import asyncio
import sys
from datetime import datetime
from typing import Any

from .config import config
from .engine import engine, build_agent_response, codex_runner, gemini_runner
from .prompts import build_deliberation_prompt
from .models import (
    Agent,
    ClaudeOpinion,
    CouncilResponse,
    CouncilRound,
    CouncilMetadata,
)


def _log(msg: str):
    """Log progress to stderr for CLI visibility."""
    print(msg, file=sys.stderr, flush=True)


class Council:
    """
    Orchestrates multi-agent deliberation between Codex and Gemini.

    The council process:
    1. Round 1: Both agents answer the question in parallel
    2. Round 2 (optional): Agents see each other's answers and revise/critique
    """

    def __init__(self, context: Any = None, task_engine: Any = None):
        """
        Initialize the Council.

        Args:
            context: MCP server context for notifications
            task_engine: Optional TaskEngine instance (uses global engine if not provided).
                        This enables dependency injection for testing.
        """
        self.context = context
        self._engine = task_engine if task_engine is not None else engine
        self.log_entries: list[str] = []

    def log(self, msg: str):
        """Add to log and print to stderr."""
        self.log_entries.append(msg)
        _log(msg)

    async def deliberate(
        self,
        prompt: str,
        working_directory: str | None = None,
        claude_opinion: str | None = None,
        deliberate: bool = True,
        critique: bool = True,
        timeout: int | None = None,
    ) -> CouncilResponse:
        """
        Run a council deliberation session.

        Args:
            prompt: The question or task to deliberate on
            working_directory: Working directory context for agents
            claude_opinion: Optional Claude opinion to share with agents
            deliberate: If True, run a second round where agents see each other's answers
            critique: If True, Round 2 asks agents to find flaws instead of revise
            timeout: Timeout per agent in seconds

        Returns:
            CouncilResponse with all rounds and metadata
        """
        if timeout is None:
            timeout = config.default_timeout

        council_start = datetime.now()

        # === Round 1: Parallel initial queries ===
        if claude_opinion and claude_opinion.strip():
            self.log(f"Claude's opinion received ({len(claude_opinion)} chars)")
        self.log("Round 1: querying Codex and Gemini...")

        round_1 = await self._run_round_1(prompt, working_directory, timeout)

        round_2 = None
        if deliberate:
            round_2 = await self._run_round_2(
                prompt=prompt,
                working_directory=working_directory,
                round_1=round_1,
                claude_opinion=claude_opinion,
                critique=critique,
                timeout=timeout,
            )

        # Build Claude opinion object if provided
        claude_opinion_obj = None
        if claude_opinion and claude_opinion.strip():
            claude_opinion_obj = ClaudeOpinion(
                content=claude_opinion.strip(),
                provided_at=council_start.isoformat(),
            )

        return CouncilResponse(
            prompt=prompt,
            working_directory=working_directory,
            deliberation=deliberate,
            critique=critique,
            claude_opinion=claude_opinion_obj,
            round_1=round_1,
            round_2=round_2,
            metadata=CouncilMetadata(
                total_duration_seconds=(datetime.now() - council_start).total_seconds(),
                rounds=2 if deliberate else 1,
                log=self.log_entries,
            ),
        )

    async def _run_round_1(
        self,
        prompt: str,
        working_directory: str | None,
        timeout: int,
    ) -> CouncilRound:
        """Run the first round of parallel queries."""
        round1_start = datetime.now()

        codex_task = self._engine.create_task(
            command=f"council_{Agent.CODEX.value}",
            args={"prompt": prompt, "working_directory": working_directory},
            context=self.context,
        )
        gemini_task = self._engine.create_task(
            command=f"council_{Agent.GEMINI.value}",
            args={"prompt": prompt, "working_directory": working_directory},
            context=self.context,
        )

        async def run_codex():
            await self._engine.run_agent(
                codex_task, codex_runner, mode="exec",
                prompt=prompt, working_directory=working_directory, enable_search=config.codex.enable_search
            )
            elapsed = (datetime.now() - round1_start).total_seconds()
            status = "completed" if codex_task.status == "completed" else "failed"
            self.log(f"Codex {status} ({elapsed:.1f}s)")

        async def run_gemini():
            await self._engine.run_agent(
                gemini_task, gemini_runner, mode="exec",
                prompt=prompt, working_directory=working_directory
            )
            elapsed = (datetime.now() - round1_start).total_seconds()
            status = "completed" if gemini_task.status == "completed" else "failed"
            self.log(f"Gemini {status} ({elapsed:.1f}s)")

        codex_task.async_task = asyncio.create_task(run_codex())
        gemini_task.async_task = asyncio.create_task(run_gemini())

        # Wait for both tasks with timeout
        done, pending = await asyncio.wait(
            [codex_task.async_task, gemini_task.async_task],
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )

        # Kill subprocesses for any tasks that timed out
        for task in [codex_task, gemini_task]:
            if task.async_task in pending:
                self.log(f"{task.command} timed out")
                await self._engine.kill_task_subprocess(task)
                task.status = "failed"
                task.error = f"Timed out after {timeout} seconds"
                task.completion_time = datetime.now()

        round1_elapsed = (datetime.now() - round1_start).total_seconds()
        self.log(f"Round 1 complete ({round1_elapsed:.1f}s)")

        return CouncilRound(
            codex=build_agent_response(codex_task, Agent.CODEX),
            gemini=build_agent_response(gemini_task, Agent.GEMINI),
        )

    async def _run_round_2(
        self,
        prompt: str,
        working_directory: str | None,
        round_1: CouncilRound,
        claude_opinion: str | None,
        critique: bool,
        timeout: int,
    ) -> CouncilRound:
        """Run the second round of deliberation."""
        self.log("Round 2: deliberation phase...")

        codex_content = round_1.codex.content or round_1.codex.error or "(no response)"
        gemini_content = round_1.gemini.content or round_1.gemini.error or "(no response)"
        claude_content = claude_opinion.strip() if claude_opinion else None

        deliberation_prompt = build_deliberation_prompt(
            original_prompt=prompt,
            codex_answer=codex_content,
            gemini_answer=gemini_content,
            claude_answer=claude_content,
            critique=critique,
        )

        round2_start = datetime.now()

        codex_delib_task = self._engine.create_task(
            command=f"council_{Agent.CODEX.value}_delib",
            args={"prompt": deliberation_prompt, "working_directory": working_directory},
            context=self.context,
        )
        gemini_delib_task = self._engine.create_task(
            command=f"council_{Agent.GEMINI.value}_delib",
            args={"prompt": deliberation_prompt, "working_directory": working_directory},
            context=self.context,
        )

        async def run_codex_delib():
            await self._engine.run_agent(
                codex_delib_task, codex_runner, mode="exec",
                prompt=deliberation_prompt, working_directory=working_directory, enable_search=config.codex.enable_search
            )
            elapsed = (datetime.now() - round2_start).total_seconds()
            self.log(f"Codex revised ({elapsed:.1f}s)")

        async def run_gemini_delib():
            await self._engine.run_agent(
                gemini_delib_task, gemini_runner, mode="exec",
                prompt=deliberation_prompt, working_directory=working_directory
            )
            elapsed = (datetime.now() - round2_start).total_seconds()
            self.log(f"Gemini revised ({elapsed:.1f}s)")

        codex_delib_task.async_task = asyncio.create_task(run_codex_delib())
        gemini_delib_task.async_task = asyncio.create_task(run_gemini_delib())

        # Wait for both tasks with timeout
        done, pending = await asyncio.wait(
            [codex_delib_task.async_task, gemini_delib_task.async_task],
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )

        # Kill subprocesses for any tasks that timed out
        for task in [codex_delib_task, gemini_delib_task]:
            if task.async_task in pending:
                self.log(f"{task.command} timed out")
                await self._engine.kill_task_subprocess(task)
                task.status = "failed"
                task.error = f"Timed out after {timeout} seconds"
                task.completion_time = datetime.now()

        round2_elapsed = (datetime.now() - round2_start).total_seconds()
        self.log(f"Round 2 complete ({round2_elapsed:.1f}s)")

        return CouncilRound(
            codex=build_agent_response(codex_delib_task, Agent.CODEX),
            gemini=build_agent_response(gemini_delib_task, Agent.GEMINI),
        )
