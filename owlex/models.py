"""
Data models for owlex - Task management and API responses.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Agent(str, Enum):
    """Available AI agents."""
    CODEX = "codex"
    GEMINI = "gemini"


@dataclass
class Task:
    """Represents a background task for CLI execution."""
    task_id: str
    status: str  # TaskStatus value
    command: str
    args: dict
    start_time: datetime
    context: Any | None = field(default=None, repr=False)  # MCP Context
    completion_time: datetime | None = None
    result: str | None = None
    error: str | None = None
    async_task: asyncio.Task | None = field(default=None, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    # Streaming support
    output_lines: list[str] = field(default_factory=list)
    stream_complete: bool = False


# === Pydantic Response Models ===

class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    error_code: str | None = None
    details: dict[str, Any] | None = None


class TaskResponse(BaseModel):
    """Response for task operations (start, get, wait)."""
    success: bool
    task_id: str | None = None
    status: str | None = None
    message: str | None = None
    content: str | None = None
    error: str | None = None
    duration_seconds: float | None = None


class AgentResponse(BaseModel):
    """Response from a single agent in council."""
    agent: str
    status: str
    content: str | None = None
    error: str | None = None
    duration_seconds: float | None = None
    task_id: str


class ClaudeOpinion(BaseModel):
    """Claude's initial opinion provided before council deliberation."""
    content: str
    provided_at: str  # ISO timestamp


class CouncilRound(BaseModel):
    """A single round of council deliberation."""
    codex: AgentResponse
    gemini: AgentResponse


class CouncilMetadata(BaseModel):
    """Metadata for council session."""
    total_duration_seconds: float
    rounds: int
    log: list[str] = []  # Progress log entries


class CouncilResponse(BaseModel):
    """Structured response from council_ask."""
    prompt: str
    working_directory: str | None = None
    deliberation: bool
    claude_opinion: ClaudeOpinion | None = None
    round_1: CouncilRound
    round_2: CouncilRound | None = None
    metadata: CouncilMetadata
