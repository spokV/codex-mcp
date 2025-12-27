"""SQLite schema for usage tracking and quality ratings."""

import aiosqlite
from pathlib import Path

DB_PATH = Path.home() / ".owlex" / "usage.db"

SCHEMA = """
-- Usage tracking table
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Call identification
    task_id TEXT NOT NULL UNIQUE,

    -- Provider/model info
    provider_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    provider_type TEXT NOT NULL,  -- 'cli' or 'openai_api' or 'openrouter'

    -- Token counts
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,

    -- Cost (in USD)
    estimated_cost REAL,

    -- Timing
    timestamp TEXT NOT NULL,
    duration_seconds REAL,

    -- Request context
    prompt_preview TEXT,  -- First 200 chars of prompt
    working_directory TEXT,

    -- Status
    status TEXT NOT NULL,  -- 'completed', 'failed', 'timeout'
    error_message TEXT,

    -- OpenRouter specific
    generation_id TEXT,

    -- Metadata
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Quality ratings table (from Claude auto-rating)
CREATE TABLE IF NOT EXISTS ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Link to usage
    usage_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,

    -- Ratings (1-5 scale)
    helpfulness INTEGER CHECK (helpfulness BETWEEN 1 AND 5),
    accuracy INTEGER CHECK (accuracy BETWEEN 1 AND 5),
    completeness INTEGER CHECK (completeness BETWEEN 1 AND 5),

    -- Overall score (computed average)
    overall_score REAL,

    -- Optional notes from Claude
    notes TEXT,

    -- Timestamp
    rated_at TEXT DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (usage_id) REFERENCES usage(id)
);

-- Provider configs cache (for tracking which providers were used)
CREATE TABLE IF NOT EXISTS providers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL,
    base_url TEXT,
    default_model TEXT,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP,
    last_used TEXT
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_usage_provider ON usage(provider_name);
CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_usage_model ON usage(model_name);
CREATE INDEX IF NOT EXISTS idx_usage_status ON usage(status);
CREATE INDEX IF NOT EXISTS idx_ratings_usage_id ON ratings(usage_id);
CREATE INDEX IF NOT EXISTS idx_ratings_task_id ON ratings(task_id);
"""


async def init_database():
    """Initialize the database with schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        await db.executescript(SCHEMA)
        await db.commit()


def get_connection() -> aiosqlite.Connection:
    """Get async database connection context manager.

    Usage: async with get_connection() as db:
    """
    return aiosqlite.connect(str(DB_PATH))
