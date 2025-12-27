"""Usage tracking and rating operations for owlex."""

from datetime import datetime, timedelta
from typing import Tuple

from .schema import get_connection, init_database


async def ensure_initialized():
    """Ensure database is initialized."""
    await init_database()


async def record_usage(
    task_id: str,
    provider_name: str,
    model_name: str,
    provider_type: str,
    status: str,
    prompt: str | None = None,
    working_directory: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    total_tokens: int | None = None,
    estimated_cost: float | None = None,
    duration_seconds: float | None = None,
    error_message: str | None = None,
    generation_id: str | None = None,
) -> int:
    """Record a provider call in the database.

    Returns the usage record ID.
    """
    prompt_preview = prompt[:200] if prompt else None
    timestamp = datetime.now().isoformat()

    async with get_connection() as db:
        cursor = await db.execute(
            """
            INSERT INTO usage (
                task_id, provider_name, model_name, provider_type,
                input_tokens, output_tokens, total_tokens, estimated_cost,
                timestamp, duration_seconds, prompt_preview, working_directory,
                status, error_message, generation_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                provider_name,
                model_name,
                provider_type,
                input_tokens,
                output_tokens,
                total_tokens,
                estimated_cost,
                timestamp,
                duration_seconds,
                prompt_preview,
                working_directory,
                status,
                error_message,
                generation_id,
            ),
        )
        await db.commit()

        # Update provider last_used
        await db.execute(
            """
            INSERT INTO providers (name, type, last_used)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET last_used = ?
            """,
            (provider_name, provider_type, timestamp, timestamp),
        )
        await db.commit()

        return cursor.lastrowid


async def record_rating(
    task_id: str,
    helpfulness: int,
    accuracy: int,
    completeness: int,
    notes: str | None = None,
) -> bool:
    """Record a quality rating for a provider response.

    Returns True if rating was recorded successfully.
    """
    if not all(1 <= r <= 5 for r in [helpfulness, accuracy, completeness]):
        raise ValueError("All ratings must be between 1 and 5")

    overall_score = (helpfulness + accuracy + completeness) / 3.0

    async with get_connection() as db:
        # Find the usage record
        cursor = await db.execute(
            "SELECT id FROM usage WHERE task_id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return False

        usage_id = row[0]

        # Check if already rated
        cursor = await db.execute(
            "SELECT id FROM ratings WHERE task_id = ?", (task_id,)
        )
        existing = await cursor.fetchone()
        if existing:
            # Update existing rating
            await db.execute(
                """
                UPDATE ratings SET
                    helpfulness = ?, accuracy = ?, completeness = ?,
                    overall_score = ?, notes = ?, rated_at = ?
                WHERE task_id = ?
                """,
                (
                    helpfulness,
                    accuracy,
                    completeness,
                    overall_score,
                    notes,
                    datetime.now().isoformat(),
                    task_id,
                ),
            )
        else:
            # Insert new rating
            await db.execute(
                """
                INSERT INTO ratings (
                    usage_id, task_id, helpfulness, accuracy, completeness,
                    overall_score, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (usage_id, task_id, helpfulness, accuracy, completeness, overall_score, notes),
            )

        await db.commit()
        return True


def _parse_period(period: str) -> Tuple[str, str]:
    """Parse period string to start and end dates.

    Supported formats:
    - 'this_month': Current month
    - 'last_month': Previous month
    - 'all_time': All records
    - 'YYYY-MM': Specific month
    """
    now = datetime.now()

    if period == "this_month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            end = now.replace(year=now.year + 1, month=1, day=1)
        else:
            end = now.replace(month=now.month + 1, day=1)
    elif period == "last_month":
        first_of_this = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = first_of_this
        if now.month == 1:
            start = now.replace(year=now.year - 1, month=12, day=1)
        else:
            start = now.replace(month=now.month - 1, day=1)
    elif period == "all_time":
        start = datetime(2000, 1, 1)
        end = datetime(2100, 1, 1)
    else:
        # Try YYYY-MM format
        try:
            year, month = map(int, period.split("-"))
            start = datetime(year, month, 1)
            if month == 12:
                end = datetime(year + 1, 1, 1)
            else:
                end = datetime(year, month + 1, 1)
        except (ValueError, AttributeError):
            # Default to this month
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if now.month == 12:
                end = now.replace(year=now.year + 1, month=1, day=1)
            else:
                end = now.replace(month=now.month + 1, day=1)

    return start.isoformat(), end.isoformat()


async def get_provider_stats(
    period: str = "this_month",
    provider: str | None = None,
) -> list[dict]:
    """Get usage statistics for providers.

    Returns list of dicts with:
    - provider_name
    - model_name
    - call_count
    - total_tokens
    - total_cost
    - avg_rating
    - avg_duration
    """
    start_date, end_date = _parse_period(period)

    async with get_connection() as db:
        query = """
            SELECT
                u.provider_name,
                u.model_name,
                COUNT(*) as call_count,
                SUM(CASE WHEN u.status = 'completed' THEN 1 ELSE 0 END) as success_count,
                SUM(u.total_tokens) as total_tokens,
                SUM(u.estimated_cost) as total_cost,
                AVG(r.overall_score) as avg_rating,
                AVG(u.duration_seconds) as avg_duration
            FROM usage u
            LEFT JOIN ratings r ON u.id = r.usage_id
            WHERE u.timestamp >= ? AND u.timestamp < ?
        """
        params = [start_date, end_date]

        if provider:
            query += " AND u.provider_name = ?"
            params.append(provider)

        query += " GROUP BY u.provider_name, u.model_name ORDER BY call_count DESC"

        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    return [
        {
            "provider_name": row[0],
            "model_name": row[1],
            "call_count": row[2],
            "success_count": row[3],
            "total_tokens": row[4],
            "total_cost": row[5],
            "avg_rating": row[6],
            "avg_duration": row[7],
        }
        for row in rows
    ]


async def get_recent_usage(limit: int = 10) -> list[dict]:
    """Get recent usage records."""
    async with get_connection() as db:
        cursor = await db.execute(
            """
            SELECT
                u.task_id, u.provider_name, u.model_name, u.status,
                u.total_tokens, u.estimated_cost, u.duration_seconds,
                u.timestamp, r.overall_score
            FROM usage u
            LEFT JOIN ratings r ON u.id = r.usage_id
            ORDER BY u.timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

    return [
        {
            "task_id": row[0],
            "provider_name": row[1],
            "model_name": row[2],
            "status": row[3],
            "total_tokens": row[4],
            "estimated_cost": row[5],
            "duration_seconds": row[6],
            "timestamp": row[7],
            "rating": row[8],
        }
        for row in rows
    ]
