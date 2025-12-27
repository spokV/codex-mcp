"""
Prompt templates for council deliberation.
Centralized prompt management for consistency and testability.
"""

# === Round 2: Deliberation Prompts ===

DELIBERATION_INTRO_REVISE = (
    "You previously answered a question. Now review all council members' "
    "answers and provide your revised opinion."
)

DELIBERATION_INTRO_CRITIQUE = (
    "You previously answered a question. Now act as a senior code reviewer "
    "and critically analyze the other council members' answers."
)

DELIBERATION_INSTRUCTION_REVISE = (
    "Please provide your revised answer after considering the other perspectives. "
    "Note any points of agreement or disagreement."
)

DELIBERATION_INSTRUCTION_CRITIQUE = (
    "Act as a senior reviewer. Identify bugs, security vulnerabilities, "
    "architectural flaws, incorrect assumptions, or gaps in the other answers. "
    "Be specific and critical. For code suggestions, look for edge cases, "
    "error handling issues, and potential failures. Do not just agree - find problems."
)


def build_deliberation_prompt(
    original_prompt: str,
    codex_answer: str,
    gemini_answer: str,
    claude_answer: str | None = None,
    critique: bool = False,
) -> str:
    """
    Build the deliberation prompt for round 2.

    Args:
        original_prompt: The original question asked
        codex_answer: Codex's round 1 answer
        gemini_answer: Gemini's round 1 answer
        claude_answer: Optional Claude opinion to include
        critique: If True, use critique mode prompts

    Returns:
        Complete deliberation prompt string
    """
    if critique:
        intro = DELIBERATION_INTRO_CRITIQUE
        instruction = DELIBERATION_INSTRUCTION_CRITIQUE
    else:
        intro = DELIBERATION_INTRO_REVISE
        instruction = DELIBERATION_INSTRUCTION_REVISE

    parts = [
        intro,
        "",
        "ORIGINAL QUESTION:",
        original_prompt,
    ]

    if claude_answer:
        parts.extend(["", "CLAUDE'S ANSWER:", claude_answer])

    parts.extend([
        "",
        "CODEX'S ANSWER:",
        codex_answer,
        "",
        "GEMINI'S ANSWER:",
        gemini_answer,
        "",
        instruction,
    ])

    return "\n".join(parts)
