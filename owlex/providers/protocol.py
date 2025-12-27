"""Provider protocol and data structures for owlex."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ProviderResult:
    """Standard result from any provider call."""

    content: str
    provider_name: str
    model_name: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    duration_seconds: float | None = None
    cost: float | None = None
    raw_response: dict | None = field(default=None, repr=False)
    generation_id: str | None = None  # For OpenRouter generation stats


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    type: str  # "cli", "openai_api", "openrouter"
    base_url: str | None = None
    api_key_env: str | None = None
    default_model: str | None = None
    timeout: int = 300
    # Cost tracking (per 1k tokens)
    cost_per_1k_input: float | None = None
    cost_per_1k_output: float | None = None
    # OpenRouter specific
    site_url: str | None = None
    app_name: str | None = None
    # Extra config
    extra: dict | None = None


@runtime_checkable
class Provider(Protocol):
    """Protocol for AI provider implementations."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'codex', 'kimi')."""
        ...

    @property
    def provider_type(self) -> str:
        """Provider type: 'cli', 'openai_api', or 'openrouter'."""
        ...

    @property
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        ...

    @property
    def config(self) -> ProviderConfig:
        """Get provider configuration."""
        ...

    async def call(
        self,
        prompt: str,
        model: str | None = None,
        working_directory: str | None = None,
        **kwargs,
    ) -> ProviderResult:
        """Execute a call to the provider."""
        ...

    async def resume(
        self,
        prompt: str,
        session_ref: str,
        model: str | None = None,
        working_directory: str | None = None,
        **kwargs,
    ) -> ProviderResult:
        """Resume a previous session (if supported)."""
        ...
