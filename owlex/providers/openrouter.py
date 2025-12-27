"""OpenRouter provider for owlex - access to 500+ models."""

import os
from datetime import datetime

import httpx

from .protocol import Provider, ProviderConfig, ProviderResult


class OpenRouterProvider:
    """Provider for OpenRouter API - unified access to 500+ models.

    Special features:
    - Model routing: :nitro (fastest), :floor (cheapest)
    - Generation stats: Query /api/v1/generation for exact cost
    - Extensive model catalog: Claude, GPT, Gemini, Kimi, etc.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def provider_type(self) -> str:
        return "openrouter"

    @property
    def is_available(self) -> bool:
        """Check if API key is configured."""
        if not self._config.api_key_env:
            return False
        return bool(os.environ.get(self._config.api_key_env))

    @property
    def config(self) -> ProviderConfig:
        return self._config

    def _get_api_key(self) -> str:
        """Get API key from environment."""
        if not self._config.api_key_env:
            raise ValueError(f"No api_key_env configured for provider {self.name}")
        key = os.environ.get(self._config.api_key_env)
        if not key:
            raise ValueError(f"API key not found in environment variable {self._config.api_key_env}")
        return key

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client with OpenRouter-specific headers."""
        if self._client is None or self._client.is_closed:
            headers = {
                "Authorization": f"Bearer {self._get_api_key()}",
                "Content-Type": "application/json",
            }
            # OpenRouter requires HTTP-Referer
            if self._config.site_url:
                headers["HTTP-Referer"] = self._config.site_url
            # Optional but recommended
            if self._config.app_name:
                headers["X-Title"] = self._config.app_name

            self._client = httpx.AsyncClient(
                base_url=self._config.base_url or self.BASE_URL,
                timeout=httpx.Timeout(self._config.timeout),
                headers=headers,
            )
        return self._client

    async def get_generation_stats(self, generation_id: str) -> dict:
        """Get detailed generation stats including exact cost.

        OpenRouter provides this endpoint to get precise token counts
        and costs after generation completes.
        """
        client = await self._get_client()
        response = await client.get(f"/generation?id={generation_id}")
        response.raise_for_status()
        return response.json()

    async def list_models(self) -> list[dict]:
        """List all available models on OpenRouter."""
        client = await self._get_client()
        response = await client.get("/models")
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    async def call(
        self,
        prompt: str,
        model: str | None = None,
        working_directory: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> ProviderResult:
        """Execute a call to OpenRouter API.

        Args:
            prompt: The prompt to send
            model: Model identifier (e.g., 'anthropic/claude-sonnet-4', 'openai/gpt-4o')
                   Can include routing suffix: :nitro (fastest) or :floor (cheapest)
            working_directory: Optional context for the request
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
        """
        client = await self._get_client()
        use_model = model or self._config.default_model

        if not use_model:
            raise ValueError(f"No model specified for provider {self.name}")

        # Build messages
        messages = [{"role": "user", "content": prompt}]

        # Add system message if working_directory provided
        if working_directory:
            system_msg = f"Working directory context: {working_directory}"
            messages.insert(0, {"role": "system", "content": system_msg})

        payload = {
            "model": use_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start_time = datetime.now()
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        duration = (datetime.now() - start_time).total_seconds()

        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")

        # OpenRouter provides generation ID for detailed stats
        generation_id = data.get("id")

        content = data["choices"][0]["message"]["content"]

        # Try to get exact cost from generation stats
        cost = None
        if generation_id:
            try:
                stats = await self.get_generation_stats(generation_id)
                cost = stats.get("data", {}).get("total_cost")
            except Exception:
                # Fallback to estimated cost if stats unavailable
                pass

        return ProviderResult(
            content=content,
            provider_name=self.name,
            model_name=use_model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration_seconds=duration,
            cost=cost,
            raw_response=data,
            generation_id=generation_id,
        )

    async def resume(
        self,
        prompt: str,
        session_ref: str,
        model: str | None = None,
        working_directory: str | None = None,
        **kwargs,
    ) -> ProviderResult:
        """API providers don't support session resume - just call with context."""
        return await self.call(prompt, model=model, working_directory=working_directory, **kwargs)

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
