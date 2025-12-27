"""OpenAI-compatible API provider for owlex."""

import os
from datetime import datetime

import httpx

from .protocol import Provider, ProviderConfig, ProviderResult


class OpenAIAPIProvider:
    """Provider for OpenAI-compatible APIs (Kimi, MiniMax, DeepSeek, etc.)."""

    def __init__(self, config: ProviderConfig):
        self._config = config
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def provider_type(self) -> str:
        return "openai_api"

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
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout),
                headers={
                    "Authorization": f"Bearer {self._get_api_key()}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    def _calculate_cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        """Calculate cost based on token usage."""
        if input_tokens is None or output_tokens is None:
            return None
        if self._config.cost_per_1k_input is None or self._config.cost_per_1k_output is None:
            return None

        input_cost = (input_tokens / 1000) * self._config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self._config.cost_per_1k_output
        return input_cost + output_cost

    async def call(
        self,
        prompt: str,
        model: str | None = None,
        working_directory: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs,
    ) -> ProviderResult:
        """Execute a call to the OpenAI-compatible API."""
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
        cost = self._calculate_cost(input_tokens, output_tokens)

        content = data["choices"][0]["message"]["content"]

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
        # For API providers, we could potentially store conversation history
        # For now, just make a new call
        return await self.call(prompt, model=model, working_directory=working_directory, **kwargs)

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
