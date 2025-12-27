"""Provider registry for owlex - discovers and manages providers."""

import json
from pathlib import Path
from typing import Dict

from .protocol import Provider, ProviderConfig
from .codex import CodexProvider
from .gemini import GeminiProvider
from .openai_api import OpenAIAPIProvider
from .openrouter import OpenRouterProvider


class ProviderRegistry:
    """Manages provider discovery and instantiation.

    Loads providers from:
    1. Built-in CLI providers (codex, gemini)
    2. User config file (~/.owlex/providers.json)
    """

    CONFIG_PATH = Path.home() / ".owlex" / "providers.json"

    def __init__(self):
        self._providers: Dict[str, Provider] = {}
        self._initialized = False

    def load_config(self) -> dict:
        """Load provider configuration from ~/.owlex/providers.json."""
        if not self.CONFIG_PATH.exists():
            return {}
        try:
            return json.loads(self.CONFIG_PATH.read_text())
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load providers config: {e}")
            return {}

    def save_config(self, config: dict):
        """Save provider configuration to ~/.owlex/providers.json."""
        self.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.CONFIG_PATH.write_text(json.dumps(config, indent=2))

    def initialize(self):
        """Initialize registry with built-in and configured providers."""
        if self._initialized:
            return

        # Register built-in CLI providers
        self._providers["codex"] = CodexProvider()
        self._providers["gemini"] = GeminiProvider()

        # Load API providers from config
        config = self.load_config()
        for name, provider_config in config.items():
            ptype = provider_config.get("type", "openai_api")

            pconfig = ProviderConfig(
                name=name,
                type=ptype,
                base_url=provider_config.get("base_url"),
                api_key_env=provider_config.get("api_key_env"),
                default_model=provider_config.get("default_model"),
                timeout=provider_config.get("timeout", 300),
                cost_per_1k_input=provider_config.get("cost_per_1k_input"),
                cost_per_1k_output=provider_config.get("cost_per_1k_output"),
                site_url=provider_config.get("site_url"),
                app_name=provider_config.get("app_name"),
                extra=provider_config.get("extra"),
            )

            if ptype == "openrouter":
                self._providers[name] = OpenRouterProvider(pconfig)
            elif ptype == "openai_api":
                self._providers[name] = OpenAIAPIProvider(pconfig)
            # CLI providers are only codex/gemini which are built-in

        self._initialized = True

    def get_provider(self, name: str) -> Provider | None:
        """Get a provider by name."""
        self.initialize()
        return self._providers.get(name)

    def list_available(self) -> list[str]:
        """List names of available (configured) providers."""
        self.initialize()
        return [name for name, p in self._providers.items() if p.is_available]

    def list_all(self) -> list[dict]:
        """List all providers with availability status."""
        self.initialize()
        return [
            {
                "name": name,
                "type": p.provider_type,
                "available": p.is_available,
                "model": p.config.default_model,
            }
            for name, p in self._providers.items()
        ]

    def add_provider(
        self,
        name: str,
        provider_type: str,
        base_url: str | None = None,
        api_key_env: str | None = None,
        default_model: str | None = None,
        **kwargs,
    ) -> bool:
        """Add a new provider to the config file.

        Returns True if added successfully.
        """
        config = self.load_config()

        provider_config = {
            "type": provider_type,
        }
        if base_url:
            provider_config["base_url"] = base_url
        if api_key_env:
            provider_config["api_key_env"] = api_key_env
        if default_model:
            provider_config["default_model"] = default_model

        # Add any extra kwargs
        for k, v in kwargs.items():
            if v is not None:
                provider_config[k] = v

        config[name] = provider_config
        self.save_config(config)

        # Re-initialize to pick up new provider
        self._initialized = False
        self.initialize()

        return name in self._providers

    def remove_provider(self, name: str) -> bool:
        """Remove a provider from the config file.

        Note: Cannot remove built-in providers (codex, gemini).
        Returns True if removed successfully.
        """
        if name in ["codex", "gemini"]:
            return False

        config = self.load_config()
        if name not in config:
            return False

        del config[name]
        self.save_config(config)

        # Re-initialize
        self._initialized = False
        self.initialize()

        return name not in self._providers


# Global registry instance
registry = ProviderRegistry()
