"""Factory for instantiating and managing LLM clients."""

from typing import Dict, List, Optional, Type
from metaqore.llm.client.interface import LLMClient, LLMProvider


class LLMClientFactory:
    """
    Central registry for LLM clients.
    Applies SecureGateway policies for provider selection.
    """

    _adapters: Dict[LLMProvider, Type[LLMClient]] = {}
    _instances: Dict[LLMProvider, LLMClient] = {}

    @classmethod
    def register_adapter(cls, provider: LLMProvider, adapter_class: Type[LLMClient]) -> None:
        """Register a new adapter class."""
        cls._adapters[provider] = adapter_class

    @classmethod
    def get_client(
        cls, provider: LLMProvider, secure_gateway=None, task_metadata: Optional[Dict] = None
    ) -> LLMClient:
        """
        Get or create a client instance, applying security policies.

        Args:
            provider: Requested provider
            secure_gateway: Optional SecureGateway for policy enforcement
            task_metadata: Metadata for policy evaluation

        Returns:
            Initialized LLMClient instance
        """
        # Apply security policy if provided
        if secure_gateway and task_metadata:
            allowed = secure_gateway.get_allowed_providers(task_metadata)
            if provider not in allowed:
                # Fallback to first allowed provider
                provider = allowed[0] if allowed else LLMProvider.MOCK

        # Return cached instance or create new
        if provider not in cls._instances:
            adapter_class = cls._adapters.get(provider)
            if not adapter_class:
                raise ValueError(f"No adapter registered for {provider}")

            instance = adapter_class()

            # Load provider-specific config
            config = cls._load_provider_config(provider)
            instance.initialize(config)

            cls._instances[provider] = instance

        return cls._instances[provider]

    @classmethod
    def get_client_for_job(cls, gateway_job, secure_gateway=None) -> LLMClient:
        """
        Higher-level method to get appropriate client for a GatewayJob.
        Implements the routing logic from configuration.
        """
        # Extract routing hints from job
        provider_hint = gateway_job.provider_hint
        skill_metadata = gateway_job.payload.get("metadata", {})

        # Determine sensitivity level
        has_private = skill_metadata.get("has_private_data", False)
        has_sensitive = skill_metadata.get("has_sensitive_data", False)

        # Apply routing policy
        if has_private or has_sensitive:
            # High sensitivity → local providers only
            provider = LLMProvider.LLAMA_CPP
        else:
            # Use hint or default
            try:
                provider = LLMProvider(provider_hint)
            except ValueError:
                provider = LLMProvider.LLAMA_CPP

        return cls.get_client(provider, secure_gateway, skill_metadata)

    @classmethod
    def _load_provider_config(cls, provider: LLMProvider) -> Dict:
        """Load provider-specific configuration."""
        # TODO: Implement config loading from YAML
        # For now, return empty config
        return {}
