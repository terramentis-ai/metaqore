"""Abstract interface for all LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum
from datetime import datetime, timezone


class LLMProvider(Enum):
    """Registered LLM providers for routing and analytics."""

    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    MOCK = "mock"  # For testing


@dataclass(frozen=True)
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    provider: LLMProvider
    model: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    usage: Dict[str, int] = None  # tokens, etc.

    # PSMP Artifact Context
    artifact_context: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base client that all providers must implement."""

    @property
    @abstractmethod
    def provider(self) -> LLMProvider:
        """Return the provider type."""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the client with provider-specific configuration.
        Called once during application startup.
        """
        pass

    @abstractmethod
    def generate(
        self, prompt: str, *, agent_name: str, metadata: Dict[str, Any], **kwargs
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: Assembled prompt from PromptAssemblyEngine
            agent_name: Name of the agent making the request
            metadata: Context from GatewayJob (PSMP project_id, HMCP level, etc.)
            **kwargs: Provider-specific parameters (temperature, max_tokens)

        Returns:
            Standardized LLMResponse
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider-specific configuration."""
        pass

    def prepare_artifact_context(
        self, response: LLMResponse, job_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare PSMP artifact metadata from the response.
        Overrideable for provider-specific enhancements.
        """
        return {
            "provider": self.provider.value,
            "model": response.model,
            "usage": response.usage or {},
            "job_metadata": job_metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
