"""Mock adapter implementing the LLMClient interface for testing."""

import asyncio
from typing import Any, Dict, Optional
from metaqore.llm.client.interface import LLMClient, LLMProvider, LLMResponse


class MockAdapter(LLMClient):
    """Mock adapter that replicates current MockLLMClient behavior."""

    def __init__(self):
        self._initialized = False

    @property
    def provider(self) -> LLMProvider:
        return LLMProvider.MOCK

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the mock adapter."""
        self._initialized = True

    def generate(
        self, prompt: str, *, agent_name: str, metadata: Dict[str, Any], **kwargs
    ) -> LLMResponse:
        """
        Generate a mock completion.
        Simulates current MockLLMClient behavior.
        """
        if not self._initialized:
            raise RuntimeError("MockAdapter not initialized")

        # Simulate processing time
        import time

        time.sleep(0.01)

        # Generate mock response based on prompt
        if "error" in prompt.lower():
            return LLMResponse(
                content="",
                provider=self.provider,
                model="mock-model",
                success=False,
                error="Simulated error for testing",
                metadata={"agent": agent_name},
            )
        else:
            mock_content = f"Mock response for {agent_name}: {prompt[:50]}..."
            return LLMResponse(
                content=mock_content,
                provider=self.provider,
                model="mock-model",
                success=True,
                metadata={"agent": agent_name, "usage": {"tokens": len(prompt.split())}},
                usage={"tokens": len(prompt.split())},
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Mock config validation - always succeeds."""
        return True
