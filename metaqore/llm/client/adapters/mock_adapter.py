"""Mock adapter implementing the LLMClient interface for testing."""

import asyncio
import time
from typing import Any, Dict, Optional
from metaqore.llm.client.interface import LLMClient, LLMProvider, LLMResponse

# Import governance components for metrics and events
try:
    from metaqore_governance_core.event_bus import event_bus, Event, EventTypes
    from metaqore.metrics.aggregator import get_metrics_aggregator
except ImportError:
    # Fallback for when governance-core is not available
    event_bus = None
    Event = None
    EventTypes = None
    get_metrics_aggregator = lambda: None


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
        
        start_time = time.time()
        request_id = metadata.get("request_id", f"req_{int(start_time)}")
        
        # Publish request started event
        if event_bus and Event and EventTypes:
            event_bus.publish(Event(
                event_type=EventTypes.LLM_REQUEST_STARTED,
                source=f"llm_adapter.{self.provider.value}",
                data={
                    "request_id": request_id,
                    "provider": self.provider.value,
                    "model": "mock-model",
                    "agent_name": agent_name,
                    "prompt_length": len(prompt),
                    "metadata": metadata,
                },
                correlation_id=metadata.get("correlation_id"),
            ))

        # Simulate processing time
        time.sleep(0.01)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000

        # Generate mock response based on prompt
        if "error" in prompt.lower():
            # Record failed request metrics
            if get_metrics_aggregator():
                aggregator = get_metrics_aggregator()
                aggregator.record_api_latency(f"llm_{self.provider.value}_failed", latency_ms)
            
            # Publish failure event
            if event_bus and Event and EventTypes:
                event_bus.publish(Event(
                    event_type=EventTypes.LLM_REQUEST_FAILED,
                    source=f"llm_adapter.{self.provider.value}",
                    data={
                        "request_id": request_id,
                        "provider": self.provider.value,
                        "model": "mock-model",
                        "latency_ms": latency_ms,
                        "error": "Simulated error for testing",
                        "success": False,
                    },
                    correlation_id=metadata.get("correlation_id"),
                ))
            
            return LLMResponse(
                content="",
                provider=self.provider,
                model="mock-model",
                success=False,
                error="Simulated error for testing",
                metadata={"agent": agent_name, "latency_ms": latency_ms},
                artifact_context={
                    "provider": self.provider.value,
                    "model": "mock-model",
                    "error": "Simulated error for testing",
                    "latency_ms": latency_ms,
                    "request_id": request_id,
                    "timestamp": metadata.get("timestamp"),
                }
            )
        else:
            mock_content = f"Mock response for {agent_name}: {prompt[:50]}..."
            
            # Record metrics
            if get_metrics_aggregator():
                aggregator = get_metrics_aggregator()
                aggregator.record_api_latency(f"llm_{self.provider.value}", latency_ms)
            
            # Prepare artifact context
            artifact_context = self.prepare_artifact_context(
                LLMResponse(
                    content=mock_content,
                    provider=self.provider,
                    model="mock-model",
                    success=True,
                    usage={"tokens": len(prompt.split())},
                ),
                metadata
            )
            artifact_context.update({
                "latency_ms": latency_ms,
                "request_id": request_id,
                "agent_name": agent_name,
            })
            
            llm_response = LLMResponse(
                content=mock_content,
                provider=self.provider,
                model="mock-model",
                success=True,
                metadata={"agent": agent_name, "usage": {"tokens": len(prompt.split())}, "latency_ms": latency_ms},
                usage={"tokens": len(prompt.split())},
                artifact_context=artifact_context,
            )
            
            # Publish completion event
            if event_bus and Event and EventTypes:
                event_bus.publish(Event(
                    event_type=EventTypes.LLM_REQUEST_COMPLETED,
                    source=f"llm_adapter.{self.provider.value}",
                    data={
                        "request_id": request_id,
                        "provider": self.provider.value,
                        "model": "mock-model",
                        "latency_ms": latency_ms,
                        "tokens_used": len(prompt.split()),
                        "success": True,
                        "artifact_context": artifact_context,
                    },
                    correlation_id=metadata.get("correlation_id"),
                ))
            
            return llm_response

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Mock config validation - always succeeds."""
        return True
